"""R5：把冻结的 ref 模型 weight-only INT8。量化模块与 RL 模块的交点。

这个实验在 V100 上测的是什么
----------------------------
RL 后训练时显存里同时装着：policy（参数 + 梯度 + 优化器状态）、**冻结的 ref**、
可能还有 reward model。ref 只做前向、不要梯度、不要优化器状态，
所以它是**最先该被压缩的那一块**：把 ref 的线性层从 fp16 换成 int8，
ref 那部分显存直接减半，而且不影响 policy 的任何数值路径——除了 KL 项。

于是问题变成一个可判定的取舍：
    **省下的 ref 显存，值不值 KL 项引入的偏差？**

三个判据（本模块的三个返回字段）：
  1. kl_rel_bias        KL 均值的相对偏差。只看这个不够——均值可以碰巧对上。
  2. sign_agreement     逐 token 的 log_ratio 符号一致率。
                        符号决定 KL 惩罚把这个 token 往上推还是往下压；
                        符号翻了，惩罚方向就反了。**这是比均值更硬的判据。**
  3. grad_cosine        用 int8-ref 与 fp16-ref 分别算完整 RL loss 后，
                        policy 梯度的余弦相似度。这是最终判据：
                        梯度方向一致，训练轨迹才一致。

在 V100 上这三个数都能实测，因为它们只需要前向 + 一次反向。

不能测什么
----------
- **不能测速度**。V100 没有 INT8 Tensor Core，int8 ref 的前向比 fp16 ref **更慢**
  （多一次 dequant）。这个实验换来的是显存，不是时间。
- 不能只跑一个 batch 就定论：sign_agreement 在 log_ratio 接近 0 的 token 上
  天然不稳定（policy 刚从 ref 复制出来时几乎所有 token 都是这种情况）。
  报告里同时给出 |log_ratio| 的分位数，才能判断「符号翻转」是不是无关紧要的翻转。

默认跳过 lm_head
----------------
MiniMind 的 lm_head 与 embedding 绑权重。量化 lm_head 会同时改掉输入侧的
embedding 表现，那就不再是「只量化 ref 的线性层」这个受控实验了。
所以 skip_name_contains 默认含 "lm_head"；要量化它得显式打开，并说明理由。
"""

from __future__ import annotations

import copy
from typing import Any, Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from mm_quant.weight_only import (
    module_weight_bytes,
    quantize_linear_modules,
    top1_agreement,
)

from .kl import kl_k3, log_ratio
from .logprob import token_logprobs

__all__ = [
    "quantize_reference_model",
    "ref_kl_report",
    "policy_grad_cosine",
    "format_ref_report",
]


def quantize_reference_model(
    ref_model: nn.Module,
    granularity: str = "per_channel",
    group_size: int = 128,
    compute_dtype: torch.dtype = torch.float16,
    skip_name_contains: Sequence[str] = ("lm_head",),
    deepcopy: bool = True,
) -> Tuple[nn.Module, Dict[str, Any]]:
    """把 ref 模型的 nn.Linear 换成 W8A16Linear，并冻结全部参数。

    deepcopy=True（默认）返回一个新模型，原模型不动——R5 需要 fp16-ref 与
    int8-ref **同时存在**才能比，就地修改会把对照组毁掉。

    compute_dtype 的两种用法，别混：
      torch.float16（默认）V100 上的真实配置。此时观测到的偏差 =
                           量化误差 + fp16 计算误差，两者叠加。
      torch.float32        **受控实验**用。此时 int8-ref 与 fp-ref 的唯一差别
                           就是权重表示，观测到的偏差全部归因于量化。
                           本机 CPU 单测一律用这一档；V100 上两档都跑，
                           两者之差就是 fp16 计算贡献的那一份。
    """
    model = copy.deepcopy(ref_model) if deepcopy else ref_model
    before = module_weight_bytes(model, reference_dtype=compute_dtype)
    model, replaced = quantize_linear_modules(
        model,
        granularity=granularity,
        group_size=group_size,
        compute_dtype=compute_dtype,
        skip_name_contains=skip_name_contains,
    )
    after = module_weight_bytes(model, reference_dtype=compute_dtype)
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()
    info = {
        "n_replaced": len(replaced),
        "replaced": replaced,
        "skipped_contains": list(skip_name_contains),
        "linear_bytes_before": before["stored_bytes"],
        "linear_bytes_after": after["stored_bytes"],
        "weight_bytes_ratio": after["weight_bytes_ratio"],
        "reference_dtype": after["reference_dtype"],
    }
    return model, info


@torch.no_grad()
def ref_kl_report(
    logits_policy: torch.Tensor,
    logits_ref_fp: torch.Tensor,
    logits_ref_int8: torch.Tensor,
    labels: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    compute_dtype: torch.dtype = torch.float32,
) -> Dict[str, Any]:
    """KL 相对偏差 + 符号一致率 + logits 层面的对照。

    全部在 compute_dtype（默认 fp32）里算：这一步的目的是量「量化引入的偏差」，
    不能让 fp16 的舍入混进来当噪声。想看 fp16 叠加效应，把 compute_dtype 调成
    torch.float16 再跑一遍，两次结果的差就是「精度」贡献的那一份。
    """
    lp_policy = token_logprobs(logits_policy, labels, compute_dtype=compute_dtype)
    lp_ref_fp = token_logprobs(logits_ref_fp, labels, compute_dtype=compute_dtype)
    lp_ref_q = token_logprobs(logits_ref_int8, labels, compute_dtype=compute_dtype)

    lr_fp = log_ratio(lp_policy, lp_ref_fp, compute_dtype=compute_dtype)
    lr_q = log_ratio(lp_policy, lp_ref_q, compute_dtype=compute_dtype)
    k3_fp = kl_k3(lr_fp)
    k3_q = kl_k3(lr_q)

    m = mask.to(torch.bool) if mask is not None else torch.ones_like(lr_fp, dtype=torch.bool)
    n = int(m.sum())
    if n == 0:
        raise ValueError("ref_kl_report 收到全 False 的 mask")

    lr_fp_m = lr_fp[m].to(torch.float32)
    lr_q_m = lr_q[m].to(torch.float32)
    k3_fp_m = k3_fp[m].to(torch.float32)
    k3_q_m = k3_q[m].to(torch.float32)

    mean_fp = float(k3_fp_m.mean())
    mean_q = float(k3_q_m.mean())
    sign_fp = torch.sign(lr_fp_m)
    sign_q = torch.sign(lr_q_m)
    nonzero = sign_fp != 0

    abs_lr = lr_fp_m.abs()
    q50 = float(abs_lr.median()) if abs_lr.numel() else float("nan")

    return {
        "n_tokens": n,
        "kl_k3_mean_ref_fp": mean_fp,
        "kl_k3_mean_ref_int8": mean_q,
        "kl_rel_bias": (abs(mean_q - mean_fp) / abs(mean_fp)) if mean_fp != 0 else float("inf"),
        "kl_max_abs_token_diff": float((k3_q_m - k3_fp_m).abs().max()),
        "sign_agreement": (
            float((sign_fp[nonzero] == sign_q[nonzero]).to(torch.float32).mean())
            if int(nonzero.sum()) > 0
            else float("nan")
        ),
        "n_sign_comparable": int(nonzero.sum()),
        "log_ratio_abs_median_ref_fp": q50,
        "log_ratio_max_abs_diff": float((lr_q_m - lr_fp_m).abs().max()),
        "ref_logits_top1_agreement": top1_agreement(
            logits_ref_fp, logits_ref_int8, mask=mask
        )["top1_agreement"],
    }


def policy_grad_cosine(
    policy: nn.Module,
    ref_fp: nn.Module,
    ref_int8: nn.Module,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    advantages: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    beta: float = 0.04,
    compute_dtype: torch.dtype = torch.float32,
) -> Dict[str, Any]:
    """两次完整的 RL loss 反向，比 policy 梯度的余弦相似度。

    loss = -mean(A * logp) + beta * mean(k3(log(pi/ref)))

    这是 R5 的最终判据。两条路径唯一的差别是 ref 模型的权重表示，
    所以余弦相似度偏离 1 的部分**全部**来自量化。

    余弦相似度看哪一档：
      > 0.999   量化 ref 对梯度方向的影响小于常见的 seed 抖动，可以上。
      0.99–0.999 需要配一个短的对照训练再决定。
      < 0.99    别用，或者换更细的粒度（per_group）再测。
    这三档是**工程经验阈值**，不是定理；写在这里是为了让判断有据可依，
    真正的判据是 Day 5 的对照训练。
    """
    if mask is None:
        mask = torch.ones_like(labels, dtype=torch.bool)

    def _grads(ref: nn.Module) -> torch.Tensor:
        policy.zero_grad(set_to_none=True)
        logits_p = policy(input_ids)
        with torch.no_grad():
            logits_r = ref(input_ids)
        lp = token_logprobs(logits_p, labels, compute_dtype=compute_dtype)
        lr_ref = token_logprobs(
            logits_r.to(compute_dtype), labels, compute_dtype=compute_dtype
        )
        lr = lp - lr_ref
        m = mask.to(torch.bool)
        pg = -(advantages.to(compute_dtype) * lp)[m].mean()
        kl = kl_k3(lr)[m].mean()
        loss = pg + beta * kl
        loss.backward()
        parts = [
            p.grad.detach().reshape(-1).to(torch.float32)
            for p in policy.parameters()
            if p.grad is not None
        ]
        if not parts:
            raise RuntimeError("policy 没有任何参数拿到梯度：检查 requires_grad 与 loss 依赖")
        return torch.cat(parts)

    g_fp = _grads(ref_fp)
    g_q = _grads(ref_int8)
    policy.zero_grad(set_to_none=True)

    n_fp = float(torch.linalg.vector_norm(g_fp))
    n_q = float(torch.linalg.vector_norm(g_q))
    # 两个梯度几乎重合时，fp32 下的 cosine_similarity 会给出 1 + 1e-5 这种
    # 数学上不可能的值。夹回 [-1, 1]，否则报告里出现 >1 会让人怀疑代码而不是舍入。
    cos = float(torch.nn.functional.cosine_similarity(g_fp, g_q, dim=0).clamp(-1.0, 1.0))
    return {
        "n_grad_elements": int(g_fp.numel()),
        "grad_cosine": cos,
        "grad_norm_ref_fp": n_fp,
        "grad_norm_ref_int8": n_q,
        "grad_norm_rel_diff": (abs(n_q - n_fp) / n_fp) if n_fp > 0 else float("inf"),
        "grad_max_abs_diff": float((g_fp - g_q).abs().max()),
        "beta": float(beta),
    }


def format_ref_report(kl_report: Dict[str, Any], bytes_info: Dict[str, Any]) -> str:
    lines = [
        "ref 模型 weight-only INT8（替换 {} 个 Linear，跳过含 {} 的）".format(
            bytes_info["n_replaced"], bytes_info["skipped_contains"]
        ),
        "  线性层字节 {} -> {}（相对 {} 参考 {:.3f}）".format(
            bytes_info["linear_bytes_before"],
            bytes_info["linear_bytes_after"],
            bytes_info["reference_dtype"],
            bytes_info["weight_bytes_ratio"],
        ),
        "  KL(k3) 均值 fp-ref {:.6e} -> int8-ref {:.6e}，相对偏差 {:.3%}".format(
            kl_report["kl_k3_mean_ref_fp"],
            kl_report["kl_k3_mean_ref_int8"],
            kl_report["kl_rel_bias"],
        ),
        "  log_ratio 符号一致率 {:.4f}（可比 token {}，|log_ratio| 中位数 {:.3e}）".format(
            kl_report["sign_agreement"],
            kl_report["n_sign_comparable"],
            kl_report["log_ratio_abs_median_ref_fp"],
        ),
        "  ref logits top-1 一致率 {:.4f}".format(kl_report["ref_logits_top1_agreement"]),
    ]
    return "\n".join(lines)
