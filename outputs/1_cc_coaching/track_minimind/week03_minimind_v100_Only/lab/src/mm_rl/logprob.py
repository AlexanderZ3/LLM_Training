"""R1：logprob 与 ratio 在 fp16 / fp32 两条路径下的差异，以及**归因**。

这个实验在 V100 上测的是什么
----------------------------
PPO/GRPO 的每一步都建立在 ratio = exp(logp_new - logp_old) 上。
V100 上只有 fp16（sm70 无 BF16），而 fp16 的 eps = 2^-10 ≈ 9.8e-4。
所以问题是：**在什么条件下，fp16 引入的误差已经大到能改变 clip 的判定？**
clip_eps 通常是 0.2；如果 logp 的绝对误差能到 1e-3 量级，ratio 的误差就是
约 1e-3 的相对量，离 0.2 还远——但如果 old_logp 是**用 fp16 缓存**的，
误差会被 exp 放大，并且在长序列上按 token 累加进序列级 ratio 里。

torch 的 AMP 有一条保护：log_softmax / softmax / cross_entropy 在 autocast 下
会自动提升到 fp32 输出。所以「在 autocast 里算 logprob」通常是安全的。
**真正的坑在 autocast 之外**：ratio = exp(logp - logp_old) 这一句如果写在
autocast 区外、或者 logp_old 是从 fp16 buffer 里读回来的，就没有任何保护。

不能测什么
----------
- CPU 的 fp16 是软件模拟，**舍入行为与 V100 的 Tensor Core 不同**。本机能验的是
  「归因逻辑对不对」，不能验「V100 上误差有多大」。后者必须在 V100 上跑。
- 不能只看一个 batch 就下结论：ratio 的尾部是长尾，max|ratio-1| 的方差很大。

归因的四种结论
--------------
IDENTICAL       两条路径逐位相同。
MASK_MISMATCH   两条路径的有效 token 集合不同。这是**第一个要排除的**：
                它会伪装成「精度问题」，但量级完全对不上。
PRECISION       差值落在 dtype 能解释的范围内（见下面的界）。
PATH_MISMATCH   差值超出该界。说明两条路径算的不是同一个东西：
                不同的 attention 后端、不同的 mask、不同的 temperature、
                或者 old_logp 来自另一个模型版本。

精度界怎么来的
--------------
一次 log_softmax 的相对误差约 O(eps)，再经过 T 个 token 的独立累加，
误差按 sqrt(T) 增长（随机游走）而不是 T。这里取一个保守的常数 k（默认 8）：
    bound = k * eps(dtype) * (1 + max|logp|)
超过这个界就不该再用「精度」解释。k 是可调的，调它要说明理由。
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Tuple

import torch
import torch.nn.functional as F

__all__ = [
    "ATTRIBUTIONS",
    "token_logprobs",
    "dual_precision_logprobs",
    "ratio_stats",
    "precision_bound",
    "attribute_divergence",
    "determinism_check",
]

ATTRIBUTIONS: Tuple[str, ...] = (
    "IDENTICAL",
    "MASK_MISMATCH",
    "PRECISION",
    "PATH_MISMATCH",
)


def _quantile(x: torch.Tensor, q: float) -> float:
    """torch.quantile 对元素数有上限（约 1.6e7），超了就用排序取分位。

    RL 里一个 rollout batch 的 token 数很容易超过这个上限，
    直接调 torch.quantile 会在 V100 上抛 RuntimeError 而不是在本机——
    这种「只在大规模下才炸」的坑必须在写的时候就堵掉。
    """
    flat = x.reshape(-1)
    n = flat.numel()
    if n == 0:
        return float("nan")
    if n <= 16_000_000:
        return float(torch.quantile(flat.to(torch.float32), q))
    idx = min(n - 1, max(0, int(round(q * (n - 1)))))
    return float(flat.to(torch.float32).sort().values[idx])


def token_logprobs(
    logits: torch.Tensor,
    labels: torch.Tensor,
    compute_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """gather 出每个 label token 的 log p。返回 [B, T]，dtype 与 log_softmax 相同。

    compute_dtype 指定 log_softmax 在哪个 dtype 里算：
      None          用 logits 自己的 dtype（AMP 之外的真实行为）
      torch.float32 强制 fp32（AMP 之内 torch 自动做的事）
    把它做成显式参数，是为了让 R1 能把「autocast 保护住了」和「没保护住」两种情况
    在同一段代码里跑出来对比，而不是靠改环境。
    """
    x = logits if compute_dtype is None else logits.to(compute_dtype)
    logp = F.log_softmax(x, dim=-1)
    return logp.gather(-1, labels.unsqueeze(-1)).squeeze(-1)


def dual_precision_logprobs(
    logits_fp32: torch.Tensor,
    labels: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """同一批 logits 走三条路径，返回三个 [B, T] 的 logprob（都上抬回 fp32 存放）。

    fp32          参考：fp32 logits + fp32 log_softmax。
    fp16_compute  logits 降到 fp16 再算 log_softmax（fp16 域内计算）。
                  这是「autocast 保护失效」的情形。
    fp16_stored   fp32 算完 logprob，再用 fp16 缓存、读回来。
                  这是 RL 里最常见的那一种：old_logp 存进 buffer 时被降精度。
    """
    ref = token_logprobs(logits_fp32.to(torch.float32), labels, compute_dtype=torch.float32)
    comp = token_logprobs(logits_fp32.to(torch.float16), labels, compute_dtype=None).to(
        torch.float32
    )
    stored = ref.to(torch.float16).to(torch.float32)
    return {"fp32": ref, "fp16_compute": comp, "fp16_stored": stored}


def ratio_stats(
    logp_new: torch.Tensor,
    logp_old: torch.Tensor,
    clip_eps: float = 0.2,
    mask: Optional[torch.Tensor] = None,
    compute_dtype: torch.dtype = torch.float32,
) -> Dict[str, float]:
    """ratio = exp(logp_new - logp_old) 的统计量。

    compute_dtype 控制**减法和 exp** 在哪个 dtype 里做。fp16 时这一步没有
    autocast 的自动提升保护，是 R1 真正要暴露的地方。
    """
    a = logp_new.to(compute_dtype)
    b = logp_old.to(compute_dtype)
    log_r = a - b
    r = torch.exp(log_r)
    r32 = r.to(torch.float32)
    log_r32 = log_r.to(torch.float32)
    if mask is not None:
        m = mask.to(torch.bool)
        r32 = r32[m]
        log_r32 = log_r32[m]
    n = int(r32.numel())
    if n == 0:
        raise ValueError("ratio_stats 收到 0 个有效 token（mask 全 False？）")
    clipped = (r32 > 1.0 + clip_eps) | (r32 < 1.0 - clip_eps)
    return {
        "n_tokens": float(n),
        "compute_eps": float(torch.finfo(compute_dtype).eps),
        "mean_ratio": float(r32.mean()),
        "max_abs_ratio_minus_1": float((r32 - 1.0).abs().max()),
        "p99_abs_ratio_minus_1": _quantile((r32 - 1.0).abs(), 0.99),
        "frac_clipped": float(clipped.to(torch.float32).mean()),
        "max_abs_log_ratio": float(log_r32.abs().max()),
        "n_nonfinite": float((~torch.isfinite(r32)).sum()),
    }


def precision_bound(
    logp: torch.Tensor, dtype: torch.dtype = torch.float16, k: float = 8.0
) -> float:
    """k * eps(dtype) * (1 + max|logp|)。见模块 docstring 的推导。"""
    eps = float(torch.finfo(dtype).eps)
    return float(k) * eps * (1.0 + float(logp.detach().to(torch.float32).abs().max()))


def attribute_divergence(
    logp_a: torch.Tensor,
    logp_b: torch.Tensor,
    mask_a: Optional[torch.Tensor] = None,
    mask_b: Optional[torch.Tensor] = None,
    dtype: torch.dtype = torch.float16,
    k: float = 8.0,
) -> Dict[str, Any]:
    """把两条 logprob 路径的差异归到四类之一。返回 verdict + 支撑数字。

    判定顺序是固定的，先排除 mask 再谈精度——反过来会把 mask bug 当成
    「fp16 不稳定」，然后去调 GradScaler，越调越乱。
    """
    if logp_a.shape != logp_b.shape:
        return {
            "verdict": "PATH_MISMATCH",
            "reason": "形状不同：{} vs {}".format(tuple(logp_a.shape), tuple(logp_b.shape)),
            "max_abs_diff": float("nan"),
            "bound": float("nan"),
        }

    a = logp_a.detach().to(torch.float32)
    b = logp_b.detach().to(torch.float32)

    ma = torch.ones_like(a, dtype=torch.bool) if mask_a is None else mask_a.to(torch.bool)
    mb = torch.ones_like(b, dtype=torch.bool) if mask_b is None else mask_b.to(torch.bool)
    if not torch.equal(ma, mb):
        diff_positions = int((ma ^ mb).sum())
        return {
            "verdict": "MASK_MISMATCH",
            "reason": "两条路径的有效 token 集合不同，差 {} 个位置".format(diff_positions),
            "n_valid_a": int(ma.sum()),
            "n_valid_b": int(mb.sum()),
            "n_mask_diff": diff_positions,
            "max_abs_diff": float((a - b).abs().max()),
            "bound": precision_bound(a, dtype=dtype, k=k),
        }

    sel = ma
    if int(sel.sum()) == 0:
        raise ValueError("attribute_divergence 收到全 False 的 mask，无法归因")
    d = (a - b)[sel].abs()
    max_diff = float(d.max())
    bound = precision_bound(a[sel], dtype=dtype, k=k)

    if max_diff == 0.0:
        verdict = "IDENTICAL"
        reason = "两条路径逐位相同"
    elif max_diff <= bound:
        verdict = "PRECISION"
        reason = "最大差 {:.3e} <= {} 的精度界 {:.3e}".format(
            max_diff, str(dtype).replace("torch.", ""), bound
        )
    else:
        verdict = "PATH_MISMATCH"
        reason = "最大差 {:.3e} 超出 {} 的精度界 {:.3e}（{:.1f} 倍），不能用精度解释".format(
            max_diff, str(dtype).replace("torch.", ""), bound, max_diff / bound
        )
    return {
        "verdict": verdict,
        "reason": reason,
        "n_valid": int(sel.sum()),
        "max_abs_diff": max_diff,
        "mean_abs_diff": float(d.mean()),
        "bound": bound,
        "ratio_to_bound": max_diff / bound if bound > 0 else float("inf"),
    }


def determinism_check(fn: Callable[[], torch.Tensor], runs: int = 2) -> Dict[str, Any]:
    """同一个函数跑 runs 次，看结果是否逐位一致。

    这是归因链上被最常跳过的一步：在把差异归给「fp16 精度」之前，
    先证明**同一条路径自己是确定的**。不确定的话（非确定性 kernel、
    atomicAdd、cudnn benchmark 选到不同算法），任何两次比较都没有意义。
    V100 上跑 attention/reduction 时这一步必须做。
    """
    outs = [fn().detach().to(torch.float32) for _ in range(max(2, runs))]
    ref = outs[0]
    all_equal = all(torch.equal(ref, o) for o in outs[1:])
    max_diff = max(float((ref - o).abs().max()) for o in outs[1:])
    return {
        "runs": len(outs),
        "bitwise_identical": bool(all_equal),
        "max_abs_diff_across_runs": max_diff,
    }
