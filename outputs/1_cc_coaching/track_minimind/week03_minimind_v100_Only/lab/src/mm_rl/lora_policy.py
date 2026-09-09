"""R6：LoRA policy + 冻结 backbone，以及四项字节账的 full vs LoRA 对照。

这个实验在 V100 上测的是什么
----------------------------
V100 是 32 GB/卡、8 卡。RL 后训练要在一张卡上同时放 policy + ref
（+ 可能的 reward model），字节账立刻变成主要矛盾。LoRA 改变的是**其中三项**：

    参数      backbone 仍然常驻（没省），多出 2*r*(d_in+d_out) 的 A/B
    梯度      只有 LoRA 参数有梯度 —— 这一项从 4P 掉到 4*p_lora
    优化器    Adam 的 m/v 只跟着可训练参数走 —— 从 8P 掉到 8*p_lora
    激活      冻结层不需要为「权重梯度」保存输入；但仍需要权重本身来算
              输入梯度，所以省的是 **saved-for-weight-grad** 那一部分

第四项是最容易讲错的一项。本模块用**前向钩子实测**：统计 autograd 需要为
「计算权重梯度」而保留的输入张量字节数，而不是背一个公式。
它仍然是估算（autograd 的实际保留策略比这复杂），标注为 `估算`。

**关键不变量（本模块最强的一条断言）**
--------------------------------------
LoRA 的 B 零初始化 => B(A(x)) 恒为 0 => step 0 的 policy 输出与 backbone
（也就是 ref）**逐位相同** => log_ratio 恒为 0 => 三个 KL 估计量**精确等于 0**。

这是整个 lab 里唯一一个「应当精确为 0」的断言（不是「小于 1e-6」）。
它一旦不成立，说明下面某一条被破坏了：
  - B 不是零初始化；
  - dropout 在 train 模式下被激活（LoRA 分支的 dropout 作用在输入上，
    但输出乘 B=0 后仍是 0，所以这一条不会破坏不变量；破坏它的是把
    dropout 加在**主分支**上）；
  - scaling 里出现了 NaN（r=0 时 alpha/r 是 inf，0*inf = NaN）；
  - 有人顺手改了 backbone 的权重。
所以这条不变量是 RL 训练开始前的**一次性冒烟测试**：
它过了，才说明 policy 和 ref 真的是同一个起点。
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from .kl import kl_estimators, log_ratio
from .logprob import token_logprobs

__all__ = [
    "LoRALinear",
    "apply_lora",
    "lora_parameters",
    "trainable_parameter_count",
    "byte_ledger",
    "step0_identity_check",
    "format_byte_ledger",
]


class LoRALinear(nn.Module):
    """base(x) + (alpha/r) * B(A(dropout(x)))，base 冻结。

    A 用 kaiming_uniform（与 microsoft/LoRA 参考实现一致），**B 恒零初始化**。
    B 零初始化是不变量的来源，不提供「随机初始化 B」的选项——
    想看破坏不变量会发生什么，用 break_zero_init() 显式地破坏它。
    """

    def __init__(
        self,
        base: nn.Linear,
        r: int = 8,
        alpha: int = 16,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if r < 1:
            raise ValueError("LoRA 的 r 必须 >= 1，收到 {}（r=0 会让 alpha/r 变 inf）".format(r))
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.r = int(r)
        self.alpha = int(alpha)
        self.scaling = float(alpha) / float(r)
        self.lora_dropout = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()
        self.lora_A = nn.Linear(base.in_features, r, bias=False)
        self.lora_B = nn.Linear(r, base.out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        delta = self.lora_B(self.lora_A(self.lora_dropout(x)))
        # B=0 时 delta 逐元素精确为 0，out + 0.0 在 IEEE 下等于 out（含 -0.0 的情况
        # 也不会改变后续 logits 的比较结果），这就是 step-0 不变量成立的原因。
        return out + self.scaling * delta

    def break_zero_init(self, std: float = 0.02, seed: Optional[int] = None) -> None:
        """故意破坏 B 的零初始化。只给「让不变量失败」的演示用。"""
        gen = None
        if seed is not None:
            gen = torch.Generator(device="cpu").manual_seed(seed)
        with torch.no_grad():
            noise = torch.randn(self.lora_B.weight.shape, generator=gen)
            self.lora_B.weight.copy_(noise.to(self.lora_B.weight.device) * std)

    def extra_repr(self) -> str:
        return "r={}, alpha={}, scaling={:.4g}".format(self.r, self.alpha, self.scaling)


def apply_lora(
    model: nn.Module,
    target_suffixes: Sequence[str] = ("q_proj", "v_proj"),
    r: int = 8,
    alpha: int = 16,
    dropout: float = 0.0,
    freeze_base: bool = True,
) -> Tuple[nn.Module, List[str]]:
    """就地把匹配的 nn.Linear 换成 LoRALinear，其余参数冻结。返回 (model, 替换的名字)。

    target_suffixes 默认 ("q_proj", "v_proj")：这是 LoRA 论文里性价比最高的一组，
    也是 MiniMind 注意力里的命名。要全覆盖就传 ("q_proj","k_proj","v_proj","o_proj")。
    """
    if freeze_base:
        for p in model.parameters():
            p.requires_grad_(False)
    replaced: List[str] = []
    for parent_name, parent in list(model.named_modules()):
        for attr, child in list(parent.named_children()):
            if not isinstance(child, nn.Linear):
                continue
            full = "{}.{}".format(parent_name, attr) if parent_name else attr
            if not any(full.endswith(s) for s in target_suffixes):
                continue
            setattr(parent, attr, LoRALinear(child, r=r, alpha=alpha, dropout=dropout))
            replaced.append(full)
    if not replaced:
        raise ValueError(
            "没有任何 Linear 匹配 {}。先 print([n for n,_ in model.named_modules()]) "
            "看真实命名。".format(list(target_suffixes))
        )
    return model, replaced


def lora_parameters(model: nn.Module) -> List[torch.nn.Parameter]:
    return [p for n, p in model.named_parameters() if ("lora_A" in n or "lora_B" in n)]


def trainable_parameter_count(model: nn.Module) -> Dict[str, int]:
    seen = set()
    trainable = 0
    total = 0
    for p in model.parameters():
        if id(p) in seen:
            continue
        seen.add(id(p))
        total += p.numel()
        if p.requires_grad:
            trainable += p.numel()
    return {"total": total, "trainable": trainable, "frozen": total - trainable}


class _ActivationAccountant:
    """用前向钩子估「为了算权重梯度而必须保留的输入」有多少字节。

    规则：对每个 nn.Linear（含 LoRALinear 内部的 base / lora_A / lora_B），
    只要它的 weight.requires_grad 为 True，autograd 就必须保留这一层的输入
    才能算 dW = grad_out^T @ x。weight 冻结时这份输入不需要被保留。

    这是**估算**：真实的 autograd 还会保留 softmax 输出、silu 输入等等，
    那些与 LoRA 与否无关，两边同时存在，做差时抵消。
    """

    def __init__(self) -> None:
        self.bytes_required = 0
        self.bytes_all = 0
        self.handles: List[Any] = []

    def _hook(self, mod: nn.Module, inputs, _out):
        if not inputs:
            return
        x = inputs[0]
        if not isinstance(x, torch.Tensor):
            return
        nbytes = x.numel() * x.element_size()
        self.bytes_all += nbytes
        w = getattr(mod, "weight", None)
        if isinstance(w, torch.Tensor) and w.requires_grad:
            self.bytes_required += nbytes

    def attach(self, model: nn.Module) -> None:
        for _, mod in model.named_modules():
            if isinstance(mod, nn.Linear):
                self.handles.append(mod.register_forward_hook(self._hook))

    def detach(self) -> None:
        for h in self.handles:
            h.remove()
        self.handles = []


def byte_ledger(
    model: nn.Module,
    input_ids: torch.Tensor,
    param_dtype_bytes: int = 4,
    grad_dtype_bytes: int = 4,
    optimizer_states: int = 2,
    optimizer_state_bytes: int = 4,
) -> Dict[str, Any]:
    """四项字节账：参数 / 梯度 / 优化器状态 / 激活（估算）。

    参数量按去重后的 numel 算（tie_embeddings 时 lm_head 与 embedding 是同一张量）。
    梯度与优化器状态只算 requires_grad=True 的参数——这正是 LoRA 省下来的地方。
    激活项用一次真实前向的钩子统计，标注为 `估算`。

    默认 dtype 字节数是 fp32（4）。V100 的 AMP 训练里参数主副本是 fp32、
    autocast 产生 fp16 的临时张量，所以「参数 4 B、梯度 4 B、Adam 状态 2x4 B」
    是主副本口径；显存里还有 fp16 的权重拷贝，那一项由 Day 3 的 FSDP 实测覆盖，
    不在本函数范围内。
    """
    counts = trainable_parameter_count(model)
    acct = _ActivationAccountant()
    acct.attach(model)
    was_training = model.training
    model.train()
    try:
        with torch.enable_grad():
            model(input_ids)
    finally:
        acct.detach()
        model.train(was_training)

    param_bytes = counts["total"] * param_dtype_bytes
    grad_bytes = counts["trainable"] * grad_dtype_bytes
    opt_bytes = counts["trainable"] * optimizer_states * optimizer_state_bytes
    return {
        "n_params_total": counts["total"],
        "n_params_trainable": counts["trainable"],
        "trainable_ratio": (
            counts["trainable"] / counts["total"] if counts["total"] else float("nan")
        ),
        "param_bytes": param_bytes,
        "grad_bytes": grad_bytes,
        "optimizer_bytes": opt_bytes,
        "activation_bytes_estimate": acct.bytes_required,
        "activation_bytes_all_linear_inputs": acct.bytes_all,
        "total_bytes": param_bytes + grad_bytes + opt_bytes + acct.bytes_required,
        "note": "activation_* 为估算：只数 requires_grad 的 Linear 输入",
    }


@torch.no_grad()
def step0_identity_check(
    policy: nn.Module,
    ref: nn.Module,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> Dict[str, Any]:
    """LoRA step-0 不变量：policy 与 ref 的 logits 逐位相同，三个 KL 估计量精确为 0。

    返回的 exact_zero_kl 必须是 True。它是 bool 不是「小于某个阈值」——
    因为 B=0 时 delta 是精确的 0，不是很小的数。
    """
    policy_training = policy.training
    ref_training = ref.training
    policy.eval()
    ref.eval()
    try:
        logits_p = policy(input_ids)
        logits_r = ref(input_ids)
    finally:
        policy.train(policy_training)
        ref.train(ref_training)

    logits_equal = bool(torch.equal(logits_p, logits_r))
    lp = token_logprobs(logits_p, labels, compute_dtype=torch.float32)
    lr_ = token_logprobs(logits_r, labels, compute_dtype=torch.float32)
    log_r = log_ratio(lp, lr_, compute_dtype=torch.float32)
    ests = kl_estimators(log_r)
    m = mask.to(torch.bool) if mask is not None else torch.ones_like(log_r, dtype=torch.bool)

    max_abs = {k: float(v[m].abs().max()) for k, v in ests.items()}
    return {
        "logits_bitwise_equal": logits_equal,
        "max_abs_logit_diff": float((logits_p - logits_r).abs().max()),
        "max_abs_log_ratio": float(log_r[m].abs().max()),
        "max_abs_kl": max_abs,
        "exact_zero_kl": all(v == 0.0 for v in max_abs.values()),
        "n_tokens": int(m.sum()),
    }


def format_byte_ledger(full: Dict[str, Any], lora: Dict[str, Any]) -> str:
    rows = [
        ("参数", "param_bytes"),
        ("梯度", "grad_bytes"),
        ("优化器状态", "optimizer_bytes"),
        ("激活(估算)", "activation_bytes_estimate"),
        ("合计", "total_bytes"),
    ]
    lines = [
        "四项字节账（单位 B；激活列标注为 `估算`）",
        "{:<14} {:>14} {:>14} {:>10}".format("项", "full", "lora", "lora/full"),
    ]
    lines.append("-" * len(lines[-1]))
    for label, key in rows:
        f = full[key]
        l = lora[key]
        lines.append(
            "{:<14} {:>14,} {:>14,} {:>10}".format(
                label, f, l, "{:.4f}".format(l / f) if f else "-"
            )
        )
    lines.append(
        "可训练参数 {:,} / {:,} = {:.4%}".format(
            lora["n_params_trainable"], lora["n_params_total"], lora["trainable_ratio"]
        )
    )
    return "\n".join(lines)
