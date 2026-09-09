"""Q4：W8A16 weight-only PTQ。int8 存权重，用前 dequant 回 fp16 再做 GEMM。

这个实验在 V100 上测的是什么
----------------------------
测两件事，两件都真实：
1. **字节账**：权重常驻显存从 2 B/元素降到约 1.03 B/元素（int8 码 + per-channel scale）。
   对 ref 模型、对推理侧的 KV 之外的部分，这是实打实的显存。
2. **误差账**：同一批 prompt 下，逐 token top-1 一致率掉多少。

不能测什么
----------
- **不能测速度收益。** V100 没有 INT8 Tensor Core，所以这里的算子链是
  「int8 -> fp16 dequant -> fp16 GEMM」。相比直接存 fp16，多了一次 dequant，
  必然更慢。W8A16 在 A100/H100/消费卡上之所以能加速，是因为 decode 阶段
  GEMM 是**访存受限**的，读权重的字节数减半 > dequant 的算力开销；
  V100 上这个权衡同样存在，但没有 INT8 Tensor Core 也没有成熟的融合 kernel，
  本实现的逐次 dequant 不融合进 GEMM，所以拿不到那个收益。
  **不要用本模块的墙钟时间下任何速度结论。**
- torch 2.1 里 torch._weight_int8pack_mm 不存在（2.3 才有，且是 CPU/MPS 侧），
  torchao 要求 torch >= 2.5。所以「融合 dequant+GEMM」这条路在目标机上关闭。
  torch._int_mm 在 2.1 存在但没有算力检查、且有已知正确性缺陷
  （pytorch#107671），是否可用由 lab/scripts/probe_int8_caps.py 现场探。

命名
----
W8A16 = 权重 8 bit，激活 16 bit。激活不量化，所以不需要校准集，
也不会有激活 outlier 问题（那是 Q6 的题目）。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .quantizers import Blocking, compute_qparams, _quantize_blocks, _storage_dtype

__all__ = [
    "W8A16Linear",
    "quantize_linear_modules",
    "module_weight_bytes",
    "weight_bytes_ratio",
    "top1_agreement",
    "logit_divergence",
]


class W8A16Linear(nn.Module):
    """int8 权重 + fp16 计算的 Linear 替身。

    存储：qweight (int8, 块视图) + scale/zero_point (fp32)。
    前向：dequant -> compute_dtype -> F.linear。dequant 每次前向都做一遍，
    这是「省显存不省时间」的直接体现，不要把它优化掉——它就是本实验的观测对象。
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        granularity: str = "per_channel",
        axis: int = 0,
        group_size: int = 128,
        scheme: str = "symmetric",
        num_bits: int = 8,
        compute_dtype: torch.dtype = torch.float16,
    ) -> None:
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.granularity = granularity
        self.axis = int(axis)
        self.group_size = int(group_size)
        self.scheme = scheme
        self.num_bits = int(num_bits)
        self.compute_dtype = compute_dtype
        self._blocking = Blocking(
            (self.out_features, self.in_features),
            granularity,
            axis=axis,
            group_size=group_size,
        )
        qmin, qmax = (-127, 127) if scheme == "symmetric" else (-128, 127)
        self.register_buffer(
            "qweight",
            torch.zeros(
                (self._blocking.num_blocks, self._blocking.block_numel),
                dtype=_storage_dtype(qmin, qmax),
            ),
        )
        self.register_buffer(
            "scale", torch.ones((self._blocking.num_blocks, 1), dtype=torch.float32)
        )
        self.register_buffer(
            "zero_point", torch.zeros((self._blocking.num_blocks, 1), dtype=torch.float32)
        )
        if bias:
            self.register_buffer("bias", torch.zeros(self.out_features, dtype=torch.float32))
        else:
            self.bias = None

    # --- 构造 ---------------------------------------------------------------
    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        granularity: str = "per_channel",
        axis: int = 0,
        group_size: int = 128,
        scheme: str = "symmetric",
        num_bits: int = 8,
        compute_dtype: torch.dtype = torch.float16,
    ) -> "W8A16Linear":
        mod = cls(
            linear.in_features,
            linear.out_features,
            bias=linear.bias is not None,
            granularity=granularity,
            axis=axis,
            group_size=group_size,
            scheme=scheme,
            num_bits=num_bits,
            compute_dtype=compute_dtype,
        )
        with torch.no_grad():
            # 权重先上 fp32 再算 qparams。原权重是 fp16 时这一步是必须的。
            w = linear.weight.detach().to(torch.float32)
            wb = mod._blocking.to_blocks(w)
            qp = compute_qparams(wb, scheme=scheme, num_bits=num_bits)
            codes = _quantize_blocks(wb, qp).to(mod.qweight.dtype)
            mod.qweight.copy_(codes)
            mod.scale.copy_(qp.scale)
            mod.zero_point.copy_(qp.zero_point)
            if linear.bias is not None:
                mod.bias.copy_(linear.bias.detach().to(torch.float32))
        mod.to(linear.weight.device)
        return mod

    # --- 前向 ---------------------------------------------------------------
    def dequantized_weight(self, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        deq = (self.qweight.to(torch.float32) - self.zero_point) * self.scale
        w = self._blocking.from_blocks(deq)
        return w.to(dtype if dtype is not None else self.compute_dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dt = self.compute_dtype
        w = self.dequantized_weight(dt)
        b = self.bias.to(dt) if self.bias is not None else None
        return F.linear(x.to(dt), w, b)

    # --- 字节账 -------------------------------------------------------------
    def stored_bytes(self) -> int:
        n = self.qweight.numel() * self.qweight.element_size()
        n += self.scale.numel() * self.scale.element_size()
        if self.scheme == "asymmetric":
            n += self.zero_point.numel() * self.zero_point.element_size()
        if self.bias is not None:
            n += self.bias.numel() * self.bias.element_size()
        return n

    def extra_repr(self) -> str:
        return (
            "in_features={}, out_features={}, granularity={}, group_size={}, "
            "scheme={}, compute_dtype={}".format(
                self.in_features,
                self.out_features,
                self.granularity,
                self.group_size,
                self.scheme,
                str(self.compute_dtype).replace("torch.", ""),
            )
        )


def _iter_linears(model: nn.Module) -> List[Tuple[str, nn.Module, str, nn.Linear]]:
    """返回 (full_name, parent_module, attr_name, linear)。"""
    found: List[Tuple[str, nn.Module, str, nn.Linear]] = []
    for parent_name, parent in model.named_modules():
        for attr, child in list(parent.named_children()):
            if isinstance(child, nn.Linear):
                full = "{}.{}".format(parent_name, attr) if parent_name else attr
                found.append((full, parent, attr, child))
    return found


def quantize_linear_modules(
    model: nn.Module,
    granularity: str = "per_channel",
    axis: int = 0,
    group_size: int = 128,
    scheme: str = "symmetric",
    num_bits: int = 8,
    compute_dtype: torch.dtype = torch.float16,
    skip_name_contains: Sequence[str] = (),
    min_numel: int = 0,
) -> Tuple[nn.Module, List[str]]:
    """就地把 model 里的 nn.Linear 换成 W8A16Linear。返回 (model, 被替换的名字)。

    skip_name_contains: 名字里含任一子串就跳过。常用于跳过 lm_head——
    它和 embedding 绑权重时，量化会同时改掉输入侧的 embedding 表现。
    """
    replaced: List[str] = []
    for full, parent, attr, linear in _iter_linears(model):
        if any(s in full for s in skip_name_contains):
            continue
        if linear.weight.numel() < min_numel:
            continue
        q = W8A16Linear.from_linear(
            linear,
            granularity=granularity,
            axis=axis,
            group_size=group_size,
            scheme=scheme,
            num_bits=num_bits,
            compute_dtype=compute_dtype,
        )
        setattr(parent, attr, q)
        replaced.append(full)
    return model, replaced


def module_weight_bytes(
    model: nn.Module, reference_dtype: torch.dtype = torch.float16
) -> Dict[str, Any]:
    """数「线性层权重」这一项的字节账。只数 Linear / W8A16Linear，不数 embedding。

    reference_dtype 是对照基准：V100 上跑 fp16 训练/推理时权重就是 fp16，
    所以默认拿 fp16 当分母，而不是 fp32——拿 fp32 当分母会把收益吹大一倍。
    """
    ref_elem = torch.finfo(reference_dtype).bits // 8
    stored = 0
    reference = 0
    n_linear = 0
    n_quant = 0
    for _, module in model.named_modules():
        if isinstance(module, W8A16Linear):
            n_quant += 1
            stored += module.stored_bytes()
            reference += module.in_features * module.out_features * ref_elem
            if module.bias is not None:
                reference += module.out_features * module.bias.element_size()
        elif isinstance(module, nn.Linear):
            n_linear += 1
            w_bytes = module.weight.numel() * module.weight.element_size()
            stored += w_bytes
            reference += module.weight.numel() * ref_elem
            if module.bias is not None:
                stored += module.bias.numel() * module.bias.element_size()
                reference += module.bias.numel() * module.bias.element_size()
    return {
        "n_linear_fp": n_linear,
        "n_linear_quantized": n_quant,
        "stored_bytes": stored,
        "reference_bytes": reference,
        "reference_dtype": str(reference_dtype).replace("torch.", ""),
        "weight_bytes_ratio": (stored / reference) if reference else float("nan"),
    }


def weight_bytes_ratio(
    model: nn.Module, reference_dtype: torch.dtype = torch.float16
) -> float:
    return float(module_weight_bytes(model, reference_dtype=reference_dtype)["weight_bytes_ratio"])


def top1_agreement(
    logits_a: torch.Tensor,
    logits_b: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> Dict[str, Any]:
    """逐 token top-1 一致率。这是 Q4 的主指标，比看 loss 值有信息量。

    为什么不看 loss：weight-only 量化后 loss 通常只涨千分之几，
    但 argmax 翻转率可能已经到百分之几——生成任务上后者才是用户看到的东西。
    """
    a = logits_a.detach().to(torch.float32)
    b = logits_b.detach().to(torch.float32)
    if a.shape != b.shape:
        raise ValueError("logits 形状不一致：{} vs {}".format(tuple(a.shape), tuple(b.shape)))
    ta = a.argmax(dim=-1)
    tb = b.argmax(dim=-1)
    same = (ta == tb)
    if mask is not None:
        m = mask.to(torch.bool)
        if m.shape != same.shape:
            raise ValueError(
                "mask 形状 {} 与 token 形状 {} 不一致".format(tuple(m.shape), tuple(same.shape))
            )
        n = int(m.sum())
        agree = float((same & m).sum()) / n if n else float("nan")
    else:
        n = int(same.numel())
        agree = float(same.sum()) / n if n else float("nan")
    return {"n_tokens": n, "top1_agreement": agree, "flip_rate": 1.0 - agree if n else float("nan")}


def logit_divergence(logits_a: torch.Tensor, logits_b: torch.Tensor) -> Dict[str, float]:
    """logits 层面的偏差。和 top1_agreement 一起看：

    max_abs 很小但 flip_rate 不小 => 说明模型本来就在很多位置上「难分伯仲」，
    这时候量化只是压垮骆驼的最后一根稻草，不是量化本身太粗。
    """
    a = logits_a.detach().to(torch.float32)
    b = logits_b.detach().to(torch.float32)
    diff = (a - b).abs()
    la = torch.log_softmax(a, dim=-1)
    lb = torch.log_softmax(b, dim=-1)
    kl = (la.exp() * (la - lb)).sum(dim=-1)
    return {
        "max_abs_logit_diff": float(diff.max()),
        "mean_abs_logit_diff": float(diff.mean()),
        "mean_kl_a_to_b": float(kl.mean()),
        "max_kl_a_to_b": float(kl.max()),
    }
