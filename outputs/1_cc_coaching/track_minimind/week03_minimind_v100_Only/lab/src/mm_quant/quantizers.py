"""INT8 量化器：对称/非对称 x per-tensor / per-channel / per-group。

这个实验在 V100 上测的是什么
----------------------------
测的是**误差**，不是速度。V100（sm70）只有 FP16 Tensor Core，INT8 的 IMMA 指令
从 Turing（sm75）才有，所以这里量化出来的 int8 权重在 V100 上**不可能**变快——
它只可能变小。因此本模块的全部产出是「误差 vs 存储」这一条曲线，
这条曲线换到 A100/H100 上数值会变，方法完全不变。

不能测什么
----------
- 不能测 INT8 GEMM 吞吐（V100 没有 IMMA）。
- 不能测 torch.ao.quantization 的真 INT8 后端在 GPU 上的行为：fbgemm/qnnpack/
  x86/onednn 全部是 CPU 后端，convert_fx(...) 的产物 .cuda() 会失败。
  这一条由 lab/scripts/probe_int8_caps.py 现场记录，不在这里假设。

一个贯穿全模块的设计点
----------------------
**scale 永远在 fp32 域计算。** 输入是 fp16 权重也先 .float() 再求 amax 和做除法。
理由：fp16 的最小正规数是 6.1e-5，amax / 127 这一步除法对 amax 很小的层
会直接下溢成 0；scale=0 之后整块权重全变 0，而 loss 曲线在前几十步看不出来。
这是 Q5（QAT）里唯一一个必须记住的实现细节，所以在最底层的量化器里就固定住，
上层无法绕过。

术语
----
codes    量化后的整数码（int8 存储），本模块内部一律以 [num_blocks, block_numel] 的块视图保存。
block    一个共享同一组 (scale, zero_point) 的元素集合。三种粒度只是三种分块方式。
scheme   'symmetric'：zero_point 恒为 0，码范围 [-127, 127]（放弃 -128 换取 0 精确可表示）。
         'asymmetric'：码范围 [-128, 127]，zero_point 由 min/max 决定。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple

import torch

__all__ = [
    "GRANULARITIES",
    "SCHEMES",
    "Blocking",
    "QParams",
    "QuantizedTensor",
    "compute_qparams",
    "quantize_tensor",
    "quantize_dequantize",
    "relative_error",
    "mse",
    "code_range",
    "scale_storage_bytes",
]

GRANULARITIES: Tuple[str, ...] = ("per_tensor", "per_channel", "per_group")
SCHEMES: Tuple[str, ...] = ("symmetric", "asymmetric")

# scale 的下限。低于这个值说明整块权重都是 0（或被 clip 成 0），直接除会得到 inf。
_MIN_SCALE = 1.0e-8


def code_range(num_bits: int, scheme: str) -> Tuple[int, int]:
    """返回 (qmin, qmax)。

    对称方案故意用 [-(2^(b-1)-1), 2^(b-1)-1]（8 bit 即 [-127, 127]）而不是
    [-128, 127]。代价是浪费一个码位，收益是 scale = amax/127 时正负两侧对称，
    并且实数 0 精确映射到码 0——权重里大量的近零值不会被系统性地推向一侧。
    """
    if num_bits < 2 or num_bits > 8:
        raise ValueError("num_bits 只支持 2..8，收到 {}".format(num_bits))
    if scheme not in SCHEMES:
        raise ValueError("scheme 只支持 {}，收到 {!r}".format(SCHEMES, scheme))
    if scheme == "symmetric":
        hi = 2 ** (num_bits - 1) - 1
        return -hi, hi
    return -(2 ** (num_bits - 1)), 2 ** (num_bits - 1) - 1


class Blocking:
    """把任意形状的张量摊成 [num_blocks, block_numel] 的块视图，并且能摊回去。

    三种粒度共用这一条路径，这样「换粒度」在代码里只是换一个字符串，
    误差对比才是同一套算术下的对比。

    per_tensor   1 个块，包含全部元素。
    per_channel  沿 axis 分块：第 c 块 = 该轴下标为 c 的整个切片。
                 对 nn.Linear 的 weight [out_features, in_features]，axis=0 就是
                 「每个输出通道一组 scale」，这是权重量化的标准做法。
    per_group    先摊成 [rows, last_dim]，再把 last_dim 按 group_size 切段。
                 对 [out, in] 的权重就是「沿输入维每 g 个元素一组」。
                 last_dim 不能整除时右侧补 0；补的 0 不改变 amax（整组为 0 时由
                 _MIN_SCALE 兜底），dequant 时切掉。
    """

    def __init__(
        self,
        shape: Sequence[int],
        granularity: str,
        axis: int = 0,
        group_size: int = 128,
    ) -> None:
        if granularity not in GRANULARITIES:
            raise ValueError(
                "granularity 只支持 {}，收到 {!r}".format(GRANULARITIES, granularity)
            )
        self.shape: Tuple[int, ...] = tuple(int(s) for s in shape)
        if len(self.shape) == 0:
            raise ValueError("Blocking 需要至少 1 维张量，收到标量")
        self.granularity = granularity
        self.ndim = len(self.shape)
        self.numel = 1
        for s in self.shape:
            self.numel *= s
        self.axis = int(axis) % self.ndim
        self.group_size = int(group_size)
        # per_group 专用字段，先给出与其它粒度一致的默认值，便于 describe() 统一处理。
        self.rows = 0
        self.groups_per_row = 0
        self.padded_last = 0
        self.last = self.shape[-1]
        self.pad = 0

        if self.granularity == "per_tensor":
            self.num_blocks = 1
            self.block_numel = self.numel
        elif self.granularity == "per_channel":
            self.num_blocks = self.shape[self.axis]
            self.block_numel = self.numel // self.num_blocks
        else:
            if self.group_size < 1:
                raise ValueError("group_size 必须 >= 1，收到 {}".format(group_size))
            self.rows = self.numel // self.last
            self.groups_per_row = int(math.ceil(self.last / self.group_size))
            self.padded_last = self.groups_per_row * self.group_size
            self.pad = self.padded_last - self.last
            self.num_blocks = self.rows * self.groups_per_row
            self.block_numel = self.group_size

    # --- 摊平 / 还原 ---------------------------------------------------------
    def to_blocks(self, x: torch.Tensor) -> torch.Tensor:
        if tuple(x.shape) != self.shape:
            raise ValueError(
                "形状不符：Blocking 建于 {}，收到 {}".format(self.shape, tuple(x.shape))
            )
        if self.granularity == "per_tensor":
            return x.reshape(1, -1)
        if self.granularity == "per_channel":
            return x.movedim(self.axis, 0).reshape(self.num_blocks, -1)
        flat = x.reshape(self.rows, self.last)
        if self.pad:
            flat = torch.nn.functional.pad(flat, (0, self.pad))
        return flat.reshape(self.num_blocks, self.group_size)

    def from_blocks(self, xb: torch.Tensor) -> torch.Tensor:
        if self.granularity == "per_tensor":
            return xb.reshape(self.shape)
        if self.granularity == "per_channel":
            moved_shape = (self.shape[self.axis],) + tuple(
                s for i, s in enumerate(self.shape) if i != self.axis
            )
            return xb.reshape(moved_shape).movedim(0, self.axis).reshape(self.shape)
        flat = xb.reshape(self.rows, self.padded_last)
        if self.pad:
            flat = flat[:, : self.last]
        return flat.reshape(self.shape)

    def describe(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "granularity": self.granularity,
            "shape": list(self.shape),
            "num_blocks": self.num_blocks,
            "block_numel": self.block_numel,
        }
        if self.granularity == "per_channel":
            d["axis"] = self.axis
        if self.granularity == "per_group":
            d["group_size"] = self.group_size
            d["pad"] = self.pad
        return d


@dataclass(frozen=True)
class QParams:
    """一组量化参数。scale / zero_point 一律 fp32，形状 [num_blocks, 1]。

    zero_point 用 fp32 存整数值而不是 int32，是为了让
    round(x / scale) + zero_point 全程留在同一个浮点域里；
    真正落盘时再转 int，本模块的 nbytes 统计按 int32 算。
    """

    scale: torch.Tensor
    zero_point: torch.Tensor
    qmin: int
    qmax: int
    scheme: str
    num_bits: int
    clip_ratio: float

    def nbytes(self) -> int:
        """scale + zero_point 的存储开销。对称方案 zero_point 恒 0，不占空间。"""
        n = self.scale.numel() * 4
        if self.scheme == "asymmetric":
            n += self.zero_point.numel() * 4
        return n


def compute_qparams(
    xb: torch.Tensor,
    scheme: str = "symmetric",
    num_bits: int = 8,
    clip_ratio: float = 1.0,
) -> QParams:
    """按块算 (scale, zero_point)。输入是块视图 [num_blocks, block_numel]。

    clip_ratio（Q2 的 alpha）：先把动态范围乘以 alpha 再算 scale。
    alpha < 1 表示故意截掉尾部——rounding 误差下降、clipping 误差上升，
    两者的和有一个最小值，那个 alpha 就是这一层的最优截断点。

    **这里强制 .float()。** 上层传 fp16 权重进来也一样在 fp32 域求 amax 和做除法。
    """
    if xb.dim() != 2:
        raise ValueError(
            "compute_qparams 需要 2 维块视图，收到 {}".format(tuple(xb.shape))
        )
    if not (0.0 < clip_ratio <= 1.0):
        raise ValueError("clip_ratio 必须落在 (0, 1]，收到 {}".format(clip_ratio))
    x = xb.detach().to(torch.float32)
    qmin, qmax = code_range(num_bits, scheme)

    if scheme == "symmetric":
        amax = x.abs().amax(dim=1, keepdim=True) * clip_ratio
        scale = (amax / float(qmax)).clamp_min(_MIN_SCALE)
        zero_point = torch.zeros_like(scale)
    else:
        lo = x.amin(dim=1, keepdim=True) * clip_ratio
        hi = x.amax(dim=1, keepdim=True) * clip_ratio
        # 把 0 纳入范围：否则 pad 出来的 0、以及 ReLU 后的 0 会被映射到一个非零码，
        # 稀疏结构在量化后就消失了。
        lo = torch.minimum(lo, torch.zeros_like(lo))
        hi = torch.maximum(hi, torch.zeros_like(hi))
        scale = ((hi - lo) / float(qmax - qmin)).clamp_min(_MIN_SCALE)
        zero_point = torch.round(float(qmin) - lo / scale).clamp_(
            float(qmin), float(qmax)
        )

    return QParams(
        scale=scale,
        zero_point=zero_point,
        qmin=qmin,
        qmax=qmax,
        scheme=scheme,
        num_bits=num_bits,
        clip_ratio=float(clip_ratio),
    )


def _quantize_blocks(xb: torch.Tensor, qp: QParams) -> torch.Tensor:
    x = xb.detach().to(torch.float32)
    codes = torch.round(x / qp.scale) + qp.zero_point
    return codes.clamp_(float(qp.qmin), float(qp.qmax))


def _dequantize_blocks(codes: torch.Tensor, qp: QParams) -> torch.Tensor:
    return (codes.to(torch.float32) - qp.zero_point) * qp.scale


def _storage_dtype(qmin: int, qmax: int) -> torch.dtype:
    if qmin >= -128 and qmax <= 127:
        return torch.int8
    return torch.int16


@dataclass
class QuantizedTensor:
    """量化结果。codes 是块视图下的整数张量，形状 [num_blocks, block_numel]。"""

    codes: torch.Tensor
    qparams: QParams
    blocking: Blocking

    def dequantize(self, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        deq = _dequantize_blocks(self.codes, self.qparams)
        return self.blocking.from_blocks(deq).to(dtype)

    def code_bytes(self) -> int:
        """整数码的字节数。含 per_group 的 padding，因为 padding 也要真存。"""
        return self.codes.numel() * self.codes.element_size()

    def payload_bytes(self) -> int:
        """码 + scale(+zp) 的总字节数，也就是真正要存进磁盘/显存的量。"""
        return self.code_bytes() + self.qparams.nbytes()

    def bytes_report(self, reference_dtype: torch.dtype = torch.float16) -> Dict[str, Any]:
        ref = self.blocking.numel * (torch.finfo(reference_dtype).bits // 8)
        payload = self.payload_bytes()
        return {
            "reference_dtype": str(reference_dtype).replace("torch.", ""),
            "reference_bytes": ref,
            "code_bytes": self.code_bytes(),
            "scale_bytes": self.qparams.nbytes(),
            "payload_bytes": payload,
            "ratio_vs_reference": payload / ref if ref else float("nan"),
            "scale_share": (self.qparams.nbytes() / payload) if payload else float("nan"),
        }


def quantize_tensor(
    x: torch.Tensor,
    scheme: str = "symmetric",
    granularity: str = "per_tensor",
    axis: int = 0,
    group_size: int = 128,
    num_bits: int = 8,
    clip_ratio: float = 1.0,
) -> QuantizedTensor:
    """一次完整的量化。返回的 codes 是整数 dtype，可以直接当作存储表示。"""
    blocking = Blocking(x.shape, granularity, axis=axis, group_size=group_size)
    xb = blocking.to_blocks(x.detach())
    qp = compute_qparams(xb, scheme=scheme, num_bits=num_bits, clip_ratio=clip_ratio)
    codes = _quantize_blocks(xb, qp).to(_storage_dtype(qp.qmin, qp.qmax))
    return QuantizedTensor(codes=codes, qparams=qp, blocking=blocking)


def quantize_dequantize(
    x: torch.Tensor,
    scheme: str = "symmetric",
    granularity: str = "per_tensor",
    axis: int = 0,
    group_size: int = 128,
    num_bits: int = 8,
    clip_ratio: float = 1.0,
    out_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """量化-反量化往返（fake quant 的前向）。默认返回 fp32。"""
    qt = quantize_tensor(
        x,
        scheme=scheme,
        granularity=granularity,
        axis=axis,
        group_size=group_size,
        num_bits=num_bits,
        clip_ratio=clip_ratio,
    )
    return qt.dequantize(dtype=out_dtype if out_dtype is not None else torch.float32)


def relative_error(x: torch.Tensor, x_hat: torch.Tensor) -> float:
    """||x - x_hat||_F / ||x||_F。全程 fp32，避免用 fp16 去度量 fp16 的误差。"""
    a = x.detach().to(torch.float32)
    b = x_hat.detach().to(torch.float32)
    denom = torch.linalg.vector_norm(a)
    if float(denom) == 0.0:
        return 0.0
    return float(torch.linalg.vector_norm(a - b) / denom)


def mse(x: torch.Tensor, x_hat: torch.Tensor) -> float:
    a = x.detach().to(torch.float32)
    b = x_hat.detach().to(torch.float32)
    return float(((a - b) ** 2).mean())


def scale_storage_bytes(
    shape: Sequence[int],
    granularity: str,
    axis: int = 0,
    group_size: int = 128,
    scheme: str = "symmetric",
) -> Dict[str, int]:
    """只算存储账，不做实际量化。用于在选粒度之前先估「scale 要多占多少」。"""
    blocking = Blocking(shape, granularity, axis=axis, group_size=group_size)
    per_block = 4 if scheme == "symmetric" else 8
    code_bytes = blocking.num_blocks * blocking.block_numel  # int8 = 1 B/元素
    return {
        "num_blocks": blocking.num_blocks,
        "code_bytes": code_bytes,
        "scale_bytes": blocking.num_blocks * per_block,
        "payload_bytes": code_bytes + blocking.num_blocks * per_block,
    }
