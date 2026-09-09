"""Q3：block-wise 8-bit Adam 优化器状态。自己写，不依赖 bitsandbytes。

这个实验在 V100 上测的是什么
----------------------------
测的是**字节账里最容易被忽略的那一项**。一个 P 参数的模型用 fp32 AdamW：
参数 4P + 梯度 4P + exp_avg 4P + exp_avg_sq 4P = 16P 字节，优化器状态独占一半。
把两个状态各压到 8 bit，优化器那 8P 变成约 2.004P
（1 B/元素 + 每 2048 元素 4 B 的 scale），整机字节账从 16P 掉到约 10P。
这一条在 V100 上是**真的**：它只涉及存储和逐元素运算，不需要 INT8 Tensor Core。

不能测什么
----------
- 不能测速度收益。这里每步都要 dequant -> 更新 -> requant，逐元素开销比 fp32 Adam
  **更大**；V100 上不会更快。省的是显存，不是时间。
- bitsandbytes 的 AdamW8bit 用融合 CUDA kernel，它的速度结论和本实现无关。
  本实现只是把「为什么必须这样量化」摊开给人看。

两个设计要点（本模块的全部内容）
--------------------------------
**要点 1：必须 per-block。**
exp_avg_sq 是梯度平方的滑动平均，跨层跨参数量级横跨好几个数量级。
用一个全局 scale 时 scale = max(v)/255，绝大多数元素 round 到 0，
dequant 回来 sqrt(0)=0，分母只剩 eps，步长炸成 lr/eps。
按 2048 元素一块各自算 scale，跨块的量级差异就被消掉了。

**要点 2：per-block 还不够，exp_avg_sq 还必须用非线性映射。**
这一条是本机 CPU 实测出来的（见 tests/test_quant_opt8bit.py::test_linear_sq_mapping_diverges）：
即使按 2048 一块算 scale，**块内**的 v 仍然跨好几个数量级
（同一层里不同权重收到的梯度量级差异极大）。线性 8-bit 在块内依然把小值压成 0，
一个 30 步的最小二乘问题上 loss 会从 0.002 涨到 3973。
解决办法是把码到值的映射改成幂律：
    量化   q = round(255 * (v / amax)^(1/P))
    反量化 v = (q / 255)^P * amax
P>1 时码空间在小值一侧变密。P=2..6 都能让 Adam 收敛到与 fp32 相当，
默认取 P=4。bitsandbytes 用的是等价思路（dynamic 数据类型：动态指数 + 线性尾数），
这里用幂律是因为它一行就能写清楚、且完全可逆。

exp_avg（一阶矩）是有符号、以 0 为中心的量，线性对称映射就够，不需要幂律。
"""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, Iterable, Optional, Tuple

import torch
from torch.optim import Optimizer

__all__ = [
    "DEFAULT_BLOCK_SIZE",
    "DEFAULT_SQ_POWER",
    "MAPPINGS",
    "blockwise_quantize",
    "blockwise_dequantize",
    "compare_state_quantization",
    "Adam8bit",
    "state_bytes_report",
    "try_import_bnb",
]

DEFAULT_BLOCK_SIZE = 2048
DEFAULT_SQ_POWER = 4.0
MAPPINGS: Tuple[str, ...] = ("linear", "power")
_MIN_SCALE = 1.0e-30


def _block_view(flat: torch.Tensor, block_size: int) -> Tuple[torch.Tensor, int, int]:
    """把一维张量补齐成 [nblocks, block_size]。返回 (视图, nblocks, pad)。

    padding 只用于**求块统计量和做逐元素运算**，不进入存储：
    codes 只保存前 n 个元素。否则一个 64 元素的 bias 会被存成 2048 字节，
    字节账立刻算错——这正是「按块」实现最容易踩的坑。
    """
    n = flat.numel()
    nblocks = int(math.ceil(n / block_size)) if n else 0
    pad = nblocks * block_size - n
    if pad:
        flat = torch.cat([flat, flat.new_zeros(pad)])
    return flat.view(max(nblocks, 1), block_size) if n else flat.view(0, block_size), nblocks, pad


def blockwise_quantize(
    x: torch.Tensor,
    block_size: int = DEFAULT_BLOCK_SIZE,
    signed: bool = True,
    mapping: str = "linear",
    power: float = DEFAULT_SQ_POWER,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """按 block_size 切块量化。返回 (codes[n], scales[nblocks])。

    signed=True   对称线性，码范围 [-127, 127]，int8。用于 exp_avg。
    signed=False  非负，码范围 [0, 255]，uint8。用于 exp_avg_sq。
    mapping       'linear' 或 'power'（幂律，仅对 signed=False 有意义）。

    scales **永远在 fp32 域算**，和 quantizers.py 同一条规矩。
    """
    if mapping not in MAPPINGS:
        raise ValueError("mapping 只支持 {}，收到 {!r}".format(MAPPINGS, mapping))
    if mapping == "power" and signed:
        raise ValueError(
            "幂律映射是为非负的 exp_avg_sq 设计的；有符号的 exp_avg 用 linear。"
        )
    if power <= 0:
        raise ValueError("power 必须为正，收到 {}".format(power))
    flat = x.detach().reshape(-1).to(torch.float32)
    n = flat.numel()
    blocks, nblocks, _pad = _block_view(flat, block_size)
    if n == 0:
        return flat.to(torch.int8 if signed else torch.uint8), flat.new_zeros(0)

    if signed:
        amax = blocks.abs().amax(dim=1, keepdim=True)
        scales = amax.clamp_min(_MIN_SCALE)
        codes = torch.round(blocks / scales * 127.0).clamp_(-127.0, 127.0).to(torch.int8)
    else:
        amax = blocks.clamp_min(0.0).amax(dim=1, keepdim=True)
        scales = amax.clamp_min(_MIN_SCALE)
        u = (blocks.clamp_min(0.0) / scales).clamp_(0.0, 1.0)
        if mapping == "power":
            u = u.pow(1.0 / power)
        codes = torch.round(u * 255.0).clamp_(0.0, 255.0).to(torch.uint8)
    return codes.reshape(-1)[:n].contiguous(), scales.reshape(-1)


def blockwise_dequantize(
    codes: torch.Tensor,
    scales: torch.Tensor,
    block_size: int,
    shape: Tuple[int, ...],
    signed: bool = True,
    mapping: str = "linear",
    power: float = DEFAULT_SQ_POWER,
) -> torch.Tensor:
    """blockwise_quantize 的逆。返回 fp32，形状还原成 shape。"""
    n = codes.numel()
    if n == 0:
        return torch.zeros(shape, dtype=torch.float32, device=codes.device)
    blocks, _nb, _pad = _block_view(codes.to(torch.float32), block_size)
    s = scales.reshape(-1, 1)
    if signed:
        vals = blocks / 127.0 * s
    else:
        u = blocks / 255.0
        if mapping == "power":
            u = u.pow(power)
        vals = u * s
    return vals.reshape(-1)[:n].reshape(shape)


def _roundtrip(
    x: torch.Tensor,
    block_size: int,
    signed: bool,
    mapping: str,
    power: float,
) -> torch.Tensor:
    codes, scales = blockwise_quantize(
        x, block_size=block_size, signed=signed, mapping=mapping, power=power
    )
    return blockwise_dequantize(
        codes, scales, block_size, tuple(x.shape), signed=signed, mapping=mapping, power=power
    )


def compare_state_quantization(
    v: torch.Tensor,
    block_size: int = DEFAULT_BLOCK_SIZE,
    power: float = DEFAULT_SQ_POWER,
) -> Dict[str, Any]:
    """对 exp_avg_sq 比三种方案：全局线性 / 按块线性 / 按块幂律。

    主指标是 zero_frac 与 median_rel_err：
      zero_frac      dequant 之后原本非零的元素里，有多少变成了精确的 0。
                     v=0 => 分母只剩 eps => 那一维的自适应学习率被关掉。
      median_rel_err 中位相对误差。用中位数不用均值：v 的分布是长尾，
                     均值会被少数大值主导，正好掩盖掉小值被压死这件事。
    """
    x = v.detach().reshape(-1).to(torch.float32).clamp_min(0.0)
    nonzero = x > 0
    n_nonzero = int(nonzero.sum())

    amax = float(x.max())
    gscale = max(amax / 255.0, _MIN_SCALE)
    global_linear = torch.round(x / gscale).clamp_(0.0, 255.0) * gscale
    block_linear = _roundtrip(x, block_size, False, "linear", power)
    block_power = _roundtrip(x, block_size, False, "power", power)

    def _stats(a: torch.Tensor) -> Dict[str, float]:
        d = torch.linalg.vector_norm(x)
        rel_fro = float(torch.linalg.vector_norm(x - a) / d) if float(d) > 0 else 0.0
        if n_nonzero:
            rel = ((a - x)[nonzero] / x[nonzero]).abs()
            return {
                "rel_err_frobenius": rel_fro,
                "median_rel_err": float(rel.median()),
                "max_rel_err": float(rel.max()),
                "zero_frac": float((a[nonzero] == 0).to(torch.float32).mean()),
            }
        return {
            "rel_err_frobenius": rel_fro,
            "median_rel_err": 0.0,
            "max_rel_err": 0.0,
            "zero_frac": 0.0,
        }

    nblocks = int(math.ceil(x.numel() / block_size))
    return {
        "numel": int(x.numel()),
        "block_size": block_size,
        "num_blocks": nblocks,
        "power": float(power),
        "dynamic_range_decades": (
            float(torch.log10(x[nonzero].max() / x[nonzero].min()))
            if n_nonzero > 1 and float(x[nonzero].min()) > 0
            else 0.0
        ),
        "global_linear": _stats(global_linear),
        "block_linear": _stats(block_linear),
        "block_power": _stats(block_power),
        "global_scale_bytes": 4,
        "block_scale_bytes": nblocks * 4,
    }


class Adam8bit(Optimizer):
    """AdamW（decoupled weight decay），可选把两个状态存成 block-wise 8-bit。

    quantize_state=False 时逐行对齐 torch.optim.AdamW 的单张量实现，
    用来证明「本实现的 Adam 数学是对的」；打开量化后再看差多少，
    这样量化引入的误差和实现 bug 就分开了。

    sq_mapping='power'（默认）是唯一能收敛的配置；
    sq_mapping='linear' 保留下来专门用于**复现发散**，见模块 docstring 要点 2。

    min_quantize_numel: 元素数少于这个值的参数不量化（bias、norm 权重等）。
    它们的 scale 开销占比高、数值又敏感，量化收益接近 0。
    """

    def __init__(
        self,
        params: Iterable[Any],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        block_size: int = DEFAULT_BLOCK_SIZE,
        quantize_state: bool = True,
        min_quantize_numel: int = 4096,
        sq_mapping: str = "power",
        sq_power: float = DEFAULT_SQ_POWER,
    ) -> None:
        if lr <= 0.0:
            raise ValueError("lr 必须为正，收到 {}".format(lr))
        if not 0.0 <= betas[0] < 1.0 or not 0.0 <= betas[1] < 1.0:
            raise ValueError("betas 必须落在 [0,1)，收到 {}".format(betas))
        if eps <= 0.0:
            raise ValueError("eps 必须为正，收到 {}".format(eps))
        if block_size < 1:
            raise ValueError("block_size 必须 >= 1，收到 {}".format(block_size))
        if sq_mapping not in MAPPINGS:
            raise ValueError("sq_mapping 只支持 {}，收到 {!r}".format(MAPPINGS, sq_mapping))
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            block_size=block_size,
            quantize_state=quantize_state,
            min_quantize_numel=min_quantize_numel,
            sq_mapping=sq_mapping,
            sq_power=sq_power,
        )
        super().__init__(params, defaults)

    # --- 状态读写：量化与否在这两个函数里收口 ---------------------------------
    @staticmethod
    def _read(state: Dict[str, Any], key: str, shape: Tuple[int, ...],
              signed: bool, block_size: int, mapping: str, power: float) -> torch.Tensor:
        dense = state.get(key, None)
        if dense is not None:
            return dense
        return blockwise_dequantize(
            state[key + "_codes"],
            state[key + "_scales"],
            block_size,
            shape,
            signed=signed,
            mapping=mapping,
            power=power,
        )

    @staticmethod
    def _write(state: Dict[str, Any], key: str, value: torch.Tensor, quantize: bool,
               signed: bool, block_size: int, mapping: str, power: float) -> None:
        if not quantize:
            state[key] = value
            state[key + "_codes"] = None
            state[key + "_scales"] = None
            return
        codes, scales = blockwise_quantize(
            value, block_size=block_size, signed=signed, mapping=mapping, power=power
        )
        state[key] = None
        state[key + "_codes"] = codes
        state[key + "_scales"] = scales

    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], Any]] = None) -> Optional[Any]:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            lr = group["lr"]
            eps = group["eps"]
            wd = group["weight_decay"]
            block_size = group["block_size"]
            mapping = group["sq_mapping"]
            power = group["sq_power"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                if p.grad.is_sparse:
                    raise RuntimeError("Adam8bit 不支持稀疏梯度")
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["quantized"] = bool(group["quantize_state"]) and (
                        p.numel() >= int(group["min_quantize_numel"])
                    )
                    zeros = torch.zeros_like(p, dtype=torch.float32)
                    self._write(state, "exp_avg", zeros, state["quantized"], True,
                                block_size, "linear", power)
                    self._write(state, "exp_avg_sq", zeros, state["quantized"], False,
                                block_size, mapping, power)

                state["step"] += 1
                t = state["step"]
                quantize = state["quantized"]
                shape = tuple(p.shape)

                m = self._read(state, "exp_avg", shape, True, block_size, "linear", power)
                v = self._read(state, "exp_avg_sq", shape, False, block_size, mapping, power)
                g32 = p.grad.detach().to(torch.float32)

                # decoupled weight decay：先缩参数，再走 Adam 更新（与 torch.optim.AdamW 同序）
                if wd != 0.0:
                    p.mul_(1.0 - lr * wd)

                m = m.mul(beta1).add_(g32, alpha=1.0 - beta1)
                v = v.mul(beta2).addcmul_(g32, g32, value=1.0 - beta2)

                bc1 = 1.0 - beta1 ** t
                bc2 = 1.0 - beta2 ** t
                denom = (v.sqrt() / math.sqrt(bc2)).add_(eps)
                update = (m / denom).mul_(-(lr / bc1))
                p.add_(update.to(p.dtype))

                self._write(state, "exp_avg", m, quantize, True, block_size, "linear", power)
                self._write(state, "exp_avg_sq", v, quantize, False, block_size, mapping, power)

        return loss


def state_bytes_report(optimizer: Optimizer) -> Dict[str, Any]:
    """数优化器状态到底占了多少字节，并给出相对 fp32 Adam 的比值。

    分子只算真正常驻的张量（codes + scales，或未量化时的 fp32 张量），
    不算 step 计数这种标量。分母是 2 个 fp32 状态 = 8 字节/参数元素。
    理论值：(1 + 4/block_size) * 2 / 8 = 0.2505（block_size=2048）。
    """
    total = 0
    quantized_numel = 0
    plain_numel = 0
    param_numel = 0
    for group in optimizer.param_groups:
        for p in group["params"]:
            st = optimizer.state.get(p, None)
            if not st:
                continue
            param_numel += p.numel()
            if st.get("quantized", False):
                quantized_numel += p.numel()
            else:
                plain_numel += p.numel()
            for key in ("exp_avg", "exp_avg_sq"):
                dense = st.get(key, None)
                if dense is not None:
                    total += dense.numel() * dense.element_size()
                    continue
                for suffix in ("_codes", "_scales"):
                    tensor = st.get(key + suffix, None)
                    if tensor is not None:
                        total += tensor.numel() * tensor.element_size()
    fp32_bytes = param_numel * 8
    return {
        "param_numel": param_numel,
        "quantized_numel": quantized_numel,
        "unquantized_numel": plain_numel,
        "state_bytes": total,
        "fp32_state_bytes": fp32_bytes,
        "ratio_vs_fp32": (total / fp32_bytes) if fp32_bytes else float("nan"),
    }


def try_import_bnb() -> Tuple[Optional[Any], str]:
    """可选对照：bitsandbytes。装不上是**预期结果**，不是失败。

    已核验（2026-09-08）：AdamW8bit 要求算力 >= 6.0，V100（sm70）满足；
    与 torch 2.1 相容的最高版本是 0.45.5（0.46+ 要求 torch >= 2.2）。
    但内网装带编译产物的包本身有风险，所以主线一律用本文件的纯 PyTorch 实现，
    bnb 只在装得上时用来交叉验证「我写的这个是不是同一个东西」。
    """
    try:
        import bitsandbytes as bnb  # type: ignore
    except Exception as exc:  # ImportError / OSError（缺 CUDA 运行库时是后者）
        return None, "{}: {}".format(type(exc).__name__, exc)
    return bnb, "bitsandbytes {}".format(getattr(bnb, "__version__", "unknown"))
