"""Q5：per-channel 权重 fake-quant + STE，两条路径对照。

这个实验在 V100 上测的是什么
----------------------------
**QAT 在 V100 上是真跑的，这是量化模块里唯一一件在这台机器上完整成立的事。**
理由：torch 2.1 的 aten/src/ATen/native/quantized/cuda/FakeQuantizeCore.cu 与
FusedObsFakeQuant.cu 提供了 FakeQuantize / FusedMovingAvgObsFakeQuantize 的
per-tensor 与 per-channel **前向和反向** CUDA kernel，这些 kernel 是普通
CUDA core 上的逐元素运算，不需要 INT8 Tensor Core。
所以 fake-quant 训练在 sm70 上照常跑，只是训练本身不会因为量化而变快。

不能测什么
----------
- 不能把 QAT 的产物转成真 INT8 模型再在 GPU 上跑：torch.ao.quantization 的
  convert 后端（fbgemm/qnnpack/x86/onednn）全是 CPU 后端，convert_fx(...) 的
  产物 .cuda() 会失败。QAT 在 V100 上的意义是「训出一组对量化鲁棒的权重」，
  部署侧的 INT8 执行要换机器。这一条由 probe_int8_caps.py 现场记录。

两条路径为什么要对照
--------------------
path A（ste）：自己写的 autograd.Function。看得见 STE 是怎么把
               round 的零梯度替换成恒等梯度的，也看得见 clipped 模式下
               超出 [qmin,qmax] 的元素梯度被置 0。
path B（ao）： torch.ao.quantization 的 FakeQuantize / FusedMovingAvgObsFakeQuantize。
               这是目标机上真正会用的实现，有 CUDA kernel。

对照的判据是**前向逐位相等**（同一组 qparams 下），以及**反向梯度逐位相等**。
两条路径对不上时，先怀疑 qmin/qmax 约定：ao 的对称 per-channel observer 用
scale = amax / ((quant_max - quant_min) / 2)，只有把 quant_min/quant_max
设成 -127/127 才等于本仓库 quantizers.code_range 的 amax/127。
设成 -128/127 就变成 amax/127.5，两条路径会差一点点——这不是 bug，是约定不同。

一个真实的坑：x/s 和 x*(1/s) 不是同一件事
------------------------------------------
aten 的 fake_quantize_* kernel 里算的是 `round(x * (1/scale))`，先取倒数再乘。
第一版实现写成了 `round(x / scale)`——数学上更准（少一次舍入），
但在 round 的半整数边界上会和 aten 差一个码。本机实测（64x512 的权重，
逐通道尺度跨 2 个数量级）：32768 个元素里差 1 个，最大差值等于该通道的一个
scale（0.386）。

后果不是「精度差一点」，而是**「自写实现 vs 框架实现」这条对照失效**：
你以为在比两种量化方法，实际上在比两种舍入顺序。
所以本模块默认 inv_scale=True，与 aten 逐位对齐；
想复现这个差异就传 inv_scale=False。

AMP 下的硬规则
--------------
qparams（amax、除法）**强制在 fp32 域算**，实现方式是在 forward 里开一个
autocast(enabled=False) 的区域并把权重 .float()。fp16 的最小正规数是 6.1e-5，
amax/127 对小 amax 的层会下溢成 0，整层权重被量化成 0 而 loss 曲线看不出来。
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .quantizers import code_range

__all__ = [
    "STE_MODES",
    "fake_quant_ste",
    "weight_qparams_fp32",
    "FakeQuantLinear",
    "make_ao_fake_quant",
    "compare_backends",
    "qat_train_steps",
]

STE_MODES: Tuple[str, ...] = ("clipped", "passthrough")


class _FakeQuantPerChannelSTE(torch.autograd.Function):
    """per-channel 对称 fake-quant，反向用 straight-through estimator。

    前向：x_hat = clamp(round(x/s) + zp, qmin, qmax) * s ... (zp=0 时)
    反向：
      clipped      —— 只有量化前落在 [qmin, qmax] 内的元素传梯度，其余置 0。
                       这与 torch 的 fake_quantize_*_cachemask 反向一致。
      passthrough  —— 所有元素都传梯度（教学对照用）。生产里不要用：
                       被 clip 的权重会一直收到梯度往外推，scale 跟着涨，
                       形成正反馈。
    """

    @staticmethod
    def forward(  # type: ignore[override]
        ctx: Any,
        x: torch.Tensor,
        scale: torch.Tensor,
        zero_point: torch.Tensor,
        axis: int,
        qmin: int,
        qmax: int,
        ste_mode: str,
        inv_scale: bool,
    ) -> torch.Tensor:
        shape = [1] * x.dim()
        shape[axis] = -1
        s = scale.reshape(shape).to(torch.float32)
        z = zero_point.reshape(shape).to(torch.float32)
        x32 = x.to(torch.float32)
        # inv_scale=True 时用 x * (1/s)，与 aten 的 fake_quantize_* kernel 逐位一致；
        # False 时用 x / s，数学上更准（少一次舍入）但与 aten 会在 round 的
        # 半整数边界上偶发不一致。见模块 docstring「一个真实的坑」。
        scaled = x32 * s.reciprocal() if inv_scale else x32 / s
        codes_raw = torch.round(scaled) + z
        codes = codes_raw.clamp(float(qmin), float(qmax))
        out = (codes - z) * s
        if ste_mode == "clipped":
            mask = (codes_raw >= float(qmin)) & (codes_raw <= float(qmax))
        else:
            mask = torch.ones_like(codes_raw, dtype=torch.bool)
        ctx.save_for_backward(mask)
        return out.to(x.dtype)

    @staticmethod
    def backward(ctx: Any, grad_out: torch.Tensor):  # type: ignore[override]
        (mask,) = ctx.saved_tensors
        grad_x = grad_out * mask.to(grad_out.dtype)
        return grad_x, None, None, None, None, None, None, None


def fake_quant_ste(
    x: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor,
    axis: int = 0,
    qmin: int = -127,
    qmax: int = 127,
    ste_mode: str = "clipped",
    inv_scale: bool = True,
) -> torch.Tensor:
    if ste_mode not in STE_MODES:
        raise ValueError("ste_mode 只支持 {}，收到 {!r}".format(STE_MODES, ste_mode))
    return _FakeQuantPerChannelSTE.apply(
        x, scale, zero_point, axis, qmin, qmax, ste_mode, inv_scale
    )


def weight_qparams_fp32(
    w: torch.Tensor,
    axis: int = 0,
    num_bits: int = 8,
    scheme: str = "symmetric",
) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
    """按输出通道算 (scale, zero_point)，**强制 fp32**。返回 (scale, zp, qmin, qmax)。

    这个函数不接受「用输入的 dtype 算」这种选项。想看 fp16 域算 scale 会怎样，
    用 scripts/run_quant_suite.py --experiments q5 --scale-in-fp16 复现故障，
    而不是把这里改成可配置——默认值必须是对的那个。
    """
    qmin, qmax = code_range(num_bits, scheme)
    w32 = w.detach().to(torch.float32)
    dims = [d for d in range(w32.dim()) if d != (axis % w32.dim())]
    if scheme == "symmetric":
        amax = w32.abs().amax(dim=dims)
        scale = (amax / float(qmax)).clamp_min(1e-8)
        zp = torch.zeros_like(scale)
    else:
        lo = torch.minimum(w32.amin(dim=dims), torch.zeros(1, device=w32.device))
        hi = torch.maximum(w32.amax(dim=dims), torch.zeros(1, device=w32.device))
        scale = ((hi - lo) / float(qmax - qmin)).clamp_min(1e-8)
        zp = torch.round(float(qmin) - lo / scale).clamp_(float(qmin), float(qmax))
    return scale, zp, qmin, qmax


def make_ao_fake_quant(
    num_channels: int,
    axis: int = 0,
    num_bits: int = 8,
    fused: bool = False,
) -> Tuple[Optional[nn.Module], str]:
    """构造 torch.ao.quantization 的 per-channel 对称 fake-quant 模块。

    fused=False -> FakeQuantize + PerChannelMinMaxObserver
    fused=True  -> FusedMovingAvgObsFakeQuantize（V100 上走 FusedObsFakeQuant.cu）

    返回 (module 或 None, 说明)。torch.ao.quantization 在新版 torch 上带
    DeprecationWarning、在极简安装里可能缺失，所以这里 try/except 而不是硬 import：
    目标机是 torch 2.1，这条路径是可用的；本机是 torch 2.14，只是会警告。
    """
    if num_bits != 8:
        return None, "ao 路径只对照 8 bit，收到 num_bits={}".format(num_bits)
    qmin, qmax = code_range(num_bits, "symmetric")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from torch.ao.quantization import (  # type: ignore
                FakeQuantize,
                FusedMovingAvgObsFakeQuantize,
            )
            from torch.ao.quantization.observer import (  # type: ignore
                MovingAveragePerChannelMinMaxObserver,
                PerChannelMinMaxObserver,
            )

            if fused:
                mod: nn.Module = FusedMovingAvgObsFakeQuantize(
                    observer=MovingAveragePerChannelMinMaxObserver,
                    quant_min=qmin,
                    quant_max=qmax,
                    dtype=torch.qint8,
                    qscheme=torch.per_channel_symmetric,
                    ch_axis=axis,
                    averaging_constant=1.0,
                )
                note = "FusedMovingAvgObsFakeQuantize(per_channel_symmetric, [{}, {}])".format(
                    qmin, qmax
                )
            else:
                mod = FakeQuantize(
                    observer=PerChannelMinMaxObserver,
                    quant_min=qmin,
                    quant_max=qmax,
                    dtype=torch.qint8,
                    qscheme=torch.per_channel_symmetric,
                    ch_axis=axis,
                )
                note = "FakeQuantize(PerChannelMinMaxObserver, [{}, {}])".format(qmin, qmax)
        return mod, note
    except Exception as exc:
        return None, "{}: {}".format(type(exc).__name__, exc)


class FakeQuantLinear(nn.Module):
    """带权重 fake-quant 的 Linear。backend 决定用哪条路径。

    backend:
      'ste'       本仓库的 _FakeQuantPerChannelSTE
      'ao'        torch.ao.quantization.FakeQuantize
      'ao_fused'  torch.ao.quantization.FusedMovingAvgObsFakeQuantize
      'none'      不量化（对照组，用来分离「量化误差」和「训练本身的噪声」）

    激活不量化：Q5 只回答「权重按 per-channel 伪量化训练，模型能不能学回来」。
    """

    def __init__(
        self,
        linear: nn.Linear,
        backend: str = "ste",
        axis: int = 0,
        num_bits: int = 8,
        ste_mode: str = "clipped",
        scheme: str = "symmetric",
    ) -> None:
        super().__init__()
        if backend not in ("ste", "ao", "ao_fused", "none"):
            raise ValueError("未知 backend: {!r}".format(backend))
        self.linear = linear
        self.backend = backend
        self.axis = int(axis)
        self.num_bits = int(num_bits)
        self.ste_mode = ste_mode
        self.scheme = scheme
        self.ao_module: Optional[nn.Module] = None
        self.backend_note = "ste"
        if backend in ("ao", "ao_fused"):
            mod, note = make_ao_fake_quant(
                linear.out_features, axis=axis, num_bits=num_bits, fused=(backend == "ao_fused")
            )
            if mod is None:
                raise RuntimeError(
                    "backend={} 不可用：{}。改用 backend='ste'。".format(backend, note)
                )
            self.ao_module = mod
            self.backend_note = note

    def quantized_weight(self) -> torch.Tensor:
        w = self.linear.weight
        # qparams 与 fake-quant 全部在 fp32 域完成：显式关掉 autocast。
        with torch.autocast(device_type=w.device.type, enabled=False):
            w32 = w.to(torch.float32)
            if self.backend == "none":
                return w32
            if self.backend == "ste":
                scale, zp, qmin, qmax = weight_qparams_fp32(
                    w32, axis=self.axis, num_bits=self.num_bits, scheme=self.scheme
                )
                return fake_quant_ste(
                    w32, scale, zp, axis=self.axis, qmin=qmin, qmax=qmax, ste_mode=self.ste_mode
                )
            assert self.ao_module is not None  # backend 已在 __init__ 校验
            return self.ao_module(w32)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        wq = self.quantized_weight()
        b = self.linear.bias
        # 回到调用侧的 dtype：AMP 下 x 是 fp16，wq 是 fp32，
        # F.linear 在 autocast 区域内会把两者都降到 fp16 再算，这正是目标行为。
        return F.linear(x, wq.to(x.dtype) if not torch.is_autocast_enabled() else wq, b)


def compare_backends(
    w: torch.Tensor,
    axis: int = 0,
    num_bits: int = 8,
    grad_out: Optional[torch.Tensor] = None,
    inv_scale: bool = True,
) -> Dict[str, Any]:
    """同一组权重上跑 ste 与 ao 两条路径，比前向与反向。

    inv_scale=True（默认）时，forward_exact_equal_vs_aten 与
    backward_exact_equal_vs_aten 在 CPU 与 CUDA 上都应当为 True。
    传 inv_scale=False 就能复现「x/s vs x*(1/s)」那一个码的差异，
    此时 n_mismatch_vs_aten 是个位数、max_abs_forward_diff 约等于一个 scale。

    与 ao_fused（FusedMovingAvgObsFakeQuantize）**不要求逐位相等**：
    它的 observer 算 qparams 的路径不同，scale 会差千分之几，
    体现在输出上就是几个码步。判据是差值 <= 几个 max_scale，不是等于 0。
    """
    out: Dict[str, Any] = {"axis": axis, "num_bits": num_bits, "inv_scale": inv_scale}
    w32 = w.detach().to(torch.float32)

    w_a = w32.clone().requires_grad_(True)
    scale, zp, qmin, qmax = weight_qparams_fp32(w_a, axis=axis, num_bits=num_bits)
    out["max_scale"] = float(scale.max())
    y_a = fake_quant_ste(
        w_a, scale, zp, axis=axis, qmin=qmin, qmax=qmax,
        ste_mode="clipped", inv_scale=inv_scale,
    )

    # torch 自带的 per-channel fake-quant（有 CUDA kernel，也有反向）作为第三方裁判
    w_t = w32.clone().requires_grad_(True)
    y_t = torch.fake_quantize_per_channel_affine(
        w_t, scale, zp.to(torch.int32), axis, qmin, qmax
    )

    g = grad_out if grad_out is not None else torch.randn_like(w32)
    y_a.backward(g)
    y_t.backward(g)

    out["forward_exact_equal_vs_aten"] = bool(torch.equal(y_a.detach(), y_t.detach()))
    out["max_abs_forward_diff_vs_aten"] = float((y_a.detach() - y_t.detach()).abs().max())
    out["n_mismatch_vs_aten"] = int((y_a.detach() != y_t.detach()).sum())
    out["numel"] = int(w32.numel())
    out["backward_exact_equal_vs_aten"] = bool(torch.equal(w_a.grad, w_t.grad))
    out["max_abs_backward_diff_vs_aten"] = float((w_a.grad - w_t.grad).abs().max())

    mod, note = make_ao_fake_quant(w32.shape[axis], axis=axis, num_bits=num_bits, fused=False)
    out["ao_note"] = note
    if mod is None:
        out["ao_available"] = False
    else:
        out["ao_available"] = True
        mod = mod.to(w32.device)
        y_ao = mod(w32.clone())
        out["forward_exact_equal_vs_ao"] = bool(torch.equal(y_a.detach(), y_ao.detach()))
        out["max_abs_forward_diff_vs_ao"] = float((y_a.detach() - y_ao.detach()).abs().max())
        out["ao_scale_max_abs_diff"] = float((mod.scale.reshape(-1) - scale.reshape(-1)).abs().max())

    fused_mod, fused_note = make_ao_fake_quant(
        w32.shape[axis], axis=axis, num_bits=num_bits, fused=True
    )
    out["ao_fused_note"] = fused_note
    if fused_mod is None:
        out["ao_fused_available"] = False
    else:
        try:
            fused_mod = fused_mod.to(w32.device)
            y_f = fused_mod(w32.clone())
            out["ao_fused_available"] = True
            out["max_abs_forward_diff_vs_ao_fused"] = float(
                (y_a.detach() - y_f.detach()).abs().max()
            )
        except Exception as exc:
            out["ao_fused_available"] = False
            out["ao_fused_note"] = "{}: {}".format(type(exc).__name__, exc)
    return out


def qat_train_steps(
    model: nn.Module,
    batches,
    loss_fn,
    lr: float = 1e-3,
    steps: int = 5,
    device: str = "cpu",
    amp: bool = False,
    scaler=None,
) -> Dict[str, Any]:
    """在 fake-quant 模型上跑固定步数，返回每步 loss 与梯度范数。

    amp=True 时用 autocast(float16)；V100 上这是唯一可用的低精度路径
    （BF16 在 sm70 上直接报错）。CPU 上 autocast(float16) 也能跑，
    但 CPU 的 fp16 是软件模拟，**数值行为与 V100 不同**，
    所以本机跑通只证明「代码路径连得上」，不证明数值稳定性。
    """
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    losses = []
    gnorms = []
    it = iter(batches)
    for _ in range(steps):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(batches)
            batch = next(it)
        opt.zero_grad(set_to_none=True)
        if amp:
            with torch.autocast(device_type=device, dtype=torch.float16):
                loss = loss_fn(model, batch)
        else:
            loss = loss_fn(model, batch)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
        else:
            loss.backward()
        total = 0.0
        for p in model.parameters():
            if p.grad is not None:
                total += float(p.grad.detach().to(torch.float32).pow(2).sum())
        gnorms.append(total ** 0.5)
        if scaler is not None:
            scaler.step(opt)
            scaler.update()
        else:
            opt.step()
        losses.append(float(loss.detach().to(torch.float32)))
    return {
        "steps": steps,
        "losses": losses,
        "grad_norms": gnorms,
        "loss_first": losses[0] if losses else float("nan"),
        "loss_last": losses[-1] if losses else float("nan"),
        "amp": bool(amp),
        "device": device,
    }
