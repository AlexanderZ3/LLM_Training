"""Q5 的单测：自写 STE 与 torch.ao / aten fake-quant 的对照。

本机 CPU 能验的：
- 自写 STE 的**前向与反向**与 aten 的 fake_quantize_per_channel_affine 逐位相同；
- ao 的 FakeQuantize（PerChannelMinMaxObserver, [-127,127]）与之逐位相同；
- x/s 与 x*(1/s) 的差异确实存在（这是把「自写 vs 框架」这条对照做成
  精确等式的前提，写错了对照就变噪声）；
- scale 强制 fp32 的保护起作用。

本机**不能**验的（下面有 skipif）：
- 这些 kernel 在 CUDA 上的行为与数值；
- AMP（autocast float16 + GradScaler）下的真实溢出行为——
  torch 2.1 的 GradScaler 只有 CUDA 实现，CPU 上只能 enabled=False。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from mm_quant import qat  # noqa: E402

CUDA = torch.cuda.is_available()
NO_CUDA_REASON = (
    "本机没有 GPU；fake-quant 的 CUDA kernel（FakeQuantizeCore.cu / FusedObsFakeQuant.cu）"
    "与 AMP 的真实行为必须在公司 8xV100 上验证"
)


def _weight(rows: int = 64, cols: int = 512, seed: int = 0, device: str = "cpu") -> torch.Tensor:
    g = torch.Generator(device="cpu").manual_seed(seed)
    w = torch.randn((rows, cols), generator=g) * torch.logspace(0.0, 2.0, rows).reshape(-1, 1)
    idx = torch.randint(0, cols, (rows, 4), generator=g)
    w.scatter_(1, idx, torch.randn((rows, 4), generator=g) * 20.0)
    return w.to(device)


def test_weight_qparams_are_fp32_and_per_output_channel():
    w = _weight(rows=16, cols=64).to(torch.float16)
    scale, zp, qmin, qmax = qat.weight_qparams_fp32(w, axis=0)
    assert scale.dtype is torch.float32
    assert zp.dtype is torch.float32
    assert scale.shape == (16,)
    assert (qmin, qmax) == (-127, 127)
    assert float(scale.min()) > 0.0


def test_ste_forward_and_backward_match_aten_bitwise():
    """自写 STE 必须与 aten 逐位相同。这是「两条路径对照」有意义的前提。"""
    w = _weight()
    rep = qat.compare_backends(w, axis=0)
    assert rep["forward_exact_equal_vs_aten"] is True
    assert rep["backward_exact_equal_vs_aten"] is True
    assert rep["n_mismatch_vs_aten"] == 0


def test_ao_fake_quantize_matches_our_qparams_bitwise():
    """ao 的 FakeQuantize + PerChannelMinMaxObserver + [-127,127] 与我们完全一致。

    如果换成 quant_min=-128，ao 的对称 scale 变成 amax/127.5，就对不上了。
    这条测试固定住的是「约定要一致」这件事，不是「实现一样」。
    """
    rep = qat.compare_backends(_weight(), axis=0)
    assert rep["ao_available"] is True
    assert rep["ao_scale_max_abs_diff"] == 0.0
    assert rep["forward_exact_equal_vs_ao"] is True


def test_fused_observer_is_close_but_not_bitwise_identical():
    """FusedMovingAvgObsFakeQuantize 的 qparams 路径不同，差几个码是**正常**的。

    把它当成「必须逐位相等」去断言，只会在 V100 上莫名其妙地红。
    正确的判据是差值不超过几个 scale。
    """
    rep = qat.compare_backends(_weight(), axis=0)
    if not rep.get("ao_fused_available", False):
        pytest.skip("FusedMovingAvgObsFakeQuantize 不可用：{}".format(rep.get("ao_fused_note")))
    assert rep["max_abs_forward_diff_vs_ao_fused"] <= 3.0 * rep["max_scale"]


def test_div_vs_reciprocal_differ_on_round_half_boundaries():
    """inv_scale=False（x/s）与 aten（x*(1/s)）会差极少数几个码。

    差的元素比例极小（这里 32768 个里差 1 个），但**不是 0**。
    这就是为什么 inv_scale 默认是 True：否则「自写 vs 框架」的对照会被
    舍入顺序污染，看起来像量化方法的差异。
    """
    rep = qat.compare_backends(_weight(), axis=0, inv_scale=False)
    assert rep["forward_exact_equal_vs_aten"] is False
    assert 0 < rep["n_mismatch_vs_aten"] <= max(8, rep["numel"] // 1000)
    assert rep["max_abs_forward_diff_vs_aten"] <= rep["max_scale"] * 1.001


def test_clipped_ste_zeroes_gradients_outside_the_range():
    w = _weight(rows=8, cols=256, seed=2).requires_grad_(True)
    scale, zp, qmin, qmax = qat.weight_qparams_fp32(w, axis=0)
    tight = scale * 0.4  # 人为收紧 scale，逼出 clip
    y = qat.fake_quant_ste(w, tight, zp, axis=0, qmin=qmin, qmax=qmax, ste_mode="clipped")
    y.backward(torch.ones_like(y))
    zero_frac = float((w.grad == 0).to(torch.float32).mean())
    assert zero_frac > 0.1, "收紧 scale 之后应当有相当比例的元素被 clip 掉梯度"

    w2 = _weight(rows=8, cols=256, seed=2).requires_grad_(True)
    y2 = qat.fake_quant_ste(w2, tight, zp, axis=0, qmin=qmin, qmax=qmax, ste_mode="passthrough")
    y2.backward(torch.ones_like(y2))
    assert float((w2.grad == 0).to(torch.float32).mean()) == 0.0
    assert torch.equal(w2.grad, torch.ones_like(w2))


def test_ste_mode_is_validated():
    w = torch.randn(4, 16, requires_grad=True)
    scale, zp, qmin, qmax = qat.weight_qparams_fp32(w, axis=0)
    with pytest.raises(ValueError):
        qat.fake_quant_ste(w, scale, zp, axis=0, qmin=qmin, qmax=qmax, ste_mode="hard")


def test_fake_quant_linear_backend_none_is_the_identity_control():
    lin = nn.Linear(32, 16)
    x = torch.randn(5, 32)
    plain = qat.FakeQuantLinear(lin, backend="none")
    assert torch.allclose(plain(x), lin(x), atol=1e-6)


def test_fake_quant_linear_unknown_backend_raises():
    with pytest.raises(ValueError):
        qat.FakeQuantLinear(nn.Linear(4, 4), backend="tensorrt")


@pytest.mark.parametrize("backend", ["ste", "ao"])
def test_qat_training_reduces_loss_on_cpu(backend):
    """CPU 上跑几步，证明 fake-quant 之后梯度还能流、loss 还会降。

    这只证明**代码路径连得上**，不证明 V100 上的收敛质量。
    """
    torch.manual_seed(0)
    base = nn.Sequential(nn.Linear(64, 128), nn.SiLU(), nn.Linear(128, 32))
    model = nn.Sequential(
        qat.FakeQuantLinear(base[0], backend=backend),
        nn.SiLU(),
        qat.FakeQuantLinear(base[2], backend=backend),
    )
    torch.manual_seed(1)
    batches = [(torch.randn(32, 64), torch.randn(32, 32)) for _ in range(4)]

    def loss_fn(m, b):
        xb, yb = b
        return ((m(xb) - yb) ** 2).mean()

    out = qat.qat_train_steps(model, batches, loss_fn, lr=5e-3, steps=12, device="cpu")
    assert out["loss_last"] < out["loss_first"]
    assert all(g > 0.0 for g in out["grad_norms"])
    assert out["amp"] is False


def test_compare_backends_accepts_fp16_weights_without_underflowing_scale():
    """权重是 fp16 也要在 fp32 域算 qparams。amax 很小的层是这条保护的目标。"""
    w = (torch.randn(8, 128) * 1e-4).to(torch.float16)
    scale, _zp, _qmin, _qmax = qat.weight_qparams_fp32(w, axis=0)
    assert scale.dtype is torch.float32
    assert float(scale.min()) > 0.0
    rep = qat.compare_backends(w, axis=0)
    assert rep["forward_exact_equal_vs_aten"] is True


@pytest.mark.skipif(not CUDA, reason=NO_CUDA_REASON)
def test_fake_quant_cuda_forward_and_backward():
    """CUDA 上的 fake-quant 前反向。**这条只有在 V100 上跑才有意义。**"""
    w = _weight(device="cuda")
    rep = qat.compare_backends(w, axis=0)
    assert rep["forward_exact_equal_vs_aten"] is True
    assert rep["backward_exact_equal_vs_aten"] is True


@pytest.mark.skipif(not CUDA, reason=NO_CUDA_REASON)
def test_qat_under_amp_float16_on_cuda():
    """autocast(float16) + GradScaler 下的 QAT。V100 上不能用 bfloat16。"""
    from mm_rl.scaler_rules import make_grad_scaler

    torch.manual_seed(0)
    base = nn.Sequential(nn.Linear(128, 256), nn.SiLU(), nn.Linear(256, 64)).cuda()
    model = nn.Sequential(
        qat.FakeQuantLinear(base[0], backend="ste"),
        nn.SiLU(),
        qat.FakeQuantLinear(base[2], backend="ste"),
    ).cuda()
    batches = [(torch.randn(32, 128, device="cuda"), torch.randn(32, 64, device="cuda"))
               for _ in range(4)]

    def loss_fn(m, b):
        xb, yb = b
        return ((m(xb) - yb) ** 2).mean()

    scaler, _api = make_grad_scaler("cuda", enabled=True)
    out = qat.qat_train_steps(
        model, batches, loss_fn, lr=1e-3, steps=8, device="cuda", amp=True, scaler=scaler
    )
    assert out["amp"] is True
    assert all(torch.isfinite(torch.tensor(l)) for l in out["losses"])
