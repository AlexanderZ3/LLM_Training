"""Q3 的单测：block-wise 8-bit Adam。

本机 CPU 就能得出的三条结论，全部固定在下面的断言里：
1. 优化器状态字节比 ≈ 0.25（理论值 (1 + 4/2048) * 2 / 8 = 0.2505）。
2. **per-block 线性量化对 exp_avg_sq 不够**——Adam 会发散。必须用非线性映射。
3. quantize_state=False 时本实现与 torch.optim.AdamW 数值一致（证明 Adam 数学没写错）。

不能在本机得出的：速度。本实现每步都 dequant/requant，在任何设备上都比
fp32 Adam 慢；bitsandbytes 的融合 kernel 才有速度可谈，而那是另一套代码。
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable, List, Tuple

import pytest
import torch
import torch.nn as nn

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from mm_quant import opt8bit as O8  # noqa: E402


def _train(factory: Callable[[List[torch.nn.Parameter]], torch.optim.Optimizer],
           steps: int = 40, d: int = 64, seed: int = 7) -> Tuple[List[float], torch.optim.Optimizer, torch.Tensor]:
    torch.manual_seed(seed)
    lin = nn.Linear(d, d)
    target = torch.randn(4 * d, d)
    inputs = torch.randn(4 * d, d)
    opt = factory(list(lin.parameters()))
    losses: List[float] = []
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        loss = ((lin(inputs) - target) ** 2).mean()
        loss.backward()
        opt.step()
        losses.append(float(loss.detach()))
    return losses, opt, lin.weight.detach().clone()


def _long_tail_state(nblocks: int = 4, block: int = 2048, decades: float = 9.0, seed: int = 0):
    """量级跨 decades 个数量级的 exp_avg_sq：不同块之间差好几个数量级。

    真实训练里这来自「不同层的梯度量级不同」和「embedding 里高频/低频 token
    收到的梯度差几个数量级」。
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    parts = [
        torch.rand(block, generator=g) * (10.0 ** (-decades * i / max(nblocks - 1, 1)))
        for i in range(nblocks)
    ]
    return torch.cat(parts)


@pytest.mark.parametrize("signed,mapping", [(True, "linear"), (False, "linear"), (False, "power")])
def test_blockwise_roundtrip_preserves_shape_and_codes_are_8bit(signed, mapping):
    x = torch.rand(5000) if not signed else torch.randn(5000)
    codes, scales = O8.blockwise_quantize(x, block_size=2048, signed=signed, mapping=mapping)
    assert codes.numel() == x.numel(), "codes 只存 n 个元素，padding 不能进存储"
    assert codes.dtype is (torch.int8 if signed else torch.uint8)
    assert scales.numel() == 3  # ceil(5000/2048)
    back = O8.blockwise_dequantize(
        codes, scales, 2048, tuple(x.shape), signed=signed, mapping=mapping
    )
    assert back.shape == x.shape
    assert torch.isfinite(back).all()


def test_blockwise_quantize_rejects_power_mapping_for_signed_state():
    with pytest.raises(ValueError):
        O8.blockwise_quantize(torch.randn(64), signed=True, mapping="power")


def test_blockwise_beats_global_for_cross_block_dynamic_range():
    """要点 1：per-block 解决的是**跨块**的量级差异。"""
    v = _long_tail_state()
    rep = O8.compare_state_quantization(v, block_size=2048)
    assert rep["dynamic_range_decades"] > 5.0
    assert rep["global_linear"]["zero_frac"] > 0.5, "全局线性量化必须把大量小值压成 0"
    assert rep["block_linear"]["zero_frac"] < 0.05, "按块之后跨块的量级差异应当消失"


def test_power_mapping_beats_block_linear_for_within_block_dynamic_range():
    """要点 2：per-block 还不够，块内也有长尾，需要非线性映射。"""
    v = _long_tail_state()
    rep = O8.compare_state_quantization(v, block_size=2048, power=4.0)
    assert rep["block_power"]["zero_frac"] == 0.0
    assert rep["block_power"]["max_rel_err"] < rep["block_linear"]["max_rel_err"]
    assert rep["block_power"]["max_rel_err"] < 0.5


def test_adam8bit_without_quantization_matches_torch_adamw():
    """quantize_state=False 时必须和 torch.optim.AdamW 数值一致。

    这条测试把「Adam 数学写错」和「量化引入误差」彻底分开：
    它过了，后面 loss 的任何差异就只能是量化造成的。
    """
    ref_losses, _, ref_w = _train(lambda ps: torch.optim.AdamW(ps, lr=1e-2, weight_decay=0.01))
    own_losses, _, own_w = _train(
        lambda ps: O8.Adam8bit(ps, lr=1e-2, weight_decay=0.01, quantize_state=False)
    )
    assert own_w.shape == ref_w.shape
    max_abs = float((ref_w - own_w).abs().max())
    scale = float(ref_w.abs().max())
    assert max_abs / scale < 1e-5, "相对偏差 {:.2e} 太大，Adam 的更新公式与 AdamW 不一致".format(
        max_abs / scale
    )
    assert own_losses[-1] == pytest.approx(ref_losses[-1], rel=1e-4)


def test_adam8bit_power_mapping_converges_close_to_fp32():
    ref_losses, _, _ = _train(lambda ps: torch.optim.AdamW(ps, lr=1e-2, weight_decay=0.01))
    q_losses, _, _ = _train(
        lambda ps: O8.Adam8bit(
            ps, lr=1e-2, weight_decay=0.01, min_quantize_numel=1,
            sq_mapping="power", sq_power=4.0,
        )
    )
    assert q_losses[-1] < q_losses[0] * 0.8, "量化 Adam 必须真的在优化（loss 要明显下降）"
    assert q_losses[-1] == pytest.approx(ref_losses[-1], rel=0.05), (
        "幂律映射下 40 步之后的 loss 应当与 fp32 AdamW 在 5% 以内"
    )


def test_linear_sq_mapping_diverges():
    """**这条测试断言的是一个失败**：per-block 线性映射的 exp_avg_sq 会让 Adam 发散。

    如果哪天它开始通过了，说明有人偷偷改了 exp_avg_sq 的量化方式，
    那么模块 docstring 里的「要点 2」也要跟着改。
    """
    ref_losses, _, _ = _train(lambda ps: torch.optim.AdamW(ps, lr=1e-2, weight_decay=0.01))
    lin_losses, _, _ = _train(
        lambda ps: O8.Adam8bit(
            ps, lr=1e-2, weight_decay=0.01, min_quantize_numel=1, sq_mapping="linear"
        )
    )
    assert lin_losses[-1] > ref_losses[-1] * 100.0, (
        "线性映射的 exp_avg_sq 本应发散；现在没有发散，说明实现变了"
    )


def test_state_bytes_ratio_is_about_one_quarter():
    """(1 B/元素 + 每 2048 元素 4 B scale) x 2 个状态 / (4 B x 2 个 fp32 状态)。"""
    p = torch.nn.Parameter(torch.randn(1024, 256))
    opt = O8.Adam8bit([p], lr=1e-3, block_size=2048, min_quantize_numel=1)
    p.grad = torch.randn_like(p)
    opt.step()
    rep = O8.state_bytes_report(opt)
    assert rep["quantized_numel"] == 1024 * 256
    expected = (1.0 + 4.0 / 2048.0) * 2.0 / 8.0
    assert rep["ratio_vs_fp32"] == pytest.approx(expected, rel=0.01)
    assert 0.24 < rep["ratio_vs_fp32"] < 0.26


def test_state_bytes_ratio_is_one_when_quantization_is_off():
    p = torch.nn.Parameter(torch.randn(512, 128))
    opt = O8.Adam8bit([p], lr=1e-3, quantize_state=False)
    p.grad = torch.randn_like(p)
    opt.step()
    assert O8.state_bytes_report(opt)["ratio_vs_fp32"] == pytest.approx(1.0, rel=1e-9)


def test_small_parameters_are_skipped_by_min_quantize_numel():
    """bias / norm 权重不该被量化：scale 开销占比高、数值又敏感。"""
    small = torch.nn.Parameter(torch.randn(64))
    big = torch.nn.Parameter(torch.randn(4096))
    opt = O8.Adam8bit([small, big], lr=1e-3, min_quantize_numel=4096)
    small.grad = torch.randn_like(small)
    big.grad = torch.randn_like(big)
    opt.step()
    rep = O8.state_bytes_report(opt)
    assert rep["unquantized_numel"] == 64
    assert rep["quantized_numel"] == 4096


def test_adam8bit_rejects_bad_hyperparameters():
    p = torch.nn.Parameter(torch.randn(8))
    for kwargs in (
        {"lr": 0.0},
        {"betas": (1.0, 0.999)},
        {"eps": 0.0},
        {"block_size": 0},
        {"sq_mapping": "log"},
    ):
        with pytest.raises(ValueError):
            O8.Adam8bit([p], **kwargs)


def test_adam8bit_rejects_sparse_gradients():
    p = torch.nn.Parameter(torch.randn(8))
    opt = O8.Adam8bit([p], lr=1e-3)
    idx = torch.tensor([[0, 2]])
    p.grad = torch.sparse_coo_tensor(idx, torch.tensor([1.0, 2.0]), (8,))
    with pytest.raises(RuntimeError):
        opt.step()


def test_bnb_import_is_optional_and_never_raises():
    """bitsandbytes 装不上是**预期结果**，不能让任何东西崩。"""
    mod, note = O8.try_import_bnb()
    assert isinstance(note, str) and note
    if mod is None:
        assert "Error" in note or "No module" in note
    else:
        assert "bitsandbytes" in note
