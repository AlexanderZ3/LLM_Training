"""Q1/Q2 的单测：误差表、outlier 指标、截断阈值扫描与 MSE 分解。

一条本机就能得出的结论（被下面的测试固定住）：
**8 bit 下截断几乎不划算，4 bit 下才划算。**
Gaussian 权重在 8 bit 时 rounding 误差已经很小，为了压它而多付的 clipping 误差
更大；到 4 bit 时 rounding 误差涨了 16 倍，截断就开始赚了。
这条结论与 GPU 无关，A100/H100 上一样。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from mm_quant import error_budget as EB  # noqa: E402


def _weights(seed: int = 0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    w1 = torch.randn((32, 512), generator=g) * torch.logspace(0.0, 2.0, 32).reshape(-1, 1)
    idx = torch.randint(0, 512, (32, 4), generator=g)
    w1.scatter_(1, idx, torch.randn((32, 4), generator=g) * 20.0)
    w2 = torch.randn((48, 256), generator=g)
    return [("q_proj.weight", w1), ("gate_proj.weight", w2)]


def test_outlier_ratio_is_at_least_one_and_grows_with_planted_outliers():
    g = torch.Generator(device="cpu").manual_seed(1)
    clean = torch.randn((16, 1024), generator=g)
    dirty = clean.clone()
    dirty[0, 0] = float(clean.abs().max()) * 50.0
    assert EB.outlier_ratio(clean) >= 1.0
    assert EB.outlier_ratio(dirty) > EB.outlier_ratio(clean) * 10.0


def test_outlier_ratio_of_all_zero_is_inf_not_nan():
    assert EB.outlier_ratio(torch.zeros(4, 4)) == float("inf")


def test_channelwise_relative_error_shapes_and_bounds():
    w = torch.randn(8, 64)
    same = EB.channelwise_relative_error(w, w, axis=0)
    assert same["max"] == 0.0
    assert same["n_channels"] == 8.0
    zero = EB.channelwise_relative_error(w, torch.zeros_like(w), axis=0)
    assert zero["min"] == pytest.approx(1.0, rel=1e-6)


def test_error_table_is_monotone_in_granularity_and_carries_the_byte_ledger():
    rows = EB.error_table(_weights())
    assert len(rows) == 2
    for r in rows:
        g = r["granularity"]
        assert g["per_tensor"]["rel_err"] >= g["per_channel"]["rel_err"]
        assert g["per_channel"]["rel_err"] >= g["per_group"]["rel_err"]
        # 粒度越细，scale 占的比例越大：这就是「误差换存储」的那条曲线
        assert (
            g["per_tensor"]["scale_share"]
            < g["per_channel"]["scale_share"]
            < g["per_group"]["scale_share"]
        )
        assert 0.49 < g["per_tensor"]["ratio_vs_reference"] < 0.55
        assert g["per_group"]["ratio_vs_reference"] < 0.55


def test_error_table_text_contains_every_layer_and_granularity():
    rows = EB.error_table(_weights())
    text = EB.format_error_table(rows)
    for name, _ in _weights():
        assert name in text
    for g in ("per_tensor", "per_channel", "per_group"):
        assert g in text


def test_clipping_sweep_decomposition_adds_up():
    """mse_total ≈ mse_clip + mse_round：交叉项必须比总量小几个数量级。

    两项支撑集几乎不相交（被截断的元素落到码的端点上，rounding 误差近 0），
    所以交叉项不是 0 但很小。它大到和主项同量级时，说明分解写错了。
    """
    g = torch.Generator(device="cpu").manual_seed(2)
    w = torch.randn((64, 1024), generator=g)
    sweep = EB.clipping_sweep(w, granularity="per_channel", num_bits=4)
    for row in sweep:
        assert abs(row["cross"]) < 1e-3 * max(row["mse_total"], 1e-12)
        assert row["mse_clip"] >= 0.0
        assert row["mse_round"] >= 0.0
    # alpha=1 时没有截断
    a1 = next(r for r in sweep if r["alpha"] == 1.0)
    assert a1["mse_clip"] == 0.0
    assert a1["clipped_frac"] == 0.0


def test_clipping_error_increases_and_rounding_error_decreases_with_smaller_alpha():
    g = torch.Generator(device="cpu").manual_seed(4)
    w = torch.randn((64, 1024), generator=g)
    sweep = EB.clipping_sweep(w, alphas=(1.0, 0.9, 0.8, 0.7, 0.6, 0.5), granularity="per_channel")
    clips = [r["mse_clip"] for r in sweep]
    rounds = [r["mse_round"] for r in sweep]
    assert clips == sorted(clips), "alpha 变小，clipping 误差必须单调上升"
    assert rounds == sorted(rounds, reverse=True), "alpha 变小，rounding 误差必须单调下降"


def test_clipping_pays_off_at_4bit_but_not_at_8bit():
    """本机可得的结论：8 bit 最优 alpha 就是 1.0，4 bit 才需要截断。"""
    g = torch.Generator(device="cpu").manual_seed(5)
    w = torch.randn((64, 1024), generator=g)

    best8 = EB.best_alpha(EB.clipping_sweep(w, granularity="per_channel", num_bits=8))
    assert best8["alpha"] >= 0.9
    assert best8["mse_reduction_vs_alpha1"] < 1.1

    best4 = EB.best_alpha(EB.clipping_sweep(w, granularity="per_channel", num_bits=4))
    assert best4["alpha"] < 1.0
    assert best4["mse_reduction_vs_alpha1"] > 1.2
    assert 0.0 < best4["clipped_frac"] < 0.1


def test_clipping_sweep_rejects_asymmetric_scheme():
    with pytest.raises(ValueError):
        EB.clipping_sweep(torch.randn(8, 64), scheme="asymmetric")


def test_best_alpha_rejects_empty_sweep():
    with pytest.raises(ValueError):
        EB.best_alpha([])
