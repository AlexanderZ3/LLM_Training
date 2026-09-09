"""Q6 的单测：激活 outlier 统计与幅度迁移。

最重要的一条是 test_smoothing_is_an_exact_identity_in_fp32：
幅度迁移在 fp32 下**不改变模型的数学**，只改变误差落在谁头上。
这条不成立就说明 s 的广播维度搞反了——一个纯代码 bug，
而它在带量化的端到端指标里会伪装成「alpha 调不好」。

本机可得的结论：alpha 有一个最优点，且明显优于不迁移（alpha=0）。
本机**不能**得出的：真实模型的 outlier 是不是结构性的。
那要用真实激活统计，只能在 V100 上跑。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from mm_quant import quantizers as Q  # noqa: E402
from mm_quant import smooth  # noqa: E402


def _activations(n_tokens=128, channels=128, outlier_idx=(3, 17, 55), mag=30.0, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn((n_tokens, channels), generator=g)
    x[:, list(outlier_idx)] *= mag
    return x


def test_channel_amax_shape_and_dtype():
    x = _activations().to(torch.float16)
    a = smooth.channel_amax(x)
    assert a.shape == (128,)
    assert a.dtype is torch.float32


def test_outlier_stats_find_the_planted_channels():
    planted = (3, 17, 55)
    st = smooth.activation_outlier_stats(_activations(outlier_idx=planted), sigma_mult=6.0)
    found = set(int(i) for i in st["outlier_channels"].tolist())
    assert set(planted).issubset(found)
    assert st["n_outlier_channels"] < 20, "6 sigma 阈值不该把大半个层都判成 outlier"
    assert st["amax_over_median"] > 5.0
    assert 0.0 < st["outlier_element_ratio"] < 0.2


def test_overlap_rate_is_one_for_structural_outliers_and_low_for_random_ones():
    """结构性 outlier（同一批 channel）重合率为 1；随机 outlier 重合率低。

    这就是「离线幅度迁移能不能成立」的判据本身。
    """
    a = smooth.activation_outlier_stats(_activations(outlier_idx=(3, 17, 55), seed=1))
    b = smooth.activation_outlier_stats(_activations(outlier_idx=(3, 17, 55), seed=2))
    same = smooth.overlap_rate(a["outlier_channels"], b["outlier_channels"])
    assert same["jaccard"] > 0.8

    c = smooth.activation_outlier_stats(_activations(outlier_idx=(7, 40, 99), seed=3))
    diff = smooth.overlap_rate(a["outlier_channels"], c["outlier_channels"])
    assert diff["jaccard"] < same["jaccard"]


def test_overlap_rate_of_empty_sets_is_one_not_nan():
    empty = torch.zeros(0, dtype=torch.long)
    r = smooth.overlap_rate(empty, empty)
    assert r["jaccard"] == 1.0
    assert r["coverage_a_in_b"] == 1.0


def test_smoothing_is_an_exact_identity_in_fp32():
    """(X/s) @ (W*s).T == X @ W.T。维度搞反这条立刻挂。"""
    x = _activations()
    g = torch.Generator(device="cpu").manual_seed(4)
    w = torch.randn((96, 128), generator=g)
    ref = x @ w.t()
    for alpha in (0.0, 0.3, 0.5, 0.8, 1.0):
        s = smooth.smoothing_scales(smooth.channel_amax(x), w.abs().amax(dim=0), alpha=alpha)
        xs, ws = smooth.apply_smoothing(x, w, s)
        assert Q.relative_error(ref, xs @ ws.t()) < 1e-5, "alpha={} 下迁移不再是恒等变换".format(alpha)


def test_smoothing_moves_magnitude_from_activation_to_weight():
    x = _activations()
    g = torch.Generator(device="cpu").manual_seed(5)
    w = torch.randn((96, 128), generator=g)
    a0 = float(smooth.channel_amax(x).max())
    w0 = float(w.abs().max())
    s = smooth.smoothing_scales(smooth.channel_amax(x), w.abs().amax(dim=0), alpha=0.8)
    xs, ws = smooth.apply_smoothing(x, w, s)
    assert float(smooth.channel_amax(xs).max()) < a0
    assert float(ws.abs().max()) > w0


def test_apply_smoothing_rejects_wrong_dimensions():
    x = torch.randn(8, 16)
    w = torch.randn(4, 16)
    with pytest.raises(ValueError):
        smooth.apply_smoothing(x, w, torch.ones(4))     # s 长度对不上 in_features
    with pytest.raises(ValueError):
        smooth.apply_smoothing(x, torch.randn(2, 3, 4), torch.ones(4))
    with pytest.raises(ValueError):
        smooth.apply_smoothing(torch.randn(8, 9), w, torch.ones(16))


def test_smoothing_scales_validate_alpha_and_shapes():
    with pytest.raises(ValueError):
        smooth.smoothing_scales(torch.ones(4), torch.ones(4), alpha=1.5)
    with pytest.raises(ValueError):
        smooth.smoothing_scales(torch.ones(4), torch.ones(5), alpha=0.5)


def test_alpha_sweep_has_an_interior_optimum_better_than_no_smoothing():
    """存在 0 < alpha < 1 让 W8A8 的输出误差明显低于 alpha=0。"""
    x = _activations()
    g = torch.Generator(device="cpu").manual_seed(6)
    w = torch.randn((96, 128), generator=g)
    rows = smooth.alpha_sweep(x, w)

    for r in rows:
        assert r["identity_rel_err"] < 1e-5, "扫描里每个 alpha 的未量化输出都必须与参考一致"

    at0 = next(r for r in rows if r["alpha"] == 0.0)
    best = min(rows, key=lambda r: r["output_rel_err"])
    assert 0.0 < best["alpha"] < 1.0
    assert best["output_rel_err"] < at0["output_rel_err"] * 0.6


def test_alpha_sweep_trades_activation_error_for_weight_error():
    x = _activations()
    g = torch.Generator(device="cpu").manual_seed(7)
    w = torch.randn((96, 128), generator=g)
    rows = smooth.alpha_sweep(x, w, alphas=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0))
    act = [r["act_rel_err"] for r in rows]
    wt = [r["weight_rel_err"] for r in rows]
    assert act == sorted(act, reverse=True), "alpha 增大，激活误差必须单调下降"
    assert wt == sorted(wt), "alpha 增大，权重误差必须单调上升"


def test_format_alpha_sweep_lists_every_alpha():
    x = _activations()
    w = torch.randn(96, 128)
    rows = smooth.alpha_sweep(x, w, alphas=(0.0, 0.5, 1.0))
    text = smooth.format_alpha_sweep(rows, "t")
    assert text.count("\n") >= 5
    for a in ("0.00", "0.50", "1.00"):
        assert a in text
