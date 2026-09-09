"""R2 的单测：三个 KL 估计量在不同 dtype 下的行为。

被固定住的数学事实：
- k3 = expm1(log_r) - log_r **恒 >= 0**，对 fp64 / fp32 / fp16 都成立；
  出现负值就是实现写错了（多半是用了 exp(x)-1 而不是 expm1）。
- k2 = 0.5 * log_r^2 恒 >= 0。
- k1 = -log_r 可以为负，且在对称的 log_r 分布下负值比例约 0.5。
- log_r 在 fp16 里做减法会丢精度，k2/k3 是二次量，误差被放大。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from mm_rl import kl as KL  # noqa: E402


def _log_ratios(n: int = 200_000, sigma: float = 0.05, seed: int = 0) -> torch.Tensor:
    g = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randn(n, generator=g) * sigma


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32, torch.float16])
@pytest.mark.parametrize("sigma", [1e-6, 1e-3, 0.05, 1.0])
def test_k3_is_never_negative(dtype, sigma):
    """e^x - 1 >= x 对所有实数成立；用 expm1 之后浮点也保持这一点。"""
    lr = _log_ratios(sigma=sigma).to(dtype)
    k3 = KL.kl_k3(lr)
    assert bool((k3 >= 0).all()), "k3 出现负值：{}".format(float(k3.min()))


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32, torch.float16])
def test_k2_is_never_negative(dtype):
    lr = _log_ratios().to(dtype)
    assert bool((KL.kl_k2(lr) >= 0).all())


def test_expm1_is_required_naive_exp_minus_one_would_go_negative():
    """对照：exp(x)-1 在 |x| 极小时会失去全部有效数字。

    这条测试证明 kl_k3 里选 expm1 不是风格问题。
    """
    lr = (torch.randn(100_000) * 1e-5).to(torch.float32)
    naive = (torch.exp(lr) - 1.0) - lr
    assert bool((naive < 0).any()), "exp(x)-1 的实现应当能造出负的 k3"
    assert bool((KL.kl_k3(lr) >= 0).all())


def test_k1_is_symmetric_around_zero_and_takes_negative_values():
    lr = _log_ratios()
    st = KL.estimator_stats(KL.kl_k1(lr))
    assert 0.45 < st["frac_negative"] < 0.55
    assert abs(st["mean"]) < 0.01


def test_k3_and_k2_agree_to_second_order_for_small_log_ratios():
    """|log_r| 很小时 k3 ≈ k2 = log_r^2/2，差距就是三阶项 x^3/6。

    断言直接对着解析的三阶界写，而不是拍一个魔法常数——
    这样它在换 sigma、换样本数之后依然成立。
    """
    lr = _log_ratios(sigma=1e-3).to(torch.float64)
    gap = float((KL.kl_k3(lr) - KL.kl_k2(lr)).abs().max())
    third_order_bound = float(lr.abs().max()) ** 3 / 6.0
    assert gap <= third_order_bound * 1.5
    assert gap > 0.0

    lr_big = _log_ratios(sigma=1.0).to(torch.float64)
    assert float((KL.kl_k3(lr_big) - KL.kl_k2(lr_big)).abs().max()) > 1e-2


def test_log_ratio_dtype_is_honoured():
    a = torch.randn(16)
    b = torch.randn(16)
    assert KL.log_ratio(a, b, compute_dtype=torch.float16).dtype is torch.float16
    assert KL.log_ratio(a, b, compute_dtype=torch.float64).dtype is torch.float64


def test_estimator_stats_rejects_empty_selection():
    with pytest.raises(ValueError):
        KL.estimator_stats(torch.randn(4), mask=torch.zeros(4, dtype=torch.bool))


def test_precision_report_shows_fp32_is_essentially_exact_and_fp16_is_not():
    """fp32 相对 fp64 的偏差可以忽略；fp16 的偏差要大好几个数量级。

    这是「advantage / KL 这类 O(N) 规约必须在 fp32 里做」这条工程结论的依据。
    """
    g = torch.Generator(device="cpu").manual_seed(1)
    lp_new = -(torch.rand(64, 32, generator=g) * 8.0 + 1.0)
    lp_ref = lp_new + torch.randn(64, 32, generator=g) * 0.002

    rep = KL.precision_report(lp_new, lp_ref)
    f32 = rep["by_dtype"]["float32"]
    f16 = rep["by_dtype"]["float16"]
    for est in KL.ESTIMATORS:
        assert f32[est]["rel_bias_vs_reference"] < 1e-4
        assert f16[est]["rel_bias_vs_reference"] > f32[est]["rel_bias_vs_reference"]
    # k1 是一次量，fp16 下的**相对**偏差比二次量更夸张：它的均值本身接近 0
    assert f16["k1"]["rel_bias_vs_reference"] > f16["k3"]["rel_bias_vs_reference"]
    assert f16["k3"]["frac_negative"] == 0.0


def test_precision_report_respects_the_mask():
    g = torch.Generator(device="cpu").manual_seed(2)
    lp_new = -(torch.rand(8, 10, generator=g) * 4.0)
    lp_ref = lp_new + 0.01
    mask = torch.zeros(8, 10, dtype=torch.bool)
    mask[:, 5:] = True
    rep = KL.precision_report(lp_new, lp_ref, mask=mask)
    assert rep["n_tokens"] == 40
    assert rep["reference"]["k1"]["n"] == 40.0


def test_format_precision_report_lists_every_dtype_and_estimator():
    g = torch.Generator(device="cpu").manual_seed(3)
    lp_new = -(torch.rand(4, 8, generator=g) * 4.0)
    text = KL.format_precision_report(KL.precision_report(lp_new, lp_new + 0.01))
    for token in ("float64", "float32", "float16", "k1", "k2", "k3"):
        assert token in text
