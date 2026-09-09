"""R4 的单测：组内 advantage 归一化的灾难性抵消。

本机 CPU 就能得出、也被下面断言固定住的三条结论：
1. reward 量级 1、组内差异 1e-3 时，fp16 下**组内排序**已经被打乱一半以上；
2. 组内差异掉到 1e-4 时，绝大多数组的 fp16 std 精确为 0（整组没有梯度信号）；
3. 同样的差异，把 base 挪到 0，fp16 就完全没有问题——
   **问题不在 fp16，在那个大偏置项**。

fp16 在 1.0 附近的 ulp = 2^-10 ≈ 9.77e-4：组内差异只值一个 ulp。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from mm_rl import advantage as ADV  # noqa: E402


def test_group_advantages_are_centred_and_unit_scaled_in_fp32():
    r = ADV.cancellation_case(32, 8, base=0.0, spread=1.0, seed=0)
    a = ADV.group_advantages(r, eps=0.0, dtype=torch.float64)
    assert a.shape == r.shape
    assert float(a.mean(dim=-1).abs().max()) < 1e-10
    assert float(a.std(dim=-1, unbiased=True).sub(1.0).abs().max()) < 1e-10


def test_group_advantages_requires_2d_input():
    with pytest.raises(ValueError):
        ADV.group_advantages(torch.randn(8))


def test_unbiased_std_requires_group_size_at_least_two():
    with pytest.raises(ValueError):
        ADV.group_mean_std(torch.randn(4, 1), unbiased=True)
    st = ADV.group_mean_std(torch.randn(4, 1), unbiased=False)
    assert float(st["std"].abs().max()) == 0.0


def test_zero_variance_group_gives_zero_advantage_not_nan():
    r = torch.ones(4, 8)
    a = ADV.group_advantages(r, eps=1e-4)
    assert torch.isfinite(a).all()
    assert float(a.abs().max()) == 0.0


def test_eps_zero_on_a_zero_variance_group_produces_nan_which_is_why_eps_exists():
    r = torch.ones(4, 8)
    a = ADV.group_advantages(r, eps=0.0)
    assert bool(torch.isnan(a).any()), "eps=0 时 0/0 必然出 NaN，这就是 eps 存在的理由"


def test_fp16_destroys_within_group_ranking_when_the_base_dominates():
    """base=1.0、spread=1e-3：fp16 的排序一致率明显掉下来，fp32 完好。"""
    r = ADV.cancellation_case(64, 8, base=1.0, spread=1e-3, seed=1)
    rep = ADV.advantage_report(r)
    f32 = rep["by_dtype"]["float32"]
    f16 = rep["by_dtype"]["float16"]
    assert f32["rank_match_vs_reference"] > 0.999
    assert f16["rank_match_vs_reference"] < 0.8
    assert f16["max_abs_diff_vs_reference"] > 0.5
    assert f16["n_distinct_advantage"] < f32["n_distinct_advantage"] / 2


def test_fp16_std_collapses_to_zero_when_the_spread_is_a_tenth_of_an_ulp():
    """spread=1e-4（约 0.1 个 ulp）：多数组的 fp16 std 精确为 0，整组不再产生信号。"""
    r = ADV.cancellation_case(64, 8, base=1.0, spread=1e-4, seed=1)
    rep = ADV.advantage_report(r)
    assert rep["by_dtype"]["float32"]["zero_std_group_ratio"] == 0.0
    assert rep["by_dtype"]["float16"]["zero_std_group_ratio"] > 0.5


def test_removing_the_offset_makes_fp16_fine_again():
    """**问题不在 fp16，在 base。** 同样的 spread，base=0 时 fp16 完全够用。"""
    r = ADV.cancellation_case(64, 8, base=0.0, spread=1e-3, seed=1)
    rep = ADV.advantage_report(r)
    f16 = rep["by_dtype"]["float16"]
    assert f16["zero_std_group_ratio"] == 0.0
    assert f16["rank_match_vs_reference"] > 0.99
    assert f16["max_abs_diff_vs_reference"] < 0.5


def test_report_carries_the_scale_information_needed_to_read_it():
    r = ADV.cancellation_case(16, 4, base=2.0, spread=0.01, seed=2)
    rep = ADV.advantage_report(r)
    assert rep["n_groups"] == 16 and rep["group_size"] == 4
    assert rep["reward_mean_abs"] == pytest.approx(2.0, abs=0.05)
    assert rep["reward_within_group_std_mean"] == pytest.approx(0.01, rel=0.5)
    assert rep["by_dtype"]["float16"]["eps_of_dtype"] == pytest.approx(2.0 ** -10, rel=1e-6)


def test_cancellation_case_rejects_nonpositive_spread():
    with pytest.raises(ValueError):
        ADV.cancellation_case(spread=0.0)


def test_cancellation_case_is_reproducible_from_the_seed():
    a = ADV.cancellation_case(8, 4, seed=3)
    b = ADV.cancellation_case(8, 4, seed=3)
    assert torch.equal(a, b)
    assert not torch.equal(a, ADV.cancellation_case(8, 4, seed=4))


def test_format_advantage_report_mentions_every_dtype():
    rep = ADV.advantage_report(ADV.cancellation_case(8, 4, seed=0))
    text = ADV.format_advantage_report(rep)
    assert "float32" in text and "float16" in text
    assert "zero_std" in text and "rank_match" in text


def test_no_nonfinite_advantages_in_any_dtype():
    for base, spread in ((0.0, 1e-3), (1.0, 1e-3), (1.0, 1e-4), (100.0, 1e-2)):
        rep = ADV.advantage_report(ADV.cancellation_case(32, 8, base=base, spread=spread, seed=0))
        for name, e in rep["by_dtype"].items():
            assert e["n_nonfinite"] == 0, "{} 下出现了非有限的 advantage".format(name)
