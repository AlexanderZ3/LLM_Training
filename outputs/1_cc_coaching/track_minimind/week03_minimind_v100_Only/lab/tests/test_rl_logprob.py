"""R1 的单测：logprob / ratio 的精度差异与四类归因。

本机可得的结论（都固定在断言里）：
- 归因顺序必须是「先排除 mask，再谈精度」；
- fp16 的减法能把 log_ratio 直接压成 0——**ratio 恒等于 1，学习信号消失**，
  而这时 max|ratio-1| = 0 看起来「非常稳定」，是最危险的一种假象；
- fp32 logits + fp32 log_softmax 的路径在 CPU 上逐位可复现。

本机**不能**得出的：V100 上误差的真实量级。CPU 的 fp16 是软件模拟，
且真实模型的 logits 分布与 TinyCausalLM 完全不同。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from mm_rl import logprob as LP  # noqa: E402
from mm_rl.toy import TinyCausalLM, ToyConfig, make_batch  # noqa: E402


@pytest.fixture(scope="module")
def fixture_logits():
    model = TinyCausalLM(ToyConfig(), seed=1234).eval()
    b = make_batch(4, 16, 256, seed=5)
    with torch.no_grad():
        logits = model(b["input_ids"]).to(torch.float32)
    return model, b, logits


def test_token_logprobs_shape_and_values_are_negative(fixture_logits):
    _model, b, logits = fixture_logits
    lp = LP.token_logprobs(logits, b["labels"], compute_dtype=torch.float32)
    assert lp.shape == b["labels"].shape
    assert float(lp.max()) <= 0.0
    # gather 出来的必须等于手工索引
    manual = torch.log_softmax(logits, dim=-1)
    expected = manual[0, 0, int(b["labels"][0, 0])]
    assert float(lp[0, 0]) == pytest.approx(float(expected), rel=1e-6)


def test_dual_precision_paths_differ_but_stay_within_the_fp16_bound(fixture_logits):
    _model, b, logits = fixture_logits
    d = LP.dual_precision_logprobs(logits, b["labels"])
    assert set(d) == {"fp32", "fp16_compute", "fp16_stored"}
    for name in ("fp16_compute", "fp16_stored"):
        v = LP.attribute_divergence(d["fp32"], d[name], b["mask"], b["mask"],
                                    dtype=torch.float16)
        assert v["verdict"] == "PRECISION"
        assert 0.0 < v["max_abs_diff"] <= v["bound"]


def test_attribution_reports_identical_for_the_same_tensor(fixture_logits):
    _model, b, logits = fixture_logits
    lp = LP.token_logprobs(logits, b["labels"], compute_dtype=torch.float32)
    v = LP.attribute_divergence(lp, lp.clone(), b["mask"], b["mask"])
    assert v["verdict"] == "IDENTICAL"
    assert v["max_abs_diff"] == 0.0


def test_mask_mismatch_is_detected_before_precision(fixture_logits):
    """mask 错位必须先被抓到，**即使两条 logprob 完全相同**。

    反过来（先谈精度）会把 mask bug 误判成 fp16 不稳定，
    然后去调 GradScaler，越调越远。
    """
    _model, b, logits = fixture_logits
    lp = LP.token_logprobs(logits, b["labels"], compute_dtype=torch.float32)
    v = LP.attribute_divergence(lp, lp.clone(), b["mask"], ~b["mask"])
    assert v["verdict"] == "MASK_MISMATCH"
    assert v["n_mask_diff"] == int(b["mask"].numel())


def test_path_mismatch_is_flagged_when_the_gap_exceeds_the_bound(fixture_logits):
    _model, b, logits = fixture_logits
    lp = LP.token_logprobs(logits, b["labels"], compute_dtype=torch.float32)
    v = LP.attribute_divergence(lp, lp * 1.05, b["mask"], b["mask"], dtype=torch.float16)
    assert v["verdict"] == "PATH_MISMATCH"
    assert v["ratio_to_bound"] > 1.0


def test_attribution_handles_shape_mismatch_without_crashing():
    v = LP.attribute_divergence(torch.zeros(2, 3), torch.zeros(2, 4))
    assert v["verdict"] == "PATH_MISMATCH"
    assert "形状不同" in v["reason"]


def test_attribution_rejects_an_all_false_mask():
    lp = torch.randn(2, 3)
    m = torch.zeros(2, 3, dtype=torch.bool)
    with pytest.raises(ValueError):
        LP.attribute_divergence(lp, lp.clone(), m, m)


def test_precision_bound_scales_with_dtype_and_magnitude():
    small = torch.full((4,), -1.0)
    large = torch.full((4,), -20.0)
    assert LP.precision_bound(large) > LP.precision_bound(small)
    assert LP.precision_bound(small, dtype=torch.float16) > LP.precision_bound(
        small, dtype=torch.float32
    )


def test_fp16_subtraction_can_collapse_the_ratio_to_exactly_one(fixture_logits):
    """**R1 的核心发现**：在 fp16 里做 logp - logp_old，差可能被舍成 0。

    此时 ratio 恒等于 1、max|r-1| = 0、frac_clipped = 0——所有指标都
    「非常健康」，而策略梯度的比值项已经没有任何信息了。
    fp32 下同一批数据的 max|r-1| 是 1e-3 量级。
    """
    _model, b, logits = fixture_logits
    d = LP.dual_precision_logprobs(logits, b["labels"])
    s32 = LP.ratio_stats(d["fp32"], d["fp16_stored"], mask=b["mask"],
                         compute_dtype=torch.float32)
    s16 = LP.ratio_stats(d["fp32"], d["fp16_stored"], mask=b["mask"],
                         compute_dtype=torch.float16)
    assert s32["max_abs_ratio_minus_1"] > 0.0
    assert s16["max_abs_ratio_minus_1"] < s32["max_abs_ratio_minus_1"]
    assert s16["compute_eps"] > s32["compute_eps"]
    assert s32["n_nonfinite"] == 0.0 and s16["n_nonfinite"] == 0.0


def test_ratio_stats_counts_clipped_tokens():
    old = torch.zeros(4, 8)
    new = torch.zeros(4, 8)
    new[0, :] = 1.0  # exp(1) ≈ 2.72，远超 1 + 0.2
    st = LP.ratio_stats(new, old, clip_eps=0.2)
    assert st["frac_clipped"] == pytest.approx(0.25, rel=1e-6)
    assert st["max_abs_ratio_minus_1"] > 1.7


def test_ratio_stats_rejects_an_empty_mask():
    lp = torch.zeros(2, 3)
    with pytest.raises(ValueError):
        LP.ratio_stats(lp, lp, mask=torch.zeros(2, 3, dtype=torch.bool))


def test_determinism_check_catches_a_nondeterministic_function(fixture_logits):
    """归因前必须先证明「同一条路径自己是确定的」。"""
    model, b, _logits = fixture_logits
    det = LP.determinism_check(lambda: model(b["input_ids"]))
    assert det["bitwise_identical"] is True
    assert det["max_abs_diff_across_runs"] == 0.0

    nondet = LP.determinism_check(lambda: torch.randn(4, 4))
    assert nondet["bitwise_identical"] is False
    assert nondet["max_abs_diff_across_runs"] > 0.0
