"""误差账：方法本身必须正确 —— 两条曲线的初始权重相同、数据相同、只有精度不同。

本机的 fp16 是 CPU 软件模拟，数值行为与 V100 的 Tensor Core 不同。
所以这里断言的是**方法的性质**（同精度必须完全重合、异精度必须能算出差值、
scaler 轨迹能被记录），不是任何具体的 max_dloss 数值。
那个数值只能在 V100 上取。
"""

from __future__ import annotations

import copy

import pytest
import torch

from mm_v100 import common as C
from mm_v100 import model as M
from mm_v100 import numeric_ref as NR

CFG = {
    "name": "unit_tiny",
    "model": {"vocab_size": 64, "hidden_size": 32, "num_hidden_layers": 2,
              "num_attention_heads": 4, "num_key_value_heads": 2,
              "intermediate_size": 64, "max_seq_len": 32},
    "train": {"seq_len": 16, "global_batch": 4, "accum": 1, "lr": 5e-4,
              "grad_clip": 1.0, "data_seed": 1234, "num_samples": 128},
}


def test_same_dtype_twice_gives_identical_curves():
    """方法自检：把「低精度」那一路也设成 fp32，两条曲线必须逐位相同。

    这条不成立就说明对照方法本身有问题（初始权重没对齐，或者数据取错了），
    那时候任何 fp16 结论都是噪声。
    """
    report = NR.run(CFG, device="cpu", steps=3, tol=1e-6, seed=7,
                    low_dtype="float32")
    assert report["compare"]["max_dloss"] == 0.0
    assert report["compare"]["first_diverge_step"] is None
    assert report["fp32"]["losses"] == report["low"]["losses"]


def test_fp16_curve_runs_and_produces_a_comparison():
    report = NR.run(CFG, device="cpu", steps=3, tol=1e-6, seed=7,
                    low_dtype="float16")
    cmp = report["compare"]
    assert cmp["steps_compared"] == 3
    assert len(cmp["deltas"]) == 3
    assert cmp["max_dloss"] >= 0.0
    assert report["low"]["scale_trajectory"][0] == C.DEFAULT_INIT_SCALE
    assert report["fp32"]["scale_trajectory"][0] == 1.0  # fp32 不启用 scaler


def test_deepcopy_not_reseed_is_used_for_the_second_model():
    """两条曲线的初始权重必须逐位相同。这里直接验 run_curve 的前提。"""
    a, _ = M.build_model(CFG["model"], device="cpu", seed=7)
    b = copy.deepcopy(a)
    for key in a.state_dict():
        assert torch.equal(a.state_dict()[key], b.state_dict()[key])


def test_first_diverge_step_respects_tolerance():
    ref = {"losses": [1.0, 2.0, 3.0], "grad_norms": [], "scale_trajectory": [1.0],
           "skipped": [False], "skip_steps": 0, "scaler": {}}
    low = {"losses": [1.0, 2.001, 3.5], "grad_norms": [],
           "scale_trajectory": [1.0], "skipped": [False], "skip_steps": 0,
           "scaler": {}}
    loose = NR.compare_curves(ref, low, tol=0.01)
    tight = NR.compare_curves(ref, low, tol=1e-4)
    assert loose["first_diverge_step"] == 3
    assert tight["first_diverge_step"] == 2
    assert loose["max_dloss"] == pytest.approx(0.5)


def test_verdict_flags_monotone_scale_decay():
    cmp = NR.compare_curves(
        {"losses": [1.0, 1.0], "grad_norms": [], "scale_trajectory": [1.0, 1.0],
         "skipped": [], "skip_steps": 0, "scaler": {}},
        {"losses": [1.0, 1.0], "grad_norms": [],
         "scale_trajectory": [65536.0, 32768.0], "skipped": [True, True],
         "skip_steps": 2, "scaler": {}},
        tol=1e-3)
    verdict = NR.verdict(cmp)
    assert verdict["pass"] is False
    assert any("故障 B" in p for p in verdict["problems"])


def test_verdict_flags_nonfinite_reference():
    cmp = NR.compare_curves(
        {"losses": [float("nan")], "grad_norms": [], "scale_trajectory": [1.0],
         "skipped": [], "skip_steps": 0, "scaler": {}},
        {"losses": [1.0], "grad_norms": [], "scale_trajectory": [1.0],
         "skipped": [], "skip_steps": 0, "scaler": {}},
        tol=1e-3)
    verdict = NR.verdict(cmp)
    assert verdict["pass"] is False
    assert any("fp32" in p for p in verdict["problems"])


def test_skip_ratio_threshold():
    cmp = NR.compare_curves(
        {"losses": [1.0] * 4, "grad_norms": [], "scale_trajectory": [1.0],
         "skipped": [], "skip_steps": 0, "scaler": {}},
        {"losses": [1.0] * 4, "grad_norms": [],
         "scale_trajectory": [65536.0] * 4, "skipped": [True] * 3 + [False],
         "skip_steps": 3, "scaler": {}},
        tol=1e-3)
    assert cmp["skip_ratio"] == pytest.approx(0.75)
    assert NR.verdict(cmp)["pass"] is False


def test_bfloat16_low_path_is_rejected():
    with pytest.raises(RuntimeError):
        NR.run(CFG, device="cpu", steps=1, low_dtype="bfloat16")


def test_render_text_lists_every_step():
    report = NR.run(CFG, device="cpu", steps=2, seed=3, low_dtype="float32")
    text = NR.render_text(report)
    assert "max_dloss" in text
    assert "first_diverge_step" in text
    assert text.count("\n") >= 5
