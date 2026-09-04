"""Day 2/5：tiny 配置 3 步 loss 有限；save → resume 后下一 step 的 loss 与连续训练完全相等；float16 带 GradScaler；mask 故障症状。

依赖 MINIMIND_ROOT（模型类与 dataset 类来自 MiniMind）与 datasets 包；缺失则 skip。CPU 即可。
"""
from __future__ import annotations

import json
import math

import pytest

pytest.importorskip("transformers")
pytest.importorskip("datasets")


def _run(minimind_root, tmp_path, *extra):
    from mm_probe.bounded_train import build_parser, run

    argv = ["--config", "tiny_cpu", "--minimind-root", str(minimind_root), "--save-dir", str(tmp_path), "--quiet", *extra]
    return run(build_parser().parse_args(argv))


LOG_KEYS = {"step", "loss", "lr", "grad_norm", "scaler_scale", "tokens_per_s", "peak_mem_mb"}


def test_sft_three_steps_finite(minimind_root, tmp_path):
    s = _run(minimind_root, tmp_path, "--stage", "sft", "--max-steps", "3")
    rows = s["rows"]
    assert len(rows) == 3 and [r["step"] for r in rows] == [1, 2, 3]
    assert all(math.isfinite(r["loss"]) for r in rows)
    assert all(LOG_KEYS <= set(r) for r in rows)
    assert all(r["n_label_tokens"] > 0 for r in rows)
    lines = [json.loads(l) for l in open(s["log_path"], encoding="utf-8") if l.strip()]
    assert len(lines) == 3 and lines[-1]["step"] == 3
    assert rows[0]["peak_mem_mb"] is None or rows[0]["peak_mem_mb"] >= 0


def test_pretrain_overfit_float16_scaler(minimind_root, tmp_path):
    s = _run(minimind_root, tmp_path, "--stage", "pretrain", "--max-steps", "3", "--overfit-n", "2", "--dtype", "float16")
    rows = s["rows"]
    assert all(math.isfinite(r["loss"]) for r in rows)
    assert all(r["scaler_scale"] is not None and r["scaler_scale"] > 0 for r in rows), "float16 必须带 GradScaler"
    idx = [tuple(r["sample_indices"]) for r in rows]
    assert all(set(i) <= {0, 1} for i in idx), "--overfit-n 2 只能用前 2 条样本"


def test_save_resume_equivalence(minimind_root, tmp_path):
    a_dir, b_dir = tmp_path / "a", tmp_path / "b"
    a = _run(minimind_root, a_dir, "--stage", "sft", "--max-steps", "4", "--save-every", "2")
    ckpt = a_dir / "ckpt_step2.pt"
    assert ckpt.exists() and (a_dir / "latest.pt").exists()
    b = _run(minimind_root, b_dir, "--stage", "sft", "--max-steps", "4", "--resume", str(ckpt))
    assert [r["step"] for r in b["rows"]] == [3, 4]
    for ra, rb in zip(a["rows"][2:], b["rows"]):
        assert ra["sample_indices"] == rb["sample_indices"], "数据游标必须恢复到同一位置"
        assert ra["loss"] == pytest.approx(rb["loss"], abs=1e-6), f"step {ra['step']}: 连续 {ra['loss']} vs 恢复 {rb['loss']}"
        assert ra["lr"] == pytest.approx(rb["lr"])
    import torch

    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    for key in ("model", "optimizer", "scheduler", "scaler", "rng", "step", "cursor"):
        assert key in ck
    assert ck["step"] == 2 and ck["cursor"]["pos"] >= 0


def test_dpo_stage_starts_at_ln2(minimind_root, tmp_path):
    s = _run(minimind_root, tmp_path, "--stage", "dpo", "--max-steps", "2")
    rows = s["rows"]
    assert rows[0]["loss"] == pytest.approx(math.log(2), abs=1e-4), "ref = policy 副本时第一步 DPO loss = ln2"
    assert rows[0]["reward_margin"] == pytest.approx(0.0, abs=1e-5)
    assert math.isfinite(rows[1]["loss"])


def test_mask_fault_symptoms(minimind_root, tmp_path):
    ok = _run(minimind_root, tmp_path / "ok", "--stage", "sft", "--max-steps", "1")
    bad = _run(minimind_root, tmp_path / "bad", "--stage", "sft", "--max-steps", "1", "--mask-fault", "assistant_all_ignored")
    assert math.isnan(bad["rows"][0]["loss"]), "labels 全 -100 → cross_entropy 均值为 nan（故障签名）"
    assert bad["rows"][0]["n_label_tokens"] == 0
    leak = _run(minimind_root, tmp_path / "leak", "--stage", "sft", "--max-steps", "1", "--mask-fault", "user_in_loss")
    assert leak["rows"][0]["n_label_tokens"] > ok["rows"][0]["n_label_tokens"]
    assert math.isfinite(leak["rows"][0]["loss"])
