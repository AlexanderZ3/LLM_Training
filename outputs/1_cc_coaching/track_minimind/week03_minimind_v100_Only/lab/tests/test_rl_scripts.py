"""run_rl_suite.py 的入口测试 + TinyCausalLM 的确定性测试。

入口层是开环交付最容易崩的一层：参数解析、路径合同、落盘、退出码。
数值算得再对，入口一崩在 V100 上什么都测不到。

TinyCausalLM 的确定性是 R1 归因链的前提：把差异归给「精度」之前，
必须先证明同一条路径自己是逐位可复现的。
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

_LAB = Path(__file__).resolve().parents[1]
_SRC = _LAB / "src"
_SCRIPTS = _LAB / "scripts"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import run_rl_suite  # noqa: E402

from mm_rl.toy import TinyCausalLM, ToyConfig, make_batch  # noqa: E402


def _run(*args: str, env=None) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(_SCRIPTS / "run_rl_suite.py"), *args],
        capture_output=True, text=True, encoding="utf-8", errors="replace", env=env,
    )


# --- TinyCausalLM ---------------------------------------------------------- #
def test_toy_model_forward_shape_and_determinism():
    m = TinyCausalLM(ToyConfig(), seed=0).eval()
    b = make_batch(3, 9, 256, seed=1)
    with torch.no_grad():
        a1 = m(b["input_ids"])
        a2 = m(b["input_ids"])
    assert a1.shape == (3, 9, 256)
    assert torch.equal(a1, a2), "同一条路径必须逐位可复现，否则 R1 的归因无从谈起"


def test_toy_model_is_causal():
    """改动第 t 个位置的输入，不能影响第 t 之前位置的 logits。"""
    m = TinyCausalLM(ToyConfig(), seed=0).eval()
    ids = torch.randint(0, 256, (1, 12))
    with torch.no_grad():
        base = m(ids)
        changed = ids.clone()
        changed[0, 8] = (int(changed[0, 8]) + 7) % 256
        after = m(changed)
    assert torch.equal(base[:, :8], after[:, :8])
    assert not torch.equal(base[:, 8:], after[:, 8:])


def test_toy_model_ties_embeddings_and_counts_parameters_once():
    m = TinyCausalLM(ToyConfig(tie_embeddings=True), seed=0)
    assert m.lm_head.weight is m.embed_tokens.weight
    naive = sum(p.numel() for p in m.parameters())
    assert m.num_parameters() <= naive


def test_toy_config_validates_head_divisibility():
    with pytest.raises(ValueError):
        ToyConfig(hidden_size=65, num_heads=4)
    with pytest.raises(ValueError):
        ToyConfig(num_heads=4, num_kv_heads=3)


def test_make_batch_masks_out_the_prompt():
    b = make_batch(2, 10, 64, seed=0, prompt_len=4)
    assert b["input_ids"].shape == (2, 10)
    assert b["labels"].shape == (2, 10)
    assert b["mask"].dtype is torch.bool
    assert int(b["mask"][:, :4].sum()) == 0
    assert int(b["mask"][:, 4:].sum()) == 2 * 6


def test_make_batch_is_reproducible():
    a = make_batch(2, 6, 32, seed=3)
    b = make_batch(2, 6, 32, seed=3)
    assert torch.equal(a["input_ids"], b["input_ids"])


# --- run_rl_suite.py ------------------------------------------------------- #
def test_rl_suite_help_exits_zero():
    r = _run("--help")
    assert r.returncode == 0
    assert "--experiments" in r.stdout


def test_rl_suite_dry_run_writes_nothing(tmp_path):
    env = dict(os.environ)
    env["MM_RUNS_ROOT"] = str(tmp_path)
    r = _run("--dry-run", env=env)
    assert r.returncode == 0, r.stderr
    assert "--dry-run" in r.stdout
    assert list(tmp_path.iterdir()) == []


def test_rl_suite_rejects_unknown_experiment():
    r = _run("--experiments", "r9")
    assert r.returncode == 2
    assert "未知实验" in r.stderr


def test_rl_suite_rejects_missing_config():
    r = _run("--config", "lab/configs/nope.json")
    assert r.returncode == 2
    assert "找不到配置文件" in r.stderr


def test_rl_suite_rejects_cuda_when_unavailable():
    if torch.cuda.is_available():
        pytest.skip("这台机器有 CUDA，测不到「请求 cuda 但没有 cuda」这条分支")
    r = _run("--device", "cuda", "--dry-run")
    assert r.returncode == 2
    assert "没有 CUDA" in r.stderr


def test_rl_suite_runs_all_six_experiments_and_writes_logs(tmp_path):
    env = dict(os.environ)
    env["MM_RUNS_ROOT"] = str(tmp_path)
    r = _run("--quiet", env=env)
    assert r.returncode == 0, r.stderr
    summary = tmp_path / "rl_smoke" / "logs" / "rl_suite_summary.json"
    jsonl = tmp_path / "rl_smoke" / "logs" / "rl_suite.jsonl"
    assert summary.is_file() and jsonl.is_file()
    payload = json.loads(summary.read_text(encoding="utf-8"))
    assert set(payload["results"]) == set(run_rl_suite.EXPERIMENTS)
    for name, rec in payload["results"].items():
        assert rec["status"] == "ok", "{} 失败：{}".format(name, rec["error"])
    assert len(jsonl.read_text(encoding="utf-8").strip().splitlines()) == 6

    # 抽查硬结论，防止「跑通但算错」
    r1 = payload["results"]["r1"]["result"]
    assert r1["attribution"]["injected_mask_shift"]["verdict"] == "MASK_MISMATCH"
    assert r1["attribution"]["injected_path_mismatch"]["verdict"] == "PATH_MISMATCH"
    assert r1["determinism"]["bitwise_identical"] is True

    r3 = payload["results"]["r3"]["result"]
    assert r3["all_violations_caught"] is True
    assert r3["correct_iteration"]["n_violations"] == 0

    r4 = payload["results"]["r4"]["result"]
    hard = r4["cases"]["base=1.0_spread=0.0001"]["by_dtype"]
    assert hard["float16"]["zero_std_group_ratio"] > 0.5
    assert hard["float32"]["zero_std_group_ratio"] == 0.0

    r6 = payload["results"]["r6"]["result"]
    assert r6["step0_invariant"]["exact_zero_kl"] is True
    assert r6["step0_invariant_after_breaking_zero_init"]["exact_zero_kl"] is False


def test_rl_suite_single_experiment_selection(tmp_path):
    env = dict(os.environ)
    env["MM_RUNS_ROOT"] = str(tmp_path)
    r = _run("--experiments", "r6", "--quiet", env=env)
    assert r.returncode == 0, r.stderr
    payload = json.loads(
        (tmp_path / "rl_smoke" / "logs" / "rl_suite_summary.json").read_text(encoding="utf-8")
    )
    assert set(payload["results"]) == {"r6"}
