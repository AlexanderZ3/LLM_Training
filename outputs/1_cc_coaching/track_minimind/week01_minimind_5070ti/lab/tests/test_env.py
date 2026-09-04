"""Day 0：probe_env.py 可运行、退出码 0、JSON 字段齐全（无 GPU 时 GPU 字段为 null）。"""
from __future__ import annotations

import json
import subprocess
import sys

REQUIRED_KEYS = [
    "python", "torch", "torch_cuda_version", "cuda_available", "gpu_name", "compute_capability", "arch_list",
    "bf16_supported", "sdpa", "cuda_matmul_ok", "transformers", "datasets",
    "minimind_root", "minimind_commit", "pinned_commit", "commit_matches", "data_files", "warnings",
]


def test_probe_env_runs_and_has_fields(lab_root, tmp_path):
    out = tmp_path / "probe.json"
    cmd = [sys.executable, str(lab_root / "scripts" / "probe_env.py"), "--out", str(out), "--skip-hash"]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stderr[-2000:]
    data = json.loads(out.read_text(encoding="utf-8"))
    missing = [k for k in REQUIRED_KEYS if k not in data]
    assert not missing, f"probe.json 缺字段: {missing}"
    assert data["pinned_commit"] == "7a6fddd63a30c06b2fdd5fac4089922b29bc841b"
    if not data["cuda_available"]:
        assert data["gpu_name"] is None and data["compute_capability"] is None and data["bf16_supported"] is None
    assert isinstance(data["warnings"], list)
    assert set(data["data_files"]) <= {"pretrain_t2t_mini.jsonl", "sft_t2t_mini.jsonl", "dpo.jsonl", "rlaif.jsonl"}


def test_probe_reports_commit_when_root_set(minimind_root):
    from mm_probe.minimind_env import PINNED_COMMIT, git_head

    head = git_head(minimind_root)
    assert head is None or len(head) == 40
    if head is not None:
        assert head == PINNED_COMMIT, f"MiniMind 未锁定在 {PINNED_COMMIT}: HEAD={head}"
