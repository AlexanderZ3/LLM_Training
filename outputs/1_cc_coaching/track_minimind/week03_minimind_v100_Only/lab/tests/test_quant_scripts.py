"""脚本层的单测：probe_int8_caps.py 与 run_quant_suite.py。

这些测试守的是「开环交付」最容易崩的那一层——不是数值，是**入口**：
参数解析、路径合同、输出落盘、退出码。代码在没有 GPU 的 Windows 上写、
第一次在 Linux + V100 上运行，入口一崩就什么都测不到。

probe_int8_caps.py 的核心约定：**它是信息收集器，不是门**。
任何探针失败都不能让它非零退出，否则「V100 上 convert_fx 不能 .cuda()」
这条预期内的失败会把整条流水线卡住。
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

_LAB = Path(__file__).resolve().parents[1]
_SRC = _LAB / "src"
_SCRIPTS = _LAB / "scripts"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import probe_int8_caps  # noqa: E402
import run_quant_suite  # noqa: E402


def _run(script: str, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(_SCRIPTS / script), *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )


def test_probe_help_exits_zero():
    r = _run("probe_int8_caps.py", "--help")
    assert r.returncode == 0
    assert "--only" in r.stdout


def test_probe_runs_and_exits_zero_even_without_gpu(tmp_path):
    """没有 GPU 时也必须退出 0：失败项本身就是要记录的信息。"""
    out = tmp_path / "caps.json"
    r = _run("probe_int8_caps.py", "--out", str(out), "--quiet")
    assert r.returncode == 0, r.stderr
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["schema"] == "mm_quant.int8_caps.v1"
    for name in probe_int8_caps.CHECKS:
        assert name in payload["results"]
        assert "status" in payload["results"][name]
    assert payload["summary"]


def test_probe_rejects_unknown_check_name():
    r = _run("probe_int8_caps.py", "--only", "does_not_exist")
    assert r.returncode == 2
    assert "未知探针名" in r.stderr


def test_probe_skip_marks_checks_as_skipped(tmp_path):
    out = tmp_path / "caps.json"
    r = _run("probe_int8_caps.py", "--out", str(out), "--skip", "bnb,convert_fx", "--quiet")
    assert r.returncode == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["results"]["bnb"]["status"] == "skipped"
    assert payload["results"]["convert_fx"]["status"] == "skipped"
    assert payload["results"]["environment"]["status"] == "ok"


def test_probe_guard_never_propagates_exceptions():
    def boom():
        raise RuntimeError("模拟探针内部炸了")

    rec = probe_int8_caps._guard(boom)
    assert rec["status"] == "error"
    assert rec["exception"] == "RuntimeError"


def test_probe_environment_reports_int8_tensor_core_flag():
    env = probe_int8_caps.probe_environment()
    assert env["status"] == "ok"
    assert "torch" in env
    if env["cuda_available"]:
        # V100 是 7.0，Turing 才是 7.5：这个布尔决定「INT8 提速是不是假的」
        assert isinstance(env["has_int8_tensor_core"], bool)
    else:
        assert env["device_count"] == 0


def test_probe_weight_int8pack_mm_records_expected_absence_on_torch_21():
    rec = probe_int8_caps.probe_weight_int8pack_mm()
    assert rec["expected_on_torch_2_1"] is False
    assert isinstance(rec["exists"], bool)


def test_quant_suite_help_exits_zero():
    r = _run("run_quant_suite.py", "--help")
    assert r.returncode == 0
    assert "--experiments" in r.stdout


def test_quant_suite_dry_run_does_not_write_anything(tmp_path):
    """--dry-run 只打印计划：MM_RUNS_ROOT 下不能多出任何文件。"""
    import os

    env = dict(os.environ)
    env["MM_RUNS_ROOT"] = str(tmp_path)
    r = subprocess.run(
        [sys.executable, str(_SCRIPTS / "run_quant_suite.py"), "--dry-run"],
        capture_output=True, text=True, encoding="utf-8", errors="replace", env=env,
    )
    assert r.returncode == 0, r.stderr
    assert "--dry-run" in r.stdout
    assert list(tmp_path.iterdir()) == []


def test_quant_suite_rejects_unknown_experiment():
    r = _run("run_quant_suite.py", "--experiments", "q9")
    assert r.returncode == 2
    assert "未知实验" in r.stderr


def test_quant_suite_rejects_missing_config():
    r = _run("run_quant_suite.py", "--config", "lab/configs/does_not_exist.json")
    assert r.returncode == 2
    assert "找不到配置文件" in r.stderr


def test_quant_suite_rejects_cuda_when_unavailable():
    import torch

    if torch.cuda.is_available():
        pytest.skip("这台机器有 CUDA，测不到「请求 cuda 但没有 cuda」这条分支")
    r = _run("run_quant_suite.py", "--device", "cuda", "--dry-run")
    assert r.returncode == 2
    assert "没有 CUDA" in r.stderr


def test_quant_suite_runs_all_six_experiments_and_writes_logs(tmp_path):
    """整套 Q1–Q6 在 CPU 上跑通，产物落到 MM_RUNS_ROOT 下。"""
    import os

    env = dict(os.environ)
    env["MM_RUNS_ROOT"] = str(tmp_path)
    r = subprocess.run(
        [sys.executable, str(_SCRIPTS / "run_quant_suite.py"), "--quiet"],
        capture_output=True, text=True, encoding="utf-8", errors="replace", env=env,
    )
    assert r.returncode == 0, r.stderr
    summary = tmp_path / "quant_smoke" / "logs" / "quant_suite_summary.json"
    jsonl = tmp_path / "quant_smoke" / "logs" / "quant_suite.jsonl"
    assert summary.is_file()
    assert jsonl.is_file()
    payload = json.loads(summary.read_text(encoding="utf-8"))
    assert set(payload["results"]) == set(run_quant_suite.EXPERIMENTS)
    for name, rec in payload["results"].items():
        assert rec["status"] == "ok", "{} 失败：{}".format(name, rec["error"])
    assert len(jsonl.read_text(encoding="utf-8").strip().splitlines()) == 6

    # 抽查几条硬结论，防止套件「跑通但算错」
    q1 = payload["results"]["q1"]["result"]
    assert q1["granularity_monotone"] is True
    q3 = payload["results"]["q3"]["result"]
    assert q3["final_loss"]["adam8bit_linear"] > q3["final_loss"]["adam8bit_power"] * 100
    assert 0.24 < q3["state_bytes_big_param"]["ratio_vs_fp32"] < 0.26
    q5 = payload["results"]["q5"]["result"]
    assert q5["backend_comparison"]["forward_exact_equal_vs_aten"] is True
    q6 = payload["results"]["q6"]["result"]
    assert q6["improvement_factor"] > 1.5
