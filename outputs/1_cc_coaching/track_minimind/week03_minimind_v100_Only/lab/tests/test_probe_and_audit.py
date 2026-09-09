"""Day 0 的两件事：探针不能崩、审计的四个计数要数得对。

探针在没有 GPU 的机器上必须**照常返回**（各项标未知），而不是抛异常。
本机就是那种机器，所以这组测试同时也是探针的真实运行证据。
"""

from __future__ import annotations

import json

import pytest
import torch

from mm_v100 import compat_audit as CA
from mm_v100 import probe


# ---------------------------------------------------------------------------
# 探针
# ---------------------------------------------------------------------------


def test_collect_runs_without_gpu_and_is_json_serializable(tmp_path):
    report = probe.collect(runs_dir=str(tmp_path))
    text = json.dumps(report, ensure_ascii=False)
    assert len(text) > 100
    for key in ("versions", "devices", "sdpa", "int8", "grad_scaler",
                "dcp_api", "fsdp_api", "disk", "checks"):
        assert key in report


def test_int8_probe_never_raises(tmp_path):
    """V100 上 torch._int_mm 大概率失败；那是预期结果，不能让探针挂掉。"""
    info = probe.int8_info()
    assert info["callable_on_device"] in (True, False)
    if info["callable_on_device"] is False:
        assert "reason" in info


def test_sdpa_reports_unknown_without_cuda():
    info = probe.sdpa_info()
    if not torch.cuda.is_available():
        assert info["functional"] == {"flash": None, "mem_efficient": None,
                                      "math": None}
        assert "未知" in info["functional_note"]


def test_device_info_distinguishes_unknown_from_false():
    """没有 GPU 时 bf16_known 必须是 False —— 未知不等于不支持。"""
    info = probe.device_info()
    if not torch.cuda.is_available():
        assert info["bf16_known"] is False
        assert info["device_count"] == 0


def test_disk_reports_tier_not_exact_size(tmp_path):
    info = probe.disk_info(str(tmp_path))
    assert info["known"] is True
    assert info["tier"] in ("<10GB", "10-50GB", "50-200GB", ">200GB")
    # 精确容量属于公司机器信息，不能出现在输出里。
    assert not any(isinstance(v, float) for v in info.values())


def test_space_tier_boundaries():
    assert probe.space_tier(5.0) == "<10GB"
    assert probe.space_tier(10.0) == "10-50GB"
    assert probe.space_tier(50.0) == "50-200GB"
    assert probe.space_tier(500.0) == ">200GB"


def test_grad_scaler_defaults_match_our_constants():
    info = probe.grad_scaler_defaults()
    assert info["expected"]["init_scale"] == 65536.0
    assert info["expected"]["growth_interval"] == 2000
    if info["source"] not in (None, "unavailable"):
        assert info.get("matches_expected") is True


def test_render_text_has_no_paths(tmp_path):
    report = probe.collect(runs_dir=str(tmp_path))
    text = probe.render_text(report)
    assert str(tmp_path) not in text
    assert "PASS" in text or "FAIL" in text


def test_checks_flag_missing_gpu_as_failure(tmp_path):
    """本机没有 GPU 时 eight_gpus 必然 FAIL。这不是 bug，是「未知」的表达方式。"""
    report = probe.collect(runs_dir=str(tmp_path))
    if not torch.cuda.is_available():
        assert report["all_checks_pass"] is False
        assert "eight_gpus" in report["blocking_failures"]


# ---------------------------------------------------------------------------
# 兼容审计
# ---------------------------------------------------------------------------

SAMPLE_BF16 = '''
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--dtype", type=str, default="bfloat16")
with torch.cuda.amp.autocast(dtype=torch.bfloat16):
    pass
'''

SAMPLE_NEW_API = '''
from torch.distributed.fsdp import fully_shard
mesh = init_device_mesh("cuda", (8,))
model = torch.compile(model)
'''

SAMPLE_OK = '''
scaler = torch.cuda.amp.GradScaler()
with torch.cuda.amp.autocast(dtype=torch.float16):
    loss = model(x)
scaler.scale(loss).backward()
scaler.step(opt)
scaler.update()
'''


def test_scan_detects_bfloat16_default_and_missing_scaler():
    result = CA.scan_text(SAMPLE_BF16, "trainer/train_x.py")
    assert result["defaults_bfloat16"] is True
    assert result["dtype_defaults"] == ["bfloat16"]
    assert result["uses_autocast"] is True
    assert result["has_gradscaler"] is False


def test_scan_accepts_correct_fp16_pattern():
    result = CA.scan_text(SAMPLE_OK, "trainer/train_ok.py")
    assert result["defaults_bfloat16"] is False
    assert result["has_gradscaler"] is True
    assert result["scaler"]["scaler.step"] == 1


def test_scan_detects_newer_apis_with_line_numbers():
    result = CA.scan_text(SAMPLE_NEW_API, "model/new.py")
    symbols = {a["symbol"] for a in result["newer_apis"]}
    assert "fully_shard" in symbols
    assert "device_mesh" in symbols
    assert result["compile_lines"]
    for api in result["newer_apis"]:
        assert all(isinstance(n, int) and n > 0 for n in api["lines"])


def test_summarize_gives_the_four_counts():
    results = [CA.scan_text(SAMPLE_BF16, "trainer/a.py"),
               CA.scan_text(SAMPLE_NEW_API, "model/b.py"),
               CA.scan_text(SAMPLE_OK, "trainer/c.py")]
    summary = CA.summarize(results)
    assert summary["n_files_scanned"] == 3
    assert summary["n_default_bfloat16"] == 1
    assert summary["n_autocast_no_scaler"] == 1
    assert summary["n_torch_compile"] == 1
    assert summary["n_newer_api_hits"] >= 2
    assert "trainer/a.py" in summary["files_default_bfloat16"]


def test_audit_on_synthetic_tree(tmp_path):
    root = tmp_path / "fake_minimind"
    (root / "trainer").mkdir(parents=True)
    (root / "model").mkdir(parents=True)
    (root / "trainer" / "train_pretrain.py").write_text(SAMPLE_BF16,
                                                        encoding="utf-8")
    (root / "model" / "model_minimind.py").write_text(SAMPLE_OK,
                                                      encoding="utf-8")
    report = CA.audit(str(root), subdirs=("trainer", "model"))
    assert report["summary"]["n_files_scanned"] == 2
    md = CA.render_markdown(report, "fake_minimind")
    assert "n_default_bfloat16" in md
    assert "train_pretrain.py" in md
    # 报告里只能有文件名与行号，不能有源码片段。
    assert "add_argument" not in md


def test_audit_on_empty_tree_gives_actionable_error(tmp_path):
    with pytest.raises(FileNotFoundError) as exc:
        CA.audit(str(tmp_path))
    assert "MINIMIND_ROOT" in str(exc.value)


def test_newer_api_table_covers_the_symbols_we_actually_use():
    """本 lab 自己避开的那些 2.1 之后的 API，清单里必须都有。"""
    symbols = {row[0] for row in CA.NEWER_APIS}
    for name in ("fully_shard", "DTensor", "device_mesh", "sdpa_kernel",
                 "torch.amp.GradScaler"):
        assert name in symbols
