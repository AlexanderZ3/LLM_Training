"""dtype 守卫：bfloat16 必须在进入 autocast 之前就被拒绝。

这条守卫的价值在于**报错的位置**。如果放任 bfloat16 传到 autocast，
错误会在训练循环第一步的 CUDA 里出现（`Current CUDA Device does not support bfloat16`），
此时已经建好了 8 个进程、加载了数据、占了显存。在参数解析阶段就拒绝，
公司机器上 10 分钟的运行预算不会被浪费。
"""

from __future__ import annotations

import pytest
import torch

from mm_dist import common as C
from mm_dist import train_fsdp


def test_float32_and_float16_accepted_on_cpu():
    assert C.assert_dtype_supported("float32", "cpu") is torch.float32
    assert C.assert_dtype_supported("fp32", "cpu") is torch.float32
    assert C.assert_dtype_supported("float16", "cpu") is torch.float16
    assert C.assert_dtype_supported("fp16", "cpu") is torch.float16


def test_bfloat16_rejected_on_cpu_with_v100_message():
    with pytest.raises(RuntimeError) as exc:
        C.assert_dtype_supported("bfloat16", "cpu")
    message = str(exc.value)
    assert "bfloat16" in message
    assert "V100" in message
    assert "float16" in message


def test_bfloat16_rejected_on_low_capability(monkeypatch):
    """把设备伪装成 sm70，确认走的是 capability 分支而不是 'not CUDA' 分支。"""
    monkeypatch.setattr(C, "device_capability", lambda device: (7, 0))
    with pytest.raises(RuntimeError) as exc:
        C.assert_dtype_supported("bfloat16", "cuda:0")
    assert "7.0" in str(exc.value)


def test_bfloat16_accepted_on_high_capability(monkeypatch):
    monkeypatch.setattr(C, "device_capability", lambda device: (8, 0))
    assert C.assert_dtype_supported("bfloat16", "cuda:0") is torch.bfloat16


def test_unknown_dtype_raises_value_error():
    with pytest.raises(ValueError):
        C.assert_dtype_supported("float8", "cpu")


def test_grad_scaler_disabled_on_cpu():
    scaler = C.make_grad_scaler(torch.float16, "cpu")
    assert scaler.is_enabled() is False
    tensor = torch.ones(3)
    assert torch.equal(scaler.scale(tensor), tensor)


def test_autocast_context_is_noop_on_cpu():
    ctx = C.autocast_context(torch.float16, "cpu")
    with ctx:
        out = torch.ones(2, 2) @ torch.ones(2, 2)
    assert out.dtype is torch.float32


def test_mixed_precision_none_on_cpu_and_fp32():
    assert train_fsdp.make_mixed_precision("float16", "cpu") is None
    assert train_fsdp.make_mixed_precision("float32", "cuda:0") is None
    mp = train_fsdp.make_mixed_precision("float16", "cuda:0")
    assert mp is not None
    assert mp.param_dtype is torch.float16
    assert mp.reduce_dtype is torch.float16


def test_sdpa_context_is_noop_without_cuda():
    with C.sdpa_context("auto", "cpu"):
        assert True
    with C.sdpa_context("math", "cpu"):
        assert True


def test_cosine_lr_matches_minimind_formula():
    """与 trainer/trainer_utils.py 的 get_lr 逐点一致。"""
    import math
    base, total = 5e-4, 100
    for step in (0, 1, 50, 99, 100):
        expected = base * (0.1 + 0.45 * (1 + math.cos(math.pi * step / total)))
        assert abs(C.cosine_lr(step, total, base) - expected) < 1e-15
