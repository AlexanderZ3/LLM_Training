"""dtype 守卫：bfloat16 必须在进入 autocast 之前就被拒绝。

守卫的价值在**报错的位置**。放任 bfloat16 传到 autocast，错误会在训练第一步的
CUDA 里出现（Current CUDA Device does not support bfloat16），此时 8 个进程已经
建好、数据已经加载、显存已经占上。在参数解析阶段拒绝，公司机器 10 分钟的
运行预算不会被浪费。
"""

from __future__ import annotations

import pytest
import torch

from mm_v100 import common as C


def test_float32_and_float16_accepted_on_cpu():
    assert C.assert_dtype_supported("float32", "cpu") is torch.float32
    assert C.assert_dtype_supported("fp32", "cpu") is torch.float32
    assert C.assert_dtype_supported("float16", "cpu") is torch.float16
    assert C.assert_dtype_supported("fp16", "cpu") is torch.float16


def test_bfloat16_rejected_on_cpu_with_actionable_message():
    with pytest.raises(RuntimeError) as exc:
        C.assert_dtype_supported("bfloat16", "cpu")
    message = str(exc.value)
    assert "bfloat16" in message
    assert "V100" in message
    # 报错必须给出可以直接执行的下一步，而不只是说明为什么不行。
    assert "--dtype" in message and "float16" in message


def test_bfloat16_rejected_on_sm70(monkeypatch):
    """把设备伪装成 sm70，确认走的是 capability 分支而不是 not-CUDA 分支。"""
    monkeypatch.setattr(C, "device_capability", lambda device: (7, 0))
    with pytest.raises(RuntimeError) as exc:
        C.assert_dtype_supported("bfloat16", "cuda:0")
    assert "7.0" in str(exc.value)


def test_bfloat16_accepted_on_sm80(monkeypatch):
    monkeypatch.setattr(C, "device_capability", lambda device: (8, 0))
    assert C.assert_dtype_supported("bfloat16", "cuda:0") is torch.bfloat16


def test_unknown_dtype_raises_value_error():
    with pytest.raises(ValueError):
        C.assert_dtype_supported("float8", "cpu")


def test_bounded_train_rejects_bfloat16_before_building_model():
    """入口脚本层面的守卫：--dtype bfloat16 不能走到训练循环里。"""
    from mm_v100 import bounded_train as BT

    args = BT.build_parser().parse_args(
        ["--config", "tiny_cpu", "--dtype", "bfloat16", "--force-cpu",
         "--max-steps", "1"])
    with pytest.raises(RuntimeError) as exc:
        BT.train(args)
    assert "V100" in str(exc.value)


def test_autocast_context_is_noop_for_float32():
    ctx = C.autocast_context(torch.float32, "cpu")
    with ctx:
        out = torch.ones(2, 2) @ torch.ones(2, 2)
    assert out.dtype is torch.float32


def test_scaler_adapter_disabled_for_float32():
    scaler = C.ScalerAdapter(torch.float32, "cpu")
    assert scaler.is_enabled() is False
    tensor = torch.ones(3)
    assert torch.equal(scaler.scale(tensor), tensor)


def test_scaler_adapter_uses_pure_python_on_cpu_for_fp16():
    """CPU 上必须落到纯 Python 实现，否则跳步逻辑根本不会触发。"""
    scaler = C.ScalerAdapter(torch.float16, "cpu")
    assert scaler.is_enabled() is True
    assert scaler.backend == "pure_python"
    assert scaler.get_scale() == C.DEFAULT_INIT_SCALE


def test_pure_python_scaler_skips_step_on_inf_and_halves_scale():
    param = torch.nn.Parameter(torch.ones(4))
    optimizer = torch.optim.SGD([param], lr=1.0)
    scaler = C.ScalerAdapter(torch.float16, "cpu")
    before_scale = scaler.get_scale()
    param.grad = torch.full((4,), float("inf"))
    applied = scaler.step_and_update(optimizer)
    assert applied is False
    assert scaler.last_step_skipped is True
    assert scaler.skipped_steps == 1
    assert scaler.get_scale() == pytest.approx(before_scale * 0.5)
    assert torch.equal(param.detach(), torch.ones(4))


def test_pure_python_scaler_applies_step_on_finite_grad():
    param = torch.nn.Parameter(torch.ones(4))
    optimizer = torch.optim.SGD([param], lr=1.0)
    scaler = C.ScalerAdapter(torch.float16, "cpu")
    param.grad = torch.full((4,), scaler.get_scale())  # 相当于未缩放的 1.0
    applied = scaler.step_and_update(optimizer)
    assert applied is True
    assert scaler.applied_steps == 1
    assert torch.allclose(param.detach(), torch.zeros(4))


def test_scaler_state_dict_round_trip():
    a = C.ScalerAdapter(torch.float16, "cpu")
    a.skipped_steps = 3
    a.applied_steps = 5
    a.impl.update(new_scale=1024.0)
    b = C.ScalerAdapter(torch.float16, "cpu")
    b.load_state_dict(a.state_dict())
    assert b.get_scale() == 1024.0
    assert b.skipped_steps == 3
    assert b.applied_steps == 5
