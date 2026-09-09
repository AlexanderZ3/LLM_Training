"""字节账：param/grad/optim 手算必须**精确等于**实测；activation 只验标度律。

为什么这样分
------------
前三项是恒等式：给定参数量、dtype、优化器，字节数没有解释空间。手算与实测
差一个字节都说明有一项没算（最常见的是 AdamW 每个张量的 step 标量）。
activation 取决于 autograd 保存了哪些中间张量，随算子实现变化，所以它只能是
估算；能验证的是它对 batch 严格线性——那条性质与实现无关。
"""

from __future__ import annotations

import pytest
import torch

from mm_v100 import common as C
from mm_v100 import ledger as LG
from mm_v100 import model as M

TINY = {"vocab_size": 128, "hidden_size": 32, "num_hidden_layers": 2,
        "num_attention_heads": 4, "num_key_value_heads": 2,
        "intermediate_size": 64, "max_seq_len": 64}


def make(batch: int, seq: int, device: str = "cpu"):
    model, cfg = M.build_model(TINY, device=device, seed=3)
    ids = torch.randint(0, TINY["vocab_size"], (batch, seq))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    return model, ids, optimizer


def measure_tiny(batch: int, seq: int, dtype_name: str = "float32"):
    model, ids, optimizer = make(batch, seq)
    return LG.measure(model, ids, ids.clone(), optimizer,
                      dtype_name=dtype_name, device="cpu")


def test_param_grad_optim_hand_equals_measured():
    model, ids, optimizer = make(2, 16)
    measured = LG.measure(model, ids, ids.clone(), optimizer,
                          dtype_name="float32", device="cpu")
    hand = LG.hand_ledger(TINY, world_size=1, mode="ddp", batch_per_rank=2,
                          seq_len=16, param_dtype="float32",
                          compute_dtype="float32")
    assert measured["param_bytes"] == hand["param_bytes"]
    assert measured["grad_bytes"] == hand["grad_bytes"]
    assert measured["optim_bytes"] == hand["optim_bytes"]
    assert LG.compare(hand, measured)["exact_items_match"] is True


def test_optim_bytes_includes_adam_step_scalars():
    """漏掉 step 标量时手算会小一点点，看起来像测量误差，其实是漏项。"""
    hand = LG.hand_ledger(TINY, world_size=1, mode="ddp", batch_per_rank=1,
                          seq_len=8)
    breakdown = hand["params_breakdown"]
    naive = breakdown["total"] * 4 * 2
    assert hand["optim_bytes"] == naive + breakdown["n_tensors"] * LG.ADAM_STEP_BYTES
    assert hand["optim_bytes"] > naive


def test_sharding_divides_the_right_terms():
    common = dict(model_cfg=TINY, batch_per_rank=1, seq_len=8, world_size=8)
    ddp = LG.hand_ledger(mode="ddp", **common)
    sgo = LG.hand_ledger(mode="shard_grad_op", **common)
    fs = LG.hand_ledger(mode="full_shard", **common)
    # SHARD_GRAD_OP 只切梯度与优化器状态，参数仍是整份。
    assert sgo["param_bytes"] == ddp["param_bytes"]
    assert sgo["grad_bytes"] * 8 == ddp["grad_bytes"]
    # FULL_SHARD 三项都切。
    assert fs["param_bytes"] * 8 == ddp["param_bytes"]
    assert fs["grad_bytes"] == sgo["grad_bytes"]
    assert fs["optim_bytes"] < ddp["optim_bytes"]
    # 只有 FULL_SHARD 需要 all_gather buffer。
    assert ddp["all_gather_buffer_bytes"] == 0
    assert fs["all_gather_buffer_bytes"] > 0


def test_all_gather_buffer_grows_when_wrap_degenerates():
    """auto_wrap 退化成一个根单元时，buffer 等于整份模型 —— FSDP 白做。"""
    fine = LG.hand_ledger(TINY, world_size=8, mode="full_shard",
                          batch_per_rank=1, seq_len=8, n_units=3)
    degenerate = LG.hand_ledger(TINY, world_size=8, mode="full_shard",
                                batch_per_rank=1, seq_len=8, n_units=1)
    # 差几个字节是 int() 截断，不是模型差异；用绝对容差而不是相对容差。
    assert abs(degenerate["all_gather_buffer_bytes"]
               - 3 * fine["all_gather_buffer_bytes"]) <= 3
    assert degenerate["all_gather_buffer_bytes"] == pytest.approx(
        degenerate["params_total"] * 2, rel=1e-9)


def test_activation_is_linear_in_batch_hand_and_measured():
    hand1 = LG.hand_ledger(TINY, batch_per_rank=2, seq_len=16)
    hand2 = LG.hand_ledger(TINY, batch_per_rank=4, seq_len=16)
    assert hand2["activation_bytes"] == pytest.approx(
        2 * hand1["activation_bytes"], rel=1e-9)

    m1 = measure_tiny(2, 16)
    m2 = measure_tiny(4, 16)
    ratio = m2["activation_bytes"] / m1["activation_bytes"]
    assert ratio == pytest.approx(2.0, rel=0.05), (
        "activation 对 batch 不是线性的，ratio=" + str(ratio))


def test_activation_grows_with_seq_len():
    m_short = measure_tiny(2, 16)
    m_long = measure_tiny(2, 32)
    assert m_long["activation_bytes"] >= 2 * m_short["activation_bytes"] * 0.95


def test_param_and_grad_do_not_grow_with_batch():
    """字节账里唯一随 batch 变的只有 activation。这条是四项定义的直接推论。"""
    m1 = measure_tiny(2, 16)
    m2 = measure_tiny(8, 16)
    assert m1["param_bytes"] == m2["param_bytes"]
    assert m1["grad_bytes"] == m2["grad_bytes"]
    assert m1["optim_bytes"] == m2["optim_bytes"]


def test_saved_activation_meter_excludes_parameters():
    model, ids, _ = make(2, 16)
    with LG.SavedActivationMeter(LG.param_storage_ptrs(model)) as meter:
        out = model(ids, labels=ids.clone())
        with_exclusion = meter.total_bytes()
        out.loss.backward()
    model2, ids2, _ = make(2, 16)
    with LG.SavedActivationMeter(set()) as meter2:
        out2 = model2(ids2, labels=ids2.clone())
        without_exclusion = meter2.total_bytes()
        out2.loss.backward()
    assert with_exclusion < without_exclusion


def test_optimizer_state_bytes_is_zero_before_first_step():
    model, ids, optimizer = make(2, 8)
    assert LG.optimizer_state_bytes(optimizer) == 0
    model(ids, labels=ids.clone()).loss.backward()
    optimizer.step()
    assert LG.optimizer_state_bytes(optimizer) > 0


def test_gate_config_hand_ledger_numbers():
    """Gate 的 A0 题：8 卡 FULL_SHARD 下每卡四项。

    P = 63,912,192，fp32 参数：
      DDP        param = 255,648,768  grad = 255,648,768  optim = 511,297,536+
      FULL_SHARD 各除以 8
    """
    cfg = C.load_config("v100_768")["model"]
    ddp = LG.hand_ledger(cfg, world_size=8, mode="ddp", batch_per_rank=8,
                         seq_len=512)
    fs = LG.hand_ledger(cfg, world_size=8, mode="full_shard", batch_per_rank=8,
                        seq_len=512)
    assert ddp["params_total"] == 63912192
    assert ddp["param_bytes"] == 63912192 * 4
    assert ddp["grad_bytes"] == 63912192 * 4
    assert fs["param_bytes"] == 63912192 * 4 // 8
    assert fs["n_units"] == 9  # 8 层 + 根
    # 优化器状态比参数本身还大，这是「模型不大却塞不下」的主因。
    assert ddp["optim_bytes"] > ddp["param_bytes"]


def test_render_markdown_has_three_columns():
    hand = LG.hand_ledger(TINY, batch_per_rank=2, seq_len=16)
    measured = measure_tiny(2, 16)
    md = LG.render_markdown(hand, measured)
    assert "手算" in md and "实测" in md and "偏差" in md
    for item in LG.ROWS:
        assert item in md


def test_unknown_mode_and_dtype_raise():
    with pytest.raises(ValueError):
        LG.hand_ledger(TINY, mode="magic")
    with pytest.raises(ValueError):
        LG.dtype_bytes("int4")
