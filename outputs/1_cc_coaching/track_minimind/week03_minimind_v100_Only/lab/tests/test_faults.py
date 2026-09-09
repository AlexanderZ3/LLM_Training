"""三档静默故障：每一档都必须被**唯一**一个不变量抓住，其余两个必须保持正常。

「被抓住」不够，还要「不误伤」——否则判别表就不可判定，Day 2 的跨场景 debug
题就没有唯一答案。所以每个端到端用例都同时断言 3 个不变量，不只是它自己那个。
"""

from __future__ import annotations

import os

import pytest
import torch

import conftest as H
from mm_v100 import data_contract as DC
from mm_v100 import faults as FA

STEPS = 3
GLOBAL_BATCH = 8


# ---------------------------------------------------------------------------
# 纯函数
# ---------------------------------------------------------------------------


def test_label_shift_keeps_target_count_but_breaks_alignment():
    """静默的定义：计数不变、量级不变，只有对齐这一条能看出来。"""
    ids = torch.arange(1, 13).reshape(2, 6)
    labels = ids.clone()
    labels[:, :2] = DC.IGNORE_INDEX
    faulty = FA.apply_label_fault(ids, labels, "label_shift")
    assert int((faulty != DC.IGNORE_INDEX).sum()) == int(
        (labels != DC.IGNORE_INDEX).sum())
    assert DC.label_alignment_violations(ids, labels) == 0
    assert DC.label_alignment_violations(ids, faulty) > 0


def test_none_mode_returns_equal_copy():
    ids = torch.arange(1, 9).reshape(2, 4)
    labels = ids.clone()
    out = FA.apply_label_fault(ids, labels, "none")
    assert torch.equal(out, labels)
    assert out.data_ptr() != labels.data_ptr()


def test_unknown_fault_mode_raises():
    ids = torch.zeros(1, 4, dtype=torch.long)
    with pytest.raises(ValueError):
        FA.apply_label_fault(ids, ids.clone(), "does_not_exist")


def test_should_sync_matrix():
    assert FA.should_sync_this_micro("none", True) is True
    assert FA.should_sync_this_micro("none", False) is False
    assert FA.should_sync_this_micro("ddp_no_sync", True) is False
    assert FA.should_sync_this_micro("ddp_no_sync", False) is False


def test_grad_inf_injector_poisons_exactly_one_param():
    model = torch.nn.Linear(4, 4)
    with FA.GradInfInjector(model, enabled=True) as inj:
        model(torch.ones(2, 4)).sum().backward()
        assert inj.n_fired == 1
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert any(bool(torch.isinf(g).any()) for g in grads)


def test_grad_inf_injector_disabled_is_noop():
    model = torch.nn.Linear(4, 4)
    with FA.GradInfInjector(model, enabled=False):
        model(torch.ones(2, 4)).sum().backward()
    for p in model.parameters():
        assert bool(torch.isfinite(p.grad).all())


def test_cross_rank_grad_delta_is_none_without_process_group():
    model = torch.nn.Linear(2, 2)
    model(torch.ones(1, 2)).sum().backward()
    assert FA.cross_rank_grad_delta(model, 1) is None


@pytest.mark.parametrize("mode,obs", [
    ("none", {"I1_label_align_violations": 0, "I2_optimizer_step_ratio": 1.0,
              "I2_param_delta_norm": 0.1, "I3_cross_rank_grad_delta": 0.0}),
    ("label_shift", {"I1_label_align_violations": 5,
                     "I2_optimizer_step_ratio": 1.0,
                     "I2_param_delta_norm": 0.1,
                     "I3_cross_rank_grad_delta": 0.0}),
    ("scaler_stuck", {"I1_label_align_violations": 0,
                      "I2_optimizer_step_ratio": 0.0,
                      "I2_param_delta_norm": 0.0,
                      "I3_cross_rank_grad_delta": 0.0}),
    ("ddp_no_sync", {"I1_label_align_violations": 0,
                     "I2_optimizer_step_ratio": 1.0,
                     "I2_param_delta_norm": 0.1,
                     "I3_cross_rank_grad_delta": 0.3}),
])
def test_classify_maps_each_signature_to_one_mode(mode, obs):
    assert FA.classify(obs)["verdict"] == mode


def test_decision_table_has_one_yes_per_fault_row():
    for row in FA.DECISION_TABLE_ROWS:
        name, i1, i2, i3 = row
        yes = [i1, i2, i3].count("是")
        assert yes == (0 if name == "none" else 1), row


# ---------------------------------------------------------------------------
# 端到端
# ---------------------------------------------------------------------------


def fault_worker(rank: int, world_size: int, port: int, out_dir: str,
                 config_path: str, mode: str, dtype: str) -> None:
    H.worker_setup(rank, world_size, port)
    from mm_v100 import bounded_train as BT

    args = BT.build_parser().parse_args([
        "--config", config_path,
        "--out-dir", os.path.join(out_dir, "logs"),
        "--placement", "ddp" if world_size > 1 else "single",
        "--stage", "sft",
        "--backend", "gloo",
        "--force-cpu",
        "--dtype", dtype,
        "--global-batch", str(GLOBAL_BATCH),
        "--accum", "1",
        "--max-steps", str(STEPS),
        "--seed", "42",
        "--seed-per-rank", "0",
        "--fault", mode,
        "--log-name", "fault_" + mode,
        "--log-interval", str(STEPS),
    ])
    summary = BT.train(args)
    H.write_result(out_dir, rank, {
        "mode": mode,
        "classification": summary["classification"],
        "losses": summary["losses"],
        "skipped_steps": summary["skipped_steps"],
    })


def run_fault(spawn_dist, tmp_path, config_path, mode, world_size,
              dtype="float16"):
    out = str(tmp_path / (mode + "_ws" + str(world_size)))
    results = spawn_dist(fault_worker, world_size, out,
                         extra=(config_path, mode, dtype))
    return results[0]["classification"], results[0]


def test_control_group_is_clean(tmp_path, tiny_config_path, spawn_dist):
    cls, res = run_fault(spawn_dist, tmp_path, tiny_config_path, "none", 2)
    obs = cls["observations"]
    assert cls["verdict"] == "none", cls
    assert obs["I1_label_align_violations"] == 0
    assert obs["I2_optimizer_step_ratio"] == pytest.approx(1.0)
    assert obs["I3_cross_rank_grad_delta"] == pytest.approx(0.0, abs=1e-9)
    assert res["skipped_steps"] == 0


def test_label_shift_only_trips_i1(tmp_path, tiny_config_path, spawn_dist):
    cls, _ = run_fault(spawn_dist, tmp_path, tiny_config_path, "label_shift", 2)
    obs = cls["observations"]
    assert cls["verdict"] == "label_shift", cls
    assert obs["I1_label_align_violations"] > 0
    assert obs["I2_optimizer_step_ratio"] == pytest.approx(1.0)
    assert obs["I2_param_delta_norm"] > 0.0
    assert obs["I3_cross_rank_grad_delta"] == pytest.approx(0.0, abs=1e-9)
    assert cls["all_hits"] == ["label_shift"]


def test_scaler_stuck_only_trips_i2(tmp_path, tiny_config_path, spawn_dist):
    cls, res = run_fault(spawn_dist, tmp_path, tiny_config_path,
                         "scaler_stuck", 2)
    obs = cls["observations"]
    assert cls["verdict"] == "scaler_stuck", cls
    assert obs["I1_label_align_violations"] == 0
    assert obs["I2_optimizer_step_ratio"] == pytest.approx(0.0)
    assert obs["I2_param_delta_norm"] == pytest.approx(0.0)
    assert res["skipped_steps"] == STEPS
    assert cls["all_hits"] == ["scaler_stuck"]


def test_ddp_no_sync_only_trips_i3(tmp_path, tiny_config_path, spawn_dist):
    cls, _ = run_fault(spawn_dist, tmp_path, tiny_config_path, "ddp_no_sync", 2)
    obs = cls["observations"]
    assert cls["verdict"] == "ddp_no_sync", cls
    assert obs["I1_label_align_violations"] == 0
    assert obs["I2_optimizer_step_ratio"] == pytest.approx(1.0)
    assert obs["I3_cross_rank_grad_delta"] > 1e-6
    assert cls["all_hits"] == ["ddp_no_sync"]


def test_loss_alone_cannot_separate_the_faults(tmp_path, tiny_config_path,
                                               spawn_dist):
    """本周的主命题：四档的 loss 都是有限值、都在同一量级。

    如果这条断言失败（比如某一档的 loss 变成 nan），那这一档就不再「静默」，
    判别表也就没有存在的必要了——所以它是一条关于**实验设计**的断言。
    """
    losses = {}
    for mode in ("none", "label_shift", "scaler_stuck", "ddp_no_sync"):
        _, res = run_fault(spawn_dist, tmp_path, tiny_config_path, mode, 2)
        losses[mode] = res["losses"]
    for mode, values in losses.items():
        assert all(v == v for v in values), (mode, values)  # 非 nan
        assert all(0.0 < v < 20.0 for v in values), (mode, values)
