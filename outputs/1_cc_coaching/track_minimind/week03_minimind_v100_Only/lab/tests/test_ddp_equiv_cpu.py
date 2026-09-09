"""不变量 I1（等价）：固定全局批时，world_size 变了 loss 不变。

对照组
------
* world_size=1, accum=2 -> micro = 8/(1*2) = 4
* world_size=2, accum=1 -> micro = 8/(2*1) = 4

两边 micro 相同（batch 维上的归约顺序一致），消费的全局样本集合相同
（micro_batch_indices 保证），精度都是 float32，seed 都不带 rank 偏移
（--equiv-check 强制这两点）。所以前几步的 loss 应该逐位接近。

判定阈值 1e-4：比 gloo 的 float32 allreduce 归约顺序差异大两个数量级，
又比「数据切错了」的差异（通常 >1e-2）小两个数量级。
"""

from __future__ import annotations

import os

import pytest

import conftest as H

TOL = 1e-4
STEPS = 3
GLOBAL_BATCH = 8


def ddp_worker(rank: int, world_size: int, port: int, out_dir: str,
               config_path: str, accum: int) -> None:
    """在一个 gloo 进程里跑 bounded_train.train()，把逐步 loss 写成 JSON。"""
    H.worker_setup(rank, world_size, port)
    from mm_v100 import bounded_train as BT

    args = BT.build_parser().parse_args([
        "--config", config_path,
        "--out-dir", os.path.join(out_dir, "ws" + str(world_size)),
        "--placement", "ddp" if world_size > 1 else "single",
        "--backend", "gloo",
        "--force-cpu",
        "--dtype", "float32",
        "--global-batch", str(GLOBAL_BATCH),
        "--accum", str(accum),
        "--max-steps", str(STEPS),
        "--seed", "42",
        "--seed-per-rank", "0",
        "--equiv-check",
        "--log-name", "equiv_ws" + str(world_size),
        "--log-interval", str(STEPS),
    ])
    summary = BT.train(args)
    H.write_result(out_dir, rank, {
        "world_size": world_size,
        "losses": summary["losses"],
        "micro_batch": summary["micro_batch"],
        "steps": summary["steps"],
    })


def test_fixed_global_batch_equivalence(tmp_path, tiny_config_path, spawn_dist):
    out_a = str(tmp_path / "a")
    out_b = str(tmp_path / "b")
    res_a = spawn_dist(ddp_worker, 1, out_a, extra=(tiny_config_path, 2))
    res_b = spawn_dist(ddp_worker, 2, out_b, extra=(tiny_config_path, 1))

    assert res_a[0]["micro_batch"] == res_b[0]["micro_batch"] == 4
    losses_a = res_a[0]["losses"]
    losses_b = res_b[0]["losses"]
    assert len(losses_a) == len(losses_b) == STEPS

    deltas = [abs(a - b) for a, b in zip(losses_a, losses_b)]
    assert max(deltas) < TOL, (
        "world_size 1 与 2 的 loss 不等价：deltas=" + str(deltas)
        + "  A=" + str(losses_a) + "  B=" + str(losses_b))


def test_all_ranks_report_the_same_global_loss(tmp_path, tiny_config_path,
                                               spawn_dist):
    """全局 loss 是 all_reduce 之后再除的，所以每个 rank 打出来的必须一样。

    这条与上一条是不同的东西：上一条验「换 world_size 结果不变」，
    这一条验「同一次运行里各 rank 看到的是同一个数」。后者不成立时，
    rank0 的日志会变得没有代表性——那正是故障 C 的伪装方式。
    """
    out = str(tmp_path / "same")
    results = spawn_dist(ddp_worker, 2, out, extra=(tiny_config_path, 1))
    assert len(results) == 2
    for a, b in zip(results[0]["losses"], results[1]["losses"]):
        assert a == pytest.approx(b, abs=1e-9)
