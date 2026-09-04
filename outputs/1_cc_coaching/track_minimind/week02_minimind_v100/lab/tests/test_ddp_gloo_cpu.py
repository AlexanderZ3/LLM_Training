"""不变量 I1：固定 global batch 时，world_size 变了 loss 不变。

对照组
------
* ``world_size=1, accum=4`` → micro = 8/(1*4) = 2
* ``world_size=2, accum=2`` → micro = 8/(2*2) = 2

两边 micro 相同（batch 维度上的归约顺序一致），消费的全局样本集合相同
（``common.micro_batch_indices`` 保证），精度都是 float32，seed 都不带 rank 偏移
（``--equiv-check`` 会强制这两点）。所以前几步的 loss 应该逐位接近，
判定阈值 1e-4——它比 gloo 的 float32 allreduce 归约顺序差异大两个数量级，
又比"数据切错了"的差异（通常 >1e-2）小两个数量级。
"""

from __future__ import annotations

import json
import os

import pytest

import conftest as H

TOL = 1e-4
STEPS = 3


def ddp_worker(rank: int, world_size: int, port: int, out_dir: str,
               config: str, accum: int) -> None:
    """在一个 gloo 进程里跑 train_ddp.train()，把逐步 loss 写成 JSON。"""
    H.worker_setup(rank, world_size, port)
    from mm_dist import train_ddp

    args = train_ddp.build_parser().parse_args([
        "--config", config,
        "--out-dir", os.path.join(out_dir, "logs_ws" + str(world_size)),
        "--backend", "gloo",
        "--force-cpu",
        "--dtype", "float32",
        "--global-batch", "8",
        "--accum", str(accum),
        "--seq-len", "32",
        "--max-steps", str(STEPS),
        "--num-samples", "256",
        "--seed", "42",
        "--seed-per-rank", "0",
        "--equiv-check",
        "--log-name", "equiv_ws" + str(world_size),
    ])
    summary = train_ddp.train(args)
    H.write_result(out_dir, rank, {
        "world_size": world_size,
        "losses": summary["losses"],
        "micro_batch": summary["micro_batch"],
        "log_path": summary["log_path"],
    })


def test_fixed_global_batch_equivalence(tmp_path, tiny_config, minimind_root,
                                        spawn_dist):
    dir_one = str(tmp_path / "ws1")
    dir_two = str(tmp_path / "ws2")
    res_one = spawn_dist(ddp_worker, 1, dir_one, extra=(tiny_config, 4))
    res_two = spawn_dist(ddp_worker, 2, dir_two, extra=(tiny_config, 2))

    assert res_one[0]["micro_batch"] == res_two[0]["micro_batch"] == 2, (
        "micro batch 必须相同，否则比较的是两个不同的数值问题")
    losses_one = res_one[0]["losses"]
    for rank_result in res_two:
        assert rank_result["losses"] == res_two[0]["losses"], (
            "同一次 world_size=2 运行里，各 rank 记录的全局 loss 必须完全相同"
            "（它是 all_reduce 之后的量）")
    losses_two = res_two[0]["losses"]

    assert len(losses_one) == len(losses_two) == STEPS
    deltas = [abs(a - b) for a, b in zip(losses_one, losses_two)]
    assert max(deltas) < TOL, (
        "world_size 1 vs 2 的逐步 loss 差 " + json.dumps(deltas)
        + " 超过容差 " + str(TOL) + "；先查 micro_batch_indices 的样本集合是否一致")


def test_micro_batch_indices_partition_global_batch():
    """I1 的数据侧：任意 (W, accum) 切法的并集都等于同一段全局索引区间。"""
    from mm_dist import common as C

    global_batch, step = 8, 3
    expected = set(range((step - 1) * global_batch, step * global_batch))
    for world_size, accum in ((1, 1), (1, 4), (2, 2), (4, 1), (8, 1), (2, 4)):
        collected = []
        for rank in range(world_size):
            for acc_idx in range(accum):
                collected.extend(C.micro_batch_indices(
                    step, global_batch, world_size, accum, rank, acc_idx))
        assert len(collected) == global_batch, (world_size, accum)
        assert set(collected) == expected, (world_size, accum)


def test_micro_batch_indices_rejects_non_divisible():
    from mm_dist import common as C

    with pytest.raises(ValueError):
        C.micro_batch_indices(1, 8, 3, 1, 0, 0)


def test_compare_logs_reports_pass_and_fail(tmp_path):
    """日志比对工具本身的行为：同一条日志比自己必然 PASS，差 0.5 必然 FAIL。"""
    from mm_dist import common as C
    from mm_dist.train_ddp import compare_logs

    path_a = str(tmp_path / "a.jsonl")
    path_b = str(tmp_path / "b.jsonl")
    with open(path_a, "w", encoding="utf-8") as fh:
        for step in range(1, 4):
            fh.write(json.dumps({"step": step, "loss": 8.0 - 0.1 * step}) + "\n")
    with open(path_b, "w", encoding="utf-8") as fh:
        for step in range(1, 4):
            fh.write(json.dumps({"step": step, "loss": 8.5 - 0.1 * step}) + "\n")

    same = compare_logs(path_a, path_a, tol=TOL)
    assert same["pass"] is True
    assert same["max_abs_delta"] == 0.0
    diff = compare_logs(path_a, path_b, tol=TOL)
    assert diff["pass"] is False
    assert abs(diff["max_abs_delta"] - 0.5) < 1e-9
    assert len(C.read_jsonl(path_a)) == 3
