"""通信账：计数器要在真实 gloo 通信下数得准，手算表要与实测对得上。

CPU + gloo 能验的是「计数器本身对不对」和「手算公式的结构对不对」。
NCCL 上的实际次数（尤其是 DDP 的 C++ reducer 那部分）只能在 V100 上取。
"""

from __future__ import annotations

import os

import pytest
import torch
import torch.distributed as dist

import conftest as H
from mm_v100 import collectives as CO


def test_meter_counts_and_bytes_without_process_group():
    """不建进程组也要能包住函数名；调用会失败，但计数逻辑本身可以单独验。"""
    meter = CO.CollectiveMeter()
    assert meter.total_calls() == 0
    assert meter.summary()["counts"] == {}


def test_meter_restores_original_functions():
    original = dist.all_reduce
    with CO.CollectiveMeter():
        assert dist.all_reduce is not original
    assert dist.all_reduce is original


def collective_worker(rank: int, world_size: int, port: int,
                      out_dir: str) -> None:
    H.worker_setup(rank, world_size, port)
    import torch.distributed as d

    from mm_v100 import collectives as C2
    from mm_v100 import common as CC

    info = CC.init_dist(backend="gloo", force_cpu=True)
    with C2.CollectiveMeter() as meter:
        t = torch.ones(256, dtype=torch.float32) * (rank + 1)
        d.all_reduce(t, op=d.ReduceOp.SUM)
        d.broadcast(t, src=0)
        gathered = [torch.zeros_like(t) for _ in range(world_size)]
        d.all_gather(gathered, t)
        summary = meter.summary()
    H.write_result(out_dir, rank, {
        "summary": summary,
        "reduced_value": float(t[0].item()),
        "n_gathered": len(gathered),
    })
    CC.cleanup_dist(info)


def test_meter_counts_real_gloo_collectives(tmp_path, spawn_dist):
    out = str(tmp_path / "collectives")
    results = spawn_dist(collective_worker, 2, out)
    summary = results[0]["summary"]
    assert summary["counts"]["all_reduce"] == 1
    assert summary["counts"]["broadcast"] == 1
    assert summary["counts"]["all_gather"] == 1
    assert summary["total_calls"] == 3
    # 每次 payload 都是 256 个 float32 = 1024 字节。
    assert summary["bytes"]["all_reduce"] == 256 * 4
    assert summary["bytes"]["all_gather"] == 256 * 4
    assert summary["total_bytes"] == 3 * 256 * 4
    # 顺带确认通信真的发生了：1+2=3。
    assert results[0]["reduced_value"] == pytest.approx(3.0)


def train_step_worker(rank: int, world_size: int, port: int, out_dir: str,
                      config_path: str) -> None:
    """在一个真实的 DDP 训练步里数通信。"""
    H.worker_setup(rank, world_size, port)
    from mm_v100 import bounded_train as BT

    args = BT.build_parser().parse_args([
        "--config", config_path,
        "--out-dir", os.path.join(out_dir, "logs"),
        "--placement", "ddp",
        "--backend", "gloo",
        "--force-cpu",
        "--dtype", "float32",
        "--global-batch", "8",
        "--accum", "2",
        "--max-steps", "2",
        "--seed", "42",
        "--seed-per-rank", "0",
        "--count-collectives", "1",
        "--log-name", "coll",
        "--log-interval", "2",
    ])
    BT.train(args)
    from mm_v100 import common as CC

    rows = [r for r in CC.read_jsonl(
        os.path.join(out_dir, "logs", "coll_rank" + str(rank) + ".jsonl"))
        if r.get("event") != "header"]
    H.write_result(out_dir, rank,
                   {"collectives": [r["collectives"] for r in rows]})


def test_training_step_records_collectives(tmp_path, tiny_config_path,
                                           spawn_dist):
    out = str(tmp_path / "train_coll")
    results = spawn_dist(train_step_worker, 2, out, extra=(tiny_config_path,))
    per_step = results[0]["collectives"]
    assert len(per_step) == 2
    for step in per_step:
        # 每步至少有一次 all_reduce（loss 的全局归约由本 lab 自己发出，
        # 一定经过 Python 侧，所以一定数得到）。
        assert step["counts"].get("all_reduce", 0) >= 1
        assert step["total_bytes"] > 0


def test_ddp_expected_bucket_math():
    exp = CO.ddp_expected(n_params=100_000_000, grad_dtype_bytes=4,
                          bucket_cap_mb=25.0)
    assert exp["grad_bytes_per_rank"] == 400_000_000
    # 400 MB / 25 MB = 16 个 bucket
    assert exp["expected_all_reduce_calls"] == 16


def test_fsdp_expected_counts_by_strategy():
    full = CO.fsdp_expected(1_000_000, n_units=9, world_size=8,
                            sharding="full_shard")
    sgo = CO.fsdp_expected(1_000_000, n_units=9, world_size=8,
                           sharding="shard_grad_op")
    none = CO.fsdp_expected(1_000_000, n_units=9, world_size=8,
                            sharding="no_shard")
    assert full["expected_all_gather_calls"] == 18   # forward + backward
    assert full["expected_reduce_scatter_calls"] == 9
    assert sgo["expected_all_gather_calls"] == 9     # backward 不再 all_gather
    assert sgo["expected_reduce_scatter_calls"] == 9
    assert none["expected_all_gather_calls"] == 0
    # 每次 all_gather 发出的是自己那一片：unit_params / world_size。
    assert full["all_gather_payload_bytes_each"] == int(1_000_000 / 9 / 8 * 2)


def test_fsdp_expected_rejects_unknown_strategy():
    with pytest.raises(ValueError):
        CO.fsdp_expected(1000, n_units=2, world_size=2, sharding="magic")


def test_compare_to_expected_flags_wrap_degeneration():
    """wrap 退化成 1 个单元时，实测 all_gather 次数只有期望的 1/9。"""
    expected = CO.fsdp_expected(1_000_000, n_units=9, world_size=8,
                                sharding="full_shard")
    measured = {"counts": {"all_gather_into_tensor": 2,
                           "reduce_scatter_tensor": 1},
                "bytes": {}, "total_calls": 3, "total_bytes": 0}
    cmp = CO.compare_to_expected(measured, expected)
    assert cmp["all_within_tol"] is False
    row = next(r for r in cmp["rows"] if r["op"] == "all_gather_into_tensor")
    assert row["ratio"] < 0.2


def test_render_text_contains_totals():
    measured = {"counts": {"all_reduce": 3}, "bytes": {"all_reduce": 12},
                "total_calls": 3, "total_bytes": 12}
    text = CO.render_text(measured, CO.ddp_expected(100, 4))
    assert "TOTAL" in text
    assert "all_reduce" in text
