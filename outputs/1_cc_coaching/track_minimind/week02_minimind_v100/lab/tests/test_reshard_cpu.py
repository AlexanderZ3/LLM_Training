"""分片 checkpoint 的 CPU 侧验证。

本机能验证什么、不能验证什么
----------------------------
**不能**：FSDP1 需要 CUDA 设备。torch 的 ``_init_device_handle`` 在参数全在 CPU 上时
会回落到 ``torch.cuda.current_device()``，没有 GPU 就抛
``FSDP needs a non-CPU accelerator device``。torch 2.1.0 与本机的新版 torch 在这一点上
行为一致，所以"FSDP 存 → 换 world_size 取"这条完整链路只能在公司 8×V100 上跑
（``mm_dist.ckpt_reshard`` 的 ``require_cuda_for_fsdp`` 会在建 FSDP 之前就说明这件事）。

**能**：
1. ``torch.distributed.checkpoint`` 的 ``save_state_dict`` / ``load_state_dict`` +
   ``FileSystemWriter`` / ``FileSystemReader`` 本身能不能跨 world_size 读写——
   2 进程写、1 进程读，值必须一致。这是 reshard 机制的地基。
2. MiniMind 的 step 换算公式。
3. FSDP 入口在 CPU 上给出的是可执行的错误信息，而不是深处的 AttributeError。
"""

from __future__ import annotations

import os

import pytest
import torch

import conftest as H

FSDP_CPU_MARKERS = ("non-CPU accelerator", "FSDP1 需要 CUDA")


def dcp_save_worker(rank: int, world_size: int, port: int, out_dir: str,
                    ckpt_dir: str) -> None:
    """每个 rank 写一份自己的张量；dcp 负责把它们合成一个可跨 world_size 读的目录。"""
    H.worker_setup(rank, world_size, port)
    import torch.distributed as dist

    from mm_dist import ckpt_reshard

    dist.init_process_group("gloo")
    state = {
        "shared": torch.arange(8, dtype=torch.float32),
        "per_rank_" + str(rank): torch.full((4,), float(rank) + 0.5),
    }
    ckpt_reshard._dcp_save(state, ckpt_dir)
    H.write_result(out_dir, rank, {"ok": True, "keys": sorted(state)})
    dist.barrier()
    dist.destroy_process_group()


def dcp_load_worker(rank: int, world_size: int, port: int, out_dir: str,
                    ckpt_dir: str, saved_world_size: int) -> None:
    H.worker_setup(rank, world_size, port)
    import torch.distributed as dist

    from mm_dist import ckpt_reshard

    dist.init_process_group("gloo")
    state = {"shared": torch.zeros(8, dtype=torch.float32)}
    for r in range(saved_world_size):
        state["per_rank_" + str(r)] = torch.zeros(4, dtype=torch.float32)
    ckpt_reshard._dcp_load(state, ckpt_dir)
    H.write_result(out_dir, rank, {
        "shared": state["shared"].tolist(),
        "per_rank": {str(r): state["per_rank_" + str(r)].tolist()
                     for r in range(saved_world_size)},
    })
    dist.destroy_process_group()


def test_dcp_roundtrip_across_world_sizes(tmp_path, spawn_dist):
    """2 进程写、1 进程读：分片 checkpoint 的存取本身与 world_size 解耦。"""
    ckpt_dir = str(tmp_path / "dcp_ckpt")
    saved = spawn_dist(dcp_save_worker, 2, str(tmp_path / "save"),
                       extra=(ckpt_dir,))
    assert all(r["ok"] for r in saved)
    assert os.path.isdir(ckpt_dir) and os.listdir(ckpt_dir)

    loaded = spawn_dist(dcp_load_worker, 1, str(tmp_path / "load"),
                        extra=(ckpt_dir, 2))
    result = loaded[0]
    assert result["shared"] == list(range(8))
    assert result["per_rank"]["0"] == [0.5] * 4
    assert result["per_rank"]["1"] == [1.5] * 4


def fsdp_cpu_guard_worker(rank: int, world_size: int, port: int, out_dir: str,
                          config: str) -> None:
    """在 CPU 上调 FSDP 入口，记录它给出的错误信息。"""
    H.worker_setup(rank, world_size, port)
    from mm_dist import ckpt_reshard

    args = ckpt_reshard.build_parser().parse_args([
        "save", "--config", config, "--ckpt-dir", os.path.join(out_dir, "ckpt"),
        "--out-dir", os.path.join(out_dir, "logs"), "--backend", "gloo",
        "--force-cpu", "--dtype", "float32", "--global-batch", "8",
        "--accum", "1", "--seq-len", "32", "--max-steps", "1",
        "--num-samples", "64",
    ])
    message = ""
    kind = "no_error"
    try:
        ckpt_reshard.cmd_save(args)
    except SystemExit as exc:
        kind = "SystemExit"
        message = str(exc)
    except RuntimeError as exc:
        kind = "RuntimeError"
        message = str(exc)
    H.write_result(out_dir, rank, {"kind": kind, "message": message[:600]})


def test_fsdp_entrypoint_explains_cpu_limitation(tmp_path, tiny_config,
                                                 minimind_root, spawn_dist):
    """CPU 上必须是一句能照着做的话，而不是 FSDP 内部的 RuntimeError。"""
    results = spawn_dist(fsdp_cpu_guard_worker, 1, str(tmp_path / "guard"),
                         extra=(tiny_config,))
    res = results[0]
    if torch.cuda.is_available():
        pytest.skip("本机有 CUDA，这条测试只在无 GPU 的机器上有意义")
    assert res["kind"] == "SystemExit", (
        "CPU 上 FSDP 入口应当抛 SystemExit 并解释原因，实际是 "
        + res["kind"] + ": " + res["message"])
    assert "FSDP1 需要 CUDA" in res["message"]
    assert "8xV100" in res["message"] or "V100" in res["message"]


def test_full_fsdp_reshard_requires_gpu():
    """完整的 FSDP 8→4 卡恢复链路在本机无法执行，如实 skip 而不是假装通过。"""
    if not torch.cuda.is_available():
        pytest.skip(
            "FSDP1 需要 CUDA：本机无 GPU，'2 卡保存 → 1 卡 verify loss 连续' "
            "这条链路必须在公司 8xV100 上执行："
            "torchrun --nproc_per_node 8 -m mm_dist.ckpt_reshard save ... 然后 "
            "torchrun --nproc_per_node 4 -m mm_dist.ckpt_reshard verify ...")
    from mm_dist import ckpt_reshard

    assert hasattr(ckpt_reshard, "cmd_verify")


def test_step_conversion_matches_minimind_formula():
    """``step * saved_ws // current_ws``：8 卡的 step 4 用 4 卡恢复应变成 8。"""
    for saved_ws, current_ws, step, expected in (
            (8, 4, 4, 8), (8, 8, 7, 7), (2, 1, 4, 8), (4, 8, 10, 5), (3, 2, 7, 10)):
        converted = step * saved_ws // current_ws
        assert converted == expected, (saved_ws, current_ws, step, converted)


def test_meta_json_contract():
    """verify 依赖 meta.json 里的这几个键；改名就会静默失效，所以钉死。"""
    from mm_dist import ckpt_reshard

    assert ckpt_reshard.META_NAME == "meta.json"
    assert ckpt_reshard.FULL_NAME == "full_state.pt"
    parser = ckpt_reshard.build_parser()
    args = parser.parse_args(["verify", "--config", "x.json"])
    assert args.command == "verify"
    assert args.format == "sharded"
    assert args.tol == 1e-3
    formats = [a for a in parser._actions if a.dest == "format"][0].choices
    assert set(formats) == {"sharded", "full"}


def test_dcp_save_load_helpers_exist():
    """本 lab 走的是 torch 2.1 的 save_state_dict/load_state_dict + FileSystem*。"""
    import torch.distributed.checkpoint as dcp

    assert hasattr(dcp, "FileSystemWriter")
    assert hasattr(dcp, "FileSystemReader")
    assert hasattr(dcp, "save_state_dict") or hasattr(dcp, "save"), (
        "torch " + torch.__version__ + " 既没有 save_state_dict 也没有 save")
