"""Day 3 分片 checkpoint：8 卡写 -> 4 卡读，step 换算，loss 连续性判定。

来源
----
从 week02 lab/src/mm_dist/ckpt_reshard.py 复制并裁剪：保留 dcp_save/dcp_load 的
双 API 兼容写法、MiniMind 的 step 换算、loss 连续性判据；去掉了必须有 CUDA 的
FSDP 上下文管理（那部分移到 fsdp_sharded_save/load 两个函数里，明确标注只能在
V100 上跑）。新增 FlatShardCheckpoint —— 一个不依赖 FSDP、CPU 上能真跑的分片格式。

两条路径
--------
A. FlatShardCheckpoint（本文件自带，CPU/GPU 通用，本机已实测）
   把整份参数摊平成一个一维 fp32 向量，按 world_size 均分，每 rank 写自己那片。
   读的时候把所有片拼回整份，再按新的 world_size 重新切。
   这就是 resharding 的全部内容：**分片是存储布局，不是模型语义。**
   本机 2 进程写 / 1 进程读的往返等价是真测试，不是 smoke。

B. FSDP SHARDED_STATE_DICT + torch.distributed.checkpoint（只能在 V100 上跑）
   生产路径。它多做的两件事是：优化器状态也分片、每片带 metadata 描述自己在
   全局张量里的位置。API 名在 torch 2.1 与更新版本之间变过，dcp_save/dcp_load
   两个都试。

step 换算
---------
MiniMind trainer/trainer_utils.py L110-113：step = step * saved_ws // current_ws。
这条换算只在「每 step 每 rank 样本数不变」时成立。本 lab 固定 global_batch，
真正消费的样本数与 world_size 无关，所以换算后的 step 只用来对齐 MiniMind 的
日志语义，不改变数据游标——consumed_samples 才是数据侧的真相。

loss 连续性怎么判才有意义
-------------------------
必须固定三件事才可比：用哪一批数据、什么精度、怎么归约。所以 verify 用的是
「第 saved_step+1 步那一批全局 batch 的 forward-only loss」，数据由
micro_batch_indices 从 step 算出来，与 world_size 无关；loss 先 all_reduce(SUM)
再除以 world_size*accum，也与 world_size 无关。剩下的差异就只剩分片恢复本身。
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist

from . import common as C

META_NAME = "meta.json"
SHARD_PREFIX = "shard_"


# ---------------------------------------------------------------------------
# 摊平 / 还原
# ---------------------------------------------------------------------------


def flatten_state(model: torch.nn.Module) -> Tuple[torch.Tensor,
                                                   List[Dict[str, Any]]]:
    """把 state_dict 摊平成一维 fp32 向量 + 索引表。

    统一转 fp32：分片格式要能在 fp16 训练与 fp32 参考之间来回读，
    存储精度不跟着计算精度走，否则「换 world_size 恢复」会掺进一次精度损失，
    loss 连续性就分不清是分片错了还是精度掉了。
    """
    index: List[Dict[str, Any]] = []
    chunks: List[torch.Tensor] = []
    offset = 0
    for name, tensor in model.state_dict().items():
        flat = tensor.detach().reshape(-1).float().cpu()
        index.append({"name": name, "shape": list(tensor.shape),
                      "offset": offset, "numel": int(flat.numel()),
                      "dtype": str(tensor.dtype)})
        chunks.append(flat)
        offset += int(flat.numel())
    if not chunks:
        raise ValueError("模型 state_dict 是空的，没有东西可存")
    return torch.cat(chunks), index


def unflatten_into(model: torch.nn.Module, flat: torch.Tensor,
                   index: List[Dict[str, Any]]) -> None:
    """把一维向量按索引表写回模型。形状/总长不匹配时报错说清楚差在哪。"""
    expected = sum(item["numel"] for item in index)
    if int(flat.numel()) != expected:
        raise ValueError(
            "扁平向量长度 " + str(int(flat.numel())) + " 与索引表要求的 "
            + str(expected) + " 不一致。\n"
            "下一步：多半是分片文件缺了一片，或者 world_size 与 meta.json 里"
            "记录的对不上。先看 meta.json 的 world_size 与 shard_bounds。")
    state = model.state_dict()
    new_state: Dict[str, torch.Tensor] = {}
    for item in index:
        piece = flat[item["offset"]: item["offset"] + item["numel"]]
        target = state[item["name"]]
        new_state[item["name"]] = piece.reshape(item["shape"]).to(
            dtype=target.dtype, device=target.device)
    model.load_state_dict(new_state)


def shard_bounds(total: int, world_size: int, rank: int) -> Tuple[int, int]:
    """均分，余数分给前面的 rank。返回 [start, end)。

    余数处理必须和读端完全一致，否则拼回来的向量会错位——那种错误表现为
    loss 突然跳到随机初始化的水平，而不是报错。
    """
    if world_size <= 0 or not (0 <= rank < world_size):
        raise ValueError("rank/world_size 不合法：rank=" + str(rank)
                         + " world_size=" + str(world_size))
    base = total // world_size
    extra = total % world_size
    start = rank * base + min(rank, extra)
    end = start + base + (1 if rank < extra else 0)
    return start, end


# ---------------------------------------------------------------------------
# FlatShardCheckpoint
# ---------------------------------------------------------------------------


def save_flat_shard(out_dir: str, model: torch.nn.Module, rank: int,
                    world_size: int, meta_extra: Optional[Dict[str, Any]] = None,
                    write_meta: bool = True) -> str:
    """每 rank 写自己那一片。rank0 额外写 meta.json（索引表 + 训练元信息）。"""
    os.makedirs(out_dir, exist_ok=True)
    flat, index = flatten_state(model)
    start, end = shard_bounds(int(flat.numel()), world_size, rank)
    piece = flat[start:end].clone()
    path = os.path.join(out_dir, SHARD_PREFIX + str(rank) + ".pt")
    torch.save({"rank": int(rank), "world_size": int(world_size),
                "start": int(start), "end": int(end), "data": piece}, path)
    if write_meta and rank == 0:
        meta = dict(meta_extra or {})
        meta.update({
            "format": "flat_shard/1",
            "world_size": int(world_size),
            "total_numel": int(flat.numel()),
            "index": index,
            "env": C.env_summary(),
        })
        with open(os.path.join(out_dir, META_NAME), "w", encoding="utf-8") as fh:
            json.dump(meta, fh, ensure_ascii=False, indent=2, sort_keys=True)
    return path


def read_meta(ckpt_dir: str) -> Dict[str, Any]:
    path = os.path.join(ckpt_dir, META_NAME)
    if not os.path.isfile(path):
        raise FileNotFoundError(
            "找不到 " + path + "。\n"
            "下一步：先用 save_flat_shard(..., rank=0) 或 "
            "lab/scripts/reshard_check.py save 生成。")
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def load_flat_full(ckpt_dir: str) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """把所有分片拼回整份向量。缺片时报出缺的是哪几个 rank。"""
    meta = read_meta(ckpt_dir)
    saved_ws = int(meta["world_size"])
    total = int(meta["total_numel"])
    full = torch.zeros(total, dtype=torch.float32)
    missing: List[int] = []
    for r in range(saved_ws):
        path = os.path.join(ckpt_dir, SHARD_PREFIX + str(r) + ".pt")
        if not os.path.isfile(path):
            missing.append(r)
            continue
        blob = torch.load(path, map_location="cpu", weights_only=False)
        full[int(blob["start"]): int(blob["end"])] = blob["data"]
    if missing:
        raise FileNotFoundError(
            "缺少分片：" + ", ".join(SHARD_PREFIX + str(r) + ".pt" for r in missing)
            + "\n下一步：这些 rank 在保存时可能已经挂了。检查它们的 stderr；"
              "本 lab 的 save 是每 rank 各写各的，任何一个 rank 失败都会缺一片。")
    return full, meta


def load_flat_into(model: torch.nn.Module, ckpt_dir: str) -> Dict[str, Any]:
    """以当前 world_size 加载（每个进程都读整份再自己切，语义最简单）。"""
    full, meta = load_flat_full(ckpt_dir)
    unflatten_into(model, full, meta["index"])
    return meta


def reshard_slice(ckpt_dir: str, new_world_size: int,
                  rank: int) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """按新的 world_size 取出本 rank 应得的那一片。resharding 的核心就是这一句。"""
    full, meta = load_flat_full(ckpt_dir)
    start, end = shard_bounds(int(full.numel()), new_world_size, rank)
    return full[start:end].clone(), meta


# ---------------------------------------------------------------------------
# step 换算
# ---------------------------------------------------------------------------


def convert_step(saved_step: int, saved_ws: int, current_ws: int) -> Dict[str, Any]:
    """复刻 MiniMind trainer_utils.py L110-113 的换算，并显式给出它的前提。"""
    if current_ws <= 0:
        raise ValueError("current_ws 必须 > 0")
    converted = (int(saved_step) * int(saved_ws) // int(current_ws)
                 if saved_ws != current_ws else int(saved_step))
    return {
        "saved_step": int(saved_step),
        "saved_world_size": int(saved_ws),
        "current_world_size": int(current_ws),
        "converted_step": int(converted),
        "assumption": "每 step 每 rank 样本数不变；固定 global_batch 时该假设不成立",
        "data_side_truth": "consumed_samples = saved_step * global_batch，与 ws 无关",
    }


# ---------------------------------------------------------------------------
# loss 连续性
# ---------------------------------------------------------------------------


def global_forward_loss(model: torch.nn.Module, dataset, step: int,
                        global_batch: int, accum: int, world_size: int,
                        rank: int, device: str, dtype_name: str = "float32",
                        sdpa_backend: str = "auto") -> float:
    """第 step 步那一批全局 batch 的 forward-only loss，与 world_size 无关。"""
    torch_dtype = C.assert_dtype_supported(dtype_name, device)
    was_training = model.training
    model.eval()
    total = torch.zeros(1, device=device, dtype=torch.float32)
    with torch.no_grad():
        for acc_idx in range(accum):
            idx = C.micro_batch_indices(step, global_batch, world_size, accum,
                                        rank, acc_idx)
            input_ids, labels = C.collate_indices(dataset, idx, device)
            with C.sdpa_context(sdpa_backend, device):
                with C.autocast_context(torch_dtype, device):
                    out = model(input_ids, labels=labels)
            total[0] += out.loss.detach().float()
    if world_size > 1 and dist.is_available() and dist.is_initialized():
        dist.all_reduce(total, op=dist.ReduceOp.SUM)
    if was_training:
        model.train()
    return float(total[0].item() / float(world_size * accum))


def verify_continuity(expected_loss: float, actual_loss: float,
                      tol: float = 1e-4) -> Dict[str, Any]:
    """恢复后同一批数据的 loss 应当与保存时算的那个几乎相同。"""
    delta = abs(float(actual_loss) - float(expected_loss))
    return {
        "expected_loss": float(expected_loss),
        "actual_loss": float(actual_loss),
        "abs_delta": delta,
        "tol": float(tol),
        "pass": bool(delta < tol),
        "hint": ("失败时第一个检查：meta.json 的 total_numel 与当前模型的参数总数"
                 "是否一致（配置改过？）；第二个检查：shard_bounds 的余数分配"
                 "在读写两端是否一致。"),
    }


# ---------------------------------------------------------------------------
# torch.distributed.checkpoint（V100 路径，本机未执行）
# ---------------------------------------------------------------------------


def dcp_save(state_dict: Dict[str, Any], path: str) -> str:
    """torch 2.1 的写法是 save_state_dict + FileSystemWriter。

    2.2 之后 save_state_dict 被标 deprecated（本机的新版 torch 会打 FutureWarning），
    此时退回同语义的 save。两个都试是因为这份代码要同时在 2.1 和本机跑。
    """
    import torch.distributed.checkpoint as dcp

    os.makedirs(path, exist_ok=True)
    writer = dcp.FileSystemWriter(path)
    fn = getattr(dcp, "save_state_dict", None)
    if fn is not None:
        try:
            fn(state_dict=state_dict, storage_writer=writer)
            return path
        except TypeError:
            pass
    dcp.save(state_dict, storage_writer=writer)
    return path


def dcp_load(state_dict: Dict[str, Any], path: str) -> Dict[str, Any]:
    import torch.distributed.checkpoint as dcp

    reader = dcp.FileSystemReader(path)
    fn = getattr(dcp, "load_state_dict", None)
    if fn is not None:
        try:
            fn(state_dict=state_dict, storage_reader=reader)
            return state_dict
        except TypeError:
            pass
    dcp.load(state_dict, storage_reader=reader)
    return state_dict


FSDP_ONLY_NOTE = (
    "这条路径需要 CUDA + 已初始化的进程组；cc 的开发机没有 GPU，"
    "所以它在本机从未执行过。第一次在 V100 上跑之前，先用 "
    "lab/scripts/reshard_check.py --format flat 把 flat_shard 路径跑通，"
    "确认数据切分与 loss 连续性判据本身没问题，再换 --format fsdp。")


def fsdp_sharded_save(model, optimizer, path: str) -> str:
    """FSDP SHARDED_STATE_DICT + DCP。只能在 CUDA 上跑。"""
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp import (ShardedOptimStateDictConfig,
                                        ShardedStateDictConfig, StateDictType)

    with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT,
                              ShardedStateDictConfig(offload_to_cpu=True),
                              ShardedOptimStateDictConfig(offload_to_cpu=True)):
        state = {"model": model.state_dict(),
                 "optim": FSDP.optim_state_dict(model, optimizer)}
    return dcp_save(state, path)


def fsdp_sharded_load(model, optimizer, path: str) -> None:
    """以当前 world_size 读回 FSDP 分片 checkpoint。只能在 CUDA 上跑。"""
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp import (ShardedOptimStateDictConfig,
                                        ShardedStateDictConfig, StateDictType)

    with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT,
                              ShardedStateDictConfig(offload_to_cpu=True),
                              ShardedOptimStateDictConfig(offload_to_cpu=True)):
        state = {"model": model.state_dict(),
                 "optim": FSDP.optim_state_dict(model, optimizer)}
        dcp_load(state, path)
        model.load_state_dict(state["model"])
        flat = FSDP.optim_state_dict_to_load(model, optimizer, state["optim"])
        optimizer.load_state_dict(flat)
