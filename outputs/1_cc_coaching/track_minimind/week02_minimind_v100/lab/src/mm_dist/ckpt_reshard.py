"""FSDP 分片 checkpoint：保存 → 换 world_size 恢复 → loss 连续性 verify。

三个子命令
----------
``save``    跑 --max-steps 步，保存 checkpoint，并把"如果再跑一步会得到的 loss"
            （即第 max_steps+1 步那一批全局 batch 的 forward-only loss）写进 meta.json。
``load``    以当前 world_size 加载 checkpoint，打印换算后的 step，不训练。
``verify``  加载后立刻算同一批（第 saved_step+1 步）的全局 loss，与 meta.json 里
            记录的 ``next_step_loss`` 比较；差值 < --tol 即判 PASS。

为什么这样验证才有意义
----------------------
"恢复后 loss 没有跳变"必须固定住三件事才可比：(1) 用哪一批数据，(2) 用什么精度，
(3) 怎么归约。本文件用 :func:`mm_dist.common.micro_batch_indices` 决定数据，
所以第 s 步的全局样本集合与 world_size 无关；loss 先 all_reduce(SUM) 再除以
world_size*accum，也与 world_size 无关。剩下的差异就只剩"分片恢复是否正确"。

两种 state dict 形态
--------------------
``--format sharded``（默认）：``StateDictType.SHARDED_STATE_DICT`` +
``torch.distributed.checkpoint.save_state_dict/load_state_dict`` +
``FileSystemWriter/FileSystemReader``。每 rank 只写自己那片，可以用任意
world_size 读回——这是 8 卡训练、4 卡恢复的正路。
``--format full``（回退）：``StateDictType.FULL_STATE_DICT`` +
``FullStateDictConfig(offload_to_cpu=True, rank0_only=True)``，rank0 用
``torch.save`` 写一个完整文件。世界大小无关，但 rank0 要装得下整份模型 + 优化器。

MiniMind 的 step 换算
---------------------
``trainer/trainer_utils.py`` L110–113：``step = step * saved_ws // current_ws``。
本文件在 ``load``/``verify`` 里复刻这条换算并打印，同时提醒它只在
"每 step 每 rank 样本数不变"时成立（01_FOUNDATIONS.md 失败模式 #10）。

只用 torch 2.1 API：``StateDictType``、``FullStateDictConfig``、
``ShardedStateDictConfig``、``FSDP.state_dict_type``、``FSDP.optim_state_dict``、
``FSDP.optim_state_dict_to_load``、``torch.distributed.checkpoint``。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.fsdp import FullOptimStateDictConfig, FullStateDictConfig
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import (
    ShardedOptimStateDictConfig,
    ShardedStateDictConfig,
    StateDictType,
)

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mm_dist import common as C  # noqa: E402
from mm_dist.train_fsdp import (  # noqa: E402
    SHARDING_STRATEGIES,
    build_auto_wrap_policy,
    make_fsdp_scaler,
    make_mixed_precision,
    require_cuda_for_fsdp,
)

META_NAME = "meta.json"
FULL_NAME = "full_state.pt"


# ---------------------------------------------------------------------------
# 组装
# ---------------------------------------------------------------------------


class ReshardContext:
    """一次 save/load/verify 共用的运行上下文。"""

    def __init__(self, args: argparse.Namespace) -> None:
        self.info = C.init_dist(backend=args.backend, timeout_s=args.nccl_timeout_s,
                                force_cpu=args.force_cpu)
        if not self.info.initialized:
            raise SystemExit(
                "ckpt_reshard 需要已初始化的进程组。单进程也要用 torchrun 启动：\n"
                "  torchrun --nproc_per_node 1 -m mm_dist.ckpt_reshard save ...")
        self.device = "cpu" if args.force_cpu else self.info.device
        require_cuda_for_fsdp(self.device)
        self.dtype_name = args.dtype
        C.assert_dtype_supported(self.dtype_name, self.device)
        C.set_seed(args.seed, per_rank=False)

        model, lm_config, cfg, mm = C.build_model(
            args.config, minimind_root=args.minimind_root,
            device="cpu" if str(self.device).startswith("cuda") else self.device)
        self.lm_config = lm_config
        self.cfg = cfg
        self.mm = mm

        train_cfg = cfg["train"]
        self.seq_len = int(args.seq_len or train_cfg["seq_len"])
        self.global_batch = int(args.global_batch or train_cfg["global_batch"])
        self.accum = int(args.accum or train_cfg.get("accum", 1))
        self.base_lr = float(args.lr if args.lr is not None
                             else train_cfg.get("lr", 5e-4))
        self.grad_clip = float(train_cfg.get("grad_clip", 1.0))
        denom = self.info.world_size * self.accum
        if self.global_batch % denom != 0:
            raise SystemExit(
                "global_batch=" + str(self.global_batch)
                + " 必须能被 world_size*accum=" + str(denom) + " 整除。")
        self.denom = denom
        self.micro = self.global_batch // denom

        fsdp_kwargs: Dict[str, Any] = dict(
            sharding_strategy=SHARDING_STRATEGIES[args.sharding],
            auto_wrap_policy=build_auto_wrap_policy(mm.MiniMindBlock),
            mixed_precision=make_mixed_precision(self.dtype_name, self.device),
            use_orig_params=True,
            limit_all_gathers=True,
            sync_module_states=True,  # full 形态 rank0_only 恢复时必须为 True
        )
        if str(self.device).startswith("cuda"):
            fsdp_kwargs["device_id"] = torch.device(self.device)
        self.model = FSDP(model, **fsdp_kwargs)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.base_lr)
        self.scaler = make_fsdp_scaler(self.dtype_name, self.device)
        self.dataset = C.build_dataset(
            args.dataset, seq_len=self.seq_len,
            vocab_size=int(lm_config.vocab_size),
            minimind_root=args.minimind_root, data_path=args.data_path,
            num_samples=int(args.num_samples),
            seed=int(train_cfg.get("data_seed", 1234)))
        self.sdpa_backend = args.sdpa_backend

    # -- 计算 ------------------------------------------------------------

    def global_loss_at(self, step: int) -> float:
        """第 ``step`` 步那一批全局 batch 的 forward-only loss（与 world_size 无关）。"""
        self.model.eval()
        sums = torch.zeros(1, device=self.device, dtype=torch.float32)
        with torch.no_grad():
            for acc_idx in range(self.accum):
                idx = C.micro_batch_indices(step, self.global_batch,
                                            self.info.world_size, self.accum,
                                            self.info.rank, acc_idx)
                input_ids, labels = C.collate_indices(self.dataset, idx, self.device)
                with C.sdpa_context(self.sdpa_backend, self.device):
                    res = self.model(input_ids, labels=labels)
                aux = res.aux_loss if res.aux_loss is not None else res.loss.new_zeros(())
                sums[0] += (res.loss + aux).detach().float()
        if self.info.world_size > 1:
            dist.all_reduce(sums, op=dist.ReduceOp.SUM)
        self.model.train()
        return float(sums[0] / float(self.denom))

    def train_steps(self, n_steps: int, log_interval: int = 1) -> List[float]:
        """跑 n_steps 个 optimizer step，返回每步全局 loss。"""
        self.model.train()
        losses: List[float] = []
        for step in range(1, n_steps + 1):
            lr = C.cosine_lr(step, n_steps, self.base_lr)
            for group in self.optimizer.param_groups:
                group["lr"] = lr
            self.optimizer.zero_grad(set_to_none=True)
            sums = torch.zeros(1, device=self.device, dtype=torch.float32)
            for acc_idx in range(self.accum):
                idx = C.micro_batch_indices(step, self.global_batch,
                                            self.info.world_size, self.accum,
                                            self.info.rank, acc_idx)
                input_ids, labels = C.collate_indices(self.dataset, idx, self.device)
                with C.sdpa_context(self.sdpa_backend, self.device):
                    res = self.model(input_ids, labels=labels)
                    aux = (res.aux_loss if res.aux_loss is not None
                           else res.loss.new_zeros(()))
                    loss = res.loss + aux
                self.scaler.scale(loss / self.accum).backward()
                sums[0] += loss.detach().float()
            if self.info.world_size > 1:
                dist.all_reduce(sums, op=dist.ReduceOp.SUM)
            self.scaler.unscale_(self.optimizer)
            self.model.clip_grad_norm_(self.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            value = float(sums[0] / float(self.denom))
            losses.append(value)
            if step % max(log_interval, 1) == 0:
                C.log_main("[train] step=%d loss=%.6f lr=%.3e" % (step, value, lr))
        return losses

    def close(self) -> None:
        C.cleanup_dist(self.info)


# ---------------------------------------------------------------------------
# 保存 / 加载
# ---------------------------------------------------------------------------


def _dcp_save(state_dict: Dict[str, Any], path: str) -> None:
    """torch 2.1 的写法：save_state_dict + FileSystemWriter。

    torch 2.1 里 ``save_state_dict`` 是公开 API；在更新的 torch 上它仍在但被标记
    deprecated（本机 2.14 会打 FutureWarning），此时退回到同语义的 ``save``。
    """
    writer = dcp.FileSystemWriter(path)
    fn = getattr(dcp, "save_state_dict", None)
    if fn is not None:
        try:
            fn(state_dict=state_dict, storage_writer=writer)
            return
        except TypeError:
            pass
    dcp.save(state_dict, storage_writer=writer)


def _dcp_load(state_dict: Dict[str, Any], path: str) -> None:
    reader = dcp.FileSystemReader(path)
    fn = getattr(dcp, "load_state_dict", None)
    if fn is not None:
        try:
            fn(state_dict=state_dict, storage_reader=reader)
            return
        except TypeError:
            pass
    dcp.load(state_dict, storage_reader=reader)


def save_checkpoint(ctx: ReshardContext, out_dir: str, fmt: str,
                    meta_extra: Dict[str, Any]) -> str:
    """写 checkpoint 与 meta.json，返回 checkpoint 目录。"""
    os.makedirs(out_dir, exist_ok=True)
    if fmt == "sharded":
        with FSDP.state_dict_type(
                ctx.model, StateDictType.SHARDED_STATE_DICT,
                ShardedStateDictConfig(offload_to_cpu=True),
                ShardedOptimStateDictConfig(offload_to_cpu=True)):
            state = {
                "model": ctx.model.state_dict(),
                "optim": FSDP.optim_state_dict(ctx.model, ctx.optimizer),
            }
        _dcp_save(state, out_dir)
    elif fmt == "full":
        with FSDP.state_dict_type(
                ctx.model, StateDictType.FULL_STATE_DICT,
                FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
                FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=True)):
            model_sd = ctx.model.state_dict()
            optim_sd = FSDP.optim_state_dict(ctx.model, ctx.optimizer)
        if ctx.info.is_main:
            torch.save({"model": model_sd, "optim": optim_sd},
                       os.path.join(out_dir, FULL_NAME))
    else:
        raise ValueError("--format 只能是 sharded / full，收到 " + repr(fmt))
    if ctx.info.is_main:
        meta = dict(meta_extra)
        meta.update({
            "format": fmt,
            "world_size": ctx.info.world_size,
            "global_batch": ctx.global_batch,
            "accum": ctx.accum,
            "micro_batch": ctx.micro,
            "seq_len": ctx.seq_len,
            "dtype": ctx.dtype_name,
            "minimind_commit": ctx.mm.commit,
            "env": C.env_summary(),
        })
        with open(os.path.join(out_dir, META_NAME), "w", encoding="utf-8") as fh:
            json.dump(meta, fh, ensure_ascii=False, indent=2, sort_keys=True)
    if ctx.info.world_size > 1:
        dist.barrier()
    return out_dir


def load_checkpoint(ctx: ReshardContext, ckpt_dir: str) -> Dict[str, Any]:
    """按 meta.json 里的 format 加载；返回 meta 字典（已补 converted_step）。"""
    meta_path = os.path.join(ckpt_dir, META_NAME)
    if not os.path.isfile(meta_path):
        raise FileNotFoundError(
            "找不到 " + meta_path + "；请先用 `ckpt_reshard save --out-dir ...` 生成。")
    with open(meta_path, "r", encoding="utf-8") as fh:
        meta = json.load(fh)
    fmt = meta.get("format", "sharded")

    if fmt == "sharded":
        with FSDP.state_dict_type(
                ctx.model, StateDictType.SHARDED_STATE_DICT,
                ShardedStateDictConfig(offload_to_cpu=True),
                ShardedOptimStateDictConfig(offload_to_cpu=True)):
            state = {
                "model": ctx.model.state_dict(),
                "optim": FSDP.optim_state_dict(ctx.model, ctx.optimizer),
            }
            _dcp_load(state, ckpt_dir)
            ctx.model.load_state_dict(state["model"])
            flat = FSDP.optim_state_dict_to_load(
                ctx.model, ctx.optimizer, state["optim"])
            ctx.optimizer.load_state_dict(flat)
    elif fmt == "full":
        blob_path = os.path.join(ckpt_dir, FULL_NAME)
        if not os.path.isfile(blob_path):
            raise FileNotFoundError("找不到 " + blob_path)
        blob = torch.load(blob_path, map_location="cpu")
        with FSDP.state_dict_type(
                ctx.model, StateDictType.FULL_STATE_DICT,
                FullStateDictConfig(offload_to_cpu=True, rank0_only=False),
                FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=False)):
            ctx.model.load_state_dict(blob["model"])
            flat = FSDP.optim_state_dict_to_load(
                ctx.model, ctx.optimizer, blob["optim"])
            ctx.optimizer.load_state_dict(flat)
    else:
        raise ValueError("meta.json 里的 format=" + repr(fmt) + " 不认识")

    saved_ws = int(meta.get("world_size", 1))
    current_ws = ctx.info.world_size
    saved_step = int(meta.get("step", 0))
    converted = saved_step * saved_ws // current_ws if saved_ws != current_ws else saved_step
    meta["saved_world_size"] = saved_ws
    meta["current_world_size"] = current_ws
    meta["converted_step"] = converted
    if saved_ws != current_ws:
        C.log_main(
            "[reshard] GPU 数量变化(%d→%d)，按 MiniMind trainer_utils.py L110-113 "
            "换算：step %d -> %d。注意该换算只在'每 step 每 rank 样本数不变'时成立；"
            "本 lab 固定 global_batch，所以真正消费的样本数不变，换算后的 step 只用于"
            "对齐 MiniMind 的日志语义，不改变数据游标。"
            % (saved_ws, current_ws, saved_step, converted))
    return meta


# ---------------------------------------------------------------------------
# 子命令
# ---------------------------------------------------------------------------


def cmd_save(args: argparse.Namespace) -> int:
    ctx = ReshardContext(args)
    t0 = time.perf_counter()
    losses = ctx.train_steps(int(args.max_steps), log_interval=args.log_interval)
    tail = losses[-20:] if losses else [0.0]
    next_loss = ctx.global_loss_at(int(args.max_steps) + 1)
    meta = {
        "step": int(args.max_steps),
        "losses": losses,
        "loss_tail_min": min(tail),
        "loss_tail_max": max(tail),
        "next_step_loss": next_loss,
        "next_step": int(args.max_steps) + 1,
        "sharding": args.sharding,
        "config": os.path.abspath(args.config),
        "train_wall_s": time.perf_counter() - t0,
    }
    save_checkpoint(ctx, args.ckpt_dir, args.format, meta)
    C.log_main("[save] format=%s world_size=%d step=%d next_step_loss=%.6f -> %s"
               % (args.format, ctx.info.world_size, int(args.max_steps),
                  next_loss, os.path.abspath(args.ckpt_dir)))
    ctx.close()
    return 0


def cmd_load(args: argparse.Namespace) -> int:
    ctx = ReshardContext(args)
    meta = load_checkpoint(ctx, args.ckpt_dir)
    C.log_main("[load] ok  saved_ws=%d current_ws=%d saved_step=%d converted_step=%d"
               % (meta["saved_world_size"], meta["current_world_size"],
                  int(meta.get("step", 0)), meta["converted_step"]))
    ctx.close()
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    ctx = ReshardContext(args)
    meta = load_checkpoint(ctx, args.ckpt_dir)
    next_step = int(meta.get("next_step", int(meta.get("step", 0)) + 1))
    expected = float(meta["next_step_loss"])
    actual = ctx.global_loss_at(next_step)
    delta = abs(actual - expected)
    in_band = (float(meta.get("loss_tail_min", expected)) - args.tol
               <= actual
               <= float(meta.get("loss_tail_max", expected)) + args.tol)
    ok = bool(delta < args.tol)
    result = {
        "saved_world_size": meta["saved_world_size"],
        "current_world_size": meta["current_world_size"],
        "next_step": next_step,
        "expected_loss": expected,
        "actual_loss": actual,
        "abs_delta": delta,
        "tol": args.tol,
        "within_tail_band": in_band,
        "pass": ok,
    }
    if ctx.info.is_main:
        print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2),
              flush=True)
        print("[verify] %s  |delta|=%.3e tol=%.1e (saved_ws=%d -> current_ws=%d)"
              % ("PASS" if ok else "FAIL", delta, args.tol,
                 meta["saved_world_size"], meta["current_world_size"]), flush=True)
        if args.result_json:
            os.makedirs(os.path.dirname(os.path.abspath(args.result_json)) or ".",
                        exist_ok=True)
            with open(args.result_json, "w", encoding="utf-8") as fh:
                json.dump(result, fh, ensure_ascii=False, indent=2, sort_keys=True)
    ctx.close()
    return 0 if ok else 1


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="FSDP 分片 checkpoint 保存 / 跨 world_size 恢复 / loss 连续性验证",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    C.add_common_args(p)
    p.add_argument("command", choices=["save", "load", "verify"],
                   help="save=训练并保存；load=只加载；verify=加载后比 loss")
    p.add_argument("--ckpt-dir", type=str, default="./runs/ckpt",
                   help="checkpoint 目录（公司环境请指定公司内部路径）")
    p.add_argument("--format", type=str, default="sharded",
                   choices=["sharded", "full"],
                   help="sharded=SHARDED_STATE_DICT+distributed.checkpoint；"
                        "full=FULL_STATE_DICT(rank0_only) 回退")
    p.add_argument("--sharding", type=str, default="full_shard",
                   choices=sorted(SHARDING_STRATEGIES))
    p.add_argument("--tol", type=float, default=1e-3,
                   help="verify 的 loss 容差；fp32/CPU 建议 1e-4，fp16/CUDA 建议 1e-2")
    p.add_argument("--result-json", type=str, default=None,
                   help="verify 结果另存为 JSON（rank0 写）")
    p.add_argument("--global-batch", type=int, default=None)
    p.add_argument("--accum", type=int, default=None)
    p.add_argument("--seq-len", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--dataset", type=str, default="tiny", choices=["tiny", "real"])
    p.add_argument("--data-path", type=str, default=None)
    p.add_argument("--num-samples", type=int, default=8192)
    p.add_argument("--sdpa-backend", type=str, default="auto",
                   choices=["auto", "math", "mem_efficient", "flash"])
    p.add_argument("--log-interval", type=int, default=1)
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.config:
        raise SystemExit("必须给 --config configs/<name>.json")
    handlers = {"save": cmd_save, "load": cmd_load, "verify": cmd_verify}
    return handlers[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
