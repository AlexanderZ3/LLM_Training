"""FSDP1 三种 sharding 策略对照（FULL_SHARD / SHARD_GRAD_OP / NO_SHARD）。

为什么与 train_ddp.py 分开写
---------------------------
两者的数据切分、loss 归约方式、日志字段完全一致（见下方 record 字典），
唯一差别是参数放置：DDP 每卡整份参数 + 梯度 allreduce；FSDP 把 flat parameter
按 world_size 切片，forward/backward 前 all_gather、backward 后 reduce_scatter。
因此同一份 configs/*.json、同一个 --global-batch，可以直接用
``python -m mm_dist.train_ddp --compare-json ddp.jsonl fsdp.jsonl`` 逐步比 loss。

只使用 torch 2.1.0 已有的 API
-----------------------------
``FullyShardedDataParallel``、``ShardingStrategy``、``MixedPrecision``、
``transformer_auto_wrap_policy``、``apply_activation_checkpointing``、
``ShardedGradScaler``、``FSDP.clip_grad_norm_``。
**不使用** ``fully_shard`` / ``DTensor`` / ``device_mesh``（那些是 FSDP2 的东西）。

wrap 粒度
---------
``transformer_auto_wrap_policy(transformer_layer_cls={MiniMindBlock})``。
MiniMind 的 block 类名是 ``MiniMindBlock``（model/model_minimind.py L178），
每个 block 含 ``self_attn`` + ``mlp``（dense 时 FeedForward，MoE 时 MOEFeedForward）。
wrap 粒度决定 all_gather 的次数与每次的字节：8 层 + 根 = 9 个 FSDP 单元。
如果 wrap 退化成"只有一个根 FSDP"，forward 峰值 = 整份模型，FSDP 就没有省显存
（01_FOUNDATIONS.md 第 6 节失败模式 #9）。

用法
----
8 卡 FULL_SHARD::

    torchrun --nproc_per_node 8 -m mm_dist.train_fsdp \
        --config configs/v100_dense.json --out-dir <公司内路径> \
        --sharding full_shard --dtype float16 --global-batch 64 --max-steps 20

同 global batch 的 DDP 对照::

    torchrun --nproc_per_node 8 -m mm_dist.train_ddp \
        --config configs/v100_dense.json --out-dir <公司内路径> \
        --dtype float16 --global-batch 64 --max-steps 20 --log-name ddp_ws8
"""

from __future__ import annotations

import argparse
import functools
import os
import sys
import time
from contextlib import nullcontext
from typing import Any, Dict, List, Optional

import torch
import torch.distributed as dist
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointImpl,
    apply_activation_checkpointing,
    checkpoint_wrapper,
)
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

if __package__ in (None, ""):  # 允许 `python src/mm_dist/train_fsdp.py`
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mm_dist import common as C  # noqa: E402
from mm_dist.train_ddp import compare_logs  # noqa: E402

SHARDING_STRATEGIES = {
    "full_shard": ShardingStrategy.FULL_SHARD,
    "shard_grad_op": ShardingStrategy.SHARD_GRAD_OP,
    "no_shard": ShardingStrategy.NO_SHARD,
}

FSDP_REQUIRES_CUDA = (
    "FSDP1 需要 CUDA 设备，纯 CPU 上跑不了。\n"
    "原因：torch 的 _init_device_handle 在参数全在 CPU 上时会回落到 "
    "torch.cuda.current_device()，没有 GPU 就抛 "
    "'FSDP needs a non-CPU accelerator device'。torch 2.1.0 与本机的新版 torch "
    "在这一点上行为一致，所以这不是版本问题，也不是本 lab 的限制。\n"
    "本机（无 GPU）能验证的是 DDP 等价、EP 通信、router 统计、GRPO 分组逻辑；\n"
    "FSDP / 分片 checkpoint / activation checkpointing 三个门只能在公司 8xV100 上跑：\n"
    "  torchrun --nproc_per_node 8 -m mm_dist.train_fsdp --config configs/v100_dense.json ...")


def require_cuda_for_fsdp(device: str) -> None:
    """FSDP 入口的统一前置检查：不是 CUDA 就立刻退出并说清楚为什么。"""
    if not str(device).startswith("cuda"):
        raise SystemExit(FSDP_REQUIRES_CUDA)


# ---------------------------------------------------------------------------
# 集合通信计数
# ---------------------------------------------------------------------------


class CollectiveCounter:
    """统计一段代码里各类集合通信被调用了多少次。

    做法：临时替换 ``torch.distributed`` 上的公共函数名。torch 2.1 的 FSDP
    (``torch/distributed/fsdp/_flat_param.py``) 通过 ``dist.all_gather_into_tensor``
    / ``dist.reduce_scatter_tensor`` 这样的模块属性调用，所以替换模块属性能拦到。

    只包公共名，不包 ``_all_gather_base`` / ``_reduce_scatter_base``：后者在 2.1 里
    是前者的 deprecated 包装，两个都包会重复计数。
    """

    NAMES = (
        "all_gather_into_tensor",
        "reduce_scatter_tensor",
        "all_reduce",
        "all_gather",
        "broadcast",
        "all_to_all_single",
    )

    def __init__(self) -> None:
        self.counts: Dict[str, int] = {n: 0 for n in self.NAMES}
        self._orig: Dict[str, Any] = {}
        self._active = False

    def reset(self) -> None:
        for n in self.NAMES:
            self.counts[n] = 0

    def total(self) -> int:
        return sum(self.counts.values())

    def snapshot(self) -> Dict[str, int]:
        return dict(self.counts)

    def _wrap(self, name: str, fn):
        def wrapper(*a, **kw):
            self.counts[name] += 1
            return fn(*a, **kw)

        functools.update_wrapper(wrapper, fn)
        return wrapper

    def __enter__(self) -> "CollectiveCounter":
        if self._active:
            return self
        for n in self.NAMES:
            fn = getattr(dist, n, None)
            if fn is None:
                continue
            self._orig[n] = fn
            setattr(dist, n, self._wrap(n, fn))
        self._active = True
        return self

    def __exit__(self, *exc) -> bool:
        for n, fn in self._orig.items():
            setattr(dist, n, fn)
        self._orig.clear()
        self._active = False
        return False


# ---------------------------------------------------------------------------
# 包装
# ---------------------------------------------------------------------------


def build_auto_wrap_policy(block_cls) -> Any:
    """按 MiniMindBlock 切 FSDP 单元。"""
    return functools.partial(transformer_auto_wrap_policy,
                             transformer_layer_cls={block_cls})


def maybe_activation_checkpoint(model, block_cls, enabled: bool) -> int:
    """在 FSDP 包装**之前**给每个 MiniMindBlock 套 activation checkpoint。

    返回被套上的模块数。NO_REENTRANT 版本与 FSDP 的兼容性最好（2.1 文档推荐）。
    """
    if not enabled:
        return 0
    wrapped = [0]

    def check_fn(module) -> bool:
        hit = isinstance(module, block_cls)
        if hit:
            wrapped[0] += 1
        return hit

    apply_activation_checkpointing(
        model,
        checkpoint_wrapper_fn=functools.partial(
            checkpoint_wrapper, checkpoint_impl=CheckpointImpl.NO_REENTRANT),
        check_fn=check_fn,
    )
    return wrapped[0]


def make_mixed_precision(dtype_name: str, device: str) -> Optional[MixedPrecision]:
    """V100 只允许 fp16；fp32 或 CPU 时返回 None（不套 MixedPrecision）。"""
    if not str(device).startswith("cuda"):
        return None
    if dtype_name in ("float16", "fp16"):
        return MixedPrecision(param_dtype=torch.float16,
                              reduce_dtype=torch.float16,
                              buffer_dtype=torch.float16)
    return None


def make_fsdp_scaler(dtype_name: str, device: str):
    """FSDP + fp16 用 ShardedGradScaler；其余走 common.make_grad_scaler。

    普通 GradScaler 的 ``unscale_`` 只看本 rank 的梯度，FULL_SHARD 下每 rank 只有
    梯度分片，inf 检查必须跨 rank 归约——这正是 ShardedGradScaler 做的事。
    """
    if str(device).startswith("cuda") and dtype_name in ("float16", "fp16"):
        from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler
        return ShardedGradScaler(enabled=True)
    return C.make_grad_scaler(
        torch.float16 if dtype_name in ("float16", "fp16") else torch.float32, device)


def count_fsdp_units(module) -> int:
    """统计 FSDP 单元数量（含根）。退化为 1 说明 auto_wrap_policy 没生效。"""
    return sum(1 for m in module.modules() if isinstance(m, FSDP))


# ---------------------------------------------------------------------------
# 训练
# ---------------------------------------------------------------------------


def train(args: argparse.Namespace) -> Dict[str, Any]:
    info = C.init_dist(backend=args.backend, timeout_s=args.nccl_timeout_s,
                       force_cpu=args.force_cpu)
    if not info.initialized:
        raise SystemExit(
            "train_fsdp 需要已初始化的进程组。即使只用 1 个进程也要通过 torchrun 启动：\n"
            "  torchrun --nproc_per_node 1 -m mm_dist.train_fsdp --config <cfg> ...")
    device = "cpu" if args.force_cpu else info.device
    require_cuda_for_fsdp(device)

    equiv = bool(args.equiv_check)
    dtype_name = "float32" if equiv else args.dtype
    C.assert_dtype_supported(dtype_name, device)
    per_rank_seed = False if equiv else bool(args.seed_per_rank)
    C.set_seed(args.seed, per_rank=per_rank_seed)

    strategy = SHARDING_STRATEGIES[args.sharding]

    # 模型先建在 CPU（CUDA 时由 FSDP 的 device_id 迁移），避免 8 卡各建一份整模型
    model, lm_config, cfg, mm = C.build_model(
        args.config, minimind_root=args.minimind_root,
        device="cpu" if str(device).startswith("cuda") else device)

    train_cfg = cfg["train"]
    seq_len = int(args.seq_len or train_cfg["seq_len"])
    global_batch = int(args.global_batch or train_cfg["global_batch"])
    accum = int(args.accum or train_cfg.get("accum", 1))
    base_lr = float(args.lr if args.lr is not None else train_cfg.get("lr", 5e-4))
    grad_clip = float(train_cfg.get("grad_clip", 1.0))

    denom = info.world_size * accum
    if global_batch % denom != 0:
        raise SystemExit(
            "global_batch=" + str(global_batch) + " 必须能被 world_size*accum="
            + str(denom) + " 整除。")
    micro = global_batch // denom

    n_ac = maybe_activation_checkpoint(model, mm.MiniMindBlock,
                                       bool(args.activation_checkpointing))

    mp = make_mixed_precision(dtype_name, device)
    fsdp_kwargs: Dict[str, Any] = dict(
        sharding_strategy=strategy,
        auto_wrap_policy=build_auto_wrap_policy(mm.MiniMindBlock),
        mixed_precision=mp,
        use_orig_params=True,
        limit_all_gathers=True,
        sync_module_states=bool(args.sync_module_states),
    )
    if str(device).startswith("cuda"):
        fsdp_kwargs["device_id"] = torch.device(device)
    fsdp_model = FSDP(model, **fsdp_kwargs)
    n_units = count_fsdp_units(fsdp_model)

    dataset = C.build_dataset(
        args.dataset, seq_len=seq_len, vocab_size=int(lm_config.vocab_size),
        minimind_root=args.minimind_root, data_path=args.data_path,
        num_samples=int(args.num_samples),
        seed=int(train_cfg.get("data_seed", 1234)))

    optimizer = torch.optim.AdamW(fsdp_model.parameters(), lr=base_lr)
    scaler = make_fsdp_scaler(dtype_name, device)
    C.reset_peak_mem(device)

    name = args.log_name or ("fsdp_" + args.sharding + "_ws" + str(info.world_size))
    logger = C.JsonlLogger(args.out_dir, name, rank=info.rank)
    total, trainable = C.count_params(model)
    header = {
        "step": 0,
        "event": "header",
        "runner": "fsdp",
        "sharding": args.sharding,
        "world_size": info.world_size,
        "backend": info.backend,
        "device_type": "cuda" if str(device).startswith("cuda") else "cpu",
        "dtype": dtype_name,
        "global_batch": global_batch,
        "accum": accum,
        "micro_batch": micro,
        "seq_len": seq_len,
        "params_total": total,
        "params_trainable": trainable,
        "use_moe": bool(lm_config.use_moe),
        "equiv_check": equiv,
        "seed": args.seed,
        "seed_per_rank": per_rank_seed,
        "fsdp_units": n_units,
        "activation_checkpointed_blocks": n_ac,
        "mixed_precision": mp is not None,
        "minimind_commit": mm.commit,
        "env": C.env_summary(),
    }
    logger.write(header)
    C.log_main("[header] " + str({k: header[k] for k in (
        "sharding", "world_size", "dtype", "global_batch", "accum", "micro_batch",
        "params_total", "fsdp_units", "activation_checkpointed_blocks")}))
    if n_units <= 1 and info.world_size > 1:
        C.log_main("[warn] fsdp_units=1：auto_wrap_policy 没有切开任何 block，"
                   "forward 峰值等于整份模型，显存不会低于 DDP（失败模式 #9）。")

    fsdp_model.train()
    counter = CollectiveCounter()
    skipped_steps = 0
    losses: List[float] = []
    total_steps = max(int(args.max_steps), 1)
    with counter:
        for step in range(1, total_steps + 1):
            counter.reset()
            t0 = time.perf_counter()
            lr = C.cosine_lr(step, total_steps, base_lr)
            for group in optimizer.param_groups:
                group["lr"] = lr
            optimizer.zero_grad(set_to_none=True)

            sums = torch.zeros(3, device=device, dtype=torch.float32)
            for acc_idx in range(accum):
                idx = C.micro_batch_indices(step, global_batch, info.world_size,
                                            accum, info.rank, acc_idx)
                input_ids, labels = C.collate_indices(dataset, idx, device)
                is_last = acc_idx == accum - 1
                sync_ctx = nullcontext() if is_last else fsdp_model.no_sync()
                with sync_ctx:
                    with C.sdpa_context(args.sdpa_backend, device):
                        res = fsdp_model(input_ids, labels=labels)
                        logits_loss = res.loss
                        aux_loss = res.aux_loss
                        if aux_loss is None:
                            aux_loss = logits_loss.new_zeros(())
                        loss = logits_loss + aux_loss
                    scaler.scale(loss / accum).backward()
                sums[0] += loss.detach().float()
                sums[1] += logits_loss.detach().float()
                sums[2] += aux_loss.detach().float()

            if info.world_size > 1:
                dist.all_reduce(sums, op=dist.ReduceOp.SUM)
            sums = sums / float(denom)

            scaler.unscale_(optimizer)
            grad_norm = fsdp_model.clip_grad_norm_(grad_clip)
            scale_before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale() < scale_before:
                skipped_steps += 1

            dt = max(time.perf_counter() - t0, 1e-9)
            coll = counter.snapshot()
            record = {
                "step": step,
                "loss": float(sums[0]),
                "logits_loss": float(sums[1]),
                "aux_loss": float(sums[2]),
                "lr": lr,
                "grad_norm": float(grad_norm),
                "scaler_scale": float(scaler.get_scale()),
                "skipped_steps": skipped_steps,
                "tokens_per_s": global_batch * seq_len / dt,
                "step_time_s": dt,
                "peak_mem_mb": C.peak_mem_mb(device),
                "world_size": info.world_size,
                "sharding": args.sharding,
                "collectives_total": counter.total(),
                "collectives": coll,
            }
            logger.write(record)
            losses.append(record["loss"])
            if step % max(int(args.log_interval), 1) == 0 or step == total_steps:
                C.log_main(
                    "[step %d/%d] sharding=%s loss=%.4f logits=%.4f aux=%.6f lr=%.3e "
                    "gn=%.3f scale=%.0f skipped=%d tok/s=%.0f peak_mem_mb=%.1f "
                    "coll=%d (ag=%d rs=%d ar=%d)"
                    % (step, total_steps, args.sharding, record["loss"],
                       record["logits_loss"], record["aux_loss"], lr,
                       record["grad_norm"], record["scaler_scale"], skipped_steps,
                       record["tokens_per_s"], record["peak_mem_mb"],
                       record["collectives_total"],
                       coll["all_gather_into_tensor"], coll["reduce_scatter_tensor"],
                       coll["all_reduce"]))

    logger.close()
    summary = {
        "sharding": args.sharding,
        "world_size": info.world_size,
        "steps": total_steps,
        "losses": losses,
        "skipped_steps": skipped_steps,
        "log_path": logger.path,
        "fsdp_units": n_units,
        "peak_mem_mb": C.peak_mem_mb(device),
    }
    C.cleanup_dist(info)
    return summary


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="MiniMind FSDP1 三策略对照（FULL_SHARD / SHARD_GRAD_OP / NO_SHARD）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    C.add_common_args(p)
    p.add_argument("--sharding", type=str, default="full_shard",
                   choices=sorted(SHARDING_STRATEGIES),
                   help="FSDP 分片策略")
    p.add_argument("--activation-checkpointing", type=int, default=0, choices=[0, 1],
                   help="1=给每个 MiniMindBlock 套 NO_REENTRANT checkpoint")
    p.add_argument("--sync-module-states", type=int, default=1, choices=[0, 1],
                   help="1=从 rank0 广播初始权重（多卡必须一致）")
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
    p.add_argument("--log-name", type=str, default=None)
    p.add_argument("--equiv-check", action="store_true",
                   help="等价模式：强制 float32、非 per-rank seed（与 DDP 对照用）")
    p.add_argument("--compare-json", nargs=2, metavar=("A", "B"), default=None,
                   help="只比对两条 JSONL 日志（不训练）")
    p.add_argument("--compare-key", type=str, default="loss")
    p.add_argument("--compare-tol", type=float, default=1e-4)
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.compare_json:
        result = compare_logs(args.compare_json[0], args.compare_json[1],
                              key=args.compare_key, tol=args.compare_tol)
        return 0 if result["pass"] else 1
    if not args.config:
        raise SystemExit("训练模式必须给 --config configs/<name>.json")
    train(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
