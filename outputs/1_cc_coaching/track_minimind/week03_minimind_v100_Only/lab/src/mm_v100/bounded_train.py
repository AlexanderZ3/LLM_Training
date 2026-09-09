"""Day 1-5 的统一有界训练入口：单卡 / DDP / FSDP 共用一个循环，同一份日志字段。

来源
----
从 week01 lab/src/mm_probe/bounded_train.py（有界步数、JSONL 仪表、checkpoint/
resume、数据游标）与 week02 lab/src/mm_dist/train_ddp.py + train_fsdp.py
（fixed-global-batch 切分、全局 loss 归约、FSDP 包装）合并裁剪而成。
去掉了 MiniMind 仓库依赖、MoE、activation checkpointing 扫描、profiler。

为什么三种放置共用一个循环
--------------------------
只有当数据切分、loss 归约、日志字段完全一致时，两条曲线才可以逐步相减。
分成三个脚本写，早晚会在某一处产生差异，而那个差异会伪装成「FSDP 的数值问题」。

一个 step 的固定顺序
--------------------
1. 算 lr（cosine，与 MiniMind get_lr 同公式）
2. zero_grad
3. accum 次：取样本 -> autocast 前向 -> scaler.scale(loss/accum).backward()
   （前 accum-1 次在 no_sync 里；ddp_no_sync 故障下全部在 no_sync 里）
4. all_reduce 把各 rank 的 loss 求和，再除以 world_size*accum -> 全局 loss
5. scaler.unscale_ -> 记 grad_norm -> clip -> scaler.step+update
6. 记三个不变量与一行 JSONL

第 4 步的顺序很重要：打印 rank0 自己的 micro loss 是比不出等价性的。

硬上限
------
--max-steps 是硬上限，没有 epoch 概念。公司机器每次运行 <=10 分钟这条约束
只能靠步数上限落实，不能靠「跑一会儿手动 Ctrl-C」。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from contextlib import nullcontext
from typing import Any, Dict, List, Optional

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

if __package__ in (None, ""):  # 允许 python src/mm_v100/bounded_train.py
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mm_v100 import collectives as CO  # noqa: E402
from mm_v100 import common as C  # noqa: E402
from mm_v100 import faults as FA  # noqa: E402
from mm_v100 import model as M  # noqa: E402

PLACEMENTS = ("single", "ddp", "fsdp")
STAGES = ("pretrain", "sft")

FSDP_NEEDS_CUDA = (
    "FSDP1 需要 CUDA 设备，纯 CPU 上跑不了。\n"
    "原因：torch 的 _init_device_handle 在参数全在 CPU 上时会回落到 "
    "torch.cuda.current_device()，没有 GPU 就抛 "
    "'FSDP needs a non-CPU accelerator device'。torch 2.1.0 与本机的新版 torch "
    "在这一点上行为一致，所以这不是版本问题。\n"
    "本机能验的是 --placement single 与 --placement ddp（gloo）；"
    "FSDP 相关的门只能在公司 8xV100 上跑：\n"
    "  torchrun --nproc_per_node 8 -m mm_v100.bounded_train "
    "--placement fsdp --config v100_768 ...")


class ParamProbe:
    """用一小片参数判断「参数到底动了没有」。

    为什么不整份 diff：57M 参数的 fp32 快照要 228 MB，8 卡上就是 1.8 GB，
    为了一个布尔量不值得。这里只抓前 n_params 个参数张量的前 n_elems 个元素。
    代价是理论上存在「只有没被抓到的参数变了」的漏检；但故障 B 是全局跳步，
    一个都不会动，所以这个探针对它是充分的。局限写在这里，不写进结论。
    """

    def __init__(self, model: torch.nn.Module, n_params: int = 8,
                 n_elems: int = 4096) -> None:
        self.slices: List[torch.nn.Parameter] = []
        for p in model.parameters():
            if p.requires_grad:
                self.slices.append(p)
            if len(self.slices) >= int(n_params):
                break
        self.n_elems = int(n_elems)
        self.prev = self._snapshot()

    def _snapshot(self) -> List[torch.Tensor]:
        return [p.detach().reshape(-1)[: self.n_elems].float().clone()
                for p in self.slices]

    def delta_norm(self) -> float:
        current = self._snapshot()
        total = 0.0
        for a, b in zip(current, self.prev):
            total += float(torch.linalg.vector_norm(a - b, 2.0).item()) ** 2
        self.prev = current
        return float(total ** 0.5)


def unwrap(model: torch.nn.Module) -> torch.nn.Module:
    inner = model
    while hasattr(inner, "module"):
        inner = inner.module
    return inner


def build_placement(model: torch.nn.Module, placement: str, info: C.DistInfo,
                    device: str, dtype_name: str, sharding: str):
    """按 placement 包装模型，返回 (wrapped, n_units)。

    n_units 只有 FSDP 有意义（字节账里 all_gather buffer 的分母），
    其余返回 1。
    """
    if placement == "single" or info.world_size <= 1:
        if placement == "fsdp":
            # 单进程 FSDP 也要走 FSDP 路径，否则字节账测的是 DDP 的账。
            pass
        else:
            return model, 1
    if placement == "ddp":
        device_ids = [info.local_rank] if str(device).startswith("cuda") else None
        return DistributedDataParallel(model, device_ids=device_ids), 1
    if placement == "fsdp":
        if not str(device).startswith("cuda"):
            raise SystemExit(FSDP_NEEDS_CUDA)
        if not info.initialized:
            raise SystemExit(
                "FSDP 需要已初始化的进程组。单进程也要用 torchrun 启动：\n"
                "  torchrun --nproc_per_node 1 -m mm_v100.bounded_train "
                "--placement fsdp ...")
        import functools

        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        from torch.distributed.fsdp import MixedPrecision, ShardingStrategy
        from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

        strategies = {
            "full_shard": ShardingStrategy.FULL_SHARD,
            "shard_grad_op": ShardingStrategy.SHARD_GRAD_OP,
            "no_shard": ShardingStrategy.NO_SHARD,
        }
        if sharding not in strategies:
            raise ValueError("--sharding 只能是 " + " / ".join(sorted(strategies)))
        mp = None
        if dtype_name in ("float16", "fp16"):
            # V100 只有 fp16；reduce_dtype 也设 fp16，否则梯度归约会退回 fp32，
            # 通信量翻倍而账上看不出来。
            mp = MixedPrecision(param_dtype=torch.float16,
                                reduce_dtype=torch.float16,
                                buffer_dtype=torch.float16)
        wrapped = FSDP(
            model,
            sharding_strategy=strategies[sharding],
            auto_wrap_policy=functools.partial(
                transformer_auto_wrap_policy, transformer_layer_cls={M.Block}),
            mixed_precision=mp,
            use_orig_params=True,
            limit_all_gathers=True,
            device_id=torch.device(device),
        )
        n_units = sum(1 for m in wrapped.modules() if isinstance(m, FSDP))
        return wrapped, n_units
    return model, 1


def make_scaler(placement: str, dtype: torch.dtype, device: str):
    """FSDP + fp16 要用 ShardedGradScaler：普通 GradScaler 的 inf 检查只看本 rank
    的那一片梯度，而 FULL_SHARD 下每 rank 只有分片，必须跨 rank 归约。"""
    if (placement == "fsdp" and dtype is torch.float16
            and str(device).startswith("cuda")):
        from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler

        adapter = C.ScalerAdapter(dtype, device)
        adapter.impl = ShardedGradScaler(enabled=True)
        adapter.backend = "sharded"
        return adapter
    return C.ScalerAdapter(dtype, device)


def clip_grad(model: torch.nn.Module, placement: str, max_norm: float) -> None:
    if placement == "fsdp" and hasattr(model, "clip_grad_norm_"):
        model.clip_grad_norm_(max_norm)
        return
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)


def save_checkpoint(path: str, model, optimizer, scaler, step: int,
                    args: argparse.Namespace, info: C.DistInfo,
                    extra: Dict[str, Any]) -> str:
    """保存 model / optimizer / scaler / RNG / step / 数据游标 / 配置。

    数据游标在本 lab 里就是 step 本身（样本下标由 micro_batch_indices 从 step
    算出来），但仍然显式存下来：换 world_size 恢复时要靠它判断消费到哪了。
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    payload = {
        "model": unwrap(model).state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "rng": C.rng_state(),
        "step": int(step),
        "data_cursor": {
            "next_step": int(step) + 1,
            "global_batch": int(args.global_batch),
            "accum": int(args.accum),
            "world_size": int(info.world_size),
            "consumed_samples": int(step) * int(args.global_batch),
        },
        "scheduler": {"kind": "cosine_minimind", "total_steps": int(args.max_steps),
                      "base_lr": float(args.lr)},
        "config_path": args.config,
        "dtype": args.dtype,
        "placement": args.placement,
        "extra": extra,
    }
    torch.save(payload, path)
    return os.path.abspath(path)


def load_checkpoint(path: str, model, optimizer, scaler) -> Dict[str, Any]:
    blob = torch.load(path, map_location="cpu", weights_only=False)
    unwrap(model).load_state_dict(blob["model"])
    optimizer.load_state_dict(blob["optimizer"])
    scaler.load_state_dict(blob["scaler"])
    C.set_rng_state(blob["rng"])
    return blob


def train(args: argparse.Namespace) -> Dict[str, Any]:
    info = C.init_dist(backend=args.backend, timeout_s=args.nccl_timeout_s,
                       force_cpu=args.force_cpu)
    device = "cpu" if args.force_cpu else info.device

    equiv = bool(args.equiv_check)
    dtype_name = "float32" if equiv else args.dtype
    torch_dtype = C.assert_dtype_supported(dtype_name, device)
    per_rank_seed = False if equiv else bool(args.seed_per_rank)
    C.set_seed(args.seed, per_rank=per_rank_seed)

    cfg = C.load_config(args.config)
    train_cfg = cfg["train"]
    args.global_batch = int(args.global_batch or train_cfg["global_batch"])
    args.accum = int(args.accum or train_cfg.get("accum", 1))
    args.seq_len = int(args.seq_len or train_cfg["seq_len"])
    args.lr = float(args.lr if args.lr is not None else train_cfg.get("lr", 5e-4))
    grad_clip = float(args.grad_clip if args.grad_clip is not None
                      else train_cfg.get("grad_clip", 1.0))

    denom = info.world_size * args.accum
    if args.global_batch % denom != 0:
        raise SystemExit(
            "global_batch=" + str(args.global_batch) + " 必须能被 world_size*accum="
            + str(info.world_size) + "*" + str(args.accum) + "=" + str(denom)
            + " 整除。\n下一步：调 --global-batch 或 --accum，让 micro 是整数。")
    micro = args.global_batch // denom

    model, model_cfg = M.build_model(cfg["model"],
                                     device="cpu" if args.placement == "fsdp"
                                     else device,
                                     seed=args.seed)
    # 默认把样本数放大到「每步都是新样本」，因为 fixed-global-batch 等价实验
    # 要求第 s 步消费的样本集合与 world_size 无关，重复消费会把结论弄脏。
    # 但 CLAUDE.md 第 7 节的 overfit 门恰恰要反过来：反复喂同一小批直到 loss→0，
    # 用「模型能不能记住 128 条」来证明前反向链路是通的。所以给一个显式开关。
    # TinyDataset.__getitem__ 本来就对索引取模，打开开关即可循环。
    n_samples = (int(args.num_samples) if args.allow_sample_reuse
                 else max(int(args.num_samples),
                          int(args.max_steps) * args.global_batch))
    dataset = C.TinyDataset(
        num_samples=n_samples,
        seq_len=args.seq_len, vocab_size=int(model_cfg.vocab_size),
        seed=int(train_cfg.get("data_seed", 1234)),
        label_ignore_prefix=(int(train_cfg.get("label_ignore_prefix",
                                               args.seq_len // 4))
                             if args.stage == "sft" else 0))

    wrapped, n_units = build_placement(model, args.placement, info, device,
                                       dtype_name, args.sharding)
    optimizer = torch.optim.AdamW(wrapped.parameters(), lr=args.lr)
    scaler = make_scaler(args.placement, torch_dtype, device)
    probe = ParamProbe(unwrap(wrapped))

    out_dir = args.out_dir
    if out_dir is None:
        from mm_v100 import paths as P

        out_dir = str(P.run_dir(args.run_tag or ("day_" + args.stage)))
    os.makedirs(out_dir, exist_ok=True)

    start_step = 0
    resume_meta: Optional[Dict[str, Any]] = None
    if args.resume:
        resume_meta = load_checkpoint(args.resume, wrapped, optimizer, scaler)
        start_step = int(resume_meta["step"])
        C.log_main("[resume] 从 " + args.resume + " 恢复，step=" + str(start_step)
                   + " data_cursor=" + json.dumps(resume_meta["data_cursor"]))

    params = M.count_params(unwrap(wrapped))
    logger = C.JsonlLogger(out_dir, args.log_name or ("metrics_" + args.stage),
                           rank=info.rank)
    header = {
        "step": 0,
        "event": "header",
        "config": cfg["name"],
        "stage": args.stage,
        "placement": args.placement,
        "sharding": args.sharding if args.placement == "fsdp" else None,
        "n_fsdp_units": n_units,
        "world_size": info.world_size,
        "backend": info.backend,
        "device_type": "cuda" if str(device).startswith("cuda") else "cpu",
        "dtype": dtype_name,
        "global_batch": args.global_batch,
        "accum": args.accum,
        "micro_batch": micro,
        "seq_len": args.seq_len,
        "params_total": params["total"],
        "fault": args.fault,
        "equiv_check": equiv,
        "seed": args.seed,
        "seed_per_rank": per_rank_seed,
        "scaler": scaler.describe(),
        "resumed_from_step": start_step,
        "env": C.env_summary(),
    }
    logger.write(header)
    C.log_main("[header] " + json.dumps(
        {k: header[k] for k in ("config", "stage", "placement", "world_size",
                                "dtype", "global_batch", "accum", "micro_batch",
                                "seq_len", "params_total", "fault")},
        ensure_ascii=False))

    if args.dry_run:
        idx = C.micro_batch_indices(1, args.global_batch, info.world_size,
                                    args.accum, info.rank, 0)
        input_ids, labels = C.collate_indices(dataset, idx, device)
        with C.autocast_context(torch_dtype, device):
            out = wrapped(input_ids, labels=labels)
        C.log_main("[dry-run] forward ok  logits=" + str(tuple(out.logits.shape))
                   + "  loss=" + ("%.5f" % float(out.loss.detach().float().item())))
        logger.write({"step": 0, "event": "dry_run", "ok": True,
                      "logits_shape": list(out.logits.shape),
                      "loss": float(out.loss.detach().float().item())})
        logger.close()
        C.cleanup_dist(info)
        return {"dry_run": True, "log_path": logger.path,
                "params_total": params["total"], "micro_batch": micro,
                "n_fsdp_units": n_units}

    injector = FA.GradInfInjector(unwrap(wrapped),
                                  enabled=(args.fault == "scaler_stuck"))
    total_steps = int(args.max_steps)
    # resume 之后 scaler 里的计数是上一轮的，step_ratio 必须只统计本轮。
    applied_at_start = scaler.applied_steps
    wrapped.train()
    losses: List[float] = []
    records: List[Dict[str, Any]] = []
    C.reset_peak_mem(device)
    t_run0 = time.perf_counter()
    ckpt_paths: List[str] = []
    try:
        for step in range(start_step + 1, total_steps + 1):
            t0 = time.perf_counter()
            lr = C.cosine_lr(step, total_steps, args.lr)
            for group in optimizer.param_groups:
                group["lr"] = lr
            optimizer.zero_grad(set_to_none=True)
            sums = torch.zeros(1, device=device, dtype=torch.float32)
            n_label_tokens = 0
            align_viol = 0
            meter = CO.CollectiveMeter(record_calls=False)
            ctx = meter if args.count_collectives else nullcontext()
            with ctx:
                for acc_idx in range(args.accum):
                    idx = C.micro_batch_indices(step, args.global_batch,
                                                info.world_size, args.accum,
                                                info.rank, acc_idx)
                    input_ids, labels = C.collate_indices(dataset, idx, device)
                    labels = FA.apply_label_fault(input_ids, labels, args.fault)
                    is_last = acc_idx == args.accum - 1
                    do_sync = FA.should_sync_this_micro(args.fault, is_last)
                    sync_ctx = nullcontext()
                    if (isinstance(wrapped, DistributedDataParallel)
                            and not do_sync):
                        sync_ctx = wrapped.no_sync()
                    with sync_ctx:
                        with C.sdpa_context(args.sdpa_backend, device):
                            with C.autocast_context(torch_dtype, device):
                                out = wrapped(input_ids, labels=labels)
                                loss = out.loss
                        scaler.scale(loss / args.accum).backward()
                    sums[0] += loss.detach().float()
                    n_label_tokens += int((labels != -100).sum().item())
                    align_viol += FA.label_alignment_violations(input_ids, labels)

                if info.initialized and info.world_size > 1:
                    dist.all_reduce(sums, op=dist.ReduceOp.SUM)
                scaler.unscale_(optimizer)
                gnorm = C.grad_global_norm(unwrap(wrapped).parameters())
                clip_grad(wrapped, args.placement, grad_clip)
                applied = scaler.step_and_update(optimizer)

            # I3 的 all_gather 放在计数上下文之外：它是仪表自己的通信，
            # 算进通信账会污染「一个 step 有几次集合通信」这个结论。
            grad_delta = FA.cross_rank_grad_delta(unwrap(wrapped),
                                                  info.world_size)
            delta_norm = probe.delta_norm()
            global_loss = float(sums[0].item() / float(denom))
            losses.append(global_loss)
            dt = max(time.perf_counter() - t0, 1e-9)
            executed = step - start_step
            obs = {
                "I1_label_align_violations": int(align_viol),
                "I2_optimizer_step_ratio": (
                    float(scaler.applied_steps - applied_at_start)
                    / max(executed, 1)),
                "I2_param_delta_norm": float(delta_norm),
                "I3_cross_rank_grad_delta": grad_delta,
            }
            record = {
                "step": step,
                "loss": global_loss,
                "lr": lr,
                "grad_norm": gnorm,
                "scaler_scale": scaler.get_scale(),
                "step_applied": bool(applied),
                "skipped_steps": scaler.skipped_steps,
                "n_label_tokens": n_label_tokens,
                "tokens_per_s": args.global_batch * args.seq_len / dt,
                "step_time_s": dt,
                "peak_mem_mb": C.peak_mem_mb(device),
                "world_size": info.world_size,
                "elapsed_s": time.perf_counter() - t_run0,
            }
            record.update(obs)
            if args.count_collectives:
                record["collectives"] = meter.summary()
            logger.write(record)
            records.append(record)
            if step % max(int(args.log_interval), 1) == 0 or step == total_steps:
                C.log_main(
                    "[%s ws=%d step %d/%d] loss=%.5f lr=%.3e gn=%.3f scale=%.0f "
                    "applied=%s I1=%d I2=%.2f I3=%s tok/s=%.0f mem=%.0fMB"
                    % (args.placement, info.world_size, step, total_steps,
                       global_loss, lr, gnorm, record["scaler_scale"],
                       record["step_applied"], obs["I1_label_align_violations"],
                       obs["I2_optimizer_step_ratio"],
                       obs["I3_cross_rank_grad_delta"], record["tokens_per_s"],
                       record["peak_mem_mb"]))
            if args.save_every and step % args.save_every == 0 and info.is_main:
                ckpt_paths.append(save_checkpoint(
                    os.path.join(out_dir, "ckpt", "step" + str(step) + ".pt"),
                    wrapped, optimizer, scaler, step, args, info,
                    {"loss": global_loss}))
    finally:
        injector.remove()
        logger.close()

    if args.save_final and info.is_main:
        ckpt_paths.append(save_checkpoint(
            os.path.join(out_dir, "ckpt", "final.pt"), wrapped, optimizer,
            scaler, total_steps, args, info,
            {"loss": losses[-1] if losses else None}))

    last_obs = records[-1] if records else {}
    summary = {
        "config": cfg["name"],
        "stage": args.stage,
        "placement": args.placement,
        "world_size": info.world_size,
        "micro_batch": micro,
        "n_fsdp_units": n_units,
        "steps": len(records),
        "losses": losses,
        "log_path": logger.path,
        "out_dir": os.path.abspath(out_dir),
        "ckpt_paths": ckpt_paths,
        "params_total": params["total"],
        "skipped_steps": scaler.skipped_steps,
        "applied_steps": scaler.applied_steps,
        "fault": args.fault,
        "classification": FA.classify({
            "I1_label_align_violations": last_obs.get(
                "I1_label_align_violations", 0),
            "I2_optimizer_step_ratio": last_obs.get(
                "I2_optimizer_step_ratio", 1.0),
            "I2_param_delta_norm": last_obs.get("I2_param_delta_norm", 1.0),
            "I3_cross_rank_grad_delta": last_obs.get(
                "I3_cross_rank_grad_delta"),
        }) if records else None,
    }
    C.cleanup_dist(info)
    return summary


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="有界训练：单卡 / DDP / FSDP 统一入口（--max-steps 是硬上限）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    C.add_common_args(p)
    p.add_argument("--placement", type=str, default="single", choices=PLACEMENTS,
                   help="参数放置：single / ddp / fsdp")
    p.add_argument("--sharding", type=str, default="full_shard",
                   choices=["full_shard", "shard_grad_op", "no_shard"],
                   help="仅 --placement fsdp 生效")
    p.add_argument("--stage", type=str, default="pretrain", choices=STAGES,
                   help="pretrain=全部非 pad 进 loss；sft=前 1/4 位置当 prompt 屏蔽")
    p.add_argument("--fault", type=str, default="none", choices=FA.FAULT_MODES,
                   help="故障注入档位；none 是对照组")
    p.add_argument("--global-batch", type=int, default=None,
                   help="全局 batch（覆盖配置里的 train.global_batch）")
    p.add_argument("--accum", type=int, default=None, help="梯度累积步数")
    p.add_argument("--seq-len", type=int, default=None, help="序列长度")
    p.add_argument("--lr", type=float, default=None, help="基础学习率")
    p.add_argument("--grad-clip", type=float, default=None, help="梯度裁剪阈值")
    p.add_argument("--num-samples", type=int, default=4096,
                   help="TinyDataset 样本数；默认不足 max_steps*global_batch 时自动放大")
    p.add_argument("--allow-sample-reuse", action="store_true",
                   help="允许样本被重复消费（num_samples 不再自动放大）。"
                        "overfit 门要用它：反复喂同一小批直到 loss 掉到接近 0")
    p.add_argument("--log-interval", type=int, default=1)
    p.add_argument("--log-name", type=str, default=None,
                   help="JSONL 文件名前缀，默认 metrics_<stage>")
    p.add_argument("--count-collectives", type=int, default=0, choices=[0, 1],
                   help="1=每步统计集合通信次数与字节（有 Python 侧开销）")
    p.add_argument("--equiv-check", action="store_true",
                   help="等价模式：强制 float32、关掉 per-rank seed")
    p.add_argument("--save-every", type=int, default=0,
                   help=">0 时每 N 步存一个 ckpt")
    p.add_argument("--save-final", type=int, default=0, choices=[0, 1],
                   help="1=结束时存 ckpt/final.pt")
    p.add_argument("--resume", type=str, default=None,
                   help="从 bounded_train 存的 .pt 恢复")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    summary = train(args)
    if C.is_main_process():
        print(json.dumps({k: v for k, v in summary.items() if k != "losses"},
                         ensure_ascii=False, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
