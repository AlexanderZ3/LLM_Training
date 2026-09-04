"""单卡 / 多卡 DDP 统一入口 + fixed-global-batch 等价检查 + 日志比对。

设计要点
--------
1. ``--global-batch G`` 是唯一的"批量真相"。micro_batch = G // (world_size * accum)，
   不整除直接报错。任何 (W, accum) 组合只要 G 与 micro 相同，第 s 步消费的样本集合
   完全一致（见 :func:`mm_dist.common.micro_batch_indices`），这是不变量 I1 的前提。
2. 日志里的 ``loss`` 是 **全局** loss：先在本 rank 上把 accum 个 micro 的 loss 求和，
   再 all_reduce(SUM)，最后除以 (world_size*accum)。只有这样 W=1 与 W=8 的 loss
   才可以逐步直接比较；打印 rank0 自己的 micro loss 是比不出等价性的。
3. 梯度累积用 DDP 的 ``no_sync()``：前 accum-1 个 micro 只在本地累加，最后一个
   micro 触发一次 allreduce。数学上等于对全部 W*accum 个 chunk 求平均。
4. 只用 torch 2.1 就有的 API：``torch.cuda.amp.autocast`` / ``GradScaler`` /
   ``DistributedDataParallel`` / ``torch.backends.cuda.sdp_kernel``。

用法
----
训练（单进程）::

    python -m mm_dist.train_ddp --config configs/tiny_cpu.json --out-dir ./runs \
        --dtype float32 --global-batch 8 --accum 4 --max-steps 3 --force-cpu

训练（2 进程 gloo）::

    torchrun --nproc_per_node 2 -m mm_dist.train_ddp --config configs/tiny_cpu.json \
        --out-dir ./runs --dtype float32 --global-batch 8 --accum 2 --max-steps 3 \
        --backend gloo --force-cpu --equiv-check

比对两条日志::

    python -m mm_dist.train_ddp --compare-json ./runs/a_rank0.jsonl ./runs/b_rank0.jsonl
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from contextlib import nullcontext
from typing import Any, Dict, List, Optional

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

if __package__ in (None, ""):  # 允许 `python src/mm_dist/train_ddp.py`
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mm_dist import common as C  # noqa: E402


# ---------------------------------------------------------------------------
# 日志比对
# ---------------------------------------------------------------------------


def compare_logs(path_a: str, path_b: str, key: str = "loss",
                 tol: float = 1e-4) -> Dict[str, Any]:
    """逐 step 比较两条 JSONL 的 ``key`` 字段，返回统计并打印明细。"""
    rec_a = {r["step"]: r for r in C.read_jsonl(path_a) if "step" in r and key in r}
    rec_b = {r["step"]: r for r in C.read_jsonl(path_b) if "step" in r and key in r}
    steps = sorted(set(rec_a) & set(rec_b))
    if not steps:
        raise ValueError(
            "两条日志没有公共 step。A 有 " + str(sorted(rec_a)[:5])
            + "，B 有 " + str(sorted(rec_b)[:5])
        )
    deltas: List[float] = []
    print("step |        A |        B |   |delta|")
    print("-----+----------+----------+----------")
    for s in steps:
        va = float(rec_a[s][key])
        vb = float(rec_b[s][key])
        d = abs(va - vb)
        deltas.append(d)
        print("%4d | %8.5f | %8.5f | %.3e" % (s, va, vb, d))
    max_delta = max(deltas)
    result = {
        "key": key,
        "steps_compared": len(steps),
        "max_abs_delta": max_delta,
        "mean_abs_delta": sum(deltas) / len(deltas),
        "tol": tol,
        "pass": bool(max_delta < tol),
        "a": path_a,
        "b": path_b,
    }
    print("")
    print("steps=%d  max|delta|=%.3e  mean|delta|=%.3e  tol=%.1e  -> %s"
          % (result["steps_compared"], result["max_abs_delta"],
             result["mean_abs_delta"], tol, "PASS" if result["pass"] else "FAIL"))
    return result


# ---------------------------------------------------------------------------
# 训练
# ---------------------------------------------------------------------------


def train(args: argparse.Namespace) -> Dict[str, Any]:
    """跑 ``--max-steps`` 个 optimizer step，返回汇总字典。"""
    info = C.init_dist(backend=args.backend, timeout_s=args.nccl_timeout_s,
                       force_cpu=args.force_cpu)
    device = info.device
    if args.force_cpu:
        device = "cpu"

    equiv = bool(args.equiv_check)
    dtype_name = "float32" if equiv else args.dtype
    torch_dtype = C.assert_dtype_supported(dtype_name, device)
    per_rank_seed = False if equiv else bool(args.seed_per_rank)
    C.set_seed(args.seed, per_rank=per_rank_seed)

    model, lm_config, cfg, mm = C.build_model(
        args.config, minimind_root=args.minimind_root, device=device)
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
            + str(info.world_size) + "*" + str(accum) + "=" + str(denom) + " 整除；"
            "请调整 --global-batch 或 --accum。"
        )
    micro = global_batch // denom

    dataset = C.build_dataset(
        args.dataset, seq_len=seq_len, vocab_size=int(lm_config.vocab_size),
        minimind_root=args.minimind_root, data_path=args.data_path,
        num_samples=int(args.num_samples), seed=int(train_cfg.get("data_seed", 1234)))

    ddp_model = model
    if info.initialized and info.world_size > 1:
        device_ids = [info.local_rank] if str(device).startswith("cuda") else None
        ddp_model = DistributedDataParallel(model, device_ids=device_ids)

    optimizer = torch.optim.AdamW(ddp_model.parameters(), lr=base_lr)
    scaler = C.make_grad_scaler(torch_dtype, device)
    C.reset_peak_mem(device)

    name = args.log_name or ("ddp_ws" + str(info.world_size))
    logger = C.JsonlLogger(args.out_dir, name, rank=info.rank)
    total, trainable = C.count_params(model)
    header = {
        "step": 0,
        "event": "header",
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
        "minimind_commit": mm.commit,
        "env": C.env_summary(),
    }
    logger.write(header)
    C.log_main("[header] " + str({k: header[k] for k in (
        "world_size", "dtype", "global_batch", "accum", "micro_batch",
        "seq_len", "params_total", "use_moe")}))

    ddp_model.train()
    skipped_steps = 0
    losses: List[float] = []
    total_steps = max(int(args.max_steps), 1)
    for step in range(1, total_steps + 1):
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
            sync_ctx = nullcontext()
            if isinstance(ddp_model, DistributedDataParallel) and not is_last:
                sync_ctx = ddp_model.no_sync()
            with sync_ctx:
                with C.sdpa_context(args.sdpa_backend, device):
                    with C.autocast_context(torch_dtype, device):
                        res = ddp_model(input_ids, labels=labels)
                        logits_loss = res.loss
                        aux_loss = res.aux_loss
                        if aux_loss is None:
                            aux_loss = logits_loss.new_zeros(())
                        loss = logits_loss + aux_loss
                scaler.scale(loss / accum).backward()
            sums[0] += loss.detach().float()
            sums[1] += logits_loss.detach().float()
            sums[2] += aux_loss.detach().float()

        if info.initialized and info.world_size > 1:
            dist.all_reduce(sums, op=dist.ReduceOp.SUM)
        sums = sums / float(denom)

        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(ddp_model.parameters(), grad_clip)
        scale_before = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        if scaler.get_scale() < scale_before:
            skipped_steps += 1

        dt = max(time.perf_counter() - t0, 1e-9)
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
        }
        logger.write(record)
        losses.append(record["loss"])
        if equiv and step <= int(args.equiv_steps):
            C.log_main("[equiv] ws=%d step=%d loss=%.8f grad_norm=%.6f"
                       % (info.world_size, step, record["loss"], record["grad_norm"]))
        elif step % max(int(args.log_interval), 1) == 0 or step == total_steps:
            C.log_main("[step %d/%d] loss=%.4f logits=%.4f aux=%.6f lr=%.3e "
                       "gn=%.3f scale=%.0f skipped=%d tok/s=%.0f peak_mem_mb=%.1f"
                       % (step, total_steps, record["loss"], record["logits_loss"],
                          record["aux_loss"], lr, record["grad_norm"],
                          record["scaler_scale"], skipped_steps,
                          record["tokens_per_s"], record["peak_mem_mb"]))

    logger.close()
    summary = {
        "world_size": info.world_size,
        "steps": total_steps,
        "losses": losses,
        "skipped_steps": skipped_steps,
        "log_path": logger.path,
        "micro_batch": micro,
    }
    C.cleanup_dist(info)
    return summary


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="MiniMind DDP 训练 / fixed-global-batch 等价检查 / 日志比对",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    C.add_common_args(p)
    p.add_argument("--global-batch", type=int, default=None,
                   help="全局 batch（覆盖配置里的 train.global_batch）")
    p.add_argument("--accum", type=int, default=None, help="梯度累积步数")
    p.add_argument("--seq-len", type=int, default=None, help="序列长度")
    p.add_argument("--lr", type=float, default=None, help="基础学习率")
    p.add_argument("--dataset", type=str, default="tiny", choices=["tiny", "real"])
    p.add_argument("--data-path", type=str, default=None,
                   help="--dataset real 时的 jsonl 路径")
    p.add_argument("--num-samples", type=int, default=8192,
                   help="TinyDataset 的样本数（必须 >= max_steps*global_batch 才不回绕）")
    p.add_argument("--sdpa-backend", type=str, default="auto",
                   choices=["auto", "math", "mem_efficient", "flash"],
                   help="仅 CUDA 生效；V100 在 torch 2.1 下没有 flash")
    p.add_argument("--log-interval", type=int, default=1)
    p.add_argument("--log-name", type=str, default=None,
                   help="日志文件名前缀，默认 ddp_ws<W>")
    p.add_argument("--equiv-check", action="store_true",
                   help="等价模式：强制 float32、非 per-rank seed，打印前 N 步 loss")
    p.add_argument("--equiv-steps", type=int, default=5)
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
