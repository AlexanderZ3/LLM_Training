"""故障注入：kill 一个 rank，以及 NCCL 超时/错误处理的环境变量助手。

为什么用 ``os._exit(1)``
------------------------
``sys.exit`` 抛 ``SystemExit``，会走 atexit、``__del__``、以及 torch 的
``destroy_process_group``，那样其余 rank 会收到一次"体面"的断开，复现不出真实的
掉卡。``os._exit(1)`` 立刻结束进程，不跑任何清理——其余 rank 看到的是
socket 断开或 collective 永远等不到，这才是要练的那种故障。

其余 rank 会怎样（01_FOUNDATIONS.md 失败模式 #2 / #11）
-----------------------------------------------------
* 默认（``TORCH_NCCL_ASYNC_ERROR_HANDLING`` 未设）：其余 rank 卡在下一次集合通信，
  一直等到进程组 timeout（``init_process_group(timeout=...)``，torch 默认 1800 秒）。
  ``nvidia-smi`` 上看利用率可能是 100%（spin wait），也可能是 0%。
* 设为 ``1``：watchdog 检测到超时后直接让进程崩溃，退出码非 0，
  外层 ``torchrun`` 能感知并结束整个 job——这才有可能自动重启。

torch 2.1 的变量名
------------------
2.1 同时认 ``NCCL_ASYNC_ERROR_HANDLING`` 与 ``TORCH_NCCL_ASYNC_ERROR_HANDLING``
（后者是 2.2 起的正式名，2.1 已经在读）。本文件两个都设，避免版本差异。
``NCCL_BLOCKING_WAIT`` / ``TORCH_NCCL_BLOCKING_WAIT`` 同理。

用法
----
两卡 CPU 复现（本机可跑）::

    torchrun --nproc_per_node 2 -m mm_dist.faults demo \
        --config configs/tiny_cpu.json --out-dir ./runs --force-cpu --backend gloo \
        --dtype float32 --global-batch 8 --accum 1 --max-steps 6 \
        --kill-rank 1 --after-step 3 --nccl-timeout-s 60

八卡 V100::

    torchrun --nproc_per_node 8 -m mm_dist.faults demo \
        --config configs/v100_dense.json --out-dir <公司内路径> \
        --dtype float16 --global-batch 64 --max-steps 20 \
        --kill-rank 3 --after-step 10 --nccl-timeout-s 120
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
from torch.nn.parallel import DistributedDataParallel

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mm_dist import common as C  # noqa: E402

ASYNC_ERROR_VARS = ("TORCH_NCCL_ASYNC_ERROR_HANDLING", "NCCL_ASYNC_ERROR_HANDLING")
BLOCKING_WAIT_VARS = ("TORCH_NCCL_BLOCKING_WAIT", "NCCL_BLOCKING_WAIT")
WATCHED_VARS = ASYNC_ERROR_VARS + BLOCKING_WAIT_VARS + (
    "NCCL_DEBUG", "NCCL_DEBUG_SUBSYS", "NCCL_IB_DISABLE", "NCCL_P2P_DISABLE",
    "TORCH_DISTRIBUTED_DEBUG",
)


# ---------------------------------------------------------------------------
# kill
# ---------------------------------------------------------------------------


def kill_rank(rank: int, after_step: int, current_rank: int, current_step: int,
              flush_paths: Optional[List[str]] = None) -> bool:
    """当 ``current_rank == rank`` 且 ``current_step >= after_step`` 时立刻杀掉本进程。

    返回值只有在**不该杀**时才有意义（返回 False）；该杀时函数不返回。
    ``flush_paths`` 里的文件描述符由调用方在进入前关闭；这里只把 stdout/stderr
    刷出去，保证"我死在第几步"这行日志不丢。
    """
    if int(rank) < 0 or int(current_rank) != int(rank):
        return False
    if int(current_step) < int(after_step):
        return False
    msg = ("[fault] rank %d 在 step %d 执行 os._exit(1)（模拟掉卡）。"
           "其余 rank 现在会卡在下一次集合通信，直到进程组 timeout 或 "
           "TORCH_NCCL_ASYNC_ERROR_HANDLING=1 触发 watchdog 崩溃。"
           % (int(current_rank), int(current_step)))
    print(msg, flush=True)
    sys.stdout.flush()
    sys.stderr.flush()
    for path in (flush_paths or []):
        try:
            with open(path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps({"event": "killed", "rank": int(current_rank),
                                     "step": int(current_step)},
                                    ensure_ascii=False) + "\n")
        except OSError:
            pass
    os._exit(1)
    return True  # 不可达；留着让静态检查看到返回类型一致


class KillRankInjector:
    """把 :func:`kill_rank` 包成训练循环里一行调用。"""

    def __init__(self, rank: int, after_step: int, current_rank: int,
                 log_path: Optional[str] = None) -> None:
        self.rank = int(rank)
        self.after_step = int(after_step)
        self.current_rank = int(current_rank)
        self.log_path = log_path
        self.armed = self.rank >= 0

    def maybe_kill(self, step: int) -> None:
        if not self.armed:
            return
        kill_rank(self.rank, self.after_step, self.current_rank, step,
                  flush_paths=[self.log_path] if self.log_path else None)

    def describe(self) -> Dict[str, Any]:
        return {"kill_rank": self.rank, "after_step": self.after_step,
                "armed": self.armed}


# ---------------------------------------------------------------------------
# NCCL 环境
# ---------------------------------------------------------------------------


def set_nccl_timeout_env(async_error_handling: bool = True,
                         blocking_wait: bool = False,
                         debug: Optional[str] = None,
                         overwrite: bool = False) -> Dict[str, str]:
    """设置 NCCL 错误处理相关环境变量，返回**本次实际写入**的键值。

    必须在 ``init_process_group`` 之前调用——进程组建好之后再改这些变量不生效。
    ``overwrite=False`` 时不覆盖用户/公司启动脚本已经设好的值（公司环境优先）。

    注意：超时**时长**不是环境变量，而是
    ``dist.init_process_group(timeout=timedelta(seconds=...))``；本 lab 所有入口
    都有 ``--nccl-timeout-s``，默认 600 秒而不是 torch 的 1800 秒，
    好让公司机器上的故障演练在 10 分钟预算内出结果。
    """
    written: Dict[str, str] = {}
    pairs: List[tuple] = []
    for name in ASYNC_ERROR_VARS:
        pairs.append((name, "1" if async_error_handling else "0"))
    for name in BLOCKING_WAIT_VARS:
        pairs.append((name, "1" if blocking_wait else "0"))
    if debug:
        pairs.append(("NCCL_DEBUG", str(debug)))
    for key, value in pairs:
        if overwrite or key not in os.environ:
            os.environ[key] = value
            written[key] = value
    return written


def nccl_env_report() -> Dict[str, Optional[str]]:
    """当前进程里与 NCCL 故障处理相关的环境变量快照（不含任何拓扑信息）。"""
    return {name: os.environ.get(name) for name in WATCHED_VARS}


def recommended_env_lines(timeout_s: int = 600) -> List[str]:
    """打印给用户抄进启动脚本的几行（bash 语法）。"""
    return [
        "export TORCH_NCCL_ASYNC_ERROR_HANDLING=1   # 超时改为崩溃，不要静默 hang",
        "export NCCL_ASYNC_ERROR_HANDLING=1         # torch 2.1 的旧名，一起设",
        "export TORCH_NCCL_BLOCKING_WAIT=0          # 与 async 错误处理互斥，保持 0",
        "export NCCL_DEBUG=WARN                     # INFO 会打印拓扑，公司环境慎用",
        "# 超时时长不是环境变量，用命令行参数：--nccl-timeout-s " + str(int(timeout_s)),
    ]


# ---------------------------------------------------------------------------
# demo：一个会在指定 step 掉卡的最小 DDP 训练
# ---------------------------------------------------------------------------


def cmd_demo(args: argparse.Namespace) -> int:
    set_nccl_timeout_env(async_error_handling=bool(args.async_error_handling),
                         blocking_wait=False,
                         debug=args.nccl_debug, overwrite=bool(args.overwrite_env))
    info = C.init_dist(backend=args.backend, timeout_s=args.nccl_timeout_s,
                       force_cpu=args.force_cpu)
    device = "cpu" if args.force_cpu else info.device
    torch_dtype = C.assert_dtype_supported(args.dtype, device)
    C.set_seed(args.seed, per_rank=False)

    model, lm_config, cfg, mm = C.build_model(
        args.config, minimind_root=args.minimind_root, device=device)
    train_cfg = cfg["train"]
    seq_len = int(args.seq_len or train_cfg["seq_len"])
    global_batch = int(args.global_batch or train_cfg["global_batch"])
    accum = int(args.accum or train_cfg.get("accum", 1))
    base_lr = float(train_cfg.get("lr", 5e-4))
    denom = info.world_size * accum
    if global_batch % denom != 0:
        raise SystemExit("global_batch 必须能被 world_size*accum 整除")

    dataset = C.build_dataset("tiny", seq_len=seq_len,
                              vocab_size=int(lm_config.vocab_size),
                              minimind_root=args.minimind_root,
                              num_samples=int(args.num_samples),
                              seed=int(train_cfg.get("data_seed", 1234)))
    ddp_model = model
    if info.initialized and info.world_size > 1:
        device_ids = [info.local_rank] if str(device).startswith("cuda") else None
        ddp_model = DistributedDataParallel(model, device_ids=device_ids)
    optimizer = torch.optim.AdamW(ddp_model.parameters(), lr=base_lr)
    scaler = C.make_grad_scaler(torch_dtype, device)

    logger = C.JsonlLogger(args.out_dir, args.log_name or "fault_kill", rank=info.rank)
    injector = KillRankInjector(args.kill_rank, args.after_step, info.rank,
                                log_path=logger.path)
    logger.write({"step": 0, "event": "header", "runner": "faults.demo",
                  "world_size": info.world_size, "dtype": args.dtype,
                  "nccl_timeout_s": args.nccl_timeout_s,
                  "nccl_env": nccl_env_report(),
                  "injector": injector.describe(), "env": C.env_summary()})
    C.log_main("[fault] " + json.dumps(injector.describe()) + "  nccl_env="
               + json.dumps(nccl_env_report()))

    steps = max(int(args.max_steps), 1)
    t_start = time.perf_counter()
    for step in range(1, steps + 1):
        injector.maybe_kill(step)  # 先杀再算：被杀的 rank 不参与本步的任何 collective
        t0 = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        total = torch.zeros(1, device=device, dtype=torch.float32)
        for acc_idx in range(accum):
            idx = C.micro_batch_indices(step, global_batch, info.world_size,
                                        accum, info.rank, acc_idx)
            input_ids, labels = C.collate_indices(dataset, idx, device)
            with C.autocast_context(torch_dtype, device):
                res = ddp_model(input_ids, labels=labels)
                aux = res.aux_loss if res.aux_loss is not None else res.loss.new_zeros(())
                loss = res.loss + aux
            scaler.scale(loss / accum).backward()
            total[0] += loss.detach().float()
        if info.world_size > 1:
            dist.all_reduce(total, op=dist.ReduceOp.SUM)
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(ddp_model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        record = {"step": step, "loss": float(total[0] / denom),
                  "step_time_s": time.perf_counter() - t0,
                  "elapsed_s": time.perf_counter() - t_start,
                  "world_size": info.world_size,
                  "peak_mem_mb": C.peak_mem_mb(device)}
        logger.write(record)
        print("[fault demo step %d/%d] rank=%d loss=%.4f elapsed=%.1fs"
              % (step, steps, info.rank, record["loss"], record["elapsed_s"]),
              flush=True)

    logger.close()
    C.cleanup_dist(info)
    return 0


def cmd_env(args: argparse.Namespace) -> int:
    report = {
        "current": nccl_env_report(),
        "recommended_lines": recommended_env_lines(args.nccl_timeout_s),
        "note": ("超时时长由 init_process_group(timeout=) 决定，本 lab 用 "
                 "--nccl-timeout-s；环境变量只控制'超时之后做什么'。"),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), flush=True)
    for line in report["recommended_lines"]:
        print(line, flush=True)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="故障注入（kill rank）与 NCCL 超时/错误处理环境助手",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    C.add_common_args(p)
    p.add_argument("command", choices=["demo", "env"],
                   help="demo=跑一段会掉卡的 DDP 训练；env=只打印/建议环境变量")
    p.add_argument("--kill-rank", type=int, default=-1,
                   help="要杀掉的 rank；-1 表示不杀")
    p.add_argument("--after-step", type=int, default=3,
                   help="从第几步（含）开始杀")
    p.add_argument("--async-error-handling", type=int, default=1, choices=[0, 1],
                   help="1=设 TORCH_NCCL_ASYNC_ERROR_HANDLING=1（超时改为崩溃）")
    p.add_argument("--overwrite-env", type=int, default=0, choices=[0, 1],
                   help="1=覆盖已存在的环境变量；默认不覆盖公司启动脚本的设置")
    p.add_argument("--nccl-debug", type=str, default=None,
                   help="NCCL_DEBUG 取值；INFO 会打印拓扑，公司环境建议留空或 WARN")
    p.add_argument("--global-batch", type=int, default=None)
    p.add_argument("--accum", type=int, default=None)
    p.add_argument("--seq-len", type=int, default=None)
    p.add_argument("--num-samples", type=int, default=8192)
    p.add_argument("--log-name", type=str, default=None)
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "env":
        return cmd_env(args)
    if not args.config:
        raise SystemExit("demo 必须给 --config configs/<name>.json")
    return cmd_demo(args)


if __name__ == "__main__":
    raise SystemExit(main())
