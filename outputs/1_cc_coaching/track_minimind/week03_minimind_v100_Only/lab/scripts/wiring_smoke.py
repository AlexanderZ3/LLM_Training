"""Day 0 的唯一早期闸门：world_size 1 -> 2 -> 8，每档 2 步，只看「能不能起来」。

用法::

    # 本机（CPU + gloo，验的是脚本接线，不是 NCCL）
    python lab/scripts/wiring_smoke.py --sizes 1,2 --force-cpu --config tiny_cpu

    # V100（NCCL）
    python lab/scripts/wiring_smoke.py --sizes 1,2,8 --config smoke_v100 \\
        --dtype float16

退出码：0 = 每一档都跑完了 --steps 步；1 = 有一档失败（失败的 world_size 打在最后）。

为什么这一步不可裁
------------------
本周所有 GPU 代码在 cc 那边一行都没在 GPU 上跑过。从 1 卡到 8 卡之间，
会坏的东西按出现顺序是：import、配置解析、模型构建、单卡前反向、进程组建立、
NCCL 握手、集合通信、显存。一次跑 8 卡失败时，这 8 件事你分不清是哪一件。
按 1 -> 2 -> 8 走，失败点直接被夹在两档之间。

耗时：三档合计 2-3 分钟，远小于「Day 3 才发现 NCCL 起不来」的代价。
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from typing import Any, Dict, List

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(os.path.dirname(_HERE), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import torch  # noqa: E402

from mm_v100 import bounded_train as BT  # noqa: E402
from mm_v100 import cli  # noqa: E402
from mm_v100 import common as C  # noqa: E402


def worker(rank: int, world_size: int, port: int, argv: List[str]) -> None:
    C.set_worker_env(rank, world_size, port)
    if _SRC not in sys.path:
        sys.path.insert(0, _SRC)
    from mm_v100 import bounded_train as inner

    inner.main(list(argv))


def child_argv(world_size: int, args: argparse.Namespace,
               out_dir: str) -> List[str]:
    argv = [
        "--config", args.config,
        "--out-dir", out_dir,
        "--placement", ("ddp" if world_size > 1 else "single"),
        "--dtype", args.dtype,
        "--max-steps", str(args.steps),
        "--global-batch", str(args.global_batch),
        "--accum", "1",
        "--seed", "42",
        "--seed-per-rank", "0",
        "--log-name", "smoke_ws" + str(world_size),
        "--log-interval", str(args.steps),
    ]
    if args.force_cpu:
        argv.append("--force-cpu")
    if args.backend != "auto":
        argv += ["--backend", args.backend]
    return argv


def run_one(world_size: int, args: argparse.Namespace,
            out_dir: str) -> Dict[str, Any]:
    argv = child_argv(world_size, args, out_dir)
    t0 = time.perf_counter()
    ok = True
    error = None
    try:
        if world_size == 1:
            BT.main(argv)
        elif args.launcher == "spawn":
            import torch.multiprocessing as mp

            port = C.free_port()
            mp.spawn(worker, args=(world_size, port, argv), nprocs=world_size,
                     join=True)
        else:
            cmd = [sys.executable, "-m", "torch.distributed.run",
                   "--nproc_per_node", str(world_size),
                   os.path.join(_HERE, "train_bounded.py")] + argv
            env = dict(os.environ)
            env["PYTHONPATH"] = _SRC + os.pathsep + env.get("PYTHONPATH", "")
            proc = subprocess.run(cmd, env=env, check=False)
            # 显式检查退出码，不靠 shell 的 set -e（CLAUDE.md 3.2）。
            if proc.returncode != 0:
                ok = False
                error = "torchrun 退出码 " + str(proc.returncode)
    except Exception as exc:  # noqa: BLE001 - smoke 要报出全部失败类型
        ok = False
        error = type(exc).__name__ + ": " + str(exc).splitlines()[0][:300]
    dt = time.perf_counter() - t0
    log_path = os.path.join(out_dir, "smoke_ws" + str(world_size)
                            + "_rank0.jsonl")
    n_steps = 0
    if os.path.isfile(log_path):
        n_steps = len([r for r in C.read_jsonl(log_path)
                       if r.get("event") != "header" and "loss" in r])
    completed = ok and n_steps >= args.steps
    return {
        "world_size": world_size,
        "ok": bool(completed),
        "steps_logged": n_steps,
        "wall_s": dt,
        "error": error,
    }


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="wiring smoke：world_size 逐档放大，每档 --steps 步",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--sizes", type=str, default="1,2",
                   help="要试的 world_size，逗号分隔（V100 上用 1,2,8）")
    p.add_argument("--config", type=str, default="tiny_cpu")
    p.add_argument("--dtype", type=str, default="float32",
                   choices=["float32", "float16", "bfloat16"])
    p.add_argument("--steps", type=int, default=2, help="每档跑几步")
    p.add_argument("--global-batch", type=int, default=8,
                   help="必须能被最大的 world_size 整除")
    p.add_argument("--launcher", type=str, default="spawn",
                   choices=["spawn", "torchrun"],
                   help="spawn=本脚本自己起进程（本机唯一可用）；"
                        "torchrun=调 torch.distributed.run")
    p.add_argument("--backend", type=str, default="auto",
                   choices=["auto", "gloo", "nccl"])
    p.add_argument("--force-cpu", action="store_true")
    p.add_argument("--out-dir", type=str, default=None)
    p.add_argument("--run-tag", type=str, default="day0_probe")
    p.add_argument("--dry-run", action="store_true",
                   help="只打印每档将要执行的参数，不真跑")
    return p


def main(argv=None) -> int:
    cli.setup()
    args = build_parser().parse_args(argv)
    sizes = [int(s) for s in args.sizes.split(",") if s.strip()]
    for ws in sizes:
        if args.global_batch % ws != 0:
            raise SystemExit(
                "global_batch=" + str(args.global_batch) + " 不能被 world_size="
                + str(ws) + " 整除。下一步：把 --global-batch 调成它们的公倍数。")
    available = torch.cuda.device_count() if torch.cuda.is_available() else 0
    out_dir = cli.resolve_out_dir(args.out_dir, args.run_tag)

    if args.dry_run:
        for ws in sizes:
            print("[dry-run] world_size=%d argv=%s"
                  % (ws, " ".join(child_argv(ws, args, out_dir))))
        print("[dry-run] 本机可见 GPU 数 = %d" % available)
        return 0

    results: List[Dict[str, Any]] = []
    for ws in sizes:
        if not args.force_cpu and available and ws > available:
            results.append({"world_size": ws, "ok": False, "steps_logged": 0,
                            "wall_s": 0.0,
                            "error": "本机只有 " + str(available) + " 张卡"})
            print("[smoke ws=%d] 跳过：本机只有 %d 张卡" % (ws, available))
            continue
        print("[smoke ws=%d] 开始" % ws, flush=True)
        res = run_one(ws, args, out_dir)
        results.append(res)
        print("[smoke ws=%d] %s  steps=%d  %.1fs%s"
              % (ws, "PASS" if res["ok"] else "FAIL", res["steps_logged"],
                 res["wall_s"], ("  " + str(res["error"])) if res["error"] else ""),
              flush=True)
        if not res["ok"]:
            print("[smoke] 在 world_size=%d 处停止：更大的规模只会掩盖这个问题。" % ws)
            break
    ok = all(r["ok"] for r in results) and len(results) == len(sizes)
    payload = {"schema": "mm_v100.wiring_smoke/1", "sizes": sizes,
               "config": args.config, "dtype": args.dtype,
               "steps_per_size": args.steps, "results": results,
               "all_pass": ok, "visible_gpus": available}
    path = cli.write_json(payload, os.path.join(out_dir, "wiring_smoke.json"))
    print("[smoke] -> " + path)
    if not ok:
        failed = [str(r["world_size"]) for r in results if not r["ok"]]
        print("[smoke] 未通过的 world_size：" + ", ".join(failed))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
