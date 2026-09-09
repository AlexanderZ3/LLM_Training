"""Day 2 故障演练：依次跑 none / label_shift / scaler_stuck / ddp_no_sync，
只凭三个不变量把它们区分开，输出一张判别表。

用法::

    # 本机（2 进程 gloo，四档全跑，含 ddp_no_sync）
    python lab/scripts/run_faults.py --config tiny_cpu --nproc 2 --force-cpu \\
        --dtype float16 --max-steps 4

    # V100 8 卡
    python lab/scripts/run_faults.py --config smoke_v100 --nproc 8 \\
        --dtype float16 --max-steps 6 --launcher torchrun

退出码：0 = 四档各自被正确的不变量抓住；1 = 有一档没被抓住或被抓错。

为什么这条命令要自己起子进程
----------------------------
ddp_no_sync 这一档只有 world_size>=2 才有内容。把「起几个进程」这件事写进
Python，本机（没有 torchrun）和 V100（有 torchrun）能用同一条命令的同一套逻辑，
只是 --launcher 不同。
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Dict, List

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(os.path.dirname(_HERE), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from mm_v100 import bounded_train as BT  # noqa: E402
from mm_v100 import cli  # noqa: E402
from mm_v100 import common as C  # noqa: E402
from mm_v100 import faults as FA  # noqa: E402

MODES = ("none", "label_shift", "scaler_stuck", "ddp_no_sync")


def worker(rank: int, world_size: int, port: int, argv: List[str]) -> None:
    """spawn 的每个进程执行的函数（必须是模块顶层函数）。"""
    C.set_worker_env(rank, world_size, port)
    if _SRC not in sys.path:
        sys.path.insert(0, _SRC)
    from mm_v100 import bounded_train as inner

    inner.main(list(argv))


def read_last_record(log_path: str) -> Dict[str, Any]:
    rows = C.read_jsonl(log_path)
    steps = [r for r in rows if r.get("event") != "header" and "step" in r]
    if not steps:
        raise RuntimeError(
            "日志 " + log_path + " 里没有 step 记录。\n"
            "下一步：看同目录 stdout；多半是第一步就抛异常了。")
    return steps[-1]


def run_one(mode: str, args: argparse.Namespace, out_dir: str) -> Dict[str, Any]:
    """跑一档，返回最后一步的三个不变量与分类结论。"""
    tag = "fault_" + mode
    child_argv = [
        "--config", args.config,
        "--out-dir", out_dir,
        "--placement", args.placement,
        "--stage", args.stage,
        "--dtype", args.dtype,
        "--max-steps", str(args.max_steps),
        "--global-batch", str(args.global_batch),
        "--accum", str(args.accum),
        "--seed", str(args.seed),
        "--seed-per-rank", "0",
        "--fault", mode,
        "--log-name", tag,
        "--log-interval", str(args.max_steps),
    ]
    if args.force_cpu:
        child_argv.append("--force-cpu")
    if args.backend != "auto":
        child_argv += ["--backend", args.backend]

    if args.nproc > 1:
        import torch.multiprocessing as mp

        port = C.free_port()
        mp.spawn(worker, args=(args.nproc, port, child_argv),
                 nprocs=args.nproc, join=True)
    else:
        BT.main(child_argv)

    log_path = os.path.join(out_dir, tag + "_rank0.jsonl")
    last = read_last_record(log_path)
    obs = {k: last.get(k) for k in ("I1_label_align_violations",
                                    "I2_optimizer_step_ratio",
                                    "I2_param_delta_norm",
                                    "I3_cross_rank_grad_delta")}
    result = FA.classify(obs)
    result.update({
        "mode": mode,
        "final_loss": last.get("loss"),
        "grad_norm": last.get("grad_norm"),
        "scaler_scale": last.get("scaler_scale"),
        "log_path": log_path,
        "detected_correctly": result["verdict"] == mode,
    })
    return result


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="三档静默故障注入 + 三不变量判别表",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--config", type=str, default="tiny_cpu")
    p.add_argument("--placement", type=str, default="ddp",
                   choices=["single", "ddp", "fsdp"])
    p.add_argument("--stage", type=str, default="sft",
                   choices=["pretrain", "sft"])
    p.add_argument("--dtype", type=str, default="float16",
                   choices=["float32", "float16", "bfloat16"],
                   help="scaler_stuck 这一档需要 float16 才有内容")
    p.add_argument("--nproc", type=int, default=2,
                   help="进程数；ddp_no_sync 这一档需要 >=2")
    p.add_argument("--launcher", type=str, default="spawn",
                   choices=["spawn", "torchrun"],
                   help="spawn=本脚本自己起进程；torchrun=只提示命令，不代跑")
    p.add_argument("--max-steps", type=int, default=4)
    p.add_argument("--global-batch", type=int, default=8)
    p.add_argument("--accum", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--backend", type=str, default="gloo",
                   choices=["auto", "gloo", "nccl"])
    p.add_argument("--force-cpu", action="store_true")
    p.add_argument("--modes", type=str, default=",".join(MODES),
                   help="要跑的档位，逗号分隔")
    p.add_argument("--out-dir", type=str, default=None)
    p.add_argument("--run-tag", type=str, default="day2_sft")
    p.add_argument("--dry-run", action="store_true",
                   help="只打印将要跑的四条子命令和判别表模板，不真跑")
    return p


def main(argv=None) -> int:
    cli.setup()
    args = build_parser().parse_args(argv)
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    for m in modes:
        if m not in MODES:
            raise SystemExit("不认识的档位 " + repr(m) + "；可选 " + ", ".join(MODES))
    out_dir = cli.resolve_out_dir(args.out_dir, args.run_tag)

    if args.launcher == "torchrun":
        print("torchrun 模式下本脚本不代跑，请逐条执行：")
        for m in modes:
            print("  torchrun --nproc_per_node %d lab/scripts/train_bounded.py "
                  "--config %s --out-dir %s --placement %s --stage %s "
                  "--dtype %s --max-steps %d --global-batch %d --accum %d "
                  "--fault %s --log-name fault_%s --seed-per-rank 0"
                  % (args.nproc, args.config, out_dir, args.placement,
                     args.stage, args.dtype, args.max_steps, args.global_batch,
                     args.accum, m, m))
        print("")
        print("跑完之后用同一个 --out-dir 再执行一次本脚本（--launcher spawn "
              "--modes 留空不跑）来汇总，或直接看各 jsonl 的最后一行。")
        return 0

    if args.dry_run:
        print("[dry-run] 将跑的档位：" + ", ".join(modes))
        print("[dry-run] out_dir=" + out_dir + " nproc=" + str(args.nproc))
        print("")
        print(FA.render_decision_table())
        return 0

    if args.nproc < 2 and "ddp_no_sync" in modes:
        print("[warn] nproc=1 时 ddp_no_sync 没有内容（I3 恒为 None），已跳过该档",
              flush=True)
        modes = [m for m in modes if m != "ddp_no_sync"]

    results = [run_one(m, args, out_dir) for m in modes]
    print("")
    print("== Day 2 判别表（实测）==")
    print("%-14s %-10s %-14s %-14s %-16s %s"
          % ("mode", "I1", "I2_ratio", "I2_delta", "I3", "verdict"))
    for r in results:
        obs = r["observations"]
        print("%-14s %-10s %-14.3f %-14.3e %-16s %s"
              % (r["mode"], obs.get("I1_label_align_violations"),
                 float(obs.get("I2_optimizer_step_ratio") or 0.0),
                 float(obs.get("I2_param_delta_norm") or 0.0),
                 str(obs.get("I3_cross_rank_grad_delta")), r["verdict"]))
    print("")
    print(FA.render_decision_table())
    ok = all(r["detected_correctly"] for r in results)
    payload = {
        "schema": "mm_v100.run_faults/1",
        "config": args.config,
        "nproc": args.nproc,
        "dtype": args.dtype,
        "max_steps": args.max_steps,
        "results": results,
        "all_detected": ok,
    }
    path = cli.write_json(payload, os.path.join(out_dir, "fault_table.json"))
    print("[faults] -> " + path)
    if not ok:
        wrong = [r["mode"] + "->" + r["verdict"]
                 for r in results if not r["detected_correctly"]]
        print("[faults] 判别错误：" + ", ".join(wrong))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
