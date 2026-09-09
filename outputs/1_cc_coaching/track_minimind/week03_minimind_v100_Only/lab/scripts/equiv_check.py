"""Day 2 等价检查：跑两个 world_size 的固定全局批训练，逐步比 loss。

用法::

    # 本机：1 进程 accum=2 vs 2 进程 accum=1，全局批都是 8
    python lab/scripts/equiv_check.py --config tiny_cpu --force-cpu \\
        --global-batch 8 --a-nproc 1 --a-accum 2 --b-nproc 2 --b-accum 1

    # 只比对两条已有日志（V100 上 torchrun 跑完之后）
    python lab/scripts/equiv_check.py --compare A.jsonl B.jsonl --tol 1e-4

退出码：0 = max|delta| < --tol；1 = 超出容差。

容差为什么取 1e-4
-----------------
它比 gloo/NCCL 的 float32 归约顺序差异大两个数量级，又比「数据切错了」的差异
（通常 >1e-2）小两个数量级。所以落在 1e-4 以内基本只能是归约顺序，
超出 1e-2 基本只能是切分或 seed 出了问题。中间那一段要看 grad_norm 才能定性。
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


def worker(rank: int, world_size: int, port: int, argv: List[str]) -> None:
    C.set_worker_env(rank, world_size, port)
    if _SRC not in sys.path:
        sys.path.insert(0, _SRC)
    from mm_v100 import bounded_train as inner

    inner.main(list(argv))


def compare_logs(path_a: str, path_b: str, key: str = "loss",
                 tol: float = 1e-4) -> Dict[str, Any]:
    """逐 step 比较两条 JSONL 的某个字段。"""
    def load(path: str) -> Dict[int, Dict[str, Any]]:
        return {int(r["step"]): r for r in C.read_jsonl(path)
                if r.get("event") != "header" and "step" in r and key in r}

    rec_a, rec_b = load(path_a), load(path_b)
    steps = sorted(set(rec_a) & set(rec_b))
    if not steps:
        raise ValueError(
            "两条日志没有公共 step。A 有 " + str(sorted(rec_a)[:5])
            + "，B 有 " + str(sorted(rec_b)[:5]) + "。\n"
            "下一步：确认两次运行的 --max-steps 一样，且都写到了 rank0 的日志。")
    deltas: List[float] = []
    print("step |         A |         B |   |delta|")
    print("-----+-----------+-----------+----------")
    for s in steps:
        va, vb = float(rec_a[s][key]), float(rec_b[s][key])
        d = abs(va - vb)
        deltas.append(d)
        print("%4d | %9.6f | %9.6f | %.3e" % (s, va, vb, d))
    max_delta = max(deltas)
    result = {
        "key": key,
        "steps_compared": len(steps),
        "max_abs_delta": max_delta,
        "mean_abs_delta": sum(deltas) / len(deltas),
        "tol": float(tol),
        "pass": bool(max_delta < tol),
        "a": os.path.abspath(path_a),
        "b": os.path.abspath(path_b),
    }
    print("")
    print("steps=%d  max|delta|=%.3e  mean|delta|=%.3e  tol=%.1e  -> %s"
          % (result["steps_compared"], result["max_abs_delta"],
             result["mean_abs_delta"], tol, "PASS" if result["pass"] else "FAIL"))
    return result


def run_side(label: str, nproc: int, accum: int, args: argparse.Namespace,
             out_dir: str) -> str:
    name = "equiv_" + label
    child_argv = [
        "--config", args.config,
        "--out-dir", out_dir,
        "--placement", "ddp" if nproc > 1 else "single",
        "--stage", args.stage,
        "--dtype", "float32",
        "--equiv-check",
        "--max-steps", str(args.max_steps),
        "--global-batch", str(args.global_batch),
        "--accum", str(accum),
        "--seed", str(args.seed),
        "--seed-per-rank", "0",
        "--log-name", name,
        "--log-interval", str(args.max_steps),
    ]
    if args.force_cpu:
        child_argv.append("--force-cpu")
    if args.backend != "auto":
        child_argv += ["--backend", args.backend]
    if nproc > 1:
        import torch.multiprocessing as mp

        port = C.free_port()
        mp.spawn(worker, args=(nproc, port, child_argv), nprocs=nproc, join=True)
    else:
        BT.main(child_argv)
    return os.path.join(out_dir, name + "_rank0.jsonl")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="fixed-global-batch 等价检查（两个 world_size 的 loss 逐步相减）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--compare", nargs=2, metavar=("A_JSONL", "B_JSONL"),
                   default=None, help="只比对两条已有日志，不训练")
    p.add_argument("--config", type=str, default="tiny_cpu")
    p.add_argument("--stage", type=str, default="pretrain",
                   choices=["pretrain", "sft"])
    p.add_argument("--global-batch", type=int, default=8,
                   help="两侧必须相同，这是等价的前提")
    p.add_argument("--a-nproc", type=int, default=1)
    p.add_argument("--a-accum", type=int, default=2)
    p.add_argument("--b-nproc", type=int, default=2)
    p.add_argument("--b-accum", type=int, default=1)
    p.add_argument("--max-steps", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--tol", type=float, default=1e-4)
    p.add_argument("--key", type=str, default="loss",
                   help="要比的字段；loss 之外还可以比 grad_norm")
    p.add_argument("--backend", type=str, default="gloo",
                   choices=["auto", "gloo", "nccl"])
    p.add_argument("--force-cpu", action="store_true")
    p.add_argument("--out-dir", type=str, default=None)
    p.add_argument("--run-tag", type=str, default="day2_sft")
    p.add_argument("--dry-run", action="store_true",
                   help="只打印两侧的 (world_size, accum, micro) 组合，不训练")
    return p


def main(argv=None) -> int:
    cli.setup()
    args = build_parser().parse_args(argv)
    if args.compare:
        result = compare_logs(args.compare[0], args.compare[1],
                              key=args.key, tol=args.tol)
        return 0 if result["pass"] else 1

    for label, nproc, accum in (("A", args.a_nproc, args.a_accum),
                                ("B", args.b_nproc, args.b_accum)):
        denom = nproc * accum
        if args.global_batch % denom != 0:
            raise SystemExit(
                "侧 " + label + "：global_batch=" + str(args.global_batch)
                + " 不能被 nproc*accum=" + str(denom) + " 整除")
    micro_a = args.global_batch // (args.a_nproc * args.a_accum)
    micro_b = args.global_batch // (args.b_nproc * args.b_accum)
    print("A: world_size=%d accum=%d micro=%d" % (args.a_nproc, args.a_accum,
                                                  micro_a))
    print("B: world_size=%d accum=%d micro=%d" % (args.b_nproc, args.b_accum,
                                                  micro_b))
    if micro_a != micro_b:
        print("[warn] 两侧 micro_batch 不同（%d vs %d）。batch 维度上的归约顺序"
              "因此不同，loss 会有 1e-6 量级的额外差异；要严格等价请让 micro 相同。"
              % (micro_a, micro_b))
    if args.dry_run:
        print("[dry-run] 未训练")
        return 0

    out_dir = cli.resolve_out_dir(args.out_dir, args.run_tag)
    log_a = run_side("A", args.a_nproc, args.a_accum, args, out_dir)
    log_b = run_side("B", args.b_nproc, args.b_accum, args, out_dir)
    print("")
    result = compare_logs(log_a, log_b, key=args.key, tol=args.tol)
    result.update({"micro_a": micro_a, "micro_b": micro_b,
                   "a_world_size": args.a_nproc, "b_world_size": args.b_nproc})
    path = cli.write_json(result, os.path.join(out_dir, "equiv_table.json"))
    print("[equiv] -> " + path)
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
