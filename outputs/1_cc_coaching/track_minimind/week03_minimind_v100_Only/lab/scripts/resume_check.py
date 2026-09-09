"""Day 3/4 恢复等价：连续训 N 步 vs 训 K 步存盘再恢复训到 N 步，逐步 loss 必须相等。

用法::

    python lab/scripts/resume_check.py --config tiny_cpu --force-cpu \\
        --total-steps 6 --break-at 3

退出码：0 = 恢复后的每一步 loss 与连续训练相等（容差 --tol）；1 = 不等。

这条检查为什么必须做
--------------------
checkpoint 里少存任何一样东西，恢复后的曲线都会「看起来还行」但不相等：
少存 optimizer      -> Adam 的一阶二阶矩清零，恢复后头几步 loss 明显抬高
少存 scaler         -> fp16 的 scale 回到 65536，可能连跳几步
少存 RNG            -> dropout / 数据采样不同（本 lab dropout=0，所以这一项
                       在这里抓不到，但在真实训练里会）
少存 data cursor    -> 从头吃数据，loss 曲线出现「重新变简单」的台阶
所以判据是**逐步相等**，不是「趋势接近」。
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


def run(argv: List[str]) -> Dict[str, Any]:
    parsed = BT.build_parser().parse_args(argv)
    return BT.train(parsed)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="checkpoint / resume 等价：连续训练 vs 断点恢复的逐步 loss",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--config", type=str, default="tiny_cpu")
    p.add_argument("--stage", type=str, default="pretrain",
                   choices=["pretrain", "sft"])
    p.add_argument("--dtype", type=str, default="float32",
                   choices=["float32", "float16", "bfloat16"])
    p.add_argument("--total-steps", type=int, default=6)
    p.add_argument("--break-at", type=int, default=3,
                   help="在第几步之后存盘并重启")
    p.add_argument("--global-batch", type=int, default=8)
    p.add_argument("--accum", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--tol", type=float, default=1e-6,
                   help="逐步 loss 的容差；单进程 fp32 下应当是 0")
    p.add_argument("--force-cpu", action="store_true")
    p.add_argument("--out-dir", type=str, default=None)
    p.add_argument("--run-tag", type=str, default="day3_dist")
    p.add_argument("--dry-run", action="store_true",
                   help="只打印两条路线的步数安排，不训练")
    return p


def main(argv=None) -> int:
    cli.setup()
    args = build_parser().parse_args(argv)
    if not 0 < args.break_at < args.total_steps:
        raise SystemExit("--break-at 必须在 (0, --total-steps) 之间")
    out_dir = cli.resolve_out_dir(args.out_dir, args.run_tag)

    base = ["--config", args.config, "--out-dir", out_dir,
            "--placement", "single", "--stage", args.stage,
            "--dtype", args.dtype, "--global-batch", str(args.global_batch),
            "--accum", str(args.accum), "--seed", str(args.seed),
            "--seed-per-rank", "0", "--log-interval", str(args.total_steps)]
    if args.force_cpu:
        base.append("--force-cpu")

    if args.dry_run:
        print("[dry-run] 连续路线：1..%d" % args.total_steps)
        print("[dry-run] 断点路线：1..%d 存盘，恢复后 %d..%d"
              % (args.break_at, args.break_at + 1, args.total_steps))
        print("[dry-run] out_dir=" + out_dir)
        return 0

    cont = run(base + ["--max-steps", str(args.total_steps),
                       "--log-name", "resume_continuous"])
    part1 = run(base + ["--max-steps", str(args.total_steps),
                        "--log-name", "resume_part1",
                        "--save-every", str(args.break_at)])
    # 注意：part1 用的也是 --max-steps total，只是我们只关心它在 break_at 存的盘。
    # lr 调度依赖 total_steps，两条路线必须用同一个 total，否则 lr 不同、loss 必不同。
    ckpt = os.path.join(out_dir, "ckpt", "step" + str(args.break_at) + ".pt")
    if not os.path.isfile(ckpt):
        raise SystemExit("没有生成 checkpoint：" + ckpt)
    resumed = run(base + ["--max-steps", str(args.total_steps),
                          "--log-name", "resume_part2", "--resume", ckpt])

    rows_c = {int(r["step"]): r for r in C.read_jsonl(
        os.path.join(out_dir, "resume_continuous_rank0.jsonl"))
        if r.get("event") != "header"}
    rows_r = {int(r["step"]): r for r in C.read_jsonl(
        os.path.join(out_dir, "resume_part2_rank0.jsonl"))
        if r.get("event") != "header"}
    steps = sorted(set(rows_c) & set(rows_r))
    deltas = [abs(float(rows_c[s]["loss"]) - float(rows_r[s]["loss"]))
              for s in steps]
    print("")
    print("== Day 3 resume 等价 ==")
    print("step | 连续训练  | 恢复之后  |   |delta|")
    print("-----+-----------+-----------+----------")
    for s, d in zip(steps, deltas):
        print("%4d | %9.6f | %9.6f | %.3e"
              % (s, rows_c[s]["loss"], rows_r[s]["loss"], d))
    max_delta = max(deltas) if deltas else 0.0
    ok = bool(steps) and max_delta <= args.tol
    print("")
    print("compared_steps=%d  max|delta|=%.3e  tol=%.1e -> %s"
          % (len(steps), max_delta, args.tol, "PASS" if ok else "FAIL"))
    payload = {
        "schema": "mm_v100.resume_check/1",
        "config": args.config,
        "total_steps": args.total_steps,
        "break_at": args.break_at,
        "checkpoint": ckpt,
        "compared_steps": steps,
        "max_abs_delta": max_delta,
        "tol": args.tol,
        "pass": ok,
        "continuous_losses": cont["losses"],
        "part1_losses": part1["losses"],
        "resumed_losses": resumed["losses"],
    }
    path = cli.write_json(payload, os.path.join(out_dir, "resume_check.json"))
    print("[resume] -> " + path)
    if not ok:
        print("[resume] 先查 checkpoint 里 optimizer / scaler / rng / "
              "data_cursor 四样是不是都存了，再查两条路线的 --max-steps 是否相同"
              "（lr 调度依赖它）。")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
