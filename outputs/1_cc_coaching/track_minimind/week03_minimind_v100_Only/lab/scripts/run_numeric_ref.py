"""Day 1 第二条命令：fp32 参考曲线 vs fp16 AMP 曲线，产出 numeric_ref.json。

用法::

    # 本机（CPU 上 fp16 autocast 也能跑，用来验方法本身）
    python lab/scripts/run_numeric_ref.py --config tiny_cpu --device cpu --steps 6

    # V100
    python lab/scripts/run_numeric_ref.py --config v100_768 --device cuda:0 \\
        --steps 20 --tol 1e-3

退出码：0 = 判定通过；1 = 有问题（问题与首个检查点打在最后）。

一条注意
--------
CPU 上的 fp16 是**软件模拟**的，数值行为与 V100 的 Tensor Core 不一样。
本机跑通只证明这套对照方法本身没写错；max_dloss 的具体数值必须在 V100 上重取。
"""

from __future__ import annotations

import argparse
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(os.path.dirname(_HERE), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from mm_v100 import cli  # noqa: E402
from mm_v100 import common as C  # noqa: E402
from mm_v100 import numeric_ref as NR  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="fp32 vs fp16 AMP 逐步误差账（max_dloss / first_diverge_step / scale）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--config", type=str, default="tiny_cpu",
                   help="configs/ 下的名字或完整路径")
    p.add_argument("--device", type=str, default="cpu",
                   help="cpu 或 cuda:0")
    p.add_argument("--steps", type=int, default=6, help="对照步数（硬上限）")
    p.add_argument("--tol", type=float, default=1e-3,
                   help="first_diverge_step 的判定阈值")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--low-dtype", type=str, default="float16",
                   choices=["float16", "bfloat16"],
                   help="低精度那一路；bfloat16 会被 dtype 守卫在 V100 上拒绝")
    p.add_argument("--sdpa-backend", type=str, default="auto",
                   choices=["auto", "math", "mem_efficient", "flash"])
    p.add_argument("--pure-python-scaler", action="store_true",
                   help="强制用纯 Python scaler（CPU 上验跳步逻辑时需要）")
    p.add_argument("--out-dir", type=str, default=None)
    p.add_argument("--run-tag", type=str, default="day1_pretrain")
    p.add_argument("--dry-run", action="store_true",
                   help="只跑 1 步对照，确认两条路都能前反向")
    return p


def main(argv=None) -> int:
    cli.setup()
    args = build_parser().parse_args(argv)
    device = args.device
    if device.startswith("cuda") and not __import__("torch").cuda.is_available():
        print("[warn] 请求 " + device + " 但本机没有 CUDA，改用 cpu", flush=True)
        device = "cpu"
    steps = 1 if args.dry_run else int(args.steps)
    cfg = C.load_config(args.config)
    report = NR.run(cfg, device=device, steps=steps, tol=args.tol,
                    seed=args.seed, sdpa_backend=args.sdpa_backend,
                    low_dtype=args.low_dtype,
                    force_pure_python_scaler=args.pure_python_scaler)
    print(NR.render_text(report))
    if args.dry_run:
        print("[dry-run] 两条路径各跑 1 步成功，未写文件")
        return 0
    out_dir = cli.resolve_out_dir(args.out_dir, args.run_tag)
    path = cli.write_json(report, os.path.join(out_dir, "numeric_ref.json"))
    print("[numeric_ref] -> " + path)
    return 0 if report["verdict"]["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
