"""Day 0 第一条命令：把这台机器的真实能力写成 probe.json。

用法::

    python lab/scripts/probe_env.py                       # 写到 $MM_RUNS_ROOT/day0_probe
    python lab/scripts/probe_env.py --out-dir /tmp/probe  # 指定目录
    python lab/scripts/probe_env.py --json-only           # 只打 JSON，不写文件

退出码：0 = 所有检查通过；1 = 有检查未通过（未通过的项名会打在最后一行）。
未通过不一定是故障：本机没有 GPU 时 eight_gpus / bf16 两项必然 FAIL，那是「未知」，
不是「不支持」。所以这条命令的结论只在目标机上有意义。
"""

from __future__ import annotations

import argparse
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(os.path.dirname(_HERE), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from mm_v100 import cli  # noqa: E402
from mm_v100 import paths as P  # noqa: E402
from mm_v100 import probe  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="V100 能力探针：卡数/capability/bf16/SDPA/int8/NCCL/DCP/磁盘",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--out-dir", type=str, default=None,
                   help="产物目录；默认 $MM_RUNS_ROOT/day0_probe")
    p.add_argument("--run-tag", type=str, default="day0_probe",
                   help="run 目录名")
    p.add_argument("--json-only", action="store_true",
                   help="只把 JSON 打到 stdout，不写任何文件")
    p.add_argument("--quiet", action="store_true", help="不打人读摘要")
    p.add_argument("--dry-run", action="store_true",
                   help="只打印将要写入的路径与要采集的项，不真的采集")
    return p


def main(argv=None) -> int:
    cli.setup()
    args = build_parser().parse_args(argv)
    out_dir = None if args.json_only else cli.resolve_out_dir(args.out_dir,
                                                              args.run_tag)
    if args.dry_run:
        print("[dry-run] 将写入目录：" + str(out_dir))
        print("[dry-run] 采集项：versions / devices / gpu_exclusive / sdpa / "
              "int8 / grad_scaler / dcp_api / fsdp_api / disk")
        print(P.describe())
        return 0
    minimind_root = os.environ.get("MINIMIND_ROOT", "").strip() or None
    report = probe.collect(runs_dir=str(P.runs_root()),
                           minimind_root=minimind_root)
    if not args.quiet:
        print(probe.render_text(report))
        print("")
    if out_dir is not None:
        path = cli.write_json(report, os.path.join(out_dir, "probe.json"))
        print("[probe] -> " + path)
    else:
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    if not report["all_checks_pass"]:
        print("[probe] 未通过：" + ", ".join(report["blocking_failures"]))
    return 0 if report["all_checks_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
