"""Day 0 第二条命令：静态扫描 MiniMind，输出四个兼容性计数。

只读 $MINIMIND_ROOT 下的 trainer/ model/ dataset/ 里的 .py，不 import、不执行、
不写回。用法::

    export MINIMIND_ROOT=/path/to/minimind
    python lab/scripts/audit_compat.py

退出码：0 = 没有 torch 2.1 缺失 API 的命中；1 = 有命中（那意味着直接跑原脚本会
AttributeError，必须先改或先绕开）。默认 bfloat16 与缺 GradScaler 不影响退出码——
它们是「必须加参数」而不是「跑不起来」。
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
from mm_v100 import compat_audit as CA  # noqa: E402
from mm_v100 import paths as P  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="MiniMind x torch 2.1 / V100 静态兼容审计（只读）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--minimind-root", type=str, default=None,
                   help="MiniMind 仓库根目录；默认读环境变量 MINIMIND_ROOT")
    p.add_argument("--subdirs", type=str, default="trainer,model,dataset",
                   help="要扫的子目录，逗号分隔")
    p.add_argument("--out-dir", type=str, default=None,
                   help="产物目录；默认 $MM_RUNS_ROOT/day0_probe")
    p.add_argument("--run-tag", type=str, default="day0_probe")
    p.add_argument("--quiet", action="store_true", help="不打 Markdown 正文")
    p.add_argument("--dry-run", action="store_true",
                   help="只解析路径与要扫的文件列表，不读文件内容")
    return p


def main(argv=None) -> int:
    cli.setup()
    args = build_parser().parse_args(argv)
    root = args.minimind_root or os.environ.get("MINIMIND_ROOT", "").strip()
    if not root:
        print("环境变量 MINIMIND_ROOT 没设，也没给 --minimind-root。\n"
              "下一步：在 V100 上 git clone MiniMind 后\n"
              "  export MINIMIND_ROOT=/你克隆到的路径\n"
              "本周锁定的 commit 见 lab/README.md 第 1 节。", file=sys.stderr)
        return 2
    root = os.path.abspath(os.path.expanduser(root))
    subdirs = tuple(s.strip() for s in args.subdirs.split(",") if s.strip())
    if args.dry_run:
        files = CA.iter_target_files(root, subdirs)
        print("[dry-run] 根目录：" + root)
        print("[dry-run] 将扫描 " + str(len(files)) + " 个文件：")
        for f in files:
            print("  " + os.path.relpath(f, root))
        return 0
    report = CA.audit(root, subdirs)
    summary = report["summary"]
    markdown = CA.render_markdown(report, os.path.basename(root))
    if not args.quiet:
        print(markdown)
    out_dir = cli.resolve_out_dir(args.out_dir, args.run_tag)
    md_path = cli.write_text(markdown, os.path.join(out_dir, "compat_audit.md"))
    js_path = cli.write_json(report, os.path.join(out_dir, "compat_audit.json"))
    print("")
    print("[audit] n_default_bfloat16=%d  n_autocast_no_scaler=%d  "
          "n_torch_compile=%d  n_newer_api_hits=%d"
          % (summary["n_default_bfloat16"], summary["n_autocast_no_scaler"],
             summary["n_torch_compile"], summary["n_newer_api_hits"]))
    print("[audit] -> " + md_path)
    print("[audit] -> " + js_path)
    print("[audit] 数据根目录（供核对）：" + str(P.data_root()))
    return 0 if summary["n_newer_api_hits"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
