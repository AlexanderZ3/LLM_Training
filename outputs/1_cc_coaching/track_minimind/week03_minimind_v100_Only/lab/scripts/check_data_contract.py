"""Day 1 第一条命令：把一条 JSONL 走到 (input_ids, labels)，并检查两个不变量。

用法::

    # 用真数据（先跑过 check_data_layout.py 确认字节数）
    python lab/scripts/check_data_contract.py --file sft_t2t_mini.jsonl --stage sft

    # 没有数据/没有网络时用内置 fixture，链路照样验完
    python lab/scripts/check_data_contract.py --toy sft

退出码：0 = 两个不变量都通过；1 = 有违例（违例明细打在最后）。

不打印任何样本内容
------------------
输出只有计数、比例和字段名。所以这条命令的输出可以直接贴给 cc，
不构成数据外带。
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
from mm_v100 import data_contract as DC  # noqa: E402
from mm_v100 import paths as P  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="JSONL 数据契约检查（字段 / 编码 / label 对齐 / token 计数）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--file", type=str, default=None,
                   help="$MM_DATA_ROOT/minimind_dataset 下的文件名，或完整路径")
    p.add_argument("--group", type=str, default="minimind_dataset",
                   help="数据组目录名")
    p.add_argument("--toy", type=str, default=None,
                   choices=sorted(DC.TOY_SETS),
                   help="不用真数据，用内置 fixture（离线可用）")
    p.add_argument("--stage", type=str, default=None,
                   choices=sorted(DC.STAGE_FIELDS),
                   help="显式指定阶段；不给就按第一行的字段自动判断")
    p.add_argument("--tokenizer-dir", type=str, default=None,
                   help="真 tokenizer 目录；不给就用内置 SimpleTokenizer")
    p.add_argument("--max-len", type=int, default=128, help="截断/填充长度")
    p.add_argument("--limit", type=int, default=8, help="只看前 N 条")
    p.add_argument("--out-dir", type=str, default=None,
                   help="产物目录；默认 $MM_RUNS_ROOT/day1_pretrain")
    p.add_argument("--run-tag", type=str, default="day1_pretrain")
    p.add_argument("--dry-run", action="store_true",
                   help="只解析路径与 tokenizer 来源，不读数据")
    return p


def main(argv=None) -> int:
    cli.setup()
    args = build_parser().parse_args(argv)
    out_dir = cli.resolve_out_dir(args.out_dir, args.run_tag)

    tok_dir = args.tokenizer_dir
    if tok_dir is None:
        try:
            candidate = P.weights_dir("minimind_tokenizer", must_exist=False)
            tok_dir = str(candidate) if candidate.is_dir() else None
        except P.PathContractError:
            tok_dir = None
    tokenizer, tok_source = DC.load_tokenizer(tok_dir)

    if args.toy:
        data_path = DC.write_toy_fixture(
            os.path.join(out_dir, "toy_" + args.toy + ".jsonl"), args.toy,
            repeat=2)
    elif args.file:
        if os.path.isfile(args.file):
            data_path = os.path.abspath(args.file)
        else:
            data_path = str(P.dataset_file(args.file, args.group))
    else:
        print("必须给 --file <名字> 或 --toy <阶段>。\n"
              "下一步：先跑 python lab/scripts/check_data_layout.py 看有哪些文件；"
              "没有数据就用 --toy sft。", file=sys.stderr)
        return 2

    if args.dry_run:
        print("[dry-run] tokenizer 来源：" + tok_source)
        print("[dry-run] 数据文件：" + data_path)
        print("[dry-run] 产物目录：" + out_dir)
        return 0

    report = DC.inspect_jsonl(data_path, tokenizer, max_len=args.max_len,
                              limit=args.limit, stage=args.stage)
    report["tokenizer_source"] = tok_source
    report["data_file"] = os.path.basename(data_path)

    agg = report["aggregate"]
    print("== Day 1 数据契约 ==")
    print("file=%s  stage=%s  tokenizer=%s  max_len=%d"
          % (report["data_file"], report["stage"], tok_source, args.max_len))
    print("first_line_keys=" + ", ".join(report["first_line_keys"]))
    print("")
    print("%6s %10s %10s %14s %12s" % ("idx", "n_nonpad", "n_label",
                                       "align_viol", "label_ratio"))
    for i, s in enumerate(report["per_sample"]):
        ratio = s["n_label_tokens"] / max(s["n_nonpad"], 1)
        print("%6d %10d %10d %14d %12.3f"
              % (i, s["n_nonpad"], s["n_label_tokens"],
                 s["label_align_violations"], ratio))
    print("")
    print("I1 label_align_violations_total = %d（正常值 0）"
          % agg["label_align_violations_total"])
    print("I2 n_label_tokens_mean = %.1f  n_nonpad_mean = %.1f  ratio = %.3f"
          % (agg["n_label_tokens_mean"], agg["n_nonpad_mean"],
             agg["label_ratio_mean"]))
    verdict = report["invariants"]
    print("verdict = " + ("PASS" if verdict["pass"] else "FAIL"))
    for problem in verdict["problems"]:
        print("  - " + problem)
    path = cli.write_json(report, os.path.join(out_dir, "data_contract.json"))
    print("[data_contract] -> " + path)
    return 0 if verdict["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
