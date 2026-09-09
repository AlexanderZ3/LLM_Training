"""Day 4 独立 eval：8 条固定 prompt 生成，跨 checkpoint 并排对照。不看 loss。

用法::

    # 单个 checkpoint 的生成结果
    python lab/scripts/eval_compare.py --config tiny_cpu --tag base

    # 训练前 vs 训练后
    python lab/scripts/eval_compare.py --config tiny_cpu --tag before
    python lab/scripts/eval_compare.py --config tiny_cpu --tag after \\
        --checkpoint $MM_RUNS_ROOT/day4_profile/ckpt/final.pt
    python lab/scripts/eval_compare.py --compare \\
        $MM_RUNS_ROOT/day4_profile/eval_before.jsonl \\
        $MM_RUNS_ROOT/day4_profile/eval_after.jsonl

退出码：恒为 0（除非文件读不到）。这一天不产生 PASS/FAIL——
生成结果的好坏在 64M + 有界训练下没有可比的判据，它提供的是**定性证据**。
"""

from __future__ import annotations

import argparse
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(os.path.dirname(_HERE), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import torch  # noqa: E402

from mm_v100 import cli  # noqa: E402
from mm_v100 import common as C  # noqa: E402
from mm_v100 import data_contract as DC  # noqa: E402
from mm_v100 import eval_generate as EG  # noqa: E402
from mm_v100 import model as M  # noqa: E402
from mm_v100 import paths as P  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="固定 8 条 prompt 的贪心生成与并排对照（不看 loss）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--compare", nargs=2, metavar=("A_JSONL", "B_JSONL"),
                   default=None, help="只做两份结果的并排对照")
    p.add_argument("--config", type=str, default="tiny_cpu")
    p.add_argument("--checkpoint", type=str, default=None,
                   help="bounded_train 存的 .pt；不给就用随机初始化")
    p.add_argument("--tag", type=str, default="base",
                   help="输出文件名 eval_<tag>.jsonl")
    p.add_argument("--mode", type=str, default="chat", choices=["chat", "raw"],
                   help="chat=ChatML 模板；raw=直接续写（适合 pretrain 权重）")
    p.add_argument("--max-new-tokens", type=int, default=32)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--tokenizer-dir", type=str, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", type=str, default=None)
    p.add_argument("--run-tag", type=str, default="day4_profile")
    p.add_argument("--dry-run", action="store_true",
                   help="只跑第 1 条 prompt、只生成 2 个 token")
    return p


def main(argv=None) -> int:
    cli.setup()
    args = build_parser().parse_args(argv)
    if args.compare:
        result = EG.compare(args.compare[0], args.compare[1])
        print(result["text"])
        return 0

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("[warn] 请求 " + device + " 但本机没有 CUDA，改用 cpu")
        device = "cpu"
    C.set_seed(args.seed, per_rank=False)
    cfg = C.load_config(args.config)
    model, _model_cfg = M.build_model(cfg["model"], device=device,
                                      seed=args.seed)
    if args.checkpoint:
        blob = torch.load(args.checkpoint, map_location="cpu",
                          weights_only=False)
        state = blob["model"] if isinstance(blob, dict) and "model" in blob else blob
        model.load_state_dict(state)
        print("[eval] 载入 " + args.checkpoint)

    tok_dir = args.tokenizer_dir
    if tok_dir is None:
        try:
            candidate = P.weights_dir("minimind_tokenizer", must_exist=False)
            tok_dir = str(candidate) if candidate.is_dir() else None
        except P.PathContractError:
            tok_dir = None
    tokenizer, tok_source = DC.load_tokenizer(
        tok_dir, vocab_size=int(cfg["model"]["vocab_size"]))
    print("[eval] tokenizer=" + tok_source)

    prompts = EG.PROMPTS[:1] if args.dry_run else None
    report = EG.run(model, tokenizer, tag=args.tag, mode=args.mode,
                    max_new_tokens=2 if args.dry_run else args.max_new_tokens,
                    device=device, prompts=prompts)
    report["checkpoint"] = args.checkpoint or "random_init"
    report["tokenizer_source"] = tok_source
    print(EG.render_text(report))
    if args.dry_run:
        print("[dry-run] 未写文件")
        return 0
    out_dir = cli.resolve_out_dir(args.out_dir, args.run_tag)
    jsonl = EG.write_jsonl(report, os.path.join(out_dir,
                                                "eval_" + args.tag + ".jsonl"))
    summary = cli.write_json(report["aggregate"],
                             os.path.join(out_dir,
                                          "eval_" + args.tag + "_summary.json"))
    print("[eval] -> " + jsonl)
    print("[eval] -> " + summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
