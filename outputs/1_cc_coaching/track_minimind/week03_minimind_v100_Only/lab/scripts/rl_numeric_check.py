"""Day 5 RL 数值链：DPO 的 ln2 与 ref 梯度断言 + GRPO 的两个统计量。

用法::

    # 本机（fixture 数据，秒级）
    python lab/scripts/rl_numeric_check.py --config tiny_cpu

    # V100（真 dpo.jsonl）
    python lab/scripts/rl_numeric_check.py --config v100_768 --device cuda:0 \\
        --dpo-file dpo.jsonl --n-pairs 4

退出码：0 = D1/D2/G2 三条断言都成立（G1 高时状态是 INCONCLUSIVE，仍返回 0）；
1 = 有断言不成立。

为什么 G1 高不算失败
--------------------
zero_std_group_ratio 高说明规则 reward 分不开这一组生成，那是**实验设计**的
局限，不是管线错误。Day 5 预注册了这个出口：>0.8 判 INCONCLUSIVE，
写进证据的 failure_class 是 insufficient，不是 model。
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

import torch  # noqa: E402

from mm_v100 import cli  # noqa: E402
from mm_v100 import common as C  # noqa: E402
from mm_v100 import data_contract as DC  # noqa: E402
from mm_v100 import model as M  # noqa: E402
from mm_v100 import paths as P  # noqa: E402
from mm_v100 import rl_numeric as RL  # noqa: E402


def load_dpo_pairs(path: str, n: int) -> List[Dict[str, Any]]:
    rows = DC.read_first_records(path, limit=n)
    for i, row in enumerate(rows):
        missing = [f for f in ("chosen", "rejected") if f not in row]
        if missing:
            raise SystemExit(
                "第 " + str(i + 1) + " 行缺字段 " + ", ".join(missing)
                + "。DPO 数据必须同时有 chosen 与 rejected。")
    return rows


def build_batch(tokenizer, pairs: List[Dict[str, Any]],
                max_len: int) -> Dict[str, torch.Tensor]:
    encs = [DC.encode_dpo(tokenizer, p["chosen"], p["rejected"], max_len)
            for p in pairs]
    keys = ("input_ids_chosen", "labels_chosen",
            "input_ids_rejected", "labels_rejected")
    return {k: torch.stack([e[k] for e in encs]) for k in keys}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="DPO(ln2 / ref 梯度) 与 GRPO(zero_std / ratio@step0) 的解析断言",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--config", type=str, default="tiny_cpu")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--dpo-file", type=str, default=None,
                   help="$MM_DATA_ROOT 下的 dpo jsonl 文件名或完整路径；"
                        "不给就用内置 fixture")
    p.add_argument("--n-pairs", type=int, default=2, help="取几对 chosen/rejected")
    p.add_argument("--max-len", type=int, default=64)
    p.add_argument("--beta", type=float, default=0.15)
    p.add_argument("--tol", type=float, default=1e-5,
                   help="init_loss 与 ln2 的容差")
    p.add_argument("--group-size", type=int, default=4,
                   help="GRPO 每个 prompt 的生成数 G")
    p.add_argument("--n-groups", type=int, default=4, help="GRPO 组数")
    p.add_argument("--max-new-tokens", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--tokenizer-dir", type=str, default=None)
    p.add_argument("--out-dir", type=str, default=None)
    p.add_argument("--run-tag", type=str, default="day5_rl")
    p.add_argument("--dry-run", action="store_true",
                   help="只跑 DPO 的两条断言，跳过 GRPO 采样")
    return p


def main(argv=None) -> int:
    cli.setup()
    args = build_parser().parse_args(argv)
    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("[warn] 请求 " + device + " 但本机没有 CUDA，改用 cpu")
        device = "cpu"
    C.set_seed(args.seed, per_rank=False)
    cfg = C.load_config(args.config)
    out_dir = cli.resolve_out_dir(args.out_dir, args.run_tag)

    tok_dir = args.tokenizer_dir
    if tok_dir is None:
        try:
            candidate = P.weights_dir("minimind_tokenizer", must_exist=False)
            tok_dir = str(candidate) if candidate.is_dir() else None
        except P.PathContractError:
            tok_dir = None
    tokenizer, tok_source = DC.load_tokenizer(
        tok_dir, vocab_size=int(cfg["model"]["vocab_size"]))

    if args.dpo_file:
        path = (args.dpo_file if os.path.isfile(args.dpo_file)
                else str(P.dataset_file(args.dpo_file)))
        pairs = load_dpo_pairs(path, args.n_pairs)
        data_label = os.path.basename(path)
    else:
        path = DC.write_toy_fixture(os.path.join(out_dir, "toy_dpo.jsonl"),
                                    "dpo", repeat=4)
        pairs = load_dpo_pairs(path, args.n_pairs)
        data_label = "fixture"

    batch = build_batch(tokenizer, pairs, args.max_len)
    model, _cfg = M.build_model(cfg["model"], device=device, seed=args.seed)
    dpo = RL.dpo_init_check(model, batch, beta=args.beta, device=device,
                            tol=args.tol)

    ratio = None
    grpo = None
    if not args.dry_run:
        x = batch["input_ids_chosen"].to(device)
        y = batch["labels_chosen"].to(device)
        ratio = RL.ratio_at_step0(model(x).logits, x, y)

        reward = RL.RuleReward()
        groups: List[List[float]] = []
        model.eval()
        for gi in range(args.n_groups):
            question = "GRPO probe prompt " + str(gi)
            prompt_ids = DC.tokenizer_encode(
                tokenizer, DC.render_chat([{"role": "user",
                                            "content": question}],
                                          add_generation_prompt=True))
            base = torch.tensor([prompt_ids], dtype=torch.long, device=device)
            texts: List[str] = []
            for k in range(args.group_size):
                # 真实 GRPO 靠随机采样得到一组不同的生成。本 lab 不引入采样，
                # 改用「同一贪心轨迹的不同截断长度」来构造组内差异：它是确定性的
                # （证据可复现），而且恰好命中规则 reward 的长度项——
                # 这也顺带暴露了规则 reward 的分辨率有多差，正是 G1 要量化的事。
                out = model.greedy_generate(
                    base, max_new_tokens=max(args.max_new_tokens - k * 3, 2),
                    eos_token_id=getattr(tokenizer, "eos_token_id", None))
                new_ids = out[0, base.shape[1]:].tolist()
                texts.append(DC.SimpleTokenizer().decode(new_ids)
                             if isinstance(tokenizer, DC.SimpleTokenizer)
                             else tokenizer.decode(new_ids))
            groups.append([reward.score(question, t) for t in texts])
        grpo = RL.grpo_group_stats(groups)

    report = RL.summarize(dpo, grpo, ratio)
    report["data"] = data_label
    report["tokenizer_source"] = tok_source
    report["config"] = cfg["name"]
    report["device"] = device
    print(RL.render_text(report))
    if args.dry_run:
        print("[dry-run] 跳过 GRPO 采样，未写文件")
        return 0 if all(c["ok"] for c in report["checks"]) else 1
    path_out = cli.write_json(report, os.path.join(out_dir, "rl_numeric.json"))
    print("[rl_numeric] -> " + path_out)
    return 0 if report["status"] != "FAIL-SYSTEM" else 1


if __name__ == "__main__":
    raise SystemExit(main())
