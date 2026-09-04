"""固定 8 条 prompt 的贪心生成，用于跨阶段（pretrain → sft → dpo → grpo / 正常 vs mask 错位）对照。

- `--checkpoint` 接受 MiniMind `{weight}_{hidden}.pth`（half state_dict）或 bounded_train `.pt`；`none` 用随机初始化。
- `--mode chat`：用 tokenizer.apply_chat_template(add_generation_prompt=True) 渲染（会带上 `<think>\\n\\n</think>\\n\\n` 前缀，
  与 MiniMind chat_template 一致）；`--mode raw`：直接以原文续写（适合 pretrain 权重）。
- 贪心：`do_sample=False`（MiniMind generate 的 argmax 分支），top_k=0，temperature=1.0。
- 输出 JSONL 每行：tag/prompt_id/prompt/rendered/output_text/n_new_tokens/ended_with_eos/output_ids。
- `--compare a.jsonl b.jsonl` 并排打印两次运行的输出（Day 3 正常 vs 错位对照）。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

PROMPTS: List[str] = [
    "你好，请介绍一下你自己。",
    "请用一句话解释什么是学习率。",
    "北京是哪个国家的首都？",
    "写一个 Python 函数，返回两个数的和。",
    "What is gradient clipping?",
    "Explain why padding tokens are masked out of the loss.",
    "1+1 等于几？",
    "The capital of France is",
]


def render(tokenizer, prompt: str, mode: str) -> str:
    if mode == "raw":
        return prompt
    return tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)


@torch.no_grad()
def greedy_generate(model, tokenizer, rendered: str, max_new_tokens: int, device: str) -> Dict[str, Any]:
    ids = tokenizer(rendered, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    out = model.generate(inputs=ids, max_new_tokens=max_new_tokens, temperature=1.0, top_p=1.0, top_k=0,
                         eos_token_id=tokenizer.eos_token_id, do_sample=False, repetition_penalty=1.0, use_cache=True)
    new_ids = out[0, ids.shape[1]:].tolist()
    ended = tokenizer.eos_token_id in new_ids
    if ended:
        new_ids = new_ids[: new_ids.index(tokenizer.eos_token_id) + 1]
    text = tokenizer.decode(new_ids, skip_special_tokens=True)
    return {"output_text": text, "n_new_tokens": len(new_ids), "ended_with_eos": ended, "output_ids": new_ids}


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def compare_outputs(path_a: str, path_b: str, width: int = 60) -> str:
    a = {r["prompt_id"]: r for r in load_jsonl(path_a)}
    b = {r["prompt_id"]: r for r in load_jsonl(path_b)}
    lines = [f"A={path_a}\nB={path_b}"]
    for pid in sorted(set(a) | set(b)):
        ra, rb = a.get(pid), b.get(pid)
        lines.append(f"\n[{pid}] {(ra or rb)['prompt']}")
        lines.append(f"  A({ra['n_new_tokens'] if ra else '-'} tok, eos={ra['ended_with_eos'] if ra else '-'}): {(ra['output_text'][:width] if ra else '-')!r}")
        lines.append(f"  B({rb['n_new_tokens'] if rb else '-'} tok, eos={rb['ended_with_eos'] if rb else '-'}): {(rb['output_text'][:width] if rb else '-')!r}")
        if ra and rb:
            lines.append(f"  identical={ra['output_ids'] == rb['output_ids']}")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="固定 8 条 prompt 贪心生成 → JSONL")
    p.add_argument("--config", default="tiny_cpu")
    p.add_argument("--checkpoint", default="none")
    p.add_argument("--mode", choices=["chat", "raw"], default="chat")
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--out", default=None, help="缺省 lab/runs/eval_<tag>.jsonl")
    p.add_argument("--tag", default="run")
    p.add_argument("--device", default=None)
    p.add_argument("--minimind-root", default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--compare", nargs=2, metavar=("A_JSONL", "B_JSONL"), default=None, help="只做两次输出的并排对照")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    from .minimind_env import (add_minimind_to_path, build_model, find_minimind_root, load_config,
                               load_state_dict_any, load_tokenizer, utf8_stdout)

    utf8_stdout()
    args = build_parser().parse_args(argv)
    if args.compare:
        print(compare_outputs(args.compare[0], args.compare[1]))
        return 0
    root = find_minimind_root(args.minimind_root)
    add_minimind_to_path(root)
    cfg = load_config(args.config)
    device = args.device or cfg.get("train", {}).get("device", "cpu")
    if "cuda" in device and not torch.cuda.is_available():
        device = "cpu"
    torch.manual_seed(args.seed)
    tokenizer = load_tokenizer(root)
    model, _ = build_model(cfg["model"], device)
    if args.checkpoint != "none":
        model.load_state_dict(load_state_dict_any(args.checkpoint), strict=False)
    model.eval()
    out_path = Path(args.out) if args.out else Path(__file__).resolve().parents[2] / "runs" / f"eval_{args.tag}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for i, prompt in enumerate(PROMPTS):
            rendered = render(tokenizer, prompt, args.mode)
            g = greedy_generate(model, tokenizer, rendered, args.max_new_tokens, device)
            row = {"tag": args.tag, "checkpoint": args.checkpoint, "config": cfg["name"], "mode": args.mode,
                   "prompt_id": i, "prompt": prompt, "rendered": rendered, **g}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(f"[{i}] {prompt}\n    -> ({g['n_new_tokens']} tok, eos={g['ended_with_eos']}) {g['output_text'][:120]!r}")
    print(f"written {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
