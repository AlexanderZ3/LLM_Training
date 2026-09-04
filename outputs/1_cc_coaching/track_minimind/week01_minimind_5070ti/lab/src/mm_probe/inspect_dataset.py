"""数据 → tokenizer → token/label 因果链检查（复现 MiniMind `dataset/lm_dataset.py` 的规则）。

复现的规则（MiniMind commit 7a6fddd，已逐行对照）：
- PretrainDataset: `[bos] + tokenizer(text)[:max_len-2] + [eos]`，右侧 pad；labels = input_ids，pad 位置 -100。
- SFTDataset: `apply_chat_template(conversations)` → 文本；`tokenizer(text).input_ids[:max_len]` 右侧 pad；
  labels 默认全 -100；只在每个 `<|im_start|>assistant\\n` 之后、到 `<|im_end|>\\n`（含）为止的位置
  复制 input_ids（`generate_labels`）。模型内部再做 shift（logits[:, :-1] vs labels[:, 1:]）。
- MiniMind 在 __getitem__ 里有两个随机分支：20% 概率加 system prompt、80% 概率删掉空 `<think>\\n\\n</think>\\n\\n`。
  本模块把它们改成确定性开关（--add-system / --keep-empty-think），便于对照；默认与"最常见分支"一致
  （不加 system、删空 think）。

纯函数 `generate_labels` 不依赖 tokenizer，可用合成 id 做单测。
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

IGNORE_INDEX = -100
EMPTY_THINK = "<think>\n\n</think>\n\n"
SYSTEM_PROMPT_FIXED = "You are minimind, a small but useful language model."  # MiniMind SYSTEM_PROMPTS 列表中的一条
_BLOCK_RE = re.compile(r"<\|im_start\|>(system|user|assistant)\n(.*?)<\|im_end\|>\n?", re.DOTALL)


# ---------------------------------------------------------------------------
# 纯规则（不需要 tokenizer）
# ---------------------------------------------------------------------------
def generate_labels(input_ids: Sequence[int], bos_seq: Sequence[int], eos_seq: Sequence[int], max_length: int) -> List[int]:
    """逐字复现 MiniMind `SFTDataset.generate_labels`。

    bos_seq = tokenizer('<|im_start|>assistant\\n') 的 id 序列（5 个 token: 1,1388,570,811,234）
    eos_seq = tokenizer('<|im_end|>\\n') 的 id 序列（2 个 token: 2,234）
    规则：匹配到 bos_seq 后，从其后第一个 token 开始，直到匹配到 eos_seq 为止；
    label 覆盖 [start, end + len(eos_seq)) ∩ [0, max_length) —— 即 eos_seq 本身也进 loss；
    若到序列末尾都没遇到 eos_seq（被截断），则一直标到末尾。
    """
    input_ids = list(input_ids)
    bos_seq = list(bos_seq)
    eos_seq = list(eos_seq)
    labels = [IGNORE_INDEX] * len(input_ids)
    i = 0
    while i < len(input_ids):
        if input_ids[i:i + len(bos_seq)] == bos_seq:
            start = i + len(bos_seq)
            end = start
            while end < len(input_ids):
                if input_ids[end:end + len(eos_seq)] == eos_seq:
                    break
                end += 1
            for j in range(start, min(end + len(eos_seq), max_length)):
                labels[j] = input_ids[j]
            i = end + len(eos_seq) if end < len(input_ids) else len(input_ids)
        else:
            i += 1
    return labels


def loss_mask_from_labels(labels: Sequence[int]) -> List[int]:
    """DPODataset.generate_loss_mask 与 generate_labels 规则相同，只是输出 0/1。"""
    return [0 if x == IGNORE_INDEX else 1 for x in labels]


def char_segments(prompt: str) -> List[str]:
    """把渲染后的 chat 文本按字符标注归属：system/user/assistant（内容）或 template（标记）。"""
    seg = ["template"] * len(prompt)
    for m in _BLOCK_RE.finditer(prompt):
        role = m.group(1)
        for c in range(m.start(2), m.end(2)):
            seg[c] = role
    return seg


# ---------------------------------------------------------------------------
# 需要 tokenizer 的编码函数
# ---------------------------------------------------------------------------
def special_sequences(tokenizer) -> Tuple[List[int], List[int]]:
    """返回 (bos_seq, eos_seq)，与 SFTDataset.__init__ 完全一致。"""
    bos_seq = tokenizer(f"{tokenizer.bos_token}assistant\n", add_special_tokens=False).input_ids
    eos_seq = tokenizer(f"{tokenizer.eos_token}\n", add_special_tokens=False).input_ids
    return bos_seq, eos_seq


def pretrain_encode(tokenizer, text: str, max_length: int) -> Dict[str, Any]:
    """复现 PretrainDataset.__getitem__。"""
    raw = tokenizer(str(text), add_special_tokens=False).input_ids
    truncated = len(raw) > max_length - 2
    tokens = [tokenizer.bos_token_id] + raw[: max_length - 2] + [tokenizer.eos_token_id]
    pad = tokenizer.pad_token_id
    input_ids = tokens + [pad] * (max_length - len(tokens))
    labels = [x if x != pad else IGNORE_INDEX for x in input_ids]
    segments = ["template"] + ["text"] * (len(tokens) - 2) + ["template"] + ["pad"] * (max_length - len(tokens))
    return {"input_ids": input_ids, "labels": labels, "segments": segments, "truncated": truncated,
            "raw_len": len(raw) + 2, "prompt": str(text)}


def sft_render(tokenizer, conversations: List[Dict[str, str]], add_system: bool = False, keep_empty_think: bool = False) -> str:
    """复现 create_chat_prompt + pre/post_processing_chat，但把随机分支改成确定开关。"""
    msgs = [dict(m) for m in conversations]
    if add_system and msgs and msgs[0].get("role") != "system":
        msgs = [{"role": "system", "content": SYSTEM_PROMPT_FIXED}] + msgs
    prompt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
    if not keep_empty_think and EMPTY_THINK in prompt:
        prompt = prompt.replace(EMPTY_THINK, "")
    return prompt


def sft_encode(tokenizer, conversations: List[Dict[str, str]], max_length: int,
               add_system: bool = False, keep_empty_think: bool = False) -> Dict[str, Any]:
    """复现 SFTDataset.__getitem__，并附带每个 token 的段落归属。"""
    prompt = sft_render(tokenizer, conversations, add_system=add_system, keep_empty_think=keep_empty_think)
    enc = tokenizer(prompt, return_offsets_mapping=True)
    full_ids = list(enc.input_ids)
    offsets = list(enc.offset_mapping)
    truncated = len(full_ids) > max_length
    input_ids = full_ids[:max_length]
    pad = tokenizer.pad_token_id
    n_real = len(input_ids)
    input_ids = input_ids + [pad] * (max_length - n_real)
    bos_seq, eos_seq = special_sequences(tokenizer)
    labels = generate_labels(input_ids, bos_seq, eos_seq, max_length)
    cseg = char_segments(prompt)
    segments = []
    for k in range(max_length):
        if k >= n_real:
            segments.append("pad")
            continue
        s, e = offsets[k]
        segments.append(cseg[s] if s < len(cseg) and e > s else "template")
    return {"input_ids": input_ids, "labels": labels, "segments": segments, "truncated": truncated,
            "raw_len": len(full_ids), "prompt": prompt}


def token_rows(tokenizer, encoded: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows = []
    for idx, (tid, lab, seg) in enumerate(zip(encoded["input_ids"], encoded["labels"], encoded["segments"])):
        rows.append({"idx": idx, "token": tokenizer.decode([tid]), "id": tid, "label": lab, "segment": seg})
    return rows


def sample_stats(encoded: Dict[str, Any], eos_token_id: int) -> Dict[str, Any]:
    labels = encoded["labels"]
    segs = encoded["segments"]
    n_nonpad = sum(1 for s in segs if s != "pad")
    n_label = sum(1 for x in labels if x != IGNORE_INDEX)
    first = next((i for i, x in enumerate(labels) if x != IGNORE_INDEX), None)
    eos_in_loss = any(x == eos_token_id for x in labels)
    per_seg = {}
    for lab, seg in zip(labels, segs):
        d = per_seg.setdefault(seg, {"tokens": 0, "in_loss": 0})
        d["tokens"] += 1
        d["in_loss"] += int(lab != IGNORE_INDEX)
    return {
        "n_nonpad": n_nonpad,
        "n_in_loss": n_label,
        "loss_token_ratio": (n_label / n_nonpad) if n_nonpad else 0.0,
        "first_loss_pos": first,
        "eos_in_loss": eos_in_loss,
        "truncated": bool(encoded["truncated"]),
        "raw_len": encoded["raw_len"],
        "per_segment": per_seg,
    }


def aggregate(stats_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(stats_list)
    if n == 0:
        return {"n_samples": 0}
    return {
        "n_samples": n,
        "mean_loss_token_ratio": sum(s["loss_token_ratio"] for s in stats_list) / n,
        "truncation_ratio": sum(1 for s in stats_list if s["truncated"]) / n,
        "eos_in_loss_ratio": sum(1 for s in stats_list if s["eos_in_loss"]) / n,
        "first_loss_pos": [s["first_loss_pos"] for s in stats_list],
        "mean_raw_len": sum(s["raw_len"] for s in stats_list) / n,
    }


def print_table(rows: List[Dict[str, Any]], stream=None) -> None:
    stream = stream or sys.stdout
    stream.write(f"{'idx':>4} | {'token':<18} | {'id':>5} | {'label':>6} | segment\n")
    stream.write("-" * 56 + "\n")
    for r in rows:
        tok = r["token"].replace("\n", "\\n")
        stream.write(f"{r['idx']:>4} | {tok:<18} | {r['id']:>5} | {r['label']:>6} | {r['segment']}\n")


def read_jsonl(path: str, limit: int) -> List[Dict[str, Any]]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
            if len(out) >= limit:
                break
    return out


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="复现 MiniMind tokenization 与 loss-mask 规则，逐 token 打印并统计")
    p.add_argument("--stage", choices=["pretrain", "sft"], default="sft")
    p.add_argument("--data-path", default=None, help="JSONL 路径；缺省且未给 --fixture 时用 $MINIMIND_ROOT/dataset 的 mini 文件")
    p.add_argument("--fixture", action="store_true", help="用内置 3 条样本（无网络、无数据文件时）")
    p.add_argument("--n", type=int, default=3, help="检查前 N 条样本")
    p.add_argument("--max-seq-len", type=int, default=None, help="缺省：pretrain 340 / sft 768（MiniMind 默认）")
    p.add_argument("--minimind-root", default=None)
    p.add_argument("--add-system", action="store_true", help="对应 MiniMind 20%% 概率分支：加一条 system")
    p.add_argument("--keep-empty-think", action="store_true", help="对应 MiniMind 20%% 概率分支：保留空 think 标签")
    p.add_argument("--no-table", action="store_true", help="只打印统计不打印逐 token 表")
    p.add_argument("--out", default=None, help="把统计写成 JSON")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    from . import fixtures
    from .minimind_env import find_minimind_root, load_tokenizer, utf8_stdout

    utf8_stdout()
    args = build_parser().parse_args(argv)
    root = find_minimind_root(args.minimind_root)
    tokenizer = load_tokenizer(root)
    max_len = args.max_seq_len or (340 if args.stage == "pretrain" else 768)

    if args.fixture:
        samples = fixtures.FIXTURES[args.stage][: args.n]
        source = "fixture"
    else:
        path = args.data_path or str(root / "dataset" / ("pretrain_t2t_mini.jsonl" if args.stage == "pretrain" else "sft_t2t_mini.jsonl"))
        samples = read_jsonl(path, args.n)
        source = path

    stats_list = []
    for i, s in enumerate(samples):
        if args.stage == "pretrain":
            enc = pretrain_encode(tokenizer, s["text"], max_len)
        else:
            enc = sft_encode(tokenizer, s["conversations"], max_len, add_system=args.add_system, keep_empty_think=args.keep_empty_think)
        st = sample_stats(enc, tokenizer.eos_token_id)
        stats_list.append(st)
        print(f"\n=== sample {i} ({source}) stage={args.stage} max_seq_len={max_len} ===")
        print("rendered prompt:", repr(enc["prompt"][:300]) + (" ..." if len(enc["prompt"]) > 300 else ""))
        if not args.no_table:
            shown = [r for r in token_rows(tokenizer, enc) if r["segment"] != "pad"]
            print_table(shown)
            n_pad = max_len - len(shown)
            if n_pad:
                print(f"(+{n_pad} pad tokens, label=-100, segment=pad)")
        print("stats:", json.dumps(st, ensure_ascii=False))

    agg = aggregate(stats_list)
    print("\n=== aggregate ===")
    print(json.dumps(agg, ensure_ascii=False, indent=2))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump({"stage": args.stage, "max_seq_len": max_len, "source": source, "samples": stats_list, "aggregate": agg}, f, ensure_ascii=False, indent=2)
        print(f"written {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
