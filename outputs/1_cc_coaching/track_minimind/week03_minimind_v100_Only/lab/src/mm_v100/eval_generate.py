"""Day 4 独立 eval：8 条固定 prompt 并排生成。**不看 loss。**

来源
----
从 week01 lab/src/mm_probe/eval_generate.py 复制并裁剪：保留固定 prompt 集、
贪心解码、JSONL 输出、并排对照；去掉了 MiniMind 模型/权重加载（改用本包的
build_model + 可选 checkpoint），新增角色泄漏检测与聚合指标。

为什么这一天不许看 loss
-----------------------
loss 下降只证明优化器在拟合某个目标。它不能回答：模型学会在该停的地方停了吗？
它会不会把 user 的话当成自己要生成的内容？这两个问题只能靠看生成结果。
Day 2 的故障 A（label 错位）和 user_in_loss 形态，在 loss 曲线上都很正常，
但在这里会立刻暴露成「复述提问」和「不产生结束符」。

三个可判定的观测
----------------
ended_with_eos  生成里出现了结束符。SFT 之后这个比例应该明显高于 pretrain 之后。
                恒为 False 说明 assistant 段的结束符没进 loss。
role_leak       生成文本里出现了 <|im_start|> 或角色名开头。出现即说明模型在
                续写模板而不是在回答——典型的 mask 边界算错。
echo_ratio      生成内容与 prompt 的字符 3-gram 重合比例。接近 1 说明在复述提问。

这三个都是**定性判据**，不是分数。小模型 + 有界训练本来就不该有可比的质量分，
把它们写成分数是把 smoke 提升成业务结论。
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Iterable, List, Optional, Sequence

import torch

from .data_contract import (IM_START, SimpleTokenizer, render_chat,
                            tokenizer_encode)

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

ROLE_MARKERS = (IM_START, "<|im_start|>user", "<|im_start|>system",
                "\nuser\n", "\nsystem\n")


def char_ngrams(text: str, n: int = 3) -> List[str]:
    stripped = "".join(text.split())
    return [stripped[i:i + n] for i in range(max(len(stripped) - n + 1, 0))]


def echo_ratio(prompt: str, output: str, n: int = 3) -> float:
    """生成内容里有多少比例的 3-gram 直接来自 prompt。"""
    p = set(char_ngrams(prompt, n))
    o = char_ngrams(output, n)
    if not o or not p:
        return 0.0
    return sum(1 for g in o if g in p) / len(o)


def has_role_leak(text: str) -> bool:
    return any(marker in text for marker in ROLE_MARKERS)


def distinct_ratio(ids: Sequence[int]) -> float:
    if not ids:
        return 0.0
    return len(set(int(i) for i in ids)) / len(ids)


def decode(tok, ids: Iterable[int]) -> str:
    if isinstance(tok, SimpleTokenizer):
        return tok.decode(ids, skip_special=False)
    return tok.decode(list(ids), skip_special_tokens=False)


@torch.no_grad()
def generate_one(model, tok, prompt: str, mode: str, max_new_tokens: int,
                 device: str) -> Dict[str, Any]:
    """一条 prompt 的贪心生成。mode=chat 走 ChatML 模板，raw 直接续写。"""
    rendered = (prompt if mode == "raw"
                else render_chat([{"role": "user", "content": prompt}],
                                 add_generation_prompt=True))
    ids = tokenizer_encode(tok, rendered)
    if not ids:
        raise ValueError("prompt 编码成了空序列：" + repr(prompt))
    max_len = int(getattr(model.config, "max_seq_len", 512))
    ids = ids[-(max_len - 1):]
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)
    eos_id = getattr(tok, "eos_token_id", None)
    out = model.greedy_generate(input_ids, max_new_tokens=max_new_tokens,
                                eos_token_id=eos_id)
    new_ids = out[0, input_ids.shape[1]:].tolist()
    ended = bool(eos_id is not None and eos_id in new_ids)
    if ended:
        new_ids = new_ids[: new_ids.index(eos_id) + 1]
    text = decode(tok, new_ids)
    return {
        "prompt": prompt,
        "mode": mode,
        "rendered_len": len(ids),
        "output_text": text,
        "n_new_tokens": len(new_ids),
        "ended_with_eos": ended,
        "role_leak": has_role_leak(text),
        "echo_ratio": echo_ratio(prompt, text),
        "distinct_ratio": distinct_ratio(new_ids),
        "output_ids": new_ids,
    }


def run(model, tok, tag: str, mode: str = "chat", max_new_tokens: int = 32,
        device: str = "cpu",
        prompts: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    """跑完 8 条 prompt，返回逐条结果 + 聚合指标。"""
    model.eval()
    rows: List[Dict[str, Any]] = []
    for i, prompt in enumerate(prompts or PROMPTS):
        row = generate_one(model, tok, prompt, mode, max_new_tokens, device)
        row["prompt_id"] = i
        row["tag"] = tag
        rows.append(row)
    n = len(rows)
    return {
        "schema": "mm_v100.eval_generate/1",
        "tag": tag,
        "mode": mode,
        "max_new_tokens": int(max_new_tokens),
        "rows": rows,
        "aggregate": {
            "n_prompts": n,
            "eos_rate": sum(1 for r in rows if r["ended_with_eos"]) / n,
            "role_leak_rate": sum(1 for r in rows if r["role_leak"]) / n,
            "mean_new_tokens": sum(r["n_new_tokens"] for r in rows) / n,
            "mean_echo_ratio": sum(r["echo_ratio"] for r in rows) / n,
            "mean_distinct_ratio": sum(r["distinct_ratio"] for r in rows) / n,
        },
    }


def write_jsonl(report: Dict[str, Any], path: str) -> str:
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for row in report["rows"]:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    return os.path.abspath(path)


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def compare(path_a: str, path_b: str, width: int = 60) -> Dict[str, Any]:
    """两次运行的并排对照（例如 pretrain vs SFT，或正常 vs 故障）。"""
    a = {r["prompt_id"]: r for r in read_jsonl(path_a)}
    b = {r["prompt_id"]: r for r in read_jsonl(path_b)}
    common = sorted(set(a) & set(b))
    lines: List[str] = ["A=" + path_a, "B=" + path_b, ""]
    identical = 0
    for pid in common:
        ra, rb = a[pid], b[pid]
        same = ra["output_ids"] == rb["output_ids"]
        identical += int(same)
        lines.append("[" + str(pid) + "] " + ra["prompt"])
        lines.append("  A(%d tok, eos=%s, leak=%s): %r"
                     % (ra["n_new_tokens"], ra["ended_with_eos"],
                        ra["role_leak"], ra["output_text"][:width]))
        lines.append("  B(%d tok, eos=%s, leak=%s): %r"
                     % (rb["n_new_tokens"], rb["ended_with_eos"],
                        rb["role_leak"], rb["output_text"][:width]))
        lines.append("  identical=" + str(same))
    summary = {
        "n_common": len(common),
        "n_identical": identical,
        "identical_ratio": (identical / len(common)) if common else 0.0,
        "a_eos_rate": (sum(1 for r in a.values() if r["ended_with_eos"])
                       / max(len(a), 1)),
        "b_eos_rate": (sum(1 for r in b.values() if r["ended_with_eos"])
                       / max(len(b), 1)),
        "a_role_leak_rate": (sum(1 for r in a.values() if r["role_leak"])
                             / max(len(a), 1)),
        "b_role_leak_rate": (sum(1 for r in b.values() if r["role_leak"])
                             / max(len(b), 1)),
    }
    lines.append("")
    lines.append(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return {"summary": summary, "text": "\n".join(lines)}


def render_text(report: Dict[str, Any], width: int = 70) -> str:
    agg = report["aggregate"]
    lines = ["== Day 4 独立 eval  tag=" + str(report["tag"]) + " =="]
    for row in report["rows"]:
        lines.append("[%d] %s" % (row["prompt_id"], row["prompt"]))
        lines.append("    -> (%d tok, eos=%s, leak=%s, echo=%.2f) %r"
                     % (row["n_new_tokens"], row["ended_with_eos"],
                        row["role_leak"], row["echo_ratio"],
                        row["output_text"][:width]))
    lines.append("")
    lines.append("eos_rate=%.2f role_leak_rate=%.2f mean_new_tokens=%.1f "
                 "mean_echo=%.2f mean_distinct=%.2f"
                 % (agg["eos_rate"], agg["role_leak_rate"],
                    agg["mean_new_tokens"], agg["mean_echo_ratio"],
                    agg["mean_distinct_ratio"]))
    lines.append("注意：这些是定性判据，不是质量分数。")
    return "\n".join(lines)
