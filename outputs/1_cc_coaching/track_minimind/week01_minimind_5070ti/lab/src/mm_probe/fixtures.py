"""内置 toy fixture：无网络、无 GB 级数据时用于单测与 CPU dry run。

字段格式与 MiniMind 官方数据一致（核验于 commit 7a6fddd）：
- pretrain: {"text": str}
- sft:      {"conversations": [{"role": "user"|"assistant"|"system", "content": str}, ...]}
- dpo:      {"chosen": [msgs], "rejected": [msgs]}
- rlaif:    {"conversations": [msgs]}（最后一条 assistant 会被 RLAIFDataset 丢弃，只保留 prompt）
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

PRETRAIN_SAMPLES: List[Dict] = [
    {"text": "今天天气不错，适合出门散步。"},
    {"text": "MiniMind is a tiny decoder-only language model trained from scratch."},
    {"text": "梯度裁剪把梯度范数限制在阈值以内，避免单步更新过大。"},
    {"text": "The quick brown fox jumps over the lazy dog. 0123456789"},
]

SFT_SAMPLES: List[Dict] = [
    {"conversations": [
        {"role": "user", "content": "你好"},
        {"role": "assistant", "content": "你好！有什么可以帮你？"},
    ]},
    {"conversations": [
        {"role": "user", "content": "1+1=?"},
        {"role": "assistant", "content": "2"},
        {"role": "user", "content": "那 2+2 呢？"},
        {"role": "assistant", "content": "4"},
    ]},
    {"conversations": [
        {"role": "system", "content": "You are minimind, a small but useful language model."},
        {"role": "user", "content": "Explain gradient clipping in one sentence."},
        {"role": "assistant", "content": "Gradient clipping rescales the gradient so its norm never exceeds a threshold."},
    ]},
]

DPO_SAMPLES: List[Dict] = [
    {"chosen": [
        {"role": "user", "content": "请用一句话介绍北京。"},
        {"role": "assistant", "content": "北京是中国的首都，也是政治和文化中心。"},
    ], "rejected": [
        {"role": "user", "content": "请用一句话介绍北京。"},
        {"role": "assistant", "content": "不知道。"},
    ]},
    {"chosen": [
        {"role": "user", "content": "What is 3*4?"},
        {"role": "assistant", "content": "3*4 = 12."},
    ], "rejected": [
        {"role": "user", "content": "What is 3*4?"},
        {"role": "assistant", "content": "3*4 = 7."},
    ]},
    {"chosen": [
        {"role": "user", "content": "写一个 Python 打印 hello 的语句。"},
        {"role": "assistant", "content": "print(\"hello\")"},
    ], "rejected": [
        {"role": "user", "content": "写一个 Python 打印 hello 的语句。"},
        {"role": "assistant", "content": "echo hello"},
    ]},
]

RLAIF_SAMPLES: List[Dict] = [
    {"conversations": [
        {"role": "user", "content": "用两句话解释什么是学习率。"},
        {"role": "assistant", "content": ""},
    ]},
    {"conversations": [
        {"role": "user", "content": "Why do we mask padding tokens in the loss?"},
        {"role": "assistant", "content": ""},
    ]},
]

FIXTURES: Dict[str, List[Dict]] = {
    "pretrain": PRETRAIN_SAMPLES,
    "sft": SFT_SAMPLES,
    "dpo": DPO_SAMPLES,
    "rlaif": RLAIF_SAMPLES,
}


def write_fixture_jsonl(stage: str, out_path: str | Path) -> Path:
    """把某阶段的 fixture 写成 JSONL（UTF-8，不转义中文）。已存在且内容相同则不重写。"""
    if stage not in FIXTURES:
        raise ValueError(f"未知 stage: {stage}，可选 {sorted(FIXTURES)}")
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    content = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in FIXTURES[stage])
    if out.exists() and out.read_text(encoding="utf-8") == content:
        return out
    out.write_text(content, encoding="utf-8")
    return out
