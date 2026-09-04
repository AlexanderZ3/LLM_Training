"""SFT label-mask 故障注入（Day 3 故障练习）。

两种错位：
- `assistant_all_ignored`：assistant 段也全部 -100 → 没有任何 target；MiniMind 的
  `F.cross_entropy(..., ignore_index=-100)` 对 0 个有效元素取均值 → loss = nan，梯度 nan。
- `user_in_loss`：所有非 pad token（含 user/system/模板）都进 loss → loss 数值与正常曲线不可比，
  模型学会复述 user 段与模板。

`apply_fault(labels, mode, input_ids=None, pad_token_id=0)` 是纯函数，接受 list 或 torch.Tensor，
返回同类型。`none` 原样返回副本。
"""
from __future__ import annotations

import argparse
import sys
from typing import List, Optional, Sequence, Union

IGNORE_INDEX = -100
FAULT_MODES = ("none", "assistant_all_ignored", "user_in_loss")

FAULT_DESCRIPTIONS = {
    "none": "正常：只有 assistant 内容 + <|im_end|>\\n 进 loss。",
    "assistant_all_ignored": "labels 全 -100：无有效 target；期望症状 loss=nan（或某些实现下 0），grad_norm=nan/0，权重不变。",
    "user_in_loss": "所有非 pad token 进 loss：期望症状 loss 起点更低且下降更快（模板 token 极易预测），固定 prompt 生成会复述 user/模板文本。",
}


def _is_tensor(x) -> bool:
    return hasattr(x, "clone") and hasattr(x, "dtype")


def apply_fault(labels: Union[Sequence[int], "torch.Tensor"], mode: str,
                input_ids: Optional[Union[Sequence[int], "torch.Tensor"]] = None,
                pad_token_id: int = 0):
    """返回注入故障后的 labels（新对象，不改原值）。"""
    if mode not in FAULT_MODES:
        raise ValueError(f"未知 mode={mode!r}，可选 {FAULT_MODES}")
    if _is_tensor(labels):
        import torch

        out = labels.clone()
        if mode == "none":
            return out
        if mode == "assistant_all_ignored":
            out.fill_(IGNORE_INDEX)
            return out
        if input_ids is None:
            raise ValueError("user_in_loss 需要 input_ids")
        ids = input_ids if _is_tensor(input_ids) else torch.as_tensor(list(input_ids), dtype=out.dtype)
        return torch.where(ids != pad_token_id, ids, torch.full_like(ids, IGNORE_INDEX))
    labels_l: List[int] = list(labels)
    if mode == "none":
        return labels_l
    if mode == "assistant_all_ignored":
        return [IGNORE_INDEX] * len(labels_l)
    if input_ids is None:
        raise ValueError("user_in_loss 需要 input_ids")
    ids_l = [int(x) for x in (input_ids.tolist() if _is_tensor(input_ids) else input_ids)]
    return [x if x != pad_token_id else IGNORE_INDEX for x in ids_l]


def count_in_loss(labels: Union[Sequence[int], "torch.Tensor"]) -> int:
    if _is_tensor(labels):
        return int((labels != IGNORE_INDEX).sum().item())
    return sum(1 for x in labels if x != IGNORE_INDEX)


def main(argv: Optional[List[str]] = None) -> int:
    from . import fixtures
    from .inspect_dataset import print_table, sft_encode, token_rows
    from .minimind_env import find_minimind_root, load_tokenizer, utf8_stdout

    utf8_stdout()
    p = argparse.ArgumentParser(description="在 fixture SFT 样本上演示 label-mask 故障注入")
    p.add_argument("--mode", choices=FAULT_MODES, default="user_in_loss")
    p.add_argument("--index", type=int, default=1)
    p.add_argument("--max-seq-len", type=int, default=64)
    p.add_argument("--minimind-root", default=None)
    args = p.parse_args(argv)
    root = find_minimind_root(args.minimind_root)
    tok = load_tokenizer(root)
    enc = sft_encode(tok, fixtures.SFT_SAMPLES[args.index]["conversations"], args.max_seq_len)
    faulty = apply_fault(enc["labels"], args.mode, input_ids=enc["input_ids"], pad_token_id=tok.pad_token_id)
    print(f"mode={args.mode}: {FAULT_DESCRIPTIONS[args.mode]}")
    print(f"in-loss tokens: normal={count_in_loss(enc['labels'])}  faulty={count_in_loss(faulty)}")
    rows = token_rows(tok, enc)
    for r, f in zip(rows, faulty):
        r["label"] = f"{r['label']}->{f}" if r["label"] != f else str(r["label"])
    print_table([r for r in rows if r["segment"] != "pad"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
