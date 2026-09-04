"""Day 1：SFT label-mask 规则。

第一组：不依赖 tokenizer 的纯规则单测（合成 id）。
第二组：依赖 MINIMIND_ROOT 的 tokenizer 单测（未设置则 skip）——fixture 样本的 assistant 段进 loss、user/template 段为 -100、
`<|im_end|>\\n` 处理与 MiniMind SFTDataset.generate_labels 逐位一致。
"""
from __future__ import annotations

import pytest

from mm_probe import fixtures
from mm_probe.inspect_dataset import (IGNORE_INDEX, aggregate, char_segments, generate_labels, pretrain_encode,
                                      sample_stats, sft_encode, special_sequences)

BOS = [1, 9, 9, 5]  # 合成的 "<|im_start|>assistant\n"
EOS = [2, 5]        # 合成的 "<|im_end|>\n"


def test_rule_single_turn_synthetic():
    #        user block           assistant block            pad
    ids = [1, 7, 5, 30, 31, 2, 5] + BOS + [40, 41, 42] + EOS + [0, 0]
    labels = generate_labels(ids, BOS, EOS, max_length=len(ids))
    start = 7 + len(BOS)
    expect = [IGNORE_INDEX] * len(ids)
    for j in range(start, start + 3 + len(EOS)):
        expect[j] = ids[j]
    assert labels == expect
    assert labels[start + 3] == 2 and labels[start + 4] == 5, "<|im_end|> 与其后的 \\n 都进 loss"
    assert all(x == IGNORE_INDEX for x in labels[:start]), "assistant 头部模板与 user 段为 -100"
    assert labels[-2:] == [IGNORE_INDEX, IGNORE_INDEX]


def test_rule_multi_turn_and_truncation_synthetic():
    ids = BOS + [40] + EOS + [1, 7, 5, 8, 2, 5] + BOS + [50, 51]  # 第二段没有 eos（被截断）
    labels = generate_labels(ids, BOS, EOS, max_length=len(ids))
    assert labels[4:7] == [40, 2, 5]
    assert labels[7:13] == [IGNORE_INDEX] * 6
    assert labels[-2:] == [50, 51], "截断后没有 eos 的 assistant 段一直标到末尾"
    # max_length 小于序列时只标到 max_length
    labels2 = generate_labels(ids, BOS, EOS, max_length=5)
    assert labels2[4] == 40 and all(x == IGNORE_INDEX for x in labels2[5:])


def test_rule_no_assistant_gives_all_ignore():
    ids = [1, 7, 5, 30, 2, 5, 0, 0]
    assert generate_labels(ids, BOS, EOS, 8) == [IGNORE_INDEX] * 8


def test_char_segments_regex():
    prompt = "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\nyo<|im_end|>\n"
    seg = char_segments(prompt)
    assert seg[prompt.index("hi")] == "user"
    assert seg[prompt.index("yo")] == "assistant"
    assert seg[0] == "template" and seg[-1] == "template"


# ---------------------------------------------------------------------------
# 依赖 tokenizer
# ---------------------------------------------------------------------------
def test_special_sequences_match_minimind(tokenizer):
    bos_seq, eos_seq = special_sequences(tokenizer)
    assert bos_seq[0] == tokenizer.bos_token_id == 1
    assert eos_seq == [tokenizer.eos_token_id, tokenizer("\n", add_special_tokens=False).input_ids[0]]
    assert tokenizer.decode(bos_seq) == "<|im_start|>assistant\n"


@pytest.mark.parametrize("idx", [0, 1, 2])
def test_fixture_sft_mask_semantics(tokenizer, idx):
    enc = sft_encode(tokenizer, fixtures.SFT_SAMPLES[idx]["conversations"], max_length=128)
    ids, labels, segs = enc["input_ids"], enc["labels"], enc["segments"]
    assert not enc["truncated"]
    for t, lab, seg in zip(ids, labels, segs):
        if seg in ("user", "system", "pad"):
            assert lab == IGNORE_INDEX, f"{seg} 段 token {t} 不应进 loss"
        if seg == "assistant":
            assert lab == t, "assistant 内容必须进 loss（label = input_id）"
    # `<|im_end|>\n` 紧跟 assistant 内容 → 进 loss；紧跟 user 内容 → -100
    for k in range(1, len(ids)):
        if ids[k] == tokenizer.eos_token_id:
            prev_seg = segs[k - 1]
            if prev_seg == "assistant":
                assert labels[k] == tokenizer.eos_token_id and labels[k + 1] == ids[k + 1]
            elif prev_seg in ("user", "system"):
                assert labels[k] == IGNORE_INDEX and labels[k + 1] == IGNORE_INDEX
    st = sample_stats(enc, tokenizer.eos_token_id)
    assert st["eos_in_loss"] is True
    n_turns = sum(1 for m in fixtures.SFT_SAMPLES[idx]["conversations"] if m["role"] == "assistant")
    assert sum(1 for x in labels if x == tokenizer.eos_token_id) == n_turns
    # 第一个进 loss 的位置正好在 "<|im_start|>assistant\n"（5 个 token）之后
    bos_seq, _ = special_sequences(tokenizer)
    first = st["first_loss_pos"]
    assert ids[first - len(bos_seq):first] == bos_seq


def test_labels_identical_to_minimind_sftdataset(tokenizer, minimind_root, tmp_path):
    """与 MiniMind 原 SFTDataset.generate_labels 逐位一致（需要 datasets 包）。"""
    pytest.importorskip("datasets")
    import random
    import sys

    if str(minimind_root) not in sys.path:
        sys.path.insert(0, str(minimind_root))
    from dataset.lm_dataset import SFTDataset  # type: ignore

    path = fixtures.write_fixture_jsonl("sft", tmp_path / "sft.jsonl")
    ds = SFTDataset(str(path), tokenizer, max_length=96)
    for i, s in enumerate(fixtures.SFT_SAMPLES):
        enc = sft_encode(tokenizer, s["conversations"], 96)
        assert ds.generate_labels(enc["input_ids"]) == enc["labels"]
        random.seed(1234)  # 让 MiniMind 的随机分支走"不加 system、删空 think"（概率 0.8×0.8）
        x, y = ds[i]
        if x.tolist() == enc["input_ids"]:
            assert y.tolist() == enc["labels"]


def test_pretrain_encode_matches_minimind(tokenizer, minimind_root, tmp_path):
    pytest.importorskip("datasets")
    import sys

    if str(minimind_root) not in sys.path:
        sys.path.insert(0, str(minimind_root))
    from dataset.lm_dataset import PretrainDataset  # type: ignore

    path = fixtures.write_fixture_jsonl("pretrain", tmp_path / "pre.jsonl")
    ds = PretrainDataset(str(path), tokenizer, max_length=32)
    for i, s in enumerate(fixtures.PRETRAIN_SAMPLES):
        enc = pretrain_encode(tokenizer, s["text"], 32)
        x, y = ds[i]
        assert x.tolist() == enc["input_ids"]
        assert y.tolist() == enc["labels"]
        assert enc["input_ids"][0] == tokenizer.bos_token_id
        assert enc["labels"][enc["raw_len"] - 1] == tokenizer.eos_token_id, "pretrain 的 eos 进 loss"


def test_aggregate_truncation_ratio(tokenizer):
    encs = [sft_encode(tokenizer, s["conversations"], 16) for s in fixtures.SFT_SAMPLES]
    agg = aggregate([sample_stats(e, tokenizer.eos_token_id) for e in encs])
    assert agg["n_samples"] == 3 and agg["truncation_ratio"] == 1.0
