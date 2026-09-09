"""数据契约的两个不变量，以及 label 语义与 MiniMind X/Y 形式之间的等价。"""

from __future__ import annotations

import json

import pytest
import torch

from mm_v100 import data_contract as DC


@pytest.fixture
def tok() -> DC.SimpleTokenizer:
    return DC.SimpleTokenizer(vocab_size=256)


def test_simple_tokenizer_is_deterministic_across_instances(tok):
    """crc32 而不是 hash()：后者带进程级随机盐，换进程结果就变。"""
    other = DC.SimpleTokenizer(vocab_size=256)
    assert tok.encode("梯度裁剪 abc") == other.encode("梯度裁剪 abc")


def test_simple_tokenizer_keeps_special_markers_atomic(tok):
    ids = tok.encode(DC.IM_START + "user\nhi" + DC.IM_END)
    assert ids[0] == tok.im_start_id
    assert ids[-1] == tok.im_end_id


def test_pretrain_labels_cover_all_nonpad(tok):
    enc = DC.encode_pretrain(tok, "梯度累积", max_len=32)
    stats = DC.token_stats(enc["input_ids"], enc["labels"],
                           DC.tokenizer_pad_id(tok))
    assert stats["label_align_violations"] == 0
    assert stats["n_label_tokens"] == stats["n_nonpad"]
    assert stats["n_label_tokens"] > 0


def test_sft_labels_are_a_strict_subset_of_nonpad(tok):
    messages = [{"role": "user", "content": "什么是学习率？"},
                {"role": "assistant", "content": "步长的比例系数。"}]
    enc = DC.encode_sft(tok, messages, max_len=64)
    stats = DC.token_stats(enc["input_ids"], enc["labels"],
                           DC.tokenizer_pad_id(tok))
    assert stats["label_align_violations"] == 0
    assert 0 < stats["n_label_tokens"] < stats["n_nonpad"]


def test_sft_labels_land_exactly_on_assistant_content(tok):
    """逐位检查 mask 边界：user 段一个都不能进 loss，assistant 段一个都不能漏。"""
    messages = [{"role": "user", "content": "AB"},
                {"role": "assistant", "content": "CD"}]
    enc = DC.encode_sft(tok, messages, max_len=64)
    ids = enc["input_ids"].tolist()
    labels = enc["labels"].tolist()
    spans = DC.assistant_spans(messages, tok)
    active = {i for i, v in enumerate(labels) if v != DC.IGNORE_INDEX}
    expected = set()
    for start, end in spans:
        expected.update(range(start, end))
    assert active == expected
    for i in active:
        assert labels[i] == ids[i]


def test_multi_turn_marks_every_assistant_turn(tok):
    messages = [{"role": "user", "content": "q1"},
                {"role": "assistant", "content": "a1"},
                {"role": "user", "content": "q2"},
                {"role": "assistant", "content": "a2"}]
    enc = DC.encode_sft(tok, messages, max_len=96)
    spans = DC.assistant_spans(messages, tok)
    assert len(spans) == 2
    n_active = int((enc["labels"] != DC.IGNORE_INDEX).sum())
    assert n_active == sum(end - start for start, end in spans)


def test_alignment_violation_detects_one_step_shift(tok):
    enc = DC.encode_pretrain(tok, "abcdefg", max_len=16)
    shifted = torch.roll(enc["labels"], shifts=1, dims=-1)
    assert DC.label_alignment_violations(enc["input_ids"], enc["labels"]) == 0
    assert DC.label_alignment_violations(enc["input_ids"], shifted) > 0


def test_shape_mismatch_gives_actionable_error(tok):
    enc = DC.encode_pretrain(tok, "abc", max_len=16)
    with pytest.raises(DC.DataContractError) as exc:
        DC.label_alignment_violations(enc["input_ids"], enc["labels"][:-1])
    assert "to_minimind_pair" in str(exc.value)


def test_to_minimind_pair_preserves_target_count(tok):
    messages = [{"role": "user", "content": "q"},
                {"role": "assistant", "content": "aaaa"}]
    enc = DC.encode_sft(tok, messages, max_len=48)
    pair = DC.to_minimind_pair(enc["input_ids"], enc["labels"])
    assert pair["x"].shape[0] == enc["input_ids"].shape[0] - 1
    assert pair["y"].shape[0] == pair["x"].shape[0]
    # MiniMind 形式下 (y, loss_mask) 里被算进 loss 的 token，
    # 必须与本 lab 形式下 labels != -100 的 token 一一对应（差一格移位）。
    ours = [enc["input_ids"][i].item()
            for i in range(1, enc["labels"].shape[0])
            if enc["labels"][i] != DC.IGNORE_INDEX]
    theirs = [int(pair["y"][i]) for i in range(pair["y"].shape[0])
              if int(pair["loss_mask"][i]) == 1]
    assert ours == theirs


def test_check_invariants_flags_empty_targets():
    verdict = DC.check_invariants(
        {"n_total": 10, "n_nonpad": 8, "n_label_tokens": 0, "n_pad": 2,
         "label_align_violations": 0}, "sft")
    assert verdict["pass"] is False
    assert any("nan" in p for p in verdict["problems"])


def test_check_invariants_flags_prompt_in_loss():
    verdict = DC.check_invariants(
        {"n_total": 10, "n_nonpad": 8, "n_label_tokens": 8, "n_pad": 2,
         "label_align_violations": 0}, "sft")
    assert verdict["pass"] is False
    assert any("user_in_loss" in p for p in verdict["problems"])


def test_toy_fixture_round_trip(tmp_path, tok):
    for stage in ("pretrain", "sft", "dpo"):
        path = DC.write_toy_fixture(str(tmp_path / (stage + ".jsonl")), stage)
        report = DC.inspect_jsonl(path, tok, max_len=64, limit=4)
        assert report["stage"] == stage
        assert report["invariants"]["pass"] is True, report["invariants"]
        assert report["aggregate"]["label_align_violations_total"] == 0


def test_inspect_jsonl_rejects_unknown_schema(tmp_path, tok):
    path = tmp_path / "bad.jsonl"
    path.write_text(json.dumps({"foo": 1}) + "\n", encoding="utf-8")
    with pytest.raises(DC.DataContractError) as exc:
        DC.inspect_jsonl(str(path), tok)
    assert "conversations" in str(exc.value)


def test_inspect_jsonl_rejects_broken_json(tmp_path, tok):
    path = tmp_path / "broken.jsonl"
    path.write_text('{"text": "abc"\n', encoding="utf-8")
    with pytest.raises(DC.DataContractError) as exc:
        DC.inspect_jsonl(str(path), tok)
    assert "二进制" in str(exc.value)


def test_empty_file_gives_actionable_error(tmp_path, tok):
    path = tmp_path / "empty.jsonl"
    path.write_text("", encoding="utf-8")
    with pytest.raises(DC.DataContractError) as exc:
        DC.inspect_jsonl(str(path), tok)
    assert "check_data_layout" in str(exc.value)


def test_dpo_uses_identical_mask_rule_for_both_sides(tok):
    pair = DC.encode_dpo(tok,
                         [{"role": "user", "content": "q"},
                          {"role": "assistant", "content": "same"}],
                         [{"role": "user", "content": "q"},
                          {"role": "assistant", "content": "same"}],
                         max_len=48)
    # 两侧内容相同 -> mask 必须逐位相同，否则序列 logp 不可比。
    assert torch.equal(pair["labels_chosen"], pair["labels_rejected"])
