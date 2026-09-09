"""独立 eval 的三个判据：eos、角色泄漏、复述。都是定性判据，不产生质量分数。"""

from __future__ import annotations


import pytest

from mm_v100 import data_contract as DC
from mm_v100 import eval_generate as EG
from mm_v100 import model as M

TINY = {"vocab_size": 128, "hidden_size": 32, "num_hidden_layers": 1,
        "num_attention_heads": 4, "num_key_value_heads": 2,
        "intermediate_size": 64, "max_seq_len": 128}


@pytest.fixture
def tok():
    return DC.SimpleTokenizer(vocab_size=TINY["vocab_size"])


def test_prompt_set_is_fixed_and_has_eight_entries():
    """固定 8 条：跨 checkpoint 对照的前提是 prompt 完全不变。"""
    assert len(EG.PROMPTS) == 8
    assert len(set(EG.PROMPTS)) == 8


def test_role_leak_detects_template_markers():
    assert EG.has_role_leak("正常回答") is False
    assert EG.has_role_leak("正常回答" + DC.IM_START + "user") is True
    assert EG.has_role_leak("blah\nuser\nblah") is True


def test_echo_ratio_is_one_for_exact_repeat():
    prompt = "什么是学习率"
    assert EG.echo_ratio(prompt, prompt) == pytest.approx(1.0)
    assert EG.echo_ratio(prompt, "完全无关的内容 xyz") == pytest.approx(0.0)


def test_echo_ratio_handles_short_strings():
    assert EG.echo_ratio("ab", "ab") == 0.0  # 不足一个 3-gram 时返回 0
    assert EG.echo_ratio("", "abc") == 0.0


def test_distinct_ratio():
    assert EG.distinct_ratio([1, 1, 1, 1]) == pytest.approx(0.25)
    assert EG.distinct_ratio([1, 2, 3, 4]) == pytest.approx(1.0)
    assert EG.distinct_ratio([]) == 0.0


def test_run_produces_one_row_per_prompt_and_aggregates(tok):
    model, _ = M.build_model(TINY, device="cpu", seed=4)
    report = EG.run(model, tok, tag="unit", mode="chat", max_new_tokens=4,
                    device="cpu")
    assert len(report["rows"]) == 8
    agg = report["aggregate"]
    assert agg["n_prompts"] == 8
    for key in ("eos_rate", "role_leak_rate", "mean_new_tokens",
                "mean_echo_ratio", "mean_distinct_ratio"):
        assert key in agg
    # 每条都必须带上三个判据，缺一个的话 Day 4 就没有可提交的证据。
    for row in report["rows"]:
        assert set(("ended_with_eos", "role_leak", "echo_ratio")) <= set(row)


def test_raw_mode_does_not_add_chat_template(tok):
    model, _ = M.build_model(TINY, device="cpu", seed=4)
    chat = EG.generate_one(model, tok, "hi", "chat", 2, "cpu")
    raw = EG.generate_one(model, tok, "hi", "raw", 2, "cpu")
    assert raw["rendered_len"] < chat["rendered_len"]


def test_generation_is_deterministic(tok):
    """贪心解码必须可复现，否则跨 checkpoint 的对照没有意义。"""
    model, _ = M.build_model(TINY, device="cpu", seed=4)
    a = EG.generate_one(model, tok, EG.PROMPTS[0], "chat", 6, "cpu")
    b = EG.generate_one(model, tok, EG.PROMPTS[0], "chat", 6, "cpu")
    assert a["output_ids"] == b["output_ids"]


def test_write_and_compare_round_trip(tmp_path, tok):
    model_a, _ = M.build_model(TINY, device="cpu", seed=4)
    model_b, _ = M.build_model(TINY, device="cpu", seed=5)
    ra = EG.run(model_a, tok, tag="a", max_new_tokens=4, device="cpu")
    rb = EG.run(model_b, tok, tag="b", max_new_tokens=4, device="cpu")
    pa = EG.write_jsonl(ra, str(tmp_path / "a.jsonl"))
    pb = EG.write_jsonl(rb, str(tmp_path / "b.jsonl"))
    rows = EG.read_jsonl(pa)
    assert len(rows) == 8
    result = EG.compare(pa, pb)
    assert result["summary"]["n_common"] == 8
    assert 0.0 <= result["summary"]["identical_ratio"] <= 1.0
    assert "identical=" in result["text"]


def test_compare_with_itself_is_fully_identical(tmp_path, tok):
    model, _ = M.build_model(TINY, device="cpu", seed=4)
    report = EG.run(model, tok, tag="self", max_new_tokens=4, device="cpu")
    path = EG.write_jsonl(report, str(tmp_path / "self.jsonl"))
    result = EG.compare(path, path)
    assert result["summary"]["identical_ratio"] == pytest.approx(1.0)


def test_render_text_marks_results_as_qualitative(tok):
    model, _ = M.build_model(TINY, device="cpu", seed=4)
    report = EG.run(model, tok, tag="q", max_new_tokens=2, device="cpu")
    text = EG.render_text(report)
    assert "eos_rate" in text
    assert "不是质量分数" in text


def test_empty_prompt_is_rejected(tok):
    model, _ = M.build_model(TINY, device="cpu", seed=4)
    with pytest.raises(ValueError):
        EG.generate_one(model, tok, "", "raw", 2, "cpu")
