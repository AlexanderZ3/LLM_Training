"""RL 数值链的四条解析断言。这些等式与模型大小、训练步数无关，必须精确成立。"""

from __future__ import annotations

import math

import pytest
import torch

from mm_v100 import data_contract as DC
from mm_v100 import model as M
from mm_v100 import rl_numeric as RL

TINY = {"vocab_size": 128, "hidden_size": 32, "num_hidden_layers": 2,
        "num_attention_heads": 4, "num_key_value_heads": 2,
        "intermediate_size": 64, "max_seq_len": 64}


@pytest.fixture
def batch():
    tok = DC.SimpleTokenizer(vocab_size=TINY["vocab_size"])
    pairs = DC.TOY_DPO
    encs = [DC.encode_dpo(tok, p["chosen"], p["rejected"], 48) for p in pairs]
    keys = ("input_ids_chosen", "labels_chosen",
            "input_ids_rejected", "labels_rejected")
    return {k: torch.stack([e[k] for e in encs]) for k in keys}


def test_d1_initial_dpo_loss_is_exactly_ln2(batch):
    """policy 与 ref 权重相同 -> logits_dpo 恒为 0 -> loss = -logsigmoid(0) = ln2。"""
    model, _ = M.build_model(TINY, device="cpu", seed=13)
    out = RL.dpo_init_check(model, batch, beta=0.15, device="cpu", tol=1e-5)
    assert out["D1_init_loss_is_ln2"] is True
    assert out["init_loss"] == pytest.approx(math.log(2.0), abs=1e-5)
    assert out["max_abs_logits_dpo"] == pytest.approx(0.0, abs=1e-4)


def test_d2_ref_model_gets_no_gradients(batch):
    model, _ = M.build_model(TINY, device="cpu", seed=13)
    out = RL.dpo_init_check(model, batch, beta=0.15, device="cpu")
    # None，不是 0：0 说明 ref 参与了 autograd 图，只是这一步梯度恰好为零。
    assert out["D2_ref_grads_all_none"] is True
    assert out["policy_has_grad"] is True


def test_dpo_loss_sign_and_margin():
    """chosen 的 logp 更高 -> logits_dpo > 0 -> loss < ln2，margin 为正。"""
    pol_c = torch.tensor([1.0, 2.0])
    pol_r = torch.tensor([0.0, 0.0])
    ref_c = torch.zeros(2)
    ref_r = torch.zeros(2)
    out = RL.dpo_loss(pol_c, pol_r, ref_c, ref_r, beta=0.15)
    assert float(out["loss"]) < math.log(2.0)
    assert torch.all(out["reward_margin"] > 0)
    assert float(out["accuracy"]) == 1.0


def test_dpo_loss_beta_scales_the_margin():
    args = (torch.tensor([1.0]), torch.tensor([0.0]),
            torch.zeros(1), torch.zeros(1))
    a = RL.dpo_loss(*args, beta=0.1)
    b = RL.dpo_loss(*args, beta=0.2)
    assert float(b["reward_margin"]) == pytest.approx(
        2 * float(a["reward_margin"]))


def test_sequence_logp_only_counts_unmasked_positions():
    logits = torch.zeros(1, 5, 4)
    ids = torch.tensor([[0, 1, 2, 3, 0]])
    labels_all = ids.clone()
    labels_half = ids.clone()
    labels_half[:, :3] = DC.IGNORE_INDEX
    full = float(RL.sequence_logp(logits, ids, labels_all))
    half = float(RL.sequence_logp(logits, ids, labels_half))
    # 均匀 logits 下每个位置贡献 log(1/4)；4 个 vs 2 个有效位置。
    assert full == pytest.approx(4 * math.log(0.25))
    assert half == pytest.approx(2 * math.log(0.25))


def test_g2_ratio_at_step0_is_exactly_one(batch):
    model, _ = M.build_model(TINY, device="cpu", seed=13)
    x = batch["input_ids_chosen"]
    y = batch["labels_chosen"]
    out = RL.ratio_at_step0(model(x).logits, x, y)
    assert out["G2_ratio_is_one"] is True
    assert out["ratio_mean"] == pytest.approx(1.0, abs=1e-6)


def test_g1_zero_std_ratio_counts_degenerate_groups():
    groups = [[1.0, 1.0, 1.0], [1.0, 2.0, 3.0], [0.0, 0.0, 0.0],
              [-1.0, 0.0, 1.0]]
    stats = RL.grpo_group_stats(groups)
    assert stats["n_groups"] == 4
    assert stats["zero_std_groups"] == 2
    assert stats["zero_std_group_ratio"] == pytest.approx(0.5)
    assert stats["inconclusive"] is False


def test_g1_flags_inconclusive_above_threshold():
    groups = [[1.0, 1.0]] * 9 + [[1.0, 2.0]]
    stats = RL.grpo_group_stats(groups)
    assert stats["zero_std_group_ratio"] == pytest.approx(0.9)
    assert stats["inconclusive"] is True


def test_advantage_is_zero_for_degenerate_group():
    adv = RL.group_advantage([[2.0, 2.0, 2.0]])
    assert all(a == pytest.approx(0.0) for a in adv[0])


def test_advantage_is_standardized_within_group():
    adv = RL.group_advantage([[0.0, 1.0, 2.0]], eps=0.0)
    assert sum(adv[0]) == pytest.approx(0.0, abs=1e-9)
    assert adv[0][0] < 0 < adv[0][2]


def test_rule_reward_is_bounded_and_deterministic():
    reward = RL.RuleReward()
    question = "什么是梯度裁剪？"
    good = reward.score(question, "梯度裁剪把全局梯度范数压到阈值以内，防止一步走太远。")
    empty = reward.score(question, "   ")
    repeat = reward.score(question, "梯度裁剪" * 40)
    for value in (good, empty, repeat):
        assert -3.0 <= value <= 3.0
    assert empty == -1.0
    assert repeat < good
    assert reward.score(question, "abc") == reward.score(question, "abc")


def test_rule_reward_resolution_is_coarse():
    """规则 reward 的值域只有几个离散点 —— 这正是 G1 会偏高的原因。"""
    reward = RL.RuleReward()
    values = {reward.score("问题", "回答" * n) for n in range(1, 30)}
    assert len(values) <= 5


def test_summarize_status_transitions(batch):
    model, _ = M.build_model(TINY, device="cpu", seed=13)
    dpo = RL.dpo_init_check(model, batch, device="cpu")
    ratio = RL.ratio_at_step0(model(batch["input_ids_chosen"]).logits,
                              batch["input_ids_chosen"],
                              batch["labels_chosen"])
    good = RL.summarize(dpo, RL.grpo_group_stats([[1.0, 2.0], [0.0, 3.0]]),
                        ratio)
    assert good["status"] == "PASS"
    incon = RL.summarize(dpo, RL.grpo_group_stats([[1.0, 1.0]] * 10), ratio)
    assert incon["status"] == "INCONCLUSIVE"
    broken = dict(dpo)
    broken["D1_init_loss_is_ln2"] = False
    bad = RL.summarize(broken, None, ratio)
    assert bad["status"] == "FAIL-SYSTEM"


def test_grpo_group_stats_rejects_empty():
    with pytest.raises(ValueError):
        RL.grpo_group_stats([])
