"""R6 的单测：LoRA policy 的四项字节账与 step-0 不变量。

**test_step0_kl_is_exactly_zero 是整个 lab 里唯一一条断言「精确等于 0」的测试。**
B 零初始化 => B(A(x)) 逐元素精确为 0 => y + 0.0 == y（IEEE 下成立）
=> policy 与 ref 的 logits 逐位相同 => log_ratio 恒为 0 => k1/k2/k3 全为 0。
它不是「小于 1e-6」，因为这里根本没有舍入误差可言。

这条不变量是 RL 训练开始前的一次性冒烟：它过了，才说明 policy 和 ref
真的是同一个起点；没过就别往下跑，KL 项从第 0 步就是错的。
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from mm_rl import lora_policy as LORA  # noqa: E402
from mm_rl.toy import TinyCausalLM, ToyConfig, make_batch  # noqa: E402


def _setup(r: int = 8, alpha: int = 16, dropout: float = 0.0, seed: int = 99):
    base = TinyCausalLM(ToyConfig(), seed=seed)
    ref = copy.deepcopy(base).eval()
    policy = copy.deepcopy(base)
    policy, targets = LORA.apply_lora(
        policy, ("q_proj", "v_proj"), r=r, alpha=alpha, dropout=dropout
    )
    b = make_batch(2, 12, 256, seed=seed + 1)
    return base, ref, policy, targets, b


def test_lora_b_is_zero_initialised_and_a_is_not():
    _base, _ref, policy, _t, _b = _setup()
    mods = [m for m in policy.modules() if isinstance(m, LORA.LoRALinear)]
    assert mods
    for m in mods:
        assert float(m.lora_B.weight.detach().abs().max()) == 0.0
        assert float(m.lora_A.weight.detach().abs().max()) > 0.0


def test_step0_kl_is_exactly_zero():
    """**唯一一条精确为 0 的断言。** 不是「小于阈值」。"""
    _base, ref, policy, _t, b = _setup()
    inv = LORA.step0_identity_check(policy, ref, b["input_ids"], b["labels"], b["mask"])
    assert inv["logits_bitwise_equal"] is True
    assert inv["max_abs_logit_diff"] == 0.0
    assert inv["max_abs_log_ratio"] == 0.0
    assert inv["max_abs_kl"] == {"k1": 0.0, "k2": 0.0, "k3": 0.0}
    assert inv["exact_zero_kl"] is True


@pytest.mark.parametrize("r,alpha", [(1, 1), (4, 8), (16, 32), (64, 64)])
def test_step0_invariant_holds_for_every_rank_and_alpha(r, alpha):
    """不变量与 r / alpha 无关：只要 B 是 0，scaling 乘什么都还是 0。"""
    _base, ref, policy, _t, b = _setup(r=r, alpha=alpha)
    inv = LORA.step0_identity_check(policy, ref, b["input_ids"], b["labels"], b["mask"])
    assert inv["exact_zero_kl"] is True


def test_step0_invariant_survives_dropout_because_the_dropout_is_on_the_lora_branch():
    """dropout 加在 LoRA 分支的输入上，乘 B=0 之后仍是 0——不变量不受影响。

    真正会破坏不变量的是把 dropout 加到主分支上。这条测试把这个区别钉死。
    """
    _base, ref, policy, _t, b = _setup(dropout=0.5)
    policy.train()
    inv = LORA.step0_identity_check(policy, ref, b["input_ids"], b["labels"], b["mask"])
    assert inv["exact_zero_kl"] is True


def test_breaking_zero_init_breaks_the_invariant():
    """反向验证：不变量不是「恒真的废话」，破坏零初始化它就该挂。"""
    _base, ref, policy, _t, b = _setup()
    for m in policy.modules():
        if isinstance(m, LORA.LoRALinear):
            m.break_zero_init(0.02, seed=0)
    inv = LORA.step0_identity_check(policy, ref, b["input_ids"], b["labels"], b["mask"])
    assert inv["exact_zero_kl"] is False
    assert inv["logits_bitwise_equal"] is False
    assert inv["max_abs_kl"]["k3"] > 0.0


def test_apply_lora_freezes_the_backbone_and_trains_only_a_and_b():
    _base, _ref, policy, targets, _b = _setup()
    assert len(targets) == 4  # 2 层 x (q_proj, v_proj)
    trainable = {n for n, p in policy.named_parameters() if p.requires_grad}
    assert trainable
    assert all(("lora_A" in n or "lora_B" in n) for n in trainable)
    counts = LORA.trainable_parameter_count(policy)
    assert 0.0 < counts["trainable"] / counts["total"] < 0.1


def test_lora_parameters_helper_matches_requires_grad():
    _base, _ref, policy, _t, _b = _setup()
    by_helper = {id(p) for p in LORA.lora_parameters(policy)}
    by_flag = {id(p) for p in policy.parameters() if p.requires_grad}
    assert by_helper == by_flag


def test_apply_lora_raises_when_nothing_matches():
    model = TinyCausalLM(ToyConfig(), seed=0)
    with pytest.raises(ValueError) as exc:
        LORA.apply_lora(model, ("attention.query",))
    assert "named_modules" in str(exc.value)


def test_lora_rank_zero_is_rejected_before_it_becomes_inf():
    """r=0 会让 scaling = alpha/0 = inf，0 * inf = NaN。要在构造时就拦住。"""
    with pytest.raises(ValueError):
        LORA.LoRALinear(nn.Linear(8, 8), r=0, alpha=8)


def test_byte_ledger_shows_gradient_and_optimizer_savings_but_not_parameter_savings():
    """LoRA 省的是梯度与优化器状态，**不省参数**——backbone 还在显存里。"""
    base, _ref, policy, _t, b = _setup()
    full = LORA.byte_ledger(copy.deepcopy(base), b["input_ids"])
    lora = LORA.byte_ledger(policy, b["input_ids"])

    assert lora["param_bytes"] > full["param_bytes"], "LoRA 多了 A/B，参数只会更多"
    assert lora["grad_bytes"] < full["grad_bytes"] * 0.1
    assert lora["optimizer_bytes"] < full["optimizer_bytes"] * 0.1
    assert lora["activation_bytes_estimate"] < full["activation_bytes_estimate"]
    assert lora["total_bytes"] < full["total_bytes"] * 0.5
    assert full["trainable_ratio"] == pytest.approx(1.0, rel=1e-9)
    assert lora["trainable_ratio"] < 0.1


def test_byte_ledger_arithmetic_is_internally_consistent():
    base, _ref, _policy, _t, b = _setup()
    led = LORA.byte_ledger(copy.deepcopy(base), b["input_ids"])
    assert led["param_bytes"] == led["n_params_total"] * 4
    assert led["grad_bytes"] == led["n_params_trainable"] * 4
    assert led["optimizer_bytes"] == led["n_params_trainable"] * 8
    assert led["total_bytes"] == (
        led["param_bytes"] + led["grad_bytes"] + led["optimizer_bytes"]
        + led["activation_bytes_estimate"]
    )
    assert "估算" in led["note"]


def test_byte_ledger_restores_the_training_flag():
    base, _ref, _policy, _t, b = _setup()
    m = copy.deepcopy(base).eval()
    LORA.byte_ledger(m, b["input_ids"])
    assert m.training is False


def test_lora_actually_learns_and_moves_away_from_the_reference():
    """训一步之后 B 不再是 0，KL 也就不再是 0——不变量只在 step 0 成立。"""
    _base, ref, policy, _t, b = _setup()
    opt = torch.optim.SGD(LORA.lora_parameters(policy), lr=1.0)
    logits = policy(b["input_ids"])
    loss = logits.pow(2).mean()
    loss.backward()
    grads = [p.grad for p in LORA.lora_parameters(policy)]
    assert all(g is not None for g in grads)
    assert any(float(g.abs().max()) > 0 for g in grads)
    opt.step()
    inv = LORA.step0_identity_check(policy, ref, b["input_ids"], b["labels"], b["mask"])
    assert inv["exact_zero_kl"] is False


def test_format_byte_ledger_lists_all_four_rows():
    base, _ref, policy, _t, b = _setup()
    text = LORA.format_byte_ledger(
        LORA.byte_ledger(copy.deepcopy(base), b["input_ids"]),
        LORA.byte_ledger(policy, b["input_ids"]),
    )
    for row in ("参数", "梯度", "优化器状态", "激活(估算)", "合计"):
        assert row in text
