"""Day 4：DPO 数学的手算小例子（不依赖 MiniMind）。"""
from __future__ import annotations

import math

import pytest
import torch

from mm_probe.dpo_check import dpo_loss, dpo_loss_minimind, param_hash, per_token_logps, sequence_logp


def test_per_token_logps_manual():
    # B=1, T=2, V=3
    logits = torch.tensor([[[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]])
    labels = torch.tensor([[0, 1]])
    lp = per_token_logps(logits, labels)
    assert lp[0, 0].item() == pytest.approx(1.0 - math.log(math.e + 2.0), abs=1e-6)
    assert lp[0, 1].item() == pytest.approx(-math.log(3.0), abs=1e-6)
    seq = sequence_logp(logits, labels, torch.tensor([[1, 0]]))
    assert seq[0].item() == pytest.approx(1.0 - math.log(math.e + 2.0), abs=1e-6), "mask=0 的位置不计入序列 logp"


def test_dpo_loss_hand_example():
    pc, pr, rc, rr = torch.tensor([-1.0]), torch.tensor([-3.0]), torch.tensor([-1.5]), torch.tensor([-2.5])
    out = dpo_loss(pc, pr, rc, rr, beta=0.15)
    # pi_logratio = 2, ref_logratio = 1 → logits_dpo = 1 → loss = -logsigmoid(0.15) = log(1 + e^-0.15)
    assert out["logits_dpo"].item() == pytest.approx(1.0)
    assert out["loss"].item() == pytest.approx(math.log(1 + math.exp(-0.15)), abs=1e-6)
    assert out["reward_margin"].item() == pytest.approx(0.15)
    assert out["chosen_reward"].item() == pytest.approx(0.15 * 0.5)
    assert out["rejected_reward"].item() == pytest.approx(0.15 * -0.5)
    assert out["accuracy"].item() == 1.0


def test_dpo_loss_is_ln2_when_policy_equals_ref():
    x = torch.tensor([-4.0, -7.5])
    y = torch.tensor([-9.0, -2.0])
    out = dpo_loss(x, y, x, y, beta=0.15)
    assert out["loss"].item() == pytest.approx(math.log(2.0), abs=1e-6)
    assert torch.all(out["reward_margin"] == 0)


def test_minimind_signature_matches():
    torch.manual_seed(0)
    ref = torch.randn(4, 6)   # [2B, T]，前 2 行 chosen，后 2 行 rejected
    pol = torch.randn(4, 6)
    mask = (torch.rand(4, 6) > 0.3).float()
    loss_a = dpo_loss_minimind(ref, pol, mask, beta=0.15)
    rs, ps = (ref * mask).sum(1), (pol * mask).sum(1)
    loss_b = dpo_loss(ps[:2], ps[2:], rs[:2], rs[2:], beta=0.15)["loss"]
    assert loss_a.item() == pytest.approx(loss_b.item(), abs=1e-6)


def test_beta_scales_gradient_signal():
    pc, pr, rc, rr = torch.tensor([-1.0]), torch.tensor([-3.0]), torch.tensor([-1.5]), torch.tensor([-2.5])
    small = dpo_loss(pc, pr, rc, rr, beta=0.05)["loss"].item()
    large = dpo_loss(pc, pr, rc, rr, beta=0.5)["loss"].item()
    assert large < small < math.log(2.0), "logits_dpo>0 时 beta 越大 loss 越小"


def test_param_hash_changes_only_when_params_change():
    m = torch.nn.Linear(3, 2)
    h0 = param_hash(m)
    assert param_hash(m) == h0
    with torch.no_grad():
        m.weight.add_(1e-3)
    assert param_hash(m) != h0
