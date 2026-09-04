"""Day 3：apply_fault 纯函数（list 与 tensor 两种输入）。"""
from __future__ import annotations

import pytest
import torch

from mm_probe.mask_fault import FAULT_MODES, IGNORE_INDEX, apply_fault, count_in_loss

IDS = [1, 7, 5, 30, 2, 5, 1, 9, 9, 5, 40, 41, 2, 5, 0, 0]
LABELS = [IGNORE_INDEX] * 10 + [40, 41, 2, 5] + [IGNORE_INDEX] * 2


def test_modes_listed():
    assert FAULT_MODES == ("none", "assistant_all_ignored", "user_in_loss")


def test_none_is_copy():
    out = apply_fault(LABELS, "none")
    assert out == LABELS and out is not LABELS


def test_assistant_all_ignored_list_and_tensor():
    assert apply_fault(LABELS, "assistant_all_ignored") == [IGNORE_INDEX] * len(LABELS)
    t = apply_fault(torch.tensor(LABELS), "assistant_all_ignored")
    assert torch.all(t == IGNORE_INDEX) and count_in_loss(t) == 0


def test_user_in_loss_list_and_tensor():
    out = apply_fault(LABELS, "user_in_loss", input_ids=IDS, pad_token_id=0)
    assert out[:14] == IDS[:14] and out[14:] == [IGNORE_INDEX, IGNORE_INDEX]
    t = apply_fault(torch.tensor(LABELS), "user_in_loss", input_ids=torch.tensor(IDS), pad_token_id=0)
    assert t.tolist() == out
    assert count_in_loss(t) == 14 > count_in_loss(LABELS) == 4


def test_user_in_loss_requires_input_ids():
    with pytest.raises(ValueError):
        apply_fault(LABELS, "user_in_loss")


def test_unknown_mode():
    with pytest.raises(ValueError):
        apply_fault(LABELS, "bogus")


def test_original_not_mutated():
    src = torch.tensor(LABELS)
    _ = apply_fault(src, "assistant_all_ignored")
    assert src.tolist() == LABELS
