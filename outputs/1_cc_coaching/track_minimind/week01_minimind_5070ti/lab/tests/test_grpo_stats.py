"""Day 5：GRPO 组内统计（reward 全等 → advantage 全 0；与 MiniMind 的 (r-mean)/(std+1e-4) 一致）。"""
from __future__ import annotations

import math

import pytest

from mm_probe.grpo_stats import group_advantage, group_stats, parse_debug_log, summarize


def test_all_equal_rewards_give_zero_advantage():
    adv = group_advantage([[1.5, 1.5, 1.5, 1.5]])
    assert adv == [[0.0, 0.0, 0.0, 0.0]]
    st = group_stats([[2.0, 2.0]])[0]
    assert st["zero_std"] is True and st["std"] == 0.0


def test_advantage_matches_minimind_formula():
    g = [1.0, 2.0, 3.0]
    m, s = 2.0, math.sqrt(2.0 / 3.0)  # population std (unbiased=False)
    adv = group_advantage([g], eps=1e-4)[0]
    assert adv == pytest.approx([(x - m) / (s + 1e-4) for x in g])
    torch = pytest.importorskip("torch")
    r = torch.tensor(g)
    ref = ((r - r.mean()) / (r.std(unbiased=False) + 1e-4)).tolist()
    assert adv == pytest.approx(ref, abs=1e-6)


def test_flat_input_with_num_generations():
    flat = [1.0, 3.0, 5.0, 5.0]
    adv = group_advantage(flat, num_generations=2)
    assert len(adv) == 4 and adv[0] == pytest.approx(-adv[1]) and adv[2] == adv[3] == 0.0
    with pytest.raises(ValueError):
        group_advantage(flat)  # 扁平输入缺 num_generations


def test_summarize_zero_std_ratio_and_len():
    recs = [
        {"step": 1, "rewards": [[1.0, 1.0], [0.0, 2.0]], "lengths": [[10, 20], [30, 40]]},
        {"step": 2, "rewards": [0.5, 0.5, 0.5, 0.5], "num_generations": 2, "avg_len": 7.0},
    ]
    s = summarize(recs)
    assert s["n_groups"] == 4 and s["zero_std_group_ratio"] == pytest.approx(0.75)
    assert s["per_step"][0]["zero_std_groups"] == 1 and s["per_step"][0]["avg_len"] == 25.0
    assert s["per_step"][1]["adv_std"] == 0.0
    assert s["mean_len"] == pytest.approx((10 + 20 + 30 + 40 + 7) / 5)


def test_parse_debug_log(tmp_path):
    log = "\n".join([
        "Model Params: 25.83M",
        "[DEBUG] step=1, sample[0]",
        "-" * 100,
        "============================== [DEBUG] sample[0] CONTEXT_BEGIN ==============================",
        "<|im_start|>user\nhi<|im_end|>",
        "=============================== [DEBUG] sample[0] CONTEXT_END ===============================",
        "============================ [DEBUG] gen[0] RESPONSE_BEGIN ============================",
        "hello there",
        "============================= [DEBUG] gen[0] RESPONSE_END =============================",
        "[DEBUG] gen[0] reward=1.2500",
        "============================ [DEBUG] gen[1] RESPONSE_BEGIN ============================",
        "yo",
        "============================= [DEBUG] gen[1] RESPONSE_END =============================",
        "[DEBUG] gen[1] reward=-0.7500",
        "=" * 100,
        "Epoch:[1/1](1/100), Reward: 0.2500, KL_ref: 0.0010, Adv Std: 0.9999, Adv Mean: 0.0000, Actor Loss: -0.0100, Avg Response Len: 12.50, Learning Rate: 0.00000030",
    ])
    p = tmp_path / "grpo.log"
    p.write_text(log, encoding="utf-8")
    recs = parse_debug_log(str(p))
    assert len(recs) == 1 and recs[0]["step"] == 1
    assert recs[0]["rewards"] == [[1.25, -0.75]]
    assert recs[0]["avg_len"] == 12.5
    assert recs[0]["resp_chars"] == [[len("hello there") + 1, len("yo") + 1]]
    s = summarize(recs)
    assert s["zero_std_group_ratio"] == 0.0 and s["per_step"][0]["mean_reward"] == pytest.approx(0.25)
