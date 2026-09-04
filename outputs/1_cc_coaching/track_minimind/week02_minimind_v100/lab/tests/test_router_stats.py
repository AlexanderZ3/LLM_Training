"""router 统计与热点注入：计数口径、bias 效果、aux loss 公式。

这些都在单进程 CPU 上跑得完，因为 router 是**每张卡都有完整副本**的部分——
它的行为与 world_size 无关。需要多卡才能看到的是负载不均导致的等待时间，
那部分在公司机器上由 ``scripts/run_moe_sweep.*`` 采集。
"""

from __future__ import annotations

import csv
import os

import pytest
import torch
import torch.nn.functional as F

from mm_dist import common as C
from mm_dist import router_stats
from mm_dist.ep_moe import aux_loss_value


@pytest.fixture()
def moe_model(moe_config, minimind_root):
    C.set_seed(1234, per_rank=False)
    model, lm_config, cfg, mm = C.build_model(moe_config, device="cpu")
    return model, lm_config, mm


def test_find_moe_layers_counts_every_block(moe_model):
    model, lm_config, mm = moe_model
    layers = router_stats.find_moe_layers(model, mm.MOEFeedForward)
    assert len(layers) == int(lm_config.num_hidden_layers)
    for name, mod in layers:
        assert name.endswith(".mlp")
        assert isinstance(mod.gate, torch.nn.Linear)
        assert mod.gate.bias is None, "MiniMind 的 router 是 bias=False 的线性层"


def test_collector_counts_match_tokens(moe_model):
    """每层 token 计数之和 = token 数 × top_k；这是统计口径的自检。"""
    model, lm_config, mm = moe_model
    collector = router_stats.RouterStatsCollector(model, moe_cls=mm.MOEFeedForward)
    collector.attach()
    model.eval()
    batch, seq = 2, 16
    ids = torch.randint(0, int(lm_config.vocab_size), (batch, seq))
    with torch.no_grad():
        model(ids, labels=ids)
    collector.detach()
    summary = collector.summary()
    expected = batch * seq * int(lm_config.num_experts_per_tok) \
        * int(lm_config.num_hidden_layers)
    assert sum(summary["expert_counts"]) == expected
    assert abs(sum(summary["expert_fractions"]) - 1.0) < 1e-9
    assert abs(summary["mean_load_ratio"] - 1.0) < 1e-6
    assert 0.0 <= summary["router_entropy"] <= summary["router_entropy_max"] + 1e-9


def test_router_bias_creates_hot_expert(moe_model):
    """给 expert 2 的 logit 加 50，所有 token 都应该被路由到它。"""
    model, lm_config, mm = moe_model
    hot = 2
    collector = router_stats.RouterStatsCollector(model, moe_cls=mm.MOEFeedForward)
    handle = router_stats.inject_router_bias(model, expert_id=hot, bias=50.0,
                                             moe_cls=mm.MOEFeedForward)
    collector.attach()
    model.eval()
    ids = torch.randint(0, int(lm_config.vocab_size), (2, 16))
    with torch.no_grad():
        model(ids, labels=ids)
    collector.detach()
    handle.remove()
    summary = collector.summary()
    e = summary["num_experts"]
    assert summary["expert_fractions"][hot] > 0.99, summary["expert_fractions"]
    assert abs(summary["max_load_ratio"] - e) < 0.05, (
        "全部坍缩到一个专家时 max_load_ratio 应接近 E=" + str(e))
    assert summary["router_entropy"] < 0.05, "坍缩时 router 熵应接近 0"


def test_bias_zero_is_a_noop(moe_model):
    """bias=0 必须与不注入完全等价，否则扫描的基线就不干净。"""
    model, lm_config, mm = moe_model
    model.eval()
    ids = torch.randint(0, int(lm_config.vocab_size), (2, 8))
    with torch.no_grad():
        base = model(ids, labels=ids).logits.clone()
    handle = router_stats.inject_router_bias(model, expert_id=1, bias=0.0,
                                             moe_cls=mm.MOEFeedForward)
    with torch.no_grad():
        biased = model(ids, labels=ids).logits
    handle.remove()
    assert torch.allclose(base, biased, atol=0, rtol=0)


def test_bias_handle_removes_hooks(moe_model):
    model, lm_config, mm = moe_model
    layers = router_stats.find_moe_layers(model, mm.MOEFeedForward)
    before = len(layers[0][1].gate._forward_hooks)
    handle = router_stats.inject_router_bias(model, expert_id=0, bias=1.0,
                                             moe_cls=mm.MOEFeedForward)
    assert len(layers[0][1].gate._forward_hooks) == before + 1
    handle.remove()
    assert len(layers[0][1].gate._forward_hooks) == before


def test_invalid_expert_id_raises(moe_model):
    model, lm_config, mm = moe_model
    handle = router_stats.inject_router_bias(model, expert_id=99, bias=1.0,
                                             moe_cls=mm.MOEFeedForward)
    ids = torch.randint(0, int(lm_config.vocab_size), (1, 4))
    with pytest.raises(ValueError):
        model(ids, labels=ids)
    handle.remove()


def test_aux_loss_matches_minimind_expression():
    """逐字对照 MiniMind L169-171 的 (load*scores.mean(0)).sum()*E*coef。"""
    torch.manual_seed(3)
    n, e, k, coef = 32, 4, 2, 5e-4
    logits = torch.randn(n, e)
    scores = F.softmax(logits, dim=-1)
    _, idx = torch.topk(scores, k=k, dim=-1, sorted=False)
    load = F.one_hot(idx, e).float().mean(0)
    expected = (load * scores.mean(0)).sum() * e * coef
    assert abs(float(aux_loss_value(scores, idx, e, coef)) - float(expected)) < 1e-12


def test_parse_float_list():
    assert router_stats._parse_float_list("0,0.5,1,2,4") == [0.0, 0.5, 1.0, 2.0, 4.0]
    assert router_stats._parse_float_list(" 0 , 1e-3 ") == [0.0, 0.001]
    with pytest.raises(ValueError):
        router_stats._parse_float_list("")


def test_sweep_writes_csv(tmp_path, moe_config, minimind_root):
    """2 个 bias × 2 个 aux_coef，每组 1 步；只验证 CSV 的形状与列名。"""
    out_dir = str(tmp_path / "sweep")
    csv_path = str(tmp_path / "sweep" / "moe_sweep.csv")
    args = router_stats.build_parser().parse_args([
        "sweep", "--config", moe_config, "--out-dir", out_dir,
        "--force-cpu", "--backend", "gloo", "--dtype", "float32",
        "--bias-list", "0,2", "--aux-coef-list", "0,0.01",
        "--global-batch", "4", "--accum", "1", "--seq-len", "16",
        "--max-steps", "1", "--num-samples", "64", "--csv", csv_path,
    ])
    assert router_stats.cmd_sweep(args) == 0
    assert os.path.isfile(csv_path)
    with open(csv_path, "r", encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 4
    assert set(rows[0]) == set(router_stats.CSV_COLUMNS)
    biases = sorted({float(r["bias"]) for r in rows})
    coefs = sorted({float(r["aux_coef"]) for r in rows})
    assert biases == [0.0, 2.0]
    assert coefs == [0.0, 0.01]
    for row in rows:
        assert float(row["mean_load_ratio"]) == pytest.approx(1.0, abs=1e-6)
        assert 1.0 <= float(row["max_load_ratio"]) <= float(row["num_experts"]) + 1e-6
        assert len(row["expert_fractions"].split("|")) == int(row["num_experts"])
    hot = [r for r in rows if float(r["bias"]) == 2.0]
    cold = [r for r in rows if float(r["bias"]) == 0.0]
    assert min(float(r["max_load_ratio"]) for r in hot) \
        >= min(float(r["max_load_ratio"]) for r in cold) - 1e-6, (
        "bias=2 的负载至少不该比 bias=0 更均衡")
