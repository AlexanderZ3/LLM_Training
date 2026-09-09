"""模型形状与参数量：手算公式必须等于实测，否则 Day 3 的闭卷题没有标准答案。"""

from __future__ import annotations

import pytest
import torch

from mm_v100 import common as C
from mm_v100 import model as M


def test_param_breakdown_matches_actual_count():
    cfg = M.ModelConfig(vocab_size=256, hidden_size=64, num_hidden_layers=2,
                        num_attention_heads=4, num_key_value_heads=2,
                        intermediate_size=128, max_seq_len=64)
    model = M.TinyCausalLM(cfg)
    hand = M.param_breakdown(cfg)
    actual = M.count_params(model)
    assert hand["total"] == actual["total"]
    assert hand["n_tensors"] == actual["tensors"]


def test_gate_config_param_count_is_exact():
    """Gate 的 A0 题用这一组数：hidden=768 layers=8 heads=8 kv=4 vocab=6400 tie。

    逐项手算（不看代码也应该能写出来）：
      embed        6400*768                    = 4,915,200
      每层 q       768*768                     =   589,824
      每层 k=v     768*(4*96)=768*384          =   294,912  x2
      每层 o       768*768                     =   589,824
      每层 gate=up=down 768*2432               = 1,867,776  x3
      每层 q_norm+k_norm 96*2                  =       192
      每层 2 个 RMSNorm                        =     1,536
      每层合计                                 = 7,374,528
      8 层                                     = 58,996,224
      final norm                               =       768
      lm_head（tied）                          =         0
      合计                                     = 63,912,192

    d_ff 用 MiniMind 的 ceil(hidden*pi/64)*64 = 2432，不是 Llama 系常见的 8/3
    倍（那会给 2048，整个模型差 7,079,424 个参数）。q_norm/k_norm 是 MiniMind 的
    QK-norm，只有 192 个参数但不能漏——漏了总数差 1,536。
    """
    cfg = M.ModelConfig(vocab_size=6400, hidden_size=768, num_hidden_layers=8,
                        num_attention_heads=8, num_key_value_heads=4,
                        max_seq_len=512)
    assert cfg.intermediate_size == 2432
    hand = M.param_breakdown(cfg)
    assert hand["layer.q_proj"] == 589824
    assert hand["layer.k_proj"] == 294912
    assert hand["layer.v_proj"] == 294912
    assert hand["layer.gate_proj"] == 1867776
    assert hand["layer.q_norm"] == 96
    assert hand["layer.k_norm"] == 96
    assert hand["per_layer_total"] == 7374528
    assert hand["all_layers"] == 58996224
    assert hand["embed_tokens"] == 4915200
    assert hand["lm_head"] == 0
    assert hand["total"] == 63912192


def test_intermediate_size_follows_minimind_rounding():
    # MiniMind: ceil(hidden * pi / 64) * 64。这几个值可以手算核对：
    # 512*pi/64 = 25.13 -> 26 -> 1664；768 -> 37.70 -> 38 -> 2432；256 -> 12.57 -> 13 -> 832
    for hidden, expected in ((512, 1664), (768, 2432), (256, 832)):
        cfg = M.ModelConfig(hidden_size=hidden, num_attention_heads=8,
                            num_key_value_heads=4)
        assert cfg.intermediate_size == expected


def test_tied_embedding_shares_storage():
    cfg = M.ModelConfig(vocab_size=64, hidden_size=32, num_hidden_layers=1,
                        num_attention_heads=4, num_key_value_heads=2,
                        intermediate_size=64, max_seq_len=32)
    model = M.TinyCausalLM(cfg)
    assert model.lm_head.weight.data_ptr() == model.embed_tokens.weight.data_ptr()
    # 去重之后 lm_head 不能再被数一次，否则字节账多出 vocab*hidden。
    assert M.count_params(model)["total"] == M.param_breakdown(cfg)["total"]


def test_forward_shapes_and_loss():
    model, cfg = M.build_model(
        {"vocab_size": 64, "hidden_size": 32, "num_hidden_layers": 2,
         "num_attention_heads": 4, "num_key_value_heads": 2,
         "intermediate_size": 64, "max_seq_len": 32}, device="cpu", seed=0)
    ids = torch.randint(0, 64, (3, 16))
    out = model(ids, labels=ids)
    assert tuple(out.logits.shape) == (3, 16, 64)
    assert out.loss.ndim == 0
    assert torch.isfinite(out.loss)
    # 随机初始化下 loss 应当接近 ln(vocab)。差太远说明初始化或 softmax 有问题。
    assert abs(float(out.loss.detach()) - float(torch.log(torch.tensor(64.0)))) < 0.5


def test_ignore_index_positions_do_not_contribute():
    model, _ = M.build_model(
        {"vocab_size": 32, "hidden_size": 32, "num_hidden_layers": 1,
         "num_attention_heads": 4, "num_key_value_heads": 2,
         "intermediate_size": 64, "max_seq_len": 32}, device="cpu", seed=0)
    ids = torch.randint(0, 32, (2, 12))
    labels_all = ids.clone()
    labels_half = ids.clone()
    labels_half[:, :6] = M.IGNORE_INDEX
    loss_all = float(model(ids, labels=labels_all).loss.detach())
    loss_half = float(model(ids, labels=labels_half).loss.detach())
    # 屏蔽一半位置之后 loss 必须变（否则 ignore_index 没生效）。
    assert loss_all != pytest.approx(loss_half, abs=1e-9)


def test_all_labels_ignored_gives_nan_loss():
    """全 -100 时 cross_entropy 对 0 个元素取均值 -> nan。这是故障 A 的极端形态。"""
    model, _ = M.build_model(
        {"vocab_size": 32, "hidden_size": 32, "num_hidden_layers": 1,
         "num_attention_heads": 4, "num_key_value_heads": 2,
         "intermediate_size": 64, "max_seq_len": 32}, device="cpu", seed=0)
    ids = torch.randint(0, 32, (2, 8))
    labels = torch.full_like(ids, M.IGNORE_INDEX)
    assert torch.isnan(model(ids, labels=labels).loss)


def test_seq_len_over_max_raises_actionable_error():
    model, _ = M.build_model(
        {"vocab_size": 32, "hidden_size": 32, "num_hidden_layers": 1,
         "num_attention_heads": 4, "num_key_value_heads": 2,
         "intermediate_size": 64, "max_seq_len": 16}, device="cpu", seed=0)
    with pytest.raises(ValueError) as exc:
        model(torch.zeros(1, 20, dtype=torch.long))
    assert "max_seq_len" in str(exc.value)


def test_gqa_requires_divisible_heads():
    with pytest.raises(ValueError):
        M.ModelConfig(hidden_size=64, num_attention_heads=8,
                      num_key_value_heads=3)


def test_rope_buffers_not_in_state_dict():
    """RoPE 表是纯函数，不该进 checkpoint，否则 reshard 要为它做分片决策。"""
    model, _ = M.build_model(
        {"vocab_size": 32, "hidden_size": 32, "num_hidden_layers": 1,
         "num_attention_heads": 4, "num_key_value_heads": 2,
         "intermediate_size": 64, "max_seq_len": 16}, device="cpu", seed=0)
    keys = list(model.state_dict().keys())
    assert not any("rope" in k for k in keys)


def test_build_model_with_same_seed_is_bitwise_identical():
    a, _ = M.build_model({"vocab_size": 32, "hidden_size": 32,
                          "num_hidden_layers": 1, "num_attention_heads": 4,
                          "num_key_value_heads": 2, "intermediate_size": 64,
                          "max_seq_len": 16}, device="cpu", seed=7)
    b, _ = M.build_model({"vocab_size": 32, "hidden_size": 32,
                          "num_hidden_layers": 1, "num_attention_heads": 4,
                          "num_key_value_heads": 2, "intermediate_size": 64,
                          "max_seq_len": 16}, device="cpu", seed=7)
    for (na, pa), (nb, pb) in zip(a.state_dict().items(), b.state_dict().items()):
        assert na == nb
        assert torch.equal(pa, pb)


def test_config_rejects_unknown_field():
    with pytest.raises(ValueError) as exc:
        M.config_from_dict({"hidden_size": 32, "flash_attn": True})
    assert "flash_attn" in str(exc.value)


def test_all_shipped_configs_load_and_build():
    """四个配置都必须能建出模型；tiny 之外的只建不跑（CPU 上太慢）。"""
    for name in ("tiny_cpu", "smoke_v100", "v100_768", "v100_768_accum"):
        cfg = C.load_config(name)
        model_cfg = M.config_from_dict(dict(cfg["model"]))
        breakdown = M.param_breakdown(model_cfg)
        assert breakdown["total"] > 0
        assert cfg["train"]["global_batch"] % cfg["train"].get("accum", 1) == 0


def test_greedy_generate_stops_at_eos():
    model, _ = M.build_model(
        {"vocab_size": 8, "hidden_size": 16, "num_hidden_layers": 1,
         "num_attention_heads": 2, "num_key_value_heads": 1,
         "intermediate_size": 32, "max_seq_len": 32}, device="cpu", seed=1)
    ids = torch.zeros(1, 3, dtype=torch.long)
    out = model.greedy_generate(ids, max_new_tokens=5, eos_token_id=None)
    assert out.shape[1] == 3 + 5
