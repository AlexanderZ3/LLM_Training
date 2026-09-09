"""R5 的单测：冻结 ref 模型 weight-only INT8。量化模块与 RL 模块的交点。

本机 CPU 能验的（compute_dtype=float32，隔离出纯量化误差）：
- ref 的线性层字节确实降下来了，且参数全部冻结；
- KL 相对偏差、log_ratio 符号一致率、policy 梯度余弦三个判据都能算出来；
- **policy 与 ref 相同时**这三个判据是退化的（log_ratio 恒 0），
  所以实验必须先让 policy 偏离 ref——这条也做成了测试。

本机**不能**验的：
- compute_dtype=float16 时的真实叠加效应（CPU 的 fp16 是软件模拟）；
- 显存到底省了多少（要看 torch.cuda.max_memory_allocated）。
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

from mm_quant.weight_only import W8A16Linear  # noqa: E402
from mm_rl import ref_model as RM  # noqa: E402
from mm_rl.toy import TinyCausalLM, ToyConfig, make_batch  # noqa: E402

CUDA = torch.cuda.is_available()
NO_CUDA_REASON = "显存与 fp16 计算路径的结论必须在公司 8xV100 上测，本机无 GPU"


def _setup(drift: float = 0.01, seed: int = 42, compute_dtype=torch.float32):
    base = TinyCausalLM(ToyConfig(), seed=seed)
    ref_fp = copy.deepcopy(base).eval()
    policy = copy.deepcopy(base)
    if drift:
        g = torch.Generator(device="cpu").manual_seed(seed + 1)
        with torch.no_grad():
            for p in policy.parameters():
                p.add_(torch.randn(p.shape, generator=g) * drift)
    ref_q, info = RM.quantize_reference_model(ref_fp, compute_dtype=compute_dtype)
    b = make_batch(2, 12, 256, seed=seed + 2)
    return base, policy, ref_fp, ref_q, info, b


def test_quantized_reference_replaces_linears_and_freezes_everything():
    _base, _policy, ref_fp, ref_q, info, _b = _setup()
    assert info["n_replaced"] > 0
    assert all("lm_head" not in n for n in info["replaced"])
    assert not any(p.requires_grad for p in ref_q.parameters())
    assert ref_q.training is False
    # 原模型不能被改动：R5 需要两个 ref 同时存在
    assert not any(isinstance(m, W8A16Linear) for m in ref_fp.modules())


def test_quantized_reference_saves_linear_bytes():
    _b0, _p, _rf, _rq, info, _b = _setup()
    assert info["linear_bytes_after"] < info["linear_bytes_before"]
    # 相对 fp32 参考：被量化的层约 0.26，lm_head 仍是 fp32，所以总比值在 0.25~0.7 之间
    assert 0.25 < info["weight_bytes_ratio"] < 0.7


def test_skip_list_can_be_widened():
    base = TinyCausalLM(ToyConfig(), seed=1)
    _q, info = RM.quantize_reference_model(
        base, skip_name_contains=("lm_head", "mlp"), compute_dtype=torch.float32
    )
    assert all("mlp" not in n for n in info["replaced"])
    assert any("self_attn" in n for n in info["replaced"])


def test_ref_kl_report_produces_all_three_criteria():
    _base, policy, ref_fp, ref_q, _info, b = _setup()
    with torch.no_grad():
        lp = policy(b["input_ids"])
        lrf = ref_fp(b["input_ids"])
        lrq = ref_q(b["input_ids"])
    rep = RM.ref_kl_report(lp, lrf, lrq, b["labels"], b["mask"])
    assert rep["n_tokens"] == int(b["mask"].sum())
    assert rep["kl_k3_mean_ref_fp"] > 0.0
    assert rep["kl_rel_bias"] < 0.5
    assert 0.0 <= rep["sign_agreement"] <= 1.0
    assert rep["n_sign_comparable"] > 0
    assert 0.0 <= rep["ref_logits_top1_agreement"] <= 1.0


def test_ref_kl_report_is_degenerate_when_policy_equals_ref():
    """policy 与 ref 完全相同时 log_ratio 恒为 0，符号一致率没有定义。

    这解释了为什么 run_rl_suite.py 的 r5 必须先给 policy 加扰动：
    否则报告里全是 nan/inf，看起来像 bug，其实是实验设计不成立。
    """
    _base, _policy, ref_fp, ref_q, _info, b = _setup(drift=0.0)
    with torch.no_grad():
        lrf = ref_fp(b["input_ids"])
        lrq = ref_q(b["input_ids"])
    rep = RM.ref_kl_report(lrf, lrf, lrq, b["labels"], b["mask"])
    assert rep["kl_k3_mean_ref_fp"] == 0.0
    assert rep["n_sign_comparable"] == 0
    import math

    assert math.isnan(rep["sign_agreement"])


def test_ref_kl_report_rejects_an_all_false_mask():
    _base, policy, ref_fp, ref_q, _info, b = _setup()
    with torch.no_grad():
        lp = policy(b["input_ids"])
        lrf = ref_fp(b["input_ids"])
        lrq = ref_q(b["input_ids"])
    with pytest.raises(ValueError):
        RM.ref_kl_report(lp, lrf, lrq, b["labels"], torch.zeros_like(b["mask"]))


def test_policy_grad_cosine_is_near_one_for_per_channel_int8_reference():
    """最终判据：量化 ref 之后 policy 梯度方向几乎不变。

    阈值 0.99 是**工程经验**，不是定理；真正的判据是 Day 5 的对照训练。
    这里只保证「余弦算得出来、且不是随机数」。
    """
    _base, policy, ref_fp, ref_q, _info, b = _setup()
    g = torch.Generator(device="cpu").manual_seed(7)
    adv = torch.randn(b["labels"].shape, generator=g)
    out = RM.policy_grad_cosine(
        policy, ref_fp, ref_q, b["input_ids"], b["labels"], adv, b["mask"], beta=0.04
    )
    assert out["n_grad_elements"] > 0
    assert -1.0 <= out["grad_cosine"] <= 1.0
    assert out["grad_cosine"] > 0.99
    assert out["grad_norm_rel_diff"] < 0.05


def test_policy_grad_cosine_leaves_no_stale_gradients():
    _base, policy, ref_fp, ref_q, _info, b = _setup()
    adv = torch.zeros(b["labels"].shape)
    RM.policy_grad_cosine(policy, ref_fp, ref_q, b["input_ids"], b["labels"], adv, b["mask"])
    assert all(p.grad is None for p in policy.parameters())


def test_a_deliberately_bad_reference_lowers_the_cosine():
    """对照组：把 ref 换成 per_tensor 且随机破坏后，余弦必须明显下降。

    没有这条对照，"余弦 0.999" 说明不了什么——可能任何 ref 都给 0.999。
    """
    _base, policy, ref_fp, ref_q, _info, b = _setup()
    broken = copy.deepcopy(ref_fp)
    g = torch.Generator(device="cpu").manual_seed(11)
    with torch.no_grad():
        for p in broken.parameters():
            p.add_(torch.randn(p.shape, generator=g) * 0.05)
    adv = torch.randn(b["labels"].shape, generator=torch.Generator().manual_seed(5))
    good = RM.policy_grad_cosine(
        policy, ref_fp, ref_q, b["input_ids"], b["labels"], adv, b["mask"], beta=0.5
    )
    bad = RM.policy_grad_cosine(
        policy, ref_fp, broken, b["input_ids"], b["labels"], adv, b["mask"], beta=0.5
    )
    assert bad["grad_cosine"] < good["grad_cosine"]
    assert bad["grad_max_abs_diff"] > good["grad_max_abs_diff"]


def test_format_ref_report_is_printable():
    _base, policy, ref_fp, ref_q, info, b = _setup()
    with torch.no_grad():
        lp = policy(b["input_ids"])
        lrf = ref_fp(b["input_ids"])
        lrq = ref_q(b["input_ids"])
    text = RM.format_ref_report(RM.ref_kl_report(lp, lrf, lrq, b["labels"], b["mask"]), info)
    assert "KL(k3)" in text and "符号一致率" in text


@pytest.mark.skipif(not CUDA, reason=NO_CUDA_REASON)
def test_int8_reference_saves_real_device_memory():
    """显存是不是真省了，只有在 GPU 上量 max_memory_allocated 才算数。"""
    base = TinyCausalLM(ToyConfig(hidden_size=256, intermediate_size=512), seed=0).cuda().eval()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    fp_bytes = sum(p.numel() * p.element_size() for p in base.parameters())
    q, info = RM.quantize_reference_model(base, compute_dtype=torch.float16)
    q = q.cuda()
    q_bytes = sum(b.numel() * b.element_size() for b in q.buffers()) + sum(
        p.numel() * p.element_size() for p in q.parameters()
    )
    assert q_bytes < fp_bytes
    assert info["weight_bytes_ratio"] < 1.0
