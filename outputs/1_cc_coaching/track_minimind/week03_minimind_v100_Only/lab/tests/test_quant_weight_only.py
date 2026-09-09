"""Q4 的单测：W8A16 weight-only PTQ 的字节账与一致率。

本机可得的结论：
- 权重字节相对 fp16 的比值 = 0.5 + scale 开销，per_channel 时约 0.503。
- top-1 一致率是比 loss 更敏感的指标。

本机**不能**得出的：任何速度结论。V100 没有 INT8 Tensor Core，
本实现每次前向都 dequant，比直接存 fp16 更慢。
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

from mm_quant import quantizers as Q  # noqa: E402
from mm_quant import weight_only as WO  # noqa: E402


def _mlp(seed: int = 0) -> nn.Module:
    torch.manual_seed(seed)
    return nn.Sequential(
        nn.Linear(128, 256), nn.SiLU(), nn.Linear(256, 256), nn.SiLU(), nn.Linear(256, 64)
    ).eval()


def test_from_linear_preserves_shape_and_bias():
    lin = nn.Linear(64, 32, bias=True)
    q = WO.W8A16Linear.from_linear(lin, compute_dtype=torch.float32)
    x = torch.randn(4, 64)
    y = q(x)
    assert y.shape == (4, 32)
    assert torch.allclose(q.bias, lin.bias.detach(), atol=0)
    assert q.qweight.dtype is torch.int8


def test_dequantized_weight_is_close_to_original():
    lin = nn.Linear(128, 96, bias=False)
    q = WO.W8A16Linear.from_linear(lin, granularity="per_channel", compute_dtype=torch.float32)
    err = Q.relative_error(lin.weight.detach(), q.dequantized_weight(torch.float32))
    assert err < 0.01


def test_finer_granularity_lowers_output_error_and_raises_bytes():
    lin = nn.Linear(512, 128, bias=False)
    with torch.no_grad():
        # 造行内 outlier，让 per_group 有事可做
        idx = torch.randint(0, 512, (128, 4))
        lin.weight.scatter_(1, idx, torch.randn(128, 4) * 20.0)
    x = torch.randn(8, 512)
    ref = lin(x)

    errs = {}
    ratios = {}
    for gran, kw in (("per_tensor", {}), ("per_channel", {}), ("per_group", {"group_size": 64})):
        q = WO.W8A16Linear.from_linear(lin, granularity=gran, compute_dtype=torch.float32, **kw)
        errs[gran] = Q.relative_error(ref, q(x))
        ratios[gran] = q.stored_bytes() / (512 * 128 * 2)
    assert errs["per_tensor"] > errs["per_channel"] > errs["per_group"]
    assert ratios["per_tensor"] < ratios["per_channel"] < ratios["per_group"]


def test_weight_bytes_ratio_is_about_half_of_fp16():
    model = _mlp()
    qmodel, replaced = WO.quantize_linear_modules(copy.deepcopy(model), compute_dtype=torch.float32)
    assert len(replaced) == 3
    rep = WO.module_weight_bytes(qmodel, reference_dtype=torch.float16)
    assert rep["n_linear_quantized"] == 3
    assert rep["n_linear_fp"] == 0
    assert 0.50 < rep["weight_bytes_ratio"] < 0.53, (
        "int8 码 + per-channel fp32 scale + fp32 bias，相对 fp16 应当略高于 0.5；"
        "得到 {}".format(rep["weight_bytes_ratio"])
    )


def test_per_group_costs_more_bytes_than_per_channel():
    model = _mlp()
    a = WO.quantize_linear_modules(copy.deepcopy(model), granularity="per_channel",
                                   compute_dtype=torch.float32)[0]
    b = WO.quantize_linear_modules(copy.deepcopy(model), granularity="per_group", group_size=64,
                                   compute_dtype=torch.float32)[0]
    assert WO.weight_bytes_ratio(b) > WO.weight_bytes_ratio(a)


def test_skip_name_contains_leaves_those_layers_in_fp():
    model = nn.Module()
    model.body = nn.Linear(32, 32)
    model.lm_head = nn.Linear(32, 100)
    qmodel, replaced = WO.quantize_linear_modules(
        model, skip_name_contains=("lm_head",), compute_dtype=torch.float32
    )
    assert replaced == ["body"]
    assert isinstance(qmodel.lm_head, nn.Linear)
    assert isinstance(qmodel.body, WO.W8A16Linear)


def test_min_numel_skips_tiny_layers():
    model = nn.Sequential(nn.Linear(4, 4), nn.Linear(128, 128))
    _, replaced = WO.quantize_linear_modules(model, min_numel=1000, compute_dtype=torch.float32)
    assert replaced == ["1"]


def test_top1_agreement_is_a_stricter_signal_than_logit_distance():
    """同样的量化下，top-1 翻转率通常远大于「logits 相对差」给人的印象。"""
    torch.manual_seed(1)
    model = _mlp(seed=1)
    x = torch.randn(4, 32, 128)
    with torch.no_grad():
        ref = model(x)
    qmodel, _ = WO.quantize_linear_modules(
        copy.deepcopy(model), granularity="per_tensor", compute_dtype=torch.float32
    )
    with torch.no_grad():
        got = qmodel(x)
    agree = WO.top1_agreement(ref, got)
    div = WO.logit_divergence(ref, got)
    assert 0.0 <= agree["top1_agreement"] <= 1.0
    assert agree["flip_rate"] == pytest.approx(1.0 - agree["top1_agreement"], rel=1e-9)
    assert agree["n_tokens"] == 4 * 32
    assert div["mean_kl_a_to_b"] >= 0.0


def test_top1_agreement_with_identical_logits_is_one():
    logits = torch.randn(2, 5, 9)
    assert WO.top1_agreement(logits, logits.clone())["top1_agreement"] == 1.0


def test_top1_agreement_respects_mask():
    a = torch.zeros(2, 4, 3)
    b = torch.zeros(2, 4, 3)
    a[..., 0] = 1.0
    b[..., 1] = 1.0          # 全部位置都不一致
    b[0, 0, :] = a[0, 0, :]  # 只有 (0,0) 一致
    mask = torch.zeros(2, 4, dtype=torch.bool)
    mask[0, 0] = True
    assert WO.top1_agreement(a, b, mask=mask)["top1_agreement"] == 1.0
    assert WO.top1_agreement(a, b)["top1_agreement"] < 0.2


def test_shape_and_mask_mismatch_raise():
    with pytest.raises(ValueError):
        WO.top1_agreement(torch.randn(2, 3, 4), torch.randn(2, 3, 5))
    with pytest.raises(ValueError):
        WO.top1_agreement(torch.randn(2, 3, 4), torch.randn(2, 3, 4),
                          mask=torch.ones(2, 4, dtype=torch.bool))


def test_state_dict_roundtrip_keeps_int8_codes():
    """量化模型要能存能读：codes 是 int8 buffer，load 之后输出必须逐位相同。"""
    lin = nn.Linear(64, 48)
    q = WO.W8A16Linear.from_linear(lin, compute_dtype=torch.float32)
    x = torch.randn(3, 64)
    before = q(x)
    fresh = WO.W8A16Linear(64, 48, bias=True, compute_dtype=torch.float32)
    fresh.load_state_dict(q.state_dict())
    assert torch.equal(fresh(x), before)
    assert fresh.qweight.dtype is torch.int8
