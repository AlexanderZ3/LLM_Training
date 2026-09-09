"""mm_quant.quantizers 的单测：分块、往返、粒度单调性、axis 用对没有。

这些断言全部在 CPU 上成立，与 GPU 无关——量化误差是算术性质。
GPU 相关的只有 fake-quant kernel（见 test_quant_qat.py），那部分有 skipif。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from mm_quant import quantizers as Q  # noqa: E402
from mm_quant.error_budget import channelwise_relative_error  # noqa: E402


def _heterogeneous_weight(rows: int = 32, cols: int = 512, seed: int = 0) -> torch.Tensor:
    """逐输出通道尺度跨 2 个数量级 + 每行若干随机 outlier。

    两种结构分别对应 per-channel 和 per-group 各自能救的那一类误差，
    所以三条粒度曲线才会明显分开。纯 randn 是分不开的——那也是一条结论。
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    w = torch.randn((rows, cols), generator=g) * torch.logspace(0.0, 2.0, rows).reshape(-1, 1)
    idx = torch.randint(0, cols, (rows, 4), generator=g)
    w.scatter_(1, idx, torch.randn((rows, 4), generator=g) * 20.0)
    return w


@pytest.mark.parametrize(
    "shape,granularity,kwargs",
    [
        ((7, 130), "per_tensor", {}),
        ((7, 130), "per_channel", {"axis": 0}),
        ((7, 130), "per_channel", {"axis": 1}),
        ((7, 130), "per_group", {"group_size": 128}),   # 130 不整除 128，走 padding 分支
        ((4, 5, 96), "per_group", {"group_size": 32}),  # 3 维
        ((4, 5, 96), "per_channel", {"axis": 2}),
    ],
)
def test_blocking_roundtrip_is_exact(shape, granularity, kwargs):
    """摊平再摊回来必须逐位等于原张量，包括 per_group 需要 padding 的情况。"""
    x = torch.randn(shape)
    b = Q.Blocking(shape, granularity, **kwargs)
    assert torch.equal(b.from_blocks(b.to_blocks(x)), x)


def test_blocking_rejects_wrong_shape():
    b = Q.Blocking((4, 8), "per_channel", axis=0)
    with pytest.raises(ValueError):
        b.to_blocks(torch.randn(4, 9))


def test_code_range_symmetric_drops_one_code():
    assert Q.code_range(8, "symmetric") == (-127, 127)
    assert Q.code_range(8, "asymmetric") == (-128, 127)
    assert Q.code_range(4, "symmetric") == (-7, 7)


def test_codes_are_int8_and_within_range():
    w = _heterogeneous_weight()
    qt = Q.quantize_tensor(w, granularity="per_channel")
    assert qt.codes.dtype is torch.int8
    assert int(qt.codes.min()) >= -127
    assert int(qt.codes.max()) <= 127


def test_relative_error_decreases_monotonically_with_granularity():
    """per_tensor >= per_channel >= per_group。这是 Q1 的主结论。

    不等式的方向来自 scale 的定义：块越小，块内 amax 越小，
    对称量化的逐元素误差上界 scale/2 就越小。
    """
    w = _heterogeneous_weight()
    errs = {
        g: Q.relative_error(w, Q.quantize_dequantize(w, granularity=g, group_size=128))
        for g in ("per_tensor", "per_channel", "per_group")
    }
    assert errs["per_tensor"] > errs["per_channel"] > errs["per_group"]
    # 差距要足够大才说明分块真的起了作用，而不是浮点噪声
    assert errs["per_tensor"] / errs["per_channel"] > 1.5
    assert errs["per_channel"] / errs["per_group"] > 1.05


def test_per_channel_axis_must_match_the_axis_the_scale_varies_along():
    """axis 用错会被逐通道误差立刻抓到，但**整体 Frobenius 误差抓不到**。

    构造：尺度只沿 dim 0 变化（跨 2 个数量级），沿 dim 1 是同分布的。
    axis=0（对）  每个通道各自一个 scale，逐通道相对误差都很小。
    axis=1（错）  每一列的 amax 被最大的那一行主导，小尺度行被压成粗糙的几个码，
                  逐通道相对误差飙到 0.5 以上。
    而 Frobenius 相对误差两者接近——因为范数被大尺度行主导。
    这条测试同时验了「axis 用对了」和「为什么必须看逐通道指标」。
    """
    g = torch.Generator(device="cpu").manual_seed(3)
    rows, cols = 24, 384
    w = torch.randn((rows, cols), generator=g) * torch.logspace(0.0, 3.0, rows).reshape(-1, 1)

    right = Q.quantize_dequantize(w, granularity="per_channel", axis=0)
    wrong = Q.quantize_dequantize(w, granularity="per_channel", axis=1)

    ch_right = channelwise_relative_error(w, right, axis=0)
    ch_wrong = channelwise_relative_error(w, wrong, axis=0)

    assert ch_right["max"] < 0.05
    assert ch_wrong["max"] > 0.5
    assert ch_wrong["max"] / ch_right["max"] > 20.0

    fro_right = Q.relative_error(w, right)
    fro_wrong = Q.relative_error(w, wrong)
    assert abs(fro_right - fro_wrong) / fro_right < 0.5, (
        "整体 Frobenius 误差在两种 axis 下接近，正是这条测试要证明的："
        "只看一个总的相对误差抓不到 axis 用反"
    )


def test_asymmetric_beats_symmetric_on_nonnegative_data():
    """ReLU 之后的激活是单侧分布，非对称方案能多拿一个 bit。"""
    g = torch.Generator(device="cpu").manual_seed(11)
    x = torch.rand((16, 256), generator=g) * 3.0
    sym = Q.relative_error(x, Q.quantize_dequantize(x, scheme="symmetric", granularity="per_channel"))
    asym = Q.relative_error(x, Q.quantize_dequantize(x, scheme="asymmetric", granularity="per_channel"))
    assert asym < sym
    assert sym / asym > 1.5


def test_scale_is_computed_in_fp32_even_for_fp16_input():
    """输入是 fp16 也要在 fp32 域算 scale。

    构造一个 amax 极小的块：amax=6e-5（fp16 的最小正规数附近）。
    在 fp16 域算 amax/127 会下溢成 0（或次正规数），量化结果全是 0；
    在 fp32 域算就没事。这条测试守的是 quantizers.compute_qparams 里的 .float()。
    """
    w16 = (torch.randn(4, 256) * 1e-5).to(torch.float16)
    assert float(w16.abs().max()) < 1e-3
    qp = Q.compute_qparams(w16.reshape(1, -1), scheme="symmetric")
    assert qp.scale.dtype is torch.float32
    assert float(qp.scale.min()) > 0.0

    w_hat = Q.quantize_dequantize(w16, granularity="per_channel")
    assert float(w_hat.abs().max()) > 0.0, "整块被量化成 0 说明 scale 下溢了"
    assert Q.relative_error(w16, w_hat) < 0.05

    # 对照：如果在 fp16 域做同一个除法，scale 会掉到 0
    naive_scale_fp16 = (w16.abs().max() / torch.tensor(127.0, dtype=torch.float16))
    assert float(naive_scale_fp16) == 0.0 or float(naive_scale_fp16) < 1e-6


def test_zero_tensor_does_not_produce_nan_or_inf():
    w = torch.zeros(8, 64)
    out = Q.quantize_dequantize(w, granularity="per_group", group_size=32)
    assert torch.isfinite(out).all()
    assert float(out.abs().max()) == 0.0


def test_bytes_report_matches_hand_arithmetic():
    """字节账要能手算对上：int8 码 + 每块 4 B 的 fp32 scale，分母取 fp16。"""
    w = torch.randn(64, 512)
    qt = Q.quantize_tensor(w, granularity="per_channel", axis=0)
    rep = qt.bytes_report(reference_dtype=torch.float16)
    assert rep["code_bytes"] == 64 * 512
    assert rep["scale_bytes"] == 64 * 4
    assert rep["reference_bytes"] == 64 * 512 * 2
    expected = (64 * 512 + 64 * 4) / (64 * 512 * 2)
    assert rep["ratio_vs_reference"] == pytest.approx(expected, rel=1e-9)
    assert 0.5 < rep["ratio_vs_reference"] < 0.51


def test_per_group_scale_overhead_is_visible_in_the_ledger():
    """per_group(g=128) 的 scale 开销约 4/128 = 3.1% 的码字节，账上要看得见。"""
    w = torch.randn(64, 512)
    qt = Q.quantize_tensor(w, granularity="per_group", group_size=128)
    rep = qt.bytes_report(reference_dtype=torch.float16)
    assert rep["scale_share"] == pytest.approx(4.0 / (128.0 + 4.0), rel=0.02)
    assert rep["ratio_vs_reference"] < 0.53


def test_scale_storage_bytes_agrees_with_actual_quantization():
    shape = (48, 320)
    for gran, kw in (("per_tensor", {}), ("per_channel", {"axis": 0}),
                     ("per_group", {"group_size": 64})):
        est = Q.scale_storage_bytes(shape, gran, **kw)
        qt = Q.quantize_tensor(torch.randn(shape), granularity=gran, **kw)
        assert est["code_bytes"] == qt.code_bytes()
        assert est["scale_bytes"] == qt.qparams.nbytes()


def test_clip_ratio_reduces_rounding_error_and_creates_clipping_error():
    w = torch.randn(16, 1024)
    full = Q.quantize_dequantize(w, granularity="per_channel", num_bits=4, clip_ratio=1.0)
    clipped = Q.quantize_dequantize(w, granularity="per_channel", num_bits=4, clip_ratio=0.7)
    # 截断之后，落在阈值内的元素误差变小
    thr = w.abs().amax(dim=1, keepdim=True) * 0.7
    inside = w.abs() <= thr
    assert float(((w - clipped)[inside] ** 2).mean()) < float(((w - full)[inside] ** 2).mean())
    # 而落在阈值外的元素误差变大
    outside = ~inside
    assert float(((w - clipped)[outside] ** 2).mean()) > float(((w - full)[outside] ** 2).mean())


def test_invalid_arguments_raise_with_actionable_messages():
    with pytest.raises(ValueError):
        Q.code_range(9, "symmetric")
    with pytest.raises(ValueError):
        Q.Blocking((4, 4), "per_block")
    with pytest.raises(ValueError):
        Q.compute_qparams(torch.randn(4, 4), clip_ratio=0.0)
    with pytest.raises(ValueError):
        Q.compute_qparams(torch.randn(4, 4, 4))
