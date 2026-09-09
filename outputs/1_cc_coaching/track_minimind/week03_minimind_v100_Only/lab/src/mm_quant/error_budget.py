"""Q1 + Q2：量化的误差预算。

这个实验在 V100 上测的是什么
----------------------------
Q1：给定一组权重，**换粒度**能把相对误差压到多少，代价是多少字节的 scale。
Q2：给定一层权重，**截断阈值 alpha** 怎么在 clipping 误差和 rounding 误差之间取平衡。

两个都是纯算术，和 GPU 型号无关；在 V100 上跑和在 CPU 上跑得到同一张表。
真正只能在 V100 上做的是**在真实 MiniMind 权重上跑**这张表——因为权重的
outlier 结构是模型自己的性质，随机张量替代不了。所以本模块的函数全部接受
「一组 (name, tensor)」，本机用随机张量验逻辑，V100 上换成真权重出结论。

不能测什么
----------
- 不能从这张表推出任何速度结论。V100 没有 INT8 Tensor Core。
- 不能从「权重量化误差小」推出「模型精度不掉」：误差会被后续层放大，
  端到端结论必须由 Q4 的逐 token 一致率和独立 eval 给出。

两个核心量
----------
outlier_ratio = max|w| / mean|w|
    量的是「这块权重里最大的那个值，比典型值大多少倍」。对称量化的 scale 由
    max|w| 独占决定，所以这个比值直接就是「有多少码位被一个离群点浪费掉了」。
    比值 20 意味着典型权重只用到 127/20 ≈ 6 个码位，等价于 3 bit 不到。

MSE 分解
    err_total  = w - Q(w)
    err_clip   = w - clamp(w, -T, T)          T = alpha * max|w|
    err_round  = clamp(w, -T, T) - Q(w)
    两项支撑集几乎不相交（被截断的元素，其码正好落在端点上，rounding 误差近 0），
    所以 mse_total ≈ mse_clip + mse_round，交叉项 cross 打印出来供核对。
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Sequence, Tuple

import torch

from .quantizers import (
    Blocking,
    quantize_dequantize,
    quantize_tensor,
    mse,
    relative_error,
)

__all__ = [
    "DEFAULT_GRANULARITIES",
    "DEFAULT_ALPHAS",
    "outlier_ratio",
    "channelwise_relative_error",
    "granularity_row",
    "error_table",
    "clipping_sweep",
    "best_alpha",
    "format_error_table",
    "format_alpha_sweep",
]

DEFAULT_GRANULARITIES: Tuple[str, ...] = ("per_tensor", "per_channel", "per_group")
DEFAULT_ALPHAS: Tuple[float, ...] = (
    1.0,
    0.95,
    0.9,
    0.85,
    0.8,
    0.75,
    0.7,
    0.6,
    0.5,
    0.4,
    0.3,
)


def outlier_ratio(w: torch.Tensor) -> float:
    """max|w| / mean|w|。恒 >= 1；越大说明动态范围被少数元素撑开得越厉害。"""
    a = w.detach().to(torch.float32).abs()
    m = float(a.mean())
    if m == 0.0:
        return float("inf")
    return float(a.max()) / m


def channelwise_relative_error(
    w: torch.Tensor, w_hat: torch.Tensor, axis: int = 0
) -> Dict[str, float]:
    """**逐通道**的相对误差，返回 mean / max / min。

    为什么要有这个而不是只看整体 Frobenius 相对误差：
    Frobenius 范数被幅度最大的那些通道主导。一个权重里如果各输出通道的尺度
    差 1000 倍，那么把 scale 的轴用错（axis 选反）之后，小尺度通道的相对误差
    会飙到 50%+，但整体 Frobenius 相对误差**几乎不动**——因为那些通道对范数
    没有贡献。用错轴这种 bug 只有逐通道看才抓得到。
    """
    a = w.detach().to(torch.float32)
    b = w_hat.detach().to(torch.float32)
    if a.shape != b.shape:
        raise ValueError("形状不一致：{} vs {}".format(tuple(a.shape), tuple(b.shape)))
    ax = axis % a.dim()
    a2 = a.movedim(ax, 0).reshape(a.shape[ax], -1)
    b2 = b.movedim(ax, 0).reshape(a.shape[ax], -1)
    num = torch.linalg.vector_norm(a2 - b2, dim=1)
    den = torch.linalg.vector_norm(a2, dim=1)
    safe = den > 0
    if int(safe.sum()) == 0:
        return {"mean": 0.0, "max": 0.0, "min": 0.0, "n_channels": float(a.shape[ax])}
    rel = torch.zeros_like(num)
    rel[safe] = num[safe] / den[safe]
    return {
        "mean": float(rel[safe].mean()),
        "max": float(rel[safe].max()),
        "min": float(rel[safe].min()),
        "n_channels": float(a.shape[ax]),
    }


def granularity_row(
    w: torch.Tensor,
    granularity: str,
    scheme: str = "symmetric",
    axis: int = 0,
    group_size: int = 128,
    num_bits: int = 8,
    reference_dtype: torch.dtype = torch.float16,
) -> Dict[str, Any]:
    """单个 (张量, 粒度) 组合的一行：相对误差 + 存储账。"""
    qt = quantize_tensor(
        w,
        scheme=scheme,
        granularity=granularity,
        axis=axis,
        group_size=group_size,
        num_bits=num_bits,
    )
    w_hat = qt.dequantize()
    rep = qt.bytes_report(reference_dtype=reference_dtype)
    per_ch = channelwise_relative_error(w, w_hat, axis=axis)
    return {
        "granularity": granularity,
        "num_blocks": qt.blocking.num_blocks,
        "rel_err": relative_error(w, w_hat),
        "rel_err_channel_mean": per_ch["mean"],
        "rel_err_channel_max": per_ch["max"],
        "mse": mse(w, w_hat),
        "code_bytes": rep["code_bytes"],
        "scale_bytes": rep["scale_bytes"],
        "payload_bytes": rep["payload_bytes"],
        "scale_share": rep["scale_share"],
        "ratio_vs_reference": rep["ratio_vs_reference"],
    }


def error_table(
    named_weights: Iterable[Tuple[str, torch.Tensor]],
    granularities: Sequence[str] = DEFAULT_GRANULARITIES,
    scheme: str = "symmetric",
    axis: int = 0,
    group_size: int = 128,
    num_bits: int = 8,
    reference_dtype: torch.dtype = torch.float16,
) -> List[Dict[str, Any]]:
    """Q1 的主表：逐层 x 逐粒度。

    每层一条记录，里面带 outlier_ratio 和 per-granularity 的子表。
    named_weights 可以来自 model.named_parameters()，也可以是构造张量。
    """
    rows: List[Dict[str, Any]] = []
    for name, w in named_weights:
        if w.dim() < 1 or w.numel() == 0:
            continue
        entry: Dict[str, Any] = {
            "name": name,
            "shape": list(w.shape),
            "numel": int(w.numel()),
            "outlier_ratio": outlier_ratio(w),
            "granularity": {},
        }
        for g in granularities:
            entry["granularity"][g] = granularity_row(
                w,
                g,
                scheme=scheme,
                axis=axis,
                group_size=group_size,
                num_bits=num_bits,
                reference_dtype=reference_dtype,
            )
        rows.append(entry)
    return rows


def clipping_sweep(
    w: torch.Tensor,
    alphas: Sequence[float] = DEFAULT_ALPHAS,
    granularity: str = "per_channel",
    scheme: str = "symmetric",
    axis: int = 0,
    group_size: int = 128,
    num_bits: int = 8,
) -> List[Dict[str, Any]]:
    """Q2：扫 alpha，把 MSE 拆成 clipping 项与 rounding 项。

    截断阈值 T 是**按块**算的：T_block = alpha * max|w_block|，和 scale 用的是
    同一个 amax。这样 alpha 的含义在三种粒度下一致。
    """
    if scheme != "symmetric":
        raise ValueError(
            "clipping_sweep 只对 symmetric 有明确的 clip/round 分解；"
            "非对称方案的截断要分左右两侧，不在本实验范围内。"
        )
    blocking = Blocking(w.shape, granularity, axis=axis, group_size=group_size)
    wb = blocking.to_blocks(w.detach().to(torch.float32))
    amax = wb.abs().amax(dim=1, keepdim=True)

    out: List[Dict[str, Any]] = []
    for alpha in alphas:
        thr = amax * float(alpha)
        wb_clip = torch.clamp(wb, -thr, thr)
        w_hat = quantize_dequantize(
            w,
            scheme=scheme,
            granularity=granularity,
            axis=axis,
            group_size=group_size,
            num_bits=num_bits,
            clip_ratio=float(alpha),
        )
        wb_hat = blocking.to_blocks(w_hat)

        e_total = wb - wb_hat
        e_clip = wb - wb_clip
        e_round = wb_clip - wb_hat
        mse_total = float((e_total ** 2).mean())
        mse_clip = float((e_clip ** 2).mean())
        mse_round = float((e_round ** 2).mean())
        clipped = (wb.abs() > thr)
        denom = float((wb ** 2).sum())
        out.append(
            {
                "alpha": float(alpha),
                "mse_total": mse_total,
                "mse_clip": mse_clip,
                "mse_round": mse_round,
                "cross": mse_total - mse_clip - mse_round,
                "rel_err": (
                    float(torch.linalg.vector_norm(e_total) / (denom ** 0.5))
                    if denom > 0
                    else 0.0
                ),
                "clipped_frac": float(clipped.to(torch.float32).mean()),
            }
        )
    return out


def best_alpha(sweep: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """从扫描结果里挑 mse_total 最小的那一行。"""
    if not sweep:
        raise ValueError("best_alpha 收到空的扫描结果")
    best = min(sweep, key=lambda r: r["mse_total"])
    baseline = next((r for r in sweep if r["alpha"] == 1.0), None)
    gain = (
        (baseline["mse_total"] / best["mse_total"])
        if baseline is not None and best["mse_total"] > 0
        else float("nan")
    )
    return {
        "alpha": best["alpha"],
        "mse_total": best["mse_total"],
        "mse_reduction_vs_alpha1": gain,
        "clipped_frac": best["clipped_frac"],
    }


def format_error_table(rows: Sequence[Dict[str, Any]]) -> str:
    """把 error_table 的结果打成等宽表。只含形状与数值，不含权重内容。"""
    header = (
        "{:<24} {:>13} {:>8}  {:<12} {:>9} {:>10} {:>8} {:>8}".format(
            "layer", "shape", "outlier", "granularity", "rel_err", "worst_ch", "scale%", "vs fp16"
        )
    )
    lines = [header, "-" * len(header)]
    for r in rows:
        shape = "x".join(str(s) for s in r["shape"])
        first = True
        for g, gr in r["granularity"].items():
            lines.append(
                "{:<24} {:>13} {:>8}  {:<12} {:>9.5f} {:>10.5f} {:>7.2f}% {:>8.3f}".format(
                    r["name"] if first else "",
                    shape if first else "",
                    "{:.1f}".format(r["outlier_ratio"]) if first else "",
                    g,
                    gr["rel_err"],
                    gr["rel_err_channel_max"],
                    100.0 * gr["scale_share"],
                    gr["ratio_vs_reference"],
                )
            )
            first = False
    return "\n".join(lines)


def format_alpha_sweep(sweep: Sequence[Dict[str, Any]], title: str = "") -> str:
    header = "{:>6} {:>12} {:>12} {:>12} {:>11} {:>10}".format(
        "alpha", "mse_total", "mse_clip", "mse_round", "cross", "clipped%"
    )
    lines = []
    if title:
        lines.append(title)
    lines.append(header)
    lines.append("-" * len(header))
    for r in sweep:
        lines.append(
            "{:>6.2f} {:>12.3e} {:>12.3e} {:>12.3e} {:>11.2e} {:>9.2f}%".format(
                r["alpha"],
                r["mse_total"],
                r["mse_clip"],
                r["mse_round"],
                r["cross"],
                100.0 * r["clipped_frac"],
            )
        )
    return "\n".join(lines)
