"""Q6：激活 outlier 统计 + 幅度迁移（SmoothQuant 式）。

这个实验在 V100 上测的是什么
----------------------------
1. **激活 outlier 是不是「结构性」的**：同样那几个 channel 在不同 batch 上反复
   成为 outlier，还是每个 batch 随机换一批。跨 batch 重合率就是这个问题的答案。
   重合率高 => outlier 绑在 channel 上 => 可以用一组固定的 per-channel 系数
   把幅度从激活挪到权重（这就是幅度迁移能成立的前提）。
   重合率低 => 只能在线求 scale，离线迁移无效。
2. **迁移强度 alpha 的取值**：s_j = max|X_j|^alpha / max|W_j|^(1-alpha)。
   alpha=0 完全不迁移，alpha=1 把激活压平但把权重撑爆。中间有一个最优点。

这两条是纯数值性质，在 V100 上跑和在 CPU 上跑结论一致；
唯一必须在 V100 上做的是**用真实模型、真实数据的激活**来统计，
因为「outlier 结构性」是模型自己的性质，随机张量造不出来。

不能测什么
----------
- 不能测 W8A8 的速度。V100 没有 INT8 Tensor Core，SmoothQuant 的全部速度收益
  来自 INT8 GEMM，在这台机器上一分钱都拿不到。这里做 Q6 只有一个目的：
  搞清楚**激活量化难在哪**，为将来换到 sm80+ 的机器做准备。
- 不能把这里的 alpha 直接搬到别的模型：alpha 是逐层、逐模型的。

数学不变量（本模块最重要的一条）
--------------------------------
(X / s) @ (W * s).T == X @ W.T           s 沿输入维广播
迁移在 fp32 下是**恒等变换**，它不改变模型的数学，只改变「误差落在谁头上」。
所以任何 alpha 下，未量化时的输出必须与原始输出一致（到 fp32 舍入）。
这条不变量是 Q6 的第一条单测；它不成立就说明 s 的广播维度搞反了。
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

import torch

from .quantizers import quantize_dequantize, relative_error

__all__ = [
    "DEFAULT_ALPHAS",
    "channel_amax",
    "activation_outlier_stats",
    "overlap_rate",
    "smoothing_scales",
    "apply_smoothing",
    "alpha_sweep",
    "format_alpha_sweep",
]

DEFAULT_ALPHAS: Tuple[float, ...] = (0.0, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)


def _as_tokens(x: torch.Tensor) -> torch.Tensor:
    """把 [..., C] 摊成 [N, C]。激活统计一律按「token x channel」看。"""
    if x.dim() < 2:
        raise ValueError("激活至少要 2 维 [..., C]，收到 {}".format(tuple(x.shape)))
    return x.reshape(-1, x.shape[-1]).to(torch.float32)


def channel_amax(x: torch.Tensor) -> torch.Tensor:
    """每个 channel 的 max|x|，fp32，形状 [C]。"""
    return _as_tokens(x).abs().amax(dim=0)


def activation_outlier_stats(x: torch.Tensor, sigma_mult: float = 6.0) -> Dict[str, Any]:
    """按 |x| > sigma_mult * sigma 判定 outlier channel。

    sigma 取**全张量**的标准差（不是逐 channel），因为要回答的问题是
    「哪些 channel 相对于整层的典型幅度异常大」。逐 channel 归一化会把
    outlier 自己归一掉，正好丢掉要找的东西。

    outlier_channels 的判据是「该 channel 的 max|x| 超过阈值」，
    只要该 channel 存在一个极端 token 就算——这与 per-tensor 激活量化的
    痛点一致：per-tensor scale 由全局 amax 决定，一个 token 就能毁掉整层。
    """
    t = _as_tokens(x)
    sigma = float(t.std())
    thr = sigma_mult * sigma
    amax = t.abs().amax(dim=0)
    idx = torch.nonzero(amax > thr, as_tuple=False).reshape(-1)
    n_elem_out = int((t.abs() > thr).sum())
    return {
        "n_tokens": int(t.shape[0]),
        "n_channels": int(t.shape[1]),
        "sigma": sigma,
        "sigma_mult": float(sigma_mult),
        "threshold": thr,
        "n_outlier_channels": int(idx.numel()),
        "outlier_channel_ratio": float(idx.numel()) / float(t.shape[1]),
        "outlier_element_ratio": n_elem_out / float(t.numel()),
        "channel_amax_max": float(amax.max()),
        "channel_amax_median": float(amax.median()),
        "amax_over_median": (
            float(amax.max() / amax.median()) if float(amax.median()) > 0 else float("inf")
        ),
        "outlier_channels": idx,
    }


def overlap_rate(idx_a: torch.Tensor, idx_b: torch.Tensor) -> Dict[str, float]:
    """两个 batch 的 outlier channel 集合的重合率。

    jaccard      |A∩B| / |A∪B|，对称，主指标。
    coverage_a   |A∩B| / |A|，「上一个 batch 的 outlier 有多少在这个 batch 还是 outlier」。
    两个都接近 1 => outlier 是结构性的，离线迁移成立。
    """
    a = set(int(i) for i in idx_a.reshape(-1).tolist())
    b = set(int(i) for i in idx_b.reshape(-1).tolist())
    inter = len(a & b)
    union = len(a | b)
    return {
        "n_a": float(len(a)),
        "n_b": float(len(b)),
        "n_intersection": float(inter),
        "jaccard": (inter / union) if union else 1.0,
        "coverage_a_in_b": (inter / len(a)) if a else 1.0,
        "coverage_b_in_a": (inter / len(b)) if b else 1.0,
    }


def smoothing_scales(
    act_amax: torch.Tensor,
    weight_amax: torch.Tensor,
    alpha: float = 0.5,
    eps: float = 1e-5,
) -> torch.Tensor:
    """s_j = max|X_j|^alpha / max|W_j|^(1-alpha)，形状 [C]，fp32。

    alpha=0 => s=1/max|W_j|^1 ... 注意这仍然不是「不迁移」。
    要「完全不迁移」得让 s 恒为 1，所以这里对 alpha=0 与 alpha=1 做了归一化处理：
    先按公式算，再整体除以几何均值，使 s 的量级稳定在 1 附近。
    不归一化时 s 的绝对量级会随层的幅度漂移，per-tensor 激活量化的对比就不公平了。
    """
    if not (0.0 <= alpha <= 1.0):
        raise ValueError("alpha 必须落在 [0, 1]，收到 {}".format(alpha))
    a = act_amax.detach().to(torch.float32).clamp_min(eps)
    w = weight_amax.detach().to(torch.float32).clamp_min(eps)
    if a.shape != w.shape:
        raise ValueError(
            "act_amax {} 与 weight_amax {} 形状不一致".format(tuple(a.shape), tuple(w.shape))
        )
    s = a.pow(alpha) / w.pow(1.0 - alpha)
    # 几何均值归一化：log 域取均值再指数回去，避免大 C 时连乘溢出。
    gm = torch.exp(torch.log(s.clamp_min(eps)).mean())
    return (s / gm).clamp_min(eps)


def apply_smoothing(
    x: torch.Tensor, w: torch.Tensor, s: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """返回 (X/s, W*s)。W 形状 [out, in]，s 沿 in 维广播。

    维度约定一旦搞反，不变量单测会立刻失败——这就是那条单测存在的理由。
    """
    if w.dim() != 2:
        raise ValueError("权重必须是 [out, in]，收到 {}".format(tuple(w.shape)))
    if s.numel() != w.shape[1]:
        raise ValueError(
            "s 的长度 {} 必须等于 in_features {}".format(int(s.numel()), int(w.shape[1]))
        )
    if x.shape[-1] != w.shape[1]:
        raise ValueError(
            "激活最后一维 {} 必须等于 in_features {}".format(int(x.shape[-1]), int(w.shape[1]))
        )
    s32 = s.detach().to(torch.float32).reshape(-1)
    x_s = x.to(torch.float32) / s32
    w_s = w.to(torch.float32) * s32.reshape(1, -1)
    return x_s, w_s


def alpha_sweep(
    x: torch.Tensor,
    w: torch.Tensor,
    alphas: Sequence[float] = DEFAULT_ALPHAS,
    act_granularity: str = "per_tensor",
    weight_granularity: str = "per_channel",
    group_size: int = 128,
    num_bits: int = 8,
    eps: float = 1e-5,
) -> List[Dict[str, Any]]:
    """扫 alpha，看端到端输出误差。

    量化方案固定为「激活 per-tensor 对称 + 权重 per-channel 对称」——
    这是 W8A8 里最难的那一档，也是 outlier 最致命的那一档。
    参考输出是 fp32 的 X @ W.T（未量化），所有 alpha 共用同一个参考。
    """
    x32 = x.to(torch.float32)
    w32 = w.to(torch.float32)
    ref = x32.reshape(-1, x32.shape[-1]) @ w32.t()
    a_amax = channel_amax(x32)
    w_amax = w32.abs().amax(dim=0)

    rows: List[Dict[str, Any]] = []
    for alpha in alphas:
        s = smoothing_scales(a_amax, w_amax, alpha=float(alpha), eps=eps)
        x_s, w_s = apply_smoothing(x32, w32, s)
        exact = x_s.reshape(-1, x_s.shape[-1]) @ w_s.t()

        x_q = quantize_dequantize(
            x_s, granularity=act_granularity, group_size=group_size, num_bits=num_bits
        )
        w_q = quantize_dequantize(
            w_s,
            granularity=weight_granularity,
            axis=0,
            group_size=group_size,
            num_bits=num_bits,
        )
        out_q = x_q.reshape(-1, x_q.shape[-1]) @ w_q.t()
        rows.append(
            {
                "alpha": float(alpha),
                "s_max": float(s.max()),
                "s_min": float(s.min()),
                "act_rel_err": relative_error(x_s, x_q),
                "weight_rel_err": relative_error(w_s, w_q),
                "output_rel_err": relative_error(ref, out_q),
                "identity_rel_err": relative_error(ref, exact),
                "act_amax_after": float(channel_amax(x_s).max()),
                "weight_amax_after": float(w_s.abs().max()),
            }
        )
    return rows


def format_alpha_sweep(rows: Sequence[Dict[str, Any]], title: str = "") -> str:
    header = "{:>6} {:>11} {:>11} {:>12} {:>12} {:>12}".format(
        "alpha", "act_err", "w_err", "out_err", "act_amax", "w_amax"
    )
    lines = []
    if title:
        lines.append(title)
    lines.append(header)
    lines.append("-" * len(header))
    for r in rows:
        lines.append(
            "{:>6.2f} {:>11.5f} {:>11.5f} {:>12.5f} {:>12.4f} {:>12.4f}".format(
                r["alpha"],
                r["act_rel_err"],
                r["weight_rel_err"],
                r["output_rel_err"],
                r["act_amax_after"],
                r["weight_amax_after"],
            )
        )
    return "\n".join(lines)
