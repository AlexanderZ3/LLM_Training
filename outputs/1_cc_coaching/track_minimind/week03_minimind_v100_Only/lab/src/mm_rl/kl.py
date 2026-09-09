"""R2：三个 KL 估计量在 fp16 与 fp32 下的行为。

这个实验在 V100 上测的是什么
----------------------------
GRPO / PPO 的 KL 惩罚项通常不用真 KL，而用一个基于采样的估计量。三个常见写法
（记 r = p_new / p_ref，log_r = logp_new - logp_ref）：

    k1 = -log_r                      无偏，但方差大，**可以为负**
    k2 = (log_r)^2 / 2               恒非负，有偏（对 KL(p_ref || p_new) 的二阶近似）
    k3 = (r - 1) - log_r             无偏且恒非负（Schulman 的低方差估计量）

在 V100 上问的问题是：**fp16 把 log_r 算坏之后，这三个量各自坏到什么程度。**
关键在于 log_r 是两个几乎相等的数相减（policy 刚从 ref 初始化时 log_r ≈ 0），
是典型的灾难性抵消场景。fp16 的 eps = 9.8e-4：当 |logp| 在 5 左右时，
logp 的 ulp 约 3.9e-3，两数相减后 log_r 的**绝对**误差就是这个量级，
而 log_r 本身可能只有 1e-3——信噪比小于 1。

k2 和 k3 都是 log_r 的二次量，误差被平方放大；k1 是一次量，相对更耐受。
这就是为什么「换个 KL 估计量」在 fp32 上是口味问题，在 fp16 上是数值问题。

不能测什么
----------
- CPU 的 fp16 是软件模拟，误差的**分布**与 V100 上不同。本机验的是公式与
  统计口径，量级要在 V100 上重测。
- 这里不评价哪个估计量「更好」——那取决于 beta 的取值和 reward 的尺度，
  是 Day 5 的实验，不是本模块的结论。

恒等式（单测断言的依据）
------------------------
k3 = expm1(log_r) - log_r >= 0 对所有实数 log_r 成立（e^x - 1 >= x）。
用 torch.expm1 而不是 torch.exp(x) - 1：后者在 |x| 很小时会把有效数字全丢光，
得到的 k3 可能是负数。**k3 出现负值 = 实现写错了，不是精度问题。**
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

import torch

__all__ = [
    "ESTIMATORS",
    "log_ratio",
    "kl_k1",
    "kl_k2",
    "kl_k3",
    "kl_estimators",
    "estimator_stats",
    "precision_report",
    "format_precision_report",
]

ESTIMATORS = ("k1", "k2", "k3")


def log_ratio(
    logp_new: torch.Tensor,
    logp_ref: torch.Tensor,
    compute_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """log r = logp_new - logp_ref，在 compute_dtype 里做减法。

    这一句就是灾难性抵消发生的地方。把 dtype 做成参数，是为了让
    「fp16 减法」和「fp32 减法」能在同一段代码里对比。
    """
    return logp_new.to(compute_dtype) - logp_ref.to(compute_dtype)


def kl_k1(log_r: torch.Tensor) -> torch.Tensor:
    return -log_r


def kl_k2(log_r: torch.Tensor) -> torch.Tensor:
    return 0.5 * log_r * log_r


def kl_k3(log_r: torch.Tensor) -> torch.Tensor:
    """(r - 1) - log_r，用 expm1 保证小 |log_r| 下的精度与非负性。"""
    return torch.expm1(log_r) - log_r


def kl_estimators(log_r: torch.Tensor) -> Dict[str, torch.Tensor]:
    return {"k1": kl_k1(log_r), "k2": kl_k2(log_r), "k3": kl_k3(log_r)}


def estimator_stats(
    values: torch.Tensor, mask: Optional[torch.Tensor] = None
) -> Dict[str, float]:
    """统计量一律在 fp32 里算：用 fp16 去统计 fp16 的误差没有意义。"""
    v = values.to(torch.float32)
    if mask is not None:
        v = v[mask.to(torch.bool)]
    n = int(v.numel())
    if n == 0:
        raise ValueError("estimator_stats 收到 0 个有效元素")
    return {
        "n": float(n),
        "mean": float(v.mean()),
        "std": float(v.std(unbiased=True)) if n > 1 else 0.0,
        "max_abs": float(v.abs().max()),
        "frac_negative": float((v < 0).to(torch.float32).mean()),
        "n_nonfinite": float((~torch.isfinite(v)).sum()),
    }


def precision_report(
    logp_new: torch.Tensor,
    logp_ref: torch.Tensor,
    dtypes: Sequence[torch.dtype] = (torch.float32, torch.float16),
    mask: Optional[torch.Tensor] = None,
    reference_dtype: torch.dtype = torch.float64,
) -> Dict[str, Any]:
    """对每个 dtype 算一遍三个估计量，并与 reference_dtype 的结果比相对偏差。

    参考用 fp64 而不是 fp32：这样 fp32 自己的偏差也能被量出来。
    在 V100 上跑时 fp64 很慢（1:2 的 FP64:FP32 比例其实是 V100 的强项，
    但仍比 fp32 慢），所以这个函数只对**一个 batch 的采样**跑，不进训练循环。
    """
    ref_log_r = log_ratio(logp_new, logp_ref, compute_dtype=reference_dtype)
    ref_vals = kl_estimators(ref_log_r)

    out: Dict[str, Any] = {
        "reference_dtype": str(reference_dtype).replace("torch.", ""),
        "n_tokens": int(mask.sum()) if mask is not None else int(logp_new.numel()),
        "reference": {k: estimator_stats(v, mask) for k, v in ref_vals.items()},
        "by_dtype": {},
    }
    for dt in dtypes:
        name = str(dt).replace("torch.", "")
        lr = log_ratio(logp_new, logp_ref, compute_dtype=dt)
        vals = kl_estimators(lr)
        entry: Dict[str, Any] = {
            "eps": float(torch.finfo(dt).eps),
            "log_ratio": estimator_stats(lr, mask),
        }
        for k in ESTIMATORS:
            st = estimator_stats(vals[k], mask)
            ref_mean = out["reference"][k]["mean"]
            st["rel_bias_vs_reference"] = (
                abs(st["mean"] - ref_mean) / abs(ref_mean) if ref_mean != 0.0 else float("inf")
            )
            diff = (vals[k].to(torch.float32) - ref_vals[k].to(torch.float32)).abs()
            if mask is not None:
                diff = diff[mask.to(torch.bool)]
            st["max_abs_err_vs_reference"] = float(diff.max())
            entry[k] = st
        out["by_dtype"][name] = entry
    return out


def format_precision_report(report: Dict[str, Any]) -> str:
    lines = [
        "KL 估计量 x dtype（参考 = {}，{} 个有效 token）".format(
            report["reference_dtype"], report["n_tokens"]
        ),
        "{:<10} {:<4} {:>12} {:>12} {:>13} {:>11}".format(
            "dtype", "est", "mean", "std", "rel_bias", "frac_neg"
        ),
    ]
    lines.append("-" * len(lines[-1]))
    for k in ESTIMATORS:
        st = report["reference"][k]
        lines.append(
            "{:<10} {:<4} {:>12.5e} {:>12.5e} {:>13} {:>10.3f}".format(
                report["reference_dtype"], k, st["mean"], st["std"], "-", st["frac_negative"]
            )
        )
    for name, entry in report["by_dtype"].items():
        for k in ESTIMATORS:
            st = entry[k]
            lines.append(
                "{:<10} {:<4} {:>12.5e} {:>12.5e} {:>13.3e} {:>10.3f}".format(
                    name, k, st["mean"], st["std"], st["rel_bias_vs_reference"],
                    st["frac_negative"],
                )
            )
    return "\n".join(lines)
