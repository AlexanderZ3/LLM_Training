"""R4：GRPO 的组内 advantage 归一化在 fp16 下的灾难性抵消。

这个实验在 V100 上测的是什么
----------------------------
GRPO 不用 critic，advantage 直接来自同一个 prompt 的 G 个采样：

    A_i = (r_i - mean(r)) / (std(r) + eps)

这一行有两个数值陷阱，在 V100 的 fp16 下都真实存在：

1. **灾难性抵消**：r_i - mean(r)。当 reward 的**量级**远大于**组内差异**时，
   两个几乎相等的 fp16 数相减，有效位数被吃光。
   fp16 在 1.0 附近的 ulp 是 2^-10 ≈ 9.8e-4；如果 reward 都在 1.0 附近、
   组内差异只有 1e-3，那么 r_i - mean 只剩下不到 1 位有效数字，
   **组内排序都会被打乱**——而 GRPO 全靠这个排序提供学习信号。

2. **std = 0**：组内所有采样拿到同一个 reward 时 std 精确为 0，
   A 全为 0（被 eps 兜住），这一组对梯度没有贡献。
   规则型 reward（值域只有几个离散值）下这个比例可能很高。
   zero_std_group_ratio 是 Day 5 预注册的 INCONCLUSIVE 判据之一。

不能测什么
----------
- CPU 的 fp16 是软件模拟。**排序被打乱的比例**这个结论在本机与 V100 上
  量级应当一致（都是 IEEE half 的舍入），但不保证逐位相同。
- 不能从这里推出「应该用多大的 beta / clip_eps」——那要真实 rollout。

结论的形状
----------
这不是「fp16 不能用」，而是「**advantage 归一化必须在 fp32 里做**」。
它的计算量相对于前向反向可以忽略，没有任何理由省这一步。
真正要在 fp16 里算的是矩阵乘，不是这种 O(G) 的规约。
"""

from __future__ import annotations

from typing import Any, Dict, Sequence

import torch

__all__ = [
    "group_mean_std",
    "group_advantages",
    "advantage_report",
    "cancellation_case",
    "format_advantage_report",
]


def group_mean_std(
    rewards: torch.Tensor,
    dtype: torch.dtype = torch.float32,
    unbiased: bool = True,
) -> Dict[str, torch.Tensor]:
    """按最后一维（组内）算 mean / std，**显式在 dtype 里做每一步**。

    不用 torch.std：它对 half 的 CPU/CUDA 实现可能内部提升到 fp32，
    那样就测不到 fp16 的真实行为了。手写 sum / 平方和，
    让每一次加法都真的发生在 dtype 里——这是本实验的观测对象。
    """
    x = rewards.to(dtype)
    n = x.shape[-1]
    if n < 2 and unbiased:
        raise ValueError("unbiased=True 需要组大小 >= 2，收到 {}".format(n))
    mean = x.sum(dim=-1, keepdim=True) / n
    d = x - mean
    denom = (n - 1) if unbiased else n
    var = (d * d).sum(dim=-1, keepdim=True) / denom
    std = var.sqrt()
    return {"mean": mean, "std": std, "centered": d}


def group_advantages(
    rewards: torch.Tensor,
    eps: float = 1e-4,
    dtype: torch.dtype = torch.float32,
    unbiased: bool = True,
) -> torch.Tensor:
    """(r - mean) / (std + eps)，形状与 rewards 相同，dtype 为入参 dtype。

    rewards 形状 [n_groups, group_size]。eps 加在 std 上（不是 var 上），
    与 TRL 的 GRPOTrainer 一致；它同时承担「std=0 时不除零」的职责，
    所以 eps 的取值直接决定了 zero-std 组会被放大到多少——
    eps 太小时那些组的 A 会炸，太大时正常组的 A 被系统性压小。
    """
    if rewards.dim() != 2:
        raise ValueError("rewards 必须是 [n_groups, group_size]，收到 {}".format(tuple(rewards.shape)))
    st = group_mean_std(rewards, dtype=dtype, unbiased=unbiased)
    eps_t = torch.tensor(eps, dtype=dtype, device=rewards.device)
    return st["centered"] / (st["std"] + eps_t)


def _rank_match(a: torch.Tensor, b: torch.Tensor) -> float:
    """组内排序一致率。按 argsort 的逐位置比较——这比相关系数更贴近 GRPO 真实用法：

    advantage 的绝对值影响步长，**符号和相对次序**决定了哪个采样被推上去。
    次序错了，学习信号的方向就错了。
    """
    ra = a.argsort(dim=-1)
    rb = b.argsort(dim=-1)
    return float((ra == rb).to(torch.float32).mean())


def advantage_report(
    rewards: torch.Tensor,
    eps: float = 1e-4,
    dtypes: Sequence[torch.dtype] = (torch.float32, torch.float16),
    reference_dtype: torch.dtype = torch.float64,
    unbiased: bool = True,
) -> Dict[str, Any]:
    """对每个 dtype 算一遍 advantage，与 reference_dtype 对照。

    主指标：
      zero_std_group_ratio  该 dtype 下 std 精确为 0 的组占比
      rank_match            组内排序与参考一致的位置占比（1.0 = 完全一致）
      max_abs_diff          与参考的 advantage 最大绝对差
    """
    if rewards.dim() != 2:
        raise ValueError("rewards 必须是 [n_groups, group_size]")
    ref = group_advantages(rewards, eps=eps, dtype=reference_dtype, unbiased=unbiased)
    ref32 = ref.to(torch.float32)
    ref_std = group_mean_std(rewards, dtype=reference_dtype, unbiased=unbiased)["std"]

    out: Dict[str, Any] = {
        "n_groups": int(rewards.shape[0]),
        "group_size": int(rewards.shape[1]),
        "eps": float(eps),
        "reference_dtype": str(reference_dtype).replace("torch.", ""),
        "reward_mean_abs": float(rewards.to(torch.float32).abs().mean()),
        "reward_within_group_std_mean": float(ref_std.to(torch.float32).mean()),
        "by_dtype": {},
    }
    for dt in dtypes:
        name = str(dt).replace("torch.", "")
        st = group_mean_std(rewards, dtype=dt, unbiased=unbiased)
        adv = group_advantages(rewards, eps=eps, dtype=dt, unbiased=unbiased)
        adv32 = adv.to(torch.float32)
        zero_std = (st["std"].to(torch.float32) == 0).reshape(-1)
        out["by_dtype"][name] = {
            "eps_of_dtype": float(torch.finfo(dt).eps),
            "zero_std_group_ratio": float(zero_std.to(torch.float32).mean()),
            "max_abs_advantage": float(adv32.abs().max()),
            "mean_abs_advantage": float(adv32.abs().mean()),
            "n_distinct_advantage": int(torch.unique(adv32).numel()),
            "rank_match_vs_reference": _rank_match(adv32, ref32),
            "max_abs_diff_vs_reference": float((adv32 - ref32).abs().max()),
            "n_nonfinite": int((~torch.isfinite(adv32)).sum()),
        }
    return out


def cancellation_case(
    n_groups: int = 64,
    group_size: int = 8,
    base: float = 1.0,
    spread: float = 1e-3,
    seed: int = 0,
    device: str = "cpu",
) -> torch.Tensor:
    """构造「reward 量级 = base、组内差异 = spread」的灾难性抵消场景。

    默认 base=1.0、spread=1e-3：fp16 在 1.0 附近的 ulp 是 9.8e-4，
    也就是说组内的全部差异只值**一个 ulp**。这是 R4 的主场景。

    把 spread 调到 1e-4 会更极端：绝大多数组的 fp16 std 直接变成 0。
    把 base 调到 0 则完全没有问题——**问题不在 fp16，在 base**。
    这一点是 R4 想让人记住的：偏置项能把一个健康的量变成噪声。
    """
    if spread <= 0:
        raise ValueError("spread 必须为正，收到 {}".format(spread))
    g = torch.Generator(device="cpu").manual_seed(seed)
    noise = torch.randn((n_groups, group_size), generator=g)
    return (base + noise * spread).to(device)


def format_advantage_report(report: Dict[str, Any]) -> str:
    lines = [
        "组数 {}  组大小 {}  eps {:g}  reward 量级 {:.4g}  组内 std {:.4g}".format(
            report["n_groups"],
            report["group_size"],
            report["eps"],
            report["reward_mean_abs"],
            report["reward_within_group_std_mean"],
        ),
        "{:<10} {:>13} {:>13} {:>11} {:>12} {:>13}".format(
            "dtype", "zero_std%", "max|A|", "distinct", "rank_match", "max_diff"
        ),
    ]
    lines.append("-" * len(lines[-1]))
    for name, e in report["by_dtype"].items():
        lines.append(
            "{:<10} {:>12.2f}% {:>13.4g} {:>11d} {:>12.4f} {:>13.4g}".format(
                name,
                100.0 * e["zero_std_group_ratio"],
                e["max_abs_advantage"],
                e["n_distinct_advantage"],
                e["rank_match_vs_reference"],
                e["max_abs_diff_vs_reference"],
            )
        )
    return "\n".join(lines)
