"""教学版专家并行（Expert Parallel）：MiniMind ``MOEFeedForward`` 的 EP 改写。

对照关系
--------
MiniMind 的 MoE 层（model/model_minimind.py L148–L176）里**没有 MoEGate 类**：
router 就是 ``MOEFeedForward.gate = nn.Linear(hidden_size, num_experts, bias=False)``，
forward 里 ``softmax -> topk -> 逐 expert index_add_``，全部专家都在本卡上。
本文件把"逐 expert index_add_"换成六步 EP 流水，专家分散到 ep_size 个 rank 上。

六步 forward（与 01_FOUNDATIONS.md 2.4 节逐条对应）
--------------------------------------------------
1. router：``scores = softmax(gate(x_flat))``，``topk`` 取 k 个专家与权重。
2. 计数：``counts_mat[e]`` = 本 rank 要发给全局专家 e 的 token 数（长度 E）；
   先做一次 ``all_to_all_single`` 交换这张表，得到"我会收到多少"。
   **这一步必须先于数据交换**：接收缓冲区的长度只能由对端告诉我。
3. permutation：``perm = argsort(expert_of, stable=True)``。按全局专家 id 排序，
   由于 ``dest_rank = expert_of // experts_per_rank`` 单调，排序后同时满足
   "按目标 rank 分块"和"块内按本地专家分组"两个条件，所以只需要一次元数据交换。
4. dispatch：``all_to_all_single(recv, send, output_split_sizes, input_split_sizes)``。
5. 本地 expert：接收缓冲区按 ``recv_counts_mat`` 切成 (来源 rank × 本地专家) 的块，
   同一本地专家跨来源 rank 的块拼在一起做一次 FFN。
6. combine + unpermute：逆向 ``all_to_all_single``（两个 split 列表对调），
   再 ``y.index_add_(0, token_of[perm], back * weight)``。

必须处理的三种情况
------------------
* **某 expert 收到 0 token**：``all_to_all_single`` 的 split size 允许为 0；
  但是本地专家如果一个 token 都没收到，它的参数就不在计算图里，DDP/FSDP 的
  梯度归约会因为"部分 rank 有梯度、部分没有"而 hang。解决办法与 MiniMind L166
  一致：训练态下加一项 ``0 * sum(p.sum() for p in expert.parameters())``。
* **token 数不整除**：split sizes 本来就是不等长列表，无需 padding。
  唯一的硬约束是不变量 I2：rank r 的 ``input_split_sizes[j]`` 必须等于
  rank j 的 ``output_split_sizes[r]``——由第 2 步的元数据交换保证。
* **capacity factor**：``--capacity-factor 0`` = dropless（不丢 token）；
  ``>0`` 时每个专家最多收 ``ceil(cf * N * k / E)`` 个 token，超出的按到达顺序丢弃，
  被丢的 token 该专家那一路输出为 0（其余专家的贡献仍保留）。

EP 约束
-------
``ep_size <= num_experts`` 且 ``num_experts % ep_size == 0``。
8 卡 EP=8 需要把 config 里的 ``num_experts`` 改成 8（默认只有 4）。

只用 torch 2.1 API：``dist.all_to_all_single``、``dist.new_group``、``dist.barrier``、
``torch.cuda.Event``。没有 DTensor、没有 device_mesh。
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mm_dist import common as C  # noqa: E402

# 计时阶段名；--time-breakdown 打开时每个 forward 都会填满这张表
PHASES = (
    "router",
    "topk",
    "count_exchange",
    "permute",
    "wait_before_dispatch",
    "dispatch_a2a",
    "expert_compute",
    "combine_a2a",
    "unpermute",
)


# ---------------------------------------------------------------------------
# 计时
# ---------------------------------------------------------------------------


class PhaseTimer:
    """按阶段累计耗时。CUDA 用 ``torch.cuda.Event``，CPU 退化到 ``perf_counter``。

    CUDA 上必须用 Event：kernel 是异步下发的，``perf_counter`` 只能量到 launch 时间，
    量不到 all_to_all 真正的等待。Event 的代价是每次 ``elapsed_time`` 都要
    ``synchronize``，所以只在 ``--time-breakdown`` 打开时启用。
    """

    def __init__(self, enabled: bool = False, device: str = "cpu") -> None:
        self.enabled = bool(enabled)
        self.device = str(device)
        self.use_cuda = self.enabled and self.device.startswith("cuda") \
            and torch.cuda.is_available()
        self.totals: Dict[str, float] = {p: 0.0 for p in PHASES}
        self.calls: Dict[str, int] = {p: 0 for p in PHASES}
        self._open: Optional[Tuple[str, Any]] = None

    def start(self, phase: str) -> None:
        if not self.enabled:
            return
        if phase not in self.totals:
            self.totals[phase] = 0.0
            self.calls[phase] = 0
        if self.use_cuda:
            ev = torch.cuda.Event(enable_timing=True)
            ev.record()
            self._open = (phase, ev)
        else:
            self._open = (phase, time.perf_counter())

    def stop(self) -> None:
        if not self.enabled or self._open is None:
            return
        phase, marker = self._open
        if self.use_cuda:
            end = torch.cuda.Event(enable_timing=True)
            end.record()
            torch.cuda.synchronize()
            ms = float(marker.elapsed_time(end))
        else:
            ms = (time.perf_counter() - float(marker)) * 1000.0
        self.totals[phase] += ms
        self.calls[phase] += 1
        self._open = None

    def reset(self) -> None:
        for k in list(self.totals):
            self.totals[k] = 0.0
            self.calls[k] = 0
        self._open = None

    def report(self) -> Dict[str, float]:
        return {k: round(v, 4) for k, v in self.totals.items()}


class _Phase:
    """``with timer.phase("dispatch_a2a"):`` 的语法糖。"""

    def __init__(self, timer: PhaseTimer, name: str) -> None:
        self.timer = timer
        self.name = name

    def __enter__(self):
        self.timer.start(self.name)
        return self

    def __exit__(self, *exc):
        self.timer.stop()
        return False


def phase(timer: PhaseTimer, name: str) -> _Phase:
    return _Phase(timer, name)


# ---------------------------------------------------------------------------
# 可反向传播的 all_to_all_single
# ---------------------------------------------------------------------------


class AllToAllSingle(torch.autograd.Function):
    """``dist.all_to_all_single`` 的自动求导包装。

    **这是必须自己写的一层。** ``torch.distributed.all_to_all_single`` 是就地写
    输出缓冲区的通信原语，autograd 完全看不见它：不包一层的话，反向传播会在
    dispatch 处静默断掉，专家权重照样有梯度（本地 FFN 那一段），但 router 和
    下游 attention 拿到的梯度是错的，loss 还会正常下降——这是最难发现的一类 bug。

    反向就是把两个 split 列表对调再做一次 all_to_all：
    前向把我的 token 送到别人那里，反向就把别人算出的梯度送回来。
    所以一个 MoE 层一步的通信次数是 4 次 all_to_all（dispatch/combine 各自的前反向）
    加 1 次计数交换，与 01_FOUNDATIONS.md 3.1 节的表一致。
    """

    @staticmethod
    def forward(ctx, send: torch.Tensor, out_splits: List[int],
                in_splits: List[int], group: Any) -> torch.Tensor:
        ctx.out_splits = list(out_splits)
        ctx.in_splits = list(in_splits)
        ctx.group = group
        recv = send.new_empty((int(sum(out_splits)),) + tuple(send.shape[1:]))
        dist.all_to_all_single(recv, send.contiguous(), list(out_splits),
                               list(in_splits), group=group)
        return recv

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        grad_input = grad_output.new_empty(
            (int(sum(ctx.in_splits)),) + tuple(grad_output.shape[1:]))
        dist.all_to_all_single(grad_input, grad_output.contiguous(),
                               ctx.in_splits, ctx.out_splits, group=ctx.group)
        return grad_input, None, None, None


# ---------------------------------------------------------------------------
# EP 分组
# ---------------------------------------------------------------------------


def validate_ep(ep_size: int, num_experts: int, world_size: int) -> None:
    """EP 的三条硬约束，任何一条不满足都在建组前报错。"""
    if ep_size < 1:
        raise ValueError("ep_size 必须 >= 1，收到 " + str(ep_size))
    if ep_size > num_experts:
        raise ValueError(
            "ep_size=" + str(ep_size) + " > num_experts=" + str(num_experts)
            + "：每个 rank 至少要持有 1 个专家。8 卡 EP=8 需要把 config 的 "
              "model.num_experts 改成 8。")
    if num_experts % ep_size != 0:
        raise ValueError(
            "num_experts=" + str(num_experts) + " 不能被 ep_size=" + str(ep_size)
            + " 整除；本教学版不做不均匀切分。")
    if world_size % ep_size != 0:
        raise ValueError(
            "world_size=" + str(world_size) + " 不能被 ep_size=" + str(ep_size)
            + " 整除；EP 组要由连续的 ep_size 个 rank 组成。")


def make_ep_groups(ep_size: int, world_size: int, rank: int):
    """按连续 rank 切 EP 组，返回 ``(my_group, my_ep_rank, all_group_ranks)``。

    ``dist.new_group`` 要求**所有**进程按相同顺序调用，即使自己不是成员
    （torch 2.1 distributed 文档）——所以这里对每个组都调用一次，再挑出自己那个。
    """
    all_ranks = [list(range(g * ep_size, (g + 1) * ep_size))
                 for g in range(world_size // ep_size)]
    if ep_size == 1 or not (dist.is_available() and dist.is_initialized()):
        return None, 0 if ep_size == 1 else rank % ep_size, all_ranks
    my_group = None
    for ranks in all_ranks:
        grp = dist.new_group(ranks=ranks)
        if rank in ranks:
            my_group = grp
    return my_group, rank % ep_size, all_ranks


# ---------------------------------------------------------------------------
# EP MoE
# ---------------------------------------------------------------------------


class EPMoEFeedForward(nn.Module):
    """MiniMind ``MOEFeedForward`` 的专家并行版本。

    参数放置：``gate`` 在每个 rank 上都是完整副本（router 很小，复制比通信便宜）；
    ``experts`` 只保留本 rank 负责的 ``num_experts // ep_size`` 个。
    ``local_expert_ids`` 记录本地专家对应的全局 id，方便 router_stats 对账。
    """

    def __init__(self, config, gate: nn.Linear, local_experts: nn.ModuleList,
                 ep_size: int = 1, ep_rank: int = 0, ep_group: Any = None,
                 capacity_factor: float = 0.0, time_breakdown: bool = False,
                 device: str = "cpu") -> None:
        super().__init__()
        self.config = config
        self.num_experts = int(config.num_experts)
        self.top_k = int(config.num_experts_per_tok)
        self.norm_topk_prob = bool(getattr(config, "norm_topk_prob", True))
        self.aux_coef = float(getattr(config, "router_aux_loss_coef", 0.0))
        validate_ep(ep_size, self.num_experts, max(ep_size, 1))
        self.ep_size = int(ep_size)
        self.ep_rank = int(ep_rank)
        self.ep_group = ep_group
        self.experts_per_rank = self.num_experts // self.ep_size
        self.capacity_factor = float(capacity_factor)
        self.gate = gate
        self.experts = local_experts
        if len(self.experts) != self.experts_per_rank:
            raise ValueError(
                "local_experts 数量 " + str(len(self.experts)) + " != experts_per_rank "
                + str(self.experts_per_rank))
        self.local_expert_ids = list(range(self.ep_rank * self.experts_per_rank,
                                           (self.ep_rank + 1) * self.experts_per_rank))
        self.timer = PhaseTimer(enabled=time_breakdown, device=device)
        self.aux_loss = torch.zeros(())
        self.last_stats: Dict[str, Any] = {}

    # -- 构造 ------------------------------------------------------------

    @classmethod
    def from_dense(cls, dense, ep_size: int = 1, ep_rank: int = 0,
                   ep_group: Any = None, capacity_factor: float = 0.0,
                   time_breakdown: bool = False,
                   device: str = "cpu") -> "EPMoEFeedForward":
        """从一个完整的 ``MOEFeedForward`` 切出本 rank 的 EP 视图。

        专家模块是**引用**而不是拷贝，所以 EP 版与 dense 版共享同一份权重，
        :func:`equivalent_to_dense` 的比较才是纯粹的算法比较。
        """
        config = dense.config
        num_experts = int(config.num_experts)
        validate_ep(ep_size, num_experts, max(ep_size, 1))
        per = num_experts // ep_size
        local = nn.ModuleList([dense.experts[ep_rank * per + i] for i in range(per)])
        return cls(config, dense.gate, local, ep_size=ep_size, ep_rank=ep_rank,
                   ep_group=ep_group, capacity_factor=capacity_factor,
                   time_breakdown=time_breakdown, device=device)

    # -- 通信原语 --------------------------------------------------------

    def _exchange_counts(self, counts_mat: torch.Tensor) -> torch.Tensor:
        """交换 ``[E]`` 的计数表，得到"每个来源 rank 发给我哪几个本地专家多少 token"。

        等长 all_to_all：每个 rank 收发 ``experts_per_rank`` 个 int64。
        返回值 shape 仍是 ``[E]``，语义变成
        ``recv[j*experts_per_rank + e] = 来自 rank j、给我第 e 个本地专家的数量``。
        """
        if self.ep_size == 1 or self.ep_group is None and not _dist_ready():
            return counts_mat.clone()
        recv = torch.empty_like(counts_mat)
        dist.all_to_all_single(recv, counts_mat.contiguous(), group=self.ep_group)
        return recv

    def _a2a(self, send: torch.Tensor, out_splits: List[int],
             in_splits: List[int]) -> torch.Tensor:
        """带 split sizes 的 all_to_all_single，附不变量 I2 的本地断言。"""
        if int(sum(in_splits)) != int(send.shape[0]):
            raise RuntimeError(
                "input_split_sizes 之和 " + str(int(sum(in_splits)))
                + " != 发送张量行数 " + str(int(send.shape[0]))
                + "（不变量 I2 的本地部分被破坏，继续跑会 hang 而不是报错）")
        if self.ep_size == 1 or (self.ep_group is None and not _dist_ready()):
            return send
        return AllToAllSingle.apply(send, list(out_splits), list(in_splits),
                                    self.ep_group)

    # -- forward ---------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        squeeze_back = False
        if x.dim() == 2:
            x = x.unsqueeze(0)
            squeeze_back = True
        bsz, seq_len, hidden = x.shape
        x_flat = x.reshape(-1, hidden)
        n_tokens = x_flat.shape[0]
        t = self.timer

        # 1. router
        with phase(t, "router"):
            logits = self.gate(x_flat)
            scores = F.softmax(logits, dim=-1)

        # 2. topk
        with phase(t, "topk"):
            topk_weight, topk_idx = torch.topk(scores, k=self.top_k, dim=-1,
                                               sorted=False)
            if self.norm_topk_prob:
                topk_weight = topk_weight / (topk_weight.sum(dim=-1, keepdim=True)
                                             + 1e-20)
            token_of = torch.arange(n_tokens, device=x_flat.device
                                    ).repeat_interleave(self.top_k)
            expert_of = topk_idx.reshape(-1)
            weight_of = topk_weight.reshape(-1)

        # 2b. capacity（0 = dropless）
        dropped = 0
        if self.capacity_factor > 0.0:
            keep, dropped = _apply_capacity(expert_of, self.num_experts,
                                            self.capacity_factor, n_tokens,
                                            self.top_k)
            token_of = token_of[keep]
            expert_of = expert_of[keep]
            weight_of = weight_of[keep]

        # 3. 计数 + 元数据交换
        with phase(t, "count_exchange"):
            counts_mat = torch.bincount(expert_of, minlength=self.num_experts
                                        ).to(torch.int64)
            recv_counts_mat = self._exchange_counts(counts_mat)
            in_splits = counts_mat.view(self.ep_size, self.experts_per_rank
                                        ).sum(dim=1).tolist()
            out_splits = recv_counts_mat.view(self.ep_size, self.experts_per_rank
                                              ).sum(dim=1).tolist()

        # 4. permutation
        with phase(t, "permute"):
            perm = torch.argsort(expert_of, stable=True)
            send_tokens = token_of[perm]
            send_weight = weight_of[perm]
            send_x = x_flat.index_select(0, send_tokens)

        # 5. dispatch
        if t.enabled and _dist_ready() and self.ep_size > 1:
            with phase(t, "wait_before_dispatch"):
                dist.barrier(group=self.ep_group)
        with phase(t, "dispatch_a2a"):
            recv_x = self._a2a(send_x, out_splits, in_splits)

        # 6. 本地 expert
        with phase(t, "expert_compute"):
            recv_expert = _local_expert_ids(recv_counts_mat, self.ep_size,
                                            self.experts_per_rank, recv_x.device)
            out_x = torch.zeros_like(recv_x)
            per_expert_tokens: List[int] = []
            for local_id, expert in enumerate(self.experts):
                sel = (recv_expert == local_id).nonzero().flatten()
                per_expert_tokens.append(int(sel.numel()))
                if sel.numel() > 0:
                    out_x.index_copy_(0, sel, expert(recv_x.index_select(0, sel)
                                                     ).to(out_x.dtype))
                elif self.training:
                    # 0 token 的专家：不进计算图会让梯度归约参与者不一致而 hang。
                    # 与 MiniMind MOEFeedForward L166 的 0*sum(p) 同样的用意。
                    out_x = out_x + 0.0 * sum(p.sum() for p in expert.parameters())

        # 7. combine（两个 split 列表对调）
        with phase(t, "combine_a2a"):
            back = self._a2a(out_x, in_splits, out_splits)

        # 8. unpermute + 加权
        with phase(t, "unpermute"):
            y = torch.zeros_like(x_flat)
            if back.shape[0] > 0:
                y.index_add_(0, send_tokens,
                             (back * send_weight.unsqueeze(1)).to(y.dtype))

        # aux loss：公式与 MiniMind L169-171 完全一致
        if self.training and self.aux_coef > 0:
            self.aux_loss = aux_loss_value(scores, topk_idx, self.num_experts,
                                           self.aux_coef)
        else:
            self.aux_loss = scores.new_zeros(1).squeeze()

        self.last_stats = {
            "n_tokens": int(n_tokens),
            "top_k": self.top_k,
            "ep_size": self.ep_size,
            "ep_rank": self.ep_rank,
            "capacity_factor": self.capacity_factor,
            "dropped_pairs": int(dropped),
            "input_split_sizes": [int(v) for v in in_splits],
            "output_split_sizes": [int(v) for v in out_splits],
            "recv_tokens": int(recv_x.shape[0]),
            "per_local_expert_tokens": per_expert_tokens,
            "global_expert_counts": [int(v) for v in counts_mat.tolist()],
            "aux_loss": float(self.aux_loss.detach()),
            "time_ms": t.report() if t.enabled else {},
        }

        out = y.view(bsz, seq_len, hidden)
        return out.squeeze(0) if squeeze_back else out


def _dist_ready() -> bool:
    return dist.is_available() and dist.is_initialized()


def _apply_capacity(expert_of: torch.Tensor, num_experts: int,
                    capacity_factor: float, n_tokens: int,
                    top_k: int) -> Tuple[torch.Tensor, int]:
    """每个专家最多保留 ``ceil(cf * N*k / E)`` 个 (token, expert) 对。

    返回 ``(保留位置的索引, 被丢弃的对数)``。丢弃按到达顺序（token 序），
    与 Switch Transformer 的 "first-come, first-served" 一致。
    """
    capacity = int(math.ceil(capacity_factor * n_tokens * top_k / max(num_experts, 1)))
    capacity = max(capacity, 1)
    order = torch.argsort(expert_of, stable=True)
    sorted_experts = expert_of[order]
    counts = torch.bincount(sorted_experts, minlength=num_experts)
    starts = torch.cumsum(counts, dim=0) - counts
    rank_in_expert = torch.arange(sorted_experts.numel(),
                                  device=expert_of.device) - starts[sorted_experts]
    keep_sorted = rank_in_expert < capacity
    keep = order[keep_sorted]
    keep, _ = torch.sort(keep)
    dropped = int(expert_of.numel() - keep.numel())
    return keep, dropped


def _local_expert_ids(recv_counts_mat: torch.Tensor, ep_size: int,
                      experts_per_rank: int, device) -> torch.Tensor:
    """接收缓冲区里每一行属于哪个**本地**专家。

    接收顺序是 (来源 rank j 外层, 本地专家 e 内层)，因为发送端按全局专家 id 排序，
    而全局专家 id = 目标 rank * experts_per_rank + 本地专家 id。
    """
    pattern = torch.arange(experts_per_rank, device=device).repeat(ep_size)
    return torch.repeat_interleave(pattern, recv_counts_mat.to(device).flatten())


def aux_loss_value(scores: torch.Tensor, topk_idx: torch.Tensor,
                   num_experts: int, coef: float) -> torch.Tensor:
    """MiniMind 的 aux loss：``(load * scores.mean(0)).sum() * E * coef``。

    ``load = one_hot(topk_idx, E).float().mean(0)``（对 token 维求均值，top_k>1 时
    再对 k 维求均值——MiniMind 用的是 ``mean(0)``，即对第 0 维；本函数完全照抄）。
    完全均衡时 ``load = scores.mean(0) = 1/E``，乘 E 后 = 1/E * E = 1（乘 coef 前）。
    完全坍缩到一个专家时该值趋近 E。
    """
    load = F.one_hot(topk_idx, num_experts).float().mean(0)
    return (load * scores.mean(0)).sum() * num_experts * coef


# ---------------------------------------------------------------------------
# 等价性检查
# ---------------------------------------------------------------------------


def equivalent_to_dense(dense, x: torch.Tensor, tol: float = 1e-5,
                        capacity_factor: float = 0.0) -> Dict[str, Any]:
    """``ep_size=1`` 时 EP 版与原始 ``MOEFeedForward`` 的逐元素比较。

    ep_size=1 不发生任何集合通信（``_a2a`` 直接返回输入），所以这条检查在单进程、
    无进程组的环境里也能跑——它验证的是 permute/unpermute/加权合并这条链路，
    不是通信本身。通信正确性由 ``tests/test_ep_cpu.py`` 的 2 进程往返测试负责。
    """
    was_training = dense.training
    dense.eval()
    ep = EPMoEFeedForward.from_dense(dense, ep_size=1, ep_rank=0, ep_group=None,
                                     capacity_factor=capacity_factor)
    ep.eval()
    with torch.no_grad():
        y_dense = dense(x)
        y_ep = ep(x)
    if was_training:
        dense.train()
    diff = (y_dense - y_ep).abs()
    max_abs = float(diff.max()) if diff.numel() else 0.0
    denom = float(y_dense.abs().max()) if y_dense.numel() else 1.0
    return {
        "max_abs_diff": max_abs,
        "rel_diff": max_abs / max(denom, 1e-12),
        "tol": tol,
        "pass": bool(max_abs < tol),
        "dtype": str(y_dense.dtype),
        "shape": list(y_dense.shape),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_ep_layer_from_config(args, device: str, ep_size: int, ep_rank: int,
                               ep_group) -> Tuple[Any, EPMoEFeedForward]:
    """按 config 造一个 MOEFeedForward 与它的 EP 视图（共享权重）。"""
    mm = C.import_minimind(args.minimind_root)
    cfg = C.load_config(args.config)
    model_kwargs = dict(cfg["model"])
    if not model_kwargs.get("use_moe", False):
        raise SystemExit(
            "config " + args.config + " 的 model.use_moe=false；"
            "EP 实验请用 configs/moe_tiny_cpu.json 或 configs/v100_moe.json。")
    hidden_size = model_kwargs.pop("hidden_size")
    num_hidden_layers = model_kwargs.pop("num_hidden_layers")
    model_kwargs.pop("use_moe")
    if args.num_experts is not None:
        model_kwargs["num_experts"] = int(args.num_experts)
    lm_config = mm.MiniMindConfig(hidden_size=hidden_size,
                                  num_hidden_layers=num_hidden_layers,
                                  use_moe=True, **model_kwargs)
    dense = mm.MOEFeedForward(lm_config).to(device)
    ep = EPMoEFeedForward.from_dense(dense, ep_size=ep_size, ep_rank=ep_rank,
                                     ep_group=ep_group,
                                     capacity_factor=args.capacity_factor,
                                     time_breakdown=bool(args.time_breakdown),
                                     device=device)
    return dense, ep


def cmd_equiv(args: argparse.Namespace) -> int:
    device = "cpu" if args.force_cpu or not torch.cuda.is_available() else "cuda:0"
    C.set_seed(args.seed, per_rank=False)
    dense, _ = build_ep_layer_from_config(args, device, 1, 0, None)
    x = torch.randn(args.batch, args.seq_len, dense.config.hidden_size, device=device)
    result = equivalent_to_dense(dense, x, tol=args.tol,
                                 capacity_factor=args.capacity_factor)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True), flush=True)
    print("[equiv] ep_size=1 vs MOEFeedForward -> %s  max|diff|=%.3e tol=%.1e"
          % ("PASS" if result["pass"] else "FAIL", result["max_abs_diff"], args.tol),
          flush=True)
    return 0 if result["pass"] else 1


def cmd_run(args: argparse.Namespace) -> int:
    info = C.init_dist(backend=args.backend, timeout_s=args.nccl_timeout_s,
                       force_cpu=args.force_cpu)
    device = "cpu" if args.force_cpu else info.device
    C.assert_dtype_supported(args.dtype, device)
    C.set_seed(args.seed, per_rank=False)
    ep_size = int(args.ep_size)
    if ep_size > info.world_size:
        raise SystemExit(
            "--ep-size " + str(ep_size) + " > world_size " + str(info.world_size)
            + "；EP 组必须由真实存在的 rank 组成。")
    group, ep_rank, all_ranks = make_ep_groups(ep_size, info.world_size, info.rank)
    dense, ep = build_ep_layer_from_config(args, device, ep_size, ep_rank, group)
    ep.train()

    logger = C.JsonlLogger(args.out_dir, args.log_name or ("ep" + str(ep_size)),
                           rank=info.rank)
    logger.write({
        "step": 0,
        "event": "header",
        "runner": "ep_moe",
        "ep_size": ep_size,
        "ep_rank": ep_rank,
        "ep_groups": all_ranks,
        "world_size": info.world_size,
        "num_experts": ep.num_experts,
        "experts_per_rank": ep.experts_per_rank,
        "local_expert_ids": ep.local_expert_ids,
        "capacity_factor": args.capacity_factor,
        "dtype": args.dtype,
        "device_type": "cuda" if str(device).startswith("cuda") else "cpu",
        "env": C.env_summary(),
    })

    hidden = dense.config.hidden_size
    steps = max(int(args.max_steps), 1)
    for step in range(1, steps + 1):
        ep.timer.reset()
        x = torch.randn(args.batch, args.seq_len, hidden, device=device)
        t0 = time.perf_counter()
        with C.autocast_context(C.assert_dtype_supported(args.dtype, device), device):
            y = ep(x)
        loss = y.float().pow(2).mean() + ep.aux_loss.float()
        loss.backward()
        for p in ep.parameters():
            p.grad = None
        dt = time.perf_counter() - t0
        rec = {"step": step, "step_time_s": dt,
               "peak_mem_mb": C.peak_mem_mb(device),
               "loss_proxy": float(loss.detach())}
        rec.update(ep.last_stats)
        logger.write(rec)
        if step % max(int(args.log_interval), 1) == 0 or step == steps:
            msg = ("[ep step %d/%d] rank=%d ep_rank=%d recv=%d per_expert=%s "
                   "dropped=%d t=%.4fs"
                   % (step, steps, info.rank, ep_rank, rec["recv_tokens"],
                      rec["per_local_expert_tokens"], rec["dropped_pairs"], dt))
            if args.time_breakdown:
                msg += " time_ms=" + json.dumps(rec["time_ms"], sort_keys=True)
            print(msg, flush=True)
    logger.close()
    C.cleanup_dist(info)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="教学版专家并行（EP）：六步 all_to_all dispatch/combine",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    C.add_common_args(p)
    p.add_argument("command", choices=["run", "equiv"],
                   help="run=按 --ep-size 跑若干步并记录分解耗时；equiv=ep_size=1 数值等价检查")
    p.add_argument("--ep-size", type=int, default=1, help="EP 组大小（<=num_experts 且整除）")
    p.add_argument("--num-experts", type=int, default=None,
                   help="覆盖 config 里的 model.num_experts（EP=8 时需要设成 8）")
    p.add_argument("--capacity-factor", type=float, default=0.0,
                   help="0=dropless；>0 时每专家上限 ceil(cf*N*k/E)")
    p.add_argument("--time-breakdown", action="store_true",
                   help="记录 router/topk/计数/permute/等待/dispatch/expert/combine/unpermute 耗时")
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--seq-len", type=int, default=64)
    p.add_argument("--tol", type=float, default=1e-5, help="equiv 的容差")
    p.add_argument("--log-interval", type=int, default=1)
    p.add_argument("--log-name", type=str, default=None)
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.config:
        raise SystemExit("必须给 --config configs/moe_tiny_cpu.json（或 v100_moe.json）")
    if args.command == "equiv":
        return cmd_equiv(args)
    return cmd_run(args)


if __name__ == "__main__":
    raise SystemExit(main())
