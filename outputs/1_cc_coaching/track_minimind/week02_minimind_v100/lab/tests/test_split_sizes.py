"""split sizes 与 permutation 的纯逻辑测试（不需要进程组）。

这些是 EP 里最容易写错、又最难在多卡上定位的部分：写错了不报错，直接 hang。
在单进程里用"手工构造的计数矩阵"把不变量 I2 检查一遍，比在 8 卡上试快得多。
"""

from __future__ import annotations

import torch

from mm_dist import ep_moe


def simulate_counts(assignments, ep_size, experts_per_rank):
    """给定每个 rank 的 (token→全局专家) 分配，返回每 rank 的计数表 ``[E]``。"""
    num_experts = ep_size * experts_per_rank
    return [torch.bincount(torch.tensor(a, dtype=torch.long),
                           minlength=num_experts).to(torch.int64)
            for a in assignments]


def exchange(counts_mats):
    """模拟等长 all_to_all：``recv[j*epr+e] = counts_mats[j][my_rank*epr+e]``。"""
    ep_size = len(counts_mats)
    total = counts_mats[0].numel()
    epr = total // ep_size
    out = []
    for me in range(ep_size):
        recv = torch.zeros(total, dtype=torch.int64)
        for src in range(ep_size):
            recv[src * epr:(src + 1) * epr] = \
                counts_mats[src][me * epr:(me + 1) * epr]
        out.append(recv)
    return out


def test_invariant_i2_input_matches_output_splits():
    """rank r 的 input_split_sizes[j] 必须等于 rank j 的 output_split_sizes[r]。"""
    ep_size, epr = 4, 2
    assignments = [
        [0, 0, 1, 3, 5, 7, 7, 7],
        [2, 2, 2, 2, 4, 6, 0, 1],
        [7, 6, 5, 4, 3, 2, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],   # 全部打到 expert 0（热点）
    ]
    counts = simulate_counts(assignments, ep_size, epr)
    recvs = exchange(counts)
    in_splits = [c.view(ep_size, epr).sum(1).tolist() for c in counts]
    out_splits = [r.view(ep_size, epr).sum(1).tolist() for r in recvs]
    for r in range(ep_size):
        for j in range(ep_size):
            assert in_splits[r][j] == out_splits[j][r], (r, j, in_splits, out_splits)
    for r in range(ep_size):
        assert sum(in_splits[r]) == len(assignments[r])


def test_zero_split_is_legal():
    """某个 rank 一个 token 都不发/不收时，split size 为 0 且总和仍然自洽。"""
    ep_size, epr = 2, 2
    counts = simulate_counts([[0, 0, 1, 1], [0, 1, 0, 1]], ep_size, epr)
    recvs = exchange(counts)
    in_splits = [c.view(ep_size, epr).sum(1).tolist() for c in counts]
    out_splits = [r.view(ep_size, epr).sum(1).tolist() for r in recvs]
    assert in_splits == [[4, 0], [4, 0]]
    assert out_splits == [[4, 4], [0, 0]]
    assert sum(out_splits[1]) == 0


def test_local_expert_ids_layout():
    """接收缓冲区的行序必须是 (来源 rank 外层, 本地专家 内层)。"""
    recv_counts = torch.tensor([2, 1, 0, 3], dtype=torch.int64)  # ep=2, epr=2
    ids = ep_moe._local_expert_ids(recv_counts, ep_size=2, experts_per_rank=2,
                                   device=torch.device("cpu"))
    assert ids.tolist() == [0, 0, 1, 1, 1, 1]
    assert int(ids.numel()) == int(recv_counts.sum())


def test_local_expert_ids_all_zero():
    recv_counts = torch.zeros(4, dtype=torch.int64)
    ids = ep_moe._local_expert_ids(recv_counts, ep_size=2, experts_per_rank=2,
                                   device=torch.device("cpu"))
    assert ids.numel() == 0


def test_permutation_is_stable_and_groups_by_destination():
    """argsort(stable=True) 必须同时满足：按目标 rank 分块、块内保持 token 原序。"""
    expert_of = torch.tensor([3, 0, 2, 0, 1, 3, 1], dtype=torch.long)
    perm = torch.argsort(expert_of, stable=True)
    sorted_experts = expert_of[perm].tolist()
    assert sorted_experts == sorted(sorted_experts)
    assert perm.tolist() == [1, 3, 4, 6, 2, 0, 5]
    # 同一个专家内部，原始 token 下标必须递增（stable 的含义）
    for expert in range(4):
        positions = [int(p) for p in perm if int(expert_of[p]) == expert]
        assert positions == sorted(positions)


def test_apply_capacity_keeps_first_come():
    """capacity 生效时按 token 序保留前 c 个，其余丢弃。"""
    expert_of = torch.tensor([0, 0, 0, 0, 1, 1, 2], dtype=torch.long)
    keep, dropped = ep_moe._apply_capacity(expert_of, num_experts=4,
                                           capacity_factor=0.5, n_tokens=8,
                                           top_k=1)
    # capacity = ceil(0.5 * 8 * 1 / 4) = 1
    assert dropped == 4
    assert keep.tolist() == [0, 4, 6]
    assert torch.equal(keep, torch.sort(keep).values), "保留的索引必须仍是升序"


def test_apply_capacity_minimum_one():
    """极小的 capacity_factor 也至少给每个专家留 1 个位置，不能算出 0。"""
    expert_of = torch.tensor([0, 0, 1], dtype=torch.long)
    keep, dropped = ep_moe._apply_capacity(expert_of, num_experts=4,
                                           capacity_factor=1e-6, n_tokens=3,
                                           top_k=1)
    assert keep.numel() == 2
    assert dropped == 1


def test_make_ep_groups_layout_without_process_group():
    """未初始化进程组时，make_ep_groups 只返回分组表，不建组也不报错。"""
    group, ep_rank, all_ranks = ep_moe.make_ep_groups(4, 8, rank=6)
    assert group is None
    assert ep_rank == 2
    assert all_ranks == [[0, 1, 2, 3], [4, 5, 6, 7]]
    group, ep_rank, all_ranks = ep_moe.make_ep_groups(2, 8, rank=5)
    assert ep_rank == 1
    assert all_ranks == [[0, 1], [2, 3], [4, 5], [6, 7]]


def test_aux_loss_balanced_equals_one():
    """完全均衡时 aux_loss / coef == 1；完全坍缩时趋近 E。"""
    num_experts = 4
    n = 64
    balanced_idx = torch.arange(n) % num_experts
    scores = torch.full((n, num_experts), 1.0 / num_experts)
    value = ep_moe.aux_loss_value(scores, balanced_idx.unsqueeze(1), num_experts, 1.0)
    assert abs(float(value) - 1.0) < 1e-6

    collapsed_idx = torch.zeros(n, 1, dtype=torch.long)
    scores_collapsed = torch.zeros(n, num_experts)
    scores_collapsed[:, 0] = 1.0
    collapsed = ep_moe.aux_loss_value(scores_collapsed, collapsed_idx,
                                      num_experts, 1.0)
    assert abs(float(collapsed) - num_experts) < 1e-6


def test_phase_timer_cpu_fallback():
    """没有 CUDA 时 PhaseTimer 退化到 perf_counter，仍然要能累计。"""
    timer = ep_moe.PhaseTimer(enabled=True, device="cpu")
    assert timer.use_cuda is False
    with ep_moe.phase(timer, "router"):
        torch.randn(64, 64) @ torch.randn(64, 64)
    report = timer.report()
    assert report["router"] >= 0.0
    assert timer.calls["router"] == 1
    timer.reset()
    assert timer.report()["router"] == 0.0


def test_phase_timer_disabled_is_free():
    timer = ep_moe.PhaseTimer(enabled=False, device="cpu")
    with ep_moe.phase(timer, "dispatch_a2a"):
        pass
    assert timer.calls["dispatch_a2a"] == 0
