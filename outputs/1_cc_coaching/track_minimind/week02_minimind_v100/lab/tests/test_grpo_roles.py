"""GRPO 角色放置：分组划分的纯逻辑 + 广播后参数一致的 CPU 简化版。

分组划分是"写错了会 hang、写对了没人夸"的那类代码，所以单独抽成纯函数
``plan_role_groups`` 来测。广播一致性对应 01_FOUNDATIONS.md 失败模式 #12：
同步版 GRPO 每轮 broadcast 之后，rollout 卡上的权重必须与 policy 卡逐位相同，
否则 rollout 用的是上上轮的策略，advantage 的分布就不是当前策略的。
"""

from __future__ import annotations

import pytest
import torch

import conftest as H

from mm_dist import grpo_roles


# ---------------------------------------------------------------------------
# 分组划分（纯逻辑）
# ---------------------------------------------------------------------------


def test_plan_role_groups_eight_gpus_default():
    plan = grpo_roles.plan_role_groups(8, 4, 2, 2)
    assert plan == {"policy": [0, 1, 2, 3], "rollout": [4, 5], "reward": [6, 7]}
    assert grpo_roles.role_of_rank(plan, 0) == "policy"
    assert grpo_roles.role_of_rank(plan, 4) == "rollout"
    assert grpo_roles.role_of_rank(plan, 7) == "reward"


def test_plan_role_groups_covers_all_ranks_without_overlap():
    for world_size, split in ((3, (1, 1, 1)), (8, (4, 2, 2)), (8, (2, 3, 3)),
                              (16, (8, 4, 4))):
        plan = grpo_roles.plan_role_groups(world_size, *split)
        flat = plan["policy"] + plan["rollout"] + plan["reward"]
        assert sorted(flat) == list(range(world_size))
        assert len(set(flat)) == world_size


def test_plan_role_groups_rejects_bad_splits():
    with pytest.raises(ValueError):
        grpo_roles.plan_role_groups(8, 4, 2, 1)      # 加起来不等于 8
    with pytest.raises(ValueError):
        grpo_roles.plan_role_groups(8, 8, 0, 0)      # 有角色分到 0 卡
    with pytest.raises(ValueError):
        grpo_roles.plan_role_groups(2, 1, 1, 1)      # 卡数不够三个角色


def test_role_of_rank_rejects_out_of_range():
    plan = grpo_roles.plan_role_groups(8, 4, 2, 2)
    with pytest.raises(ValueError):
        grpo_roles.role_of_rank(plan, 8)


# ---------------------------------------------------------------------------
# GRPO 的数学
# ---------------------------------------------------------------------------


def test_group_advantages_zero_mean_within_group():
    rewards = torch.tensor([1.0, 2.0, 3.0, 4.0, 10.0, 10.0, 10.0, 10.0])
    adv = grpo_roles.group_advantages(rewards, group_size=4)
    assert adv.shape == rewards.shape
    first, second = adv[:4], adv[4:]
    assert abs(float(first.mean())) < 1e-6
    assert abs(float(first.std(unbiased=False)) - 1.0) < 1e-4
    # 一组内 reward 全相同 -> 优势必须是 0（不能除出 inf）
    assert torch.allclose(second, torch.zeros(4), atol=1e-5)
    assert torch.isfinite(adv).all()


def test_group_advantages_rejects_non_divisible():
    with pytest.raises(ValueError):
        grpo_roles.group_advantages(torch.zeros(7), group_size=4)


def test_rule_reward_range_and_determinism():
    torch.manual_seed(5)
    gen = torch.randint(0, 100, (4, 8 + 6))
    r1 = grpo_roles.rule_reward(gen, prompt_len=8, vocab_size=100, rule_mod=7)
    r2 = grpo_roles.rule_reward(gen, prompt_len=8, vocab_size=100, rule_mod=7)
    assert torch.equal(r1, r2), "规则奖励必须完全确定，否则 advantage 不可比"
    assert r1.shape == (4,)
    assert float(r1.min()) >= 0.0 and float(r1.max()) <= 1.0


def test_rule_reward_penalises_repetition():
    """全是同一个 token 的序列，distinct_ratio 最低。"""
    repeated = torch.full((1, 4 + 6), 14, dtype=torch.long)   # 14 % 7 == 0
    varied = torch.tensor([[0, 0, 0, 0, 7, 14, 21, 28, 35, 42]], dtype=torch.long)
    r_rep = float(grpo_roles.rule_reward(repeated, 4, 100, 7)[0])
    r_var = float(grpo_roles.rule_reward(varied, 4, 100, 7)[0])
    assert r_var > r_rep, (r_var, r_rep)
    assert abs(r_rep - (0.5 * (1 / 6) + 0.5 * 1.0)) < 1e-6


def test_param_checksum_detects_difference():
    torch.manual_seed(1)
    a = torch.nn.Linear(4, 4)
    torch.manual_seed(2)
    b = torch.nn.Linear(4, 4)
    assert grpo_roles.param_checksum(a) != grpo_roles.param_checksum(b)
    b.load_state_dict(a.state_dict())
    assert abs(grpo_roles.param_checksum(a) - grpo_roles.param_checksum(b)) < 1e-12


# ---------------------------------------------------------------------------
# 广播一致性（2 进程 gloo）
# ---------------------------------------------------------------------------


def broadcast_worker(rank: int, world_size: int, port: int, out_dir: str) -> None:
    """每个 rank 用不同的种子初始化；从 rank0 广播后 checksum 必须相同。"""
    H.worker_setup(rank, world_size, port)
    import torch.distributed as dist

    from mm_dist import grpo_roles as G

    dist.init_process_group("gloo")
    torch.manual_seed(100 + rank)
    model = torch.nn.Sequential(torch.nn.Linear(8, 8), torch.nn.Linear(8, 4))
    before = G.param_checksum(model)
    n = G.broadcast_policy_weights(model, src=0)
    after = G.param_checksum(model)
    H.write_result(out_dir, rank, {
        "rank": rank, "before": before, "after": after, "tensors": n,
        "first_row": model[0].weight[0].tolist(),
    })
    dist.barrier()
    dist.destroy_process_group()


def test_broadcast_makes_all_ranks_identical(tmp_path, spawn_dist):
    results = spawn_dist(broadcast_worker, 2, str(tmp_path / "bcast"))
    assert len(results) == 2
    r0, r1 = results
    assert r0["tensors"] == r1["tensors"] == 4
    assert abs(r0["after"] - r1["after"]) < 1e-9, (
        "广播之后两个 rank 的参数 checksum 仍不同：" + str(results))
    assert abs(r0["before"] - r0["after"]) < 1e-12, "src=0 自己的参数不应被改动"
    assert abs(r1["before"] - r1["after"]) > 1e-6, "rank1 的参数应当被覆盖"
    assert r0["first_row"] == r1["first_row"]


def role_group_worker(rank: int, world_size: int, port: int, out_dir: str) -> None:
    """三个角色组各建一次 new_group，然后在自己的组里 all_reduce 一个标记。"""
    H.worker_setup(rank, world_size, port)
    import torch.distributed as dist

    from mm_dist import grpo_roles as G

    dist.init_process_group("gloo")
    plan = G.plan_role_groups(world_size, 2, 1, 1)
    groups = G.make_role_groups(plan)
    role = G.role_of_rank(plan, rank)
    marker = torch.tensor([float(rank)])
    dist.all_reduce(marker, group=groups[role])
    H.write_result(out_dir, rank, {"rank": rank, "role": role,
                                   "group_sum": float(marker[0]),
                                   "plan": plan})
    dist.barrier()
    dist.destroy_process_group()


def test_role_groups_are_usable_for_collectives(tmp_path, spawn_dist):
    """4 进程 → policy[0,1] / rollout[2] / reward[3]；组内 all_reduce 必须只算组内。"""
    results = spawn_dist(role_group_worker, 4, str(tmp_path / "groups"))
    by_rank = {r["rank"]: r for r in results}
    assert by_rank[0]["role"] == by_rank[1]["role"] == "policy"
    assert by_rank[2]["role"] == "rollout"
    assert by_rank[3]["role"] == "reward"
    assert by_rank[0]["group_sum"] == 1.0   # 0 + 1，只在 policy 组内
    assert by_rank[1]["group_sum"] == 1.0
    assert by_rank[2]["group_sum"] == 2.0   # 单成员组
    assert by_rank[3]["group_sum"] == 3.0


# ---------------------------------------------------------------------------
# 单进程 replicate 放置的端到端 smoke
# ---------------------------------------------------------------------------


def test_replicate_placement_runs_one_step(tmp_path, tiny_config, minimind_root):
    """单进程 replicate + 规则奖励，跑 1 步，检查日志字段齐全。"""
    from mm_dist import common as C

    out_dir = str(tmp_path / "grpo")
    args = grpo_roles.build_parser().parse_args([
        "--config", tiny_config, "--out-dir", out_dir,
        "--placement", "replicate", "--force-cpu", "--backend", "gloo",
        "--dtype", "float32", "--max-steps", "1",
        "--n-prompts", "2", "--group-size", "2", "--prompt-len", "8",
        "--max-gen-len", "4", "--reward", "rule", "--lr", "1e-5",
    ])
    summary = grpo_roles.run(args)
    assert summary["placement"] == "replicate"
    records = C.read_jsonl(summary["log_path"])
    assert records[0]["event"] == "header"
    assert records[0]["max_gen_len"] == 4
    step = records[1]
    for key in ("loss", "reward_mean", "rollout_ms", "reward_ms", "update_ms",
                "sync_ms", "role_idle_ms", "step_time_s", "peak_mem_mb",
                "param_checksum"):
        assert key in step, key
    assert step["reward_mean"] >= 0.0


def test_generation_is_greedy_and_repeatable(tiny_config, minimind_root):
    """同样的权重与 prompt，两次贪心生成必须逐 token 相同（有无 KV cache 也一致）。"""
    from mm_dist import common as C

    C.set_seed(7, per_rank=False)
    model, lm_config, cfg, mm = C.build_model(tiny_config, device="cpu")
    prompts = grpo_roles.make_prompts(1, 2, 2, 6, int(lm_config.vocab_size),
                                      "cpu", seed=42)
    out_a = grpo_roles.greedy_generate(model, prompts, 5, use_cache=True)
    out_b = grpo_roles.greedy_generate(model, prompts, 5, use_cache=True)
    out_c = grpo_roles.greedy_generate(model, prompts, 5, use_cache=False)
    assert out_a.shape == (4, 11)
    assert torch.equal(out_a, out_b)
    assert torch.equal(out_a, out_c), "KV cache 路径与不带 cache 的路径必须给出同一串 token"
    assert torch.equal(out_a[:, :6], prompts)


def test_make_prompts_groups_share_the_same_prompt(tiny_config, minimind_root):
    """GRPO 的组：同一个 prompt 复制 G 份，否则组内比较没有意义。"""
    from mm_dist import common as C

    C.set_seed(7, per_rank=False)
    prompts = grpo_roles.make_prompts(3, n_prompts=2, group_size=3, prompt_len=5,
                                      vocab_size=100, device="cpu", seed=42)
    assert prompts.shape == (6, 5)
    assert torch.equal(prompts[0], prompts[1]) and torch.equal(prompts[1], prompts[2])
    assert torch.equal(prompts[3], prompts[4]) and torch.equal(prompts[4], prompts[5])
    assert not torch.equal(prompts[0], prompts[3])
