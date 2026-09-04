"""GRPO 三种角色放置的同步版框架：replicate / policy_fsdp / split_roles。

要回答的问题
------------
同样 8 张 V100，policy / rollout / reward 三个角色放在哪里，峰值显存和 step 时间
差多少？本文件把三种放置写成同一个循环的三个分支，日志字段一致，可直接对比。

三种放置
--------
``replicate``    每张卡都有 policy（DDP 包）、都做 rollout、都做 reward。
                 显存 = policy + optimizer + rollout 激活 + reward，全在一张卡上；
                 没有跨角色通信，只有 DDP 的梯度 allreduce。
``policy_fsdp``  policy 用 FSDP1 分片到全部 8 卡，rollout 和 reward 仍在每张卡上做。
                 参数/优化器显存除以 8，代价是生成时每层都要 all_gather。
``split_roles``  ``dist.new_group`` 划三组，8 卡默认 4/2/2：
                 rank 0-3 policy，rank 4-5 rollout，rank 6-7 reward。
                 一轮 = rollout 生成 → reward 打分 → policy 更新 → broadcast 权重。

同步版的含义
------------
每一轮里三组严格串行：rollout 生成完才打分，打分完才更新，更新完才广播。
所以 rollout 用的权重永远是上一轮结束时的权重，不存在异步 RL 的 off-policy 漂移
（01_FOUNDATIONS.md 失败模式 #12 检查的就是这一点：广播后 policy 与 rollout 的
参数 checksum 必须相等）。代价是 rollout 卡和 reward 卡在别人干活时是闲着的——
``role_idle_ms`` 字段量的就是这个。

为什么所有跨角色通信都走全局组
------------------------------
``broadcast`` 只要有一个成员没调用就会挂住。让全部 8 个 rank 都参与同一次
``dist.broadcast``（src 是发送方那一组的第一个 rank），比在子组之间点对点传递
更难写错，也让"谁在等谁"能被 barrier 计时直接量出来。

GRPO 的教学简化（明确写出来，不是省略）
--------------------------------------
每轮 rollout 之后只做 **一次** policy 更新，所以重要性采样比 ``ratio = 1``，
PPO 的 clip 项恒不激活，目标函数退化为 ``-A * logpi``。
参考模型的 KL 项默认关闭（``--kl-coef 0``）；打开时参考的是**冻结的初始 policy**
（训练开始前 ``copy.deepcopy`` 的一份，全程不更新）——对着当前 policy 算 KL 恒等于 0，
那不是正则化。代价是多一份完整权重，所以只在 ``--kl-coef > 0`` 时才建这份拷贝。
组相对优势：同一个 prompt 的 G 个样本，``A_i = (r_i - mean(r)) / (std(r) + 1e-6)``。

只用 torch 2.1 API：``dist.new_group``、``dist.broadcast``、``dist.barrier``、
``DistributedDataParallel``、FSDP1。
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mm_dist import common as C  # noqa: E402
from mm_dist.train_fsdp import (  # noqa: E402
    SHARDING_STRATEGIES,
    build_auto_wrap_policy,
    make_mixed_precision,
    require_cuda_for_fsdp,
)

PLACEMENTS = ("replicate", "policy_fsdp", "split_roles")
ROLES = ("policy", "rollout", "reward")


# ---------------------------------------------------------------------------
# 角色分组（纯逻辑，可单独单测）
# ---------------------------------------------------------------------------


def plan_role_groups(world_size: int, n_policy: int = 4, n_rollout: int = 2,
                     n_reward: int = 2) -> Dict[str, List[int]]:
    """把 ``[0, world_size)`` 按 policy / rollout / reward 顺序切成三段连续 rank。

    8 卡默认 4/2/2 → policy=[0,1,2,3]、rollout=[4,5]、reward=[6,7]。
    连续切分的理由：同一角色内部要做 allreduce，连续 rank 在多数拓扑上互连更好
    （公司机器非全 NVLink，具体分组质量以探针为准）。
    """
    if world_size < 3:
        raise ValueError(
            "split_roles 至少需要 3 个 rank（每个角色 1 个），收到 world_size="
            + str(world_size))
    for name, n in (("policy", n_policy), ("rollout", n_rollout), ("reward", n_reward)):
        if n < 1:
            raise ValueError(name + " 的 rank 数必须 >= 1，收到 " + str(n))
    total = n_policy + n_rollout + n_reward
    if total != world_size:
        raise ValueError(
            "policy+rollout+reward=" + str(total) + " 必须等于 world_size="
            + str(world_size) + "（本教学版不允许角色共卡）")
    ranks = list(range(world_size))
    return {
        "policy": ranks[:n_policy],
        "rollout": ranks[n_policy:n_policy + n_rollout],
        "reward": ranks[n_policy + n_rollout:],
    }


def role_of_rank(plan: Dict[str, List[int]], rank: int) -> str:
    for role in ROLES:
        if rank in plan[role]:
            return role
    raise ValueError("rank " + str(rank) + " 不在任何角色组里：" + json.dumps(plan))


def make_role_groups(plan: Dict[str, List[int]]) -> Dict[str, Any]:
    """为三个角色各建一个 ``dist.new_group``。

    所有进程必须按**相同顺序**调用 ``new_group``，即使自己不是成员——
    这是 torch 2.1 distributed 文档的硬要求，漏掉会在第一次集合通信时挂住。
    """
    if not (dist.is_available() and dist.is_initialized()):
        return {role: None for role in ROLES}
    groups: Dict[str, Any] = {}
    for role in ROLES:  # 固定顺序：policy -> rollout -> reward
        groups[role] = dist.new_group(ranks=plan[role])
    return groups


# ---------------------------------------------------------------------------
# 权重同步
# ---------------------------------------------------------------------------


def param_checksum(model) -> float:
    """所有参数的 float64 求和；用于验证 broadcast 之后各 rank 参数一致。"""
    total = 0.0
    with torch.no_grad():
        for p in model.parameters():
            total += float(p.detach().double().sum())
    return total


def broadcast_policy_weights(model, src: int, group: Any = None) -> int:
    """把 ``src`` 的参数广播给组内所有 rank，返回广播的张量个数。

    ``model`` 必须是**未分片**的完整模型（split_roles 里 rollout 卡也持有完整
    policy 副本）。FSDP 分片过的模型不能这样广播：每个 rank 的 ``p.data`` 是不同的
    分片，广播会把 rank0 的分片盖到所有人身上。
    """
    if not (dist.is_available() and dist.is_initialized()):
        return 0
    count = 0
    with torch.no_grad():
        for p in model.parameters():
            dist.broadcast(p.data, src=src, group=group)
            count += 1
        for b in model.buffers():
            if b.is_floating_point() or b.dtype in (torch.int32, torch.int64):
                dist.broadcast(b.data, src=src, group=group)
                count += 1
    return count


# ---------------------------------------------------------------------------
# 生成与打分
# ---------------------------------------------------------------------------


@torch.no_grad()
def greedy_generate(model, input_ids: torch.Tensor, max_new_tokens: int,
                    use_cache: bool = True) -> torch.Tensor:
    """贪心生成，返回 ``[B, prompt_len + max_new_tokens]``。

    贪心（argmax）而不是采样：本周比较的是三种**放置**的显存与时间，
    生成必须在不同放置下逐 token 完全一致，否则 reward 不可比。
    """
    model.eval()
    seq = input_ids
    past = None
    step_input = input_ids
    for _ in range(int(max_new_tokens)):
        if use_cache:
            out = model(step_input, past_key_values=past, use_cache=True)
            past = out.past_key_values
        else:
            out = model(seq, use_cache=False)
        nxt = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        seq = torch.cat([seq, nxt], dim=1)
        step_input = nxt
    model.train()
    return seq


def rule_reward(generated: torch.Tensor, prompt_len: int, vocab_size: int,
                rule_mod: int = 7) -> torch.Tensor:
    """内置规则奖励（默认）：不依赖任何外部模型，完全确定。

    ``r = 0.5 * distinct_ratio + 0.5 * target_ratio``

    * ``distinct_ratio``：生成段里不重复 token 的比例——惩罚复读；
    * ``target_ratio``：生成段里满足 ``id % rule_mod == 0`` 的 token 比例——
      一个可验证的、与语义无关的"任务"，作用是让 reward 有可优化的方向，
      好让 GRPO 的 advantage 不是常数。

    这不是语言质量指标，只是放置实验的负载与信号源；任何"模型变好了"的结论
    都不能从这里得出。
    """
    gen = generated[:, prompt_len:]
    if gen.numel() == 0:
        return torch.zeros(generated.shape[0], dtype=torch.float32,
                           device=generated.device)
    length = gen.shape[1]
    distinct = torch.tensor(
        [torch.unique(row).numel() / float(length) for row in gen],
        dtype=torch.float32, device=gen.device)
    target = (gen % max(int(rule_mod), 1) == 0).float().mean(dim=1).to(torch.float32)
    return 0.5 * distinct + 0.5 * target


class ModelReward:
    """``--reward model`` 的打分器：用本地已有的 HF 序列分类模型。

    公司环境里 ``--reward-path`` 指向公司内部已下载好的目录；本文件不下载、
    不联网、不建议上传任何东西。MiniMind 的 tokenizer 与 RM 的 tokenizer 通常不同，
    所以流程是"MiniMind ids → 文本 → RM tokenizer → RM 打分"。
    """

    def __init__(self, reward_path: str, minimind_root: Optional[str],
                 device: str, max_length: int = 256) -> None:
        if not os.path.isdir(reward_path):
            raise FileNotFoundError(
                "--reward-path " + str(reward_path) + " 不是一个已存在的目录。"
                "请指向本机/公司内部已有的 HF 模型目录；本 lab 不下载模型。")
        from transformers import (AutoModelForSequenceClassification,
                                  AutoTokenizer)
        root = C.resolve_minimind_root(minimind_root)
        self.gen_tokenizer = AutoTokenizer.from_pretrained(
            os.path.join(root, "model"))
        self.rm_tokenizer = AutoTokenizer.from_pretrained(reward_path)
        self.rm = AutoModelForSequenceClassification.from_pretrained(
            reward_path).to(device).eval()
        self.device = device
        self.max_length = int(max_length)

    @torch.no_grad()
    def __call__(self, generated: torch.Tensor, prompt_len: int) -> torch.Tensor:
        texts = self.gen_tokenizer.batch_decode(generated[:, prompt_len:],
                                                skip_special_tokens=True)
        batch = self.rm_tokenizer(texts, return_tensors="pt", padding=True,
                                  truncation=True, max_length=self.max_length)
        batch = {k: v.to(self.device) for k, v in batch.items()}
        logits = self.rm(**batch).logits
        return logits[:, 0].float() if logits.dim() == 2 else logits.float()


def group_advantages(rewards: torch.Tensor, group_size: int) -> torch.Tensor:
    """GRPO 的组相对优势：同一 prompt 的 G 个样本内做标准化。

    ``rewards`` 形状 ``[n_prompts * group_size]``，按 prompt 连续排列。
    """
    if rewards.numel() % group_size != 0:
        raise ValueError(
            "rewards 数量 " + str(int(rewards.numel())) + " 不能被 group_size="
            + str(group_size) + " 整除")
    grouped = rewards.view(-1, group_size)
    mean = grouped.mean(dim=1, keepdim=True)
    std = grouped.std(dim=1, unbiased=False, keepdim=True)
    return ((grouped - mean) / (std + 1e-6)).reshape(-1)


def sequence_logprob(model, sequences: torch.Tensor, prompt_len: int
                     ) -> torch.Tensor:
    """生成段每个 token 的平均 log pi(a|s)，形状 ``[B]``（带梯度）。"""
    out = model(sequences)
    logits = out.logits[:, :-1, :]
    targets = sequences[:, 1:]
    logprobs = F.log_softmax(logits.float(), dim=-1)
    picked = logprobs.gather(2, targets.unsqueeze(-1)).squeeze(-1)
    gen_part = picked[:, prompt_len - 1:]
    return gen_part.mean(dim=1)


# ---------------------------------------------------------------------------
# 主循环
# ---------------------------------------------------------------------------


def build_policy(args, device: str, placement: str, info: C.DistInfo):
    """按放置方式构造 policy（以及给 rollout 用的引用）。"""
    model, lm_config, cfg, mm = C.build_model(
        args.config, minimind_root=args.minimind_root,
        device="cpu" if (placement == "policy_fsdp"
                         and str(device).startswith("cuda")) else device)
    raw = model
    if placement == "policy_fsdp":
        if not info.initialized:
            raise SystemExit("policy_fsdp 需要 torchrun 启动的进程组")
        require_cuda_for_fsdp(device)
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        fsdp_kwargs: Dict[str, Any] = dict(
            sharding_strategy=SHARDING_STRATEGIES[args.sharding],
            auto_wrap_policy=build_auto_wrap_policy(mm.MiniMindBlock),
            mixed_precision=make_mixed_precision(args.dtype, device),
            use_orig_params=True,
            limit_all_gathers=True,
            sync_module_states=True,
        )
        if str(device).startswith("cuda"):
            fsdp_kwargs["device_id"] = torch.device(device)
        wrapped = FSDP(model, **fsdp_kwargs)
    elif placement == "replicate" and info.initialized and info.world_size > 1:
        device_ids = [info.local_rank] if str(device).startswith("cuda") else None
        wrapped = DistributedDataParallel(model, device_ids=device_ids)
    else:
        wrapped = model
    return wrapped, raw, lm_config, cfg, mm


def make_prompts(step: int, n_prompts: int, group_size: int, prompt_len: int,
                 vocab_size: int, device: str, seed: int) -> torch.Tensor:
    """确定性 prompt：只由 (seed, step, prompt 序号) 决定，与放置方式无关。"""
    rows = []
    for p in range(n_prompts):
        gen = torch.Generator()
        gen.manual_seed((seed * 7919 + step * 104729 + p) % (2 ** 62))
        ids = torch.randint(1, max(vocab_size, 2), (prompt_len,), generator=gen,
                            dtype=torch.long)
        for _ in range(group_size):
            rows.append(ids)
    return torch.stack(rows).to(device)


def run(args: argparse.Namespace) -> Dict[str, Any]:
    info = C.init_dist(backend=args.backend, timeout_s=args.nccl_timeout_s,
                       force_cpu=args.force_cpu)
    device = "cpu" if args.force_cpu else info.device
    torch_dtype = C.assert_dtype_supported(args.dtype, device)
    C.set_seed(args.seed, per_rank=False)
    placement = args.placement

    plan: Optional[Dict[str, List[int]]] = None
    groups: Dict[str, Any] = {role: None for role in ROLES}
    my_role = "all"
    if placement == "split_roles":
        if not info.initialized:
            raise SystemExit(
                "split_roles 需要 torchrun 启动的多进程；单进程请用 --placement replicate")
        plan = plan_role_groups(info.world_size, args.n_policy, args.n_rollout,
                                args.n_reward)
        groups = make_role_groups(plan)
        my_role = role_of_rank(plan, info.rank)

    policy, raw_policy, lm_config, cfg, mm = build_policy(args, device, placement, info)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=float(args.lr))
    scaler = C.make_grad_scaler(torch_dtype, device)
    C.reset_peak_mem(device)

    # KL 参考模型：GRPO 的 KL 项要对着**冻结的初始 policy**，不是当前 policy。
    # 对着当前 policy 算出来恒等于 0，那不是正则化，是自欺。代价是多一份完整权重，
    # 所以默认 --kl-coef 0 时不建这份拷贝。
    ref_model = None
    if float(args.kl_coef) > 0.0:
        ref_model = copy.deepcopy(raw_policy).to(device).eval()
        for p in ref_model.parameters():
            p.requires_grad_(False)

    reward_fn: Any = None
    if args.reward == "model":
        if not args.reward_path:
            raise SystemExit("--reward model 必须同时给 --reward-path <本地 HF 目录>")
        if placement != "split_roles" or my_role == "reward":
            reward_fn = ModelReward(args.reward_path, args.minimind_root, device)

    n_prompts = max(int(args.n_prompts), 1)
    group_size = max(int(args.group_size), 2)
    prompt_len = max(int(args.prompt_len), 1)
    total_seqs = n_prompts * group_size
    seq_total_len = prompt_len + int(args.max_gen_len)

    logger = C.JsonlLogger(args.out_dir, args.log_name or ("grpo_" + placement),
                           rank=info.rank)
    logger.write({
        "step": 0, "event": "header", "runner": "grpo_roles",
        "placement": placement, "role": my_role, "role_plan": plan,
        "world_size": info.world_size, "dtype": args.dtype,
        "device_type": "cuda" if str(device).startswith("cuda") else "cpu",
        "n_prompts": n_prompts, "group_size": group_size,
        "prompt_len": prompt_len, "max_gen_len": int(args.max_gen_len),
        "reward": args.reward, "reward_path": args.reward_path,
        "kl_coef": float(args.kl_coef),
        "params_total": C.count_params(raw_policy)[0],
        "minimind_commit": mm.commit, "env": C.env_summary(),
    })
    C.log_main("[grpo] placement=%s world_size=%d role_plan=%s"
               % (placement, info.world_size, json.dumps(plan)))

    steps = max(int(args.max_steps), 1)
    summaries: List[Dict[str, Any]] = []
    for step in range(1, steps + 1):
        t_step = time.perf_counter()
        prompts = make_prompts(step, n_prompts, group_size, prompt_len,
                               int(lm_config.vocab_size), device, args.seed)
        sequences = torch.zeros(total_seqs, seq_total_len, dtype=torch.long,
                                device=device)
        rewards = torch.zeros(total_seqs, dtype=torch.float32, device=device)
        idle_ms = 0.0

        # --- 1. rollout ------------------------------------------------
        t0 = time.perf_counter()
        do_rollout = placement != "split_roles" or my_role == "rollout"
        if do_rollout:
            with C.autocast_context(torch_dtype, device):
                sequences = greedy_generate(policy, prompts,
                                            int(args.max_gen_len),
                                            use_cache=bool(args.use_cache))
        rollout_ms = (time.perf_counter() - t0) * 1000.0
        if placement == "split_roles":
            t_idle = time.perf_counter()
            dist.broadcast(sequences, src=plan["rollout"][0])
            idle_ms += (time.perf_counter() - t_idle) * 1000.0

        # --- 2. reward -------------------------------------------------
        t0 = time.perf_counter()
        do_reward = placement != "split_roles" or my_role == "reward"
        if do_reward:
            if reward_fn is not None:
                rewards = reward_fn(sequences, prompt_len).to(device).float()
            else:
                rewards = rule_reward(sequences, prompt_len,
                                      int(lm_config.vocab_size),
                                      int(args.rule_mod)).to(device).float()
        reward_ms = (time.perf_counter() - t0) * 1000.0
        if placement == "split_roles":
            t_idle = time.perf_counter()
            dist.broadcast(rewards, src=plan["reward"][0])
            idle_ms += (time.perf_counter() - t_idle) * 1000.0

        # --- 3. policy update ------------------------------------------
        t0 = time.perf_counter()
        advantages = group_advantages(rewards.float(), group_size)
        loss_value = 0.0
        do_update = placement != "split_roles" or my_role == "policy"
        if do_update:
            optimizer.zero_grad(set_to_none=True)
            with C.autocast_context(torch_dtype, device):
                logp = sequence_logprob(policy, sequences, prompt_len)
                loss = -(advantages.detach() * logp).mean()
                if ref_model is not None:
                    with torch.no_grad():
                        ref_logp = sequence_logprob(ref_model, sequences, prompt_len)
                    loss = loss + float(args.kl_coef) * (logp - ref_logp).pow(2).mean()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            if placement == "policy_fsdp":
                policy.clip_grad_norm_(float(args.grad_clip))
            else:
                torch.nn.utils.clip_grad_norm_(policy.parameters(),
                                               float(args.grad_clip))
            if placement == "split_roles" and len(plan["policy"]) > 1:
                # policy 组内手写梯度 allreduce（这组没有 DDP 包）
                for p in policy.parameters():
                    if p.grad is not None:
                        dist.all_reduce(p.grad, op=dist.ReduceOp.SUM,
                                        group=groups["policy"])
                        p.grad /= float(len(plan["policy"]))
            scaler.step(optimizer)
            scaler.update()
            loss_value = float(loss.detach())
        update_ms = (time.perf_counter() - t0) * 1000.0

        # --- 4. 权重同步 -------------------------------------------------
        t0 = time.perf_counter()
        n_bcast = 0
        checksum_after = 0.0
        if placement == "split_roles":
            n_bcast = broadcast_policy_weights(raw_policy, src=plan["policy"][0])
            checksum_after = param_checksum(raw_policy)
        elif placement == "replicate":
            checksum_after = param_checksum(raw_policy)
        sync_ms = (time.perf_counter() - t0) * 1000.0

        dt = time.perf_counter() - t_step
        record = {
            "step": step,
            "placement": placement,
            "role": my_role,
            "loss": loss_value,
            "reward_mean": float(rewards.mean()),
            "reward_std": float(rewards.std(unbiased=False)),
            "advantage_absmean": float(advantages.abs().mean()),
            "rollout_ms": rollout_ms,
            "reward_ms": reward_ms,
            "update_ms": update_ms,
            "sync_ms": sync_ms,
            "role_idle_ms": idle_ms,
            "step_time_s": dt,
            "peak_mem_mb": C.peak_mem_mb(device),
            "broadcast_tensors": n_bcast,
            "param_checksum": checksum_after,
            "world_size": info.world_size,
        }
        logger.write(record)
        summaries.append(record)
        print("[grpo %s step %d/%d] rank=%d role=%s loss=%.4f r=%.4f "
              "rollout=%.1fms reward=%.1fms update=%.1fms sync=%.1fms "
              "idle=%.1fms peak_mem_mb=%.1f"
              % (placement, step, steps, info.rank, my_role, loss_value,
                 record["reward_mean"], rollout_ms, reward_ms, update_ms,
                 sync_ms, idle_ms, record["peak_mem_mb"]), flush=True)

    logger.close()
    out = {
        "placement": placement,
        "role": my_role,
        "steps": steps,
        "mean_step_time_s": sum(r["step_time_s"] for r in summaries) / len(summaries),
        "peak_mem_mb": C.peak_mem_mb(device),
        "log_path": logger.path,
    }
    C.cleanup_dist(info)
    return out


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="GRPO 三种角色放置（replicate / policy_fsdp / split_roles）同步版",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    C.add_common_args(p)
    p.add_argument("--placement", type=str, default="replicate", choices=PLACEMENTS)
    p.add_argument("--sharding", type=str, default="full_shard",
                   choices=sorted(SHARDING_STRATEGIES),
                   help="policy_fsdp 时 policy 的分片策略")
    p.add_argument("--n-policy", type=int, default=4, help="split_roles 的 policy 卡数")
    p.add_argument("--n-rollout", type=int, default=2, help="split_roles 的 rollout 卡数")
    p.add_argument("--n-reward", type=int, default=2, help="split_roles 的 reward 卡数")
    p.add_argument("--n-prompts", type=int, default=2)
    p.add_argument("--group-size", type=int, default=4, help="GRPO 每个 prompt 的样本数 G")
    p.add_argument("--prompt-len", type=int, default=16)
    p.add_argument("--max-gen-len", type=int, default=64, help="贪心生成的新 token 数")
    p.add_argument("--use-cache", type=int, default=1, choices=[0, 1],
                   help="1=生成时用 KV cache")
    p.add_argument("--reward", type=str, default="rule", choices=["rule", "model"])
    p.add_argument("--reward-path", type=str, default=None,
                   help="--reward model 时的本地 HF 目录（不下载、不上传）")
    p.add_argument("--rule-mod", type=int, default=7,
                   help="规则奖励里 target token 的模数")
    p.add_argument("--kl-coef", type=float, default=0.0)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--log-name", type=str, default=None)
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.config:
        raise SystemExit("必须给 --config configs/<name>.json")
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
