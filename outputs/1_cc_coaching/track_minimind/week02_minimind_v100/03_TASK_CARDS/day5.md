# Week M02 · Day 5 — 把 GRPO 的三个模型换三种放法量显存与时间，再 kill 掉一个 rank 并从分片 checkpoint 恢复

| 字段 | 值 |
| --- | --- |
| 主要产物 | 一张三行放置对照表（`replicate` / `policy_fsdp` / `split_roles` 的 `mem_ratio`、`step_time_ratio`、`idle_fraction`） |
| 次要产物 | 故障演练证据（`killed_at_step`、两个退出码）与步骤 6 的 8→4 分片恢复回归（门 10；两者都允许 `INCONCLUSIVE`，主要产物不受影响） |
| 估时 | `120 分钟`（步骤估时之和） |
| 环境 | `公司 V100` |
| AI 辅助等级要求 | `A2`（三种放置由脚本跑；"瓶颈在哪一步"与"同步版为什么必然空闲"要能自己论证） |
| 前置 | Day 2 的证据：`verify_pass = true`（分片 checkpoint 能跨 world_size 恢复）；Day 4 的 `mean_load_ratio_is_one = true` |
| 本卡对应门 | `eval` → `恢复/故障` |
| 可裁剪项 | 步骤 3 里的 `split_roles`（允许判 `INCONCLUSIVE`）与步骤 5 的 `async=0` 用例；`policy_fsdp` 与步骤 6 不可裁，Gate 依赖它们 |

## 为什么做（三句以内）

GRPO 一轮里同时存在三个模型：要训练的 policy、冻结的 reference、冻结的 reward，MiniMind 当前的做法是每张卡都加载全部三个。今天要量的是"把 policy 分片"和"把角色拆到不同卡上"这两步各自换来了多少显存、付出了多少等待。最后一步是本周唯一的故障练习：训练中 kill 一个 rank，从 Day 2 那条分片 checkpoint 链路以 4 卡恢复并验证 loss 连续。

## 步骤

### 1. 先过角色分组与故障注入的 CPU 单测门（10 分钟）· `可直接执行`

```bash
export MINIMIND_ROOT=<MiniMind 仓库绝对路径>
export PYTHONPATH=$PWD/lab/src:$PYTHONPATH
python -m pytest lab/tests/test_grpo_roles.py lab/tests/test_faults.py -q
```

- 预期：`test_grpo_roles.py` 14 个用例、`test_faults.py` 13 个用例全部通过（含 2 进程 broadcast 后 `param_checksum` 一致、4 进程角色分组、一个真的会 `os._exit(1)` 的子进程且 exitcode == 1）。
- 不对时先查：`test_grpo_roles.py` 里与 `policy_fsdp` 相关的用例报 `FSDP needs a non-CPU accelerator device` → 那是无 GPU 机器上的预期行为；在公司机器上它不该出现，出现了说明进程没看到 CUDA 设备。

### 2. 第一次跑 `run_grpo_roles.sh`：先用 `MAX_STEPS=2` 试线路（10 分钟）· `模板`

```bash
OUT_DIR=<公司内路径>/day5/wiring MAX_STEPS=2 MAX_GEN_LEN=16 \
  bash lab/scripts/run_grpo_roles.sh
```

脚本里的 `torchrun` 命令在写课程的那台机器上没有端到端执行过（无 GPU，torch wheel 缺 libuv），所以第一次运行先压到 2 步、生成 16 个 token。

- 预期：三次 `torchrun`（`replicate` / `policy_fsdp` / `split_roles`）都进入循环并各写 2 行；header 里 `placement`、`role`、`role_plan`、`reward` 都有值；`REWARD` 默认是 `rule`，用的是内置规则奖励 `r = 0.5 × distinct_ratio + 0.5 × target_ratio`，不依赖任何外部模型。
- 不对时先查：`policy_fsdp` 起不来而另外两种能起来 → 看 FSDP 的 wrap 是否与 Day 2 一致；`split_roles` 卡住 → `dist.new_group` 要求所有进程按相同顺序调用（即使自己不是成员），`grpo_roles.make_role_groups` 是对每个组都调一次再挑自己那个（故障树 F2 第 4 条）。

### 3. 三种放置各跑 5 步（35 分钟）· `模板`

```bash
OUT_DIR=<公司内路径>/day5/gate9 MAX_STEPS=5 MAX_GEN_LEN=64 \
  N_POLICY=4 N_ROLLOUT=2 N_REWARD=2 DTYPE=float16 \
  bash lab/scripts/run_grpo_roles.sh
```

`DTYPE=float16` 原样落到三条 `torchrun` 命令里的 `--dtype float16`；V100 上没有 bf16，改成 `bfloat16` 会在建模型之前被拒绝。

- 预期（`估算`）：`<公司内路径>/day5/gate9/` 下出现 `grpo_{replicate,policy_fsdp,split_roles}_rank<r>.jsonl`；`replicate` 的 `peak_mem_mb` 最高且 `role_idle_ms` ≈ 0；`policy_fsdp` 的参数与优化器显存约为 `replicate` 的 1/8，代价是生成时每层 all_gather 使 `rollout_ms` 上升；`split_roles` 下 8 卡分成 4/2/2，同步版三组严格串行，三组的 idle 之和接近 `2/3 × step_time`；总耗时 6–10 分钟（`估算`）。
- 不对时先查：单次运行超过 12 分钟就 Ctrl-C，先把 `MAX_GEN_LEN` 减半；时间不够时优先保 `policy_fsdp`（Gate 依赖它），`split_roles` 允许判 `INCONCLUSIVE`。

### 4. 算比值并验证权重同步的正确性（20 分钟）· `可直接执行`

算 `mem_ratio_policy_fsdp_vs_replicate`、`mem_ratio_split_roles_vs_replicate`、两个 `step_time_ratio`、三个角色的 `idle_fraction = role_idle_ms / (step_time_s × 1000)`；再核对 `param_checksum`。

- 预期：`split_roles` 下同一 step 里 broadcast 之后**所有 rank 的 `param_checksum` 必须相同**（写课程的机器上用 2 进程简化版验证过 broadcast 后 checksum 一致，`本机实测`）；`reward_std_nonzero_steps` 记下有多少步的 advantage 不是全 0——`reward_std` 为 0 的那一步 advantage 全 0，什么也没学到。`loss` 会在 0 附近正负波动，它**不是**越小越好。
- 不对时先查：`param_checksum` 在各 rank 之间不同 → 按故障树 F12：先确认 `broadcast_policy_weights` 是在 policy 更新**之后**调用的，再确认广播的是未分片的模型副本（FSDP 分片过的模型每个 rank 的 `p.data` 是不同的分片，广播会把 rank0 的分片盖到所有人身上），最后确认浮点 buffer 也广播了。

### 5. kill-rank 故障注入，两个用例对比（25 分钟）· `模板`

```bash
OUT_DIR=<公司内路径>/day5/gate10 KILL_RANK=3 AFTER_STEP=5 NCCL_TIMEOUT_S=60 \
  bash lab/scripts/fault_kill_rank.sh
```

脚本跑两个用例，只差 `TORCH_NCCL_ASYNC_ERROR_HANDLING`：`=1` 时 watchdog 在超时后让其余 rank 崩溃、`torchrun` 退出码非 0；`=0` 时其余 rank 静默阻塞到进程组 timeout。`DTYPE` 默认 `float16`，落到命令上是 `--dtype float16`。

- 预期：被杀 rank 的日志最后一行是 `{"event": "killed", "rank": 3, "step": 5}`；两个用例的 `torchrun` 退出码与从被杀到退出的墙钟秒数都被脚本打印出来，它们的比值就是"崩溃 vs 干等"的差别。写课程的机器上用 2 进程 gloo 复现过同一形态：rank 1 在 step 3 以退出码 1 结束，rank 0 阻塞到超时后被拆掉，全程 11.4 秒（`本机实测`）。
- 不对时先查：kill 之后其余 rank 一直不退出 → 先跑 `python -m mm_dist.faults env --nccl-timeout-s 60` 看当前环境变量与建议值；注意超时时长不是环境变量，环境变量只决定"超时之后做什么"（故障树 F11）。

### 6. 独立跑一次 8→4 分片恢复回归（15 分钟）· `模板`

```bash
OUT_DIR=<公司内路径>/day5/gate6b SAVE_NPROC=8 LOAD_NPROC=4 TOL=1e-2 \
  bash lab/scripts/run_reshard.sh
```

- 预期：`<公司内路径>/day5/gate6b/verify_8to4.json` 里 `"pass": true`、`abs_delta < tol`、`within_tail_band` 为 true（恢复后的 loss 落在中断前最后 20 步的 `[min, max]` 内）；日志里那行 `[reshard] GPU 数量变化(8→4)，... step 20 -> 40` 与 `step*saved_ws//current_ws` 一致。
- 不对时先查：`abs_delta` 超过 `tol` 但模型能加载 → 先确认恢复端的 `--global-batch` 与保存端相同（脚本用 `--accum $((SAVE_NPROC / LOAD_NPROC))` 保住全局批不变），`next_step_loss` 是用保存时的权重在第 21 步那一批全局 batch 上算的 forward-only loss，数据由 `micro_batch_indices` 决定、与 world_size 无关，剩下的差异只可能来自分片恢复本身。

### 7. 填证据字段（5 分钟）· `可直接执行`

对着 `02_LAB_GUIDE.md` 第 6.7 与 6.8 节填：四个比值、三个 `idle_fraction`、`checksum_identical_after_broadcast`、`reward_std_nonzero_steps`；以及 `killed_at_step`、`exit_code_async_on`、`exit_code_async_off`、`wall_seconds_ratio`、`survivor_last_step`、`resume_after_fault_pass`。

- 预期：`policy_fsdp` 那一档必须有完整证据（Gate 依赖它）；`split_roles` 那一档允许写 `INCONCLUSIVE` 并在 `notes` 里说明是时间预算还是别的原因。
- 不对时先查：想把三种放置的绝对显存与绝对 step time 一起记下来时停一下——出公司的只有两两比值、占比和布尔。

## 公司路径差异

- 全部步骤在公司 8×V100 上执行；`policy_fsdp` 与 `run_reshard.sh` 都要 CUDA，写课程的机器上无法验证（那里只验过 `replicate` 的单进程端到端、`split_roles` 的分组与 broadcast 逻辑、kill rank 的真实退出码）。
- **fp16 不是 bf16**：步骤 3、5、6 的 `DTYPE=float16` 都落成 `--dtype float16`；`TOL=1e-2` 是为 fp16 定的。
- 奖励默认用内置规则奖励（`REWARD=rule`），不依赖 1.8B 模型；确实需要模型奖励时用 `REWARD=model REWARD_PATH=<公司内部已有的 HF 目录>`，本 lab 不下载模型。
- 日志、checkpoint、绝对显存、绝对 step time、生成出来的 token 全部留在公司内；离开公司机器的只有下面证据字段里的比值、占比、计数与布尔。

## 今日自测（闭卷，2 题，只记录能/不能）

1. `M02-P-03`（三个模型的显存账；角色放置方案；同步版一轮的完整时序与权重同步字节数；瓶颈在哪一步）
2. `M02-T-04`（全复制 + Policy-FSDP 与角色拆分的取舍；同步版空闲的直接来源；时间不够时保留哪一部分）

## 证据字段（填完即可归档到 memory/cc/evidence/）

```json
{
  "date": "YYYY-MM-DD",
  "week": 2,
  "day": 5,
  "card": "day5",
  "skill": "grpo_role_placement_and_fault_recovery",
  "env": "v100",
  "ai_level": "A2",
  "config_id": "v100_dense.json + placement×3 + reward=rule + fp16 + kill_rank=3",
  "primary_artifact": "三行放置对照表 + <公司内路径>/day5/gate6b/verify_8to4.json",
  "observations": {
    "mem_ratio_policy_fsdp_vs_replicate": "<比值，预期 <1>",
    "mem_ratio_split_roles_vs_replicate": "<比值>",
    "step_time_ratio_policy_fsdp_vs_replicate": "<比值>",
    "step_time_ratio_split_roles_vs_replicate": "<比值>",
    "idle_fraction_policy": "<占比>",
    "idle_fraction_rollout": "<占比>",
    "idle_fraction_reward": "<占比>",
    "checksum_identical_after_broadcast": "true | false",
    "reward_std_nonzero_steps": 0,
    "killed_at_step": 5,
    "exit_code_async_on": 0,
    "exit_code_async_off": 0,
    "wall_seconds_ratio": "<async 关 / async 开>",
    "survivor_last_step": 0,
    "resume_after_fault_pass": "true | false",
    "within_tail_band": "true | false"
  },
  "status": "PASS | FAIL-SYSTEM | INCONCLUSIVE",
  "failure_class": "none | system | measurement | model | insufficient",
  "self_check": {"M02-P-03": "能 | 不能", "M02-T-04": "能 | 不能"},
  "time_spent_min": 0,
  "notes": "<一句话：policy_fsdp 是否有完整证据、split_roles 的判定与原因，不含绝对显存与绝对秒数>"
}
```
