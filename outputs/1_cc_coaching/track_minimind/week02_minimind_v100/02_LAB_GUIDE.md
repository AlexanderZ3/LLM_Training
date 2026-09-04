# Week M02 Lab Guide — 同一个 64M MiniMind 在 8×V100 上的四种放置方式

> 代码基线：MiniMind commit `7a6fddd63a30c06b2fdd5fac4089922b29bc841b`
> 实验包：`lab/`（安装与目录说明见 `lab/README.md`）
> 理论对照：`01_FOUNDATIONS.md`；本周目标与逐日地图：`00_WEEK_CARD.md`
> 命令等级约定：`可直接执行` = 复制即可跑（前提已在该条注明）；`模板` = 必须替换尖括号里的值或环境变量后才能跑；`伪代码` = 描述逻辑，不能直接跑。
> 数值标注约定：`本机实测` = 2026-09-05 在 `ResearchAgentPy310`（Windows / CPU / torch 2.14.0+cpu）上真实跑出来的输出；`估算` = 由已确认参数推出的量级，**未在任何 GPU 上验证**；`待目标机` = 只有公司 8×V100 能产出。

---

## 0. 平台与预算

### 0.1 三台机器各自负责什么

| 环境 | 本周负责的门 | 明确不负责的事 |
| --- | --- | --- |
| 公司 8×V100 32 GB（sm70，torch 2.1.0） | 门 2 单 batch fp16、门 3 smoke、门 4 DDP 等价、门 5 FSDP 对照、门 6 reshard、门 7 EP 扫描、门 8 router 扫描、门 9 GRPO 放置、门 10 故障 | 不新建 conda、不升级 torch、不装包；不产出任何要外发的文件 |
| 本机 CPU（`ResearchAgentPy310`） | 门 0 静态、门 1 CPU 单测（含 2 进程 DDP 等价、2 进程 EP 往返、4 进程角色分组、kill rank 真实退出码） | FSDP（需要 CUDA）、fp16、显存、吞吐、NCCL、torchrun（本机 wheel 无 libuv） |
| 个人 5070 Ti | 本周只做单卡回归，不产出多卡结论 | 不能冒充 8 卡结果 |

判据：**凡是结论里带显存绝对值、tokens/s 绝对值、fp16 数值行为、NCCL 行为的，本机一律不产出。** 本机 `peak_mem_mb` 恒为 `0.0`，`probe.json` 的 `arch_list` 是空列表 `[]`（`本机实测`）。

### 0.2 公司边界（这一节比性能重要）

1. **产物只写 `--out-dir` / `-OutDir`。** `lab/` 里没有任何隐含写入位置，也没有任何上传、同步、外发代码。公司机器上把 `--out-dir` 指向公司内部路径。
2. **不建议、不实现任何绕过公司网络/存储/审计/安装政策的做法。** 需要 reward 模型时用 `--reward model --reward-path <公司内部已有目录>`，本 lab 不下载模型。
3. **从公司带出的只有抽象值**：比值、布尔判定、计数、step 号。具体清单见第 6 节。
4. **不带出**：loss 绝对值序列、tokens/s 绝对值、显存绝对值、profiler trace、截图、`nvidia-smi topo -m` 矩阵、数据样本、权重、checkpoint、CSV 原文、JSONL 原文。
5. `scripts/probe_company.py` 的 `--redact` 默认开启：不写主机名、不写绝对路径；拓扑矩阵**在任何情况下都不写进 probe.json**，只输出连接类型计数直方图与"是否全对称"。

### 0.3 每次运行的时间上限与止损

公司机器上**每次运行 ≤ 10 分钟**。所有入口都有 `--max-steps`，所有脚本都设了 `--nccl-timeout-s`（默认 600，故障演练压到 60）。

| 门 | 默认参数 | 预算（`估算`） | 超时怎么砍 |
| --- | --- | --- | --- |
| 门 4 DDP 等价 | `--max-steps 20` × 3 次 | 6–9 分钟 | 先砍到 `--max-steps 10`，再去掉 ws=2 那次 |
| 门 5 FSDP 对照 | `--max-steps 20` × 4 次 | 8–12 分钟 | 去掉 `no_shard`（它只是对照的对照） |
| 门 6 reshard | save 20 步 + verify 1 次前向 | 3–6 分钟 | `--max-steps 10` |
| 门 7 EP 扫描 | `--max-steps 10` × 4 个 EP | 5–8 分钟 | 只跑 EP=1 与 EP=8 两端 |
| 门 8 router 扫描 | 5×4 组合 × 10 步 | 6–10 分钟 | `--bias-list 0,1,4`、`--aux-coef-list 0,0.01` |
| 门 9 GRPO 放置 | 3 种 × 5 步 × 生成 64 token | 6–10 分钟 | 保 `policy_fsdp`（Gate 依赖），`split_roles` 允许 `INCONCLUSIVE` |
| 门 10 故障 | 2 个用例 × timeout 60s | 3–5 分钟 | 只跑 `async=1` 那个用例 |

止损规则：**一次运行超过 12 分钟就 Ctrl-C，先把 `--max-steps` 减半再跑**，不要等它自己结束。前一门没有证据就不开下一门（`CLAUDE.md` 第 6 节第 6 条）。

---

## 1. 数据与模型来源

### 1.1 代码

| 项 | 值 |
| --- | --- |
| 仓库 | `https://github.com/jingyaogong/minimind.git` |
| commit | `7a6fddd63a30c06b2fdd5fac4089922b29bc841b`（`已确认`，本机克隆后 `git rev-parse HEAD` 核对一致） |
| license | Apache-2.0（MiniMind 仓库根 `LICENSE`） |
| 引用方式 | **不复制进 lab**；用 `MINIMIND_ROOT` 环境变量或 `--minimind-root` 指定 |
| 本 lab 用到的符号 | `MiniMindConfig`、`MiniMindForCausalLM`、`MiniMindModel`、`MiniMindBlock`、`Attention`、`FeedForward`、`MOEFeedForward`、`RMSNorm`（`common.import_minimind` 一次性导入） |

关键事实（`已确认`，读源码核对）：MiniMind **没有 `MoEGate` 类**，router 就是 `MOEFeedForward.gate = nn.Linear(hidden_size, num_experts, bias=False)`（`model/model_minimind.py` L151）；aux loss 公式在 L169–171 是 `(load * scores.mean(0)).sum() * num_experts * router_aux_loss_coef`；空专家分支在 L166 是 `y[0,0] += 0 * sum(p.sum() for p in expert.parameters())`。

`trainer/trainer_utils.py` L110–113：从 `_resume.pth` 恢复时 `step = step * saved_ws // current_ws`。本 lab 的 `ckpt_reshard` 复刻这条换算并在日志里打印。

### 1.2 模型配置（本 lab 自带，不下载权重）

| 配置 | 用途 | 关键参数 | 参数量 |
| --- | --- | --- | --- |
| `configs/tiny_cpu.json` | CPU 单测 / 本机 smoke | hidden 64, 2 层, dense, vocab 6400 | 483,776（`本机实测`，train_ddp header 的 `params_total`） |
| `configs/moe_tiny_cpu.json` | CPU 上的 MoE / EP / router 单测 | hidden 64, 2 层, 4 experts, top-1 | 同量级 |
| `configs/v100_dense.json` | 公司主线 dense | hidden 768, 8 层, seq 512, G=64 | 63.9M（`01_FOUNDATIONS.md` 1.1 节，`计算`） |
| `configs/v100_moe.json` | 公司主线 MoE | 同上 + 4 experts top-1, aux coef 5e-4 | 198M 总 / 约 64M 激活（`计算`） |

EP=8 需要 8 个专家（约束 `ep_size <= num_experts` 且整除）。脚本用 `--num-experts 8` 在命令行覆盖，**不改 config 文件**，这样同一份 config 能同时服务 EP=1/2/4/8。

### 1.3 数据

默认 `--dataset tiny`：`common.TinyDataset` 的合成 token，第 i 条样本只由 `(data_seed, i)` 决定。

为什么本周主要用合成数据，而不是 MiniMind 的语料：fixed-global-batch 等价实验要求"第 s 步消费的样本集合与 world_size 无关"。`common.micro_batch_indices` 把第 s 步的全局区间 `[(s-1)G, sG)` 切成 `world_size × accum` 个等长 chunk，任何 (W, accum) 组合都取到同一批样本——这是不变量 I1 的数据侧条件，用可按索引复现的合成数据最干净，也不需要在公司机器上准备 GB 级语料。

需要真实语料时：`--dataset real --data-path <MINIMIND_ROOT>/dataset/pretrain_t2t_mini.jsonl`，它走 MiniMind 自己的 `dataset/lm_dataset.py::PretrainDataset` + `<MINIMIND_ROOT>/model` 下的 tokenizer。**需要 `datasets` 包**；公司环境没有就留在 `tiny`，本周的所有结论都不依赖真实语料。

### 1.4 Reward 模型（门 9，可选）

`--reward rule`（默认）：内置规则奖励，`r = 0.5 × distinct_ratio + 0.5 × target_ratio`（`target` = 生成 token 里 `id % --rule-mod == 0` 的比例）。完全确定，不依赖任何外部模型。**它不是语言质量指标**，只是让 advantage 非零的信号源。

`--reward model --reward-path <目录>`：用本地已有的 HF 序列分类模型。流程是 MiniMind ids → 文本（MiniMind tokenizer 解码）→ RM tokenizer → RM 打分。目录必须已存在，本 lab 不下载。

### 1.5 离线替代

无网络时全套仍可跑：`--dataset tiny` 不需要下载数据，`--reward rule` 不需要 reward 模型，模型权重是随机初始化的（本周比较的是**放置方式**，不是训练效果）。唯一必须先拿到的是 MiniMind 仓库本身。

---

## 2. 文件依赖图

### 2.1 谁依赖谁

```text
                          MINIMIND_ROOT（外部仓库，只读）
                                    │
                                    ▼
                     src/mm_dist/common.py
      （init_dist / assert_dtype_supported / set_seed / build_model /
        TinyDataset / micro_batch_indices / JsonlLogger / cosine_lr /
        sdpa_context / peak_mem_mb / add_common_args）
          │                │              │              │
          │                │              │              │
          ▼                ▼              ▼              ▼
   train_ddp.py      ep_moe.py      router_stats.py    faults.py
   （compare_logs）   （AllToAllSingle,  （复用 ep_moe.       （kill_rank,
          │            EPMoEFeedForward,  aux_loss_value）    NCCL env）
          │            equivalent_to_dense）      ▲
          ▼                    ▲                 │
   train_fsdp.py               └─────────────────┘
   （SHARDING_STRATEGIES, build_auto_wrap_policy,
     make_mixed_precision, make_fsdp_scaler,
     require_cuda_for_fsdp, CollectiveCounter）
          │                              │
          ├──────────────┬───────────────┘
          ▼              ▼
   ckpt_reshard.py   grpo_roles.py
   （_dcp_save/_dcp_load,  （plan_role_groups, make_role_groups,
     save/load/verify）     broadcast_policy_weights, greedy_generate,
                            group_advantages, rule_reward, ModelReward）
```

脚本层（不被 `src/` 依赖，只调用它）：

```text
scripts/probe_company.py          → common.{TARGET_TORCH_VERSION, MINIMIND_COMMIT,
                                            resolve_minimind_root, minimind_commit}
scripts/audit_minimind_compat.py  → common.{MINIMIND_COMMIT, resolve_minimind_root,
                                            minimind_commit}  ← 只读 MiniMind 源码
scripts/run_ddp_equiv.*           → mm_dist.train_ddp
scripts/run_fsdp_vs_ddp.*         → mm_dist.train_ddp + mm_dist.train_fsdp
scripts/run_reshard.*             → mm_dist.ckpt_reshard
scripts/run_ep_scan.*             → mm_dist.ep_moe
scripts/run_moe_sweep.*           → mm_dist.router_stats
scripts/run_grpo_roles.*          → mm_dist.grpo_roles
scripts/fault_kill_rank.*         → mm_dist.faults（+ 恢复演练接 run_reshard.*）
```

测试层：

```text
tests/conftest.py          提供 SRC_DIR 注入、MINIMIND_ROOT 定位、spawn_dist（mp.spawn）
  ├ test_dtype_guard.py    common + train_fsdp.make_mixed_precision（无进程组）
  ├ test_ddp_gloo_cpu.py   train_ddp.train / compare_logs（1 与 2 进程 gloo）
  ├ test_ep_cpu.py         ep_moe + router_stats.inject_router_bias（1 与 2 进程）
  ├ test_split_sizes.py    ep_moe 纯逻辑（无进程组）
  ├ test_reshard_cpu.py    ckpt_reshard._dcp_save/_dcp_load（2→1 进程）+ CPU 守卫
  ├ test_router_stats.py   router_stats 全部（无进程组）
  ├ test_grpo_roles.py     grpo_roles（无进程组 + 2/4 进程 gloo）
  └ test_faults.py         faults（含一个真的会 os._exit(1) 的子进程）
```

### 2.2 每个本地路径由哪一步创建

| 路径 | 由哪一步创建 | 说明 |
| --- | --- | --- |
| `<MINIMIND_ROOT>/` | 门 0 的 `git clone` + `git checkout` | 外部仓库，本 lab 只读 |
| `<out>/probe.json` | 门 0：`scripts/probe_company.py --out <...>` | 抽象化环境快照 |
| `<out>/compat_audit.md` | 门 0：`scripts/audit_minimind_compat.py --out <...>` | 静态审计报告 |
| `<out>/equiv_ws{1,2,8}_rank<r>.jsonl` | 门 4：`run_ddp_equiv.*` | `--log-name equiv_ws<W>` 决定前缀 |
| `<out>/ddp_ws8_rank<r>.jsonl` | 门 5：`run_fsdp_vs_ddp.*` 的第一步 | FSDP 的对照基线 |
| `<out>/fsdp_{full_shard,shard_grad_op,no_shard}_ws8_rank<r>.jsonl` | 门 5 | 三种策略各一条 |
| `<out>/ckpt_sharded/`（含 `meta.json` 与 dcp 分片文件） | 门 6：`ckpt_reshard save` | `--format full` 时改为 `full_state.pt` + `meta.json` |
| `<out>/verify_8to4.json` | 门 6：`ckpt_reshard verify --result-json` | 只有这个文件里的字段可以带出 |
| `<out>/ep{1,2,4,8}_rank<r>.jsonl` | 门 7：`run_ep_scan.*` | 含 `time_ms` 分解 |
| `<out>/moe_sweep.csv` + `moe_sweep_rank0.jsonl` | 门 8：`run_moe_sweep.*` 的 sweep | CSV 列见 `router_stats.CSV_COLUMNS` |
| `<out>/hot_expert_rank<r>.jsonl` | 门 8 的单点复现 | 逐步曲线 |
| `<out>/grpo_{replicate,policy_fsdp,split_roles}_rank<r>.jsonl` | 门 9：`run_grpo_roles.*` | 每种放置一条 |
| `<out>/fault_async{0,1}_rank<r>.jsonl` | 门 10：`fault_kill_rank.*` | 被杀的 rank 的最后一行是 `{"event":"killed"}` |

**没有任何路径是"假设已经存在"的**：上表每一行都能回溯到前面某一条命令。

---

## 3. 逐门执行

顺序固定，前一门没有证据不开下一门。命令里 `<公司内路径>`、`<MINIMIND 绝对路径>` 必须替换。

### 门 0 — 环境与来源锁定（Day 0，公司 V100）

等级：`模板`。

```bash
git clone https://github.com/jingyaogong/minimind.git
cd minimind && git checkout 7a6fddd63a30c06b2fdd5fac4089922b29bc841b && cd ..
export MINIMIND_ROOT=<MiniMind 绝对路径>

python lab/scripts/probe_company.py --out <公司内路径>/probe.json
python lab/scripts/audit_minimind_compat.py --out <公司内路径>/compat_audit.md
```

**预期观测**（`待目标机`）：

- `versions.torch` == `2.1.0`，`versions.nccl` 有值；
- `devices.device_count` == 8，`devices.homogeneous` == true，每卡 `capability` == `"7.0"`；
- `devices.bf16_supported` == **false**（这是本周所有 `--dtype float16` 的依据）；
- `arch.has_sm_70` == true（若为 false，说明这份 wheel 根本不支持 V100，立刻停下）；
- `sdpa.functional`：`math` 与 `mem_efficient` 为 true，`flash` 大概率为 false（torch 2.1 的 flash 后端只覆盖 sm75–sm90，`已确认`）；
- `topology.distinct_link_types` 若多于 1 种，说明非全对称，门 7 的 EP 分组要注意跨慢链路；
- `minimind.commit_matches` == true。

`compat_audit.md` 的预期（`本机实测`，扫的是同一个 commit，与公司机器上结果一致，因为它是静态扫描）：**9 个 trainer 脚本的 `--dtype` 默认值是 `bfloat16`**（`train_agent / train_distillation / train_dpo / train_full_sft / train_grpo / train_grpo_rule / train_lora / train_ppo / train_pretrain`）；**4 个脚本用了 autocast 但没有 GradScaler**（`train_agent / train_grpo / train_grpo_rule / train_ppo`）；**8 处 `torch.compile` 调用点**；**内置符号清单范围内 0 处 torch 2.1 不存在的 API**。

失败时第一个检查点：`probe_company.py` 退出码非 0 → 看 stdout 里哪一条 `[FAIL]`，把它记成任务卡上的偏差，不要就地装包或改环境。

### 门 1 — CPU 单测（Day 0–1，本机即可）

等级：`可直接执行`（前提：已设 `MINIMIND_ROOT`，已装本机依赖）。

```powershell
$env:MINIMIND_ROOT = "<MiniMind 绝对路径>"
& "D:\Software\Large\Anconda\envs\ResearchAgentPy310\python.exe" -m pytest lab/tests -q
```

**预期观测**（`本机实测` 2026-09-05）：`73 passed, 1 skipped`，耗时约 88 秒。唯一的 skip 是 `test_full_fsdp_reshard_requires_gpu`，原因是 FSDP 需要 CUDA。

失败时第一个检查点：全部用例都 skip 且提示找不到 MiniMind → `MINIMIND_ROOT` 没生效；`PermissionError` 指向临时目录 → 加 `--basetemp <可写目录>`。

### 门 2 — 单 batch 前反向 + fp16（Day 1，公司 V100）

等级：`模板`。先在 1 卡上把 fp16 跑通，再谈多卡。

```bash
export MINIMIND_ROOT=<MiniMind 绝对路径>
export PYTHONPATH=$PWD/lab/src:$PYTHONPATH

# 2a) FP32 reference（1 步）
python -m mm_dist.train_ddp --config lab/configs/v100_dense.json \
  --out-dir <公司内路径>/gate2 --dtype float32 \
  --global-batch 8 --accum 1 --max-steps 1

# 2b) 同样的 batch 换 fp16
python -m mm_dist.train_ddp --config lab/configs/v100_dense.json \
  --out-dir <公司内路径>/gate2 --dtype float16 \
  --global-batch 8 --accum 1 --max-steps 1 --log-name fp16_1step

# 2c) 确认 bf16 被显式拒绝（应当立刻报错退出，不进训练循环）
python -m mm_dist.train_ddp --config lab/configs/v100_dense.json \
  --out-dir <公司内路径>/gate2 --dtype bfloat16 --max-steps 1
```

**预期观测**（`估算`）：2a 与 2b 的 `loss` 相对差 < 1%（随机初始化的模型，第一步 loss ≈ `ln(6400)` ≈ 8.76；本机 CPU 上 dense tiny 的第一步是 8.7691，`本机实测`）；2b 的 `scaler_scale` 起始为 65536，`skipped_steps` 为 0 或 1（第一步跳一次是正常的，见 4.2）；2c 应当**在建模型之前**就抛 `RuntimeError`，消息里含 "V100"、"compute capability 7.0"、"请改用 --dtype float16"。

失败时第一个检查点：2b 的 loss 是 `nan` → 见故障树 F8；2c 没有报错反而开始训练 → `assert_dtype_supported` 没被调用，检查是不是绕过了 `train_ddp.train()`。

### 门 3 — smoke（Day 1，公司 V100，2 卡）

等级：`模板`。

```bash
torchrun --nproc_per_node 2 -m mm_dist.train_ddp \
  --config lab/configs/v100_dense.json --out-dir <公司内路径>/gate3 \
  --dtype float16 --global-batch 16 --accum 1 --max-steps 5 \
  --nccl-timeout-s 600 --log-name smoke_ws2
```

**预期观测**：5 步都有日志行；`world_size` 字段为 2；两个 rank 的 `loss` 字段**完全相同**（它是 all_reduce 之后的量）；`tokens_per_s` 有值。

失败时第一个检查点：卡住不动 → 见故障树 F2；两个 rank 的 loss 不同 → all_reduce 没生效，检查 `--backend`。

### 门 4 — DDP fixed-global-batch 等价（Day 1，公司 V100）

等级：`模板`。

```bash
OUT_DIR=<公司内路径>/gate4 MAX_STEPS=20 bash lab/scripts/run_ddp_equiv.sh
```

脚本内部跑三次（W=1/accum=8、W=2/accum=4、W=8/accum=1，micro 都是 8），再用 `train_ddp --compare-json` 逐步比。

**预期观测**：三次的 header 里 `micro_batch` 必须都是 8；`compare_logs` 输出的 `max|delta| < 1e-4`，判定行是 `PASS`。本机 CPU 上同类实验（tiny 配置，W=1/accum=4 vs W=2/accum=2，micro 都是 2）的实际结果是三步 loss **完全相等**（`本机实测`，`test_fixed_global_batch_equivalence` 通过）。

失败时第一个检查点：第 1 步就差 > 1e-2 → 见故障树 F3。

### 门 5 — FSDP 三策略 vs DDP（Day 2，公司 V100，8 卡）

等级：`模板`。**本机跑不了**（FSDP 需要 CUDA）。

```bash
OUT_DIR=<公司内路径>/gate5 MAX_STEPS=20 bash lab/scripts/run_fsdp_vs_ddp.sh
# 打开 activation checkpointing 再跑一次对照
OUT_DIR=<公司内路径>/gate5_ac AC=1 MAX_STEPS=20 bash lab/scripts/run_fsdp_vs_ddp.sh
```

**预期观测**（全部 `估算`，来自 `01_FOUNDATIONS.md` 3.3/3.4 与 5.1）：

- `fsdp_units` == 9（8 个 `MiniMindBlock` + 根）。**如果是 1，auto_wrap_policy 没生效，后面所有数都不用看了**（故障树 F9）。
- `collectives_total`：DDP 约 11（`bucket_cap_mb=25`，255.6 MB 梯度）；FULL_SHARD 约 27（forward all_gather 9 + backward all_gather 9 + reduce_scatter 9）；SHARD_GRAD_OP 约 18（forward 后不 reshard，backward 前不再 all_gather）；NO_SHARD 与 DDP 同量级。
- `mem_ratio = fsdp.peak_mem_mb / ddp.peak_mem_mb`：FULL_SHARD 应 **< 1**；若 ≥ 1 见故障树 F9。
- `step_time_ratio = fsdp.step_time_s / ddp.step_time_s`：64M 这个尺寸上预期 **> 1**（FSDP 更慢）。这不是字节量问题——每 rank 实际字节 FSDP 335 MB < DDP 447 MB——而是 27 次 vs 11 次的**次数与延迟**问题，以及 all_gather 必须在层计算之前完成、掩盖不住。这条因果链是 Gate 的 A1 题。
- `--activation-checkpointing 1` 后：`activation_checkpointed_blocks` == 8，`peak_mem_mb` 下降，`step_time_s` 上升（多一次前向）。

### 门 6 — 8→4 卡 checkpoint 恢复（Day 2，公司 V100）

等级：`模板`。

```bash
OUT_DIR=<公司内路径>/gate6 SAVE_NPROC=8 LOAD_NPROC=4 TOL=1e-2 \
  bash lab/scripts/run_reshard.sh
# 回退路径也跑一次
OUT_DIR=<公司内路径>/gate6_full FORMAT=full TOL=1e-2 bash lab/scripts/run_reshard.sh
```

**预期观测**：`verify` 输出 `"pass": true`，`abs_delta < tol`；日志里有一行 `[reshard] GPU 数量变化(8→4)，... step 20 -> 40`（换算公式 `step*saved_ws//current_ws`，与 MiniMind `trainer_utils.py` L110–113 一致）。

验证逻辑本身（为什么这样比才有意义）：`save` 时记录的 `next_step_loss` 是**用保存时的权重、在第 21 步那一批全局 batch 上的 forward-only loss**；`verify` 加载后算同一批。数据由 `micro_batch_indices` 决定，与 world_size 无关；loss 先 all_reduce(SUM) 再除以 `world_size*accum`，也与 world_size 无关。剩下的差异只可能来自分片恢复本身。

本机能验证的只有地基部分：`torch.distributed.checkpoint` 的 `save_state_dict`/`load_state_dict` + `FileSystemWriter/Reader` 能不能 2 进程写、1 进程读——`本机实测` 通过（`test_dcp_roundtrip_across_world_sizes`）。

### 门 7 — EP 扫描（Day 3，公司 V100，8 卡）

等级：`模板`。

```bash
# 7a) 先过数值等价（单进程，无通信）——这一步本机也能跑
python -m mm_dist.ep_moe equiv --config lab/configs/v100_moe.json \
  --out-dir <公司内路径>/gate7 --num-experts 8 --batch 2 --seq-len 64 \
  --tol 1e-5 --force-cpu --dtype float32

# 7b) EP=1/2/4/8 扫描
OUT_DIR=<公司内路径>/gate7 NUM_EXPERTS=8 MAX_STEPS=10 bash lab/scripts/run_ep_scan.sh
```

**预期观测**：

- 7a：`[equiv] ep_size=1 vs MOEFeedForward -> PASS  max|diff|=0.000e+00`（`本机实测`：moe_tiny 配置下 max|diff| 恰好为 0）。
- 7b 每个 EP 规模：`input_split_sizes` 之和 == 本 rank 的 token-expert 对数；`output_split_sizes[j]` 必须等于 rank j 的 `input_split_sizes[me]`（不变量 I2，日志里两边都有，可以直接对账）。
- `time_ms` 分解：随 EP 增大，`dispatch_a2a` + `combine_a2a` 的占比上升，`expert_compute` 的占比下降。EP=1 时两个 a2a 恒为 0（代码走的是"不发生通信"的分支）。
- `wait_before_dispatch` 在各 rank 之间的极差就是木桶效应的大小：负载均衡时接近 0，热点时最慢的那张卡接近 0 而其余卡很大。
- `recv_tokens` 的最大/最小比值：均衡期望 ≈ 1，热点时 ≈ E。

一个 MoE 层一步的通信是 **1 次计数交换 + 4 次 all_to_all**（dispatch/combine 各自的前向与反向）。反向那两次来自 `ep_moe.AllToAllSingle` 这个自定义 autograd Function——`dist.all_to_all_single` 本身不可微，不包一层的话反向会在 dispatch 处静默断掉，专家权重仍有梯度、loss 仍会下降，但 router 与下游拿到的梯度是错的。

### 门 8 — 热点专家与 aux 系数扫描（Day 4，公司 V100）

等级：`模板`。

```bash
OUT_DIR=<公司内路径>/gate8 MAX_STEPS=10 \
  BIAS_LIST=0,0.5,1,2,4 AUX_COEF_LIST=0,0.001,0.01,0.1 \
  bash lab/scripts/run_moe_sweep.sh
```

**预期观测**（趋势是 `估算`，口径是 `本机实测`）：

- `bias` 增大 → `max_load_ratio` 从 ≈1 升向 E，`router_entropy` 从 ≈`log E` 降向 0，`mean_wait_ms` 在各 rank 之间分化。本机用 bias=50 做极端验证，`expert_fractions[hot] > 0.99`、`max_load_ratio ≈ E`、`router_entropy < 0.05`（`本机实测`）。
- `aux_coef` 增大 → `max_load_ratio` 回落到 ≈1，但 `mean_logits_loss` 可能变差且不收窄（故障树 F7）。
- 自检项：`mean_load_ratio` 恒等于 1.0（口径没写错的证据）；`sum(expert_fractions)` == 1。

### 门 9 — GRPO 三种角色放置（Day 5，公司 V100）

等级：`模板`。

```bash
OUT_DIR=<公司内路径>/gate9 MAX_STEPS=5 MAX_GEN_LEN=64 \
  bash lab/scripts/run_grpo_roles.sh
```

**预期观测**（`估算`）：

- `replicate`：`peak_mem_mb` 最高（policy + optimizer + rollout 激活 + reward 全在一张卡），`role_idle_ms` ≈ 0（没有跨角色等待）。
- `policy_fsdp`：参数/优化器显存约为 replicate 的 1/8，代价是生成时每层 all_gather，`rollout_ms` 上升。
- `split_roles`：8 卡分成 4/2/2；每张卡的 `role_idle_ms` 就是它等别的角色的时间。同步版下三组严格串行，所以三组的 idle 之和接近 `2/3 × step_time`——这正是"同步版角色拆分的代价"。
- `param_checksum`：`split_roles` 下 broadcast 之后**所有 rank 必须相同**（失败模式 #12 的检查）。本机用 2 进程简化版验证过 broadcast 后 checksum 一致（`本机实测`，`test_broadcast_makes_all_ranks_identical`）。

Gate 只依赖 `policy_fsdp`；`split_roles` 允许判 `INCONCLUSIVE`。

### 门 10 — kill rank 与恢复（Day 5，公司 V100）

等级：`模板`。

```bash
OUT_DIR=<公司内路径>/gate10 KILL_RANK=3 AFTER_STEP=5 NCCL_TIMEOUT_S=60 \
  bash lab/scripts/fault_kill_rank.sh
```

脚本跑两个用例，只差 `TORCH_NCCL_ASYNC_ERROR_HANDLING`：

- `=1`：watchdog 在 timeout 后让其余 rank 崩溃，torchrun 退出码非 0；
- `=0`：其余 rank 静默阻塞到进程组 timeout。

**预期观测**：被杀 rank 的日志最后一行是 `{"event": "killed", "rank": 3, "step": 5}`；两个用例的墙钟时间之比反映"崩溃 vs 干等"的差别。本机用 2 进程 gloo 复现过：rank 1 在 step 3 以退出码 1 结束，rank 0 阻塞到超时后被拆掉，全程 11.4 秒（`本机实测`）。

然后接门 6 的恢复演练：用 4 卡从最后一个 checkpoint `verify`，确认 loss 连续。

---

## 4. 日志与曲线怎么读

所有 JSONL 每行一个 JSON，第一行是 `event: "header"`，之后每个 optimizer step 一行。`--out-dir` 下按 `<log-name>_rank<r>.jsonl` 命名。

### 4.1 header 行（所有 runner 共有）

| 字段 | 含义 | 正常范围 / 检查点 |
| --- | --- | --- |
| `event` | 恒为 `"header"` | — |
| `world_size` | 进程数 | 与 `--nproc_per_node` 一致 |
| `backend` | `nccl` / `gloo` / `none` | 公司机器上应为 `nccl` |
| `device_type` | `cuda` / `cpu` | — |
| `dtype` | 实际使用的精度 | 公司机器上应为 `float16`；`--equiv-check` 会强制成 `float32` |
| `global_batch` / `accum` / `micro_batch` | 批量三元组 | `micro = global_batch / (world_size × accum)`；等价实验里三次运行的 `micro_batch` 必须相同 |
| `seq_len` | 序列长度 | — |
| `params_total` / `params_trainable` | 参数量 | dense v100 配置约 63.9M（`计算`）；tiny_cpu 是 483,776（`本机实测`） |
| `use_moe` | 是否 MoE | — |
| `seed` / `seed_per_rank` | 种子与是否 `42+rank` | **等价实验必须 `seed_per_rank=false`**；`seed_per_rank=true` 是复刻 MiniMind `train_pretrain.py` L112 的行为 |
| `minimind_commit` | 实际使用的 commit | 必须是 `7a6fddd6...` |
| `env` | python / torch / cuda 可用性 / 卡数 / 时间戳 | `torch` 与 `target_torch` 不一致时要在证据里注明 |

`train_fsdp` 额外有：`runner: "fsdp"`、`sharding`、`fsdp_units`（应为层数+1）、`activation_checkpointed_blocks`、`mixed_precision`。
`ep_moe` 额外有：`runner: "ep_moe"`、`ep_size`、`ep_rank`、`ep_groups`、`num_experts`、`experts_per_rank`、`local_expert_ids`、`capacity_factor`。
`grpo_roles` 额外有：`runner: "grpo_roles"`、`placement`、`role`、`role_plan`、`n_prompts`、`group_size`、`prompt_len`、`max_gen_len`、`reward`、`kl_coef`。
`faults.demo` 额外有：`runner: "faults.demo"`、`nccl_timeout_s`、`nccl_env`、`injector`。

### 4.2 训练 step 行（`train_ddp` / `train_fsdp`）

| 字段 | 含义 | 怎么读（范围为 `估算`） |
| --- | --- | --- |
| `step` | 第几个 optimizer step，1-based | — |
| `loss` | **全局** loss = (logits_loss + aux_loss) 先在本 rank 累加 accum 个 micro，再 all_reduce(SUM)，最后除以 `world_size × accum` | 随机初始化的第一步 ≈ `ln(vocab)` ≈ 8.76（vocab 6400）。**只有这个量在不同 world_size 之间可比**；打印 rank0 自己的 micro loss 比不出等价性 |
| `logits_loss` | 交叉熵部分 | dense 时等于 `loss` |
| `aux_loss` | MoE 的负载均衡项，已乘 `router_aux_loss_coef` | 均衡时 ≈ `coef × 1`；坍缩时趋向 `coef × E`。除以 coef 后落在 [1, E] |
| `lr` | 本步学习率 | 与 MiniMind `get_lr` 一致：`base×(0.1+0.45×(1+cos(π·step/total)))`，所以 step=0 时是 `base×1.0`、step=total 时是 `base×0.1` |
| `grad_norm` | clip **之前**的梯度范数（`clip_grad_norm_` 的返回值） | 稳定训练里应在同一量级缓慢下降；突然跳到 `inf`/`nan` 见 F8 |
| `scaler_scale` | GradScaler 当前 scale | fp16 起始 65536；CPU/fp32 恒为 1.0 |
| `skipped_steps` | 累计跳步次数（scale 变小的次数） | 前几步跳 1–2 次正常；**持续每步都跳**说明发散，见 F8 |
| `tokens_per_s` | `global_batch × seq_len / step_time` | 只在公司机器上有意义；本机恒为 CPU 数量级 |
| `step_time_s` | 本步墙钟 | 第 1 步含 CUDA 上下文与 cudnn 预热，**从第 3 步起算平均** |
| `peak_mem_mb` | `torch.cuda.max_memory_allocated`，MB | 本机恒为 `0.0`（无 CUDA） |
| `world_size` | 冗余记录，便于合并多次运行的日志 | — |

`train_fsdp` 额外：`sharding`、`collectives_total`、`collectives`（按类型的字典：`all_gather_into_tensor` / `reduce_scatter_tensor` / `all_reduce` / `all_gather` / `broadcast` / `all_to_all_single`）。计数由 `CollectiveCounter` 通过临时替换 `torch.distributed` 上的公共函数名实现，**只包公共名**（`_all_gather_base` 在 torch 2.1 里是公共名的 deprecated 包装，两个都包会重复计数）。

### 4.3 EP step 行（`ep_moe run`）

| 字段 | 含义 | 怎么读 |
| --- | --- | --- |
| `n_tokens` | 本 rank 本步的 token 数 = batch × seq_len | — |
| `input_split_sizes` | 我发给每个 EP 组员的 token 数，长度 `ep_size` | 之和 == 本 rank 的 (token, expert) 对数（top-1 时就是 `n_tokens`） |
| `output_split_sizes` | 我从每个组员收到的 token 数 | **不变量 I2**：我的 `input_split_sizes[j]` == rank j 的 `output_split_sizes[me]`。不一致不会报错，会 hang |
| `recv_tokens` | `sum(output_split_sizes)` | 各 rank 之间的极差就是负载不均的直接量度 |
| `per_local_expert_tokens` | 本地每个专家收到的 token 数 | 出现 0 是**正常**的（代码里有 `0×sum(p)` 分支保证梯度参与者一致） |
| `global_expert_counts` | 本 rank 路由到每个全局专家的 token 数，长度 E | 除以总数就是本 rank 看到的 `f_e` |
| `dropped_pairs` | 因 capacity 被丢弃的 (token, expert) 对数 | `--capacity-factor 0`（dropless）时恒为 0 |
| `capacity_factor` | 当前设置 | `>0` 时每专家上限 `ceil(cf × N × k / E)`，至少 1 |
| `aux_loss` | 本层 aux loss | 同 4.2 |
| `time_ms` | 九个阶段的耗时字典 | 见下 |

`time_ms` 的九个阶段：`router`（gate 线性层 + softmax）、`topk`、`count_exchange`（元数据 all_to_all）、`permute`（argsort + index_select）、`wait_before_dispatch`（dispatch 前的一次 barrier，量的是"我比别人早到多久"）、`dispatch_a2a`、`expert_compute`（本地 FFN）、`combine_a2a`、`unpermute`（加权 index_add）。

CUDA 上用 `torch.cuda.Event` 计时（kernel 是异步下发的，`perf_counter` 只量得到 launch 时间，量不到 all_to_all 真正的等待）；CPU 上退化到 `perf_counter`。代价是每次 `elapsed_time` 都要 `synchronize`，所以只在 `--time-breakdown` 打开时启用。本机 EP=1 的一次实测分解（`本机实测`，单位 ms，仅示意量级，CPU 上无参考价值）：`expert_compute` 1.76、`router` 1.12、`topk` 0.60、`permute` 0.29、`count_exchange` 0.18、`unpermute` 0.15、两个 a2a < 0.02。

### 4.4 router 统计（`router_stats stats` 的 step 行 / `sweep` 的 CSV）

| 字段 | 含义 | 正常范围 |
| --- | --- | --- |
| `expert_counts` / `expert_fractions` | 每专家 token 数 / 占比（对所有 MoE 层求和） | `sum(fractions) == 1`；counts 之和 == `token 数 × top_k × 层数` |
| `max_load_ratio` | `max(f_e) × E` | 完全均衡 = 1；完全坍缩 = E |
| `min_load_ratio` | `min(f_e) × E` | 均衡 = 1；有专家饿死 = 0 |
| `mean_load_ratio` | 恒等于 1.0 | **自检项**：不等于 1 说明统计口径写错了 |
| `router_entropy` / `router_entropy_max` | 平均 router 分布的熵 / `log E` | 均衡时接近 `log E`（E=4 时 1.386）；坍缩时接近 0 |
| `aux_loss_unit_coef` | coef=1 时的 aux loss | 落在 [1, E]，乘上 `router_aux_loss_coef` 就是训练里加的那一项 |
| `per_layer_fractions` | 逐层的占比 | 某一层坍缩、其余层均衡时只有这里看得出来 |
| `compute_ms` / `wait_ms` | 本 rank 的前反向耗时 / 紧随其后的 barrier 耗时 | 木桶效应：最慢的 rank `wait_ms` ≈ 0，其余 rank 的 `wait_ms` 就是被拖住的时间 |

CSV 的列固定为 `router_stats.CSV_COLUMNS`；`expert_fractions` 在 CSV 里用 `|` 连接成一个字段。

### 4.5 GRPO step 行（`grpo_roles`）

| 字段 | 含义 | 怎么读 |
| --- | --- | --- |
| `placement` / `role` | 放置方式 / 本 rank 的角色 | `replicate` 与 `policy_fsdp` 下 `role` 恒为 `"all"` |
| `loss` | `-(A × logπ).mean()`（+ 可选 KL） | 会在 0 附近正负波动，**不是**越小越好；advantage 均值为 0，所以 loss 的符号不代表好坏 |
| `reward_mean` / `reward_std` | 本步所有样本的奖励 | 规则奖励落在 [0, 1]；`reward_std` 为 0 时 advantage 全 0，这一步什么也没学到 |
| `advantage_absmean` | 优势绝对值均值 | 组内 reward 全相同时为 0 |
| `rollout_ms` / `reward_ms` / `update_ms` / `sync_ms` | 四段耗时 | `split_roles` 下非本角色的段接近 0 |
| `role_idle_ms` | 等别的角色的时间（跨角色 broadcast 的耗时之和） | 同步版的代价，占 `step_time_s` 的比例是门 9 的主要观测 |
| `broadcast_tensors` | 本步广播的张量个数 | `split_roles` 下 = 参数数 + 浮点 buffer 数 |
| `param_checksum` | 全部参数的 float64 求和 | **broadcast 之后所有 rank 必须相同**（失败模式 #12） |

### 4.6 verify 结果（`ckpt_reshard verify --result-json`）

`saved_world_size`、`current_world_size`、`next_step`、`expected_loss`、`actual_loss`、`abs_delta`、`tol`、`within_tail_band`（恢复后的 loss 是否落在中断前最后 20 步的 [min, max] 内）、`pass`。

---

## 5. 故障树

逐条对应 `01_FOUNDATIONS.md` 第 6 节的 12 条失败模式。格式：症状 → 首个检查 → 下一步。

### F1 — 启动即 `RuntimeError: Current CUDA Device does not support bfloat16`（失败模式 #1）

**首个检查**：`python -c "import torch; print(torch.cuda.is_bf16_supported())"`，V100 上应为 `False`；再看命令行有没有 `--dtype bfloat16`。

**下一步**：改成 `--dtype float16`。如果这个错误来自 MiniMind 自己的脚本而不是 lab，那是因为 MiniMind 的 9 个 trainer 默认值就是 `bfloat16`（门 0 的审计报告第 1 节列了名单），每次运行都要显式覆盖。lab 的入口会在建模型**之前**就用 `common.assert_dtype_supported` 拒绝，错误信息里直接给出替代命令；如果你看到的是 CUDA 深处抛出的原始错误，说明有代码绕过了这道守卫。

### F2 — 某 rank 打印 NCCL timeout / watchdog 后退出，或所有 rank 静止（失败模式 #2）

**首个检查**：`py-spy dump --pid <各 rank 的 pid>`，看每个 rank 卡在哪个 collective。**卡在不同 collective = 调用序列不一致**；卡在同一个 = 等一个永远来不了的字节数。

**下一步**：
1. 设 `TORCH_NCCL_ASYNC_ERROR_HANDLING=1`，让超时变成崩溃而不是静默 hang（`lab/scripts/fault_kill_rank.*` 的两个用例就是在对比这两种行为）；
2. 把 `--nccl-timeout-s` 从 600 压到 60，先让它快点失败；
3. 检查 MoE 的空专家分支：某 rank 的某个专家一个 token 都没收到时，如果没有 `0×sum(p)` 那一项，它的参数不进计算图，梯度归约的参与者就不一致。MiniMind L166 与 `ep_moe.py` 的本地 expert 循环里都有这一项；
4. 检查 `dist.new_group`：**所有进程必须按相同顺序调用，即使自己不是成员**。`ep_moe.make_ep_groups` 与 `grpo_roles.make_role_groups` 都是对每个组都调一次再挑自己那个。

### F3 — 不同 world_size 的 loss 在第 1 步就差 > 1e-2（失败模式 #3，破坏 I1）

**首个检查**：三次运行 header 里的 `micro_batch` 是否相同。不同就不是同一个数值问题（batch 维度上的归约顺序变了）。

**下一步**：
1. `seed_per_rank` 必须是 `false`。MiniMind `train_pretrain.py` L112 用的是 `setup_seed(42+rank)`，那会让每个 rank 的 dropout / 初始化不同——做等价实验必须关掉，`--equiv-check` 会自动关；
2. `dtype` 必须是 `float32`。fp16 下不同的归约顺序会带来 1e-3 量级的差异，比判定阈值还大；`--equiv-check` 会强制 fp32；
3. 打印每步各 rank 的样本索引并集，与 `[(s-1)G, sG)` 对账。`tests/test_ddp_gloo_cpu.py::test_micro_batch_indices_partition_global_batch` 就是在做这件事，本机对 6 种 (W, accum) 组合都验证过；
4. 看 `scaler_scale`：两次运行的 scale 序列不同意味着跳步位置不同，loss 自然对不上。

### F4 — 从 8 卡分片 ckpt 以 4 卡恢复时 `size mismatch` 或 flat_param 长度错（失败模式 #4）

**首个检查**：`meta.json` 里的 `format` 与恢复时用的是否一致；恢复时的 FSDP wrap 结构（`--sharding`、config、`--activation-checkpointing`）是否与保存时**完全相同**。分片是 `(world_size, wrap 粒度)` 的函数，wrap 变了分片就对不上。

**下一步**：
1. `--format sharded` 才是跨 world_size 的正路（`SHARDED_STATE_DICT` + `torch.distributed.checkpoint`）；
2. 优化器状态必须走 `FSDP.optim_state_dict_to_load(model, optim, sd)` 重新切分，直接 `optimizer.load_state_dict(sd)` 一定错；
3. 实在不行用 `--format full` 回退（`FULL_STATE_DICT` + `rank0_only`），代价是 rank0 要装得下整份模型 + 优化器；
4. 恢复端的 FSDP 构造必须带 `sync_module_states=True`，否则 rank0_only 的完整 state dict 广播不出去。

### F5 — EP 训练 hang，无报错（失败模式 #5，破坏 I2）

**首个检查**：每个 rank 打印 `input_split_sizes` 与 `output_split_sizes`（`ep_moe` 的日志里本来就有），逐对核对：rank r 的 `input_split_sizes[j]` 必须 == rank j 的 `output_split_sizes[r]`。

**下一步**：
1. `ep_moe._a2a` 里有本地断言 `sum(input_split_sizes) == send.shape[0]`，它能抓住一半的错误（本地不自洽）；跨 rank 的不一致只能靠上面的对账；
2. 先用 gloo 2 进程 CPU 复现：`pytest lab/tests/test_ep_cpu.py -q`。本机的 `test_ep_two_ranks_roundtrip` 会把两个 rank 的 split 表交叉断言一遍；
3. 检查计数交换是不是漏了。接收缓冲区的长度**只能由对端告诉我**，所以数据 all_to_all 之前必须先有一次元数据 all_to_all；
4. 检查 `--ep-size` 与 `--num-experts`：`ep_size <= num_experts` 且整除，`world_size` 能被 `ep_size` 整除。`ep_moe.validate_ep` 会在建组前报错，如果你是在通信时才挂住，说明绕过了它。

### F6 — step time 随训练变长，`wait_ms` 集中在除某一个 rank 之外的所有 rank（失败模式 #6）

**首个检查**：`router_stats` 的 `max_load_ratio`。≈1 是均衡，趋向 E 是坍缩。

**下一步**：四步定位链（Gate 的跨场景 debug 题）：`wait_ms` 分布 → `expert_fractions`（哪个专家是热点）→ `aux_loss / coef`（落在 [1, E] 的哪里）→ `router_entropy`（是否被拉平）。确认是热点后，加大 `router_aux_loss_coef`（门 8 的扫描就是在找这个系数），并注意 F7。

### F7 — 提高 λ 后 aux_loss 降到 ≈1 但 logits_loss 比 dense 差且不收窄（失败模式 #7）

**首个检查**：对比 `--aux-coef 0` 与 dense 基线的 `mean_logits_loss`。

**下一步**：看 `router_entropy` 是不是被推到接近 `log E`——那意味着 router 被强制均匀分配，失去了"分工"这个 MoE 存在的理由。均衡与分工是一对权衡：主指标看 `mean_logits_loss`，`max_load_ratio` 只是约束条件，不是优化目标。门 8 的 CSV 把这两列放在一起就是为了这个判断。

### F8 — scale 从 65536 一路减半到 <1，loss 打印 NaN（失败模式 #8）

**首个检查**：`skipped_steps` 是不是每步都在涨；`grad_norm` 是不是 `inf`/`nan`。

**下一步**：
1. 先回 `--dtype float32` 跑同样的配置。如果 fp32 也发散，那是模型/学习率问题，不是 fp16 问题；
2. fp32 正常而 fp16 发散：降 `--lr`，或检查有没有在 autocast 里做了不该做的事（fp16 的动态范围是 6e-5 到 65504）；
3. FSDP + fp16 必须用 `ShardedGradScaler`（`train_fsdp.make_fsdp_scaler` 已经处理）：普通 `GradScaler.unscale_` 只看本 rank 的梯度分片，inf 检查必须跨 rank 归约，否则各 rank 对"要不要跳这一步"的判断会不一致。

### F9 — FSDP 显存**高于** DDP（失败模式 #9）

**首个检查**：日志 header 的 `fsdp_units`。应为 `层数 + 1`（v100_dense 是 9）。等于 1 说明 `auto_wrap_policy` 没切开任何 block，整个模型是一个 FSDP 单元，forward 峰值就是全模型——那 FSDP 什么也没省。`train_fsdp` 在 `fsdp_units <= 1 且 world_size > 1` 时会主动打一行 `[warn]`。

**下一步**：
1. 确认 `transformer_auto_wrap_policy(transformer_layer_cls={MiniMindBlock})` 里的类是从 `common.import_minimind()` 拿到的**同一个类对象**——从别处 import 的同名类不是同一个对象，`isinstance` 会失败；
2. 确认 `limit_all_gathers=True`（限制同时在飞的 all_gather 数量，防止预取把显存吃光）；
3. 确认 `use_orig_params` 的设置与优化器构造顺序一致；
4. 用 `--sharding no_shard` 跑一次作对照：它的显存应当与 DDP 接近，如果 `no_shard` 就已经比 DDP 高，问题不在分片策略。

### F10 — 恢复后 step 数不对（失败模式 #10）

**首个检查**：`ckpt_reshard load/verify` 打印的那行 `[reshard] GPU 数量变化(A→B) ... step X -> Y`，核对 `Y == X * A // B`。

**下一步**：这条换算（MiniMind `trainer_utils.py` L110–113）**只在"每 step 每 rank 样本数不变"时成立**。本 lab 固定 `--global-batch`，所以真正消费的样本数不随 world_size 变，换算后的 step 只用于对齐 MiniMind 的日志语义，不改变数据游标。如果你改的是 micro batch 而不是 accum，global batch 就变了，这时要按 **token 数**重算而不是按 step 数换算。

### F11 — kill 一个 rank 后其余 rank 不退出（失败模式 #11）

**首个检查**：`TORCH_NCCL_ASYNC_ERROR_HANDLING` 的值（`python -m mm_dist.faults env` 会打印当前值与建议值）。

**下一步**：没设时其余 rank 会阻塞到 `init_process_group(timeout=...)`（本 lab 由 `--nccl-timeout-s` 控制，默认 600 秒；torch 默认是 1800 秒）。设成 1 之后 watchdog 会让它们崩溃，torchrun 才能感知并结束整个 job。门 10 的两个用例就是在量这个差别。注意**超时时长不是环境变量**，环境变量只决定"超时之后做什么"。

### F12 — 角色拆分版 rollout 用的权重滞后一轮（失败模式 #12）

**首个检查**：`grpo_roles` 日志里各 rank 的 `param_checksum`。同一 step 里所有 rank 必须相同。

**下一步**：
1. 确认 `broadcast_policy_weights` 是在 policy 更新**之后**调用的（`run()` 里的第 4 段）；
2. 确认广播的是**未分片**的模型：FSDP 分片过的模型不能这样广播，每个 rank 的 `p.data` 是不同的分片，广播会把 rank0 的分片盖到所有人身上。`split_roles` 用的是普通模型副本，正是为了这个；
3. 确认 buffers 也广播了（`broadcast_policy_weights` 会广播浮点 buffer 与整型 buffer）。

### F13 — 本机 `torchrun` 报 `use_libuv was requested but PyTorch was built without libuv support`（环境问题，非代码缺陷）

**首个检查**：是不是在本机 Windows 上跑 `torchrun`。

**下一步**：这台机器的 torch 2.14 CPU wheel 编译时没带 libuv，而 torchrun 的 rendezvous 会创建 libuv 版 TCPStore，`USE_LIBUV=0` 压不住（该开关只影响用户自建的 TCPStore）。本机的多进程验证一律走 `torch.multiprocessing.spawn`——`pytest lab/tests` 就是这么做的。公司 Linux 机器上 `torchrun` 照常可用，`scripts/*.sh` 与 `scripts/*.ps1` 里的命令是给目标机的。

### F14 — pytest 报 `PermissionError` 指向临时目录（环境问题）

**首个检查**：报错路径是不是 `.../Temp/pytest-of-<user>`。

**下一步**：加 `--basetemp <一个可写目录>`。本机验证时用的就是这个办法。

---

## 6. 证据清单

本周需要提交的字段。**全部是抽象值**：比值、布尔、计数、step 号。不提交任何绝对性能数字、日志原文、CSV 原文、拓扑矩阵、截图。

### 6.1 环境与来源（门 0）

| 字段 | 类型 | 来源 |
| --- | --- | --- |
| `torch_matches_target` | bool | `probe.json` → `versions.torch_matches_target` |
| `device_count` | int | `probe.json` → `devices.device_count` |
| `capability_uniform` | bool | `probe.json` → `devices.homogeneous` |
| `bf16_supported` | bool（预期 false） | `probe.json` → `devices.bf16_supported` |
| `has_sm_70` | bool | `probe.json` → `arch.has_sm_70` |
| `sdpa_functional` | 三个 bool（flash / mem_efficient / math） | `probe.json` → `sdpa.functional` |
| `topology_link_types` | 字符串列表（如 `["NV1","NV2"]`） | `probe.json` → `topology.distinct_link_types` |
| `topology_fully_symmetric` | bool | `probe.json` → `topology.fully_symmetric` |
| `minimind_commit_matches` | bool | `probe.json` → `minimind.commit_matches` |
| `bf16_default_scripts` | int（预期 9） | `compat_audit.md` 第 1 节的行数 |
| `autocast_without_scaler` | int（预期 4） | `compat_audit.md` 第 2 节结论行 |
| `torch_compile_sites` | int（预期 8） | `compat_audit.md` 第 3 节的行数 |
| `newer_api_hits` | int（预期 0） | `compat_audit.md` 第 4 节 |

### 6.2 DDP 等价（门 4）

| 字段 | 类型 |
| --- | --- |
| `micro_batch_identical` | bool（三次运行的 `micro_batch` 是否相同） |
| `max_abs_delta_ws1_ws2` | float（科学计数，只报数量级即可） |
| `max_abs_delta_ws1_ws8` | float |
| `tol` | float（1e-4） |
| `equiv_pass` | bool |
| `steps_compared` | int |

### 6.3 FSDP 对照（门 5）

| 字段 | 类型 |
| --- | --- |
| `fsdp_units` | int（预期 9） |
| `collectives_total_ddp` / `_full_shard` / `_shard_grad_op` / `_no_shard` | int ×4 |
| `step_time_ratio_full_shard_vs_ddp` | float（比值，不报绝对秒数） |
| `step_time_ratio_shard_grad_op_vs_ddp` | float |
| `mem_ratio_full_shard_vs_ddp` | float（比值，不报绝对 MB） |
| `mem_ratio_shard_grad_op_vs_ddp` | float |
| `ac_blocks` | int（打开 activation checkpointing 时预期 8） |
| `ac_mem_ratio` / `ac_step_time_ratio` | float ×2（同配置 AC 开 vs 关） |

### 6.4 checkpoint 恢复（门 6）

| 字段 | 类型 |
| --- | --- |
| `saved_world_size` / `current_world_size` | int ×2 |
| `format` | `"sharded"` / `"full"` |
| `abs_delta` / `tol` | float ×2 |
| `verify_pass` | bool |
| `within_tail_band` | bool |
| `converted_step_correct` | bool（`step*saved_ws//current_ws` 是否与日志一致） |

### 6.5 EP（门 7）

| 字段 | 类型 |
| --- | --- |
| `ep1_equiv_max_abs_diff` / `ep1_equiv_pass` | float / bool |
| `split_invariant_holds` | bool（I2 对账结果） |
| `a2a_time_fraction[ep]` | 每个 EP 规模一个 float：`(dispatch+combine) / sum(time_ms)` |
| `expert_compute_fraction[ep]` | 同上 |
| `wait_spread_ratio[ep]` | float：`max(wait_before_dispatch) / mean(...)` 跨 rank |
| `recv_token_imbalance[ep]` | float：`max(recv_tokens) / min(recv_tokens)` 跨 rank |
| `zero_token_expert_observed` | bool |

### 6.6 router 扫描（门 8）

| 字段 | 类型 |
| --- | --- |
| `max_load_ratio[bias][coef]` | 二维表（每格一个 float，落在 [1, E]） |
| `router_entropy[bias][coef]` | 二维表（每格一个 float，落在 [0, log E]） |
| `logits_loss_ratio[bias][coef]` | 二维表（相对 `bias=0, coef=0` 的比值，**不报绝对 loss**） |
| `wait_ms_ratio[bias][coef]` | 二维表（相对基线的比值） |
| `mean_load_ratio_is_one` | bool（统计口径自检） |
| `expert_fractions_sum_is_one` | bool（每格的 `sum(expert_fractions)` == 1） |
| `hot_expert_reproduced` | bool（`bias` 最大那档的 `max_load_ratio` 显著高于 `bias=0`） |
| `sweep_cells_completed` | int（实际跑完的格子数，满格 = len(BIAS_LIST) × len(AUX_COEF_LIST)） |

### 6.7 GRPO 放置（门 9）

| 字段 | 类型 |
| --- | --- |
| `mem_ratio_policy_fsdp_vs_replicate` | float |
| `mem_ratio_split_roles_vs_replicate` | float |
| `step_time_ratio_policy_fsdp_vs_replicate` | float |
| `step_time_ratio_split_roles_vs_replicate` | float |
| `idle_fraction[role]` | 三个 float（`role_idle_ms / step_time_s / 1000`） |
| `checksum_identical_after_broadcast` | bool |
| `reward_std_nonzero_steps` | int（有多少步的 advantage 不是全 0） |

### 6.8 故障（门 10）

| 字段 | 类型 |
| --- | --- |
| `killed_at_step` | int |
| `exit_code_async_on` / `exit_code_async_off` | int ×2 |
| `wall_seconds_ratio` | float（async 关 / async 开） |
| `survivor_last_step` | int |
| `resume_after_fault_pass` | bool（接门 6 的 verify） |

### 6.9 本机验证（贯穿全周）

| 字段 | 类型 |
| --- | --- |
| `pytest_passed` / `pytest_skipped` | int ×2 |
| `py_compile_all_pass` | bool |
| `skip_reasons` | 字符串列表 |

---

## 7. 本机验证状态（诚实记录）

环境：Windows 11，`D:\Software\Large\Anconda\envs\ResearchAgentPy310\python.exe`（Python 3.10.20），torch 2.14.0+cpu，transformers 4.57.6，numpy 2.2.6，pytest 9.1.1，**无 NVIDIA GPU**。MiniMind 克隆于 scratchpad，`git rev-parse HEAD` == `7a6fddd63a30c06b2fdd5fac4089922b29bc841b`（核对一致）。日期 2026-09-05。

### 7.1 pytest（已执行，通过）

```text
$ python -m pytest lab/tests -q --basetemp <可写目录>
..................................................s..................... [ 97%]
..                                                                       [100%]
73 passed, 1 skipped, 3 warnings in 88.15s (0:01:28)
```

3 条 warning 全部是同一条 `FutureWarning: torch.cuda.amp.GradScaler(args...) is deprecated`，来自 `common.py:279`。这是**有意保留**的：目标机是 torch 2.1.0，那里 `torch.amp.GradScaler('cuda', ...)` 的新签名还不存在，改成新写法会在公司机器上直接报错。

逐文件结果：

| 文件 | 用例数 | 结果 |
| --- | --- | --- |
| `test_dtype_guard.py` | 10 | 全部通过 |
| `test_ddp_gloo_cpu.py` | 4 | 全部通过（含 1 进程 vs 2 进程 gloo 的 I1 等价） |
| `test_ep_cpu.py` | 7 | 全部通过（含 2 进程 EP 往返、空专家） |
| `test_split_sizes.py` | 11 | 全部通过 |
| `test_reshard_cpu.py` | 6 | 5 通过，1 skip |
| `test_router_stats.py` | 9 | 全部通过 |
| `test_grpo_roles.py` | 14 | 全部通过（含 2 进程 broadcast、4 进程角色分组） |
| `test_faults.py` | 13 | 全部通过（含真的 `os._exit(1)` 的子进程，exitcode == 1） |

### 7.2 skip 项与原因（只有 1 条）

| 用例 | 原因 |
| --- | --- |
| `test_reshard_cpu.py::test_full_fsdp_reshard_requires_gpu` | FSDP1 需要 CUDA 设备。torch 的 `_init_device_handle` 在参数全在 CPU 上时回落到 `torch.cuda.current_device()`，无 GPU 时抛 `FSDP needs a non-CPU accelerator device`；torch 2.1.0 与本机 2.14 行为一致。完整的"2 卡保存 → 1 卡 verify loss 连续"链路必须在公司 8×V100 上执行。 |

另外两条与 MiniMind 相关的 skip **没有触发**（因为本机已克隆 MiniMind）：`conftest.py` 的 `minimind_root` fixture 在找不到仓库时会 skip 所有依赖它的用例并给出 clone 命令。

### 7.3 py_compile（已执行，全部通过）

```text
$ find lab -name "*.py" | xargs python -m py_compile
（无输出 = 全部通过）
```

覆盖 20 个文件：`src/mm_dist/` 下 9 个（`__init__`、`common`、`train_ddp`、`train_fsdp`、`ckpt_reshard`、`ep_moe`、`router_stats`、`grpo_roles`、`faults`）、`scripts/` 下 2 个（`probe_company`、`audit_minimind_compat`）、`tests/` 下 9 个（`conftest` + 8 个 test 文件）。

### 7.4 本机实际执行过的 CLI（全部成功）

| 命令 | 结果 |
| --- | --- |
| `python -m mm_dist.{train_ddp,train_fsdp,ckpt_reshard,ep_moe,router_stats,grpo_roles,faults} --help` | 7 个入口全部正常打印帮助 |
| `python -m mm_dist.train_ddp --config configs/tiny_cpu.json --force-cpu --dtype float32 --global-batch 8 --accum 4 --seq-len 32 --max-steps 3` | 3 步完成；`params_total=483776`，loss 8.7691 / 8.7633 / 8.7780，`scaler_scale=1`（CPU 上 scaler disabled），`peak_mem_mb=0.0` |
| `python -m mm_dist.ep_moe equiv --config configs/moe_tiny_cpu.json --force-cpu --dtype float32 --tol 1e-5` | `PASS  max|diff|=0.000e+00` |
| `python -m mm_dist.ep_moe run --ep-size 1 --time-breakdown --max-steps 2` | 2 步完成；`per_expert=[15,17,14,18]`、`dropped=0`，`time_ms` 九个阶段都有值 |
| `python -m mm_dist.faults env --nccl-timeout-s 60` | 打印当前 NCCL 环境变量与 5 行建议 |
| `python lab/scripts/probe_company.py --out <scratchpad>/probe.json` | 生成 probe.json；退出码 1（本机无 GPU，4 项 FAIL：torch 版本、sm_70、卡数、同构性；2 项 PASS：bf16 不支持、MiniMind commit） |
| `python lab/scripts/audit_minimind_compat.py --out <scratchpad>/compat_audit.md` | 扫描 15 个文件：默认 bfloat16 = 9，autocast 无 scaler = 4，torch.compile 调用点 = 8，新 API 命中 = 0 |
| 2 进程 gloo 的 `faults.cmd_demo`（经 `mp.spawn`，非 torchrun） | rank 1 在 step 3 以退出码 1 结束；rank 0 阻塞至超时后被拆掉；全程 11.4 秒 |

### 7.5 未能在本机验证的项（全部 `待目标机`）

| 项 | 原因 |
| --- | --- |
| `train_fsdp.py` 的训练路径（三种 sharding、`CollectiveCounter` 的实际计数、activation checkpointing） | FSDP1 需要 CUDA |
| `ckpt_reshard.py` 的 FSDP 保存/恢复/verify 完整链路 | 同上（其地基 `_dcp_save`/`_dcp_load` 的 2→1 进程往返已在本机通过） |
| `grpo_roles.py --placement policy_fsdp` | 同上（`replicate` 的单进程端到端已在本机通过；`split_roles` 的分组与 broadcast 逻辑已在本机通过） |
| 任何 fp16 / GradScaler 的真实数值行为（跳步、scale 衰减、溢出） | 本机 CPU 上 `autocast` 是 no-op、`GradScaler` 是 disabled |
| 任何 NCCL 行为（timeout、watchdog、async error handling 的实际效果） | 本机只有 gloo |
| 任何显存与吞吐数字（`peak_mem_mb`、`tokens_per_s`） | 本机 `peak_mem_mb` 恒为 0.0 |
| SDPA 三后端的实际可用性 | 需要 CUDA；`probe_company.py` 在无 CUDA 时把三项都记为 `null`（未知，不是不支持） |
| `scripts/*.sh` 与 `scripts/*.ps1` 里的 `torchrun` 命令 | 本机 torch wheel 无 libuv，torchrun 无法启动（见故障树 F13）。脚本的参数拼装经过人工核对，但**未在任何机器上端到端执行过**；第一次在公司机器上运行时请先用 `MAX_STEPS=2` 试一遍 |
| 8 卡拓扑对 EP 分组的影响 | 需要公司机器的 `nvidia-smi topo -m` |

### 7.6 结论

- **可以现在就做的**：门 0（探针 + 审计）、门 1（CPU 单测）——两者本机/公司都能跑，且都已在本机验证过命令本身。
- **必须在公司机器上做的**：门 2 到门 10 的全部。
- **绝不能声称的**：任何 fp16、显存、吞吐、NCCL、FSDP、8 卡的结论。本文件里所有这类数字都标了 `估算` 或 `待目标机`。
