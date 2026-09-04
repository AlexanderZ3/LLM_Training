# Week M02 — MiniMind 八卡 V100 训练系统实验台（一页周卡）

> 状态：`BUILT / RUNTIME-UNVERIFIED`（文档与 CPU 门通过；NCCL / fp16 / 8 卡门待公司机器）  
> 生成日期：2026-09-04 · 构建完成：2026-09-05 · 生成方：cc · 轨道：`track_minimind`  
> 来源：`input_info/minimind_5070ti_v100.md`（用户意图）+ 2026-09-04 来源审计

## 本周唯一主问题

同一个 64M MiniMind 在 8×V100 上，DDP / FSDP / EP（专家并行）/ GRPO 角色拆分各自把参数、梯度、token 和通信放在哪里，哪一个决定了 step time？

## 核心实现 · 关键故障练习 · Gate

| 项 | 内容 |
| --- | --- |
| 核心实现 | `lab/src/mm_dist/ep_moe.py`：在 MiniMind 的 `MOEFeedForward`（4 experts，top-1，`router_aux_loss_coef=5e-4`，无 capacity/drop/EP）之上实现教学版 EP：router → 每 rank token 计数 → permutation → `all_to_all_single` → 本地 expert → 逆 all_to_all → unpermute 加权合并；带每 rank 计算/等待计时 |
| 关键故障练习 | 训练中 kill 一个 rank，从 FSDP 分片 checkpoint 以 4 卡恢复并验证 loss 连续 |
| Gate | A0 闭卷 30 分钟：画出 EP=4 / DP=2 时每张卡的 group 归属与一次 MoE 层的通信序列；A1 解释“64M 上 FSDP 比 DDP 慢”的原因链；跨场景 debug：热点专家导致 step time 拉长；两个不同日期的证据 |

## 先修

- Week M01 Gate `PASS`。本周 lab 自带日志与吞吐统计（`mm_dist/common.py`），不 import M01 的 `mm_probe/`；两周目录相互独立。
- 公司环境已用 Day 0 探针 `已确认`（当前全部为 `用户自述待核验`：8×V100 32 GB、PyTorch 2.1.0、既有 Conda）。

## 硬事实（已核验，2026-09-04，决定本周所有命令）

- **BF16 在 V100 + torch 2.1 直接报错**：`torch.cuda.is_bf16_supported()` 要求 major ≥ 8；`autocast(dtype=bfloat16)` 抛 `RuntimeError: Current CUDA Device does not support bfloat16`。MiniMind 所有脚本默认 `--dtype bfloat16`，本周一律 `--dtype float16`；`train_grpo.py` 没有 GradScaler，需要在 lab 里补。
- **SDPA 后端**：torch 2.1 的 flash 后端只覆盖 sm75–sm90，V100（sm70）走 mem-efficient（支持 fp16/fp32）或 math；无 FlashAttention-2。
- **torch 2.1 FSDP API 齐全**：`ShardingStrategy.FULL_SHARD/SHARD_GRAD_OP/HYBRID_SHARD`、`MixedPrecision`、`StateDictType.FULL/LOCAL/SHARDED` + `state_dict_type()`、`optim_state_dict/optim_state_dict_to_load`、`torch.distributed.checkpoint` save/load、`apply_activation_checkpointing`；`all_to_all_single(output, input, output_split_sizes, input_split_sizes, group)`。
- **不用 DeepEP**（要求 SM90、torch ≥ 2.10）、**不用 MegaBlocks**（要求 torch 2.7.x）。Tutel 要求 torch ≥ 2.0 但 README 未列 Volta；DeepSpeed-MoE `torch>=2.0`，API `MoE(hidden_size, expert, num_experts, ep_size, k, capacity_factor, min_capacity, drop_tokens, use_tutel, ...)` 返回 `(output, l_aux, exp_counts)`。两者只作为 Day 3–4 的可选对照，主线是自写教学版 EP。
- **torch.compile**：Triton 2.1 声称 CC ≥ 7.0，V100 满足但未实测；本周默认 `use_compile=0`。
- MiniMind 多卡：所有脚本 `init_distributed_mode()` 读 `RANK/LOCAL_RANK`，NCCL，`DistributedSampler + DDP`；`_resume.pth` 保存 `world_size`，恢复时 step 按比例换算。
- 公司机器的 torch 2.1.0 wheel 的 CUDA 版本、`get_arch_list()` 是否含 sm_70、NCCL 版本、拓扑、transformers 版本：全部 `未知`，Day 0 探针。

## 逐日地图

| Day | 目标 | 主要产物 | 门 | 估时 | 环境 |
| --- | --- | --- | --- | --- | --- |
| 0 | 公司探针 + MiniMind 对 torch 2.1 的兼容审计 | `lab/scripts/probe_company.py`（版本、8 卡 capability、bf16=False、NCCL、拓扑抽象化）+ `<公司内路径>/day0/compat_audit.md`（bf16→fp16、compile 关、SDPA 回退、依赖 dry-run；审计输出留在公司内，不落仓库） | 静态 | 90 | 公司 V100 |
| 1 | 单卡 FP16 → 2 卡 → 8 卡 DDP fixed-global-batch 等价 | `mm_dist/train_ddp.py` + `tests/test_ddp_gloo_cpu.py` + `scripts/run_ddp_equiv.*` | CPU 单测 → FP32 ref → FP16 → 2 卡 → 8 卡 | 90 | 公司 V100（本机 CPU gloo 2 进程先过） |
| 2 | FSDP FULL_SHARD vs DDP 对照 + 8→4 卡 checkpoint 恢复 | `mm_dist/train_fsdp.py` + `mm_dist/ckpt_reshard.py` + `tests/test_reshard_cpu.py` | 有界训练 → 恢复 | 120 | 公司 V100 |
| 3 | 教学版 EP + 8 卡运行 EP=1/2/4/8 | `mm_dist/ep_moe.py` + `tests/test_ep_cpu.py`（gloo 2 进程 permute/unpermute 往返等价，EP=1 与原实现数值一致）+ `scripts/run_ep_scan.*` | CPU 单测 → 单 batch → 短 smoke | 120 | 公司 V100 |
| 4 | 热点专家（router bias 0/0.5/1/2/4）与 aux-loss 系数（0/1e-3/1e-2/1e-1）扫描 | `mm_dist/router_stats.py` + `scripts/run_moe_sweep.*` | 有界训练 → 消融 | 90 | 公司 V100 |
| 5 | GRPO Policy-FSDP + 角色拆分同步版（rank0–3 policy、4–5 rollout、6–7 reward）+ kill-rank 故障注入 | `mm_dist/grpo_roles.py` + `scripts/fault_kill_rank.*` | eval → 恢复/故障 | 120 | 公司 V100 |

总估时：约 10.5 小时 + Gate 1 小时。超预算先砍 Day 4 的 capacity 扩展与 Day 5 的 rollout 权重同步演示；角色拆分允许 `INCONCLUSIVE`，Policy-FSDP 必须有证据。

## 三类环境边界

| 环境 | 本周用途 | 限制 |
| --- | --- | --- |
| 公司 8×V100 32 GB | 主线 | 不新建 conda、不升级 torch 2.1、FP16 不 BF16；每次运行 ≤ 10 分钟；日志、trace、checkpoint、图片、拓扑与性能数字**留在公司内**，只带出任务卡证据字段中的抽象值（比值、是否连续、次数） |
| 本机 CPU（ResearchAgentPy310） | 所有分布式脚本先过 gloo 2 进程单测 | 无 GPU；不能验证 NCCL/FP16 |
| 个人 5070 Ti | 只做单卡回归 | 不能冒充多卡结果 |
| 可选 H100 | 本周不需要 | — |

## 本机验证状态

构建完成日：2026-09-05。本机（Windows，conda `ResearchAgentPy310`，Python 3.10.20，torch 2.14.0+cpu，**无 GPU**）实测：

| 检查 | 结果 |
| --- | --- |
| `py_compile` 全部 20 个 `.py` | 通过 |
| `pytest lab/tests -q` | `73 passed, 1 skipped`（3 条 warning 均为 `torch.cuda.amp.GradScaler` 的 FutureWarning，有意保留，目标机 2.1.0 无新签名） |
| DDP 固定全局批等价 | 1 进程 vs 2 进程 gloo，前三步 loss 完全相等 |
| 专家并行数值等价 | `ep_size=1` 与稠密实现 `max\|diff\| = 0.000e+00` |
| 专家并行往返 | 2 进程下 permute → all_to_all → 逆 → unpermute 顺序完全恢复；某专家收到 0 token 时全部参数仍有梯度 |
| 角色分组 | 4 进程下组内 all_reduce 不跨组；broadcast 后参数 checksum 一致 |
| 故障注入 | `kill_rank` 子进程真实 exitcode=1；2 进程 gloo 掉卡演练 11.4 秒后其余 rank 退出 |
| 兼容审计 | 扫出 9 个默认 `bfloat16` 的脚本、4 处用 autocast 但无 GradScaler（含 `train_grpo.py`）、8 处 `torch.compile`、0 处 torch 2.1 不存在的 API |

**跳过的 1 项**：`test_full_fsdp_reshard_requires_gpu`。FSDP1 需要 CUDA，参数全在 CPU 上时 torch 会抛 `FSDP needs a non-CPU accelerator device`；2.1.0 与本机 2.14 行为一致，不是版本问题。已加 `train_fsdp.require_cuda_for_fsdp()` 提前给出可执行报错，并把 CPU 侧能验的部分改成真测试（分布式 checkpoint 的 2 进程写、1 进程读往返，通过）。

**本机限制导致未端到端执行的部分**：这台机器的 torch CPU wheel 编译时没带 libuv，`torchrun` 的 rendezvous 直接失败且 `USE_LIBUV=0` 压不住，所以 14 个 `scripts/run_*.{sh,ps1}` 的 torchrun 命令行只做了人工参数核对。**首次在公司机器上跑任一 `run_*` 脚本时先用 `MAX_STEPS=2` 试跑。**

**仍未验证（必须在公司 8×V100 上执行）**：FSDP 训练路径、reshard 完整链路、`policy_fsdp` 放置、fp16 与 GradScaler 的真实数值行为、NCCL 通信、显存与吞吐数字、SDPA 三后端实际命中。`02_LAB_GUIDE.md` 中标 `估算` 的数字全部属于此类。

**实现层面的一个真实缺陷已修复**：`dist.all_to_all_single` 不可微。教学版专家并行为此包了自定义 autograd Function（`AllToAllSingle`），否则梯度会在 dispatch 处静默断掉、专家参数永远不更新且不报错。

## 已知限制与风险

- 公司 torch 2.1.0 与当前 MiniMind commit 的依赖（transformers 4.57.6 等）可能不兼容；Day 0 审计给出“公司兼容路径 / 当前公开路径”两列，必要时 vendored 最小模型文件。
- 公司拓扑非全 NVLink（用户自述 NV1/NV2 混合），EP 分组对 all_to_all 影响大；只记录抽象结论。
- 8 卡配额未知；所有脚本设 `--max-steps` 上限与 NCCL timeout。
- 角色拆分是最重的一天；Gate 不依赖它。

## 来源

- `input_info/minimind_5070ti_v100.md`
- PyTorch v2.1.0 源码（`torch/cuda/__init__.py`、`autocast_mode.py`、`sdp_utils.cpp`）、PyTorch 2.1 FSDP / distributed.checkpoint 文档、DeepEP README、MegaBlocks setup.py、Tutel README、DeepSpeed MoE 文档（核验 2026-09-04）
