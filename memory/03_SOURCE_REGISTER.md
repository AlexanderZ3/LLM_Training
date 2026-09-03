# 来源登记与事实状态

> 建立日期：2026-09-03。`input_info/` 均为只读课程来源，不是用户训练证据。

## 本地来源

| 来源 | 用途 | 状态/限制 |
| --- | --- | --- |
| `Overall_robot_wam_va_vla_finance_12_week_plan_v100_fp16_2026-09.md` | W03–W14 的总体资源、节奏、Gate 与系统规范 | 来源计划；其中硬件/拓扑数值待现场核验 |
| `week01_pure_llm_tinystories_from_scratch.md` | W01 MiniGPT、BPE、CE、AMP、resume | 已用于课程扩展；命令未实机验证 |
| `week02_pure_vlm_nanovlm_public_vqa.md` | W02 vision tokens、projector、VQA | 已用于课程扩展；模型/数据 revision 使用时再核验 |
| `week03_v100_environment_topology_fp16_baseline.md` | W03 环境、拓扑、NCCL、FP16 | 已用于课程扩展；现场输出尚无 |
| `week04_tinyflowpolicy_complete_training_loop.md` | W04 conditional flow matching 与 action chunk | 已用于课程扩展；数据/rollout 待执行 |
| `week05_taskp_data_contract_alignment_reproducibility.md` | W05 schema、lag、split、RNG | 已用于课程扩展；Task-P 尚未由用户最终选择 |
| `week06_single_gpu_profiling_memory_optimization.md` | W06 时间/显存模型、Profiler/Nsight | 已用于课程扩展；性能阈值不是实测 |
| `week07_ddp_correctness_topology_scaling.md` | W07 fixed-batch 等价与 1/2/4/8 卡 | 已用于课程扩展；拓扑分组待现场确认 |
| `week08_fsdp_sharding_checkpoint.md` | W08 FSDP、hybrid shard、checkpoint | 已用于课程扩展；PyTorch 2.1 能力必须按版本分支 |
| `week09_diffusion_vs_flow_matching_controlled_experiment.md` | W09 公平 objective/sampler 对照 | 已用于课程扩展；算法收益未知 |
| `week10_smolvla_450m_v100_finetuning.md` | W10 SmolVLA 兼容与微调 | 已用于课程扩展；仓库 API/revision 使用时再核验 |
| `week11_mini_wam_world_supervision_ablation.md` | W11 action/world/shuffle 消融 | 已用于课程扩展；不存在结果结论 |
| `week12_finexec_06b_sft_cpt_evaluator.md` | W12 schema、executor、SFT/CPT | 已用于课程扩展；不构成金融能力或投资结果 |
| `week13_finexec_scaling_peft_fsdp_grpo_smoke.md` | W13 放大、PEFT、可选 GRPO | 已用于课程扩展；量化/kernel 兼容待审计 |
| `week14_capstone_reproduce_scale_recover_report.md` | W14 clean-room、scale、recover、报告 | 已用于课程扩展；Capstone 尚未选择/执行 |

## 2026-09-03 已核验的官方事实

| 事实 | 状态 | 官方来源 |
| --- | --- | --- |
| Codex 项目指令使用 `AGENTS.md`，按目录层级发现和合并 | 已确认 | [OpenAI AGENTS.md](https://learn.chatgpt.com/docs/agent-configuration/agents-md) |
| 项目级 custom agents 位于 `.codex/agents/*.toml`，需 `name/description/developer_instructions` | 已确认 | [OpenAI Subagents](https://learn.chatgpt.com/docs/agent-configuration/subagents) |
| V100/Volta 有 FP16 Tensor Core mixed-precision 路径，loss scaling 用于缓解小梯度下溢 | 已确认 | [NVIDIA mixed precision](https://docs.nvidia.com/deeplearning/performance/mixed-precision-training/index.html) |
| CUDA 13 移除 Maxwell/Pascal/Volta 的离线编译和库支持；12.x 仍可用于这些架构 | 已确认 | [CUDA 13 release notes](https://docs.nvidia.com/cuda/archive/13.0.0/cuda-toolkit-release-notes/index.html) |
| PyTorch 2.1 提供 CUDA autocast/GradScaler 与 FSDP API | 已确认 | [PyTorch 2.1 AMP](https://docs.pytorch.org/docs/2.1/amp.html)、[PyTorch 2.1 FSDP](https://docs.pytorch.org/docs/2.1/fsdp.html) |

## 使用时必须重新核验

- PyTorch、CUDA、NCCL、Nsight 与 GPU driver 的现场组合；
- nanoVLM、LeRobot、SmolVLA、Transformers、Datasets、PEFT、TRL、bitsandbytes 的当前安装与 API；
- 数据集 revision、license、字段和下载命令；
- 公司机器真实 GPU 型号/显存/compute capability、拓扑、网络、容器可见性与审批规则；
- RTX 5070 Ti 当前 PyTorch wheel/driver 支持；
- 租用 H100 平台的网络、持久盘、计费和数据销毁规则。

不得因链接在 2026-09-03 可访问，就声称未来使用时仍兼容。
