# 14 周基础训练课程总入口

> `01_FOUNDATIONS.md` 与 `02_LAB_GUIDE.md` 的发布标准见 [详细操作手册验收规范](00_MANUAL_ACCEPTANCE_STANDARD.md)。它们要从数据/模型获取写到代码、训练、评测和恢复，不是 roadmap。

> 生成日期：2026-09-03  
> 状态：课程教程已生成；**尚未代表任何训练或能力 Gate 已通过**。  
> 使用顺序：共享协议 → 当周基础 → 当周实验 → 闭卷口试 → 参考答案。

## 开始前

1. 先读 [共享实验与证据协议](00_SHARED_PROTOCOL.md)。
2. 不要一次性浏览参考答案。先完成当周实践证据，再闭卷答题。
3. 所有 `<PLACEHOLDER>`、`/path/to/...`、模型/数据 ID 都必须按现场批准路径替换；带占位符的命令是模板，不可原样运行。
4. 公司环境中的代码、数据、配置、日志、loss 曲线、trace、checkpoint、图片、机器拓扑和性能数字全部留在公司内。
5. 教程中的阈值是课程 Gate 或来源计划假设，不是已获得的实验结果。

## 课程导航

| 周 | 主题 | 基础 | 实践 | 闭卷题 | 参考答案 |
| ---: | --- | --- | --- | --- | --- |
| 01 | TinyStories MiniGPT | [基础](week01_tinystories_minigpt/01_FOUNDATIONS.md) | [实践](week01_tinystories_minigpt/02_LAB_GUIDE.md) | [拷问](week01_tinystories_minigpt/03_ORAL_EXAM.md) | [答案](week01_tinystories_minigpt/04_REFERENCE_ANSWERS.md) |
| 02 | nanoVLM + VQA | [基础](week02_nanovlm_vqa/01_FOUNDATIONS.md) | [实践](week02_nanovlm_vqa/02_LAB_GUIDE.md) | [拷问](week02_nanovlm_vqa/03_ORAL_EXAM.md) | [答案](week02_nanovlm_vqa/04_REFERENCE_ANSWERS.md) |
| 03 | V100 / FP16 / 拓扑 | [基础](week03_v100_fp16_topology/01_FOUNDATIONS.md) | [实践](week03_v100_fp16_topology/02_LAB_GUIDE.md) | [拷问](week03_v100_fp16_topology/03_ORAL_EXAM.md) | [答案](week03_v100_fp16_topology/04_REFERENCE_ANSWERS.md) |
| 04 | TinyFlowPolicy | [基础](week04_tinyflowpolicy/01_FOUNDATIONS.md) | [实践](week04_tinyflowpolicy/02_LAB_GUIDE.md) | [拷问](week04_tinyflowpolicy/03_ORAL_EXAM.md) | [答案](week04_tinyflowpolicy/04_REFERENCE_ANSWERS.md) |
| 05 | 数据合同与可复现 | [基础](week05_data_contract_repro/01_FOUNDATIONS.md) | [实践](week05_data_contract_repro/02_LAB_GUIDE.md) | [拷问](week05_data_contract_repro/03_ORAL_EXAM.md) | [答案](week05_data_contract_repro/04_REFERENCE_ANSWERS.md) |
| 06 | 单卡 profiling | [基础](week06_single_gpu_profiling/01_FOUNDATIONS.md) | [实践](week06_single_gpu_profiling/02_LAB_GUIDE.md) | [拷问](week06_single_gpu_profiling/03_ORAL_EXAM.md) | [答案](week06_single_gpu_profiling/04_REFERENCE_ANSWERS.md) |
| 07 | DDP scaling | [基础](week07_ddp_scaling/01_FOUNDATIONS.md) | [实践](week07_ddp_scaling/02_LAB_GUIDE.md) | [拷问](week07_ddp_scaling/03_ORAL_EXAM.md) | [答案](week07_ddp_scaling/04_REFERENCE_ANSWERS.md) |
| 08 | FSDP 与 checkpoint | [基础](week08_fsdp_checkpoint/01_FOUNDATIONS.md) | [实践](week08_fsdp_checkpoint/02_LAB_GUIDE.md) | [拷问](week08_fsdp_checkpoint/03_ORAL_EXAM.md) | [答案](week08_fsdp_checkpoint/04_REFERENCE_ANSWERS.md) |
| 09 | Diffusion vs Flow | [基础](week09_diffusion_vs_flow/01_FOUNDATIONS.md) | [实践](week09_diffusion_vs_flow/02_LAB_GUIDE.md) | [拷问](week09_diffusion_vs_flow/03_ORAL_EXAM.md) | [答案](week09_diffusion_vs_flow/04_REFERENCE_ANSWERS.md) |
| 10 | SmolVLA | [基础](week10_smolvla/01_FOUNDATIONS.md) | [实践](week10_smolvla/02_LAB_GUIDE.md) | [拷问](week10_smolvla/03_ORAL_EXAM.md) | [答案](week10_smolvla/04_REFERENCE_ANSWERS.md) |
| 11 | Mini-WAM | [基础](week11_mini_wam/01_FOUNDATIONS.md) | [实践](week11_mini_wam/02_LAB_GUIDE.md) | [拷问](week11_mini_wam/03_ORAL_EXAM.md) | [答案](week11_mini_wam/04_REFERENCE_ANSWERS.md) |
| 12 | FinExec SFT/CPT | [基础](week12_finexec_sft_cpt/01_FOUNDATIONS.md) | [实践](week12_finexec_sft_cpt/02_LAB_GUIDE.md) | [拷问](week12_finexec_sft_cpt/03_ORAL_EXAM.md) | [答案](week12_finexec_sft_cpt/04_REFERENCE_ANSWERS.md) |
| 13 | FinExec 放大/PEFT/GRPO | [基础](week13_finexec_scaling_peft_grpo/01_FOUNDATIONS.md) | [实践](week13_finexec_scaling_peft_grpo/02_LAB_GUIDE.md) | [拷问](week13_finexec_scaling_peft_grpo/03_ORAL_EXAM.md) | [答案](week13_finexec_scaling_peft_grpo/04_REFERENCE_ANSWERS.md) |
| 14 | Capstone | [基础](week14_capstone/01_FOUNDATIONS.md) | [实践](week14_capstone/02_LAB_GUIDE.md) | [拷问](week14_capstone/03_ORAL_EXAM.md) | [答案](week14_capstone/04_REFERENCE_ANSWERS.md) |

## 先修关系

```text
W01 LLM ──┬── W02 VLM
          └── W03 数值/硬件 ── W04 Flow 训练闭环 ── W05 数据合同
                                                │
W06 单卡 Profile ←──────────────────────────────┘
  │
  ├── W07 DDP ── W08 FSDP
  │                 │
  └── W09 算法对照 ─┴─ W10 SmolVLA ── W11 Mini-WAM
                                      │
W12 FinExec 基线 ── W13 放大 ─────────┴── W14 Capstone
```

W12 可在 W10–W11 期间做轻量阅读，但正式多卡实验应先通过 W07–W08。

## 每周固定节奏

| 天 | 核心动作 | 证据 |
| --- | --- | --- |
| Day 1 | 数学/数据/shape 合同，1–5 step dry run | shape、单测、预期 loss |
| Day 2 | 最小实现与 tiny-set overfit | 过拟合曲线和固定 batch |
| Day 3 | FP32 reference / 主实验 | config、manifest、metrics |
| Day 4 | FP16、resume、分布式或 profile | 数值/恢复/性能证据 |
| Day 5 | 独立 eval、故障注入、口试、decision | rubric、错误分类、回归日期 |

每天标准 60–90 分钟；第二小时只做一个 stretch task。训练作业可以在离开后运行，但必须有 `max_steps`、资源限制、保存周期和停止条件，并遵守共享资源政策。

## Gate 语义

- `PASS`：本次预注册问题在限定范围内得到足够证据。
- `FAIL-MODEL`：系统与测量可信，但模型/算法假设未成立；仍是有效实验。
- `FAIL-SYSTEM`：环境、数据、数值、分布式或恢复链路使模型结论不可判定。
- `INCONCLUSIVE`：样本、方差、预算或评测不足，不能支持方向性结论。

这些标签不能直接转换为“已掌握”。能力 Gate 还要求 A0/A1 独立表现、跨场景 debug/transfer、trade-off 解释以及不同日期的复测。

## 课程结构校验

在项目根目录运行：

```powershell
powershell -ExecutionPolicy Bypass -File scripts/validate_curriculum.ps1
powershell -ExecutionPolicy Bypass -File scripts/validate_markdown_examples.ps1
```

Linux/已有 Python 环境可运行 `python scripts/validate_curriculum.py`。

第一个脚本检查 14 周 × 4 文件、关键章节、题答 ID 和本地链接；第二个检查 Markdown fence，并解析 JSON、PowerShell 与可用时的 Bash 代码块。脚本通过不代表 Python 训练代码、公式、依赖兼容或模型结果已在目标机器验证。
