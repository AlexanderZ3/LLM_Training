# LLM Training Learning OS

这是一个 14 周大模型训练陪跑项目：从 TinyStories 上的 MiniGPT 与 nanoVLM 起步，逐步进入 FP16、数据合同、profiling、DDP/FSDP、生成式机器人策略、VLA/WAM、金融 SFT/CPT/PEFT，最后以可恢复、可解释的多卡训练 Capstone 收口。

目标不是“把示例跑起来”，而是建立完整 ownership：

```text
数学目标 → 数据与 shape → forward/loss → 数值稳定性
→ checkpoint/resume → 分布式正确性 → 性能归因 → 独立评测与架构解释
```

## 当前交付

- 14 个周目录，每周 4 份教程：基础知识、实践手册、闭卷拷问、参考答案。
- 1 个主代理契约与 6 个项目级 Codex 专职 agent。
- 可复用的输入处理、每周训练、内容刷新和公司环境边界 workflow。
- 学员 profile、项目 state、决策记录、证据 schema 与确定性课程校验器。
- 首轮详细教程质量审查：文档/静态门通过，目标机器训练执行仍待完成。

课程总入口：[`outputs/0_basic_training/00_README.md`](outputs/0_basic_training/00_README.md)  
详细手册发布门：[`outputs/0_basic_training/00_MANUAL_ACCEPTANCE_STANDARD.md`](outputs/0_basic_training/00_MANUAL_ACCEPTANCE_STANDARD.md)
质量审查报告：[`outputs/02_2026-09-03_详细教程质量审查报告.md`](outputs/02_2026-09-03_详细教程质量审查报告.md)

## 14 周地图

| 周 | 主线 | 本周必须拥有的能力 |
| ---: | --- | --- |
| 01 | TinyStories MiniGPT | 从 token 到 logits/loss/生成的纯 LLM 闭环 |
| 02 | nanoVLM + 公开 VQA | 视觉 token、projector、文本解码和多模态评测 |
| 03 | V100 / FP16 / NCCL | 环境、拓扑、mixed precision 与通信基线 |
| 04 | TinyFlowPolicy | flow matching 动作生成与完整训练/采样闭环 |
| 05 | 数据合同与复现 | 时间对齐、split、防泄漏、RNG 与 hash |
| 06 | 单卡 profiling | step 时间、显存模型、PyTorch/Nsight 证据 |
| 07 | DDP scaling | fixed global batch 等价、1/2/4/8 卡扩展 |
| 08 | FSDP/checkpoint | sharding、wrap、通信与可恢复状态 |
| 09 | Diffusion vs Flow | 公平对照、采样成本与受控结论 |
| 10 | SmolVLA | 真实 VLA checkpoint 的兼容、微调与评测 |
| 11 | Mini-WAM | action/world/shuffle 三组消融与 OOD 证据 |
| 12 | FinExec 0.6B | evaluator、mask/packing、SFT 与短 CPT |
| 13 | FinExec 放大 | FSDP/LoRA、规模瓶颈与可选 GRPO smoke |
| 14 | Capstone | clean-room 复现、1→8 卡、故障恢复与答辩 |

## 使用方法

1. 打开 `00_INPUT.md`，在顶部标记块内填写本次需求；默认 `TYPE: TASK / DEPTH: AUTO / MODE: COACH`。
2. 让 Codex 处理顶部需求块。它会按 `AGENTS.md` 读取状态、执行、校验、更新进度并归档输入。
3. 学习某周时先读 `01_FOUNDATIONS.md`，沿具体数据/模型对象复算公式、shape、dtype 与失败机制；再按 `02_LAB_GUIDE.md` 从空目录完成下载、代码、smoke、训练、eval 和恢复。它们是操作手册，不是 roadmap。
4. 完成证据后闭卷回答 `03_ORAL_EXAM.md`；提交答案或自评前不要打开 `04_REFERENCE_ANSWERS.md`。
5. 每个 Gate 只接受真实运行或现场解释证据。教程、源计划和 AI 生成代码都不是通过证明。

校验课程结构：

```powershell
powershell -ExecutionPolicy Bypass -File scripts/validate_curriculum.ps1
powershell -ExecutionPolicy Bypass -File scripts/validate_markdown_examples.ps1
```

有可用 Python 的环境也可运行 `python scripts/validate_curriculum.py`。

第一个校验器检查 56 份周文件、章节、题答映射与本地链接；第二个校验 Markdown fence，并解析 JSON、PowerShell 与可用时的 Bash 代码块。两者都只读，不运行训练、不访问公司资产，也不联网；它们不能替代目标机上的 Python/CUDA smoke test。

## 三类执行环境

| 环境 | 定位 | 约束 |
| --- | --- | --- |
| 公司 8×V100 | FP16、NCCL、DDP/FSDP、profiling 主实验室 | 用户自述待探针核验；沿用现有 Conda/PyTorch 2.1；任何代码、数据、日志、trace、checkpoint、图片不得导出 |
| 个人 RTX 5070 Ti 16GB | 公开数据缩小复现、Blackwell 兼容验证、公开作品集 | 完整权限，但显存较小；结果不能冒充公司多卡结果 |
| 可选 4×/8×H100 | 只有较小资源无法回答的扩展问题 | 先通过资源门，预注册预算、停止条件和产物 |

V100 支持 FP16 Tensor Core 与 AMP。项目输入中“V100 不支持 FP16”按“通常不支持原生 BF16”纠正；真实环境仍以 Week 03 探针为准。

## 项目结构

```text
.
├── .codex/
│   ├── config.toml
│   └── agents/                 # 6 个项目级专职 agent
├── input_info/                 # 原始计划，只读
├── memory/                     # profile、状态与稳定决策
├── outputs/
│   ├── 00_2026-09-03_教练系统实施规划.md
│   ├── 01_2026-09-03_技术事实与兼容性核验.md
│   ├── 02_2026-09-03_详细教程质量审查报告.md
│   └── 0_basic_training/       # 14 周 × 4 份教程
├── schemas/                    # evidence/run manifest 结构
├── scripts/                    # 只含确定性工具
├── workflows/                  # 可复用协作流程
├── 00_INPUT.md                 # 高频任务入口
├── 01_PROGRESS.md              # 处理日志
└── AGENTS.md                   # 项目运行契约
```

## Agent 设计

本项目采用“1 个主代理 + 6 个可复用专职 agent”，而不是为 14 周各建一个 agent。角色按职责复用，能减少周次之间的标准漂移；同时最多并行 3 个子代理，以便主代理保留最终整合与写入控制。

角色和调用顺序见 [`outputs/00_2026-09-03_教练系统实施规划.md`](outputs/00_2026-09-03_教练系统实施规划.md)。项目级 agent 定义采用官方支持的 `.codex/agents/*.toml`，每个文件都包含 `name`、`description` 和 `developer_instructions`。

## 证据与状态

- 所有硬件、拓扑和环境描述在探针前均标记为“用户自述待核验”。
- 基线诊断前不生成精确 mastery 分数。
- `PASS / FAIL-MODEL / FAIL-SYSTEM / INCONCLUSIVE` 是实验标签，不是能力等级。
- AI 辅助从 A0（独立闭卷）到 A4（AI 主导）记录；核心周 Gate 默认需要 A0/A1 证据。
- 长期事实见 `memory/00_PROFILE.md`，当前动作见 `memory/01_STATE.md`。

## 重要来源

课程基于 `input_info/` 中的总体计划和 14 份周手册扩展。会变化的技术事实应在使用当天重新核验；本次 Codex 配置依据 [OpenAI AGENTS.md 文档](https://learn.chatgpt.com/docs/agent-configuration/agents-md) 与 [OpenAI Subagents 文档](https://learn.chatgpt.com/docs/agent-configuration/subagents)，核验日期为 2026-09-03。
