# cc 学员 Profile

> 建立：2026-09-04 · 维护方：cc（Claude Code）  
> 只记录本项目需要的稳定信息；无现场证据的事实保留 `用户自述待核验`。与 Codex 的 `memory/00_PROFILE.md` 相互独立。

## 目标（已确认，来自 00_INPUT.md 历史需求）

- 用 14 周从基础深度学习与小型语言模型训练起步，建立大模型训练的端到端 ownership。
- 不满足于修改他人代码；要从算法与 infra 联合视角理解设计、loss/log、数值稳定性、分布式、profiling、checkpoint/recovery 与评测。
- 长期目标：资深训练工程师/训练架构师，能把能力迁移到金融、投行等垂直领域的专有模型训练。
- 对 cc 的额外要求（2026-09-04）：执行层面要足够细，能回答“今天具体做什么、跑什么命令、看到什么算对”。

## 环境（用户自述待核验）

| 环境 | 事实 | 约束 |
| --- | --- | --- |
| 公司 | 8× NVIDIA V100；既有 Conda；PyTorch 2.1.0 与配套 CUDA；可按公司流程 pip 安装 | 不新建虚拟环境；不擅自升级；数据、代码、图片、日志、trace、checkpoint、拓扑、性能数字不出公司；只能带出总结性文字 |
| 个人 | RTX 5070 Ti 16 GB 台式工作站；完整权限 | 显存 16 GB；用于公开数据缩小复现与 Blackwell 兼容验证 |
| 可选 | 租用 4×/8× H100；完整权限 | 只有较小资源无法回答的问题才使用；先过资源门 |

## 本机（运行 cc 的 Windows 机器，`已确认`，探针日期 2026-09-04）

- Anaconda 23.10.0，根目录 `D:\Software\Large\Anconda`；环境：`torchdiff`（py3.10，torch CPU）、`parttime`（py3.10，torch CPU）、`py311_env`、`ResearchAgentPy310`、`stock_research_agent`。
- 无 NVIDIA GPU；`nvidia-smi` 不存在。本机只做 CPU 单测、语法检查、dry run。
- PATH 上的 `python` 是商店占位（退出码 9009）。用户指令（2026-09-04）：Python 一律用 conda 环境 `ResearchAgentPy310`（`D:\Software\Large\Anconda\envs\ResearchAgentPy310`）；入口 `.claude/scripts/cc_py.ps1`。该环境初始无 torch，需要时用 CPU 版 wheel 安装。
- 这台机器不是 5070 Ti 工作站；GPU 证据必须来自 5070 Ti 或公司机器。

## 关键纠错

- 用户输入写“V100 不支持 FP16”；官方资料支持 V100 FP16 Tensor Core 路径，合理解释是“通常无原生 BF16”。按此处理，真实能力以 Week 03 探针为准。
- 输入中的 `nanpGPT` 按 nanoGPT 类小型 GPT 项目理解。

## 已有背景（用户自述待核验）

- 有公司大模型训练体感，但只了解局部；有 infra 方面积累（算子、并行、profiling 认知）。
- 缺少的是“亲手控制一次完整训练全流程”的经验。

## 当前能力证据

- 截至 2026-09-04，cc 侧没有任何 A0/A1 基线、代码、日志、口试或 debugging 证据。
- `input_info/` 与 `outputs/0_basic_training/` 是课程来源与 Codex 的产物，不是学员掌握证据。
- 因此不给 K/A/D/E 或 mastery 分数。

## 时间假设

- 来源计划按工作日每天 1–2 小时、每周 5–10 小时设计；用户尚未向 cc 单独确认。开始 Week 01 时确认。
