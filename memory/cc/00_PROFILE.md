# cc 学员 Profile

> 建立：2026-09-04 · 维护方：cc（Claude Code）  
> 只记录本项目需要的稳定信息；无现场证据的事实保留 `用户自述待核验`。与 Codex 的 `memory/00_PROFILE.md` 相互独立。

## 目标（已确认，来自 00_INPUT.md 历史需求）

- 用 14 周从基础深度学习与小型语言模型训练起步，建立大模型训练的端到端 ownership。
- 不满足于修改他人代码；要从算法与 infra 联合视角理解设计、loss/log、数值稳定性、分布式、profiling、checkpoint/recovery 与评测。
- 长期目标：资深训练工程师/训练架构师，能把能力迁移到金融、投行等垂直领域的专有模型训练。
- 对 cc 的额外要求（2026-09-04）：执行层面要足够细，能回答“今天具体做什么、跑什么命令、看到什么算对”。

## 环境（2026-09-08 重大变更）

| 环境 | 事实 | 约束 |
| --- | --- | --- |
| 公司 8×V100 | **唯一的 GPU 执行环境**。8× NVIDIA V100（sm_70）；既有 Conda；PyTorch 2.1.0 与配套 CUDA | **内网**：`pip install` 与 `git clone` 可用，但**有依赖的包容易失败**；**权重必须外网下好再拷进去**；服务器**只进不出**。不新建虚拟环境；不擅自升级 torch；FP16 不能 BF16；数据、代码、图片、日志、trace、checkpoint、拓扑、性能数字不出公司，只带出证据字段里的抽象值 |
| 个人 5070 Ti | **2026-09-08 起不再可用**（用户当次确认） | 不要再写进任何执行路径 |
| 租用 H100 | 当前不在计划内 | — |

**这个变更的直接后果**：cc 的交付从此是**开环**的——代码在这台有外网、无 GPU 的 Windows 机器上写和静态验证，第一次真实运行发生在 V100 上。所以每个周包都必须有一个早期 wiring 闸门，且所有 GPU 侧数字必须标 `估算`。

## 本机（运行 cc 的 Windows 机器，`已确认`，探针日期 2026-09-04）

- **换机器了。** 项目根目录现在是 `C:\Users\zzz_7893\Desktop\0_Projects\LLM_Training`；旧记录里的 `D:\zz\00_RealProjects\0_LLM_Training`、`D:` 盘、Anaconda 根 `D:\Software\Large\Anconda`、环境 `ResearchAgentPy310`、用户名 `13289` **在这台机器上都不存在**。
- 现在是 miniconda，根目录 `C:\Users\zzz_7893\miniconda3`，`conda` **不在 PATH 上**（只能用绝对路径调解释器）。环境：`rfm`、`work`（都是 py3.11.16）、`base`（py3.14.7，空）。
- **用户指令（2026-09-08）：科研类的都用 conda 环境 `rfm`，以后不要再问。** 解释器 `C:\Users\zzz_7893\miniconda3\envs\rfm\python.exe`。cc 已于 2026-09-08 往该环境装入 CPU 版 torch 2.14.0。
- 无 NVIDIA GPU；`nvidia-smi` 不存在。本机只做 CPU 单测、语法检查、dry run。
- pytest 默认临时目录**正常**，不需要 `--basetemp`（那是旧机器的问题）。
- 控制台代码页是 cp1252，直接 print 中文会抛 `UnicodeEncodeError`。交付脚本要在入口调 UTF-8 兜底（范例：week03 周包 `lab/src/mm_v100/console.py` 的 `use_utf8()`）。

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
