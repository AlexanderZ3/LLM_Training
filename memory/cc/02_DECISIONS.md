# cc 稳定决策记录

## 2026-09-04 — 与 Codex 系统的隔离

- cc 与 Codex 各自拥有独立的契约、agent、skill/workflow、schema、脚本和记忆；互不引用对方的规则文件。
- cc 专属路径：`CLAUDE.md`、`.claude/`、`outputs/1_cc_coaching/`、`memory/cc/`。
- 共享路径：`00_INPUT.md`（新增 `AGENT:` 字段路由）、`01_PROGRESS.md`（标题带 `[cc]`/`[codex]`）、`input_info/`（只读）、`outputs/` 根目录（分目录存放）。
- 用户已批准以上三条（2026-09-04）。

## 2026-09-04 — 交付单位

- cc 的最小交付单位是“可执行的一天”：一张任务卡 + `lab/` 可运行代码 + 证据字段。
- 代码放仓库不放 Markdown；任务卡 ≤ 1 页、≤ 7 步、每步有命令/预期/检查点。
- 周包固定结构：周卡、Foundations、Lab Guide、6 张任务卡、口试、参考答案、`lab/`。

## 2026-09-04 — 课程顺序与门

- 沿用 `input_info/` 的 14 周顺序；周次是 Gate 顺序，不是硬日历。
- 训练门顺序固定：静态检查 → dry run → overfit → FP32 ref → FP16 → resume → 2 卡等价 → scaling → eval。
- 主线默认在个人 5070 Ti；公司 V100 只用于多卡与 profiling；H100 是 gated stretch。

## 2026-09-04 — 硬件/版本

- V100 支持 FP16 Tensor Core；“不支持 FP16”按 BF16 笔误纠正，现场探针为准。
- 公司 PyTorch 2.1 是固定兼容线；教程同时保留公司路径与当前公开路径。

## 2026-09-04 — 证据

- 基线前不给精确 mastery 分数。
- `PASS/FAIL-MODEL/FAIL-SYSTEM/INCONCLUSIVE` 描述一次运行，不描述掌握。
- 公司原始证据不进本仓库；只记录获准的抽象总结。

## 2026-09-04 — 本机 Python 走 conda

- 用户指令：以后使用 Python 一律用 conda 环境。
- 用户指定环境 `ResearchAgentPy310`（Python 3.10），入口 `.claude/scripts/cc_py.ps1`；裸 `python` 禁用。
- 本机无 GPU；该环境安装 CPU 版 torch 只用于单测与 dry run。
- `00_INPUT.md` 的 `AGENT` 字段接受 `cc` / `claude` / `claude code` 作为 cc 的别名。

## 2026-09-05 — MiniMind 两周热身轨道

- 在 14 周主线之前加一条 `track_minimind` 轨道：M01 单机 5070 Ti 走通全链路，M02 公司 8×V100 做分布式实验台。它不替代主线 Week 01。
- 轨道内口试 ID 前缀用 `M01-`/`M02-`；校验器加 `-Track` 与 `-IdPrefix` 参数支持子轨道。
- MiniMind 不复制进仓库，通过 `MINIMIND_ROOT` 引用锁定 commit `7a6fddd63a30c06b2fdd5fac4089922b29bc841b`。

## 2026-09-05 — V100 上一律 fp16

- MiniMind 所有训练脚本默认 `--dtype bfloat16`，在 V100 + torch 2.1 上直接抛 RuntimeError（`is_bf16_supported()` 要求 compute capability ≥ 8）。
- M02 的所有命令显式 `--dtype float16`；`train_grpo.py` 用 autocast 但没有 GradScaler，已在 lab 中补齐。
- M02 代码只用 torch 2.1 存在的接口，禁用 `fully_shard`、`DTensor`、`device_mesh`、`torch.accelerator`。

## 2026-09-05 — 参数量以实测为准

- 首版 M02 参数量表漏算 `q_norm`+`k_norm`（每层 192）。以 `sum(p.numel() for p in MiniMindForCausalLM(cfg).parameters())` 实测为准：dense 63,912,192、MoE 198,416,640。
- 凡是能在本机实例化算出的数字，不接受手推结果；手推只用于解释公式。

## 2026-09-04 — 暂缓项

- ~~`input_info/minimind_5070ti_v100.md` 暂不处理~~ → 已于 2026-09-05 处理完毕。
