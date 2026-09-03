# 稳定决策记录

## 2026-09-03 — 项目边界

- 本项目是独立的 LLM Training Learning OS。
- 所有副作用严格限制在 `D:\zz\00_RealProjects\0_LLM_Training\`。
- `input_info/` 保持只读；扩展教程写入 `outputs/`。

## 2026-09-03 — 课程顺序

- 采用输入材料的 14 周顺序：纯 LLM → 纯 VLM → V100/FP16 → TinyFlow → 数据合同 → profiling → DDP → FSDP → diffusion/flow → SmolVLA → Mini-WAM → FinExec 0.6B → 放大/PEFT/可选 GRPO → Capstone。
- 该顺序将前两周基础补课放在原 12 周训练系统计划之前。
- 14 周是 Gate 顺序，不是硬日历；前置正确性失败可延长。

## 2026-09-03 — 教练架构

- 采用 1 个主代理 + 6 个项目级专职 agent。
- 角色按稳定职责而非周次拆分：source audit、curriculum、theory、lab、exam、reliability。
- 最多并行 3 个子代理；最终状态与输入归档只由主代理写。
- 配置使用官方项目级 `.codex/agents/*.toml` 结构，不自创不可执行的 agent manifest。

## 2026-09-03 — 教程合同

- 每周固定 4 份：`01_FOUNDATIONS.md`、`02_LAB_GUIDE.md`、`03_ORAL_EXAM.md`、`04_REFERENCE_ANSWERS.md`。
- 问题与答案分离，答案 ID 必须一一对应；默认先闭卷。
- 教程必须连接数学、shape、实现、数值与 infra，并包含止损、证据和三类资源边界。
- 文档生成不等于训练完成；所有命令默认待用户现场执行。

## 2026-09-03 — 硬件/版本策略

- V100 支持 FP16；“不支持 FP16”按 BF16 笔误纠正。
- 公司 PyTorch 2.1 为固定兼容线，不为教程强行升级。
- 当前官方文档只用于事实核验和个人新环境分支，不直接替代公司 API。
- H100 是 gated stretch resource，不是课程默认依赖。

## 2026-09-03 — 证据策略

- 基线前不产生精确 mastery 分数。
- PASS/FAIL-MODEL/FAIL-SYSTEM/INCONCLUSIVE 描述 run，不描述长期掌握。
- 源计划、AI 产物、未执行命令和用户自评都不能单独成为 mastery 证据。
- 公司内部原始证据不进入本仓库；只在获准时记录抽象且不可重识别的总结。
