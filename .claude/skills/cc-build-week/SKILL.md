---
name: cc-build-week
description: 用 cc 多智能体流水线生成第 NN 周完整周包到 outputs/1_cc_coaching/weekNN_<slug>/：来源审计 → 课程规划 → 理论+实验并行 → 任务卡 → 口试 → 可靠性审查 → 校验。用法 /cc-build-week 01。
---

# /cc-build-week NN — 生成一周的周包

参数：周次 `NN`（01–14）。slug 见下表；已存在的周包默认做增量修订，不整体覆盖。

子轨道：`/cc-build-week 01 --track track_minimind` 把周包放在 `outputs/1_cc_coaching/track_minimind/weekNN_*/`，口试 ID 前缀改用 `M`（校验时加 `-Track track_minimind -IdPrefix M`）。子轨道的 slug 见该轨道的 `00_TRACK_README.md`。

| NN | slug | NN | slug |
| --- | --- | --- | --- |
| 01 | pure_llm_minigpt | 08 | fsdp_checkpoint |
| 02 | pure_vlm_nanovlm | 09 | diffusion_vs_flow |
| 03 | v100_fp16_topology | 10 | smolvla_finetune |
| 04 | tinyflowpolicy | 11 | mini_wam_ablation |
| 05 | data_contract_repro | 12 | finexec_sft_cpt |
| 06 | single_gpu_profiling | 13 | finexec_scale_peft_grpo |
| 07 | ddp_scaling | 14 | capstone |

## 阶段 0：准备（主会话）

1. 读 `memory/cc/01_STATE.md`、`00_PROFILE.md`；读 `input_info/weekNN_*.md` 与总体计划中对应段落。
2. 确认前一周 Gate 状态。若未通过且用户没有明确要求，先提示，再继续（生成周包不等于允许跳门）。
3. 创建目录骨架：

```text
outputs/1_cc_coaching/weekNN_<slug>/{03_TASK_CARDS,lab/{configs,src,tests,scripts}}
```

## 阶段 1：审计与规划（可并行，都是只读）

- 派 `cc-source-auditor`：本周涉及的数据/模型/包/硬件事实表。
- 派 `cc-curriculum-planner`：周规格与逐日地图。给它审计结果作为输入之一。

主会话合并两者，写出 `00_WEEK_CARD.md`（使用 `.claude/templates/WEEK_CARD.md`）。这是后续所有 agent 的共同输入。

## 阶段 2：理论与实验（并行，互斥文件）

- 派 `cc-theory-tutor`：只写 `01_FOUNDATIONS.md`。
- 派 `cc-lab-builder`：只写 `lab/` 与 `02_LAB_GUIDE.md`。

两者都拿到 `00_WEEK_CARD.md` 与审计事实表。并发不超过 3。

## 阶段 3：任务卡（依赖 lab/ 已存在）

派 `cc-task-card-writer`：写 `03_TASK_CARDS/day0.md` … `day5.md`。任务里附上 `lab/README.md` 与 `lab/scripts/` 清单，要求每个命令都能在其中找到。

## 阶段 4：口试

派 `cc-examiner`：写 `04_ORAL_EXAM.md` 与 `05_REFERENCE_ANSWERS.md`。

## 阶段 5：审查与校验

1. 派 `cc-reliability-reviewer`：输出问题清单与结论。
2. 主会话按严重度修复 P0/P1；P2/P3 视时间修复或记录到周卡“已知限制”。
3. 运行：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .claude/scripts/validate_cc_week.ps1 -Week NN
```

4. 校验不过不交付。

## 阶段 6：收口（主会话）

- 更新 `memory/cc/01_STATE.md`（当前周包状态：`BUILT / RUNTIME-UNVERIFIED`）。
- `01_PROGRESS.md` 追加 `[cc]` 记录。
- 如本次由 `/cc-input` 触发，回到那里做归档。

## 写入互斥表

| 文件 | 唯一写者 |
| --- | --- |
| `00_WEEK_CARD.md` | 主会话 |
| `01_FOUNDATIONS.md` | cc-theory-tutor |
| `02_LAB_GUIDE.md`、`lab/**` | cc-lab-builder |
| `03_TASK_CARDS/*.md` | cc-task-card-writer |
| `04_ORAL_EXAM.md`、`05_REFERENCE_ANSWERS.md` | cc-examiner |
| `memory/cc/*`、`01_PROGRESS.md`、`00_INPUT.md` | 主会话 |

## 诚实条款

本机可能没有 Python/CUDA。周包完成状态最高只能是 `BUILT / RUNTIME-UNVERIFIED`，直到用户在目标机上提交真实证据。
