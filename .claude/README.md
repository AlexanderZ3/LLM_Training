# .claude/ — Claude Code 陪跑教练系统

这是 cc（Claude Code）在本仓库的全部配置。它与 Codex 的 `.codex/` + `AGENTS.md` 完全独立，互不引用。契约在仓库根目录的 `CLAUDE.md`。

## 文件地图

```text
.claude/
├── README.md                 # 本文件
├── agents/                   # 8 个子智能体（frontmatter + 系统提示）
│   ├── cc-source-auditor.md        只读：版本/硬件/许可事实表
│   ├── cc-curriculum-planner.md    只读：周规格与逐日地图
│   ├── cc-theory-tutor.md          写 01_FOUNDATIONS.md
│   ├── cc-lab-builder.md           写 lab/ + 02_LAB_GUIDE.md
│   ├── cc-task-card-writer.md      写 03_TASK_CARDS/dayN.md
│   ├── cc-examiner.md              写 04/05 口试与答案
│   ├── cc-reliability-reviewer.md  只读：审查 + 跑校验脚本
│   └── cc-evidence-coach.md        只读：分类用户证据
├── skills/                   # 7 个斜杠命令
│   ├── cc-input/        处理 00_INPUT.md 的 AGENT: cc 需求
│   ├── cc-build-week/   生成周包（多智能体流水线）
│   ├── cc-day/          按天陪跑
│   ├── cc-evidence/     记录并分类证据
│   ├── cc-gate/         周末口试与 Gate
│   ├── cc-refresh/      版本/事实刷新
│   └── cc-status/       当前位置与唯一下一步
├── templates/
│   ├── TASK_CARD.md            任务卡模板（颗粒度合同的载体）
│   ├── WEEK_CARD.md            一页周卡模板
│   ├── EXPERIMENT_CONTRACT.md  长跑前实验合同
│   └── PROGRESS_ENTRY.md       01_PROGRESS.md 的 [cc] 记录格式
├── schemas/
│   └── evidence.schema.json    证据 JSON 结构
└── scripts/
    └── validate_cc_week.ps1    周包静态校验器（PowerShell 5.1）
```

相关目录：`outputs/1_cc_coaching/`（交付）、`memory/cc/`（状态与证据）。

## 一次典型的周循环

```text
/cc-build-week 01      生成 Week 01 周包（审计→规划→理论∥实验→任务卡→口试→审查→校验）
/cc-day 01 0           Day 0：环境探针 + readiness
   …用户执行任务卡，提交证据…
/cc-evidence           归档证据，分类 PASS/FAIL-*/INCONCLUSIVE
/cc-day 01 1 … 5
/cc-gate 01            口试 + Gate 判定 → PASS / EXTEND
/cc-status             任何时候：我在哪、下一步唯一动作
```

## 与 Codex 的边界

| | cc | Codex |
| --- | --- | --- |
| 契约 | `CLAUDE.md` | `AGENTS.md` |
| 智能体 | `.claude/agents/*.md` | `.codex/agents/*.toml` |
| 流程 | `.claude/skills/` | `workflows/` |
| schema / 脚本 | `.claude/schemas/`、`.claude/scripts/` | `schemas/`、`scripts/` |
| 记忆 | `memory/cc/` | `memory/*.md` |
| 交付 | `outputs/1_cc_coaching/` | `outputs/0_basic_training/`、`outputs/0N_*.md` |
| 共享 | `00_INPUT.md`（`AGENT:` 路由）、`01_PROGRESS.md`（`[cc]`/`[codex]` 标签）、`input_info/`（只读） | 同左 |

## 校验

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .claude/scripts/validate_cc_week.ps1 -Week 01
```
