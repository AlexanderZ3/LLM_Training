---
name: cc-input
description: 处理 00_INPUT.md 顶部标记块中 AGENT 为 cc 或 both 的最新需求：解析、定义验收、执行、校验、更新 memory/cc 与 01_PROGRESS.md（[cc] 标签）、归档输入。用户说“处理 00_INPUT”或“/cc-input”时使用。
---

# /cc-input — 从 `00_INPUT.md` 到可验收交付

严格按 `CLAUDE.md` 第 4 节协议执行。以下是逐步操作。

## 1. Parse

读取 `00_INPUT.md` 中 `<!-- VOICE_ROUTER:INPUT:BEGIN -->` 与 `<!-- VOICE_ROUTER:INPUT:END -->` 之间的文本，保存原文副本。解析：

```text
TYPE:  TASK | REFINE          默认 TASK
DEPTH: AUTO | DEEP | SCRIPT   默认 AUTO
MODE:  COACH | BUILD | DAY | GATE | REVIEW | RESEARCH | REFINE   默认 COACH
AGENT: cc | codex | both      默认 cc
```

- `AGENT: codex`：不处理。告诉用户这条需求标记给 Codex，然后停止。
- 输入块为空：告诉用户没有待处理需求，给出 `/cc-status` 的结果，停止。

## 2. Context

读取 `memory/cc/01_STATE.md`、`memory/cc/00_PROFILE.md`、`memory/cc/02_DECISIONS.md`，以及需求直接引用的 `input_info/` 文件和 `outputs/1_cc_coaching/` 历史输出。

## 3. Acceptance

在执行前写出：将创建/修改的文件、期望行为、需要的证据、不允许发生的副作用。把它作为本次任务的验收单。

## 4. Route

按 MODE 路由：

| MODE | 走哪个 skill 或流程 |
| --- | --- |
| BUILD（周包） | `/cc-build-week NN` 流程 |
| DAY | `/cc-day NN D` 流程 |
| GATE | `/cc-gate NN` 流程 |
| RESEARCH | 派 `cc-source-auditor`，结果写 `outputs/1_cc_coaching/NN_<date>_cc_<topic>.md` |
| REFINE | 修改 `CLAUDE.md`、`.claude/` 下文件；改完自检一致性 |
| COACH / REVIEW | 主会话直接完成；必要时派 `cc-evidence-coach` |

## 5. Validate

涉及周包时运行：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .claude/scripts/validate_cc_week.ps1 -Week NN
```

校验失败不得归档为完成；修复后重跑。

## 6. State + Progress

- 更新 `memory/cc/01_STATE.md`：当前周/天、Gate 状态、唯一下一步。
- 在 `01_PROGRESS.md` 的 `## Log` 之后、最新一条之前插入一条记录，标题格式：

```markdown
## YYYY-MM-DD [cc] - 标题
`TYPE: …` · `DEPTH: …` · `MODE: …`
### 输入 / 动作 / 输出 / 决策 / 下一步
```

只追加，不改 Codex 的 `[codex]` 或无标签记录。

## 7. Archive

把原需求完整移动到 `00_INPUT.md` 的“历史归档区”，作为新小节 `### YYYY-MM-DD [cc] - 标题`，附处理状态与主要交付；恢复顶部空白输入块（保留 `TYPE/DEPTH/MODE/AGENT` 四行与注释）。最后再读一遍确认输入块为空。

## 失败处理

- 输入不完整但可安全推断：写明默认值并继续。
- 缺少会改变方案的用户选择：完成不依赖该选择的部分，把阻断问题写进“下一步”。
- 官方资料与源计划冲突：保留两个版本，按环境分支，不静默覆盖。
- 写入目标越过 `CLAUDE.md` 第 0 节边界：立即停止该动作并报告。

## 完成定义

交付物存在且可读、校验通过、`memory/cc` 与进度一致、`input_info/` 未改、没有虚构运行证据、输入已归档且新输入块为空。
