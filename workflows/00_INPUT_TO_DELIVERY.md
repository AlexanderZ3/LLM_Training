# Workflow 00：从 `00_INPUT.md` 到可验收交付

## 触发

用户要求“处理 `00_INPUT.md` 顶部最新输入需求块”。

## 状态机

1. **Parse**：读取 marker 内文本；解析 TYPE/DEPTH/MODE，保留原文副本。
2. **Context**：读取 `memory/01_STATE.md`、profile、直接相关源文件和最近进度。
3. **Acceptance**：在执行前列出文件、行为、证据和不允许发生的副作用。
4. **Audit**：涉及当前事实时委派 `source_auditor`，只使用官方一手资料形成事实表。
5. **Plan**：跨文件/多阶段任务由 `curriculum_architect` 给先修与 Gate；主代理锁定互斥写入范围。
6. **Produce**：理论、实验和考核可在不同文件或周目录中并行；同一文件只有一个 owner。教程型任务还必须逐条执行 [`00_MANUAL_ACCEPTANCE_STANDARD.md`](../outputs/0_basic_training/00_MANUAL_ACCEPTANCE_STANDARD.md)，不能把 roadmap 扩写当作教程。
7. **Review**：`reliability_reviewer` 检查阻断项；主代理处理冲突。Lab 额外做“命令依赖追踪”：每个本地路径必须能回溯到前文的创建、下载或固定官方仓库步骤。
8. **Validate**：运行 `scripts/validate_curriculum.ps1`（Windows）或同等 Python 校验器及必要内容抽查；不把脚本通过等同于技术正确。
9. **State**：更新 memory 与进度；只记录实际完成的交付，不记录未发生的训练。
10. **Archive**：把原需求完整移动到历史区，恢复空白输入块；最后再读取确认。

## 失败处理

- 输入不完整但可安全推断：明确默认值并继续。
- 缺少会改变方案的用户选择：完成不依赖该选择的部分，再将阻断问题留为下一步。
- 官方资料与源计划冲突：保留两个版本，按环境分支，不静默覆盖。
- 校验失败：不得归档为完成；修复后重跑。
- 写入目标越过项目根目录：立即停止该动作。

## 完成定义

交付物存在且可读、校验通过、状态/进度一致、原始来源未改、没有虚构运行证据、输入已归档且新输入块为空。
