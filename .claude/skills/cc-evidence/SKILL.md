---
name: cc-evidence
description: 记录并分类一条用户提交的证据（测试输出、日志摘要、口试回答、实验结论）：敏感性检查 → cc-evidence-coach 分类 → 写 memory/cc/evidence JSON → 更新证据索引与状态。用户说“记录证据”“我跑完了，结果是…”时使用。
---

# /cc-evidence — 记录一条证据

## 1. 敏感性检查（先于一切）

证据文本若包含以下任一项，停止并请用户改写为抽象摘要：公司内部路径、内部模型/项目/客户名、真实机器拓扑或带宽数字、原始 loss 曲线图片、trace 文件、checkpoint 内容、可拼接重识别内部系统的多个字段。

个人 5070 Ti 或租用环境上用公开数据产生的证据不受此限。

## 2. 定位

确认证据对应的周/天/任务卡与步骤。用户没说时问一句。读取对应 `03_TASK_CARDS/dayD.md` 的证据字段块，核对用户是否填齐。

## 3. 分类

派 `cc-evidence-coach`，输入：证据文本、任务卡、相关 `lab/` 脚本、`memory/cc/03_EVIDENCE_LOG.md` 最近记录。得到 status、failure_class、依据、能力观察、最小提示、回归任务、建议 JSON 字段。

## 4. 写入

- 文件：`memory/cc/evidence/YYYY-MM-DD_wNN_dD_<short>.json`，结构遵守 `.claude/schemas/evidence.schema.json`。
- `memory/cc/03_EVIDENCE_LOG.md` 追加一行：日期 | 周/天 | skill | status | failure_class | AI 等级 | 回归日期 | 文件名。
- 若 status 为 `FAIL-*` 或 `INCONCLUSIVE`：在 `memory/cc/01_STATE.md` 的“待闭合”列表加一项，并把回归任务写进下一步。

## 5. 反馈用户

只给：判定与依据、最小提示、回归任务、下一步唯一动作。不给分数，不给完整答案（除非用户明确要求）。

## 规则

- `PASS` 只描述这次运行；不写“已掌握”。
- 用户自述 AI 等级与证据不一致（如自述 A0 但代码风格明显 AI 生成）时，如实记录“等级存疑”并说明依据。
- 证据不足就写 `INCONCLUSIVE` 并说明缺哪一个实验，不猜。
