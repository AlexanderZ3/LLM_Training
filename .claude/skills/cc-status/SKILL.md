---
name: cc-status
description: 读取 memory/cc 状态与证据日志，报告当前周/天、Gate 状态、未闭合项，并给出唯一下一步动作。用户问“我现在到哪了”“下一步做什么”时使用。
---

# /cc-status — 当前位置与唯一下一步

## 步骤

1. 读 `memory/cc/01_STATE.md`、`03_EVIDENCE_LOG.md` 最近 10 行、`00_INPUT.md` 顶部输入块是否为空。
2. 检查对应周包是否存在（`outputs/1_cc_coaching/weekNN_*/`）以及最近一次校验结果。
3. 输出：

```markdown
## 当前位置
周 / 天 / 周包状态 / Gate 状态

## 未闭合项
- FAIL-* 或 INCONCLUSIVE 证据及回归日期
- 待处理的 00_INPUT 需求（若有且 AGENT 为 cc/both）

## 唯一下一步
一句话 + 对应命令（/cc-day NN D 或 /cc-build-week NN 或 /cc-gate NN 或 /cc-evidence）
```

## 规则

- 只给一个下一步，不给菜单。
- 状态文件与证据日志不一致时，以证据日志为准并指出不一致。
- 不修改任何文件。
