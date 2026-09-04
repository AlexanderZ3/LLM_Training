---
name: cc-gate
description: 第 NN 周的口试与 Gate 判定：检查证据齐全性 → 六层各抽题闭卷口试 → 0–4 评分与 K/A/D/E 观察 → PASS/EXTEND 判定 → 薄弱项与回归任务 → 更新状态。用法 /cc-gate 01。
---

# /cc-gate NN — 周末口试与 Gate

## 1. 证据齐全性检查

读 `memory/cc/03_EVIDENCE_LOG.md`，对照本周 `00_WEEK_CARD.md` 的 Gate 要求核对：

- 至少一次 A0/A1 限时表现；
- 至少一次 debugging 或跨场景迁移证据；
- 两个不同日期的证据；
- 没有未闭合的 `FAIL-*`。

缺项时不进入口试，先列出缺什么、用哪张卡补。用户坚持要考时，口试可以做，但 Gate 结论最高为 `EXTEND`。

## 2. 口试

从 `04_ORAL_EXAM.md` 六层各抽 1–2 题（共 6–10 题），优先选与本周证据中薄弱环节相关的题。逐题：

1. 呈现题目与限时；
2. 用户闭卷作答；
3. 对照 `05_REFERENCE_ANSWERS.md` 的评分点打 0–4 分，记录 K/A/D/E 观察；
4. 答错时只给一次最小提示并允许补答（补答分数单独记录）；
5. 追问一次。

不在口试过程中展示参考答案全文。

## 3. 判定

```text
PASS    ：证据齐全 + 口试无 0–1 分题 + 能解释本周关键 trade-off
EXTEND  ：证据缺项或有 0–1 分题；给出回归任务与复考日期，本周延长
BLOCKED ：环境或许可阻断，非能力问题；记录阻断原因
```

不因为日历到了而 PASS。基线诊断前不换算精确 mastery。

## 4. 输出

写 `outputs/1_cc_coaching/weekNN_<slug>/06_GATE_RECORD_YYYY-MM-DD.md`：证据清单、逐题得分与观察、判定、薄弱项（最多 3 个）、回归任务（做什么、何时、验收什么）、下一周入口条件。

## 5. 状态

- `memory/cc/01_STATE.md`：本周 Gate 状态、下一周或回归任务作为唯一下一步。
- `memory/cc/03_EVIDENCE_LOG.md`：口试作为一条证据记录（skill=oral_gate）。
- `01_PROGRESS.md`：追加 `[cc]` 记录。
