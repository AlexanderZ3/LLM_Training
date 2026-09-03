# Workflow 01：每周陪跑闭环

## 0. 进入条件

- 用户选择 Week NN；
- 前一周 Gate 已通过，或明确记录为何进行受控补课/并行；
- 执行环境、可用时间、数据许可和 AI 辅助等级已声明。
- 首次执行前已阅读该周 Foundations、Lab 和 [`详细手册验收规范`](../outputs/0_basic_training/00_MANUAL_ACCEPTANCE_STANDARD.md)；Lab 中所有数据/模型来源、revision、文件与命令依赖均已解析。

## 1. Readiness（10–20 分钟）

使用该周 `03_ORAL_EXAM.md` 的 3–5 个先修题做 A0/A1 基线。记录能否独立回答，不先打开参考答案。

输出：`READY / NEEDS-REMEDIATION / BLOCKED-BY-ENV`，以及最多三个薄弱点。

## 2. 实验合同（10 分钟）

用户先写：

```text
hypothesis:
independent_variable:
controlled_variables:
input/output shape:
primary_metric:
correctness_invariant:
stop_condition:
decision_rule:
AI assistance: A0-A4
```

没有这份合同不启动长跑。

## 3. 分级运行

```text
static/shape test
→ 1–5 step dry run
→ tiny-set overfit
→ FP32 reference
→ FP16 stability
→ checkpoint/resume
→ distributed equivalence
→ performance/ablation
```

每过一门再解锁下一门。任意 NaN、数据泄漏、evaluator 错误或 resume 不完整都优先归为 correctness blocker。

## 4. 证据回看

教练要求用户提交获准的摘要字段，而不是敏感原件：配置标识、数据 hash 的非敏感替代标识、step 范围、关键统计、状态与异常描述。

分类顺序：

1. 测量错误；
2. 数据/任务合同错误；
3. 数值/优化错误；
4. 分布式/系统错误；
5. 模型假设失败；
6. 证据不足。

## 5. Socratic debug

用户答错时：复述现象 → 要求给一个不变量 → 给最小提示 → 让用户提出区分两个假设的实验。除非用户明确要求，不直接给完整修复。

## 6. Oral Gate

从六层各取一题：Recall、Explain、Apply、Debug、Design、Trade-off。答案按 0–4 评分，并分别记录 K/A/D/E；基线前不换算精确 mastery。

## 7. 收口

记录本周最小证据、FAIL-MODEL/FAIL-SYSTEM/INCONCLUSIVE/PASS、AI 等级、薄弱项、回归日期和下一次唯一动作。

只有不同日期的独立证据与迁移/debugging 证据齐备，才允许标记 Gate 完成。
