# 实验合同（长跑前必填）

> 没有这份合同不启动超过 5 分钟的训练。用户自己填写；cc 只检查完整性与可判定性。

```text
date:
week/day:
hypothesis:               # 一句可证伪的陈述
independent_variable:     # 只改这一项
controlled_variables:     # seed、数据子集、global batch、lr、精度、步数……
input/output shape:
primary_metric:           # 名称 + 计算口径 + 在哪个 split 上
correctness_invariant:    # 例如：valid loss 不用于任何训练决策；resume 后 step k 的 loss 与连续训练相等
stop_condition:           # 步数/时间/指标阈值/NaN
decision_rule:            # 结果为 X 则结论 A，为 Y 则结论 B，否则 INCONCLUSIVE
budget:                   # 时间、显存、（若租用）费用上限
ai_assistance:            # A0–A4
env:                      # 5070ti / v100 / h100
```

## cc 检查项

- [ ] 假设可证伪，decision_rule 覆盖三种结果；
- [ ] 只有一个自变量；
- [ ] primary_metric 的口径与 `lab/` 中日志字段一致；
- [ ] stop_condition 含 NaN/发散处理；
- [ ] 公司环境时，产物路径全部在公司内。
