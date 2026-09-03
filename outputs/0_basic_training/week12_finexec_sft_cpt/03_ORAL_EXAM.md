# Week 12 拷问篇：FinExec Evaluator、SFT 与 CPT

> 先闭卷，不看答案篇。每题 0–3 分：0=不会，1=背名词，2=机制正确，3=能给 schema、不变量、失败证据和权衡。共 72 分；建议 54 分且 Debug/Design 各至少 70% 才进入放大训练。

## Recall

### W12-Q01

写出 FinExec 输出的四个字段、类型和最重要的不变量。

### W12-Q02

列出 evaluator waterfall 的至少八个阶段，并说明 first failure 与全部 flags 的区别。

### W12-Q03

写出 target-only causal LM loss，说明 mask 和分母。

### W12-Q04

区分 program result、scaled result、reported answer。

## Explain

### W12-Q05

为什么 gold evaluator 100% 必须先于训练？100% 的范围是什么？

### W12-Q06

为什么 execution accuracy 通常比 program string exact match 更合理？

### W12-Q07

为什么 answer 正确但 program 无效仍应判过程失败？

### W12-Q08

为什么短 CPT 不能证明模型获得金融知识？

## Apply

### W12-Q09

给一条 prompt 40 token、JSON 20 token、padding 4 token 的样本，写出 labels 哪些位置为 -100，EOS 是否计 loss。

### W12-Q10

为 divide(subtract(128.4,97.1),97.1) 画 AST，并说明 scale=percent 的结果链。

### W12-Q11

设计 30 个 evaluator 单测应覆盖的类别，至少列出十类。

### W12-Q12

设计 base→SFT 与 base→short CPT→SFT 的公平比较，哪些变量必须固定？

## Debug

### W12-Q13

teacher-forced loss 快速下降但 valid JSON 仍接近 0，按优先级排查。

### W12-Q14

关闭 packing 时正常，开启后 execution accuracy 暴跌。如何证明是否跨样本泄漏？

### W12-Q15

gold evaluator 为 98.7%。你能否先训练？如何处理异常 gold？

### W12-Q16

多卡 fixed-global-batch 的前 20 step loss 与单卡偏差 8%，如何拆分数据、数学和系统根因？

## Design

### W12-Q17

设计一个不用 Python eval/exec 的 DSL parser/executor 安全边界。

### W12-Q18

设计一份 dataset manifest，使 evidence、split、normalization 和 lineage 可审计。

### W12-Q19

设计 checkpoint selection 与最终 test 协议，避免 test 调参。

### W12-Q20

设计一个能区分“只学格式”和“学会程序推理”的评测面板。

## Trade-off

### W12-Q21

严格 JSON schema 的收益与训练代价是什么？何时应该简化 schema？

### W12-Q22

关闭 packing 会损失什么，保住什么？什么证据足以重新开启？

### W12-Q23

0.6B 在单卡装得下时，DDP 与 FSDP 应如何选择？

### W12-Q24

SFT 提高 valid JSON 20pp，但 execution 只提高 1pp、evidence F1 下降。你如何定性并选择下一步？
