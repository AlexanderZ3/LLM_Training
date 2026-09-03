# Week 13 拷问篇：Scaling、FSDP、LoRA 与 GRPO

> 先闭卷，不看答案篇。每题 0–3 分；满分要求公式/机制、证据与取舍。总分 72；54 分且 Debug/Design 各至少 70% 才进入 Capstone。

## Recall

### W13-Q01

写出 mixed-precision AdamW 的模型状态显存组成，为什么常用 12–16 byte/parameter 只是估算？

### W13-Q02

写出 LoRA 的更新公式、A/B shape 和参数量。

### W13-Q03

列出 FSDP Full Shard 在 forward/backward 的主要 collective。

### W13-Q04

写出 GRPO 组内标准化 advantage 的最小公式。

## Explain

### W13-Q05

为什么 DDP 不会帮助 4B base 分摊单卡参数显存？

### W13-Q06

为什么 FSDP 能省 model-state 显存，却不保证解决 activation OOM？

### W13-Q07

为什么不同 tokenizer 的 raw train loss 不适合直接比较模型规模？

### W13-Q08

为什么 reward total 上升仍可能是 reward hacking？

## Apply

### W13-Q09

对一个 out=4096、in=4096 的 Linear，计算 rank 8 和 rank 16 的 LoRA 参数量。

### W13-Q10

设计 0.6B 与 1.7B 的公平 SFT 比较，列出至少八个固定项。

### W13-Q11

给出一个 FSDP 50-step smoke 的执行顺序与 PASS 条件。

### W13-Q12

为 FinExec 写六个 reward 分项，并说明依赖顺序。

## Debug

### W13-Q13

LoRA loss 完全不动，trainable parameter 百分比看似正常。如何逐层定位？

### W13-Q14

FSDP 相比 DDP 几乎不省峰值显存。给出至少五个可检验原因。

### W13-Q15

GRPO valid program 上升，但 evidence F1 下降、response length 上升。你如何判定和止损？

### W13-Q16

放大模型吞吐远低于纸面预测，如何区分 activation、communication、data、generation 和 checkpoint？

## Design

### W13-Q17

设计一张 model memory budget，必须包含哪些不按参数线性缩放的项？

### W13-Q18

设计 adapter checkpoint manifest，怎样避免 base/adapter/tokenizer 错配？

### W13-Q19

设计 GRPO adversarial reward test suite，至少覆盖八种攻击。

### W13-Q20

设计一个 5070 Ti、V100、H100 跨硬件实验，避免把 precision 与硬件差异误判为算法收益。

## Trade-off

### W13-Q21

1.7B 全参 FSDP 与 4B FP16 LoRA 如何选择？

### W13-Q22

LoRA rank 8 与 16 的收益、显存、通信和过拟合风险如何权衡？

### W13-Q23

什么条件下值得做 QLoRA？V100 上依赖不兼容时为何应回退？

### W13-Q24

放大模型执行准确率只提高 0.6pp，但显存增 2.3 倍、吞吐降 65%。如何给结论并决定下一步？
