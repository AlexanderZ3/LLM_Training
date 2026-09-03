# Week 11 拷问篇：Mini-WAM

> 规则：先闭卷作答，不看 04_REFERENCE_ANSWERS.md。每题先给结论，再画 shape/梯度/证据链。建议限时 45 分钟。  
> 评分：每题 0–3 分；0=无答案，1=名词，2=机制正确，3=能给不变量、失败证据和取舍。总分 72；54 分且 Debug/Design 均不低于 70% 才建议进入 Week 12。

## Recall

### W11-Q01

写出 z_t、z_future、z_pred、action_chunk 的典型 shape，并说明 k 与 H_a 是否必须相等。

### W11-Q02

写出 L_total 和 masked normalized MSE 的公式。分母为什么不能固定为 batch size？

### W11-Q03

stop-gradient 具体切断哪条梯度路径？它与 encoder.eval 有何区别？

### W11-Q04

列出 A/B/C 三组实验的唯一区别和各自回答的问题。

## Explain

### W11-Q05

为什么 last-frame baseline 对慢变化机器人视频尤其重要？

### W11-Q06

为什么 shuffled-action 比 zero-action 或 no-action 更强？

### W11-Q07

为什么 L_future 降低不能推出 rollout success 提高？

### W11-Q08

解释 λ_world 如何通过共享参数制造正迁移或梯度冲突。

## Apply

### W11-Q09

给定 B=8、V=2、N=196、D=768、H_a=16、A=7，写出 patch latent 和 action 的 shape，并说明 mask 如何广播。

### W11-Q10

设计一个无 fixed point 的 batch shuffle；batch size 为 1 时怎么办？

### W11-Q11

设计一个只用 256 windows 的 overfit gate，列出至少五项通过条件。

### W11-Q12

给出公司 V100 上 PyTorch 2.1.0 的正确 FP16 更新顺序，并指出 clip 的位置。

## Debug

### W11-Q13

B/C 的 L_future 曲线几乎重合。按优先级给出定位步骤。

### W11-Q14

L_future 很低，但 prediction variance 接近 0。你如何判断是表示塌缩、mask 错误还是数据静止？

### W11-Q15

训练 loss 正常，dev future error 突然很差。列出数据、模型和系统三类根因及证据。

### W11-Q16

FP16 中 scaler 连续回退且 world-head grad 出现 Inf，如何最小化定位？

## Design

### W11-Q17

设计一份 pre-registration，使你看完结果后无法偷偷换指标或调 λ。

### W11-Q18

若 encoder 很大且冻结，设计 latent cache；列出 cache manifest 的必要字段和失效条件。

### W11-Q19

设计一个证明 world error 能做 early failure score 的独立评测。

### W11-Q20

设计 A/B/C 在单卡与四卡之间的公平迁移，说明 global batch、seed 和 samples seen 如何处理。

## Trade-off

### W11-Q21

patch latent 与 pooled latent 各有什么学习能力和 Infra 成本？

### W11-Q22

future offset 太短和太长分别会导致什么偏差？如何只用一个主 offset 做决定？

### W11-Q23

何时应冻结 encoder，何时允许联合训练？给出风险和证据门槛。

### W11-Q24

出现“B 的 future error 优于 A/C，但 action/OOD 与 A 无差异且成本增加 35%”时，你会给什么结论标签，下一步是什么？
