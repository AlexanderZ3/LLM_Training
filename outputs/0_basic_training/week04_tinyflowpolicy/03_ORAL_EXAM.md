# Week 04 拷问篇：TinyFlowPolicy

> 先闭卷，禁止查看答案。每题必须说清 shape、证据和边界。

## Recall

1. **W04-R01**：定义 `x0,x1,t,xt,ut,vθ` 及各自 shape。
2. **W04-R02**：写线性 path 的 flow target 与 masked loss。
3. **W04-R03**：写 Euler 从 t=0 到1的更新式。
4. **W04-R04**：列出 EMA 与 resume 必存状态。

## Explain

5. **W04-E01**：为什么训练 flow matching 不需求 ODE 解？
6. **W04-E02**：为什么模型必须以 t 为条件？
7. **W04-E03**：为何 action chunk 通常可用双向 attention，而语言 token 需要 causal attention？
8. **W04-E04**：为什么 offline action MSE 低不保证 rollout success？

## Apply

9. **W04-A01**：B=8,H=32,Da=7，mask 有 200 个有效 timestep，masked MSE 分母是多少？
10. **W04-A02**：长度3、H=4、actions `[10,11,12]`，逐 k 写 target 与 mask。
11. **W04-A03**：为恒定 velocity field 设计 solver oracle，并给预期误差。
12. **W04-A04**：设计 BC 与 Flow 的公平预算表，列出必须固定的变量。

## Debug

13. **W04-D01**：flow loss 明显下降但采样发散，给出有顺序的诊断树。
14. **W04-D02**：短 episode 的 loss 总是更低，如何证明/修复 mask denominator bug？
15. **W04-D03**：correct/zero/shuffled condition 三组相同，列出四个解释与实验。
16. **W04-D04**：20+resume 后 raw loss接近但 EMA eval偏离，排查哪些状态？

## Design

17. **W04-G01**：设计 two-stage overfit gate，分离 target实现与随机训练难度。
18. **W04-G02**：设计 solver steps—延迟—质量的受控实验。
19. **W04-G03**：设计一个只用终端统计、不导出轨迹图的 toy rollout报告。
20. **W04-G04**：设计 condition 接口，使未来能从 state扩展到image/language而不改训练器。

## Trade-off

21. **W04-T01**：one-step BC 与 stochastic Flow 各有什么建模/系统代价？
22. **W04-T02**：prepend condition、FiLM/AdaLN、cross-attention 如何取舍？
23. **W04-T03**：EMA decay 大小、更新频率和 FP32 shadow 的权衡是什么？
24. **W04-T04**：增加 action horizon 与增加 solver steps 分别影响训练/推理什么？

## 评分

每题 0–4，总分 96；`≥77` 且 Debug/Design 各≥60%通过。把 padding 分母写成固定 `B×H×Da`、EMA 在 skipped step 更新、或用单 seed offline MSE声称闭环更好，均为关键错误。

