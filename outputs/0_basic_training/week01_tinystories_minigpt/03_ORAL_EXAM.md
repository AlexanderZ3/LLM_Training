# Week 01 拷问篇：TinyStories MiniGPT

> 先闭卷，建议 45 分钟。不得查看 `04_REFERENCE_ANSWERS.md`。每题先说结论，再画 shape/状态流，最后给可证伪实验。问题本身不包含答案。

## Recall

1. **W01-R01**：写出 next-token 训练中 `input_ids`、`labels`、`logits` 的 shape 与错位关系。
2. **W01-R02**：列出一个可等价 resume 的 checkpoint 至少要保存哪些状态。
3. **W01-R03**：写出多头注意力从 `[B,T,D]` 到 scores 的完整 reshape/transpose。
4. **W01-R04**：autocast、GradScaler、unscale、gradient clipping 的正确顺序是什么？

## Explain

5. **W01-E01**：为什么 causal mask 必须在 softmax 前施加？
6. **W01-E02**：为什么 byte BPE 可以做到无 OOV，同时 tokenizer 仍可能不可复现？
7. **W01-E03**：为什么扩大词表可能既加速又减慢训练？
8. **W01-E04**：为什么 AdamW 的 decoupled weight decay 不等价于在 loss 中加 L2？

## Apply

9. **W01-A01**：`B=8,T=256,D=384,Nh=6,V=8192`，逐项给出 Q/K/V、scores、logits shape，并指出最大的序列相关张量。
10. **W01-A02**：micro-batch=16、accumulation=4、单卡，200 optimizer steps 共处理多少序列与最多多少 token？scheduler 应走多少步？
11. **W01-A03**：设计一个 128-window overfit gate，明确冻结变量、成功判据与失败后的下一检查。
12. **W01-A04**：设计 token-weighted validation NLL/PPL 聚合，处理最后一个短 batch。

## Debug

13. **W01-D01**：loss 长期接近 `ln(V)`；给出按信息增益排序的诊断树。
14. **W01-D02**：FP32 正常，FP16 每隔几步 scale 减半并 skip；你需要哪些日志来定位首个坏模块？
15. **W01-D03**：20+resume 后 step 21 loss 差 8%，但模型权重成功加载；如何分层排查？
16. **W01-D04**：validation PPL 改善很多，生成却重复严重；如何判断是评测、数据、模型还是解码问题？

## Design

17. **W01-G01**：为只能输入、不能输出的公司环境设计一个终端自解释训练入口和证据包。
18. **W01-G02**：设计 tokenizer 版本契约，使两个月后能判断两个 checkpoint 是否可比较。
19. **W01-G03**：在 V100 32GB 与 5070 Ti 16GB 间设计公开复现，怎样避免把硬件差异误写成算法差异？
20. **W01-G04**：设计一个 attention oracle，既检查数值也检查 mask 语义。

## Trade-off

21. **W01-T01**：learned position embedding 与 RoPE 在本周为何可能做不同选择？
22. **W01-T02**：weight tying 的收益、限制及检查方法是什么？
23. **W01-T03**：更大 batch、更多 accumulation、更多训练 step 三者怎样改变优化与吞吐解释？
24. **W01-T04**：为什么本周应先做 FP32 reference 再做 FP16，而不是直接追求最快吞吐？

## 评分规则

每题 0–4 分：0 无法回答；1 复述名词；2 原理基本正确；3 能给 shape/公式/实验；4 能指出边界、失败证据和 trade-off。总分 96；`≥77` 且 Debug/Design 均不低于 60% 才建议进入下一周。核心证据需 A0/A1 完成。

