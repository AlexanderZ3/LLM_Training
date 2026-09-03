# Week 02 拷问篇：nanoVLM / VQA

> 闭卷 45 分钟，禁止先看答案。每题要包含 shape、可观测证据或受控对照；只背术语不算通过。

## Recall

1. **W02-R01**：列出从 `pixel_values` 到 `combined_embeds` 的全部 shape。
2. **W02-R02**：attention mask 与 label mask 分别控制什么？
3. **W02-R03**：freeze、`no_grad()`、`eval()` 的区别是什么？
4. **W02-R04**：projector-only checkpoint 与可恢复训练 checkpoint 分别至少含什么？

## Explain

5. **W02-E01**：为什么 visual token 必须映射到 language hidden dimension？
6. **W02-E02**：在 decoder-only prefix 方案中，answer token 如何看到图像而图像 token 看不到答案？
7. **W02-E03**：为什么只训练 projector 能过拟合仍不足以证明视觉能力？
8. **W02-E04**：为什么 answer-only loss 优于让 question 也贡献 loss？

## Apply

9. **W02-A01**：`B=4,Nv=196,Dv=768,Dl=576,T=64`，给出投影前后、拼接与 labels shape，并计算每样本总序列长度。
10. **W02-A02**：设计 20 个 evaluator 单测，覆盖 normalization、EOS/PAD 与 invalid generation。
11. **W02-A03**：构造双 LR optimizer groups，怎样证明 frozen 参数没有被误更新？
12. **W02-A04**：设计固定子集，确保每次运行取得同样的 5k train/500 val，而不是只依赖 random seed。

## Debug

13. **W02-D01**：loss finite 且下降，但 projector grad 始终为 0，按顺序排查。
14. **W02-D02**：original 与 shuffled-image EM 相同，列出至少四个可证伪解释。
15. **W02-D03**：加载 222M checkpoint 出现大量 shape mismatch，哪些信息先核对，哪些“修复”禁止使用？
16. **W02-D04**：teacher-forcing loss 正常而 greedy 输出为空或只复述 prompt，如何定位？

## Design

17. **W02-G01**：设计 base/projector-only/light-unfreeze × 四种视觉条件的最小评测矩阵。
18. **W02-G02**：设计一个 collator 的 shape/mask 契约，能处理不同答案长度与 padding。
19. **W02-G03**：在不能外传任何图像/日志的公司环境里，如何留下可复核证据？
20. **W02-G04**：设计 image-token 数与显存/延迟关系的受控实验。

## Trade-off

21. **W02-T01**：保留所有 patch tokens 与只用 pooled token 的优劣是什么？
22. **W02-T02**：projector-only 与轻量解冻分别适合回答什么问题？
23. **W02-T03**：降低 batch、降低分辨率、缩短 text、减少解冻层，各自损失什么？
24. **W02-T04**：为什么 V100 上选择 FP16 原生 attention fallback，而不是强装 BF16/FlashAttention-2？

## 评分

每题 0–4，总分 96。`≥77` 且 Debug、Design 各 `≥10/16` 才通过；出现“shuffle 不降仍声称学会视觉”或“忽略 checkpoint shape mismatch”任一关键错误，本周 Gate 不通过。记录 AI 辅助等级。

