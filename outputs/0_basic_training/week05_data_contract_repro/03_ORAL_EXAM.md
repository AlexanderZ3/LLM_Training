# Week 05 拷问篇：数据合同 / 对齐 / 可复现

> 先闭卷，禁止看答案。回答必须区分“已确认、用户自述待核验、推断、未知”。

## Recall

1. **W05-R01**：列出最小 episode schema 与每个核心字段类型。
2. **W05-R02**：Task Card、Data Card、manifest 各自负责什么？
3. **W05-R03**：需要固定哪些 RNG 与 sampler/DataLoader 状态？
4. **W05-R04**：列出 validator 至少 12 类检查。

## Explain

5. **W05-E01**：为什么按 frame 随机 split 会产生严重泄漏？
6. **W05-E02**：为什么 normalization 必须只使用 train？
7. **W05-E03**：为什么相同 seed 不保证 bitwise deterministic？
8. **W05-E04**：为什么 dataset 名与 random seed 不足以重建数据版本？

## Apply

9. **W05-A01**：为 `o_k→a_{k+lag}` 定义正负号，并画 lag=+2 的时间线。
10. **W05-A02**：长度3、H=4 的 episode，逐k给 action window/mask，并说明终止语义。
11. **W05-A03**：设计 object-pose 单轴 OOD split，同时防 episode/object/seed泄漏。
12. **W05-A04**：设计12类 corruption suite，使每种错误有唯一预期错误码。

## Debug

13. **W05-D01**：训练loss很好但rollout几乎不动，按数据层优先给诊断树。
14. **W05-D02**：同seed第一batch一致，第7batch开始不同，如何定位 worker/augmentation/sampler？
15. **W05-D03**：lag相关峰在-1和+1接近，如何避免事后挑选？
16. **W05-D04**：发现官方train与test存在近重复轨迹，如何处理与报告？

## Design

17. **W05-G01**：设计 point-in-time data manifest，保证两个月后能唯一重建。
18. **W05-G02**：设计 step25 checkpoint/resume，使下一批sample IDs与增强参数一致。
19. **W05-G03**：设计一个不读取/导出原始图像也能复核的数据质量终端报告。
20. **W05-G04**：设计 Task-P→Task-D 可共享接口，同时防止动作语义偷换。

## Trade-off

21. **W05-T01**：absolute 与 delta action 的建模、归一化和部署权衡是什么？
22. **W05-T02**：strict deterministic 与高性能数据管线如何取舍？
23. **W05-T03**：streaming 与本地冻结数据集的复现/磁盘/随机访问权衡是什么？
24. **W05-T04**：RoboTwin、PushT、合成 episode 三个入口如何按一周目标选择？

## 评分

每题0–4，总分96；`≥77` 且 Debug/Design 各≥60%才通过。frame随机split、test统计参与normalization、真实lag在合成oracle失败时仍给确定结论，任一出现即Gate失败。

