# Week 08 拷问篇：FSDP 与 Checkpoint

> 先闭卷，禁止查看回答篇。建议 55 分钟。所有答案必须区分 PyTorch 2.1 FSDP1 与当前新 API，不能假设命令已跑。资料核验日期：2026-09-03。

## Recall

### W08-R01

写出 DDP、N-way Full Shard、S-way Hybrid 的每卡 model-state 粗略公式。

### W08-R02

列出 FSDP unit 在 forward/backward 的主要 collective。

### W08-R03

一个可恢复 checkpoint 至少保存哪些状态？

### W08-R04

定义 shard degree 与 replica degree；2×4 Hybrid 各是多少？

## Explain

### W08-E01

为什么 FSDP 不自动解决 activation OOM？

### W08-E02

为什么 wrap unit 太大和太小都会变慢或增大峰值？

### W08-E03

为什么 completion marker 必须最后写？

### W08-E04

为什么 8 卡保存→4 卡加载比 8→8 resume 难？

## Apply

### W08-A01

计算 0.6B、1.5B、4B 在 16P 粗估下的 DDP、/4、/8 model-state。

### W08-A02

给定 W shape [4096,4096]，描述 8-way shard、forward all-gather 与 backward reduce-scatter 的逻辑 shape。

### W08-A03

设计 DDP 与 FSDP 一步等价实验。

### W08-A04

设计同 world-size 的 50 step 保存→100 step 恢复测试，定义 next-batch 验收。

## Debug

### W08-D01

启用 Full Shard 后 peak memory 只下降 5%。给出排查顺序。

### W08-D02

checkpoint 目录存在，但加载时某些 rank hang。如何判断是否不完整提交？

### W08-D03

FSDP loss 与 DDP 在第一步就显著不同。列出至少六个检查点。

### W08-D04

Hybrid 的吞吐低于 Full Shard，是否证明 Hybrid 无用？怎样归因？

## Design

### W08-G01

设计 block-level wrap coverage 报告，如何处理 tied embedding/head？

### W08-G02

设计一个 crash-safe 的分布式 checkpoint 协议。

### W08-G03

设计 8-way Full 与 2×4 Hybrid 的公平拓扑对照。

### W08-G04

公司 PyTorch 2.1 不支持你想要的新 DCP/FSDP2 示例时，设计降级方案。

## Trade-off

### W08-T01

比较 Full Shard 与 2×4 Hybrid 的显存、通信范围和适用条件。

### W08-T02

比较 FSDP state sharding 与 activation checkpointing。

### W08-T03

比较 full、sharded、local state dict 的可移植性与峰值风险。

### W08-T04

什么时候 1.5B 短测有价值，什么时候应停留在 0.6B？

## 评分

每题 0–3，共 72。58+ 可独立；44–57 需 review；30–43 只会配置；低于 30 回到 DDP 与状态模型。

若建议升级公司 PyTorch 绕过约束、手工拼大 shard、忽略 completion marker，或外带公司代码/数据/日志/trace/图片/checkpoint/指标，相关题为 0 分并触发复训。
