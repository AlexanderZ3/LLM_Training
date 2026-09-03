# Week 14 拷问篇：Capstone 答辩

> 先闭卷，不看答案篇。建议 45 分钟口述，再用 15 分钟补图。每题 0–3 分；总分 72。建议至少 58 分，且 Debug/Design 任何题不得为 0。

## Recall

### W14-Q01

区分 repeatability、reproducibility 和 determinism。

### W14-Q02

列出完整 resume checkpoint 至少十类状态。

### W14-Q03

写出 global batch、scaling efficiency 和 relative error 公式。

### W14-Q04

列出 run manifest 的最小 lineage 字段。

## Explain

### W14-Q05

为什么“能 load checkpoint”不等于“能正确 resume”？

### W14-Q06

为什么多卡结果不要求逐位一致，但仍必须做 fixed-global-batch 等价？

### W14-Q07

为什么小模型 8 卡效率低不能直接判为 NCCL 故障？

### W14-Q08

为什么算法未赢仍可能是成功 Capstone？

## Apply

### W14-Q09

单卡 micro-batch=8、accumulation=4。四卡和八卡保持 global batch=32 时，给出可行 micro-batch/accumulation 组合。

### W14-Q10

画出从 claim 到 checkpoint、config、data、eval 的 lineage。

### W14-Q11

设计一个不破坏真实 checkpoint 的 incomplete-checkpoint 测试。

### W14-Q12

设计 1/4/8 卡 correctness→scaling 的执行顺序和每步 Gate。

## Debug

### W14-Q13

resume 后第一步 loss 跳高 20%，按优先级定位。

### W14-Q14

四卡 loss 与单卡偏差大，但每个 rank 参数彼此一致。可能是什么？

### W14-Q15

step p50 正常而 p95 每 100 step 尖峰，如何归因？

### W14-Q16

同一 checkpoint 两次 eval 指标不同，如何判断是 generation、数据、环境还是代码 lineage 问题？

## Design

### W14-Q17

设计原子 checkpoint 提交协议，说明 COMPLETE marker 和 manifest 的作用。

### W14-Q18

设计 pre-registration，防止看到 test 后换主指标。

### W14-Q19

设计一个不依赖网络/Codex/手改源码的 clean-room 测试。

### W14-Q20

设计公司 V100 结果与个人 5070 Ti 公开复现之间的安全隔离。

## Trade-off

### W14-Q21

Capstone 应选 Mini-WAM 还是 FinExec？给出可执行选择标准。

### W14-Q22

严格 determinism、训练吞吐与跨卡扩展之间如何权衡？

### W14-Q23

DDP 与 FSDP 的 checkpoint 格式应怎样选，何时允许 world-size 迁移？

### W14-Q24

若系统 Gate 全过但 treatment 主指标无提升，报告应怎样写，下一步是什么？
