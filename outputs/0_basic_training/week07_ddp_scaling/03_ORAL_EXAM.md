# Week 07 拷问篇：DDP 正确性与 Scaling

> 先闭卷，禁止打开回答篇。建议 50 分钟；必须写公式、样本不变量和 rank 控制流。题目不提供答案。资料核验日期：2026-09-03。

## Recall

### W07-R01

写出 global batch 公式，并解释三个因子。

### W07-R02

定义 world_size、rank、local_rank 与 CUDA_VISIBLE_DEVICES 映射。

### W07-R03

定义强扩展、弱扩展、speedup 与 E_N。

### W07-R04

列出一个最小 DDP 进程从启动到 cleanup 的顺序。

## Explain

### W07-E01

为什么 DDP 不节省单卡 model-state 显存？

### W07-E02

为什么 DistributedSampler 每个 epoch 要调用 set_epoch？

### W07-E03

DDP 梯度平均与 local mean loss 如何组合成 global mean？何时不成立？

### W07-E04

为什么 NV2 pair 更快不保证 8 卡效率更高？

## Apply

### W07-A01

构造 global batch=64 的 1/2/4/8 卡配置，并指出强扩展到 8 卡的风险。

### W07-A02

写出 K=4 gradient accumulation 的 no_sync 控制流。

### W07-A03

设计单卡与两卡一步等价测试，包括样本、随机性、loss 与参数检查。

### W07-A04

给定 throughput_1=1000、throughput_4=3100、throughput_8=4700，计算 E4/E8 并按源计划的弱扩展诊断门槛解释。

## Debug

### W07-D01

第二个 epoch 的每 rank 样本顺序与第一个完全相同，但程序无报错。根因和检测是什么？

### W07-D02

两卡训练 loss 比单卡低很多，且 LR 相同。给出分层排查顺序。

### W07-D03

8 卡 hang，rank 6 最后记录在 backward，其他 rank 已进入 eval barrier。怎样定位与安全止损？

### W07-D04

FP16 下 rank 3 跳过 optimizer step，其他 rank 更新。会发生什么，如何检测和修复？

## Design

### W07-G01

设计 sample audit，使 train padding 与 validation 重复计数都可被发现。

### W07-G02

设计 NV2/NV1/SYS 与两个四卡域的拓扑实验，列出必须固定的变量。

### W07-G03

设计 bucket_cap_mb A/B。什么证据出现前不应调 bucket？

### W07-G04

设计 rank-local 日志与 rank0 汇总协议，避免并发覆盖和局部成功假象。

## Trade-off

### W07-T01

比较每个 micro-step 同步与 no_sync accumulation 的正确性、通信和调试成本。

### W07-T02

比较固定 global batch 与固定 per-GPU batch 的研究问题，为什么不能混报？

### W07-T03

比较四卡长训与八卡低效长训的资源价值；何时 8 卡只保留为 topology sample？

### W07-T04

何时应该设置 find_unused_parameters，何时它会掩盖模型控制流问题？

## 评分

每题 0–3 分，共 72。3 分需答案、公式/shape、证据与边界齐全；2 分为正确但不完整；1 分仅术语；0 分为关键错误。

- 58–72：可以独立做 DDP scaling；
- 44–57：需 review；
- 30–43：只会启动，不足以证明正确；
- 0–29：回到 manual all-reduce。

建议导出公司代码、数据、日志、NCCL trace、图片、checkpoint 或指标，或声称 DDP 汇聚 256GB 透明显存，相关题直接 0 分。
