# Week 06 拷问篇：单卡 Profiling

> 规则：先闭卷，禁止打开 04_REFERENCE_ANSWERS.md。建议 45 分钟作答，白板写公式与 shape。每题使用唯一编号；回答“看 GPU utilization”不构成证据。资料核验日期：2026-09-03。

## Recall

### W06-R01

写出一个训练 step 的主要时间分解，并指出哪些阶段可能重叠。

### W06-R02

allocated、reserved、peak allocated 分别是什么？

### W06-R03

对 P 个参数，普通 FP32 参数、梯度与 AdamW 两个 moment 的粗略下界是多少 bytes/parameter？哪些部分未包含？

### W06-R04

定义 self CUDA time、total CUDA time、warmup 和 timed steps。

## Explain

### W06-E01

为什么直接用 Python perf_counter 包住一个 CUDA forward 可能严重低估耗时？

### W06-E02

为什么 Transformer 序列长度翻倍，显存可能超过两倍？

### W06-E03

为什么 activation checkpointing 降低峰值显存，却可能降低吞吐？

### W06-E04

为什么 V100 上调用 PyTorch SDPA 不能证明已使用官方 FlashAttention-2？

## Apply

### W06-A01

给出 MiniGPT 的 residual、Q/K/V、attention score、MLP intermediate 与 logits shape；使用 B、T、D、H、Dh、V 表示。

### W06-A02

设计 batch 1/2/4/8 与 sequence 128/256/512 的显存扫描。如何保证能区分两个轴？

### W06-A03

你观察到 data_wait 占 35%。设计一个只检验 DataLoader workers 的 A/B。

### W06-A04

为一次 100 timed-step 测试定义最小日志 schema 和最终 summary。

## Debug

### W06-D01

reserved 为 29 GB、allocated 为 19 GB。列出至少四个假设，并说明验证顺序。

### W06-D02

开启 profiler 后吞吐下降 40%。这是模型回归吗？怎样判断？

### W06-D03

nvidia-smi 长期显示 99%，但 tokens/s 仍很低。你会检查哪些证据？

### W06-D04

FP16 run 在 step 37 出现 skipped optimizer step，loss 尚为有限值。如何定位，而不是直接降 batch？

## Design

### W06-G01

设计一个 ranges 层级，使数据、训练阶段和模型模块都能映射到 profiler，同时避免每步同步。

### W06-G02

设计一份能够在无 Nsight 权限时仍区分 data-bound、launch-bound 与 compute-bound 的 fallback。

### W06-G03

为视觉动作模型推导分辨率、摄像头数和 action horizon 对 token 数与显存的影响，并设计安全扫描顺序。

### W06-G04

设计公司 V100 与个人 5070 Ti 的公开复现实验，如何避免泄露和错误硬件归因？

## Trade-off

### W06-T01

比较增大 micro-batch、gradient accumulation 与 activation checkpointing 对吞吐、显存和训练数学的影响。

### W06-T02

比较 torch.profiler、Nsight Systems 与 Nsight Compute 的观察粒度和开销。

### W06-T03

何时 8% 的稳定吞吐提升可以接受，何时即使 15% 提升也应回滚？

### W06-T04

如果热门优化没有提升，怎样把负结果写成有价值的工程证据？

## 评分方式

每题 0–3 分：0 为无回答/关键错误；1 为术语级；2 为正确且有证据路径；3 为能写出公式、shape、不变量、边界与止损。总分 72：

- 58–72：可独立执行；
- 44–57：需小提示；
- 30–43：能使用工具但归因不足；
- 0–29：回到基础篇与最小实验。

任何建议导出公司代码、数据、日志、trace、图片或 checkpoint，或在 V100 上强行使用 BF16/官方 FA2，相关题记 0 分并触发安全复训。
