# Week 06 基础篇：从一个训练 step 推导时间与显存

> 发布状态：从零执行手册的原理篇。下文所有耗时、显存与输出均为公式、估算或示例，不是用户实测。公司 8×V100 32GB/Linux/PyTorch 2.1 为用户自述待核验。核验日期：2026-09-03。

## 1. 本周问题、产物与衔接

本周回答：“单卡一步训练的时间和显存到底花在哪里，某个优化是否真的命中瓶颈？”

最终产物由 Lab 从空目录生成：

- 一个只依赖 Python、PyTorch 的完整 TinyLM 训练器；
- train、profile、memory scan、checkpoint/resume、eval 五条可运行路径；
- metrics.json、checkpoint.pt、短 profiler trace；
- batch/sequence 显存表和一次单变量 A/B；
- decision.md，结论只能是 PASS、FAIL-SYSTEM、FAIL-MODEL 或 INCONCLUSIVE。

它承接前几周的训练循环，并给 Week 07 DDP 提供“单卡正确性与吞吐口径”。Week 07 不直接复制此脚本，避免隐式依赖；它会用同一不变量重新实现分布式版本。

## 2. 前置知识与三条环境边界

读者需会 Python、PyTorch Module、forward/backward、AdamW、基础 shell。不了解 CUDA 异步时先掌握：Python 发出 kernel 后通常立即返回，GPU 在另一条时间线上执行。

| 资源路径 | 本周角色 | 不能做 |
|---|---|---|
| 公司 8×V100/Linux | 只占一张卡建立 V100 FP16 主证据 | 不导出公司代码、数据、日志、trace、图片、checkpoint、配置或性能数字 |
| 个人 RTX 5070 Ti | 用同一合成数据从零重跑公开缩小版 | 不把新架构 kernel/吞吐外推给 V100 |
| 可选 H100 | 有现成授权时做额外 backend 对照 | 不替代 V100 主验收，不为本周专门申请 |

V100 是 Volta SM70，可使用 FP16 Tensor Core；通常没有原生 BF16 Tensor Core 路径。公司 PyTorch 2.1 是约束，不强升 torch/CUDA。官方 FlashAttention-2 CUDA 路径面向 Ampere、Ada、Hopper；本周代码使用 PyTorch 2.1 内置 Transformer/eager 路径。

## 3. 数据、模型、版本与许可

核心实验不下载外部数据或模型，避免把 I/O、许可和网络混入 profiling 基线。

| 对象 | 来源与固定方式 | 许可/限制 |
|---|---|---|
| 合成 token | Lab 的 SyntheticTokens 由 seed=17、vocab、seq、sample_id 决定 | 运行时生成，无第三方数据许可；不得称为语言建模质量证据 |
| TinyLM | Lab 内完整教学实现，结构与 CLI 写入 metrics.json | 本项目教学代码；对外发布前由项目所有者选择许可 |
| PyTorch | 官方 tag v2.1.0 的 API 文档；实机版本必须精确记录 | PyTorch 官方 BSD-style license；公司安装规则优先 |
| 可选 Week 01 权重 | 只有提供 checkpoint SHA-256、模型 config 和授权时才能替换 | 缺任一项即不用；公司权重不得外带 |

因此“下载和离线替代”在核心路径中等价：无需下载；完全离线仍能生成固定合成样本。合成数据只能通过代码/系统门，不能支撑模型效果结论。

输入样例：

    input_ids shape  = [B,T] = [2,128], dtype=int64
    targets shape    = [2,128], dtype=int64
    logits shape     = [2,128,8192], dtype=float32/float16

targets 是 input_ids 向左移动一位的确定性序列。

## 4. 从输入到 loss 的完整数据流

1. SyntheticTokens 根据 sample_id 和 seed 构造长度 T+1 的 token。
2. x=tokens[:-1]，y=tokens[1:]。
3. embedding 把 [B,T] 变成 [B,T,D]。
4. 每个 Transformer block 执行 self-attention 与 MLP。
5. tied LM head 输出 [B,T,V]。
6. cross entropy 把 logits 展平为 [B×T,V]，target 为 [B×T]。
7. FP16 路径中 autocast 控制算子 dtype，GradScaler 缩放 loss 后 backward。
8. unscale 后 clip，optimizer step；若 overflow，step 被跳过。
9. checkpoint 保存 model/optimizer/scaler/RNG/step/config，并以最后发布的 sidecar manifest 绑定 checkpoint SHA-256、代码 SHA、格式版本与 torch 版本；任何 `torch.load` 前先核验。

## 5. Shape、dtype 与 worked example

设 B=2、T=128、D=512、H=8、Dh=64、V=8192、MLP 倍率 4：

| 张量 | shape | 元素数 | FP16 仅按张量体积 |
|---|---:|---:|---:|
| residual | [2,128,512] | 131,072 | 0.25 MiB |
| Q/K/V 各自 | [2,8,128,64] | 131,072 | 0.25 MiB |
| attention score | [2,8,128,128] | 262,144 | 0.50 MiB |
| MLP intermediate | [2,128,2048] | 524,288 | 1.00 MiB |
| logits | [2,128,8192] | 2,097,152 | 4.00 MiB |

这些只是单张量，不含 backward 保存、多层存活期、allocator 和 workspace。T 从 128 变 256 时 residual 约 2 倍，显式 attention score 约 4 倍。

## 6. 核心公式

一步时间：

    T_step = T_data + T_h2d + T_forward + T_backward
           + T_unscale_clip + T_optimizer + T_other

阶段会重叠，不能机械求和。吞吐：

    samples_per_second = B / T_step
    tokens_per_second = B × T / T_step

普通 FP32 参数、梯度和 AdamW 两个 moment 的粗略 model-state 下界：

    M_state ≈ P × (4 + 4 + 4 + 4) = 16P bytes

还需加 activation、临时 buffer、CUDA context、allocator 与 profiler 开销。

算术强度：

    I = FLOPs / bytes_moved
    attainable ≤ min(peak_compute, I × memory_bandwidth)

它用于提出 compute-bound 或 memory-bound 假设，不用于代替实测。

## 7. 最小可运行原理例子

以下命令不依赖本地文件，可在已有 PyTorch 环境运行；它只验证 shape/loss，不是性能结果：

    python - <<'PY'
    import torch
    B,T,D,V = 2,8,16,32
    torch.manual_seed(17)
    x = torch.randint(0,V,(B,T))
    y = torch.roll(x,-1,1)
    emb = torch.nn.Embedding(V,D)
    head = torch.nn.Linear(D,V)
    logits = head(emb(x))
    loss = torch.nn.functional.cross_entropy(logits.reshape(-1,V), y.reshape(-1))
    print({"x": list(x.shape), "logits": list(logits.shape),
           "loss_finite": bool(torch.isfinite(loss))})
    PY

预计示例字段：x=[2,8]、logits=[2,8,32]、loss_finite=True。具体 loss 数值是待验证输出。

## 8. 正确计时与 Profiler

- CPU wall 用 perf_counter。
- GPU step 用 CUDA events；记录所有 event 后在窗口末同步，避免每步 synchronize。
- 正式吞吐使用 warmup 后、取 batch 之前开始且最终 GPU 同步后结束的端到端 wall 窗口；它包含 DataLoader、H2D、forward/backward/step。CUDA event 的 kernel 时间与端到端 tokens/s 必须分列，不能相互冒充。
- warmup 排除 context 初始化、kernel cache、allocator 和 optimizer lazy init。
- 报 p50/p95，不只报均值。
- profiler 的 record_shapes、profile_memory、with_stack 都会改变 workload；短 trace 定位，正式吞吐关闭 profiler。
- self CUDA time 是 op 自身；total CUDA time 包含子调用，嵌套项不可直接相加。

## 9. 显存与算法—infra 联动

- allocated 是活跃 tensor；reserved 是 caching allocator 保留块；reserved 大不自动等于泄漏。
- optimizer moment 常在第一次 step 才创建，所以必须测首步之后。
- activation checkpointing 少存中间量、backward 重算，属于 compute-memory trade-off。
- batch 常近线性增加 activation；sequence 对显式 attention 有二次轴。
- FP16 overflow 会使 optimizer step 被跳过；必须记录 scale、skip、grad norm。
- 通过减小 batch 消除 NaN 不能替代定位错误算子。

## 10. 正确性不变量

任一失败都停止正式性能结论：

- x/y/logits shape 与 vocab 边界正确；
- loss、grad、parameter 全为有限值；
- clip 发生在 scaler.unscale_ 之后；
- resume 在反序列化前核验非空 checkpoint、sidecar SHA、代码/torch/格式版本，再恢复 model、optimizer、scaler、RNG 和 step；DataLoader 使用独立 generator，重建 iterator 不得推进模型全局 RNG；
- resume 下一固定 batch 的 loss 与连续 run 在预注册容差内；
- eval 使用独立 seed 与不相交 sample-ID 区间；strict load 成功不等于 eval 独立，训练样本重用即为泄漏；
- A/B 只改变一个变量；
- warmup、timed steps、seed、model、data、precision 一致；
- profiler-on 吞吐不当作正式吞吐；
-主 V100 run peak reserved <30.5GB。

## 11. 失败模式与诊断树

| 症状 | 最小检查 | 候选根因 | 修复与回归 |
|---|---|---|---|
| Python 计时异常小 | CUDA event 对照 | 异步 launch | 仅窗口边界同步，再跑 20+100 |
| reserved≫allocated | memory_summary、固定 shape | 缓存、碎片、旧引用 | 去除引用/动态 shape；不用 empty_cache 伪装 |
| profiler 后慢很多 | on/off 同配置 | tracing 开销 | 缩 active steps；正式关闭 |
| step 0 很慢 | 分离 warmup | context/cache/lazy state | 丢弃 warmup，首步另报 |
| FP16 skip | scale、grad norm、输入范围 | overflow、loss异常 | FP32 tiny reference、定位模块 |
| GPU util 高但吞吐低 | top ops、调用数、tokens/s | 小 kernel/低效 shape | 单变量 shape A/B |
| eval 无法加载 | checkpoint key/config | 仅权重或架构不一致 | 使用完整 checkpoint，重跑独立 eval |

## 12. 自检题

1. P=120M 时 16P 粗估是多少 GB？遗漏哪些项？
2. B=4、H=8、T=512 时 attention score 有多少元素？
3. 为什么每个 Python 函数时间相加可能大于或小于 T_step？
4. sequence 翻倍时哪些张量线性、哪些二次？
5. 为什么 optimizer 首步必须包含在显存扫描的 warmup 中？
6. 设计 workers=0 与 workers=4 的单变量 A/B。
7. 何时 8% 提升可接受，何时 15% 也应回滚？
8. 为什么 V100 调用 SDPA API不等于使用官方 FA2？

## 13. 官方资料

- [PyTorch v2.1 profiler](https://docs.pytorch.org/docs/2.1/profiler.html)
- [PyTorch v2.1 CUDA memory](https://docs.pytorch.org/docs/2.1/notes/cuda.html#cuda-memory-management)
- [PyTorch v2.1 AMP examples](https://docs.pytorch.org/docs/2.1/notes/amp_examples.html)
- [PyTorch v2.1 SDPA](https://docs.pytorch.org/docs/2.1/generated/torch.nn.functional.scaled_dot_product_attention.html)
- [NVIDIA Volta tuning guide](https://docs.nvidia.com/cuda/volta-tuning-guide/)
- [FlashAttention 官方仓库硬件要求](https://github.com/Dao-AILab/flash-attention)

链接与可变事实核验日期：2026-09-03。
