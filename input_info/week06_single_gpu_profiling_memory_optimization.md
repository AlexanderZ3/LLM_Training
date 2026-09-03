# 第 06 周执行手册：单卡 Profiling、显存模型与一次可信优化

> 对应原 12 周主线第 4 周。标准投入 7.5 小时。  
> 载荷：第 01 周 120M MiniGPT + 第 04 周 TinyFlow/DiT surrogate。  
> 本周目标：能回答一步训练的时间和显存花在哪里，并用 A/B 实验验证一个优化。

每日时间盒：10 分钟写性能假设，50–60 分钟采样/实验，15–20 分钟做终端汇总；第 2 小时只深挖一个最大瓶颈。

## 1. 工具与官方资料

| 工具 | 用途 | 获取 |
|---|---|---|
| PyTorch Profiler | op、shape、CPU/CUDA time、memory | https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html |
| torch.cuda.memory | allocated/reserved/peak/summary | https://docs.pytorch.org/docs/stable/cuda.html#memory-management |
| Nsight Systems | CPU、CUDA、NCCL 时间线 | 通常由管理员安装；https://developer.nvidia.com/nsight-systems |
| Nsight Compute | 单 kernel 指标 | 通常由管理员安装；https://developer.nvidia.com/nsight-compute |
| CS336 A2 Systems | profiling、benchmark、memory/distributed 思路 | https://github.com/stanford-cs336/assignment2-systems |

V100 适配：

- 做 CS336 A2 的 benchmark/memory/distributed 思路；
- 不做官方 FlashAttention-2 Triton 目标，因为官方 FA2 路径要求 Ampere 及更新架构；
- 不因完成作业而安装非官方 SM70 FlashAttention fork；
- attention 优化可比较 PyTorch 当前版本支持的原生实现、naive reference 和 checkpointing。

## 2. Profiling 的实验纪律

每个结果必须固定：

- 模型参数与结构；
- batch、sequence/action horizon、分辨率；
- precision；
- gradient accumulation；
- checkpointing；
- logging/eval/checkpoint 频率；
- warmup step；
- timed step；
- 数据来源；
- GPU 当时共享负载。

统一测量：

    warmup = 20 steps
    timed = 50–100 steps
    repeat = 3

同时报告：

- end-to-end step p50/p95；
- GPU-active time；
- data wait；
- forward/backward/optimizer；
- tokens/s 或 windows/s；
- peak allocated/reserved；
- loss scale 与 skipped steps。

## 3. 目录

    week06-profile/
      src/
        ranges.py
        bench.py
        memory_model.py
        summarize_profile.py
      configs/
        lm_120m.yaml
        tinyflow.yaml
      scripts/
        run_torch_profiler.sh
        run_nsys.sh
        run_ncu.sh
      reports/
        baseline.json
        optimization_ab.json
        profile_report.md

不要依赖导出 trace 才能理解结果。summarize_profile 必须在终端打印 top ops 和阶段表。

## 4. 先建立纸面时间/显存模型

### 参数状态

普通 AdamW 的粗略下界：

- FP32 parameter：4 bytes/param；
- gradient：4；
- first moment：4；
- second moment：4；
- 合计约 16 bytes/param；
- AMP autocast 通常不等于参数永久 FP16；
- 框架临时 buffer、allocator、activation 另算。

120M 参数仅 model states 粗估约 1.9GB；真实峰值通常由 activation、attention、临时 buffer 和 batch 决定。

### activation

不要背一个常数。记录每层主要 tensor：

- residual stream B×T×D；
- Q/K/V；
- attention score/probability B×heads×T×T，若实现显式物化；
- MLP intermediate B×T×4D；
- image/action tokens；
- backward 保存的中间结果。

本周要比较理论增长：

- batch 近线性；
- hidden 近线性或矩阵计算更高阶；
- sequence 对 residual 近线性，对显式 attention score 近二次；
- image resolution 通过 patch 数进入 sequence。

## 5. 每日安排

### 周一：插入 ranges 与可信计时

目标：把一步拆成可测的阶段。

使用 record_function 或 NVTX 标记：

    dataloader_wait
    h2d
    zero_grad
    forward
    loss
    backward
    unscale_clip
    optimizer_step
    scaler_update
    ema
    logging
    eval
    checkpoint

模型内部再标：

    embedding
    attention
    mlp
    condition_encoder
    action_head or lm_head

计时器要求：

- CPU wall time 用 perf_counter；
- GPU 段用 CUDA events，或同步后测；
- warmup 前不统计；
- eval/checkpoint 与 steady-state 分开；
- 不每 step synchronize 破坏真实性；
- 只在 benchmark 边界同步。

当天输出：

| phase | p50 ms | p95 ms | share |
|---|---:|---:|---:|
| data | | | |
| h2d | | | |
| forward | | | |
| backward | | | |
| optimizer | | | |
| other | | | |

单测：人为 sleep/矩阵 workload，确认 range 顺序和汇总不会漏 step。

验收：

- 阶段时间总和与 end-to-end 在可解释误差内；
- GPU asynchronous 问题被处理；
- cold start 与 steady-state 分开；
- 可从同一入口开启/关闭 profiling。

### 周二：PyTorch Profiler 的最小有效 trace

目标：找 top CUDA ops，而不是生成几 GB trace。

建议 schedule：

    wait 1
    warmup 2
    active 5–10
    repeat 1

配置：

- CPU + CUDA activities；
- record_shapes 只在短 trace 开；
- profile_memory 开；
- with_stack 谨慎，开销高；
- tensorboard trace 可保存本机，但终端 summary 必须存在。

终端至少打印：

    key_averages().table(sort_by="self_cuda_time_total", row_limit=20)
    key_averages().table(sort_by="cuda_memory_usage", row_limit=20)

分别 profile：

1. MiniGPT forward+backward；
2. TinyFlow forward+backward；
3. 只做 dataloader；
4. 只做 optimizer step。

对 top 10 op 写：

- 所属模块；
- 调用次数；
- self CUDA time；
- total CUDA time；
- input shape；
- 是否预期；
- 可以优化还是结构必需。

验收：

- trace 只覆盖计划 steps；
- top op 能映射回模型模块；
- profiler 开销通过有/无 profiler A/B 被量化；
- 不把 profiler 模式吞吐当正式训练吞吐。

可选第二小时：用 CS336 A2 的 benchmark 题型重写你自己的 model size/sequence sweep。

### 周三：显存 envelope 与 OOM 前停止

目标：用扫描建立可预测显存边界。

扫描只改一个变量：

#### MiniGPT

- batch：1/2/4/8/16/...；
- sequence：128/256/512/1024；
- checkpointing off/on。

#### TinyFlow/VLM-style

- batch；
- action horizon；
- image resolution 或 visual token 数；
- condition encoder freeze/unfreeze。

每点：

    reset_peak_memory_stats
    warmup 5
    measure 10
    record max_memory_allocated
    record max_memory_reserved

达到 reserved 28–30GB 后停止，不以真正 OOM 为每点终点。目标给通信/eval/波动留余量，主训练 reserved <30.5GB。

对照纸面模型：

| config | predicted GB | allocated GB | reserved GB | error % | explanation |
|---|---:|---:|---:|---:|---|

必须解释：

- allocator fragmentation；
- activation；
- temporary buffer；
- optimizer state lazy initialization；
- first step 峰值；
- eval/generation KV cache；
- checkpointing 的 compute-memory tradeoff。

验收：

- 找出至少一个近线性和一个非线性增长轴；
- 理论与实测偏差有归因；
- OOM envelope 可用于下一周 batch 选择；
- 没有通过反复 empty_cache 把不可行配置伪装成稳定。

### 周四：Nsight Systems/Compute 或等价 fallback

目标：从 op 表升级到时间线。

先检查：

    nsys --version
    ncu --version

若 nsys 可用：

    nsys profile \
      --trace=cuda,nvtx,osrt \
      --sample=none \
      --cpuctxsw=none \
      --output=runs/nsys_lm \
      python -m src.bench --config configs/lm_120m.yaml --warmup 20 --steps 10

具体 flags 以本机版本 help 为准。trace 留本机。

回答：

- CPU launch gap 是否明显；
- H2D 是否与计算重叠；
- kernel 是否大量碎片化；
- forward/backward 各占多少；
- optimizer 是否造成长 gap；
- DataLoader 是否让 GPU idle；
- 周期性 logging/eval/checkpoint 是否阻塞。

若 ncu 可用，只选 top 1 kernel 和 1–3 次 launch，不 profile 整个训练：

- achieved occupancy；
- memory throughput；
- compute throughput；
- Tensor Core/HMMA 迹象；
- launch dimensions。

如果工具不可用：

- 使用 torch profiler + CUDA events；
- 把 Nsight 标记为环境缺口；
- 不请求 root 安装、不自行下载未批准二进制。

验收：

- 有一张本机时间线或等价阶段证据；
- 能判断 compute-bound、memory-bound、launch-bound 或 data-bound；
- 不只引用 nvidia-smi utilization。

### 周五：只优化一个最大瓶颈并复测

目标：证明优化因果，而非堆开关。

按 profiling 证据只选一个：

- DataLoader worker/pin/prefetch；
- batch/gradient accumulation；
- activation checkpointing；
- 去除不必要同步；
- optimizer zero_grad set_to_none；
- 减少 logging/eval 频率；
- shape 对齐；
- 原生 attention 可用后端；
- 缓存固定 preprocessing。

预注册：

    假设
    目标阶段
    预计改善
    可能副作用
    不变量
    回滚条件

A/B：

- 相同模型/数据/seed/precision；
- 20 warmup + 100 timed；
- 各重复 3 次；
- 报 p50/p95、吞吐、显存、loss 差异；
- 优化后必须重新 correctness 20–100 step。

PASS 条件：

- 吞吐提升至少 10%，且数值/显存无不可接受回归；或
- 严格实验证明热门优化在该 workload 上无效，并能解释瓶颈没有被命中。

负结果是有效结果；没有对照的“感觉更快”不是。

## 6. 本周考核

| 项目 | PASS |
|---|---|
| ranges | data/H2D/FWD/BWD/optimizer/eval/checkpoint 可分 |
| timing | warmup、同步、p50/p95、重复规则正确 |
| profiler | top CUDA op 与 memory op 可映射回模块 |
| memory | batch/sequence/horizon/resolution 至少两轴 envelope |
| theory | 参数状态与 activation 粗算可解释实测 |
| timeline | Nsight 或等价证据能判断主要瓶颈 |
| optimization | 单变量 A/B，≥10% 改善或可信负结果 |
| correctness | 优化后 loss/grad/checkpoint smoke 仍通过 |

## 7. 故障判断

### GPU utilization 低

依次检查：

1. batch/sequence 是否过小；
2. data wait；
3. CPU launch gap；
4. 每 step synchronize/item/logging；
5. kernel 碎片；
6. H2D；
7. 共享机器干扰。

### reserved 远大于 allocated

- allocator cache 与 fragmentation；
- 不同 shape 导致 block 复用差；
- 是否持有旧 tensor reference；
- eval/generation 后 cache；
- 使用 memory_snapshot 需谨慎，结果留本机。

### profiler 后模型变慢

这是正常可能：record_shapes、stack、memory 都有开销。正式性能计时必须关闭 profiler。

## 8. 闭卷口试

1. self CUDA time 与 total CUDA time 有何区别？
2. 为什么 Python wall time 会低估异步 GPU 段？
3. warmup 要排除哪些一次性成本？
4. 参数、梯度、optimizer、activation 分别怎样增长？
5. sequence length 为什么可能出现二次显存增长？
6. allocated 与 reserved 为什么不同？
7. checkpointing 为什么省内存但可能降吞吐？
8. GPU utilization 100% 为什么仍可能优化？
9. profiler 本身如何改变 workload？
10. 何种 A/B 才能支持“这个优化有效”？
