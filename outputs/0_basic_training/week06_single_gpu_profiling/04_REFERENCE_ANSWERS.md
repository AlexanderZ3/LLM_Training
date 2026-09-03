# Week 06 回答篇：单卡 Profiling

> 必须先闭卷完成 03_ORAL_EXAM.md，再使用本篇校准。每题 3 分：答案核心 1 分，证据/公式 1 分，边界或止损 1 分。硬件与拓扑为用户自述待核验；不把计划命令写成实测结果。资料核验日期：2026-09-03。

## Recall

### W06-R01

参考答案：T_step 可拆为 data wait、H2D、forward、loss、backward、unscale/clip、optimizer、scaler update 与其他周期任务。H2D 与 compute、CPU launch 与 GPU execution 可能重叠，嵌套 profiler range 也会重复计时，所以阶段和不必严格等于 wall time。

评分点：写出主要阶段；指出异步/重叠；区分 steady-state 与 eval/checkpoint。  
常见误区：把所有 range 机械相加。  
追问：怎样用 CUDA event 与 perf_counter 分别测量？

### W06-R02

参考答案：allocated 是活跃 tensor 当前占用；reserved 是 caching allocator 向 CUDA 保留的块；peak allocated 是 reset 后 allocated 的最大值。reserved 可包含空闲缓存，不能单独证明泄漏。

评分点：三者定义；allocator 关系；说明需同时看峰值。  
常见误区：reserved 等于模型 tensor 总和。  
追问：empty_cache 能解决什么，不能解决什么？

### W06-R03

参考答案：粗略为 16P bytes：FP32 parameter、gradient、Adam first moment、second moment 各 4P。未含 activation、临时 workspace、CUDA context、allocator、通信 bucket、可能的 master copy 和框架对象；实际 dtype/optimizer 会变。

评分点：16P；四项拆分；列出至少三项遗漏。  
常见误区：认为 AMP 自动把 16P 减半。  
追问：为什么应在第一次 optimizer step 后测？

### W06-R04

参考答案：self CUDA 是算子自身 CUDA 活动，total 包含子调用；warmup 排除初始化、autotune、缓存与惰性状态成本；timed steps 是正式统计窗口。

评分点：定义准确；指出嵌套；说明 warmup 目的。  
常见误区：把第一个 step 纳入均值。  
追问：cold start 是否完全应丢弃？

## Explain

### W06-E01

参考答案：CUDA launch 通常异步，perf_counter 可能只覆盖 CPU 入队。应使用 CUDA events，或在测量边界 synchronize；不能每步同步，因为会破坏 overlap 和真实吞吐。

评分点：异步；正确测量；指出同步副作用。  
常见误区：在每个小算子后 synchronize。  
追问：端到端 wall time 为何仍有价值？

### W06-E02

参考答案：residual 与 QKV 对 T 近线性，但显式 attention logits/probability shape 为 [B,H,T,T]，对 T 二次；backward 还要保存中间量。具体后端若不物化完整矩阵，增长会不同，所以需 profile。

评分点：[B,H,T,T]；二次关系；后端边界。  
常见误区：所有 Transformer 显存都严格 O(T²)。  
追问：视觉分辨率如何通过 patch token 放大这个问题？

### W06-E03

参考答案：checkpointing 不保存部分 forward 激活，backward 时重算，因此降低 activation memory、增加 forward-like FLOPs 和 launch；若释放的显存未用于更大有效 batch，吞吐常下降。

评分点：保存/重算机制；显存收益；性能代价。  
常见误区：认为 checkpointing 同时必然更快。  
追问：dropout/RNG 对重算正确性有什么要求？

### W06-E04

参考答案：SDPA 是统一 API，PyTorch 会基于版本、设备、dtype、shape、mask 等选后端。V100 是 SM70，而官方 FlashAttention-2 CUDA 支持从 Ampere 等更新架构开始；2.1 可能回退 math/eager 或其他兼容实现，须看 profiler 与后端日志。

评分点：API 与 kernel 分离；SM70/FA2 边界；验证方法。  
常见误区：函数名含 attention 就等于 FlashAttention。  
追问：H100 上相同调用为什么也仍需验证后端？

## Apply

### W06-A01

参考答案：residual [B,T,D]；Q/K/V 各 [B,H,T,Dh] 且 D=H×Dh；score [B,H,T,T]；MLP 中间 [B,T,4D]（倍率依模型而定）；logits [B,T,V]。

评分点：shape 全部正确；指出 D=H×Dh；说明 4D 是常见而非绝对。  
常见误区：把 score 写成 [B,T,D]。  
追问：哪两个张量最容易因 T 或 V 变大成为峰值？

### W06-A02

参考答案：先固定 T=256 扫 B=1/2/4/8，再固定 B=1 扫 T=128/256/512；每点相同模型、seed、precision、warmup 5、measure 10，并在独立构建/释放模型后 reset peak。达到 reserved 28–30 GB 即止损，不做 B×T 全网格。

评分点：单变量；统一测量；安全阈值。  
常见误区：同时增加 batch 和 sequence。  
追问：如何计算 prediction error 并解释偏差？

### W06-A03

参考答案：A workers=0，B workers=4；固定样本顺序、batch、模型、seed、pin/prefetch 之外所有项、20 warmup+100 timed，各 3 次。比较 data ratio、step p50/p95、吞吐、CPU/内存，并做 loss/sample ID 回归。

评分点：唯一变量；重复和指标；correctness。  
常见误区：B 同时打开 pin_memory 与 persistent_workers。  
追问：若共享机器噪声大于提升怎么办？

### W06-A04

参考答案：step 日志包括 run_id、step、samples、loss、lr、grad norm 前后、loss scale、skip、各阶段 ms、allocated/reserved peak；summary 给 p50/p95、吞吐、阶段占比、峰值、skip、top ops 与状态标签。

评分点：数值、性能、显存三类；有效更新；状态标签。  
常见误区：只记录平均 step time。  
追问：哪些字段不能从公司机器导出？

## Debug

### W06-D01

参考答案：依次查：是否仍持有 tensor 引用；动态 shape 是否造成不同 block；eval/generation KV cache 是否未释放；optimizer/state 首步峰值；allocator cache/碎片。先在阶段边界记录 allocated/reserved，再用 memory_summary；只在公司内部且必要时用 snapshot。

评分点：至少四个假设；从低侵入证据开始；不滥用 empty_cache。  
常见误区：直接判定 CUDA 泄漏。  
追问：如何构造固定 shape 对照？

### W06-D02

参考答案：不一定。record_shapes、profile_memory、stack 和 trace 写盘都会改变 workload。用相同 config 做 profiler on/off，正式吞吐以 off 为准；同时确认 loss 与样本一致。

评分点：识别测量开销；on/off A/B；正式口径。  
常见误区：优化模型以“修复”profiler 开销。  
追问：怎样缩短 trace 又保留代表性？

### W06-D03

参考答案：检查 tokens/s 的单位和 global batch，kernel 时长/碎片、Tensor Core 迹象、memory throughput、CPU launch gap、同步、shape 对齐、forward/backward 比例。99% 只说明采样窗口内繁忙，不代表接近硬件有效上限。

评分点：不依赖 utilization；列三类证据；联系 workload。  
常见误区：认为 GPU 100% 就无法优化。  
追问：roofline 需要哪些数据？

### W06-D04

参考答案：记录 step 37 的 scale、unscale 后 grad norm、各模块非有限检查、输入范围与 loss 分项；与 FP32 小 batch reference 对齐；确认 clip 在 unscale 后且 skipped step 不推进 scheduler/计数错误。先定位算子，再决定降 scale、FP32 敏感运算或修数据。

评分点：GradScaler 顺序；模块定位；reference。  
常见误区：只降 batch 或吞掉 NaN。  
追问：连续 skip 与偶发 skip 的验收差异？

## Design

### W06-G01

参考答案：外层 data_wait/H2D/train_step，train_step 下分 zero_grad/FWD/loss/BWD/unscale_clip/optimizer；模型内分 embedding/attention/MLP/condition/action head。CPU 用 record_function，GPU 段用 event 或 profiler；只在整个 timed window 两端同步。

评分点：两级 range；阶段完整；避免逐步同步。  
常见误区：range 交叉或命名随 step 变化。  
追问：如何验证汇总器没有漏 step？

### W06-G02

参考答案：用 data-only benchmark 测供应能力；用预加载 GPU tensor 排除 DataLoader；用 record_function+CUDA event 看 CPU gap 与 GPU段；torch.profiler 短 trace 看调用次数和 kernel；增大 workload 的 A/B 判断 launch amortization。三组证据共同分类。

评分点：至少三个正交对照；可区分三类 bound；有局限声明。  
常见误区：无 Nsight 就直接猜。  
追问：怎样排除共享负载？

### W06-G03

参考答案：visual tokens=C(Himg/p)(Wimg/p)，总 S 加语言/状态/动作 token；显式 score 约 [B,heads,S,S]。先 batch=1 固定 C/horizon 扫低分辨率，再固定分辨率扫 C，最后扫 horizon；每点先预测，reserved 到 28–30 GB 止损。

评分点：公式；非线性；单变量安全扫描。  
常见误区：同时增加摄像头与分辨率。  
追问：冻结 vision encoder 改变哪些显存项？

### W06-G04

参考答案：公司 V100 内部完成真实 baseline，但不导出任何代码、数据、日志、trace、图片、checkpoint 或数值；个人 5070 Ti 用公开代码/合成数据从零重跑缩小版，独立 manifest；分别报告 GPU/torch/后端，不比较成硬件排行榜。

评分点：边界完整；重新生成；硬件结论分离。  
常见误区：手抄公司汇总数字到个人报告。  
追问：哪些抽象经验可以带走？

## Trade-off

### W06-T01

参考答案：增 micro-batch 往往提高利用率但增加 activation 且改变 global batch；accumulation 在相同 global batch 下省峰值但增加串行 micro-step，DDP 时影响同步；checkpointing 省 activation、增加重算。三者都需固定或显式重定义训练数学。

评分点：三者的显存/吞吐；global batch；correctness。  
常见误区：把 accumulation 当免费增大 batch。  
追问：如何选优化顺序？

### W06-T02

参考答案：torch.profiler 看框架 op、shape、memory，易集成但 tracing 有开销；Nsight Systems 看 CPU/CUDA/NCCL 时间线与 overlap；Nsight Compute 深挖少数 kernel 的 occupancy、带宽和指令，开销最高。先粗后细。

评分点：粒度区分；开销；选择顺序。  
常见误区：用 ncu profile 整个训练。  
追问：哪个工具最适合发现 DataLoader gap？

### W06-T03

参考答案：10% 是计划门槛而非自然定律。8% 若重复稳定、成本低、p95/显存/数值不回归且测量噪声小，可作为有条件 PASS；15% 若改变样本、有效 batch、skip、loss 或挤到显存红线，应回滚。

评分点：统计稳定；correctness；资源余量。  
常见误区：只按单一百分比裁决。  
追问：如何预注册最小可检测效应？

### W06-T04

参考答案：记录假设、目标阶段、A/B 不变量、三次结果、置信范围和 profiler 证据；说明为什么瓶颈未被命中及何种 workload 下可能有效。负结果是对适用边界的证据，不是“开关没用”的普遍断言。

评分点：完整实验链；限定结论；下一步。  
常见误区：从一个 workload 否定整个技术。  
追问：什么情况下应写 INCONCLUSIVE 而非负结果？

## 复训门槛

低于 44/72：重做 W06-A02、W06-D04、W06-G02 三题并完成一次 A0/A1 限时复测。出现数据外带、BF16/FA2 硬件边界错误时，无论总分多少都先做安全与兼容性复训。
