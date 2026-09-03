# Week 03 回答篇：V100 / Topology / NCCL / FP16

> 先闭卷。每题 4 分：机制、证据、边界、处置各 1 分。

## Recall

### W03-R01
答案：driver 是内核/设备驱动；`torch.version.cuda` 是 wheel 构建时 toolkit；`nvcc` 是本机扩展编译器。三者可不同。评分点：三定义 3；关系 1。误区：nvidia-smi 的 CUDA 就是 torch runtime。追问：何时需要 nvcc？

### W03-R02
答案：V100=Volta、SM70；支持 FP16/Tensor Core，通常无原生 BF16 Tensor Core。必须由现场探针确认设备与 build 支持。评分点：四点。误区：V100 不支持 FP16。追问：BF16 动态范围为何更大？

### W03-R03
答案：NV1/NV2 表示 NVLink 数/等级；PIX 是近端 PCIe bridge 路径；SYS 跨 host bridge/CPU interconnect。具体字符串以工具帮助为准。评分点：四项。误区：SYS=网卡。追问：PHB 与 NODE 呢？

### W03-R04
答案：allocated 是活跃 tensors；reserved 是 caching allocator 从 CUDA 保留池，包含未当前使用块。评分点：两定义 2；OOM/碎片意义 2。误区：reserved 都是泄漏。追问：怎样 reset peak stats？

## Explain

### W03-E01
答案：DDP 每 rank 复制完整模型、梯度和通常 optimizer state，每卡仍受 32GB 限制；只有 sharding/parallelism 分摊特定状态。评分点：复制 2；限制 1；对照 1。误区：显存自动池化。追问：FSDP 不分摊什么？

### W03-E02
答案：collective 可经多跳、多 channel、ring/tree 沿其他 NVLink 边传输；点对边只描述直接路径。需 NCCL graph/整体 benchmark。评分点：图 2；NCCL 1；证据 1。误区：最差 pair 决定全部。追问：何时 SYS 仍成瓶颈？

### W03-E03
答案：algbw 是 payload/time；busbw按 collective 通信量归一，便于接近链路利用口径；换算随 collective/N变化，correctness另看 error。评分点：两定义 2；换算边界 1；correctness 1。误区：busbw 就物理峰值。追问：all-gather 因子？

### W03-E04
答案：小/不对齐 GEMM、CPU/data/launch 主导、autocast未覆盖、转换开销、低 occupancy 都会遮蔽 Tensor Core。评分点：四因。误区：不快即无 FP16能力。追问：用什么 profiler 证据？

## Apply

### W03-A01
答案：进程内逻辑 0→物理4，逻辑1→物理6；打印 env visible order、local/global rank、current_device、PCI bus ID/UUID（本机保留）。评分点：映射 2；证据 2。误区：rank0=物理0。追问：torchrun 如何设设备？

### W03-A02
答案：固定 binary/collective/message sizes/warmup/repeats/GPU空闲条件；只换 visible pair；记录 correctness、time、algbw/busbw、p50/波动与 mapping。评分点：控制 2；观测 2。误区：不同 pair 用不同消息大小。追问：为什么重复3次？

### W03-A03
答案：`busbw=X×2×7/8=1.75X`，适用于 nccl-tests 的 ring all-reduce口径；其他 collective/实现需查对应定义。评分点：式 2；数值 1；边界 1。误区：任何 collective乘2。追问：N=2 呢？

### W03-A04
答案：固定参数量/近似FLOPs、batch/T、数据、precision与计时；只改 D/head 对齐；warmup后测 p50/p95/tokens/s，并用 profiler看 GEMM dtype/shape。评分点：控制 2；指标 1；内核证据 1。误区：同时改模型深度。追问：参数量不完全相等怎么办？

## Debug

### W03-D01
答案：停止多卡；重复该卡 FP32/FP16、换 seed/shape；与同配置其他卡比；记录错误/温度/ECC等批准可见信息，提交平台最小复现。评分点：止损 1；隔离 2；升级 1。误区：跳过坏卡继续声称全机通过。追问：如何排实现共因？

### W03-D02
答案：禁 P2P只是绕开症状且降低/改变路径；先收集 timeout/rank/mapping/NCCL info，最小化 pair/size，验证驱动/链路并报平台。评分点：不掩盖 1；最小化 2；处置 1。误区：把 workaround 写默认。追问：何时可临时禁用？

### W03-D03
答案：用 synthetic nccl-tests排除数据；交换时序、重复3次、确认 mapping/消息；检查 CPU affinity/共享进程；若稳定差再看 graph/topology。评分点：四层。误区：一次结果归因 NVLink。追问：四卡 A/B 哪些理论上相同？

### W03-D04
答案：记录 `FAIL-VISIBILITY` 与命令退出码；不把 NUMA输出当事实、不越权 mount；NVLink correctness 独立记录为通过；请求管理员提供批准的只读信息。评分点：分离结论 2；权限 1；后续影响 1。误区：警告=NVLink坏。追问：会影响哪个性能实验？

## Design

### W03-G01
答案：版本、设备逐卡、拓扑证据、rank mapping、CPU可见性、NCCL矩阵、FP32/FP16、内存 envelope、禁用项、共享负载、未知项、状态标签。评分点：四类以上覆盖。误区：只放截图/峰值。追问：如何版本化？

### W03-G02
答案：逐卡 reference→pair correctness→4A/4B→8卡→固定 workload重复性能；任一 correctness失败停止，性能污染可 INCONCLUSIVE。评分点：阶梯 2；门槛 1；标签 1。误区：直接8卡长跑。追问：何时加入训练载荷？

### W03-G03
答案：原始日志/trace/topology本机；本机生成带 config/hash/命令/时间/退出码的 summary 与 decision；外部只谈通用方法，不传具体数字/路径/图。评分点：内部复核 2；边界 2。误区：脱敏后随意传性能。追问：公开作品如何获得？

### W03-G04
答案：独立 debug config、很短步数、先保存reference；人为放大 finite loss；观测scale/skip/参数finite；关闭注入后恢复；设置max_steps和nonfinite硬停。评分点：隔离 1；观测 2；恢复 1。误区：直接制造NaN写入权重。追问：scheduler是否前进？

## Trade-off

### W03-T01
答案：自动选择能适配拓扑/size且是可靠默认；手工固定适合诊断单变量，但可能压错算法、损失泛化。先观察再受控强制。评分点：双方 2；顺序 2。误区：Ring永远最好。追问：Tree适合什么情形？

### W03-T02
答案：nccl-tests给collective correctness/上限，不能预测模型计算重叠；Transformer反映真实compute/data/bucket，却难隔离链路。二者共同归因。评分点：各能/不能共4点。误区：任一替代另一。追问：训练达不到 busbw为何正常？

### W03-T03
答案：严格模式适合单测/resume归因，可能降低性能；性能模式代表生产吞吐但需容差/多次运行。先严格建立reference，再冻结高性能配置。评分点：双方2；阶段策略2。误区：同seed必bitwise。追问：哪些算子可能非确定？

### W03-T04
答案：升级会同时改变ABI、kernel、NCCL/依赖与复现基线，风险超出本周；2.1足以完成核心实验。缺功能记录边界，在批准隔离环境另测。评分点：风险2；目标1；回退1。误区：新版本一定更快更正确。追问：何时值得升级？

## 自评

记录总分、六级分项、AI辅助等级、是否出现安全/事实错误、最小复现实验与回归日期。Hardware Card 只有现场证据才能从“待核验”改成“已确认”。
