# Week 07 回答篇：DDP 正确性与 Scaling

> 先闭卷，再校准。每题 3 分：核心 1、证据/公式 1、边界/止损 1。公司硬件与拓扑为用户自述待核验。资料核验日期：2026-09-03。

## Recall

### W07-R01

参考答案：global_batch=micro_batch_per_gpu×gradient_accumulation_steps×world_size。micro 是每 rank 每次 forward 的样本数，accum 是一次 optimizer update 累积次数，world_size 是并行进程数。

评分点：公式；三个定义；指出有效 mask/uneven batch 例外。  
常见误区：漏乘 world_size。  
追问：skipped AMP step 是否计入有效 batch？

### W07-R02

参考答案：world_size 是 process group 总进程数；rank 是全局编号；local_rank 是节点内编号并用于选择可见 device。CUDA_VISIBLE_DEVICES=0,5 时 local_rank 0/1 映射物理 0/5。

评分点：四者；给映射例；不二次硬编码。  
常见误区：local_rank 永远等于物理 GPU ID。  
追问：多节点时 rank 与 local_rank 如何不同？

### W07-R03

参考答案：弱扩展固定每卡工作、global batch 随 N 增；强扩展固定总工作、每卡工作下降。speedup_N=throughput_N/throughput_1，E_N=speedup_N/N。

评分点：两个定义；两公式；口径限制。  
常见误区：不同 global batch 曲线直接比较 loss。  
追问：强扩展为何更容易通信主导？

### W07-R04

参考答案：读 local_rank→set CUDA device→init_process_group(NCCL)→设 model/data seed→模型移本卡→DDP wrap→DistributedSampler/DataLoader→set_epoch→train→rank0 汇总/checkpoint 与对称 barrier→finally destroy process group。

评分点：顺序；sampler；cleanup。  
常见误区：模型先落到默认 GPU0。  
追问：optimizer 应围绕哪些参数创建？

## Explain

### W07-E01

参考答案：每 rank 拥有完整参数、梯度与 optimizer state；DDP 只同步梯度，没有对这些状态做 sharding。因此单卡仍受 32GB 限制，8 卡不是透明 256GB。

评分点：三类状态；通信角色；显存结论。  
常见误区：数据被切分所以模型也被切分。  
追问：FSDP 与 DDP 的核心差异？

### W07-E02

参考答案：sampler 以 epoch 参与确定性 shuffle；不 set_epoch 会每个 epoch 使用同一排列。所有 rank 必须用一致 epoch 值，才能既共同洗牌又保持分区协调。

评分点：epoch seed；跨 rank 协调；检测 IDs。  
常见误区：每 rank 随机 seed 即可。  
追问：validation 应否 shuffle？

### W07-E03

参考答案：每 rank 相同样本数、local loss 是 mean 时，DDP 平均局部梯度等价 global mean。若有效 token/mask denominator 或 batch 大小不同，rank mean 等权会误权重；应全局汇总 numerator/denominator 或保证计数相同。

评分点：成立条件；反例；修复。  
常见误区：任何 mean 都自动全局正确。  
追问：sum reduction 时 LR/除数如何处理？

### W07-E04

参考答案：pair 只测一条局部链路；8 卡 collective 跨多个路径，且计算/通信比、bucket overlap、CPU 与数据都会变。更快 pair 不代表全图 channel 或每卡 workload 有利。

评分点：局部与全局；overlap；workload。  
常见误区：从单 pair 带宽预测线性 E8。  
追问：NCCL tests 与训练吞吐各回答什么？

## Apply

### W07-A01

参考答案：例如 micro=8：N=1 accum=8；N=2 accum=4；N=4 accum=2；N=8 accum=1，均 global=64。8 卡每 rank 只算 8 个样本，可能 workload 太小、collective/launch 主导。

评分点：四点正确；global 固定；风险。  
常见误区：accum 使用非整数仍声称等价。  
追问：不能整除时怎样重新选 micro/global？

### W07-A02

参考答案：前 3 个 micro-step 在 model.no_sync 上下文中 backward，每次 loss/4；第 4 个不用 no_sync，触发梯度同步；然后 unscale、clip、step、update、zero_grad。

评分点：K-1；loss 除 K；step 顺序。  
常见误区：最后一步仍 no_sync。  
追问：为什么 zero_grad 不应在 micro-step 中间调用？

### W07-A03

参考答案：固定初始化、global batch、20 step 的 global sample IDs 与顺序、dropout/RNG、optimizer/LR、loss reduction；先 tiny FP32，比较每步 loss、grad、update 后完整参数 max diff，再 FP16 看 scaler/skip。

评分点：数据；参数/数值；FP32→FP16 层次。  
常见误区：只比最终 loss。  
追问：浮点 reduction order 造成的容差怎样预注册？

### W07-A04

参考答案：E4=3100/(4×1000)=77.5%；E8=4700/(8×1000)=58.75%。对合格的 300M+ 计算密集弱扩展，E4 达到 75% 合理门槛；E8 低于 60% 通过线但高于 50% 止损线，应先诊断再决定长跑。

评分点：两个计算；门槛准确；限定 workload。  
常见误区：称 E8 58.75% 为硬件不合格。  
追问：若是强扩展该如何表述？

## Debug

### W07-D01

参考答案：最可能忘记 sampler.set_epoch(epoch)。记录每 rank 每 epoch 前 20 IDs 与全 epoch hash；在 DataLoader 消费前设置相同 epoch。程序不会报错，因为数据仍合法，只是随机顺序未变化。

评分点：根因；检测；修复时机。  
常见误区：换随机 seed 掩盖。  
追问：重复顺序一定导致模型错误吗？

### W07-D02

参考答案：先核对 global batch/accum与 LR；再核对样本集合/顺序、loss mean/sum 与 mask denominator；随后 dropout/RNG、sampler padding、BatchNorm；最后看 AMP skips 与参数差。用 tiny FP32 一步缩小范围。

评分点：分层顺序；数学优先；reference。  
常见误区：先调学习率。  
追问：loss 低为何不一定是好事？

### W07-D03

参考答案：控制流不对称：rank 6 尚在 backward，其他 rank 进入 barrier。检查 rank 6 是否 OOM/异常/等待 collective及最后 range；设短 timeout、捕获 rank-local 错误并终止整个作业。修复所有 rank 对称完成 train/eval 控制流，不能只让 rank0 任意 barrier。

评分点：识别不对称；rank-local 证据；安全整体终止。  
常见误区：继续等待或随机禁用 P2P。  
追问：rank0-only eval 怎样设计同步？

### W07-D04

参考答案：rank 3 不更新而其他 rank 更新，参数/optimizer state 立即分叉，下一次 collective 的梯度来自不同模型。记录 per-rank scale、skip 与 checksum；使 overflow 判定/step 决策跨 rank 一致，并排查触发 overflow 的输入/算子。

评分点：后果；检测；同步决策。  
常见误区：认为下轮 all-reduce 会自动修复参数。  
追问：scheduler 是否也必须只在有效 step 推进？

## Design

### W07-G01

参考答案：每 rank 输出 epoch、position、sample_id、is_padding；rank0 聚合 union、intersection、计数与 hash。train 明确 drop_last/padding 策略；validation 以原始 sample ID 去重或用无 padding sampler，并按 numerator/denominator 聚合。

评分点：rank-local IDs；集合审计；validation 去重。  
常见误区：只比较每 rank 数量。  
追问：大数据集如何不用保存全部 ID？

### W07-G02

参考答案：当日 topo 确认后选 NV2/NV1/SYS pair及两四卡域；固定模型、每卡 micro、precision、数据、warmup/timed、seed、logging、共享负载，3 repeats。报告 compute、exposed comm、throughput、correctness，不先强制 NCCL 算法。

评分点：实际核验；固定变量；通信证据。  
常见误区：只跑每种 pair 一次。  
追问：如何处理两四卡域不对称？

### W07-G03

参考答案：只有 trace 显示 all-reduce 暴露或 bucket ready/launch 不合理时，测试默认 vs 一个依据充分的 bucket size；固定训练所有项，短 profiler 加正式 off-profiler 计时，并回归参数/loss。太小会 collective 碎，太大会延迟首个同步和增加缓冲。

评分点：触发证据；单变量；两端代价。  
常见误区：网格扫很多 bucket。  
追问：bucket 与 gradient accumulation 如何交互？

### W07-G04

参考答案：每 rank 独立 JSONL 路径，含 run_id/rank/world/range/异常；rank0 仅在全 rank 完成标志/collective 确认后写 summary 和完成 marker。checkpoint 采用临时目录、所有必需 shard/状态就绪后最后写 marker。

评分点：防覆盖；全局完成；异常可定位。  
常见误区：rank0 正常退出就标 PASS。  
追问：怎样避免 barrier 本身掩盖异常？

## Trade-off

### W07-T01

参考答案：每 micro-step 同步最易验证但通信 K 次；no_sync 只在最后同步，通信少但 loss/K、最后同步和异常控制更易错。先用同步版本作 reference，再启用 no_sync 并比较梯度/参数。

评分点：通信；正确性；验证顺序。  
常见误区：no_sync 自动完成除 K。  
追问：什么时候每步同步成本可忽略？

### W07-T02

参考答案：固定 per-GPU batch 是弱扩展，研究资源增加后的总吞吐，但 global batch 变化会改变优化；固定 global batch 是强扩展，研究同一工作加速，但每卡计算变小。两者因问题和 loss 语义不同必须分表。

评分点：研究问题；数学变化；分表。  
常见误区：把两个曲线拼一个 E_N。  
追问：LR scaling 属于哪个实验的变量？

### W07-T03

参考答案：若四卡 correctness、吞吐良好，而八卡因 450M 以下小负载进入通信主导或 E8<50%，四卡更节省 GPU-hours；八卡只做 20–100 step topology/profile 样本。若需要更大 global throughput且 E8达门槛，才长训。

评分点：GPU-hours；效率止损；8卡样本价值。  
常见误区：为了使用所有卡而改变 global batch。  
追问：如何量化 time-to-target 而非瞬时吞吐？

### W07-T04

参考答案：模型确有合法的条件分支导致部分参数在某些 batch 未参与时可启用并承担额外图遍历成本；若参数本应始终使用，开启它会掩盖 bug。先审计分支与 unused 参数集合，再决定。

评分点：合法场景；成本；不掩盖 bug。  
常见误区：遇到 hang 就永远开启。  
追问：static_graph 在 2.1 下何时可考虑？

## 复训

低于 44/72：闭卷重写最小 DDP trainer，完成 W07-A03、W07-D03、W07-G01 的 A0/A1 复测。安全边界或“8 卡=256GB”错误必须先复训再进入 FSDP。
