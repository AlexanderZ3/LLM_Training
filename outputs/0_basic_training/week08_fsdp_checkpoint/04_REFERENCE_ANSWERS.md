# Week 08 回答篇：FSDP 与 Checkpoint

> 先完成闭卷。每题 3 分：核心、证据/计算、边界各 1 分。公司硬件和拓扑均为用户自述待核验。资料核验日期：2026-09-03。

## Recall

### W08-R01

参考答案：普通 FP32 AdamW 粗估 DDP 每卡 16P bytes；N-way Full Shard 理想为 16P/N；Hybrid 的 shard degree 为 S 时为 16P/S，不除 replica degree。

评分点：三个公式；说明理想下界；列出 activation/buffer 不含。  
常见误区：2×4 Hybrid 除以 8。  
追问：mixed precision 为何改变常数？

### W08-R02

参考答案：每个 wrapped unit 计算前 all-gather 参数；backward 后 reduce-scatter 梯度，随后每 rank 更新本地 state shard。具体 reshard/prefetch 时机由配置与版本决定。

评分点：两 collective；顺序；版本边界。  
常见误区：forward 后只需 all-reduce。  
追问：参数何时可能继续驻留完整形式？

### W08-R03

参考答案：model、optimizer、scheduler、GradScaler、Python/NumPy/torch CPU/CUDA RNG、sampler 状态、global step/updates/samples、config/data/code revision、world/strategy/wrap/precision、metadata/完成标记。

评分点：训练状态；随机与数据位置；分布式元数据。  
常见误区：只保存 model weights。  
追问：EMA 若存在放在哪里？

### W08-R04

参考答案：shard degree 是一个逻辑状态被切成几份；replica degree 是完整 shard 集复制几组。2×4 的本设计为 shard=4、replica=2。

评分点：定义；数值；每卡状态=/4。  
常见误区：把“2×4”固定解释为 API mesh 维度顺序。  
追问：如何在代码中断言进程组？

## Explain

### W08-E01

参考答案：FSDP 切参数、梯度、optimizer state；residual [B,T,D]、score [B,H,T,T] 等 activation 仍由每 rank 的本地 batch/sequence 产生。需 checkpointing、缩 batch/sequence 或更合适 attention。

评分点：对象区分；shape；替代措施。  
常见误区：所有 tensor 都被 shard。  
追问：activation 何时反而因 all-gather buffer 更难判断？

### W08-E02

参考答案：unit 太大导致 full-param all-gather buffer 大、overlap 粗；太小导致 collective/launch/metadata 过多。block-level 常是平衡起点，最终由 per-unit memory 与 trace 验证。

评分点：两端后果；平衡；证据。  
常见误区：unit 越小越省显存且总是更快。  
追问：limit_all_gathers 影响什么？

### W08-E03

参考答案：目录和部分 shard 可能在作业中断后存在；若 marker 提前写，loader 会把不完整集合当成功。所有 rank 写完并经 coordinator 校验后最后提交 marker，loader 先检查它。

评分点：部分写；最后提交；loader 拒绝。  
常见误区：目录存在即完成。  
追问：如何避免覆盖上一个完整 checkpoint？

### W08-E04

参考答案：8→8 可按相同分片布局加载；8→4 需把 model 和 optimizer shard 重新映射，涉及 metadata、flattening、parameter identity 与 API 支持。2.1 state dict 跨版本也不保证兼容。

评分点：reshard；optimizer 难点；版本。  
常见误区：拼接文件再均分即可。  
追问：为何 full state 可作小模型验证但有峰值风险？

## Apply

### W08-A01

参考答案：0.6B：DDP 9.6GB、/4 2.4GB、/8 1.2GB；1.5B：24GB、6GB、3GB；4B：64GB、16GB、8GB。均不含 activation/buffer 等。

评分点：九个数；单位；边界。  
常见误区：把 GiB 与 GB 混而不说明。  
追问：32GB 卡上 4B /8 为什么仍未必安全？

### W08-A02

参考答案：M=16,777,216 elements；理想每 rank 2,097,152 elements 的平坦 shard（实现可能 padding）。forward all-gather 重建逻辑 [4096,4096]；backward 各 rank 产生完整逻辑贡献，reduce-scatter 后留下对应 flat grad shard。

评分点：numel；本地长度；两 collective shape。  
常见误区：每 rank 只计算输出的八分之一。  
追问：这与 tensor parallel 有何不同？

### W08-A03

参考答案：相同初始化、global batch与样本顺序、loss reduction、optimizer/LR；tiny FP32 先跑一步，比较 global loss、grad norm与更新后的 full关键参数；再 FP16 比 scale/skip，容差预注册。full materialization 只在小模型/受控区间。

评分点：固定数学；FP32→FP16；峰值安全。  
常见误区：只比较 100 step 最终 loss。  
追问：tied weight 如何检查？

### W08-A04

参考答案：连续 reference 跑100；实验跑50写临时目录、全 shard与 marker后退出，再8卡恢复到100。比较 step51 fixed batch loss相对差<1%，并核对 optimizer/scheduler/scaler/RNG/sampler/step和最终参数。

评分点：连续 reference；完整状态；next-batch。  
常见误区：load成功就PASS。  
追问：如何验证没有重复消费样本？

## Debug

### W08-D01

参考答案：先确认 wrap coverage/strategy；在 optimizer首步后测；分解 activation与model-state；检查 full state dict/reference 是否驻留；看 all-gather buffer/unit过大；检查 use_orig_params/shared/frozen；再看 allocator。小模型也可能被固定开销主导。

评分点：顺序；至少五项；不先改参数。  
常见误区：直接断言FSDP失效。  
追问：怎样测 optimizer lazy init？

### W08-D02

参考答案：先检查 completion marker、metadata与预期shard集合/大小；看每rank最后range和路径是否一致；确认所有参与rank共同调用DCP及process group正确。缺marker/任何shard直接拒绝，不进入collective load。

评分点：marker；rank路径；调用对称。  
常见误区：在hang中反复重试同一坏目录。  
追问：Hybrid checkpoint为何需正确shard group？

### W08-D03

参考答案：核对初始化seed、global batch/sample IDs、loss reduction、mixed precision/reduce dtype、tied weights、wrap coverage/use_orig_params、global grad clip、optimizer参数身份、scaler skip。用tiny FP32一步隔离。

评分点：六项；reference；先correctness。  
常见误区：放宽容差到通过。  
追问：哪一项会从step0就产生差异？

### W08-D04

参考答案：不能。Hybrid少一半sharding、显存更高；其收益依赖高频collective是否留在局部域及replica同步成本。核对mesh实际成员、all-gather/reduce与replica通信、batch/sequence、峰值和共享负载；可能是workload或映射不合适。

评分点：非普遍结论；mesh验证；综合显存/通信。  
常见误区：只看tok/s。  
追问：何时即使更慢也选Hybrid？

## Design

### W08-G01

参考答案：列每unit路径、unique parameter ID/name、numel、dtype、trainable、shared关系；聚合unique numel等于全模型。tied embedding/head记录同一对象，只由兼容边界管理，并做forward/update后仍共享的断言。

评分点：unique覆盖；共享处理；粒度统计。  
常见误区：按名字相加导致重复计数。  
追问：frozen/trainable混合如何审计？

### W08-G02

参考答案：新step写独立.tmp目录；各rank写local shard并fsync/返回；coordinator验证manifest、shard数/hash；barrier后原子重命名或最后写COMPLETED；更新latest指针最后进行。loader只读完整marker并校验revision。

评分点：临时目录；全局验证；原子提交。  
常见误区：先更新latest。  
追问：磁盘满时如何保留上个checkpoint？

### W08-G03

参考答案：相同0.6B、global batch、precision、wrap、warmup/timed、seed、checkpoint关闭；Full shard=8，Hybrid shard=4 replica=2。打印mesh/assert group，记录peak、吞吐、all-gather/reduce/replica share、loss，三次重复。

评分点：公平；group证据；多指标。  
常见误区：Hybrid使用更大batch却直接比吞吐。  
追问：怎样把“容纳更大batch”作为另一问题报告？

### W08-G04

参考答案：停在FSDP1；导入/签名探测2.1 API；完成8-way Full、同规模DCP或该版本state-dict路径；Hybrid若缺API标ENV-BLOCKED，可做两个独立4-way概念测量但不冒充Hybrid。新框架只读，不升级生产环境。

评分点：尊重版本；可交付回退；诚实标签。  
常见误区：pip升级整个torch。  
追问：怎样记录未来迁移需求？

## Trade-off

### W08-T01

参考答案：Full /8显存最低但高频shard collective覆盖8卡；2×4 Hybrid /4显存较高，力图让频繁通信留在四卡域并跨域做replica同步。模型容纳优先选Full；拓扑跨域昂贵且/4已够时Hybrid可能更快。

评分点：显存；通信；选择条件。  
常见误区：Hybrid总是Full的性能升级。  
追问：实际拓扑与mesh错配会怎样？

### W08-T02

参考答案：FSDP切model state并引入collective；activation checkpointing少存激活并以重算换显存。model-state主导先FSDP，activation主导先checkpointing/shape；可组合但需要独立A/B。

评分点：对象；成本；诊断选择。  
常见误区：二者等价。  
追问：组合后如何定位新峰值？

### W08-T03

参考答案：full state易于普通模型消费但materialization峰值高、常rank0 CPU offload；sharded state并行I/O和内存友好、依赖分布式metadata/API做reshard；local state最贴近当前布局、可移植性最低。2.1具体支持必须实测。

评分点：三者；峰值；移植性。  
常见误区：local shard可以单文件独立加载。  
追问：发布推理权重为何常另做受控full export？

### W08-T04

参考答案：0.6B已通过预算、一步等价、200步稳定、resume和profile后，1.5B短测可验证规模转折；若任一正确性/恢复失败、接近显存红线或时间预算不足，应停在0.6B。规模不是本周北极星。

评分点：前置门槛；止损；学习目标。  
常见误区：0.6B没通就用1.5B“证明FSDP”。  
追问：4B为何只做paper budget？

## 复训

低于44/72：重做W08-A03、W08-D02、W08-G02并完成一次tiny checkpoint roundtrip。任何版本强升、手工拼shard或数据外带倾向，先做安全复训。
