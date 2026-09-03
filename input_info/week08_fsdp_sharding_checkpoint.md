# 第 08 周执行手册：FSDP、状态切分、Hybrid Shard 与 checkpoint

> 对应原 12 周主线第 6 周。标准投入 7.5 小时。  
> 主载荷：0.6B decoder-only Transformer；1.5B 只在 0.6B 全部通过后短测。  
> 本周目标：亲自验证 DDP、Full Shard、2×4 Hybrid Shard 的显存、通信和恢复差异。

每日时间盒：10 分钟做纸面预算，50–60 分钟完成一个 sharding/recovery 实验，15–20 分钟核对显存与状态；第 2 小时只做一个可选策略。

## 1. 官方资源与版本纪律

| 资源 | 用途 | 地址 |
|---|---|---|
| PyTorch FSDP | FSDP1 API；以本机版本对应文档为准 | https://docs.pytorch.org/docs/stable/fsdp.html |
| Distributed Checkpoint | sharded state 保存/加载 | https://docs.pytorch.org/docs/stable/distributed.checkpoint.html |
| CS336 A2 | memory-efficient/distributed 练习 | https://github.com/stanford-cs336/assignment2-systems |
| Picotron | 教学级并行模块；完成自己的 FSDP 后阅读 | https://github.com/huggingface/picotron |
| Nanotron | 更完整预训练框架，作为后续阅读 | https://github.com/huggingface/nanotron |
| TorchTitan | PyTorch-native 大规模系统阅读 | https://github.com/pytorch/torchtitan |

重要边界：

- 本机若是经典 torch.distributed.fsdp.FullyShardedDataParallel，就使用 FSDP1；
- 不为追最新 FSDP2/TorchTitan 升级整套 torch/CUDA；
- 当前 TorchTitan main 建议 PyTorch nightly，且现代 CUDA/内核路径未必支持 V100；本周只做代码架构阅读；
- Nanotron 默认安装示例含 FlashAttention 等新 kernel，V100 不照抄 fused-kernel 安装；
- Picotron 可读，但其 H100 benchmark 不能当你的 V100 目标。

## 2. 先做内存预算

对 P 个参数，普通 FP32 AdamW 粗估：

| 状态 | bytes/param |
|---|---:|
| parameter | 4 |
| gradient | 4 |
| Adam first moment | 4 |
| Adam second moment | 4 |
| 小计 | 16 |

AMP、框架和 optimizer 实现可能增加/改变副本，最终以实测为准。

理论 model-state：

- DDP：每卡约 16P bytes；
- 8-way Full Shard：理想下每卡约 16P/8，加 all-gather 临时 buffer；
- 2×4 Hybrid Shard：每卡约 16P/4，四卡内 shard、跨两组 replica；
- activation 不因 model-state sharding 自动消失。

必须为 0.6B、1.5B、4B 写纸面表，但本周只承诺 0.6B：

| params | DDP state lower bound | /4 shard | /8 shard | activation estimate | safe? |
|---:|---:|---:|---:|---:|---|
| 0.6B | | | | | |
| 1.5B | | | | | |
| 4B | | | | | paper only |

## 3. 目录

    week08-fsdp/
      src/
        model_600m.py
        wrap_policy.py
        train_ddp.py
        train_fsdp.py
        memory_report.py
        checkpoint_io.py
        compare_states.py
      configs/
        ddp_600m.yaml
        full_shard_600m.yaml
        hybrid_2x4_600m.yaml
      tests/
        test_wrap_coverage.py
        test_one_step_equivalence.py
        test_checkpoint_roundtrip.py
      reports/
        memory_budget.md
        fsdp_comparison.csv
        decision.md

## 4. wrap policy 原则

目标是以 Transformer block 为 FSDP unit。错误极端：

- 整个模型一个 unit：all-gather 粗、峰值可能大、overlap 差；
- 每个 tiny module 一个 unit：collective 太碎；
- embedding/LM head weight tying 被错误拆分；
- 某些参数没被任何 unit 正确管理；
- frozen/trainable 参数混合造成版本限制或非预期行为。

本周先打印：

- 每个 FSDP unit 名称；
- 参数量；
- 是否 trainable；
- shared parameter；
- 总和是否等于模型参数总量；
- 最大/最小 unit。

## 5. 每日安排

### 周一：纸面预算、0.6B 模型与 DDP reference

目标：先建立不切分的数学 reference。

建议 0.6B surrogate：

- decoder-only Transformer；
- vocab 可缩小到 16k–32k；
- sequence 256–512；
- hidden/layers/heads 调到实算约 0.6B；
- synthetic tokens 或第 01 周 tokenized data；
- tied embeddings；
- 不依赖 FlashAttention。

任务：

1. 参数量按模块打印；
2. 计算 DDP model-state 下界；
3. 估 activation；
4. 单卡 3-step forward/backward；
5. 如单卡 optimizer OOM，这是预期信息，先用较小模型建立 DDP reference；
6. 2 卡 DDP 20-step reference，保存每步 loss/grad norm；
7. 固定 batch 保存一份 reference outputs。

验收：

- 纸面预算能提前预测是否 OOM；
- 模型参数量误差 <1% 相对目标；
- DDP reference 数值稳定；
- 记录 optimizer first step 后的真实峰值，因为 state 常 lazy init；
- 说明哪些显存不在 16P 粗估中。

可选第二小时：用 meta device 或空权重初始化研究 rank0 内存峰值，但不得破坏主路径。

### 周二：2 卡 Full Shard 与一步等价性

目标：先在最小 world size 上把 FSDP 配对。

配置：

- 2 卡 NV2 0,1；
- transformer-block auto wrap；
- mixed precision param/reduce/buffer 以本机 API 支持为准；
- V100 只用 FP16，不用 BF16；
- use_orig_params 是否使用必须记录；
- activation checkpointing 先关闭；
- CPU offload 先关闭。

启动先 3 step dry run，再 20/100 step：

    CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 src/train_fsdp.py \
      --config configs/full_shard_600m.yaml \
      --max_steps 100

一步等价：

1. 相同初始化；
2. 相同 global batch；
3. DDP/reference 与 FSDP；
4. FP32 tiny config 先比较；
5. 再比较 FP16；
6. 汇总 global loss；
7. 检查 update 后 full state 的关键 tensor。

记录：

- peak allocated/reserved；
- forward/backward/optimizer；
- all-gather/reduce-scatter；
- unit count；
- params per unit；
- loss/grad norm；
- scaler/skipped。

PASS：

- 100 step 无 NaN/死锁/rank divergence；
- peak model-state memory 明显低于 DDP；
- 固定一步在容差内；
- 能从 profiler 区分 all-gather 与 reduce-scatter。

### 周三：8-way Full Shard vs 2×4 Hybrid Shard

目标：让 sharding 设计匹配实际拓扑。

先做 4 卡 Full Shard，再 8 卡 Full Shard。每个只做 20-step smoke，通过后 100 timed steps。

Hybrid 逻辑：

- shard rows：0,1,2,3 与 4,5,6,7；
- replicate columns：0/4、1/5、2/6、3/7；
- shard 高频 all-gather/reduce-scatter 留在各四卡 NVLink 域；
- replica 维通过跨组直连完成梯度同步；
- 每卡只得到 4-way state sharding，不是 8-way。

具体 DeviceMesh/HYBRID_SHARD API 随 torch 版本变化。先：

    python -c "import torch; print(torch.__version__)"
    python -c "from torch.distributed.fsdp import ShardingStrategy; print(list(ShardingStrategy))"

若本机版本支持 DeviceMesh/2D hybrid，按对应文档实现；若不支持：

- 完成 8-way Full Shard；
- 用两个独立 4-way Full Shard 组做概念/通信测量；
- 将 Hybrid 标记为 ENV-BLOCKED；
- 不升级生产环境。

比较：

| strategy | shard degree | replica degree | peak GB | tok/s | all-gather share | reduce share |
|---|---:|---:|---:|---:|---:|---:|
| DDP | 1 | 8 | | | | |
| Full Shard | 8 | 1 | | | | |
| Hybrid | 4 | 2 | | | | |

必须解释：

- Full Shard 内存更省但参数收集范围更大；
- Hybrid 内存少省一半但可能减少跨全图频繁参数通信；
- workload/sequence/batch 会改变胜负；
- 不能只看吞吐，不看是否容纳更大 batch/model。

### 周四：sharded checkpoint、同 world-size resume 与迁移

目标：把 FSDP 训练变成可恢复系统。

checkpoint 必须恢复：

- sharded model；
- optimizer state；
- scheduler；
- GradScaler；
- RNG；
- sampler；
- global step/tokens seen；
- config/data revision；
- FSDP strategy/wrap policy/world size。

流程：

1. 8 卡跑 50 step；
2. 每 rank 写合法 shard，使用框架支持的 distributed checkpoint；
3. barrier；
4. 退出；
5. 8 卡恢复到 100；
6. fixed next-batch loss 比较；
7. 模拟 incomplete checkpoint：少一个 completion marker 时 loader 必须拒绝；
8. 保存完成 marker 必须最后写。

world-size 迁移作为 stretch：

- 尝试 8 卡保存 → 4 卡加载的官方受支持路径；
- 或导出 full state 只用于小模型验证；
- 不手工拼接大 checkpoint；
- 若版本不支持 optimizer reshard，明确限制。

checkpoint 性能：

- 保存 wall time；
- 每 shard 大小；
- rank 间是否均衡；
- 保存期间 GPU idle；
- 磁盘空间；
- retention 规则。

PASS：

- 同 world size resume；
- next fixed-batch loss 相对偏差 <1%；
- optimizer/scaler/RNG/step 连续；
- incomplete checkpoint 不被当作成功；
- checkpoint 不由所有 rank 同时覆盖一个文件。

### 周五：activation checkpointing、故障恢复与架构阅读

目标：区分 model-state OOM 与 activation OOM。

A/B：

- FSDP Full Shard；
- activation checkpointing off/on；
- 同 batch/sequence/global batch；
- 20 warmup + 100 timed；
- peak memory、throughput、forward/backward time；
- loss/grad correctness。

判断：

- model-state 主导：FSDP 收益大；
- activation 主导：sharding 后仍 OOM，checkpointing/sequence/batch 更重要；
- all-gather buffer 主导：wrap unit 可能过大；
- collectives 过碎：wrap unit 可能过小。

安全故障：

- 在 20–50 step 后正常发送终止信号或停止作业；
- 只从上一个完整 checkpoint 恢复；
- 不通过 kill -9 制造共享系统风险，除非公司规范允许；
- 验证没有丢 step/重复消费不可接受数据。

最后只读比较：

1. Picotron 的 data_parallel.py / tensor_parallel.py 文件布局；
2. Nanotron 的 trainer/config/parallel context；
3. TorchTitan 的 train spec、parallelisms、checkpoint 结构。

写一页“从我的 600M trainer 到生产框架多了什么”，至少包括：

- config validation；
- elastic/fault tolerance；
- distributed checkpoint；
- metrics；
- multiple parallel dimensions；
- kernel abstraction；
- data pipeline；
- test matrix。

不要运行最新 TorchTitan/Nanotron 的 H100 优化路径。

## 6. 本周硬验收

| 项目 | PASS |
|---|---|
| budget | 0.6B/1.5B/4B model-state 与 activation 纸面预算 |
| wrap | 所有参数覆盖；unit 粒度合理；共享权重处理明确 |
| correctness | DDP vs FSDP 一步等价在容差内 |
| stability | 0.6B FSDP 200 step 无 NaN/死锁/divergence |
| memory | FSDP 显著降低每卡 model-state 峰值 |
| strategy | Full Shard 与 Hybrid 实测，或 Hybrid 明确 ENV-BLOCKED |
| profile | all-gather/reduce-scatter 可识别 |
| checkpoint | 同 world-size 完整恢复 model/optim/scaler/RNG/step |
| failure | incomplete checkpoint 被拒绝；安全中断可恢复 |
| mastery | 能解释 sharding 没有自动解决 activation OOM |

## 7. 常见故障

### FSDP 没省内存

- wrap policy 没生效；
- 模型太小，临时 buffer/allocator 主导；
- 测在 optimizer state 初始化之前；
- full state dict 常驻；
- use_orig_params/共享权重；
- CPU/GPU reference 未清理；
- activation 主导。

### load checkpoint 卡住

- world size/strategy 不匹配；
- rank 路径不一致；
- 某 shard 缺失；
- barrier 位置；
- 所有 rank 是否共同调用 API；
- completion marker 是否错误。

### loss 与 DDP 不一致

- mixed precision reduction；
- global batch；
- loss reduction；
- init seed；
- tied weights；
- optimizer state；
- grad clipping 的 global norm 语义；
- scaler skip。

## 8. 闭卷口试

1. DDP、Full Shard、Hybrid 各复制/切分什么？
2. FSDP forward 前为什么需要 all-gather？
3. backward 为什么需要 reduce-scatter？
4. wrap unit 太大/太小各有什么后果？
5. 为什么 FSDP 不自动消除 activation OOM？
6. 2×4 Hybrid 为什么只有四路 state sharding？
7. sharded checkpoint 为什么需要 completion marker？
8. world-size 迁移为何比同规模 resume 难？
9. activation checkpointing 的计算代价来自哪里？
10. 在这台 topology 上为何 TP8 通常不如 TP4×另一维自然？
