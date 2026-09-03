# 第 07 周执行手册：DDP 从原语到 1/2/4/8 卡正确性与扩展

> 对应原 12 周主线第 5 周。标准投入 7.5 小时。  
> 主载荷：300M–500M decoder Transformer 或 DiT surrogate。  
> 本周目标：不仅会 torchrun，而是能解释数据切分、梯度同步、global batch、拓扑和 scaling。

每日时间盒：10 分钟写 invariant，50–60 分钟运行一个规模点，15–20 分钟核对 ranks/样本/数值；第 2 小时只增加一个拓扑或效率对照。

## 1. 官方资源

| 资源 | 用途 | 地址 |
|---|---|---|
| PyTorch DDP | API 与行为 | https://docs.pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html |
| PyTorch distributed | process group/collective | https://docs.pytorch.org/docs/stable/distributed.html |
| CS336 A2 | 从自己的 A1 模型做 distributed systems | https://github.com/stanford-cs336/assignment2-systems |
| NCCL tests | 第 03 周通信上限 | https://github.com/NVIDIA/nccl-tests |
| Picotron | 教学级 DP/TP/PP/CP 参考；本周只读 data parallel | https://github.com/huggingface/picotron |

本周先亲手写最小 all-reduce，再使用 DDP。Picotron 在你完成主实验后才读，不用它替代自己的理解。

## 2. 核心公式

    global_batch
      = micro_batch_per_gpu
      × gradient_accumulation_steps
      × world_size

弱扩展：

    micro_batch_per_gpu 固定
    global_batch 随 world_size 增长

强扩展：

    global_batch 固定
    每卡工作量随 world_size 下降

效率：

    E_N = throughput_N / (N × throughput_1)

必须分开报告弱扩展和强扩展，不能把 global batch 同时变化的 loss/吞吐放在一个结论里。

## 3. 目录

    week07-ddp/
      src/
        dist_env.py
        collective_probe.py
        manual_grad_allreduce.py
        train_ddp.py
        sampler_audit.py
        compare_checkpoints.py
      configs/
        model_350m.yaml
        strong_scaling.yaml
        weak_scaling.yaml
      scripts/
        run_1gpu.sh
        run_2gpu.sh
        run_4gpu.sh
        run_8gpu.sh
      reports/
        scaling.csv
        topology_pairs.csv
        scaling_report.md

每个 rank 只写自己的 rank-local 日志，rank 0 写汇总。文件命名包含 run_id/rank，避免并发覆盖。

## 4. 启动前的统一进程规则

每个进程：

1. 从 LOCAL_RANK 选择当前 CUDA device；
2. init_process_group backend=nccl；
3. seed 分成 model seed 与 data seed；
4. 构造模型并移到本地 device；
5. DDP wrap；
6. DistributedSampler；
7. 每 epoch 调 sampler.set_epoch(epoch)；
8. 只有 rank 0 做公共 checkpoint/eval 汇总；
9. finally destroy_process_group。

注意：设置 CUDA_VISIBLE_DEVICES=0,5 后，进程内 local rank 0/1 对应物理 GPU0/5。不要在代码里再次硬编码物理号。

## 5. 每日安排

### 周一：collective 原语与手工梯度平均

目标：先看到 DDP 帮你做的核心动作。

Part A：collective_probe

- 每 rank 创建值等于 rank 的 tensor；
- all_reduce sum/mean；
- broadcast；
- all_gather；
- reduce_scatter 若版本支持；
- barrier；
- 检查每个 rank 的期望结果。

启动：

    CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 src/collective_probe.py

Part B：manual_grad_allreduce

1. 两个 rank 使用相同初始模型；
2. 各读不同 micro-batch；
3. backward 后逐参数 all_reduce grad；
4. 除以 world size；
5. optimizer step；
6. 与单进程 global batch reference 比较。

必须处理：

- loss reduction 是 mean 还是 sum；
- uneven final batch；
- accumulation；
- grad 为 None；
- 参数 iteration order 一致。

通过后才换成 DDP，并对比 step 后参数：

    max_abs_param_diff < 预注册容差

FP32 tiny model 目标 1e-6 量级或更紧；FP16 AMP 另设合理容差。

当天口试：

- DDP 默认平均还是求和梯度；
- local loss 与 global loss 如何汇总；
- 为什么 all_reduce 后还要除 world size；
- manual all-reduce 在何处容易死锁。

可选第二小时：用 Gloo/CPU 跑同一单测，帮助区分 collective 语义与 CUDA/NCCL。

### 周二：固定 global batch 的 1 卡 vs 2 卡等价性

目标：证明 DDP 不改变训练数学问题。

配置例：

| world size | micro batch | accum | global batch |
|---:|---:|---:|---:|
| 1 | 16 | 4 | 64 |
| 2 | 16 | 2 | 64 |

如果显存不允许，等比例缩小。样本顺序必须可追踪。

实验顺序：

1. 单卡 FP32 20 step reference；
2. 两卡 NV2 0,1 FP32 20 step；
3. 对比每步 global sample IDs；
4. 对比 loss；
5. 对比 step 后参数 hash/最大差；
6. 再跑 FP16 AMP 100 step；
7. 检查 scaler 同步策略与 overflow。

DistributedSampler 审计：

- 每 rank 打印前 20 sample IDs；
- 合并后无重复/漏样；
- epoch 边界调用 set_epoch；
- drop_last/padding 行为明确；
- validation 不重复计数。

gradient accumulation：

- 非最后 micro-step 使用 DDP.no_sync；
- 最后一步触发同步；
- loss 除 accum steps；
- profiler 证明 all-reduce 次数减少；
- 数值结果与每 micro-step 同步版本一致。

PASS：

- 前 20 step loss 相对偏差目标 <2%；
- FP32 reference 参数差异在预设容差；
- 样本不重复不遗漏；
- global batch 真正相同；
- FP16 100 step 无 rank divergence。

### 周三：NV2/NV1/SYS pair 与两个四卡组

目标：把拓扑差异与模型计算区分开。

固定每卡 micro-batch，做弱扩展/拓扑对照：

    CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 src/train_ddp.py ...
    CUDA_VISIBLE_DEVICES=0,3 torchrun --standalone --nproc_per_node=2 src/train_ddp.py ...
    CUDA_VISIBLE_DEVICES=0,5 torchrun --standalone --nproc_per_node=2 src/train_ddp.py ...

然后：

    CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 src/train_ddp.py ...
    CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --standalone --nproc_per_node=4 src/train_ddp.py ...

每次：

- 20 warmup；
- 100 timed；
- 3 repeats；
- 同模型、batch、precision；
- steady-state 不含 eval/checkpoint；
- 记录 compute、exposed communication、step p50/p95；
- 记录当时共享负载。

若权限允许，分组绑核 A/B：

- GPU0–3 进程/DataLoader 对应 CPU 0-17,36-53；
- GPU4–7 对应 CPU 18-35,54-71；
- 全程只用公司允许的 taskset/launcher 方式；
- hwloc 警告存在时不自称完成 NUMA 精确优化。

预期不是硬规格：

- NV2 pair 应优于 SYS pair；
- A/B 四卡组应大致对称；
- 若模型计算很重，pair 差异可能被隐藏；
- 若小 batch，通信/launch 可能主导。

PASS：

- 三种 pair 与两个四卡组均完成；
- 数值正确；
- 差异用 communication share/NCCL 上限解释；
- 没有先强制 NCCL_ALGO 或禁用 P2P。

### 周四：8 卡弱/强扩展与通信 overlap

目标：形成完整 1/2/4/8 scaling 曲线。

8 卡启动：

    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    torchrun --standalone --nproc_per_node=8 src/train_ddp.py \
      --config configs/model_350m.yaml \
      --precision fp16 \
      --max_steps 150

第一次只做 20 step smoke；通过后再正式计时。

必须运行：

1. 弱扩展：per-GPU micro batch 固定；
2. 强扩展：global batch 固定；
3. accumulation/no_sync A/B；
4. 一次 NCCL_DEBUG=INFO/GRAPH 短跑；
5. profiler active 5–10 step；
6. 默认 bucket vs 一个有依据的 bucket_cap_mb 对照，仅当 trace 显示通信暴露。

记录：

| N | mode | global batch | throughput | E_N | comm ratio | peak GB | loss |
|---:|---|---:|---:|---:|---:|---:|---:|

适用于 300M+ 计算密集弱扩展的诊断目标：

- E2 ≥85% 合理；
- E4 ≥75% 合理；
- E8 ≥60% 通过，≥70% 良好；
- E8 <50% 停止长跑，先诊断。

强扩展不使用这些同一阈值。

E8 低时按顺序：

1. workload 是否太小；
2. global/per-GPU batch 是否搞错；
3. DataLoader/CPU affinity；
4. all-reduce 是否暴露；
5. accumulation/no_sync；
6. bucket；
7. 周期性 checkpoint/eval/log；
8. 共享负载；
9. NCCL graph。

### 周五：故障注入、Scaling Report 与闭卷重建

目标：能够识别最常见的“训练能跑但结果错”。

安全故障实验，逐个短跑：

- 忘记 sampler.set_epoch；
- 每 rank 使用相同 data seed/样本；
- loss 未除 accumulation；
- no_sync 覆盖最后一个 micro-step；
- 只 rank0 保存但其他 rank 未同步；
- 不同 rank 条件分支导致某参数 unused。

每个故障回答：

- 指标怎样表现；
- 为什么可能不报错；
- 如何用 invariant/test 抓到；
- 正确修复。

Scaling Report：

1. workload；
2. global batch 公式；
3. correctness；
4. sample audit；
5. NV2/NV1/SYS；
6. 4A/4B；
7. 1/2/4/8 弱扩展；
8. 强扩展；
9. profiler/NCCL 解释；
10. 最佳 config 与适用边界。

闭卷重写最小 DDP trainer，不需要完整代码，但必须包括 local rank、device、process group、sampler、DDP、no_sync、rank0 checkpoint、cleanup。

## 6. 本周硬验收

| 项目 | PASS |
|---|---|
| 原语 | all_reduce/broadcast/gather 结果正确 |
| 理解 | 手工 grad all-reduce 与单卡 global batch reference 对齐 |
| 数据 | 各 rank 样本无重复/漏样；set_epoch 正确 |
| 等价 | 1/2 卡固定 global batch 前 20 step loss 偏差 <2% 或有证据 |
| 稳定 | 1/2/4/8 卡各至少 100 稳定 steps |
| 参数 | 同一 run 各 rank update 后参数一致 |
| 拓扑 | NV2/NV1/SYS、4A/4B、8-card 全部完成 |
| scaling | 弱/强扩展分开；E_N 公式正确 |
| profiling | 通信暴露比例可解释 |
| 掌控力 | 不看框架能写出最小 DDP 训练骨架 |

## 7. 常见故障

### hang

- 是否所有 rank 进入同样 collective；
- 某 rank 是否 OOM/exception；
- evaluation/checkpoint 是否只有部分 rank barrier；
- unused parameters/条件分支；
- timeout 前保存 rank-local 最小状态；
- 不用随机设置 NCCL 环境变量碰运气。

### 多卡 loss 与单卡差很多

- global batch；
- LR scaling；
- loss reduction；
- data order；
- accumulation division；
- dropout/RNG；
- BatchNorm；
- GradScaler skip 是否各 rank 一致。

### 8 卡比 4 卡还慢

- workload 太小；
- strong scaling 每卡无计算；
- DataLoader/CPU；
- logging rank；
- all-reduce 暴露；
- checkpoint/eval；
- 共享机器；
- NCCL topology。

## 8. 闭卷口试

1. DDP 为什么不节省单卡 model state 显存？
2. global batch 如何由三个量决定？
3. 强扩展与弱扩展分别回答什么？
4. DistributedSampler 为什么要 set_epoch？
5. no_sync 应包哪些 micro-steps？
6. 梯度平均与 loss reduction 如何配合？
7. 为什么不同 rank 的 scaler skip 必须一致？
8. bucket 太大或太小各有什么问题？
9. NV2 pair 更快为什么不保证 E8 更高？
10. 什么证据能区分 NCCL 瓶颈与 workload 太小？
