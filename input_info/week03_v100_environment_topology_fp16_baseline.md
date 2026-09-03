# 第 03 周执行手册：V100 环境、拓扑、NCCL 与 FP16 基线

> 对应原 12 周主线第 1 周。标准投入 7.5 小时。  
> 本周载荷：第 01 周 MiniGPT 的可缩放版本；单机 8×V100 32GB。  
> 本周结果：一份可信 Hardware Card，而不是一堆 nvidia-smi 截图。

每日时间盒：10 分钟写今天唯一要回答的问题，50–60 分钟实验，15–20 分钟记录；第 2 小时只用于一个对照或故障实验。

## 1. 本周主问题

这台机器对训练而言到底有什么能力边界？你需要用实验证明：

- 8 张卡是否都能做正确计算；
- FP16 Tensor Core 路径是否真正工作；
- NV2、NV1、SYS pair 的 collective 差异；
- GPU0–3、GPU4–7 两个四卡组是否近似对称；
- NCCL 是否自动利用 hybrid cube-mesh NVLink 图；
- hwloc/sysfs 警告影响什么、不影响什么；
- 当前软件栈能否作为后续 11 周的冻结基线。

## 2. 已知拓扑事实

| 范围 | 已知事实 |
|---|---|
| GPU 0–3 | CPU affinity 0-17,36-53；四卡内均为 NV1/NV2 |
| GPU 4–7 | CPU affinity 18-35,54-71；四卡内均为 NV1/NV2 |
| 跨组直连 | 0↔4 NV1、1↔5 NV1、2↔6 NV2、3↔7 NV2 |
| 对照 pair | 0↔1 NV2；0↔3 NV1；0↔5 SYS |
| NIC0 | GPU2/3 为 PIX；本计划单机训练基本不受它影响 |
| 警告 | hwloc 无法读取完整 sysfs CPU topology，NUMA 输出不可盲信 |

存在全 NVLink 8 卡环，例如：

    0 → 1 → 5 → 7 → 3 → 2 → 6 → 4 → 0

这只证明路径存在，不要求你手工强制 NCCL ring。默认让 NCCL 自己选算法和 channel。

## 3. 官方资料与获取

| 资源 | 用途 | 地址 |
|---|---|---|
| NVIDIA nccl-tests | collective correctness 与 algbw/busbw | https://github.com/NVIDIA/nccl-tests |
| PyTorch AMP | autocast 与 GradScaler 规则 | https://docs.pytorch.org/docs/stable/amp.html |
| CUDA release notes | 确认 CUDA 13 对旧架构的兼容边界 | https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html |
| FlashAttention 官方仓库 | 核对 FA2 支持 Ampere/Ada/Hopper，不用于 V100 | https://github.com/Dao-AILab/flash-attention |
| PyTorch distributed docs | process group 与 collectives | https://docs.pytorch.org/docs/stable/distributed.html |

不要因为文档最新版与机器版本不同就升级生产环境。本周先记录版本，再以机器已安装版本对应文档为准。

## 4. 建立本周目录

    week03-hardware/
      env/
        probe_env.py
        env_snapshot.txt
      topology/
        topology_card.md
        nccl_matrix.csv
      src/
        cuda_correctness.py
        matmul_bench.py
        transformer_bench.py
        distributed_probe.py
      configs/
        transformer_30m.yaml
        transformer_120m.yaml
      reports/
        hardware_card.md
        decision.md

所有输出留在公司机器。终端只需要打印脱敏汇总；不把具体性能数字转发到外部聊天。

## 5. 每日安排

### 周一：冻结环境事实与最小 CUDA 正确性

目标：先证明每张卡独立正确，再谈通信。

环境探针至少打印：

    Python version
    torch version
    torch.version.cuda
    cuDNN version
    NCCL version
    driver version
    visible GPU count
    GPU name and total memory
    compute capability
    torch.cuda.get_arch_list()
    torch.cuda.is_bf16_supported()

建议命令：

    nvidia-smi
    nvidia-smi topo -m
    python -m torch.utils.collect_env
    python env/probe_env.py
    lscpu -e=CPU,NODE,SOCKET
    numactl -H
    sed -n '/Cpus_allowed_list/p' /proc/self/status

如果 lscpu/numactl 因容器权限失败，记录 FAIL-VISIBILITY，不尝试 mount 或绕过权限。

为每张卡跑相同 correctness：

1. 固定 seed 生成两个矩阵；
2. GPU FP32 matmul 与 CPU/FP64 reference 比较；
3. GPU FP16 matmul 结果转 FP32 后比较；
4. forward/backward 各 20 次；
5. 检查 NaN/Inf 和最大相对误差。

示例启动：

    for gpu in 0 1 2 3 4 5 6 7; do
      CUDA_VISIBLE_DEVICES=$gpu python src/cuda_correctness.py --steps 20
    done

若 shell 规范不允许 loop，逐卡运行。不要后台同时压满 8 卡，除非调度规则允许。

当天 PASS：

- 8 卡均可见且显存约 32GB；
- compute capability 为 7.0；
- BF16 支持结果符合 V100 预期；
- 8 卡 FP32/FP16 correctness 全通过；
- CUDA 版本、驱动、torch、NCCL 被写入 manifest。

可选第二小时：用 256/512/1024/2048 方阵比较 FP32 与 FP16，并记录小矩阵为什么可能没有加速。

### 周二：理解拓扑而不是只抄矩阵

目标：形成可操作的 rank mapping。

核心任务：

1. 把 topo 矩阵录入 topology_card；
2. 标注 NV2、NV1、SYS 三类 pair；
3. 标注两个 CPU affinity 域；
4. 画出四条跨域直连；
5. 写出后续 2/4/8 卡首选映射；
6. 检查 P2P access：

    python -c "import torch; print([[torch.cuda.can_device_access_peer(i,j) for j in range(8)] for i in range(8)])"

7. 检查 CPU 可见性，但不把 NUMA Affinity 0 当真相；
8. 写出三个实验假设：
   - NV2 pair 的 collective 上限应优于 SYS pair；
   - 两个四卡 NVLink 组在空闲机器上应大致对称；
   - 8 卡结果由 NCCL 图算法决定，不能由一个 pair 直接推断。

拓扑卡必须包含：

| 实验 | CUDA_VISIBLE_DEVICES | 目的 |
|---|---|---|
| 2 卡 NV2 | 0,1 | 最佳同组 pair |
| 2 卡 NV1 | 0,3 | 同组较弱 link |
| 2 卡 SYS | 0,5 | 跨 CPU 对照 |
| 4 卡 A | 0,1,2,3 | 第一 NVLink 四卡组 |
| 4 卡 B | 4,5,6,7 | 第二 NVLink 四卡组 |
| 8 卡 | 0,1,2,3,4,5,6,7 | 全图 |

当天口试：

- PIX、SYS、NV1、NV2 分别表示什么；
- 为什么 GPU2/3 靠近 NIC0 对单机 all-reduce 不重要；
- 为什么 0,5 是 SYS 却不说明 8 卡 NCCL 只能走 SYS；
- rank 编号如何受 CUDA_VISIBLE_DEVICES 重映射影响。

### 周三：nccl-tests correctness 与通信上限

目标：用标准工具建立 collective 上限。

获取：

    git clone --depth 1 https://github.com/NVIDIA/nccl-tests.git third_party/nccl-tests

构建前先检查 CUDA_HOME、mpicc 是否存在。单机不需要 MPI 时：

    cd third_party/nccl-tests
    make MPI=0

如果编译权限/工具链缺失，优先请求管理员提供已批准 binary；不要改系统编译器。可用 PyTorch distributed_probe 作为 correctness fallback，但周结论注明没有标准 busbw。

对每个映射至少测试 all_reduce：

    CUDA_VISIBLE_DEVICES=0,1 ./build/all_reduce_perf -b 8M -e 1G -f 2 -g 2
    CUDA_VISIBLE_DEVICES=0,3 ./build/all_reduce_perf -b 8M -e 1G -f 2 -g 2
    CUDA_VISIBLE_DEVICES=0,5 ./build/all_reduce_perf -b 8M -e 1G -f 2 -g 2
    CUDA_VISIBLE_DEVICES=0,1,2,3 ./build/all_reduce_perf -b 8M -e 1G -f 2 -g 4
    CUDA_VISIBLE_DEVICES=4,5,6,7 ./build/all_reduce_perf -b 8M -e 1G -f 2 -g 4
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 ./build/all_reduce_perf -b 8M -e 1G -f 2 -g 8

参数需要按可用显存/工具版本调整。不要把 -e 设到导致机器 OOM。每个配置重复 3 次，记录：

- size；
- time；
- algbw；
- busbw；
- out-of-bounds/error count；
- p50 与波动；
- 当时是否有共享负载。

再选一个中等 size 测 all_gather 和 reduce_scatter，供 FSDP 周参考。

一次且仅一次打开：

    NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=GRAPH CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 ./build/all_reduce_perf -b 256M -e 256M -f 2 -g 8

日志留本机。检查是否识别 NVLink/P2P、channel/ring/tree，但不强制 NCCL_ALGO。

PASS：

- 所有 collective correctness 通过；
- NV2/NV1/SYS 差异被量化；
- 四卡 A/B 的差异有重复测量；
- 8 卡得到 algbw/busbw 上限；
- 能解释 algbw 与 busbw 的区别。

### 周四：30M/120M Transformer 的 FP32/FP16 基线

目标：把通信之外的计算基线固定下来。

复用第 01 周模型，准备两种规模：

| 模型 | 用途 |
|---|---|
| 30M 左右 | correctness、shape、快速迭代 |
| 100M–150M | Tensor Core、显存、后续 scaling |

固定 synthetic token dataset，排除 I/O。每组：

- 20 warmup steps；
- 100 timed steps；
- FP32；
- FP16 AMP；
- hidden/head dimensions 对齐版本；
- 一个故意不友好的 shape 对照；
- batch 逐步增加到 reserved 约 28–30GB 之前停止。

计时需要 CUDA events 或显式 synchronize，不能只用未同步 Python time。

记录：

| config | precision | samples/s | tokens/s | step p50/p95 | peak allocated | peak reserved | final scale | skipped |
|---|---|---:|---:|---:|---:|---:|---:|---:|

FP16 correctness：

- 同权重、同 batch，FP32/FP16 forward loss 相对差异有界；
- 200 step 无 NaN/Inf；
- GradScaler warmup 后不持续回退；
- clip 在 unscale 后；
- 标准参数仍以 FP32 保存，不用 model.half() 代替 AMP。

如果 FP16 不快：

1. 看模型/矩阵是否太小；
2. 看 hidden/head 是否对齐；
3. 看 data/CPU launch 是否主导；
4. 看 autocast 是否覆盖 GEMM；
5. 看 GPU 利用率和 profiler，而不是下结论“V100 Tensor Core 无用”。

### 周五：Hardware Card 与故障演练

目标：得到后续所有实验引用的冻结基线。

做一次安全故障演练：

- 给 FP16 loss 人为乘一个很大系数；
- 观察 GradScaler 回退/skipped step；
- 恢复正常 loss；
- 验证训练可继续且参数没有 NaN；
- 删除故障注入，不进入后续配置。

Hardware Card 必须包含：

1. 硬件/软件版本；
2. CUDA 13/FlashAttention-2/BF16 禁用项；
3. 8 卡正确性；
4. topology 与首选 rank mapping；
5. NCCL pair/4-card/8-card 摘要；
6. FP32/FP16 baseline；
7. hwloc/NUMA 可见性限制；
8. 稳定 batch/memory envelope；
9. 后续实验冻结的默认环境；
10. 已知未知项。

最后闭卷写：

- FP16 一步更新的完整顺序；
- DDP 前为什么要先测 NCCL；
- 2 卡 NV2、2 卡 SYS、4 卡、8 卡分别回答什么；
- CUDA runtime、driver、torch build 三者关系。

## 6. 本周硬验收

| 项目 | PASS |
|---|---|
| 可见性 | 8 卡均可独立完成 forward/backward |
| 精度 | 120M 左右载荷 FP16 200 step 无 NaN/Inf |
| AMP | GradScaler、unscale、clip、step 顺序正确 |
| NCCL | NV2/NV1/SYS、4A/4B、8-card correctness 全通过 |
| 拓扑 | 能解释全图而不是只会背 pair |
| 性能 | FP32/FP16 有严格相同 workload 的对照 |
| 兼容 | 不使用 BF16、FA2、CUDA 13 专用路径 |
| 报告 | Hardware Card 可让未来的你复现实验条件 |

## 7. 止损与升级规则

- 任一卡 correctness 失败：不进入多卡训练，先报平台；
- NCCL 出现 data error/hang：记录最小复现，不强制禁用 P2P 掩盖；
- CUDA 13 环境：不自行降级或重装，走公司环境流程；
- hwloc 警告：不尝试越权读取 host sysfs；
- 共享机器负载波动大：性能结论标记 INCONCLUSIVE，correctness 可继续；
- nccl-tests 无法编译：用 PyTorch collective probe 完成正确性，并把带宽上限列为缺口。

## 8. 本周闭卷口试

1. 单卡 32GB 为什么不等于 DDP 可见 256GB？
2. NVLink pair 性能与 8 卡 collective 有何区别？
3. algbw 和 busbw 分别回答什么？
4. 为什么 rank 0 不一定对应物理 GPU0？
5. BF16 与 FP16 的指数范围差异为什么重要？
6. GradScaler 为什么能缓解 underflow，却不能修复所有 overflow？
7. FP16 GEMM 为什么对 shape 对齐敏感？
8. allocated 与 reserved 显存有什么区别？
9. 何时 NUMA/CPU affinity 会影响 DataLoader？
10. 什么证据足以把故障归为 FAIL-SYSTEM？
