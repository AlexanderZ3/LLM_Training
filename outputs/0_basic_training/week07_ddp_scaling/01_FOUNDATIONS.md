# Week 07 基础篇：从两次局部 backward 到可信 DDP Scaling

> 发布状态：从零执行手册原理篇；数值与拓扑描述均非实测。公司 8×V100 32GB/Linux/PyTorch 2.1、两组四卡 NVLink 域为用户自述待核验。核验日期：2026-09-03。

## 1. 本周问题与最终产物

本周回答：“DDP 是否保持了训练数学，1/2/4/8 卡的速度变化究竟来自计算还是通信？”

Lab 从空目录创建一个完整脚本，产出：

- Gloo/CPU 与 NCCL/GPU collective 自测；
- deterministic global-batch 切片与 sample-id audit；
- FP32 单卡/两卡等价、FP16 1/2/4/8 卡 smoke；
- 强扩展、弱扩展、topology pair 和短 profiler；
- crash-safe、按 step 版本化的 DDP checkpoint/resume，与 uninterrupted-vs-resume 等价检查和独立 eval；
- 安全的重复样本失败注入；
- scaling.csv/decision.md 所需 JSON 证据。

它使用与 Week 06 相同的时间、显存、FP16 口径，但代码完全自包含，缺少上周目录也能执行。Week 08 会以此 DDP 结果作为 FSDP 数学 reference。

## 2. 环境与来源/许可

| 对象 | 来源/版本 | 许可与限制 |
|---|---|---|
| PyTorch DDP/torchrun | 官方 pytorch/pytorch v2.1.x API；实机记录精确 patch 版本 | BSD-style；公司已装版本优先 |
| NCCL | 随批准的 torch/CUDA 环境；记录 torch.cuda.nccl.version | NVIDIA NCCL 许可；不自行替换 |
| 合成 token | Lab 由 sample_id、seed 公式生成 | 无第三方数据；只能证明系统正确性 |
| TinyLM/脚本 | 本手册完整实现，SHA-256 固定 | 项目内部教学代码 |

不下载外部模型/数据；网络不可达不影响核心实验。若使用 Week 06 checkpoint，必须另记录文件 SHA/config/授权，本周主证据不依赖它。

三条资源：

- 公司 8×V100：NCCL 与 scaling 主路径；所有代码、数据、日志、NCCL 输出、trace、图片、checkpoint、性能数字留内部。
- 个人 5070 Ti：单 GPU 与 CPU/Gloo 两进程语义练习；不能制造多卡结论。
- 可选 H100：获授权才做独立对照；其 BF16/新 kernel 不替代 V100 FP16。

V100 支持 FP16 AMP、通常没有原生 BF16；卡数增加不能修复不兼容 kernel。公司 PyTorch 2.1 不强升。

## 3. 进程与设备

单机 N 卡采用一进程一卡：

    WORLD_SIZE = N
    RANK ∈ [0,N)
    LOCAL_RANK ∈ [0,N)

CUDA_VISIBLE_DEVICES=0,5 后，local rank 0/1 映射物理 GPU 0/5。程序只使用 local_rank 选择当前可见设备，不再次硬编码物理 ID。

每 rank 的模型参数 W shape 完整相同。设 W=[Dout,Din]，局部梯度 g_r 同 shape：

    g_global = (1/N) × Σ g_r

DDP bucket 是多个梯度展平后的 1-D buffer；它尽量在 backward 尚未结束时 all-reduce 已 ready bucket。

## 4. Global batch 与 worked example

    global_batch = micro_batch_per_rank × accumulation_steps × world_size

固定 global batch=64：

| world | micro | accum | global |
|---:|---:|---:|---:|
| 1 | 16 | 4 | 64 |
| 2 | 16 | 2 | 64 |
| 4 | 8 | 2 | 64 |
| 8 | 8 | 1 | 64 |

Lab 的 deterministic batch 用 global sample offset：

    base = update × global_batch
         + micro_index × (micro × world)
         + rank × micro

例：world=2、micro=2、accum=2、update=0，各局部 sample：

    micro0 rank0=[0,1], rank1=[2,3]
    micro1 rank0=[4,5], rank1=[6,7]

union 正好 [0..7]，无交集。这个公式让单卡与多卡看到同一 global sample 集。

## 5. Mean loss 与 accumulation

各 rank 样本数/有效 token 数相同，local loss 是 mean，DDP 又平均梯度时，得到 global mean 梯度。若 mask denominator 不等，rank mean 的等权平均不再等价；必须汇总 numerator/denominator。

K 次 accumulation：

    for k=0..K-1:
        backward(local_mean_loss / K)
        synchronize only when k=K-1
    optimizer.step()

前 K-1 次使用 no_sync，最后一次必须同步。漏除 K 会把梯度放大；最后一次也 no_sync 会让各 rank 参数分叉。

## 6. 最小可运行 collective 例子

以下命令可在已有 torch 环境用 CPU/Gloo 两进程运行，不依赖本地文件：

    torchrun --standalone --nproc-per-node=2 - <<'PY'
    import os, torch
    import torch.distributed as dist
    dist.init_process_group("gloo")
    rank=dist.get_rank()
    x=torch.tensor(float(rank))
    dist.all_reduce(x)
    print({"rank":rank,"sum":x.item()})
    dist.destroy_process_group()
    PY

部分 torchrun/shell 不支持 stdin 脚本时，使用 Lab 已创建的 src/ddp_lab.py。预计每 rank 的 sum=1；这是示例预期，不是实测。

## 7. Shape 与 dtype

Lab formal 候选 D=1024、H=16、Dh=64、T=256、V=16384、layers=24，参数约 3 亿量级；真实 parameters 字段为准。

| 张量 | shape |
|---|---|
| ids/targets | [micro,T] int64 |
| residual | [micro,T,D] fp16/fp32 |
| Q/K/V | [micro,H,T,Dh] |
| score（若物化） | [micro,H,T,T] |
| logits | [micro,T,V] |
| parameter gradient | 与参数同逻辑 shape |
| bucket | [若干参数 numel 总和] |

V100 主路径 autocast FP16+GradScaler；loss 计算可显式 FP32，clip 必须在 unscale 后。某 rank 跳步而其他 rank 更新会立即分叉，必须记录 per-rank scale/skip/checksum。

## 8. 强弱扩展

弱扩展固定每卡 micro，global batch 随 N 增；强扩展固定 global batch，每卡计算下降：

    speedup_N = throughput_N / throughput_1
    E_N = throughput_N / (N × throughput_1)

只对同一口径计算。对 300M+、计算密集、固定每卡 micro 的弱扩展，源计划诊断线为 E2≥85%、E4≥75%、E8≥60%通过、≥70%良好；E8<50%停止长跑。它们不是硬件规格，也不适用于强扩展/小模型。

## 9. 拓扑与通信

用户自述候选：0–3、4–7 为四卡域；0,1 可作 NV2，0,3 可作 NV1，0,5 可作 SYS 对照。每次必须用当日 nvidia-smi topo -m 重判，不能把候选当事实。

    exposed_comm = collective_time - overlap_with_backward

pair 带宽不是 8 卡效率。全图 NCCL channel、bucket ready 顺序、每卡计算、CPU/DataLoader 与共享负载都会影响 E8。先用 NCCL 默认拓扑发现，再做一次 INFO/GRAPH 短跑；不先强制 Ring 或禁 P2P。

## 10. 正确性不变量

- 所有 rank 模型结构、参数顺序、初始化相同；
- current device 在 NCCL 初始化前设置；
- 每 update 的 global sample union 正确、intersection 为空；
- global batch、loss reduction、loss/K 正确；
- 最后 micro-step 同步；
-所有 rank scale/skip 与 update 后 parameter checksum 一致；
- rank-local 文件不覆盖；内容先写临时版本目录，manifest/hash 核完后原子发布目录，最后以含格式版本、torch patch 与 manifest hash 的 `COMPLETED` 绑定本次提交；loader 不把旧 marker 与新内容混用；
- eval 使用与训练轨迹严格不相交的 sample-ID 区间，并同时报告训练上界与评估区间；
- correctness 未过，不计算效率。

性能报告分两种时钟：`steady_tokens_per_s` 只覆盖预热后的训练 update，开始/结束处同步，且窗口内不做日志、profile 或 checkpoint；`run_tokens_per_s_including_aux` 才包含日志与保存。扩展效率只使用前者，Profiler run 只诊断。

## 11. 失败诊断树

| 症状 | 最小检查 | 候选根因 | 修复/回归 |
|---|---|---|---|
| hang | 各 rank 最后 phase | collective 次序不同、某 rank OOM | 全作业终止；2卡短跑重现 |
| loss 与单卡偏离 | sample IDs/global batch | reduction、accum、RNG | FP32一步对照 |
| 参数分叉 | checksum/skip | no_sync末步、局部overflow | 修同步，20步回归 |
| 第二 epoch 顺序相同 | sampler epoch/IDs | 忘 set_epoch | set_epoch 后审计 |
| 8卡慢于4卡 | compute/comm、NCCL图 | 负载小、拓扑、I/O | 停长跑，逐项A/B |
| 文件损坏 | rank路径/marker | 并发覆盖、部分成功 | rank-local+最后marker |

不要随机堆 NCCL 环境变量。先确定最早分叉的 rank/phase。

## 12. 自检题

1. 手算 world=4、micro=2、accum=3 的 global batch。
2. 为什么 unequal valid-token count 破坏简单 rank mean？
3. 写 K=4 的 no_sync 边界。
4. throughput1=1000、throughput4=3000 时 E4？
5. 为什么 DDP 不节省单卡 model-state？
6. local_rank 与物理 GPU 何时不相等？
7. 设计 sample union/intersection 审计。
8. NV2 pair 更快为什么不能推出 E8？

## 13. 官方资料

- [PyTorch DDP 2.1](https://docs.pytorch.org/docs/2.1/generated/torch.nn.parallel.DistributedDataParallel.html)
- [PyTorch distributed 2.1](https://docs.pytorch.org/docs/2.1/distributed.html)
- [torchrun 2.1](https://docs.pytorch.org/docs/2.1/elastic/run.html)
- [DistributedSampler 2.1](https://docs.pytorch.org/docs/2.1/data.html#torch.utils.data.distributed.DistributedSampler)
- [NCCL 官方文档](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/)
- [NCCL tests](https://github.com/NVIDIA/nccl-tests)

链接与可变事实核验日期：2026-09-03。
