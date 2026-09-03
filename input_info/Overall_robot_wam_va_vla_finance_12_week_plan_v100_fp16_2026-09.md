# 12 周机器人 WAM / VA / VLA 与金融训练工程计划：8×V100 32GB FP16 版

> 版本：V3.1，2026-09-02；已纳入实际 `nvidia-smi topo -m` 拓扑。  
> 训练资源：公司内 8×NVIDIA V100 32GB；只允许向机器输入，训练结果、日志、trace、checkpoint 和数据不得导出。  
> 已知拓扑：8 卡 hybrid cube-mesh 型 NVLink 图；GPU 0–3 与 4–7 分属两个 CPU affinity 域；NIC0 与 GPU2/3 为 PIX。  
> 个人时间：每个工作日 1–2 小时，每周 5 天；按每周 5–10 小时、标准 7.5 小时、12 周约 90 小时设计。  
> 延续项目：机器人主线 `Dex-WAM`；金融副线 `FinExec-Reasoner`。  
> 核心目的：补齐独立完成“数据 → loss → FP16 稳定性 → checkpoint/resume → DDP/FSDP → Profiling → 评测与消融”的完整训练工程闭环。

---

## 一、先给结论

这台 8×V100 32GB 机器非常适合你当前最需要补的训练工程练习，而且比单张 5070 Ti 更适合系统学习多卡训练与通信 Profiling。

但新版不能把旧方案中的 `8×H100` 机械替换成 `8×V100`。正确调整是：

1. **V100 是训练系统实验室，不是前沿大模型成绩机器。**
   - 适合：FP16 AMP、loss scaling、DDP、FSDP、NCCL、checkpoint、数据管线、显存分析、训练吞吐、通信与计算重叠。
   - 可以做：TinyFlowPolicy、约 450M 的 MultiTask DiT / SmolVLA 小规模微调、Mini-WAM、0.6B–1.5B 全参训练练习、4B LoRA/QLoRA 或短程 FSDP smoke test。
   - 不作为主目标：5B–10B 前沿 VLA/WAM 的正式训练、8B 长程全参训练、27B CPT、π0.5/LingBot-VA 的强行多卡部署。

2. **32GB 是单卡显存，不等于每个任务都能看到 256GB。**
   - DDP 在每张卡上复制完整模型，单卡仍受 32GB 限制。
   - 只有 FSDP/ZeRO/TP 等切分策略会分摊部分模型状态；激活、临时 all-gather buffer 和某些非切分对象仍会占每卡显存。
   - 实际拓扑明显优于普通 8 卡 PCIe 服务器：GPU 0–3 与 GPU 4–7 各自构成全 NVLink 四卡组，两组间还有 `0↔4`、`1↔5`、`2↔6`、`3↔7` 四条直接 NVLink 边。
   - 因此 8 卡 DDP/NCCL 值得认真练习，但仍需拓扑感知；不能把任意两张卡视为等价。

3. **全程使用 FP16 mixed precision，不使用 BF16。**
   - 使用 `autocast(float16) + GradScaler`；保存并恢复 scaler state。
   - gradient clipping 必须发生在 `scaler.unscale_(optimizer)` 之后。
   - 自定义归一化、reduction、loss 和数值敏感计算必要时显式转 FP32。
   - 不把整个模型简单 `model.half()` 后就当作正确 AMP 实现。

4. **V100 是 Volta、计算能力 7.0，软件栈要主动兼容旧架构。**
   - CUDA 13 已移除 Volta 的离线编译和库支持，新环境优先固定在经验证的 CUDA 12.x 或现有可用的 11.8 环境。
   - 官方 FlashAttention-2 不支持 V100；项目若硬依赖 FA2、BF16、TF32、FP8 或 Hopper/Ampere 专用 kernel，必须改为兼容后端或缩小为 surrogate 实验。

5. **“只能输入、不能输出”会改变产物定义。**
   - 原始 trace、checkpoint、日志、训练数据全部留在公司机器内。
   - 计划中的验收优先使用终端表格和本机报告，不设计任何绕过公司边界的导出方式。
   - 若最终需要公开作品集，只在个人 5070 Ti 上用公开数据重跑一个缩小版；不复制公司机器的文件、指标或内部数据。

因此，这 12 周的目标不再是“同时做出两个大模型项目的完整研究成绩”，而是：

> 以 Dex-WAM 和 FinExec-Reasoner 为真实载荷，系统掌握单卡数值正确性、多卡等价性、显存与吞吐、通信瓶颈、故障恢复和受控消融。

---

## 二、相对 V2 的具体变化

| V2 设计 | V100 FP16 版调整 | 原因 |
|---|---|---|
| 每周约 26 小时、周末整天 | 每周 5–10 小时，仅 5 个工作日 | 匹配真实可投入时间 |
| RTX 5070 Ti + 单卡/多卡 H100 | 公司 8×V100 为主要练习资源 | 把重点转向训练系统与 Profiling |
| BF16 为默认训练精度 | FP16 AMP + GradScaler | V100 没有原生 BF16 Tensor Core 支持 |
| FlashAttention 等新 kernel 可默认使用 | 官方 FA2 禁用，使用已验证兼容后端 | V100 为 SM70 |
| 第 6 周运行 π0.5/LingBot-VA | 改为兼容性审计和 SmolVLA/Mini-WAM 实训 | 多卡不能自动解决单进程模型兼容与显存问题 |
| 第 9 周 8×H100 VLA-JEPA 正式训练 | 改为 1/2/4/8×V100 DDP/FSDP scaling lab | 学习目标是正确性、扩展和瓶颈归因 |
| 4×H100 上 8B SFT/CPT/GRPO | 0.6B–1.5B 全参闭环；4B 只做 PEFT 或短程 FSDP | 适应算力、软件和 90 小时时间预算 |
| 机器人正式工业成绩 + 金融完整成绩 | 一个训练工程 capstone + 两个领域载荷 | 避免在时间缩减后留下多个半成品 |

保留不变的内容：

- `Task-P`：公开、低成本、可重复的代理任务，用于训练代码、算法和消融。
- `Task-D`：真实工业灵巧手任务，用于最终业务价值验证。
- `Dex-WAM` 仍回答“世界监督是否改善动作学习/OOD 鲁棒性”。
- `FinExec-Reasoner` 仍回答“证据定位 → 可执行程序 → 数值答案”的训练与评测闭环。
- 每次实验仍必须有 config、数据版本、checkpoint、eval 和明确结论标签。

本版只要求公司机器完成 Task-P 和公开金融数据练习。Task-D 数据只有在公司与兼职双方均明确授权时才能进入该机器；否则 Task-D 保留为接口设计，不进行数据迁移。

---

## 三、V100 的能力边界

### 3.1 适合承担的任务

| 任务 | 推荐卡数 | 训练方式 | 本计划定位 |
|---|---:|---|---|
| 10M–50M TinyFlowPolicy | 1 | FP32 对照 + FP16 AMP | 学懂完整训练闭环 |
| 100M–500M Transformer/DiT surrogate | 1/2/4/8 | DDP | Scaling 与通信分析载荷 |
| MultiTask DiT / SmolVLA 约 450M | 1；短测 2/4/8 | FP16、冻结部分 encoder、DDP | 真实 VLA 训练链路 |
| Mini-WAM 20M–100M world head | 1/2/4 | FP16、联合 loss | 世界监督受控消融 |
| 0.6B–1.5B LLM | 1/2/4/8 | DDP 或 FSDP 全参短训 | CPT/SFT 与 sharding 练习 |
| 4B LLM | 1 或 4/8 | LoRA/QLoRA；或 FSDP 短 smoke | 显存与并行能力验证 |
| 8B LLM | 8 | 仅可选 FSDP 50–100 step | 内存可行性实验，不是正式训练目标 |

模型状态粗略估算可先按 AdamW 约 `12–16 bytes/parameter` 估计，再加激活、临时 buffer、通信 bucket 和框架开销。这个估算用于排除明显不可行配置，最终以实测峰值显存为准。

### 3.2 不应依赖的能力

- BF16、TF32、FP8；
- FlashAttention-2/3/4 官方 CUDA kernel；
- Hopper/Ampere 专用 Transformer Engine 路径；
- 把 8 卡显存当作透明统一显存；
- 依赖最新 CUDA 13 编译 Volta kernel；
- 用一周时间追逐只在新架构上成立的极致 benchmark。

### 3.3 已确认拓扑：双 CPU affinity 域 + hybrid cube-mesh 型 NVLink

根据实际矩阵，可以将机器概括为：

| 层级 | GPU/连接 | 结论 |
|---|---|---|
| CPU affinity 域 A | GPU 0–3；CPU `0-17,36-53` | 四卡内部任意两卡均为 NV1/NV2 |
| CPU affinity 域 B | GPU 4–7；CPU `18-35,54-71` | 四卡内部任意两卡均为 NV1/NV2 |
| 跨域直连 | `0↔4 NV1`、`1↔5 NV1`、`2↔6 NV2`、`3↔7 NV2` | 两个四卡组不是孤岛 |
| 其他跨域 pair | 多数为 `SYS` | 任意点对 benchmark 会有显著差异 |
| NIC0 | 对 GPU2/3 为 `PIX`，其余为 `SYS` | 单机训练基本无关；多机/RDMA 时重要 |

这张图不是“任意两卡全互联”，但它可以形成只走 NVLink 的 8 卡环。例如：

```text
0 → 1 → 5 → 7 → 3 → 2 → 6 → 4 → 0
```

上面只说明存在这样的路径，不要求手工把 NCCL ring 固定成这一条。默认先让 NCCL 自动发现拓扑，再通过一次 `NCCL_DEBUG=INFO`、必要时加 `NCCL_DEBUG_SUBSYS=GRAPH` 检查它实际选择的 channel/ring/tree。不要一开始强制 `NCCL_ALGO=Ring` 或禁用 P2P。

#### 拓扑感知的卡组选择

| 练习 | 首选映射 | 对照映射 | 目的 |
|---|---|---|---|
| 2 卡最佳链路 | `0,1` 或 `2,6`，均为 NV2 | `0,3` 为 NV1；`0,5` 为 SYS | 分离 NV2/NV1/SYS 影响 |
| 4 卡训练 | `0,1,2,3` | `4,5,6,7` | 两个全 NVLink 四卡组做对称性检查 |
| 8 卡 DDP | `0–7` | 4 卡最佳组 | 验证 NCCL 是否利用整张 NVLink 图 |
| 2×4 Hybrid Shard | shard rows：`0–3`、`4–7`；replicate columns：`0/4`、`1/5`、`2/6`、`3/7` | 8-way Full Shard | 降低频繁参数 all-gather 的拓扑代价 |
| 可选 TP4×PP2 | TP 组放在 `0–3` 与 `4–7`；PP 对应 rank 走 `0↔4` 等直连 | TP8 | 让高频 TP 通信留在四卡 NVLink 组内 |
| 可选 MoE | EP/All-to-All 优先限制在四卡组，另一维做 DP | EP8 | 避免大量随机跨域 `SYS` 流量 |

#### 新的扩展效率门槛

以下只用于 **300M+、计算密集、固定每卡 micro-batch 的弱扩展**；强扩展和小模型不能套用：

- 2 卡 NV2 pair：`E2 ≥85%` 为合理目标；
- 4 卡单 affinity 域：`E4 ≥75%` 为合理目标；
- 8 卡全机：`E8 ≥60%` 视为通过，`≥70%` 表现良好；
- `E8 <50%` 时停止长跑，先检查 batch 是否太小、CPU 绑核、DataLoader、NCCL ring/tree、通信暴露和周期性 checkpoint/eval。

这些是训练 workload 的诊断阈值，不是硬件带宽规格。最终要用 `nccl-tests` 的 `busbw` 作为 collective 上限，并解释训练达不到该上限的计算与同步原因。

#### `hwloc` 警告的处理

输出中的：

```text
hwloc/linux: failed to find sysfs cpu topology directory
```

不代表 NVLink 失效，但说明当前容器/环境无法完整读取 CPU topology sysfs。因此：

1. 暂时只把 `CPU Affinity` 的两组 CPU 列表当作可用事实；
2. 不把当前所有 GPU 显示的 `NUMA Affinity 0` 当成可靠结论；
3. 在权限允许时检查 `lscpu -e=CPU,NODE,SOCKET`、`numactl -H` 和 `/proc/self/status` 中的 `Cpus_allowed_list`；
4. 若 sysfs 被容器裁剪，交由管理员按公司规范暴露必要的只读拓扑信息，不绕过权限；
5. 训练 GPU0–3 时把 DataLoader/CPU 工作线程限制在第一组 CPU，GPU4–7 时限制在第二组；全 8 卡时再做分组绑核对照。

---

## 四、公司机器的输入/输出边界

### 4.1 可以做

- 按公司流程输入公开代码、公开数据、小型合成数据生成器和获批准的软件依赖；
- 在机器内保存 checkpoint、trace、日志、Profiler 报告和实验表；
- 在本机终端打印汇总指标并完成分析；
- 使用不含公司数据的预制模板记录 PASS / FAIL / INCONCLUSIVE；
- 在获授权前提下使用机器已有的容器、镜像、数据和调度系统。

### 4.2 不做

- 不通过截图、网盘、个人邮箱、聊天工具、外部 API 或其他方式带出训练结果；
- 不把公司数据、内部模型、拓扑细节或性能数据放入个人作品集；
- 不在未授权情况下将兼职 Task-D 数据输入公司机器；
- 不为了方便安装而绕过网络、镜像、USB 或账户权限；
- 不把公司机器上的指标转述成可公开比较的 benchmark。

### 4.3 适应单向环境的工作方法

每周输入的代码包要做到“配置驱动 + 终端自解释”，至少包含：

```text
v100-training-lab/
  README_OFFLINE.md
  env/
    probe_env.py
    requirements-lock.txt
  configs/
  scripts/
    smoke_single_gpu.py
    launch_ddp.sh
    launch_fsdp.sh
    run_profile.sh
  src/
    data/
    models/
    training/
    eval/
    profiling/
  tests/
  reports/templates/
  third_party_manifest.md
```

每个训练入口都应支持：

```text
--max_steps
--seed
--precision fp32|fp16
--resume_from
--profile_steps
--world_size
--output_dir
--dry_run
```

脚本必须在终端打印一个短汇总，而不是只能靠导出 trace 才能理解：

```json
{
  "status": "PASS",
  "step_time_ms_p50": 0,
  "samples_per_second": 0,
  "peak_memory_gb": 0,
  "grad_norm_p95": 0,
  "loss_scale_final": 0,
  "skipped_steps": 0,
  "data_ratio": 0,
  "communication_ratio": 0
}
```

---

## 五、统一时间节奏

### 5.1 每天 1–2 小时的使用方法

每个工作日采用同一结构：

1. `10 分钟`：写今天唯一要回答的问题；
2. `40–60 分钟`：完成核心任务；
3. `10–20 分钟`：读取结果、记录结论和下一个动作；
4. 若当天有第 2 小时：只做一个 stretch task，不扩展新方向。

每周三档工作量：

| 档位 | 周投入 | 要求 |
|---|---:|---|
| 最低 | 5 小时 | 完成五个核心任务和一个 smoke run |
| 标准 | 7.5 小时 | 完成核心任务、一次正式对照和周结论 |
| 上限 | 10 小时 | 增加一次 seed、一个规模点或一个故障实验 |

模型运行可以在你离开后继续，但只允许启动有明确 `max_steps`、保存周期和资源上限的作业，并遵守公司的共享机器与调度规则。无人值守长跑不替代实验设计。

### 5.2 每周固定产物

所有产物留在公司机器内：

1. `config.yaml`：唯一实验配置；
2. `run_manifest.json`：代码版本、环境、数据 hash、seed、world size；
3. `metrics.json`：训练、数值稳定性、显存、吞吐和评测；
4. `decision.md`：一页以内，写假设、证据、结论、下一步；
5. 一个标签：`PASS`、`FAIL-MODEL`、`FAIL-SYSTEM` 或 `INCONCLUSIVE`。

---

## 六、12 周总览

| 周 | 主问题 | 主要载荷 | 多卡/Profiling重点 | 标准工时 |
|---:|---|---|---|---:|
| 1 | 机器到底是什么能力边界？ | 合成 Transformer | 拓扑、NCCL、FP32/FP16 基线 | 7.5h |
| 2 | 一个训练循环怎样完整闭环？ | TinyFlowPolicy | AMP、GradScaler、overfit、resume | 7.5h |
| 3 | 性能问题是否其实来自数据？ | Task-P 数据合同 | 对齐、split、RNG、可复现 | 7.5h |
| 4 | 单卡时间和显存花在哪里？ | TinyFlow/DiT surrogate | PyTorch Profiler、Nsight、内存扫描 | 7.5h |
| 5 | DDP 扩展是否正确且有效？ | 300M–500M DiT/Transformer | 1/2/4/8 卡、NCCL、bucket | 7.5h |
| 6 | FSDP 如何改变显存与通信？ | 0.6B–1.5B Transformer | Full shard、checkpoint、world-size 迁移 | 7.5h |
| 7 | Diffusion 与 Flow 如何公平比较？ | MultiTask DiT / Task-P | 受控目标函数对照 | 7.5h |
| 8 | 真正 VLA 能否在 V100 稳定微调？ | SmolVLA 约 450M | FP16 兼容、冻结策略、DDP 短测 | 7.5h |
| 9 | world loss 是否提供有效监督？ | Mini-WAM | action/world/shuffle 消融 | 7.5h |
| 10 | 金融 SFT/CPT 管线是否正确？ | FinExec 0.6B | evaluator、packing、DDP/FSDP | 7.5h |
| 11 | 模型放大后瓶颈怎样变化？ | FinExec 1.5B/4B | PEFT 或短程 FSDP、可选 GRPO smoke | 7.5h |
| 12 | 能否独立复现、恢复和解释？ | 机器人或金融二选一 | 1→8 卡 capstone、故障恢复、总报告 | 7.5h |

机器人约占 52 小时，金融约占 15 小时，通用训练系统约占 23 小时。第 10 周之前不做金融 Agent；全计划不做金融 Agent/RAG/交易系统。

---

## 七、逐周执行计划

### 第 1 周：V100 环境、拓扑和 FP16 基线

**周目标**：先证明机器、软件栈和通信是可信的，避免之后把环境问题误判成模型问题。

| 工作日 | 60–90 分钟核心任务 | 可选第 2 小时 |
|---|---|---|
| 周一 | 运行环境探针：GPU 型号、32GB 显存、SM70、驱动、CUDA、PyTorch、NCCL；验证 `torch.cuda.is_bf16_supported()` 为 false | 检查 `torch.cuda.get_arch_list()` 是否含 `sm_70` |
| 周二 | 把已知矩阵整理成 Topology Card；确认两个 CPU affinity 域、四条跨域 NVLink 与 NIC0 位置 | 检查 `hwloc`/sysfs/NUMA 可见性，不擅自修系统 |
| 周三 | `nccl-tests` 分别测试 NV2 pair `0,1`、NV1 pair `0,3`、SYS pair `0,5`、四卡 `0–3`、四卡 `4–7` 和八卡 | 加 all-gather/reduce-scatter 与 P2P bandwidth/latency |
| 周四 | 在单卡跑 100M 左右 Transformer 的 FP32 与 FP16 AMP 各 200 step | 用对齐和未对齐 hidden size 比较 Tensor Core 利用 |
| 周五 | 形成 Hardware Card v1：显存、FP16、NV2/NV1/SYS、四卡/八卡 NCCL 与已知不兼容项 | 用一次 NCCL GRAPH 日志确认自动选择的拓扑路径 |

**必须记录**：

- FP32/FP16 step time、samples/s、峰值显存；
- 初始/最终 loss scale、overflow/skipped step；
- NCCL `algbw` 和 `busbw`，不要只看一个带宽数字；
- NV2/NV1/SYS 三类 pair 的 latency、`algbw` 和 `busbw`；
- 两个四卡组是否近似对称；
- 8 卡 collective 是否明显偏离四卡组外推；
- CUDA 是否为 13.x；若是，不在本周擅自重装，先走公司许可并准备 CUDA 12.x 兼容环境。

**验收门槛**：

- 8 卡均可见；NV2、NV1、SYS、两个四卡组和 8 卡 collective correctness 均通过；
- 能解释“很多 pair 是 SYS”为什么不等于“8 卡 all-reduce 必须走 SYS”；
- `hwloc` 警告被记录并完成可见性检查，不把错误 NUMA 信息写进后续结论；
- FP16 200 step 无 NaN/Inf，参数无非有限值；
- 能解释 FP16 为何需要 GradScaler，以及 V100 为什么不能把 BF16 当作默认精度；
- 若 FP16 未明显快于 FP32，能用 shape、kernel 或 GPU 利用率证据解释，而不是直接判断 V100 没有 Tensor Core 收益。

### 第 2 周：TinyFlowPolicy 与完整训练闭环

**周目标**：在最小模型上独立拥有数据、forward、loss、backward、optimizer、EMA、采样、checkpoint 和 eval。

| 工作日 | 核心任务 | 可选第 2 小时 |
|---|---|---|
| 周一 | 推导 conditional flow matching；写清 `x0/x1/t/vθ` 与 action chunk shape | 加 MSE behavior cloning 对照 |
| 周二 | 完成 10M–50M TinyFlowPolicy 和合成/PushT window dataset | 为 padding/mask 写单测 |
| 周三 | FP32 过拟合 128–256 个窗口 | 扫描 action horizon 16/32 |
| 周四 | 改成 FP16 AMP；保存 model、optimizer、scheduler、EMA、scaler、RNG 和 global step | 测试 gradient accumulation |
| 周五 | 中断后 resume，同一 fixed batch 对比连续训练；运行短 rollout/eval | 打印按模块的耗时表 |

**FP16 正确模板**：

```python
with torch.autocast(device_type="cuda", dtype=torch.float16):
    output = model(batch)

# 自定义数值敏感 loss 在 FP32 中计算；标准 loss 也应确认 autocast 策略。
loss = compute_loss(output.float(), batch)

scaler.scale(loss).backward()
scaler.unscale_(optimizer)
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
scaler.step(optimizer)
scaler.update()
```

**验收门槛**：

- 小样本过拟合成功；
- resume 后下一个固定 batch 的 loss 相对偏差 `<1%`；
- checkpoint 包含 scaler 和 RNG state，不只保存 model weights；
- warmup 后 skipped optimizer steps 占比目标 `<1%`；
- 能逐张量解释一个 batch 如何变成 action chunk，以及噪声如何被采样器还原为动作。

### 第 3 周：Task-P 数据合同、对齐与可复现

**周目标**：证明模型看到的是正确任务，排除“训练正常但数据含义错误”。

Task-P 仍优先选择 RoboTwin 2.0 中一个与 Task-D 接近的单任务；若环境导入成本过高，用已批准的 LIBERO 单任务或合成 episode 替代。12 周内只固定一个 Task-P。

| 工作日 | 核心任务 | 可选第 2 小时 |
|---|---|---|
| 周一 | 冻结 episode schema、camera order、state/action 单位、坐标系、rotation 表示 | 写 Task Card |
| 周二 | 实现 validator：边界、时间戳、NaN/Inf、动作范围、静止比例 | 加可视化/终端分位数摘要 |
| 周三 | 做 camera/action lag sweep；确定补偿 lag | 对比 absolute 与 delta action |
| 周四 | 固定 train/dev/test，检查 episode、对象、seed 泄漏；统计量只来自 train | 加 hash manifest |
| 周五 | 两次同 seed 训练和一次不同 seed 训练；分析可复现边界 | 恢复 DataLoader sampler state |

**验收门槛**：

- validator 通过率 100%，NaN/Inf 为 0；
- split 无 episode、对象实例或 seed 重叠；
- 时间对齐误差小于一个 control period，或配置中明确记录补偿值；
- 同环境、同 seed 的前 20 step loss 符合预设容差；
- Data Card 与 Task Card 能让你两周后不看代码也能解释每个 tensor 的业务含义。

### 第 4 周：单卡 Profiling 与显存模型

**周目标**：不只会“看 GPU 利用率”，而是能回答 step time 和显存分别由什么构成。

| 工作日 | 核心任务 | 可选第 2 小时 |
|---|---|---|
| 周一 | 给 data、H2D、forward、loss、backward、optimizer、eval 加明确 range | 给 attention/MLP/world head 加子 range |
| 周二 | 用 PyTorch Profiler 跑 warmup + 10 个 active step，终端打印 CUDA time top-10 | 打开 shapes 和 memory 分析 |
| 周三 | 若已安装 Nsight Systems，用 CLI 采集 CUDA/NCCL 时间线并留在本机 | 用 Nsight Compute 查看一个最重 kernel |
| 周四 | 扫描 batch、sequence/action horizon、image resolution，形成 OOM envelope | 比较 checkpointing 开/关 |
| 周五 | 只优化一个最大瓶颈，再用完全相同配置复测 | 比较 pin memory、worker、prefetch |

**时间测量规则**：

- 丢弃冷启动、编译和前 20 个 warmup step；
- 用 CUDA event 或 profiler 测 GPU 段，不能只用未经同步的 Python wall time；
- 同时报告 p50/p95 step time；
- 显存同时报告 allocated 与 reserved；
- 目标峰值 reserved `<30.5GB`，给通信、评测和偶发波动留余量。

**验收门槛**：

- 能把 step time 分为 data / forward / backward / optimizer / other；
- 能说出显存中参数、梯度、optimizer state、activation、bucket 的大致占比；
- 对一个瓶颈获得 `≥10%` 吞吐改进，或以受控实验证明常见优化在该负载上无效；
- trace 留在公司机器内；终端汇总足以支持结论。

### 第 5 周：DDP 正确性与 1/2/4/8 卡扩展

**周目标**：理解 DDP 复制了什么、同步了什么，以及 global batch 改变为何会污染 scaling 结论。

本周不要用 20M TinyFlow 做主 benchmark；使用 300M–500M Transformer/DiT surrogate，使单卡计算量足以暴露真实通信比例。

| 工作日 | 核心任务 | 可选第 2 小时 |
|---|---|---|
| 周一 | 写清 micro-batch、gradient accumulation、world size、global batch 关系 | 验证 DistributedSampler 不重复样本 |
| 周二 | 1 卡 vs 2 卡固定 global batch 正确性；2 卡先用 NV2 `0,1` | 用 NV1 `0,3` 与 SYS `0,5` 做同配置对照 |
| 周三 | 四卡 `0–3` 与 `4–7` 分别做弱扩展，再运行 8 卡 `0–7` | 增加强扩展：固定 global batch |
| 周四 | 用 profiler/Nsight 定位 all-reduce；检查 NCCL 自动 ring/tree 与 exposed communication ratio | 调 bucket 或 accumulation，但不先硬编码算法 |
| 周五 | 写 Scaling Report v1：NV2/NV1/SYS → 4卡组 → 8卡全图 | 对最佳配置复测 3 次并加入 CPU 绑核 A/B |

**必须区分两种效率**：

```text
弱扩展：每卡 micro-batch 固定，global batch 随卡数增长
强扩展：global batch 固定，每卡工作量随卡数下降

效率 E_N = throughput_N / (N × throughput_1)
```

**验收门槛**：

- 1/2/4/8 卡都能完成至少 100 个稳定 step；
- 每个 epoch 样本数正确，无 rank 重复或漏样本；
- 同一 run 内各 rank 更新后参数一致；
- 固定 global batch 的 1 卡与多卡 loss 趋势偏差可解释，前 20 step 目标 `<2%`；
- NV2 pair 应优于同规模 SYS pair；若没有，先排查 GPU 负载、P2P、进程映射和测量方式；
- 两个四卡组结果应大致对称，差异明显时检查 CPU/DataLoader/共享负载；
- 8 卡效率按第 3.3 节新门槛判断；不因小模型低效率误判 NCCL 故障。

### 第 6 周：FSDP、模型状态切分与 checkpoint

**周目标**：亲自验证 DDP 与 FSDP 的显存、通信和 checkpoint 差异。

| 工作日 | 核心任务 | 可选第 2 小时 |
|---|---|---|
| 周一 | 按参数、梯度、optimizer、activation 写 0.6B/1.5B 内存预算 | 加 4B 理论预算，不启动训练 |
| 周二 | 2 卡 Full Shard smoke test；核对 wrap policy 和 mixed precision | 对比 no-shard/DDP |
| 周三 | 比较 8-way Full Shard 与拓扑感知 2×4 Hybrid Shard；记录峰值显存和通信 | 调 activation checkpointing |
| 周四 | 保存 sharded checkpoint；测试同 world size resume | 尝试 8 卡保存、1/4 卡加载的受支持路径 |
| 周五 | 输出 DDP vs Full Shard vs 2×4 Hybrid Shard：显存、吞吐、通信和 checkpoint | 做一次安全的中断后恢复 |

**验收门槛**：

- FSDP 训练 200 step 无 NaN、死锁和 rank divergence；
- 相同模型下，FSDP 显著降低每卡 model-state 内存；若没有，定位 wrap 或配置错误；
- checkpoint 可恢复 optimizer、scheduler、GradScaler、RNG 和 step；
- 能解释为何 FSDP 节省模型状态显存，但不自动消除 activation OOM；
- 能从 trace 中区分 DDP all-reduce 与 FSDP all-gather/reduce-scatter。
- 能解释 2×4 Hybrid Shard 为何通常比 8-way Full Shard 少做跨整个拓扑的参数收集，但只提供约四路而非八路状态切分；
- 若当前 PyTorch/FSDP 版本不支持方便的 2D mesh，记录为环境边界，不为完成练习升级整套生产环境。

### 第 7 周：MultiTask DiT 的 Diffusion vs Flow Matching 对照

**周目标**：把前 6 周训练系统能力放回机器人算法问题中。

| 工作日 | 核心任务 | 可选第 2 小时 |
|---|---|---|
| 周一 | 固定唯一变量：objective 与 sampler；其他模型、数据、global batch、预算一致 | 推导 DDPM/DDIM 与 Flow ODE |
| 周二 | 将 Task-P 适配到 MultiTask DiT；先做 1 卡 20-step dry run | 扫描 FP16 不稳定算子 |
| 周三 | 跑 diffusion 与 flow 两个等预算配置 | 加第二个 seed |
| 周四 | 2/4/8 卡短 DDP，比较两种 objective 的吞吐和通信占比 | 扫采样步数 5/10/20 |
| 周五 | 比较 offline action error、短 rollout success、latency、jerk、显存 | 补第三 seed |

**验收门槛**：

- 两种 objective 都能在 FP16 下稳定收敛；
- global batch、训练样本/step 数和模型容量严格一致；
- 必做 2 个 seed，时间充足再补第 3 个；
- 不用单次最好 loss 宣称算法优劣；
- 能说明采样步数减少为何可能提高实时性，却不必然提高闭环成功率。

### 第 8 周：SmolVLA 450M 的 V100 FP16 微调

**周目标**：完成一次真实 VLA checkpoint 的数据、processor、训练、resume 与 eval 链路。

| 工作日 | 核心任务 | 可选第 2 小时 |
|---|---|---|
| 周一 | 做兼容性审计：BF16 hard-code、FA2、CUDA arch、processor、依赖版本 | 建兼容性表而不是盲目安装 |
| 周二 | 单卡 batch 1/2/4 memory scan；默认冻结 vision encoder | 扫 action horizon |
| 周三 | 运行 200–500 step FP16 微调并 resume | 比较只训 action expert 与解冻顶层 |
| 周四 | 先在四卡组 `0–3` 做 100-step DDP；只有单卡/四卡正确且负载足够大时再跑 8 卡 | 在 `4–7` 复测对称性或比较 gradient accumulation |
| 周五 | base vs fine-tuned 固定评测；按模块 Profile vision/VLM/action head | 写 Model Card v1 |

**兼容原则**：

- 配置显式 `bf16=false, fp16=true`；
- 不安装官方 FlashAttention-2 后强行运行；
- 优先使用 PyTorch 原生 attention/SDPA 的可用后端或项目自带兼容实现；
- 若官方仓库某版本强依赖 Ampere+ kernel，退回已验证 commit 或用同结构 surrogate，不把整周消耗在编译第三方 fork；
- 任何 fallback 都写入 manifest。

**验收门槛**：

- 单卡完成可 resume 的真实 checkpoint 微调；
- FP16 无非有限 loss/grad/param；
- processor、normalization 和 camera order 在 train/eval 完全一致；
- 峰值 reserved `<30.5GB`；
- 至少形成一次 base vs fine-tuned 受控比较；数据不足时标记 `INCONCLUSIVE`，不制造成功率结论。
- 450M 模型若在四卡后已经进入通信主导，不要求用 8 卡长训；8 卡只保留为 topology/profile 样本。

### 第 9 周：Mini-WAM 世界监督消融

**周目标**：回答原项目的核心问题——future latent loss 是有用监督，还是只增加一个容易下降的辅助 loss。

| 工作日 | 核心任务 | 可选第 2 小时 |
|---|---|---|
| 周一 | 冻结视觉 encoder、future offset 和 last-frame baseline | 比较 `t+4/t+8/t+16` |
| 周二 | 增加 20M–100M action-conditioned future latent predictor | 加 world-error 失败分数 |
| 周三 | Run A：action-only；Run B：action + world | 第二 seed |
| 周四 | Run C：shuffled-action world；用 2/4 卡 DDP 跑等预算 | Profile world head 额外成本 |
| 周五 | 比较 future error、action error、ID/OOD proxy、失败相关性 | 做小规模闭环 rollout |

联合目标保持简单：

```text
L_total = L_action + λ_world × L_future
```

**验收门槛**：

- world predictor 相对 last-frame baseline 的 future latent error 目标改善 `≥10%`；
- 正确 action conditioning 必须优于 shuffled-action；
- 加 world loss 后 ID 指标不下降超过 3 个百分点；
- 至少一个 OOD proxy 改善才记录为初步正结果；
- 若不成立但三组消融完整，标记 `FAIL-MODEL`，它仍是有效结果。

本周结果只是小规模研究证据，不替代 Task-D 真机验证，也不直接作为论文结论。

### 第 10 周：FinExec 0.6B 的 evaluator、SFT 与 CPT 预演

**周目标**：用一个可程序验证的金融任务练习 LLM 数据、mask、packing、全参训练和评测，而不是做金融 Agent。

固定输出：

```json
{
  "evidence": ["table:r3c2", "text:p4s1"],
  "program": "divide(subtract(128.4, 97.1), 97.1)",
  "answer": 32.23,
  "scale": "percent"
}
```

| 工作日 | 核心任务 | 可选第 2 小时 |
|---|---|---|
| 周一 | 冻结 FinQA/TAT-QA split、JSON schema 和 deterministic executor | 建六类错误桶 |
| 周二 | evaluator 在 gold program 上达到 100%；检查 scale/unit | 加无效引用和除零测试 |
| 周三 | 0.5B–0.6B 模型单卡 FP16 SFT，检查 label mask/packing | 过拟合 128 样本 |
| 周四 | 2/4/8 卡 DDP 或 FSDP 短训；固定 global batch | 运行 0.5M–2M token CPT rehearsal |
| 周五 | 比较 base、SFT、短 CPT→SFT；输出 error waterfall | 补第二 seed |

**验收门槛**：

- gold evaluator 100% 通过；
- invalid JSON、无效 program、错误 scale 均能被单独统计；
- 训练只对目标 token 计 loss，padding/prompt mask 正确；
- SFT valid JSON rate 目标 `≥90%`；
- 短 CPT 只用于验证流程、遗忘和 checkpoint，不因小 token 预算宣称金融知识提升。

### 第 11 周：模型放大、PEFT/FSDP 与可选 GRPO smoke

**周目标**：观察模型从 0.6B 放大到 1.5B/4B 后，显存、吞吐、通信和后训练稳定性怎样变化。

| 工作日 | 核心任务 | 可选第 2 小时 |
|---|---|---|
| 周一 | 为 1.5B 与 4B 写内存预算、预计 step time 和止损点 | 为 8B 写纸面预算，不训练 |
| 周二 | 1.5B 全参 FSDP 或 4B LoRA/QLoRA 做 50-step smoke | 比较 LoRA rank 8/16 |
| 周三 | 运行 200–500 step SFT；记录 FP16 scale、grad norm、吞吐 | 做 1/2/4/8 卡规模点 |
| 周四 | 在最佳 SFT 上构造分项可验证 reward | 0.6B 上做 50–100 step GRPO smoke |
| 周五 | 比较模型规模、训练方式和收益；决定金融线是否值得继续 | 检查 reward hacking |

**优先级与回退**：

1. 首选：1.5B 全参 FSDP，依赖最少、最适合学习 sharding；
2. 若 bitsandbytes/PEFT 已有经批准且支持 V100 的环境：做 4B QLoRA；计算 dtype 必须为 FP16；
3. 若第三方量化 kernel 不兼容：回退 4B LoRA 或 1.5B 全参，不花整周编译非官方 Volta fork；
4. GRPO 是 stretch task，不挤占 SFT、evaluator 和 Profiling；
5. 不用交易 PnL 作 reward。

**验收门槛**：

- 至少一个放大配置完成可 resume 的 200 step；
- 实测显存与纸面预算差异得到解释；
- 若执行 GRPO，必须分别记录 format、evidence、execution、answer、scale reward，以及 KL、entropy、response length；
- GRPO 出现 valid program 上升但 evidence F1 明显下降时，判为目标错配或 reward hacking；
- 不把 50–100 step smoke 宣称为最终算法收益。

### 第 12 周：训练系统 Capstone、故障恢复与总结

**周目标**：用一个载荷证明你能独立复现、扩展、Profile、恢复并解释完整训练任务。

Capstone 二选一：

- 推荐：`Mini-WAM action-only vs action+world`；
- 备选：`FinExec 0.6B SFT vs short CPT→SFT`。

不要两个都做。

| 工作日 | 核心任务 | 可选第 2 小时 |
|---|---|---|
| 周一 | 冻结代码、数据 hash、config、seed、最大 step、验收问题 | 建 clean output directory |
| 周二 | 单卡跑正确性参考；保存 checkpoint 和终端 profile summary | 重复一次确认方差 |
| 周三 | 先跑四卡 `0–3`，再跑八卡等价性 + scaling；保留本机 trace | 在 `4–7` 复测四卡对称性 |
| 周四 | 在安全边界内中断作业并从 checkpoint 恢复；完成 eval | 若选 FSDP，比较 Full Shard 与 2×4 Hybrid Shard |
| 周五 | 写 6–10 页本机总结与 12 周能力矩阵；决定下一阶段 | 在个人 5070 Ti 规划公开缩小复现 |

**最终验收门槛**：

- 从一个入口命令即可完成 train/resume/eval/profile；
- 每个结果都能对应 config、代码版本、数据 hash、seed、checkpoint 与 eval command；
- 单卡/多卡的数值差异处于预注册容差内或有证据解释；
- 能用 profiler 证据说明主瓶颈是计算、通信、数据、内存还是同步；
- 中断后恢复不会丢失 optimizer、scheduler、scaler、RNG 和 step；
- 最终报告明确区分 `FAIL-MODEL` 与 `FAIL-SYSTEM`；
- 公司机器上的全部原始产物留在公司内部。

---

## 八、统一 FP16 与分布式规范

### 8.1 FP16 必记的观测量

每个 run 至少记录：

- `loss` 与各分项 loss；
- `GradScaler.get_scale()`；
- skipped optimizer step 数；
- clip 前后的 global grad norm；
- NaN/Inf 首次出现 step 与模块；
- learning rate；
- activation/parameter 的极值或分位数抽样；
- peak allocated/reserved memory。

止损条件：

- 连续 3 次 scale 回退并伴随非有限梯度；
- 参数首次出现 NaN/Inf；
- 通过降低 batch 掩盖数值问题却未定位算子；
- FP16 与 FP32 小样本参考的 loss 轨迹出现无法解释的持续偏离。

### 8.2 DDP/FSDP 的正确性先于速度

多卡正式计时前必须依次通过：

```text
单卡 overfit
→ 2 卡 NV2 fixed-batch equivalence
→ 2 卡 NV1/SYS topology control
→ 4 卡 `0–3` smoke
→ 4 卡 `4–7` symmetry check
→ 8 卡 smoke
→ 1/2/4/8 正式 profile
```

禁止一开始直接 `torchrun --nproc_per_node=8` 跑长任务。

### 8.3 每次性能对照必须固定

- 模型结构与参数量；
- 数据样本、sequence/action horizon、分辨率；
- global batch 或明确标注弱扩展；
- precision；
- warmup 与 timed step 数；
- gradient accumulation；
- checkpointing；
- eval 与 logging 频率；
- GPU 时钟/共享负载条件；
- seed 和软件版本。

---

## 九、Profiling 判断树

遇到吞吐低时按下列顺序处理：

1. **进程是否放在了正确的 GPU/CPU 拓扑上？**
   - 先区分 NV2、NV1、SYS；检查 `CUDA_VISIBLE_DEVICES`、rank mapping、CPU affinity 和 NCCL 选择。
2. **GPU 是否在等待数据？**
   - 看 data time、H2D、worker、pin memory、磁盘吞吐。
3. **GPU kernel 是否太碎？**
   - 看大量短 kernel、CPU launch gap、过小 batch/sequence。
4. **Tensor Core 是否真正使用？**
   - 检查 FP16 autocast、GEMM shape、hidden/head dimension 对齐。
5. **backward 是否占绝对多数？**
   - 看 attention/MLP/activation checkpointing 成本。
6. **多卡是否暴露通信？**
   - 看 all-reduce/all-gather/reduce-scatter 与 GPU idle gap。
7. **显存是否被 optimizer 或 activation 主导？**
   - optimizer 主导优先 FSDP；activation 主导优先 checkpointing、batch/sequence 调整。
8. **checkpoint/eval/logging 是否周期性阻塞？**
   - 分开测 steady-state 与端到端训练吞吐。

优化只接受 A/B 对照结果，不接受“某开关理论上更快”。

---

## 十、每周实验规模上限

为了让 1–2 小时/天可持续，统一限制：

| 类型 | 建议规模 |
|---|---|
| dry run | 1–5 step，只查 shape/依赖 |
| smoke test | 20–50 step，只查能否跑通 |
| correctness run | 100–300 step，检查等价/稳定/resume |
| performance run | 20 warmup + 50–100 timed step |
| 小模型算法 run | 1k–5k step，最多两个主配置 |
| 多卡规模点 | 1/2/4/8 卡，每点最多 100–300 step |

每周最多：

- 两个正式主配置；
- 一个关键消融；
- 一个 stretch task；
- 不做无假设的超参数网格。

---

## 十一、最终能力与产物

12 周结束后，应能不依赖他人完成并解释：

### 11.1 训练正确性

- 从数据 schema 到 loss mask；
- FP16 autocast、GradScaler、clip 与 overflow；
- checkpoint/resume 与 RNG；
- overfit、固定 batch、seed 和 split；
- 算法失败与系统失败的区分。

### 11.2 分布式训练

- DDP 的数据切分、global batch 与 all-reduce；
- FSDP 的参数 all-gather、梯度 reduce-scatter、optimizer sharding；
- 1/2/4/8 卡数值等价和扩展效率；
- topology、NCCL、bucket 与 overlap；
- sharded checkpoint 与恢复。

### 11.3 Profiling

- PyTorch Profiler 的 op、shape、memory 表；
- Nsight Systems 的 CPU/CUDA/NCCL 时间线；
- Nsight Compute 的单 kernel 诊断；
- step time、吞吐、显存和通信模型；
- 用 A/B 测试验证优化。

### 11.4 领域载荷

- Dex-WAM：TinyFlow → MultiTask DiT → SmolVLA → Mini-WAM 消融；
- FinExec：evidence/program schema → evaluator → 0.6B SFT/CPT → 可选 1.5B/4B/GRPO smoke；
- Task-P 与 Task-D 的接口边界；
- 为什么 V100 结果不能替代前沿 H100 训练或真实机器人验证。

### 11.5 可公开材料的处理

公司机器内形成的是个人能力证据，不自动等于可公开作品集。若用于求职：

1. 只在个人 5070 Ti 上使用公开代码和公开数据；
2. 重跑一个缩小版 TinyFlow/Mini-WAM 或 FinExec 0.6B；
3. 重新生成公开日志、图表和报告；
4. 不引用、搬运或暗示公司 V100 的内部结果。

---

## 十二、明确不做

这 12 周不做：

- BF16、TF32、FP8 训练；
- CUDA 13 上为 V100 强行编译新 kernel；
- 官方 FlashAttention-2/3/4 的 V100 强行适配；
- π0.5、LingBot-VA 5B 的强行多卡部署；
- 8B/27B 的长程全参 CPT；
- 金融 Agent、RAG、交易系统和 PnL reward；
- 未授权的 Task-D 或公司数据迁移；
- 从公司机器带出日志、trace、checkpoint 或性能数据；
- 同时维护多个 Task-P、多个 VLA checkpoint 或大规模网格搜索；
- 只看 GPU utilization、只看 loss、只看单次最好结果。

---

## 十三、12 周后的决策门

### 继续深入训练系统/大模型工程，如果

- 能独立定位至少一次真实通信或数值瓶颈；
- 能稳定完成 DDP/FSDP checkpoint/resume；
- 能将一个优化转化为可复现实验，而不是经验判断；
- 对 V100 与 H100 的差异已经能落到 kernel、精度、带宽和显存模型。

### 继续投入 Dex-WAM，如果

- action+world 相对 action-only 在至少一个 OOD proxy 上有稳定信号；
- shuffled-action 对照证明模型确实使用动作条件；
- Task-D 能合法获得数据并定义自动/半自动成功判据。

### 继续投入 FinExec，如果

- evaluator 稳定且能捕获 evidence/program/scale 错误；
- SFT/CPT/可选 GRPO 的贡献能被单独归因；
- 金融方向不会挤占机器人 WAM 主线。

如果以上算法信号不足，但训练系统闭环完整，这 12 周仍然成功：当前计划的首要验收对象是你的训练 ownership，而不是制造一个漂亮的 benchmark 数字。

---

## 十四、主要参考资料

- [NVIDIA CUDA 13.0 Release Notes：Volta 的离线编译与库支持已从 CUDA 13 移除](https://docs.nvidia.com/cuda/archive/13.0.3/cuda-toolkit-release-notes/index.html)
- [NVIDIA Volta Tuning Guide](https://docs.nvidia.com/cuda/volta-tuning-guide/)
- [PyTorch Automatic Mixed Precision](https://docs.pytorch.org/docs/stable/amp.html)
- [PyTorch AMP Examples](https://docs.pytorch.org/docs/stable/notes/amp_examples.html)
- [PyTorch DistributedDataParallel](https://docs.pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html)
- [PyTorch FullyShardedDataParallel](https://docs.pytorch.org/docs/stable/fsdp.html)
- [PyTorch Profiler Recipe](https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html)
- [NVIDIA NCCL Tests](https://github.com/NVIDIA/nccl-tests)
- [NVIDIA Nsight Systems User Guide](https://docs.nvidia.com/nsight-systems/UserGuide/index.html)
- [NVIDIA Nsight Compute CLI](https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html)
- [FlashAttention 官方仓库：FA2 CUDA 支持从 Ampere 开始](https://github.com/Dao-AILab/flash-attention)
- [LeRobot SmolVLA](https://huggingface.co/blog/smolvla)
- [LeRobot MultiTask DiT](https://huggingface.co/docs/lerobot/multi_task_dit)
- [LeRobot VLA-JEPA](https://huggingface.co/docs/lerobot/main/vla_jepa)
- [FinQA](https://github.com/czyssrs/FinQA)
- [TAT-QA](https://github.com/NExTplusplus/TAT-QA)

---

## 十五、一句话执行原则

> 用单卡证明数值与数据正确，用 2 卡证明分布式等价，用 4/8 卡学习扩展与通信；V100 上追求的是可解释、可恢复、可复现的训练能力，而不是用旧架构硬追 H100 上的模型规模和吞吐。
