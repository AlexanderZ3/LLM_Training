# Week M03 — MiniMind 纯 V100 全链路训练系统（一页周卡）

> 状态：`BUILT / RUNTIME-UNVERIFIED`（本机静态与 CPU 门通过；**所有 GPU 结论待公司 8×V100**）
> 生成日期：2026-09-08 · 生成方：cc · 轨道：`track_minimind`
> 来源：用户 2026-09-08 输入（`已确认`）+ 2026-09-04 V100/torch 2.1 审计 + 2026-09-08 量化/RL/内网审计

## 0. 这一周和前两周的关系

**week01（5070 Ti）与 week02（8×V100）的两步走计划作废。** 用户 2026-09-08 起不再有 5070 Ti，唯一 GPU 环境是公司内网 8×V100。本周**自包含**：不引用、不 import、不依赖前两周的任何产物或 Gate 状态，代码是从那两周已通过 CPU 单测的实现里**复制并裁剪**进来的（来源在 `02_LAB_GUIDE.md` 第 0 节逐文件标注）。

前两周的目录保留作为历史记录，不再维护，也不作为本周的先修。

## 1. 本周唯一主问题

在 8×V100（sm70：有 FP16 Tensor Core，**无 BF16、无 FlashAttention-2、无 INT8 Tensor Core**）上，同一个 64M MiniMind 从一条 JSONL 走到 GRPO 的全过程里，三本账各是多少：

| 账 | 问的是什么 |
| --- | --- |
| **字节账** | 参数 / 梯度 / 优化器状态 / 激活，每张卡各占多少字节 |
| **误差账** | fp16 相对 fp32 从哪一步开始偏、偏多少、偏的方向 |
| **通信账** | 一个 step 内几次集合通信、什么类型、每次多大 |

哪一本账决定了它**能不能跑**、**跑多快**、以及**学习信号还在不在**？

这三本账是可迁移的：换到 A100/H100 上数字会变，记账方法不变。这正是本周不追求"在 V100 上跑出快结果"的原因——V100 给不了的是速度，给得了的是**把每一个字节和每一次通信都数清楚**的训练。

## 2. 核心实现 · 关键故障练习 · Gate

| 项 | 内容 |
| --- | --- |
| 核心实现 | `lab/src/mm_v100/`：三本账的仪表（字节账 hook、fp32/fp16 双路参考、collectives 计数器）+ 三档静默故障注入 + DDP/FSDP 统一入口。零新依赖，只 `torch` / `numpy` / 标准库 |
| 关键故障练习 | **「loss 曲线看起来正常，但学习信号已经断了」的三档判别**：(A) label mask 错位、(B) fp16 GradScaler 持续跳步、(C) DDP 梯度未同步。三档共用同一个脚本、同一个 seed、同一份数据，**只凭日志的三个不变量**把它们区分开 |
| Gate | 见第 6 节 |

## 3. 硬事实（决定本周所有命令）

### 3.1 V100 + torch 2.1（`已确认`，2026-09-04 审计，本周沿用）

- **BF16 在 V100 + torch 2.1 直接报错**：`torch.cuda.is_bf16_supported()` 要求 major ≥ 8；`autocast(dtype=bfloat16)` 抛 `RuntimeError`。MiniMind 全部 9 个 trainer 默认 `--dtype bfloat16`，**本周一律 `--dtype float16`**；`train_grpo.py` 没有 GradScaler，lab 里补上。
- **SDPA 后端**：torch 2.1 的 flash 后端只覆盖 sm75–sm90，V100（sm70）走 mem-efficient 或 math，**没有 FlashAttention-2**。
- **torch 2.1 FSDP API 齐全**：`ShardingStrategy`、`MixedPrecision`、`StateDictType` + `state_dict_type()`、`optim_state_dict`、`torch.distributed.checkpoint`、`all_to_all_single`。
- **不用 DeepEP**（要 SM90）、**不用 MegaBlocks**（要 torch 2.7.x）。
- **`torch.compile` 本周关闭**：Triton 2.1 声称 CC ≥ 7.0，V100 满足但未实测，属纯风险项。
- MiniMind 锁定 commit `7a6fddd63a30c06b2fdd5fac4089922b29bc841b`（Apache-2.0）。
- 公司机器的 torch 2.1.0 wheel 的 CUDA 版本、`get_arch_list()` 是否含 sm_70、NCCL 版本、拓扑、transformers 版本：全部 `未知`，Day 0 探针回答。

### 3.2 量化在 V100 上的边界（`已确认`，2026-09-08 联网核验）

完整事实表与来源 URL 见 `06_QUANT_LOWBIT.md` 第 1 节。一句话总结：**8-bit 里"省显存"的部分在 V100 上是真的，"提速"的部分是假的。**

| 在 V100 上 | 能不能 | 依据 |
| --- | --- | --- |
| INT8 Tensor Core 矩阵乘 | **不能** | IMMA 指令从 Turing sm75 起；Volta 的 Tensor Core 只接受 FP16 输入 |
| `bitsandbytes` 的 `LLM.int8()` / `load_in_8bit=True` | **不能** | 官方门槛 CC ≥ 7.5，V100 是 7.0；代码里 `supports_igemmlt()` 直接判 `< (7,5)` 返回 False |
| `torch.ao.quantization` 的 `convert_fx` 产物 | **不能上 GPU** | fbgemm / qnnpack / x86 / onednn 全是 CPU 后端 |
| `torchao` | **完全不能** | 最低 torch 2.5 |
| **fake-quant QAT 训练** | **能，而且是真 CUDA kernel** | torch 2.1 的 `FakeQuantizeCore.cu` 与 `FusedObsFakeQuant.cu` 提供 per-tensor / per-channel 的前向与反向 |
| **8-bit 优化器（`AdamW8bit`）** | **能** | 官方门槛 CC ≥ 6.0。这是本周唯一能拿到真实 8-bit kernel 且真实省显存的地方 |
| **NF4 / FP4（QLoRA）** | **能** | 官方门槛 CC ≥ 6.0。但 `bnb_4bit_compute_dtype` 只能是 `float16`，不能照抄文档里的 `bfloat16` |
| `torch._int_mm` | **未知，必须探针** | 2.1 里存在且**没有 CC 检查**，行为取决于 cuBLASLt；且有已知正确性缺陷（pytorch#107671），即使不报错也要与 fp32 参考比对 |

所以量化模块把量化还原成它本来的两件事：**一个误差预算问题 + 一个内存带宽问题**。这两件事在 V100 上完全真实、可精确测量、结论可迁移到 A100/H100。核心实现全部用纯 PyTorch 自己写（`bitsandbytes` 只作可选对照），这既绕开内网装编译型包的风险，也把交付等级从 A3 拉到 A1——自己写过一遍的量化器和调用过一次 API 的量化器，不是一回事。

### 3.3 RL 在 V100 上的边界（`已确认`，2026-09-08 联网核验）

完整事实表见 `07_RL_LOWPRECISION.md` 第 1 节。

- **不装 vLLM / verl / OpenRLHF。** vLLM 每个版本都硬钉一个精确 torch 版本（连 0.2.7 都要 `torch==2.1.2` 而非 2.1.0），且官方最低 CC 7.5；verl 要 CUDA ≥12.8；OpenRLHF 绑 vllm 0.27.1。**这三个的共同点值得记住：现代 RL 框架都把推理引擎当硬依赖，而推理引擎是整个生态里对 GPU 架构和 torch 版本最挑剔的一环。**
- `trl` 可用区间 `0.14.0`（`GRPOTrainer` 起点）到 `0.22.2`，配 `transformers 4.55.x`。`GRPOConfig.use_vllm` 默认就是 `False`，走 transformers 原生 `generate` 是官方支持的路径，不是将就。
- **fp16 未必是 RL 的劣势。** torch AMP 文档警告 bf16 预训练模型在 fp16 下会梯度**上溢**；但另一方面有 2025 年的工作（arXiv:2510.26788）主张 RL 场景下 fp16 比 bf16 更稳，因为 bf16 的舍入误差破坏了训练与推理的一致性。**这是活跃的开放问题，本周把它当作一个可证伪的 hypothesis（R 模块），不当结论。**

### 3.4 内网（`已确认`，用户 2026-09-08 输入 + 2026-09-08 联网核验）

`pip install` 与 `git clone` 可用，但**有依赖的包容易失败**；**权重必须外网下好再拷进去**；服务器只进不出。完整应对见 `08_INTRANET_SETUP.md`。要点：

- **主线 Day 0–5 不需要装任何新包。** 两个扩展模块的核心路径也不需要，只有可选对照项才用到 `bitsandbytes==0.45.5`。
- 内网装包失败几乎都是同一个原因：**某个包为了满足依赖把 torch 2.1.0 换掉了。** 对策是 constraints 文件钉死 torch，让冲突在解析阶段可见地失败，而不是 `--no-deps` 静默留下坏环境。
- 关键版本上限：`transformers < 4.56`、`bitsandbytes <= 0.45.5`、`trl <= 0.22.2`、**`numpy < 2`**（torch ≤2.1 的 `.numpy()` 在 numpy 2.x 下失效）。

## 4. 逐日地图

| Day | 目标 | 主要产物 | 门 | 估时 |
| --- | --- | --- | --- | --- |
| 0 | 把 V100 真实能力、MiniMind 对 torch 2.1 的兼容性、拷进来的数据、`torchrun` 能不能起来，四件事一次钉死 | `$MM_RUNS_ROOT/day0_probe/evidence.json` + 兼容审计表 | 静态合同 / shape 检查 | 100 |
| 1 | 在单卡上把数据契约和 fp32↔fp16 的误差账同时立住 | `day1_pretrain/numeric_ref.json` | CPU 单测 → 单 batch 前反向 → 128 样本 overfit → FP32 ref → FP16 AMP | 110 |
| 2 | 让 2 卡 / 8 卡在固定全局批下走出同一条曲线，然后把它**弄坏三次**并只凭日志区分开 | `day2_sft/equiv_table.json` + **一张手写三行判别表** | 2 卡等价 → 8 卡等价 → 故障注入 | 120 |
| 3 | **先闭卷手算**每卡四项字节账，再用 FSDP 实测打出来，最后 8 卡分片 checkpoint 换 4 卡恢复 | `day3_dist/byte_ledger.md`（手算列 + 实测列 + 偏差列） | 有界训练 → checkpoint/resume 等价 | 120 |
| 4 | 有界 pretrain → SFT，导出 Day 5 的输入，并做**独立 eval**（不是看 loss） | `day4_profile/eval_compare.jsonl` + `full_sft_768.pth` | 有界训练 → 独立 eval | 110 |
| 5 | 低精度下走完 DPO 与 GRPO 的数值链，kill 一个 rank 从分片 checkpoint 恢复 | `day5_rl/rl_numeric.json` | smoke → eval → 恢复/故障测试 | 120 |

**核心 6 天：680 分钟（11 h 20 m）+ Gate 60 分钟 = 740 分钟（12 h 20 m）。**

**排期（用户 2026-09-09 确认：每周 5 次、每次 3 小时 = 15 小时）**：每次 180 分钟，比最长的一张卡（120 分钟）还多 60 分钟，所以**一次一张卡，走满配，不需要降级**。多出来的 60 分钟不是余量浪费——它正好留给内网首次运行的摩擦：排队、装包失败、路径没对齐、第一次 `torchrun` 起不来。

| 第几次 | 内容 | 分钟 | 余量 |
| --- | --- | --- | --- |
| 1 | Day 0 | 100 | 80 |
| 2 | Day 1 | 110 | 70 |
| 3 | Day 2 | 120 | 60 |
| 4 | Day 3 | 120 | 60 |
| 5 | Day 4 | 110 | 70 |
| 6 | Day 5 + Gate | 120 + 60 | 0 |

**六次做完**：本周 5 次跑 Day 0–Day 4，下周第 1 次跑 Day 5 + Gate。第 6 次排得满，那天不要再安排别的。

降级配（515 分钟 + Gate）作为**预案**保留：某一天被排队或故障吃掉时间时，按各张卡的「可裁剪项」砍，砍完 Gate 四项仍成立。不可裁项见下。

**绝对不可裁**（裁了 Gate 不成立）：Day 0 的 wiring smoke、Day 2 的三档故障注入、Day 3 的闭卷手算字节账、Day 3 的 8→4 reshard、Day 4 的 checkpoint 导出、Day 5 的 kill-rank 恢复。

## 5. 两个扩展模块（Gate 之后，不进 Gate）

用户 2026-09-08 明确的下一步方向。**本周交付它们的完整规格 + 可运行 lab 代码 + CPU 单测；实验执行排在 Gate 通过之后**，因为两者合计 17 小时，超过主线本身，塞进本周等于把三周内容压成一周。

| 模块 | 文档 | 代码 | 实验 | 估时 |
| --- | --- | --- | --- | --- |
| 超大模型 8-bit 量化 / QAT | `06_QUANT_LOWBIT.md` | `lab/src/mm_quant/` | Q1–Q6 | 480 分钟 |
| 低精度 + RL 后训练 | `07_RL_LOWPRECISION.md` | `lab/src/mm_rl/` | R1–R6 | 540 分钟 |

两个模块的交点是 **R5：把冻结的 ref 模型量化到 INT8**——它同时是显存问题和 KL 偏差问题，是"低精度 + RL 一起训练"这条路上第一个真正要做的判断。

## 6. Gate

| 项 | 要求 |
| --- | --- |
| A0 闭卷 30 分钟 | 给定 `hidden=768, layers=8, heads=8, kv_heads=4, vocab=6400, tie_embeddings=True, B, T, world_size=8`，手写 DDP 与 FSDP `FULL_SHARD` 下**每卡**四项字节账，并写出一个 step 内的集合通信序列（类型 + 次数 + 单次张量字节量） |
| A1 45 分钟（只查官方文档） | 讲完三档静默故障各自的「不变量 → 最小检查 → 判据」，每档一条命令；再解释「为什么 V100 上 INT8 权重量化省显存但不省时间」 |
| 跨场景 debug 20 分钟 | 考官给一份只保留数字列的日志，限时反推是哪一档故障，并说出哪一对特征是**充分的**、哪一对只是**必要的** |
| 证据 | 两个不同日期：Day 0–2 至少一条，Day 3–5 至少一条 |

**Gate 明确不依赖**：MoE/EP、量化模块 Q、RL 模块 R、GRPO 是否有正向结果。这些都是加分。

## 7. 先修

**无。本周自包含。** 需要的只有：

1. 公司 8×V100 的登录与排队权限；
2. MiniMind 已 `git clone` 到 V100（锁定 commit 见 3.1）；
3. 数据已按 `datasets/README.md` 第 2 节拷到位——Day 0 第一步就验证这件事。

## 8. 三类环境边界

| 环境 | 本周用途 | 限制 |
| --- | --- | --- |
| 公司 8×V100 32 GB（内网） | **唯一执行环境** | 不新建 conda、不升级 torch 2.1、FP16 不 BF16；每次运行 ≤ 10 分钟；日志、trace、checkpoint、拓扑与性能数字**留在公司内**，只带出证据字段里的抽象值（比值、布尔、计数） |
| 本机 Windows（conda `rfm`） | 写代码、CPU 单测、静态检查 | **无 GPU**。这是**开环**的一端：这里写的 GPU 代码一行都没在 GPU 上跑过 |
| 个人 5070 Ti | **不再可用**（2026-09-08） | — |
| 租用 H100 | 本周不使用 | — |

## 9. 本机验证状态

构建完成日：2026-09-08。本机（Windows 11，conda `rfm`，Python 3.11.16，torch 2.14.0+cpu，**无 GPU**）实测：

| 检查 | 结果 |
| --- | --- |
| `pytest lab/tests -q` | **`368 passed, 6 skipped, 1 warning`**，退出码 0 |
| 结构校验 `validate_cc_week.ps1 -Week 03 -Track track_minimind -IdPrefix M` | **`PASS`（0 error 0 warning，口试 27 题 / 27 答）** |
| `compileall lab/src lab/scripts` | 退出码 0 |
| Day 1 / 2 / 3 的 `run_day.py` 编排 | 本机 CPU 实跑通过（等价 `max|delta|=0.000e+00`；故障判别表三档各自命中唯一不变量） |
| Gate A0 题的答案链 | 独立复算一致（`d_ff=3264` → 每层 12,650,624 → 160,197,120） |
| 路径合同解析与错误信息 | 通过（四个根目录、清单加载、缺文件/错字节的报错文案） |
| `check_data_layout.py` 四种状态 | 通过（`OK` / `MISSING` / `SIZE-MISMATCH` / `SHA-MISMATCH` 各构造一次） |
| `check_weights_layout.py` 两种内网失败 | 通过（分片索引引用了不存在的分片、`auto_map` 的远程代码 `.py` 缺失） |
| 参数量手算链 | 通过（`d_ff=2432`、`P_ℓ=7,374,528`、`P=63,912,192` 独立复算一致） |

**6 个跳过的用例全部是 `skipif(not torch.cuda.is_available())`**，逐条原因：

| 用例 | 为什么本机验不了 |
| --- | --- |
| `test_quant_qat.py` ×2 | fake-quant 的 CUDA kernel 与 AMP 的真实行为 |
| `test_reshard_cpu.py` ×1 | FSDP `SHARDED_STATE_DICT` 需要 CUDA |
| `test_rl_scaler_rules.py` ×2 | torch 的 `GradScaler` 只有 CUDA 实现；无 GPU 时 `enabled=False`、scale 恒为 1，测不到溢出/backoff/跳步 |
| `test_rl_ref_model.py` ×1 | 显存与 fp16 计算路径的结论 |

那 1 条 warning 来自 `Adam8bit` **故意**拒绝稀疏梯度的那个测试，是 torch 的 sparse invariant 提示，不是缺陷。

**仍未验证（必须在公司 8×V100 上执行）**：NCCL 通信与 `torchrun` 启动、FP16 与 GradScaler 的真实数值行为、FSDP 训练与 reshard 全链路、显存与吞吐数字、SDPA 后端实际命中、量化与 RL 模块的所有 GPU 侧结论。文档里标 `估算` 的数字全部属于此类。

**开环交付的固有风险**：本周所有 GPU 代码在 cc 这边一行都没跑过。**Day 0 的 wiring smoke（`world_size` 1 → 2 → 8，各 2 步）是唯一的早期闸门。** 跳过它，风险会一路累积到 Day 3 才爆发。

## 10. 已知限制与风险

1. **时间预算未对齐**：满配 12.3 h，用户每周可投入小时数从未向 cc 确认。默认按降级配 9.6 h 排，确认后再调。
2. **数据许可**：数据集卡片标了 `cc-by-nc-2.0`（**非商用**）。前两周在个人机器上无碍；本周把这批数据放到**公司机器**上，这条边界是否被公司用途触及，**需要用户按公司政策自行判断**。cc 只标出这一条，不做法律判断。
3. **`zero_std_group_ratio` 可能高到让 GRPO 无结论**：本周用规则 reward，值域只有 `[-3, 3]`，分辨率更差。Day 5 预注册 `INCONCLUSIVE` 出口，Gate 不依赖 GRPO 有正向结果。
4. **8 卡是否独占未知**：非独占时 Day 3 的所有 `step_time_ratio` 标为 `measurement` 类不确定，只保留 `mem_ratio` 与 `collectives_total` 作为结论依据。
5. **MoE / 专家并行整体移出主线**：两天的量塞不进 6 天，且用户 2026-09-08 的新优先级是量化与 RL。降级为可选的「读 `ep_moe.py` + 跑 CPU 单测」，代码保留在 week02 目录。
6. **profiling / `torch.compile` 删除**：torch 2.1 + Volta 上 Triton 未实测，属纯风险项。本周用 collectives 计数与比值代替 profiler，不损失结论。

## 11. 来源

- 用户 2026-09-08 输入（`00_INPUT.md` 归档区）
- PyTorch v2.1.0 源码与文档、NVIDIA Volta/Turing 架构白皮书、各库官方文档——逐条 URL 与核验日期见 `06_QUANT_LOWBIT.md` 第 1 节、`07_RL_LOWPRECISION.md` 第 1 节、`08_INTRANET_SETUP.md` 第 1 节
- MiniMind 仓库（Apache-2.0），锁定 commit 见 3.1
