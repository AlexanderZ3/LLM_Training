# 扩展模块 Q — 超大模型 8-bit 量化与 QAT（V100 版）

> 生成日期：2026-09-08 · 生成方：cc · 状态：`BUILT / RUNTIME-UNVERIFIED`
> 定位：**Gate 之后**的独立模块，不进 Gate。6 个实验共约 480 分钟，建议拆 4 个 session。
> 前置：主线 Day 1 的 `numeric_ref.json`（误差基线）、Day 3 的 `byte_ledger.md`（显存基线）、Day 4 的 `full_sft_768.pth`（被量化对象）。

## 0. 这个模块的立论

你说接下来要做超大模型的 8-bit 量化。那么第一件要面对的事是：

> **V100 没有 INT8 Tensor Core。任何以"提速"为卖点的 8-bit 实验，在这台机器上都是玩具。**

这听起来像坏消息，其实不是。因为量化本来就是两件事，而"提速"只是其中一件的下游结果：

| 量化真正是什么 | 在 V100 上能不能测 |
| --- | --- |
| **一个误差预算问题**：8 bit 的分辨率有限，把它花在哪里 | **能，完全真实** |
| **一个内存带宽问题**：权重字节减半，搬运减半 | **能，完全真实** |
| 一个 TOPS 问题：INT8 矩阵乘比 FP16 快几倍 | 不能，硬件没有 |

前两件事的结论**可以直接迁移到 A100/H100**——误差预算是数学，带宽账是字节数，都不依赖 Tensor Core。第三件事在任何有 IMMA 的卡上都是"装上库、跑一下、看数字"，反而是最不需要训练的部分。

所以这个模块不教你调 `load_in_8bit=True`。它教你**自己写一遍量化器**，然后回答：这 8 个 bit 花在哪里最值，以及怎么知道自己花错了。

## 1. 事实表（`已确认`，2026-09-08 联网核验）

### 1.1 V100 上什么能跑什么不能

| 能力 | V100（sm_70）| 依据 | 来源 |
| --- | --- | --- | --- |
| INT8 Tensor Core（IMMA） | **没有** | Volta Tuning Guide：Tensor Core "matrix multiply inputs A and B are FP16 matrices"，全篇无 INT8。Turing Tuning Guide §1.4.2："Turing adds acceleration for integer matrix multiply operations"——**INT8 Tensor Core 是 sm_75 起的能力** | [Volta](https://docs.nvidia.com/cuda/volta-tuning-guide/index.html) · [Turing](https://docs.nvidia.com/cuda/turing-tuning-guide/index.html) |
| `bitsandbytes` `LLM.int8()` | **不能** | 官方安装文档表格：LLM.int8() 需 **CC 7.5+（Turing）**。代码 `supports_igemmlt()`：`if get_device_capability() < (7,5): return False` | [bnb 安装](https://huggingface.co/docs/bitsandbytes/main/en/installation) |
| `bitsandbytes` **8-bit 优化器** | **能** | 同一张表：8-bit optimizers/quantization 需 **CC 6.0+（Pascal）** | 同上 |
| `bitsandbytes` **NF4 / FP4（QLoRA）** | **能** | 同一张表：NF4/FP4 需 **CC 6.0+** | 同上 |
| `torch.ao.quantization` 的真 INT8 后端 | **只有 CPU** | torch 2.1 量化文档：后端为 x86/fbgemm/onednn（服务器 CPU）、qnnpack/xnnpack（移动 CPU）；GPU 一栏只有 "(early prototype) ... via TensorRT through fx2trt"。FAQ："We don't have official GPU support yet" | [torch 2.1 quantization](https://docs.pytorch.org/docs/2.1/quantization.html) |
| **fake-quant 的 CUDA kernel** | **有，前向反向都有** | v2.1.0 源码 `aten/src/ATen/native/quantized/cuda/FakeQuantizeCore.cu` 定义 `fake_quantize_tensor_cachemask_kernel_cuda`、`fake_quant_per_channel_cachemask_cuda`、`_fake_quantize_grad_learnable_*_cuda`；`FusedObsFakeQuant.cu` 实现 `fused_moving_avg_obs_fake_quant_cuda` | [FakeQuantizeCore.cu](https://github.com/pytorch/pytorch/blob/v2.1.0/aten/src/ATen/native/quantized/cuda/FakeQuantizeCore.cu) · [FusedObsFakeQuant.cu](https://github.com/pytorch/pytorch/blob/v2.1.0/aten/src/ATen/native/quantized/cuda/FusedObsFakeQuant.cu) |
| `torch._int_mm` | **存在，但行为未知** | v2.1.0 `Blas.cpp` 有 `_int_mm_out_cuda`，守卫是 `CUDA_VERSION >= 11070`，走 `cublasLtMatmul` + `CUDA_R_8I`。**全函数没有任何 compute capability 的 `TORCH_CHECK`**，V100 上的行为完全取决于 cuBLASLt 运行时 | [Blas.cpp](https://raw.githubusercontent.com/pytorch/pytorch/v2.1.0/aten/src/ATen/native/cuda/Blas.cpp) |
| `torch._int_mm` 的正确性 | **有已知缺陷** | pytorch#107671："`torch._int_mm` may yield wrong results starting cuda 12.1 update 1"，A100 上以 (m=17,k=16,n=16) 复现，milestone 挂在 2.1.0。**torch 2.1.0 的 cu121 wheel 正在这个区间** | [#107671](https://github.com/pytorch/pytorch/issues/107671) |
| `torch._weight_int8pack_mm` | **2.1 里不存在** | 其 CPU kernel 文件在 tag v2.2.0 是 404，v2.3.0 才有。且是 CPU/MPS 侧算子 | [v2.3.0](https://github.com/pytorch/pytorch/blob/v2.3.0/aten/src/ATen/native/cpu/int8mm_kernel.cpp) |
| `torchao` | **完全不可用** | 官方兼容表最老一行是 `torchao 0.12.0 → torch 2.7.1/2.6.0/2.5.0`。最低 torch 2.5 | [pytorch/ao#2919](https://github.com/pytorch/ao/issues/2919) |
| `__dp4a` 在 sm_70 | **未知** | 抓了 CUDA Programming Guide 的 compute-capabilities 附录与 PTX ISA 目录，文档被拆分/截断，没取到 target 说明；检索只找到二手论坛帖。**必须实测，不要在文档里写死** | — |

### 1.2 版本约束

`bitsandbytes` 与 torch 2.1 相容的**最高版本是 `0.45.5`**（0.46.0 起 `torch>=2.2`，0.48.0 起 `torch>=2.4`）。来源：[PyPI 0.45.5](https://pypi.org/pypi/bitsandbytes/0.45.5/json) · [0.46.0](https://pypi.org/pypi/bitsandbytes/0.46.0/json)。

`0.45.5` 的 wheel 是否含 sm_70 cubin：`推断`。当前文档的构建矩阵列出 `sm60, sm70, sm75, sm80, sm86, sm89, sm90`，但未在 0.45.5 的 CMakeLists 里逐字核对。**Q0 探针会回答。**

### 1.3 一句话结论

在 8×V100 + torch 2.1 上做"超大模型 8-bit QAT"：

- **能真跑的**：fake-quant 伪量化训练（权重仍是 fp16/fp32，前向插 quant-dequant，反向走 STE，CUDA kernel 齐全）；observer 统计与逐层敏感度分析；`bnb` 的 8-bit 优化器状态；NF4 4-bit 权重 + QLoRA。
- **只能模拟的**：任何 INT8 Tensor Core GEMM；`LLM.int8()`；`convert_fx` 的产物上 GPU；`torchao` 全部。

**所以本模块把"8-bit QAT"明确定位为：fake-quant 训练 + 误差/敏感度分析 + 真实的显存账。加速收益只能来自优化器状态与 4-bit 权重，不来自 INT8 矩阵乘。**

## 2. Q0 — 能力探针（先跑这个，30 分钟）

在做任何实验之前，把上表所有 `未知` 一次问完。

```bash
python lab/scripts/probe_int8_caps.py --json > $MM_RUNS_ROOT/ext_qat/int8_caps.json
```

`可直接执行`。它逐项 try/except，**任何一项失败都不会让脚本非零退出**——它是信息收集器，不是门。会回答：

| 字段 | 含义 |
| --- | --- |
| `int_mm_callable` | `torch._int_mm` 能不能调 |
| `int_mm_matches_fp32` | 调通了的话，结果和 fp32 参考**相不相等**（呼应 #107671，不能默认它对） |
| `fakequant_cuda_fwd_bwd` | `FakeQuantize` 与 `FusedMovingAvgObsFakeQuantize` 在 CUDA 上前反向是否可用 |
| `convert_fx_cuda_fails_as_expected` | `convert_fx` 产物 `.cuda()` 是否如预期失败（**这个"失败"本身是证据**） |
| `bnb_version` / `bnb_adamw8bit` / `bnb_nf4` / `bnb_int8` | bnb 装没装上，三类功能各自可用与否 |
| `arch_list_has_sm70` / `torch_cuda_version` | wheel 是 cu118 还是 cu121（决定 #107671 是否适用） |

**这份 JSON 是后面每个实验的前提。** 如果 `bnb_adamw8bit` 是 false，Q3 就只跑自写实现，不做对照——那不是失败，是把"内网装不上编译型包"这件事记录成了事实。

## 3. 六个实验

每个实验都按 `CLAUDE.md` 第 13 节的实验合同写：hypothesis、controlled variables、metric、stop condition、decision rule。

### Q1 — 误差粒度分解（CPU，45 分钟）

**Hypothesis**：每个 Linear 的 per-tensor 对称 INT8 误差主要由少数 outlier 通道的 clipping 贡献。换成 per-channel（axis = 输出通道）后相对误差下降一个数量级以上；per-group(g=128) 再降，但收益递减。

**Controlled variables**：同一份权重、同一个对称量化器、只变粒度。

**Metric**：每层 `rel_err = ‖W−Q(W)‖_F / ‖W‖_F` × 三种粒度；每层 outlier 指标 `max|w| / mean|w|`；per-group 的 scale 存储开销占比。

**Stop condition**：拿到全部 L 层三种粒度的表。

**Decision rule**：能指出 `rel_err` 与 outlier 比值的秩相关方向为正 → PASS。**per-channel 没降 → 第一个检查点是 `axis` 是不是对应输出通道**，这是最常见的实现错误，而且它不会报错，只会让你得到一个"per-channel 没用"的错误结论。

### Q2 — clipping vs rounding 的误差预算（CPU，45 分钟）

**Hypothesis（已被本机实测修正，见下）**：截断阈值存在一个最优 `α·max`，`α ≤ 1`；**位宽越低，最优 α 越小**。

**Metric**：扫 `α ∈ {0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0}` 的 `MSE(α)`，在 **8 / 4 / 3 bit 三个位宽**各跑一条；把 MSE 分解成 **clipping 项**与 **rounding 项**两列。

**本机实测结果（`已确认`，2026-09-08，CPU）**——这条和最初的假设不一样，要按实测的来：

| 位宽 | 最优 α | 相对 α=1.0 的 MSE 改善 |
| --- | --- | --- |
| 8 bit | **1.0（没有收益）** | 1.00× |
| 4 bit | ≈0.7 | 1.55× |
| 3 bit | ≈0.6 | 2.29× |

**所以"8-bit 权重量化要调截断阈值"这个说法，在这个模型的权重分布上是站不住的。** 8 bit 给了 256 个格子，权重分布又不像激活那样长尾，rounding 项一直压着 clipping 项，最优解就是不截断。收益要到 4 bit 以下才出现——那时格子少到 rounding 项主导，牺牲少数 outlier 换全体分辨率才划算。

**Decision rule**：拿到三个位宽的曲线，**并能解释为什么 8 bit 没有内点极小值而 3 bit 有**。这个"负结果"比原本预期的正结果更有价值：它告诉你截断阈值调优是**低位宽**的工具，不是 8-bit 的工具。你以后做 4-bit 时会再遇到它。

激活的 `α*` 预期比权重小（激活长尾更重），但激活分布依赖具体数据，**本机未测**，留到 V100 上用真实 batch 验。

### Q3 — 8-bit 优化器状态（V100，90 分钟）**收益 100% 真实**

这是本模块唯一一个在 V100 上能拿到**真实 8-bit kernel 且真实省显存**的实验。

**Hypothesis**：把 Adam 的 `exp_avg` / `exp_avg_sq` 做 block-wise（block=2048）8-bit 量化，优化器状态字节降到约 **0.25**；20 步内的 `|Δloss|` 小于 fp16 AMP 自身引入误差（Day 1 的 `fp16_vs_fp32_max_dloss`）的 2 倍。

**Metric**：`optimizer_state_bytes_ratio`（期望 ≈ 0.2505 = (2×1 + 2×4/2048)/8）；20 步逐步 `|Δloss|`；`exp_avg_sq` 的 per-block 动态范围直方图。

**Decision rule**：ratio ∈ [0.24, 0.27] **且** `|Δloss| ≤ 2×` Day 1 基线 → PASS。

**这里有一个本机实测推翻了直觉的点，是整个实验的题眼。**

直觉的答案是"`exp_avg_sq` 非负长尾，所以要 per-block 缩放"。**per-block 缩放是必要的，但不够。** 本机实测（`已确认`，2026-09-08，CPU）：block=2048 的 per-block **线性** 8-bit 量化仍然发散——

| 优化器状态量化方式 | 末步 loss |
| --- | --- |
| fp32 AdamW（基线） | **0.0186** |
| per-block 线性 8-bit | **3973 / 77330**（两次不同 seed，都发散） |
| per-block 幂律映射 8-bit | 收敛到 fp32 的 5% 以内 |

原因是**块内动态范围本身就跨好几个数量级**。缩放只是把整块平移到 `[0,1]`，块内部分布依然极度偏斜：线性映射下 255 个格子里，绝大多数梯度平方值仍然挤在最低的几个格子里，`sqrt(exp_avg_sq)` 一除就炸。

`lab/src/mm_quant/opt8bit.py` 的 `Adam8bit` 因此默认用**幂律映射** `q = round(255 · u^(1/P))`，P=4——把格子按幂律铺开，小值区间分到更多格子。`test_linear_sq_mapping_diverges` 把线性映射的发散**断言下来**，保证这个失败随时可复现，而不是变成一句口口相传的经验。

**Decision rule 补充**：如果你自己实现的版本 `|Δloss|` 大一个数量级，检查顺序是：① 有没有 per-block 缩放 → ② **块内用的是线性还是幂律映射** → ③ `exp_avg`（有符号，近似对称）和 `exp_avg_sq`（非负长尾）是不是用了同一套映射（它们不该用同一套）。

**可选对照**：`bnb.optim.AdamW8bit`（Q0 探针显示可用时）。装不上是预期结果，不是失败。

### Q4 — W8A16 weight-only PTQ 的速度归因（V100，90 分钟）

**Hypothesis**：对 64M MiniMind 做 weight-only INT8（int8 存储、用前 dequant 回 fp16），8 条固定 prompt 的 greedy 输出几乎不变；但在 V100 上**不会变快**，因为多了一个未融合的 dequant kernel，且没有 IMMA。真实收益只有权重字节减半。

**Metric**：`weight_bytes_ratio`（期望 ≈ 0.5 相对 fp16 存储）；8 条 prompt 的 token 级一致率；固定 512 条 held-out 的 perplexity 差；`decode_ms_per_token` 三档（fp16 / int8-per-tensor / int8-per-channel）。

**Decision rule**：拿到三个数字**并写出"慢在哪里"**——额外 kernel、无融合、batch=1 decode 本来就是 memory-bound。

**如果 int8 反而更快** → 去查是不是 dequant 被融合进了同一个 kernel。那才是真收益的机制，必须和"IMMA 加速"区分清楚——它们是完全不同的两件事，混淆了会让你在 H100 上做出错误预期。

### Q5 — STE / fake-quant QAT 在 fp16 AMP 下（V100，120 分钟）

**Hypothesis**：在 SFT 阶段插入 per-channel 权重 fake-quant（前向量化-反量化，反向 STE）继续训 200 步后再 PTQ，末 20 步平均 loss 落在 `fp16-SFT` 与 `直接 PTQ` 之间，且明显靠近前者。

**Metric**：三条曲线的末 20 步均值（`fp16-SFT` / `PTQ-直接` / `QAT-200步后-PTQ`）；`grad_norm` 是否因 STE 出现尖峰；GradScaler `skip_rate` 是否上升。

**Decision rule**：QAT 后 loss 明显靠近 fp16-SFT → PASS。

**QAT 比直接 PTQ 还差 → 第一个检查点是：fake-quant 的 scale 是否在 fp32 域计算。**

这是本模块最重要的一条教学点，值得单独说清楚。autocast 会把 `clamp` / `round` 的输入降到 fp16。fp16 的 eps 是 2⁻¹⁰ ≈ 9.8e-4，只有大约 3 位十进制有效数字。如果 scale 自身用 fp16 表示，它的表示误差就有 1e-3 量级——而你整个 8-bit 量化的分辨率也不过是 1/256 ≈ 3.9e-3。**scale 的误差会吃掉量化预算的四分之一**，而且完全静默：不报错、不出 NaN，只是量化效果莫名其妙地差。

`lab/src/mm_quant/quantizers.py` 里 scale 的计算强制留在 fp32，就是为了这一条。

**两条路径对照**：自写 STE，以及 torch 的 `FakeQuantize` / `FusedMovingAvgObsFakeQuantize`（一条走 cachemask kernel、一条走融合 kernel）。这是 torch 2.1 的真实实现差异，对照本身有信息量。构建期在本机 CPU 上做这个对照，已经挖出两个具体的坑：

**坑一：`round(x/scale)` 和 `round(x·(1/scale))` 不是一回事。** aten 的 fake-quant 内部算的是 `round(x * (1/scale))`——先求倒数再乘。数学上等价，浮点上不等价：先算 `1/scale` 引入一次舍入，结果与直接相除在约 **1/32768** 的元素上差一整个 scale step。如果你按"数学上更对"的写法实现自写 STE，它就会和框架对不上，而你会以为是自己写错了。`lab/src/mm_quant/qat.py` 的 `fake_quant_ste(..., inv_scale=True)` 默认复现 aten 的写法；传 `inv_scale=False` 可以把这个差异复现出来看。

**坑二：`FusedMovingAvgObsFakeQuantize` 与 `FakeQuantize` + `PerChannelMinMaxObserver` 不是位相等的。** 融合路径的 qparams 算出来的 scale 比 `amax/127` 差约 **0.6%**。所以这个对照的判据必须是"**相差不超过几个 scale step**"，**不能写成相等**。要求位相等的检查会在 V100 上失败，而原因不是 bug。

### Q6 — 激活 outlier 与幅度迁移（V100，90 分钟）

SmoothQuant 的思路，不装包，自己实现。

**Hypothesis**：hidden activation 存在**持续性** outlier 通道（跨 batch 重合，不是样本噪声）；用 `s_j = max|X_j|^α / max|W_j|^(1−α)` 把幅度从激活迁到权重后，W8A8 的端到端 MSE 在某个 `α ∈ (0,1)` 取最小。

**Metric**：`outlier_channels`（`|x| > 6σ` 的通道数）与**跨 batch 重合率**；W8A8 端到端 MSE vs `α ∈ {0, 0.25, 0.5, 0.75, 1.0}`；平滑后权重 `rel_err` 的上升量。

**Decision rule**：重合率 > 0.8 **且** MSE(α) 有内点极小 → PASS。

重合率低 → 结论是"这个 64M 模型没有强 outlier 结构"，判 `INCONCLUSIVE` 但**同样归档**。这是关于**模型规模**的真结论：outlier 结构随规模增强，小模型上看不到它恰恰是预期的。这个负结果对你以后上超大模型有直接价值——它告诉你不能拿小模型的量化结论外推。

## 4. 时间与顺序

| Session | 内容 | 环境 | 分钟 |
| --- | --- | --- | --- |
| 0 | Q0 探针 | V100 | 30 |
| 1 | Q1 + Q2 | CPU | 90 |
| 2 | Q3 | V100 | 90 |
| 3 | Q4 + Q6 | V100 | 180 |
| 4 | Q5 | V100 | 120 |

合计 510 分钟。Q1/Q2 在 CPU 上做，可以在等 GPU 排队时先做完。

## 5. 迁移到超大模型时会变的三件事

这个模块用 64M 模型建立方法。换到你真正要量化的那个超大模型时，下面三件事会变，**方法不变但结论要重测**：

1. **outlier 结构会变强。** Q6 在 64M 上很可能拿不到强重合率，在几十 B 上大概率能。所以 Q6 的 `INCONCLUSIVE` 不代表 SmoothQuant 类方法没用。
2. **per-channel 的 scale 存储开销占比会变小。** 权重矩阵越大，每个输出通道一个 scale 的相对开销越低，per-group 的收益递减点会往后移。
3. **优化器状态在总显存里的占比会变。** 小模型上激活占大头，超大模型上参数+优化器状态占大头——Q3 的收益在大模型上更显著。

**一件不会变的事**：Q5 的 scale 必须在 fp32 域算。这跟模型多大没关系，是 fp16 的表示能力决定的。

## 6. 边界

- V100 上的所有结论**不包含任何 INT8 GEMM 加速**。要测那个，需要 sm_75 以上的卡。
- 如果 `qat_base/` 放的是公司自己的模型，那么它、以及它派生出的量化产物、误差统计、逐层敏感度表，**全部留在 V100 上**。带出来的只有证据字段里的抽象值。
- 选基座时先查它是不是 bf16 预训练的——见 `weights/README.md` 第 2 节那个坑。
