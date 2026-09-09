# 扩展模块 R — 低精度 + RL 后训练（V100 版）

> 生成日期：2026-09-08 · 生成方：cc · 状态：`BUILT / RUNTIME-UNVERIFIED`
> 定位：**Gate 之后**的独立模块，不进 Gate。6 个实验共约 540 分钟，建议拆 4 个 session。
> 前置：主线 Day 5 的 `rl_numeric.json`；R5 还需要 Q 模块的 `weight_only.py`。

## 0. 这个模块的立论

你说以后要做 RL 后训练，而且很可能是**低精度 + RL 一起**。那么这个模块只围绕一句话展开：

> **低精度 + RL 的所有坑，都发生在"两个几乎相等的对数概率相减"这一个动作上。**

SFT 的 loss 是一个绝对量。fp16 的 1e-3 相对误差在那里无所谓——loss 从 8.7 降到 2.3，误差在小数点后第三位。

RL 不是。`ratio = exp(logp_new − logp_old)` 和 `KL` 都是**两个几乎相等的数的差**。当 `logp_new` 和 `logp_old` 只差 1e-4，而 fp16 的 eps 是 9.8e-4 时，这个差值里**一位有效数字都不剩**。灾难性抵消（catastrophic cancellation）把量化噪声放大到和信号同量级。

这不是"精度低一点，效果差一点"。这是**信号被噪声完全淹没，但训练照常进行、loss 照常输出、没有任何报错**。

所以这个模块的六个实验，本质上都在回答同一个问题的不同侧面：**哪些量必须留在 fp32 域，怎么证明。**

## 1. 事实表（`已确认`，2026-09-08 联网核验）

### 1.1 工具链在 V100 + torch 2.1 上的边界

| 工具 | 能不能用 | 卡在哪里 | 来源 |
| --- | --- | --- | --- |
| **`trl`** | **能，`0.14.0` ≤ trl ≤ `0.22.2`** | 0.14.0 是 `GRPOTrainer` 的起点；0.23.0 起要 transformers ≥4.56.1，而 transformers ≥4.56 要 torch ≥2.2 | [trl v0.14.0 release](https://github.com/huggingface/trl/releases/tag/v0.14.0) · [PyPI](https://pypi.org/pypi/trl/0.22.2/json) |
| **GRPO 是否强依赖 vLLM** | **不依赖** | `GRPOConfig.use_vllm` 默认 `False`；vLLM 只在 "Speed up training with vLLM" 一节作为可选加速 | [TRL GRPO 文档](https://huggingface.co/docs/trl/en/grpo_trainer) |
| `vllm` | **不能** | **每个版本都硬钉一个精确 torch 版本**：0.2.7 要 `torch==2.1.2`（不是 2.1.0），最新版要 `torch==2.13.0`。且官方最低 CC **7.5**，V100 是 7.0 | [PyPI 0.2.7](https://pypi.org/pypi/vllm/0.2.7/json) · [vLLM 安装](https://docs.vllm.ai/en/latest/getting_started/installation/gpu.html) |
| `verl` | **不能** | 要求 CUDA ≥**12.8**、vllm ≥0.18 | [verl 安装](https://verl.readthedocs.io/en/latest/start/install.html) |
| `OpenRLHF` | **不能** | 绑定 vllm 0.27.1 | [README](https://raw.githubusercontent.com/OpenRLHF/OpenRLHF/main/README.md) |
| `deepspeed` | 声明层全系列写 `torch>=2.0.0` 无上界；**实跑建议 0.12.x–0.14.x** | 会 JIT 编译 CUDA 算子，声明宽松 ≠ 能跑。判据是现场 `ds_report` | [PyPI](https://pypi.org/pypi/deepspeed/json) |

**这三个不可用的框架有一个共同点，值得记住**：现代 RL 训练框架都把推理引擎当作硬依赖，而推理引擎是整个生态里对 GPU 架构和 torch 版本最挑剔的一环。在老卡的内网上做 RL，路线必须是「训练框架 + transformers 原生 generate」。这不是将就——`use_vllm=False` 是官方支持的默认路径。

### 1.2 fp16 下的数值事实

| 事实 | 内容 | 来源 |
| --- | --- | --- |
| fp16 的三个数 | 最小正规数 2⁻¹⁴ ≈ 6.1e-5；最大 65504；**eps = 2⁻¹⁰ ≈ 9.8e-4**（约 3 位十进制有效数字） | IEEE 754 binary16 |
| autocast 会自动提升哪些算子 | `softmax`、`log_softmax`、`cross_entropy` 都在 torch 2.1 的 **"CUDA Ops that can autocast to float32"** 列表里 | [torch 2.1 amp](https://docs.pytorch.org/docs/2.1/amp.html) |
| **但这不够** | `ratio = exp(logp − logp_old)`、KL 的减法，如果写在 autocast 区**外**，或者用了 fp16 缓存的 `old_logp`，精度就在那里丢掉了。autocast 只保护它列表里的算子，保护不了你自己写的减法 | 同上 |
| **bf16 预训练模型的陷阱** | torch AMP 文档原文警告："most bf16-pretrained models cannot operate in the fp16 numerical range of max 65504 and will cause gradients to **overflow** instead of underflow"，且 `GradScaler` 不保证 scale 保持 >1 | 同上 |
| GradScaler 的三条硬规则 | ① 多个 loss → 各自 `scaler.scale()`；② 多个 optimizer → 各自 `scaler.step()`；③ **`scaler.update()` 每个 iteration 只调一次，在所有 optimizer step 之后**。梯度累积期间 scale 必须保持不变 | [AMP examples](https://docs.pytorch.org/docs/2.1/notes/amp_examples.html) |
| 训练/推理精度不一致会破坏 RL | TRL 文档专设 "Dealing with the Training-Inference Mismatch" 一节："This mismatch leads to a biased gradient update which has been observed to destabilize training"，默认开启 Truncated Importance Sampling | [TRL GRPO](https://huggingface.co/docs/trl/en/grpo_trainer) |
| ref 模型的精度缓解手段 | `disable_dropout`（防止同一输入产生不同 logprob）；`cast_lm_head_to_fp32`（带 ref model 时推荐，仅在 `tie_word_embeddings=False` 时可用）；`beta=0.0` 时干脆不加载 ref model | 同上 |

### 1.3 一个开放问题：fp16 对 RL 未必是劣势

主流实践把 bf16 当作"更稳"的选择（DeepSpeed 文档明写 bf16 不需要 loss scaling）。但 2025 年有工作提出相反主张：

> arXiv:2510.26788《Defeating the Training-Inference Mismatch via FP16》：*"The widely adopted BF16, despite its large dynamic range, introduces large rounding errors that breaks the consistency between training and inference"*，并称改回 FP16 能得到 "more stable optimization, faster convergence, and stronger performance"。

**这是活跃的开放问题，不是定论。** 但它对你的处境有直接意义：V100 强制 fp16，这个约束**未必是纯粹的损失**。R1 把它做成一个可证伪的 hypothesis，而不是接受一个安慰性的说法。

`来源计划假设`：该论文的结论尚未成为共识，本模块引用它作为假设来源，不作为结论。

## 2. 六个实验

### R1 — on-policy ratio 恒等性（V100，90 分钟）

**Hypothesis**：完全 on-policy 时，第一次内层更新前 `ratio ≡ 1`。fp16 下若生成走 `padding_side=left` 的批量路径、训练走非 padding 路径，`ratio` 会**系统性**偏离 1，偏离量足以让一部分 token 在 step 0 就被 clip。

**Metric**：`ratio` 分布（mean / std / p99 / max）；`frac_clipped_at_step0`；**同一批数据在 fp32 下重算 logp** 的 ratio 分布。

**Decision rule**——这是本实验的全部价值所在，两个结论必须分开：

| 观测 | 结论 | 下一步 |
| --- | --- | --- |
| fp32 下 `max|ratio−1| < 1e-4`，fp16 下显著更大 | **精度导致** | 把 logprob 链路提到 fp32 |
| **fp32 下也偏离** | **kernel / mask 路径不一致** | 查 MiniMind `Attention.forward` 的两条分支——`attention_mask` 全 1 才走 SDPA，含 0 时走手写分支，两条路径数值不同 |

**把这两个结论混在一起，这个实验就白做了。** 一次 fp32 重算就能把它们分开，代价极小。

### R2 — KL 三个估计量在 fp16 下的偏差与方差（V100，60 分钟）

**Hypothesis**：`k1 = −log r` 方差最大且**可以为负**；`k2 = (log r)²/2`；`k3 = (r−1) − log r` 非负且方差最小。三者的 fp16-vs-fp32 相对偏差差一个数量级。

**Metric**：同一批 rollout 上 `k1/k2/k3` 的均值与标准差各两组（fp16 / fp32）；`k1 < 0` 的比例。

**Decision rule**：三个估计量的 fp16-vs-fp32 相对偏差都有数字，**且能解释为什么 `k3` 是 KL 的无偏且非负估计**——因为 `E[r] = 1` 所以 `E[r−1] = 0`，于是 `E[k3] = E[−log r] = KL`。解释不出来就是没懂，数字再漂亮也不算。

### R3 — GradScaler 在 RL 里丢掉的是一整批生成（V100，90 分钟）

SFT 里跳一步只是少更新一次。**RL 里跳一步，丢掉的是一整轮昂贵的 rollout。**

**Hypothesis**：`skip_rate_grpo > skip_rate_sft`（同 `init_scale=65536`），因为 `advantage × logp` 的动态范围比 SFT 的 cross-entropy 更大。

**Metric**：`skip_rate_sft` vs `skip_rate_grpo`；`scale` 轨迹；每次跳步对应被丢弃的生成 token 数；用 hook 定位 **inf/nan 首次出现在哪个张量**。

**Decision rule**：拿到两个 skip_rate + 至少一次跳步的定位张量。`skip_rate_grpo > 0.2` → 把 `init_scale` 降到 1024 重跑并记录变化。这就是"低精度 RL 必须单独调 scaler"的直接证据。

**同时验三条硬规则**（`lab/src/mm_rl/scaler_rules.py` 里做成可失败的单测，CPU 上就能验调用次数逻辑）：GRPO 的一次优化步包含一次 group rollout + 多次 minibatch 前反向。**如果每个 minibatch 都调 `scaler.update()`，scale 会被反复重置，等价于把 loss scaling 打坏**——而且不报错。

### R4 — advantage 归一化的灾难性抵消（V100 + CPU 对照，60 分钟）

**Hypothesis**：组内 `(r − mean)/(std + 1e-4)`，当 reward 量级为 1 而组内差异为 1e-3 时，fp16 只有约 3 位十进制有效数字，`r − mean` 只剩 0–1 位有效数字，`std` 与 advantage 双双退化成量化噪声。

**Metric**：`zero_std_group_ratio`（fp16 算 vs fp32 算，两组）；`max|advantage|` 两组；把归一化改到 fp32 后的差异。

**Decision rule**：能给出"advantage 与所有归一化统计量必须在 fp32 域计算"的实测依据——两组 `max|adv|` 相差 ≥ 10×，或 fp16 组出现 inf。两组一样 → 结论是"该 reward 分布下不敏感"，**同样归档**。

这个实验在 CPU 上就能构造出灾难性抵消的场景，不必等 GPU。

### R5 — 把冻结的 ref 模型量化到 INT8（V100，120 分钟）**两个模块的交点**

这是"低精度 + RL 一起训练"这条路上第一个真正要做的判断：**ref 模型能不能省？**

**Hypothesis**：ref 模型 weight-only INT8 可省约一半 ref 显存；引入的 KL 偏差是**系统性**的（同一 prompt 上符号一致，不会互相抵消）；若相对偏差超过 KL 系数对应的容忍度，int8 ref 会改变优化方向。

**Metric**：`ref_weight_bytes_ratio`；`|KL_int8ref − KL_fp16ref| / KL_fp16ref` 的均值与 p99；**偏差符号一致率**；**policy 梯度的余弦相似度**（int8-ref vs fp16-ref）。

**Decision rule**：

- 余弦 > 0.99 → int8 ref 可用，写进"超大模型 RL 的显存账"结论。
- 余弦 < 0.9 → 判不可用，并写出原因：**KL 是两个对数概率的小差值，量化噪声在这里不被抵消而是被放大。**

为什么"符号一致率"是关键指标：如果偏差是随机的，它在大量样本上会互相抵消，对期望梯度影响有限；如果偏差是系统性的（同一 prompt 上符号一致），它就是一个**偏置**，会持续把优化推向一个错误方向。这两种情况的数值大小可能一样，后果完全不同。

**跑之前必须知道的一个退化情形**：如果 policy 和 ref 是同一份权重（还没开始训练时就是这样），那么 `log_ratio ≡ 0`，于是 `sign_agreement` 变成 `nan`、`kl_rel_bias` 变成 `inf`。**这不是 bug，是 0/0。** `lab/src/mm_rl/ref_model.py` 的实验入口会先加一个 `policy_drift` 扰动把两者拉开再测。看到 nan/inf 时第一件事是确认 policy 有没有真的偏离 ref，而不是去查量化代码。

### R6 — LoRA policy + 冻结低精度 backbone（V100，120 分钟）

**Hypothesis**：只训 LoRA 时，梯度 + 优化器状态显存降到 <5%，但**激活显存不降**。fp16 下 LoRA 的 `B` 初始化为 0，因此 step 0 的 policy 与 ref **完全相同**，`ratio ≡ 1` 且 `KL ≡ 0` 是一条**可精确判定**的自检不变量。

**Metric**：四项字节账（param / grad / optim / activation）full vs LoRA 的比值；**step 0 的 `max|KL|` 是否精确为 0**；100 步后 KL 的增长曲线。

**Decision rule**：`step 0 的 max|KL|` 不是精确 0 → **立刻停下**，按顺序查三件事：

1. LoRA 的 `B` 是不是零初始化；
2. ref 是不是与 policy 共享同一份 backbone；
3. dropout 有没有关掉（TRL 文档专门提供 `disable_dropout`，就是因为同一输入产生不同 logprob 会污染 KL）。

**这是整个模块里唯一一个"应当精确为 0"的断言。** 大部分数值检查只能给出"差不多对"的区间判断，容易自欺；这一条不行——它要么是 0，要么你的实现有 bug。诊断价值极高，所以放在最后作为整套实现的总校验。

它同时补回了主线里砍掉的 LoRA。

## 3. 时间与顺序

| Session | 内容 | 环境 | 分钟 |
| --- | --- | --- | --- |
| 1 | R1 | V100 | 90 |
| 2 | R2 + R4 | V100（R4 的 fp32 对照可 CPU） | 120 |
| 3 | R3 + R6 | V100 | 210 |
| 4 | R5 | V100 | 120 |

合计 540 分钟。R4 的灾难性抵消构造、R6 的 LoRA 零初始化不变量在 CPU 上就能先验，建议排队等 GPU 时先做。

## 4. 这个模块留下的可迁移结论

做完六个实验，你应该能回答下面这些问题——它们和模型多大、用哪个框架都没关系：

1. RL 训练里哪些量**必须**留在 fp32 域，为什么（答案至少包括：advantage 与所有归一化统计量、logprob 的差、KL）。
2. 为什么 GradScaler 在 RL 里比在 SFT 里更危险，`update()` 该在哪里调。
3. 怎么用**一次 fp32 重算**把"精度问题"和"kernel/mask 路径问题"分开。
4. 怎么判断一个近似（比如 int8 ref）是引入了**噪声**还是引入了**偏置**，以及为什么后者严重得多。
5. 什么样的自检不变量是"精确为 0"型的，为什么这类断言比区间判断可靠。

## 5. 边界

- 本模块的所有结论来自 64M MiniMind + 规则 reward。**不能提升为对超大模型或真实任务的结论**——小规模代理任务上的 RL 结果尤其不可外推。
- `zero_std_group_ratio` 很可能高到让部分实验 `INCONCLUSIVE`：规则 reward 值域只有 `[-3, 3]`，分辨率本来就差。这是**预注册的合法出口**，不是失败。
- 运行产物留在 V100 上，只带出证据字段里的抽象值。
- 本周不装 vLLM / verl / OpenRLHF。理由必须落到"会改动 torch 2.1.0"这一条上——这是你将来在别的内网环境也会反复遇到的判断。
