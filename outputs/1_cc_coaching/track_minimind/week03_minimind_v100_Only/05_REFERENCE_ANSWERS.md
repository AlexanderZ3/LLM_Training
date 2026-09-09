# Week M03 Reference Answers — 逐 ID 参考答案与评分标准

> 配套 `04_ORAL_EXAM.md`，ID 一一对应，共 **27 题**。**默认闭卷作答后再看。**
> 每题给：关键评分点（3–5 条）· 0–4 分 rubric · 常见误区 · 追问 · 定位（`01_FOUNDATIONS.md` 章节号或 `lab/` 文件名）。
>
> **通用高分标准**（所有层通用，`Debug` / `Design` / `Trade-off` 三层强制）：**有假设、有证据、有替代解释、有失败边界、有可验证下一步**。五条里少于三条的答案，即使结论正确也封顶 2 分。只背术语不给推导、不给数字、不给失败边界的答案封顶 2 分。
>
> **数值约定**：MB = 10⁶ B，GB = 10⁹ B，MiB = 2²⁰，GiB = 2³⁰。本文所有数字都是由 `01_FOUNDATIONS.md` 第 1–4 节的配置**推出的 `计算` 值**，或标注为 `估算`。本周截至出卷时**没有任何 V100 实测数据**；答案里出现"我实测到"即判 0 分。**唯一例外**是 `06_QUANT_LOWBIT.md` 里两条**本机 CPU 实测**结论（截断阈值 α、8-bit 优化器状态的幂律映射），引用时必须标明是 CPU 实测、量级待 V100 复测。
>
> **模型来源边界**：本周 `lab/src/mm_v100/model.py` 是周包自带的、与 MiniMind 形状对齐的实现，**不是 MiniMind 本体**，不含 MiniMind 的权重与 tokenizer。字节账 / 通信账 / 数值账的结论成立且可迁移；但"直接加载 MiniMind 发布的 checkpoint"这一步不成立。答案里把权重来源说清楚的加分，默认可以加载的扣分。
>
> **公司边界**：评分时把"答案里出现了绝对毫秒 / 绝对 GB / 拓扑矩阵 / 日志原文"视为一次边界违规，该题封顶 2 分，并在证据里记 `FAIL-SYSTEM`。

---

## Recall

### M03-R-01

**关键评分点**

1. IEEE 754 binary16（1 符号 + 5 指数 + 10 尾数）：**最小正规数** $2^{-14} = 6.1035\times10^{-5}$；**最大值** $(2-2^{-10})\cdot2^{15} = \mathbf{65504}$；**1.0 处 ULP（eps）** $2^{-10} = 9.7656\times10^{-4}$。补充两个：**最小 subnormal** $2^{-24} = 5.9605\times10^{-8}$（小于它直接 flush to zero）、**单位舍入** $u = 2^{-11} = 4.8828\times10^{-4}$。动态范围 $65504/5.96\times10^{-8}\approx1.1\times10^{12}$。
2. `GradScaler(init_scale=65536.0, growth_factor=2.0, backoff_factor=0.5, growth_interval=2000)`，另有 `enabled=True`。四个数值默认值必须都对。
3. `GradScaler(enabled=(args.dtype == 'float16'))`。含义：**只有 fp16 路径 scaler 才真正生效**；bf16 路径下它是空壳（bf16 指数位与 fp32 相同，不需要 loss scaling）。在 V100 上因为只能走 fp16，所以 scaler **永远是开的**，于是"跳步"这一整套状态机在本周全程在线——这正是三档静默故障里 B 档存在的前提。

**0–4 分 rubric**

- 0：三个 fp16 的数说不全，或把 eps 与最小正规数混为一谈。
- 1：三个数对，`GradScaler` 默认值答不全（少于三个）。
- 2：第 1、2 问全对。
- 3：三问全对，含 `enabled` 的取法与它在 V100 上恒为真。
- 4：以上全对，并主动指出 eps（$2^{-10}$）与单位舍入（$2^{-11}$）差一倍且各自用在什么场合（前者是相邻数间隔，后者是舍入到最近的最大相对误差），以及"V100 上 scaler 恒开"直接推出"必须记录落地率而不是只看 loss"这条纪律。

**常见误区**

- 把 eps 写成 $2^{-11}$（那是单位舍入 $u$），或把最小正规数写成最小 subnormal。
- 把 `growth_interval` 记成 2000 步"翻倍一次"而不是"连续 2000 次成功后才试一次翻倍"——后者才是语义。
- 认为 bf16 下 `GradScaler` 会报错。它不报错，只是被 `enabled=False` 关成空壳。

**追问**

- 从 65536 一路减半到 $2^{-24}$ 需要多少次？一个 scale 已经掉到 $10^{-7}$ 的 run 还有救吗？为什么？
- `growth_interval=2000` 这个默认值，对一个总共只跑 200 步的 smoke 意味着什么？

**定位**：`01_FOUNDATIONS.md` §3.1、§3.4；`lab/src/mm_v100/numeric_ref.py`。

---

### M03-R-02

**关键评分点**

1. **FP16 Tensor Core（HMMA，FP32 累加）：有**（Volta 架构，SXM2 峰值约 125 TFLOPS FP16 TC / 15.7 TFLOPS FP32）。**BF16：没有**——`torch.cuda.is_bf16_supported()` 要求 `get_device_properties(...).major >= 8`，V100 的 major = 7。**INT8 Tensor Core（IMMA）：没有**——整数 MMA 从 Turing sm75 起，Volta 的 Tensor Core 只接受 FP16 输入。
2. torch 2.1 的 **flash 后端只覆盖 sm75–sm90**，V100 是 **sm70（CC 7.0）**，因此只能落到 **mem-efficient** 或 **math**，**不可能**是 flash。本周没有 FlashAttention-2。
3. `bitsandbytes`：`LLM.int8()` 门槛 **CC ≥ 7.5** → **不能**（`supports_igemmlt()` 里 `< (7,5)` 直接返回 False）；**8-bit 优化器**门槛 **CC ≥ 6.0** → **能**，这是本周唯一能拿到真实 8-bit kernel 且真实省显存的地方；**NF4 / FP4** 门槛 **CC ≥ 6.0** → **能**，但 `bnb_4bit_compute_dtype` 只能是 `float16`，不能照抄文档里的 `bfloat16`。
4. `convert_fx` 的产物：**不能上 GPU**（fbgemm / qnnpack / x86 / onednn 全是 CPU 后端）。`torchao`：**完全不能**（最低 torch 2.5）。`torch._int_mm`：**存在且没有任何 compute capability 的 `TORCH_CHECK`**，行为取决于 cuBLASLt 运行时，判 `未知`、**必须探针**；而且有已知正确性缺陷（pytorch#107671），**即使不报错也要与 fp32 参考比对**。
5. 加分：`torch.ao` 的 **fake-quant CUDA kernel 是有的**（`FakeQuantizeCore.cu` / `FusedObsFakeQuant.cu`，前向反向齐全），所以 QAT 在 V100 上是"真跑"而不是模拟。

**0–4 分 rubric**

- 0：认为 V100 支持 BF16，或认为能用 flash 后端。
- 1：知道 no-BF16 / no-flash，但说不出 CC 判据。
- 2：第 1、2 问正确并给出 CC 数字。
- 3：四问基本正确，含 bnb 三档门槛（7.5 / 6.0 / 6.0）。
- 4：以上全对，且把 `torch._int_mm` 标成 `未知` 而不是"能"或"不能"，并主动说出"没有 CC 检查"和"有已知正确性缺陷"是两件独立的风险；再指出 fake-quant kernel 存在这一点让 QAT 成为 V100 上少数几件"真跑"的量化工作之一。

**常见误区**

- 把 `MiniMindConfig.flash_attn=True` 当成"启用了 FlashAttention 库"。它只是"是否调用 `F.scaled_dot_product_attention`"的开关。
- 把 `bitsandbytes` 当成一个整体判"能"或"不能"。它的三类功能门槛不同，必须分开。
- 因为 `torch._int_mm` 在 V100 上不报错就当成"能用"。没有 CC 检查恰恰意味着**没人保证它对**。

**追问**

- `bnb_4bit_compute_dtype` 如果照抄文档写成 `bfloat16`，在 V100 上会在哪一步失败？这个失败是响的还是静默的？
- 你会用什么最小实验判定 `torch._int_mm` 在这台机器上可不可信？停止条件是什么？

**定位**：`00_WEEK_CARD.md` §3.1、§3.2；`01_FOUNDATIONS.md` §7.1；`06_QUANT_LOWBIT.md` §1.1；`lab/scripts/probe_int8_caps.py`。

---

### M03-R-03

**关键评分点**

1. `hidden_size=768`、`num_hidden_layers=8`、`vocab_size=6400`、`num_attention_heads=8`、`num_key_value_heads=4`（GQA，$n_{rep}=2$）、`head_dim = 768/8 = 96`、`intermediate_size = ceil(768\pi/64)\times64 = \lceil37.699\rceil\times64 = 38\times64 = \mathbf{2432}$、`tie_word_embeddings=True`、`dropout=0.0`、`rms_norm_eps=1e-6`、`rope_theta=1e6`、`use_moe=False`。
2. 单层（所有 `nn.Linear` 都 `bias=False`）：`q_proj` $768\times768 = 589{,}824$；`k_proj` $768\times384 = 294{,}912$；`v_proj` $294{,}912$；`o_proj` $589{,}824$（注意力小计 **1,769,472**）；`q_norm` 96 + `k_norm` 96 = **192**；`input_layernorm` 768 + `post_attention_layernorm` 768 = **1,536**；`gate_proj`/`up_proj`/`down_proj` 各 $768\times2432 = 1{,}867{,}776$，小计 **5,603,328**。$P_\ell = 1{,}769{,}472 + 5{,}603{,}328 + 1{,}536 + 192 = \mathbf{7{,}374{,}528}$。
3. $P = Vd + 8P_\ell + d = 4{,}915{,}200 + 58{,}996{,}224 + 768 = \mathbf{63{,}912{,}192}$。`lm_head.weight` 与 embedding **共享存储**，贡献 0。**不进 `model.parameters()` 的**：RoPE 的 `freqs_cos` / `freqs_sin`（是 buffer 不是 parameter），因此也不进梯度、不进优化器状态、不进 DDP 的桶。
4. tie 的三个后果：① 参数量少 4,915,200（占 7.7%）；② 每卡少 $16\times4{,}915{,}200 = 78{,}643{,}200$ B $= 78.6$ MB 的 fp32 训练态；③ 并行放置上——DDP 里它是**一个** parameter、落在**一个** bucket，只 all_reduce 一次，没有重复归约；FSDP 里共享参数**必须落在同一个 unit**，按 block 粒度 wrap 时 `embed_tokens` 与 `lm_head` 都留在 root unit，天然合法（否则 FSDP 在构造时报共享参数错误）。

**0–4 分 rubric**

- 0：`intermediate_size` 用了别的规则（`8/3·d` 或 `4d`），或总参数量偏差超过 1%。
- 1：十二个值大体对，但 $P_\ell$ 或 $P$ 有一项算错。
- 2：$d_{ff}=2432$、$P_\ell = 7{,}374{,}528$、$P = 63{,}912{,}192$ 三个数全对。
- 3：以上全对，且答出 buffer 不进 `parameters()`。
- 4：以上全对，并说清 tie 的三个后果（尤其第三条的 FSDP unit 约束），以及"如果手算结果**大于** 63,912,192，第一件事是检查有没有把 buffer 数进去"。

**常见误区（这两条按主会话 2026-09-08 的确认单列）**

- **用 $8/3$ 规则算 $d_{ff}$**：$768\times8/3 = 2048$，FFN 每层少 $3\times768\times(2432-2048) = 884{,}736$，8 层共少 **7,077,888**，总数错成 56,834,304。MiniMind 用的是 $\lceil d\pi/64\rceil\times64$，把中间维定在 $\approx\pi d$ 再对齐到 64 的倍数（便于 Tensor Core 的 tile 划分）。
- **漏掉 QK-norm**：`q_norm` / `k_norm` 各 96，每层 **192**，8 层共少 **1,536**。
- 两个误区叠加时总差 $7{,}077{,}888 + 1{,}536 = \mathbf{7{,}079{,}424}$，得到 56,832,768。看到这个数就知道犯了哪两个错。
- 把 `lm_head` 再数一遍 4,915,200（忘了 tie）。

**追问**

- $2432/768 = 3.1\overline{6}$ 这个比值在本周的哪一本账里以系数形式出现过？如果 $d_{ff}$ 改成 2048，那个系数变成多少？
- 如果 `tie_word_embeddings` 改成 `False`，FSDP 的 wrap 策略需要跟着改吗？为什么？

**定位**：`01_FOUNDATIONS.md` §1.1、§1.2；`lab/src/mm_v100/model.py`；`lab/tests/test_model_shapes.py`。

---

### M03-R-04

**关键评分点**

1. DDP：`bucket_cap_mb=25`（单位是 **MiB**，即 26,214,400 B）、`gradient_as_bucket_view=False`、`find_unused_parameters=False`、`broadcast_buffers=True`、`static_graph=False`。另有一个更小的首桶常数 `_DEFAULT_FIRST_BUCKET_BYTES` = 1 MiB（`推断`，C++ 侧常量）。
2. FSDP：`backward_prefetch=BackwardPrefetch.BACKWARD_PRE`（**默认开**）、`forward_prefetch=False`（**默认关**，这是本周通信账里最重要的一个默认值）、`limit_all_gathers=True`、`use_orig_params=False`、`sync_module_states=False`。
3. `StateDictType`：`FULL_STATE_DICT` / `SHARDED_STATE_DICT` / `LOCAL_STATE_DICT`。
4. NCCL process group 默认 `timeout = 1800 s`（30 分钟）。把"超时后静默等待"变成"超时后崩溃"的环境变量是 `TORCH_NCCL_ASYNC_ERROR_HANDLING=1`（2.1 同时仍接受旧名 `NCCL_ASYNC_ERROR_HANDLING`）。

**0–4 分 rubric**

- 0：`bucket_cap_mb` 的单位说成 MB（$10^6$），或不知道 `gradient_as_bucket_view` 默认是 False。
- 1：DDP 一组基本对，FSDP 一组答不全。
- 2：前两问全对。
- 3：四问全对。
- 4：以上全对，并主动指出两条可执行推论：① `gradient_as_bucket_view=True` 是 DDP 路径上最便宜的一笔优化（一个布尔参数换 255.65 MB/卡，占 DDP 静态开销的 18.2%）；② `forward_prefetch=False` 与 `backward_prefetch=BACKWARD_PRE` 的不对称，直接对应"FSDP 前向的 9 次 all-gather 没有预取、反向的 9 次有"，是"FSDP 慢"的第一号候选来源。

**常见误区**

- 把 `bucket_cap_mb=25` 当成 25×10⁶ B。差 4.9%，会让桶数算错一个。
- 以为 `forward_prefetch` 默认是 True（"反向都预取了，前向当然也预取"）。恰恰相反。
- 把 `LOCAL_STATE_DICT` 当成"最通用"的那个。它是最快但**最不通用**的：key 是 `_flat_param`，长度是 $W$ 的函数。

**追问**

- `limit_all_gathers=True` 默认开着，它带来的是什么代价、防的是什么？在 64M 上关掉它可能观察到什么？
- 默认 30 分钟的超时，在一次只跑 10 分钟的 smoke 上意味着什么？你会怎么改这个默认值？

**定位**：`01_FOUNDATIONS.md` §4.2、§4.4(a)、§8.1、附录 A；`lab/src/mm_v100/collectives.py`。

---

### M03-R-05

**关键评分点**

1. `numpy < 2`（**必须**；torch ≤2.1 的 `.numpy()` / `.from_numpy()` 在 numpy 2.x 下失效，且 1.x 编译的扩展与 2.x 二进制不兼容）；`transformers >= 4.46.0, < 4.56.0`（4.56.0 起 `setup.py` 声明 `torch>=2.2`）；`trl >= 0.14.0, <= 0.22.2`（0.14.0 是 `GRPOTrainer` 的起点；0.23.0 起要 transformers ≥4.56.1）；`bitsandbytes <= 0.45.5`（0.46.0 起 `torch>=2.2`，0.48.0 起 `torch>=2.4`）。加分项：`tokenizers >=0.21,<0.22`、`huggingface-hub >=0.34.0,<1.0`、`accelerate >=1.4.0`、`peft >=0.14.0`、`datasets >=3.0.0`、`deepspeed` 声明宽松但实跑建议 0.12.x–0.14.x（唯一判据是现场 `ds_report`）。
2. 明确不能用：`torchao`（最低 torch 2.5）、`vllm`（每个版本硬钉一个**精确**的 torch 版本，连 0.2.7 都要 `torch==2.1.2` 而不是 2.1.0；且官方最低 CC 7.5）、`verl`（要 CUDA ≥12.8、vllm ≥0.18）、`OpenRLHF`（绑 vllm 0.27.1）。**后三个的共同结构性原因：现代 RL 训练框架都把推理引擎当硬依赖，而推理引擎是整个生态里对 GPU 架构和 torch 版本最挑剔的一环。** 所以在老卡的内网上做 RL，路线只能是"训练框架 + transformers 原生 `generate`"——`GRPOConfig.use_vllm` 默认就是 `False`，这是官方支持的路径，不是将就。
3. 单一根因：**某个包为了满足自己的依赖把 torch 2.1.0 换掉了。** 对策是 **constraints 文件**（`pip install -c constraints-v100.txt ...`，文件里钉 `torch==2.1.0` 与 `numpy<2`）——它只声明"要装就必须是这个版本"，不触发安装，让冲突在**依赖解析阶段可见地失败**。不该用 `--no-deps`：它能挡住 pip 动 torch，但同时关掉了所有依赖检查，装完的环境可能缺包，而且要到 import 或运行时才炸——把一个响的失败换成了一个静默的坏环境。
4. `HF_HUB_OFFLINE=1`（缓存里没有就**立刻报错**，正是内网想要的行为）。`TRANSFORMERS_OFFLINE` 在 transformers 4.55 的 `utils/hub.py` 里已经**不再被读取**，写了没用。

**0–4 分 rubric**

- 0：说不出 `numpy<2` 或 `transformers<4.56` 这两条上限中的任何一条。
- 1：四个上限里对两条，不能说清卡在哪里。
- 2：四个上限全对且各给了依据。
- 3：加上四个禁用包与 constraints 机制。
- 4：以上全对，并说出"三个 RL 框架的共同结构性原因"，以及为什么 `--no-deps` 是把响的失败换成静默的坏环境（这是本题的题眼）。

**常见误区**

- 认为 `pip install --no-deps` 是"保护 torch"的正确做法。
- 认为 `vllm` 只是版本高一点，降级就能用。它钉的是**精确**版本，且 CC 门槛也不满足，两条独立的硬约束。
- 把 `TRANSFORMERS_OFFLINE=1` 当成必需项。

**追问**

- wheelhouse 的四个参数（`--platform` / `--python-version` / `--implementation` / `--abi`）如果对准了外网机器而不是 V100，症状是什么？在哪一步暴露？
- 本周主线 Day 0–5 需要装新包吗？如果不需要，为什么还要把这份矩阵背下来？

**定位**：`08_INTRANET_SETUP.md` §1、§2、§4；`00_WEEK_CARD.md` §3.4；`07_RL_LOWPRECISION.md` §1.1。

---

## Explain

### M03-E-01

**关键评分点**

1. 未训练模型的 logits 近似与输入无关且各类别等价 → softmax 输出接近均匀分布 $1/V$ → $\mathcal{L} = -\mathbb{E}[\log p(y)] = -\log(1/V) = \ln V$。代入 $V = 6400$：$\ln 6400 = \ln 64 + \ln 100 = 6\ln2 + \ln100 = 4.158883 + 4.605170 = \mathbf{8.76405}$。这不是经验值，是可以在纸上算完的结构量。
2. bits $= 8.76405/\ln2 = 8.76405/0.693147 = \mathbf{12.6439} = \log_2 6400$；PPL $= e^{8.76405} = \mathbf{6400}$。常数是 $\ln 2 = 0.693147$，它是 nats↔bits 的唯一换算常数。必须背是因为论文里报 bits/byte、代码里算 nats，两者差一个 $\ln 2$；混淆会让你把一个 1.44 倍的差异当成模型问题。
3. **能验**的是"模型接线"：`vocab_size` 是否对、tie 是否真的生效（`lm_head.weight.data_ptr() == embed_tokens.weight.data_ptr()`）、初始化尺度是否异常（`logits.std()`）、logits 是否被意外缩放。**不能验**的是"数据接线"，尤其是掩码：**随机初始化下不论掩掉哪些位置，剩下位置的 loss 都是 $\ln V$**——因为均匀分布对任何子集都是均匀的。所以 label mask 错位、双重位移、pad 未掩，这个检查全部免疫。判定阈值 $8.764 \pm 0.05$。
4. $8.76405\in[2^3,2^4)$，该区间的 fp16 ULP 是 $2^3\cdot2^{-10} = 2^{-7} = 7.8125\times10^{-3}$。**在 fp16 里 8.764 和 8.768 是同一个数。** 而本周的 fixed-global-batch 等价判定要分辨 $10^{-4}$ 量级的 loss 差异，比 fp16 的分辨率细两个数量级。所以 loss、`log_softmax`、`cross_entropy` 必须在 fp32——这也正是 PyTorch 把它们放进 autocast fp32 列表的原因。

**0–4 分 rubric**

- 0：说不出 $\ln V$ 的来历，或把它当成经验值。
- 1：知道 $\ln 6400 = 8.764$ 但给不出推导。
- 2：第 1、2 问正确（含 bits 与 PPL 换算）。
- 3：加上第 3 问的"验接线不验掩码"，且能给出一个具体免疫例子。
- 4：以上全对，并给出第 4 问的定量论证（$2^{-7}$ vs $10^{-4}$，差两个数量级），且能说出"$\ln V$ 检查模型接线、D1/D2/D3 检查数据接线，两者不可互相替代"。

**常见误区**

- 认为 loss 从 8.764 开始就说明"数据没问题"。它只说明 logits 层接线没问题。
- 把 PPL 与 loss 混用而不注明底数，或忘了 $\ln2$ 这个换算。
- 认为"loss 在 fp32 里算"是性能考虑。它是分辨率考虑。

**追问**

- 第 0 步 loss 是 8.2 而不是 8.76，你会按什么顺序查哪三件事？
- 除了 $\ln V$，本周还有哪两个**可以从数据算出来的**锚点？它们各自能抓什么？

**定位**：`01_FOUNDATIONS.md` §3.2、§5.4、§9.2 第 4 行；`lab/scripts/run_numeric_ref.py`。

---

### M03-E-02

**关键评分点**

1. $\text{目标可优化}\Rightarrow\text{曲线好看}$，但 $\text{曲线好看}\not\Rightarrow\text{目标正确}$。交叉熵对**任何一个定义一致的目标**都是良定的可优化目标；SGD 最小化的是你实际写下来的那个东西。曲线的**形状**（平滑、单调下降）由可优化性决定；曲线的**渐近值**由你选中的那个目标的条件熵决定。而你对"正确目标的条件熵应该是多少"通常**没有独立估计**——这才是它能骗过人的根本原因。
2. 三个算例：
   - **A 双重位移**（数据集也做了一次位移）→ 实际最小化 $H(x_{t+2}\mid x_{\le t})$。跳一格预测同样可学，曲线一样平滑单调，只是收敛到**更高**的平台（自然语言上大致 +0.5–1.5 nats，`估算`）。
   - **B 掩码差一位**（忘了 `loss_mask[:, 1:]`）→ 计分位置整体平移一格，$n_{\text{label\_tokens}}$ **数量完全不变**，收敛值**几乎不动**。
   - **C pad 未掩**（`labels = input_ids` 直接用）→ loss **更低**，看起来更好。带数字：$T=512$、真实平均长度 200 token → 被计分位置中 pad 占比 $f = (511-199)/511 = 0.611$；pad→pad 是确定性的，训几百步后 $L_{\text{pad}}\approx0.005$；若真实 $L_{\text{real}} = 3.0$，则 $L_{\text{report}} = 0.389\times3.0 + 0.611\times0.005 = \mathbf{1.17}$，报出 PPL $e^{1.17} = 3.2$ 而真实是 $e^{3.0} = 20.1$。**一个 64M 模型在中文网页文本上报 PPL 3.2 是刺眼的——问题是很多人不知道该期待什么。**
3. 查不出来的是 **B（掩码差一位）**：D1（$n_{\text{label}} = n_{\text{nonpad}} - b$）成立，D2 的 $\rho$ 也在正常带内，因为它只是集合平移不改变计数。抓得住它的是 **D3 的第一条断言** `(labels[labels != -100] == input_ids[labels != -100]).all()`（钉死"labels 是 input_ids 的逐位置拷贝，只是部分被掩"这个契约），或者**生成测试**。
4. 因为任何"从前缀确定性地映射到某个 token"的任务都是**可记忆的**：预测 $x_{t+1}$ 是这样，预测 $x_{t+2}$ 也是这样。双重位移的模型同样能把那条序列压到 0。**有效检查是生成**：过拟合一条序列到 loss < 0.05 后，用它的前 10 个 token 做贪心解码——接线正确则逐 token 复现原序列；位移错一格则复现的是**偏移一格**的版本（跳过或重复一个 token）。这是一个**二值、无歧义、不需要参考曲线**的判据。
5. 一句可迁移的规律：**曲线类检查判优化，张量断言与行为检查判正确性。**

**0–4 分 rubric**

- 0：认为曲线平滑下降就说明训练管线正确。
- 1：知道曲线不可靠，但说不出形状与渐近值分别由什么决定。
- 2：给出三个算例与各自方向，指出 C 会让 loss 变好。
- 3：加上 C 的带数字算例，且答出 B 是两个不变量都查不出的那一档。
- 4：以上全对，且给出"过拟合到 0 无效 / 生成有效"的机制解释与二值判据，并主动总结出第 5 条规律。

**常见误区**

- 认为双重位移会让 loss 明显变差或不收敛。它只是平台高一点。
- 认为 `n_label_tokens` 数量对了就说明掩码对了。B 档正是计数不变而集合平移。
- 把 C 档的低 loss 解读成"模型学得好"。

**追问**

- 算例 C 里的 $f = 0.611$ 依赖"真实平均长度 200"这个假设。如果真实长度分布很宽，你会怎么改这个估计，让它成为一个可报告的量？
- 三个算例里哪一个在 SFT 阶段更常见、哪一个在 pretrain 阶段更常见？为什么？

**定位**：`01_FOUNDATIONS.md` §5.2、§5.3、§5.5、§6.3；`lab/src/mm_v100/data_contract.py`；`lab/scripts/check_data_contract.py`。

---

### M03-E-03

**关键评分点**

1. 每 micro-batch 每 rank 线上字节（ring）：DDP 做 1 次 all_reduce（负载 $4P$），$\frac{2(W-1)}{W}\cdot4P = 8P\cdot\frac{W-1}{W}$。FSDP `FULL_SHARD` 做 3 次单向通信（fwd all-gather、bwd all-gather、reduce-scatter），fp16-reduce 时三次负载都是 $2P$：$3\cdot\frac{W-1}{W}\cdot2P = 6P\cdot\frac{W-1}{W}$。
   $$\frac{\text{FSDP}_{\text{fp16-reduce}}}{\text{DDP}} = \frac{6P}{8P} = \mathbf{0.75},\qquad \frac{\text{FSDP}_{\text{fp32-reduce}}}{\text{DDP}} = \frac{2\cdot2P + 4P}{8P} = \frac{8P}{8P} = \mathbf{1.00}$$
   **$(W-1)/W$ 在分子分母完全抵消 → 这两个比值与 world_size 无关。** 它们只取决于"FSDP 三次单向 vs DDP 一次双向"和两边各自的 dtype。
2. 具体数（$W=8$，64M）：$t_{\text{DDP}} = 154\alpha + 447.39\,\text{MB}/\beta$，$t_{\text{FSDP}} = 189\alpha + 335.54\,\text{MB}/\beta$（154 = 11 桶×14，189 = 27 次×7）。FSDP 纯线上时间更短的条件：
   $$189\alpha + \frac{335.54}{\beta} < 154\alpha + \frac{447.39}{\beta}\iff 35\alpha < \frac{111.85\,\text{MB}}{\beta}\iff \boxed{\alpha < \frac{3.196\,\text{MB}}{\beta}}$$
   代入 $\beta=10$ GB/s → 临界 $\alpha = 320\ \mu$s；$\beta=25$ GB/s → 128 $\mu$s。单节点 NCCL 的集合通信启动延迟通常在 **5–30 μs** 量级（`估算`），比临界值小**一个数量级**。
3. **"慢在字节量"不成立。** FSDP 的线上字节比 DDP **少 25%**，ring step 只多 23%，按纯线上时间算 FSDP 应该更快。这个结论依赖的前提是：ring 算法、单节点、$\alpha$ 在 5–30 μs 量级、$\beta$ 不低于 PCIe Gen3 ×16 的实测量级。前提不成立（例如跨节点、或 $\alpha$ 实测到 300 μs 以上）则结论要重判——这正是 Day 0 wiring smoke 要测的。
4. 三个真正来源，各配一个开关：
   - **(a) 前向 all-gather 的暴露时间。** `forward_prefetch=False` 是默认值，unit $\ell$ 的 all-gather 要进入它的 forward 才发起；反向侧 `BACKWARD_PRE` 默认开着，所以**反向的 9 次有预取、前向的 9 次没有**。DDP 完全没这问题（通信全在反向，由梯度就绪事件触发）。开关：`forward_prefetch=True`。定量：$t_{\text{ag}}(\text{block})\approx12.91\,\text{MB}/\beta = 1.29$ ms（$\beta=10$ GB/s），要藏住需 $N > 2600$ token；WE1 的 $N=8192$ 是它的 3.1 倍 → **带宽上完全藏得住，藏不住只可能是没预取**。
   - **(b) 每-unit 的主机侧开销。** flat_param 的 unshard/reshard、`use_orig_params=False` 下的视图重建、per-unit 的 CUDA event record/wait、`limit_all_gathers=True` 的 rate limiter 同步。9 unit × 3 阶段 = 27 组簿记，全在主机侧；DDP 只有 11 次桶就绪回调。开关：`use_orig_params`、`limit_all_gathers`、以及"把 wrap 粒度调粗"。
   - **(c) 梯度累积的不对称。** DDP 的 `no_sync()` 免费（梯度本来就累加在 `param.grad` 上）；FSDP 在 `FULL_SHARD` 下的 `no_sync()` 必须把**未分片的**梯度留在卡上跨 micro-batch，多占 $2P$（fp16）或 $4P$，而且只省掉 reduce_scatter（27→18，省 33%），两次 all-gather 一次都省不掉。$A=4$ 时：DDP+`no_sync` 每 micro-batch 摊 111.85 MB / 2.75 次，FSDP 是 335.54 MB / 27 次——**字节 3 倍、次数 10 倍**。开关：`ShardingStrategy` 换 `SHARD_GRAD_OP`、或 DDP 侧加 `no_sync`。
5. **只有换 batch size 才能分离的是"每 step 固定的常数开销"（(b) 与集合通信启动延迟）。** 任何"每 step 固定"的开销都会被更大的 $b$ 摊薄；任何"与字节成正比"的开销不会。只在一个 batch size 下看比值，这两类完全不可区分——所以 $b$ 翻倍（16→32）后比值向 1 收敛是**决定性实验**。

**0–4 分 rubric**

- 0：直接说"FSDP 通信更多所以更慢"，不给字节账。
- 1：给出了 0.75 / 1.00 中的一个，但推导不完整或没说与 $W$ 无关。
- 2：两个比值与"与 $W$ 无关"的抵消论证都对。
- 3：加上临界不等式 $\alpha < 3.196\,\text{MB}/\beta$ 与"慢在字节量不成立"的判断。
- 4：以上全对，三个来源各配一个具体开关，且指出 $b$ 扫描是决定性实验并说清为什么（常数开销 vs 正比开销的可分离性）。

**常见误区**

- 把 27 > 11 直接当成"通信更多"。次数多不等于字节多；FSDP 的每次通信是单向且负载减半。
- 忘了 fp32-reduce 会把比值从 0.75 抬回 1.00——把 `reduce_dtype` 的选择当成纯数值问题，不记进通信账。
- 认为 `forward_prefetch` 默认是开的。

**追问**

- 如果实测 `forward_prefetch=True` 之后比值只改善了一点点，你的下一个假设是什么？用哪个实验证伪它？
- 在 64M 上 FSDP 只把每卡总量从 7.3 GB 降到 6.1 GB。那为什么本周还要练 FSDP？

**定位**：`01_FOUNDATIONS.md` §4.2、§4.3、§4.4、§4.5、§2.5；`lab/src/mm_v100/collectives.py`；`lab/scripts/byte_ledger.py --show-collectives`。

---

### M03-E-04

**关键评分点**

1. 不缩放时 fp16 的梯度窗口是 $[2^{-24},\,65504] = [5.96\times10^{-8},\,6.55\times10^{4}]$，而真实 LLM 的梯度分量绝大多数落在 $[10^{-9},10^{-2}]$——**窗口上半段完全浪费，下半段不够用**。乘上 $s$ 相当于把窗口整体左移 $\log_2 s$ 位：
   $$\left[\frac{2^{-24}}{s},\ \frac{65504}{s}\right]\xrightarrow{s=2^{16}}[9.09\times10^{-13},\ \mathbf{0.99951}]$$
   上界 **0.99951** 紧贴 1.0，而 MiniMind 的 `--grad_clip` 默认正是 **1.0**。这不是巧合：$2^{16}$ 的设计意图就是让 fp16 的溢出上限恰好落在"单位范数梯度"这个自然尺度上。把 `grad_clip` 改成 10 → 需要把上界推到 10 附近 → $s\approx65504/10 = 6550$ → `init_scale` 往**小**调到 $2^{13} = 8192$ 附近。
2. 四个行为后果：
   - **跳步时 `loss.item()` 照常有值**（loss 是前向算的，与优化器步不步无关）→ **跳步不会在 loss 曲线上留任何痕迹**，只让曲线"下降得慢"。这是"静默"的第一来源。
   - **跳步时学习率调度照常推进**（`get_lr` 以 micro-batch 计数为时钟）→ 连跳 500 步，lr 已按余弦衰减走了 500 步，参数一步没动。这让故障**越拖越难恢复**。
   - **跳步后 `optimizer.zero_grad()` 照常执行** → 那一组 micro-batch 的梯度被丢弃，算力白花。
   - **DDP 下各 rank 会一起跳步**（all_reduce 后 `param.grad` 逐位相同 → `found_inf` 相同）→ 副本仍然同步，不会额外触发跨 rank 不一致，于是连 I3 这条线索也是干净的。
3. **FSDP 下第 4 条不成立**：每个 rank 只持有自己的分片，`found_inf` 各不相同 → 普通 `GradScaler` 会让**一部分 rank 跳步、一部分不跳**，各 rank 的参数分道扬镳，而且不报错。替代品是 `torch.distributed.fsdp.sharded_grad_scaler.ShardedGradScaler`，它把 `found_inf` **跨 rank 归约**后再统一决定跳不跳。机制层面的理由是：`GradScaler` 的正确性依赖"所有副本看到同一个 `found_inf`"，DDP 靠梯度归约免费提供了这个前提，FSDP 不提供。
4. 健康跳步率：启动阶段若 $s\cdot\max|g| > 65504$，需要 $k = \lceil\log_2(s\max|g|/65504)\rceil$ 次减半，损失 $k$ 步（$k$ 通常 0–5）；稳态每 2000 步试一次翻倍，若溢出再损失 1 步，$N$ 步内约 $N/2000$ 步。
   $$\text{健康跳步率}\approx\frac{k + N/2000}{N}\xrightarrow{N=10^4,\ k=3}\approx 0.08\%$$
   报警阈值：**落地率 < 95% 报警；连续 50 次尝试落地率 = 0 且 scale 单调下降 → 判 B 档。** 补充：从 65536 减半到 1 要 16 次、到 $2^{-24}$ 要 40 次；**一个 scale 已经掉到 $10^{-7}$ 的 run 一定是坏的，且不是缩放能救的**——inf 来自前向本身。

**0–4 分 rubric**

- 0：说 `init_scale=65536` 是"经验值"或"随便取的 2 的幂"。
- 1：知道要左移窗口，但算不出上界 0.99951 或说不出与 `grad_clip` 的配套关系。
- 2：第 1 问完整（含改 `grad_clip` 后的调整方向与量级）。
- 3：加上四个行为后果，且指出前两条是"静默"的根源。
- 4：以上全对，且给出 FSDP 下的机制性理由（不只是"要用 ShardedGradScaler"）与第 4 问的量化判据。

**常见误区**

- 只答"防止梯度下溢"，说不出为什么恰好是 $2^{16}$。
- 认为跳步会让 loss 曲线出现尖峰或平台。全跳时曲线是**平坦**的，半跳时只是**慢一半到两倍半**，两种都不难看。
- 认为 FSDP 下用普通 `GradScaler` 会报错。它不报错，这正是问题所在。

**追问**

- `unscale_` 写在 `clip_grad_norm_` **之后**会发生什么？把等效阈值算出来。这个错误在曲线上看得见吗？
- 一个 run 的 scale 稳定在 $2^{12}$ 不动、落地率 100%，这健康吗？说明什么？

**定位**：`01_FOUNDATIONS.md` §3.4、§1.4、§6.4、§9.1 第 12/13 行；`lab/src/mm_rl/scaler_rules.py`。

---

### M03-E-05

**关键评分点**

1. $k_1 = -\log r$：**无偏**（$\mathbb{E}_{p_{new}}[-\log r] = \mathrm{KL}(p_{new}\Vert p_{ref})$），方差大，**可以为负**（单个样本上 $r>1$ 时）。$k_2 = (\log r)^2/2$：**恒非负**，**有偏**（是 KL 的二阶近似）。$k_3 = (r-1) - \log r$：**无偏且恒非负**，方差最小（Schulman 的低方差估计量）。
2. **无偏**：在 $p_{new}$ 下 $\mathbb{E}[r] = \mathbb{E}_{p_{new}}[p_{ref}/p_{new}] = \sum p_{ref} = 1$，故 $\mathbb{E}[r-1] = 0$，于是 $\mathbb{E}[k_3] = \mathbb{E}[-\log r] = \mathrm{KL}$。**前提**：期望是在采样分布 $p_{new}$ 下取的，且 $p_{ref}$ 是一个归一化的分布（求和为 1）。**非负**：令 $x = \log r$，$k_3 = e^{x} - 1 - x\ge0$ 对所有实数 $x$ 成立（$e^x\ge1+x$ 是 $e^x$ 的凸性/一阶泰勒下界）。**前提**：只需 $x$ 是实数，与分布无关——所以它是一条**逐样本**成立的恒等式，不是期望意义上的。
3. 用 `torch.expm1(log_r)`：当 $|x|$ 很小时 `exp(x) - 1` 里 $e^x\approx1$，两个几乎相等的数相减是灾难性抵消，有效数字全丢，算出来的 $k_3$ 甚至可能是**负数**。`expm1` 是为这个区间专门设计的。**因此：$k_3$ 出现负值 = 实现写错了，不是精度问题**——因为非负性是逐样本的代数恒等式，任何浮点精度都不该破坏符号（除非用了错误的写法）。这是一条"精确可判定"的自检。
4. DPO 初始 loss：policy 与 ref 逐位相同时 $\pi_{\text{logratios}} = \text{ref}_{\text{logratios}}$ → `logits_dpo` $\equiv 0$ → $\mathcal{L} = -\log\sigma(\beta\cdot0) = -\log(1/2) = \ln 2 = \mathbf{0.693147}$。**与 `beta` 无关**（$\beta$ 乘的是 0）。偏离时按顺序查：① chosen 与 rejected 的 mask 是不是由**同一个** encode 路径生成的；② ref 是不是 `deepcopy` 而不是"同 seed 重建"——重建在 CUDA 上不保证逐位相同，那样 $\ln2$ 会差在小数点后第三位，**看起来像数值问题，其实是方法问题**。

**0–4 分 rubric**

- 0：写不出三个估计量，或认为 $k_1$ 有偏。
- 1：三个式子对，性质说错一项以上。
- 2：三个式子与性质全对，$k_3$ 的无偏性证明正确。
- 3：加上非负性证明并说清两条证明的前提不同（一条是期望意义、一条是逐样本代数）。
- 4：以上全对，且答出 `expm1` 的理由与"负 $k_3$ = 实现问题"的判定逻辑，以及 DPO 的 $\ln2$ 与 `beta` 无关、两条检查的顺序与"deepcopy vs 重建"这个方法性陷阱。

**常见误区**

- 认为 $k_3$ 非负是"因为它是 KL 的估计，KL 非负"。KL 非负是期望意义的；$k_3$ 的非负是逐样本的，两者强度不同。
- 用 `torch.exp(log_r) - 1` 写 $k_3$ 然后把负值归咎于 fp16。
- 认为 DPO 初始 loss 依赖 `beta`。

**追问**

- 在 fp16 下 $k_2$ 和 $k_3$ 的相对偏差为什么会比 $k_1$ 更大？用"$\log r$ 的绝对误差"这一个量解释。
- 如果 policy 与 ref 还完全相同，$\mathrm{sign\_agreement}$ 与 $\mathrm{kl\_rel\_bias}$ 会算出什么？这是不是 bug？

**定位**：`07_RL_LOWPRECISION.md` §2 的 R2；`lab/src/mm_rl/kl.py`（模块 docstring 的恒等式与断言依据）；`lab/src/mm_v100/rl_numeric.py` 的 `dpo_loss` / `dpo_init_check`。

---

## Apply

### M03-A-01 —— **Gate 题**

> 这一题的评分方式与其他题不同：**每一项独立判对错，最后按项数给分**。允许 ±0.5% 的算术误差（手算），但**结构错误**（漏项、公式错、比值错）不允许。

**关键评分点**

1. **形状与参数量**
   - $d_h = 1024/16 = \mathbf{64}$；$d_{ff} = \lceil1024\pi/64\rceil\times64 = \lceil50.2655\rceil\times64 = 51\times64 = \mathbf{3264}$。
   - 单层：`q_proj` $1024\times1024 = 1{,}048{,}576$；`k_proj` $1024\times256 = 262{,}144$；`v_proj` $262{,}144$；`o_proj` $1{,}048{,}576$（注意力小计 **2,621,440**）；`q_norm`+`k_norm` $= 64+64 = 128$；两个 hidden 级 norm $= 2048$；FFN $3\times1024\times3264 = 10{,}027{,}008$。
     $$P_\ell = 2{,}621{,}440 + 10{,}027{,}008 + 2{,}048 + 128 = \mathbf{12{,}650{,}624}$$
   - $P = Vd + 12P_\ell + d = 8{,}388{,}608 + 151{,}807{,}488 + 1{,}024 = \mathbf{160{,}197{,}120}$（tie，`lm_head` 贡献 0）。
   - 常用常数：$2P = 320{,}394{,}240$ B $= 320.39$ MB；$4P = 640{,}788{,}480$ B $= 640.79$ MB；$8P = 1{,}281{,}576{,}960$ B；$16P = 2{,}563{,}153{,}920$ B $= 2.563$ GB。
2. **DDP 每卡字节账**
   | 项 | 公式 | MB |
   | --- | --- | --- |
   | param（fp32 master） | $4P$ | 640.79 |
   | grad（fp32） | $4P$ | 640.79 |
   | Adam $m,v$ | $8P$ | 1281.58 |
   | **漏记项 1：DDP bucket**（`gradient_as_bucket_view=False`） | $4P$ | 640.79 |
   | **漏记项 2：autocast fp16 权重缓存** | $2P_{\text{linear}}$ | 320.34 |
   | 静态小计 | | **3524.29** |
   | 激活 | 见第 4 点 | 4655.68 |
   | CUDA context + 碎片 `估算` | | ~1000 |
   | **每卡合计** | | **≈ 9.18 GB** |
   - $P_{\text{linear}} = 12\times(2{,}621{,}440+10{,}027{,}008) + 8{,}388{,}608 = 160{,}169{,}984$；×2 B = 320,339,968 B = **320.34 MB**。校验：$P - P_{\text{linear}} = 27{,}136 = 12\times2176 + 1024$，正好是全部 norm 参数（norm 走 fp32 路径，不生成 fp16 拷贝）。
   - 加分：指出 `gradient_as_bucket_view=True` 一个布尔参数省掉 640.79 MB。
3. **FSDP `FULL_SHARD` 每卡字节账**
   | 项 | 公式 | MB |
   | --- | --- | --- |
   | 分片 fp32 param | $4P/W$ | 80.10 |
   | 分片 fp32 grad | $4P/W$ | 80.10 |
   | 分片 fp32 Adam $m,v$ | $8P/W$ | 160.20 |
   | **常驻小计** | $16P/W$ | **320.39** |
   | 瞬时：当前 + 预取 unit 的 fp16 全量参数 | $2\times2P_\ell$ | 50.60 |
   | 瞬时：root unit 的 fp16 全量参数 | $2\times(8{,}388{,}608+1{,}024)$ | 16.78 |
   | 瞬时：反向中当前 unit 的 fp16 全量梯度 | $2P_\ell$ | 25.30 |
   | **峰值静态** | | **≈ 413.1** |
   | 激活 | | 4655.68 |
   | CUDA context `估算` | | ~1000 |
   | **每卡合计** | | **≈ 6.07 GB** |
   - 记忆点：$W=8$ 时 $16P/W = 2P$，FSDP 每卡常驻恰好等于整模型的一份 fp16 拷贝。
4. **激活**（$N = bT = 8\times512 = 4096$ token/卡/micro-batch）
   - 每层系数的一般式：$59 + 8\cdot\frac{d_{ff}}{d}$ B/(token·$d$)。MiniMind 默认 $d_{ff}/d = 3.1\overline{6}$ 给 84.333；**本题 $d_{ff}/d = 3264/1024 = 3.1875$，系数变成 $59 + 25.5 = \mathbf{84.5}$**。差 0.2%，但**必须指出它变了**——直接背 84.333 而不检查 $d_{ff}/d$ 的答案在这一项判错。
   - $\text{Act} = N\big(84.5\,dL + 12V\big) = 4096\times(84.5\times1024\times12 + 12\times8192) = 4096\times(1{,}038{,}336 + 98{,}304) = 4096\times1{,}136{,}640 = \mathbf{4{,}655{,}677{,}440}$ B $= 4.656$ GB。
   - 前提要写出来：fp16 AMP、mem-efficient 后端、**无激活重计算**；且这个系数**与 $T$ 无关**（表里没有 $[b,H,T,T]$ 项）。
5. **一个 optimizer step 的集合通信序列**（$A = 4$）
   - **DDP**：桶数 $n = 1 + \lceil(4P - 1\,\text{MiB})/25\,\text{MiB}\rceil = 1 + \lceil639{,}739{,}904/26{,}214{,}400\rceil = 1 + \lceil24.404\rceil = \mathbf{26}$。
     | 写法 | 类型 | 次数 | 每 rank 线上字节 | ring steps | 单 chunk |
     | --- | --- | --- | --- | --- | --- |
     | 每 micro-batch | `all_reduce` | 26 | $\frac{14}{8}\times640.79 = 1121.38$ MB | $26\times14 = 364$ | $26.214/8 = 3.28$ MB |
     | **当前训练循环（无梯度累积不同步）** | `all_reduce` | **104** | **4485.5 MB** | 1456 | 3.28 MB |
     | **加上累积不同步后** | `all_reduce` | **26** | **1121.38 MB** | 364 | 3.28 MB |
   - **FSDP `FULL_SHARD`**（按 block auto-wrap → 12 block unit + 1 root unit = **13 unit**）：
     | 阶段 | 类型 | 次数/micro-batch | 总负载 | 每 rank 字节 | ring steps |
     | --- | --- | --- | --- | --- | --- |
     | forward 进 unit 前 gather 参数 | `all_gather` | 13 | $2P$ = 320.39 MB | 280.34 MB | 91 |
     | backward 重新 gather（前向后已 reshard） | `all_gather` | 13 | 320.39 MB | 280.34 MB | 91 |
     | backward 梯度归约回分片（`reduce_dtype=fp16`） | `reduce_scatter` | 13 | 320.39 MB | 280.34 MB | 91 |
     | **合计/micro-batch** | | **39** | | **841.03 MB** | **273** |
     | **合计/optimizer step（$A=4$）** | | **156** | | **3364.1 MB** | 1092 |
     - 单 ring step chunk：block unit $2P_\ell/8 = 25{,}301{,}248/8 = 3.16$ MB；root unit $16{,}779{,}264/8 = 2.10$ MB。
     - 若 `reduce_dtype=fp32`：reduce_scatter 负载变 $4P$，每 rank 560.69 MB，合计 1121.37 MB/micro-batch —— 与 DDP **完全相等**，比值 1.00。
     - 校验：$841.03/1121.38 = \mathbf{0.75}$，与 $6P/8P$ 一致。
6. **能不能跑**：32 GiB = 34.36 GB。DDP ≈ 9.18 GB、FSDP ≈ 6.07 GB，**两个都跑得下**，都还剩三倍以上余量。**卡住的（如果要卡）是激活那一项**：静态在 DDP 下只有 3.52 GB，激活 4.66 GB 已经是它的 1.3 倍；把 $b$ 从 8 加到 48 时激活到 27.9 GB 才会顶到上限，而静态几乎不变。**推论：这个规模上省显存的杠杆是激活重计算，不是 FSDP。**

**0–4 分 rubric**

- 0：$d_{ff}$ 或 $P$ 算错（结构性错误），后面全部作废。
- 1：$P$ 正确，但 DDP / FSDP 的四格账缺两项以上（典型是漏 bucket 与 fp16 权重缓存），或通信序列只写了次数没写字节。
- 2：$P$、两份四格账、激活估计式都正确，通信序列写出了次数与每 rank 字节，但漏了 ring step / chunk，或没区分"当前写法"与"加累积不同步"。
- 3：六项全部答出且数字正确（允许 ±0.5% 手算误差），含 0.75 比值的校验。
- 4：以上全对，**并且**：① 主动指出 $d_{ff}/d$ 变了导致激活系数从 84.333 变成 84.5；② 指出 $16P/W$ 在 $W=8$ 时恰等于 $2P$；③ 第 6 点给出"杠杆在激活重计算"的推论；④ 每个 `估算` 项都标了标签，没有把 CUDA context 说成精确值。

**常见误区**

- **漏 DDP 的 $4P$ bucket**（默认 `gradient_as_bucket_view=False`）→ 静态少算 640.79 MB，18.2%。
- **漏 autocast 的 fp16 权重缓存** → 少算 320.34 MB；或者反过来，误以为 AMP 会把 param/grad 变成 fp16，把静态算成 $8P$。**autocast 不改参数 dtype**，静态恒为 $16P$。
- **直接搬 84.333** 而不检查 $d_{ff}/d$。
- **unit 数记成 12**（忘了 root unit）→ 通信次数算成 36 而不是 39。
- 把 FSDP 的三次通信都按双向 all_reduce 的 $\frac{2(W-1)}{W}$ 算 → 比值算成 1.5 而不是 0.75。
- 把 32 GB 当成 $32\times10^9$ B 而不是 32 GiB。

**追问**

- 如果把 `tie_word_embeddings` 改成 `False`，上面哪几个数字会变？变多少？FSDP 的 wrap 策略要不要跟着改？
- 把 $b$ 从 8 提到 32、$A$ 从 4 降到 1（全局 batch 不变），第 5 点的两张表各有哪几个数字会变、哪几个不变？这个对比说明了什么？

**定位**：`01_FOUNDATIONS.md` §1.2、§2.1–§2.6、§4.2、§4.3；`lab/src/mm_v100/ledger.py` 的 `hand_ledger()`；命令 `python lab/scripts/byte_ledger.py --mode fsdp --world-size 8 --hand-only --show-collectives`。

---

### M03-A-02

**关键评分点**

1. 约束：$G = W\cdot A\cdot b$ → $A\cdot b = G/W = 768/8 = \mathbf{96}$。整数解：$(b,A)\in\{(8,12),(12,8),(16,6),(24,4),(32,3),(48,2),(96,1)\}$（还有 $(6,16)$、$(4,24)$ 等）。
2. 每 token 激活 $= 84.333\,dL + 12V = 84.333\times768\times8 + 12\times6400 = 518{,}144 + 76{,}800 = 594{,}944$ B $\approx0.595$ MB。
   $$b\times512\times594{,}944\le5.0\times10^{9}\iff b\le\frac{5.0\times10^{9}}{304{,}611{,}328} = 16.41\ \Rightarrow\ \boxed{b\le16}$$
   满足的解：$(8,12)$、$(12,8)$、$(16,6)$（以及更小的 $b$）。
3. 选 $(b,A) = (\mathbf{16},\mathbf{6})$。理由必须覆盖三条随谁变：
   - **激活字节 $\propto b$**（也 $\propto T$，但 $T$ 固定）：$b=16$ 用满 4.87 GB 预算里的 4.87 GB，$b=8$ 只用一半，浪费。
   - **每 optimizer step 的集合通信次数 $\propto A$**（当前训练循环没有梯度累积不同步）：$A=6$ → $11\times6 = 66$ 次 all_reduce；$A=12$ → 132 次。字节量同比翻倍。
   - **每 step 的固定主机侧开销（簿记、kernel launch、集合通信启动延迟）$\propto A$ 而与 $b$ 无关**：更大的 $b$ 把这些常数摊薄。这正是 Foundations 里 E4 实验的原理。
   三条都指向"在激活预算内取最大的 $b$"。
4. **会变，至少要重新论证。** 加上梯度累积不同步之后，每个 optimizer step 的 all_reduce 从 $11A$ 降到 **11 次（与 $A$ 无关）**，第二条理由消失。剩下第一条（激活 $\propto b$）与第三条（主机侧固定开销 $\propto A$）。第三条仍然偏向大 $b$，但强度弱得多——通信/计算比从 0.39–0.91 掉到 0.05–0.11，重叠质量不再是瓶颈。此时如果有别的理由偏好小 $b$（例如要留显存给激活重计算之外的东西、或要更细的梯度噪声粒度），$(8,12)$ 也可以接受。**高分答案会明确说"结论方向不变但论据强度变了"，而不是简单答"不变"。**

**0–4 分 rubric**

- 0：约束式写错（例如把 $G$ 写成 $A\cdot b$ 漏了 $W$）。
- 1：约束对，激活上限算错或没给算式。
- 2：前两问全对，第 3 问给了选择但理由只覆盖一条。
- 3：三条随谁变全部覆盖。
- 4：以上全对，且第 4 问指出"方向不变、论据强度变了"，并把通信/计算比从 0.39–0.91 降到 0.05–0.11 这个量级说出来。

**常见误区**

- 把 $G$ 理解成 token 数而不是序列数。本周 $G$ 的单位是**序列**。
- 认为"$A$ 越大越省显存"。$A$ 不影响激活峰值（激活由单个 micro-batch 的 $b\cdot T$ 决定），只影响通信次数。
- 只用一条理由（通常是"激活"）就下结论。

**追问**

- 如果 $T$ 从 512 改成 340（MiniMind 默认），第 2 问的 $b$ 上限变成多少？$G$ 的约束式要不要改？
- $b$ 和 $A$ 都不变、只把 $W$ 从 8 降到 4，全局 batch 会怎样？要保住 $G$ 你会动哪个？

**定位**：`01_FOUNDATIONS.md` §0.3（WE0/WE1/WE2）、§2.3、§4.2、§4.5 的 E4、§4.6。

---

### M03-A-03

**关键评分点**

1. 保持"**已消费的样本数**"不变（$\text{step}\cdot bW$）：$\text{step}' = \text{step}\cdot\frac{bW}{b'W'}$。$b' = b = 16$ → $\text{step}' = 1200\times\frac{8}{4} = \mathbf{2400}$。
2. 保持 $G = WAb$：$W'A'b' = WAb$ → $4\cdot A'\cdot16 = 8\cdot4\cdot16$ → $A' = \mathbf{8} = 2A$。验证：$G = 8\times4\times16 = 512$；$G' = 4\times8\times16 = 512$ ✓。
3. 只改 $W$ 不改 $A$：$G$ 从 512 塌到 **256**；$\text{step}' = 2\,\text{step}$ 仍然对齐了数据位置，但 optimizer step 数变成 $2\text{step}/A = 2\times$ 原来的。**Adam 在同样的数据上做了两倍的更新，每次用一半的 batch。** 这不是"恢复"，是换了一个训练配方。曲线看不出来是因为：更小的 batch + 更多的 step 同样是一个可优化的配置，曲线照样平滑下降（`01_FOUNDATIONS.md` §5.3 的原理）。
4. 学习率相位**不会偏**（在 $b'=b$ 的前提下）。`get_lr(epoch*iters + step, epochs*iters, lr)` 用 micro-batch 计数作时钟，而 $\text{iters} = \lceil|D|/(bW)\rceil\propto1/W$。8→4 时 $\text{iters}' = 2\,\text{iters}$、$\text{step}' = 2\,\text{step}$，比值 $\text{step}'/\text{iters}' = \text{step}/\text{iters}$ 不变 → 相位自动保持。**前提**：$b' = b$；若 $b$ 也改了，这条就不成立，要重新推。
5. `FULL_STATE_DICT`（unflattened & not sharded）→ **能**换成任意 $W'$，rank 0 峰值 $4P = 255.65$ MB，本模型完全放得下，逻辑最简单。`SHARDED_STATE_DICT`（unflattened but sharded）→ **能**，配 `torch.distributed.checkpoint` 的 `save_state_dict`/`load_state_dict` 做 reshard，各 rank 写自己那份、无 all_gather。`LOCAL_STATE_DICT`（no transformation）→ **不能**，key 是 `_flat_param`，长度是 $W$ 的函数。**本周练 `SHARDED_STATE_DICT` 而不是最省事的 `FULL_STATE_DICT`，因为它是唯一能扩展到 7B/27B 的路径**——27B 上 `FULL_STATE_DICT` 的 rank 0 峰值是 108 GB，装不下。用一个 64M 模型练一条为 27B 准备的路径。
   优化器侧：必须用 `FSDP.optim_state_dict(model, optim)` / `FSDP.optim_state_dict_to_load(...)`；**普通的 `optimizer.state_dict()` 在 FSDP 下换 $W$ 恢复一定失败**（key 是 flat_param 的下标）。
6. 还必须恢复的三样：
   - `GradScaler` 的 `scale` 与 `growth_tracker`——不恢复则从 65536 重新搜索，前若干步白跳。
   - `DistributedSampler` 的 `epoch`（`set_epoch`）与 epoch 内偏移——不恢复则数据顺序变了，"接着训"变成"重新采样"。
   - Adam 的 `exp_avg` / `exp_avg_sq` **以及每参数的 `step` 计数**（偏差修正 $\hat m = m/(1-\beta_1^t)$ 用的就是它）。
7. 判据组合：**8→4 reshard 用 R2 + R3 + R4。**
   - **R2（充分）：5 步重放等价。** $\max_i|\ell_i-\ell_i'|/\ell_i < 10^{-4}$。跨 $W$ 时改成"从 reshard 后的状态跑 5 步"与"从 `FULL_STATE_DICT` 路径恢复到同一个 $W'$ 再跑 5 步"对比。
   - **R3（必要）：逐 FQN 的优化器状态范数**，$\big|\|m_{\text{fqn}}\|_2^{\text{after}} - \|m_{\text{fqn}}\|_2^{\text{before}}\big|/\|m\|_2^{\text{before}} < 10^{-6}$。$\|\cdot\|_2$ 对分片方式不变，所以跨 $W$ 可比；**逐 FQN 而不是全局求和**，是为了抓住"参数之间错位"（全局范数对这种错误免疫）。
   - **R4（必要）：`scaler.get_scale()` 与 Adam 每参数 `step` 在前后逐位相等。**
   - **R1（落带判据）只是必要**：一个把优化器状态丢光的恢复，参数完全正确，第一步 loss 必然落在带内，$m,v$ 归零的影响要 50–200 步才显形。

**0–4 分 rubric**

- 0：`step'` 的换算方向搞反（写成 $\text{step}/2$）。
- 1：第 1、2 问对，但说不清"只改 $W$ 不改 $A$"的后果。
- 2：前 4 问全对（含 lr 相位的推导与它的前提）。
- 3：加上三种 `StateDictType` 的判定与必须一起恢复的三样。
- 4：以上全对，且第 7 问给出 R2 充分 / R1·R3·R4 必要的分层，并说出本周选 `SHARDED_STATE_DICT` 的理由是"练一条为 27B 准备的路径"（附 108 GB 这个数）。

**常见误区**

- 认为改了 $W$ 就自动改了 $A$。MiniMind 的 `lm_checkpoint` 只按 $\text{step}' = \text{step}\cdot W_{\text{saved}}/W_{\text{cur}}$ 换算，**只覆盖了数据位置这一行**，且假设 $b'=b$。
- 用 `optimizer.state_dict()` 保存 FSDP 优化器状态。
- 只用 R1（loss 落在带内）就判定恢复成功。

**追问**

- 用 `--from_weight pretrain` 而不是 `--from_resume 1` 续训会同时做哪两件坏事？曲线上看得见吗？
- R2 的 $10^{-4}$ 这个阈值怎么推出来的？如果实测到 $10^{-2}$，你的第一个假设是什么？

**定位**：`01_FOUNDATIONS.md` §8.1–§8.4、§9.1 第 10/11/17 行；`lab/src/mm_v100/ckpt_reshard.py`；命令 `python lab/scripts/reshard_check.py --save-nproc 8 --load-nproc 4 --format sharded`。

---

### M03-A-04

**关键评分点**

1. mem-efficient 后端每 token $= 84.333\,dL + 12V = 518{,}144 + 76{,}800 = \mathbf{594{,}944}$ B。每条序列（$T=1024$）$= 1024\times594{,}944 = 609{,}222{,}656$ B。
   $$b\le\frac{24\times10^{9}}{609{,}222{,}656} = 39.40\ \Rightarrow\ \boxed{b_{\max} = 39}$$
2. math 后端额外物化 $[b,H,T,T]$ 的 scores 与 softmax 输出。摊到每 token（$N = bT$）是 $HT$ 个元素/张量；保存两个这样的张量，且 autocast 把 softmax 提到 fp32，粗算 fp16 两份 + fp32 两份：
   $$\Delta = HT(2+2+4+4) = 12HT\ \text{B/(token·层)}\xrightarrow{H=8,T=1024}98{,}304\ \text{B}\times L{=}8 = \mathbf{786{,}432}\ \text{B/token}$$
   总计 $594{,}944 + 786{,}432 = 1{,}381{,}376$ B/token；每条序列 $1024\times1{,}381{,}376 = 1{,}414{,}529{,}024$ B。
   $$b\le\frac{24\times10^{9}}{1{,}414{,}529{,}024} = 16.97\ \Rightarrow\ \boxed{b_{\max} = 16}$$
3. 比值 $39/16\approx\mathbf{2.44}$。**$T$ 从 512 换成 1024 会让这个比值变大**（在 $T=512$ 时额外项是 $12\times8\times512\times8 = 393{,}216$，总 988,160，比值 $988{,}160/594{,}944 = 1.66$；$T=1024$ 时是 $1{,}381{,}376/594{,}944 = 2.32$）。机制：mem-efficient 下每 token 系数里**没有任何 $[b,H,T,T]$ 项，所以它与 $T$ 无关**；math 的额外项是 $12HT$，**正比于 $T$**。两者相除自然随 $T$ 增长。
4. 诊断规律：**mem-efficient 下把 $T$ 翻倍，激活应该恰好翻倍（线性，因为 $N = bT$ 翻倍而每 token 系数不变）。如果观察到激活增长明显超过 2 倍（趋近 4 倍），就该怀疑走的是 math 分支**——因为 math 的额外项对 $T$ 是二次的（$N\cdot12HT\propto bT^2$）。这是一个不需要读 traceback 就能做的判断。

**0–4 分 rubric**

- 0：不知道两个后端的激活差在哪，或把差异说成常数倍。
- 1：算出了 mem-efficient 的上限，math 的额外项写不出表达式。
- 2：两个上限都算对（允许 ±1 的取整差异）。
- 3：加上比值随 $T$ 的走向与机制解释（$T$ 无关 vs $\propto T$）。
- 4：以上全对，且给出第 4 问的可执行诊断规律（线性 vs 二次），并说明这条规律为什么比读 traceback 更早可用。

**常见误区**

- 把 $[b,H,T,T]$ 记成 $[b,T,T]$（漏 head 数）→ 额外项小 8 倍。
- 只算 fp16 的一份 scores（$2HT$）而漏掉 softmax 输出与 autocast 的 fp32 提升 → 额外项小 6 倍。
- 认为 mem-efficient 的每 token 系数也随 $T$ 变。它不随（这是本周的一个关键结构事实）。

**追问**

- 如果开了全层激活重计算，第 1、2 问的两个上限各变成多少？math 的劣势会被抹平吗？
- math 后端在本周有一个不可替代的用途，是什么？这个用途和它的激活代价之间的冲突，本周是怎么处理的？

**定位**：`01_FOUNDATIONS.md` §2.3、§2.5、§7.3、§9.1 第 8 行；`lab/scripts/run_numeric_ref.py --sdpa-backend`。

---

## Debug

### M03-D-01 —— **Gate 题**

**关键评分点**

1. 三档（`lab/src/mm_v100/faults.py` 的 mode 名）：
   - **A `label_shift`（label mask 错位）**：dataset 侧多做一次位移 / `loss_mask` 忘了切 `[:, 1:]` / `ignore_index` 用了 0 而不是 $-100$ / SFT 的 prompt 未掩。目标被换成了另一个"也能优化"的目标。
   - **B `scaler_stuck`（GradScaler 持续跳步）**：每一次 optimizer step 尝试都检出 inf/NaN → `optimizer.step()` 一次都不执行 → 参数冻结，但前向照做、loss 照打、lr 照走。
   - **C `ddp_no_sync`（DDP 梯度未同步）**：autograd hook 没注册在这条图上（走了 `model.module(...)`）/ `no_sync()` 覆盖了累积组的最后一个 micro-batch / rank 间控制流分歧 / 桶等不到就绪。每个 rank 独立优化自己那 $1/W$ 的数据，有效全局 batch 从 $Wb$ 塌到 $b$。
2. 不变量：
   - **A → I1 数据契约比** $(n_{\text{label\_tokens}},\ n_{\text{nonpad}},\ \rho)$ 与 D3 的两条断言。对象是 **dataloader 输出的两个 int64 张量本身**，位置在**模型前向之前**，不涉及浮点精度。**唯一识别性的来源**：I1 与 I2/I3 在因果上完全正交——B 只动 `GradScaler`、C 只动梯度通信，两者都不触碰 dataloader 的输出；反过来 A 不改变 scale 轨迹、也不破坏跨 rank 一致性。
   - **B → I2 缩放器状态**（`scaler.get_scale()` 与滑动窗口内的"已落地 step 数 / 尝试数"）**配合 I4 跨时间参数移动** $h(\theta_t) - h(\theta_{t-k})$。I4 用 **fp64** 累加。位置在 `scaler.step(); scaler.update()` 前后各取一次。
   - **C → I3 跨 rank 梯度一致性**。对象是**梯度张量**（不是参数，为的是把 DDP 的问题和优化器的问题分开：梯度相同而参数不同说明优化器状态或 lr 在 rank 间不一致，是另一类故障）。位置在第一个 optimizer step 之后，跑一次即可。
3. 最小检查：
   - **A（CPU 即可，不需要 GPU、不需要模型）**
     ```python
     x, y = next(iter(loader))
     assert (y[y != -100] == x[y != -100]).all(), "labels 被额外位移了"
     assert (y[x == pad_id] == -100).all(), "pad 未被掩掉"
     print((y[:, 1:] != -100).sum().item(), (x != pad_id).sum().item())   # pretrain 应满足 n_lab == n_nonpad - b
     ```
     判据：两条断言过 + $n_{\text{lab}} = n_{\text{nonpad}} - b$（pretrain）；SFT 看 $0<\rho<1$ 且批间稳定。
   - **B**
     ```python
     before = h_params()                                   # fp64 参数校验和
     scaler.step(optimizer); scaler.update()
     moved = abs(h_params() - before) > 0                   # I4
     landed_ratio = n_landed / n_attempted                  # I2
     ```
     判据：落地率 < 95% 报警；**连续 50 次尝试落地率 = 0 且 scale 单调下降 → 判 B**。
   - **C**：见第 5 点（必须逐元素，不能用范数）。判据：健康 DDP 下该量**恒等于 0.0**，任何非零即判 C。
4. loss 曲线与报错：
   | | 曲线 | 报错 |
   | --- | --- | --- |
   | A | 平滑单调下降，只是平台高一点或低一点 | 无 |
   | B（全跳） | **平坦**，在中断时刻的值附近抖动 | 无 |
   | B（半跳） | 平滑下降，只是**慢一半到两倍半** | 无 |
   | C | 平滑下降，rank 0 上看起来完全正常 | 无 |
   **loss 曲线形状对三档都无判别力**，这正是"静默"的定义。
5. **为什么"比较各 rank 的梯度范数"只是必要条件**（本题题眼）：
   - **范数是一个把整个张量压成一个标量的聚合量。** 两个方向完全不同的梯度向量可以有几乎相同的范数——尤其在 $6.4\times10^{7}$ 个分量上，各 rank 看的是同分布数据的不同分片，**幅度统计会集中（大数定律），方向不会**。范数恰好是那个会集中的统计量。
   - **all_reduce 的语义保证的是逐元素相等**（归约后每个 rank 的梯度**逐位相同**，这是 DDP 保持副本同步的前提）。判据必须建立在被保证的那个量上，而不是它的某个聚合投影。所以正确的比较在**元素级**：`lab/src/mm_v100/faults.py` 的 I3 用的是**逐元素梯度指纹的最大相对偏差**，健康 DDP 下**恒等于 `0.0`**——这是一个"精确为 0"型断言，比任何区间判断都可靠。
   - **在 64M 这样的小模型上，用范数做判据最可能的失败形式是「漏报」（假阴性）**：`ddp_no_sync` 下各 rank 的梯度范数只相差约 $3.6\times10^{-7}$ 的相对量，落在你为 fp32 噪声留的任何容差之内，于是你会得出"已同步"的结论，而实际上没同步。**误报几乎不可能**——健康时逐位相同，范数必然也相同。所以这是一个**只朝一个方向错**的坏判据，而且错的方向正是最危险的那个。
6. **必须用 fp64 的是 I4（以及 I3 若用校验和形式时的那个 $h$）**：对 $6.4\times10^{7}$ 个数做 fp32 求和，相对误差约 $\sqrt{n}\cdot2^{-24} = 8000\times5.96\times10^{-8} = \mathbf{4.8\times10^{-4}}$。而 B 档要判定的是"参数动没动"这种可能只有 $10^{-6}$ 量级的位移——fp32 的求和噪声比信号大两个数量级，会把 I4 变成一台**假阴性机器**。逐参数张量用 fp64 二范数还有第二个好处：能定位到具体是哪个张量偏了。
7. 第四种故障：**采样器未分片**（`DataLoader` 没传 `sampler=DistributedSampler(...)`，或没有每 epoch `set_epoch`）。各 rank 拿到**相同的数据** → 即使没同步，各 rank 梯度也相同 → **I3 = 0**，C 依然成立却检不出来。识别靠一对：**(I3 = 0 ∧ I5 = 0)** —— 各 rank loss 逐位相同只可能是同一批数据。此时"是否同步"已经无从区分，必须先修采样器。

**0–4 分 rubric**

- 0：三档说不全，或把判据建在 loss 曲线上。
- 1：三档名字与机理对，不变量给不全或给错对象（例如 A 档去查梯度）。
- 2：三档的不变量与最小检查都对，含 loss 曲线无判别力这一点。
- 3：加上第 5 问的"范数只是必要"论证与第 6 问的 fp64 定量理由。
- 4：以上全对，且答出第 5 问的**漏报方向性**（只朝一个方向错）与第 7 问的第四种故障及其特征对；并明确说出 I3 是"精确为 0"型断言、比区间判断可靠。

**常见误区**

- **用梯度范数或参数范数做 C 档判据。** 这是本题最重要的扣分点。
- 把 I4 用 fp32 算。
- 认为 I3 ≠ 0 是 C 的必要条件（它只是充分）。
- 认为 A 档能靠"loss 平台偏高"看出来。你没有"正确平台"的独立估计。

**追问**

- 三档同时踩到两个坑时，`classify()` 会怎么报？为什么把全部命中都列出来比只报第一个更好？
- 如果没有多卡环境，三档里哪几档能在单机 CPU 上复现？用哪个脚本？

**定位**：`01_FOUNDATIONS.md` §6.1–§6.6、§9.3；`lab/src/mm_v100/faults.py` 的 `observe()` / `classify()` / `cross_rank_grad_delta()`；命令 `python lab/scripts/run_faults.py --modes none,label_shift,scaler_stuck,ddp_no_sync --nproc 2`。

---

### M03-D-02 —— **Gate 题**

**关键评分点**

1. **判定：B 档 `scaler_stuck`（GradScaler 持续跳步）。**
2. **用的那一对是（I2 异常 ∧ I4 = 0）**，即"落地率恒为 0 且 scale 单调下降"配上"参数一步没动"。
   - I4 = 0 的意思是**参数一步没动**。能造成这个的只有三类：① 优化器没被调用（B）；② lr 恒为 0；③ 参数被 `requires_grad_(False)` 冻结。
   - 后两类的 **I2 是正常的**：scale 会稳定在 65536、落地率 100%（`scaler.step()` 照常执行，只是更新量为 0 或没有梯度）。
   - 所以 (I2 异常 ∧ I4 = 0) 把 ② ③ 全排除，唯一剩下 B。**充分。**
   - 辅助确认：I1 满足 D1（每一行 $n_{\text{lab}} = n_{\text{nonpad}} - 16$，$b=16$）→ 排除 A；I3 恒为 0（逐元素指纹）→ 无 C 的证据；I5 ≠ 0 → 各 rank 看的是不同数据，采样器正常。
3. **两列各自单独看只是必要条件**：
   - **`scale` / `landed/att`（I2）单独**：健康 run 在**启动阶段本来就会连续减半 $k$ 次**（$k = \lceil\log_2(s\max|g|/65504)\rceil$，通常 0–5），而且每 2000 步的翻倍试探也会偶发一次跳步。"scale 下降过"是 B 的必要条件，不是充分条件；**把它当判据会在每一个正常 run 上误报**。
   - **`I4 = 0` 单独**：lr 恒为 0（调度器写错、warmup 除零）或参数被冻结（迁移学习里忘了解冻）同样给 I4 = 0，而那两种的 I2 完全正常。
4. **`scale` 列区分出的两个子情形**：
   - **① 缩放溢出**：$s\cdot\max|g| > 65504$。减半几次就解决，`scale` 会**稳定在某个 $2^k$ 上**，落地率随后恢复。**这是健康的自适应，不是故障。**
   - **② 前向本身产生 inf/NaN**：不论 $s$ 多小 `found_inf` 恒真 → `scale` 从 65536 一路**单调减半、永不恢复**。
   - **这份日志属于 ②**。判据：step 1 是 $2^{16}$，step 80 是 $1.084\times10^{-19} = 2^{-63}$，正好是 **79 次连续减半、零次恢复**；而从 65536 减到 1 只要 16 次、减到 $2^{-24}$ 只要 40 次。**一个 scale 已经掉到 $10^{-19}$ 的 run 一定是坏的，且不是缩放能救的。**
5. 另外两档在这张表上的差异（各一行）：
   - **A**：`n_lab` 与 `n_nonpad` 的关系被破坏（不再是 $n_{\text{nonpad}} - 16$），或 D3 断言失败；而 `scale`、`landed/att`、`I4` 全部正常。
   - **C**：**`I3` 列不再恒为 0**（逐元素指纹出现非零相对偏差）；而 `scale`、`landed/att`、`I4`、`n_lab` 全部正常，loss 在 rank 0 上平滑下降。
6. **下一个单一测量**（放在 `scaler.scale(loss).backward()` **之前**）：
   ```python
   ok_logits = torch.isfinite(res.logits).all().item()
   ```
   - **`False` → 前向本身已经 inf/NaN**，缩放救不了。修复方向是数值/模型侧：查 RMSNorm 是否在 fp32 域算（$\bar x > 9.235$ 时 fp16 的 $\sum x^2$ 溢出）、查有没有在 fp16 域内做 `exp`（$x>11.09$ 即 inf）、查输入数据里有没有异常 token 或全 pad 的样本。
   - **`True` → 是纯缩放问题**，本应自愈；不自愈就查 `unscale_` 与 `clip_grad_norm_` 的**顺序**（clip 写在 unscale 之前等价于把阈值缩成 $1.0/65536 = 1.53\times10^{-5}$），或查有没有自定义 hook 在梯度上写入了 inf。
7. **`lr` 列的额外后果**：跳步时**学习率调度照常推进**（`get_lr` 以 micro-batch 计数为时钟）。日志里 lr 已经从 5.000e-4 衰减到 4.842e-4，走完了 80 步的余弦相位，**而参数一步没动**。所以即使现在修好，也不是"从第 1 步继续"——学习率相位已经被消耗掉了，必须连同调度器一起重置或从 checkpoint 重来。这让 B 档**越拖越难恢复**。

**0–4 分 rubric**

- 0：判成 A 或 C，或说"看不出来"。
- 1：判对是 B，但只用了 `scale` 这一列（不充分）。
- 2：判对且用了 (I2 ∧ I4) 这一对，能说出它为什么充分。
- 3：加上第 3 问的两个反例（健康启动 / lr=0 与冻结）与第 4 问的两个子情形判定。
- 4：以上全对，且给出第 5 问的两档差异行、第 6 问的单一测量与两个分支、第 7 问的 lr 相位后果。**限时 20 分钟内完成即视为 Gate 的跨场景 debug 项通过。**

**常见误区**

- 看到 loss 不下降就判 A（"数据有问题"）。A 的曲线是**平滑下降**的，不是平坦的。
- 看到 `scale` 下降就判 B 而不看 I4。这会在每一个正常 run 的前几步误报。
- 看到 I3 = 0 就宣布"DDP 没问题"。I3 = 0 不能排除 C（各 rank 同数据时 C 也给 0）；这里排除 C 靠的是 I5 ≠ 0 加上 I4 = 0 已经定位到 B。
- 忽略 `n_lab` / `n_nonpad` 两列，不做 D1 校验就断言"数据没问题"。

**追问**

- 这份日志里 `I3` 用的是逐元素指纹。如果换成"各 rank 梯度范数之差"，这张表上哪一列会失去判别力？失效的方向是漏报还是误报？
- 如果 `ok_logits` 是 `True`，你会用什么最小实验把"`unscale_`/`clip` 顺序错"和"自定义 hook 写入 inf"分开？

**定位**：`01_FOUNDATIONS.md` §3.4、§6.2、§6.4、§6.6、§9.3；`lab/src/mm_v100/faults.py` 的 `classify()`。

---

### M03-D-03

**关键评分点**

1. 手算：按 `ModuleWrapPolicy({MiniMindBlock})` auto-wrap → **8 个 block unit + 1 个 root unit = 9 unit**。`FULL_SHARD` 每 unit 每 micro-batch 三次通信（forward 前 all-gather 参数、backward 重新 all-gather、backward reduce-scatter 梯度）→ $9\times3 = \mathbf{27}$。
2. 观测值 3 说明 **`auto_wrap_policy` 根本没生效，整个模型退化成一个 root unit**（$1\times3 = 3$）。症状链：
   - 策略没匹配到任何子模块 → 没有子 unit → 只剩 root FSDP unit；
   - `FULL_SHARD` 仍然分片，但**前向要一次性 all-gather 整个模型的 fp16 参数**（$2P = 127.82$ MB 一次到位），且这份全量参数要活到该 unit 的 forward 结束才 reshard；反向再一次性 all-gather 一遍、一次性 reduce-scatter 一遍；
   - 于是瞬时峰值从"当前+预取的两个 block（29.50 MB）+ root（9.83 MB）+ 一层 fp16 梯度（14.75 MB）"变成"全模型 fp16 参数 127.82 MB + 全模型 fp16 梯度 127.82 MB" → `mem_ratio` 从预期的约 0.84 抬到 0.93；
   - 同时**没有任何逐层重叠的机会**（只有 3 次大通信，每次都是一个同步点），主机侧也没有 27 组簿记去填流水 → `step_time_ratio` 恶化到 2.4；
   - **全程不报错**，loss 曲线平滑（数学上 FSDP 与 DDP 等价，退化只影响放置与调度）。
3. 第一个检查点：**打印 FSDP unit 数**，期望 **9**。
   ```python
   print(len(FSDP.fsdp_modules(model)))     # 期望 9；打出 1 即确认 auto_wrap_policy 没生效
   ```
   等价做法：`python lab/scripts/byte_ledger.py --mode fsdp --world-size 8 --n-units 9 --show-collectives` 打出手算列，再和实测的 `collectives_total` 对账。
4. 两种常见触发：
   - **`ModuleWrapPolicy({MiniMindBlock})` 里的类对象不是同一个**：`MiniMindBlock` 从另一条 import 路径导入，`isinstance` 判定失败。**这一种完全静默**——策略是一个可调用对象，"没有任何模块匹配"是一个合法结果，FSDP 不会抱怨；它只会安静地把整个模型当成一个 unit。
   - **`size_based_auto_wrap_policy` 的 `min_num_params` 设得比 $P_\ell = 7{,}374{,}528$ 大**（例如照抄别人给 7B 模型写的 $10^{8}$）→ 没有任何 block 达到阈值。这一种同样不报错。
5. 修好之后：
   - `collectives_total` (per micro-batch) = **27**（forward 9 + backward all-gather 9 + reduce-scatter 9）；每 rank 线上字节 **335.54 MB**（fp16 reduce）或 **447.39 MB**（fp32 reduce）；ring steps 189；单 chunk $14{,}749{,}056/8 = 1.84$ MB。
   - `mem_ratio` ≈ **0.84**（FSDP 每卡约 6.1 GB / DDP 约 7.3 GB；常驻从 1406.1 MB 降到 127.82 MB，瞬时约 54 MB）。
   - `step_time_ratio`：预期显著改善。**但注意**——即使修好，它也未必 < 1.0，因为还有 `forward_prefetch=False`、每-unit 主机侧开销、梯度累积不对称这三个来源在（`01_FOUNDATIONS.md` §4.4）。**不要把"修好 wrap"和"FSDP 应该更快"绑在一起**。
6. 这个故障属于**字节账**（它直接改变了瞬时峰值的构成），但它的最响的信号在**通信账**（次数从 27 掉到 3）。**在 64M 上没有表现为 OOM**，因为退化后的峰值也只有约 383 MB 静态 + 4.87 GB 激活 ≈ 5.3 GB，离 32 GiB 还很远——**激活才是大头，静态怎么变都不足以触发 OOM**。这正是"64M 上显存不是瓶颈"这个前提带来的后果：**它让一个真实的配置错误失去了最响的那个报警**，只能靠计数器抓。

**0–4 分 rubric**

- 0：算不出 27，或认为 3 是正常的。
- 1：算出 27 并指出退化，但给不出症状链。
- 2：症状链完整（从根因到三个观测量），第一个检查点正确。
- 3：加上两种触发与"为什么静默"，以及修复后的三个数字。
- 4：以上全对，且第 6 问答出"没有 OOM 是因为激活是大头"，并主动提醒"修好 wrap 不等于 FSDP 就会更快，还有另外三个来源"。

**常见误区**

- 把 unit 数记成 8（忘了 root unit）→ 手算成 24。
- 认为 `auto_wrap_policy` 不匹配会报错。
- 认为退化成单 unit 会 OOM。在 64M 上不会。
- 把 `step_time_ratio` 恶化全部归因于这一个故障。

**追问**

- 如果 `tie_word_embeddings=True` 而 `auto_wrap_policy` 把 `embed_tokens` 和 `lm_head` 分到了不同 unit，会发生什么？这个失败是响的还是静默的？
- 在一个 27B 模型上，同样的退化会以什么形式暴露？为什么那时候反而更好排查？

**定位**：`01_FOUNDATIONS.md` §4.3、§2.4、§9.1 第 9 行、§1.2 的 tie 第三条；`lab/src/mm_v100/ledger.py`；`lab/src/mm_v100/collectives.py`。

---

### M03-D-04

**关键评分点**

1. $384.00\ \text{MiB} = 384\times2^{20} = 402{,}653{,}184$ B。除以 2 B（fp16）得 $201{,}326{,}592$ 个元素。而
   $$b\cdot H\cdot T\cdot T = 24\times8\times1024\times1024 = 201{,}326{,}592\ \checkmark$$
   即 **`[24, 8, 1024, 1024]` 的 fp16 张量**，精确吻合。要点：**这个规模的分配只可能来自一个 $[b,H,T,T]$ 的注意力分数矩阵**——它与 $d$ 无关、与参数量无关（$4P$ 才 255.65 MB），只与 $b\cdot H\cdot T^2$ 有关。反推时先除 dtype 字节数拿到元素数，再去凑 $b\cdot H\cdot T^2$，这是一条通用手法。
2. **落到了 math 后端。** 依据：只有 math 后端**物化** $[b,H,T,T]$；flash 与 mem-efficient 都是分块计算、不物化注意力矩阵（mem-efficient 只额外保存 `logsumexp [b,H,T]` fp32，本例 = $24\times8\times1024\times4 = 786{,}432$ B $= 0.79$ MB，量级差 500 倍）。而 flash 在 V100（sm70）上**根本不可用**（覆盖 sm75–sm90），所以剩下的两个后端里只有 math 会产生这个分配。
3. 每卡激活（$N = bT = 24\times1024 = 24576$ token）：
   - **mem-efficient**：$24576\times594{,}944 = 1.462\times10^{10}$ B $= \mathbf{14.62\ \text{GB}}$。加静态 1.41 GB（fp16 AMP + DDP）+ CUDA context ~1.0 GB `估算` → 约 **17.0 GB**，在 32 GiB = 34.36 GB 上**宽裕**。
   - **math**：额外 $12HT = 12\times8\times1024 = 98{,}304$ B/(token·层) $\times8$ 层 $= 786{,}432$ B/token → $24576\times786{,}432 = 1.933\times10^{10}$ B $= 19.33$ GB。合计激活 $14.62 + 19.33 = \mathbf{33.95\ \text{GB}}$，加静态与 context 约 **36.4 GB > 34.36 GB** → **OOM**。
   - 所以"$b=8,T=512$ 正常"也解释得通：那时 $N = 4096$，math 的额外项只有 $12\times8\times512\times8\times4096 = 1.61$ GB，总激活 4.05 GB，绰绰有余。**故障随 $bT^2$ 增长，不随 $bT$ 增长**。
4. 为什么 math 被选中（至少两条）：
   - **传了显式的 `attn_mask`**：mem-efficient 后端对 `attn_mask` 的支持有限制，一旦不满足就回退到 math。**这一条与 MiniMind 的 `Attention.forward` 直接相关**——它有两条分支：`attention_mask` 全 1 时走 `F.scaled_dot_product_attention(..., is_causal=True)`（不传 `attn_mask`），含 0 时走手写分支/传 mask。所以**一个含 padding 的 batch 就足以把整条路径切换掉**，而且切换是逐 batch 的、不报错。
   - **dtype / 对齐 / head_dim 不被 mem-efficient kernel 支持**（例如在 fp32 下跑、或 `head_dim` 不满足对齐要求）。
   - **手动禁用了其他后端**（`torch.backends.cuda.sdp_kernel(enable_mem_efficient=False)`），或为了做数值 reference 显式选了 math 却忘了改回来。
5. 两个修复动作，排序：
   - **① 修根因**：查清 math 为什么被选中。如果是 `attention_mask` 含 0 导致走了另一条分支，那么真正要处理的是**批内 padding 的组织方式**（长度分桶、或统一走 `is_causal` 路径），而不是强行改后端。这一步同时消除了"两条分支数值不同"这个隐患——它在 RL 里会变成 `ratio ≠ 1` 的假信号（见 `M03-D-05`）。
   - **② 绕过**：显式固定后端 `torch.backends.cuda.sdp_kernel(enable_flash=False, enable_mem_efficient=True, enable_math=False)`（本周脚本里是 `python lab/scripts/run_numeric_ref.py --sdpa-backend`），或把 $b$ / $T$ 降下来。
   - **"绕过"什么时候反而正确**：当你**故意**要用 math 后端做数值 reference 时。math 是本周唯一逐步可比对的 reference kernel，这时正确的动作不是改后端，而是**把实验规模降到 WE2（$b=8,T=512$）或更小**。
   - 第三个杠杆：**全层激活重计算**能把峰值压到一层的量级，但它不改变 math 对 $T$ 的二次依赖，只是把系数变小。
6. 硬约束：**FP32 / math 数值 reference 实验的规模被这条硬件边界限死了。** math 后端把激活压到近两倍且**随 $T$ 线性增长**（原来的每 token 系数与 $T$ 无关），所以 Day 1 的 reference 只能在小 $b$、小 $T$（WE2 或更小）上做，不能在 WE1 上做。这不是"跑不动"，是**实验设计必须迁就硬件边界**——V100 没有 flash 后端这一条，直接决定了本周 reference 实验的最大规模。

**0–4 分 rubric**

- 0：认不出 $[b,H,T,T]$，或猜"参数太多"。
- 1：反推出 shape，但说不清为什么另两个后端不会产生它。
- 2：判定 math 后端并给出依据（含 flash 在 sm70 不可用）。
- 3：加上第 3 问两个后端的激活数字与 OOM 判定，且指出故障随 $bT^2$ 而非 $bT$ 增长。
- 4：以上全对，且第 4 问答出与 `Attention.forward` 两条分支的关联、第 5 问区分"修根因 vs 绕过"并说明绕过何时正确、第 6 问给出对实验设计的硬约束。

**常见误区**

- 把 402 MB 当成"权重太大"。$4P = 255.65$ MB，跟这个数不是一回事，而且权重在第 1 步之前就分配好了。
- 认为 V100 上"至少还有 flash 可以试"。没有。
- 只给"降 batch size"这一个修复，不查为什么走了 math。降 batch 会让问题在下次改 $T$ 时原样复发。
- 把 math 后端一概当成"坏后端"。它是本周唯一的数值 reference。

**追问**

- 如果同一个 run 里有的 batch 走 mem-efficient、有的走 math，显存曲线会长什么样？这种"间歇性 OOM"该怎么定位？
- 你要怎么在不带出绝对显存数字的前提下，把"两个后端的激活差"报告出去？

**定位**：`01_FOUNDATIONS.md` §7.3、§2.3、§9.1 第 8 行；`00_WEEK_CARD.md` §3.1 的 SDPA 行；`lab/scripts/run_numeric_ref.py --sdpa-backend`。

---

### M03-D-05

**关键评分点**

1. 理论上 `ratio` **恒等于 1**（$\log r = \log p_{new} - \log p_{old} = 0$）。两个条件各自的作用：
   - **"完全 on-policy"** → 采样用的策略参数与计算 $\log p_{old}$ 用的参数是**同一份**；
   - **"第一次内层更新之前"** → 参数还没被 optimizer 动过，所以 $\log p_{new}$ 和 $\log p_{old}$ 是**同一组权重在同一批 token 上的两次前向**。
   两个条件缺一个，$\text{ratio} = 1$ 就不再是恒等式，这道题也就没有基准了。
2. 两种性质完全不同的归因：
   - **① 精度导致**：$\log p$ 是 `log_softmax` 的输出，$\log r$ 是两个几乎相等的数相减 —— **灾难性抵消**。fp16 的 eps 是 $9.8\times10^{-4}$，当 $|\log p|$ 在 5 左右时它的 ULP 约 $3.9\times10^{-3}$，两数相减后 $\log r$ 的**绝对**误差就是这个量级，而真实 $\log r$ 本身可能只有 $10^{-3}$ —— **信噪比小于 1**。
   - **② kernel / mask 路径不一致**：生成走 `padding_side=left` 的批量路径、训练走非 padding 路径；或 MiniMind `Attention.forward` 的两条分支（`attention_mask` 全 1 走 SDPA，含 0 走手写分支）在两次前向里命中了不同的一条。**两条路径的数值本来就不同**，与精度无关。
   - **混在一起这个实验就白做了**：两者都表现为"ratio 偏离 1、一部分 token 在 step 0 被 clip"，但修复方向完全相反——一个要把 logprob 链路提到 fp32，另一个要统一 mask/padding 路径。按错的方向改，现象会"改善一点"（因为 fp32 也会顺带减小路径差异的可见度）却不会消失，然后你会得出"精度问题只能缓解不能解决"这个错误结论。
3. **一次 fp32 重算**：
   - **重算什么**：对**同一批 rollout 的同一批 token id**，用**同一份权重**，把 logprob 链路（`logits` → `log_softmax` → gather → 求和）整条提到 fp32 再算一遍 $\log r$ 与 `ratio`。
   - **控制不变**：权重、token id、`attention_mask`、padding 方式、SDPA 后端、seed、`dropout` 关闭（`disable_dropout`，否则同一输入会产生不同 logprob，污染整个判定）。**唯一变量是计算精度。**
   - **比较什么**：`max|ratio − 1|` 与 `frac_clipped_at_step0` 两组数字。
   - **两种结果**：
     | 观测 | 结论 |
     | --- | --- |
     | fp32 下 $\max\lvert\text{ratio}-1\rvert < 10^{-4}$，fp16 下显著更大 | **① 精度导致** |
     | **fp32 下也偏离**（仍是 0.3 量级） | **② kernel / mask 路径不一致** |
4. 各自的下一步：
   - **①** → 把 logprob 链路强制留在 fp32：`log_softmax` 与随后的 gather / 求和都在 fp32 域；缓存的 `old_logp` 也必须以 fp32 存储（用 fp16 缓存 `old_logp` 等于在源头就把精度丢了，autocast 保护不了你自己写的减法）。同时考虑 `cast_lm_head_to_fp32`（带 ref model 时官方推荐，仅在 `tie_word_embeddings=False` 时可用 —— 本模型 tie 为 True，所以这条**不适用**，要说出来）。
   - **②** → 统一两次前向的 mask/padding 路径：让 rollout 与 training 用**同一个** batch 组织方式；或在两次前向里都强制走同一条分支（例如都不传 `attn_mask`、都用 `is_causal`）。落到代码上就是 `Attention.forward` 的分支条件与 `padding_side` 这两处。
5. **先确认的事**：`old_logp` 是不是**真的**来自那次 rollout 的同一份权重与同一条前向路径，而不是从 `generate()` 的返回值里取的（`generate` 常用 left padding、可能带采样温度/top-k 变换、可能已经是另一条 kernel 路径）。如果这一步就已经错了，你观察到的偏离与精度、与 kernel 都无关——**你会在诊断一个根本不存在的问题**。同样要确认 `dropout` 已关。
6. 代价：一次前向的 fp32 重算，一批数据、无反向、无参数更新，**分钟级**。**"代价极小"本身就是把它排在诊断第一步的理由**——它是一个能把假设空间**一刀切成两半**的实验，而且不改动任何代码路径、不引入新变量、可重复。诊断顺序的原则是"先做信息量/代价比最高的那个实验"，而不是"先做最像根因的那个猜测"。

**0–4 分 rubric**

- 0：认为 ratio 偏离 1 就是"正常的数值噪声"。
- 1：知道理论上应为 1，但只给出一种归因。
- 2：两种归因都给出并说清区别。
- 3：设计出 fp32 重算并写清控制变量与两种结果的对应结论。
- 4：以上全对，且第 4 问的修复方向落到了具体代码位置（含指出 `cast_lm_head_to_fp32` 因 tie 而不适用）、第 5 问答出 `old_logp` 来源与 dropout、第 6 问给出"信息量/代价比"这条方法论。

**常见误区**

- 直接下结论"fp16 精度不够"，不做区分实验。
- 认为 autocast 会保护 $\log r$。**autocast 只保护它列表里的算子，保护不了你自己写的减法**；而且如果 `old_logp` 是 fp16 缓存的，精度在缓存那一刻就丢了。
- 忘了关 dropout，于是同一输入两次前向本来就不同。
- 认为 `cast_lm_head_to_fp32` 是万能开关。它要求 `tie_word_embeddings=False`。

**追问**

- 如果 fp32 重算后偏离**缩小但没消失**（比如从 0.34 降到 0.02），你会怎么解读？这说明假设空间还剩什么？
- `frac_clipped_at_step0 = 0.031` 这个数在 $\varepsilon = 0.2$ 下意味着什么？如果它是 0.4 呢，你的处理顺序会变吗？

**定位**：`07_RL_LOWPRECISION.md` §0、§1.2、§2 的 R1；`lab/src/mm_rl/logprob.py`；`lab/src/mm_v100/rl_numeric.py` 的 `ratio_at_step0()`。

---

## Design

### M03-P-01

**关键评分点**

1. 三个数：$16P = 16\times27\times10^{9} = \mathbf{432\ \text{GB}}$；8 卡分摊 $432/8 = \mathbf{54\ \text{GB/卡}}$；单卡容量 32 GiB $= \mathbf{34.36\ \text{GB}}$，8 卡合计 274.9 GB。结论：**$432 > 274.9$，全机总显存都装不下纯 fp32 训练态；即使完美分片，每卡 54 GB 也超出 34.36 GB 的 1.57 倍。** 所以"FSDP 一开就行"这个答案是错的，必须叠加别的手段。
2. 四项各自可用的手段（都要在 sm70 + torch 2.1 上真的可用）：
   | 项 | 手段 | 压缩后 |
   | --- | --- | --- |
   | **参数** | FSDP `FULL_SHARD` 分片 | $4P/8 = 13.5$ GB/卡 |
   | | **冻结 backbone + 只训 LoRA**（`bnb` NF4 需 CC ≥ 6.0，V100 满足；`bnb_4bit_compute_dtype` 必须是 `float16`） | fp16 冻结分片 $2P/8 = 6.75$ GB/卡；NF4 冻结分片约 $0.5P/8 = \mathbf{1.69}$ GB/卡（+ block scale） |
   | **梯度** | FSDP 分片 | $4P/8 = 13.5$ GB/卡 |
   | | 只训 LoRA → 梯度只在适配器上（<1% 参数） | **< 0.14 GB/卡** |
   | | `MixedPrecision(reduce_dtype=fp16)` | 只影响线上字节，**不减常驻显存** |
   | **优化器状态** | FSDP 分片 | $8P/8 = 27$ GB/卡 |
   | | **8-bit Adam**（`bnb` 8-bit 优化器门槛 CC ≥ 6.0，V100 满足；这是本周唯一能拿到真实 8-bit kernel 且真实省显存的地方） | 比例 $\approx(2\times1 + 2\times4/2048)/8 = 0.2505$ → $8P\times0.2505/8 = \mathbf{6.77}$ GB/卡 |
   | | 只训 LoRA | **< 0.28 GB/卡** |
   | **激活** | **全层激活重计算** | 每层保存集合从 84.33 B/(token·$d$) 降到 4 B/(token·$d$)，峰值再加一层重算的量；在 64M 上是 4.87 GB → 约 1.36 GB（**省 3.5 GB，比 FSDP 省的 1.15 GB 更大**），代价 +33% 计算量 |
   | | 减小 $b\cdot T$ | 线性 |
   | | 确认 SDPA 不落 math 后端 | 避免近乎翻倍且 $\propto T$ 的额外项 |
3. 排序（每 GB 收益 / 风险）：
   1. **冻结 backbone + LoRA/QLoRA** —— 一次性把梯度与优化器状态砍掉两个数量级，是唯一能让 27B 真正装进 8×32 GB 的手段。**但它改变的是"你在训什么"**（从全参微调变成低秩适配），不只是"怎么训"。这是排序里唯一有这个性质的一项，必须点出来。
   2. **FSDP `FULL_SHARD`** —— 纯放置变换，数学上等价，零精度风险。
   3. **激活重计算** —— 纯时间换空间（+33% 计算量），数学上等价，零精度风险。
   4. **8-bit 优化器状态** —— 收益大（27 → 6.77 GB/卡），但有**真实的数值风险**（见第 4 点）。
   5. `reduce_dtype`、`gradient_as_bucket_view` 等 —— 前者不省常驻显存，后者是 DDP 专属，在 FSDP 路径上不适用。
   叠加后的估算：LoRA + NF4 冻结 backbone + FSDP + 激活重计算 → 静态约 **2 GB/卡**，剩下 30+ GB 全给激活，可行。
4. 失败边界（每个手段一条）：
   - **LoRA**：低秩假设不一定成立；若任务需要改变 backbone 的表示分布，LoRA 的表达力可能不够，表现为"loss 降到某个平台就不动了"。而且这个平台**你没有独立估计**（`01_FOUNDATIONS.md` §5.3 的同一个陷阱）。
   - **NF4 冻结 backbone**：必须先查基座是不是 **bf16 预训练**的 —— torch AMP 文档明确警告 bf16 预训练模型在 fp16 数值范围（max 65504）下会让梯度**上溢**而不是下溢。V100 只能 fp16，这条是硬约束。
   - **FSDP**：共享参数（tie）必须落在同一个 unit；`auto_wrap_policy` 不生效会静默退化（见 `M03-D-03`）；FSDP 路径上必须用 `ShardedGradScaler` 而不是普通 `GradScaler`。
   - **激活重计算**：+33% 计算量；重计算段内如果有非确定性算子，会破坏 R2 五步重放的可比性。
   - **8-bit 优化器状态（本机 CPU 实测的反直觉结论，必须答出）**：**per-block 缩放是必要的，但不充分。** block=2048 的 per-block **线性** 8-bit 量化**仍然发散**——末步 loss 3973 / 77330（两次不同 seed），而 fp32 基线是 0.0186。原因是**块内动态范围本身就跨好几个数量级**：缩放只是把整块平移到 $[0,1]$，块内分布依然极度偏斜，线性映射下 255 个格子里绝大多数梯度平方值仍挤在最低的几个格子，`sqrt(exp_avg_sq)` 一除就炸。必须改用**幂律映射** $q = \mathrm{round}(255\cdot u^{1/P})$（P=4）才收敛到 fp32 的 5% 以内。而且 `exp_avg`（有符号、近似对称）与 `exp_avg_sq`（非负长尾）**不该用同一套映射**。（这条是**本机 CPU 实测**，量级待 V100 复测。）
5. checkpoint 路径：**必须走 `SHARDED_STATE_DICT` + `torch.distributed.checkpoint`**。`FULL_STATE_DICT` 在 27B 上的 rank 0 峰值是 $4P = \mathbf{108\ \text{GB}}$，即使配 `offload_to_cpu=True, rank0_only=True` 也要 rank 0 的主机内存扛住 108 GB，且要做一次全模型 all_gather —— **直接不成立**。`LOCAL_STATE_DICT` 更不行（不能换 $W$ 恢复）。优化器侧必须用 `FSDP.optim_state_dict` / `optim_state_dict_to_load`。**用一个 64M 模型练这条路径，正是本周的性价比所在。**
6. 碰 GPU 之前的可验证下一步：
   ```
   python lab/scripts/byte_ledger.py --mode fsdp --world-size 8 --hand-only --show-collectives
   ```
   把 27B 的配置代进 `hand_ledger()`，先在**本机 CPU 上**打出四项手算账与集合通信序列。预期看到：常驻 $16P/8 = 54$ GB/卡（未叠加任何手段时），每卡合计 > 34.36 GB。**看到"每卡合计 < 34.36 GB"就说明账算错了**——最可能是把 param/grad 按 fp16 记（autocast 不改参数 dtype，静态恒为 $16P$），或漏了优化器的两个矩。
   加分：注明本周 `lab/src/mm_v100/model.py` 是与 MiniMind 形状对齐的自带实现、不含 MiniMind 权重，所以这条路径上的 27B 基座**必须自己指明来源**（外网下好按 40 位 commit hash 固定版本再拷进内网，`08_INTRANET_SETUP.md` §4），不能假设已有 checkpoint 可用。

**0–4 分 rubric**

- 0：没算 432 GB / 54 GB，或认为 FSDP 一开就够。
- 1：算出了三个数，四项手段只给到两项。
- 2：四项各有手段且给了数字，排序有依据。
- 3：加上每项的失败边界，含 checkpoint 路径的 108 GB 论证。
- 4：以上全对，且：① 明确指出 LoRA 改变的是"你在训什么"；② 答出 8-bit 优化器状态的幂律映射实测结论并标明是 CPU 实测；③ 给出碰 GPU 之前的具体命令与"看到什么说明账算错了"。

**常见误区**

- 把 AMP 当成能省参数显存（把静态算成 $8P$）。
- 用 `LLM.int8()` 压参数。V100 上**不能**（CC 门槛 7.5）。
- 认为 8-bit 优化器只要"加 per-block 缩放"就安全。
- 忘了检查基座是不是 bf16 预训练的。

**追问**

- 如果这 27B 是 MoE 而不是 dense，上面哪几项的收益排序会变？为什么？
- 你怎么在不带出任何绝对 GB 的前提下，把这份 27B 的可行性结论写进周报？

**定位**：`01_FOUNDATIONS.md` §2.1–§2.5、§8.1；`06_QUANT_LOWBIT.md` §1.1、§3 的 Q3；`07_RL_LOWPRECISION.md` §1.2；`08_INTRANET_SETUP.md` §4；`lab/src/mm_quant/opt8bit.py`；`lab/src/mm_v100/ledger.py`。

---

### M03-P-02

**关键评分点**

1. 实验合同五项：
   - **Hypothesis（可证伪形式）**："把冻结的 ref 模型做 weight-only INT8 之后，policy 梯度方向不变" ——**操作化为**：int8-ref 与 fp16-ref 两条路径算出的 policy 梯度的**余弦相似度 > 0.99**，且 KL 偏差的**符号一致率 ≈ 0.5**（即偏差是噪声不是偏置）。任一条不成立即证伪。
   - **Controlled variables**：同一份 policy 权重、同一批 prompt 与同一批 rollout（**不重新采样**）、同一个 `beta`、同一个 mask 与 padding 路径、`dropout` 关闭、同一个 seed。**唯一变量是 ref 模型的权重表示（fp16 vs weight-only INT8）。**
   - **Metric**：`ref_weight_bytes_ratio`（期望 ≈ 0.5 相对 fp16 存储）；$\lvert \mathrm{KL}_{\text{int8ref}} - \mathrm{KL}_{\text{fp16ref}}\rvert / \mathrm{KL}_{\text{fp16ref}}$ 的**均值与 p99**；**偏差符号一致率**；**policy 梯度的余弦相似度**。
   - **Stop condition**：跑满预定的 N 批 rollout（或 ≤ 10 分钟），拿到全部四个指标；不追加运行。
   - **Decision rule**：余弦 **> 0.99** → int8 ref 可用，写进"超大模型 RL 的显存账"结论；余弦 **< 0.9** → 判不可用，并写出原因（KL 是两个对数概率的小差值，量化噪声在这里**不被抵消而是被放大**）；落在 $[0.9, 0.99]$ → 判 `INCONCLUSIVE`，记录并注明需要更多样本或更大 `beta` 才能分辨。
2. **区分噪声 vs 偏置的量：偏差符号一致率。** 对每个 prompt（或每个 token 位置）算 $\mathrm{sign}(\mathrm{KL}_{\text{int8}} - \mathrm{KL}_{\text{fp16}})$，统计同号比例。**≈ 0.5 → 噪声；接近 1.0（或 0.0）→ 系统性偏置。** 为什么数值大小相同时后果完全不同：如果偏差是随机的，它在大量样本上会**互相抵消**，对期望梯度影响有限，表现为方差略增；如果偏差是**系统性**的（同一 prompt 上符号一致），它就是一个**偏置**，会**持续**把优化推向一个错误方向，而且样本量越大越确定地推——增加数据不但不能救，还会加固它。
3. 直接支持"优化方向没变"的量：**policy 梯度的余弦相似度**（int8-ref 梯度 vs fp16-ref 梯度，按参数拼成一个长向量算）。阈值 **0.99**。阈值来源：这是一个**预注册的工程判据**（不是从数学推出来的），依据是"余弦 0.99 对应夹角约 8.1°，在同一个下降方向的锥内"；写答案时**必须标明它是预注册阈值而不是理论值**，并说明它是在实验前定的、不允许看到数据后再改。加分：同时报 `beta` 的取值，因为 KL 项的权重直接决定这点偏差在总梯度里占多大比例。
4. **退化情形**：如果 policy 和 ref 是**同一份权重**（还没开始训练时就是这样），那么 $\log r \equiv 0$ → `sign_agreement` 变成 `nan`、`kl_rel_bias` 变成 `inf`（分母为 0）。**这不是 bug，是 0/0。** 第一件该做的事是**确认 policy 有没有真的偏离 ref**（`lab/src/mm_rl/ref_model.py` 的实验入口会先加一个 `policy_drift` 扰动把两者拉开再测），**而不是去查量化代码**。
5. 失败边界（至少两条）：
   - 即使 PASS，也**不能**推出"在超大模型上 int8 ref 同样安全"。本实验的对象是 64M + 规则 reward；量化误差结构随模型规模变化（outlier 结构随规模增强），小模型上的结论不可外推。
   - 即使 PASS，也**不能**推出"int8 ref 在训练全程都安全"。本实验只在某一个 policy-ref 距离上测过；随着训练推进 $\log r$ 变大，灾难性抵消的相对严重程度会变，结论需要在多个训练阶段复测。
   - 补充：也不能推出"省下的 ref 显存能直接换成更大的 batch"——那是另一本账，要重新算激活。
6. 可带出的证据字段（只能是比值、布尔、计数、结果类型）：`ref_weight_bytes_ratio`、`grad_cosine`、`kl_rel_bias_mean` / `kl_rel_bias_p99`、`sign_agreement`、`n_prompts`、`beta`、`policy_drift_applied`（布尔）、`status ∈ {PASS, FAIL-MODEL, INCONCLUSIVE}`、`decision_threshold_preregistered`（布尔）。**不带出**任何绝对显存数、绝对 KL 值、prompt 内容、模型输出。

**0–4 分 rubric**

- 0：只说"测一下 KL 差多少"，没有可证伪的形式。
- 1：五项合同不全，或 decision rule 没有阈值。
- 2：五项完整且有阈值。
- 3：加上噪声 vs 偏置的区分量与它的机制解释。
- 4：以上全对，且答出退化情形（0/0 而不是 bug）、两条失败边界、以及"阈值是预注册的工程判据、不允许看到数据后再改"。

**常见误区**

- 只报 KL 相对偏差的均值，不报符号一致率——均值小可能只是正负抵消。
- 看到 `nan` / `inf` 就去查量化代码。
- 把 `INCONCLUSIVE` 当成失败而不归档。它是**预注册的合法出口**。
- 重新采样一批 rollout 来做对照 —— 那样就多了一个变量，实验作废。

**追问**

- 如果余弦是 0.995 但符号一致率是 0.97，你判 PASS 还是 FAIL？为什么这两个指标可能给出相反的信号？
- 把 `beta` 从 0.04 提到 0.4，你预期上面哪几个指标会变？这对结论的适用范围意味着什么？

**定位**：`07_RL_LOWPRECISION.md` §2 的 R5、§4；`06_QUANT_LOWBIT.md` §3 的 Q4；`lab/src/mm_rl/ref_model.py`；`lab/src/mm_quant/weight_only.py`。

---

### M03-P-03

**关键评分点**

1. 固定量：$G = W\cdot A\cdot b$（单位是**序列**）。两组配置，例如 $G = 512$：**组 1** $(W,A,b) = (2, 16, 16)$；**组 2** $(W,A,b) = (8, 4, 16)$。**$b$ 必须固定**（理由见第 2、3 点：$b$ 一变，GEMM 的 shape 变、lr 相位的推导前提变、每 rank 的数值累加顺序也变）。
2. 控制变量清单（每项写清怎么控）：
   | 类别 | 怎么控 |
   | --- | --- |
   | **随机性** | 固定全局 seed；`torch.use_deterministic_algorithms(True)` 或至少固定 cuDNN/cuBLAS 的确定性开关；`dropout = 0.0`（MiniMind 默认就是 0） |
   | **初始化** | 两组从**同一个** checkpoint 冷启动（或 `sync_module_states=True` + 同 seed 构造），保证起点逐位相同 |
   | **数据顺序** | `DistributedSampler` + 每 epoch `set_epoch(同一个值)`；确保两组消费的**全局样本序列相同**（$W$ 变了但 $G$ 不变，所以每个 optimizer step 看到的样本集合应当一致） |
   | **精度与后端** | 两组都 `--dtype float16`；**显式固定 SDPA 后端**（不能一组落 mem-efficient、一组落 math），`torch.backends.cuda.sdp_kernel(...)` |
   | **学习率时钟** | `get_lr` 用 micro-batch 计数，$\text{iters}\propto1/W$；$b$ 固定时 $\text{step}/\text{iters}$ 自动对齐；**必须验证两组在同一个 optimizer step 上的 lr 逐位相等** |
   | **裁剪** | 同一个 `grad_clip = 1.0`，且 `unscale_` → `clip` → `step` 的顺序两组一致 |
   | **GradScaler** | 同一个 `init_scale` / `growth_interval`；**并且要记录两组的落地率与 scale 轨迹——如果两组跳步的步不一样，比较的就不是同一件事**。这是最容易漏的一项 |
   | **梯度累积同步** | 两组要么都加梯度累积不同步，要么都不加。这不影响数学结果，但影响数值累加顺序 |
3. **Metric**：前 $N$ 个 **optimizer step**（不是 micro-batch）上的 loss 序列，比 $\max_i\lvert\ell_i^{(2)} - \ell_i^{(8)}\rvert/\ell_i^{(2)}$。
   **阈值推导**：允许的差异来自三处——① 集合通信的归约**树形不同**（2 卡 vs 8 卡的 ring 长度不同，浮点加法不满足结合律）；② 梯度累积的**次数不同**（$A=16$ vs $A=4$，累加进 `param.grad` 的顺序与轮数不同）；③ 非确定性 kernel（本模型主要是 embedding 反向的 `index_add` 类算子）。
   **不能照搬同 $W$ 重放的 $10^{-4}$**：那个阈值成立的前提是"同 $W$、同 $b$ → GEMM shape 相同 → cuBLAS 选同一个 kernel → 累加顺序相同"，差异只来自非确定性 kernel（$\sim10^{-6}$ 相对），给一个数量级余量取 $10^{-4}$。这里 ① ② 两项**主动引入了不同的累加顺序**，误差会随累积轮数放大，所以取 $\mathbf{10^{-3}}$ 作为判据，并把阈值来源写进结论。
4. **Stop condition**：两组各跑 20 个 optimizer step 或各 ≤ 10 分钟，先到先停。**Decision rule**：$\max$ 相对差 $< 10^{-3}$ → `PASS`；$\ge10^{-3}$ 且 $<10^{-2}$ → `INCONCLUSIVE`，进第 5 步二分；$\ge10^{-2}$ → `FAIL`，一定有东西没对齐。
5. 不通过时的二分顺序（三步以内）：
   1. **先比第 0 步的 loss**。第 0 步没有任何优化器更新，两组应当**逐位相等**（同一份初始权重、同一批数据）。不相等 → 隔离掉"数据/初始化/后端"这一类，根本不是并行的问题。
   2. **再比两组的 scaler 落地率与 scale 轨迹**。不一致 → 隔离掉"跳步的步不一样"，比较的不是同一件事，先把两组的 scaler 对齐（或索性关 AMP 用 fp32 重跑这个等价实验）。
   3. **最后比逐 optimizer step 的梯度范数**（fp64）。若梯度已经对不上而 loss 还对得上，问题在通信/累积；若梯度对得上而参数对不上，问题在优化器状态或 lr。
6. 失败边界（PASS 后仍不能证明的两件事）：
   - **不能证明训练目标是正确的。** 这个实验只证明两种放置在优化**同一个**东西；如果那个东西本身错了（label mask 错位、双重位移），两组会一起错，而且一起给出漂亮的等价性。
   - **不能证明性能/扩展性。** 它是数值等价实验，不含任何吞吐结论；`step_time_ratio` 在非独占机器上甚至不能作为结论（`00_WEEK_CARD.md` §10 第 4 条）。
   - 加分第三条：也不能外推到别的 $(W,A,b)$ 组合——它只在测过的那一对上成立。

**0–4 分 rubric**

- 0：控制变量清单少于四项，或没意识到 $b$ 必须固定。
- 1：清单基本齐但阈值直接照搬 $10^{-4}$，没有推导。
- 2：清单齐全且阈值有推导（说清三处误差来源）。
- 3：加上 stop condition / decision rule 与二分顺序。
- 4：以上全对，且答出 scaler 落地率这一项（最容易漏）与两条失败边界，尤其是"两组会一起错"这一条。

**常见误区**

- 只固定 seed 就以为控住了随机性，忘了 SDPA 后端可能在两组里落到不同 kernel。
- 比较 micro-batch 级的 loss 而不是 optimizer step 级的。$A$ 不同，micro-batch 的分组不同，逐 micro-batch 比较没有意义。
- 照搬 $10^{-4}$ 阈值，然后把正常的归约顺序差异误判成 FAIL。
- 认为 PASS 就说明"分布式训练是对的"。

**追问**

- 如果把 $b$ 也一起改（$W=2,A=4,b=64$ vs $W=8,A=4,b=16$，$G$ 仍是 512），第 3 问的阈值要怎么改？为什么？
- 这个实验能不能在 CPU 上用 `gloo` 后端先跑一遍？跑通了能说明什么、不能说明什么？

**定位**：`01_FOUNDATIONS.md` §8.4 的 R2、§4.2、§3.4；`00_WEEK_CARD.md` §10 第 4 条；`lab/scripts/equiv_check.py`；`lab/tests/test_ddp_equiv_cpu.py`。

---

### M03-P-04

**关键评分点**

1. 要测的量与分离方法：
   - 测**同一种集合通信（先选 `all_reduce`）在一组负载大小 $S$ 上的耗时**，$S$ 扫 $\{1, 4, 16, 64, 256\}$ MB 量级（覆盖两个数量级）。
   - 时间模型 $t = \alpha_{\text{launch}} + n_{\text{ring}}\big(\alpha_{\text{link}} + \frac{\text{chunk}}{\beta}\big)$。把它整理成对 $S$ 的**线性形式** $t = a + S\cdot\frac{2(W-1)}{W}\cdot\frac{1}{\beta}$：
     - **截距 $a$ → $\alpha$**（每次集合通信的固定开销，含启动与 per-ring-step 延迟之和）；
     - **斜率 $\to 1/\beta$**（把 $\frac{2(W-1)}{W}$ 这个已知系数除掉之后就是有效带宽的倒数）。
   - 关键点：**必须扫 $S$**。只测一个 $S$ 无法把截距和斜率分开——这正是本题的核心方法论。
2. **至少两个 world_size（$W = 2$ 与 $W = 8$），加上 $W=1$ 做 wiring 通路检查。** 理由：① $\frac{2(W-1)}{W}$ 与 $n_{\text{ring}} = 2(W-1)$ 都是 $W$ 的函数，两个 $W$ 才能验证模型本身对不对（如果拟合出的 $\alpha$、$\beta$ 在两个 $W$ 下差很多，说明 ring 模型不适用，多半是混合拓扑）；② $W=2$ 与 $W=8$ 在混合拓扑机器上可能走完全不同的链路，$\beta$ 本来就不是一个数。命令：`python lab/scripts/wiring_smoke.py --sizes 1,2,8 --steps 2`（这同时是 Day 0 唯一的早期闸门）。
3. 变成可判定的不等式：
   $$\alpha < \frac{3.196\ \text{MB}}{\beta}$$
   （由 $35\alpha < 111.85\,\text{MB}/\beta$ 化简而来，$W=8$、64M、11 桶 vs 27 次的对比）。
   - **判为真** → FSDP 的**纯线上时间**应该比 DDP 短 → 若 Day 3 实测 FSDP 更慢，**慢的一定不是字节量**，应当直接去做 E1（`forward_prefetch`）与 E4（$b$ 翻倍），预期 `forward_prefetch=True` 能消掉大部分差距。
   - **判为假**（$\alpha$ 大到 $\beta$ 对应的临界值以上，例如跨节点或极差的 PCIe 路径）→ 次数多这件事本身就成立地解释了慢 → 优先考虑**减少集合通信次数**（wrap 粒度调粗、`SHARD_GRAD_OP`），而不是去调预取。
   **两种判定导向完全不同的 Day 3 实验顺序**，这就是这次 smoke 的价值。
4. **Stop condition**：$W\in\{1,2,8\}$ 各 2 步 wiring + 每个 $W$ 下 5 个 $S$ 各 3 次取中位数，总计 ≤ 10 分钟；拿到两组拟合参数即停。**Decision rule**：① wiring smoke 在三个 $W$ 上都能起来且 loss 有限 → 通过第一道闸门，否则**停止本周所有多卡工作**先修 `torchrun`/NCCL；② 两个 $W$ 拟合出的 $\beta$ 相差 < 20% → 认为"一个 $\beta$"的假设可用；≥ 20% → 记为混合拓扑，后续所有通信账结论都要标注"依赖 rank 分组"。
5. 两条失败边界：
   - **机器不独占**：8 卡如果不是独占，所有耗时测量都受邻居干扰，$\alpha$、$\beta$ 的拟合值**不能作为结论**，只能标为 `measurement` 类不确定。此时**还剩下可用的量**：`collectives_total`（27 / 18 / 11 这些**计数**）与 `mem_ratio`（显存是独占的，不受邻居影响）。整个 Day 3 的结论必须改成只依赖这两类量。
   - **混合拓扑**：8 卡 V100 服务器常见三种拓扑（全 NVLink 立方网、双 NVLink 组 + PCIe 跨组、纯 PCIe），$\beta$ 可差 3–5 倍，而且**同一台机器上不同 rank 对之间的 $\beta$ 可能不同**。ring 集合通信的速度由环上**最慢的一条链路**决定，所以"一个 $\beta$"这个假设本身不成立，`new_group` 的 rank 顺序变得有意义。应对：不追求测出真 $\beta$，改成测**同一个集合通信在两种 rank 分组下的耗时比**，判据是"比值是否 > 1.2"。
6. 可带出的证据字段：$\alpha$ 与 $\beta$ **本身都是绝对量，不带出**。替代量：
   - `alpha_beta_product_over_1MB`（$\alpha\beta/1\,\text{MB}$，无量纲，直接可与临界值 3.196 比较）→ 或者干脆只带一个**布尔** `crit_alpha_lt_3196kB_over_beta`；
   - `beta_ratio_w2_over_w8`（比值）、`beta_ratio_grouping_A_over_B`（不同 rank 分组的耗时比）；
   - `collectives_total`、`ring_steps`（计数）；
   - `wiring_ok_ws1` / `_ws2` / `_ws8`（布尔）、`gpu_exclusive`（布尔）、`status`。
   **不带出**：绝对毫秒、绝对 GB/s、`nvidia-smi topo -m` 的矩阵、NCCL 日志原文。

**0–4 分 rubric**

- 0：只测一个 $S$ 就想拿到 $\alpha$ 和 $\beta$。
- 1：知道要扫 $S$，但说不清截距/斜率各对应哪个。
- 2：分离方法正确，且答出至少两个 $W$ 及理由。
- 3：加上临界不等式与两种判定各自导向什么后续实验。
- 4：以上全对，且两条失败边界都给了"失效之后还剩什么可用"的具体替代（非独占 → 只用计数与 `mem_ratio`；混合拓扑 → 只用分组耗时比），并给出不含绝对量的证据字段设计。

**常见误区**

- 用一个固定大小的 all_reduce 测一次就报"带宽是多少"。那个数里混着截距。
- 把 $\alpha$、$\beta$ 当成机器的固有常数。它们是**这台机器 + 这个 rank 分组 + 这个 NCCL 版本**下的量。
- 忘了 wiring smoke 本身是闸门：跳过它，开环交付的风险会一路累积到 Day 3 才爆发。
- 在报告里写绝对 GB/s。

**追问**

- 如果拟合出的 $\alpha$ 是 400 μs（远高于单节点常见的 5–30 μs），你的第一个假设是什么？怎么证伪它？
- 这次 smoke 只测了 `all_reduce`。FSDP 用的是 `all_gather` 和 `reduce_scatter`，测一个能不能推另外两个？依据是什么、风险是什么？

**定位**：`01_FOUNDATIONS.md` §0.1、§4.1、§4.4、§4.5、§7.5；`00_WEEK_CARD.md` §4（Day 0）、§9、§10 第 4 条；`lab/scripts/wiring_smoke.py`；`lab/src/mm_v100/collectives.py`。

---

## Trade-off

### M03-T-01

**关键评分点**

1. **字节账**：
   - **静态一个字节都不省。** 在 `torch.cuda.amp.autocast` 下参数与 `param.grad` **全程 fp32**（autocast 不改参数 dtype，只在算子入口临时 cast 输入，反向梯度会被 cast 回参数 dtype 再累加）。所以 $s_p = s_g = s_o = 4$，静态恒为 $16P = 1{,}022{,}595{,}072$ B $= \mathbf{1.023}$ GB，与开不开 AMP 无关。**AMP 反而多花 $2P_{\text{linear}} = 127.80$ MB 的 fp16 权重缓存**（`cache_enabled=True` 的 weight cache，被 `linear` 的反向节点持有，活到 backward 结束）。
   - **激活省，但只省 37% 不是 50%**：fp16 AMP 4.87 GB vs 纯 fp32 6.70 GB（WE1）。原因是 AMP 引入了五份**多余的 cast 拷贝**（q/k/v 三次 + gate/up 两次，共 5 个 $U$），把本来能省的一部分吃回去了：系数从 $30.1667\times4 = 120.667$ 降到 $7\times4 + 28.1667\times2 = 84.333$，比值 0.699。
   - **反直觉的那一段：loss 头一个字节都不省。** fp16 下 logits 104.86 + 切片 104.65 + CE 入口 fp32 提升 209.31 + `log_softmax` fp32 209.31 = **628.1 MB**；纯 fp32 下 209.3 + 209.3 + 209.3 = **628.0 MB**，几乎一模一样。因为 `cross_entropy` 在 autocast 的 **fp32 列表**上，fp16 logits 省下的一半字节，被"为了进 CE 而多做的一次 fp16→fp32 拷贝"精确地吃掉了。
2. **误差账**（至少两条，各带阈值）：
   - **`GradScaler` 跳步 → 参数悄悄不更新**（B 档）。判据：落地率 < 95% 报警；连续 50 次尝试落地率 = 0 且 scale 单调下降 → 判定故障。
   - **fp16 域内的归约溢出**：RMSNorm 在 fp16 域算 $\sum x^2$ 时，$\bar x > \sqrt{65504/768} = \mathbf{9.235}$ 即溢出（MiniMind 写 `x.float()` 就是防这个）；fp16 域内 `exp` 在 $x > \ln 65504 = \mathbf{11.090}$ 时 inf。
   - 第三条加分：**优化器状态若也降到 fp16**，$\lvert g\rvert < 2^{-12} = 2.441\times10^{-4}$ 时 $g^2 < 2^{-24}$ **恰好归零** → $\sqrt{\hat v} = 0$ → 更新变成 $m/\varepsilon$。这是"优化器状态必须 fp32"最硬的理由（$(2^{-12})^2 = 2^{-24}$ 是精确等式）。
   - 第四条加分：FSDP 路径上必须用 `ShardedGradScaler`，否则各 rank `found_inf` 不同、分道扬镳且不报错。
3. **时间账**：收益来自 GEMM 走 V100 的 **FP16 Tensor Core（HMMA，FP32 累加）**。理论上界：SXM2 的 FP16 TC 峰值约 125 TFLOPS vs FP32 约 15.7 TFLOPS → **约 8×**。实际达不到，因为：① $d = 768$ 规模的 GEMM 达成算力只在 30–70 TFLOPS 区间（`估算`）；② 归约算子（norm、softmax、CE）全部留在 fp32，不受益；③ AMP 引入的五次 cast 拷贝本身要花访存时间；④ 通信不受益（DDP 的 all_reduce 用 fp32 梯度）；⑤ WE1 的 DDP 恰好落在"通信与计算同量级"的区间（通信/计算 0.39–0.91），加速计算会让通信占比更高。
4. 选纯 fp32 的场景（各指出哪本账起决定作用）：
   - **做数值 reference / 逐张量比对时**（Day 1 的 FP32 ref）——**误差账**决定：需要一个没有 AMP 噪声的基准来定义"fp16 相对 fp32 偏多少"。
   - **模型或数据处于会溢出的区域**（例如基座是 bf16 预训练的、或某层激活幅度接近 9.235/11.09 的阈值）——**误差账**决定：此时 fp16 会让梯度**上溢**而不是下溢，`GradScaler` 救不了。
   - 第三个加分场景：**显存极其宽裕而 batch 很小、GEMM 规模小到吃不到 Tensor Core 的时候**——**时间账**决定：AMP 的 cast 开销可能超过 HMMA 的收益，而字节账上 AMP 本来就只省激活的 30%。
5. 换到有 BF16 的卡上：
   - 第 1 问：静态仍是 $16P$（不变）；激活的 cast 拷贝仍在（不变）；loss 头仍不省（不变）。**字节账几乎完全不变。**
   - 第 2 问：**`GradScaler` 整个关掉**（bf16 指数位与 fp32 相同，范围 $\sim10^{38}$，不需要 loss scaling），**B 档故障整个消失**；归约溢出的两个阈值（9.235、11.09）也不再适用。**误差账的麻烦大部分消失，换来的是 $u$ 从 $2^{-11}$ 变成 $2^{-8}$（精度降 8 倍）**——bf16 用精度换范围。
   - 第 3 问：时间账的机制不变（仍走 Tensor Core），但 A100/H100 上还多了 TF32、FlashAttention-2 等路径。
   - 第 4 问：第一、二个场景仍成立但触发条件变了（不再是 fp16 的范围问题，而是 bf16 的精度问题——例如 RL 里两个几乎相等的对数概率相减，bf16 的 7 位尾数反而更糟）。
   - **一句话总结**：V100 无 bf16 → 必须 fp16 → 必须 loss scaling → 引入跳步状态机 → 引入"参数悄悄不更新"这一整类故障。**在 A100 上把 `--dtype` 改成 `bfloat16`，这一整条链条会消失，你也就永远不会学到它。**

**0–4 分 rubric**

- 0：认为 AMP 能省参数显存（把静态算成 $8P$）。
- 1：知道静态不省，但说不出激活只省 37% 或 loss 头不省。
- 2：三本账都覆盖到，含 8× 上界与达不到的理由。
- 3：加上误差账的两条带阈值判据与选 fp32 的两个场景。
- 4：以上全对，且答出 loss 头"AMP 一个字节都不省"的机制（CE 在 fp32 列表上）、第 5 问逐条对照，并说出"V100 无 bf16 → ... → 引入 B 档故障"这条因果链。

**常见误区**

- 认为 AMP 把激活减半。实际只减 30%（4.87/6.70 = 0.727）。
- 认为 AMP 的收益主要在显存。主要在时间。
- 把 bf16 说成"全面更好"。它精度更差，在 RL 的小差值场景里可能更糟。

**追问**

- 如果把 autocast 的 `cache_enabled` 关掉，字节账和时间账各会怎样？这是一个划算的交换吗？
- 纯 fp32 相对 fp16 AMP，激活只多 37%。这个 37% 是从哪五个张量省回来的？

**定位**：`01_FOUNDATIONS.md` §2.1–§2.4、§3.1、§3.3、§3.5、§7.2、§4.6。

---

### M03-T-02

**关键评分点**

1. 三个维度的排序：
   | 维度 | 排序方向 |
   | --- | --- |
   | **量化误差** | per-tensor > per-channel > per-group（**越细越小**） |
   | **scale 存储开销** | per-tensor < per-channel < per-group（**越细越大**） |
   | **kernel 实现复杂度** | per-tensor < per-channel < per-group（越细越复杂：per-tensor 一个标量广播；per-channel 沿输出维广播，还能和 GEMM 的 epilogue 融合；per-group 要在 K 维上分块，反量化不能简单地放进 epilogue） |
2. per-channel 的 scale 存储：权重 $[C_{out}, C_{in}]$，每个输出通道一个 fp16 或 fp32 scale。
   $$\frac{\text{scale 字节}}{\text{权重字节}} = \frac{C_{out}\cdot s_{\text{scale}}}{C_{out}C_{in}\cdot s_{w}} = \frac{s_{\text{scale}}}{C_{in}\cdot s_w}$$
   int8 权重（$s_w = 1$）+ fp32 scale（$s_{\text{scale}}=4$）时开销 $= 4/C_{in}$；$C_{in} = 768$ → **0.52%**；$C_{in} = 4096$ → 0.098%。
   per-group($g$)：每 $g$ 个输入元素一个 scale → 开销 $= \frac{s_{\text{scale}}}{g\cdot s_w}$，与 $C_{in}$ 无关；$g=128$、fp32 scale → **3.1%**。
   **走向**：权重矩阵越大（$C_{in}$ 越大），per-channel 的相对开销**越低**，而 per-group 的开销**不变**。迁移到超大模型时，per-channel 变得几乎免费，per-group 的收益递减点**往后移**（更晚才值得用）。
3. per-channel 收益的机理：per-tensor 用**一个** scale 覆盖整个矩阵，那个 scale 由**全局最大值**决定；只要有少数几个 outlier 通道，整个矩阵的量化步长就被它们撑大，绝大多数通道的有效比特被浪费在一个用不到的动态范围上。per-channel 把 scale 分摊到每个输出通道，outlier 通道的影响被**局限在它自己那一行**。所以误差与"outlier 指标"（`max|w| / mean|w|`）的秩相关方向为正。
   **实测"per-channel 没有收益"时的第一个检查点：`axis` 是不是对应输出通道。** 这个错误特别危险，因为它**不会报错**——沿错误的轴做 per-channel 在 shape 上照样合法，只是分组分错了，于是你得到一个"per-channel 没用"的**错误结论**，然后可能据此放弃一个本来正确的方向。
4. per-group 相对 per-channel 的收益递减点由**通道内部的动态范围**决定：如果一个输出通道内部的权重分布已经足够集中（$\max/\text{mean}$ 比值小），再切成 128 一组也没有多少可挤的空间。判"per-channel 就够了"的条件：① per-channel 的 `rel_err` 已经达到任务容忍度；② 通道内 $\max/\text{mean}$ 与整层的 $\max/\text{mean}$ 相比没有明显下降（说明 outlier 不是"通道内局部聚集"的）；③ per-group 的 kernel 复杂度或 3.1% 的 scale 开销吃掉了误差收益。
5. **V100 上能真实测量 vs 只能模拟**：
   - **能真实测量**：所有**误差**指标（`rel_err`、MSE 分解、outlier 指标、秩相关）——误差预算是数学，不依赖 Tensor Core；**权重字节**与 **HBM 读权重的字节**（带宽账）；`torch.ao` 的 **fake-quant CUDA kernel**（`FakeQuantizeCore.cu` / `FusedObsFakeQuant.cu`，前反向都有）；`bnb` 的 **8-bit 优化器**与 **NF4/FP4**（CC ≥ 6.0）。
   - **只能模拟**：任何 **INT8 Tensor Core GEMM**（无 IMMA，sm75 起才有）；`LLM.int8()`（CC 门槛 7.5）；`convert_fx` 产物上 GPU（后端全是 CPU）；`torchao`（要 torch ≥2.5）。`torch._int_mm` 属 `未知`，必须探针且**即使不报错也要与 fp32 参考比对**（pytorch#107671）。
   - 一句话：**"误差预算 + 带宽账"这两条结论可以直接迁移到 A100/H100；"TOPS 收益"这条在 V100 上根本测不了。**

**0–4 分 rubric**

- 0：三个维度的排序方向说反一个以上。
- 1：排序对，但给不出 scale 开销的表达式。
- 2：排序与开销表达式都对，含随 $C_{in}$ 的走向。
- 3：加上 per-channel 收益的机理与 `axis` 检查点，且说清这个错误为什么危险。
- 4：以上全对，且第 4 问给出可测量形式的判据、第 5 问把"能真测 / 只能模拟"分清并指出前两条可迁移。

**常见误区**

- 认为粒度越细一定越好，不算 scale 开销与 kernel 复杂度。
- 把 per-channel 的 axis 弄反，然后得出"per-channel 没用"。
- 认为 V100 上量化实验"没意义"。误差预算与带宽账两条完全真实且可迁移。

**追问**

- 如果 scale 用 fp16 存而不是 fp32，第 2 问的开销减半——这是个好交易吗？会付出什么代价？
- 激活的最优粒度和权重一样吗？为什么本周把激活的结论留到 V100 上用真实 batch 验？

**定位**：`06_QUANT_LOWBIT.md` §1.1、§3 的 Q1、§5；`01_FOUNDATIONS.md` §7.4；`lab/src/mm_quant/quantizers.py`；`lab/src/mm_quant/error_budget.py`。

---

### M03-T-03

**关键评分点**

1. 定义：**PTQ**（post-training quantization）= 训练完之后直接量化，只用少量校准数据统计 scale，不做反向。**QAT**（quantization-aware training）= 训练过程中在前向插入 quant-dequant（fake-quant），反向用 STE 直通梯度，让权重在训练时就适应量化噪声。
   在 V100 上：**QAT 是"真跑"**——torch 2.1 的 `FakeQuantizeCore.cu` 与 `FusedObsFakeQuant.cu` 提供 per-tensor / per-channel 的**前向与反向 CUDA kernel**，权重仍是 fp16/fp32，不需要 INT8 GEMM。**PTQ 的存储与误差部分也是真跑**（权重字节真的减半、误差真实可测），但 **PTQ 产物的 INT8 GEMM 加速是模拟不出来的**（无 IMMA）。
2. PTQ 就够了的判断条件（尽量可测量）：
   - **位宽够高**：8 bit 时 256 个格子对权重分布已经足够——见第 4 点的实测。
   - **量化后 `rel_err = ‖W−Q(W)‖_F/‖W‖_F` 已在任务容忍度内**，且**下游指标**（固定 held-out 的 perplexity 差、固定 prompt 的 token 级一致率）没有可测量的退化。
   - **没有强 outlier 结构**：`max|w|/mean|w|` 不高，且 per-channel 已经把误差压下去（不需要靠训练去"重排"权重分布）。
   - 加一条工程条件：**拿不到训练数据或没有训练预算**时，PTQ 是唯一选项，此时问题变成"PTQ 够不够"而不是"选哪个"。
3. QAT 的代价：训练时间之外，有一项**与 fp16 AMP 直接相关的数值代价——fake-quant 的 scale 必须在 fp32 域计算。**
   定量：autocast 会把 `clamp` / `round` 的输入降到 fp16；fp16 的 eps 是 $2^{-10}\approx9.8\times10^{-4}$（约 3 位十进制有效数字）。如果 scale 自身用 fp16 表示，它的表示误差就有 $10^{-3}$ 量级；而整个 8-bit 量化的分辨率不过 $1/256\approx3.9\times10^{-3}$。
   $$\frac{9.8\times10^{-4}}{3.9\times10^{-3}}\approx\mathbf{25\%}$$
   **scale 的表示误差会吃掉量化预算的四分之一**，而且完全静默：不报错、不出 NaN，只是量化效果莫名其妙地差。所以 `lab/src/mm_quant/quantizers.py` 强制把 scale 的计算留在 fp32。这也是"QAT 比直接 PTQ 还差"时的**第一个检查点**。
4. **截断阈值的反直觉实测结论**（`06_QUANT_LOWBIT.md` 的 Q2，**本机 CPU 实测**，2026-09-08，量级待 V100 复测）：
   | 位宽 | 最优 α | 相对 α=1.0 的 MSE 改善 |
   | --- | --- | --- |
   | **8 bit** | **1.0（没有收益）** | 1.00× |
   | 4 bit | ≈0.7 | 1.55× |
   | 3 bit | ≈0.6 | 2.29× |
   机制：8 bit 给了 256 个格子，权重分布又不像激活那样长尾，**rounding 项一直压着 clipping 项，最优解就是不截断**（没有内点极小值）。收益要到 4 bit 以下才出现——那时格子少到 rounding 项主导，牺牲少数 outlier 换全体分辨率才划算。
   **决策含义：**"8-bit 权重量化要调截断阈值"这个常见说法在这个模型的权重分布上**站不住**。截断阈值调优是**低位宽（≤4 bit）**的工具，不是 8-bit 的工具。做 8-bit 时把这份预算花在**粒度**（per-channel）上，比花在阈值上划算得多。这个"负结果"比原本预期的正结果更有价值。
   加分：激活的 $\alpha^*$ 预期比权重小（激活长尾更重），但激活分布依赖具体数据，**本机未测**，要在 V100 上用真实 batch 验。
5. 自写 fake-quant 与框架对不上的两个已知原因：
   - **`round(x/scale)` 与 `round(x·(1/scale))` 不是一回事。** aten 的 fake-quant 内部算的是 `round(x * (1/scale))`——先求倒数再乘。数学上等价，浮点上不等价：先算 `1/scale` 引入一次舍入，结果与直接相除在约 **1/32768** 的元素上差**一整个 scale step**。按"数学上更对"的写法实现，就会和框架对不上，而你会以为是自己写错了。
   - **`FusedMovingAvgObsFakeQuantize` 与 `FakeQuantize` + `PerChannelMinMaxObserver` 不是位相等的。** 融合路径算出的 scale 比 `amax/127` 差约 **0.6%**。
   **对判据的影响**：这个对照的判据必须是"**相差不超过几个 scale step**"，**不能写成位相等**。要求位相等的检查会在 V100 上失败，而失败的原因不是 bug。

**0–4 分 rubric**

- 0：说不清 QAT 与 PTQ 的区别，或认为 QAT 在 V100 上跑不了。
- 1：定义对，PTQ 够用的条件只给一条或不可测量。
- 2：定义 + 三个条件 + QAT 的时间代价。
- 3：加上 scale 必须 fp32 的定量论证（25% 这个数）。
- 4：以上全对，且答出第 4 问的三档 α 实测结论并标明是 CPU 实测、给出决策含义；第 5 问的两个坑与"判据必须是几个 scale step 而不是位相等"。

**常见误区**

- 认为"8-bit 量化当然要调截断阈值"。实测在权重上没有收益。
- 认为 QAT 在 V100 上只能模拟。fake-quant 的 CUDA kernel 前反向都有，是真跑。
- 自写 STE 与框架对不上就断定自己写错了。先查 `inv_scale` 的写法。
- 把融合路径与非融合路径的对照写成位相等断言。

**追问**

- QAT 之后 `grad_norm` 出现尖峰、`GradScaler` 的跳步率上升，你会怎么解读？这和 STE 有什么关系？
- 如果这个模型换成 4 bit，第 2 问的"PTQ 就够了"三个条件里，哪一条最先不成立？

**定位**：`06_QUANT_LOWBIT.md` §3 的 Q2 / Q5、§5；`lab/src/mm_quant/quantizers.py`；`lab/src/mm_quant/qat.py` 的 `fake_quant_ste(..., inv_scale=...)`；`lab/src/mm_quant/error_budget.py`。

---

### M03-T-04 —— **Gate 题**

**关键评分点**

1. **V100 的 Tensor Core 只接受 FP16 输入**（Volta Tuning Guide 原文："matrix multiply inputs A and B are FP16 matrices"），整数 MMA（IMMA）**不存在**。所以 INT8 权重在这台机器上只能走：
   ```
   INT8 权重 --(反量化 kernel)--> FP16 权重 --(HMMA)--> FP32 累加
   ```
   INT8 只是**存储格式**，从来没有进过计算单元。
2. 三行拆解：
   | 账 | V100 上的收益 | 为什么 |
   | --- | --- | --- |
   | **字节账** | **正**（真实） | 权重从 $2P$ 降到 $P$（+ per-channel scale，$C_{in}=768$ 时开销 0.52%） |
   | **带宽账** | **正**（真实） | 从 HBM 读权重的字节减半；对**访存受限**的场景尤其明显（batch=1 的 decode 本来就是 memory-bound） |
   | **时间账** | **零，甚至为负** | 计算仍然在 FP16 Tensor Core 上做，一次都没少；而且**多了一个未融合的反量化 kernel**——额外的 kernel launch、额外的一次读写显存。这就是它可能为**负**的确切原因 |
3. **A100/H100 上变的是第三行。** 机制：这些架构有 **INT8 Tensor Core（IMMA）**，INT8 的 GEMM 有**硬件路径**，吞吐约为 FP16 的 **2×**。这个能力**从 Turing（sm75）开始有**，Volta（sm70）没有——**差的是一整个架构代际，不是驱动、不是库、不是编译选项，装什么都补不上。**
4. 可迁移的判断规则（跑之前就能判）：
   > **先问"这个数据类型在这块卡上有没有对应的 Tensor Core 指令"，再问"我的 kernel 会不会真的用到它、反量化有没有被融合进去"。** 两个都是"是"，时间账才可能为正；只要第一个是"否"，收益就只剩字节账与带宽账，**跑之前就能下这个结论，不需要 benchmark**。
   操作化：查目标卡的 compute capability → 查该 dtype 的 MMA 支持起始代际（INT8 → sm75；BF16 → sm80）→ 查你用的库是否真的调了那条路径（例如 `bnb` 的 `supports_igemmlt()` 直接判 `< (7,5)`）。
5. 如果 V100 上实测 int8 **反而更快**：这**不是好消息，是一个必须查清的信号**。要查的是：**反量化是不是被融合进了同一个 kernel**（例如编译器把 dequant 融进了 GEMM 的 prologue，于是省下了一次显存往返，收益来自**带宽**而不是算力）。
   **为什么必须严格区分**：这两个机制的**可扩展性完全不同**。
   - "融合带来的带宽收益"在**访存受限**时才有，随 batch 增大而消失（batch 大了就变成算力受限）；
   - "IMMA 带来的算力收益"在**算力受限**时才有，随 batch 增大而**增强**。
   把前者误当成后者，你会在 H100 上按"batch 越大收益越大"去做预期，然后发现完全相反。**混淆这两件事会让你在新硬件上做出方向性错误的预期**，这比没有收益更糟。
6. 这条路线的**代价**：本周把量化还原成"一个误差预算问题 + 一个内存带宽问题"，换来的是这两条结论**完全真实、可精确测量、可直接迁移到 A100/H100**（误差预算是数学，带宽账是字节数，都不依赖 Tensor Core）。**给不出来的那一类结论是：任何关于 INT8 GEMM 端到端加速比的数字**——W8A8 相对 FP16 的实际吞吐倍数、不同 tile 大小下的达成算力、以及"量化后能不能把某个模型塞进某个延迟预算"。要回答这些，**必须有 sm75 及以上的卡**，没有任何软件替代路径（`torch.ao` 的 GPU 后端只是 early prototype 且走 TensorRT，`torchao` 要 torch ≥2.5，`bnb` 的 `LLM.int8()` 门槛 7.5）。
   加分：指出这恰恰是划算的分工——第三件事在任何有 IMMA 的卡上都是"装上库、跑一下、看数字"，**反而是最不需要训练的部分**；而误差预算与带宽账是需要练的部分，在 V100 上练完全不打折。

**0–4 分 rubric**

- 0：说"量化能加速"或说不出 V100 没有 IMMA。
- 1：知道 V100 没有 INT8 Tensor Core，但三行拆解不完整或说不出时间账可能为负。
- 2：三段式数据流 + 三行拆解全部正确，含"为负"的理由（未融合的反量化 kernel）。
- 3：加上 A100/H100 变的是哪一行、IMMA 从 sm75 起、约 2× 吞吐。
- 4：以上全对，且第 4 问给出可在跑之前使用的判断规则、第 5 问区分"融合带来的带宽收益"与"IMMA 带来的算力收益"并说清两者随 batch 的相反走向、第 6 问诚实写出这条路线给不出的那一类结论。**在 A1 条件下 20 分钟内完整讲出即视为 Gate 该项通过。**

**常见误区**

- 把"量化能加速"当成普适命题。**它是一句依赖硬件代际的话。**
- 认为可以靠"装个新版 `bitsandbytes`"或"升级驱动"在 V100 上拿到 INT8 加速。差的是硬件代际。
- 看到 int8 更快就直接当成加速成功，不查是不是融合带来的带宽收益。
- 认为"V100 上做量化没意义"。字节账与带宽账两条完全真实且可迁移，只有第三条测不了。

**追问**

- 同样这套推理，套到 **NF4 + QLoRA** 上，三行各是什么？V100 上 NF4 的时间账是正是负？
- 你怎么在不带出绝对显存与绝对延迟的前提下，把 Q4 实验（weight-only INT8 的速度归因）的结论报告出去？

**定位**：`01_FOUNDATIONS.md` §7.1 第 3 条、§7.4、§7.6；`00_WEEK_CARD.md` §3.2；`06_QUANT_LOWBIT.md` §0、§1.1、§1.3、§3 的 Q4、§6。

---

## 附：Gate 判定汇总

| Gate 项 | 对应题 | 条件 | 通过判据 |
| --- | --- | --- | --- |
| ① A0 闭卷 30 分钟字节账 + 通信序列 | `M03-A-01` | A0，纸笔，30 min | 六项全部答出且数字正确（±0.5% 手算误差），含 0.75 比值校验 → rubric ≥ 3 |
| ② A1 45 分钟三档故障 + INT8 解释 | `M03-D-01`（18 min）+ `M03-T-04`（12 min） | A1，可查官方文档与本周文件 | 两题 rubric 均 ≥ 3；`M03-D-01` 必须答出"范数只是必要条件"，`M03-T-04` 必须答出三行拆解与 sm75 代际 |
| ③ 跨场景 debug 20 分钟 | `M03-D-02` | A0，限时 20 min | 判定正确 + 说清哪一对充分、哪两列只是必要 → rubric ≥ 3 |
| 证据 | 全卷分三次作答 | — | 两个不同日期，Day 0–2 至少一条、Day 3–5 至少一条 |

**Gate 明确不依赖**：`M03-P-02`、`M03-T-02`、`M03-T-03`、`M03-E-05` 这几道来自两个扩展模块（`06_QUANT_LOWBIT.md` / `07_RL_LOWPRECISION.md`）的题，以及 GRPO 是否有正向结果。它们是加分项。
