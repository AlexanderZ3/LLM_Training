# Week M01 Reference Answers — 逐 ID 参考答案与评分标准

> 配套 `04_ORAL_EXAM.md`，ID 一一对应，共 28 题。**默认闭卷作答后再看。**
> 每题给：关键评分点 · 0–4 分 rubric · 常见误区 · 追问 · 定位（`01_FOUNDATIONS.md` 章节号 或 `lab/` 脚本名）。
> 通用高分标准（所有层通用，D/P/T 层强制）：**有假设、有证据、有替代解释、有失败边界、有可验证下一步**。五条里少于三条的答案，即使结论正确也封顶 2 分。
> 数值约定：本文件的 MB/GB 为十进制（10^6 / 10^9），GiB 为二进制（2^30），出现时会标注。

---

## Recall

### M01-R-01

**关键评分点**

1. pretrain：`tokens = [bos] + tokenizer(text, add_special_tokens=False, max_length=max_len-2, truncation=True) + [eos]`，右侧 pad 到 `max_length`；`labels = input_ids.clone()`；`labels[input_ids == pad] = -100`。**bos 与 eos 都留在 label 里**，只有 pad 被屏蔽。
2. SFT：`labels` 由 `SFTDataset.generate_labels` 生成；被置 -100 的是 system 段、user 段、`<|im_start|>assistant\n` 这 5 个头部 token、以及右侧 pad。进 loss 的只有 assistant 内容 token 加结尾的 `<|im_end|>\n` 两个 token。
3. 唯一差别是 CE 求和集合 S：pretrain 的 S 是全部非 pad 位置；SFT 的 S 是 assistant 段。损失函数、前向、归一化方式完全相同。

**0–4 分 rubric**

- 0：写不出任一条构造规则。
- 1：只答对 pretrain 或只答对 SFT 其中一边。
- 2：两边都答对大意，但漏掉"pretrain 的 bos/eos 进 loss"或漏掉"assistant 头部 5 token 不进 loss"。
- 3：三问全对，措辞准确。
- 4：三问全对，并主动指出 user/system 段虽然不进 loss 但仍在 attention 上下文里，因此仍然影响 assistant 段的条件概率。

**常见误区**

- 认为 SFT 换了损失函数（说成"只对 assistant 算 CE 是另一种 loss"）。实际是同一个 `F.cross_entropy`，只换了 `ignore_index` 的分布。
- 认为 pretrain 的 eos 不进 loss。实际 `labels` 是 `input_ids` 的完整拷贝，只屏蔽 pad，所以模型在 pretrain 阶段就学了"什么时候该出 eos"。

**追问**

- SFT 里没有 packing，一条短样本会用 pad 填满 768。这对每个 micro-batch 的 CE 分母意味着什么？
- 如果把 pretrain 的 pad 也放进 loss 会怎样？给一个能在 200 步内观察到的症状。

**定位**：`01_FOUNDATIONS.md` §1.2、§2.1、§2.3；`lab/src/mm_probe/inspect_dataset.py` 的 `pretrain_encode` / `sft_encode` / `generate_labels`。

---

### M01-R-02

**关键评分点**

1. `<|endoftext|>` = 0，同时作为 **pad 与 unk**；`<|im_start|>` = 1（bos）；`<|im_end|>` = 2（eos）。
2. `bos_id = tokenizer("<|im_start|>assistant\n") = [1, 1388, 570, 811, 234]`，长度 **5**；`eos_id = tokenizer("<|im_end|>\n") = [2, 234]`，长度 **2**。
3. ByteLevel BPE，词表 6400，`post_processor = null`，因此 `tokenizer(文本)` **不会**自动加 BOS/EOS，所有边界 token 都是模板或 dataset 显式拼上去的。

**0–4 分 rubric**

- 0：三个特殊 token id 都写不对。
- 1：只写对特殊 token id。
- 2：写对特殊 token id + 两个序列的长度，但记不住完整 id 列表。
- 3：三问全对，含完整 id 列表。
- 4：三问全对，并能说出 234 就是 `\n`、因此它同时出现在 `bos_id` 末尾和 `eos_id` 末尾，这正是必须做序列匹配而非单 token 匹配的原因之一。

**常见误区**

- 把 `bos_id` 记成只有 `<|im_start|>` 一个 token（长度 1 或 2）。实际 `assistant` 被切成 `ass`/`ist`/`ant` 三片。
- 认为 `tokenizer(prompt)` 会自动补 BOS（多数 HF tokenizer 的习惯），从而在手算 token 序列时多加一个 token。

**追问**

- `<think>`=25、`</think>`=26 是单 token；这对"思考块被 80% 概率删掉"的实现有什么便利？
- 如果 pad 和 unk 共用 id 0，`inspect_dataset.py` 统计 pad 比例时会不会把 unk 算进去？怎么区分？

**定位**：`01_FOUNDATIONS.md` §1.1；`lab/src/mm_probe/inspect_dataset.py` 的 `special_sequences`；`lab/src/mm_probe/minimind_env.py` 的 `load_tokenizer`。

---

### M01-R-03

**关键评分点**

1. `hidden_size=768`、`num_hidden_layers=8`、`num_attention_heads=8`、`num_key_value_heads=4`（`n_rep=2`）、`head_dim = 768/8 = 96`。
2. `intermediate_size = ceil(768·π/64)·64 = ceil(37.70)·64 = 2432`；`vocab_size=6400`；RoPE θ = **1e6**（预计算 32768 个位置）；`rms_norm_eps=1e-6`；`tie_word_embeddings=True`。
3. QK-norm（`q_norm`/`k_norm` 是 `RMSNorm(head_dim)`）在 **RoPE 之前**；attention 走 `F.scaled_dot_product_attention(is_causal=True)`，不依赖 flash-attn 包。
4. 总参数量 **63.9 M**（63,912,192）。

**0–4 分 rubric**

- 0：一半以上的值记错。
- 1：记住 hidden/layers/heads/vocab 四个基本值。
- 2：再加上 kv_heads、head_dim、tie_word_embeddings。
- 3：全部值正确，含 `intermediate_size=2432` 与 θ=1e6。
- 4：全部正确，且能当场用 §1.3 的分项算式复算出 63.9M（embedding 4.92M + 每层 7.37M × 8 + 末层 norm 768）。

**常见误区**

- 把 `intermediate_size` 当成 4H=3072 或 8/3·H。MiniMind 用的是 `ceil(H·π/64)·64`。
- 把 RoPE θ 记成常见的 1e4。MiniMind 用 1e6。

**追问**

- 为什么 `Model Params` 打印出来是 63.91M 而不是 68.83M？tied 权重在 `model.parameters()` 里出现几次？
- QK-norm 放在 RoPE 之后会怎样？说出一个能用单测检出的差异。

**定位**：`01_FOUNDATIONS.md` §1.3、§2.2；`lab/src/mm_probe/minimind_env.py` 的 `build_model_config`。

---

### M01-R-04

**关键评分点**

1. 四步：`scaler.scale(loss).backward()`（loss 乘 s 后反向，梯度整体放大 s 倍以避免下溢）→ `scaler.unscale_(optimizer)`（梯度除以 s，使 `clip_grad_norm_` 看到真实范数）→ `scaler.step(optimizer)`（若 unscale 时发现 inf/nan 则**跳过** `optimizer.step()`）→ `scaler.update()`（发现 inf/nan 则 s←s×0.5；连续 2000 步正常则 s←s×2）。
2. 默认常量：`init_scale = 65536 = 2^16`、`backoff_factor = 0.5`、`growth_interval = 2000`、`growth_factor = 2.0`。
3. `train_grpo.py` 没有 `GradScaler`，直接 `loss.backward()` → `clip_grad_norm_` → `optimizer.step()`。

**0–4 分 rubric**

- 0：说不出四步顺序。
- 1：能说出四步名字但讲不清各自作用。
- 2：四步顺序与作用正确，常量记不全。
- 3：四步 + 四个常量全对 + 指出 GRPO 无 scaler。
- 4：以上全对，并能说明 `unscale_` 必须在 `clip_grad_norm_` 之前的原因（否则裁剪阈值 1.0 会被 s 倍放大的梯度直接触发，等于把所有梯度按 1/s 压掉）。

**常见误区**

- 把 `clip_grad_norm_` 放在 `unscale_` 之前，或认为 `scale()` 会改变梯度方向。它只改幅值。
- 认为跳步时 lr 调度也会回退。实际 `get_lr` 以数据 step 计数，跳步不回退，因此每次跳步都是"lr 照走但权重不动"的一步。

**追问**

- 梯度累积窗口内某一个 micro-batch 出现 inf，会丢弃这一个 micro-batch 还是整个窗口？
- bf16 下 `scaler.get_scale()` 返回什么？为什么这个值可以直接用来在日志里区分是否真的开了 scaler？

**定位**：`01_FOUNDATIONS.md` §4.2、§4.5；`lab/src/mm_probe/hooks.py` 的 `grad_global_norm`。

---

### M01-R-05

**关键评分点**

1. `train_grpo.py` 默认：`loss_type='cispo'`（**不是** grpo）、`num_generations=6`、`batch_size=2`、`max_gen_len=1024`、`dtype='bfloat16'`。
2. checkpoint 两类：权重文件 `{weight}_{hidden_size}.pth`（例如 `full_sft_768.pth`），保存前做了 `half()`；恢复文件 `{...}_resume.pth`，含 `model`（half）、`optimizer`、`epoch`、`step`、`world_size`、`wandb_id`；pretrain/SFT/DPO 额外存 `scaler`，GRPO 额外存 `scheduler`。
3. 默认 lr：pretrain `5e-4`、full_sft `1e-5`、lora `1e-4`、dpo `4e-8`、grpo `3e-7`。

**0–4 分 rubric**

- 0：默认值大面积记错。
- 1：只答对 lr 一组或只答对 GRPO 一组。
- 2：三问答对两问。
- 3：三问全对。
- 4：三问全对，并能解释这五个 lr 相差四个数量级的理由不是梯度大小（AdamW 已把每步位移归一到 ≈ lr），而是**允许模型离起点漂多远**。

**常见误区**

- 以为 `train_grpo.py` 默认跑 GRPO。默认是 CISPO。
- 以为 `_resume.pth['model']` 是 fp32。它是 `half()` 之后的，这是 resume 不 bitwise 的原因之一。

**追问**

- `_resume.pth` 里存 `world_size` 是为了什么？恢复到不同卡数时 `step` 怎么换算？
- 权重文件存 half 而 optimizer 状态存 fp32，这个组合在恢复时会造成什么不一致？

**定位**：`01_FOUNDATIONS.md` §4.3、§5.4、§6 第 12 条；`00_WEEK_CARD.md`「MiniMind 关键事实」。

---

## Explain

### M01-E-01

**关键评分点**

1. 分母是 `S = { t : labels[t+1] ≠ -100 }` 的大小，即**这个 micro-batch 内所有非 -100 位置的总数**。不是 B×T（pad 与 -100 都不计），也不是"先逐样本平均再样本间平均"（跨样本是直接混在一起数的）。
2. `loss / accumulation_steps` 逐 micro-batch 反传 = 对 8 个 micro-batch 的"各自 token 均值"再取算术均值。仅当各 micro-batch 有效 token 数**完全相等**时，它才等于把所有 token 混在一起的一次均值。不等时，**有效 token 数少的 micro-batch 里的每个 token 被隐式加了更大权重**，即短回答样本被高估。
3. 随机初始化 + tied embedding 时 logits 近似均匀，`loss ≈ ln V = ln 6400 = 8.764`。用途：SFT 从 pretrain 权重起步时首步 loss 应显著低于 8.76（通常 2–4）；若仍在 8.7 附近，说明预训练权重没被真正加载（`load_state_dict(strict=False)` 静默跳过不匹配 key），即 §6 第 1 条。

**0–4 分 rubric**

- 0：说成分母是 B×T。
- 1：说对分母是非 -100 位置数，但讲不清累积。
- 2：分母正确 + 知道累积是"均值的均值"，但说不出何时不等价。
- 3：三问全对，含 ln 6400 = 8.76 与它的诊断用途。
- 4：以上全对，并给出一个具体数字例子（例如两个 micro-batch 有效 token 1000/100、loss 2.0/4.0 → 3.0 对 2.18）说明偏差方向和量级。

**常见误区**

- 认为 `ignore_index=-100` 只是"跳过不算"，忘了它同时把这些位置从**分母**里去掉，因此 pad 多少不影响 loss 数值。
- 认为梯度累积在数学上严格等价于大 batch。只有在有效 token 数相等时才等价。

**追问**

- 一个 batch 里若混了"user 300 token + assistant 20 token"和"user 20 token + assistant 300 token"两条样本，哪一条主导 loss？
- 你要怎么改动才能让累积严格等价于大 batch 的 token 加权均值？这个改动需要哪个额外统计量？

**定位**：`01_FOUNDATIONS.md` §2.1、§3.3、§6 第 1 条；`lab/src/mm_probe/inspect_dataset.py` 的 `sample_stats` / `aggregate`。

---

### M01-E-02

**关键评分点**

1. `<|im_start|>assistant\n` 里只有 `<|im_start|>`（1）和 `\n`（234）是单 token，`assistant` 被 BPE 切成 `ass`(1388)/`ist`(570)/`ant`(811) 三片，所以共 5 个 token。label 起点相对 `<|im_start|>` 偏移 **5** 位，而不是偏移 1 或 2 位。
2. 只匹配 id=1 会同时命中 `<|im_start|>user\n` 与 `<|im_start|>system\n` 的头，导致 user/system 段整段进 loss —— 就是 §6 第 3 条「模型复述/续写 user 的问题、自己生成 `<|im_start|>user`」，对应 `mask_fault.py --mode user_in_loss`。
3. 只匹配 id=2 停止，则 `<|im_end|>` 后的 `\n`（234）不进 label；模型学会输出 `<|im_end|>` 但没学会那个换行，在多轮拼接与流式解析时会出现边界残缺；更关键的是 `generate_labels` 的扫描游标 `i = end + len(eos_seq)` 会错位一格，后续轮次的 bos 匹配可能整体失准。
4. 截断时若扫到序列末尾都没匹配到 `eos_id`，`generate_labels` 会**一直标到序列末尾**（`end` 走到 `len(input_ids)`），即被截断的 assistant 段全部进 loss、但缺少 eos。

**0–4 分 rubric**

- 0：不知道 `bos_id` 是多 token 序列。
- 1：知道是 5 个 token，但说不出后果。
- 2：答对 1、2 两问。
- 3：四问全对。
- 4：四问全对，并指出 `\n`(234) 同时是 `bos_id` 的末位与 `eos_id` 的末位，这使得任何"只看单 token"的简化匹配都必然产生边界歧义。

**常见误区**

- 用解码后的字符串 `"assistant"` 去 `find`，忽略 token 边界与字符边界不一致，导致在 id 序列上定位错位。
- 认为 label 不包含 `<|im_end|>`。实际它必须包含，否则就是 §6 第 4 条"生成永不停止"。

**追问**

- 写一条能在 CPU 单测里断言"每条 SFT 样本的 label 中至少出现一次 id 2"的检查，它属于哪个测试文件？
- 若某条样本的 assistant 内容里本身出现了字符串 `<|im_start|>assistant`，会发生什么？这属于数据清洗问题还是 label 生成问题？

**定位**：`01_FOUNDATIONS.md` §1.1、§1.2、§6 第 2/3/4 条；`lab/src/mm_probe/inspect_dataset.py` 的 `generate_labels` docstring。

---

### M01-E-03

**关键评分点**

1. **GQA**：k_proj/v_proj 从 `H×H = 589,824` 降到 `H×(n_kv·d) = 768×384 = 294,912`，每层省 `2×294,912 = 589,824` 参数，8 层共省 ≈ **4.72 M**。KV cache 每 token：`8 层 × 2(k,v) × 4 heads × 96 × 2 B = 12,288 B = 12 KB`；MHA 则是 24 KB，省一半。`repeat_kv` 在 RoPE **之后**、SDPA 之前，把 k/v 在 head 维复制 `n_rep=2` 次到 `[B,8,T,96]`。代价：4 个 kv head 被 8 个 q head 共享，key/value 子空间的表达力下降。
2. **tied embedding**：省 `V·H = 4,915,200 ≈ 4.92 M`（不 tie 时总量 68.83M，tie 后 63.91M，省 7.1%）。`embed_tokens.weight` 收到两路梯度：作为输入查表的**稀疏**梯度（只有 batch 中出现过的 token 行）和作为 `lm_head` 的**稠密**梯度（softmax 对全部 6400 行都有贡献）。后者主导。代价：输入表示与输出表示被强制共享一个空间。
3. **RoPE θ=1e6**：`freq_i = θ^(−2i/d)`，θ 越大低频维波长越长（最低频波长 2π·1e6 ≫ 768），因此在 340/768 的训练长度内绝大多数维度几乎不旋转，位置分辨率主要由**高频维**承担。换来的是先天为长上下文留出的外推余量；代价是短序列上可用的位置编码维度实际变少。

**0–4 分 rubric**

- 0：三项中两项说不出机制。
- 1：能说出三个设计各自的名字与"省显存/省参数"的定性结论。
- 2：其中一项给出了正确算式。
- 3：三项都有正确算式或正确机制描述，代价说清楚。
- 4：以上全对，并指出三者的代价方向不同：GQA 与 tied 是"容量换资源"，θ=1e6 是"短序列分辨率换长序列外推"，因此在本周 T≤768 的设定下第三项的代价是被实际付出的。

**常见误区**

- 把 `repeat_kv` 当成节省计算的手段。它不省 attention 的计算量（复制后仍是 8 head 的 attention），只省参数与 KV cache。
- 认为 tied embedding 只是省参数。它显著改变了 embedding 层的梯度结构（稠密梯度主导），这是小词表模型 tie 后早期收敛更快的原因。

**追问**

- 如果把 `num_key_value_heads` 从 4 改成 8，参数量、KV cache/token、attention 的 FLOPs 分别怎么变？
- θ 从 1e6 降到 1e4，在 T=768 的训练上你预期看到什么可测量的变化？设计一个能观察到它的最小实验。

**定位**：`01_FOUNDATIONS.md` §1.3、§2.2、§4.6、§5.2；`lab/src/mm_probe/hooks.py`。

---

### M01-E-04

**关键评分点**

1. DPO 的信号是**相对偏好**：目标里出现的是 chosen 与 rejected 的 log-ratio 之**差**，只用 chosen 做 SFT 是最大化绝对似然，会同时抬高与 chosen 相似的所有输出（包括 rejected）。DPO 用一个 pair 提供一个 bit 的相对信息，并把隐式 reward 定义成 `β·log(π_θ/π_ref)`。
2. ref 模型由 `init_model` **二次加载同一份 `full_sft_768.pth`** 得到，设成 `eval()` + `requires_grad_(False)`，forward 在 `torch.no_grad()` 上下文里。ref 是隐式 reward 的**零点 / KL 锚点**；如果它也更新，锚点会随策略漂移，`log(π_θ/π_ref)` 会趋于 0，目标退化成无约束优化，SFT 学到的 chat 格式与停止行为会被冲掉。
3. `(log_probs * mask).sum(dim=1)` 是**求和**，所以序列级 log-ratio 的量级正比于该回答的有效 token 数：长回答天然有更大的 |margin|，DPO 会系统性地偏向"长的那一条"，长度成为一个未被显式建模的混杂变量。
4. 初始时 `π_θ = π_ref` → `pi_logratios − ref_logratios = 0` → `loss = −logsigmoid(0) = ln 2 = 0.693`。训练有效时：`chosen_reward = β·(logπ_θ − logπ_ref)(y_w)` 上升、`rejected_reward` 下降、`margin > 0` 的样本比例上升。

**0–4 分 rubric**

- 0：说不清 DPO 为什么要 pair。
- 1：答对 pair 的必要性。
- 2：再答对 ref 的状态与"不更新"的理由。
- 3：四问全对，含 0.693 的推导。
- 4：以上全对，并把"求和不求均值"与"β=0.15、lr=4e-8"联系起来：因为 margin 量级本身随长度放大，所以 β 与 lr 必须压得很小才不让长度效应主导权重更新。

**常见误区**

- 认为 ref 是"上一步的 policy"。它是固定的 SFT 权重，整个训练过程一次都不更新。
- 认为初始 loss 应该是 0。是 `ln 2 = 0.693`，因为 `σ(0) = 0.5`。

**追问**

- `torch.cat([x_chosen, x_rejected], dim=0)` 使 `batch_size=4` 时 policy 与 ref 各看到什么 shape？为什么这个拼接方式让实现不需要两次 forward？
- 用 §4.4 的漂移上界论证：`N = 2 万` 条 DPO 样本、`batch=4`、`lr=4e-8`，每个参数的总漂移上界是多少？相对初始 std 0.02 是多少百分比？

**定位**：`01_FOUNDATIONS.md` §2.5、§4.4、§3.1 不变量 7、§6 第 11 条。

---

### M01-E-05

**关键评分点**

1. `A_i = (r_i − mean(r)) / (std_pop(r) + 1e-4)`，`std` 用 `unbiased=False`（总体标准差），常数 `1e-4`，组大小 G = `num_generations`。
2. GRPO 的 token 级项是 PPO 式：`ℓ = −( min(ratio·A, clip(ratio, 1−ε, 1+ε)·A) − β·KL )`，ε=0.2，`ratio` 直接乘在 A 上并参与梯度。CISPO 是 `ℓ = −( sg[min(ratio, ε_high)]·A·log π_θ − β·KL )`，`ε_high=5.0`，`sg` 是 detach —— **重要性比值被截断后 detach 成一个标量权重，梯度只从 `log π_θ` 这一项流出**。
3. 默认 torch rollout 引擎下，`old_per_token_logps` 由 `rollout_engine.rollout` 在生成结束后用**同一个 policy** 重新前向算出，训练前向也用同一个 policy，且每个 rollout 只做一次 backward，因此数学上 `ratio ≡ 1`，clip 不触发；唯一偏差来自 bf16 数值与 `logits_to_keep` 切片，预期 `max|ratio−1| < 1e-2`。于是默认 CISPO 退化为「**组内归一化 REINFORCE + β·KL**」。
4. 不是同一个东西。损失里用的是 k3 估计：`KL = exp(logπ_ref − logπ_θ) − (logπ_ref − logπ_θ) − 1`，**恒 ≥ 0**；日志里的 `KL_ref` 是 `mean(logπ_ref − logπ_θ)`，是有符号的 log-ratio 均值，**可以为负**，不能当散度读。

**0–4 分 rubric**

- 0：写不出 advantage 公式。
- 1：advantage 公式正确。
- 2：再说清 CISPO 与 GRPO 的表达式差别。
- 3：四问全对，含 `ratio ≡ 1` 的理由。
- 4：以上全对，并指出 off-policy 修正只在 SGLang 引擎（权重每 `save_interval` 才同步）下真正起作用，因此在本周单卡 torch 引擎上讨论 clip 阈值是没有意义的。

**常见误区**

- 把 `std` 当成无偏（`unbiased=True`）。G 小的时候两者差别不可忽略（G=2 时无偏 std 是总体 std 的 √2 倍）。
- 把日志里的 `KL_ref` 当成 KL 散度，看到负值就以为是 bug。

**追问**

- `completion_mask` 是怎么定义的？为什么 `generate()` 对已结束序列继续填 eos 时不会污染 loss？
- 若把 `1e-4` 改成 `1e-8`，reward 接近全等但不完全相等的组会发生什么？给一个数值例子。

**定位**：`01_FOUNDATIONS.md` §2.6、§3.3；`lab/src/mm_probe/grpo_stats.py` 的 `group_advantage` / `group_stats`。

---

## Apply

### M01-A-01

**完整计算过程**

**（1）渲染后的模板字符串**

```text
<|im_start|>system\n你是一个金融助手。<|im_end|>\n<|im_start|>user\n什么是久期？<|im_end|>\n<|im_start|>assistant\n久期衡量债券价格对利率的敏感度。<|im_end|>\n<|im_start|>user\n谢谢<|im_end|>\n<|im_start|>assistant\n不客气。<|im_end|>\n
```

**（2）分段累加与位置分配**

| # | 片段 | token 数 | 位置区间 |
| --- | --- | --- | --- |
| 1 | `<\|im_start\|>system\n` | 4 | 0–3 |
| 2 | `你是一个金融助手。` | 6 | 4–9 |
| 3 | `<\|im_end\|>\n` | 2 | 10–11 |
| 4 | `<\|im_start\|>user\n` | 4 | 12–15 |
| 5 | `什么是久期？` | 5 | 16–20 |
| 6 | `<\|im_end\|>\n` | 2 | 21–22 |
| 7 | `<\|im_start\|>assistant\n` | 5 | 23–27 |
| 8 | `久期衡量债券价格对利率的敏感度。` | 11 | 28–38 |
| 9 | `<\|im_end\|>\n` | 2 | 39–40 |
| 10 | `<\|im_start\|>user\n` | 4 | 41–44 |
| 11 | `谢谢` | 1 | 45 |
| 12 | `<\|im_end\|>\n` | 2 | 46–47 |
| 13 | `<\|im_start\|>assistant\n` | 5 | 48–52 |
| 14 | `不客气。` | 3 | 53–55 |
| 15 | `<\|im_end\|>\n` | 2 | 56–57 |

`4+6+2+4+5+2+5+11+2+4+1+2+5+3+2 = 58` → **总长 58 token**；`max_seq_len=64` → pad(0) 占 **58–63**（6 个）。

**（3）label 区间**

| 位置区间 | label |
| --- | --- |
| 0–27 | -100（system 段 + 第一轮 user 段 + 第一个 assistant 头 5 token） |
| 28–38 | 自身 id（assistant-1 内容，11 个） |
| 39–40 | 2, 234（assistant-1 的 `<\|im_end\|>\n`） |
| 41–52 | -100（第二轮 user 段 + 第二个 assistant 头 5 token） |
| 53–55 | 自身 id（assistant-2 内容，3 个） |
| 56–57 | 2, 234（assistant-2 的 `<\|im_end\|>\n`） |
| 58–63 | -100（pad） |

**（4）有效 label 数** = (11+2) + (3+2) = **18**。不变量交叉验证：`Σ(assistant 内容 + 2) = 13 + 5 = 18` ✓，且 label 中包含 id 2 两次 ✓。

**（5）** 第一个 assistant 内容 token 在位置 28，位置 t 的 logits 预测 `label[t+1]`，所以由 **t = 27**（assistant 头的 `\n`）负责预测它。最后一个进入 loss 的位置是 **t = 56**（预测 `label[57] = 234`）；t=57 预测 `label[58] = -100`，不进 loss。

**（6）CE 分母** = 满足 `labels[t+1] ≠ -100` 的 t 的个数。18 个有效 label 的下标全部 ≥ 1，因此分母 = **18**。

**（7）`max_seq_len=48`**：输入被截断到位置 0–47，第二个 `<|im_start|>assistant\n`（48–52）整段消失，`generate_labels` 只能匹配到第一个 bos。有效 label 数 = **13**。注意此时截断点恰好落在 user 段的 `<|im_end|>\n` 之后，没有产生"半个 assistant 段"，所以不会触发"扫不到 eos 就标到末尾"的分支。

**（8）** 有效 label 数**本身不变**（新插入的 system 段全 -100，`Σ(assistant 内容 + 2)` 不变）。但总长度增加，若增加后超过 `max_seq_len` 而截掉了尾部的 assistant 段，则有效 label 数会减少 —— 这是唯一会变的条件。

**0–4 分 rubric**

- 0：位置分配就算错。
- 1：总长 58 与 pad 区间正确，但 label 区间划错（例如把 assistant 头也标成 label）。
- 2：label 区间与有效 label 数 18 正确，但第 5、6 问答错。
- 3：1–6 问全对。
- 4：八问全对，且第 7、8 问明确指出了"截断改变的是有效 label 数、system 插入只在触发截断时才改变它"这一条件依赖。

**常见误区**

- 把 label 起点算在位置 23（`<|im_start|>` 本身）而不是 28，即忘了 `bos_id` 长度是 5。
- 把"哪个位置有 label"和"哪个位置的 logits 进 loss"混为一谈，导致第 5 问答成 t=28。

**追问**

- 若把这条样本放进 `batch_size=16` 的 micro-batch，它对 CE 分母贡献 18，而某条长回答样本贡献 300。谁主导这一步的梯度？
- 用 `inspect_dataset.py --show-labels` 打印这条样本，你要检查哪三个断言？

**定位**：`01_FOUNDATIONS.md` §1.2（对照表）、§2.1、§2.3；`lab/src/mm_probe/inspect_dataset.py` 的 `generate_labels` / `token_rows` / `sample_stats`；`lab/src/mm_probe/fixtures.py` 的 SFT 样例。

---

### M01-A-02

**完整计算过程**

**（1）MiniMind SFT，B=16, T=768, V=6400**

```text
元素数 n = 16 × 768 × 6400 = 12,288 × 6,400 = 78,643,200
bf16 logits      = 78,643,200 × 2 B = 157,286,400 B = 157.3 MB
CE fp32（两份）  = 78,643,200 × 4 B × 2 = 629,145,600 B = 629.1 MB
合计             = 786,432,000 B = 786.4 MB
```

**（2）同 B、T，V = 151,936**

```text
n = 12,288 × 151,936 = 1,866,989,568
bf16 logits      = 1,866,989,568 × 2 B = 3,733,979,136 B = 3.73 GB
CE fp32（两份）  = 1,866,989,568 × 8 B = 14,935,916,544 B = 14.94 GB
合计             = 18,669,895,680 B = 18.67 GB      → 单这一项就超过 16 GB
```

**（3）单样本，B=1, T=768, V=151,936**

```text
n = 768 × 151,936 = 116,686,848
bf16 logits = 233,373,696 B = 0.233 GB
CE fp32 ×2  = 933,494,784 B = 0.933 GB
合计        = 1,166,868,480 B = 1.167 GB / 样本（= 1.087 GiB）
```

**（4）上限 4 GB**

```text
4.0 GB / 1.167 GB ≈ 3.43  → batch_size 最大 3
B=3 → 3.50 GB ✓      B=4 → 4.67 GB ✗
```
（换成 GiB 判据结论相同：B=3 → 3.26 GiB，B=4 → 4.35 GiB。）

**（5）** 词表比是 `151,936 / 6,400 = 23.74` 倍，而 logits/CE 项严格正比于 V。MiniMind 的 786 MB 相对静态项 1.0 GB 与激活 2.9 GB 只是配角；乘上 23.74 倍后变成 18.67 GB，超过整卡容量，从配角变成绝对主项 —— 这就是 §7 说的第 12 周 CPT 显存差异根源。

**0–4 分 rubric**

- 0：公式用错（例如漏掉 CE 的 fp32 两份）。
- 1：第 1 问正确。
- 2：第 1、2 问正确。
- 3：1–4 问数字全对，算式写出。
- 4：五问全对，且第 5 问给出了 23.74 倍这个显式比值并说明它为什么改变了"主项是谁"的结论。

**常见误区**

- 只算 bf16 logits，忘了 `log_softmax` 在 autocast 下升到 fp32、且其输出与梯度都要保留给反向（因此是 ×4 B ×2）。
- 把 MB 与 MiB 混用后得出"157 MB 与 150 MB 不一致"的结论。本周约定十进制。

**追问**

- 若把 CE 换成分块计算（每次只对词表的一个切片求 log_softmax），第 2 问的 14.94 GB 会怎么变？代价是什么？
- V=151,936 时，把 T 从 768 降到 384，第 4 问的 batch 上限变成多少？

**定位**：`01_FOUNDATIONS.md` §5.1、§7；`lab/src/mm_probe/hooks.py` 的 `peak_memory_mb`。

---

### M01-A-03

**完整计算过程**

**（1）**

```text
global batch = batch_size × accumulation_steps × world_size = 32 × 8 × 1 = 256 序列/优化步
每条序列 T=340，shift 后贡献至多 339 个位置
上界 = 256 × 339 = 86,784 个 loss 位置/优化步
```
是"上界"是因为：pad 位置 label = -100 不计入分母；文本短于 338 token 的样本会有 pad，实际有效位置数低于 339。

**（2）** `get_lr(s,S,lr) = lr·(0.1 + 0.45·(1 + cos(π·s/S)))`，lr = 5e-4：

```text
s=0    : cos(0) = 1     → 0.1 + 0.45×2 = 1.00 → 5.0e-4
s=S/2  : cos(π/2) = 0   → 0.1 + 0.45×1 = 0.55 → 2.75e-4
s=S    : cos(π) = −1    → 0.1 + 0.45×0 = 0.10 → 5.0e-5
```
无 warmup，起点就是峰值。

**（3）**

```text
(a) MiniMind 实际：每个 micro-batch 各自 loss/2 后 backward
    等效 loss = (2.0 + 4.0) / 2 = 3.0000
(b) 1100 个 token 混在一起：
    (2.0 × 1000 + 4.0 × 100) / 1100 = (2000 + 400) / 1100 = 2400 / 1100 = 2.1818
(c) 差 = 3.0000 − 2.1818 = 0.8182
```
被高估的是**有效 token 数少的 micro-batch**（此处 100 token 那个），其中每个 token 的隐含权重是 `1/(2×100)`，而另一个 micro-batch 里每个 token 只有 `1/(2×1000)`，相差 10 倍。在 SFT 里这系统性地对应"assistant 回答很短的样本"。

**（4）**

- **global batch**：`8 × 32 × 1 = 256`，不变。
- **`get_lr` 作为数据位置的函数**：不变。`s` 按 micro-batch 计数，`S = epochs × 每 epoch micro-batch 数`；batch 缩小 4 倍使 s 与 S 同时放大 4 倍，比值 `s/S` 在同一数据位置上不变。
- **第 3 问的偏差**：**变大**。micro-batch 越小，各 micro-batch 的有效 token 数相对波动越大，"均值的均值"离 token 加权均值越远。

**0–4 分 rubric**

- 0：global batch 算错。
- 1：第 1 问正确。
- 2：第 1、2 问正确。
- 3：1–3 问数字全对，指出被高估的样本类型。
- 4：四问全对，且第 4 问三个子问都答对并说明了 `s/S` 不变的理由。

**常见误区**

- 认为改 `accumulation_steps` 会改变 lr 曲线形状。它只改变每个优化步吃多少 token。
- 认为把 batch 缩小、accum 放大是"完全等价"的显存优化。global batch 与 lr 曲线确实等价，但"均值的均值"偏差和 BN 类统计（本模型没有）不等价。

**追问**

- 若 world_size 从 1 变 2 而其他不变，global batch 与 `get_lr` 分别怎么变？`_resume.pth` 的 `step` 换算规则是什么？
- 在 64M 模型上把 global batch 提高 4 倍并保持 lr 不变，你预期 loss-vs-token 曲线怎么变？设计一个 128 样本 overfit 的对照实验来验证。

**定位**：`01_FOUNDATIONS.md` §2.1、§4.3、§5.4；`lab/src/mm_probe/parse_log.py`（曲线）。

---

### M01-A-04

**完整计算过程**

**（1）G=6，r = [1.75, 1.75, 2.25, 0.75, 1.75, 1.75]**

```text
mean = (1.75×4 + 2.25 + 0.75) / 6 = (7.00 + 3.00) / 6 = 10.00 / 6 = 1.666667
偏差 : +0.083333 (×4), +0.583333, −0.916667
总体方差 = (4×0.083333² + 0.583333² + 0.916667²) / 6
         = (4×0.00694444 + 0.34027778 + 0.84027778) / 6
         = (0.02777778 + 0.34027778 + 0.84027778) / 6
         = 1.20833333 / 6 = 0.20138889
std_pop  = 0.4487637
分母     = 0.4487637 + 1e-4 = 0.4488637

A(1.75) = 0.083333 / 0.4488637 =  0.1857   （4 条）
A(2.25) = 0.583333 / 0.4488637 =  1.2996
A(0.75) = −0.916667 / 0.4488637 = −2.0422
不变量：4×0.185653 + 1.299572 − 2.042185 = −0.000001 ≈ 0 ✓
```
数学上组内和恒为 0（分子 `Σ(r_i − mean) = 0`），浮点下应满足 `|sum| < 1e-5`。

**（2）r 全 = 1.75**

```text
mean = 1.75，std_pop = 0，分母 = 0 + 1e-4 = 1e-4
A_i = 0 / 1e-4 = 0（全部 6 条）—— 是 0，不是 nan，这正是 1e-4 的作用
```
此时 token 级 loss 里策略梯度项整体为 0，只剩 **β·KL(π_θ ‖ π_ref)** 一项。梯度把策略往 **ref 方向**拉，即把这一步变成纯粹的"往回收敛到 SFT 模型"，没有任何来自 reward 的学习信号。

**（3）G=2，r = [2.25, 0.75]**

```text
mean = 1.50，偏差 ±0.75
std_pop = sqrt((0.75² + 0.75²)/2) = sqrt(0.5625) = 0.75
分母 = 0.7501
A = ±0.75 / 0.7501 = ±0.999867
```

**（4）G=2，r = [2.00, 1.90]**

```text
mean = 1.95，偏差 ±0.05
std_pop = sqrt((0.05² + 0.05²)/2) = 0.05
分母 = 0.0501
A = ±0.05 / 0.0501 = ±0.998004
```

**一般结论**：G=2 且两条 reward 不相等时，`std_pop = |r₁−r₂|/2`，于是 `A = ±(|r₁−r₂|/2) / (|r₁−r₂|/2 + 1e-4) ≈ ±1`，**与 reward 差距的绝对大小几乎无关**。组内归一化把"差多少"完全抹掉，只保留了"谁更好"这一个符号位。因此 G=2 时 GRPO 退化成一个每步幅度恒定的成对比较，reward 的量纲信息被完全丢弃；差距 0.10 与差距 1.50 得到同样强度的更新。

**0–4 分 rubric**

- 0：advantage 公式用错（例如用无偏 std 或漏掉 1e-4）。
- 1：第 1 问的 mean/std 正确但 advantage 算错。
- 2：第 1、2 问全对（含"是 0 不是 nan"）。
- 3：1–3 问数字全对。
- 4：四问全对，且第 4 问明确写出 `A ≈ ±1` 的一般式并指出"量纲信息被抹掉"这一后果。

**常见误区**

- 用样本标准差（分母 G−1）算。`unbiased=False`，分母是 G。G=6 时两者差 √(6/5) ≈ 1.095，会让全部 advantage 系统性偏小约 9%。
- 认为 std=0 时 advantage 是 nan。1e-4 保证了它是 0，因此故障表现是"静默不学习"而不是"报错"，这正是它难被发现的原因。

**追问**

- 把 1e-4 改成 1e-8，第 4 问的 A 会变成多少？这对 reward 只有微小随机差异的组意味着什么？
- G=2 时若想保留 reward 的量纲信息，你会怎么改 advantage 的定义？这个改动会破坏 GRPO 的哪个性质？

**定位**：`01_FOUNDATIONS.md` §2.6、§3.1 不变量 8、§6 第 8 条；`lab/src/mm_probe/grpo_stats.py` 的 `group_advantage` / `_pstd` / `summarize`。

---

## Debug

### M01-D-01

**参考答案要点**

**Run A（bf16 SFT，loss 与 grad_norm 同时恒为 0，权重不变）**

1. **被破坏的不变量**：`(labels != -100).sum() > 0` 对每个 micro-batch 成立（§3.1 不变量 1）。可执行判据：对连续 64 个 micro-batch 断言 `int((labels != -100).sum()) > 0`。
2. **互斥假设**：
   - H1：label 生成路径把 assistant 段也标成了 -100（`mask_fault.py --mode assistant_all_ignored` 等价的故障），有效 label 数恒为 0，CE 在空分母下返回 0（部分 PyTorch 版本返回 nan），梯度全 0。
   - H2：优化器里没有任何 `requires_grad=True` 的参数（例如误挂了 LoRA 却没把 lora 参数放进优化器，或整模型被 `requires_grad_(False)`）。
   - H3：数据本身是空的（`text`/`conversations` 字段为空），样本被 tokenize 成全 pad。
3. **最小检查**：跑 `lab/src/mm_probe/inspect_dataset.py`（`sample_stats` / `aggregate`）对前 1000 条样本统计 `n_label`（有效 label 数）与 pad 比例。
   - H1 成立：`n_label` 恒为 0，但 pad 比例正常、token 数正常。
   - H2 成立：`n_label` 正常（十几到几百），故障在模型侧 —— 再断言 `sum(p.numel() for p in model.parameters() if p.requires_grad) > 0`。
   - H3 成立：`n_label` 为 0 且非 pad token 数也接近 0。
4. **`grad_norm:0.000000` 排除了什么**：排除了"梯度爆炸/溢出"整条路线，也排除了 lr 设置问题（lr 正常且在变化）。它把范围收窄到"要么没有 target，要么没有可训参数"。**Run B 的 `grad_norm:nan`** 反过来排除了"没有学习信号"，说明前向确实产生了非零但非有限的中间量。
5. **dtype 能否消除**：Run B 可以 —— 切到 bf16 后指数位与 fp32 相同，65,504 的溢出上限消失（§4.1/§4.2）。Run A **不能** —— 它是 label 构造故障，与数值格式无关；换 dtype 后 loss 仍是 0。这本身就是一个廉价的区分实验。

**Run B（fp16 pretrain，loss 从正常值变 nan，scale 单调减半）**

1. **被破坏的不变量**：`torch.isfinite(loss)` 每步为真；`scaler.get_scale()` 的健康形态是"偶尔减半、随后每 2000 步翻倍"，不是单调减半（§4.2、§6 第 5 条）。
2. **互斥假设**：
   - H1：前向中间量溢出（`silu(gate)⊙up` 或 `q·kᵀ/√96` 超过 65,504），产生 inf logits → loss = nan。scaler 只能反复跳步，不能修复。
   - H2：反向梯度溢出，但前向仍有限 —— 这种情况 loss 本身应该是有限的，与观察到的 `loss:nan` 矛盾。
   - H3：lr 过大导致权重发散，数值溢出只是结果不是原因。
3. **最小检查**：把同一 seed、同一数据切片用 `--dtype bfloat16` 重跑 500 步，用 `lab/src/mm_probe/hooks.py` 的 `grad_global_norm` 逐步记录，并用 `parse_log.py` 画 loss 与 grad_norm。
   - H1 成立：bf16 下 loss 全程有限且平滑下降，grad_norm 有界 → 纯精度问题。
   - H3 成立：bf16 下 loss 同样发散（或 grad_norm 在 nan 出现前就已单调上升几个数量级）→ 是 lr/初始化问题，换回 fp16 也没用。
4. 见 Run A 第 4 点。
5. 见 Run A 第 5 点。

**0–4 分 rubric**

- 0：只说"数据有问题"或"精度有问题"，给不出判据。
- 1：对其中一个 Run 给出了正确的不变量。
- 2：两个 Run 都给出正确不变量，但只有一个假设、没有区分实验。
- 3：两个 Run 各给出 ≥2 个互斥假设 + 一个可执行的区分检查（脚本名 + 看哪个数字）。
- 4：以上全对，并答出第 4、5 问 —— 用 `grad_norm` 的 0 与 nan 做排除、用"换 dtype 重跑"作为区分 A 与 B 的最廉价实验。

**常见误区**

- 看到 Run A 的 loss=0 就判断"模型已经完美收敛"。0 loss 配 0 grad_norm 且生成逐字不变，是"没有 target"而不是"学会了"。
- 看到 Run B 的 scale 一路减半就去调 `init_scale`。scale 只是症状；单调减半意味着**每一步**都溢出，问题在前向或 lr，不在 scaler 参数。

**追问**

- 若 PyTorch 版本对"全部 target 都是 ignore_index"返回 nan 而不是 0，Run A 的日志会变成什么样？你的区分实验还成立吗？
- Run B 里 `grad_norm:1.0000` 恰好等于 clip 阈值。这是巧合还是必然？它提供了什么额外信息？

**定位**：`01_FOUNDATIONS.md` §4.1、§4.2、§6 第 5/6 条、§3.1 不变量 1；`lab/src/mm_probe/mask_fault.py`（`assistant_all_ignored` 的期望症状注释）、`hooks.py`、`inspect_dataset.py`、`parse_log.py`。

---

### M01-D-02

**参考答案要点**

1. **两处角色越界**：(a) 模型先原样复述了 user 的问题 `什么是久期？`，而不是直接作答；(b) 模型自己生成了 `<|im_start|>user\n什么是凸性？`，即它在扮演 user 角色继续对话。
2. **被破坏的不变量**：user/system 段与所有模板 token 必须全部为 -100（§3.1 不变量 2）。单条样本可执行断言：

   ```text
   对每条样本：labels[t] != -100  ⟹  t 落在某个 assistant 段的 [内容起点, eos 末尾] 区间内
   等价的廉价形式：有效 label 数 == Σ(assistant 内容 token 数 + 2)
   ```
   故障时有效 label 数会变成"整条序列的非 pad token 数"。
3. **为什么"初值更低 + 更平滑"是必然结果**：进入 loss 的绝大多数位置变成了模板 token（`<|im_start|>`、`user`、`\n`、`<|im_end|>`）和 user 文本。模板 token 是**完全确定**的（给定前缀，下一个 token 几乎无歧义），user 文本也比 assistant 生成更容易被拟合。分母被大量低熵位置稀释 → 平均 CE 必然更低；而这些位置的 loss 逐 batch 几乎不变 → 曲线抖动被压掉。所以这不是巧合，是"往分母里塞进一大批可预测位置"的算术后果。
4. **最小检查**：`lab/src/mm_probe/inspect_dataset.py`（`--show-labels` / `sample_stats`）统计每条样本的 **有效 label 数 / 非 pad token 数** 之比。
   - 正常：这个比值等于 assistant 段占比，SFT 数据上通常远小于 1（量级 0.1–0.4）。
   - 故障：比值 ≈ **1.0**（所有非 pad token 都进 loss）。
   判据用比值而不是绝对数，可以不依赖具体数据集的长度分布。
5. **可控复现**：`python -m mm_probe.mask_fault --mode user_in_loss`，把注入后的 labels 接进有界训练。要采集的两条曲线：
   - 曲线一：`--mode none` 的 loss-vs-step（基线）；
   - 曲线二：`--mode user_in_loss` 的 loss-vs-step。
   两条必须共享 seed、数据切片、步数、lr 调度，只差 labels 一项。再各自用同一组固定 prompt 做生成对照。
6. **只看 loss 曲线能不能区分"故障"与"数据更容易"**：单看曲线不能，两者都表现为更低更平滑。区分实验：**保持数据集完全不变**，只切换 `mask_fault --mode`，跑两条曲线。数据集是受控变量，若切换 mode 就复现出观察到的曲线形态，则故障假设成立；若两条曲线都在观察值附近（说明数据本身就容易），则是数据假设。另一个独立判据：故障运行的 **CE 分母**（有效 label 数）会比基线大一个量级，这个量与"数据难易"无关，可以直接从 `inspect_dataset.py` 读出。

**0–4 分 rubric**

- 0：只说"mask 错了"，指不出是哪一种错。
- 1：正确指认 `user_in_loss`，但给不出可执行断言。
- 2：断言正确 + 说清"初值低"的原因。
- 3：1–5 问全对，含 `mask_fault.py --mode user_in_loss` 与两条对照曲线的受控变量说明。
- 4：以上全对，且第 6 问给出了两个独立的区分判据（受控切换实验 + CE 分母量级），并说明为什么分母这个量与数据难易正交。

**常见误区**

- 把"生成复述 user"归因为"SFT 步数不够"或"数据里有复述样本"。这两个假设无法解释"模型主动生成 `<|im_start|>user` 标记"，因为正常训练下这个标记从不作为 target 出现。
- 只跑故障运行不跑基线，导致"更低更平滑"没有参照物，无法量化。

**追问**

- 若故障是"system 段进 loss 但 user 段正常"，第 1 问的两处越界还会出现吗？你的判据要怎么改？
- `user_in_loss` 下 loss 数值与正常曲线是否可比？在报告里你会怎么标注这一点？

**定位**：`01_FOUNDATIONS.md` §6 第 3 条、§2.3、§3.1 不变量 2；`lab/src/mm_probe/mask_fault.py`（`FAULT_MODES` 与 `user_in_loss` 的期望症状注释）、`inspect_dataset.py`、`parse_log.py`。

---

### M01-D-03

**参考答案要点**

1. `adv_std = 0` 意味着 **advantage 向量全为 0**。原因：组内 reward 全等 → `std_pop = 0` → `A_i = (r_i − mean) / (0 + 1e-4) = 0 / 1e-4 = 0`。分母上的 `1e-4` 保证了这是精确的 0 而不是 `0/0 = nan`，这也是故障静默的根源。
2. loss 非零且缓慢增大，说明策略梯度项已归零，**只剩 β·KL** 一项。它把 π_θ 往 π_ref（SFT 模型）方向拉。`kl_ref` 从 0.0009 单调涨到 0.0052 说明策略确实在偏离 ref、而 KL 惩罚在追着它 —— 这个方向"不对"，因为它不携带任何 reward 信息，纯粹是把模型往回收；持续下去只会消耗步数并轻微破坏已有能力，不会带来任何改进。
3. **三类独立原因**：
   - **采样问题**：`num_generations=2` 太小 + 采样温度/top-p 过窄 → 两条 response 几乎相同甚至逐字相同，reward 自然相等。
   - **reward 函数问题**：规则 reward 是**离散的形状分**（长度项 ±0.5、思考块项 +1.0/−0.5、`</think>` 计数项 +0.25、3-gram 惩罚上限 0.5），当两条 response 都落在同一档时分数完全相等；reward model 未加载意味着唯一的连续分量缺席。
   - **prompt 问题**：RLAIF prompt 过于简单或高度模板化，导致模型对同一 prompt 的输出分布退化（低熵），任何 G 下都难产生差异。
4. **实验顺序与判据**（都用 `lab/src/mm_probe/grpo_stats.py`）：
   - 第 1 步：`parse_debug_log` / `group_stats` 逐组打印 **reward 向量本身**与 `std==0` 组占比。若占比接近 100%，继续；若只有部分组为 0，问题没那么严重，转去看分量。
   - 第 2 步：打印每组的 **response 文本或其哈希**。若组内 response 逐字相同 → 采样问题（回到采样超参）；若 response 不同但 reward 相同 → 排除采样问题，进入第 3 步。
   - 第 3 步：把 `calculate_rewards` 的四个分量**拆开**记录。若四个分量逐条相同 → reward 函数分辨率不足（离散档位太粗）；若某个分量本该不同却恒定 → 该分量实现或输入有问题。
   - 第 4 步（仅在前三步都排除后）：换一批更开放的 prompt，看 `std==0` 占比是否下降 → prompt 问题。
5. **`reward` 恒 = 1.7500 的线索**：它正好是"长度项 +0.5 + 思考块项 +1.0 + 恰好一个 `</think>` +0.25 = **1.75**"，即**只拿满形状分、重复惩罚为 0、reward model 分量缺席**。结合 `gen_len` 稳定在 34–35 字，可以判断：模型输出稳定落在形状奖励的同一档内，规则 reward 已经**饱和**，它对当前策略没有任何区分度。这也说明 reward hacking 已经"完成"—— 模型拿到了形状分的上限。
6. **只改一个超参数**：把 `num_generations` 从 2 提到 6（受显存约束时至少提到 4）。理由：G 增大直接降低"整组同分"的概率，是最小侵入且不改变 reward 定义的改动。预期 `grpo_stats.py` 里**先动的是 `std==0` 组的占比**（从接近 100% 下降），随后 `adv_std` 才会离开 0，最后 `reward` 才可能变化。若提到 6 后 `std==0` 占比仍接近 100%，则采样假设被证伪，必须动 reward 函数（加入 reward model 或细化档位）。

**0–4 分 rubric**

- 0：把 `adv_std=0` 当成"训练稳定"的好现象。
- 1：说出 advantage 全 0 与"没有学习信号"。
- 2：再说清"loss 非零是因为只剩 KL 项"以及它把策略推向 ref。
- 3：给出 ≥3 个互斥原因 + 一个有序的区分实验方案，每步带判据。
- 4：以上全对，且识别出 1.7500 = 0.5+1.0+0.25 的分量拆解，并给出"先动 `std==0` 占比"这一可证伪的预期。

**常见误区**

- 认为 `adv_std=0` 会导致 loss=0，看到 loss 非零就否定这条线索。KL 项与 advantage 无关，始终存在。
- 直接跳去调 lr 或 β。在 advantage 恒为 0 时，lr 和 β 的任何调整都只是改变"往 ref 回收的速度"，不产生学习信号。

**追问**

- `std==0` 组占比多少时你判定必须干预？给出阈值和它的依据。
- 若加载了 1.8B reward model 后 `std==0` 占比降到 5%，但 `adv_std` 仍很小，你下一步查什么？

**定位**：`01_FOUNDATIONS.md` §2.6、§6 第 8/9 条、§3.1 不变量 8；`lab/src/mm_probe/grpo_stats.py` 的 `group_advantage` / `group_stats` / `summarize` / `parse_debug_log`。

---

### M01-D-04

**参考答案要点**

1. traceback 停在 `scores = F.softmax(scores.float(), ...)`，这是**手写 attention** 路径，说明没走 `F.scaled_dot_product_attention`。走 SDPA 的**完整条件**（§1.3、§5.2）是三条同时成立：
   - `attention_mask is None` 或 **全为 1**；
   - 没有 KV cache（`past_key_values` 为空 / 非增量解码）；
   - 由 `is_causal=True` 提供因果性。
   任一条不满足就落到手写分支，materialize 完整的 `[*, *, T, T]` 分数矩阵。
2. GRPO 的训练前向传入 `attention_mask = (outputs != pad)`。`batch_size=2` 时两条 prompt 长度不同，为了对齐要做 **左 pad**，mask 里就出现了 0 → 第一个条件被破坏 → 走手写 attention。`batch_size=1` 时只有一条序列，无需 pad，mask 全 1 → 走 SDPA（flash / mem-efficient kernel，不 materialize T×T）。关键不是 batch 大小本身，而是**批内 prompt 是否等长**：若两条 prompt 恰好等长，`batch_size=2` 也不会触发。
3. **shape 反推**：

   ```text
   dim0 = B × G = 2 × 6 = 12                     （每个 prompt 采 G 条 response）
   dim1 = num_attention_heads = 8                （repeat_kv 后 q/k/v 都是 8 head）
   dim2 = dim3 = max_seq_len + max_gen_len = 768 + 1024 = 1792
   scores.shape = [12, 8, 1792, 1792]
   元素数 = 12 × 8 × 1792 × 1792 = 96 × 3,211,264 = 308,281,344
   fp32   = 308,281,344 × 4 B = 1,233,125,376 B = 1.233 GB = 1.1485 GiB ≈ 1.15 GiB ✓
   与报错的 "Tried to allocate 1.15 GiB" 一致（报错用 GiB，本周表格用十进制 GB，两者不矛盾）。
   ```
   注意 bf16 的 `scores` 本身另占 616 MB，且 `F.softmax` 的 fp32 输出会被 autograd 保留给反向，所以**每层**这条路径的额外开销约 1.85 GB，8 层约 14.8 GB —— 单这一项就打爆 16 GB。
4. **为什么不是线性下降**：`batch_size` 减半只让 token 数减半（线性项：KV cache、logits、每层激活），但它同时**切换了 attention 分支**。手写分支的开销是 `O(B·G·n_h·T²)`，是 T 的平方项，且被 8 层累加；切到 SDPA 后这一项**整体消失**（flash kernel 分块计算，不保存 T×T）。所以 13.71 GB → 5.9 GB 是"线性项减半"叠加"平方项归零"的结果，不可能用单一比例解释。
5. **不变量断言与监控**：

   ```text
   断言：训练前向进入 Attention.forward 时，若 attention_mask 非 None，则 attention_mask.all() 为真
   （等价：所有参与本次前向的序列等长，无 pad）
   ```
   监控：`lab/src/mm_probe/hooks.py` 的 `peak_memory_mb(reset=True)` 分段记录 rollout / reward / policy 前向 / ref 前向 / backward 五段的峰值；`grpo_stats.py` 在 `Attention.forward` 上挂 hook，把"本次前向走的分支"（sdpa / manual）作为一列写进每步记录。两者合起来才能把"峰值升高"与"分支切换"关联上，单看显存数字无法归因。
6. **保持 `batch_size=2` 的两条路**：
   - **把批内 prompt 补到等长**（按同一目标长度做**右**侧对齐或统一截断），使 `attention_mask` 全 1 → 仍走 SDPA。代价：短 prompt 被填充或长 prompt 被截断，改变了 prompt 分布，rollout 的语义被动过，需要在报告里标注。
   - **把 `max_seq_len + max_gen_len` 压下来**（例如 256+256=512）。平方项从 1792² 降到 512²，是 12.25 倍，即使仍走手写分支也能放下。代价：限制了 response 长度，直接影响长度类 reward 分量与可学到的行为空间。
   两条都不改变分支条件的判定逻辑本身，区别在于第一条消除平方项、第二条缩小平方项。

**0–4 分 rubric**

- 0：只答"batch 太大，减小 batch"。
- 1：认出走了手写 attention 分支。
- 2：写出完整的三条 SDPA 条件 + 解释左 pad 的作用。
- 3：1–4 问全对，含 shape 反推与 1.15 GiB 的验证。
- 4：以上全对，且第 6 问给出两条代价清晰的替代方案，并指出"等长 prompt 时 `batch_size=2` 也安全"这一失败边界（即 batch 大小不是根本变量）。

**常见误区**

- 把根因说成"batch_size=2 计算量翻倍"。翻倍只解释线性项，解释不了 13.71 → 5.9 这个幅度。
- 把 GB 与 GiB 混算后认为 1.233 与 1.15 对不上，从而否定 shape 反推。两者是同一个数的不同进制表示。

**追问**

- 若只把 `max_gen_len` 从 1024 降到 256（其他不变，仍是 `batch_size=2`），`scores` 的 fp32 大小变成多少？还会 OOM 吗？
- SDPA 分支下反向的显存复杂度是 O(T) 还是 O(T²)？这决定了你能把 `max_gen_len` 推到多大。

**定位**：`01_FOUNDATIONS.md` §1.3（attention 分支条件）、§5.2（三模型同驻与左 pad 陷阱）、§6 第 7 条；`lab/src/mm_probe/hooks.py` 的 `peak_memory_mb`、`grpo_stats.py`。

---

### M01-D-05

**参考答案要点**

1. **拆分**：
   - **预期后果**：恢复后前几步出现一次性抬升（2.73 → 2.61 快速回落）。本周已知 `_resume.pth['model']` 是 `half()` 权重、优化器动量与被舍入的权重不再匹配、数据增广 RNG 流改变、SDPA 反向非确定 —— 这些都会造成短暂扰动。
   - **需要额外解释**：恢复后第 80–100 步均值 **2.441** 高于中断前最后 100 步的 **2.403**。这不是噪声：单步标准差约 0.022，100 步均值的标准误约 0.002 量级（即使考虑 loss 序列自相关，把有效样本数打折到 1/10，标准误也只有约 0.007），而差值是 0.038。更强的证据是**训练本应继续下降**：step 680–700 的均值应低于 step 500–600，实际却更高。所以存在一个持续性偏差。
2. **三个机制及其形态**：

   | 机制 | 更可能造成 |
   | --- | --- |
   | `_resume.pth['model']` 存的是 half 权重，恢复后与 fp32 优化器状态失配 | **一次性跳变后回落**（优化器几十步内重新适配） |
   | 数据增广用 `random.random()`（system prompt 20%、空思考块 80%），`SkipBatchSampler` 跳过 batch 后 RNG 流与原来不同 | **一次性跳变**（若只是样本顺序不同）或 **永久性抬高**（若增广比例被系统性改变） |
   | SDPA 反向非确定（kernel 选择与归约顺序） | 只造成**噪声级别**的不重合，既不跳变也不抬高 |

3. **精度推算**：`_resume.pth['model']` 是 **fp16（half）**。fp16 相对精度约 `9.8e-4`（尾数 10 位）。典型量级 0.02 的权重舍入后绝对误差约 `0.02 × 9.8e-4 ≈ 2e-5`，相对漂移约 **0.098 %**。对照 §4.4 的论证：DPO 里 1% 的相对漂移就足以冲掉 SFT 学到的能力，0.1% 大约小一个数量级。**判断**：这个量级足以解释 2.40 → 2.73 的**瞬时**跳变（权重被扰动、动量方向不再匹配），但**不足以解释 80–100 步后仍高出 0.038 的持续差**——因为优化器在几十步内就能吸收 0.1% 的扰动。依据是"跳变确实在 5 步内回落了一半以上"，说明扰动是可恢复的。
4. **RNG 流的影响**：`SkipBatchSampler` 只保证**跳过哪些 batch**，不保证跳过期间 `random.random()` 被消费同样的次数。恢复后每条样本落到"插 system prompt"和"删空思考块"两个分支的具体结果与原来不同。它影响的**不是单条样本的正确性**（两个分支都是合法样本），而是**数据分布**：如果实现上跳过 batch 时完全不调用增广，那么恢复后的 RNG 相位整体偏移，20%/80% 的期望比例仍然正确，只是具体落点不同 → 短期扰动；但若恢复后随机种子被重置到固定值，则可能长期抽到同一批分支，比例发生系统性偏移 → 持续抬高。
5. **恢复正确性验收判据**：

   ```text
   窗口   : 中断前最后 W=100 步 与 恢复后第 (W+1)..(2W) 步（即跳过前 100 步瞬态）
   统计量 : 两个窗口的 loss 均值 m1, m2；以及各自的块自助（block bootstrap，块长 20）标准误 se1, se2
   通过   : m2 ≤ m1 + 2·sqrt(se1² + se2²)   且   m2 的趋势斜率为负（仍在下降）
   不通过 : m2 显著高于该阈值，或斜率非负
   ```
   **不能用"逐步 loss 相等"作判据**，因为 half 舍入、RNG 相位、SDPA 反向非确定这三条都保证了 bitwise 复现在本设置下是**不可能**的；要求逐步相等会让判据永远失败，从而失去区分"正常不重合"与"真故障"的能力。
6. **永久停在 2.44 时的怀疑对象**：最可能是 **lr 调度的 step 计数没有正确恢复** —— `get_lr(s, S, lr)` 以 micro-batch 计数，若 `s` 从 0 重新开始，恢复后的 lr 会跳回峰值 5e-4（而中断点本应已衰减到更低），高 lr 会把 loss 永久抬在一个更高的平台上。次选是 optimizer 的 `m, v` 状态没被加载（等价于重新 warm up 动量）。
   **能直接证伪的检查**：打印恢复后第 1 步的 `lr` 与中断前最后 1 步的 `lr` 并比较。若恢复后 lr 明显更大（例如回到 5e-4 量级而不是接近中断点的值），假设成立；若两者连续，则转去断言 `optimizer.state_dict()['state']` 非空且 `exp_avg` 的范数与中断前同量级。

**0–4 分 rubric**

- 0：认为 resume 必须 bitwise 复现，把所有差异都判为 bug。
- 1：说出至少一个非 bitwise 的机制。
- 2：三个机制都说出，并区分了"预期跳变"与"需要解释的持续差"。
- 3：1–4 问全对，含 half 精度的定量推算与"够解释跳变、不够解释持续差"的判断。
- 4：以上全对，且第 5 问给出了带窗口长度、统计量和阈值的可执行验收判据，第 6 问给出了一个能直接证伪的检查（比较恢复前后的 lr）。

**常见误区**

- 用"loss 数值差不多"这类定性判断代替统计判据，导致 0.038 这种真实但不显眼的持续偏差被放过。
- 把 SDPA 反向非确定当成主要嫌疑。它只造成噪声级不重合，量级远小于观察到的跳变。

**追问**

- 如果 `_resume.pth` 改成保存 fp32 master 权重，文件大小和第 3 问的推算各变成什么？这个改动值得吗？
- 恢复到不同 `world_size` 时 `step = step × saved_ws // current_ws` 的换算，会对第 6 问的 lr 检查带来什么额外的混淆项？

**定位**：`01_FOUNDATIONS.md` §4.2、§4.3、§4.4、§5.4、§6 第 12 条、§8 第 9 条；`lab/src/mm_probe/parse_log.py`（`load_rows` / `numeric_fields` 做窗口统计）、`hooks.py`。

---

## Design

### M01-P-01

**参考答案要点**

1. **核心论点**：`batch_size=1` 让批内只有一条序列、无需 pad，`attention_mask` 全 1，模型走 `F.scaled_dot_product_attention` 的 flash/mem-efficient kernel，**不 materialize `[B·G, n_h, T, T]` 的分数矩阵**。`batch_size≥2` 且 prompt 不等长时出现左 pad，mask 含 0，落到手写 attention，每层多出一个 `O(T²)` 的 fp32 softmax 输出被 autograd 保留。所以改变的不是"算多少 token"，而是**显存复杂度从 O(T) 变成 O(T²)、且乘以层数**。
2. **实验设计**（至少三个配置点，其余变量全部固定：seed、数据切片、`num_generations`、`max_seq_len`、`max_gen_len`、dtype、步数）：

   | # | 配置 | 目的 |
   | --- | --- | --- |
   | C1 | `batch_size=1`，prompt 任意 | 基线，预期走 SDPA |
   | C2 | `batch_size=2`，两条 prompt **长度不同** | 预期走手写分支 |
   | C3 | `batch_size=2`，两条 prompt **人为补到等长**（mask 全 1） | 关键对照：token 数与 C2 相同，但分支应回到 SDPA |
   | C4 | `batch_size=1`，`num_generations` 翻倍使总 token 数 ≈ C2 | 排除"token 数"解释 |

   每个配置点记录：**分支标识**（`sdpa` / `manual`，来自 hook）、`max_memory_allocated`（GB）、总 token 数、每步耗时。预期结果表：C1、C3、C4 分支为 `sdpa` 且显存与 token 数近似线性；C2 分支为 `manual` 且显存显著高出（不在同一条线性关系上）。
3. **工具**：显存用 `lab/src/mm_probe/hooks.py` 的 `peak_memory_mb(device, reset=True)`，在每段（rollout / policy 前向 / backward）前后各调一次。分支证据需要在 `Attention.forward` 上挂 **forward pre-hook**，读取传入的 `attention_mask`：若为 `None` 或 `.all()` 为真则记 `sdpa`，否则记 `manual`，并把 `scores` 是否被创建（可通过 forward hook 观察中间张量或直接记录该判定结果）写入每步记录，交给 `grpo_stats.py` 汇总成一列。**关键是把"分支"变成一个可写进日志的离散变量**，否则只有显存数字，无法归因。
4. **失败边界**：若批内两条 prompt **恰好等长**，`batch_size=2` 也不会产生 pad，mask 全 1，仍走 SDPA —— 此时 C2 与 C3 无差别，实验退化，无法区分两个解释。同样，如果实现里对 mask 做了"全 1 时置 None"的短路，或数据集本身已被统一填充到固定长度，也会掩盖分支切换。设计时必须先验证 C2 确实产生了含 0 的 mask。
5. **替代解释与排除**：替代解释是"显存下降只是因为 token 数减半"。**C4 专门排除它**：C4 保持 `batch_size=1`（走 SDPA）但把 `num_generations` 翻倍，使总 token 数与 C2 相当。若 C4 的显存远低于 C2，则"token 数"解释被证伪，只剩"分支切换"能解释差值。C3 从另一侧加固：token 数与 C2 相同、分支不同、显存回落。
6. **可验证的下一步**：证明成立后，用**显式显存模型**而不是试到 OOM 来定 `num_generations` 上限。在确保走 SDPA 的前提下，显存 ≈ 静态项（policy 1.02 GB + ref 0.26 GB + reward 3.4 GB）+ 线性项（激活 240 KB/token × `B·G·(seq+gen)` + logits `B·G·T·V×bytes` + KV cache 12 KB/token）。把可用显存减去静态项与安全余量后除以每 token 的线性系数，即得 token 预算，再除以 `(seq+gen)` 得到 `B·G` 上限。用 C1/C4 两点实测的 (token 数, 峰值) 拟合出线性系数并校准该模型，然后**外推预测** G=4、G=6 的峰值，再实测验证预测误差 —— 预测误差在 15% 以内即认为模型可用。

**0–4 分 rubric**

- 0：论点仍停留在"batch 小省显存"。
- 1：论点正确（分支切换）但没有实验设计。
- 2：给出 ≥3 个配置点，含显存记录。
- 3：实验设计含分支标识的采集方式 + 明确的排除性配置点（C3 或 C4）。
- 4：以上全对，且给出失败边界（等长 prompt 时实验退化）与"用显存模型外推 + 预测误差检验"的可验证下一步。

**常见误区**

- 只记显存不记分支，导致结论无法排除"token 数"这个混杂变量。
- 忘了控制 `num_generations`：改 batch 的同时改了 G，序列数变了两次，实验失去可比性。

**追问**

- 如果只能改一处代码来永久规避这个陷阱，你会改哪里？改 dataset 的 pad 策略还是改 `Attention.forward` 的分支条件？各自的风险是什么。
- 你的显存线性模型在什么情况下会失准？至少说出两个非线性来源。

**定位**：`01_FOUNDATIONS.md` §1.3、§5.1、§5.2、§6 第 7 条；`lab/src/mm_probe/hooks.py`、`grpo_stats.py`；`00_WEEK_CARD.md`「三类环境边界」。

---

### M01-P-02

**参考答案要点**

1. **指标定义**：
   - **长度类 —— fertility**：`fertility = token 数 / 字符数`（中文按字符计），或其倒数"字/token"。分母必须明确是**去掉空白后的原文字符数**，否则中英混排时会被空格稀释。对照基准：本周核验的通用中文约 **1.5 字/token**（fertility ≈ 0.67 token/字）。
   - **覆盖类 —— byte-fallback 率 / 碎片率**：`碎片率 = 被切成 ≥3 片的目标术语数 / 目标术语总数`；ByteLevel BPE 没有真正的 UNK，罕见字会退化成多个字节级 token，所以要测的是"单个汉字被切成几个 token"的比例（`单字节化率 = 平均每汉字 token 数 > 1 的字符占比`）。
2. **语料切分（四类，必须分开统计）**：
   - **纯中文叙述**（研报正文、公告段落）：作为与通用中文的可比基线。
   - **数字与金额/百分比/日期**（`1,234.56 亿元`、`-3.72%`、`2025Q3`）：数字在 ByteLevel BPE 下常被逐位切开，fertility 与文本完全不同量级。
   - **代码型标识**（股票代码 `600519.SH`、`000001.SZ`、ISIN、科目编号）：这些是高频但字符级随机的串，最容易爆 token。
   - **专业术语与中英混排**（`久期`、`凸性`、`EBITDA`、`WACC`、`可转债`、`Basel III`）：检验词表里有没有金融领域的常用 subword。
   必须分开统计的理由：这四类的 fertility 分布差异可达数倍，混在一起取平均会被占比最大的类（通常是纯中文叙述）掩盖，从而得出"还行"的错误结论；而真正吃 token 的是数字与代码型标识。
3. **可执行步骤**：
   - 用 `lab/src/mm_probe/minimind_env.py` 的 `find_minimind_root` + `load_tokenizer(root)` 拿到固定 commit 的 tokenizer。
   - 用 `lab/src/mm_probe/inspect_dataset.py` 的 `pretrain_encode(tokenizer, text, max_length)` 逐条编码（它返回 token 数与 pad 信息），或直接用 `special_sequences` 之外的编码路径按段落调用。
   - 输出表格：每行一类语料，列为 `样本数 / 总字符数 / 总 token 数 / 字符每 token / token 每字符 / P50 / P90 / 最差样本示例`。P90 与最差样本必须列出，因为 `max_seq_len` 截断是由尾部决定的，不是由均值决定的。
4. **判定门槛与下游换算**：
   - 门槛：若某一类语料的"字/token"低于通用中文基线的 **60 %**（即 1.5 → 低于 0.9 字/token），判定为该类不适合；若整体加权后低于 70 %，判定 tokenizer 整体不适合。
   - **后果一（`max_seq_len`）**：同一段原文需要的 token 数按同样比例上升。若基线 768 token 能装 ~1150 字，fertility 差 40% 后只能装 ~690 字，意味着要么把 `max_seq_len` 提高到 1280 才能装同样内容，要么接受截断。截断直接改变 `generate_labels` 的行为（§1.2：扫不到 eos 就标到末尾），影响 label 正确性，不只是"少看点内容"。
   - **后果二（训练成本）**：前反向 FLOPs ≈ `6·N·token 数`（§5.3），token 数上升 k 倍则计算量上升 k 倍；显存侧 logits 项 `B×T×V×bytes` 也随 T 线性上升。把 fertility 比值直接乘上去即可换算训练时长与显存。
5. **三条对策与代价**：
   - **扩充/重训 tokenizer（加大词表）**：fertility 改善，但 `V` 上升会让 `B×T×V×bytes` 的 logits 项线性上升（§5.1 已给出 V=6400 → 151,936 时从 157 MB 涨到 3.73 GB 的量级），而且 `tie_word_embeddings=True` 意味着 embedding 与 lm_head 同时变大；最致命的代价是**已有的 pretrain/SFT checkpoint 全部作废**（embedding 行数变了）。MiniMind README 也明确不建议重训。
   - **只做词表增补（在原词表尾部追加金融高频词）**：保留旧 checkpoint 的前 6400 行，新行随机初始化。代价是新 token 的 embedding 与 lm_head（tied）从零开始，早期会产生高 loss 区域；且必须重跑 CPT 才能把新 token 训熟。
   - **不动 tokenizer，改用数据侧规范化**（把金额/日期/代码统一成固定格式或占位符）：零模型改动、旧 checkpoint 全部可用。代价是引入了不可逆的信息损失，且推理时必须做同样的规范化，链路复杂度上升。
6. **失败边界**：
   - 语料不代表真实分布（例如只采了研报正文而没采表格与附注），测出的 fertility 偏乐观。
   - 只看均值不看 P90/最差样本，会漏掉"少数极长样本被截断"这个真正影响 label 正确性的问题。
   - 字符数定义不一致（含不含空白、全角半角）会让不同类之间的 fertility 不可比。
   - 该测量只回答"编码效率"，**不回答"语义是否被切碎影响建模"**。fertility 正常但术语被切成无意义碎片时，这套指标给不出信号，需要额外的下游任务评测。

**0–4 分 rubric**

- 0：只说"看看 token 多不多"。
- 1：给出一个指标与公式。
- 2：给出两类指标 + 语料分类。
- 3：1–4 问全对，含可执行步骤（脚本名 + 函数名）与量化门槛。
- 4：以上全对，且第 5 问的三条对策各自给出了本周知识范围内的具体代价（logits 显存、tied embedding、checkpoint 失效），第 6 问给出 ≥3 条失败边界含"fertility 正常但语义被切碎"这一层。

**常见误区**

- 用一个全语料平均 fertility 下结论，掩盖数字与代码型标识这两类的爆炸。
- 认为"换个大词表 tokenizer"是低成本改动，忽略它会同时放大 logits 显存并作废已有 checkpoint。

**追问**

- 你的门槛 60% 是怎么来的？如果换成 80% 或 40%，第 4 问的两个后果分别变成什么？给出一个把门槛与训练预算挂钩的定量方法。
- 若必须在"截断"和"提高 `max_seq_len`"之间二选一，你用哪个观测量做决策？

**定位**：`01_FOUNDATIONS.md` §1.1、§5.1、§5.3、§7；`lab/src/mm_probe/minimind_env.py` 的 `load_tokenizer`、`inspect_dataset.py` 的 `pretrain_encode` / `sample_stats` / `aggregate`。

---

### M01-P-03

**参考答案要点**

1. **数据与 label 层（全部失效，必须重测）**：
   - `bos_id = [1,1388,570,811,234]` 与 `eos_id = [2,234]`：新 tokenizer 的 chat template 与特殊 token 完全不同。**重新确定的方法**：加载新 tokenizer，对 `apply_chat_template` 渲染出的**实际字符串**做一次编码，打印 id 序列，而不是照搬常量。
   - 特殊 token id（pad / bos / eos / think 标记）与 pad 侧：从 `tokenizer_config.json` 与 `tokenizer.padding_side` 读，不能假设 pad=0。
   - chat template 本身：assistant 段的起止标记、是否有 `<think>` 块、`add_generation_prompt` 的行为都要现场渲染一遍再看。
   - 不变量 `有效 label 数 = Σ(assistant 内容 + len(eos_seq))` 的形式保留，但 `len(eos_seq)` 这个常数必须重算。
2. **模型层（作废的形状常量）**：`768 / 8 层 / 8 heads / 4 kv_heads / head_dim 96 / intermediate 2432 / V=6400 / 63.9M / KV 12 KB per token / 激活 240 KB per token` 全部作废。**重新推导的最小输入集合**（从新 config 读 7 个字段）：

   ```text
   hidden_size H, num_hidden_layers L, num_attention_heads n_h,
   num_key_value_heads n_kv, intermediate_size I, vocab_size V, tie_word_embeddings
   → head_dim d = H / n_h
   → 参数量 = V·H·(2 − tie) + L·[ H·H + 2·H·(n_kv·d) + H·H + 3·H·I + 2H + 2d ] + H
   → KV cache/token = L × 2 × n_kv × d × bytes
   → 激活/token ≈ (10H + 3I) 元素 × bytes × L
   ```
3. **显存层（0.6B 全参 SFT 是否可行）**：
   - 静态项：`参数 + 梯度 + AdamW(m,v) ≈ 16 B/参数`（fp32 master）→ `0.6e9 × 16 B ≈ 9.6 GB`。
   - logits 项：`B×T×V×bytes`，V≈151,936、T=768、bf16 → **0.233 GB/样本**，加 CE fp32 两份共 **1.167 GB/样本**（见 M01-A-02）。
   - 激活项：随 H、L 上升，比 MiniMind 的 240 KB/token 高一个量级以上。
   → 静态 9.6 GB 已占掉 16 GB 的 60%，加上 CUDA context 0.5–1 GB，留给动态项不足 5 GB，**B=1 且 T=768 时 logits+CE 就吃掉 1.17 GB，激活再占几 GB，全参 SFT 在 16 GB 上不可行**。
   - **先动哪个旋钮**：先动**优化器状态**（用 LoRA 或 8-bit optimizer 把 9.6 GB 的静态项压到接近纯参数的 1.2–2.4 GB）。理由：静态项与 batch/序列长度**完全无关**，是唯一无法靠降 batch 缓解的部分；而 logits 与激活都能通过降 B、降 T 线性压缩。先解决不可压缩的那一项，才有腾挪空间。
4. **LoRA 层**：MiniMind 的 `apply_lora` 只挂 `in_features == out_features` 的 Linear，在 MiniMind 上恰好命中 `q_proj`（768→768）和 `o_proj`（768→768）。迁到 GQA 更激进的新基座后，`q_proj` 的输出维通常仍是 `n_h·d`、`k_proj/v_proj` 是 `n_kv·d`，且 `hidden_size` 与 `n_h·d` 可能不再相等 —— **这个条件可能一个模块都命中不到，或者只命中 `o_proj`**，导致"挂了 LoRA 但可训参数接近 0、loss 不动"，正是 §6 第 14 条的症状。**改法**：把判定规则从"形状相等"改成"**按模块名白名单**"（例如 `q_proj / k_proj / v_proj / o_proj / gate_proj / up_proj / down_proj` 中显式选择），并保留一条启动断言：`可训参数量 == r × Σ(in+out)` 且首步前 `B.weight.abs().sum() == 0`。
5. **后训练层**：
   - DPO：ref 是同尺寸的第二份权重，**只算权重不算优化器**，即额外 `0.6e9 × 4 B ≈ 2.4 GB`（fp32）或 1.2 GB（fp16/bf16 推理）。相对 MiniMind 的 256 MB 是 **约 9.4 倍**。
   - GRPO：policy（含梯度与优化器，9.6 GB）+ ref（2.4 GB）+ reward model（1.8B，fp16 约 3.4 GB，与基座无关）≈ **15.4 GB 静态**，已经等于整卡容量，动态项无处安放。
   - **最小可行方案**：policy 用 **LoRA**（静态项降到 ≈ 参数 1.2 GB（bf16）+ 极小的 LoRA 优化器状态）；**ref 直接用"关掉 LoRA 分支的同一份基座权重"**，从而完全省掉第二份 ref 权重（LoRA 的 `B` 零初始化保证关掉分支就等于基座）；reward 改用**规则 reward** 或把 1.8B reward model 放到 CPU/离线批量打分。这样静态项从 15.4 GB 降到 1.5 GB 量级，再按 M01-P-01 的显存模型定 `B·G` 上限。
6. **仪表层**：
   - **可原样复用**：`hooks.py`（`ThroughputMeter` / `peak_memory_mb` / `grad_global_norm` 都与模型结构无关）、`parse_log.py`（只解析日志文本）、`grpo_stats.py` 的 `group_advantage` / `group_stats` / `summarize`（纯数值，与 tokenizer 无关）。
   - **必须改**：
     - `minimind_env.py`：`find_minimind_root` / `load_tokenizer` / `build_model_config` / `build_model` 全部绑定 MiniMind 仓库结构，需要换成新基座的加载路径；`DATA_FILES` 的字节数校验表要换。
     - `inspect_dataset.py`：`special_sequences` 必须从新 tokenizer 现场取 `bos_seq/eos_seq`（函数签名已支持传入，改的是取值来源）；`sft_render` 里硬编码的模板与 `_BLOCK_RE` 正则要按新 chat template 重写。
     - `fixtures.py`：样例仍是 MiniMind 的 JSONL 字段名（`conversations` / `chosen` / `rejected`），若新链路沿用同样字段则可留，否则要改。
     - `mask_fault.py`：`user_in_loss` 依赖 `pad_token_id=0` 的默认值，必须改成从 tokenizer 读。
7. **验收标准（三条重新跑通的不变量断言）**：
   - `有效 label 数 == Σ(assistant 内容 token 数 + len(eos_seq))`，且 label 中至少出现一次新的 eos id（对应 §3.1 不变量 2 与 §6 第 4 条）。
   - 随机初始化下首步 `loss ≈ ln V_new`；从预训练权重起步时首步 loss 显著低于该值，且 `load_state_dict` 的 `missing_keys/unexpected_keys` 为空（对应 §6 第 1 条）。
   - LoRA 挂载后 `可训参数量 == r × Σ(in+out)` 且首步前 `B.weight.abs().sum() == 0`；一步后 `> 0`（对应 §3.1 不变量 6 与 §6 第 14 条）。

**0–4 分 rubric**

- 0：直接背 Qwen 的配置数字，或只说"改改配置就行"。
- 1：说出 tokenizer 常量与形状常量会失效。
- 2：给出重新推导所需的最小 config 字段集合 + 显存可行性判断。
- 3：1–5 问全对，含 LoRA 挂载条件的问题与最小可行 GRPO 方案。
- 4：以上全对，且第 6 问按脚本逐个给出"复用/必改 + 改动点"，第 7 问的三条断言都对应到本周已有的不变量编号。

**常见误区**

- 把 `bos_id` 之类的常量当成"改个数字"的小事。它决定 label 起点，写错就是静默的 mask 错位，loss 曲线照常下降。
- 认为 LoRA 一定能挂上。`in_features == out_features` 这个隐式假设在新基座上极可能一个都不命中，症状是"loss 不动"而不是报错。

**追问**

- 你说 ref 可以用"关掉 LoRA 分支的同一份权重"。这个做法在数学上与 DPO 论文里的 ref 完全等价吗？在什么条件下不等价？
- 若新基座的 `tie_word_embeddings=False`，第 2 问的参数量公式和第 3 问的静态显存分别怎么变？

**定位**：`01_FOUNDATIONS.md` §1.1、§1.3、§2.4、§2.5、§5.1、§5.2、§7、§6 第 1/4/14 条；`lab/src/mm_probe/minimind_env.py`、`inspect_dataset.py`、`mask_fault.py`、`fixtures.py`、`hooks.py`、`parse_log.py`、`grpo_stats.py`。

---

### M01-P-04

**参考答案要点**

1. **`mask_fault.py` 已实现的两种模式**（见 `FAULT_MODES`）：
   - `assistant_all_ignored`：把 `labels` **整条置为 -100**，没有任何 target。期望症状：loss = 0（或某些 PyTorch 版本下 nan）、`grad_norm` = 0/nan、权重不变。
   - `user_in_loss`：把**所有非 pad token**（含 user/system/模板）都写进 `labels`。期望症状：loss 起点更低、下降更快、曲线更平滑；固定 prompt 生成会复述 user/模板文本。
   （还有一个 `none` 作为直通基线。）
2. **第三种故障"丢 eos"的注入方式**：在 `generate_labels` 的输出上，对每一段被标为 label 的区间，把**区间末尾 `len(eos_seq)` 个位置**（即 `<|im_end|>` 与其后的 `\n`）重新置回 -100，其余保持不变。要保证不破坏的性质：
   - assistant **内容** token 的 label 必须原样保留（否则就退化成 `assistant_all_ignored` 的一个变体，两个故障无法区分）；
   - user/system 段与 pad 仍全为 -100（否则会混入 `user_in_loss` 的特征）；
   - 有效 label 数应恰好等于正常值减去 `assistant 段数 × len(eos_seq)`，这一条本身就是注入正确性的断言；
   - 不改变 `input_ids`，只改 `labels`，这样两条曲线的数据分布严格一致。
3. **判别特征**：

   | 故障 | 只看 loss 曲线 | 只看固定 prompt 生成 | 充分性 |
   | --- | --- | --- | --- |
   | `assistant_all_ignored` | loss 恒为 0（或 nan），`grad_norm` 恒 0 | 输出与起步 checkpoint **逐字相同** | **这一对是充分的**：权重完全不变，任何其他故障都做不到 |
   | `user_in_loss` | 初值明显更低、曲线明显更平滑 | 复述 user 文本，主动生成 `<\|im_start\|>user` 角色标记 | 曲线特征只是**必要**（数据更容易也能造成）；生成里的**角色越界标记**是充分的，因为该标记在正常训练中从不作为 target |
   | 丢 eos | 与正常曲线几乎**无法区分**（只少了每段 2 个位置） | 生成**永不停止**，一直写到 `max_new_tokens`，或不断续接新一轮 | 曲线特征完全**不充分**；生成的"永不停止"是充分的 |

4. **对照实验矩阵**：

   ```text
   运行数 : 4 条（none / assistant_all_ignored / user_in_loss / drop_eos）
   共享   : 同一 seed、同一数据切片（前 N 条固定样本）、同一 lr 调度、同一步数（建议 300–500 步有界）
   唯一变量: labels 的注入模式
   采集   : 每步 loss / grad_norm / 有效 label 数；训练结束后用同一组固定 prompt 各生成一次（同 seed、同 max_new_tokens）
   ```
   **必须有 `none` 基线的理由**：三个故障里有两个（`user_in_loss` 的"更低更平滑"、`drop_eos` 的"几乎无变化"）都是**相对判断**，没有基线就没有"更低"和"无变化"的参照；而且基线还负责证明这套有界训练本身是健康的（loss 在下降、生成正常停止），否则任何异常都可能来自环境而非注入。
5. **可能被混淆的一对**：`assistant_all_ignored` 与"数据集为空 / 所有样本被 tokenize 成全 pad"。两者都给出 loss=0、grad_norm=0、权重不变。**额外观测量**：`inspect_dataset.py` 输出的**非 pad token 数**。故障注入时非 pad token 数正常（只是 labels 被清空），数据为空时非 pad token 数也接近 0。这一个量就能拆开。
   另一对：`drop_eos` 与"训练步数不足导致模型还没学会停"。**额外观测量**：`label == eos_id[0]`（id 2）的样本比例 —— 正常应为 100%，`drop_eos` 下为 0%；这个量与训练步数完全无关。
6. **自动化 vs 人看**：
   - **自动化判据**（可写成断言，进 CI）：`inspect_dataset.py` 给出的每条样本有效 label 数、`label==2` 的样本比例、非 pad token 数；`parse_log.py` 的 `load_rows` + `numeric_fields` 提取 loss/grad_norm 序列后计算的窗口均值、标准差（平滑度）、首步值。这几个量都有明确阈值，不需要人判断。
   - **必须人看**：固定 prompt 的**生成文本**。"复述了 user 的问题"、"自己扮演 user"、"永不停止"这三类都是语义/结构判断，虽然"是否出现 `<|im_start|>user` 字面串"和"是否在 `max_new_tokens` 处截断"可以自动化成两个布尔量，但"输出是否答非所问"仍需人读。设计上应把可自动化的部分（角色标记出现、是否达到长度上限、输出是否与基线逐字相同）先抽成布尔量，把人看的范围压到最小。

**0–4 分 rubric**

- 0：说不出 `mask_fault.py` 有哪些模式。
- 1：两种已实现模式正确。
- 2：给出第三种故障的注入方式与不可破坏的性质。
- 3：1–4 问全对，含判别特征表与"必须有基线"的理由。
- 4：以上全对，且第 5 问给出了两对易混淆场景与各自的额外观测量，第 6 问明确划出了自动化边界并说明怎么把人看的范围压小。

**常见误区**

- 只依赖 loss 曲线做判别。三个故障里只有 `assistant_all_ignored` 在曲线上是充分可辨的，另外两个必须配生成结果。
- 注入 `drop_eos` 时顺手把 assistant 内容也清掉，导致它与 `assistant_all_ignored` 无法区分。

**追问**

- 若三个故障同时出现，你的矩阵还能拆开吗？需要补几条运行？
- `drop_eos` 训练出的模型，其 `<|im_end|>` 的输出概率会是多少？设计一个不需要生成、只做一次前向就能测出来的判据。

**定位**：`01_FOUNDATIONS.md` §1.2、§6 第 2/3/4 条、§8 第 8 条；`lab/src/mm_probe/mask_fault.py`（`FAULT_MODES` 与各模式的期望症状注释）、`inspect_dataset.py`、`parse_log.py`；`00_WEEK_CARD.md`「关键故障练习」。

---

### M01-P-05

**参考答案要点**

1. **可观测症状**：
   - **β 过大**：`k3 KL` 被压在接近 0 的水平且几乎不随步数增长；`Adv` 的策略梯度项被 KL 项淹没（把 loss 拆成两路记录后，KL 分量占比 > 80%）；生成分布与 ref 几乎无差别（固定 prompt 的输出与 SFT 模型逐字重合率很高）；`Reward` 长期不动即使 `adv_std > 0`。
   - **β 过小**：`k3 KL` 单调增大且没有回落；`gen_len` 或输出形态快速漂向 reward 的形状项（长度锁死在某个区间、`</think>` 固定出现一次）；能力侧固定 prompt 的输出质量下降（出现重复、答非所问）；`Reward` 上升但只有形状分量在涨。
2. **陷阱**：日志里的 `KL_ref = mean(logπ_ref − logπ_θ)` 是**有符号**的 log-ratio 均值，可以为负；损失里用的是 k3 估计 `exp(Δ) − Δ − 1`（Δ = logπ_ref − logπ_θ），**恒 ≥ 0**。如果拿日志里的 `KL_ref` 来判断"锚定强度"，正负项会互相抵消，一个策略在某些 token 上正偏、某些 token 上负偏时 `KL_ref` 可能接近 0，看起来"锚得很好"，实际 k3 KL 已经很大。**要让 `grpo_stats.py` 额外记录**：逐 token 的 k3 KL 的均值与 P90、`β·KL` 在总 loss 中的占比、以及策略梯度项与 KL 项的绝对值之比。
3. **扫描协议**：
   - β 取点：`0.0`（关掉 KL，作为下界锚点）、`0.02`、`0.05`、`0.1`（源码量级）、`0.3`，共 5 点，等比跨度覆盖两个数量级。
   - 每点步数：与有界 smoke 一致（例如 200–300 步），足以让 k3 KL 离开 0 但不至于把整卡时间吃光。
   - **固定住**：seed、prompt 集合与顺序、`num_generations`、`max_seq_len`/`max_gen_len`、lr、dtype、reward 函数与其权重。β 是唯一自变量。
   - 三类指标：
     - **策略侧**：`adv_std`、`std==0` 组占比、策略梯度项的绝对值均值。
     - **锚定侧**：k3 KL 的均值与 P90、`β·KL` 占总 loss 的比例。
     - **能力侧**：固定 prompt 集合上的生成质量（见第 4 问）。
4. **能力侧怎么测**：用 `lab/scripts/eval_generate.py` 的思路，准备一组**训练外**的固定 prompt（建议 20–40 条），在每个 β 点训练前后各跑一次，同 seed、同 `max_new_tokens`、同解码参数，做逐条对照。选取原则：覆盖 SFT 阶段已具备的通用能力（多轮对话、指令跟随、正常停止），并包含明确可判定的项（是否在 `<|im_end|>` 处停止、是否出现角色越界、是否重复）。
   **不能用训练用的 RLAIF prompt** 的理由：它们正是 reward 被优化的对象，模型在它们上的表现会随 reward 上升而上升，无法区分"真的变好"与"学会了拿分"；而 β 要防的恰恰是"在训练 prompt 上拿分、在训练外掉能力"。用训练 prompt 测能力等于把自变量和因变量混在一起。
5. **停止条件与失败边界**：
   - 停止条件：当 β 增大到某点后 k3 KL 已被压到与 β=0.3 点无显著差别、且 `Reward` 与 β=0 点相比没有任何提升时，说明已进入"过大"区间，无需再往上扫；当 β 减小到某点后能力侧指标开始退化时，说明已进入"过小"区间，无需再往下扫。选定值取"能力侧不退化的最小 β"。
   - **失败边界**：若 `adv_std ≡ 0`（即 M01-D-03 的故障没有先解决），策略梯度项恒为 0，损失里**只有** β·KL 一项，此时改变 β 只是在改变"往 ref 回收的速度"，扫描结果完全不含 reward 信息，**整个扫描无效**。因此扫描的前置条件是：`std==0` 组占比必须低于某个阈值（例如 < 30%），否则先修 advantage 再扫 β。另一个失败边界：若 reward 已被形状项饱和（见 M01-T-02），能力侧退化会被误归因给 β 而不是 reward 设计。
6. **可验证的下一步**：选定 β 后，用一个**比值不变量**做长期监控：`β·KL 项的绝对值 / 策略梯度项的绝对值` 应稳定落在选定时观测到的区间内（例如 0.1–0.5）。这个比值是无量纲的、与 reward 尺度无关，比直接盯 k3 KL 的绝对值更稳健。一旦该比值持续跌破下界，说明 KL 约束相对策略梯度失效，需要重新扫；持续超过上界则说明策略被锚死。同时保留能力侧固定 prompt 的定期回归，作为独立于训练指标的外部证据。

**0–4 分 rubric**

- 0：用"效果不好"这类不可测量的描述。
- 1：给出 β 过大/过小各一个可测量信号。
- 2：识别出 `KL_ref` 与 k3 的区别并说出它带来的陷阱。
- 3：1–4 问全对，含扫描点、固定变量、三类指标与训练外 prompt 的理由。
- 4：以上全对，且第 5 问指出"advantage 恒为 0 时扫描无效"这一前置条件，第 6 问给出一个无量纲的比值不变量作为长期监控。

**常见误区**

- 直接盯 `KL_ref` 的绝对值定 β，被正负抵消误导。
- 用训练 prompt 评估能力，把 reward 上升当成能力提升。

**追问**

- 若 β=0 时 `Reward` 反而涨得最快，你会选 β=0 吗？给出你的判据和它的风险。
- k3 KL 的 P90 与均值差距很大时说明什么？这对 β 的选择有什么额外含义。

**定位**：`01_FOUNDATIONS.md` §2.6、§6 第 8/9/10 条、§7；`lab/src/mm_probe/grpo_stats.py`、`lab/scripts/eval_generate.py`（周卡 Day 5 产物）。

---

## Trade-off

### M01-T-01

**参考答案要点**

1. **显存侧**（G 影响的项，`B·G` 是序列数）：

   | 项 | 与 G 的关系 |
   | --- | --- |
   | 生成期 KV cache（12 KB/token） | **线性** |
   | 训练前向的 logits（`B·G·T·V×bytes`）与 CE fp32 两份 | **线性** |
   | 每层激活（≈240 KB/token） | **线性** |
   | ref 前向的临时 log_softmax | **线性** |
   | 手写 attention 分支的 `scores`（`B·G·n_h·T²`） | **线性于 G、平方于 T** |
   | policy/ref/reward 的**权重、梯度、优化器状态** | **无关** |

   `batch_size=1, max_seq_len=256, max_gen_len=256` 下，每条序列 512 token：
   - G=2 → 1,024 token → 激活 ≈ 0.25 GB、logits fp32 ≈ 26 MB、KV ≈ 12 MB → 动态项约 **0.3 GB**。
   - G=6 → 3,072 token → 各项 ×3 → 动态项约 **0.9 GB**。
   静态项（policy 1.02 + ref 0.26 + reward 3.4 ≈ 4.7 GB）不随 G 变。所以 G 从 2 到 6 的总显存差约 0.6 GB —— **在这个短序列配置下 G 不是显存瓶颈**，静态项才是。这个结论在 `max_gen_len=1024` 下会反转。
2. **时间侧**：G 增大主要拖慢**生成（rollout）**与 **reward 打分**两段。§5.3：生成是自回归的，逐 token 走 8 层前向，batch 小时 GPU 利用率极低；reward model 是 1.8B、且 `get_score` 是 Python 级**逐条串行**调用。序列数从 2 变 6 意味着 reward 打分的串行次数直接 ×3，而 backward 仍然只有一次。所以随 G 上升，"生成 + 打分"在 step 时间里的占比只会更高（本来就 > 80%），backward 的占比进一步被压缩。
3. **信号侧**：由 M01-A-04 第 4 问，G=2 时只要两条 reward 不等，`A ≈ ±1`，**与 reward 差距的绝对大小无关** —— 组内归一化把量纲信息完全抹掉，只剩一个符号位，且每步更新幅度恒定。G 增大后：advantage 恢复了**连续取值**（如 G=6 的例子里出现 +1.30、+0.19、−2.04 三个量级），能区分"明显更好"与"稍微更好"；同时组均值与组标准差的估计方差下降，advantage 的方差估计更可靠。
4. **整组作废的概率**：reward 由离散的形状分档位主导（长度 ±0.5、思考块 +1.0/−0.5、`</think>` 计数 +0.25、3-gram 惩罚），取值集合很小。设两条 response 落到同一档的概率为 p，则 G=2 时整组同分概率就是 p；G=6 时要求**全部 6 条**落在同一档，概率约为 p⁵ 量级（远小于 p）。只要 p 明显小于 1，G 越大整组作废越罕见。这是 G=2 时 `std==0` 占比高的直接原因。
5. **决策**：在 16 GB、每天 2 小时的约束下选 **G=4 作为起点**，并按证据向 6 移动。判据：**`std==0` 组的占比**。
   - 若 G=4 下 `std==0` 占比 < 20%，且单步耗时使 300 步能在 40 分钟内跑完，保持 G=4。
   - 若占比 ≥ 30%，提到 G=6；若 G=6 后占比仍 ≥ 30%，说明是 reward 分辨率问题而不是 G 的问题（转 M01-T-02）。
   - **改主意的证据**：(a) `peak_memory_mb` 显示动态项已超过可用余量（此时降 G 或降 `max_gen_len`）；(b) 单步耗时使一天跑不完一次有界实验（此时 G 让位于步数）；(c) `std==0` 占比在 G=2 时就已经很低（说明 reward 分辨率足够，没必要付 G 的成本）。
6. **显存不允许提高 G 时的两条替代路**：
   - **提高 reward 的分辨率**：把离散形状分改成连续函数（例如长度项用平滑函数而非阶跃），或加载 reward model 提供连续分量。代价：reward model 占 3.4 GB 静态显存并让打分成为串行瓶颈；平滑化会改变 reward 的语义，之前的曲线不可比。
   - **提高采样多样性**：调高温度 / 放宽 top-p，让同一 prompt 的 G 条 response 更分散。代价：多样性上升会引入更多低质量样本，advantage 的方差上升、单步噪声更大；且温度是另一个需要扫的超参，扫描成本转移而非消失。
   （第三条可选：在多个连续步上累积同一 prompt 的样本再算组统计，等价于用时间换 G，代价是这些样本来自不同的 policy 版本，引入真正的 off-policy 偏差。）

**0–4 分 rubric**

- 0：只说"G 大更准但更费显存"。
- 1：正确列出 G 影响的显存项及线性关系。
- 2：再答对时间侧的归因（生成 + 打分占主导）。
- 3：1–4 问全对，含 G=2 时 `A ≈ ±1` 的结构性缺陷与整组作废概率的量级论证。
- 4：以上全对，且第 5 问给出可测量阈值与三条"改主意"的证据，第 6 问的替代方案各自代价清晰。

**常见误区**

- 认为 G 增大会线性增加 backward 成本。backward 每步仍只有一次，增加的是生成与打分。
- 忽略静态项：在短序列配置下 G=2 与 G=6 的总显存差只有 0.6 GB，而 reward model 一项就是 3.4 GB，先砍 G 是砍错了地方。

**追问**

- 把 `max_gen_len` 从 256 提到 1024，第 1 问的结论（"G 不是显存瓶颈"）还成立吗？重新算一遍。
- 若把 reward model 挪到 CPU，第 2 问的时间构成怎么变？这会不会让 G=6 变得更划算？

**定位**：`01_FOUNDATIONS.md` §2.6、§5.2、§5.3、§6 第 8 条；`lab/src/mm_probe/grpo_stats.py`、`hooks.py`（`ThroughputMeter` / `peak_memory_mb`）；`00_WEEK_CARD.md`「三类环境边界」。

---

### M01-T-02

**参考答案要点**

1. **形状类奖励的取值区间与上界**（§2.6）：
   - 长度项：20–800 字 → **+0.5**，否则 **−0.5**。
   - 思考块项：思考块 20–300 字 → **+1.0**，否则 **−0.5**；恰好一个 `</think>` 额外 **+0.25**。
   - 3-gram 重复惩罚：**0 到 −0.5**（上限 0.5 的扣分）。
   - **上界** = 0.5 + 1.0 + 0.25 + 0（不触发重复惩罚）= **1.75**。一条"写满 20 字以上、思考块 20 字以上、放恰好一个 `</think>`、不重复"的**内容完全无意义**的 response 就能拿到 1.75。（这正是 M01-D-03 日志里 `reward:1.7500` 的来源。）
2. **为什么构成直接激励**：这 1.75 分**不依赖内容质量**，只依赖可被字符统计验证的表面结构。策略梯度会优先找最容易的上升方向；形状分的梯度信号密集、确定、无噪声，而 reward model 分数稀疏且带噪。所以模型会先把形状分吃满，然后停在那里 —— 因为再往上需要真正改善内容，收益率低得多。
   **两个预期的退化形态**：
   - 生成长度**锁死**在阈值边缘附近（例如稳定在 20–40 字），既满足 +0.5 又不多花 token；M01-D-03 里 `gen_len` 稳定在 34–35 就是这个形态。
   - 思考块变成**模板化的占位内容**（凑够 20 字的套话），并且恰好放一个 `</think>` 收 +0.25，思考块与最终答案之间没有逻辑关联。
3. **形状类奖励的正面作用**：在训练早期，如果只有 reward model 分数，SFT 起步的策略在同一 prompt 上采出的 G 条 response 质量往往相近，reward model 给的分数差异很小 → **组内 reward 方差接近 0 → advantage 接近 0 → 没有学习信号**（M01-D-03 的机制）。形状项提供了粗粒度但**可靠**的方差来源：格式对不对、有没有停、长不长，这些在早期就能把明显不合格的 response 区分出来，让 advantage 离开 0。所以形状项解决的是"冷启动阶段的信号密度"问题。
4. **取舍**：形状项应当**在早期占主导、随训练推进衰减**，而不是全程固定权重。具体做法：把形状项总权重设为随步数衰减（例如从 1.0 衰减到 0.2），reward model 分数权重相应上升。
   **依据的可测量量**：
   - `std==0` 组的占比 —— 只要它保持低位（< 20%），说明信号密度够，就可以继续削减形状项权重。
   - **形状分量的饱和度** —— 当 ≥ 90% 的 response 已经拿满形状分上界时，形状项对当前策略已无区分度（它的组内方差为 0），继续给它权重只是在给所有样本加同一个常数，对 advantage 完全没有贡献（因为组内归一化会减掉均值）。**这是一个硬判据**：饱和的分量对 advantage 的贡献严格为零，此时它的权重应当降到 0。
5. **监控方案**：`grpo_stats.py` 应把总 reward 拆成 **4 路**分别记录（长度项、思考块项、重复惩罚、reward model 分数），并对每一路单独计算组内均值与组内标准差。
   **"总 reward 上升但不是进步"的判据**：总 reward 的上升量中，来自形状项的分量占比 > 70%，同时 **reward model 分量的组内均值持平或下降**。因为真正的能力提升应当体现在唯一的内容相关分量上；若形状分在涨而内容分不涨，上升的只是"拿分技巧"。补充判据：`gen_len` 分布向奖励阈值边缘收敛（方差下降、均值贴近阈值）。
6. **失败边界**：**不能证明** reward model 没被 hack。理由：
   - reward model 本身是一个 1.8B 的学习到的打分器，它对分布外输入的行为没有保证；策略完全可能找到它的高分盲区。
   - 在本周条件下（单卡、有界步数、无独立的第二个评测模型、无人工标注），唯一能拿来交叉验证的是 `eval_generate.py` 在**固定训练外 prompt** 上的生成，而这仍然要靠人读，样本量小。
   - **标注方式**：把这个结论明确标为 `未知` / `INCONCLUSIVE`，并写清缺什么证据才能升级 —— 需要一个与训练用 reward model **独立**的评测器（不同基座或人工标注），且样本量足以做统计判断。周卡已经为 Day 5 预注册了 `INCONCLUSIVE` 出口，这里正是它的用途。绝不能因为 reward 曲线上升就在报告里写"模型变好了"。

**0–4 分 rubric**

- 0：算不出形状分上界。
- 1：上界 1.75 正确。
- 2：再说清它为什么构成激励，并给出两个退化形态。
- 3：1–4 问全对，含形状项在冷启动期的正面作用与"饱和分量对 advantage 贡献为零"的硬判据。
- 4：以上全对，且第 5 问给出可执行的四路拆分与量化判据，第 6 问明确承认无法证伪并给出 `未知` 的标注方式与升级条件。

**常见误区**

- 认为把形状项权重直接设为 0 就解决了 hacking。那会让冷启动阶段的 advantage 塌回 0（M01-D-03），训练根本起不来。
- 只看总 reward 曲线做判断。组内归一化会减掉均值，任何已饱和（组内无方差）的分量对 advantage 贡献严格为零，总 reward 的绝对水平本身不携带学习信号。

**追问**

- "饱和分量对 advantage 贡献为零"这个论断，在数学上怎么从 `A = (r − mean)/(std + 1e-4)` 推出来？如果只是接近饱和（不是完全饱和）呢？
- 若把 3-gram 重复惩罚的上限从 0.5 提到 2.0，第 1 问的上界怎么变？这会不会引入新的 hack 路径？

**定位**：`01_FOUNDATIONS.md` §2.6、§6 第 8/9/10 条、§7；`lab/src/mm_probe/grpo_stats.py`（`group_stats` / `summarize`）；`00_WEEK_CARD.md`「已知限制与风险」的 `INCONCLUSIVE` 出口。

---

### M01-T-03

**参考答案要点**

1. **显存账**：

   | 项 | 全参 SFT | LoRA |
   | --- | --- | --- |
   | 参数（fp32 master） | 63.9M × 4 B = **256 MB** | 同样 **256 MB**（基座权重仍要在显存里） |
   | 梯度 | 63.9M × 4 B = **256 MB** | 0.393M × 4 B ≈ **1.6 MB** |
   | AdamW m,v | 63.9M × 8 B = **511 MB** | 0.393M × 8 B ≈ **3.1 MB** |
   | 小计（静态） | ≈ **1.02 GB** | ≈ **0.26 GB** |
   | **激活** | ≈ 240 KB/token × B·T | **完全相同** |

   **LoRA 不省的是激活**。理由：`B` 的梯度需要该 Linear 的输入、`A` 的梯度需要 `B` 上游传回的梯度，反向必须**穿过整个网络**才能到达每一层的 LoRA 模块，因此所有层的中间激活都要保留。LoRA 省的只是"梯度 + 优化器状态"这两项，即 §2.4 说的 AdamW 状态只有约 3 MB。
2. **参数账**：

   ```text
   每个 LoRA 模块 = r·(in + out) = 16·(768 + 768) = 24,576
   挂载数 = 8 层 × 2（q_proj, o_proj）= 16 个
   总计 = 393,216 ≈ 0.393 M，占 63.91M 的 0.62 %
   ```
   **挂上的**：`q_proj`（768→768）、`o_proj`（768→768）。**没挂上的**：`k_proj`/`v_proj`（768→384，`in ≠ out`）、`gate_proj`/`up_proj`（768→2432）、`down_proj`（2432→768）、以及 embedding 与 lm_head（tied）。
   "没挂上"意味着：**key/value 的投影空间不会被调整**（模型看什么、检索什么的能力被冻结）、**FFN 完全不动**（FFN 是模型存事实性知识的主要位置，占每层 5.60M / 7.37M ≈ 76% 的参数）、**输出词表分布的直接调整能力被冻结**（lm_head 与 embedding tied 且不挂 LoRA）。所以 LoRA 在这个配置下调整的是"注意力怎么加权和怎么汇聚"，不是"知道什么"和"输出什么词"。
3. **`B` 零初始化 → 初始 ΔW = 0**：
   - **训练上的好处**：训练起点与基座**完全一致**，不引入任何初始扰动，因此不需要 warmup 来"修复"随机初始化带来的破坏，可以直接用较大的 lr（1e-4，比全参 SFT 的 1e-5 大 10 倍）。
   - **调试上的好处**：给了一个**精确的、可断言的不变量** —— 首步前 `B.weight.abs().sum() == 0`，一步后 `> 0`（§3.1 不变量 6、§6 第 14 条）。这把"LoRA 到底有没有在训"从一个模糊判断变成一行断言。同时它意味着"关掉 LoRA 分支 = 基座"，可以零成本地拿到一个完全等价的对照模型（这也是 M01-P-03 里用它当 DPO ref 的依据）。
4. **本周的选择：全参 SFT**。理由：
   - **理由一（可比较性，必答）**：本周的核心练习是 mask 故障注入的**对照实验**（正常 vs `user_in_loss` vs `assistant_all_ignored`）。这些对照要求两条曲线除了 labels 之外**完全同构**。LoRA 只让 0.62% 的参数可训、且完全不动 FFN，会**压缩故障的可观测幅度** —— 例如 `user_in_loss` 造成的"loss 初值更低"在 LoRA 下会被冻结的大部分参数稀释，曲线差异变小，判别变难。全参 SFT 让故障以最大幅度表现出来，这正是故障注入练习需要的。
   - **理由二（显存不是约束）**：§5.1 估算 SFT `B=16, T=768` 全参约 4.7 GB，在 16 GB 上宽裕。LoRA 只省 0.76 GB 的静态项，而激活（2.9 GB）一分不省 —— 省下的这点显存换不来任何配置上的自由度（不能因此把 batch 翻倍）。**在瓶颈不是显存时，用 LoRA 是拿可观测性换一个不需要的收益。**
   - 理由三：全参 SFT 的产物 `full_sft_768.pth` 是 Day 4 DPO 的直接输入（policy 与 ref 都从它加载），走全参路径不需要额外的 merge 步骤。
5. **LoRA 会给出误导性结论的场景**：Day 3 的 mask 故障注入中，若用 LoRA 跑 `assistant_all_ignored`，症状（loss=0、grad_norm=0、权重不变）与"LoRA 参数没进优化器"（§6 第 14 条：LoRA loss 不动）**完全重合**。两个完全不同的故障产生同一组观测，诊断必须额外增加一步（断言可训参数量 == 393,216）才能拆开。全参 SFT 下不存在这个歧义。
6. **迁移视角：会，而且会反转结论。** 在 0.6B 及以上基座上：
   - 静态项从 1.02 GB 变成 `0.6e9 × 16 B ≈ 9.6 GB`（M01-P-03），已占 16 GB 的 60%，成为**不可压缩的主要矛盾**（它与 batch/序列长度完全无关，降 batch 也降不下来）。LoRA 把它压到接近纯参数的 1.2–2.4 GB，是唯一能在单卡上打开局面的手段。
   - 与此同时，"LoRA 不省激活"这一条**仍然成立**，只是它变成了**可压缩的次要矛盾**：激活正比于 `B·T`，可以靠降 batch、降 `max_seq_len`、或梯度检查点线性压下来。
   所以结论反转的机制是：**16 GB 上 MiniMind 的瓶颈是激活（可压缩），0.6B 的瓶颈是优化器状态（不可压缩）**；LoRA 恰好只解决后者，因此在 MiniMind 上是可有可无的，在 0.6B 上是必需的。

**0–4 分 rubric**

- 0：说不出 LoRA 不省激活。
- 1：显存账四项正确，指出激活不省。
- 2：再答对参数账与"没挂上的层意味着什么"。
- 3：1–4 问全对，且第 4 问的理由中包含"可比较性"这一条。
- 4：以上全对，且第 5 问给出具体的观测歧义场景，第 6 问用"可压缩/不可压缩矛盾"解释了结论反转的机制。

**常见误区**

- 认为 LoRA 能大幅降低训练显存峰值。它只降静态项；峰值通常由激活主导，LoRA 对峰值的改善在本周配置下不足 20%。
- 忽略 `in_features == out_features` 这个隐式挂载条件，误以为 LoRA 覆盖了 FFN。FFN 占每层 76% 的参数，一个都没挂上。

**追问**

- 若把 `apply_lora` 改成也挂 `k_proj`/`v_proj`（用两个不同形状的 A、B），可训参数量变成多少？激活显存变吗？
- 在梯度检查点开启的情况下，第 1 问的"LoRA 不省激活"这个结论还成立吗？两者是叠加还是互相替代？

**定位**：`01_FOUNDATIONS.md` §1.3、§2.4、§5.1、§3.1 不变量 6、§6 第 14 条；`lab/src/mm_probe/mask_fault.py`、`hooks.py`；`00_WEEK_CARD.md` Day 3。

---

### M01-T-04

**参考答案要点**

1. **格式差别**（§4.1）：

   | 格式 | 指数位 | 尾数位 | 最大值 | 最小正规数 | 相对精度 |
   | --- | --- | --- | --- | --- | --- |
   | fp32 | 8 | 23 | 3.4e38 | 1.2e-38 | 6e-8 |
   | fp16 | 5 | 10 | 65,504 | 6.1e-5 | 9.8e-4 |
   | bf16 | 8 | 7 | 3.4e38 | 1.2e-38 | 7.8e-3 |

   bf16 的**指数位与 fp32 完全相同**，动态范围一致，梯度（典型量级 1e-8 ~ 1e-3）不会因为格式而下溢或上溢。loss scaling 的**唯一目的**就是把梯度从 fp16 的下溢区（< 6.1e-5）搬上来；bf16 不存在这个问题，所以 `GradScaler(enabled=False)` 是安全的，此时 `scale()` 返回原 loss、`step()` 直接调 `optimizer.step()`、`get_scale()` 恒为 1.0。
2. **bf16 尾数只有 7 位（相对精度 7.8e-3）**，因此**必须在 fp32 里算的两类量**：
   - **CE / log_softmax**（pretrain、SFT 的 loss 计算）。
   - **DPO 与 GRPO 里的逐 token log-prob 及其差**（`logits_to_log_probs` 的输出、`ratio = exp(logπ_θ − logπ_old)`、k3 KL 的 `Δ = logπ_ref − logπ_θ`）。这些量本身是两个相近数的差，在 bf16 里分辨率只有约 1e-2，而我们要观察的 `max|ratio−1|` 期望小于 1e-2 —— 直接在 bf16 里算会让信号完全淹没在量化噪声里。
   **保证机制**：PyTorch 的 autocast op 列表把 `cross_entropy`、`log_softmax`、`softmax` 归入"CUDA autocast 下以 fp32 执行"，所以即使在 bf16 autocast 上下文里，这些算子也会自动升到 fp32。这是框架保证，不是代码里显式转换的。
3. **收益侧（能提前拿到的第 3 周证据）**：
   - **`scaler.get_scale()` 的完整时间序列**：健康形态是什么样、遇到偶发溢出时怎么减半再爬回来。这是 V100 周的核心观察对象，在 5070 Ti 上跑一次就能拿到基线形态并归档。
   - **跳步率**：`scaler.step()` 被跳过的步数占比，以及跳步时 lr 调度照走带来的"lr 前进但权重不动"的累计效应。这个数字直接决定 V100 上要不要调 `init_scale` 或 lr。
   - （可选第三条）bf16 与 fp16 在同 seed、同数据下的 loss 曲线差，作为"精度对收敛的影响"的量化基线。
4. **风险侧（GRPO 无 GradScaler，fp16 下两种失败，§4.5）**：
   - **下溢（静默）**：lr=3e-7 量级的策略梯度经 fp16 反向，大量梯度落到 6.1e-5 以下变成次正规数或 0。表现为 reward 曲线不动、`KL_ref` 恒 0，**看起来像"没学到"**，实际是梯度被格式吃掉。
   - **溢出（显性）**：反向中间量 > 65,504 → inf → `clip_grad_norm_` 算出 nan 范数 → 所有梯度乘 nan → `optimizer.step()` 把 nan 写进权重 → 下一次 rollout 全部生成垃圾，**不可恢复，只能回滚 checkpoint**。
   **静默的那种更危险**，理由：溢出会立刻产生 nan，是一个不可能被忽略的显性信号，且发生一次就停；下溢不产生任何异常值，日志上的一切都"正常"，你会把它误诊为"reward 设计不好"或"lr 太小"，从而把大量时间花在调 reward 和 lr 上 —— 即**它会把你引向错误的假设空间**。区分方法：`grpo_stats.py` 打印 `grad_norm` 与**零梯度参数比例**，下溢时后者会异常高。
5. **决策**：
   - **做**：pretrain 与 full SFT 用 `--dtype float16` 各跑一次有界训练（例如 300–500 步）。这两个脚本有 `GradScaler`，跳步保护存在，最坏情况是训练变慢或不收敛，不会写坏权重；而它们恰好是 `get_scale()` 曲线的最佳采集场景。
   - **不做**：GRPO 的 fp16 实验。它没有 scaler，溢出会直接把 nan 写进权重，且 GRPO 本身在 16 GB 上就处于显存边缘（§5.2），一次坏运行的代价（回滚 + 重跑 rollout）远高于收益。DPO 也可以不做（lr=4e-8 本来就在 fp16 的危险区，结果会被下溢污染，得不到干净的信号）。
   - **替代证据**：GRPO 的 fp16 风险用**推理 + 静态分析**覆盖 —— 已经从源码确认 `train_grpo.py` 没有 scaler（§4.5），并在 bf16 下用 `grpo_stats.py` 记录 `grad_norm` 与零梯度比例的**基线分布**；到 V100 周时把 fp16 下的同一组数字与这个基线对比，就能识别下溢，而不需要在本周冒险。这一条要在报告里明确标注为 `推断`，不是本周实测。
6. **`scaler.get_scale()` 的健康形态与判据**（§4.2、§6 第 5 条）：
   - **健康**：从 65536 起步，**偶尔**减半（对应偶发的前向/反向溢出），随后连续 2000 步正常就翻倍回去；长期在某个量级附近震荡，不单调下行。
   - **判定"不是精度问题"的形态**：scale **持续单调减半直到 < 1**，说明**每一步**都溢出。此时问题不在数值格式（bf16 也救不了一个每步都发散的训练），而在 lr 或初始化。判据可以写成：连续 20 次 `update()` 都发生了 backoff 且没有一次 growth。
   - **此时这次 fp16 实验不能作为第 3 周的证据**，因为采到的 `get_scale()` 曲线反映的是训练发散，不是 fp16 的精度特性；必须先在 bf16 下把同一配置跑收敛，确认 lr 与初始化没问题，再重跑 fp16。这就是本次实验的**失败边界**：只有当 bf16 对照运行是健康的，fp16 运行的 scale 曲线才有解释力。

**0–4 分 rubric**

- 0：说不清 bf16 为什么不需要 scaler。
- 1：格式表与"指数位相同"的理由正确。
- 2：再答对"哪两类量必须在 fp32 算"及 autocast 的保证机制。
- 3：1–4 问全对，含两种 fp16 失败方式与"静默的更危险"的理由。
- 4：以上全对，且第 5 问把决策拆成"做/不做/替代证据"三部分并标注了证据等级，第 6 问给出可执行的健康形态判据与明确的失败边界。

**常见误区**

- 认为 bf16 比 fp16"更精确"。bf16 的相对精度（7.8e-3）实际比 fp16（9.8e-4）**差**约 8 倍；bf16 的优势只在动态范围。
- 认为在 5070 Ti 上跑通 fp16 就等于 V100 上会跑通。5070 Ti 的 fp16 是模拟同一数值格式，但 kernel 选择、SDPA backend、算力特性都不同；本周能迁移的是**数值行为的证据**（scale 曲线、跳步率），不是性能数字。

**追问**

- 若 fp16 pretrain 跑出来的 `get_scale()` 稳定在 1024 而不是 65536，这说明什么？需要担心吗？
- 你打算怎么在报告里区分"5070 Ti 上的 fp16 数值证据"和"V100 上待验证的性能结论"？给出两条标注规则。

**定位**：`01_FOUNDATIONS.md` §4.1、§4.2、§4.5、§6 第 5 条、§7（向第 3 周的连接）；`lab/src/mm_probe/hooks.py`（`grad_global_norm`）、`grpo_stats.py`、`parse_log.py`；`00_WEEK_CARD.md`「三类环境边界」。

---

## 评分汇总表

| 层 | 题数 | ID |
| --- | --- | --- |
| Recall | 5 | M01-R-01 … M01-R-05 |
| Explain | 5 | M01-E-01 … M01-E-05 |
| Apply | 4 | M01-A-01 … M01-A-04 |
| Debug | 5 | M01-D-01 … M01-D-05 |
| Design | 5 | M01-P-01 … M01-P-05 |
| Trade-off | 4 | M01-T-01 … M01-T-04 |
| **合计** | **28** | |

**Gate 换算建议**（对应 `00_WEEK_CARD.md` 的 A1 限时 45 分钟 Gate）：

- Recall + Explain 两层平均 ≥ 3.0 分：基础常量与机制过关。
- Apply 层 M01-A-01 与 M01-A-02 两道手算题各 ≥ 3 分：Gate 题 1 与显存模型过关。
- Debug 层平均 ≥ 2.5 分且 M01-D-02 ≥ 3 分：Gate 题 3（mask 错位反推）过关。
- Design + Trade-off 两层平均 ≥ 2.5 分，且至少两题达到 4 分（即完整给出假设/证据/替代解释/失败边界/可验证下一步五要素）。
- 任一层平均 < 2.0 分：该层对应的 `01_FOUNDATIONS.md` 章节需要重读，并补一次对应的 lab 观测再重考该层。
