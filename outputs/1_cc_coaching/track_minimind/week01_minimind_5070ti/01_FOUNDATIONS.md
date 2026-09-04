# Week M01 Foundations — MiniMind 单机全链路：从一条 JSONL 到 GRPO 的因果链

> 轨道 `track_minimind` · 生成日期 2026-09-04 · 生成方 cc-theory-tutor  
> 代码基线：MiniMind commit `7a6fddd63a30c06b2fdd5fac4089922b29bc841b`（`已确认`，2026-09-04）  
> 标签约定：`已确认` = 源码/官方文档/本机运行输出支持；`推断/计算` = 由已确认事实推出的数值，附公式；`未知` = 本周实验前不给数。  
> 本机核验说明：本文中 tokenizer 相关的 token id 由本机 `ResearchAgentPy310`（transformers 4.57.6）加载固定 commit 的 `model/tokenizer.json` + `tokenizer_config.json` 实际编码得到，属于 `已确认`；显存/吞吐数字均为 `推断/计算`，等待 5070 Ti 探针。

---

## 0. 本周要拥有的能力（一段话）

拿到任意一条 MiniMind 格式的样本（`{"text"}`、`{"conversations"}`、`{"chosen","rejected"}` 或 RLAIF prompt），你能不查文档写出它的 token 序列、label 序列（哪些位置是 -100）、进入 decoder-only 模型后每一层的张量 shape、loss 的归一化分母，以及这一条样本在 pretrain / SFT / LoRA / DPO / GRPO 五种训练中分别以什么形式参与梯度；你还能用自己的仪表（`lab/src/mm_probe/`）在 5070 Ti 上把上述每一环的不变量量出来，并在 label mask 被人为错位时，只凭 loss 曲线和固定 prompt 的生成结果反推故障点。核心判断标准只有一个：**loss 下降只证明优化器在拟合某个目标；目标是否正确要靠 label、mask、reward 和评测各自的不变量来证明。**

---

## 1. 对象：数据与模型（具体 ID、样例、shape）

### 1.1 Tokenizer（`已确认`，本机核验 2026-09-04）

- 类型：ByteLevel BPE（`tokenizer.json` 中 `model.type = "BPE"`，`pre_tokenizer.type = "ByteLevel"`），词表 6400，`post_processor = null`（即 `tokenizer(prompt)` **不会**自动加 BOS/EOS）。
- 特殊 token：`<|endoftext|>`=0（pad/unk）、`<|im_start|>`=1（bos）、`<|im_end|>`=2（eos）、`<think>`=25、`</think>`=26、`<tool_call>`=21、`</tool_call>`=22。
- 常量：`bos_id = tokenizer("<|im_start|>assistant\n") = [1, 1388, 570, 811, 234]`（5 个 token：`<|im_start|>`, `ass`, `ist`, `ant`, `\n`），`eos_id = tokenizer("<|im_end|>\n") = [2, 234]`。注意 `assistant` 不是单个 token，这决定了 label 起点在第 5 个 token 之后。
- 中文粒度：`今天天气不错` → `[5640, 3660, 417, 1745]`（4 token / 6 字，约 1.5 字/token，与脚本注释“中文 1 token ≈ 1.5–1.7 字符”一致）。

### 1.2 三种数据样本与它们的 token/label（`已确认` 规则 + 本机编码）

**Pretrain**（`PretrainDataset.__getitem__`）：`tokens = [bos] + tokenizer(text, add_special_tokens=False, max_length=max_len-2, truncation=True) + [eos]`，右侧补 pad 到 `max_length`；`labels = input_ids.clone()`，`labels[input_ids == pad] = -100`。例：`{"text":"今天天气不错"}`，`max_seq_len=8` → `input_ids=[1,5640,3660,417,1745,2,0,0]`，`labels=[1,5640,3660,417,1745,2,-100,-100]`。bos 和 eos 都在 label 里。

**SFT**（`SFTDataset`）：`pre_processing_chat` 以 20% 概率在最前面插入随机 system prompt；`create_chat_prompt` 调 `tokenizer.apply_chat_template(..., add_generation_prompt=False)`；chat template 把每个 assistant 轮渲染成 `<|im_start|>assistant\n<think>\n{reasoning}\n</think>\n\n{content}<|im_end|>\n`；`post_processing_chat` 以 80% 概率把所有空思考块 `<think>\n\n</think>\n\n` 删掉；然后 `tokenizer(prompt).input_ids[:max_length]` + pad，`generate_labels` 扫描 `bos_id`，把 **`bos_id` 之后到 `eos_id` 结尾（含 `<|im_end|>\n`）** 的位置复制为 label，其余 -100。无 packing：一条样本一行，短样本用 pad 填满。

本机实际编码的两轮样本（空思考块已删除的 80% 分支）：

```text
{"conversations":[{"role":"user","content":"请介绍一下你自己。"},{"role":"assistant","content":"我是minimind。"},
                  {"role":"user","content":"再见"},{"role":"assistant","content":"再见！"}]}
渲染后：<|im_start|>user\n请介绍一下你自己。<|im_end|>\n<|im_start|>assistant\n我是minimind。<|im_end|>\n<|im_start|>user\n再见<|im_end|>\n<|im_start|>assistant\n再见！<|im_end|>\n
共 43 token；max_seq_len=64 时后 21 位为 pad(0)
```

| 位置 t | token id | 文本 | label[t] | 位置 t 的 logits 预测谁（= label[t+1]） |
| --- | --- | --- | --- | --- |
| 0–3 | 1,832,311,234 | `<\|im_start\|>`,`us`,`er`,`\n` | -100 | -100（不训练） |
| 4–9 | 960,2919,2360,441,1153,302 | 请 介绍 一下 你 自己 。 | -100 | -100 |
| 10–11 | 2,234 | `<\|im_end\|>`,`\n` | -100 | -100 |
| 12–16 | 1,1388,570,811,234 | `<\|im_start\|>`,`ass`,`ist`,`ant`,`\n` | -100 | t=16 预测 463“我”（**训练**） |
| 17–22 | 463,357,4704,467,916,302 | 我 是 min im ind 。 | 自身 id | 下一个内容 token |
| 23–24 | 2,234 | `<\|im_end\|>`,`\n` | 2, 234 | t=22 预测 2；t=23 预测 234；t=24 预测 label[25]=-100 |
| 25–37 | … | 第二轮 user 段 + assistant 头 | -100 | t=37 预测 2223“再” |
| 38–42 | 2223,1558,1364,2,234 | 再 见 ！ `<\|im_end\|>` `\n` | 自身 id | — |
| 43–63 | 0 | pad | -100 | -100 |

有效 label 数 = 8 + 5 = 13；其中 2 个是 `<|im_end|>`（模型学会停）。**不变量**：每条样本有效 label 数 = Σ(assistant 内容 token 数 + 2)，且 label 中一定包含 id 2。

**DPO**（`DPODataset`）：chosen 与 rejected 各自走 `apply_chat_template` → `tokenizer(..., truncation=True, max_length, padding='max_length')` → `generate_loss_mask`（与 SFT 同样的 bos_id/eos_id 扫描，但输出 0/1 mask）；返回 `x = ids[:-1]`, `y = ids[1:]`, `mask = loss_mask[1:]`。这里 shift 在 dataset 里做了，模型 forward 不传 `labels`。默认 `max_seq_len=1024` → `x_chosen.shape = [1023]`。

**RLAIF/GRPO**（`RLAIFDataset`）：只返回 prompt 字符串（`apply_chat_template(conversations[:-1], add_generation_prompt=True, open_thinking=随机)`），生成头为 `<|im_start|>assistant\n<think>\n`（开思考）或 `<|im_start|>assistant\n<think>\n\n</think>\n\n`（关思考）；response 由 policy 自己采样，reward 由规则 + InternLM2-1.8B reward model 打分。

### 1.3 模型：MiniMind 默认配置（`已确认`：`MiniMindConfig.__init__`）

| 符号 | 值 | 来源 |
| --- | --- | --- |
| H hidden_size | 768 | 默认 |
| L num_hidden_layers | 8 | 默认 |
| n_h / n_kv | 8 / 4 → n_rep = 2 | 默认 |
| d head_dim | 768/8 = 96 | `hidden_size // num_attention_heads` |
| I intermediate_size | ceil(768·π/64)·64 = ceil(37.70)·64 = **2432** | `math.ceil(hidden_size*math.pi/64)*64` |
| V vocab | 6400 | 默认 |
| RoPE θ | 1e6；预计算 `max_position_embeddings=32768` 个位置 | `precompute_freqs_cis` |
| RMSNorm eps | 1e-6 | `rms_norm_eps` |
| tie_word_embeddings | True（`embed_tokens.weight = lm_head.weight`） | `MiniMindForCausalLM.__init__` |
| QK-norm | 有：`q_norm/k_norm = RMSNorm(head_dim)` 在 RoPE 之前 | `Attention.__init__/forward` |
| attention | `F.scaled_dot_product_attention(is_causal=True)`；仅当 `attention_mask is None 或全 1` 且无 KV cache 时 | `Attention.forward` |

**参数量**（`推断/计算`，由上表算出）：

```text
embedding      = V·H = 6400·768                       = 4,915,200   （lm_head 与之共享，不重复计数）
attention/层   = H·H(q) + H·(n_kv·d)(k) + 同(v) + H·H(o) = 589,824+294,912+294,912+589,824 = 1,769,472
q_norm+k_norm  = 2·96                                  = 192
FFN/层         = 3·H·I = 3·768·2432                    = 5,603,328
2×RMSNorm/层   = 2·768                                 = 1,536
每层小计       = 7,374,528
8 层 + 末层 norm 768 + embedding                       = 58,996,224 + 768 + 4,915,200 = 63,912,192 ≈ 63.9 M
```

`init_model` 打印的 `Model Params` 应等于 63.91M（tied 权重在 `model.parameters()` 里只出现一次）；若你在 5070 Ti 上看到不同数字，先怀疑 `hidden_size/num_hidden_layers` 命令行参数。

---

## 2. 数学目标与推导

### 2.1 语言模型目标与 -100 归一化

对一条长度 T 的序列 x₁..x_T，模型给出条件分布 p_θ(x_{t+1} | x_{≤t})。训练目标：

```text
L = − (1 / |S|) · Σ_{t ∈ S} log p_θ(y_t | x_{≤t}),   S = { t : label[t+1] ≠ −100 }
```

代码（`MiniMindForCausalLM.forward`）：`x = logits[..., :-1, :]`, `y = labels[..., 1:]`, `F.cross_entropy(x.view(-1,V), y.view(-1), ignore_index=-100)`。默认 `reduction='mean'` → 分母 |S| 是**这个 micro-batch 内所有非 -100 位置的总数**，不是 B×T，也不是逐样本平均。推论：

- 一条 340 token 的 pretrain 样本贡献 339 个位置（bos 之后到 eos），pad 不计。
- SFT 里一条“user 300 token + assistant 20 token”的样本只贡献 22 个位置；一个 batch 的 loss 被长回答样本主导。
- 梯度累积：`loss / accumulation_steps` 后逐 micro-batch backward，等价于对 8 个 micro-batch 的“各自 token 均值”再取均值，而**不是**对 8 个 micro-batch 的所有 token 取一次均值。当各 micro-batch 有效 token 数差异大时，两者不同（常见误区）。

初始 loss 的不变量：随机初始化 + tied embedding 时 logits 近似均匀，loss ≈ ln V = ln 6400 = **8.764**。SFT 从 pretrain 权重起步的首步 loss 应明显低于 8.76（`推断`：通常 2–4）；若仍是 8.7 附近，说明权重没加载进去（见第 6 节）。

### 2.2 Decoder-only 一层的前向（符号与维度）

输入 h ∈ ℝ^{B×T×H}。

```text
RMSNorm:   n(h) = w ⊙ h / sqrt(mean_j h_j² + ε)                    （fp32 计算后 cast 回 h.dtype）
Q,K,V:     q = n(h)W_q ∈ ℝ^{B×T×n_h×d},  k,v = n(h)W_k, n(h)W_v ∈ ℝ^{B×T×n_kv×d}
QK-norm:   q ← RMSNorm_d(q), k ← RMSNorm_d(k)                       （逐 head 在 d=96 上归一化）
RoPE:      q ← q⊙cos + rotate_half(q)⊙sin，k 同；freq_i = θ^{−2i/d}, i=0..47, θ=1e6
GQA:       k,v 在 head 维 repeat n_rep=2 次 → ℝ^{B×T×8×96}
attention: softmax(q kᵀ / sqrt(96) + causal_mask) v，逐 head
o_proj:    h ← h + attn W_o
SwiGLU:    h ← h + W_down( silu(W_gate n₂(h)) ⊙ W_up n₂(h) ),  I=2432
```

RoPE 细节（`precompute_freqs_cis`/`apply_rotary_pos_emb`）：`rotate_half` 把第 i 维与第 i+48 维配对（NeoX 风格），因此 `freqs_cos` 是 `cat([cos, cos])` 形状 `[32768, 96]`。θ=1e6 意味着最低频维度的波长 2π·1e6 ≫ 训练长度 768，绝大多数维度在 340/768 内几乎不旋转——这是“小 θ 训练、大 θ 外推”之外的另一种选择：先天为长上下文留余量，代价是短序列上位置分辨率主要靠高频维。

### 2.3 SFT：同一个 CE，不同的 S

SFT 与 pretrain 的**唯一**数学差别是集合 S：pretrain 的 S 是全部非 pad 位置；SFT 的 S 是 assistant 段（内容 + `<|im_end|>\n`）。user/system/assistant 头（`<|im_start|>assistant\n` 五个 token）都不在 S 中，但它们仍在 attention 的上下文里。第 1.2 节的表说明：预测第一个内容 token 的位置是头部的 `\n`（t=16），所以“模型学会在看到 `assistant\n` 后开始答”仍然被训练到。

### 2.4 LoRA：把 ΔW 限制在秩 16

`model_lora.apply_lora`：对每个满足 `in_features == out_features` 的 `nn.Linear`（即 `q_proj`、`o_proj`，768×768；`k_proj/v_proj` 是 768→384，`gate/up/down` 是 768↔2432，均**不**挂 LoRA）替换 forward 为 `W x + B(A x)`，`A ∈ ℝ^{16×768}` 高斯 std 0.02 初始化，`B ∈ ℝ^{768×16}` 全零 → 初始 ΔW = 0，训练起点与基座完全一致。

```text
每个 LoRA 模块参数 = r·(in+out) = 16·(768+768) = 24,576
挂载数 = 8 层 × 2（q_proj, o_proj） = 16 个 → 393,216 ≈ 0.393 M
可训占比 = 0.393 / 63.91 ≈ 0.62 %                                   （推断/计算；train_lora.py 会打印这两个数）
```

`train_lora.py` 把非 lora 参数 `requires_grad=False`，优化器只吃 `lora_params`，因此 AdamW 状态只有 0.39M×8 B ≈ 3 MB；但**激活显存不变**（反向仍要穿过整个网络求 A、B 的梯度）。

### 2.5 DPO：把 reward model 藏进 log-ratio

DPO 目标（Rafailov et al., 2023）：

```text
L_DPO = − E [ log σ( β·( [log π_θ(y_w|x) − log π_ref(y_w|x)] − [log π_θ(y_l|x) − log π_ref(y_l|x)] ) ) ]
```

MiniMind（`train_dpo.dpo_loss`）的实现细节，每一项都影响数值：

1. `logits_to_log_probs`：`log_softmax(logits, dim=2)` 后 `gather` 出 y 的 log p，形状 `[2B, T−1]`。
2. `(log_probs * mask).sum(dim=1)`：**求和不求均值**，所以序列级 log-ratio 的量级 ∝ 回答 token 数；长回答的 margin 天然更大。
3. batch 前半是 chosen、后半是 rejected（`torch.cat([x_chosen, x_rejected], dim=0)`），`batch_size=4` 时 policy/ref 各看到 `[8, 1023]`。
4. `logits = pi_logratios − ref_logratios`，`loss = −logsigmoid(β·logits).mean()`，β=0.15。
5. ref 模型：`init_model` 二次加载同一份 `full_sft_768.pth`，`eval()` + `requires_grad_(False)`，forward 在 `torch.no_grad()` 里。**ref 永不更新**：它是“隐式 reward 的零点”，更新它等于让 KL 锚点漂移。

不变量：初始时 π_θ = π_ref → logits = 0 → loss = ln 2 = 0.693；训练后应看到 `chosen_reward = β·(logπ_θ − logπ_ref)(y_w)` 上升、`rejected_reward` 下降、margin > 0 的比例上升。这三个量原脚本不打印，由 `lab/src/mm_probe/dpo_check.py` 计算。

### 2.6 GRPO / CISPO：组内相对优势 + token 级策略梯度 + KL

一个 prompt 采 G=`num_generations` 条 response，reward r₁..r_G：

```text
A_i = (r_i − mean(r)) / (std_pop(r) + 1e-4)                         （train_grpo.py，unbiased=False）
ratio_{i,t} = exp( log π_θ(o_{i,t}) − log π_old(o_{i,t}) )
KL_{i,t}    = exp(log π_ref − log π_θ) − (log π_ref − log π_θ) − 1     （k3 估计，恒 ≥ 0）
GRPO:  ℓ_{i,t} = −( min(ratio·A_i, clip(ratio, 1−ε, 1+ε)·A_i) − β·KL ),  ε=0.2
CISPO: ℓ_{i,t} = −( sg[min(ratio, ε_high)]·A_i·log π_θ(o_{i,t}) − β·KL ),  ε_high=5.0, sg = detach
loss = mean_i ( Σ_t ℓ_{i,t}·m_{i,t} / Σ_t m_{i,t} )                    （先逐序列 token 均值，再序列均值）
```

要点：

- **reward 全等 ⇒ std=0 ⇒ A_i = 0/(1e-4) = 0 ⇒ 策略梯度项为零，只剩 β·KL 把策略往 ref 拉。** 这就是“reward 完全相同时没有学习信号”的精确含义；G=2 时两条 response 只要分数相同就整组作废。
- `completion_mask`：从 response 第 1 个 token 到**第一个** `<|im_end|>`（含）为 1，其后为 0；`generate()` 对已结束的序列继续填 eos，靠这个 mask 把它们排除。
- `old_per_token_logps` 由 `rollout_engine.rollout` 在生成结束后用**同一个 policy** 重新前向算出（`compute_per_token_logps`），而训练前向也用同一个 policy，且每个 rollout 只做一次 backward → 数学上 ratio ≡ 1，clip 不会触发（`推断`：唯一的偏差来自 bf16 数值和 `logits_to_keep` 切片带来的浮点差；`grpo_stats.py` 应打印 `max|ratio−1|`，预期 < 1e-2）。因此默认配置下 CISPO 退化为“组内归一化 REINFORCE + KL”，off-policy 修正只在 SGLang 引擎（权重每 `save_interval` 才同步）下真正起作用。
- 日志里的 `KL_ref` 是 `mean(logπ_ref − logπ_θ)`（有符号的 log-ratio 均值），不是上面的 k3，可以为负；不要把它当 KL 散度读。
- reward 组成（`calculate_rewards`）：长度项（20–800 字 +0.5 否则 −0.5）、思考块项（20–300 字 +1.0 否则 −0.5；恰好一个 `</think>` +0.25）、3-gram 重复惩罚（上限 0.5）、reward model 分数。长度和思考块是**可被 hack 的形状奖励**：模型可以只学会“写 20 字以上、放一个 `</think>`”就拿到 +1.75。

---

## 3. 从张量到代码

### 3.1 shape 表（pretrain 默认 B=32, T=340；SFT 默认 B=16, T=768）

| 阶段 | 张量 | pretrain shape | SFT shape | dtype（autocast bf16 下） |
| --- | --- | --- | --- | --- |
| dataset | `input_ids`, `labels` | [340], [340] | [768], [768] | int64 |
| embed | `h₀ = embed_tokens(ids)` | [32,340,768] | [16,768,768] | fp32（Embedding 不被 autocast 降级） |
| 每层 | `q` / `k` / `v`（view 后） | [32,340,8,96] / [32,340,4,96] / 同 k | [16,768,8,96] / [16,768,4,96] | bf16 |
| 每层 | `cos, sin` 切片 | [340,96] | [768,96] | fp32 buffer |
| 每层 | repeat_kv 后 transpose | q,k,v 均 [32,8,340,96] | [16,8,768,96] | bf16 |
| 每层 | SDPA 输出 → reshape | [32,340,768] | [16,768,768] | bf16 |
| 每层 | gate/up 输出 | [32,340,2432] ×2 | [16,768,2432] ×2 | bf16 |
| 每层 | 残差流 h | [32,340,768] | [16,768,768] | fp32（残差加法在 fp32 上做） |
| 输出 | `logits = lm_head(norm(h))` | [32,340,6400] | [16,768,6400] | bf16 |
| loss | `x=logits[:,:-1]`, `y=labels[:,1:]` | [32,339,6400] → view [10848,6400] | [16,767,6400] → [12272,6400] | CE 在 fp32 中计算 |

**关键不变量**（每条都对应一个 lab 检查）：

1. `labels[t] ∈ {−100} ∪ [0, 6400)`；`(labels != −100).sum() > 0` 对每个 micro-batch 成立 → `inspect_dataset.py`。
2. SFT：`labels[t] != −100 ⇒ input_ids[t] == labels[t]`；有效 label 数 = Σ(assistant 内容长 + 2) → `inspect_dataset.py`、`tests/test_label_mask.py`。
3. 右 pad + 无 attention_mask 时走 SDPA `is_causal` 路径：pad 在真实 token 之后，因果 mask 保证真实 token 看不到 pad；pad 位置的 logits 被 -100 丢弃，所以“pad 参与 attention”不污染 loss → `hooks.py` 记录 `Attention.forward` 走的分支。
4. `q_norm/k_norm` 在 RoPE 之前；顺序反了 RoPE 会被归一化抹掉旋转幅度 → 单测比较手写 attention 与模块输出。
5. tied embedding：`model.lm_head.weight.data_ptr() == model.model.embed_tokens.weight.data_ptr()` → `bounded_train.py` 启动时断言。
6. LoRA：挂载后 `sum(p.numel() for p in model.parameters() if p.requires_grad) == 393,216`，且首个 step 前 `B.weight.abs().sum() == 0` → 用 MiniMind 的 `trainer/train_lora.py`（本 lab 未封装 LoRA 入口；`apply_lora` 后自行打印可训参数量）。
7. DPO：`mask_chosen.sum(1) > 0` 且 `mask_chosen.sum(1) == 该样本 assistant token 数`；初始 loss ≈ 0.693 → `dpo_check.py`。
8. GRPO：`advantages.view(-1, G).sum(1) ≈ 0`（组内均值为零）；`completion_mask` 每行至多一个 eos 且其后全 0 → `grpo_stats.py`。

### 3.2 代码位置（函数名，指向 MiniMind 源文件与 lab）

| 环节 | MiniMind 源码（固定 commit） | lab 仪表 |
| --- | --- | --- |
| 样本 → token/label | `dataset/lm_dataset.py`: `PretrainDataset.__getitem__`, `SFTDataset.create_chat_prompt`, `SFTDataset.generate_labels`, `pre_processing_chat`, `post_processing_chat` | `lab/src/mm_probe/inspect_dataset.py`（打印第 1.2 节那张表） |
| 前向 | `model/model_minimind.py`: `RMSNorm.forward`, `precompute_freqs_cis`, `apply_rotary_pos_emb`, `repeat_kv`, `Attention.forward`, `FeedForward.forward`, `MiniMindBlock.forward`, `MiniMindModel.forward`, `MiniMindForCausalLM.forward` | `lab/src/mm_probe/hooks.py`（shape/显存/吞吐 hook） |
| 优化循环 | `trainer/train_pretrain.py`: `train_epoch`；`trainer/trainer_utils.py`: `get_lr`, `init_model`, `lm_checkpoint`, `SkipBatchSampler`, `setup_seed` | `lab/src/mm_probe/bounded_train.py`（有界训练 + 断言）、`parse_log.py` |
| SFT 故障注入 | `SFTDataset.generate_labels` | `lab/src/mm_probe/mask_fault.py`（`FAULT_MODES = none / assistant_all_ignored / user_in_loss`），也可用 `bounded_train.py --mask-fault <mode>` 直接训练对照 |
| LoRA | `model/model_lora.py`: `LoRA.forward`, `apply_lora`, `save_lora`, `merge_lora` | MiniMind `trainer/train_lora.py`（Day 3 可选扩展，本 lab 未封装入口） |
| DPO | `trainer/train_dpo.py`: `logits_to_log_probs`, `dpo_loss`, `train_epoch`；`DPODataset.generate_loss_mask` | `lab/src/mm_probe/dpo_check.py` |
| GRPO | `trainer/train_grpo.py`: `calculate_rewards`, `rep_penalty`, `grpo_train_epoch`；`trainer/rollout_engine.py`: `TorchRolloutEngine.rollout`, `compute_per_token_logps`；`trainer_utils.py`: `LMForRewardModel.get_score` | `lab/src/mm_probe/grpo_stats.py` |
| 生成/评测 | `MiniMindForCausalLM.generate` | `lab/scripts/eval_generate.py` |

### 3.3 最小片段：公式 ↔ 代码

```python
# SFT 的 CE：位置 t 的 logits 预测 labels[t+1]，-100 不进分母（MiniMindForCausalLM.forward）
x, y = logits[..., :-1, :], labels[..., 1:]
loss = F.cross_entropy(x.reshape(-1, V), y.reshape(-1), ignore_index=-100)   # 分母 = (y != -100).sum()
# DPO：序列级 log-ratio 的差再过 logsigmoid（train_dpo.dpo_loss）
loss = -F.logsigmoid(beta * ((pi_w - pi_l) - (ref_w - ref_l))).mean()
# GRPO 组内优势（train_grpo.grpo_train_epoch）
adv = (r - r.view(-1, G).mean(1).repeat_interleave(G)) / (r.view(-1, G).std(1, unbiased=False).repeat_interleave(G) + 1e-4)
```

---

## 4. 数值行为

### 4.1 精度格式（`已确认`：IEEE 754 / bf16 定义）

| 格式 | 指数位 | 尾数位 | 最大值 | 最小正规数 | 相对精度 |
| --- | --- | --- | --- | --- | --- |
| fp32 | 8 | 23 | 3.4e38 | 1.2e-38 | 6e-8 |
| fp16 | 5 | 10 | 65,504 | 6.1e-5 | 9.8e-4 |
| bf16 | 8 | 7 | 3.4e38 | 1.2e-38 | 7.8e-3 |

bf16 的动态范围与 fp32 相同，只是精度粗 3 位；fp16 精度高 3 位但 65,504 就溢出、6e-5 以下就进入次正规区。**梯度**的典型量级是 1e-8 ~ 1e-3，正好落在 fp16 的危险区。

### 4.2 fp16 溢出、GradScaler 与跳步（`已确认`：PyTorch `torch.amp.GradScaler` 文档）

MiniMind 的 pretrain/SFT/LoRA/DPO 都写成 `scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype=='float16'))`，循环是：

```text
scaler.scale(loss).backward()      # 反向前把 loss 乘 s（初始 s = 2^16 = 65536），梯度整体放大 s 倍，避免下溢
scaler.unscale_(optimizer)         # 梯度除以 s，之后 clip_grad_norm_ 才看到真实范数
clip_grad_norm_(params, 1.0)
scaler.step(optimizer)             # 若本轮 unscale 时发现 inf/nan：跳过 optimizer.step()
scaler.update()                    # 发现 inf/nan：s ← s×0.5；连续 2000 步正常：s ← s×2
```

推论与不变量：

- bf16 下 `enabled=False`：`scale()` 返回原 loss，`step()` 直接调 `optimizer.step()`，`get_scale()` 恒为 1.0。**bf16 不需要 loss scaling 的原因是指数位与 fp32 相同**，梯度不会因为格式而下溢；代价是尾数只有 7 位，逐 token 的 log-prob 差（DPO/GRPO 的 ratio）在 bf16 里分辨率约 1e-2，所以这些量必须在 fp32 里算（`log_softmax` 在 autocast 下自动升到 fp32，`已确认`：autocast op 列表）。
- fp16 下每次跳步都是一个**没有更新但 lr 调度照走**的 step：`get_lr(epoch*iters + step, …)` 以数据 step 计数，跳步不回退。`hooks.py` 应记录 `scaler.get_scale()` 序列：健康曲线是“偶尔减半、随后每 2000 步翻倍”；持续减半到 < 1 说明每步都溢出，通常是 lr 或初始化问题而非精度问题。
- 梯度累积 + 溢出：`scaler.step` 在累积窗口末尾调用一次，窗口内任一 micro-batch 产生 inf 都会导致**整个窗口**被丢弃。
- 前向溢出与 scaler 无关：fp16 下 `q·kᵀ/√96` 若 q、k 幅值大（QK-norm 已经把每个 head 归一到 √96 量级，缓解了这一点），或 SwiGLU 中 `silu(gate)⊙up` 乘积超 65,504，会直接给出 inf logits，loss = nan，scaler 只能反复跳步，不能修复。

### 4.3 lr 与 global batch

```text
global_batch(序列) = batch_size × accumulation_steps × world_size
pretrain 默认: 32 × 8 × 1 = 256 序列/优化步 → 最多 256×339 = 86,784 个 loss 位置/步（pad 越多越少）
SFT 默认:      16 × 1 × 1 = 16 序列/优化步 → 有效 token 数 = 16 条样本的 assistant 长度之和（未知，Day 1 用 inspect_dataset 统计分布）
```

`get_lr(s, S, lr) = lr·(0.1 + 0.45·(1 + cos(π·s/S)))`：s=0 时 1.0·lr，s=S/2 时 0.55·lr，s=S 时 0.1·lr；无 warmup；s 按 micro-batch 计数、S = epochs × 每 epoch micro-batch 数。改 `accumulation_steps` 不改变 lr 曲线形状，只改变每个优化步吃多少 token。因此“把 batch 从 8 提到 32 并保持 lr 不变”意味着每个优化步的梯度更平均、噪声更小，但步数减少 4 倍；在 64M 模型上通常需要按 √k 或线性规则上调 lr 才能在同样 token 数内到达同样 loss（`推断`，Day 2 可做 128 样本 overfit 对照）。

pretrain lr=5e-4、SFT lr=1e-5、LoRA lr=1e-4、DPO lr=4e-8、GRPO lr=3e-7（`已确认`：各脚本 argparse 默认）。四个后训练阶段的 lr 相差 4 个数量级，理由不是梯度大小（AdamW 对梯度做了归一化，每步每个参数的位移量级 ≈ lr，与梯度绝对值基本无关），而是**允许模型离起点漂多远**。

### 4.4 DPO 的 lr 为什么小到 4e-8（`推断`，附源码注释 `已确认`）

源码注释：`--learning_rate 4e-8 "初始学习率（建议<=5e-8避免遗忘）"`。定量理解：

```text
AdamW 每步位移 ≈ lr（梯度经 m/√v 归一化后量级为 1）
dpo.jsonl = 53.65 MB；样本数未知，设为 N；优化步数 = N/4
每个参数总漂移上界 ≈ (N/4)·4e-8 = N·1e-8；N=2万 时 ≈ 2e-4
MiniMind 线性层权重初始 std ≈ 0.02 → 相对漂移 ≈ 1%
```

DPO 的信号是**序列级、只有一个 bit（chosen 比 rejected 好）**，又对 64M 全参训练，梯度方向噪声极大；lr 再大 10 倍，1% 会变成 10%，SFT 学到的通用能力（chat 格式、停止）会被 DPO 的 log-ratio 目标冲掉，这就是“遗忘”。对照：7B 模型的 DPO 常用 5e-7；MiniMind 小 100 倍却用小 10 倍的 lr，因为小模型每个参数承载的信息更密。判定 lr 是否过小的办法不是看 loss，而是看 `dpo_check.py` 输出的 margin 分布是否在有界训练内离开 0。

### 4.5 GRPO 没有 GradScaler：fp16 下的两种失败

`train_grpo.py` 直接 `loss.backward()` → `clip_grad_norm_` → `optimizer.step()`，没有 scaler。5070 Ti 用 bf16 不受影响；但若为了准备 V100 周把 `--dtype float16` 打开：

- **下溢（静默）**：lr=3e-7 量级的策略梯度经 fp16 反向，很多梯度落到 6e-5 以下变成次正规数或 0；表现为 reward 曲线不动、`KL_ref` 恒 0，看起来像“没学到”，实际是梯度被格式吃掉。检查：`grpo_stats.py` 打印 `grad_norm` 与零梯度参数比例。
- **溢出（显性）**：反向中某个中间量 > 65,504 → inf → `clip_grad_norm_` 算出 nan 范数 → 所有梯度乘以 nan → `optimizer.step()` 把 nan 写进权重，**下一次 rollout 开始全部生成 pad/垃圾**，不可恢复，只能回滚 checkpoint。没有 scaler 意味着没有“跳过这一步”的保护。
- 前向里的 `torch.exp(kl_div)` 和 `torch.exp(logp − old_logp)` 输入来自 fp32 的 log_softmax，本身安全；真正的 fp16 风险在矩阵乘和 attention 的反向。

### 4.6 初始化与 tied embedding

`MiniMindForCausalLM` 走 transformers 的 `post_init`（`PreTrainedModel` 默认 `_init_weights`：Linear/Embedding 正态 std=`initializer_range`，默认 0.02）。tied embedding 使 `embed_tokens.weight` 同时收到两路梯度：作为输入查表的稀疏梯度（只有 batch 中出现的 token 行）和作为 `lm_head` 的稠密梯度（所有 6400 行，来自 softmax）。后者主导；这也是为什么小词表模型 tie 权重后训练早期 loss 下降更快（`推断`）。

---

## 5. 硬件与分布式代价（5070 Ti 16 GB，单卡）

### 5.1 显存模型（`推断/计算`；所有数字待 `hooks.py` 的 `torch.cuda.max_memory_allocated()` 核验）

```text
静态项（与 batch 无关）
  参数 fp32 master        63.9M × 4 B = 256 MB
  梯度 fp32               256 MB
  AdamW m,v fp32          511 MB
  小计                    ≈ 1.02 GB           （LoRA：参数 256 MB + 梯度/状态仅 0.39M×12 B ≈ 5 MB）
  RoPE buffer             32768×96×4 B×2 = 25 MB
  CUDA context + cuBLAS workspace  ≈ 0.5–1 GB（未知，探针给）

动态项（∝ B×T）
  logits bf16             B·T·V·2 B
  CE 内部 fp32 log_softmax + 其梯度   ≈ 2 × B·T·V·4 B
  每层激活（bf16，供反向）≈ (10H + 3I) 元素/token/层 ≈ 15k 元素 ≈ 30 KB；8 层 ≈ 240 KB/token
```

| 配置 | tokens/step | logits bf16 | CE fp32 ×2 | 激活 ≈ | 合计（含静态 1.0 GB，不含 context） |
| --- | --- | --- | --- | --- | --- |
| pretrain B=32,T=340 | 10,880 | 32×340×6400×2 = 139 MB | 557 MB | 2.6 GB | ≈ 4.3 GB |
| SFT B=16,T=768 | 12,288 | 157 MB | 629 MB | 2.9 GB | ≈ 4.7 GB |
| DPO B=4（×2 chosen/rejected）,T=1023 | 8,184 | 105 MB（policy）+105 MB（ref 临时） | 419 MB + ref 210 MB 临时 | 2.0 GB | ≈ 3.9 GB（+ref 参数 256 MB） |

单样本 logits 显存的公式记住即可：`B × T × V × bytes`，1 条 SFT 样本 = 768×6400×2 B = 9.8 MB（bf16），fp32 翻倍。V=6400 让 MiniMind 的 logits 很便宜；换成 Qwen 词表 151,936 时同一公式给出 233 MB/样本，logits 会成为主项——这是第 12 周 CPT 的显存差异根源。

### 5.2 GRPO 三模型同驻

```text
policy   fp32 + grad + AdamW           ≈ 1.02 GB
ref      fp32（init_model 默认 fp32）   ≈ 256 MB
reward   internlm2-1_8b-reward fp16     safetensors 3.40 GB（已确认文件大小）→ 权重 ≈ 3.4 GB + 其推理激活（未知，估 0.3–1 GB）
```

rollout 与训练前向的动态项，默认 `batch_size=2, num_generations=6, max_seq_len=768, max_gen_len=1024`：

```text
序列数   = B·G = 12；每条 ≤ 768 + 1024 = 1792 token；总 token ≤ 21,504
KV cache = 12 KB/token（8 层 × 2 × 4 kv_heads × 96 × 2 B）× 21,504 ≈ 264 MB（生成期）
训练前向 logits [12,1792,6400]：bf16 275 MB；log_softmax fp32 550 MB（保留给反向）+ 梯度 550 MB
ref 前向同尺寸 log_softmax fp32 550 MB（no_grad，临时）
激活 21,504 token × 240 KB ≈ 5.2 GB
```

还有一个源码级陷阱（`推断`，从 `Attention.forward` 的分支条件读出，未在 GPU 实测）：训练前向传入 `attention_mask = (outputs != pad)`。`batch_size=2` 时两条 prompt 长度不同 → 左 pad → mask 含 0 → **不满足 SDPA 分支条件，落到手写 attention**，materialize `scores ∈ [12, 8, 1792, 1792]`：bf16 616 MB，`F.softmax(scores.float())` 的 fp32 输出 1.23 GB 被 autograd 保存，加上 bf16 概率 616 MB，每层 ≈ 1.85 GB，8 层 ≈ 14.8 GB。这一项单独就能打爆 16 GB。`batch_size=1` 时 mask 全 1 → 走 SDPA（flash/mem-efficient kernel，不 materialize T×T）。所以周卡建议的 `batch 1 / gen 2 / seq 256` 起步不只是“少算 token”，而是**换了 attention 路径**。

推荐起步配置的估算：`batch_size=1, num_generations=2, max_seq_len=256, max_gen_len=256` → 2 条序列 × 512 token = 1,024 token → 激活 0.25 GB、logits fp32 26 MB、KV 12 MB；总量 ≈ 1.0 + 0.26 + 3.4 + 1（reward 激活/context）+ 0.3 ≈ **6 GB**（`推断`）。之后按 G=4 → 6 → 8 递增，每次记录 `max_memory_allocated` 与 `advantages_std`（见第 6 节第 8 条）。

### 5.3 时间在哪里

GRPO 每步 = 生成（自回归，逐 token 8 层前向，batch 小时 GPU 利用率极低）+ reward model 前向（1.8B，逐条 response 循环调用 `get_score`，是 Python 级串行）+ policy/ref 前向 + 一次反向。`推断`：生成与 reward 打分占 step 时间 > 80%，而 backward 不到 10%；这就是“RL 里生成比训练更拖慢”的单卡版本。`grpo_stats.py` 应分别计时这四段。

预训练吞吐：64M 模型每 token 前反向 ≈ 6·N = 6 × 63.9M ≈ 0.38 GFLOP；一个默认优化步 86.8k token ≈ 33 TFLOP。5070 Ti 的 bf16 峰值算力和实测 MFU 是 `未知`，Day 2 用 `hooks.py` 测 tokens/s 后再反推 MFU；不要用 3090 的“两小时”类比。

### 5.4 分布式（本周不做，但要预先知道差异）

MiniMind 多卡只有 DDP（`torchrun --nproc_per_node N`），每卡完整模型；`lm_checkpoint` 恢复时 `step = step × saved_ws // current_ws` 换算。GRPO 在 DDP 下每卡都复制 ref 与 1.8B reward model，显存 ×N 而不是 /N——这是第 7–8 周 FSDP/角色拆分要解决的问题。

---

## 6. 失败模式与识别方法（症状 → 不变量 → 最小检查）

| # | 症状 | 被破坏的不变量 | 最小检查（lab） |
| --- | --- | --- | --- |
| 1 | SFT 首步 loss ≈ 8.7（≈ ln 6400），像从零开始 | `init_model` 应把 `pretrain_768.pth` 全部 key 加载；`load_state_dict(strict=False)` 会静默跳过不匹配的 key | `bounded_train.py` 启动时打印 `missing_keys/unexpected_keys`；对比 `eval_generate.py` 在 pretrain 权重上的续写是否通顺 |
| 2 | SFT loss 几步内掉到 < 0.5 且生成只输出 `<\|im_end\|>` | 每样本有效 label 数 = Σ(assistant 内容 + 2)；若 `bos_id` 匹配失败只剩 eos 段进 loss | `inspect_dataset.py --stage sft --fixture --n 1` 看逐 token 表并数有效 label；`tests/test_label_mask.py` 用第 1.2 节样本断言 13 个 |
| 3 | 固定 prompt 生成时模型复述/续写 user 的问题，或自己生成 `<\|im_start\|>user` | user 段必须全 -100 | `mask_fault.py --mode user_in_loss` 对照曲线：user 进 loss 时 loss 初值更低（用户文本更可预测）、曲线更平滑；配 `eval_generate.py` 观察角色越界 |
| 4 | SFT loss 曲线正常但生成永不停止直到 max_new_tokens | label 必须包含 id 2（`<\|im_end\|>`） | `inspect_dataset.py --stage sft` 的统计里 `label==2` 的样本比例应为 100%；若比例 < 100%，多半是 `max_seq_len` 把 `<\|im_end\|>` 截掉了，用 `--max-seq-len` 加长复核 |
| 5 | fp16 下 loss 出现 nan 或 `scaler.get_scale()` 一路减半 | 前向中间量 < 65,504；scaler 跳步应是偶发 | `hooks.py` 记录每步 `get_scale()` 与 `isfinite(loss)`；先切 bf16 确认是精度而非 lr 问题 |
| 6 | loss 停在 ln V 附近不动，lr 正常 | 每个 micro-batch `(labels != -100).sum() > 0`；数据 `text` 字段非空 | `inspect_dataset.py` 扫前 1000 条：空文本比例、截断比例、pad 比例 |
| 7 | GRPO 第一步 OOM，日志停在训练前向 | `batch_size=1` 或所有 prompt 等长 → `attention_mask` 全 1 → SDPA 分支 | `grpo_stats.py` 在 `Attention.forward` 挂 hook 报告分支；`hooks.py` 打印 `max_memory_allocated` 分段峰值 |
| 8 | GRPO `Adv Std` 长期 ≈ 0，`Reward` 不动 | 组内 reward 必须有方差；G=2 时同分概率高 | `grpo_stats.py` 逐组打印 reward 向量与 `std==0` 组的比例；比例 > 50% 时升 G 或改 reward |
| 9 | `Reward` 上升但生成越来越长/越来越短、`</think>` 出现多次或空 | 形状奖励（长度 ±0.5、思考块 +1.0/+0.25）被 hack；reward model 分数分量应同步上升 | `grpo_stats.py` 把 `calculate_rewards` 的四个分量拆开记录；只看总 reward 会被骗 |
| 10 | `KL_ref` 绝对值持续增大，通用问答变差 | β·KL 应把策略锚在 ref 附近；β=0.1 是否足够取决于 lr 与步数 | `grpo_stats.py` 记录 k3 KL（非日志里的有符号均值）；`eval_generate.py` 用 SFT 前后同一组 prompt 对照 |
| 11 | DPO loss 停在 0.693 | margin = β·((πθ−ref)(y_w) − (πθ−ref)(y_l)) 应离开 0；`mask.sum(1) > 0` | `dpo_check.py` 打印 chosen/rejected reward、margin>0 比例、每样本 mask 和 |
| 12 | resume 后 loss 跳变或曲线不重合 | `_resume.pth['model']` 是 `half()` 后的权重（fp32 master 被舍入到 fp16）；数据增广用 `random.random()`（system prompt 20%、空思考块 80%）在跳过 batch 后 RNG 流不同；SDPA 反向非确定 | `bounded_train.py --resume` 对比中断前后各 20 步 loss 的均值差；预期不 bitwise，但均值差应在噪声范围内 |
| 13 | 吞吐远低于预期，GPU 利用率低 | DataLoader 在线 tokenize，`num_workers=8`；Windows 上 worker 启动/序列化开销大 | `hooks.py` 分别计时 data wait / forward / backward；`num_workers` 0/2/4 对照 |
| 14 | LoRA 训练 loss 不动 | `B` 零初始化保证首步等价基座，但 lora 参数必须 `requires_grad=True` 且在优化器里 | 在 MiniMind `train_lora.py` 里 `apply_lora` 之后打印可训参数量应为 393,216，且 1 步后 `B.weight.abs().sum() > 0` |

---

## 7. 与前后周的连接

- **向后（第 2 周 VLM）**：SFT 的 label mask 规则原样迁移，只是 user 段里多了图像 token 占位；“谁进 loss”仍是 assistant 段。第 1.2 节的表就是你要在 VLM 样本上重画的表。
- **向后（第 3 周 V100/FP16）**：第 4.2、4.5 节的 GradScaler 与“GRPO 无 scaler”在 V100 上从预防知识变成必答题；本周在 5070 Ti 上用 `--dtype float16` 跑一次有界 pretrain，把 `get_scale()` 曲线留作证据。
- **向后（第 7–8 周 DDP/FSDP）**：第 5.4 节的“每卡完整 ref + reward”是 FSDP 与角色拆分的动机；`lm_checkpoint` 的 world_size 换算是 distributed checkpoint 的雏形。
- **向后（第 12–13 周 FinExec CPT/SFT/GRPO）**：第 2.1 节的分母规则、第 5.1 节的 `B×T×V` 公式、第 2.6 节的形状奖励 hack，都在 Qwen 词表（151,936）与真实金融数据上被放大：logits 从 10 MB/样本变成 233 MB/样本，reward hacking 从“写 20 字”变成“引用格式正确但数字错误”。
- **向前（无）**：本轨道在 14 周主线之前，没有周证据前置。

---

## 8. Teach-back 清单（对应周卡 Gate）

Gate：A1 限时 45 分钟；两次不同日期证据。不看文档，你应能：

1. **手写 token/label**：给一条新的多轮 chat 样本（含 system、两轮 assistant），写出渲染后的模板字符串、`bos_id=[1,1388,570,811,234]`、`eos_id=[2,234]`，逐位置标出 -100 与有效 label，说明位置 t 的 logits 预测 label[t+1]，并算出有效 label 数（Gate 题 1）。
2. **-100 与分母**：说出 `F.cross_entropy(ignore_index=-100)` 的分母是 micro-batch 内非 -100 位置数；解释梯度累积为何是“均值的均值”；给出初始 loss ≈ ln 6400 = 8.76 及其用途。
3. **一层的 shape**：默认配置下写出 q `[B,T,8,96]`、k/v `[B,T,4,96]`、repeat_kv 后 `[B,8,T,96]`、gate/up `[B,T,2432]`、logits `[B,T,6400]`；算出 63.9M 参数与 LoRA 0.39M（0.62%）。
4. **DPO 吃什么**：输入是 chosen/rejected 两条完整对话的 `x=ids[:-1], y=ids[1:], mask`；policy 与冻结 ref 各前向一次；loss 用 assistant 段 log-prob 的**和**做序列级 log-ratio；β=0.15；初始 loss 0.693；lr 4e-8 的“漂移上界 ≈ 步数 × lr”论证（Gate 题 2 前半）。
5. **GRPO 吃什么**：输入是 prompt → G 条采样 response → 规则 + reward model 打分 → 组内 `(r−mean)/(std+1e-4)` → token 级 loss（CISPO 用 `sg[min(ratio, 5)]·A·logπ`，GRPO 用 PPO clip 0.2）+ β·k3 KL；`completion_mask` 到第一个 eos；reward 全等 ⇒ A=0 ⇒ 无学习信号；默认 torch 引擎下 ratio ≡ 1（Gate 题 2 后半）。
6. **bf16 vs fp16**：为什么 bf16 不需要 GradScaler（指数位同 fp32）、fp16 下 scaler 的四步与跳步、GRPO 没有 scaler 在 fp16 下的下溢/溢出两种结局。
7. **显存模型**：能在白板上写出 “参数+梯度+Adam = 16 B/参数 ≈ 1.0 GB”、“logits = B×T×V×bytes”、“激活 ≈ 240 KB/token”，并解释 GRPO 默认配置为何在 16 GB 上危险（三模型 + 21k token + 左 pad 触发 T×T attention）。
8. **mask 错位反推**：拿到两条 loss 曲线和固定 prompt 生成结果，指出哪条是 user 进 loss、哪条是 assistant 全 -100、哪条是丢 eos，并说出各自对应的最小检查（Gate 题 3）。
9. **checkpoint**：`{weight}_768.pth` 是 half 权重；`_resume.pth` 含 model（half）、optimizer、epoch、step、world_size、wandb_id，pretrain/SFT/DPO 额外存 scaler，GRPO 额外存 scheduler；列出三个导致恢复不 bitwise 的原因。

---

## 9. 来源（URL + 核验日期 2026-09-04）

- MiniMind 固定 commit 源码（`已确认`，WebFetch + 本机 curl 逐行阅读）：
  - `model/model_minimind.py` — https://raw.githubusercontent.com/jingyaogong/minimind/7a6fddd63a30c06b2fdd5fac4089922b29bc841b/model/model_minimind.py
  - `dataset/lm_dataset.py` — https://raw.githubusercontent.com/jingyaogong/minimind/7a6fddd63a30c06b2fdd5fac4089922b29bc841b/dataset/lm_dataset.py
  - `trainer/train_dpo.py` — https://raw.githubusercontent.com/jingyaogong/minimind/7a6fddd63a30c06b2fdd5fac4089922b29bc841b/trainer/train_dpo.py
  - `trainer/train_grpo.py` — https://raw.githubusercontent.com/jingyaogong/minimind/7a6fddd63a30c06b2fdd5fac4089922b29bc841b/trainer/train_grpo.py
  - `trainer/trainer_utils.py` — https://raw.githubusercontent.com/jingyaogong/minimind/7a6fddd63a30c06b2fdd5fac4089922b29bc841b/trainer/trainer_utils.py
  - 辅助核验：`trainer/train_pretrain.py`、`trainer/train_full_sft.py`、`trainer/train_lora.py`、`model/model_lora.py`、`trainer/rollout_engine.py`、`model/tokenizer.json`、`model/tokenizer_config.json`（同 commit，同前缀 URL）
- 周卡 `00_WEEK_CARD.md`（2026-09-04 来源审计：数据文件大小、reward 模型文件大小、5070 Ti 需 torch ≥ 2.7 cu128）
- `input_info/minimind_5070ti_v100.md`（用户意图与问题列表；其中 `max_seq_len=768` 为 pretrain 默认的误记，已按源码改为 340）
- PyTorch `torch.amp.GradScaler`（init_scale 65536、growth_factor 2.0、backoff_factor 0.5、growth_interval 2000；`step` 在检测到 inf/nan 时跳过）— https://docs.pytorch.org/docs/stable/amp.html （核验 2026-09-04）
- PyTorch autocast op 列表（`cross_entropy`、`log_softmax`、`softmax` 在 CUDA autocast 下以 fp32 执行）— https://docs.pytorch.org/docs/stable/amp.html#cuda-ops-that-can-autocast-to-float32 （核验 2026-09-04）
- PyTorch `scaled_dot_product_attention`（`is_causal` 与 `attn_mask` 语义；backend 选择）— https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html （核验 2026-09-04）
- DPO 原论文：Rafailov et al., "Direct Preference Optimization: Your Language Model is Secretly a Reward Model", 2023 — https://arxiv.org/abs/2305.18290
- GRPO 原论文：Shao et al., "DeepSeekMath", 2024（组内相对优势与 k3 KL 估计）— https://arxiv.org/abs/2402.03300
- CISPO：MiniMax-M1 技术报告，2025（clipped importance-sampling weight 的 detach 形式）— https://arxiv.org/abs/2506.13585
- RoPE：Su et al., "RoFormer", 2021 — https://arxiv.org/abs/2104.09864
- LoRA：Hu et al., 2021 — https://arxiv.org/abs/2106.09685
- bf16/fp16 格式常量：IEEE 754-2019 与 Google Brain bfloat16 定义（常识级事实，未单独联网核验）
