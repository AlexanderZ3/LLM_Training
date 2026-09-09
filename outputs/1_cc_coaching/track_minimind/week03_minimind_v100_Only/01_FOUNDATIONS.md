# Week M03 Foundations — 一个 64M MiniMind 在 8×V100 上的三本账

> 生成日期：2026-09-08 · 生成方：cc-theory-tutor · 轨道：`track_minimind`
> 对象：MiniMind commit `7a6fddd63a30c06b2fdd5fac4089922b29bc841b`（Apache-2.0）+ PyTorch **v2.1.0** 源码与文档
> 状态：**本文档中没有任何一个数字来自 V100 上的实际运行。** 标 `计算` / `推导` 的数可以在纸上复算；标 `估算` 的数依赖尚未测量的机器参数（β、η、α），Day 0–3 才会有实测列。

## 标签与单位约定（读任何一个数字之前先看这条）

| 标签 | 含义 |
| --- | --- |
| `已确认` | 有官方源码或官方文档的原文支持，本节末尾给出 URL 与核验日期 |
| `计算` | 由 `已确认` 的配置值算出，读者可以用纸笔复算到同一个数字 |
| `推导` | 由 `已确认` 的机制推出的不等式或阈值，推导链在正文里 |
| `推断` | 机制上成立但缺一手原文，给出推断链和替代解释 |
| `估算` | 依赖尚未测量的硬件参数；数量级可信，具体值待实测 |
| `未知` | 不编造；写明由哪一天的哪一步回答 |

单位：本文所有 **MB = 10⁶ 字节**，**GB = 10⁹ 字节**，**MiB = 2²⁰**，**GiB = 2³⁰**。显存容量按厂商标称的 32 GB 理解为 32 GiB = 34.36 × 10⁹ B。这条很重要：`14.75 MB` 与 `14.06 MiB` 是同一个数，混用会让你以为算错了。

数据边界（CLAUDE.md 第 2 节）：本文只给公式、字节数和判据。在 8×V100 上跑出来的日志、profile、checkpoint、拓扑矩阵和绝对性能数字**留在公司内**；能带出来的只有证据字段里的抽象量——比值、布尔、计数、第几步、`PASS/FAIL` 类型。

---

## 0. 本周的三本账是什么，为什么在 V100 上记账比追速度更值得

### 0.1 三本账的定义

同一个 64M MiniMind，从 `$MM_DATA_ROOT/minimind_dataset/pretrain_t2t_mini.jsonl` 的一行 JSON 走到一次 `optimizer.step()`，中间有三样东西是**确定的、可以在纸上算完的**：

| 账 | 记什么 | 单位 | 决定什么 |
| --- | --- | --- | --- |
| **字节账** | 参数 / 梯度 / 优化器状态 / 激活，每张卡各占多少字节 | B | **能不能跑**（OOM 与否），以及 batch 的上限 |
| **误差账** | fp16 相对 fp32 从哪一步开始偏、偏多少、偏的方向 | 相对误差、nats | **学习信号还在不在**（更新是否真的落到参数上） |
| **通信账** | 一个 step 内几次集合通信、什么类型、每次多大、几个 ring step | 次 / B | **跑多快**（通信暴露时间） |

三本账都是**结构量**：只依赖模型配置、并行方式和 dtype，不依赖机器有多快。所以它们可以在 Windows 上、在没有 GPU 的情况下，先算到底。机器决定的只有三个标量：

- $\beta$：某一对 rank 之间的有效带宽（B/s）
- $\alpha$：一次集合通信的固定延迟（s）
- $\eta F$：GEMM 的实际达成算力（FLOP/s）

**这就是本周的方法论：把可算的部分算到底，把不可算的三个标量圈出来，用最少的实验去测这三个标量，再把它们代回账里。** 这跟"跑一跑看看快不快"是两种完全不同的工作。

### 0.2 为什么偏偏在 V100 上做这件事

V100（sm70）给不了速度：没有 BF16、没有 FlashAttention-2、没有 INT8 Tensor Core、torch 锁在 2.1、不能新建环境。但它恰好把三本账里**每一本的约束都顶到明面上**：

- 没有 BF16 → 必须用 `GradScaler` → 误差账里多出"跳步"这一整套机制，而这套机制正是本周三档静默故障里的第二档。在 A100 上你会直接用 bf16，`GradScaler` 删掉，这一档故障**根本不会出现**，你也就永远不会学到它。
- 没有 INT8 Tensor Core → 量化省字节但不省时间 → 逼你把"字节账"和"时间账"当成两本账来记，而不是含糊地说"量化能加速"。
- 只有 32 GB × 8 且 64M 的模型 → 显存**不是**瓶颈 → FSDP 在这里省下的那点字节毫无意义，于是"为什么还要用 FSDP"这个问题只能靠通信账来回答，不能靠"不然装不下"来搪塞。

换句话说：**在一台足够好的机器上，三本账都是隐形的；在 V100 上，三本账全部显形。** 这一周的产出不是一个更快的 MiniMind，而是一套换到 A100/H100/多节点上数字会变、方法不变的记账能力。

### 0.3 本周的三个 worked example

后文所有具体数字都在这三组配置上算。除非另有说明，默认 **WE1**。

| ID | $b$（每卡 micro-batch） | $T$ | $W$ | $A$（accumulation） | 每卡每 micro-batch token 数 $N=bT$ | 全局 batch $G=WAb$ |
| --- | --- | --- | --- | --- | --- | --- |
| **WE0**（MiniMind 默认） | 32 | 340 | 8 | 8 | 10,880 | 2,048 seq |
| **WE1**（本周主线） | 16 | 512 | 8 | 4 | **8,192** | 512 seq |
| **WE2**（单卡 Day 1） | 8 | 512 | 1 | 1 | 4,096 | 8 seq |

WE0 的 $T=340$ 来自 `train_pretrain.py` 的 `--max_seq_len` 默认值（`已确认`）。WE1 把 $T$ 取成 512 是为了让 $N=8192$ 这个数在心算里干净；后文会证明**在 mem-efficient SDPA 后端下，激活的"每 token 字节数"与 $T$ 无关**，所以换回 340 只需把 $N$ 换掉。

### 0.4 与前两周、与后面的关系

本周自包含（`00_WEEK_CARD.md` 第 0 节）：不 import、不依赖 week01/week02 的任何产物。三本账本身是可迁移的：

- 往前：week02 的 `01_FOUNDATIONS.md` 给过 DDP/FSDP/EP 的四种放置方式；本周把其中**只与三本账有关**的部分重算了一遍，并且修正了通信账里一个方向性的结论（见第 4.5 节：FSDP 的**字节数比 DDP 少 25%**，慢的原因不在线上字节，在暴露时间与每-unit 主机侧开销）。
- 往后：`06_QUANT_LOWBIT.md` 的量化模块是**字节账 + 误差账**的一次组合练习（省字节、付误差、在 V100 上不省时间）；`07_RL_LOWPRECISION.md` 的 RL 模块是**字节账 + 通信账**的组合（三个模型同时在卡上、rollout 与 train 的通信不同构）。两个模块在 Gate 之后做。

---

## 1. 对象：MiniMind 64M 的确切配置与参数量推导

### 1.1 配置（`已确认`，`$MINIMIND_ROOT/model/model_minimind.py`，`MiniMindConfig.__init__`）

| 符号 | 字段 | 值 | 来源 |
| --- | --- | --- | --- |
| $d$ | `hidden_size` | 768 | 默认参数 |
| $L$ | `num_hidden_layers` | 8 | 默认参数 |
| $V$ | `vocab_size` | 6400 | `kwargs.get("vocab_size", 6400)` |
| $H$ | `num_attention_heads` | 8 | `kwargs.get(..., 8)` |
| $H_{kv}$ | `num_key_value_heads` | 4 | `kwargs.get(..., 4)`（GQA，$n_{rep}=H/H_{kv}=2$） |
| $d_h$ | `head_dim` | `hidden_size // num_attention_heads` = 96 | 默认表达式 |
| $d_{ff}$ | `intermediate_size` | `math.ceil(768 * math.pi / 64) * 64` | 默认表达式 |
| — | `tie_word_embeddings` | `True` | `kwargs.get(..., True)` |
| — | `dropout` | 0.0 | `kwargs.get("dropout", 0.0)` |
| — | `rms_norm_eps` | 1e-6 | |
| — | `rope_theta` | 1e6 | |
| — | `flash_attn` | `True`（含义是"走 `F.scaled_dot_product_attention`"，不是 FlashAttention 库） | |
| — | `bos_token_id` / `eos_token_id` | 1 / 2 | |
| — | `use_moe` | `False`（本周主线全程 dense，MoE 移出本周） | |

$d_{ff}$ 手算（`计算`）：

$$768\pi = 2412.743\ldots,\quad \frac{2412.743}{64}=37.699\ldots,\quad \lceil 37.699\rceil = 38,\quad 38\times 64 = \mathbf{2432}$$

这个奇怪的表达式的用意是把 SwiGLU 的中间维度定在 $\approx \pi d$（而不是常见的 $\tfrac{8}{3}d$ 或 $4d$），再对齐到 64 的倍数以便 Tensor Core 的 tile 划分。$2432 = 38\times 64$，$2432/768 = 3.1\overline{6}$ —— 后面激活账里 `3.1667` 这个系数就是它。

### 1.2 参数量：一路手算到 63,912,192

所有 `nn.Linear` 都是 `bias=False`（`已确认`，`Attention.__init__` 里四个 `nn.Linear(..., bias=False)`）。`RMSNorm` 只有一个 `weight` 向量，长度等于被归一化的最后一维。

**单层（`MiniMindBlock`）**：

| 子模块 | 形状 | 公式 | 参数量 |
| --- | --- | --- | --- |
| `self_attn.q_proj.weight` | $[H d_h, d] = [768, 768]$ | $d\cdot H d_h$ | 589,824 |
| `self_attn.k_proj.weight` | $[H_{kv} d_h, d] = [384, 768]$ | $d\cdot H_{kv} d_h$ | 294,912 |
| `self_attn.v_proj.weight` | $[384, 768]$ | $d\cdot H_{kv} d_h$ | 294,912 |
| `self_attn.o_proj.weight` | $[768, 768]$ | $H d_h\cdot d$ | 589,824 |
| `self_attn.q_norm.weight` | $[96]$ | $d_h$ | 96 |
| `self_attn.k_norm.weight` | $[96]$ | $d_h$ | 96 |
| `input_layernorm.weight` | $[768]$ | $d$ | 768 |
| `post_attention_layernorm.weight` | $[768]$ | $d$ | 768 |
| `mlp.gate_proj.weight` | $[2432, 768]$ | $d\cdot d_{ff}$ | 1,867,776 |
| `mlp.up_proj.weight` | $[2432, 768]$ | $d\cdot d_{ff}$ | 1,867,776 |
| `mlp.down_proj.weight` | $[768, 2432]$ | $d_{ff}\cdot d$ | 1,867,776 |
| **单层合计 $P_\ell$** | | $4d^2\frac{H_{kv}+H}{2H}+3dd_{ff}+2d+2d_h$ | **7,374,528** |

手算校验（`计算`）：注意力四个矩阵 $589{,}824+294{,}912+294{,}912+589{,}824 = 1{,}769{,}472$；FFN 三个矩阵 $3\times 1{,}867{,}776 = 5{,}603{,}328$；两者之和 $7{,}372{,}800$；再加四个 norm 向量 $768+768+96+96 = 1{,}728$ → $7{,}374{,}528$。

**全模型**：

| 项 | 公式 | 参数量 |
| --- | --- | --- |
| `model.embed_tokens.weight` | $Vd = 6400\times 768$ | 4,915,200 |
| 8 层 | $8P_\ell = 8\times 7{,}374{,}528$ | 58,996,224 |
| `model.norm.weight` | $d$ | 768 |
| `lm_head.weight` | **与 embedding 共享存储**（`tie_word_embeddings=True`） | 0 |
| **合计 $P$** | | **63,912,192** |

$$P = Vd + 8P_\ell + d = 4{,}915{,}200 + 58{,}996{,}224 + 768 = \mathbf{63{,}912{,}192}\approx 63.9\text{M}$$

**不在参数里的东西**：RoPE 的 `freqs_cos` / `freqs_sin` 是 buffer 不是 parameter，不进 `model.parameters()`，因此不进梯度、不进优化器状态、不进 DDP 的桶。如果你手算出来比 63,912,192 大，第一件事就是检查是不是把 buffer 数进去了。

**tie 的三个后果**（`计算` + `推断`）：

1. 参数量少 4,915,200（占 7.7%）。
2. 每卡少 $16\times 4{,}915{,}200 = 78{,}643{,}200$ B $= 78.6$ MB 的 fp32 训练态（param+grad+Adam 两个矩，见第 2 节的 $16P$ 系数）。
3. 这一个张量同时接收来自 embedding 查表和 `lm_head` 矩阵乘的梯度。在 DDP 里它是**一个** parameter、落在**一个** bucket，只 all_reduce 一次，没有重复归约的风险（`推断`：DDP 的 Reducer 按 `module.parameters()` 去重后的列表建桶）。在 FSDP 里共享参数必须落在**同一个 FSDP unit** 里；如果 `auto_wrap_policy` 把 `embed_tokens` 和 `lm_head` 分到不同 unit，FSDP 会在构造时报共享参数错误。MiniMind 的 `embed_tokens` 在 `MiniMindModel` 下、`lm_head` 在 `MiniMindForCausalLM` 下，两者都不在任何 `MiniMindBlock` 里，所以按 block 粒度 wrap 时它们都留在 root unit，天然合法（`推断`，Day 3 用 `FSDP.summon_full_params` 打印 unit 划分来核对）。

### 1.3 一条样本走完全程的 shape 表（WE1：$b=16$, $T=512$）

数据来自 `$MM_DATA_ROOT/minimind_dataset/pretrain_t2t_mini.jsonl`（1,241,043,656 B，sha256 前缀 `6dd6716c84ab3689`，`已确认` 来自 `datasets/DATA_MANIFEST.json`）。

| 阶段 | 张量 | shape | dtype | 元素数 | 字节 |
| --- | --- | --- | --- | --- | --- |
| dataloader | `input_ids` | `[16, 512]` | int64 | 8,192 | 65,536 |
| dataloader | `labels` | `[16, 512]` | int64 | 8,192 | 65,536 |
| embedding | `hidden_states` | `[16, 512, 768]` | **fp32** | 6,291,456 | 25,165,824 |
| 每层 q（proj 后） | `xq` | `[16, 512, 8, 96]` | fp16 | 6,291,456 | 12,582,912 |
| 每层 k/v（proj 后） | `xk`,`xv` | `[16, 512, 4, 96]` | fp16 | 3,145,728 各 | 6,291,456 各 |
| `repeat_kv` 后 | `xk_rep`,`xv_rep` | `[16, 512, 8, 96]` | fp16 | 6,291,456 各 | 12,582,912 各 |
| SDPA 输出 | `output` | `[16, 8, 512, 96]` | fp16 | 6,291,456 | 12,582,912 |
| FFN 中间 | `gate`,`up`,`silu`,`prod` | `[16, 512, 2432]` | fp16 | 19,922,944 各 | 39,845,888 各 |
| 残差流 | `hidden_states` | `[16, 512, 768]` | **fp32** | 6,291,456 | 25,165,824 |
| lm_head | `logits` | `[16, 512, 6400]` | fp16 | 52,428,800 | 104,857,600 |
| 位移后 | `x = logits[..., :-1, :]` | `[16, 511, 6400]` | fp16 | 52,326,400 | 104,652,800 |
| CE 前的 fp32 提升 | — | `[8176, 6400]` | **fp32** | 52,326,400 | 209,305,600 |
| loss | 标量 | `[]` | fp32 | 1 | 4 |

**这张表里有两处反直觉，是本周的两个关键事实**：

1. **残差流是 fp32，不是 fp16。** `nn.Embedding` 不在 autocast 的 fp16 列表上，输出跟权重同 dtype = fp32；`RMSNorm.forward` 写的是 `return (self.weight * self.norm(x.float())).type_as(x)`（`已确认`），`type_as(x)` 把它还原成**输入的** dtype，所以 norm 不会把残差降成 fp16；`h = x + attn_out` 里 x 是 fp32、`attn_out` 是 fp16（`o_proj` 输出），PyTorch 的类型提升给出 fp32。于是残差流从第 1 层到第 8 层全程 fp32。这直接把第 2 节激活账里六个 $N\cdot d$ 张量的单价从 2 B 抬到 4 B。
2. **GQA 在这份实现里省参数、省 KV cache，但不省激活。** `repeat_kv` 的实现是 `x[:, :, :, None, :].expand(...).reshape(...)`（`已确认`）——`reshape` 作用在 `expand` 出来的非连续张量上会**真正物化一份拷贝**。所以进入 SDPA 的 `xk_rep`/`xv_rep` 是完整的 8 头张量，各 12.58 MB，跟没有 GQA 一样。$H_{kv}=4$ 只在 `k_proj`/`v_proj` 的参数量（294,912 而不是 589,824）和推理期 KV cache 上兑现。

### 1.4 训练循环里与三本账直接相关的 8 行（`已确认`，`trainer/train_pretrain.py`）

```python
with autocast_ctx:                                  # torch.cuda.amp.autocast(dtype=dtype)
    res = model(input_ids, labels=labels)
    loss = res.loss + res.aux_loss
    loss = loss / args.accumulation_steps
scaler.scale(loss).backward()                       # 注意：不在 autocast 里，也没有 no_sync()
if step % args.accumulation_steps == 0:
    scaler.unscale_(optimizer)                      # 先 unscale
    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)   # 再 clip，顺序正确
    scaler.step(optimizer); scaler.update()
```

对三本账各自的影响：

- **字节账**：`autocast` 内的 forward 会为每个 `nn.Linear` 的权重生成 fp16 拷贝并缓存（第 2.3 节），这是一笔容易漏记的 127.8 MB。
- **误差账**：`unscale_` → `clip_grad_norm_` → `step` 的顺序是对的。如果 clip 写在 `unscale_` 之前，阈值 1.0 实际作用在 $s\cdot g$ 上，等价于把阈值缩成 $1.0/65536 = 1.53\times 10^{-5}$，梯度会被压扁到几乎为零 —— 这是一个**loss 曲线仍然平滑下降**的静默错误。
- **通信账**：`backward()` 每个 micro-batch 都调用，**没有 `model.no_sync()`**。DDP 因此在**每一个** micro-batch 都做完整的梯度 all_reduce。在 $A=4$（WE1）下通信量是理论必要量的 4 倍，$A=8$（WE0）下是 8 倍。这是第 4 节最重要的一条可执行结论。

**最小片段（只说明"缺 `no_sync` 意味着什么"，完整实现在 `lab/src/mm_v100/` 下的训练入口）**：

```python
sync = (micro_idx == accumulation_steps - 1)          # 只有最后一个 micro-batch 需要同步
ctx = nullcontext() if sync else ddp_model.no_sync()  # 其余 micro-batch 只在本地累积到 param.grad
with ctx:
    scaler.scale(loss / accumulation_steps).backward()
```

---

## 2. 字节账：param / grad / optim / activation，两种精度 × 两种放置

### 2.1 记账的四个格子与一条主公式

设 $P$ = 参数量，$W$ = world_size，$s_p$ = 参数常驻精度的字节数，$s_o$ = 优化器矩的字节数，Adam 有两个矩。

$$\underbrace{s_p P}_{\text{param}} + \underbrace{s_g P}_{\text{grad}} + \underbrace{2 s_o P}_{\text{Adam } m,v} = \text{静态字节}$$

**关键点：在 `torch.cuda.amp.autocast` 下，参数和 `param.grad` 全程是 fp32。** autocast 不改参数的 dtype，它只在算子入口把输入临时 cast 成 fp16；反向产生的梯度会被 cast 回参数的 dtype 再累加到 `param.grad`。所以 $s_p = s_g = s_o = 4$，静态字节 $= 16P$，与是否开 AMP **无关**。

$$16P = 16\times 63{,}912{,}192 = 1{,}022{,}595{,}072\ \text{B} = \mathbf{1.023\ GB}$$

这就是"AMP 不省参数显存"的确切含义。AMP 省的是激活里被 cast 成 fp16 的那部分，以及 GEMM 的时间。

四个常用常数（`计算`，后面反复用）：

| 量 | 值（B） | MB |
| --- | --- | --- |
| $2P$ | 127,824,384 | 127.82 |
| $4P$ | 255,648,768 | 255.65 |
| $8P$ | 511,297,536 | 511.30 |
| $16P$ | 1,022,595,072 | 1022.60 |
| $16P/8$ | 127,824,384 | 127.82 |

最后一行是个便于记忆的巧合：**$W=8$ 时 FSDP 的每卡常驻训练态，恰好等于整模型的一份 fp16 拷贝。**

### 2.2 静态字节：四种组合，$W=8$

| 组合 | param | grad | Adam $m,v$ | DDP bucket | autocast fp16 权重缓存 | **每卡合计** |
| --- | --- | --- | --- | --- | --- | --- |
| **FP32 纯精度 + DDP** | $4P$ = 255.65 | $4P$ = 255.65 | $8P$ = 511.30 | $4P$ = 255.65 | — | **1278.24 MB** |
| **FP16 AMP + DDP** | $4P$ = 255.65 | $4P$ = 255.65 | $8P$ = 511.30 | $4P$ = 255.65 | 127.80 | **1406.05 MB** |
| **FP16 AMP + DDP（`gradient_as_bucket_view=True`）** | 255.65 | （与 bucket 共用） | 511.30 | 255.65 | 127.80 | **1150.40 MB** |
| **FP16 AMP + FSDP `FULL_SHARD`** | $4P/W$ = 31.96 | $4P/W$ = 31.96 | $8P/W$ = 63.91 | — | 见 2.4 瞬时项 | **127.82 MB 常驻 + ~54 MB 瞬时** |

逐项来源：

- **DDP bucket 那一列**：`DistributedDataParallel.__init__` 的 `gradient_as_bucket_view=False` 是默认值（`已确认`，v2.1.0 源码签名）。默认下 Reducer 另外分配一组扁平的 bucket 缓冲区，把每个 `param.grad` **拷贝**进去再 all_reduce，所以梯度相关的显存是 $2\times 4P$。把这个参数改成 `True`，`param.grad` 变成 bucket 的 view，$4P$ 直接省掉。**这是 DDP 路径上最便宜的一笔优化：一个布尔参数换 255.65 MB/卡，占 DDP 静态开销的 18.2%。**
- **autocast fp16 权重缓存**：`torch.autocast(cache_enabled=True)` 的文档把它叫作 "weight cache"（`已确认`，PyTorch 2.1 amp 文档）；被缓存的是**叶子且 requires_grad 的**张量，也就是权重（`推断`，由 "weight cache" 的措辞与"激活是非叶子"这一事实推出）。要缓存的权重是所有会走 `linear` 的：每层 $1{,}769{,}472 + 5{,}603{,}328 = 7{,}372{,}800$，8 层 $58{,}982{,}400$，加 `lm_head` 复用的 $4{,}915{,}200$，共 $63{,}897{,}600$ 个参数 × 2 B = **127,795,200 B = 127.80 MB**（`计算`）。注意 $P - 63{,}897{,}600 = 14{,}592$ 正好是 8 层的 1,728 个 norm 参数加末层 norm 的 768 —— norm 走 fp32 路径，不生成 fp16 拷贝。这些 fp16 权重被 `linear` 的反向节点持有，所以**它们活到 backward 结束**，即使 `autocast` 上下文在 `backward()` 之前就退出了。

### 2.3 激活字节：从 op 的 saved_tensors 一项项数

这一节是字节账里唯一需要逐算子推的部分。规则只有一条：**一个张量占显存，当且仅当它被某个反向节点保存（`saved_tensors`），或者它还活在某条引用链上。** 下表数的是"被保存"的部分；纯瞬时张量另列。

单位 $U = N\cdot d$ 个元素。WE1 下 $U = 8192\times 768 = 6{,}291{,}456$ 元素 → fp16 **12.583 MB**，fp32 **25.166 MB**。

**单层（`MiniMindBlock`）保存表**（`推导`，依据各算子的反向需求；`估算`，因为 PyTorch 的实际保存集合要到 Day 3 用 `torch.cuda.max_memory_allocated()` 对账）：

| # | 张量 | 谁保存它 | 单位 | dtype |
| --- | --- | --- | --- | --- |
| 1 | 层输入残差 `x` | `input_layernorm` 里的 `pow`/`mul` | 1.000 | fp32 |
| 2 | `x * rsqrt(...)` | `weight * y` 的 `mul` | 1.000 | fp32 |
| 3 | q/k/v 三次独立的 fp16 cast | `q_proj`/`k_proj`/`v_proj` 各自 | 3.000 | fp16 |
| 4 | `xq`,`xk`,`xv` | `q_norm`/`k_norm`/`repeat_kv` | 2.000 | fp16 |
| 5 | `q_norm` 的 `x.float()` 与 `y` | 两个 `mul` | 2.000 | fp32 |
| 6 | `k_norm` 的 `x.float()` 与 `y` | 两个 `mul` | 1.000 | fp32 |
| 7 | `q_norm`/`k_norm` 输出 | RoPE 的 `mul` | 1.500 | fp16 |
| 8 | `rotate_half` 的 `cat` 结果 | `* sin` | 1.500 | fp16 |
| 9 | RoPE 输出 `q_rot`,`k_rot` | SDPA | 1.500 | fp16 |
| 10 | `repeat_kv` 物化的 `k_rep`,`v_rep` | SDPA | 2.000 | fp16 |
| 11 | SDPA 输出 | `transpose`/`reshape` 链 | 1.000 | fp16 |
| 12 | `reshape` 的连续化拷贝 | `o_proj` | 1.000 | fp16 |
| 13 | attn 后残差 `h` | `post_attention_layernorm` | 1.000 | fp32 |
| 14 | FFN norm 的 `y` | `weight * y` | 1.000 | fp32 |
| 15 | gate/up 两次 fp16 cast | `gate_proj`/`up_proj` | 2.000 | fp16 |
| 16 | `gate` 输出 | `silu` | 3.1667 | fp16 |
| 17 | `up` 输出 | `mul` | 3.1667 | fp16 |
| 18 | `silu(gate)` | `mul` | 3.1667 | fp16 |
| 19 | `silu(gate)*up` | `down_proj`（已是 fp16，cast 是 no-op） | 3.1667 | fp16 |
| | **fp32 小计** | #1,2,5,6,13,14 | **7.000** | |
| | **fp16 小计** | 其余 | **28.1667** | |

$3.1667 = d_{ff}/d = 2432/768$。SDPA（mem-efficient）额外保存的 `logsumexp` 形状 `[16,8,512]` fp32 = 262,144 B，相对 $U$ 可忽略。`o_proj` / `down_proj` 的输出以及 RoPE 里 fp16×fp32 提升出来的 fp32 中间量都是**瞬时**的（`add` 不保存操作数，`.to()` 不保存输入），只抬峰值不进保存集合。

**每层系数（`计算`）**：

$$\frac{\text{每层激活字节}}{N\cdot d} = 7\times 4 + 28.1667\times 2 = 28 + 56.333 = \mathbf{84.33\ \text{B}}$$

WE1 代入：$84.33\times 6{,}291{,}456 = 530.6$ MB/层，8 层 = **4,244.7 MB**。

**loss 头（这一段单独算，因为它按 $V$ 而不是 $d$ 缩放）**：

| 张量 | shape | dtype | 字节 |
| --- | --- | --- | --- |
| `logits` | `[16,512,6400]` | fp16 | 104,857,600 |
| `logits[..., :-1, :].contiguous()` | `[16,511,6400]` | fp16 | 104,652,800 |
| `cross_entropy` 入口的 fp32 提升 | `[8176,6400]` | fp32 | 209,305,600 |
| `log_softmax` 输出（反向要用） | `[8176,6400]` | fp32 | 209,305,600 |
| **合计** | | | **628,121,600 B = 628.1 MB** |

$$\frac{\text{loss 头字节}}{N\cdot V}\approx 2+2+4+4 = \mathbf{12\ \text{B}}$$

**这里有一个值得停下来看的结果**：把同样的账在纯 fp32 下重算 —— `logits` fp32 209.3 MB + 切片 fp32 209.3 MB + `log_softmax` fp32 209.3 MB = 628.0 MB。**几乎一模一样。** 原因是 `cross_entropy` 在 autocast 的 fp32 列表上（`已确认`），fp16 logits 省下的一半字节，被"为了进 CE 而多做的一次 fp16→fp32 拷贝"精确地吃掉了。**AMP 在 loss 头上一个字节都不省。**

**整合成一个可心算的估计式（`计算` + `估算`）**：

$$\text{Act}(N) \approx N\cdot\big(\underbrace{84.333\times d\times L}_{\text{8 层}} + \underbrace{12\times V}_{\text{loss 头}}\big) = N\times(518{,}144 + 76{,}800) = N\times 594{,}944\ \text{B}$$

（$84.333\times 768\times 8 = 84.333\times 6144 = 518{,}144$；$12\times 6400 = 76{,}800$。这个式子比逐项相加的 $4{,}244{,}635{,}648 + 628{,}121{,}600$ 大约 0.02%，差在 loss 头有两项用的是 $T-1$ 而不是 $T$。）

$$\boxed{\text{每 token 约 }0.595\ \text{MB}=0.567\ \text{MiB 激活}}$$

WE1（$N=8192$）：4.87 GB。WE0（$N=10{,}880$）：6.47 GB。WE2（$N=4096$）：2.44 GB。

**这个系数与 $T$ 无关**（在 mem-efficient SDPA 下），因为表里没有任何 $[B,H,T,T]$ 项。切到 math 后端就不是了，见第 7.3 节。

### 2.4 四种组合的完整每卡账（WE1，$W=8$）

| 组合 | 静态 | 激活 | CUDA context + 分配器碎片 | **每卡总计** |
| --- | --- | --- | --- | --- |
| FP32 + DDP | 1278.2 MB | 6.70 GB（见下） | ~1.0 GB `估算` | **~9.0 GB** |
| FP16 AMP + DDP | 1406.1 MB | 4.87 GB | ~1.0 GB `估算` | **~7.3 GB** |
| FP16 AMP + DDP + `gradient_as_bucket_view` | 1150.4 MB | 4.87 GB | ~1.0 GB | **~7.0 GB** |
| FP16 AMP + FSDP `FULL_SHARD` | 181.9 MB（见下） | 4.87 GB | ~1.0 GB | **~6.1 GB** |

FP32 激活的重算（`计算`）：fp32 模式下没有 autocast 的 cast 拷贝，#3（3 U）与 #15（2 U）这 5 U 消失；剩下的 $28.1667-5 = 23.1667$ U 从 fp16 变成 fp32。系数变成

$$(7 + 23.1667)\times 4 = 30.1667\times 4 = 120.667\ \text{B/(token}\cdot d)$$

每层 $120.667\times 6{,}291{,}456 = 759.2$ MB，8 层 6.073 GB；loss 头在 fp32 下是 209.7 + 209.3 + 209.3 = 628.0 MB（第 2.3 节已说明它与 AMP 下几乎相同）→ 合计 **6.70 GB**，即每 token 818,176 B。**纯 fp32 相对 fp16 AMP，激活只多 37%，不是想当然的 2×** —— 因为 AMP 用五份多余的 cast 拷贝换回了一部分本来能省的字节。

FSDP `FULL_SHARD` 的 181.9 MB 拆解（`计算` + `推断`）：

| 项 | 公式 | MB |
| --- | --- | --- |
| 分片 fp32 master 参数 | $4P/W$ | 31.96 |
| 分片 fp32 Adam $m,v$ | $8P/W$ | 63.91 |
| 分片 fp32 梯度（`keep_low_precision_grads=False`） | $4P/W$ | 31.96 |
| **常驻小计** | $16P/W$ | **127.82** |
| 瞬时：当前 + 预取 unit 的 fp16 全量参数（`BACKWARD_PRE` 默认开） | $2\times 2P_\ell$ | 29.50 |
| 瞬时：root unit 的 fp16 全量参数 | $2\times 4{,}915{,}968$ | 9.83 |
| 瞬时：反向中当前 unit 的 fp16 全量梯度 | $2P_\ell$ | 14.75 |
| **峰值** | | **~181.9** |

### 2.5 本节最重要的一句话

**在 64M 上，FSDP 相对 DDP 省下的是 1278 MB 里的 1150 MB，但每卡总量只从 7.3 GB 降到 6.1 GB —— 因为激活（4.87 GB）不分片，而激活是大头。**

推论（`推导`）：如果你的目标是省显存，这个模型上真正的杠杆是**激活重计算**（`torch.utils.checkpoint`）而不是 FSDP：

- 把 8 个 `MiniMindBlock` 都包上 checkpoint，保存集合从每层 84.33 B/(token·d) 降到只保留层输入 1 U fp32 = 4 B/(token·d)，8 层共 201.3 MB；
- 反向时逐层重算，峰值额外一层的 530.6 MB；
- loss 头 628.1 MB 不变；
- 激活总量 4.87 GB → **约 1.36 GB**，代价是前向多做一遍，约 +33% 计算量（`推导`：fwd:bwd ≈ 1:2，重算把 fwd 做两遍 → 4/3）。

**FSDP 省 1.15 GB，激活重计算省 3.5 GB。这就是"先算账再选工具"的意思。**

### 2.6 Gate A0 闭卷要能写出来的那张表

给定 $(d, L, H, H_{kv}, V, \text{tie}, b, T, W)$，闭卷 30 分钟内要能写出：

```
P      = V*d + L*(4*d*d*(H+H_kv)/(2H) + 3*d*d_ff + 2*d + 2*d_h) + d
静态   = 16P (DDP，每卡) | 16P/W (FSDP FULL_SHARD，每卡)
额外   = +4P  (DDP bucket, gradient_as_bucket_view=False)
       = +2*P_linear (autocast fp16 权重缓存)
激活   = b*T*(84.333*d*L + 12*V)         [fp16 AMP, mem-efficient SDPA, 无重计算]
       = b*T*(120.667*d*L + 12*V)        [纯 fp32]
       = b*T*(4*d*L + 12*V) + 一层峰值   [fp16 AMP + 全层重计算]
       + b*T*(12*H*T*L)                  [若 SDPA 落到 math 后端，见 7.3]
```

---

## 3. 误差账：fp16 的三个数、GradScaler 的四个数、以及哪些运算必须留在 fp32

### 3.1 fp16 的三个数，以及它们各自卡住什么

IEEE 754 binary16：1 符号位 + 5 指数位 + 10 尾数位。

| 名称 | 精确值 | 十进制 | 它卡住什么 |
| --- | --- | --- | --- |
| **最小正规数** | $2^{-14}$ | $6.1035\times 10^{-5}$ | 小于它进入 subnormal，有效位数逐位丢失 |
| **最小 subnormal** | $2^{-24}$ | $5.9605\times 10^{-8}$ | 小于它**直接变 0**（flush to zero） |
| **最大值** | $(2-2^{-10})\cdot 2^{15}$ | $65504$ | 大于它变 $\infty$ |
| **eps（1.0 处的 ULP）** | $2^{-10}$ | $9.7656\times 10^{-4}$ | 两个相邻 fp16 数的间隔 |
| **单位舍入 $u$** | $2^{-11}$ | $4.8828\times 10^{-4}$ | 舍入到最近时的最大相对误差 |

动态范围只有 $65504/5.96\times 10^{-8}\approx 1.1\times 10^{12}$（约 40 个二进制数量级）。对比 bf16：8 位指数，范围与 fp32 相同（$\sim 10^{38}$），但只有 7 位尾数（$u = 2^{-8}=3.9\times 10^{-3}$）。**bf16 用精度换范围，fp16 用范围换精度。深度学习的梯度分布是"范围宽、精度不敏感"的，所以 bf16 更合适 —— 而 V100 没有 bf16，这就是本周误差账所有麻烦的根源。**

四个由这三个数直接推出的阈值（`推导`，每一个都能在 30 秒内复算）：

| 阈值 | 推导 | 后果 |
| --- | --- | --- |
| $\|g\| < 2^{-12} = 2.441\times 10^{-4}$ | $g^2 < 2^{-24}$ | fp16 存 Adam 的 `exp_avg_sq` 时**恰好归零** → $\sqrt{\hat v}=0$ → 更新变成 $m/\varepsilon$ | 
| $\|g\| < 2^{-7} = 7.8125\times 10^{-3}$ | $g^2 < 2^{-14}$ | fp16 的 `exp_avg_sq` 落入 subnormal，有效位数开始掉 |
| $x > \ln 65504 = 11.090$ | $e^x$ 溢出 | 任何 fp16 域内的 `exp`（softmax 未做减最大值时）直接 inf |
| $\bar x > \sqrt{65504/768} = 9.235$ | $d\cdot \bar x^2 > 65504$ | fp16 域内算 RMSNorm 的 $\sum x^2$（$d=768$）溢出 → 这正是 MiniMind 写 `x.float()` 的原因 |

$2^{-12}$ 那一行值得停一下：$(2^{-12})^2 = 2^{-24}$ 是精确等式，不是近似。**任何绝对值小于 $2.44\times 10^{-4}$ 的梯度，在 fp16 的二阶矩里等于没有出现过。** 一个 64M 模型训练中后期的绝大多数梯度分量都在这个量级以下。这是"优化器状态必须 fp32"最硬的一条理由。

### 3.2 为什么 loss 标量也必须是 fp32

未训练模型的 loss 是 $\ln V$（第 5.4 节推）：

$$\ln 6400 = \ln 64 + \ln 100 = 6\ln 2 + \ln 100 = 6\times 0.693147 + 4.605170 = 4.158883 + 4.605170 = \mathbf{8.76405}$$

$8.76405 \in [2^3, 2^4)$，所以它附近的 fp16 ULP 是 $2^3\cdot 2^{-10} = 2^{-7} = 7.8125\times 10^{-3}$。

**在 fp16 里，8.764 和 8.768 是同一个数。** 而本周要做的 fixed-global-batch 等价判定需要分辨 $10^{-4}$ 量级的 loss 差异。所以 loss、`log_softmax`、`cross_entropy` 必须在 fp32 —— 这也正是 PyTorch 把它们放进 autocast fp32 列表的原因（`已确认`）。

顺带记住两个换算锚点：

$$\text{bits} = \frac{\text{nats}}{\ln 2} = \frac{8.76405}{0.693147} = 12.6439 = \log_2 6400,\qquad \text{PPL} = e^{8.76405} = 6400$$

$\ln 2 = 0.693147$ 之所以要背，是因为它是 nats↔bits 的唯一常数；论文里报 bits/byte、代码里算 nats，两者差一个 $\ln 2$，混淆会让你把一个 1.44 倍的差异当成模型问题。

### 3.3 autocast 的分工（`已确认`，PyTorch 2.1 amp 文档）

| 落 fp16 | 落 fp32 | 提升到最宽输入类型 |
| --- | --- | --- |
| `matmul`, `mm`, `bmm`, `addmm`, `baddbmm`, `linear`, `conv*d`, `mv`, `prelu`, RNN/LSTM/GRU cell | `softmax`, `log_softmax`, `cross_entropy`, `nll_loss`, `layer_norm`, `group_norm`, `sum`, `prod`, `cumsum`, `exp`, `log`, `log2`, `log10`, `log1p`, `pow`, `reciprocal`, `rsqrt`, `norm`, `normalize`, `dist`, `cdist`, `sinh`, `cosh`, `tan`, `acos`, `asin`, `erfinv`, 各类 loss | `addcdiv`, `addcmul`, `atan2`, `bilinear`, `cross`, `dot`, `grid_sample`, `index_put`, `scatter_add`, `tensordot` |

映射到 MiniMind 的每一处：

| 位置 | 算子 | 实际 dtype | 说明 |
| --- | --- | --- | --- |
| `embed_tokens` | `embedding` | **fp32** | 不在任何列表上 → 跟权重同 dtype |
| `input_layernorm` / `post_attention_layernorm` | 手写 `pow`+`mean`+`rsqrt`+`mul` | **fp32** | 三个算子都在 fp32 列表；MiniMind 还额外写了 `x.float()` 双保险 |
| `q_proj`/`k_proj`/`v_proj`/`o_proj` | `linear` | **fp16** | 走 V100 的 HMMA Tensor Core |
| `q_norm`/`k_norm` | 同 RMSNorm | **fp32 内部，fp16 输出**（`type_as`） | |
| RoPE | `mul`+`add` | fp16×fp32 提升为 **fp32**，末尾 `.to(q.dtype)` 回 **fp16** | |
| SDPA | mem-efficient 或 math kernel | 输入 fp16，**内部 softmax 累加 fp32**（kernel 自己保证） | |
| `gate/up/down_proj` | `linear` | **fp16** | |
| `act_fn` (SiLU) | `silu` | **fp16** | 不在 fp32 列表；SiLU 的 sigmoid 在 fp16 域内不会溢出（输入 <11.09 时） |
| `lm_head` | `linear` | **fp16** | 输出 `[16,512,6400]` |
| loss | `cross_entropy` | **fp32** | 入口做一次 fp16→fp32 拷贝 |
| 残差 `+` | `add` | **fp32**（类型提升） | |

**结论：MiniMind 里只有 GEMM 和 SiLU 真正在 fp16 域内。所有归约（norm 的平方和、softmax 的 exp 和、CE 的对数和）都在 fp32。** 这是一个设计良好的 AMP 路径；本周不需要改它，需要的是能说清每一处为什么。

### 3.4 GradScaler：四个默认值与跳步机制

`torch.cuda.amp.GradScaler(init_scale=65536.0, growth_factor=2.0, backoff_factor=0.5, growth_interval=2000, enabled=True)`（`已确认`，PyTorch 2.1 amp 文档）。

**为什么 init_scale 偏偏是 $65536 = 2^{16}$**（`推导`，这是本节最值得记住的一段）：

不做缩放时，fp16 能表示的梯度窗口是 $[2^{-24}, 65504] = [5.96\times 10^{-8},\ 6.55\times 10^{4}]$。而真实 LLM 的梯度分量绝大多数落在 $[10^{-9}, 10^{-2}]$ —— **窗口的上半段完全浪费，下半段不够用。** 乘上 $s$ 相当于把整个窗口整体左移 $\log_2 s$ 位：

$$\text{有效窗口} = \left[\frac{2^{-24}}{s},\ \frac{65504}{s}\right] \xrightarrow{s=2^{16}} [9.09\times 10^{-13},\ 0.99951]$$

窗口上界落在 **0.99951**，也就是紧贴 1.0。而 MiniMind 的 `--grad_clip` 默认是 **1.0**（`已确认`）。这不是巧合：$2^{16}$ 这个默认值的设计意图就是让 fp16 的溢出上限恰好落在"单位范数梯度"这个自然尺度上。理解了这一点，你就知道什么时候该动这个默认值：如果你把 `grad_clip` 改成 10，就应该把 `init_scale` 改成 $2^{13}$ 附近。

**跳步机制的完整状态机**（`已确认`，amp 文档 + `GradScaler` 语义）：

```
scaler.scale(loss).backward()         # 反向得到 s·∇ℓ
scaler.unscale_(optimizer)            # param.grad /= s；同时对每个 device 求 found_inf
clip_grad_norm_(params, 1.0)          # 作用在真实梯度上（顺序正确）
scaler.step(optimizer)                # found_inf 为真 → 跳过 optimizer.step()；为假 → 正常 step
scaler.update()                       # found_inf 为真 → s *= 0.5，growth_tracker = 0
                                      # 为假 → growth_tracker += 1；到 2000 → s *= 2，tracker = 0
```

**四个必须记住的行为后果**：

1. **跳步时 `loss.item()` 照常有值。** loss 是前向算出来的，跟优化器步不步没关系。所以**跳步不会在 loss 曲线上留下任何痕迹**，只会让曲线"下降得慢"。
2. **跳步时学习率调度照常推进。** MiniMind 的 `get_lr(epoch*iters + step, ...)` 以 micro-batch 计数为时钟。连续跳 500 步，lr 已经按余弦衰减走了 500 步，但参数一步没动。
3. **跳步后 `optimizer.zero_grad()` 照常执行**（在 `if step % accumulation_steps == 0` 块的末尾），那一组 micro-batch 的梯度被丢弃。
4. **DDP 下各 rank 会一起跳步。** 因为 all_reduce 之后每个 rank 的 `param.grad` 逐位相同，`found_inf` 也就相同（`推断`：NCCL all_reduce 对所有 rank 返回相同结果，这是 DDP 保持副本同步的前提）。**FSDP 下不成立**：每个 rank 只持有自己的分片，`found_inf` 各不相同，普通 `GradScaler` 会让各 rank 分道扬镳。torch 2.1 为此提供 `torch.distributed.fsdp.sharded_grad_scaler.ShardedGradScaler`，它会把 `found_inf` 跨 rank 归约。**在 FSDP 路径上用普通 `GradScaler` 本身就是一个静默 bug。**

**健康 vs 不健康的量化判据**（`推导`）：

- 启动阶段：若初始梯度的最大值使 $s\cdot\max|g| > 65504$，需要 $k=\lceil\log_2(s\max|g|/65504)\rceil$ 次减半，**损失 $k$ 步**。$k$ 通常是 0–5。
- 稳态：每 2000 步试一次翻倍，若翻倍后溢出则再损失 1 步。$N$ 步内因试探损失 $\approx N/2000$ 步。
- **健康跳步率 $\approx (k + N/2000)/N$。$N=10{,}000$、$k=3$ 时约 0.08%。**
- 判据：**跳步率 > 5%，或 scale 连续 50 次尝试单调下降而无一次成功 → 不健康。**
- 从 65536 一路减半到 1 需要 16 次；减到 $2^{-24}$ 需要 40 次。**一个 scale 已经掉到 $10^{-7}$ 的 run 一定是坏的，且不是缩放能救的**——inf 来自前向本身。

**区分"缩放溢出"和"前向 inf"的最小检查**（三行）：

```python
with torch.no_grad():                       # 在 scaler.scale(loss).backward() 之前
    ok_logits = torch.isfinite(res.logits).all().item()   # 前向是否已经 inf/NaN
ok_loss = torch.isfinite(loss).item()
# ok_logits=False → 缩放救不了，是模型/数值问题；ok_logits=True 但持续跳步 → 是 s 太大
```

### 3.5 哪些运算必须留在 fp32 域：五条，每条带阈值

| # | 必须 fp32 的对象 | 阈值推导 | 在 MiniMind 里是谁 |
| --- | --- | --- | --- |
| 1 | **master 参数** | RMSNorm 权重初值 1.0，此处 fp16 ULP $=2^{-10}=9.77\times 10^{-4}$，半 ULP $=4.88\times 10^{-4}$；Adam 的单步更新幅度 $\approx \text{lr}=5\times 10^{-4}$ —— **恰好等于一个半 ULP**。fp16 master 权重下，norm 权重的更新有一半会被舍掉 | 18 个 RMSNorm 权重（8 层×2 + 8 层×2 个 head-dim norm + 末层） |
| 2 | **Adam 二阶矩 `exp_avg_sq`** | $\|g\|<2^{-12}$ ⇒ $g^2$ 归零（3.1 节） | `optimizer.state[p]['exp_avg_sq']` |
| 3 | **归约型算子**（norm 的平方和、softmax 的 exp 和、CE 的对数和） | $d\bar x^2 > 65504$ ⇒ $\bar x > 9.235$；$e^x$ 溢出 ⇒ $x>11.09$ | `RMSNorm._norm` 的 `x.float()`、SDPA kernel 内部、`cross_entropy` |
| 4 | **loss 标量** | 8.764 处 fp16 ULP $=7.81\times 10^{-3}$，粗于等价判定所需的 $10^{-4}$（3.2 节） | `res.loss` |
| 5 | **梯度的跨卡归约（有条件）** | fp16 归约作用在 $s\cdot g$ 上；$W$ 个分量求和溢出的条件是 $s\|g\| > 65504/W$，$s=65536$、$W=8$ 时 $\|g\| > 0.125$。而 `clip_grad_norm_` 是在 `unscale_` **之后**才生效的，归约发生在它之前 | DDP 的 all_reduce 用 fp32 梯度，天然安全；FSDP 的 `MixedPrecision(reduce_dtype=...)` 需要选择 |

第 5 条是 FSDP 路径上一个真实的取舍（`推导`）：

- `reduce_dtype=torch.float16`：reduce_scatter 负载 $2P = 127.82$ MB，但有上面的溢出风险。torch 2.1 的 FSDP 用 pre-divide / post-divide 把除以 $W$ 拆到归约前后来缓解（`推断`，未逐字核验源码）。
- `reduce_dtype=torch.float32`：负载 $4P = 255.65$ MB，每 micro-batch 每卡多 111.85 MB 线上字节（第 4 节），换来完全没有归约溢出。
- **在 V100 上（没有 bf16 这个两全的选项）本周默认取 `reduce_dtype=torch.float32`，把这 111.85 MB 明确记在通信账上，而不是让它变成一个不可见的数值风险。** Day 3 用两个配置各跑一次，比较 loss 前 10 步的相对差。

---

## 4. 通信账：一个 step 内几次、什么类型、多大

### 4.1 记账的三个量

每一次集合通信要记三个数，不是一个：

$$\text{(1) 次数 } n\qquad \text{(2) 每 rank 线上字节 } \text{Bytes}_{\text{rank}}\qquad \text{(3) ring step 数与单步 chunk 大小}$$

ring 算法下（NCCL 在单节点多卡上的默认之一）：

| 类型 | ring step 数 | 每 rank 收发字节（负载 $S$） | 单 ring step 的 chunk |
| --- | --- | --- | --- |
| `all_reduce` | $2(W-1)$ | $\dfrac{2(W-1)}{W}S$ | $S/W$ |
| `all_gather` | $W-1$ | $\dfrac{W-1}{W}S$ | $S/W$ |
| `reduce_scatter` | $W-1$ | $\dfrac{W-1}{W}S$ | $S/W$ |

时间模型（`推导`）：

$$t = \alpha_{\text{launch}} + n_{\text{ring}}\left(\alpha_{\text{link}} + \frac{\text{chunk}}{\beta}\right)$$

$\alpha$ 与 $\beta$ 都是**机器量**，本周 `未知`，Day 0 的 wiring smoke 与 Day 3 的比值实验去圈定它们。字节数和次数是**结构量**，下面全部可以算死。

### 4.2 DDP 的账（$W=8$，fp32 梯度）

**桶的划分**：`bucket_cap_mb=25` 是默认值（`已确认`，v2.1.0 `DistributedDataParallel.__init__` 签名），单位是 MiB → 26,214,400 B。DDP 另有一个更小的首桶常数 `_DEFAULT_FIRST_BUCKET_BYTES` = 1 MiB（`推断`，C++ 侧常量，未逐字核验）。

$$n_{\text{bucket}} = 1 + \left\lceil\frac{255{,}648{,}768 - 1{,}048{,}576}{26{,}214{,}400}\right\rceil = 1 + \lceil 9.712\rceil = 1 + 10 = \mathbf{11}$$

（桶按参数边界切分，实际可能是 10 或 11；用 11 记账，误差 <10%。）

**每 micro-batch 的 backward**：

| 量 | 值 |
| --- | --- |
| all_reduce 次数 | 11 |
| 总负载 | $4P$ = 255.65 MB |
| **每 rank 线上字节** | $\frac{2\times 7}{8}\times 255.65 = \mathbf{447.39\ MB}$ |
| ring step 总数 | $11\times 14 = \mathbf{154}$ |
| 单 ring step chunk | $26{,}214{,}400/8 = 3.28$ MB |

**每 optimizer step**（$A$ 个 micro-batch）：

| 写法 | 集合通信次数 | 每 rank 字节 |
| --- | --- | --- |
| **MiniMind 现状（无 `no_sync()`）** | $11A$ | $447.39A$ MB |
| WE1（$A=4$） | **44** | **1789.6 MB** |
| WE0（$A=8$） | **88** | **3579.1 MB** |
| 加上 `no_sync()`（前 $A-1$ 个 micro-batch） | **11** | **447.39 MB** |

**这是本周通信账里最可执行的一条**（`已确认` 代码事实 + `计算`）：MiniMind 的训练循环没有 `model.no_sync()`，DDP 因此在每个 micro-batch 都做完整归约。加上 `no_sync()` 后，每个 optimizer step 的通信量降到 $1/A$，而**额外显存开销为零**——梯度本来就要累加到已经分配好的 `param.grad` 上。$A=8$ 时这是 8 倍的通信量差异，一行上下文管理器换来的。

### 4.3 FSDP 的账（$W=8$，按 `MiniMindBlock` auto-wrap）

**unit 划分**（`推断`，`ModuleWrapPolicy({MiniMindBlock})`）：8 个 block unit + 1 个 root unit。

| unit | 参数量 | fp16 负载 $S$ |
| --- | --- | --- |
| 每个 `MiniMindBlock` | 7,374,528 | 14,749,056 B = 14.75 MB |
| root（`embed_tokens` 4,915,200 + `model.norm` 768） | 4,915,968 | 9,831,936 B = 9.83 MB |
| **合计** | 63,912,192 | $2P$ = 127.82 MB |

**`FULL_SHARD` + `MixedPrecision(param_dtype=fp16, reduce_dtype=fp16)`，每 micro-batch**：

| 阶段 | 类型 | 次数 | 总负载 | 每 rank 字节 | ring step |
| --- | --- | --- | --- | --- | --- |
| forward（进每个 unit 前 all-gather 参数） | `all_gather` | 9 | 127.82 MB | 111.85 MB | 63 |
| backward（`FULL_SHARD` 前向后已 reshard，需重新 gather） | `all_gather` | 9 | 127.82 MB | 111.85 MB | 63 |
| backward（梯度 reduce-scatter 回分片） | `reduce_scatter` | 9 | 127.82 MB | 111.85 MB | 63 |
| **合计** | | **27** | 383.47 MB | **335.54 MB** | **189** |

单 ring step chunk = $14{,}749{,}056/8 = 1{,}843{,}632$ B = **1.84 MB**（block）。

**三种策略并排**（每 micro-batch，$W=8$）：

| 策略 | 集合通信次数 | 每 rank 字节 | ring steps | 单 chunk | 每卡常驻静态 |
| --- | --- | --- | --- | --- | --- |
| DDP | 11 | 447.39 MB | 154 | 3.28 MB | 1406.1 MB |
| DDP + `no_sync`（$A=4$，摊到每 micro-batch） | 2.75 | 111.85 MB | 38.5 | 3.28 MB | 1406.1 MB |
| FSDP `FULL_SHARD`（reduce fp16） | 27 | 335.54 MB | 189 | 1.84 MB | 127.8 MB |
| FSDP `FULL_SHARD`（reduce fp32） | 27 | 447.39 MB | 189 | 1.84/3.28 MB | 127.8 MB |
| FSDP `SHARD_GRAD_OP`（reduce fp16） | 18 | 223.69 MB | 126 | 1.84 MB | 255.6 MB |

两个精确的比值（`计算`）：

$$\frac{\text{FSDP}_{\text{fp16-reduce}}}{\text{DDP}} = \frac{3\cdot\frac{W-1}{W}\cdot 2P}{2\cdot\frac{W-1}{W}\cdot 4P} = \frac{6P}{8P} = \mathbf{0.75}$$

$$\frac{\text{FSDP}_{\text{fp32-reduce}}}{\text{DDP}} = \frac{2\cdot\frac{W-1}{W}\cdot 2P + 1\cdot\frac{W-1}{W}\cdot 4P}{2\cdot\frac{W-1}{W}\cdot 4P} = \frac{8P}{8P} = \mathbf{1.00}$$

**$(W-1)/W$ 在两边完全抵消 —— 这两个比值与 world_size 无关。** 它们只取决于"FSDP 做三次单向通信、DDP 做一次双向通信"和"两边各自用什么 dtype"。这是通信账里最值得背下来的一个结构性事实。

### 4.4 "为什么 64M 上 FSDP 比 DDP 慢"—— 先排除掉字节量

把上面的数代进时间模型（`推导`）：

$$t_{\text{DDP}} = 154\alpha + \frac{447.39\ \text{MB}}{\beta},\qquad t_{\text{FSDP}} = 189\alpha + \frac{335.54\ \text{MB}}{\beta}$$

FSDP 在**纯线上时间**上更快，当且仅当：

$$189\alpha + \frac{335.54}{\beta} < 154\alpha + \frac{447.39}{\beta}\iff 35\alpha < \frac{111.85\ \text{MB}}{\beta}\iff \alpha < \frac{3.196\ \text{MB}}{\beta}$$

代入两种可能的链路（`估算`）：

| $\beta$ | 临界 $\alpha$ |
| --- | --- |
| 10 GB/s（PCIe Gen3 ×16 的实测量级） | 320 μs |
| 25 GB/s（NVLink 单向的量级） | 128 μs |

单节点 NCCL 的一次集合通信启动延迟通常在 **5–30 μs** 量级（`估算`，Day 0 的 wiring smoke 测）。这比临界值小一个数量级。

> **所以：FSDP 的线上字节比 DDP 少 25%，ring step 只多 23%，按纯线上时间算 FSDP 应该更快。如果实测 FSDP 更慢，慢的一定不是字节量，也不是"次数×延迟"这个乘积本身。**

这跟"FSDP 在 64M 上更慢"的经验并不矛盾，只是把责任定位到了正确的地方。真正的三个来源（`推断`，每一个都对应一个可以关掉的开关）：

**(a) 前向 all-gather 的暴露时间。** `FullyShardedDataParallel.__init__` 里 `forward_prefetch: bool = False` 是**默认值**（`已确认`，v2.1.0 源码签名）。默认下，unit $\ell$ 的参数 all-gather 在进入 unit $\ell$ 的 forward 时才发起，只能与"已经排进队列的前一层计算"重叠，不能提前一整层发起。反向侧 `backward_prefetch = BackwardPrefetch.BACKWARD_PRE` 是默认开的（`已确认`，同一签名），所以反向的 9 次 all-gather有预取，前向的 9 次没有。DDP 完全没有这个问题：它的通信全在反向，且由梯度就绪事件触发，天然与反向计算重叠。

用数字看这一层能不能被藏住（`推导` + `估算`）：

$$t_{\text{ag}}(\text{block}) = \alpha + 7\left(\alpha_{\text{link}} + \frac{1.84\ \text{MB}}{\beta}\right)\approx \frac{12.91\ \text{MB}}{\beta} \xrightarrow{\beta=10\ \text{GB/s}} 1.29\ \text{ms}$$

$$t_{\text{cmp}}(\text{block, forward}) = \frac{2P_\ell N}{\eta F} = \frac{1.475\times 10^{7}\cdot N}{\eta F}\xrightarrow{\eta F = 30\ \text{TFLOPS}} 4.92\times 10^{-7}N\ \text{s}$$

要藏住需要 $4.92\times 10^{-7}N > 1.29\times 10^{-3}$，即 $N > 2{,}600$ token（$\beta=10$ GB/s、$\eta F=30$ TFLOPS 时）。**WE1 的 $N = 8192$ 是它的 3.1 倍，WE2 的 $N=4096$ 是 1.6 倍。也就是说带宽上完全藏得住，藏不住只可能是因为没有预取。** 这把一个模糊的经验变成了一个可证伪的预测和一个可以拧的旋钮：`forward_prefetch=True` 应该消掉大部分差距；如果没消掉，就是 (b) 或 (c)。

**(b) 每-unit 的主机侧开销。** FSDP 每个 unit 每个 step 要做：flat_param 的 unshard/reshard、`_use_unsharded_views` / `_use_sharded_views` 的视图重建（`use_orig_params: bool = False` 是默认值，`已确认`）、per-unit 的 CUDA event record/wait、`limit_all_gathers=True`（默认，`已确认`）带来的 rate limiter 同步。9 个 unit × 3 个阶段 = 27 组这样的簿记，全部在 Python/C++ 主机侧。当每个 block 的 GPU 计算只有几毫秒时，主机侧来不及喂满 GPU。DDP 的簿记只有 11 次桶就绪回调。

**(c) 梯度累积的不对称。** DDP 的 `no_sync()` 免费：梯度本来就累加在 `param.grad` 上，跳过 all_reduce 不需要任何额外内存。FSDP 在 `FULL_SHARD` 下的 `no_sync()` 必须把**未分片的**梯度留在卡上跨 micro-batch，多占 $2P = 127.82$ MB（fp16）或 $4P$（fp32），而且只省掉 9 次 reduce_scatter（27 → 18，省 33%），前向和反向的 all-gather 一次都省不掉。**所以在有梯度累积时，DDP+`no_sync` 的通信优势会被放大 $A$ 倍，而 FSDP 拿不到同样的放大。** $A=4$ 时：DDP+`no_sync` 每 micro-batch 摊 111.85 MB / 2.75 次，FSDP 是 335.54 MB / 27 次 —— 这时候 FSDP 的字节量是 DDP 的 3 倍，次数是 10 倍。**"FSDP 更慢"的经验多半是在有梯度累积的场景下得到的，而这一点在只比较单 micro-batch 时看不出来。**

### 4.5 Day 3 的分离实验（把上面三个来源逐个关掉）

| 实验 | 变量 | 若 step_time 比值显著改善 | 结论指向 |
| --- | --- | --- | --- |
| E1 | `forward_prefetch` False→True | 改善 | (a) 前向暴露 |
| E2 | `FULL_SHARD` → `SHARD_GRAD_OP`（27→18 次） | 改善 | 反向 all-gather 未被藏住 |
| E3 | `reduce_dtype` fp16→fp32（字节 +33%，次数不变） | 变差幅度 ≪ 33% | 不是带宽受限 |
| E4 | $b$ 翻倍 16→32（每 step 常数开销不变，计算翻倍） | 比值向 1 收敛 | (b) 主机侧/延迟类常数开销 |
| E5 | DDP 加 `no_sync` | 改善且与 $A$ 成正比 | (c) 累积不对称 |

**E4 是决定性的**：任何"每 step 固定"的开销（主机侧簿记、集合通信启动延迟）都会被更大的 $b$ 摊薄；任何"与字节成正比"的开销不会。只看一个 batch size 下的比值，无法区分这两者。

可带出公司的证据只有比值与计数：`collectives_total`（27 / 18 / 11）、`step_time_ratio`（FSDP/DDP）、`mem_ratio`。绝对毫秒和 GB 留在内网。

**最小片段（说明"次数"怎么被数出来，完整计数器在 `lab/src/mm_v100/` 的 collectives 计数模块）**：

```python
_orig = dist.all_gather_into_tensor
def counted(out, inp, *a, **kw):                       # 同样的包法用在 reduce_scatter/all_reduce 上
    COUNTER["all_gather"] += 1; COUNTER["bytes"] += inp.numel() * inp.element_size()
    return _orig(out, inp, *a, **kw)
dist.all_gather_into_tensor = counted
```

### 4.6 通信 vs 计算：一个可手算的量级判断

每 token 的训练 FLOPs（`计算`，前向 $2P$、反向 $4P$ 的标准记法）：

| 项 | 前向 FLOP/token | ×3（含反向） |
| --- | --- | --- |
| 8 层参数 GEMM | $2\times 8 P_\ell = 1.180\times 10^8$ | $3.539\times 10^8$ |
| `lm_head` | $2Vd = 9.83\times 10^6$ | $2.95\times 10^7$ |
| 注意力打分（$2\cdot 2 T d$，与参数无关） | $4\times 512\times 768\times 8\ \text{层} = 1.258\times 10^7$ | $3.77\times 10^7$ |
| **合计** | $1.404\times 10^8$ | $\mathbf{4.21\times 10^{8}}$ |

WE1 每 micro-batch 每卡：$8192\times 4.21\times 10^8 = 3.45$ TFLOP。

| $\eta F$ | 计算时间 | DDP 通信时间（$\beta=10$ GB/s） | 通信/计算 |
| --- | --- | --- | --- |
| 30 TFLOPS `估算` | 115 ms | 44.7 ms | 0.39 |
| 70 TFLOPS `估算` | 49 ms | 44.7 ms | 0.91 |

**所以 WE1 的 DDP 恰好落在"通信与计算同量级"的区间**，重叠得好不好直接决定吞吐。这也解释了为什么 `no_sync()` 在这个规模上不是微优化：它把 optimizer step 级别的通信/计算比从 0.39–0.91 降到 0.05–0.11，让重叠质量不再重要。

（V100 SXM2 的 FP16 Tensor Core 峰值约 125 TFLOPS、FP32 约 15.7 TFLOPS —— `已确认`，厂商规格；本周未重新联网核验，沿用 2026-09-04 审计。这里用 30–70 TFLOPS 作为 $d=768$ 规模 GEMM 的达成区间，标 `估算`，Day 3 用 $b$ 扫描反推。）

---

## 5. 数据契约：JSONL → token → label 的每一步，与两个不变量

### 5.1 pretrain 路径逐步（`已确认`，`$MINIMIND_ROOT/dataset/lm_dataset.py`，`PretrainDataset`）

输入文件：`$MM_DATA_ROOT/minimind_dataset/pretrain_t2t_mini.jsonl`，1,241,043,656 B。每行一个 JSON 对象，字段 `text`。

| 步 | 操作 | 输出 | 形状/长度 |
| --- | --- | --- | --- |
| 1 | `json.loads(line)` | `{"text": "..."}` | — |
| 2 | `tokenizer(str(sample['text']), add_special_tokens=False, max_length=T-2, truncation=True).input_ids` | 内容 token 列表 | $\ell_c \le T-2$ |
| 3 | `tokens = [bos] + ids + [eos]` | | $\ell = \ell_c + 2 \le T$ |
| 4 | `input_ids = tokens + [pad] * (T - ℓ)` | 右填充 | $T$ |
| 5 | `labels = input_ids` 的拷贝；`labels[input_ids == pad_token_id] = -100` | | $T$ |
| 6 | collate | `input_ids [b,T] int64`, `labels [b,T] int64` | |
| 7 | 模型内部：`x = logits[..., :-1, :]`, `y = labels[..., 1:]` | `[b,T-1,V]`, `[b,T-1]` | |
| 8 | `F.cross_entropy(x.view(-1,V), y.view(-1), ignore_index=-100)` | fp32 标量 | |

**位移只发生一次，在模型里**（`已确认`，`MiniMindForCausalLM.forward` 的那两行）。数据集**不做**位移。任何在 dataset 或 collate 里再做一次位移，就是双重位移。

### 5.2 两个不变量

**$n_{\text{nonpad}}$**：`(input_ids != pad_token_id).sum()`
**$n_{\text{label\_tokens}}$**：`(labels[:, 1:] != -100).sum()` —— 也就是 `cross_entropy` 的分母

**D1（pretrain，逐条序列）**：

$$n_{\text{label\_tokens}} = n_{\text{nonpad}} - 1$$

推导：非 pad 位置是 $0..\ell-1$，`labels` 在这些位置等于 `input_ids`，其余为 $-100$。位移后被计分的是原下标 $1..\ell-1$，共 $\ell-1 = n_{\text{nonpad}} - 1$ 个。整批则是 $n_{\text{label\_tokens}} = n_{\text{nonpad}} - b$。

**D1 的一个前提必须单独验**（`未知`，Day 0 打印）：若 `pad_token_id == eos_token_id`，第 5 步的掩码会把**真实的 EOS 也一起掩掉**，D1 变成 $n_{\text{nonpad}} - 2$，而且模型**永远学不会停止生成**。MiniMind 的 config 里 `eos_token_id=2`；`pad_token_id` 来自 tokenizer 配置，本周标 `未知`，Day 0 的探针第一件事就是打印这三个 id。

**D2（SFT，逐条序列）**：

$$0 < n_{\text{label\_tokens}} < n_{\text{nonpad}},\qquad \rho = \frac{n_{\text{label\_tokens}}}{n_{\text{nonpad}}}\ \text{在批间应当稳定}$$

`SFTDataset.generate_labels` 的逻辑（`已确认`）：labels 全部初始化为 $-100$；扫描 `input_ids`，找到 assistant 起始标记 `bos_id` 的 token 序列，从 `start = i + len(bos_id)` 一直标到第一个 `eos_id` 标记结束（含 eos 本身），把这些位置的 label 设回 `input_ids[j]`。也就是说**只有 assistant 的内容和结束标记被计分，system/user 和模板本身不计分**。

D2 的两个端点各对应一种故障：

- $\rho = 1.0$ → 掩码根本没生效，prompt 也在被训练。loss 会比正常低（prompt 里有大量模板 token，极易预测）。
- $\rho = 0$ → 标记匹配失败（换了 tokenizer、改了模板）。此时 `cross_entropy` 的分母为 0，**返回 NaN**（`推断`：`nll_loss` 的 mean reduction 除以 `total_weight = 0`）。这一档反而是响的，好办。

**D3（位移一致性，两条断言就能查完）**：

```python
assert (labels[labels != -100] == input_ids[labels != -100]).all()   # labels 未被额外位移
assert (labels[input_ids == pad_id] == -100).all()                    # pad 全部被掩掉
```

第一条是关键：它把"labels 是 input_ids 的逐位置拷贝（只是部分被掩）"这个契约钉死。数据侧任何位移都会让它失败。**这两行不需要 GPU、不需要模型、不需要跑一步训练**，在 Windows 上就能跑。

### 5.3 loss mask 错位为什么不会让 loss 曲线变难看

这是本周整个方法论的核心，值得展开成一条原理加三个算例。

**原理**：交叉熵对**任何一个定义一致的目标**都是良定的可优化目标。SGD 最小化的是你实际写下来的那个东西。曲线的**形状**由可优化性决定（平滑、单调下降），曲线的**渐近值**由你选中的那个目标的条件熵决定。而你对"正确目标的条件熵应该是多少"通常没有独立估计。

$$\text{曲线好看} \Longleftarrow \text{目标可优化} \not\Longrightarrow \text{目标正确}$$

**算例 A：双重位移。** 数据集也做了一次位移，模型内部又做一次 → 实际在最小化 $H(x_{t+2}\mid x_{\le t})$ 而不是 $H(x_{t+1}\mid x_{\le t})$。跳一格预测同样是可学的（语言里有跨一位的统计规律），曲线一样平滑单调，只是收敛到一个更高的平台。高多少？对自然语言大致是 0.5–1.5 nats（`估算`，没有本数据集上的实测）。而你并不知道"正确的平台"应该在哪里，所以看不出来。

**算例 B：掩码差一位。** 若在旧版 `loss_mask` 接口下忘了 `loss_mask[:, 1:]`，计分位置整体偏移一格：assistant 的第一个 token 不被计分，eos 之后多计一个 token。$n_{\text{label\_tokens}}$ **数量完全不变**，只是集合平移。收敛值几乎不动。**这一档 D1/D2 都查不出来**，只有 D3 的第一条断言或生成测试能抓住。

**算例 C：pad 没被掩掉（`labels = input_ids` 直接用）。** 这一档最危险，因为它让 loss **看起来更好**。设一批里被计分位置中 pad 占比 $f$，pad→pad 是确定性的，训几百步后 $L_{\text{pad}}\approx 0.005$：

$$L_{\text{report}} = (1-f)L_{\text{real}} + f L_{\text{pad}}$$

WE1 里 $T=512$、真实平均长度取 200 token → $f = (511-199)/511 = 0.611$。若真实语言建模 loss $L_{\text{real}} = 3.0$：

$$L_{\text{report}} = 0.389\times 3.0 + 0.611\times 0.005 = 1.167 + 0.003 = \mathbf{1.17}$$

报出来的困惑度 $e^{1.17} = 3.2$，真实的是 $e^{3.0} = 20.1$。**一个 64M 模型在中文网页文本上报出 PPL 3.2，只要你知道该期待什么，这就是刺眼的；问题是很多人不知道该期待什么。**

### 5.4 该期待什么：四个可计算的锚点

| 锚点 | 值 | 怎么得到 | 用途 |
| --- | --- | --- | --- |
| **均匀分布** | $\ln V = \ln 6400 = \mathbf{8.7641}$ nats = 12.644 bits，PPL = 6400 | 手算（3.2 节） | **第 0 步的 loss 必须落在这里**（±0.05）。偏离说明初始化、vocab、tie 或 logits 尺度有问题 |
| **unigram** | $H(\hat p_{\text{token}})$ | 对数据流做一遍 token 频次统计 | 任何"在学东西"的模型必须低于它 |
| **bigram** | $H(x_{t+1}\mid x_t)$ | 一遍二元频次统计 | **早期（<几百步）就低于它 → 怀疑标签泄漏/pad 污染，而不是模型好** |
| **0** | — | — | 只有泄漏能做到 |

前两个锚点里 unigram 与 bigram 熵**是可以从数据算出来的确定数**，不是猜的。它们把"loss 是多少算正常"从经验判断变成了计算。Day 1 的一个产物就是在 `pretrain_t2t_mini.jsonl` 的一个固定子集上算出这两个数，写进 `$MM_RUNS_ROOT/day1_pretrain/numeric_ref.json`。

**第 0 步 loss = 8.764 这个检查非常强，但它检查不到掩码。** 随机初始化下不论你掩掉哪些位置，剩下位置的 loss 都是 $\ln V$。所以：$\ln V$ 检查的是「模型接线」，D1/D2/D3 检查的是「数据接线」，两者不可互相替代。

### 5.5 为什么"过拟合到 0"也检查不出位移

一个常见的错误做法是：拿一条序列反复训练，看 loss 能不能降到 0，以此证明训练管线是对的。

**这个检查对位移故障完全无效。** 任何"从前缀确定性地映射到某个 token"的任务都是可记忆的：预测 $x_{t+1}$ 是这样，预测 $x_{t+2}$ 也是这样。双重位移的模型同样能把那条序列的 loss 压到 0。

**有效的行为级检查是生成。** 过拟合一条序列后，用它的前 10 个 token 做贪心解码：

- 接线正确 → 逐 token 复现原序列；
- 位移错一格 → 复现的是原序列**偏移一格**的版本（跳过或重复一个 token）。

这是一个二值、无歧义、不需要参考曲线的判据。**规律：曲线类检查判优化，张量断言与行为检查判正确性。**

---

## 6. 三档静默故障的机理：不变量 → 唯一识别性 → 最小检查

本周要能只凭日志里的几个数把三档故障分开。先把要打的量定义清楚。

### 6.1 五个可记录量

| ID | 量 | 定义 | 每步开销 |
| --- | --- | --- | --- |
| **I1** | 数据契约比 | $(n_{\text{label\_tokens}},\ n_{\text{nonpad}},\ \rho)$ | 两次 `sum()`，可忽略 |
| **I2** | 缩放器状态 | `scaler.get_scale()` 与「已落地的 optimizer step 数 / 尝试数」 | 一次读取 |
| **I3** | 跨 rank 参数一致性 | $\max_r h_r - \min_r h_r$，其中 $h_r=\sum_p \|p\|_2^{(\text{fp64})}$ | 一次 fp64 归约 + 一次 all_gather 标量 |
| **I4** | 跨时间参数移动 | $h(\theta_t) - h(\theta_{t-k})$ | 复用 I3 的 $h$ |
| **I5** | 跨 rank loss 差异 | $\max_r L_r - \min_r L_r$ | 一次 all_gather 标量 |

$h_r$ 用 **fp64** 算是必须的（`推导`）：对 $6.4\times 10^7$ 个数做 fp32 求和，相对误差约 $\sqrt{n}\cdot 2^{-24} = 8000\times 5.96\times 10^{-8} = 4.8\times 10^{-4}$；这个噪声会淹没真正的小差异，把 I3 变成一个假阴性机器。逐参数张量的 fp64 二范数还有第二个好处：定位到具体是哪个张量偏了。

> **实现说明：lab 里的 I3 比的是梯度，而且是逐元素（`本机实测`，2026-09-08）。**
>
> 上面这个基于范数的定义在数学上没问题，但构建期在本机做故障注入时发现它**抓不到故障 C**：不同 micro-batch 的梯度**方向**不同，**范数**却往往几乎相等——小模型加随机数据上实测各 rank 只差 3.58e-07，落在任何合理阈值之下。
>
> 根源是：**范数是一个把方向信息全部丢掉的标量，而故障 C 恰恰只改变方向。** 提高精度（fp32 到 fp64）救不了这一点，因为问题不是数值噪声，是这个统计量本身对要检测的差异不敏感。
>
> 所以 `lab/src/mm_v100/faults.py` 的 `cross_rank_grad_delta()` 改成 all_gather 一段**梯度指纹**（前若干个参数张量各取前若干个元素）之后**逐元素**比，返回相对偏差 `max|g_i - g_0| / mean|g_0|`。健康 DDP 下它**精确为 0.0**——因为 all_reduce 的契约就是归约后各 rank 拿到逐位相同的 buffer。
>
> 这里有一条比这个不变量本身更值得记的教训：**先问这个统计量对你要检测的差异敏不敏感，再问它的数值精度够不够。** 一个不敏感的量，算得再精确也还是个假阴性机器。

### 6.2 三档故障的真值表

| | I1 数据比 | I2 缩放/落地率 | I3 跨 rank 差 | I4 跨时间移动 | I5 跨 rank loss 差 |
| --- | --- | --- | --- | --- | --- |
| **正常** | 符合 D1/D2 | 落地率 >99% | $=0$ | $\ne 0$ | $\ne 0$ |
| **A：label mask 错位** | **异常**（或 D3 断言失败） | 正常 | $=0$ | $\ne 0$ | $\ne 0$ |
| **B：GradScaler 持续跳步** | 正常 | **落地率 ≈ 0，scale 单调下降** | $=0$ | $\mathbf{=0}$ | $\ne 0$ |
| **C：DDP 梯度未同步** | 正常 | 正常或轻微异常 | $\mathbf{\ne 0}$ | $\ne 0$ | $\ne 0$ |

三档的日志表现（这是它们"静默"的原因）：

| | loss 曲线长什么样 | 有没有报错 |
| --- | --- | --- |
| A | 平滑单调下降，只是平台高一点或低一点 | 无 |
| B（全跳） | **平坦**，在中断时刻的值附近抖动 | 无 |
| B（半跳） | 平滑下降，只是**慢一半到两倍半** | 无 |
| C | 平滑下降，rank 0 上看起来完全正常 | 无 |

### 6.3 A：label mask 错位

**机理**：第 5.3 节的三个算例。四种常见触发：dataset 侧多做一次位移；`loss_mask` 忘了切 `[:, 1:]`；`ignore_index` 用了 0 而不是 $-100$（把 token id 0 也掩掉）；SFT 的 prompt 未掩。

**为什么 I1 能唯一识别它**：I1 是**数据张量本身的性质**，在模型前向之前就能算出来。B 只动 `GradScaler`，C 只动梯度通信，两者都不触碰 dataloader 的输出。反过来，A 不改变 scale 轨迹，也不破坏跨 rank 一致性。**I1 与 I2/I3 在因果上完全正交**，这就是唯一识别性的来源。

**最小检查（CPU，不需要 GPU、不需要模型）**：

```python
x, y = next(iter(loader))                                    # [b,T] int64 两个张量
n_nonpad = (x != pad_id).sum().item(); n_lab = (y[:, 1:] != -100).sum().item()
assert (y[y != -100] == x[y != -100]).all(), "labels 被额外位移了"
assert (y[x == pad_id] == -100).all(), "pad 未被掩掉"
print(n_lab, n_nonpad, n_lab / n_nonpad)                     # pretrain: n_lab == n_nonpad - b
```

**行为级复核（需要 GPU，5 分钟）**：过拟合一条序列到 loss < 0.05，用前 10 token 贪心解码，逐 token 比对（第 5.5 节）。

### 6.4 B：GradScaler 持续跳步

**机理**：每一个 optimizer step 尝试都检出 inf/NaN。两种来源必须分开：

1. **缩放溢出**：$s\cdot\max|g| > 65504$。减半几次就解决，`scale` 会稳定在某个 $2^k$ 上。**这是健康的自适应，不是故障。**
2. **前向本身就产生 inf/NaN**：不论 $s$ 多小，`found_inf` 恒真 → `scale` 从 65536 一路单调减半，永不恢复。16 步到 1，40 步到 $2^{-24}$。

**后果**：`optimizer.step()` 一次都不执行 → 参数冻结。但是 `loss.item()` 照常打印（前向照做）、`lr` 照常按调度衰减、日志格式完全正常。

**为什么 (I2 异常, I4 = 0) 这一对是充分的**：
- I4 $= 0$ 的意思是"参数一步没动"。能造成这个的只有三类：优化器没被调用（B）、lr 恒为 0、参数被 `requires_grad_(False)` 冻结。
- 后两类的 I2 是**正常**的（scale 稳定在 65536 且落地率 100%）。
- 所以 (I2 异常 ∧ I4 = 0) 排除了后两类，唯一剩下 B。**充分。**

**为什么 I2 异常单独只是必要不充分**：健康 run 在启动阶段本来就会连续减半 $k$ 次（3.4 节），每 2000 步的翻倍试探也会偶发一次跳步。**"scale 下降过"是 B 的必要条件，不是充分条件。** 把这个当成判据会在每一个正常 run 上误报。

**为什么 I4 = 0 单独也只是必要不充分**：见上面的 lr=0 和冻结两种情况。

**最小检查**：

```python
before = h_params()                                      # fp64 参数校验和，见 6.1
scaler.step(optimizer); scaler.update()
moved = abs(h_params() - before) > 0                     # I4：这一步参数动没动
landed_ratio = n_landed / n_attempted                    # I2：滑动窗口内的落地率
finite_fwd = torch.isfinite(res.logits).all().item()     # 区分"前向 inf" vs "缩放溢出"
```

判据（`推导`，阈值来自 3.4 节）：**落地率 < 95% 报警；连续 50 次尝试落地率 = 0 且 scale 单调下降 → 判定 B；此时 `finite_fwd == False` 说明是前向 inf（缩放救不了），`True` 说明是纯缩放问题（应能自愈，若不自愈则是 `unscale_`/`clip` 顺序或自定义 hook 的问题）。**

### 6.5 C：DDP 梯度未同步

**机理**：四种常见触发（`推断`）：

1. 前向走的是 `model.module(...)` 或未包装的原模型 → DDP 的 autograd hook 根本不注册在这条图上 → 没有 all_reduce。
2. `no_sync()` 的上下文覆盖了累积组的**最后一个** micro-batch。
3. rank 间控制流分歧（某个 rank 因为数据条件跳过了 backward）。
4. 参数在某些 step 上没有梯度而 `find_unused_parameters=False` → 桶永远等不到就绪。

后果：每个 rank 独立优化自己那 $1/W$ 的数据。有效全局 batch 从 $Wb$ 塌到 $b$。rank 0 保存的 checkpoint 是一个只见过 $1/8$ token 的模型。**rank 0 的 loss 曲线完全正常**——它确实在正确地拟合它看到的数据。

**为什么 I3 ≠ 0 是充分的**：DDP 的核心契约是"归约后每个 rank 的梯度逐位相同，因此参数保持逐位相同"（`已确认`，PyTorch 2.1 DDP notes：构造时从 rank 0 广播 `state_dict`，此后靠梯度平均保持同步）。`dropout=0.0`，优化器是逐元素确定性的。所以在没有 C 的情况下 I3 恒等于 0。**观察到 I3 ≠ 0 ⇒ 一定是 C。充分。**

**为什么 I3 ≠ 0 不是必要的**：如果各 rank 恰好拿到**相同的数据**（`DistributedSampler` 没生效、或者忘了传 `sampler`、或者 `set_epoch` 的问题），那么即使没有同步，各 rank 的梯度也相同 → I3 = 0，而 C 依然成立。这一种要靠 I5 补：

$$\text{I5} = 0\ \text{（各 rank loss 完全相同）} \Longrightarrow \text{各 rank 在看同一批数据} \Longrightarrow \text{采样器坏了}$$

所以完整的判据是一对：**(I3 ≠ 0) 充分判定 C；(I3 = 0 ∧ I5 = 0) 判定"采样器未分片"这第四种故障，此时是否同步已经无从区分。**

**最小检查**（放在第一个 optimizer step 之后，跑一次即可）：

```python
g = torch.stack([p.grad.detach().double().norm() for p in model.parameters()]).norm()
buf = [torch.zeros_like(g) for _ in range(world_size)]; dist.all_gather(buf, g)
assert (max(buf) - min(buf)).item() == 0.0, "梯度未跨 rank 同步"       # 查梯度，隔离掉优化器
L = torch.tensor([loss.item()], device=g.device); Lb = [torch.zeros_like(L) for _ in range(world_size)]
dist.all_gather(Lb, L); assert (max(Lb) - min(Lb)).item() != 0.0, "各 rank 数据相同，采样器未分片"
```

查**梯度**而不是参数，是为了把 DDP 的问题和优化器的问题分开：梯度相同而参数不同，说明优化器状态或 lr 在 rank 间不一致（另一类故障）。

### 6.6 Gate 要求的"充分 / 必要"总表

| 特征对 | 关系 | 理由 |
| --- | --- | --- |
| I1 异常（或 D3 断言失败） | **充分**判 A | I1 直接测量出问题的那个对象本身 |
| (I2 异常 ∧ I4 = 0) | **充分**判 B | I4=0 的另外两种解释（lr=0、冻结）会让 I2 正常 |
| I2 异常（单独） | **仅必要** | 健康启动阶段与每 2000 步的翻倍试探都会让 scale 下降 |
| I4 = 0（单独） | **仅必要** | lr=0、参数冻结同样给 I4=0 |
| I3 ≠ 0 | **充分**判 C | DDP 的同步契约保证无 C 时 I3 恒为 0 |
| I3 = 0 | **不能**排除 C | 各 rank 拿到相同数据时 C 也给 I3=0 |
| (I3 = 0 ∧ I5 = 0) | **充分**判"采样器未分片" | 各 rank loss 逐位相同只可能是同一批数据 |
| loss 曲线形状 | **对三档都无判别力** | 第 5.3 节与 6.2 节 |

---

## 7. V100 sm70 的硬件边界，以及每条边界卡住哪一本账

### 7.1 边界清单

| # | 边界 | 事实与来源 | 卡住哪本账 |
| --- | --- | --- | --- |
| 1 | **有 FP16 Tensor Core（HMMA，FP32 累加）** | Volta 架构；SXM2 峰值约 125 TFLOPS FP16 TC / 15.7 TFLOPS FP32（`已确认`，厂商规格，沿用 2026-09-04 审计） | 时间账：fp16 相对 fp32 理论 8× |
| 2 | **无 BF16** | `torch.cuda.is_bf16_supported()` 的实现要求 `get_device_properties(...).major >= 8`；V100 是 7.0（`已确认`，PyTorch v2.1.0 `torch/cuda/__init__.py` 原文） | **误差账** |
| 3 | **无 INT8 Tensor Core（IMMA）** | 整数 MMA 从 Turing/sm75 开始（`已确认`，架构代际事实，逐条 URL 见 `06_QUANT_LOWBIT.md` 第 1 节） | **字节账 ↛ 时间账的转换** |
| 4 | **SDPA 无 flash 后端** | torch 2.1 的 flash 后端不覆盖 sm70，落 mem-efficient 或 math（`已确认`，`00_WEEK_CARD.md` 3.1，2026-09-04 审计） | **激活账** |
| 5 | **32 GB HBM2 / 约 900 GB/s** | 厂商规格（`已确认`，同 #1） | 带宽账 |
| 6 | **卡间拓扑（NVLink/PCIe 混合）** | `未知` —— Day 0 探针 | **通信账的 $\beta$** |
| 7 | **torch 锁 2.1，不新建环境** | 用户约束（`已确认`，`00_WEEK_CARD.md` 第 8 节） | 全部：无 FlashAttention-2、无 DeepEP、无 MegaBlocks、`torch.compile` 关闭 |

### 7.2 边界 #2（无 BF16）如何卡住误差账

fp16 和 bf16 的差别只有一处：指数位 5 vs 8。

| | 指数位 | 尾数位 | 动态范围 | 单位舍入 $u$ |
| --- | --- | --- | --- | --- |
| fp16 | 5 | 10 | $[6\times 10^{-8},\ 6.6\times 10^{4}]$ | $2^{-11}=4.9\times 10^{-4}$ |
| bf16 | 8 | 7 | $[1\times 10^{-38},\ 3\times 10^{38}]$ | $2^{-8}=3.9\times 10^{-3}$ |
| fp32 | 8 | 23 | $[1\times 10^{-38},\ 3\times 10^{38}]$ | $2^{-24}=6.0\times 10^{-8}$ |

梯度分布是"跨越很多数量级、但每个分量不需要很高精度"的。bf16 的范围与 fp32 相同，所以**根本不需要 loss scaling** —— `GradScaler` 在 bf16 路径上是关掉的（MiniMind 的 `GradScaler(enabled=(args.dtype == 'float16'))` 就是这个意思，`已确认`）。

**因果链**：V100 无 bf16 → 必须 fp16 → 必须 loss scaling → 引入 `GradScaler` 的跳步状态机 → 引入"参数悄悄不更新"这一整类故障（第 6.4 节的 B 档）→ 引入"必须记录落地率而不是只看 loss"这条纪律。

**在 A100 上把 `--dtype` 改成 `bfloat16`，B 档故障会整个消失。** 这就是"V100 把误差账顶到明面上"的确切含义：它逼你把一个在更好的硬件上被自动绕开的问题，亲手处理一遍。

### 7.3 边界 #4（无 flash）如何卡住激活账

三个后端的差别在于**是否物化 $[b,H,T,T]$**：

| 后端 | 是否物化注意力矩阵 | 是否可用于 sm70 | 每层额外保存 |
| --- | --- | --- | --- |
| flash | 否 | **否** | — |
| mem-efficient | 否（分块） | 是 | `logsumexp [b,H,T]` fp32，可忽略 |
| math | **是** | 是 | scores 与 softmax 输出，且 softmax 在 autocast 下走 fp32 |

math 后端的额外字节（`计算`）：$[b,H,T,T]$ 有 $bHT^2$ 个元素，摊到每个 token（$N=bT$）是 $HT$ 个元素。保存两个这样的张量（scores 与 softmax 输出），且 autocast 把 softmax 提到 fp32，粗算 fp16 两份 + fp32 两份：

$$\Delta \approx H T(2+2+4+4) = 12HT\ \text{B/token/层}\xrightarrow{H=8,T=512} 49{,}152\ \text{B/token/层}\times 8 = 393{,}216\ \text{B/token}$$

对比第 2.3 节的 594,739 B/token —— **math 后端几乎把激活翻一倍，而且这一项 $\propto T$**（原来的系数与 $T$ 无关）。WE1 下激活从 4.87 GB 涨到约 8.09 GB（`估算`）。

**这条边界的实际后果**：math 后端是本周唯一的**数值 reference**（它逐步计算，可以逐张量对比），但它把激活账压到近两倍且随 $T$ 线性增长。所以 Day 1 的 FP32 reference 只能在小 $b$、小 $T$ 上做（WE2 或更小），不能在 WE1 上做。**这是"边界卡住账本"最直接的一次体现：不是不能跑，而是 reference 实验的规模被硬件边界限死了。**

### 7.4 边界 #3（无 IMMA）如何把字节账和时间账切开

V100 的 Tensor Core 只接受 FP16 输入。INT8 的 GEMM 没有硬件路径，只能：

```
INT8 权重 --(反量化)--> FP16 权重 --(HMMA)--> FP32 累加
```

于是 INT8 权重量化在 V100 上的收益是：

| 账 | 收益 |
| --- | --- |
| **字节账** | 权重从 $2P$ 降到 $P$（+ per-channel scale），**真实收益** |
| **带宽账** | 从 HBM 读权重的字节减半，**真实收益**（对访存受限的解码尤其明显） |
| **时间账** | **零，甚至为负**（多了一次反量化 kernel） |

**在 A100/H100 上第三行会变成正的（INT8 TC 有 2× 于 FP16 的吞吐）。所以"量化能加速"是一句依赖硬件代际的话，不是一句普适的话。** 本周把量化还原成它本来的两件事——一个误差预算问题、一个内存带宽问题——正是因为在 V100 上第三条路被硬件堵死了，剩下两条反而能被干净地测量。细节在 `06_QUANT_LOWBIT.md`。

### 7.5 边界 #6（拓扑未知）如何卡住通信账

第 4 节的所有字节数和次数都是**确定的**；唯一不确定的是 $\alpha$ 和 $\beta$。8 卡 V100 服务器常见三种拓扑：全 NVLink 立方网、双 NVLink 组 + PCIe 跨组、纯 PCIe。它们的 $\beta$ 可以差 3–5 倍，而且**同一台机器上不同 rank 对之间的 $\beta$ 可能不同**（混合拓扑）。

后果：ring 集合通信的速度由环上**最慢的一条链路**决定。所以 `new_group` 的 rank 顺序在混合拓扑上是有意义的。

本周对此的处理（`已确认`，`00_WEEK_CARD.md` 第 8 节的数据边界）：拓扑矩阵本身是公司资产，不带出。**只记比值**：同一个集合通信在两种 rank 分组下的耗时比、是否 > 1.2。这样既回答了问题又不外泄。

### 7.6 一句话总结这一节

| 边界 | 它把哪本账从"隐形"变成"必须记" |
| --- | --- |
| 无 BF16 | 误差账（loss scaling 与跳步落地率） |
| 无 INT8 TC | 字节账与时间账必须分开记 |
| 无 flash | 激活账（后端选择直接改变每 token 系数与它对 $T$ 的依赖） |
| 拓扑未知 | 通信账的字节数可算、时间不可算 → 只能用比值做结论 |

---

## 8. checkpoint 与恢复：三种 StateDictType、8→4 的 step 换算、loss 连续的判据

### 8.1 三种 `StateDictType`（`已确认`，PyTorch 2.1 FSDP 文档）

| 类型 | 文档定义 | 张量形态 | 能换 world_size 恢复吗 | 保存代价 |
| --- | --- | --- | --- | --- |
| `FULL_STATE_DICT` | "all states are unflattened and not sharded" | 完整的、原始 FQN 的普通张量 | **能，换成任意 $W'$** | 每个 unit 一次 all_gather；配 `FullStateDictConfig(offload_to_cpu=True, rank0_only=True)` 时只有 rank 0 物化整模型 |
| `SHARDED_STATE_DICT` | "all states are unflattened but sharded" | 按原始 FQN 组织的分片张量（带全局 shape 与 offset） | **能**，配 `torch.distributed.checkpoint.save_state_dict` / `load_state_dict` 做 reshard | 无 all_gather，各 rank 写自己那份 |
| `LOCAL_STATE_DICT` | "no transformation will be performed" | 原始的 `flat_param` 分片，key 是 `_flat_param` | **不能**，必须同 $W$ 同 wrap 结构 | 最快 |

对 64M 的选择（`推导`）：`FULL_STATE_DICT` 的 rank 0 峰值只有 $4P = 255.65$ MB，完全放得下，恢复逻辑最简单。**但本周要练的正是 `SHARDED_STATE_DICT` + `torch.distributed.checkpoint` 的 reshard 路径**，因为它是唯一能扩展到 7B/27B 的路径（在 27B 上 `FULL_STATE_DICT` 的 rank 0 峰值是 108 GB，装不下）。用一个 64M 模型练一条为 27B 准备的路径 —— 这就是本周的性价比所在。

优化器侧（`已确认`，同一文档）：`FSDP.optim_state_dict(model, optim)` 把优化器状态从"扁平索引"转成"按参数 FQN 组织"，`FSDP.optim_state_dict_to_load(...)` 把它按当前 $W'$ 重新切分。**普通的 `optimizer.state_dict()` 在 FSDP 下换 $W$ 恢复一定失败**，因为它的 key 是 `flat_param` 的下标，而 `flat_param` 的长度是 $W$ 的函数。

### 8.2 8→4 卡 reshard 时 step 怎么换算

先把符号定死：

| 符号 | 含义 |
| --- | --- |
| $b$ | 每 rank 每 micro-batch 的序列数 |
| $A$ | `accumulation_steps` |
| $W$ | world_size |
| `step` | MiniMind 训练循环的计数器 = **每 rank 消费的 micro-batch 数** |
| `iters` | `len(loader)`，每 rank 一个 epoch 的 micro-batch 数 $= \lceil |D| / (bW)\rceil$ |
| $G = WAb$ | 全局 batch（序列数） |

三个想保持不变的量，各自给出一个换算：

| 想保持不变 | 约束 | 换算 |
| --- | --- | --- |
| **消费的样本数** $= \text{step}\cdot b W$ | — | $\text{step}' = \text{step}\cdot\dfrac{bW}{b'W'}$ |
| **optimizer step 数** $= \text{step}/A$ | — | 需要 $A' = A\cdot\dfrac{\text{step}'}{\text{step}} = A\cdot\dfrac{bW}{b'W'}$ |
| **全局 batch** $G = WAb$ | — | 需要 $W'A'b' = WAb$ |

MiniMind 的 `lm_checkpoint` 存了 `world_size`，恢复时按 $\text{step}' = \text{step}\cdot W_{\text{saved}} / W_{\text{cur}}$ 换算（`已确认`，week02 审计定位到 `trainer_utils.py`）。**这个公式只覆盖了第一行，而且假设 $b'=b$。**

**8→4 的正确做法（`推导`）**：保持 $b'=b=16$，则

$$\text{step}' = \text{step}\times\frac{8}{4} = 2\,\text{step},\qquad A' = 2A\ (\text{WE1: } 4\to 8),\qquad G' = 4\times 8\times 16 = 512 = G\ \checkmark$$

**如果只改 $W$ 不改 $A$**（最常见的错误）：$G$ 从 512 塌到 256，$\text{step}'=2\text{step}$ 仍然对齐了数据位置，但 optimizer step 数变成 $2\text{step}/A = 2\times$ 原来的。**Adam 在同样的数据上做了两倍的更新，每次用一半的 batch。** 这不是"恢复"，这是换了一个训练配方。而且 loss 曲线完全看不出来——它照样平滑下降。

**学习率相位（`推导`）**：MiniMind 的 `get_lr(epoch*iters + step, epochs*iters, lr)` 用 micro-batch 计数作时钟，而 `iters` 本身正比于 $1/W$。8→4 时 $\text{iters}' = 2\,\text{iters}$、$\text{step}' = 2\,\text{step}$，比值 $\text{step}'/\text{iters}' = \text{step}/\text{iters}$ 不变 → **相位自动保持**。这一点是对的，前提仍然是 $b'=b$。

**换算表（$b'=b$）**：

| $W\to W'$ | $\text{step}'$ | 保 $G$ 需要的 $A'$ | 保 $G$ 需要的 $b'$（若不改 $A$） |
| --- | --- | --- | --- |
| 8 → 4 | $2\,\text{step}$ | $2A$ | $2b$ |
| 8 → 2 | $4\,\text{step}$ | $4A$ | $4b$ |
| 8 → 1 | $8\,\text{step}$ | $8A$ | $8b$ |
| 4 → 8 | $\text{step}/2$ | $A/2$（$A$ 为偶数时才整除） | $b/2$ |

**还必须一起恢复的三样东西**（漏一样就不是同一次训练）：

1. `GradScaler` 的 `scale` 与 `growth_tracker`（MiniMind 把 `scaler.state_dict()` 存进了 `_resume.pth`，`已确认`）。不恢复 → 重新从 65536 开始搜索，前若干步白跳。
2. `DistributedSampler` 的 `epoch`（`set_epoch`）与 epoch 内的偏移。不恢复 → 数据顺序变了，"接着训"变成"重新采样"。
3. Adam 的 `exp_avg` / `exp_avg_sq` **以及每参数的 `step` 计数**（偏差修正 $\hat m = m/(1-\beta_1^t)$ 用的就是它）。

### 8.3 一个必须知道的陷阱：MiniMind 存了两种 checkpoint

`train_pretrain.py` 里同时做两件事（`已确认`，第 1.4 节引用的保存块）：

```python
torch.save({k: v.half().cpu() for k, v in state_dict.items()}, ckp)   # pretrain_768.pth：fp16 部署权重
lm_checkpoint(..., model=model, optimizer=optimizer, scaler=scaler, ...)  # _resume.pth：fp32 续训状态
```

- `pretrain_768.pth` 是 **`.half()` 之后的 fp16**，没有优化器状态。
- `_resume.pth` 才是完整的续训状态。

**用 `--from_weight pretrain` 续训（而不是 `--from_resume 1`）会同时做两件坏事**：丢掉全部优化器状态（Adam 的 $m,v$ 归零，前几百步等于重新预热），并且把每个权重舍入到 fp16（相对误差 $u=2^{-11}=4.88\times 10^{-4}$，第 3.1 节）。两件事都不会让 loss 曲线难看——它会有一个小凸起然后继续下降。**这是本周第四类"曲线正常但不是同一次训练"的情形。**

### 8.4 恢复后判定 loss 连续的判据

四条判据，强度递增。**只有 R2 是充分的。**

**R1（必要不充分）：落带判据。** 取中断前最后 20 个已记录 loss，$L_0'$ 落在 $[\min, \max]$ 内，或 $|L_0' - \bar L| \le 3\sigma$。

为什么不充分：一个把优化器状态丢光的恢复（8.3 节），参数完全正确，第一步 loss 必然落在带内。$m,v$ 归零的影响要 50–200 步之后才在曲线上显形。**R1 只能证明参数没坏，不能证明"这是同一次训练的继续"。**

**R2（充分）：5 步重放等价。** 在保存 checkpoint 的同时，继续跑 5 步并记下这 5 个 loss（$\ell_1..\ell_5$）。然后从 checkpoint 冷启动，用同样的 seed、同样的 sampler epoch、同样的 $W$ 跑 5 步得到 $\ell_1'..\ell_5'$。判据：

$$\max_i \frac{|\ell_i - \ell_i'|}{\ell_i} < 10^{-4}$$

这个阈值是推出来的（`推导`）：参数、梯度、loss 全在 fp32；同 $W$、同 $b$ 下 GEMM 的 shape 相同 → cuBLAS 选同一个 kernel → 累加顺序相同 → 差异只来自非确定性 kernel（本模型的前反向里 `index_add` 类算子只在 embedding 反向出现），量级 $\sim 10^{-6}$ 相对。给一个数量级的余量取 $10^{-4}$。**如果看到 $10^{-2}$，那不是数值噪声，是有东西没被恢复。**

**R3（跨 world_size 的必要条件）：优化器状态的逐 FQN 范数。** 8→4 reshard 后，对每一个参数 FQN：

$$\big|\ \|m_{\text{fqn}}\|_2^{\text{after}} - \|m_{\text{fqn}}\|_2^{\text{before}}\ \big| / \|m_{\text{fqn}}\|_2^{\text{before}} < 10^{-6}$$

$\|\cdot\|_2$ 对分片方式是不变的（把一个张量切成几块再求平方和，结果不变），所以它是一个跨 $W$ 可比的量。逐 FQN 而不是全局求和，是为了同时抓住"参数之间错位"（全局范数对这种错误免疫）。

**R4（跨 world_size 的必要条件）：scale 与 step 计数。** `scaler.get_scale()` 与 Adam 每参数的 `step` 在 reshard 前后必须逐位相等。这两个是标量/整数，不涉及分片，不相等就是恢复代码漏了。

**判据组合**：同 $W$ 恢复用 R2 + R4；8→4 reshard 用 R2（$W'$ 下无法与原 5 步逐一比对，改为"从 reshard 后的状态跑 5 步，与从 `FULL_STATE_DICT` 路径恢复到同一个 $W'$ 再跑 5 步"对比）+ R3 + R4。

**最小片段（说明 R3 怎么算，完整的 reshard 脚本在 `lab/scripts/` 下）**：

```python
with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT):
    osd = FSDP.optim_state_dict(model, optimizer)                  # 按 FQN 组织，跨 W 可比
norms = {fqn: st["exp_avg"].float().norm(dtype=torch.float64).item()
         for fqn, st in osd["state"].items()}                       # 逐 FQN 的 ||m||_2
```

---

## 9. 本周的失败模式清单与自查表

### 9.1 失败模式清单（症状 → 破坏的不变量 → 最小检查 → 属于哪本账）

| # | 症状 | 破坏的不变量 | 最小检查 | 账 |
| --- | --- | --- | --- | --- |
| 1 | 启动即 `RuntimeError`，提到 bfloat16 | dtype ∈ 硬件支持集 | `torch.cuda.is_bf16_supported()` 返回 False；命令行必须是 `--dtype float16` | 误差 |
| 2 | 第 0 步 loss 不是 $8.764\pm 0.05$ | $L_0 = \ln V$ | 打印 `vocab_size`、`logits.std()`、确认 tie 生效（`lm_head.weight.data_ptr() == embed_tokens.weight.data_ptr()`） | 误差 |
| 3 | 曲线平滑但平台偏高/偏低 | D1 / D2 / D3 | 第 6.3 节的四行断言；再做一次生成复核 | 数据 |
| 4 | 曲线平坦，无报错 | I2 落地率、I4 参数移动 | 第 6.4 节；先看 `finite_fwd` | 误差 |
| 5 | 曲线正常但收敛慢 2–3 倍 | I2 落地率（半跳） | 落地率滑动窗口 < 95% | 误差 |
| 6 | 多卡曲线与单卡不同，无报错 | I3 跨 rank 参数一致 | 第 6.5 节的梯度 all_gather 断言 | 通信 |
| 7 | 各 rank loss 完全相同 | I5 ≠ 0 | 检查 `DataLoader` 是否传了 `sampler=DistributedSampler(...)`、是否每 epoch `set_epoch` | 数据 |
| 8 | OOM 且远早于预期 | 激活估计式 | 用 $N\times 594{,}944$ B 反算；再查 SDPA 后端是否落到 math（第 7.3 节，激活近乎翻倍） | 字节 |
| 9 | FSDP 显存**高于** DDP | $m_\text{FSDP} < m_\text{DDP}$ | 检查 `auto_wrap_policy` 是否真的生效（退化成单一 root unit 时前向峰值 = 全模型）；打印 unit 数应为 9 | 字节 |
| 10 | 换到 4 卡恢复后 loss 有阶跃 | R2 五步重放 | 先查 $A'$ 是否按 8.2 节翻倍；再查 scaler 与 sampler epoch 是否恢复 | 全部 |
| 11 | 换到 4 卡恢复报 `size mismatch` / `flat_param` 长度错 | 分片 = 函数($W$, wrap 粒度) | 用了 `LOCAL_STATE_DICT`？必须换成 `SHARDED_STATE_DICT` + `torch.distributed.checkpoint`，或 `FULL_STATE_DICT(rank0_only)` | 字节 |
| 12 | FSDP 下各 rank 的 scale 不同 | 跨 rank scale 一致 | 用了普通 `GradScaler`？换 `ShardedGradScaler`（第 3.4 节第 4 点） | 误差 |
| 13 | 梯度裁剪后梯度几乎为 0 | `unscale_` 在 `clip` 之前 | 检查两行的顺序；错序等价于阈值被缩小 65536 倍（第 1.4 节） | 误差 |
| 14 | 通信占比远高于预期 | 每 optimizer step 的 all_reduce 次数 = 桶数 | 是否漏了 `no_sync()`？计数器应显示每 optimizer step 11 次而不是 $11A$ 次（第 4.2 节） | 通信 |
| 15 | 某 rank 卡住，其余 rank 静止到超时 | 所有 rank 调用相同顺序、相同数量的集合通信 | 各 rank 的 `collectives_total` 计数必须相同；设 `TORCH_NCCL_ASYNC_ERROR_HANDLING=1` 把超时变成 crash | 通信 |
| 16 | 模型不会停止生成 | `pad_token_id != eos_token_id` | Day 0 打印三个特殊 token id（第 5.2 节） | 数据 |
| 17 | 续训后前几百步明显退步 | 用的是 `_resume.pth` 而不是 `.half()` 的部署权重 | 检查用了 `--from_resume 1` 还是 `--from_weight`（第 8.3 节） | 全部 |

### 9.2 每次开跑前的六行自查表

按顺序做，前一条不过不做下一条（CLAUDE.md 第 6 节第 6 条）：

| # | 检查 | 通过判据 | 在哪台机器 |
| --- | --- | --- | --- |
| 1 | 路径合同 | `python lab/scripts/check_data_layout.py` 全 `OK` | V100 |
| 2 | 数据契约 | 第 6.3 节四行断言全过，打印 $\rho$ | 本机或 V100（CPU 即可） |
| 3 | 参数量 | `sum(p.numel() for p in model.parameters()) == 63912192` | 本机（CPU） |
| 4 | 第 0 步 loss | $\in [8.71, 8.82]$ | V100 单卡 |
| 5 | 字节账对账 | `torch.cuda.max_memory_allocated()` 与第 2.4 节手算列的偏差 < 25% | V100 单卡 |
| 6 | 通信账对账 | `collectives_total` 与第 4.3 节表格逐行相符 | V100 多卡 |

第 3、4 条是**结构性**的：它们只有一个正确答案，任何偏离都必须查清楚再往下走。第 5、6 条是**对账性**的：手算列与实测列都写进 `$MM_RUNS_ROOT/day3_dist/byte_ledger.md`，偏差列写清楚差在哪一项。

### 9.3 三档故障的口袋卡（考场里能默写出来的三行）

```
A  label mask 错位  :  I1 异常                       → 充分。曲线不会难看，只有张量断言能抓。
B  GradScaler 跳步  :  I2 异常 ∧ I4 = 0              → 充分。单看 I2 或单看 I4 都只是必要。
C  DDP 未同步       :  I3 ≠ 0                        → 充分，但不必要（各 rank 同数据时 I3 = 0）。
                       I3 = 0 ∧ I5 = 0              → 判"采样器未分片"这第四种。
```

### 9.4 Teach-back 清单（闭卷复述，不看文档）

1. **参数量手算链**：从 $d=768,L=8,H=8,H_{kv}=4,V=6400$ 出发，写出 $d_{ff}=\lceil 768\pi/64\rceil\times 64=2432$，逐个子模块写出 $P_\ell=7{,}374{,}528$，再到 $P=63{,}912{,}192$；说明 tie 省了多少参数、省了多少字节、在 FSDP 里带来什么约束。（Gate A0）
2. **字节账四格**：DDP 与 FSDP `FULL_SHARD` 下每卡的 param/grad/optim 各是 $16P$ 与 $16P/W$；说出两笔容易漏记的项（DDP 的 $4P$ bucket、autocast 的 $2P_{\text{linear}}$ 权重缓存），以及 `gradient_as_bucket_view=True` 省下的确切数字。（Gate A0）
3. **激活估计式**：$\text{Act} \approx bT(84.33\,dL + 12V)$，即每 token 约 0.595 MB；说明为什么它与 $T$ 无关、math 后端下为什么变成正比于 $T$；说明为什么 FSDP 在 64M 上只把每卡总量从 7.3 GB 降到 6.1 GB，以及真正的杠杆是激活重计算。（Gate A0）
4. **通信账的两个比值**：$\text{FSDP}_{\text{fp16-reduce}}/\text{DDP}=6P/8P=0.75$、$\text{FSDP}_{\text{fp32-reduce}}/\text{DDP}=1.00$，且 $(W-1)/W$ 抵消所以与 $W$ 无关；写出 27 / 11 / 18 三个次数和 189 / 154 / 126 三个 ring step 数；说明"FSDP 慢"不能归因于线上字节，以及三个真正来源与各自的开关。（Gate A0 + A1）
5. **fp16 的三个数与四个阈值**：$2^{-14}$、65504、$2^{-10}$；以及 $|g|<2^{-12}$ 让 `exp_avg_sq` 归零、$\bar x>9.235$ 让 fp16 域的 RMSNorm 溢出、$x>11.09$ 让 `exp` 溢出、8.764 处 fp16 ULP $=7.8\times 10^{-3}$。解释 `init_scale=65536` 为什么恰好把溢出上限推到 0.99951、为什么它跟 `grad_clip=1.0` 是配套的。（Gate A1）
6. **三档故障的判别表**：默写 9.3 的口袋卡，并说清哪一对是充分的、哪一对只是必要的，以及每一个"仅必要"的反例是什么。（Gate 跨场景 debug）
7. **数据契约**：写出 pretrain 的 D1（$n_{\text{label}} = n_{\text{nonpad}} - 1$）与 SFT 的 D2；解释为什么 $\ln 6400 = 8.764$ 能验模型接线但验不了掩码；解释为什么"过拟合到 0"检查不出位移，而生成能。（Gate A1）
8. **8→4 reshard**：写出 $\text{step}'=\text{step}\cdot W/W'$ 与保持 $G$ 所需的 $A'=2A$；说出 `FULL_STATE_DICT` / `SHARDED_STATE_DICT` / `LOCAL_STATE_DICT` 各自能不能换 $W$；说清 R1 为什么只是必要、R2 五步重放为什么充分、$10^{-4}$ 这个阈值怎么来的。（Gate A0 + A1）

---

## 附录 A. 来源（URL + 核验日期）

**MiniMind @ `7a6fddd63a30c06b2fdd5fac4089922b29bc841b`（Apache-2.0），核验日期 2026-09-08**

- `model/model_minimind.py`：https://raw.githubusercontent.com/jingyaogong/minimind/7a6fddd63a30c06b2fdd5fac4089922b29bc841b/model/model_minimind.py
  引用点：`MiniMindConfig.__init__` 全部默认值、`intermediate_size = math.ceil(hidden_size * math.pi / 64) * 64`、`Attention.__init__` 的四个 `bias=False` 线性层与 `q_norm`/`k_norm = RMSNorm(head_dim)`、`RMSNorm.forward` 的 `(self.weight * self.norm(x.float())).type_as(x)`、`repeat_kv` 的 `expand(...).reshape(...)`、`F.scaled_dot_product_attention(xq, xk, xv, dropout_p=..., is_causal=self.is_causal)`（不传 `attn_mask`）、`apply_rotary_pos_emb` 末尾的 `.to(q.dtype)`、`FeedForward.forward` 的 `down_proj(act_fn(gate_proj(x)) * up_proj(x))`、`MiniMindForCausalLM.forward` 的 `x, y = logits[..., :-1, :].contiguous(), labels[..., 1:].contiguous()` 与 `F.cross_entropy(..., ignore_index=-100)`。
- `dataset/lm_dataset.py`：https://raw.githubusercontent.com/jingyaogong/minimind/7a6fddd63a30c06b2fdd5fac4089922b29bc841b/dataset/lm_dataset.py
  引用点：`PretrainDataset.__getitem__` 的 `[bos] + tokens + [eos]` + 右填充 + `labels[input_ids == pad_token_id] = -100`；`SFTDataset.__getitem__` 与 `generate_labels` 的 `bos_id`/`eos_id` 扫描区间标注。
- `trainer/train_pretrain.py`：https://raw.githubusercontent.com/jingyaogong/minimind/7a6fddd63a30c06b2fdd5fac4089922b29bc841b/trainer/train_pretrain.py
  引用点：`train_epoch` 全文（`autocast_ctx` 块、`scaler.scale(loss).backward()`、`scaler.unscale_` → `clip_grad_norm_` → `scaler.step` → `scaler.update()` 顺序、**无 `no_sync()`**、两种 checkpoint 的保存）；argparse 默认值（`--dtype bfloat16`、`--batch_size 32`、`--max_seq_len 340`、`--accumulation_steps 8`、`--grad_clip 1.0`、`--learning_rate 5e-4`）；`GradScaler(enabled=(args.dtype == 'float16'))`。

**PyTorch v2.1.0，核验日期 2026-09-08**

- AMP 文档：https://docs.pytorch.org/docs/2.1/amp.html
  引用点：`GradScaler(init_scale=65536.0, growth_factor=2.0, backoff_factor=0.5, growth_interval=2000, enabled=True)`；autocast 的 fp16 算子列表、fp32 算子列表、"提升到最宽输入类型"列表；`cache_enabled` 的 "weight cache" 表述。
- FSDP 文档：https://docs.pytorch.org/docs/2.1/fsdp.html
  引用点：`StateDictType` 三种类型的定义原文（"unflattened and not sharded" / "unflattened but sharded" / "no transformation will be performed"）；`ShardingStrategy` 五个成员；`MixedPrecision` 的 `param_dtype` / `reduce_dtype` / `buffer_dtype` / `keep_low_precision_grads`；`state_dict_type()` / `set_state_dict_type()` / `optim_state_dict()` / `optim_state_dict_to_load()`。
- FSDP 源码签名：https://raw.githubusercontent.com/pytorch/pytorch/v2.1.0/torch/distributed/fsdp/fully_sharded_data_parallel.py
  引用点：`__init__` 的默认值 `backward_prefetch=BackwardPrefetch.BACKWARD_PRE`、**`forward_prefetch: bool = False`**、`limit_all_gathers: bool = True`、`use_orig_params: bool = False`、`sync_module_states: bool = False`、`sharding_strategy: Optional[ShardingStrategy] = None`。
- DDP 源码签名：https://raw.githubusercontent.com/pytorch/pytorch/v2.1.0/torch/nn/parallel/distributed.py
  引用点：`__init__` 的默认值 **`bucket_cap_mb=25`**、`gradient_as_bucket_view=False`、`find_unused_parameters=False`、`broadcast_buffers=True`、`static_graph=False`；`_DEFAULT_FIRST_BUCKET_BYTES` 的存在（其具体值 1 MiB 标 `推断`，未逐字核验 C++ 常量）。
- CUDA 能力判定源码：https://raw.githubusercontent.com/pytorch/pytorch/v2.1.0/torch/cuda/__init__.py
  引用点：`is_bf16_supported()` 的 `torch.cuda.get_device_properties(torch.cuda.current_device()).major >= 8 and cuda_maj_decide`。V100 的 major = 7 → 返回 False。

**沿用 2026-09-04 审计（本周未重新联网核验，标注沿用日期）**

- PyTorch 2.1 DDP notes：https://docs.pytorch.org/docs/2.1/notes/ddp.html —— Reducer、按桶索引顺序 all_reduce、构造时从 rank 0 广播 `state_dict`。
- PyTorch 2.1 distributed：https://docs.pytorch.org/docs/2.1/distributed.html —— `new_group`、默认 `timeout=1800s`、`NCCL_ASYNC_ERROR_HANDLING`。
- V100 / Volta 的硬件规格（sm70、FP16 Tensor Core、无 BF16、32 GB HBM2、约 900 GB/s、SXM2 峰值约 125 TFLOPS FP16 TC / 15.7 TFLOPS FP32）与"整数 MMA 从 sm75 开始"。逐条 URL 与核验日期见 `06_QUANT_LOWBIT.md` 第 1 节。

**本周包内来源（核验日期 2026-09-08）**

- `00_WEEK_CARD.md` 第 1、3、8 节：三本账的定义、V100 + torch 2.1 硬事实表、三类环境边界与数据不出公司的约束。
- `datasets/DATA_MANIFEST.json`：`pretrain_t2t_mini.jsonl` 的 1,241,043,656 B 与 sha256 前缀 `6dd6716c84ab3689`，dataset revision `312afb4f76391145c6902f765bb51691c09a12f5`。
- `lab/src/mm_v100/paths.py`：`MM_DATA_ROOT` / `MM_WEIGHTS_ROOT` / `MM_RUNS_ROOT` / `MINIMIND_ROOT` 四个环境变量与 `run_dir(tag)` 的 `ckpt/logs/profiles` 子目录约定。本文所有路径都按这套合同写，不出现字面绝对路径。

**未核验、由具体某一天回答的 `未知` 项**

| 项 | 回答它的步骤 |
| --- | --- |
| 公司 torch 2.1.0 wheel 的 CUDA 版本、`torch.cuda.get_arch_list()` 是否含 `sm_70` | Day 0 探针 |
| NCCL 版本与卡间拓扑（决定 $\alpha$、$\beta$） | Day 0 wiring smoke |
| `tokenizer.pad_token_id` / `bos_token_id` / `eos_token_id` 的实际值，以及 pad 是否等于 eos | Day 0 探针 |
| SDPA 实际命中哪个后端 | Day 1 |
| $\eta F$（$d=768$ 规模 GEMM 的达成算力） | Day 3 的 $b$ 扫描 |
| 8 卡是否独占（决定 `step_time_ratio` 是否可作结论） | Day 0 |
