# Week M02 Reference Answers — 逐 ID 参考答案与评分标准

> 配套 `04_ORAL_EXAM.md`，ID 一一对应，共 27 题。**默认闭卷作答后再看。**
> 每题给：关键评分点 · 0–4 分 rubric · 常见误区 · 追问 · 定位（`01_FOUNDATIONS.md` 章节号 或 `lab/` 脚本名）。
> 通用高分标准（所有层通用，D/P/T 层强制）：**有假设、有证据、有替代解释、有失败边界、有可验证下一步**。五条里少于三条的答案，即使结论正确也封顶 2 分。
> 数值约定：MB/GB 为十进制（$10^6$ / $10^9$）。本文件所有数字都是由 `01_FOUNDATIONS.md` 第 1、3 节的配置**推出的 `计算` 值**，或标注为 `估算`；本周截至出卷时**没有任何实测数据**，答案里不得出现"实测"。

---

## Recall

### M02-R-01

**关键评分点**

1. `torch.cuda.is_bf16_supported()` 判断的是 **compute capability 的 major ≥ 8**（Ampere 及以上）。V100 是 sm70（major = 7），不满足。报错发生在 **进入 `torch.cuda.amp.autocast(dtype=torch.bfloat16)` 上下文时**（不是在某个算子里，也不是在 `.to(bfloat16)` 时），抛 `RuntimeError: Current CUDA Device does not support bfloat16`。
2. MiniMind 全部训练脚本的 `--dtype` 默认是 `bfloat16`（`train_pretrain.py` L91），本周一律改 `float16`。`GradScaler(enabled=(args.dtype == 'float16'))`（L137）——只有 fp16 时 scaler 才真正生效；`bfloat16` 时 scaler 是空壳。**`train_grpo.py` 没有 `GradScaler`**，lab 里要补。
3. 降到 fp16 的：`matmul` / `linear` / `bmm` / `conv`——落到模型上就是 q/k/v/o、FFN 的 gate/up/down、router 的 `gate` linear、`lm_head`。保留 fp32 的：`softmax`、`log_softmax`、`cross_entropy`、`layer_norm`（含 RMSNorm 的 `rsqrt`、`pow`、`sum`、`exp`）。**参数与 `param.grad` 始终是 fp32**（autocast 不改参数 dtype，master weight 就是原参数）——这正是第 3.3 节按 4 B/参数算 DDP 通信量的依据。

**0–4 分 rubric**

- 0：说不出 bf16 报错的硬件条件，或以为 V100 支持 bf16。
- 1：知道 V100 不支持 bf16，但说不清判据（major ≥ 8）或报错时机。
- 2：第 1、2 问正确。
- 3：三问全对，含 autocast 的算子分工与 param/grad 是 fp32。
- 4：以上全对，并主动指出"报错在进入 autocast 时抛出"意味着**这是启动即失败、不会污染任何训练结果**，属于最好排查的一类故障；且能说出 fp32 的 param/grad 直接决定了 DDP 每步 255.6 MB 而不是 127.8 MB。

**常见误区**

- 认为可以靠 `model.to(torch.bfloat16)` 绕过 `is_bf16_supported()` 的检查。检查在 autocast 入口，绕过它只会把问题变成更隐蔽的 kernel 级失败或极慢的软件模拟，本周的做法是直接换 fp16。
- 认为 autocast 下参数变成了 fp16，于是把 DDP 通信量算成 127.8 MB。autocast 只在算子边界做 cast，参数与梯度仍是 fp32。

**追问**

- `train_grpo.py` 缺 `GradScaler`，在 fp16 下最早会以什么形式暴露出来？
- 如果某天换到支持 bf16 的卡上，第 3 问里"保留 fp32 的算子"这份清单会变吗？为什么？

**定位**：`01_FOUNDATIONS.md` §4.1、§4.2、§1.3；`00_WEEK_CARD.md` 硬事实第 1 条；`lab/` 的 `train_ddp.py`。

---

### M02-R-02

**关键评分点**

1. torch 2.1 的 **flash 后端只覆盖 sm75–sm90**；V100 是 **sm70**，不在区间内。因此 `F.scaled_dot_product_attention` 只能落到 **mem-efficient**（支持 fp16/fp32、支持 `is_causal`、分块计算不物化注意力矩阵）或 **math**（纯 PyTorch 实现，物化注意力矩阵，是数值 reference）。本周没有 FlashAttention-2。
2. `torch.backends.cuda.sdp_kernel(enable_flash=..., enable_mem_efficient=..., enable_math=...)`。
3. math 后端会物化 $[B, H, T, T] = [8, 8, 512, 512]$ 的注意力分数矩阵。fp16 下每层 $8\times8\times512\times512\times2\,\text{B} = 33{,}554{,}432\,\text{B} = \mathbf{33.5\,MB}$（`计算`），再加 fp32 的 softmax 中间量。

**0–4 分 rubric**

- 0：以为 V100 能用 flash 后端。
- 1：知道 V100 不能用 flash，但说不出 sm 区间或替代后端。
- 2：第 1、2 问正确。
- 3：三问全对，含 33.5 MB 的算式。
- 4：以上全对，并指出 `flash_attn=True` 这个配置名有误导性——它只是"是否调用 SDPA"的开关，不代表用了 FlashAttention 库；并说明本周把 math 当 FP32 reference、mem-efficient 当生产路径的实验设计意图。

**常见误区**

- 把 `MiniMindConfig.flash_attn=True` 当成"启用了 FlashAttention"。
- 把 $[B,H,T,T]$ 记成 $[B,T,T]$ 或忘记乘 head 数，把 33.5 MB 算成 4.2 MB。

**追问**

- 如果把 `T` 从 512 提到 1024，math 后端的这块显存变成多少？这是什么阶的增长？
- 你要怎么在不带出绝对显存数字的前提下，向周报报告"math 与 mem-efficient 的显存差"？

**定位**：`01_FOUNDATIONS.md` §4.3、§3.5（SDPA 后端行）；`00_WEEK_CARD.md` 硬事实第 2 条；`lab/` 的 `train_ddp.py` 的 `--sdpa-backend`。

---

### M02-R-03

**关键评分点**

1. `hidden_size=768`、`num_hidden_layers=8`、`vocab_size=6400`、`num_attention_heads=8` / `num_key_value_heads=4`、`head_dim=768/8=96`、`intermediate_size = ceil(768\cdot\pi/64)\cdot64 = 38\times64 = \mathbf{2432}`、`num_experts=4`、`num_experts_per_tok=1`、`moe_intermediate_size = 2432`（等于 `intermediate_size`）、`router_aux_loss_coef=5e-4`、`tie_word_embeddings=True`。
2. dense 每层 $= 2d^2 + 2dH_{kv}d_h + 2d + 2d_h + 3dd_{ff} = 1{,}179{,}648 + 589{,}824 + 1{,}536 + 192 + 5{,}603{,}328 = \mathbf{7{,}374{,}528}$；MoE 每层 $= 1{,}179{,}648 + 589{,}824 + 1{,}536 + 192 + 22{,}413{,}312 + 3{,}072 = \mathbf{24{,}187{,}584}$。差额来自两项：**FFN 从 1 份变 4 份**（5,603,328 → 22,413,312）与**新增 router `gate: nn.Linear(768,4)`** 的 3,072。注意 $2d_h = 192$ 是 `q_norm` 与 `k_norm` 两个 head-dim 级 RMSNorm，漏掉它每层会少 192、全模型少 1,536。
3. dense 总量 $= 8\times7{,}374{,}528 + 4{,}915{,}200(\text{embedding}) + 768(\text{末层 norm}) = \mathbf{63{,}912{,}192}$；MoE 总量 $= 8\times24{,}187{,}584 + 4{,}915{,}200 + 768 = \mathbf{198{,}416{,}640}$。两个数字已用 `sum(p.numel() for p in MiniMindForCausalLM(cfg).parameters())` 实测核对（`已确认`，2026-09-05），并与 Week M01 第 1 节一致。每 token 激活参数 = 每层 attention + norm + router + **1 个** expert $= 7{,}377{,}600$，×8 + embedding + norm $= 63{,}936{,}768 \approx \mathbf{63.9\,M}$，与 README 的"64M 激活 / 198M 总量"一致。

**0–4 分 rubric**

- 0：一半以上的值记错，或 `intermediate_size` 说不出来源。
- 1：记住基本超参，但算不出每层参数量。
- 2：基本超参 + 两个每层参数量正确。
- 3：三问全对，数字精确到个位。
- 4：以上全对，并指出 `tie_word_embeddings=True` 意味着 embedding 的 4,915,200 只计一次（否则总量会多 7.7%），以及"激活参数 ≈ dense 总量"这件事正是本周 MoE 实验能和 dense 基线直接对比的前提。

**常见误区**

- 把 `intermediate_size` 记成 $4d=3072$ 或 $\frac{8}{3}d$。MiniMind 用的是 $\lceil d\pi/64\rceil\cdot64$，结果是 2432。
- 算 GQA 的 k/v 时按 $2d^2$ 算。$H_{kv}=4$、$d_h=96$，所以是 $2\cdot768\cdot384 = 589{,}824$，只有 q/o 的一半。

**追问**

- 若把 `num_experts` 从 4 改成 8（EP=8 需要），总参数量与激活参数量各变成多少？
- 4 个 expert 的 FFN 占 MoE 层参数的百分之多少？这个比例对"专家参数必须单独走一个通信组"这件事意味着什么？

**定位**：`01_FOUNDATIONS.md` §1.1（参数量表）；`lab/` 的 `ep_moe.py`。

---

### M02-R-04

**关键评分点**

1. `ShardingStrategy.FULL_SHARD`、`SHARD_GRAD_OP`、`HYBRID_SHARD`。
2. 两者差在 **backward 前的 AllGather**：`FULL_SHARD` 在 forward 用完某子模块后立刻把完整参数 reshard 掉，反向再 AllGather 一次；`SHARD_GRAD_OP` forward 后**不 reshard**，反向直接用，省掉这一整轮。按 8 个 block + embedding 单独 wrap（共 9 个 wrap 单元）计：

| | AllGather（fwd） | AllGather（bwd） | ReduceScatter | 总次数 |
| --- | --- | --- | --- | --- |
| `FULL_SHARD` | 9 | 9 | 9 | **27** |
| `SHARD_GRAD_OP` | 9 | **0** | 9 | **18** |

3. `StateDictType.FULL_STATE_DICT` / `LOCAL_STATE_DICT` / `SHARDED_STATE_DICT`。2.1 默认 `backward_prefetch=BackwardPrefetch.BACKWARD_PRE`、`forward_prefetch=False`。

**0–4 分 rubric**

- 0：说不清 `FULL_SHARD` 与 `SHARD_GRAD_OP` 的差别。
- 1：能说出"一个 reshard 一个不 reshard"，但给不出次数。
- 2：差别与次数表正确。
- 3：三问全对，含两个 prefetch 的默认值。
- 4：以上全对，并指出 `SHARD_GRAD_OP` 省通信次数的代价是 **forward 后完整参数一直驻留显存**，所以它是"次数/显存"这条轴上的中间点，本周 Day 2 的假设正是它的 step time 介于 DDP 与 `FULL_SHARD` 之间。

**常见误区**

- 以为 `SHARD_GRAD_OP` 连 ReduceScatter 也省了。梯度仍然要 ReduceScatter，省的只是反向前的 AllGather。
- 把 wrap 单元数记成 8（忘了 embedding 单独 wrap），把总次数算成 24。

**追问**

- `HYBRID_SHARD` 在单节点 8 卡上有意义吗？它是为什么场景设计的？
- `forward_prefetch=False` 这个默认值对第 2 问的**次数**有影响吗？对什么有影响？

**定位**：`01_FOUNDATIONS.md` §3.4（FSDP 通信表）、§3.5（FSDP 行）；`lab/` 的 `train_fsdp.py`。

---

### M02-R-05

**关键评分点**

1. `dist.all_to_all_single(output, input, output_split_sizes=None, input_split_sizes=None, group=None, async_op=False)`。两个 split 列表的**长度都等于该 group 的 world size**。`input_split_sizes[j]` = 本 rank 沿 `input` 的**第 0 维**发给 rank $j$ 的行数；`output_split_sizes[j]` = 本 rank 沿 `output` 的第 0 维从 rank $j$ 接收的行数。约束：`sum(input_split_sizes) == input.shape[0]`、`sum(output_split_sizes) == output.shape[0]`。
2. `new_group(ranks)` 要求**主 group 里的所有进程都按相同顺序调用它**，即使自己不在 `ranks` 里（非成员拿到的返回值不可用于通信）。不满足会导致进程间的 group 编号错位，表现为 hang 或后续集合通信语义错误。
3. 默认 `timeout=1800s`（30 分钟）。`NCCL_ASYNC_ERROR_HANDLING=1`（`01_FOUNDATIONS.md` §6 记为 `TORCH_NCCL_ASYNC_ERROR_HANDLING=1`）把"超时后静默等待"改成"超时后 crash"。
4. `_resume.pth` 的键：`model`、`optimizer`、`epoch`、`step`、`world_size`、`scaler`。换算式：`step_new = step * saved_ws // current_ws`。

**0–4 分 rubric**

- 0：写不出 `all_to_all_single` 的参数或把两个 split 说反。
- 1：参数列表对，但说不清两个 split 各自计的是哪个方向。
- 2：第 1、2 问正确。
- 3：四问全对。
- 4：以上全对，并指出两个 split 列表**必须在所有 rank 上互相匹配**（这是 I2），以及 `new_group` 的"全员调用"要求正是 EP 代码里最容易漏的一行——非成员 rank 上那行 `new_group` 看起来是死代码，删掉就 hang。

**常见误区**

- 把 split 列表的长度记成"专家数"。它的长度是 **group 的 world size**（EP 组大小），不是 $E$；只有 $E = E_p$ 时两者碰巧相等。
- 认为 `new_group` 只需要成员调用。

**追问**

- `sum(input_split_sizes) == input.shape[0]` 这个断言能不能防住 I2 违反？为什么？
- `_resume.pth` 里存 `scaler` 是为了保住哪条不变量？

**定位**：`01_FOUNDATIONS.md` §2.4、§3.5、§6 第 2/5/10 条；`lab/` 的 `ep_moe.py`、`ckpt_reshard.py`。

---

## Explain

### M02-E-01

**关键评分点**

1. `load` $= f_e = \frac{1}{N}\sum_i\mathbb{1}[e\in\mathcal{I}_i]$（实际负载份额）；`scores.mean(0)` $= P_e = \frac{1}{N}\sum_i s_{i,e}$（平均路由概率）；`* E` 是专家数 4 的归一化因子；`* coef` 是 $\lambda = 5\times10^{-4}$。`scores` 是 `softmax(z)`，在 autocast 下 **softmax 保留在 fp32**，所以 aux loss 全程 fp32，不受 fp16 动态范围影响。
2. $E\sum_e f_eP_e$：完全均衡时 $f_e = P_e = 1/E$，$\sum_e f_eP_e = E\cdot\frac{1}{E^2} = \frac{1}{E}$，乘 $E$ 得 **1**（这是 $k=1$ 时的最小值）；完全坍缩到单个专家时 $f = P = (1,0,0,0)$，$\sum = 1$，乘 $E$ 得 **$E$ = 4**。所以这个量的取值范围是 $[1, E]$，读日志时直接当"热点程度"用。
3. 梯度只经 **$P_e$**（`scores.mean(0)`）流回 router。$f_e$ 由 `one_hot`/`topk` 的离散选择得到，**不可导**，不贡献梯度。
4. MiniMind 当前 MoE **没有**：capacity（无上限）、token dropping（不丢）、EP（无跨卡专家）、shared expert；另外也没有 `seq_aux`（是对 $B\cdot T$ 全部 token 算一个 aux，不是逐序列）。其中**没有 capacity / 不丢 token（dropless）**这一条直接导致木桶效应可观察：热点专家的 token 数无上界，最慢 rank 的计算时间就是 EP 组的 step time。

**0–4 分 rubric**

- 0：写不出公式与代码的对应。
- 1：对应关系正确，但算不出两个极端值。
- 2：第 1、2 问正确。
- 3：四问全对，含"梯度只经 $P_e$"。
- 4：以上全对，并指出 $[1,E]$ 这个范围让 `aux_loss` 成为一个**无量纲的热点读数**，因此它适合当副指标（诊断）而不适合当主指标（优化目标已经把它压向 1 了，见 M02-P-01）。

**常见误区**

- 把 `aux_loss` 打印值直接当成进入总 loss 的那一项。进总 loss 的是乘了 $\lambda=5\times10^{-4}$ 之后的值；诊断时看的是乘 $\lambda$ 之前的 $[1,E]$ 量。
- 认为 $f_e$ 也提供梯度，于是以为 aux loss 能"直接惩罚负载"。它只能通过把 $P_e$ 推向均匀来间接影响路由。

**追问**

- `MOEFeedForward` 里那句 `0*sum(p)` 形式的写法是为了解决什么问题？它和本周失败模式表第 2 条有什么关系？
- 如果改成 `seq_aux`（逐序列算 aux 再平均），在什么数据分布下结论会和现在不同？

**定位**：`01_FOUNDATIONS.md` §2.3、§4.2、§4.5、§4.6、§6 第 6/7 条；`lab/` 的 `router_stats.py`。

---

### M02-E-02

**关键评分点**

1. $\hat g=\frac{1}{W}\sum_{r=1}^{W}\sum_{a=1}^{A}\frac{1}{A}\nabla\ell_{r,a}$，其中 $\ell_{r,a}$ 是 micro-batch 内的 mask 加权 **token 均值**。$\hat g = \nabla\mathcal{L}$ **当且仅当每个 micro-batch 的有效 token 数 $\sum_t m$ 相等**（各 micro-batch 权重才等于它的 token 占比）。
2. 按 **bucket（桶）** 打包，默认 `bucket_cap_mb=25`。64M dense 模型的梯度是 fp32，总量 $63{,}912{,}192\times4\,\text{B} = 255{,}648{,}768\,\text{B} = 255.6\,\text{MB}$；`bucket_cap_mb` 的单位是 **MiB**（$25\times1024^2=26.2\,\text{MB}$），不是十进制 MB：$255{,}648{,}768/26{,}214{,}400 = 9.75$，再加上 DDP 单独的 1 MiB 首桶 → **约 11 个桶**（`计算`）。
3. 按**桶索引顺序**启动，不是就绪顺序。好处：所有 rank 的 all-reduce 发起顺序完全一致，不需要额外协商就不会错配（顺序一致本身就是集合通信的前提）。坏处：如果某个靠前索引的桶最后才就绪，后面已经就绪的桶只能等它，重叠度下降；这也是 `find_unused_parameters` 和动态分支（MoE 空专家）会引发问题的根源。
4. EP=4/DP=2 时，**专家参数的通信组不是全 8 卡**：rank 2 上的 expert 2 只和 rank 6 构成同专家副本，梯度应在 `{2,6}` 上 all-reduce；而 attention/embedding/norm 在全 8 卡上 all-reduce。一个 `DDP(model)` 只有一个 `process_group`，会把专家梯度也在 8 卡上平均，等于把 4 个不同专家的梯度混在一起——数学上错误。替代做法：把专家参数放进带 `process_group={2,6}` 的独立桶，或（本周 lab 的 `来源计划假设`）在 `ep_moe.py` 里**手写两组 `all_reduce`** 而不是套两层 DDP。

**0–4 分 rubric**

- 0：写不出 $\hat g$ 或说不出桶。
- 1：$\hat g$ 正确，但给不出等价条件。
- 2：第 1、2、3 问正确。
- 3：四问全对，含"桶索引顺序"及其两面性。
- 4：以上全对，并指出第 4 问的错误是**静默的**（不会报错、loss 照样下降），只能靠"EP=1 与 EP=4 的等价性检查"或直接比对专家参数 checksum 才能发现。

**常见误区**

- 把 `bucket_cap_mb=25` 理解成"25 MB 参数"，用 fp16 算成 128 MB / 25 ≈ 5 个桶。梯度是 fp32。
- 认为 DDP 按"梯度就绪顺序"发起 all-reduce。就绪顺序会因 rank 而异，集合通信不允许。

**追问**

- 如果专家梯度被错误地在 8 卡上 all-reduce，训练会崩吗？给一个 200 步内可观察的症状。
- `bucket_cap_mb` 调大到 100 会怎样？在本周 64M 模型上，桶数和重叠度分别怎么变？

**定位**：`01_FOUNDATIONS.md` §2.1、§3.2、§3.3、§6 第 2 条；`lab/` 的 `train_ddp.py`、`ep_moe.py`。

---

### M02-E-03

**关键评分点**

1. 六步（EP 组大小 $E_p$，每 rank 持 $E/E_p$ 个专家，$\rho(i)=\lfloor\mathcal{I}_i/(E/E_p)\rfloor$）：
   - **① 计数**：由 `topk_idx` 得 `dest_rank`，`c_r[j] = bincount(dest_rank)`；一次 `all_to_all_single` 交换得 `c'_r[j] = c_j[r]`（**通信 1**：counts 交换，传 `[E_p] int64`）。
   - **② permutation**：`perm = argsort(dest_rank, stable=True)`，`x_sorted = x_flat[perm]`，长度 $Nk$；无通信。
   - **③ dispatch**：`all_to_all_single(recv, x_sorted, output_split_sizes=c'_r, input_split_sizes=c_r)`（**通信 2**），`recv` 长度 $\sum_j c'_r[j]$。
   - **④ 本地 expert**：`recv` 按本地专家 id 再分桶（$E/E_p>1$ 时），逐桶做 FFN；无通信。
   - **⑤ combine**：逆向 `all_to_all_single(back, out, output_split_sizes=c_r, input_split_sizes=c'_r)`（**通信 3**）。
   - **⑥ unpermute + 加权**：`y.index_add_(0, token_of[perm], back * w)`；无通信。
2. **I2**：对每一对 $(r,j)$，rank $r$ 的 `input_split_sizes[j]` 必须等于 rank $j$ 的 `output_split_sizes[r]`。
3. 因为 `all_to_all_single` 的接收方是按 `output_split_sizes` **预先算好要收多少字节**才发起等待的。若发送方少发了，接收方就永远等不到那个字节数——NCCL 没有"对端声明的长度"这一层协商，只有本地声明的长度，所以不匹配不是协议错误而是**永不满足的等待**，直到 30 分钟 timeout 才被 watchdog 发现。
4. **I3**：$E_p=1$ 时，六步的输出与原 `MOEFeedForward.forward` 在 fp32 下**逐元素相等**。可以用 gloo 后端、2 进程、CPU 直接验证（`tests/test_ep_cpu.py`），不占 8 卡配额。
5. 一个 MoE 层一个训练 step：`all_to_all_single` 共 **4 次**——前向 dispatch、前向 combine、combine 的反向（= 一次 a2a）、dispatch 的反向（= 一次 a2a）；**另加 1 次 counts 交换**（不参与反向）。合计 5 次集合通信，其中 4 次搬 hidden。

**0–4 分 rubric**

- 0：六步说不全或顺序错。
- 1：六步正确，但写不出 I2。
- 2：六步 + I2 + I3 正确。
- 3：以上全对，并正确解释"为什么是 hang 不是报错"。
- 4：以上全对，且第 5 问的次数与归属完全正确，并指出**counts 交换本身也是一次集合通信，同样受 I2 之外的"全员参与"约束**——某个 rank 提前 break 出循环就会在这一步 hang。

**常见误区**

- 忘记 permutation 必须是 **stable** 排序，否则 unpermute 时 `index_add_` 的对应关系错，结果错但不报错。
- 只数 2 次 a2a（忘了反向也各有一次），把每层通信次数低估一半。

**追问**

- 第 ⑥ 步为什么用 `index_add_` 而不是直接赋值？在 top-2 下这个区别会怎样体现？
- counts 交换能不能省掉（比如用固定 capacity 预分配）？省掉之后 I2 还需要吗？

**定位**：`01_FOUNDATIONS.md` §2.4（六步与 I2/I3）、§3.1、§3.5、§6 第 5 条；`lab/` 的 `ep_moe.py`、`tests/test_ep_cpu.py`。

---

### M02-E-04

**关键评分点**

1. 对一个 wrap 单元：**forward 前** AllGather 完整参数 → 计算 → （`FULL_SHARD`）用完立刻 **reshard**，只留自己那一片；**backward 该单元开始前**再 AllGather 一次完整参数 → 算梯度 → **ReduceScatter** 梯度（每 rank 只保留自己那一片梯度）→ 再次 reshard。`SHARD_GRAD_OP` 在 forward 结束后**不 reshard**，完整参数一直留到反向用完，因此反向那次 AllGather 被省掉。
2. wrap 粒度**细**（每层一个单元）：集合通信次数**多**（每单元一轮），但同一时刻只有一个单元的完整参数在显存里，**峰值显存低**。wrap 粒度**粗**（只有一个根 FSDP）：集合通信次数**少**（每步 1 次 AllGather + 1 次 ReduceScatter），但那一次 AllGather 会把**整个模型的完整参数**拉齐，**前向峰值 = 全模型参数**，等于没省显存。这两个极端就是 M02-D-04 第 3 问的根因。
3. `forward_prefetch=False`（2.1 默认）意味着**第 $\ell$ 层的 AllGather 只能在第 $\ell-1$ 层计算结束后才发起**，forward 里通信与计算基本不重叠，每层前面挂一段暴露的等待。`BACKWARD_PRE` 在反向里补上了预取：在算第 $\ell$ 层梯度的同时，提前发起第 $\ell-1$ 层的 AllGather，所以反向的重叠好于前向。
4. 每层 AllGather 的**负载**是完整参数量 $S_\ell$（`MiniMindBlock`：$7{,}374{,}528\times2\,\text{B} = 14.75\,\text{MB}$，fp16 通信 dtype）；**每 rank 实际收发**是 $\frac{W-1}{W}S_\ell = \frac{7}{8}\times14.75 = 12.9\,\text{MB}$。两者不同是因为每 rank 本来就持有 $1/W$ 那一片，只需要从其余 $W-1$ 个 rank 收（并把自己那片发出去），不需要搬自己已有的部分。

**0–4 分 rubric**

- 0：说不清 AllGather / ReduceScatter 各自的时机。
- 1：时机正确，但说不清两种 sharding 在 forward 后的差别。
- 2：第 1、2 问正确。
- 3：四问全对，含 $\frac{W-1}{W}$ 的解释。
- 4：以上全对，并指出 wrap 粒度是**次数 ↔ 峰值显存**这条轴上唯一的旋钮，而 `forward_prefetch=False` 让"次数多"的代价在 forward 里被完整暴露——这两条合起来才是"64M 上 FSDP 慢"的机制层解释（见 M02-T-03）。

**常见误区**

- 认为 ReduceScatter 之后每 rank 有完整梯度。ReduceScatter 的产物就是分片梯度，这正是优化器状态也能分片的前提。
- 把每 rank 实际字节写成 $S_\ell$ 或 $S_\ell/W$。正确是 $\frac{W-1}{W}S_\ell$。

**追问**

- 如果把 `forward_prefetch` 打开，第 3.4 节的**次数**会变吗？你预期哪个抽象比值会动？
- `use_orig_params=True` 对 wrap 粒度和显存分别有什么影响？

**定位**：`01_FOUNDATIONS.md` §3.4、§5.1、§6 第 9 条；`lab/` 的 `train_fsdp.py`。

---

### M02-E-05

**关键评分点**

1. **I1**：若两种 $(W,A,b)$ 配置满足以下五条，则 $\hat g$ 逐步一致、loss 曲线在 fp16 舍入误差内重合：
   - ① $G = W\cdot A\cdot b$ 相同；
   - ② **每个 optimizer step 消费的样本集合相同**（不只是数量相同）；
   - ③ 每个 micro-batch 的**有效 token 数相等**（否则"均值的均值 ≠ 全局 token 均值"）；
   - ④ seed / dropout / 任何 RNG 依赖的前向路径一致；
   - ⑤ GradScaler 的 scale 值与增长计数相同，且 `clip_grad_norm_` 在 `unscale_` **之后**调用、accumulation 边界对齐。
2. `setup_seed(42 + rank)` 不破坏初值一致，是因为 **DDP 构造时会从 rank 0 广播 `state_dict()`**（2.1 DDP notes），各 rank 的随机初始化结果被覆盖掉；同时 `dropout=0.0`，前向不消耗 RNG。**只要把 `dropout` 调成 >0（或引入随机 mask / 随机数据增强），rank 间 RNG 不同就立刻破坏 I1 的 ④**。
3. **不相同**。`DistributedSampler(num_replicas=W)` 是按 $W$ 切分索引的：2 卡时每 rank 拿 1/2 的索引，8 卡时每 rank 拿 1/8；第 $k$ 个 optimizer step 上，2 卡×acc4 消费的是 2 个 rank 各 4 个 micro-batch，8 卡×acc1 是 8 个 rank 各 1 个——两者的**样本并集不同**。要让它们相同，lab 需要用**固定的全局索引表按 $G$ 顺序切**（每个 step 取全局第 $[kG,(k+1)G)$ 段，再在配置内部分配给 rank/accumulation），而不是依赖 `DistributedSampler` 的默认切分。
4. 设第 $j$ 个 micro-batch 的有效 token 数为 $n_j$、其 token 总损失为 $\Sigma_j$。micro-batch 均值再平均给出 $\frac{1}{M}\sum_j\frac{\Sigma_j}{n_j}$；全局 token 均值是 $\frac{\sum_j\Sigma_j}{\sum_j n_j}$。两者相等当且仅当所有 $n_j$ 相等（加权平均的权重从 $\frac{n_j}{\sum n}$ 变成了 $\frac{1}{M}$）。SFT 数据每条的 assistant 段长度不同 → $n_j$ 不等 → **必然**不等。系统性后果：**有效 token 数少的样本（短回答）被高估**，因为它拿到了和长样本一样的 $1/M$ 权重。

**0–4 分 rubric**

- 0：只会说"batch size 一样就等价"。
- 1：能列出 2–3 个条件。
- 2：五个条件基本齐全，第 2 问正确。
- 3：四问全对，含 `DistributedSampler` 的切分机制。
- 4：以上全对，且第 4 问给出了完整的加权对比推导并明确指出"短样本被高估"；并主动说明这一条使得 **SFT 阶段的 I1 只能是近似检查**，pretrain（定长、几乎无 padding）才适合当严格判据。

**常见误区**

- 认为 `setup_seed(42+rank)` 会让模型初值不同。DDP 的 rank 0 广播兜住了这一点；但一旦有 dropout 就兜不住前向。
- 认为"global batch 相同"就等价。样本**集合**必须相同，不只是数量。

**追问**

- 如果按第 3 问改成全局索引表，2 卡与 8 卡的 loss 还会有差吗？合理的残差量级是多少、来自哪里？
- `set_epoch(epoch)` 在这里起什么作用？漏掉它会破坏 I1 的哪一条？

**定位**：`01_FOUNDATIONS.md` §2.1（I1）、§4.7、§1.3、§6 第 3 条；`lab/` 的 `train_ddp.py`、`tests/test_ddp_gloo_cpu.py`。

---

## Apply

### M02-A-01（Gate 题）

**关键评分点**

1. EP=4/DP=2 的 8 卡表（`计算`，EP 组由连续 rank 组成，$E=4$、每 rank 持 $E/E_p=1$ 个专家）：

| rank | EP 组 | 持有 expert | 专家梯度 all-reduce 伙伴 | 非专家梯度 all-reduce 组 |
| --- | --- | --- | --- | --- |
| 0 | {0,1,2,3} | 0 | {0,4} | {0..7} |
| 1 | {0,1,2,3} | 1 | {1,5} | {0..7} |
| 2 | {0,1,2,3} | 2 | {2,6} | {0..7} |
| 3 | {0,1,2,3} | 3 | {3,7} | {0..7} |
| 4 | {4,5,6,7} | 0 | {0,4} | {0..7} |
| 5 | {4,5,6,7} | 1 | {1,5} | {0..7} |
| 6 | {4,5,6,7} | 2 | {2,6} | {0..7} |
| 7 | {4,5,6,7} | 3 | {3,7} | {0..7} |

2. 另两行配置：

| 配置 | EP 组 | 每 rank 专家数 | 专家梯度 DP 组 | 非专家 DP 组 |
| --- | --- | --- | --- | --- |
| EP=2, DP=4 | {0,1}{2,3}{4,5}{6,7} | 2 | {0,2,4,6}、{1,3,5,7} | {0..7} |
| EP=1, DP=8 | 8 个单元素组（等价于不建组） | 4 | {0..7} | {0..7} |

3. EP=8/DP=1 必须先把 `num_experts` 从 4 改成 **8**。约束：$E_p \le E$ 且 $E \bmod E_p = 0$（每 rank 持 $E/E_p$ 个专家，必须是整数且 ≥ 1）。EP=8 时每 rank 1 个专家、**没有同专家副本**，专家梯度不需要 all-reduce。
4. EP=4/DP=2 下一个 MoE 层一个 step 的通信序列（按时间顺序）：

| # | 阶段 | 类型 | process group | 传什么 |
| --- | --- | --- | --- | --- |
| 1 | fwd | `all_to_all_single` | EP 组（4 卡） | counts `[4] int64` |
| 2 | fwd | `all_to_all_single` | EP 组 | dispatch：`x_sorted` 的 hidden |
| 3 | fwd | 无 | — | 本地 expert FFN |
| 4 | fwd | `all_to_all_single` | EP 组 | combine：expert 输出回原 rank |
| 5 | bwd | `all_to_all_single` | EP 组 | combine 的反向 |
| 6 | bwd | `all_to_all_single` | EP 组 | dispatch 的反向 |
| 7 | 梯度同步 | `all_reduce` | 专家 DP 组（2 卡，如 {2,6}） | 该 rank 持有的那个 expert 的梯度 |
| 8 | 梯度同步 | `all_reduce` | 全 8 卡 | attention / embedding / norm 的梯度 |

**0–4 分 rubric**

- 0：画不出 rank 到 EP 组的映射。
- 1：EP 组划分对，但专家梯度的 DP 组写错（最常见是写成全 8 卡）。
- 2：第 1 问整表正确。
- 3：第 1、2、3 问全对，第 4 问的通信序列基本正确。
- 4：以上全对，通信序列含反向的两次 a2a 与**两组不同的梯度 all-reduce**，并主动说明第 7、8 两行为什么不能合并成一次 DDP all-reduce（对应 M02-E-02 第 4 问）。

**常见误区**

- 把"EP 组"和"专家梯度 DP 组"搞混：EP 组是**同一份 token 在其中被分发的组**（组内专家互不相同），专家 DP 组是**持有同一个 expert 副本的组**（跨 EP 组）。EP=4 时前者是 {0,1,2,3}，后者是 {0,4}。
- 忘记反向也有两次 a2a，只画 3 次通信。

**追问**

- 如果 EP 组改成 {0,2,4,6}/{1,3,5,7}，第 1 问表里哪几列会变、哪几列不变？
- EP=8 时非专家参数的 all-reduce 还需要吗？为什么？

**定位**：`01_FOUNDATIONS.md` §3.2（8 卡 rank 映射表）、§2.4、§8 第 1 条；`lab/` 的 `ep_moe.py`。

---

### M02-A-02

**关键评分点**（全部 `计算`，$B=8$、$T=512$、$d=768$、fp16 = 2 B）

1. $N_{tok} = 8\times512 = \mathbf{4096}$ token/rank/层。`x_flat` = `[4096, 768] fp16`，$4096\times768\times2 = 6{,}291{,}456\,\text{B} = \mathbf{6.29\,MB}$。
2. `scores` = `[4096, 4] fp32`（softmax 在 autocast 下升 fp32）$= 4096\times4\times4 = 65{,}536\,\text{B} = \mathbf{64\,KB}$。`c_r` = `[4] int64` $= \mathbf{32\,B}$。
3. 发送上限 **6.29 MB**。是"上限"因为 top-1 下每个 token 只产生一份副本，总量固定为 4096 个 token 的 hidden；但其中路由到**本 rank 自己那个专家**的部分是自块（本地拷贝），真正跨卡上网的字节数 ≤ 6.29 MB，具体多少取决于当步的 `c_r` 分布。
4. 接收字节：
   - （a）完全均衡：EP 组 4 个 rank 各发来 $4096/4 = 1024$ 个 token → $4\times1024 = 4096$ token $\times1536\,\text{B} = 6{,}291{,}456\,\text{B} = \mathbf{6.29\,MB}$。
   - （b）上限：4 个 rank 的全部 token 都路由到本 rank 的专家 → $E_p\times N_{tok} = 4\times4096 = 16{,}384$ token $\times1536\,\text{B} = 25{,}165{,}824\,\text{B} = \mathbf{25.2\,MB}$。
5. `gate` 输出 = `[n_recv, 2432] fp16`。上限情形 $16{,}384\times2432\times2 = 79{,}691{,}776\,\text{B} = \mathbf{79.7\,MB}$。一次 FFN 前向里有 **2 个**这样的张量（`gate_proj` 与 `up_proj` 各一个，相乘后才进 `down_proj`），即热点时约 159 MB 的激活——**这才是热点专家最先撑爆显存的地方，比 25.2 MB 的通信缓冲大 3 倍以上**。
6. top-2：
   - 发送量 $\times2 = \mathbf{12.6\,MB}$（8192 个 token-copy）。
   - 期望接收 $\times2 = \mathbf{12.6\,MB}$（8192 个 token-copy）。
   - 接收上限 **不变，仍是 25.2 MB**（16,384 token-copy）。原因：top-2 要求每个 token 选**两个不同的专家**，所以任何单个专家从一个 rank 最多收到该 rank 的全部 4096 个 token 各一份，$4\times4096 = 16{,}384$ 仍是上界——top-2 抬高了期望值，把分布推向上界，但没有抬高上界本身。

**0–4 分 rubric**

- 0：算不出 4096 token 或 6.29 MB。
- 1：第 1、2 问正确。
- 2：第 1–4 问正确（含均衡期望与上限）。
- 3：第 1–5 问全对，含 79.7 MB 与"有 2 个"。
- 4：以上全对，且第 6 问的"上限不变"给出了**因为 top-2 的两个专家必须不同**这个正确理由（而不是含糊地说"上限本来就是最坏情况"）。

**常见误区**

- 把接收上限算成 $4096\times1536$（只算了本 rank 自己的 token），漏掉 EP 组内其余 3 个 rank 也会全发过来。
- 把 top-2 的接收上限也乘 2，算成 50.3 MB。

**追问**

- 反向也要走两次 a2a，那么一个 MoE 层一步搬的 hidden 总字节数在均衡时是多少？
- 如果把 $T$ 从 512 提到 1024，第 4 问的两个数各变成多少？哪个量的**比值**不变？

**定位**：`01_FOUNDATIONS.md` §3.1（shape 与字节表）、§1.2、§5.2；`lab/` 的 `ep_moe.py`。

---

### M02-A-03

**关键评分点**（全部 `计算`，$W=8$）

1. **DDP**：梯度是 fp32 → 总负载 $63{,}912{,}192\times4\,\text{B} = 255{,}648{,}768\,\text{B} = \mathbf{255.6\,MB}$。桶数：`bucket_cap_mb=25` 是 **MiB** = 26.2 MB，$255.6/26.2 = 9.75$，加 1 MiB 首桶 → **约 11 个桶**。Ring all-reduce 每 rank 收发各 $\frac{2(W-1)}{W}\times255.6 = 1.75\times255.6 = \mathbf{447\,MB}$。
2. **FSDP**：单个 `MiniMindBlock` 的 $S_\ell = 7{,}374{,}528\times2\,\text{B} = 14{,}749{,}056\,\text{B} = \mathbf{14.75\,MB}$；embedding $= 4{,}915{,}200\times2\,\text{B} = 9{,}830{,}400\,\text{B} = \mathbf{9.83\,MB}$。每 rank 实际收发 $\frac{7}{8}S_\ell$：block **12.9 MB**、embedding **8.6 MB**。
3. **FULL_SHARD**：每步 9 次 fwd AllGather + 9 次 bwd AllGather + 9 次 ReduceScatter = **27 次**。单轮 9 个单元的负载 $= 8\times14.75 + 9.83 = 127.8\,\text{MB}$，三轮总负载 $= 3\times127.8 = \mathbf{383\,MB}$；每 rank 实际 $= \frac{7}{8}\times383 = \mathbf{335\,MB}$。
4. **SHARD_GRAD_OP**：**18 次**（9 AllGather + 9 ReduceScatter）；总负载 $= 2\times127.8 = \mathbf{256\,MB}$；每 rank 实际 $= \frac{7}{8}\times256 = \mathbf{224\,MB}$。
5. 对照表：

| 方案 | 每 rank 实际字节 | 每步集合通信次数 |
| --- | --- | --- |
| DDP | 447 MB | 约 11 |
| FSDP `FULL_SHARD` | 335 MB | 27 |
| FSDP `SHARD_GRAD_OP` | 224 MB | 18 |

只看字节量应该预测 `SHARD_GRAD_OP` 最快、`FULL_SHARD` 次之、DDP 最慢。**这与本周的工作假设相反**（假设是 DDP 最快、`FULL_SHARD` 最慢，`SHARD_GRAD_OP` 居中）。因此差异必须由**与字节量无关的项**解释：每次集合通信的固定延迟 $\alpha\times$次数、`forward_prefetch=False` 造成的不可重叠等待、以及 FSDP 的 CPU 侧 flatten/unflatten 与更多 kernel 启动。这正是 M02-T-03 的论点。

**0–4 分 rubric**

- 0：DDP 的字节数按 fp16 算，或算不出 ring 系数。
- 1：DDP 部分正确。
- 2：DDP + FSDP `FULL_SHARD` 正确。
- 3：四问数字全对，对照表完整。
- 4：以上全对，且第 5 问明确指出"字节量预测与假设相反"，并把差异归到延迟×次数 + 重叠度 + CPU 开销三类，而不是含糊地说"FSDP 有额外开销"。

**常见误区**

- 用 fp16 算 DDP 梯度（127.8 MB），进而算出 DDP 只有 224 MB，得出错误的对照。
- 把 ring all-reduce 的系数写成 $\frac{W-1}{W}$（那是 AllGather / ReduceScatter 单向的系数）。all-reduce = reduce-scatter + all-gather，所以是 $\frac{2(W-1)}{W}$。

**追问**

- 如果把 wrap 粒度改成"每 2 层一个单元"，第 3、4 问的次数和每 rank 字节各变成多少？
- `reduce_dtype` 从 fp16 改成 fp32，哪几个数会变、变多少？

**定位**：`01_FOUNDATIONS.md` §3.3、§3.4、§5.1；`lab/` 的 `train_ddp.py`、`train_fsdp.py`。

---

### M02-A-04

**关键评分点**（全部 `计算`；$N=4096$、$k=1$、$E=4$、$E_p=4$、fp16）

1. $C = \text{cf}\cdot\frac{Nk}{E} = \text{cf}\cdot\frac{4096\times1}{4} = \text{cf}\times1024$。cf = 1.0 → $C = \mathbf{1024}$；cf = 1.5 → $C = \mathbf{1536}$。
2. **cf = 1.0**：expert 0 收到 2867 > 1024，丢 $2867-1024 = \mathbf{1843}$ 个 token，占本 rank 的 $1843/4096 = \mathbf{45.0\%}$。空槽位 $=(1024-410)+(1024-410)+(1024-409) = 614+614+615 = \mathbf{1843}$。两个数**恰好相等**，因为 cf = 1.0 时总容量 $4\times1024 = 4096 = N$，token 总数与槽位总数相同，于是"挤不进去的"必然等于"没被填的"。这就是 cf = 1.0 下"丢 token"与"padding 浪费"是同一枚硬币两面的证明。
3. **cf = 1.5**：丢 $2867-1536 = \mathbf{1331}$ 个，占 $1331/4096 = \mathbf{32.5\%}$。占用槽位 $=1536+410+410+409 = 2765$，总容量 $4\times1536 = 6144$，空槽位 $= 6144-2765 = \mathbf{3379}$。丢的少了，浪费的多了：**计算预算从 4096 涨到 6144（1.5 倍），丢弃率从 45.0% 降到 32.5%**。
4. **dropless**：EP 组 4 个 rank 都发来 2867 个 token 给 expert 0 → rank 0 收到 $4\times2867 = \mathbf{11{,}468}$ token $\times1536\,\text{B} = 17{,}614{,}848\,\text{B} = \mathbf{17.6\,MB}$。与 A-02 对比：均衡期望 6.29 MB（4096 token）< 17.6 MB < 上限 25.2 MB（16,384 token）——处在期望与上限之间、明显偏向上限。
5. 热点比 $\max_e f_e\cdot E = \frac{2867}{4096}\times4 = 0.6999\times4 = \mathbf{2.80}$（均衡时应为 1.00）。cf = 1.5 + drop 下每个专家的 token 上界是 $E_p\times C = 4\times1536 = 6144$，dropless 下是 11,468 → **计算量上界改善 $11{,}468/6{,}144 = 1.87$ 倍**。代价用一个数字表示就是**丢掉了 32.5% 的 token**（它们的前向输出为 0、只走残差，且不产生 FFN 梯度）。

**0–4 分 rubric**

- 0：写不出 $C$ 的定义式或代错 $N$。
- 1：算出两个 $C$ 与 cf=1.0 的丢弃数。
- 2：第 1–3 问全对，含"丢弃数 = 空槽数"的观察。
- 3：五问数字全对。
- 4：以上全对，且第 2 问给出了"cf=1.0 时总容量恰等于 token 总数"这个**结构性理由**（而不是说"巧合"），第 5 问把 1.87 倍与 32.5% 并列成一组明确的取舍数对。

**常见误区**

- 把 $N$ 代成全局 token 数（8 卡 × 4096）。capacity 是**每 rank 每专家**的口径，$N$ 是本 rank 本层的 token 数。
- 第 4 问只算 2867 个 token（忘了 EP 组内 4 个 rank 都在发）。

**追问**

- 如果 4 个 rank 的计数向量各不相同（只有一个 rank 有热点），第 4、5 问的答案会怎么变？木桶效应还成立吗？
- 被丢的 token 在反向里对 router 有没有梯度贡献？这对"热点会不会自我强化"意味着什么？

**定位**：`01_FOUNDATIONS.md` §4.6、§3.1、§5.2、§2.3；`lab/` 的 `router_stats.py`、`ep_moe.py`。

---

## Debug

### M02-D-01

**关键评分点**

1. 破坏的是 **I2（split 一致）**：对每一对 $(r,j)$，rank $r$ 的 `input_split_sizes[j]` 必须等于 rank $j$ 的 `output_split_sizes[r]`。
2. "等约 30 分钟才退出"精确对应 NCCL process group 的**默认 `timeout=1800s`**。这条证据同时说明：进程没死、没有异常抛出、是集合通信在等一个永远不会到的字节数——即典型的 collective 不匹配，而不是死循环或数据加载卡住（后者不会在 1800 s 这个整数点上集体退出）。利用率 100% 是 NCCL 的忙等自旋，不代表在算东西。
3. 三个假设（按可能性排序）+ 区分实验：
   - **H1：counts 交换与 dispatch 的 split 不一致**（最常见：`recv_counts` 用错方向，把 `c_r` 当成了 `c'_r`）。区分实验：在 gloo/CPU 2 进程下打印每个 rank 的 `counts` 与 `recv_counts`，检查 `counts[r][j] == recv_counts[j][r]`。
   - **H2：某个 rank 提前退出了循环或跳过了某次集合通信**（例如空专家分支被 `if` 短路，或 `new_group` 没有全员调用）。区分实验：给每次集合通信前加带 rank 与序号的日志，看哪个 rank 的序号先停；或 `py-spy dump` 看各 rank 的栈停在哪个 collective。
   - **H3：EP 组构造本身错位**（各 rank 对 `new_group` 的调用顺序不同，或组内成员不一致）。区分实验：每 rank 打印 `dist.get_rank(group=ep_group)` 与 `dist.get_world_size(group=ep_group)`，检查组内编号是 0..3 的一个排列。
4. 两条断言：`assert sum(input_split_sizes) == input.shape[0]`、`assert sum(output_split_sizes) == output.shape[0]`。它们**只覆盖本地自洽性，不覆盖 I2 的全部**——I2 是跨 rank 的成对条件，本地断言看不到对端的列表。缺的那部分要靠：把 `counts` 做一次 `all_gather` 成完整的 $E_p\times E_p$ 矩阵，断言它与各 rank 的 `recv_counts` 转置一致（这一次额外通信只在 debug 模式下开）。
5. 复现方案：**本机 CPU（ResearchAgentPy310）+ gloo 后端 + 2 个进程**，跑 `tests/test_ep_cpu.py` 里的 permute/unpermute 往返用例；断言的是 I3（EP=1 与原实现逐元素相等）与第 4 问的 split 矩阵一致性。gloo 不占 8 卡配额，且同样会在 split 不一致时 hang，可以先在这里把逻辑错误全部清掉再上 V100。

**0–4 分 rubric**

- 0：只说"是通信问题"，给不出不变量。
- 1：认出 I2，但没有用上 1800 s 这条证据。
- 2：I2 + timeout 证据 + 至少两个带区分实验的假设。
- 3：五问基本完整，断言写法正确。
- 4：以上全对，且明确指出**本地断言不足以覆盖跨 rank 的 I2**并给出 all_gather 矩阵检查，同时给出不占配额的 CPU 复现路径与它能/不能验证的边界（gloo 能验逻辑，不能验 NCCL / fp16）。

**常见误区**

- 看到"利用率 100%"就判断成"在算，可能只是慢"。NCCL 等待是自旋忙等，利用率高不代表有有效计算。
- 认为加上本地的 `sum(splits) == shape[0]` 断言就能防住 hang。它防不住"两个 rank 各自自洽但互相不匹配"的情况。

**追问**

- 打开 `NCCL_ASYNC_ERROR_HANDLING=1` 之后，这个 bug 的表现会变成什么？排查成本降低了多少？
- 如果 hang 发生在**反向**的那次 a2a 而不是前向，你的定位路径会怎么改？

**定位**：`01_FOUNDATIONS.md` §2.4（I2/I3）、§6 第 2/5 条、§3.5；`lab/` 的 `ep_moe.py`、`tests/test_ep_cpu.py`。

---

### M02-D-02

**关键评分点**

1. 破坏的是"**分片布局 = 函数(world_size, wrap 结构, 参数展平顺序)**"这一前提。FSDP 的 `flat_param` 是把一个 wrap 单元内所有参数展平后按 world_size 均分的产物，8 卡存下来的每片长度 $\approx L/8$，4 卡要的是 $L/4$——直接加载必然 `size mismatch`。
2. 三种保存方式：
   - `FULL_STATE_DICT(rank0_only=True)`：rank 0 把完整未分片的 state dict 收齐落盘。恢复时任意 world size 都能加载，最简单。代价：rank 0 需要容纳全模型（64M 无压力，27B 会 OOM）、保存时有一次全量 gather。
   - `LOCAL_STATE_DICT`：每 rank 存自己的 `flat_param` 分片，最快最省，但**分片布局硬编码在文件里**，只能以相同 world size + 相同 wrap 结构恢复。8→4 卡直接不可用。
   - `SHARDED_STATE_DICT` + `torch.distributed.checkpoint.save_state_dict/load_state_dict`：存的是带全局元信息的分片，加载时由 checkpoint 层按新的 world size **重新切分**。代价：多一层 API 和文件布局。
   - 本周 lab 的默认选择：**`SHARDED_STATE_DICT` + `distributed.checkpoint`**——因为本周的关键故障练习就是 8→4 卡恢复，而它是三者里唯一"既能重切分、又能外推到 27B"的；`FULL_STATE_DICT(rank0_only)` 作为 64M 上的对照与 fallback。
3. 优化器状态用 `FSDP.optim_state_dict(...)` 保存、`FSDP.optim_state_dict_to_load(...)` 加载。跳过它们直接 `optimizer.load_state_dict()` 会把 8 卡布局的 `exp_avg` / `exp_avg_sq` 塞给 4 卡的参数分片：轻则 shape 报错，重则**形状恰好对得上但语义错位**（Adam 的一阶/二阶矩对应到了别的参数），训练不报错但 loss 立刻跳变。
4. 换算式 `step_new = step * saved_ws // current_ws`，代入 $1000\times8//4 = \mathbf{2000}$。前提：**每 step 每 rank 消费的样本数不变**（`batch_size` 与 `accumulation_steps` 都不变），此时"总消费样本数"守恒，step 只是被重新计价。若同时把 `batch_size` 从 8 改成 16，前提破坏——global batch 从 $8\times8=64$ 变成 $4\times16=64$，恰好相等，此时 `step` 应该**保持 1000 不变**，而换算式会给出 2000，是错的。一般情况下应该按**已消费的样本数（或 token 数）** 重算：$\text{step}_{new} = \frac{\text{samples\_consumed}}{W_{new}\cdot b_{new}\cdot A_{new}}$。
5. 两条可提交判据（符合公司边界）：
   - **loss 连续性**：恢复后第 1 步的 loss 落在中断前最后 20 步的 $[\min,\max]$ 区间内 → 记 `PASS/FAIL`，不记 loss 数值。
   - **优化器状态一致性**：恢复后 `exp_avg` 的全局范数与中断前最后一步的比值落在容差内 → 只记这个**比值**与 `PASS/FAIL`。（外加 `scaler.get_scale()` 恢复前后**相同**这一条布尔。）

**0–4 分 rubric**

- 0：说不清 FSDP 分片依赖 world size。
- 1：认出分片问题，但三种 `StateDictType` 说不清。
- 2：第 1、2 问正确，选了 `SHARDED_STATE_DICT` 并给了理由。
- 3：第 1–4 问全对，含 step=2000 与它的前提。
- 4：以上全对，且第 4 问识别出"`batch_size` 同时改变时换算式会给错答案"并给出按样本/token 数重算的正确形式；第 5 问的两条判据都只带出抽象量。

**常见误区**

- 认为 `optimizer.state_dict()` 可以像模型一样"存了就能换卡数加载"。优化器状态与参数分片一一绑定，必须走 FSDP 的专用 API。
- 把 step 换算方向搞反（写成 $1000\times4//8 = 500$）。卡数变少 → 每 step 消费的样本变少 → 同样的样本量对应**更多** step。

**追问**

- 如果 8 卡与 4 卡的 wrap 结构不同（比如一边按 block wrap、一边只有根 FSDP），`SHARDED_STATE_DICT` 还能恢复吗？为什么？
- 你怎么在 CPU 上（不占 8 卡）先验证 reshard 逻辑？要 mock 掉什么？

**定位**：`01_FOUNDATIONS.md` §3.5（FSDP / MiniMind checkpoint 行）、§6 第 4/10 条、§6 末段（loss 连续性）；`lab/` 的 `ckpt_reshard.py`、`tests/test_reshard_cpu.py`。

---

### M02-D-03（Gate 跨场景 debug）

**关键评分点**

1. 木桶公式：EP 组内 step time $= \max_r\big(t^{a2a} + t^{expert}_r\big)$，且 $t^{expert}_r \propto n_{recv,r}$。所以 step time 由**收到 token 最多的那个 rank**决定；其余 rank 的时间以 `t_wait` 的形式出现在 combine 的 a2a 上。症状里"$t_{compute}+t_{wait}$ 各 rank 大体相等"正是这个公式的直接推论——它证明这**不是**通信本身变慢，而是负载偏了。
2. 四步定位：
   - **① 计时分布**：看逐 rank 的 `t_compute` / `t_wait`。健康形态是 4 个 rank 的 `t_compute` 接近、`t_wait` 都很小；异常形态是 1 高 3 低 + 3 个大 `t_wait`。报告量：$\max_r t_{compute}/\text{median}_r t_{compute}$。
   - **② 每专家负载 $f_e$**：健康 $f_e \approx 1/E = 0.25$；异常是某个 $f_e$ 显著偏大。报告量：热点比 $\max_e f_e\cdot E$，健康 ≈ 1.0。
   - **③ `aux_loss`（乘 $\lambda$ 之前的量）**：健康 ≈ 1（下界），异常趋向 $E=4$。这一步用来确认 ② 不是采样噪声。
   - **④ router logits 的熵 / `scores` 分布**：坍缩时熵远低于 $\log E$，且 `gate` 权重的某一行范数持续增长（富者愈富）。这一步区分"数据本身就该这样路由"与"router 参数跑偏了"。
3. `router_stats.py` 打印的三个量与健康取值：$f_e$（$\approx 1/E = 0.25$ 各项）、热点比 $\max_e f_e\cdot E$（$\approx 1.0$，$\le 1.5$ 可接受）、`scores` 的平均熵（$\approx \log E = 1.386$ nat，被拉平时接近这个上界，坍缩时趋向 0）。
4. 两个互斥假设 + 区分实验：
   - **H-A：router 自我强化（富者愈富）**——早期随机偏置被主 loss 放大，$\lambda=5\times10^{-4}$ 太小压不住，所以要跑到几百步才显现。区分实验：把 $\lambda$ 提到 $10^{-2}$ 重跑同样步数，看热点比是否不再随步数上升。
   - **H-B：数据分布漂移**——第 300 步之后 dataloader 进入了一段内容同质的数据，token 本来就该路由到同一个专家。区分实验：固定一个 batch **反复喂**（关掉 sampler 推进）跑 500 步，若热点比仍随步数上升则是 H-A，若保持不变则是 H-B。
5. 加大 aux 系数后的主指标必须是 **`logits_loss`**（与 dense 基线以及 $\lambda=0$ 的 MoE 基线对比）。不能拿 `aux_loss` 当主指标，因为 `aux_loss` 正是被优化的那一项——把它压到 1 是**定义上必然**的，不构成"变好"的证据；只有 `logits_loss` 没有变差，"修复"才不是白赚的（否则就是用主任务质量换了均衡，见 M02-P-01）。
6. 公司边界：**能写**的字段——$\max_r t_{compute}/\text{median}_r t_{compute}$ 的比值、热点比 $\max_e f_e\cdot E$、NCCL kernel 时间占比的三档区间（<25% / 25–50% / >50%）、`PASS/FAIL`、"第 N 步后单调上升"这类形状描述。**不能写**的字段——绝对 step time（ms）、绝对显存（GB）、profiler trace 文件、`nvidia-smi topo -m` 原文、日志原文截图、checkpoint。

**0–4 分 rubric**

- 0：只说"某个专家太热"，给不出定位路径。
- 1：写出木桶公式，但四步路径不完整。
- 2：木桶公式 + 四步路径 + `router_stats.py` 的三个量。
- 3：以上全对，两个互斥假设都带可执行的区分实验。
- 4：以上全对，且第 5 问明确论证"`aux_loss` 被压低是定义上必然、不构成证据"，第 6 问的可写/不可写字段各举了两例且完全符合公司边界。

**常见误区**

- 把"3 个 rank 的 `t_wait` 大"当成通信慢，去查网络/拓扑。$t_{compute}+t_{wait}$ 各 rank 相等这条证据已经排除了通信本身变慢。
- 用 `aux_loss` 下降来宣布问题解决。它下降是被优化的直接结果，不是主任务变好的证据。

**追问**

- 如果 4 个 rank 的 `t_compute` 都变高、`t_wait` 都很小，同样是 step time 变 3 倍，你的定位路径怎么改？
- 热点比从 2.8 降到 1.2 但 `logits_loss` 比 dense 基线差且不收窄，你会怎么判定这次修复？

**定位**：`01_FOUNDATIONS.md` §5.2、§4.5、§2.3、§6 第 6/7 条、§8 第 3 条；`00_WEEK_CARD.md`「三类环境边界」；`lab/` 的 `router_stats.py`、`ep_moe.py`。

---

### M02-D-04

**关键评分点**

1. 正常情况下 FSDP 相对 DDP 应省三类量，每类**约 $W=8$ 倍**：**参数**（每 rank 只持 $1/8$）、**梯度**（ReduceScatter 后每 rank 只持 $1/8$）、**优化器状态**（Adam 的 `exp_avg`/`exp_avg_sq` 也只对自己那片存）。折算到 64M 模型：DDP 每卡约 $255.6 + 255.6 + 511.3 \approx 1.02\,\text{GB}$ 训练态，FSDP 理想情况约 $128\,\text{MB}$（`计算`）。
2. 三个可能原因，各落到具体构造参数/结构：
   - **wrap 退化成单一根 FSDP**（`auto_wrap_policy` 没生效或写错）→ 没有分层，前向那一次 AllGather 直接拉齐全模型参数，峰值 = 全模型完整参数，等于没省。
   - **`limit_all_gathers=False`**（或未设）→ 多个单元的 AllGather 缓冲同时在飞，未使用的完整参数堆积，峰值被 in-flight 缓冲抬高。
   - **`use_orig_params` 与优化器/参数分组的组合不当**（例如按名字分组 weight decay 时导致 flatten 失效或参数被重复持有），以及 `MixedPrecision` 下同时存在 fp32 master 与 fp16 通信副本。
3. 让前向峰值直接等于"整个模型完整参数"的是**第一条：wrap 退化成单一根 FSDP**。不跑满 8 卡的验证方法：在**本机 CPU + gloo 2 进程**下构造模型并 wrap，然后遍历 `FSDP.fsdp_modules(model)` 打印被 wrap 的子模块个数——期望是 9（8 个 block + embedding），若是 1 就命中；同时打印每个单元 `flat_param.numel()`，检查它们的和 $\times W$ 是否等于总参数量。这一步只验证结构，不需要 GPU。
4. 单步时间更长**不一定**同根因：
   - 若是**同一个根因**（wrap 退化）：集合通信次数会**减少**（每步只有 2 次而不是 27 次），profiler 里 NCCL kernel 占比反而低，但单次通信极大、完全无法与计算重叠 → 证据形态是"次数少 + 单次巨大 + 峰值显存 = 全模型"。
   - 若是**两个独立问题**：wrap 正常（27 次），显存高来自 in-flight 缓冲，慢来自 $27\alpha$ + `forward_prefetch=False` → 证据形态是"次数正常 + NCCL 占比高 + 峰值只比理论值高一点"。
   两者用"每步集合通信次数"这一个量就能分开。
5. 修复后重新记录的四个抽象量：$m_{FSDP}/m_{DDP}$（比值）、$r_1 = t_{FSDP}/t_{DDP}$（比值）、$r_2 = t_{SGO}/t_{DDP}$（比值）、profiler 里 NCCL kernel 时间占比的**三档区间**（<25% / 25–50% / >50%）。四个都是比值或档位，不含绝对 ms/GB，符合公司边界。

**0–4 分 rubric**

- 0：说不出 FSDP 本该省什么。
- 1：第 1 问正确，能给出一个原因。
- 2：三个原因都落到了具体参数或结构上。
- 3：第 1–4 问全对，第 4 问给出了两种情况各自的证据形态。
- 4：以上全对，且第 3 问的 CPU 验证方案具体到"数被 wrap 的单元个数 / 查 `flat_param.numel()`"这种可执行动作，第 5 问的四个量全部符合边界并说明了取值形式。

**常见误区**

- 认为 FSDP 一定省显存，于是把峰值更高归因于"测量方式不对"，不去查 wrap 结构。
- 把"集合通信次数少"当成好事。在这里次数骤降恰恰是 wrap 退化的**指纹**。

**追问**

- 如果 `activation checkpointing` 也打开了，第 1 问的三类量之外还会动哪一类显存？它会不会掩盖 wrap 退化？
- $m_{FSDP}/m_{DDP}$ 修好后你预期落在什么区间？低于多少你反而要怀疑测量口径？

**定位**：`01_FOUNDATIONS.md` §3.4、§5.1、§6 第 9 条；`lab/` 的 `train_fsdp.py`。

---

### M02-D-05

**关键评分点**

**异常 A**

1. 判据（至少两个可打印量）：① `clip_grad_norm_` 的返回值 `total_norm`——若它在 scale 开始下滑**之前**就已经是 `inf`/极大，说明梯度本身炸了，是发散；若它一直有限、只是偶尔某步 inf，那是 fp16 溢出。② `torch.isfinite(logits).all()`（或对某层激活做同样检查）——**前向就出 NaN** 说明与梯度缩放无关。日志形态区别：**偶尔跳步**= scale 掉一两档后停住并在 2000 步后回升（`growth_interval=2000`、`growth_factor=2`）；**反复跳步** = 单调减半、从不回升，scale 掉到 $<1$。本例"一路减半、从未回升、20 步后 NaN"是典型的**发散**，不是动态范围问题。
2. 换格式不能修，因为发散来自 lr / 初始化 / 数据，梯度在**任何格式下**都是发散的，fp32 只是让它慢一点变成 inf。V100 上**能换**的：fp32（math SDPA + 关掉 autocast）作为 reference；**不能换**的是 **bf16**——`is_bf16_supported()` 要求 major ≥ 8，V100 是 sm70，进 autocast 就抛 `RuntimeError`。所以本周唯一的"更宽动态范围"退路是 fp32，而 fp32 只能用来**诊断**，不是生产路径。
3. 机制：`loss.item()` 打印的是**前向算出的 loss 张量**，它在 `scaler.step()` 之前就已经存在，与是否跳步无关；`scaler.step(optimizer)` 内部检查到 inf/NaN 时**直接不调用 `optimizer.step()`**，参数保持不变，而 `lr_scheduler` 通常在循环外无条件推进，所以 lr 照常走。直接证明"参数没动"的检查：在 `scaler.step()` 前后各取某一层权重的 `.sum()` 或 checksum 做对比，或记录 `optimizer.state[p]['step']` 是否递增。
4. 破坏的是 **I1**。$10^{-2}$ **不能**用 fp16 累加顺序解释：合理差异量级是 $10^{-3}$ **相对**误差，而且更关键的是——**第 1 个 optimizer step 的 loss 是在参数完全相同（DDP 从 rank 0 广播了 `state_dict`）的模型上算出的前向结果，此时还没有任何 all-reduce 影响过参数**。所以第 1 步的 loss 差只可能来自"喂进去的样本不同"或"loss 归约口径不同"，与通信数值路径无关。这条推理是判断的核心依据。
5. 四个不合理来源 + 第 1 步就能打印的证据：
   - **样本集合不同**（`DistributedSampler` 按 $W$ 切分）→ 打印每个 step 各 rank 拿到的**样本索引并集**并比对。
   - **accumulation 边界不对齐**（2 卡的 4 个 micro-batch 与 8 卡的 8 个 rank 覆盖的不是同一批样本）→ 打印 `W * A * b` 与本 step 实际消费的样本数。
   - **scaler 初值/状态不同**（比如一边从 `_resume.pth` 恢复了 scale）→ 打印 `scaler.get_scale()`。
   - **loss 归约口径不同**（打印的是 rank 0 的局部 loss 还是全局平均）→ 打印归约前后的两个值，确认对比的是同一口径。
   （第五个候选：`clip_grad_norm_` 在 `unscale_` 之前调用——但它影响的是更新而不是第 1 步的 loss 打印，可作为"影响第 2 步起"的候选。）
6. 有没有可能同根因：**可能但不太像**。一个共同根因的候选是"两次运行喂的数据不同 + 其中一条数据路径引入了异常样本导致发散"。证伪观测：把异常 B 里的两种配置都改成**同一份固定 batch 反复喂**（关掉 sampler 推进）——若异常 B 消失而异常 A 依然出现，则两者独立；若两者同时消失，则指向数据路径这个共同根因。

**0–4 分 rubric**

- 0：把异常 A 当成"fp16 不够用，换 bf16"（在 V100 上根本不可行）。
- 1：A 的判据正确，但说不清"发散 vs 溢出"的日志形态区别。
- 2：A 的 1–3 问正确。
- 3：A、B 都答到位，B 认出 I1 并列出四个不合理来源。
- 4：以上全对，且第 4 问给出了**"第 1 步参数还完全相同、所以通信数值路径不可能是原因"**这条决定性推理；第 6 问给出了一个真正能证伪的观测。

**常见误区**

- 用 fp16 累加顺序解释 $10^{-2}$ 的第 1 步差异。量级差一个数量级，而且第 1 步的 loss 根本还没经过任何梯度通信。
- 认为跳步时 `loss.item()` 也应该异常。loss 打印与是否 `optimizer.step()` 完全解耦。

**追问**

- 如果 scale 稳定在 1024 而不是 65536，需要担心吗？这属于哪种形态？
- 异常 B 修好之后，你预期两种配置的 loss 残差落在什么量级？超过多少你会重新怀疑？

**定位**：`01_FOUNDATIONS.md` §2.2、§4.1、§4.4、§4.7、§2.1（I1）、§6 第 3/8 条；`lab/` 的 `train_ddp.py`、`tests/test_ddp_gloo_cpu.py`。

---

## Design

### M02-P-01

**关键评分点**

1. MiniMind 默认 `router_aux_loss_coef = 5e-4`，**不在**扫描点里，应该补进去（它是"当前基线"，没有它就无法判断扫描结论相对现状是改进还是退步）。假设：$\lambda$ 增大时，**热点比 $\max_e f_e\cdot E$ 单调下降**（趋向 1.0），**`logits_loss` 先基本不动、过了某点后单调上升**（变差）。两条曲线的"交叉"出现在热点比已接近 1 但 `logits_loss` 开始抬头的区间——`估算`落在 $10^{-3}\sim10^{-2}$ 之间，具体位置就是这次扫描要测的东西。
2. **主指标：`logits_loss`**（相对 dense 基线的差值或比值）。**副指标：热点比 $\max_e f_e\cdot E$**。主指标不能选 `aux_loss`，因为它是被 $\lambda$ 直接优化的那一项——加大 $\lambda$ 必然把它压低，这是定义上的必然而不是证据（同 M02-D-03 第 5 问）。
3. 检测"router logits 被拉平"的量：**`scores` 的平均熵** $-\sum_e s_{i,e}\log s_{i,e}$ 在 token 上平均。完全均衡（被拉平）时趋向上界 $\log E = \log 4 = 1.386$ nat；完全坍缩时趋向 0。它与 $f_e$ 互补：$f_e$ 看的是硬选择的结果，熵看的是软分布的形状，$\lambda$ 过大最先动的是熵。
4. 对照基线必须有两个：**dense 基线**（同参数量预算下不带 MoE 的 `logits_loss`）与 **$\lambda=0$ 的 MoE**。没有它们的话，`logits_loss` 的绝对高低无法解释——MoE 相对 dense 本来就可能有差异，而 $\lambda=0$ 给出"完全不约束"这一端。
5. 失败边界（宣布 `INCONCLUSIVE`）：① 各 $\lambda$ 之间 `logits_loss` 的差异小于同一 $\lambda$ 下**不同 seed 的重复运行**的波动；② 训练步数太短，热点比在 $\lambda=0$ 下都还没开始上升（说明还没进入"富者愈富"阶段，扫描测不到东西）；③ 出现了 GradScaler 反复跳步等数值故障，曲线不可比。
6. 可验证的下一步：若 $10^{-2}$ 最优，下一步做 **seed 重复实验（同 $\lambda$ 换 2–3 个 seed）**，用来排除"$10^{-2}$ 的优势只是单次运行的随机波动"这个替代解释；再做一次 **router bias 注入实验**（bias 0/0.5/1/2/4），检验 $\lambda=10^{-2}$ 在人为热点下是否仍能把热点比压回 1.2 以内——排除"这个 $\lambda$ 只是恰好适配了当前数据的天然分布"。

**0–4 分 rubric**

- 0：主指标选了 `aux_loss`。
- 1：主/副指标选对，但没有对照基线。
- 2：指标 + 基线 + 熵这个量都正确。
- 3：六问基本完整，失败边界具体可判。
- 4：以上全对，且第 5 问的失败边界里包含"seed 重复波动"这个统计层面的判据，第 6 问的下一步明确说明了它排除的是哪个替代解释。

**常见误区**

- 忘了把默认值 `5e-4` 补进扫描，结论无法与现状对比。
- 用单次运行的 `logits_loss` 差异下结论，不做 seed 重复。

**追问**

- 如果热点比在所有 $\lambda$ 下都接近 1.0，这次扫描说明了什么？你会改哪个实验条件重做？
- $\lambda$ 的最优值会随 `num_experts` 变化吗？从 aux loss 的取值范围 $[1,E]$ 推一下。

**定位**：`01_FOUNDATIONS.md` §4.5、§2.3、§6 第 6/7 条；`00_WEEK_CARD.md` Day 4；`lab/` 的 `router_stats.py`。

---

### M02-P-02

**关键评分点**

1. 影响的是符号 **$\beta_{path}$**（该 group 内的有效带宽），出现在 $t^{a2a} \approx \alpha + \frac{(E_p-1)}{E_p}\frac{S}{\beta_{path}}$ 里。$\beta_{path}$ 由**组内最慢的那条链路**决定：只要一对 rank 之间要走 PCIe（或跨 CPU），整个 all_to_all 的有效带宽就被它拉低——因为 all_to_all 要求组内**每一对**都完成传输。
2. 对照实验设计：
   - **跑什么**：只跑 EP=4 的短 smoke（`--max-steps` 设小），A 组与 B 组各跑一次，交替顺序跑两轮以抵消热身/漂移。
   - **控制变量**：模型配置、$B$、$T$、`num_experts`、top-k、dtype、seed、步数、是否 compile 全部固定；只改 `new_group` 的成员列表。
   - **记录**：$t^{a2a}_{groupA} / t^{a2a}_{groupB}$ 这**一个比值**（由 `ep_moe.py` 的计时器给出），外加两组各自的 `t_wait` 分布形状是否一致（用来确认差异来自通信而不是负载）。
   - **判据**：比值 **> 1.2** 判为"分组方式显著影响 all_to_all"，落在 $[0.83, 1.2]$ 内判为"本配置下不显著"。
3. 比值 ≈ 1.05 **不能**直接得出"拓扑不重要"。至少两种替代解释：① **通信没暴露**——a2a 时间被 expert 计算掩盖了，或者 a2a 的字节量太小（6.29 MB）以致 $\alpha$ 主导、$\beta_{path}$ 的差异被淹没；② **两种分组恰好跨越了同一条瓶颈链路**，$\beta_{path}$ 本来就相同（这时候要看第三种分组）；③ 测量噪声大于效应（要看重复运行的波动）。正确结论是"**在本模型规模与本字节量下**分组方式不显著"，这是一个有适用范围的结论，不是"拓扑不重要"。
4. **没有影响**。非专家参数的 all-reduce 用的是**全 8 卡**的默认 group，成员固定，与 EP 组怎么分无关。这条也是第 2 问里一个有用的内部对照：如果全 8 卡 all-reduce 的时间在两组实验间也变了，说明有其他变量没控住。
5. 固化位置与形式：把 EP 组的构造做成 `ep_moe.py` 里一个**可配置的 rank 映射函数**（例如 `--ep_group_layout {contiguous, strided}`，默认 `contiguous`），而不是把某个具体的 rank 列表或拓扑矩阵写死在代码/文档里——这样既保留结论又不带出拓扑信息。失败边界：换机器（不同 NVLink 邻接）**不成立**；EP 度改变（EP=2 或 EP=8）时组的构成变了，**需重测**；top-k 改成 2 后字节量翻倍、$\beta_{path}$ 的权重上升，结论**方向可能不变但阈值要重定**。
6. 周报可带出的字段与确切写法（示例）：
   `EP=4 分组对照：t_a2a(layoutA)/t_a2a(layoutB) = <比值>，判据阈值 1.2，结论 PASS/FAIL；非专家 all-reduce 时间比值 ≈ 1.0（对照量，确认变量受控）；重复 2 轮，方向一致。`
   不带出：任何绝对 ms、`nvidia-smi topo -m` 原文、哪几号卡之间是 NV2。

**0–4 分 rubric**

- 0：说不出 $\beta_{path}$ 或把影响归到 $\alpha$。
- 1：符号正确，但实验设计缺控制变量或判据。
- 2：第 1、2 问完整（含阈值）。
- 3：以上全对，第 3 问给出至少两个替代解释。
- 4：以上全对，且第 5 问的固化形式明确做到"保留结论不保留拓扑"，第 6 问写出了一行可直接粘进周报、完全符合边界的记录。

**常见误区**

- 用"比值接近 1"直接下"拓扑不重要"的结论，不讨论"通信是否暴露"这个前提。
- 把结论以"rank 0/1/2/3 之间是高速链路"这种形式写进 lab 或周报——这是拓扑信息，越界。

**追问**

- 如果把 $T$ 从 512 提到 2048（a2a 字节量 ×4），你预期比值会往哪个方向动？为什么这本身就是一个验证实验？
- 除了 A/B 两种分组，还有哪种分组值得测？它能排除第 3 问的哪个替代解释？

**定位**：`01_FOUNDATIONS.md` §5.1、§5.3、§3.2；`00_WEEK_CARD.md`「已知限制与风险」「三类环境边界」；`lab/` 的 `ep_moe.py`。

---

### M02-P-03

**关键评分点**

1. 显存账（`计算`；RM 大小 1.8B 为 `来源计划假设`）：
   - Policy fp32 训练态：参数 $63{,}912{,}192\times4 = 255.6\,\text{MB}$ + 梯度 $255.6\,\text{MB}$ + Adam 两个矩 $511.3\,\text{MB}$ $\approx \mathbf{1.02\,GB}$。
   - Reference fp16 冻结：$63{,}912{,}192\times2 = \mathbf{127.8\,MB}$。
   - Reward 1.8B fp16 冻结：$1.8\times10^9\times2 = \mathbf{3.6\,GB}$。
   - Policy 用 8 卡 FSDP 分片后每卡约 $1.02\,\text{GB}/8 \approx \mathbf{128\,MB}$，即**每卡省下约 0.9 GB**。**没省下的**：Reference 的 127.8 MB × 8 份、Reward 的 **3.6 GB × 8 份**，以及动态显存（训练前向激活、logits、生成期 KV cache）。也就是说 Policy-FSDP 省的那 0.9 GB 只有 RM 单份体量的四分之一——**RM 才是显存主项**。
2. 放置方案：**rank 0–3 = Policy（组内 FSDP 分片）、rank 4–5 = Rollout（持 policy 的 fp16 推理副本）、rank 6–7 = Reward（持 RM）**。把 Reward 单独拿出来收益最大，因为它是唯一的 GB 级静态项：从"8 卡各 3.6 GB"变成"2 卡各 3.6 GB"，**释放了 6 张卡上共约 21.6 GB**（`计算`），而 Policy-FSDP 在 8 张卡上总共只释放约 7.2 GB。
3. 同步版一轮时序：
   1. rollout 卡（4–5）用**上一轮**的权重生成 response（$T_g$ 次串行前向，KV cache 常驻）。
   2. rollout → reward：`send/recv`（或在 {4,5,6,7} 的组里 `broadcast`）传 `(prompt, response, logprob)`。
   3. reward 卡（6–7）前向打分，把**标量奖励**回传给 policy 组（字节量极小）。
   4. policy 组（0–3）内做 FSDP 前反向 + 更新：组内 AllGather / ReduceScatter。
   5. policy 组以 `StateDictType.FULL_STATE_DICT(rank0_only=True)` 收齐权重，`broadcast` 到 rollout 卡：**fp16 全量 $63{,}912{,}192\times2 = 127.8\,\text{MB} \approx 128\,\text{MB}$**（`计算`）。
   6. 回到第 1 步。
4. 瓶颈在**第 1 步（生成）**。理由不能只是"生成慢"，要说清机制：生成 $T_g$ 个 token 需要 **$T_g$ 次串行前向**，每次只处理 $B\cdot G$ 个 token（本周口径 $B=2$、$G=6$ → 12 个 token）。64M 模型在这种极小 batch 下，每次前向的时间由 **kernel 启动延迟**主导（每层十几个 kernel × 8 层，乘以上千个生成步），而不是 FLOPs；同时 rollout 只有 2 张卡，其余 6 张全程等待。相比之下训练一步是把 $T$ 个位置**并行**算完的一次前反向，FLOPs 虽大但一次就完。
5. 空闲形态：一轮里 rank 0–3 与 6–7 在第 1 步（最长的一步）全程空闲；rank 4–5 在第 4 步（policy 更新）空闲。可测量的抽象量：**每类角色的 `t_busy / t_round`**（三个比值），以及 **$t_{generate}/t_{round}$**。都是比值，符合公司边界。
6. 失败边界与下一步：若观测到 ① $t_{generate}/t_{round}$ 并不占主导（比如 < 0.4），或 ② 权重 broadcast 的 128 MB 成为可观开销（说明轮次太短、生成太少），或 ③ 角色拆分后 policy 组只剩 4 卡导致 FSDP 分片收益不足、反而 OOM——就退回"全复制 + Policy-FSDP"。可验证的下一步：先做一个**只测生成的微基准**（固定 prompt，扫 $T_g \in \{128, 512, 1024\}$，记 $t_{generate}$ 相对 $T_g$ 的斜率是否近似线性），用来确认"串行前向 + kernel 启动主导"这个机制成立，再决定是否值得投入角色拆分那一天。

**0–4 分 rubric**

- 0：给不出三个模型的显存账。
- 1：显存账正确，但说不出"RM 才是主项"。
- 2：第 1、2 问正确，放置方案合理。
- 3：第 1–4 问全对，时序完整且瓶颈论证到了 kernel 启动这一层。
- 4：以上全对，且第 6 问给出了明确的退回条件与一个**能先于大改动执行**的微基准作为可验证下一步。

**常见误区**

- 认为"Policy 用 FSDP 分片"是这里最重要的优化。它省的 0.9 GB/卡远小于 RM 的 3.6 GB/卡。
- 说"生成慢是因为要算很多 token"。生成的 FLOPs 其实很小，慢的原因是**串行**与**每次 batch 太小导致延迟主导**。

**追问**

- 如果把 `num_generations` 从 6 提到 12，第 1 问的哪些数变、第 4 问的瓶颈判断变不变？
- 权重同步用 `FULL_STATE_DICT` broadcast 128 MB；有没有更省的做法？它会引入什么新的一致性风险？

**定位**：`01_FOUNDATIONS.md` §5.4、§3.5（GRPO 角色行）、§6 第 12 条；`00_WEEK_CARD.md` Day 5；`lab/` 的 `grpo_roles.py`。

---

### M02-P-04

**关键评分点**

1. 实验矩阵（`计算`，来自 §3.2）：

| 配置 | EP 组 | 每 rank 专家数 | `num_experts` |
| --- | --- | --- | --- |
| EP=1, DP=8 | 8 个单元素组 | 4 | 4（不改） |
| EP=2, DP=4 | {0,1}{2,3}{4,5}{6,7} | 2 | 4（不改） |
| EP=4, DP=2 | {0,1,2,3}{4,5,6,7} | 1 | 4（不改） |
| EP=8, DP=1 | {0..7} | 1 | **必须改成 8** |

必须固定的变量（至少四个）：global batch $G$（即 $W\cdot A\cdot b$ 保持不变）、$T$、dtype（fp16）、seed、步数、top-k、SDPA 后端、`use_compile=0`。**注意 EP=8 改了 `num_experts` 后模型总参数量变了**，所以 EP=8 的 step time 不能与前三个直接比 —— 这一点必须在报告里标注，否则结论无效。
2. 假设与机制：随 EP 增大，**下降的是每 rank 的专家参数量与专家计算量**（每 rank 持 $E/E_p$ 个专家，本地 FFN 的 token 数在均衡时不变但专家权重的显存与其梯度 all-reduce 量下降）；**上升的是**：① `all_to_all` 的组规模与跨卡字节（EP=1 时根本没有跨卡 a2a），② 木桶效应的暴露程度（组越大，$\max_r$ 覆盖的 rank 越多，出现慢 rank 的概率越高，见 §5.2）。因此假设是**存在中间最优点**，不是单调。
3. 每配置记录（全部抽象形式）：step time 相对 EP=1 的比值 $t_{EP=k}/t_{EP=1}$；$\max_r t_{compute}/\text{median}_r t_{compute}$（木桶比）；$\text{mean}_r t_{wait}/t_{step}$（空等占比）；热点比 $\max_e f_e\cdot E$；profiler 里 NCCL kernel 占比的三档区间；`PASS/FAIL`（是否在 `--max-steps` 内跑完无 hang）。
4. 必须先过 **`tests/test_ep_cpu.py`**（gloo、2 进程、CPU）。它验证的是 **I3（往返等价）**：EP=1 时六步 EP 实现的输出与原 `MOEFeedForward.forward` 在 fp32 下逐元素相等；同时顺带验证 I2 的本地自洽断言。没过这个单测就上 8 卡，任何 step time 数字都没有意义（可能在算错的东西）。
5. "EP=2 最快、EP=8 最慢"的两个机制 + 区分实验：
   - **机制 α：a2a 的组规模代价**——EP=8 的 a2a 跨越全部 8 卡，$\beta_{path}$ 被最慢链路拉低，且组内每对都要传。区分实验：固定 EP=4，比较 P-02 的两种分组；若分组方式对 EP=4 就有明显影响，则 EP=8 更差可归到这条。
   - **机制 β：木桶效应随组规模放大**——EP 组越大，$\max_r$ 取遍的 rank 越多。区分实验：在各 EP 度下比较**木桶比** $\max_r t_{compute}/\text{median}_r t_{compute}$ 是否随 EP 单调上升；若上升则是这条。
   （若两个量都不动，则要怀疑 EP=8 的差异其实来自第 1 问指出的 `num_experts=8` 变更。）
6. 失败边界（对更大模型不可外推）：① 本实验的模型只有 64M、每层 7.4M，$t^{cmp}$ 与通信固定延迟同量级——大模型上计算会盖住通信，最优 EP 会右移；② 单节点 8 卡、无跨节点，跨节点 EP 的 $\beta_{path}$ 低一到两个量级，结论不可搬；③ 专家数只有 4（或 8），真实大 MoE 有 64–256 个专家，$E/E_p$ 的粒度完全不同；④ 本周 dropless 无 capacity，有 capacity 的实现里木桶效应会被 $C$ 截断，EP 的扩展曲线形状会变。

**0–4 分 rubric**

- 0：矩阵写不全或漏了 EP=8 要改 `num_experts`。
- 1：矩阵 + 固定变量正确。
- 2：加上第 2 问的"下降一项、上升两项"与"存在中间最优"的假设。
- 3：第 1–5 问完整，记录量全部是抽象形式。
- 4：以上全对，且第 1 问主动指出 **EP=8 因为改了 `num_experts` 而不可直接与前三者比**，第 6 问的失败边界至少给出三条并各自点明是哪个量的量级变化。

**常见误区**

- 直接把 EP=8 的 step time 和 EP=1 比，忘了 `num_experts` 从 4 变成了 8、模型都不是同一个。
- 只记录 step time，不记录木桶比与空等占比，结果无法解释"为什么"。

**追问**

- 如果结论是 EP=1 最快，这个实验还有价值吗？它证明了什么？
- 固定 global batch 的要求下，EP 改变时 $W$ 没变，那么真正被改变的是什么？为什么这仍然是一个有意义的对照？

**定位**：`01_FOUNDATIONS.md` §3.2、§5.1、§5.2、§2.4（I3）；`00_WEEK_CARD.md` Day 3；`lab/` 的 `ep_moe.py`、`tests/test_ep_cpu.py`。

---

## Trade-off

### M02-T-01

**关键评分点**

1. 三选二的论证（用 A-04 的数字）：
   - **cf = 1.0 + drop**：计算有上界（每专家 ≤ 1024 token/发送方）、无空等（所有专家计算量相同），但**丢了 45.0% 的 token** → 放弃"质量"。
   - **cf = 1.5 + drop**：丢弃降到 32.5%、仍有上界，但**计算预算涨到 6144（1.5×）且 3379 个槽位是 padding** → 质量与上界都打折，放弃的是"计算/显存效率"这半个，本质上是在前两项之间买了个折中位而不是同时满足三项。
   - **dropless**：不丢一个 token（质量满分），但热点 rank 要算 11,468 个 token 而均衡期望是 4096，**最慢 rank 决定 step time，其余 rank 空等** → 放弃"无空等"，且计算量无上界（显存也无上界）。
   三种取法各放弃一项，没有一种能同时拿到三项。
2. cf 从 0.75 提到 1.5：**丢 token 比例**下降；**padding 比例**上升；**峰值显存**上升，且**与 cf 成线性**（接收缓冲 = $E_p\cdot C\cdot d\cdot2\,\text{B}$，$C\propto\text{cf}$——这是四个量里唯一严格线性的）；**最慢 rank 计算量**上升（被 $C$ 钳住，$\propto \text{cf}$，直到 cf 大到不再有 token 被丢为止，之后不再随 cf 上升）。
3. dropless 等价于 **$\text{cf} = \max_e f_e\cdot E$**（本例 2.80），即"capacity 大到刚好没人被丢"；但它是**动态的、由数据决定的**，不是预设常数。它把成本从"质量损失"转移到了**时间（空等）与显存（无上界的接收缓冲与激活）**——A-02 已经算出热点时 `gate` 输出可以到 79.7 MB × 2。
4. 本周选择：**保持 dropless**（MiniMind 现状），理由：① 本周的教学目标就是**观察**木桶效应，加 capacity 会把它截断掉；② 64M 模型 + 4 个专家，热点时的绝对显存（25.2 MB 通信缓冲 + 约 160 MB 激活）在 32 GB 卡上完全放得下；③ 丢 token 会破坏 I3 与 dense 基线的可比性。变成错误选择的条件：① 专家数或 $T$ 增大到热点时激活撑爆显存（出现 OOM 或必须开 activation checkpointing）；② 目标从"观察机制"变成"最大化吞吐"，此时无上界的 step time 不可接受；③ 热点比持续 > 3 且 aux loss 压不下来，空等占比超过一半。
5. 被丢的 token 前向输出 **0**（只走残差直通，$y_i = 0$ 加上 residual）。后果：① 反向时这些 token 对**该专家的 FFN 参数**没有梯度贡献；② 对 aux loss 的影响取决于 $f_e$ 的统计口径——若 $f_e$ 按"路由决策"统计（含被丢的），aux 仍在惩罚热点；若按"实际处理"统计，则被丢的 token 反而让热点专家看起来负载正常，**aux loss 失去了它要修的信号**。这是实现 capacity 时最容易搞错的一处。
6. 换成 64 专家、跨 8 节点：会**转向有限 capacity + drop**（或 dropless + 专门的通信 kernel）。推动这个改变的量级变化是 **$\beta_{path}$**：跨节点带宽比节点内低一到两个量级，a2a 的字节量直接主导 step time，而无上界的接收缓冲意味着**无上界的跨节点流量**，风险不可接受；同时专家数从 4 到 64 让"某个专家热"的概率大增，木桶效应的期望损失也放大。

**0–4 分 rubric**

- 0：只复述"三方权衡"这句话，不给数字。
- 1：能用 A-04 的数字说明其中一两种取法。
- 2：三种取法各放弃哪一项都说清了，第 2 问的方向正确。
- 3：以上全对，识别出峰值显存是唯一线性的量，第 3 问的 dropless ↔ 动态 cf 对应正确。
- 4：以上全对，且第 5 问指出了 **$f_e$ 的统计口径会决定 aux loss 是否还有效**这个实现层陷阱，第 6 问把改变归因到 $\beta_{path}$ 的量级而不是含糊地说"更大所以更难"。

**常见误区**

- 认为 cf 越大越好（"反正不丢 token 了"）。cf 大到覆盖热点时，计算与显存就等于回到了 dropless 的最坏情况，只是多了一堆 padding。
- 认为被丢的 token 会报错或产生 NaN。它们只是输出 0 走残差，训练照常进行——这正是它危险的地方（静默的质量损失）。

**追问**

- 如果 EP 组内 4 个 rank 的热点专家**不是同一个**，木桶效应还成立吗？dropless 的 step time 由什么决定？
- 你会怎么设计一个实验，把"丢 token 造成的质量损失"单独测出来（不与 cf 带来的其他变化混淆）？

**定位**：`01_FOUNDATIONS.md` §4.6、§5.2、§5.3、§3.1、§2.3；`lab/` 的 `ep_moe.py`、`router_stats.py`。

---

### M02-T-02

**关键评分点**

1. top-1 → top-2 的四个量（基于 A-02，`计算`）：

| 量 | top-1 | top-2 | 倍数 |
| --- | --- | --- | --- |
| 单层 dispatch 发送字节 | 6.29 MB | 12.6 MB | **×2** |
| 单层期望接收字节 | 6.29 MB | 12.6 MB | **×2** |
| 单层接收字节上限 | 25.2 MB | 25.2 MB | **×1（不变）** |
| 本地 expert FLOPs | 基准 | ×2 | **×2** |

不是简单倍数的是**接收上限**。原因：top-2 要求每个 token 选**两个不同的专家**，因此单个专家从任一发送 rank 最多仍只收到该 rank 的全部 4096 个 token 各一份，$E_p\times4096 = 16{,}384$ 这个上界不变；top-2 只是把实际分布**往上界推**，没有抬高上界。
2. `norm_topk_prob=True` 在 top-2 下把两个专家的权重归一化成和为 1（$w \leftarrow w/(\sum w + 10^{-20})$），使 $y_i$ 是两个专家输出的**凸组合**，输出尺度与 dense 可比。在 top-1 下它把唯一那个权重直接归一化成 **1**，即权重项被完全消掉——所以 top-1 + `norm_topk_prob=True` 时 router 的 softmax 值**不进入前向输出**，只通过 aux loss 的 $P_e$ 影响训练。这是 top-1 一个容易被忽视的性质。
3. top-2 对负载均衡**更容易**：每个 token 贡献两次分配，$f_e$ 的估计方差更小，且单个 token 的"极端选择"被另一个专家稀释；同时 $\sum_e f_e = k = 2$，aux 的取值范围与 $k=1$ 时不同（$E\sum f_eP_e$ 在均衡时不再是 1，读日志时必须换基准，否则会误判成"变差了"）。代价是路由的区分度下降——每个 token 都会有一个"第二选择"，专家分工被软化。
4. top-2 值得付代价的两个可观测判据：① 在同一 step 预算下，top-2 的 **`logits_loss` 相对 top-1 的改善幅度大于 step time 的增幅**（即 loss-per-wallclock 更优）；② top-1 下观测到**路由不稳定**——同一 token 在相邻 step 频繁改变专家（可用相邻 step 的 `topk_idx` 一致率度量），top-2 能把这种抖动的影响减半。只说"效果更好"不给判据的答案封顶 2 分。
5. 阈值口径：应该用 **loss-per-wallclock 等价点**来定，而不是拍一个数。做法：先测 top-1 的 `logits_loss` 随 step 的下降速率，再问"top-2 每步慢 $x$ 倍时，它需要每步多降多少 loss 才能打平"；把 $x$ 定在"top-2 在同等 wallclock 下不劣于 top-1"的那个点。本周 8 卡预算下可以先用 $x = 1.5$ 作为工程上的**先验阈值**（`估算`，未实测），并在扫描后用实际比值替换。
6. 证伪"top-2 的收益来自更好的专家分工"的实验：把 top-2 的**第二个专家改成随机选择**（不用 router 的第二名），其余全部不变。若 `logits_loss` 的改善基本保留，则收益来自"每 token 更多计算 / 集成平均"，**不是**来自更好的分工；若改善消失，则分工假说成立。（补充对照：把 top-1 的 expert FFN 宽度加倍，使 FLOPs 与 top-2 相当，若它也能拿到同样改善，同样证伪分工假说。）

**0–4 分 rubric**

- 0：四个量算错，或认为接收上限也翻倍。
- 1：四个量正确，但说不清上限为什么不变。
- 2：第 1、2 问正确（含 top-1 下权重被归一化成 1 这个性质）。
- 3：第 1–4 问全对，判据具体可观测。
- 4：以上全对，且第 5 问用 loss-per-wallclock 定阈值（而不是拍数），第 6 问的证伪实验设计干净（只改一个变量，且给了补充对照）。

**常见误区**

- 认为 top-2 一定更难均衡。恰恰相反：$k$ 增大让 $f_e$ 的估计更稳，但 aux 的读数基准要跟着换。
- 忘了 top-1 下 `norm_topk_prob=True` 会把权重变成常数 1，于是以为 router 的 softmax 值在 top-1 前向里也起作用。

**追问**

- top-2 下 aux loss 的均衡值（$E\sum f_eP_e$）是多少？推一下，说明为什么日志基准必须换。
- 如果显存是瓶颈而不是时间，top-2 最先撑爆的是哪个张量？（用 A-02 第 5 问的数）

**定位**：`01_FOUNDATIONS.md` §3.1（Top-2 行）、§2.3、§1.1、§4.5；`lab/` 的 `ep_moe.py`、`router_stats.py`。

---

### M02-T-03（Gate A1 题）

**关键评分点**

1. 四段因果链（每段显式推导）：
   - **① 参数小 → 每层计算时间短**：$P_\ell = 7{,}374{,}528$，$t^{cmp}_\ell = \frac{6P_\ell N_{tok}}{\eta F}$，其中 $\eta$（小矩阵利用率）在 $d = 768$ 时**远低于 1**，进一步压低了有效算力。
   - **② 计算时间短 → 每次集合通信的固定延迟藏不住**：`FULL_SHARD` 每层 forward 前必须**等 AllGather 完成**才能算，而 2.1 的 `forward_prefetch=False` 意味着这次 AllGather 只能在上一层算完后才发起，无法与计算重叠；暴露时间 $\approx\sum_\ell\max(0,\;t^{ag}_\ell - t^{cmp}_{\ell-1})$。
   - **③ 集合通信次数多 → 固定开销累加**：`FULL_SHARD` 每步 **27 次**（9 fwd AG + 9 bwd AG + 9 RS），DDP 只有约 **11 次**，且 DDP 的 11 次全部在反向里与计算**重叠**，暴露时间只有 $\max(0,\,t_{allreduce} - t_{bwd})$。27 次 × $\alpha$ 这一项在 DDP 侧根本不存在。
   - **④ 再加 FSDP 的 CPU 侧开销**：flatten/unflatten、更多的 kernel 启动与 Python 侧簿记，在每层计算只有几毫秒量级时占比不可小视。
   合起来：$t_{FSDP} > t_{DDP}$ 的来源是**延迟 × 次数 + 重叠度差 + CPU 开销**，三项都与字节量无关。
2. 不是字节量问题的两个关键数字（A-03，`计算`）：FSDP `FULL_SHARD` 每 rank 实际收发 **335 MB**，DDP 每 rank **447 MB**。**FSDP 传的字节比 DDP 还少 25%**，却更慢——字节量假说被这两个数直接证伪。（`SHARD_GRAD_OP` 更少，只有 224 MB。）
3. 不等式：FSDP 每层暴露 $\approx\max(0,\;\alpha + \frac{W-1}{W}\frac{S_\ell}{\beta} - t^{cmp}_{\ell-1})$，其中 $t^{cmp}_\ell = \frac{6P_\ell N_{tok}}{\eta F}$、$S_\ell = 2P_\ell$。FSDP 慢于 DDP 的条件近似为
   $$27\alpha + \sum_\ell\max\Big(0,\;\frac{(W-1)}{W}\frac{2P_\ell}{\beta} - t^{cmp}_{\ell-1}\Big) + T_{cpu}\;>\;\max(0,\;t_{allreduce} - t_{bwd})$$
   翻转条件：当 $t^{cmp}_\ell$ 增长到能盖住括号内的传输项、且 $27\alpha$ 在整步时间中的份额趋近于零时，不等式左边趋于 0，FSDP 不再更慢。
4. 27B 上 DDP 直接不可行（`计算`，fp32 口径）：参数 $27\times10^9\times4 = 108\,\text{GB}$、梯度 $108\,\text{GB}$、Adam 两个矩 $216\,\text{GB}$ → **每卡 432 GB**，而 V100 单卡只有 32 GB。即使只放 fp16 参数也要 54 GB > 32 GB。DDP 要求每卡持有完整的参数 + 梯度 + 优化器状态，**这一步就装不下**，与快慢无关。这说明 FSDP 在大模型上的理由是**可行性**，不是吞吐。
5. 转折点估算（`估算`，必须写清假设）：设 $P_\ell \approx 12.4\,d^2$（本模型 $d=768$ 时给出 7.31M，与 7.37M 相符）。则
   $$\frac{t^{cmp}_\ell}{t^{transfer}_\ell} = \frac{6P_\ell N_{tok}/(\eta F)}{\frac{W-1}{W}\cdot 2P_\ell/\beta} = \frac{6N_{tok}\beta}{\frac{7}{4}\eta F}$$
   **$P_\ell$ 被约掉了** —— 在带宽主导区，单纯放大 `hidden_size` **不改变**计算与传输的比值。真正的旋钮有两个：① **$\alpha$ 的相对份额**：$t^{cmp}_\ell \propto d^2$ 而 $\alpha$ 是常数，所以 $d$ 增大确实让 $27\alpha$ 的份额趋近于零——按 $t^{cmp}\propto d^2$，$d$ 从 768 提到 3072（4×）时 $t^{cmp}$ 涨约 16×，$\alpha$ 的份额降到 1/16，这是**量级上的转折带**（`估算`，假设 $N_{tok}$、$\eta$ 不变）；② **$\eta$ 随矩阵变大而改善**，进一步压低左边。所以答案是：转折不是由字节量驱动的，而是由"$\alpha$ 份额 + $\eta$ 改善"驱动的，量级上 $d$ 需要提高约一个数量级（$10^3$ 量级 → $\sim10^4$ 量级，即 27B 类模型的 $d\approx5000$ 以上）。**所有这些都是 `估算`，$\eta$、$\beta$、$\alpha$ 三个量在公司机器上都还是未知，Day 2 才有第一批比值。**
6. 失败边界与下一步：推翻因果链的观测——① profiler 显示 NCCL kernel 时间占比 **< 25%**（那说明慢的不是通信，因果链②③错，要转去查 CPU 侧与 kernel 启动）；② `SHARD_GRAD_OP`（18 次）与 `FULL_SHARD`（27 次）的 step time **几乎相同**（说明次数不是主因）；③ 打开 `forward_prefetch=True` 后 $r_1$ 基本不动（说明重叠度不是主因）。可验证的下一步（8 卡预算内）：跑 DDP / `FULL_SHARD` / `SHARD_GRAD_OP` 三组短 smoke，只记 $r_1 = t_{FSDP}/t_{DDP}$、$r_2 = t_{SGO}/t_{DDP}$ 与 NCCL 占比三档区间——若 $r_2$ 落在 1 与 $r_1$ 之间且 NCCL 占比 > 25%，因果链成立。

**0–4 分 rubric**

- 0：把"FSDP 慢"归因于"传的字节多"。
- 1：因果链能说出两段，但没有用 335 vs 447 证伪字节量假说。
- 2：四段因果链完整 + 两个关键数字 + 27B 的显存账。
- 3：以上全对，第 3 问写出了含 $\alpha$ 与 $t^{cmp}$ 的不等式并说明翻转条件。
- 4：以上全对，且第 5 问**发现 $P_\ell$ 在比值里被约掉**、把转折归到 $\alpha$ 份额与 $\eta$ 而不是字节量，并明确标注整段是 `估算`、依赖三个未知量；第 6 问给出三条可证伪观测与一个本周能做完的实验。

**常见误区**

- 认为 FSDP 慢是因为通信量大。A-03 的数字是反的。
- 认为 27B 用 FSDP 是为了更快。是为了**装得下**——432 GB vs 32 GB，这是可行性问题。
- 在第 5 问里直接说"hidden_size 到 4096 就翻转"而不写假设、不做量纲分析——这类答案封顶 2 分（本周没有任何实测支撑一个具体数）。

**追问**

- 若把 $N_{tok}$（每 rank 每层 token 数）翻倍，第 5 问的比值怎么动？这算不算一个比放大 $d$ 更便宜的验证手段？
- `HYBRID_SHARD` 在 27B + 多节点场景里解决的是第 3 问不等式里的哪一项？

**定位**：`01_FOUNDATIONS.md` §5.1（符号模型与因果链）、§3.3、§3.4、§8 第 2 条、§7；`00_WEEK_CARD.md` Day 2；`lab/` 的 `train_fsdp.py`、`train_ddp.py`。

---

### M02-T-04

**关键评分点**

1. 静态显存（`计算`，取自 P-03）：
   - **方案甲**（每卡全复制 + Policy-FSDP）：每卡 = Policy 分片 $\approx128\,\text{MB}$ + Reference $127.8\,\text{MB}$ + Reward $3.6\,\text{GB}$ $\approx \mathbf{3.86\,GB}$，8 卡都一样。
   - **方案乙**（角色拆分）：rank 0–3 = Policy 分片（4 卡分片，$1.02/4\approx256\,\text{MB}$）$\approx\mathbf{0.26\,GB}$；rank 4–5 = policy 的 fp16 推理副本 $127.8\,\text{MB}$ + KV cache $\approx\mathbf{0.13\,GB}$ + 动态；rank 6–7 = Reward $\mathbf{3.6\,GB}$。
   - 方案乙的静态显存**显著更低**：6 张卡从 3.86 GB 降到 0.13–0.26 GB 量级，全机静态总量从 $8\times3.86 = 30.9\,\text{GB}$ 降到约 $4\times0.26 + 2\times0.13 + 2\times3.6 = 8.5\,\text{GB}$（`计算`），**省约 22 GB**，主项就是 RM 从 8 份变成 2 份。
2. 通信构成：
   - **方案甲**：只有 Policy 组内的 FSDP AllGather / ReduceScatter（8 卡）；Reference 与 Reward 都在本卡上前向，**零跨卡通信**。
   - **方案乙**：**多出**——rollout→reward 的 `(prompt, response, logprob)` 传输、reward→policy 的标量奖励回传、policy→rollout 的权重 `broadcast`（fp16 全量 $\approx128\,\text{MB}$/轮）；**少了**——FSDP 组从 8 卡缩到 4 卡，每次 AllGather/ReduceScatter 的组更小、$\frac{W-1}{W}$ 系数从 $7/8$ 降到 $3/4$。
3. 结构性缺陷：**同步**约束下，一轮里 rank 0–3（policy）与 rank 6–7（reward）在**生成阶段**全程空闲，rank 4–5（rollout）在 **policy 更新阶段**空闲。空闲的直接来源是"每轮 `broadcast` 完成后才允许生成"这条约束——它把三个阶段串成了一条链，任何时刻只有一类角色在工作，而生成阶段又是最长的一段（P-03 第 4 问）。
4. 方案甲反而更好的条件（每个都可观测判定）：
   - ① **$t_{generate}/t_{round}$ 很小**（比如 < 0.3）：此时串行链的损失不大，而方案甲能让 8 张卡全程参与训练，吞吐更高。
   - ② **显存不是约束**：观测到方案甲下峰值显存离 32 GB 还很远（$m_{peak}/32\,\text{GB}$ 明显小于 1），那省下的 22 GB 没有换来任何东西。
   - ③ **权重同步开销占比高**：观测到 $t_{broadcast}/t_{round}$ 不可小视（轮次短、生成少时会这样），方案乙多出的 128 MB/轮变成净损失。
   - ④ **policy 组缩到 4 卡后 OOM 或吞吐反降**：分片收益不足以抵消组变小。
5. rollout 卡用的是**上一轮**的权重（同步版里权重在本轮更新完成后才 broadcast 过去，见 §6 第 12 条）。去掉"broadcast 完成后才生成"这个约束，就得到**异步 rollout**：rollout 卡不等新权重，继续用手上的旧权重生成，吞吐大幅提升、空等消失。代价是引入 **off-policy 偏差**（stale policy）——用旧 policy 采的样本去更新新 policy，重要性比不再为 1，需要 importance ratio 修正 / clip，否则更新方向有偏。这正是 verl / OpenRLHF 这类框架要处理的核心问题，也是本周**明确不覆盖**的部分。
6. 时间不够时的取舍：**保留**"Policy-FSDP + kill-rank 故障注入 + 从分片 checkpoint 恢复"这一段作为证据（它是 Gate 依赖的关键故障练习，且能产出 loss 连续性这条可提交判据）；**放弃**角色拆分的权重同步演示与 rollout/reward 分卡运行。放弃后，周报里"GRPO 角色拆分"的结论标成 **`INCONCLUSIVE`**（周卡明确允许），并写清放弃原因与"下一次需要多少 8 卡分钟"这两个字段；**不允许**用方案甲的观测去冒充方案乙的结论。

**0–4 分 rubric**

- 0：算不出两种方案的静态显存差。
- 1：显存差正确，但通信构成说不清。
- 2：第 1–3 问正确，指出了空闲来自同步约束。
- 3：第 1–5 问全对，四个"甲更好"的条件都可观测判定。
- 4：以上全对，且第 5 问正确命名 off-policy / stale policy 偏差并说明修正方向，第 6 问的取舍明确保住 Gate 依赖项、把放弃项标成 `INCONCLUSIVE` 且拒绝用甲冒充乙。

**常见误区**

- 认为角色拆分一定更好。它省显存但制造空闲；只有当生成确实主导一轮时间、且显存确实是约束时才划算。
- 认为"异步 rollout 只是工程优化"。它改变了算法的 on-policy 性质，必须配 importance ratio 修正。

**追问**

- 如果 RM 从 1.8B 换成 7B，第 1 问的两个数变成多少？第 4 问的哪个条件最先失效？
- 方案乙里 policy 组只有 4 卡，FSDP 的 $\frac{W-1}{W}$ 从 7/8 变成 3/4，这对每 rank 字节是省还是费？为什么它同时也是坏消息？

**定位**：`01_FOUNDATIONS.md` §5.4、§6 第 12 条、§7；`00_WEEK_CARD.md` Day 5、「总估时」的砍单顺序；`lab/` 的 `grpo_roles.py`、`train_fsdp.py`。

---

## 评分汇总表

| 层 | 题数 | ID |
| --- | --- | --- |
| Recall | 5 | M02-R-01 … M02-R-05 |
| Explain | 5 | M02-E-01 … M02-E-05 |
| Apply | 4 | M02-A-01 … M02-A-04 |
| Debug | 5 | M02-D-01 … M02-D-05 |
| Design | 4 | M02-P-01 … M02-P-04 |
| Trade-off | 4 | M02-T-01 … M02-T-04 |
| **合计** | **27** | |

## Gate 判定建议

对应 `00_WEEK_CARD.md` 的 Gate（A0 闭卷 30 分钟画 EP=4/DP=2 的 rank 映射与 MoE 层通信序列；A1 解释"64M 上 FSDP 比 DDP 慢"的原因链；跨场景 debug 热点专家；两个不同日期的证据）：

- **Gate 题 1（rank 映射与通信序列）**：`M02-A-01` **≥ 3 分**，且第 1 问的整表与第 4 问的通信序列都正确。这是硬门槛，低于 3 分直接不过。
- **Gate 题 2（FSDP 更慢的原因链）**：`M02-T-03` **≥ 3 分**，且必须包含"335 MB vs 447 MB 证伪字节量假说"这一条。
- **Gate 题 3（跨场景 debug 热点专家）**：`M02-D-03` **≥ 3 分**，且第 5 问必须答对"主指标是 `logits_loss` 而不是 `aux_loss`"。
- **Recall + Explain**：两层合并平均 **≥ 3.0**，且 `M02-R-01`（bf16/fp16）与 `M02-R-05`（`all_to_all_single` 两个 split）各自 **≥ 3 分**——这两条是本周所有命令与所有 EP 代码的前提。
- **Apply**：`M02-A-02` 与 `M02-A-03` 两道手算题各 **≥ 3 分**（数字必须与 `01_FOUNDATIONS.md` 第 3 节的表一致）；`M02-A-04` **≥ 2 分**。
- **Debug**：五题平均 **≥ 2.5**，其中 `M02-D-01`（EP hang）与 `M02-D-02`（8→4 恢复）各 **≥ 3 分**——它们对应本周唯一的关键故障练习。
- **Design + Trade-off**：两层合并平均 **≥ 2.5**，且至少两题达到 **4 分**（即完整给出假设 / 证据 / 替代解释 / 失败边界 / 可验证下一步五要素）。
- **一票否决**：任何一题把 `估算` 或 `计算` 的数字说成"实测"，该题记 0 分；本周截至 Gate 时不存在任何实测数据。
- **任一层平均 < 2.0**：该层对应的 `01_FOUNDATIONS.md` 章节需要重读，并补一次对应的 lab 观测（CPU 单测即可，不必占 8 卡配额）后重考该层。
- **两个不同日期的证据**：R+E+A 与 D+P+T 分两次作答、日期不同，即满足周卡的这一条。
