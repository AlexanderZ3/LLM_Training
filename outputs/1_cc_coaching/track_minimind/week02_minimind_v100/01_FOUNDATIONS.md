# Week M02 Foundations — 同一个 64M MiniMind 在 8×V100 上的四种放置方式

> 生成日期：2026-09-04 · 生成方：cc-theory-tutor · 轨道：`track_minimind`
> 对象：MiniMind commit `7a6fddd`（`model/model_minimind.py`、`trainer/trainer_utils.py`、`trainer/train_pretrain.py`）+ PyTorch 2.1 官方文档（FSDP / distributed / DDP notes / amp）。所有源码引用均于 2026-09-04 核验。
> 标签约定：`已确认`（官方文档/源码）、`用户自述待核验`、`来源计划假设`、`推断`、`计算`（由已确认的配置算出的数）。

## 0. 本周要拥有的能力（一段话）

本周结束时，你应能对同一个 63.9M 参数的 MiniMind dense 模型（以及它的 198M/激活 63.9M 的 4-expert MoE 变体），不看文档说清楚：在 8×V100 上分别用 DDP、FSDP(FULL_SHARD / SHARD_GRAD_OP)、EP（专家并行）和 GRPO 角色拆分时，**参数、梯度、优化器状态、token 和每一次集合通信分别落在哪张卡、在 step 的哪个时刻发生、字节量多大、由谁决定 step time**；并能从日志里的三种不变量（loss 连续性、fixed-global-batch 等价、每 rank 的 token 计数向量）反推出 bf16 报错、GradScaler 跳步、NCCL hang、FSDP 恢复 shape 不匹配、all_to_all split 不一致和热点专家这八类故障。

**公司边界提醒（CLAUDE.md 第 2 节，本周每条命令都受约束）**：公司 8×V100 上产生的日志、trace、checkpoint、图片、拓扑细节（包括 `nvidia-smi topo -m` 的原文）和任何性能数字不得带出；本文第 5 节只给符号模型和"抽象比值"记录法，你回填证据字段时只写比值、是否连续、次数、`PASS/FAIL` 类型。不新建 conda 环境、不升级 PyTorch 2.1.0、不装 DeepEP/MegaBlocks。个人 5070 Ti 只做单卡回归，不能冒充多卡结果。

## 1. 对象：数据与模型（具体 ID、样例、shape）

### 1.1 模型：`MiniMindConfig()` 默认值（`已确认`，`model/model_minimind.py` L10–L45）

| 符号 | 字段 | 值 |
| --- | --- | --- |
| $d$ | `hidden_size` | 768 |
| $L$ | `num_hidden_layers` | 8 |
| $V$ | `vocab_size` | 6400（`tie_word_embeddings=True`） |
| $H$ / $H_{kv}$ | `num_attention_heads` / `num_key_value_heads` | 8 / 4（GQA） |
| $d_h$ | `head_dim` | 768/8 = 96 |
| $d_{ff}$ | `intermediate_size` | `ceil(768·π/64)·64` = 38·64 = **2432**（`计算`） |
| $E$ | `num_experts` | 4（MoE 时） |
| $k$ | `num_experts_per_tok` | 1 |
| — | `moe_intermediate_size` | = `intermediate_size` = 2432 |
| $\lambda$ | `router_aux_loss_coef` | 5e-4 |
| — | `norm_topk_prob` | True |
| — | `dropout` | 0.0 |
| — | `flash_attn` | True（实际是"是否调用 SDPA"，不是 FlashAttention 库） |

**每层参数量（`计算`，无 bias）**

| 子模块 | 公式 | dense 层 | MoE 层 |
| --- | --- | --- | --- |
| `q_proj`+`o_proj` | $2·d·d$ | 1,179,648 | 1,179,648 |
| `k_proj`+`v_proj` | $2·d·H_{kv}·d_h$ = 2·768·384 | 589,824 | 589,824 |
| 两个层级 RMSNorm | $2d$ | 1,536 | 1,536 |
| `q_norm`+`k_norm` | $2·d_h$ = 2·96 | 192 | 192 |
| FFN（gate/up/down） | $3·d·d_{ff}$ = 3·768·2432 | 5,603,328 | ×4 experts = 22,413,312 |
| router `gate` | $d·E$ | — | 3,072 |
| **每层合计** $P_\ell$ | | **7,374,528** | **24,187,584** |
| 8 层 + embedding(4,915,200) + 末层 norm(768) | | **63,912,192 ≈ 63.9M** | **198,416,640 ≈ 198M** |

上表已用 `MiniMindForCausalLM(MiniMindConfig(use_moe=...))` 实例化后 `sum(p.numel())` 核对（`已确认`，2026-09-05）：dense 63,912,192、MoE 198,416,640，`lm_head` 与 `embed_tokens` 共享存储（tied），因此 embedding 只计一次。与 Week M01 第 1 节的 63,912,192 一致。

MoE 每 token 激活参数 = dense 63.9M + 8×3,072 router ≈ 63.9M（`计算`），与 README 的"64M 激活/198M 总量"一致。

### 1.2 本周的 worked-example batch

固定 $B=8$、$T=512$ → 每 rank 每层 $N_{tok}=4096$ 个 token；hidden 在 autocast(fp16) 下每 token 占 $d·2\,\text{B}=1536\,\text{B}$（`计算`）。数据用 `dataset/pretrain_t2t_mini.jsonl`（MiniMind 自带，公开）；一条样本进入模型后 shape 为 `input_ids:[B,T] int64` → `hidden:[B,T,768] fp16` → `logits:[B,T,6400]`（`cross_entropy` 在 autocast 下升到 fp32，见第 4 节）。

### 1.3 训练循环里与本周直接相关的 5 行（`已确认`，`trainer/train_pretrain.py`）

- L37–L40：`loss = res.loss + res.aux_loss; loss = loss / args.accumulation_steps; scaler.scale(loss).backward()`
- L42–L47：`if step % accumulation_steps == 0: scaler.unscale_(optimizer); clip_grad_norm_(...); scaler.step(optimizer); scaler.update()`
- L112：`setup_seed(42 + rank)` — **每个 rank 的 seed 不同**
- L121–L122：`dtype = bfloat16 if args.dtype=="bfloat16" else float16; autocast_ctx = torch.cuda.amp.autocast(dtype=dtype)`；L137：`GradScaler(enabled=(args.dtype=='float16'))`
- L136、L154、L158：`DistributedSampler(train_ds)` / `DistributedDataParallel(model, device_ids=[local_rank])` / `set_epoch(epoch)`

## 2. 数学目标与推导

### 2.1 主目标与 fixed-global-batch 不变量

单步优化目标是 global batch $\mathcal{B}$（$G=|\mathcal{B}|$ 条序列）上的 token 平均交叉熵：

$$\mathcal{L}(\theta)=\frac{1}{\sum_{s\in\mathcal{B}}\sum_t m_{s,t}}\sum_{s\in\mathcal{B}}\sum_t m_{s,t}\,\big(-\log p_\theta(x_{s,t+1}\mid x_{s,\le t})\big)$$

数据并行把 $\mathcal{B}$ 切成 $W$（world_size）× $A$（accumulation）个 micro-batch，每个大小 $b$，$G=W·A·b$。DDP 算的是

$$\hat g=\frac{1}{W}\sum_{r=1}^{W}\sum_{a=1}^{A}\frac{1}{A}\nabla\ell_{r,a}$$

其中 $\ell_{r,a}$ 是 micro-batch 内的 **token 均值**（MiniMind 的 `res.loss` 是 mask 加权均值）。$\hat g=\nabla\mathcal{L}$ 当且仅当每个 micro-batch 的有效 token 数 $\sum_t m$ 相等（`推断`：pretrain 数据固定 `max_seq_len` 且几乎无 padding 时近似成立；SFT 数据每条有效 token 数不同，"micro-batch 均值的均值 ≠ 全局 token 均值"，这正是 2 卡×acc 4 与 8 卡×acc 1 不严格等价的第一个数学来源）。

**不变量 I1（fixed-global-batch）**：若 $(W,A,b)$ 两种配置使 $G$ 相同、**每个 optimizer step 消费的样本集合相同**、seed/dropout/scaler 状态相同，则 $\hat g$ 逐步一致，loss 曲线在 fp16 舍入误差内重合。

### 2.2 FP16 loss scaling

fp16 的最小正规数 $2^{-14}\approx6.1\times10^{-5}$，梯度里大量元素小于它会 flush 到 0。GradScaler 用尺度 $s$ 把 loss 放大：反传得到 $s·\nabla\ell$，`unscale_` 再除以 $s$。若 $s·\nabla\ell$ 出现 inf/NaN，则跳过 `optimizer.step()`、$s\leftarrow s·0.5$；连续 2000 步无跳步则 $s\leftarrow 2s$（`已确认`：`init_scale=65536, growth_factor=2, backoff_factor=0.5, growth_interval=2000`，PyTorch 2.1 amp 文档）。

### 2.3 MoE router 与 aux loss（MiniMind 实现，`已确认` L155–L172）

对 flatten 后的 $N$ 个 token $x_i\in\mathbb{R}^d$：

$$z_i=W_g x_i\in\mathbb{R}^E,\quad s_i=\text{softmax}(z_i),\quad (\mathcal{I}_i,w_i)=\text{topk}(s_i,k),\quad w_i\leftarrow \frac{w_i}{\sum w_i+10^{-20}}$$
$$y_i=\sum_{e\in\mathcal{I}_i} w_{i,e}\,\text{FFN}_e(x_i)$$

aux loss（Switch-Transformer 形式）：$f_e=\frac{1}{N}\sum_i \mathbb{1}[e\in\mathcal{I}_i]$（实际负载份额，`load`），$P_e=\frac{1}{N}\sum_i s_{i,e}$（平均路由概率，`scores.mean(0)`）：

$$\mathcal{L}_{aux}=\lambda\,E\sum_{e=1}^{E} f_e P_e$$

当 $f_e=P_e=1/E$ 时 $\sum f_eP_e=1/E$，乘 $E$ 后最小值为 1（$k=1$）；完全坍缩到一个专家时为 $E$。梯度只经 $P_e$ 流回 router（$f_e$ 经 one_hot 不可导）。注意 MiniMind 是 **对 $B·T$ 全部 token 算一个 aux（无 `seq_aux`）、无 capacity、无 drop、无 EP、无 shared expert**（`已确认`）。总 loss $=\mathcal{L}+\sum_{\ell}\mathcal{L}_{aux,\ell}$（`MiniMindModel.forward` 把 8 层 aux 求和，L230）。

### 2.4 EP 的 token 流（本周核心实现的数学）

EP 组大小 $E_p$，每 rank 持有 $E/E_p$ 个专家。记 rank $r$ 上 token $i$ 的目标 rank $\rho(i)=\lfloor \mathcal{I}_i / (E/E_p)\rfloor$。

1. **计数**：$c_r[j]=\#\{i: \rho(i)=j\}$，$j=0..E_p-1$；一次 `all_to_all_single` 交换 $c$ 得到 $c'_r[j]=c_j[r]$（我将收到多少）。
2. **permutation**：按 $\rho(i)$ 稳定排序得 `perm`，`x_sorted = x[perm]`，长度 $N k$。
3. **dispatch**：`all_to_all_single(recv, x_sorted, output_split_sizes=c'_r, input_split_sizes=c_r)`，`recv` 长度 $\sum_j c'_r[j]$。
4. **本地 expert**：`recv` 再按本地专家 id 分桶（$E/E_p>1$ 时）做 FFN。
5. **combine**：逆向 `all_to_all_single(back, out, output_split_sizes=c_r, input_split_sizes=c'_r)`。
6. **unpermute + 加权**：`y.index_add_(0, token_of[perm], back * w)`。

**不变量 I2（split 一致）**：对每对 $(r,j)$，rank $r$ 的 `input_split_sizes[j]` 必须等于 rank $j$ 的 `output_split_sizes[r]`；否则 NCCL 等待永不满足的字节数 → hang。
**不变量 I3（往返等价）**：EP=1 时上述 6 步的输出与原 `MOEFeedForward.forward` 在 fp32 下逐元素相等（`tests/test_ep_cpu.py` 的目标）。

## 3. 从张量到代码（shape 表、关键不变量、代码位置）

### 3.1 一次 MoE 层在 EP=4 / DP=2 上的 shape 表（B=8, T=512, top-1, fp16；`计算`）

| 步骤 | 张量 | shape / dtype | 字节 |
| --- | --- | --- | --- |
| 输入 | `x_flat` | [4096, 768] fp16 | 6.29 MB |
| router | `scores` | [4096, 4] fp32（softmax 升精度） | 64 KB |
| topk | `topk_idx`,`topk_weight` | [4096,1] int64 / fp32 | — |
| 计数 | `c_r` | [4] int64 | 32 B（先于 dispatch 单独交换一次） |
| dispatch 发送上限 | `x_sorted` | [≤4096, 768] | ≤ 6.29 MB |
| dispatch 接收上限（热点） | `recv` | [≤4·4096=16384, 768] | ≤ 25.2 MB（均衡期望 4096 → 6.29 MB） |
| 本地 expert | `gate/up` 输出 | [n_recv, 2432] fp16 | 热点时 79.7 MB |
| combine | 与 dispatch 对称 | | |
| Top-2 | 发送 8192 token-copies；接收期望 8192、上限仍 16384（每 token 两个不同专家） | | 发送翻倍 12.6 MB |
| 反向 | 两次 all_to_all 各自反向 | | 通信量再 ×2 |

一个 MoE 层一步 = 4 次 `all_to_all_single`（fwd dispatch/combine + 各自反向）+ 1 次计数交换。

### 3.2 8 卡 rank 映射（`计算`，lab 约定：EP 组由连续 rank 组成；EP=8 需 `--num-experts 8`，因为 $E_p\le E$ 且 $E\bmod E_p=0$）

| 配置 | EP 组（`new_group`） | 每 rank 专家 | 专家梯度的 DP 组（同专家副本） | 非专家参数 DP 组 |
| --- | --- | --- | --- | --- |
| EP=1, DP=8 | 8 个单元素组（等价于不建组） | 4 | 全 8 卡 | 全 8 卡 |
| EP=2, DP=4 | {0,1}{2,3}{4,5}{6,7} | 2 | {0,2,4,6}、{1,3,5,7} | 全 8 卡 |
| EP=4, DP=2 | {0,1,2,3}{4,5,6,7} | 1 | {0,4}{1,5}{2,6}{3,7} | 全 8 卡 |
| EP=8, DP=1（E=8） | {0..7} | 1 | 无（不需要 allreduce） | 全 8 卡 |

EP=4/DP=2 时 rank 2 的归属：EP 组 `{0,1,2,3}` 持有 expert 2；expert-2 的梯度与 rank 6 allreduce；attention/embedding/norm 的梯度与全部 8 卡 allreduce。**这意味着 DDP 不能原样包住整个模型**：专家参数必须放进带 `process_group={2,6}` 的桶，或在 lab 里手写两组 `all_reduce`（`来源计划假设`：`ep_moe.py` 采用手写 allreduce 而非两层 DDP）。`new_group` 要求所有 8 个进程按相同顺序调用，即使自己不是成员（`已确认`，2.1 distributed 文档）。

### 3.3 DDP 每步 allreduce 字节（`计算`）

autocast 下参数与梯度仍是 fp32，$63{,}912{,}192×4\,\text{B}=255.6\,\text{MB}$；`bucket_cap_mb=25` 默认 → 约 11 个桶。Ring allreduce 每 rank 收发各 $2(W-1)/W×255.6$ = 447 MB（W=8）。桶在反向中按**桶索引顺序**（不是就绪顺序）异步启动，与反向计算重叠（`已确认`，DDP notes）。

### 3.4 FSDP 每层 AllGather / ReduceScatter 字节（`计算`，`MixedPrecision(param_dtype=fp16, reduce_dtype=fp16)`，wrap 粒度 = 每个 `MiniMindBlock`）

| 集合通信 | 每层负载 $S_\ell$ | 每 rank 实际收/发 $(W-1)/W·S_\ell$ | 每步次数（FULL_SHARD） | 每步次数（SHARD_GRAD_OP） |
| --- | --- | --- | --- | --- |
| AllGather（forward 前） | 7,374,528×2 B = 14.75 MB | 12.9 MB | 8 + 1（embedding 9.83 MB） | 9 |
| AllGather（backward 前） | 同上 | 12.9 MB | 9 | **0**（forward 后不 reshard） |
| ReduceScatter（backward 后） | 14.75 MB | 12.9 MB | 9 | 9 |

每步集合通信 27 次（FULL_SHARD）vs DDP 约 11 次；总负载 383 MB，每 rank 实际字节 335 MB——**比 DDP 的 447 MB 还少**。所以"64M 上 FSDP 慢"不是字节量问题，是次数、延迟和暴露时间的问题（第 5 节）。

### 3.5 代码位置

| 概念 | MiniMind / PyTorch 名称 | lab 文件（`来源计划假设`，由 cc-lab-builder 同步产出） |
| --- | --- | --- |
| autocast/scaler | `torch.cuda.amp.autocast`、`GradScaler.scale/unscale_/step/update` | `lab/src/mm_dist/train_ddp.py` |
| SDPA 后端 | `F.scaled_dot_product_attention`（`Attention.forward` L125–126）；`torch.backends.cuda.sdp_kernel(enable_flash, enable_mem_efficient, enable_math)` | `train_ddp.py` 的 `--sdpa-backend` |
| DDP | `init_distributed_mode()`（`trainer_utils.py` L44–51）、`DistributedSampler`、`DistributedDataParallel` | `train_ddp.py`、`tests/test_ddp_gloo_cpu.py` |
| FSDP | `FullyShardedDataParallel(sharding_strategy, auto_wrap_policy, mixed_precision, backward_prefetch, use_orig_params, sync_module_states, device_id)`、`ShardingStrategy`、`MixedPrecision`、`StateDictType`、`FSDP.state_dict_type`、`FSDP.optim_state_dict / optim_state_dict_to_load`、`torch.distributed.checkpoint.save_state_dict/load_state_dict` | `train_fsdp.py`、`ckpt_reshard.py`、`tests/test_reshard_cpu.py` |
| MiniMind checkpoint | `lm_checkpoint()`（`trainer_utils.py` L63–117）：`_resume.pth` 含 `model/optimizer/epoch/step/world_size/scaler` | `ckpt_reshard.py` |
| MoE | `MoEGate` 不存在——router 就是 `MOEFeedForward.gate: nn.Linear(768,4)`；`MOEFeedForward.forward`、`aux_loss` | `ep_moe.py`、`router_stats.py` |
| EP 通信 | `dist.all_to_all_single(output, input, output_split_sizes, input_split_sizes, group)`、`dist.new_group(ranks)` | `ep_moe.py`、`tests/test_ep_cpu.py` |
| GRPO 角色 | `dist.new_group` ×3、`dist.broadcast`（权重同步）、`dist.send/recv` 或 `broadcast` 传 rollout | `grpo_roles.py` |

最小片段（只说明公式↔代码；完整实现在 `lab/`）：

```python
# ep_moe.py 的 dispatch 核心（伪代码级，勿直接运行）
counts = torch.bincount(dest_rank, minlength=ep_size)           # c_r
recv_counts = torch.empty_like(counts); dist.all_to_all_single(recv_counts, counts, group=ep_group)  # c'_r
perm = torch.argsort(dest_rank, stable=True); x_sorted = x_flat[perm]
recv = x_flat.new_empty(int(recv_counts.sum()), d)
dist.all_to_all_single(recv, x_sorted, recv_counts.tolist(), counts.tolist(), group=ep_group)
```

## 4. 数值行为

### 4.1 为什么 bf16 在 V100 直接报错（`已确认`，周卡硬事实；torch 2.1 `torch/cuda/__init__.py`）

`torch.cuda.is_bf16_supported()` 要求 compute capability major ≥ 8；V100 是 sm70。`autocast(dtype=torch.bfloat16)` 在进入上下文时检查并抛 `RuntimeError: Current CUDA Device does not support bfloat16`。MiniMind 全部脚本默认 `--dtype bfloat16`（`train_pretrain.py` L91），本周一律 `--dtype float16`，此时 L137 的 `GradScaler(enabled=True)` 才生效；`train_grpo.py` 无 GradScaler，lab 补。

### 4.2 autocast(float16) 的实际分工（`已确认`，2.1 amp 文档）

进 fp16：`matmul/linear/bmm/...`（q/k/v/o、gate/up/down、router gate、lm_head）。留 fp32：`softmax, log_softmax, cross_entropy, layer_norm, sum, exp, rsqrt, pow, ...`——因此 router 的 `softmax(scores)` 与 aux loss、RMSNorm 的 `rsqrt`、最终 `cross_entropy` 都在 fp32；SDPA 内部 softmax 由 kernel 自己处理。参数与 `param.grad` 始终 fp32（master weights 就是原参数），这决定了 3.3 节的 4 B/参数。

### 4.3 SDPA 后端差异（`已确认`：torch 2.1 flash 后端只覆盖 sm75–sm90；V100 走 mem-efficient 或 math）

- mem-efficient：分块计算，不物化 $[B,H,T,T]$，支持 fp16/fp32，支持 `is_causal`。
- math：物化注意力矩阵 $[8,8,512,512]$ fp16 = 33.5 MB/层，再加 fp32 softmax 中间量（`计算`）；慢且吃显存，但是数值 reference。
- 本周实验设计：先在 math 下拿 FP32 reference，再切 mem-efficient + fp16，比较 loss 差是否在 $10^{-3}$ 量级（`来源计划假设`，Day 1 门阈值）。

### 4.4 GradScaler 跳步的日志特征

正常：scale 保持 65536 或每 2000 步翻倍。异常序列：`scaler.get_scale()` 连续输出 65536→32768→16384→…（每次 inf 减半），同时 loss 被打印（`loss.item()` 不受跳步影响）但参数不变、`lr` 正常推进。**反复跳步**（scale 掉到 $<1$）意味着不是 fp16 溢出而是模型本身发散（NaN 源于前向），此时应看 `clip_grad_norm_` 返回的 total_norm 是否早已为 inf。MiniMind 把 `scaler.state_dict()` 存进 `_resume.pth`（L145 读回），恢复时 scale 连续是 I1 的一部分。

### 4.5 aux 系数 $\lambda$ 的两端

$\lambda\to0$：router 只被主 loss 驱动，早期随机偏置被放大（富者愈富），热点专家出现，$f_{max}\to1$。$\lambda$ 过大（如 1e-1）：$\mathcal{L}_{aux}$ 梯度把 $P_e$ 压向均匀，router logits 被拉平，专家失去分工，`logits_loss` 相对 dense 基线变差。MiniMind 日志已经分开打印 `logits_loss` 与 `aux_loss`（L58），Day 4 扫描 $\lambda\in\{0,10^{-3},10^{-2},10^{-1}\}$ 时以 `logits_loss` 为主指标、以 $\max_e f_e/(1/E)$（热点比）为副指标。

### 4.6 capacity factor / token dropping / dropless 的三方权衡（本周只解释）

capacity $C=\text{cf}·\frac{Nk}{E}$。cf=1 且均衡时刚好装满；热点专家超过 $C$ 的 token 被 drop（残差直通，$y_i=0$）→ 主 loss 变差但 step time 有上界；cf 加大 → padding 浪费计算、显存随 cf 线性增；dropless（MiniMind 现状、MegaBlocks）→ 不丢 token，但最慢 rank 决定 step time（第 5 节）。三者不可兼得：**质量（不丢）、显存/计算上界（有 C）、无空等（均衡）** 里最多选两个。

### 4.7 seed 与 DDP 等价

`setup_seed(42 + rank)`（L112）让各 rank 的 `torch.manual_seed` 不同。因为 `dropout=0.0`、初始化后 DDP 构造时会从 rank 0 广播 `state_dict()`（`已确认`，DDP notes），所以模型初值一致；但任何依赖 RNG 的前向（dropout>0、随机 mask）都会破坏 I1。2 卡与 8 卡比较时，合理差异来源只有 fp16 累加顺序（allreduce 与 reduce 树不同）——量级 $10^{-3}$ 相对；不合理来源：样本集合不同（`DistributedSampler` 用 `num_replicas=W` 切分，2 卡与 8 卡的第 $k$ 个 step 拿到的样本并集不同，除非 lab 用固定的全局索引表按 $G$ 切）、accumulation 边界不对齐、scaler 初值不同、`clip_grad_norm_` 在 unscale 前调用。

## 5. 硬件与分布式代价（8×V100 32 GB，非全 NVLink；只给符号模型与记录法）

### 5.1 通信/计算比模型（每层）

$$t^{cmp}_\ell=\frac{6\,P_\ell\,N_{tok}}{\eta F}\quad\quad t^{ag}_\ell=\alpha+\frac{(W-1)}{W}\frac{S_\ell}{\beta_{path}}$$

$F$ 峰值算力、$\eta$ 小矩阵利用率（$d=768$ 时远低于 1）、$\alpha$ 每次集合通信的固定延迟（NCCL launch + 同步）、$\beta_{path}$ 取决于该组内**最慢链路**（NV1/NV2/PCIe 混合拓扑下由跨 PCIe 的那一对决定，`用户自述待核验`）。

DDP：每步只有 $\approx11$ 次 allreduce，且都在反向中重叠，暴露时间 $\approx\max(0,\;t_{allreduce}-t_{bwd})$。
FSDP FULL_SHARD：每层 forward 前必须等 AllGather 完成才能算（2.1 默认 `forward_prefetch=False`，只能靠 `BACKWARD_PRE` 在反向预取），暴露时间 $\approx\sum_\ell \max(0, t^{ag}_\ell - t^{cmp}_{\ell-1}) + 27\alpha$ 级别的固定开销。当 $P_\ell$ 只有 7.4M 时 $t^{cmp}_\ell$ 与 $\alpha$ 同量级，$27\alpha$ 就能超过 DDP 的整个通信暴露——这是"64M 上 FSDP 比 DDP 慢"的原因链（`推断`）：**参数小 → 每层计算时间短 → 通信固定延迟无法被藏住 → 集合通信次数×延迟主导 → 再加 FSDP 的 CPU 侧 flatten/unflatten 与更多 kernel 启动**。SHARD_GRAD_OP 去掉 9 次反向 AllGather，应介于两者之间；这就是 Day 2 的 hypothesis。

**记录法**：只记 $r_1=t_{FSDP}/t_{DDP}$、$r_2=t_{SGO}/t_{DDP}$、$r_3=$ 单步 profiler 里 NCCL kernel 时间占比（抽象为 <25% / 25–50% / >50% 三档）、峰值显存比 $m_{FSDP}/m_{DDP}$。不记绝对 ms、不记 GB。

### 5.2 EP 的木桶效应

EP 组内 step time = $\max_r\big(t^{a2a}+t^{expert}_r\big)$，$t^{expert}_r\propto n_{recv,r}$。热点专家使某 rank 的 $n_{recv}$ 达到 $E_p·N_{tok}$（上限 16384），其他 rank 在 combine 的 `all_to_all_single` 上等待。**记录法**：每 rank 打 `t_compute_r`、`t_wait_r`（combine 前 `torch.cuda.synchronize()` 计时，lab 的 `ep_moe.py` 计时器），报告 $\max_r t_{compute}/\text{median}_r t_{compute}$ 与 $\text{mean}_r t_{wait}/t_{step}$；再报告 router bias 从 0→4 时热点比 $\max_e f_e·E$ 的变化方向。

### 5.3 拓扑感知的 EP 分组

EP=4 时两种分组 {0,1,2,3}/{4,5,6,7} 与 {0,2,4,6}/{1,3,5,7} 的 $\beta_{path}$ 可能不同（取决于 NVLink 邻接）；只记 $t_{a2a}^{groupA}/t_{a2a}^{groupB}$ 的比值与"是否 >1.2"的判断，不记拓扑矩阵。

### 5.4 GRPO 三模型显存分解（`计算`；RM 大小为 `来源计划假设`，来自 `input_info` 转述的 1.8B InternLM RM）

| 角色 | 静态显存（fp16 权重 / fp32 训练态） | 动态显存 |
| --- | --- | --- |
| Policy 63.9M | fp32 参数 256 MB + 梯度 256 MB + Adam 512 MB ≈ 1.0 GB；FSDP 8 卡后每卡 ≈ 128 MB | 训练前向激活 + logits $[B·G, T_p+T_g, 6400]$ fp32：B=2,G=6,T=1792 时 ≈ 550 MB；生成期 KV cache $2·L·B·G·T·H_{kv}·d_h·2\,\text{B}$ ≈ 8·2·12·1792·384·2 ≈ 264 MB |
| Reference 63.9M | fp16 128 MB（冻结） | 一次前向 logits 550 MB |
| Reward 1.8B | fp16 ≈ 3.6 GB（冻结） | 一次前向的激活 |

所以 "Policy-FSDP + Ref/RM 复制" 在每卡节省的只是 ~0.9 GB 的训练态，RM 的 3.6 GB 仍复制 8 份；角色拆分（rank 0–3 policy、4–5 rollout、6–7 reward）才真正把 RM 从 policy 卡上移走。同步版一轮的流程与通信：rollout 卡用**上一轮**权重生成 → 把 `(prompt, response, logprob)` 发到 reward 卡（`send/recv`）→ reward 卡回传标量奖励到 policy 组 → policy 组 FSDP 更新 → `FULL_STATE_DICT`（rank0_only）→ `broadcast` 到 rollout 卡（128 MB fp16）。**生成是瓶颈**（`推断`）：生成 $T_g$ 个 token 需要 $T_g$ 次串行前向，每次只处理 $B·G$ 个 token，64M 模型的每次前向由 kernel 启动延迟主导（每层约十几个 kernel × 8 层 × 1024 步）而非 FLOPs，且 rollout 卡只有 2 张，其余 6 张在等；训练一步的 FLOPs 只相当于生成期的一次前向×$T$ 的并行版本。

> **单位与 dtype 约定（读第 3、5 节的字节数前先看这条）**：本文所有 MB 一律是十进制 $10^6$ 字节（6.29 MB 恰好是 6.0 MiB，25.2 MB 恰好是 24.0 MiB，按 MiB 换算会以为算错）。另外，FSDP 的 335 MB 与 DDP 的 447 MB **两侧 dtype 不同**——FSDP 以 fp16 通信参数、DDP 归约 fp32 梯度，$335.5/447.4=0.75$ 恰好就是 $\frac{3\times2\,\text{B}}{2\times4\,\text{B}}$，$(W-1)/W$ 在两边完全抵消。这是 MiniMind 当前代码路径下的比较，不是“FSDP 结构上比 DDP 省带宽”的普遍结论：若 DDP 也用 fp16 归约就是 224 MB，反而比 FSDP 少三分之一。所以“64M 上 FSDP 更慢”的论据是**集合通信次数 × 延迟**，不是字节量。

## 6. 失败模式与识别方法（症状 → 不变量 → 最小检查）

| # | 症状 | 破坏的不变量 | 最小检查 |
| --- | --- | --- | --- |
| 1 | 启动即 `RuntimeError: Current CUDA Device does not support bfloat16` | dtype ∈ 硬件支持集 | `torch.cuda.is_bf16_supported()` 为 False；确认命令行有 `--dtype float16` |
| 2 | 某 rank 打印 `NCCL timeout`/watchdog 后进程退出，或所有 rank 静止到 30 分钟（默认 `timeout=1800s`）| 所有 rank 调用相同顺序、相同数量的集合通信 | 用 `py-spy dump` 看各 rank 卡在哪个 collective；设 `TORCH_NCCL_ASYNC_ERROR_HANDLING=1` 让超时改为 crash；查 `find_unused_parameters` / MoE 空专家分支（L166 的 `0*sum(p)` 就是为此） |
| 3 | 2 卡×acc4 与 8 卡×acc1 的 loss 曲线在第 1 步就差 $>10^{-2}$ | I1 | 打印每 step 各 rank 的样本索引并集、`scaler.get_scale()`、`accumulation_steps·W·b` |
| 4 | 从 8 卡分片 ckpt 以 4 卡恢复：`size mismatch` 或 `flat_param` 长度错 | 分片 = 函数(world_size, wrap 粒度) | 保存用 `SHARDED_STATE_DICT`+`torch.distributed.checkpoint` 或 `FULL_STATE_DICT(rank0_only)`；恢复时 wrap 结构完全一致；`optim_state_dict_to_load` 重新切分 |
| 5 | EP 训练 hang，无报错，`nvidia-smi` 利用率 100% 或 0% | I2（split 一致） | 每 rank 打印 `counts`、`recv_counts`；断言 `sum(input_split_sizes)==input.shape[0]`、`sum(output_split_sizes)==output.shape[0]`；先用 gloo 2 进程 CPU 复现 |
| 6 | step time 随训练进行变长，`t_wait` 集中在同一个 rank 之外的所有 rank | 负载均衡 $f_e\approx1/E$ | `router_stats.py` 打印 $f_e$、$P_e$、热点比；看 `aux_loss` 是否接近 1（均衡）还是趋向 $E$ |
| 7 | 提高 $\lambda$ 后 `aux_loss` 降到 ≈1 但 `logits_loss` 比 dense 差且不收窄 | router 分工 vs 均衡的权衡 | 对比 $\lambda=0$ 与 dense 基线；看 `scores` 的熵是否接近 $\log E$（被拉平） |
| 8 | scale 从 65536 一路减半到 <1，loss 打印 NaN | fp16 动态范围 / 模型未发散 | `clip_grad_norm_` 返回值；`torch.isfinite(logits).all()`；先回 FP32 reference 看是否也发散 |
| 9 | FSDP 显存**高于** DDP | 期望 $m_{FSDP}<m_{DDP}$ | `limit_all_gathers=True`、检查 `use_orig_params`、wrap 是否退化为单一根 FSDP（无分层则前向峰值 = 全模型） |
| 10 | 恢复后 `step` 数不对 | `_resume.pth` 的 `world_size` 换算（L110–113：`step·saved_ws//current_ws`） | 换算只对"每 step 每 rank 样本数不变"成立；global batch 改变时需按 token 数重算 |
| 11 | kill 一个 rank 后其他 rank 不退出 | 集合通信参与者数量 | 见 #2；无 `ASYNC_ERROR_HANDLING` 时其余 rank 阻塞到 timeout |
| 12 | 角色拆分版 rollout 卡权重滞后一轮 | 同步版每轮 broadcast 完成后才生成 | 在 broadcast 后校验 policy/rollout 的某层参数 checksum 相等 |

**loss 连续性作为恢复不变量**：恢复后第 1 步的 `loss` 应落在中断前最后 20 步的 [min, max] 内且 scale 相同；FSDP 8→4 卡恢复还要求 `optimizer.state_dict()` 的 `exp_avg` 范数在换算后一致（`ckpt_reshard.py` 的断言）。

## 7. 与前后周的连接

- **来自 Week M01**：M01 的单卡 FP16 经验与 FP32 reference 是 4.3 节的比较基线；本周的日志解析与吞吐统计由 `mm_dist/common.py` 自带，**不依赖 M01 的 `mm_probe/`**（两周的 lab 相互独立，无跨目录 import）。
- **去向 Week M03+**：EP 的 `new_group` 与 rank 映射表是后续 TP/PP 组合的模板；FSDP 的 `SHARDED_STATE_DICT` 恢复是 27B 级 CPT（Qwen 系列）的必备；GRPO 角色拆分对应 verl/OpenRLHF 的 Actor/Rollout/Reward 划分；capacity/dropless 权衡在 Tutel/DeepSpeed-MoE 对照（本周可选）中变成可调参数。
- **仍未覆盖**：跨节点 EP、RDMA、DeepEP/融合 kernel、异步 rollout——需 A100/H100 集群。

## 8. Teach-back 清单（闭卷复述）

1. 画出 EP=4/DP=2 的 8 卡表：每张卡的 EP 组、持有的专家、专家梯度 allreduce 伙伴、非专家 allreduce 组；再画一次 MoE 层的通信序列：计数交换 → dispatch a2a → 本地 FFN → combine a2a（反向再两次）。（Gate A0）
2. 说出"64M 上 FSDP 比 DDP 慢"的四段因果链，并指出为什么**不是**字节量（335 MB vs 447 MB）。（Gate A1）
3. 热点专家 debug：从 `t_wait` 分布 → $f_e$ → `aux_loss` 数值 → router logits 熵，四步定位，并说明 $\lambda$ 加大后主指标应看哪一个。（Gate 跨场景 debug）
4. 写出 aux loss 公式与 MiniMind 的 `(load * scores.mean(0)).sum() * E * coef` 的对应；说明均衡与坍缩时的取值。
5. 列出 fixed-global-batch 等价的五个条件，以及 SFT 数据为什么在数学上就不严格等价。
6. GradScaler 的三个数（65536 / 0.5 / 2000）与"反复跳步"和"偶尔跳步"的区别。
7. `all_to_all_single` 两个 split 列表的一致性条件，以及不一致时为什么是 hang 不是报错。
8. FSDP 三种 `StateDictType` 各自在 8→4 卡恢复时要做的事；MiniMind `_resume.pth` 里 `world_size` 的换算何时失效。

## 9. 来源（URL + 核验日期 2026-09-04）

- MiniMind `model/model_minimind.py` @7a6fddd：https://raw.githubusercontent.com/jingyaogong/minimind/7a6fddd63a30c06b2fdd5fac4089922b29bc841b/model/model_minimind.py（`MiniMindConfig` L10–45、`FeedForward` L136、`MOEFeedForward` L148–172、`MiniMindModel.forward` L230）
- MiniMind `trainer/trainer_utils.py` @7a6fddd：https://raw.githubusercontent.com/jingyaogong/minimind/7a6fddd63a30c06b2fdd5fac4089922b29bc841b/trainer/trainer_utils.py（`init_distributed_mode` L44、`setup_seed` L54、`lm_checkpoint` L63–117）
- MiniMind `trainer/train_pretrain.py` @7a6fddd：https://raw.githubusercontent.com/jingyaogong/minimind/7a6fddd63a30c06b2fdd5fac4089922b29bc841b/trainer/train_pretrain.py（L35–47、L91、L112、L121–122、L136–137、L154）
- PyTorch 2.1 FSDP：https://docs.pytorch.org/docs/2.1/fsdp.html（`ShardingStrategy`、`MixedPrecision`、`StateDictType`、`BackwardPrefetch`、`optim_state_dict_to_load`）
- PyTorch 2.1 distributed：https://docs.pytorch.org/docs/2.1/distributed.html（`all_to_all_single`、`new_group`、`timeout=1800s`、`NCCL_ASYNC_ERROR_HANDLING`、gloo 不支持 `reduce_scatter`）
- PyTorch 2.1 DDP notes：https://docs.pytorch.org/docs/2.1/notes/ddp.html（Reducer、bucket、按桶索引顺序 allreduce、rank 0 广播）
- PyTorch 2.1 amp：https://docs.pytorch.org/docs/2.1/amp.html（GradScaler 默认值、autocast fp16/fp32 op 列表、`unscale_` 后再 clip）
- 周卡 `00_WEEK_CARD.md`（bf16 RuntimeError、SDPA sm70 后端、DeepEP/MegaBlocks 不可用）
- `input_info/minimind_5070ti_v100.md`（用户意图与 8 卡实验矩阵；其中 RM 1.8B、GRPO 默认参数为 `来源计划假设`）
