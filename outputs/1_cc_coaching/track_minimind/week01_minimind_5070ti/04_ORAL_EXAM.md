# Week M01 Oral Exam — MiniMind 单机全链路：从一条 JSONL 到 GRPO

> **闭卷。提交答案或自评前不要打开 `05_REFERENCE_ANSWERS.md`。**
> 每题括号内依次是：限时（分钟）· 期望 AI 等级 · 考察维度。
> AI 等级：`A0` = 全程不查文档不问 AI；`A1` = 可查本周 `01_FOUNDATIONS.md` 与 `lab/` 源码，但必须在答案里标注查了什么。
> 维度：`K` 知识回忆 · `A` 应用推算 · `D` 故障诊断 · `E` 工程取舍。
>
> 全卷 28 题，建议分两次完成（第 1 次 R+E+A，第 2 次 D+P+T），两次日期不同即可作为 Gate 的"两个不同日期的证据"。
> 代码基线：MiniMind commit `7a6fddd63a30c06b2fdd5fac4089922b29bc841b`。所有默认值以该 commit 为准。

---

## Recall

### M01-R-01（3 min · A0 · K）

同一份 `MiniMindForCausalLM` 前向，pretrain 与 SFT 的 `labels` 构造规则不同。分别写出：

1. pretrain（`PretrainDataset.__getitem__`）的 `input_ids` 是怎么拼的（首尾各加什么、截断参数、pad 在哪一侧），`labels` 由什么得到、哪些位置被置 -100。
2. SFT（`SFTDataset`）的 `labels` 由哪个函数生成，被置 -100 的是哪几类位置。
3. 一句话说出这两者在数学目标上唯一的差别是什么。

### M01-R-02（3 min · A0 · K）

写出以下 token id：

1. `<|endoftext|>`、`<|im_start|>`、`<|im_end|>` 三个特殊 token 的 id，以及 `<|endoftext|>` 在 dataset 里同时承担哪两个角色。
2. `SFTDataset.generate_labels` 用来定位 label 起点的序列 `bos_id`（写出完整 id 列表和长度）与用来定位终点的序列 `eos_id`（同样写出完整 id 列表和长度）。
3. 该 tokenizer 的类型、词表大小，以及 `tokenizer("任意文本")` 会不会自动加上 BOS/EOS。

### M01-R-03（4 min · A0 · K）

不查文档写出 MiniMind 默认配置的下列值，每项一个数字或一个布尔：

`hidden_size` · `num_hidden_layers` · `num_attention_heads` · `num_key_value_heads` · `head_dim` · `intermediate_size` · `vocab_size` · RoPE 的 θ · `rms_norm_eps` · `tie_word_embeddings` · QK-norm 相对 RoPE 的先后顺序 · attention 实际调用的算子名。

再写出总参数量（保留到 0.1 M）。

### M01-R-04（4 min · A0 · K）

1. 按顺序写出 MiniMind pretrain/SFT/LoRA/DPO 训练循环里与 `GradScaler` 相关的四个调用，以及每个调用做了什么。
2. 写出 `GradScaler` 的四个默认常量：初始 scale、检测到 inf/nan 时的乘数、连续正常多少步后增长、增长乘数。
3. 五个训练脚本（pretrain / full_sft / lora / dpo / grpo）中，哪一个没有 `GradScaler`？

### M01-R-05（4 min · A0 · K）

1. `train_grpo.py` 的默认 `loss_type`、`num_generations`、`batch_size`、`max_gen_len`、`dtype`。
2. 训练结束落盘的两类 checkpoint 文件名格式各是什么；`_resume.pth` 里存了哪些键；其中 `model` 这一项保存前做了什么处理。
3. pretrain / full_sft / lora / dpo / grpo 五个阶段的默认学习率各是多少。

---

## Explain

### M01-E-01（7 min · A0 · K）

围绕 `F.cross_entropy(..., ignore_index=-100)` 回答三问：

1. 默认 `reduction='mean'` 时分母到底是什么集合的大小？说清它不是 B×T、也不是逐样本平均。
2. MiniMind 用 `loss / accumulation_steps` 后逐 micro-batch backward。请说明这与"把整个累积窗口的所有有效 token 放在一起取一次均值"在什么条件下等价、在什么条件下不等价，并指出不等价时哪种样本被高估。
3. 随机初始化的 MiniMind 首步 loss 应该是多少（给数值和推导）？这个数在本周被用来判断哪一类故障？

### M01-E-02（7 min · A0 · K）

`generate_labels` 是按**序列**匹配 `bos_id` 和 `eos_id`，而不是按单个 token id 查找。请解释：

1. 为什么 `<|im_start|>assistant\n` 会占 5 个 token 而不是 2 个；这直接决定了 label 起点相对 `<|im_start|>` 偏移几位。
2. 如果把起点匹配简化成"找到 id 为 1 的位置就开始标 label"，会产生什么后果？用本周的失败模式表里的哪一条来命名这个后果。
3. 如果把终点匹配简化成"遇到 id 为 2 就停"，label 里会少掉哪个 token？这对生成阶段有什么可观察的影响？
4. `generate_labels` 在一条样本被截断、扫到序列末尾都没匹配到 `eos_id` 时怎么处理？

### M01-E-03（8 min · A1 · K）

MiniMind 在 64M 规模上同时用了 GQA（`num_key_value_heads=4`）、tied embedding、RoPE θ=1e6 三个设计。对每一个分别回答"省了什么 / 代价是什么"：

1. GQA：写出它相对 MHA 在**参数量**上每层省多少、在 **KV cache** 上每 token 省多少字节（给算式），以及 `repeat_kv` 在前向里的位置。
2. tied embedding：省多少参数（给数字和占比）；`embed_tokens.weight` 因此收到哪两路梯度，哪一路主导。
3. RoPE θ=1e6：在 `max_seq_len` 只有 340/768 的训练里，这个 θ 让位置信息主要落在哪一类维度上；换取的是什么。

### M01-E-04（8 min · A0 · K）

关于 `train_dpo.py`：

1. 一条 DPO 样本为什么必须同时有 chosen 和 rejected？只用 chosen 做 SFT 与用 pair 做 DPO，目标函数上差在哪。
2. reference 模型是怎么得到的、被设成什么状态、forward 在什么上下文里跑？用一句话说清"如果 ref 也跟着更新"会破坏什么。
3. `logits_to_log_probs` 之后是 `(log_probs * mask).sum(dim=1)`。这里是求和不是求均值，请说出它对**不同长度回答**的 margin 量级有什么系统性影响。
4. 训练刚开始时 loss 应该等于多少（给数值和推导）？训练有效时，哪三个量应该往哪个方向动？

### M01-E-05（8 min · A0 · K）

关于 `train_grpo.py`：

1. 写出组内 advantage 的完整公式，包括标准差用的是有偏还是无偏、加的常数是多少。
2. 写出 CISPO 与 GRPO 在 token 级 loss 上的表达式差别（重点：重要性比值在两者中分别以什么形式出现、哪一处有 `detach`）。
3. 默认的 torch rollout 引擎下，`ratio = exp(logπ_θ − logπ_old)` 在数学上应该等于多少？为什么？这使得默认配置下的 CISPO 实际退化成什么算法？
4. 日志里打印的 `KL_ref` 与损失里用的 KL 估计量是不是同一个东西？说出两者的表达式和取值范围差别。

---

## Apply

### M01-A-01（12 min · A0 · A）

**手算题。** 下面是一条新的 SFT 样本（含 system 与两轮对话）：

```json
{"conversations":[
  {"role":"system","content":"你是一个金融助手。"},
  {"role":"user","content":"什么是久期？"},
  {"role":"assistant","content":"久期衡量债券价格对利率的敏感度。"},
  {"role":"user","content":"谢谢"},
  {"role":"assistant","content":"不客气。"}
]}
```

假设本次采样落在"空思考块被删除"的分支，`max_seq_len=64`，右侧 pad。**本题给定的分段 token 数**（直接使用，不要自己猜 tokenizer 切分）：

| 片段 | token 数 |
| --- | --- |
| `<\|im_start\|>system\n` | 4 |
| `你是一个金融助手。` | 6 |
| `<\|im_end\|>\n` | 2 |
| `<\|im_start\|>user\n` | 4 |
| `什么是久期？` | 5 |
| `<\|im_start\|>assistant\n` | 5 |
| `久期衡量债券价格对利率的敏感度。` | 11 |
| `谢谢` | 1 |
| `不客气。` | 3 |

请回答：

1. 写出渲染后的完整模板字符串（用上面的占位文本）。
2. 序列总 token 数是多少？pad 占据哪些位置下标？
3. 用"位置区间 → label 内容（-100 或自身 id）"的形式列出全部区间，不要漏掉任何一段。
4. 有效 label 数是多少？用本周的不变量 `Σ(assistant 内容 token 数 + 2)` 交叉验证。
5. 哪个位置 t 的 logits 负责预测第一个 assistant 内容 token？最后一个进入 loss 的位置 t 是哪个？
6. 这条样本单独成一个 micro-batch 时，`F.cross_entropy` 的分母是多少？
7. 如果 `max_seq_len` 改成 48，有效 label 数变成多少？为什么？
8. 如果 `pre_processing_chat` 这次又在最前面插入了一条随机 system prompt，有效 label 数会变吗？在什么条件下会变？

### M01-A-02（10 min · A0 · A）

**手算题。** 只用公式 `字节数 = B × T × V × 每元素字节数` 与本周的 CE 显存规则（bf16 logits 一份 + fp32 `log_softmax` 输出及其梯度共两份）计算，写出算式和最终数字（用十进制 MB/GB）：

1. MiniMind SFT 默认 `B=16, T=768, V=6400`，autocast bf16：logits 张量多少 MB？CE 内部 fp32 部分多少 MB？两者合计多少 MB？
2. 保持 `B=16, T=768` 不变，把词表换成 151,936（本周 §7 提到的 Qwen 量级）：logits 多少 GB？CE fp32 部分多少 GB？合计多少 GB？
3. 单条样本（`B=1, T=768`）在 V=151,936 下，logits + CE 合计多少 GB？
4. 在 16 GB 卡上，如果要求"logits + CE"这一项不超过 4 GB，V=151,936、T=768 时 `batch_size` 最大能取多少？
5. 用一句话说明：为什么在 MiniMind 上 logits 不是显存主项，而换到 Qwen 量级词表后它会变成主项。

### M01-A-03（10 min · A0 · A）

**手算题。** pretrain 默认 `batch_size=32, accumulation_steps=8, world_size=1, max_seq_len=340, learning_rate=5e-4`，调度函数 `get_lr(s, S, lr) = lr·(0.1 + 0.45·(1 + cos(π·s/S)))`。

1. 一个优化步吃多少条序列？该步进入 loss 的位置数上界是多少（给算式）？为什么是"上界"？
2. `get_lr` 在 s=0、s=S/2、s=S 三点的值各是多少（给到 2 位有效数字）？
3. 某个累积窗口只有 2 个 micro-batch：第一个有效 token 数 1000、loss 2.0；第二个有效 token 数 100、loss 4.0。分别算出：(a) MiniMind 实际反传的等效 loss；(b) 把 1100 个 token 放在一起取一次均值的 loss；(c) 两者之差。指出哪一类样本因此被高估。
4. 显存不够，把 `batch_size` 从 32 降到 8、`accumulation_steps` 从 8 提到 32。回答三问：global batch 变了吗？`get_lr` 作为"数据位置的函数"变了吗？第 3 问揭示的那个偏差是变大还是变小？

### M01-A-04（10 min · A0 · A）

**手算题。** GRPO 组内 advantage，`std` 用总体标准差（`unbiased=False`），分母加 1e-4。

1. `num_generations=6`，某个 prompt 的一组 reward 为 `[1.75, 1.75, 2.25, 0.75, 1.75, 1.75]`。算出组均值、总体标准差、6 个 advantage（保留 4 位小数），并验证不变量"组内 advantage 之和"。
2. 同一 prompt 换一次采样，6 条 reward 全部等于 1.75。算出 advantage，并写出此时 token 级 loss 里还剩下哪一项、它把策略往哪里推。
3. `num_generations=2`，两条 reward 分别为 `[2.25, 0.75]`。算出两个 advantage。
4. 再取 `num_generations=2`、reward 为 `[2.00, 1.90]`。算出两个 advantage，并据此给出一个关于"G=2 时 advantage 取值"的一般结论。这个结论对"reward 的绝对差距大小"意味着什么？

---

## Debug

### M01-D-01（12 min · A1 · D）

两次有界训练的日志片段如下。

**Run A**（`train_full_sft.py`，`--dtype bfloat16`，从 `pretrain_768.pth` 起步）：

```text
[SFT-A] Epoch:[1/1](  20/1000) loss:0.000000 lr:9.98e-06 grad_norm:0.000000 epoch_Time:11.8min
[SFT-A] Epoch:[1/1](  40/1000) loss:0.000000 lr:9.95e-06 grad_norm:0.000000 epoch_Time:11.8min
[SFT-A] Epoch:[1/1](  60/1000) loss:0.000000 lr:9.90e-06 grad_norm:0.000000 epoch_Time:11.9min
```

跑满 200 步后，用固定 prompt 生成，输出与起步 checkpoint 逐字一致。

**Run B**（`train_pretrain.py`，`--dtype float16`，随机初始化）：

```text
[PT-B] step 300 loss:6.3120 scale:65536.0 grad_norm:1.0000
[PT-B] step 340 loss:5.9014 scale:32768.0 grad_norm:1.0000
[PT-B] step 380 loss:nan    scale:8192.0  grad_norm:nan
[PT-B] step 420 loss:nan    scale:2048.0  grad_norm:nan
[PT-B] step 460 loss:nan    scale:512.0   grad_norm:nan
```

对 A 和 B **分别**回答：

1. 被破坏的不变量是哪一条（用可执行的判据表述，不要用形容词）。
2. 至少两个互相排斥的假设。
3. 一个能区分这些假设的最小检查：用 `lab/src/mm_probe/` 里的哪个脚本或哪个函数、看哪个数字、每个假设下预期看到什么。
4. Run A 的 `grad_norm:0.000000` 与 Run B 的 `grad_norm:nan` 分别排除了什么可能性？
5. 这两个故障哪一个可以靠切换 dtype 消除、哪一个不能？说明理由。

### M01-D-02（12 min · A1 · D）

`train_full_sft.py` 有界训练 500 步，与一次已知正确的对照运行放在一起：

```text
对照运行  : step 20 loss 3.41 | step 100 loss 2.62 | step 300 loss 2.11 | step 500 loss 1.94  （曲线有明显抖动）
本次运行  : step 20 loss 1.87 | step 100 loss 1.42 | step 300 loss 1.19 | step 500 loss 1.08  （曲线很平滑）
```

训练结束后，用固定 prompt 做生成：

```text
prompt : <|im_start|>user\n什么是久期？<|im_end|>\n<|im_start|>assistant\n
输出   : 什么是久期？<|im_end|>\n<|im_start|>user\n什么是凸性？<|im_end|>\n<|im_start|>assistant\n凸性是……
```

请回答：

1. 生成结果里有两处明确的角色越界，分别指出来。
2. 被破坏的不变量是什么？写成一个能对单条样本执行的断言。
3. "loss 初值更低且曲线更平滑"这一现象，用一句因果解释它为什么是这个故障的**必然**结果而不是巧合。
4. 给出确认该假设的最小检查：脚本名 + 要看的统计量 + 正常值与故障值的量级差别（不用给精确数，给出判据）。
5. 用 `lab/src/mm_probe/mask_fault.py` 做一次可控复现：写出 `--mode` 取值，以及你要采集哪两条曲线来完成对照。
6. 如果只给你 loss 曲线、不给生成结果，你还能不能区分这个故障与"用了更容易的数据集"？给出一个能区分的实验。

### M01-D-03（12 min · A1 · D）

GRPO smoke run（`batch_size=1, num_generations=2, max_seq_len=256, max_gen_len=256`，规则 reward，reward model 未加载）：

```text
[GRPO] step 10 loss:0.0031 reward:1.7500 adv_std:0.0000 kl_ref:0.00091 gen_len:34.0
[GRPO] step 20 loss:0.0044 reward:1.7500 adv_std:0.0000 kl_ref:0.00214 gen_len:35.2
[GRPO] step 30 loss:0.0068 reward:1.7500 adv_std:0.0000 kl_ref:0.00371 gen_len:34.6
[GRPO] step 40 loss:0.0092 reward:1.7500 adv_std:0.0000 kl_ref:0.00518 gen_len:35.0
```

1. `adv_std` 恒为 0 意味着 advantage 向量是什么？用 `train_grpo.py` 的公式说明为什么它是 0 而不是 nan。
2. `adv_std=0` 时 loss 却不是 0 且在缓慢增大，损失里剩下的是哪一项？它正在把策略推向哪里？这个方向对不对？
3. 列出至少三个能导致"组内 reward 全等"的独立原因（要区分"采样问题"与"reward 函数问题"）。
4. 给出区分这三个原因的实验顺序，每一步用 `lab/src/mm_probe/grpo_stats.py` 的哪个输出做判据。
5. `reward` 恒等于 1.7500 这个具体数值本身提供了什么线索？结合 `calculate_rewards` 的分量构成说明。
6. 如果你只被允许改一个超参数来让学习信号出现，你改哪个、改到多少、预期 `grpo_stats.py` 的哪个数字先动？

### M01-D-04（12 min · A1 · D）

GRPO 用**默认配置**（`batch_size=2, num_generations=6, max_seq_len=768, max_gen_len=1024, bf16`）在 5070 Ti 16 GB 上启动，第一步就崩：

```text
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 1.15 GiB
(GPU 0; 15.99 GiB total capacity; 13.71 GiB already allocated; 0.42 GiB free)
  File ".../model/model_minimind.py", line ***, in forward
    scores = F.softmax(scores.float(), dim=-1).type_as(xq)
```

把 `batch_size` 改成 1（其他不变）后能跑完 30 步，`torch.cuda.max_memory_allocated()` 峰值 5.9 GB。

1. 从 traceback 的那一行可以直接读出模型走的是哪条 attention 路径。写出触发这条路径的**完整条件**（三个子条件都要写）。
2. `batch_size=2` 为什么会满足该条件而 `batch_size=1` 不会？和 prompt 长度有什么关系？
3. 反推 `scores` 张量的 shape（四个维度都要写出来源），并算出它的 fp32 大小，验证是否与报错里的 1.15 GiB 一致。注意区分 GB 与 GiB。
4. 峰值从 13.71 GB 掉到 5.9 GB，远超"减半"。用第 1–3 问的结论解释为什么不是线性关系。
5. 写出这条故障对应的不变量断言，以及用 `lab/src/mm_probe/hooks.py` 和 `grpo_stats.py` 各采集什么来长期监控它。
6. 如果必须保持 `batch_size=2`（例如为了让每步覆盖两个 prompt），有哪两条不改变 attention 分支条件的路可以走？各自的代价是什么。

### M01-D-05（12 min · A1 · D）

有界 pretrain 跑到 step 600 手动中断，从 `_resume.pth` 恢复：

```text
中断前最后 5 步 loss : 2.412  2.398  2.405  2.389  2.401     （最后 100 步均值 2.403，标准差 0.021）
恢复后前 5 步   loss : 2.734  2.681  2.655  2.640  2.612
恢复后第 80–100 步   : 均值 2.441，标准差 0.024
```

1. 把观察到的现象拆成两部分：哪一部分是本周已知机制的**预期后果**，哪一部分需要额外解释？
2. 列出三个导致恢复后无法 bitwise 复现的机制（本周明确点过的），并对每一个说明它更可能造成"一次性跳变后回落"还是"永久性抬高"。
3. `_resume.pth['model']` 的存储精度是什么？由此推算：从 fp32 master 权重舍入到该精度，对一个典型量级 0.02 的权重带来的相对误差是多少？这个误差量级能不能解释 2.40 → 2.73 的跳变？给出你的判断和依据。
4. `SkipBatchSampler` 跳过已消费的 batch，但 `SFTDataset`/`PretrainDataset` 里有基于 `random.random()` 的数据增广。说明这会造成什么，以及它影响的是数据分布还是单条样本。
5. 设计一个可执行的"恢复正确性验收判据"：给出比较的窗口长度、统计量、通过阈值，并说明为什么不能用"逐步 loss 相等"作判据。
6. 如果恢复后 loss 永久停在 2.44 不再回到 2.40，你怀疑哪一项没被正确恢复？给出一个能直接证伪的检查。

---

## Design

### M01-P-01（15 min · A1 · E）

周卡建议 GRPO 在 16 GB 上从 `batch 1 / gen 2 / seq 256` 起步。请论证并设计验证：

1. 写出你的核心论点：`batch_size=1` 在这里改变的**不是**计算量的多少，而是什么。用两句话说清机制。
2. 设计一个能直接证明该论点的实验：至少三个配置点、每个配置点记录哪些量（至少包含一个分支标识和一个显存数字）、预期结果表格的形状。
3. 这个实验用 `lab/src/mm_probe/hooks.py` 的哪个函数取显存、需要在 `Attention.forward` 上挂什么样的 hook 才能把"走了哪条分支"变成可记录的证据？
4. 给出你的实验会失败（即无法区分两个解释）的边界条件：什么情况下 `batch_size=2` 也不会触发另一条分支？
5. 给出一个替代解释（"显存下降只是因为 token 数减半"），并说明你的实验设计中的哪一个配置点专门用来排除它。
6. 可验证的下一步：在证明成立后，你会用什么规则决定 `num_generations` 的上限，而不是靠试到 OOM？

### M01-P-02（15 min · A1 · E）

你要判断"MiniMind 自带的 6400 词表 tokenizer 是否适合中文金融文本"。已知本周核验过的事实：该 tokenizer 是 ByteLevel BPE、词表 6400、通用中文约 1.5 字/token。

1. 定义你要测量的指标：至少给出两个（一个长度类、一个覆盖类），写出各自的计算公式和分母。
2. 设计语料切分：至少列出四类金融文本片段（要覆盖数字、代码型标识、专业术语、中英混排），说明每一类为什么必须单独统计而不能混在一起算平均。
3. 给出可执行的检查步骤：用 `lab/src/mm_probe/minimind_env.py` 的哪个函数拿 tokenizer、用 `inspect_dataset.py` 的哪条路径做编码、输出什么表格。
4. 给出判定门槛：fertility 差到什么程度你才认为"不适合"？把这个门槛翻译成两个下游后果（一个关于 `max_seq_len`、一个关于训练成本），用本周的公式给出换算方式。
5. 假设测出来确实很差，列出三条可选对策，并给出各自在本周知识范围内可预见的代价（提示方向：词表大小与 logits 显存的关系、tie_word_embeddings、已有 checkpoint 的可用性）。
6. 失败边界：你的这套测量在什么情况下会给出误导性结论？

### M01-P-03（15 min · A1 · E）

把本周这条链路（数据 → label → pretrain/SFT → LoRA → DPO → GRPO + 自写仪表）迁到一个 0.6B 量级的 Qwen 基座上。**不要背 Qwen 的具体配置数字**，而是给出"必须重新读取/重新推导的量"清单。

1. 数据与 label 层：列出所有会失效的常量与规则（至少三项），并说明每一项应该用什么方法在新基座上重新确定，而不是猜。
2. 模型层：本周记住的哪些形状常量会全部作废？写出重新推导所需的最小输入集合（即从 config 里读哪几个字段就能重算出参数量、KV cache/token、激活/token）。
3. 显存层：用本周的三条公式（参数+梯度+优化器、`B×T×V×bytes`、激活/token）说明在 16 GB 上 0.6B 全参 SFT 是否可行；如果不可行，给出你会先动哪个旋钮以及为什么是它。
4. LoRA 层：`apply_lora` 的挂载条件是 `in_features == out_features`。说明这个条件迁到新基座后会带来什么问题，以及你会怎么改判定规则。
5. 后训练层：DPO 的 ref 与 GRPO 的 policy/ref/reward 三模型同驻，在新基座上分别多占多少（用参数量的倍数关系表述）。给出一个仍能在 16 GB 上做 GRPO 的最小可行方案。
6. 仪表层：`lab/src/mm_probe/` 里哪些脚本可以原样复用、哪些必须改？对必须改的每一个，写出改动点。
7. 迁移完成的验收标准：列出三条你要在新基座上重新跑通的不变量断言。

### M01-P-04（15 min · A1 · E）

Gate 的第三题是"只凭 loss 曲线与固定 prompt 生成对照反推 mask 故障"。请把它设计成一套可执行协议。

1. `lab/src/mm_probe/mask_fault.py` 目前实现了两种故障模式，写出它们的名字和各自对 `labels` 做了什么。
2. 第三种故障是"assistant 段进 loss 但丢掉结尾的 `<|im_end|>\n`"。它当前不在 `FAULT_MODES` 里。写出你会怎么注入它（描述对 `generate_labels` 输出的具体改动，以及要保证不破坏的性质）。
3. 为这三种故障各写出一条**只看 loss 曲线**的判别特征，以及一条**只看固定 prompt 生成**的判别特征。指出哪一对特征是充分的、哪一对只是必要的。
4. 设计对照实验矩阵：几个运行、每个运行的配置、共享哪些随机种子与数据切片、跑多少步、采什么日志。说明为什么必须有一条"正常"基线而不能只跑三个故障。
5. 给出一条"两个故障可能被混淆"的场景，以及你会加哪一个额外观测量来拆开它们。
6. 用 `lab/src/mm_probe/parse_log.py` 与 `inspect_dataset.py` 说明这套协议里哪一部分是自动化判据、哪一部分必须人看。

### M01-P-05（12 min · A1 · E）

GRPO 损失里有 β·KL 项把策略锚在 ref 附近。请设计一个用有界实验确定 β 的协议。

1. 分别描述 β 过大和 β 过小的可观测症状，每种至少两个可测量的信号（不要用"效果不好"这类描述）。
2. 日志里的 `KL_ref` 与损失里用的 k3 估计量不是同一个东西。说明在设计协议时这带来什么陷阱，以及你要让 `grpo_stats.py` 额外记录什么。
3. 设计扫描协议：β 取几个点、每个点跑多少步、固定住哪些变量、每个点采集哪三类指标（策略侧、锚定侧、能力侧）。
4. 能力侧指标怎么测？用 `lab/scripts/eval_generate.py` 的思路描述一组固定 prompt 应该怎么选，以及为什么不能用训练用的 RLAIF prompt。
5. 给出停止条件与失败边界：什么情况下这个扫描本身是无效的（提示方向：advantage 恒为 0 时 β 的扫描还有没有意义）。
6. 可验证的下一步：选定 β 之后，你用哪一个不变量在后续长跑中持续监控它仍然合适？

---

## Trade-off

### M01-T-01（12 min · A1 · E）

`num_generations` 从 2 提到 6。请给出完整取舍分析。

1. 显存侧：写出 G 影响哪几项显存（至少三项），每一项与 G 是什么关系（线性/平方/无关）。用本周的估算公式给出 G=2 与 G=6 在 `batch_size=1, max_seq_len=256, max_gen_len=256` 下的量级差别。
2. 时间侧：G 增大主要拖慢哪一段？结合"GRPO 每步时间构成"说明为什么这一段的占比会随 G 上升。
3. 信号侧：结合 M01-A-04 第 4 问的结论，说明 G=2 的 advantage 有什么结构性缺陷；G 增大后 advantage 的哪个性质改善了。
4. 给出"整组作废"的概率角度：为什么 G=2 时 `std=0` 的组占比会显著高于 G=6？
5. 给出你的决策：在 16 GB、每天可用 2 小时的约束下选哪个 G，写出你的判据（一个可测量的阈值），以及在什么证据出现时你会改主意。
6. 替代方案：如果显存不允许提高 G，还有哪两条路能改善 advantage 质量？各自代价是什么。

### M01-T-02（12 min · A1 · E）

MiniMind 的 `calculate_rewards` 由四个分量构成：长度项、思考块项、3-gram 重复惩罚、reward model 分数。

1. 写出前三项（形状类奖励）各自的取值区间，并算出一条"只满足形状、内容完全无意义"的 response 能拿到的分数上界。
2. 说明这个上界为什么构成 reward hacking 的直接激励。给出两个你预期会在生成样本里看到的具体退化形态。
3. 形状类奖励也有正面作用。说出它在训练早期解决了什么问题（提示方向：如果只有 reward model 分数，早期组内 reward 的方差会怎样）。
4. 给出你的取舍：形状项与 reward model 分数的相对权重，你会怎么设，依据是什么可测量的量。
5. 监控方案：`grpo_stats.py` 应该把总 reward 拆成几路记录？给出一条能在总 reward 上升时判定"这是 hacking 不是进步"的判据。
6. 失败边界：reward model 本身也可能被 hack。在本周的条件下（1.8B reward model、单卡、有界步数），你能不能证明它没被 hack？如果不能，你会怎么标注这个结论的可信度。

### M01-T-03（12 min · A1 · E）

在 16 GB 上做 SFT，选全参还是 LoRA（rank 固定 16、只挂到方阵 Linear）。

1. 显存账：分别写出全参与 LoRA 在"参数 / 梯度 / 优化器状态 / 激活"四项上的数字或关系。明确指出哪一项 LoRA **不省**，以及为什么。
2. 参数账：算出 LoRA 可训参数量与占比；说出哪几个 Linear 挂上了、哪几个没挂上，以及"没挂上"意味着模型的哪部分能力不会被调整。
3. `B` 零初始化保证首步等价基座。这个性质在训练上带来什么好处，在调试上带来什么好处？
4. 给出你的选择与判据：在本周"有界训练 + 故障注入 + 需要和对照曲线比较"的目标下，哪一个更合适？给出至少两条理由，其中一条必须与"可比较性"有关。
5. 反例：说出一个本周场景中 LoRA 会给出误导性结论的情况。
6. 迁移视角：把这条判断推到 0.6B 或更大基座上，第 1 问里那个"LoRA 不省"的项会不会变成主要矛盾？说明理由。

### M01-T-04（12 min · A1 · E）

第 3 周要上 V100（只有 fp16，没有 bf16）。现在要决定：本周是否值得在 5070 Ti 上专门用 `--dtype float16` 跑一次有界训练。

1. 先说清 bf16 与 fp16 的格式差别（指数位/尾数位/最大值/最小正规数），并据此解释为什么 bf16 下 `GradScaler(enabled=False)` 是安全的。
2. bf16 的代价是尾数只有 7 位。指出本周哪两类量必须在 fp32 里算，以及框架是靠什么机制保证的。
3. 收益侧：在 5070 Ti 上跑 fp16 能提前拿到哪些第 3 周会用到的证据？至少列两条可归档的曲线或数字。
4. 风险侧：`train_grpo.py` 没有 `GradScaler`。写出 fp16 下它的两种失败方式，并指出哪一种是静默的、为什么静默的那种更危险。
5. 给出你的决策，并把它拆成"做哪些阶段、不做哪些阶段"。说明不做的那部分你打算用什么替代证据覆盖。
6. 可验证的下一步与失败边界：你要采集的 `scaler.get_scale()` 序列，健康形态是什么样？出现什么形态时你判定"这是 lr 或初始化问题，不是精度问题"，因而这次 fp16 实验不能作为第 3 周的证据？

---

## 自评说明

- 每题先闭卷作答并写下用时，再打开 `05_REFERENCE_ANSWERS.md` 按 0–4 分 rubric 自评。
- 记录三类偏差：**记错的常量**（对应 Recall 层）、**说不清机制**（Explain/Apply 层）、**给不出可执行判据**（Debug/Design/Trade-off 层）。第三类是本周唯一真正卡 Gate 的一类。
- 任何一题给不出"假设 / 证据 / 替代解释 / 失败边界 / 可验证下一步"五要素中的三条以上，即使结论正确也按 2 分记。
