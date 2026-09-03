# Week 01 基础篇：从 TinyStories 原文到可恢复 MiniGPT

> 这是“从零实现”的原理与契约手册，不是运行报告。文中的命令、阈值和输出均为待验证计划。公司 `8×V100 32GB`、个人 `RTX 5070 Ti 16GB` 以及拓扑均来自用户自述，必须以 Day 0 探针为准。公司环境固定在 PyTorch 2.1，不强行升级；V100 支持 FP16 Tensor Core，通常不支持原生 BF16，本周禁用 BF16、TF32 和官方 FlashAttention-2。

## 1. 本周问题、边界与最终产物

本周只回答一个问题：能否从两个公开文本文件开始，亲手构造一个可解释、可测试、可恢复的 decoder-only 语言模型训练闭环？

```text
TinyStories UTF-8 文本
  → 确定性 byte-BPE（vocab/merges/special token）
  → uint16 token stream（train/valid 严格分开）
  → 连续窗口 x/y
  → token + position embedding
  → 6 层 pre-norm causal Transformer
  → [B,T,V] logits 与 next-token NLL
  → AdamW + warmup/cosine + FP16 AMP
  → 全状态 checkpoint/resume
  → token-weighted validation NLL/PPL + 固定 prompt 生成
```

最终证据不是一句“loss 降了”，而是：来源与 SHA-256 manifest、tokenizer 可逆性测试、attention oracle、128-window overfit、FP32/FP16 smoke、连续训练与 resume 对照、validation 汇总和固定 prompt 错误桶。

Week 02 会复用 tokenizer/decoder、answer-only loss mask 和 Trainer；Week 03 会复用模型作为单卡 FP16 载荷。这里如果数据、mask 或 checkpoint 不可信，后面所有吞吐和扩展结论都没有意义。

## 2. 前置知识与机器分工

读者应会基础 Python、PyTorch `nn.Module`、矩阵乘法和反向传播，但不需要会 tokenizer 或分布式。

| 平台 | 本周职责 | 明确禁区 |
|---|---|---|
| 公司 Linux / V100 | 主训练、FP16、Profiler；只用公开且获批输入 | 不外传代码、数据、日志、trace、图、checkpoint、路径、硬件数字；不登录外部追踪平台 |
| 个人 5070 Ti | 用公开数据独立重跑缩小版；验证较新消费卡环境 | 不接收或复用任何公司产物；不要为了“统一”降到公司 PyTorch 2.1 |
| 可选 H100 | 不属于本周必需资源 | 不把 H100/BF16/FA2结果混入 V100 FP16 基线 |

“不新建环境”在本教程中指复用当前已批准 Conda 环境，不创建新的 Conda 或 venv。缺包先 `pip freeze`、`pip check` 和 resolver dry-run；任何会替换 `torch`、CUDA runtime 或 torchvision ABI 的方案都应停止并交给平台管理员。

## 3. 数据、参考实现与版本锁

技术来源核验日期：**2026-09-03**。下面的 revision 是该日解析值；正式执行仍要把“请求 revision”和“下载后 resolved revision”同时写入 manifest，不能在远端变化后静默追随 `main`。

| 对象 | 固定标识 | 本周实际用途 | 许可与边界 |
|---|---|---|---|
| TinyStories | `roneneldan/TinyStories@f54c09fd23315a6f9c86f9dc80f725de7d8f9c64` | `TinyStoriesV2-GPT4-train.txt`、`TinyStoriesV2-GPT4-valid.txt` | dataset card 标注 CDLA-Sharing-1.0；执行前保存 card 并让公司流程确认 |
| CS336 A1 | commit `a158843b20107949f1a8d7df1b05cd33b9166712` | 测试思想与对照阅读，不作为本教程运行依赖 | MIT；保留版权声明 |
| minBPE | commit `1acefe89412b20245db5a22d2a02001e547dc602` | 自己实现完成后做结构对照 | MIT |
| nanoGPT | commit `3adf61e154c3fe3fca428ad6bc3818b27a3b8291` | Day 5 后对照 model/train 设计 | MIT |

正式数据只接受上述两个文件。合成故事只能通过代码正确性门；没有公开 validation 文件时，结论必须是 `INCONCLUSIVE`，不能把合成 loss 写成 TinyStories 结果。

原始文件的最小合同：UTF-8 可解码；train/valid 路径不同且 SHA-256 不同；大小大于 0；抽样文本不是 HTML 错误页；统计字节数、换行数、空白样本数。训练 tokenizer 只读取 train，valid 从不参与 merge、词表、超参选择或 normalization。

## 4. 从字节到 BPE：完整状态机

### 4.1 为什么从 byte 开始

Unicode 字符先经 UTF-8 变成 `0..255` 的 bytes，因此初始词表覆盖任何合法 UTF-8 文本。字符“你”是三个 byte，不是一个初始 token。byte fallback 的代价是初始序列较长，收益是没有普通文本 OOV。

本周 pre-tokenization 固定为 Python Unicode 正则 `\s+|[^\s]+`。它把空白块与非空白块分别作为 piece，BPE merge 不跨 piece；encode 必须使用完全相同的切分。这个选择不是 GPT-2 regex 的替代品，而是可逆、依赖少的教学合同。

### 4.2 训练

将每个 piece 编成 token tuple，记录 piece 频数。每轮：

1. 对每种 piece 统计所有相邻 pair，并乘该 piece 的语料频数；
2. 选择频数最高 pair；并列时选数值字典序最小 pair，保证确定性；
3. 分配新 ID `256 + merge_index`；
4. 在所有 piece 内从左到右非重叠替换；
5. 记录 `pair → new_id` 的先后顺序，这个顺序就是 merge rank。

若目标普通词表大小为 `V_base`，最多执行 `V_base-256` 次 merge。`<|endoftext|>` 在所有 merge 后分配独立 ID `V_base`，不参与普通文本 merge。

### 4.3 编码与解码

编码一个 piece 时，不是重新统计频率，而是反复找“当前相邻 pair 中 rank 最小者”，将其全部非重叠合并，直到没有已知 pair。解码表满足：初始 ID 映射到单字节；每个 merge token 的 bytes 是两个父 token bytes 的拼接。因此普通文本必须满足：

$$\operatorname{decode}(\operatorname{encode}(s))=s.$$

worked example：语料 pieces 中 `[97,98,97,98]`（`abab`），若第一条 merge `(97,98)→256`，编码变为 `[256,256]`；若第二条 `(256,256)→257`，最终为 `[257]`，而 `vocab[257]=b"abab"`。

### 4.4 Tokenizer 不变量

- 中英文、emoji、数字、换行、空串 round-trip 100%；
- save→load 后 token IDs 完全一致；
- 相同 corpus bytes、`max_bytes`、目标词表和 tie-break 产生相同 JSON SHA-256；
- tokenizer JSON 记录格式版本、pre-tokenizer、merges、special token；
- `max(token_id) < model.vocab_size`；
- train/valid `.bin` manifest 都绑定同一 tokenizer hash；
- special token 不被普通 decode 悄悄吞掉。

词表变大通常缩短序列，但 embedding/LM head 增大。若输入输出权重 tying，词表相关参数约为 `V×D`；不 tying则约为 `2VD`。因此压缩率、参数量与 softmax 计算需要共同报告。

## 5. 监督窗口与 loss：一位不能错

给定 token stream `[t0,t1,…,tN]`，上下文长度为 `T`，起点 `i`：

```text
x = [ti,     …, t(i+T-1)]  shape [T]
y = [t(i+1), …, t(i+T)]    shape [T]
```

batch 后 `input_ids/labels` 是 `[B,T]` 的 `torch.long`；embedding 是 `[B,T,D]`；logits 是 `[B,T,V]`。交叉熵接收 `logits.reshape(B*T,V)` 与 `labels.reshape(B*T)`。

自回归分解和 token 平均负对数似然：

$$p(x_{1:T})=\prod_{t=1}^{T}p(x_t\mid x_{<t}),\qquad
\mathcal L=\frac{\sum_{b,t}-\log p_\theta(y_{b,t}\mid x_{b,\le t})}{BT}.$$

具体例子：`B=2,T=4,V=300,D=96` 时，logits `[2,4,300]` 展平为 `[8,300]`，labels 为 `[8]`；总共监督 8 个 target token。若将 `x` 原样当 `y`，模型会学复制而不是 next-token。

validation 要累计 `loss_sum = Σ token_nll` 与 `token_count`，最后 `NLL=loss_sum/token_count`、`PPL=exp(NLL)`。不能先对每 batch 求 PPL 再平均；tokenizer、split 或 mask 不同的 PPL 也不能横比。

## 6. Decoder-only Transformer：shape 账本

固定正式配置：`V=4097`（4096 普通 token + EOT）、`T=256`、`L=6`、`D=384`、`Nh=6`、`Dh=64`、MLP hidden `4D=1536`。实际参数量由代码计算，不凭印象写“20M”。

| 边界 | 具体 shape | dtype |
|---|---:|---|
| ids | `[B,256]` | int64 |
| token/position embedding | `[B,256,384]` | FP32 参数；autocast 内激活可 FP16 |
| fused QKV | `[B,256,1152]` | 同激活 |
| split Q/K/V | 各 `[B,6,256,64]` | 同激活 |
| attention scores | `[B,6,256,256]` | FP16/内部选择；mask 后 softmax |
| attention output | `[B,256,384]` | 同激活 |
| MLP hidden | `[B,256,1536]` | 同激活 |
| logits | `[B,256,4097]` | loss 前可转 FP32 |

Scaled dot-product attention：

$$S=\frac{QK^\top}{\sqrt{D_h}}+M,\quad A=\operatorname{softmax}(S),\quad O=AV.$$

`M[i,j]=0` 当 `j≤i`，未来位置为 `-∞`。mask 必须在 softmax 前进入；softmax 后乘零会让一行概率和小于 1。`1/√Dh` 防止维度增大时点积方差爆炸。

Pre-norm block 是：

$$x' = x + \operatorname{Attention}(\operatorname{LN}(x)),\qquad
x'' = x' + \operatorname{MLP}(\operatorname{LN}(x')).$$

一层参数主项：QKV `3D²`、attention output `D²`、MLP `D·4D+4D·D=8D²`，合计约 `12D²`。attention scores/概率随 `T²` 增长，参数量不随 `T` 增长。

最小可运行的 shape/oracle 片段（完整工程在实践篇）：

```python
import math, torch
B, H, T, Dh = 2, 3, 5, 4
q = torch.randn(B, H, T, Dh)
k = torch.randn(B, H, T, Dh)
v = torch.randn(B, H, T, Dh)
scores = q @ k.transpose(-2, -1) / math.sqrt(Dh)
future = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
prob = scores.masked_fill(future, float("-inf")).softmax(-1)
out = prob @ v
assert out.shape == (B, H, T, Dh)
assert torch.all(prob.masked_select(future[None, None])) == 0
torch.testing.assert_close(prob.sum(-1), torch.ones(B, H, T))
```

## 7. 优化器、全局 batch 与 LR 时钟

AdamW 的动量：

$$m_t=\beta_1m_{t-1}+(1-\beta_1)g_t,\qquad
v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2.$$

偏置修正后做自适应更新，再解耦 weight decay：`θ ← θ - η·adaptive_update - ηλθ`。它与把 L2 项加进 loss 不等价，因为后者会被 Adam 的预条件器缩放。bias、LayerNorm 权重通常放进 no-decay 参数组。

累积 `G` 个 micro-batch 时，每个 micro loss 除以 `G`，完成 G 次 backward 后才更新。单卡每次成功更新的 token 预算是 `B×T×G`。warmup/cosine、`global_step` 和 checkpoint 间隔均按“成功 optimizer update”推进；attempted step 与 successful step 要分开记录。

## 8. FP16 AMP 与 infra 联动

V100 主路径是 FP16 mixed precision：模型主参数和 Adam 状态通常保持 FP32，autocast 选择低精度算子，GradScaler 放大 loss 以降低 FP16 梯度下溢。

```text
zero_grad
→ G 次 autocast forward + scaled backward
→ scaler.unscale_(optimizer)
→ finite/grad norm 检查
→ clip_grad_norm_
→ scaler.step
→ scaler.update
→ 仅成功时推进 LR/step
```

`model.half()` 不是替代方案。loss/reduction 可显式 `.float()`。在单卡本周使用 `torch.cuda.amp.GradScaler`；后续 FSDP/FP16 需使用与 sharding 匹配的 scaler，不应机械照搬。

显存包括参数、梯度、Adam 一二阶状态、activations、temporary workspaces 和 allocator reserved。仅用“参数数×2 bytes”估算会严重偏低。更大 vocab 增加 embedding/head；更长 T 主要放大 attention activation；gradient accumulation 降低 micro-batch activation 峰值，却不减少参数/optimizer state。

应记录：`train_nll, valid_nll, lr, grad_norm, loss_scale, step_skipped, tokens_seen, step_ms, tokens/s, peak_allocated, peak_reserved`。Profiler 只采短窗口，trace 留在执行机器本地。

## 9. Checkpoint 是状态快照，不只是权重

等价 resume 至少保存：

- model、optimizer、LR 时钟、GradScaler；
- attempted/successful step、tokens seen；
- Python、NumPy、torch CPU、所有 CUDA RNG；
- train/valid batch generator 状态；
- 完整 config、所有 `src/*.py` 的组合 hash、torch/env 摘要；
- requested/resolved 数据 revision、tokenizer JSON hash、train/valid bin 和原始文件 manifest。

写盘应先写同目录临时文件，完成后原子替换 `last.pt`。恢复时先以 CPU `map_location` 载入，先比较不可变 lineage/contract，再把 model/optimizer state 复制到既有 device；这样 torch CPU RNG 与 CPU `Generator` 收到的仍是 CPU ByteTensor。只有权重的文件应命名/标注 `weights_only`，不能用于声称优化轨迹等价。

`40 continuous` 与 `20 + resume + 20` 对照必须使用相同初始 seed 和下一批 token IDs；先比较 step 21 输入 hash、LR、loss scale，再比较 loss。若输入不同，模型 loss 的差异没有诊断价值。

## 10. 必须在正式训练前成立的不变量

1. 来源 revision、license 审批、原始 SHA-256 已落盘；
2. tokenizer 10+ 类文本 round-trip 100%，save/load IDs 相同；
3. train tokenizer 从未读取 validation；
4. `.bin` dtype 与最大 ID 匹配，train/valid hash 不同；
5. `y[t]==x[t+1]` 的人工窗口通过；
6. causal 未来概率严格为 0、每行和约 1、oracle 误差 `<1e-5`；
7. 所有应训练参数有 finite gradient；
8. 128 固定窗口能明显过拟合；失败时禁止长训；
9. FP32 20 step finite 后才进入 FP16；
10. resume 的 next-batch hash、LR、scaler、step 连续。

## 11. 失败树：症状到回归测试

| 症状 | 第一检查 | 常见根因 | 最小修复 | 回归门 |
|---|---|---|---|---|
| BPE hash 漂移 | 比 corpus/hash/目标 vocab | 无稳定遍历或 tie-break | 对 piece/pair 稳定排序 | 同输入连续训练两次 hash 相同 |
| 中文 decode 失败 | 比较原始 bytes | 按 token 单独 UTF-8 decode | 合并全部 bytes 后一次 decode | 中文/emoji round-trip |
| loss≈`ln(V)`不降 | 打印首个 x/y | 未 shift、mask 反、LR=0 | 修正最小路径 | 128-window overfit |
| train 降 valid 不降 | 查 split/hash | 重复、过拟合、预处理不一致 | 重建 manifest/split | 固定 valid NLL |
| FP16 scale 连续下降 | 同 batch FP32、首个非有限模块 | logits/grad overflow、clip 顺序错 | FP32 reduction、unscale 后 clip | FP16 200 step，warmup 后 skip<1% |
| OOM | peak allocated/reserved 与阶段 | T/B 太大、评测留图 | 先降 micro-batch/T，保持 global tokens | 20-step memory scan |
| resume 偏离 | 下一批 token hash | RNG/generator/scaler/LR 未恢复 | 补状态并统一保存时点 | 20+20 对照 |
| PPL “异常好” | token count 和 split | 平均 batch PPL、泄漏 | 聚合 token NLL | 手工小例子复算 |

故障分类：环境/程序错误是 `FAIL-SYSTEM`；系统正确但模型假设失败是 `FAIL-MODEL`；公开 validation、正式 run 或关键证据缺失是 `INCONCLUSIVE`。

## 12. Teach-back（先闭卷）

1. 对 UTF-8 字符串 `a你` 写出 byte 数量，并说明为何无 OOV。
2. 手算 `abab` 在两条 merge 后的 IDs 与 vocab bytes。
3. 从 `[2,256]` 写到 Q/K/V `[2,6,256,64]`、scores `[2,6,256,256]`、logits `[2,256,4097]`。
4. 解释为何 causal mask 要在 softmax 前，给出 softmax 后置零的反例。
5. 写出 accumulation=4 的 AMP 顺序，并圈出 LR 何时推进。
6. 解释词表从 1K 到 4K 对序列、embedding 和 softmax 的三向影响。
7. 列出 full checkpoint 与 weights-only 的区别。
8. 为“loss 不降”写出一个最多五步、每步可证伪的诊断流程。

## 13. 官方一手资料

技术链接核验日期：**2026-09-03**。

- [TinyStories dataset card](https://huggingface.co/datasets/roneneldan/TinyStories)
- [Stanford CS336 Spring 2025](https://cs336.stanford.edu/spring2025/)
- [CS336 Assignment 1 Basics](https://github.com/stanford-cs336/assignment1-basics)
- [minBPE](https://github.com/karpathy/minbpe)
- [nanoGPT](https://github.com/karpathy/nanoGPT)
- [PyTorch 2.1 AMP examples](https://pytorch.org/docs/2.1/notes/amp_examples.html)
- [PyTorch 2.1 reproducibility notes](https://pytorch.org/docs/2.1/notes/randomness.html)
