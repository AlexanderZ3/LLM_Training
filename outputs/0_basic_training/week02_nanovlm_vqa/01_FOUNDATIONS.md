# Week 02 基础篇：nanoVLM 222M 的图像前缀、答案监督与视觉反事实

> 本篇描述待执行实验，不是实测报告。公司 `V100 32GB`、个人 `RTX 5070 Ti 16GB` 和软件情况均为用户自述待核验。公司 PyTorch 2.1 是冻结约束；V100 走 FP16 + GradScaler，不用 BF16、TF32 或官方 FlashAttention-2。任何公司代码、数据、日志、图片、trace、checkpoint 和性能信息都不得导出。

## 1. 本周要构建什么

Week 01 已证明 text tokens → causal decoder → answer loss。本周增加一条视觉条件支路：

```text
RGB 224×224 → SigLIP patch tokens [B,196,768]
             → 2×2 pixel shuffle [B,49,3072]
             → modality projector [B,49,576]

question + answer → SmolLM2 tokenizer → [B,T]
visual embeddings + text embeddings → [B,49+T,576]
→ causal LM → 只在 answer+EOS 上计算 next-token loss
```

目标不是从随机初始化重训 222M 基础能力，而是复用公开 SigLIP/SmolLM2 初始化，亲手实现 projector、packing、label mask、冻结/解冻、FP16、resume、独立评测和 image counterfactual。最终产物为一套本地 VQA 训练器及 base/projector-only/light-unfreeze 三组可比证据。

Week 04 会把“image/question condition → answer”替换成“state/image/time/noise condition → action velocity”；本周最重要的迁移能力是 condition 是否真的被模型使用，而非只看训练 loss。

## 2. 前置与平台边界

需要已理解 causal LM shift、AMP、checkpoint 与 train/valid 分离。平台职责：

| 平台 | 可做 | 不做 |
|---|---|---|
| 公司 Linux / V100 | 公开且获批子集的 222M projector/light-unfreeze FP16 | 不安装 current nanoVLM 依赖树、不升级 torch、不登录 W&B/Hub、不导出任何产物 |
| 个人 5070 Ti | 从公开源独立下载并缩小复现 | 不接收公司文件/指标；不强行使用 PyTorch 2.1 |
| 可选 H100 | 仅另立报告；可审计 BF16 | 不把 H100 数字写入 V100 基线 |

不新建 Conda/venv。公司缺包先审计；`torch==2.1.x` 与 torchvision 必须 ABI 配对：2.1.0↔0.16.0、2.1.1↔0.16.1、2.1.2↔0.16.2。不能只升级 torchvision。当前 LeRobot/nanoVLM main 会漂移，本周不用 main 代码训练。

## 3. 官方对象、精确版本与许可

技术链接核验日期：**2026-09-03**。

| 对象 | 固定 revision/tag | 用途 | 许可/注意 |
|---|---|---|---|
| nanoVLM | release `v0.1`，解引用 commit `6ba9082e16f1fc8c21a1f8d0c54b26c9233c8771` | 222M 架构对照；不直接运行其 BF16/W&B 默认训练脚本 | MIT；保留声明 |
| SigLIP B/16-224 | `google/siglip-base-patch16-224@7fd15f0689c79d79e38b1c2e2e2370a7bf2761ed` | vision backbone | model card 标注 Apache-2.0；公司复核 |
| SmolLM2-135M | `HuggingFaceTB/SmolLM2-135M@93efa2f097d58c2a74874c7e644dbc9b0cee75a2` | causal LM backbone | model card 标注 Apache-2.0；公司复核 |
| cosmo2 tokenizer | `HuggingFaceTB/cosmo2-tokenizer@4ce2318a3628e77279c939ed6a9f3f03034402de` | 与 v0.1 契约对齐 | tokenizer 文件随 model card/许可登记 |
| nanoVLM-222M | `lusxvr/nanoVLM-222M@9df794e259cb178223acfb1f3bc3b29521e551c5` | weights-only/base 评测对照，可选 | MIT model card；不与本教程 HF-backbone wrapper 静默混载 |
| The Cauldron | `HuggingFaceM4/the_cauldron@847a98a779b1652d65111daf20c972dfcd333605`，config=`clevr` | 固定公开 VQA 子集 | 集合内来源许可可能不同；保存 config card 与上游 CLEVR 条款，不能假定一个 umbrella license |
| MMStar | `Lin-Chen/MMStar@bc98d668301da7b14f648724866e57302778ab27` | 可选独立 smoke | 保存 dataset card；不把小样本结果冒充完整 benchmark |

为什么锁 v0.1：官方 v0.1 是 222M 教学架构；current main 已改变规模/API。v0.1 官方 `train.py` 默认 BF16、无 GradScaler、可启用 `torch.compile` 且默认 W&B，这些都不适合公司 V100/PyTorch2.1。本教程只借其架构契约，用 Hugging Face 官方 backbone API 加载本地快照，并自写离线 Trainer。

The Cauldron 本周只取 `clevr` 固定 revision 的有界样本。若它只有 train split，则只用规范化图像 bytes 的 SHA-256 做 group split：同一图像的所有问题永远同组，answer/gold 不参与 split 或 sample ID。这样既防同图跨 train/dev，也避免标签决定数据归属；这仍不是“官方 validation”。公开源不可达时可生成彩色几何图做 correctness，但正式结论必须 `INCONCLUSIVE`。

## 4. 图像如何变成 49 个语言 token

SigLIP patch size 为 16。`224/16=14`，所以无 CLS 时共有 `14×14=196` 个视觉 token，每个 768 维。

v0.1 的 pixel shuffle factor 为 2：把空间相邻 `2×2` patch 合并到通道：

```text
[B,14,14,768]
→ [B,7,2,7,2,768]
→ permute [B,7,7,2,2,768]
→ [B,49,3072]
→ Linear(3072,576)
→ [B,49,576]
```

这里不是图像超分辨率中的 pixel shuffle；它是 token 数减少 4 倍、通道增大 4 倍的确定性重排。必须断言 196 是完全平方数且 14 可被 factor 2 整除。

projector 至少承担空间对齐：视觉输出最后维 768 与 LM hidden 576 不同，不能直接拼接。线性层约有 `3072×576≈1.77M` 个权重；只有 projector 训练时，仍需通过冻结 LM 反传输入梯度，不能把整个 LM forward 放进 `no_grad()`。

## 5. Packing、causal attention 与标签 shift

固定文本模板：`Question: {question} Answer:`，后接 answer tokens 与 EOS。右 padding，最长文本 79 token，与 49 image tokens 合计不超过 v0.1 的 128 总长度。

使用 Hugging Face `AutoModelForCausalLM(..., labels=combined_labels)` 时，模型内部执行 shift：位置 `i` 的 logits 预测 combined label `i+1`。因此 labels 与 input IDs 同长度：

```text
text input:   [prompt tokens................][answer tokens][EOS][PAD]
text labels:  [-100........................][answer tokens][EOS][-100]
combined:     49 个 image label=-100 + text labels
```

worked example：`B=2`，每张图 `Nv=49`；样本 A prompt=12 token、answer=3、EOS=1、padding=4，总文本 T=20。则：

- `pixel_values [2,3,224,224] float32`；
- `vision_hidden [2,196,768]`；
- `visual [2,49,576]`；
- `input_ids [2,20] int64`；
- `combined_embeds [2,69,576]`；
- `combined_attention_mask [2,69]`；
- `combined_labels [2,69]`；样本 A 恰有 4 个非 `-100`（3 answer + EOS）。

首个 answer token 的监督来自最后一个 prompt 位置的 logits，这是内部 shift 的结果。若你先手工 shift labels 又交给 HF causal LM，会双重 shift。若自己直接对同位置 logits/targets 求 CE，则必须手工 shift；两种合同只能选一种。

visual tokens 放在文本前，因此每个后续问题/答案 token 在 causal mask 下可看全部 image prefix；image token 彼此按顺序 causal 可见，而不是双向 vision attention。图像内部的双向关系已由 SigLIP encoder 建模。

## 6. Answer-only loss 与指标

有效集合 `S={(b,t): label[b,t] != -100}`：

$$\mathcal L=-\frac{1}{|S|}\sum_{(b,t)\in S}\log p_\theta(y_{b,t}\mid image_b,prompt_b,y_{b,<t}).$$

padding、image、question 均不进入 numerator/denominator。若一个 batch 全被 truncation 丢弃而没有有效 label，应在 collator 直接报错，不把 NaN 或零 loss 混入训练。

normalized exact match 是生成答案归一化后逐样本比较。归一化协议必须固定：Unicode NFKC、trim、lower、折叠空白、可选去英文冠词/标点。数字词到数字的规则若加入，必须写测试。报告：

- token-weighted answer NLL；
- normalized EM；
- invalid generation rate；
- mean answer tokens；
- answer type 桶（yes/no、number、short noun、multiword）。

EM 不可用 teacher-forced argmax 代替自由生成；teacher-forced token accuracy只能作为诊断。

## 7. Freeze、eval 与反事实

`requires_grad=False`：不计算该参数梯度；`torch.no_grad()`：不构建整个上下文的 autograd 图；`model.eval()`：切换 dropout/normalization 行为。三者不等价。

projector-only：vision 与 LM 参数冻结，vision forward 可 `no_grad`，但 LM forward 必须保留对 `inputs_embeds` 的梯度，让 loss 到达 projector。light-unfreeze：只训练 projector + LM 最顶 1 block；固定 SmolLM2 配置 `tie_word_embeddings=true`，所以不能单独把 `lm_head` 标为可训练——它与 input embedding 是同一个 Parameter，会把整个输入词表一起解冻。projector LR 通常比已预训练 block 高 10–100 倍。

只看 original EM 无法证明模型用图。至少做配对反事实：

| 条件 | 改什么 | 排除什么 |
|---|---|---|
| original | 正确 image/question | 主结果 |
| shuffled image | batch 内跨样本固定置换 | 语言偏置、图像恒等映射 |
| blank image | 输入统一灰/零图，仍过 vision encoder | 图像内容贡献 |
| zero visual | projector 输出置零 | 整个视觉前缀贡献 |
| question-only/base | 不经微调的对照 | 初始化与常见答案先验 |

如果 original 与 shuffled/blank 的配对差不明显，只能写“未证明视觉依赖”，不能写“模型已学会视觉”。

## 8. FP16、显存与性能

V100 用 `torch.cuda.amp.autocast(dtype=float16)` + `GradScaler`。parameters/optimizer state 通常 FP32，激活依算子选择 FP16/FP32。CE/reduction 使用 FP32。若 accumulation 内第 `i` 个 micro-batch 有 `n_i` 个 answer+EOS target，其 HF loss 是均值 `L_i`，正确的窗口目标是 `Σ(n_i L_i)/Σn_i`，不是 `ΣL_i/K`；否则长短答案会因分组方式改变权重。每个 attempt 记录 `loss_scale` 和 skip；非有限 FP16 梯度交给 GradScaler 跳过并回退，scheduler 只随成功 update 前进，连续有界次数回退才止损。

主要显存：222M 参数、梯度、Adam state、image/text activations、attention `[B,H,L,L]`、workspaces 和 allocator reserved。视觉 token 从196降49显著降低 LM 序列长度；总长度从 `T` 变 `T+49`，LM attention 近似按 `(T+49)^2` 增长。

先 projector-only batch scan，再 light-unfreeze；OOM 顺序：降低 micro-batch → 降 text max length → 增 accumulation 保持 global batch → 减少解冻层。不得以反复 `empty_cache()` 掩盖峰值。

## 9. Checkpoint 与版本耦合

full resume state：projector/vision/LM weights、optimizer param groups、scaler、successful/attempted step、有效/尝试 target tokens、LR、best metric、Python/NumPy/torch/CUDA RNG、sampler generator/offset，以及不可变 run contract。contract 必须含 config、实际 trainable 参数集合、backbone/tokenizer requested+resolved revisions、records/逐图/artifact/code SHA-256 和依赖版本；resume 在加载权重前重算并拒绝漂移。

`nanoVLM-222M` 官方 checkpoint 与本教程 wrapper 虽使用相同 backbone 思路，但 state_dict 命名和实现不同；不可 `strict=False` 静默加载。若要对照，只通过 v0.1 固定代码单独加载并记录，或写显式 key mapping + shape/hash 单测。

## 10. 正确性不变量

1. 所有 Hub对象下载到固定 revision，本地文件有 SHA-256；
2. The Cauldron config/card/上游许可经过公司确认；
3. split 只由 image hash 决定、gold 不参与，train/dev image hash 交集为空；
4. RGB/resize/normalize 使用 SigLIP processor 的固定 snapshot；
5. visual shape 严格 `[B,49,576]`；
6. labels 非 `-100` 数等于 answer+EOS token 数；
7. prompt/image/padding labels 全为 `-100`；
8. projector-only 时只有 projector 参数变化；
9. 128 样本 FP32 能显著过拟合，否则不解冻更多层；
10. FP16 200 step finite、resume next batch/param groups/scaler/LR 连续；
11. original/shuffle/blank/zero-visual 使用相同 sample IDs 和 decode 配置。

## 11. 失败树

| 症状 | 最小检查 | 根因候选 | 修复与回归 |
|---|---|---|---|
| backbone load missing keys | revision、config、shape | v0.1/main混用或快照不完整 | 固定本地 snapshot；禁止 `ignore_mismatched_sizes` |
| loss 为 NaN/0 | 每样本有效 label 数 | truncation 吃掉答案、全 -100 | collator fail-fast；20 个手工 mask case |
| loss 降但不会生成 | 比 teacher forcing/greedy prompt | EOS、shift、prompt 模板不一致 | 固定模板与内部 shift测试 |
| projector grad 为 0 | 检查 visual embeds grad | LM forward 被 no_grad、visual 未拼接 | 只 no_grad vision；单 batch反向 |
| 冻结参数变化 | step 前后 hash | optimizer 包含 frozen 参数/weight decay | 明确 param groups；bitwise 对照 |
| image shuffle 不降 | 打印 image/sample mapping | 数据语言偏置、模型忽略图 | 视觉必要子集、zero-visual 对照 |
| V100 FP16 overflow | 同 batch FP32、scale/grad | BF16代码遗留、无 scaler、reduction FP16 | 显式 FP16/scaler/FP32 CE |
| OOM | peak 阶段与 shapes | batch/text/解冻过大 | 按既定顺序减载，重跑 smoke |
| resume 偏离 | next sample IDs | sampler/RNG/param groups 漏存 | 补状态并做 50+50 对照 |

## 12. Teach-back（先闭卷）

1. 从 224、patch16 推到196，再从 factor2 推到49和3072。
2. 写出 prompt=12、answer=3、EOS=1、pad=4 时 labels 的有效位置。
3. 解释 HF 内部 shift 与手工 shift 为什么不能同时做。
4. 说明冻结 LM 时为何仍需通过 LM 对 projector 反传。
5. 分别说出 shuffle image、blank image、zero visual 排除的伪结论。
6. 解释 original EM 高但 shuffle 不降为何不能证明视觉依赖。
7. 写出 FP16 optimizer step 与成功 update 判定。
8. 列出一个可等价 resume 的 VLM checkpoint 状态。

## 13. 官方一手资料

技术链接核验日期：**2026-09-03**。

- [nanoVLM v0.1](https://github.com/huggingface/nanoVLM/releases/tag/v0.1)
- [nanoVLM 官方教程](https://huggingface.co/blog/nanovlm)
- [SigLIP B/16-224](https://huggingface.co/google/siglip-base-patch16-224)
- [SmolLM2-135M](https://huggingface.co/HuggingFaceTB/SmolLM2-135M)
- [The Cauldron](https://huggingface.co/datasets/HuggingFaceM4/the_cauldron)
- [MMStar](https://huggingface.co/datasets/Lin-Chen/MMStar)
- [PyTorch 2.1 AMP](https://pytorch.org/docs/2.1/notes/amp_examples.html)
