# 第 02 周执行手册：纯 VLM 小闭环——nanoVLM 222M 与公开 VQA

> 新增前置周 B；标准投入 7.5 小时，工作日每天 1–2 小时。  
> 机器：单张 V100 32GB；FP16 AMP；不使用 BF16、FlashAttention-2、外部 W&B 或 Hub 上传。  
> 本周定位：在进入机器人动作学习前，先掌握 image → visual tokens → projector → language decoder → answer loss 的完整多模态链路。

每日时间盒：10 分钟闭卷写预期，50–60 分钟完成核心任务，15–20 分钟跑测试并记录；若有第 2 小时，只做当天标出的可选任务，不开启新方向。

## 1. 项目选择与边界

使用 Hugging Face 官方 nanoVLM 的 v0.1 教学版本作为参考：

- 视觉骨干：SigLIP-B/16-224-85M；
- 语言骨干：SmolLM2-135M；
- projector：小型 modality projection；
- 总规模：约 222M；
- 训练数据：公开 The Cauldron 的一个小而明确的 VQA 子集；
- 独立评测：固定 validation split 的 exact match / normalized accuracy；可选 MMStar 小样本 smoke。

为什么固定 v0.1：

- 官方仓库明确说明 v0.1 与旧 222M 模型相容，代码更简单；
- 当前 main 已经历 breaking changes，默认转为 450M；不适合作为一周的基础练习入口；
- 222M 在 V100 32GB 上有充足空间做 batch 扫描、冻结消融和 FP16。

本周不是从随机初始化训练 222M 的视觉与语言基础能力。合理目标是：

1. 加载公开预训练 vision/language backbones；
2. 亲手实现或逐行重建 projector、image-token 拼接和 label mask；
3. 在小型公开 VQA 数据上训练 projector，再解冻少量顶层；
4. 完成可恢复训练和独立 evaluation。

这仍然是完整 VLM 训练，只是把大规模视觉/文本预训练作为公开初始化，避免把一周浪费在不可行的算力目标上。

## 2. 官方资源与获取位置

| 资源 | 用途 | 地址 |
|---|---|---|
| nanoVLM 官方仓库 | 纯 PyTorch、小型 VLM、训练与评测 | https://github.com/huggingface/nanoVLM |
| nanoVLM 官方教程 | 架构、data loader、双 LR、validation/MMStar | https://huggingface.co/blog/nanovlm |
| v0.1 release | 与 222M 旧模型匹配的简单代码 | https://github.com/huggingface/nanoVLM/releases/tag/v0.1 |
| SigLIP backbone | 公开视觉初始化 | https://huggingface.co/google/siglip-base-patch16-224 |
| SmolLM2-135M | 公开语言初始化 | https://huggingface.co/HuggingFaceTB/SmolLM2-135M |
| nanoVLM-222M | 官方公开训练 checkpoint，可作 base/eval 对照 | https://huggingface.co/lusxvr/nanoVLM-222M |
| The Cauldron | 公开多模态训练集合；本周只取一个配置/子集 | https://huggingface.co/datasets/HuggingFaceM4/the_cauldron |
| MMStar | 可选标准化外部评测 | https://huggingface.co/datasets/Lin-Chen/MMStar |

不要直接训练 The Cauldron 全部约百万级样本。本周先选择一个公开、字段稳定、样本规模适中的 VQA 子集。优先顺序：

1. nanoVLM v0.1 默认配置中已验证的单个 dataset config；
2. 若默认配置下载过大，使用其中前 5k–20k train 样本和固定 500–2k validation；
3. 若 The Cauldron 被代理阻止，使用公开 VQAv2 的固定小切片；
4. 仍无法获取时，用 1k 组合图形 + 自动问答合成集完成系统正确性，但周结论必须标记 INCONCLUSIVE，不能冒充公开 VQA 效果。

## 3. 安装与网络预检

创建独立环境，复用现有 torch：

    mkdir -p v100-lab/week02-vlm
    cd v100-lab/week02-vlm
    python -m venv --system-site-packages .venv
    source .venv/bin/activate
    python -m pip install --upgrade pip
    python -m pip install numpy pillow torchvision datasets transformers huggingface_hub pytest pyyaml

禁止照抄官方 README 中重新 pip install torch，也不登录 W&B/Hugging Face 上传。所有输出只落本机。

克隆固定 release：

    git clone --branch v0.1 --depth 1 https://github.com/huggingface/nanoVLM.git third_party/nanoVLM-v0.1
    cd third_party/nanoVLM-v0.1
    git rev-parse HEAD
    cd ../..

模型下载若获批准：

    hf download google/siglip-base-patch16-224 --local-dir artifacts/siglip-b16
    hf download HuggingFaceTB/SmolLM2-135M --local-dir artifacts/smollm2-135m
    hf download lusxvr/nanoVLM-222M --local-dir artifacts/nanovlm-222m

若 hf 命令不可用：

    python -c "from huggingface_hub import snapshot_download; print('hub client import ok')"

然后在公司许可范围内用 snapshot_download 或内部镜像。不要把个人 token 写入 shell history；公开模型通常不需要 token。若仓库 gated，换非 gated 公开模型，不借用个人凭据绕过策略。

## 4. 本周自己的目录

    week02-vlm/
      README_LOCAL.md
      configs/projector_only.yaml
      configs/unfreeze_top.yaml
      src/data.py
      src/vision_adapter.py
      src/projector.py
      src/vlm.py
      src/train.py
      src/eval.py
      src/generate.py
      src/checkpoint.py
      tests/test_image_tokens.py
      tests/test_packing_mask.py
      tests/test_projector_grad.py
      tests/test_eval_normalize.py
      runs/
      decision.md

你可以复用第 01 周 Trainer/checkpoint，但不要直接把 nanoVLM 的 train.py 当黑盒。今天只允许第三方代码承担：

- 预训练 backbone 定义和权重加载；
- 官方 tokenizer/image processor；
- 数据集下载接口。

下面这些必须由你自己实现或从空白逐行重建：

- projector；
- visual token 与 text token 的拼接；
- attention mask；
- labels 与 ignore index；
- 冻结/解冻参数组；
- validation loop；
- checkpoint/resume。

## 5. 固定任务协议

每条样本转换为：

    image: RGB image
    question: string
    answer: string
    sample_id: stable id
    split: train or validation

输入模板固定为：

    Question: {question} Answer:

目标仅对 answer token 计算 loss。question、image placeholder、padding 全部设为 ignore index。

核心 shape 约定：

| tensor | shape | 含义 |
|---|---|---|
| pixel_values | B×3×224×224 | 归一化图像 |
| vision_hidden | B×Nv×Dv | 视觉 patch tokens |
| projected_visual | B×Nv'×Dl | 投到语言 hidden space |
| input_ids | B×T | 问题与答案 tokens |
| combined_embeds | B×(Nv'+T)×Dl | 送入语言 decoder |
| labels | B×(Nv'+T) | 只保留答案区域 |

本周必须打印一条真实 batch 的所有 shape、有效 label 数、padding 数和 image token 数。

## 6. 每日安排

### 周一：读懂架构并建立无训练 forward

目标：让一张图片和一个问题经过完整 forward，loss 的监督位置正确。

先画出四段：

    image → vision encoder → projector → visual tokens
    text → tokenizer → text embeddings
    visual + text → decoder
    answer positions → cross entropy

核心任务：

1. 检查 v0.1 仓库的 model/config、vision、language、projector、VLM 文件；
2. 只读架构，不先运行 train.py；
3. 加载两个 backbone，全部冻结；
4. 自己写两层 projector：LayerNorm/Linear/activation/Linear；
5. 拼接 visual/text embeddings；
6. 用 2 条样本跑 CPU 或单卡 forward；
7. 打印参数总量、trainable 参数量和模块表。

测试：

- projector 输入/输出最后一维正确；
- Nv 和 T 改变时拼接仍正确；
- answer 以外 label 均为 ignore index；
- loss 与手工截取 answer logits 计算结果一致；
- 冻结 backbone 后其 grad 为 None，projector grad 非零。

验收：

- forward loss finite；
- trainable 参数量与冻结策略一致；
- 能口头解释视觉 token 为什么必须投到语言 hidden dimension；
- 能说明拼接顺序如何影响位置编码和 attention。

可选第二小时：比较 prepend visual tokens 与单一 pooled visual token，不训练，只比较 shapes、显存估计和表达能力。

### 周二：公开数据、split、collator 与 evaluator

目标：把“图片+问答”变成没有泄漏、可复现的训练样本。

核心任务：

1. 检查所选 dataset config 的 license、字段、split、样本数；
2. 固定前 5k–20k train 与 500–2k validation，但用 sample_id/hash 而非每次随机切；
3. 抽看至少 30 条样本：图像是否可读、问题/答案是否匹配；
4. 实现 collator，包括 resize/normalize、tokenize、padding、label mask；
5. 统计 question/answer token 长度 p50/p95/max；
6. 实现 answer normalization 与 exact match；
7. 对数字、大小写、标点、冠词规则写单测。

必须做三类泄漏检查：

- 相同 sample_id 不跨 split；
- 图像内容 hash 不跨 split，或明确记录数据集本身的重复；
- 同一 question+answer 文本重复比例被统计。

evaluator 最低要求：

    normalized_exact_match = correct / total
    invalid_generation_rate = invalid / total
    mean_answer_tokens

如果任务有官方 metric，保留官方 metric，同时用自己的 normalized EM 作为透明 reference。

验收：

- 50 条 batch round trip 无异常；
- train/val overlap 为 0，或重复来自官方数据且被明确记录；
- labels 中有效 token 数恰好等于 answer token 区域；
- evaluator 的人工构造 20 个 case 全通过。

可选第二小时：加入 answer type 桶：yes/no、number、short noun、multiword。

### 周三：projector-only 过拟合与训练闭环

目标：先证明多模态监督能进入正确模块。

固定 128–256 条样本，训练配置：

- vision encoder frozen；
- language decoder frozen；
- only projector trainable；
- FP32，batch 4–16；
- dropout 0；
- 不做随机数据增强；
- 500–2k step 或直到明显过拟合。

检查：

1. 同一 batch loss 是否稳定下降；
2. projector grad norm 是否非零；
3. vision/language 参数是否保持 bitwise 或数值不变；
4. shuffled-image 对照：图像打乱后验证准确率是否下降；
5. blank-image 对照：全零图像是否暴露只靠语言先验。

如果 projector-only 无法过拟合：

- 先查 answer mask；
- 检查 visual tokens 是否真的进入 decoder；
- 检查 image normalization；
- 检查 optimizer param groups；
- 不要直接解冻全部模型掩盖 bug。

验收：

- 128–256 样本 train loss 显著下降；
- 至少一部分样本能生成正确答案；
- image shuffle/blank 对照已实现；
- checkpoint 保存 trainable/frozen manifest。

可选第二小时：只解冻 LM 顶层 1 block，看 overfit 速度变化，但不替代主 run。

### 周四：FP16 正式微调、双学习率与 resume

目标：在 V100 上完成稳定的小规模真实训练。

Run A，projector-only：

- 5k–20k train samples；
- FP16 AMP；
- projector LR 1e-3 或 3e-4 起步；
- backbone LR 0；
- 1k–5k step；
- 每 250 step validation；
- 本地保存 best/last。

Run B，轻量解冻：

- projector + LM 最上方 1–2 blocks；
- 可选 vision 最后一层，若显存/时间允许；
- projector LR 高，backbone LR 低 10–100 倍；
- 与 A 相同 train samples、global batch、step。

FP16 注意：

- 显式 autocast float16；
- 使用 GradScaler；
- 不设置 bf16；
- 不安装 FlashAttention-2；
- 数值敏感 loss/reduction 可回 FP32；
- reserved 显存目标低于 30.5GB。

resume 测试：

1. 50 step 保存；
2. 退出进程；
3. 恢复到 100 step；
4. 检查 optimizer param group、scaler、LR、step、best metric；
5. 固定 batch 的下一步 loss 相对偏差目标小于 1%。

本地日志必须禁用外部上传。若原仓库默认 wandb 或 push_to_hub，显式关闭/删除调用。

验收：

- 200 step 无 NaN/Inf；
- warmup 后 skipped step <1%；
- peak reserved <30.5GB；
- resume 连续；
- Run A/B 只有冻结策略和对应 LR 不同。

可选第二小时：batch 1/2/4/8/16 memory scan，拟合 image tokens 与 peak memory 的关系。

### 周五：独立评测、反事实测试与闭卷重建

目标：证明模型是否使用图像，而不只会生成常见答案。

正式评测至少包含：

1. 原始 validation；
2. shuffled-image；
3. blank-image；
4. question-only；
5. base checkpoint；
6. Run A projector-only；
7. Run B light-unfreeze。

表格：

| 模型 | original EM | shuffled EM | blank EM | invalid rate | tokens/s | peak GB |
|---|---:|---:|---:|---:|---:|---:|
| base | | | | | | |
| projector-only | | | | | | |
| light-unfreeze | | | | | | |

关键判断：

- original 好而 shuffled 几乎不降：可能主要依赖语言偏置；
- train 高而 validation 低：数据小、冻结策略或过拟合；
- invalid rate 高：生成模板、EOS、tokenizer 或 decoding 问题；
- image shuffle 明显下降且 original 提升：初步说明视觉输入被利用。

抽 30 个错误，按以下桶分类：

- object/attribute；
- count；
- spatial relation；
- OCR/text-in-image；
- reasoning；
- formatting；
- hallucinated entity；
- question ignored。

最后 30 分钟闭卷画/写：

- vision patch 到 visual tokens；
- projector 的输入输出；
- visual/text packing；
- causal attention 中 image token 能看谁；
- answer-only label mask；
- 冻结/双 LR param groups；
- eval normalization。

完成后，才逐行对照 nanoVLM 的 train.py、data 和 model 文件，写一页差异。

## 7. 定量验收标准

| 类别 | PASS |
|---|---|
| 数据 | 固定公开 train/val；sample/image hash 泄漏检查完成 |
| packing | image/question/padding labels 全为 ignore；只训练 answer |
| gradient | projector 有梯度；冻结 backbone 无梯度且参数未变化 |
| overfit | 128–256 样本能明显过拟合 |
| FP16 | 200 step 无非有限值；warmup 后 skipped step <1% |
| resume | next fixed-batch loss 相对偏差 <1%；param groups/scaler/LR 连续 |
| eval | base/A/B + original/shuffle/blank 至少 9 个格子有结果 |
| 多模态性 | 主模型 original EM 高于 shuffled/blank；若不成立，必须判为未证明使用视觉 |
| 掌控力 | 不看代码能画出完整 VLM forward 和 loss mask |

绝对 EM 提升不设虚假硬门槛，因为你使用的数据子集、训练步数和公开 checkpoint状态会影响结果。评判重点是受控对照是否证明训练链路和视觉依赖。

## 8. 本周故障树

### 下载模型成功但 load 失败

1. 检查代码 release 与 checkpoint 版本；
2. 对照 config key 和 state_dict missing/unexpected keys；
3. 确认 v0.1 对应 222M，不拿 current main 450M config 混用；
4. 记录 commit、model revision、transformers 版本；
5. 不通过 ignore_mismatched_sizes 静默吞掉结构错误。

### CUDA OOM

依次降低：

1. batch；
2. answer/text max length；
3. image token 数或分辨率；
4. 解冻层数；
5. gradient accumulation 保持 global batch。

先记录 allocated/reserved 和 OOM 阶段。不要一上来清 cache 循环掩盖峰值。

### loss 降但生成完全错误

- 检查 teacher-forcing labels 与 generate prompt 是否一致；
- 检查 EOS 与 padding id；
- 检查 decode 是否保留 prompt；
- 分开报告 token loss 和 exact match；
- 先 greedy，再调 temperature。

### 图像打乱后几乎不变

- 检查 visual embeddings 是否进入 decoder；
- 检查 projector grad；
- 检查问题是否可由语言偏置直接猜；
- 换需要图像信息的样本子集；
- 结论写“未证明视觉依赖”，不写“VLM 已学会”。

## 9. 公司机器的边界

- 不登录 W&B；
- 不 push_to_hub；
- 不上传 checkpoint、日志、评测结果；
- 不把机器指标或路径贴入聊天；
- 只使用公开数据与获批依赖；
- 第三方 README 中的云端上传步骤全部跳过。

向 ChatGPT 求助的最小格式：

    目标：例如 answer mask 应只覆盖最后 7 个 token
    自己的预期 shape：B×(Nv+T)
    实际 shape：只写抽象维度，不写公司数据
    最小公开代码：不超过 50 行
    traceback：只留错误类型和相关栈，删除路径/环境细节

## 10. 本周闭卷口试

1. 为什么只训 projector 也可能学不到真正视觉能力？
2. visual token 在 decoder-only VLM 中如何参与 causal attention？
3. 为什么 question token 不应计入 answer loss？
4. projector LR 为什么通常高于 backbone LR？
5. freeze、no_grad 和 model.eval 有什么不同？
6. image shuffle 与 blank image 各能排除什么伪结论？
7. validation exact match 为什么要 normalization？
8. 视觉分辨率怎样影响 patch 数和 attention 成本？
9. 为什么 V100 上不应安装官方 FlashAttention-2？
10. 这个 VLM 与后面的 VLA 相比，还缺哪些状态、动作和时序接口？

## 11. 与 TinyFlowPolicy 的接口

第 03 周先做硬件/FP16 基线，第 04 周进入 TinyFlowPolicy。你将复用：

- image processor 与视觉 encoder 的理解；
- 多模态 condition packing；
- freeze/unfreeze 与不同学习率；
- answer-only mask 对应到 action/future loss mask；
- image shuffle 对应到 condition shuffle；
- base/fine-tuned 受控评测；
- AMP、checkpoint、resume 和错误桶。

到这里，三个“麻雀项目”的逻辑已经清楚：

| 小项目 | 预测对象 | 核心条件 | 可验证输出 |
|---|---|---|---|
| MiniGPT | next token | 历史 tokens | validation NLL / generation |
| nanoVLM | answer tokens | image + question | normalized EM / image counterfactual |
| TinyFlowPolicy | action velocity/action chunk | observation + time/noise | overfit / action error / rollout |
