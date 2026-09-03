# 第 01 周执行手册：纯 LLM 从零训练——TinyStories MiniGPT

> 新增前置周 A；标准投入 7.5 小时，工作日每天 1–2 小时。  
> 机器：单张 V100 32GB 即可；全程先 FP32 正确性，再 FP16 AMP。  
> 本周定位：麻雀虽小、五脏俱全。你要亲手拥有 tokenizer、数据、模型、loss、优化器、训练、评测、生成、checkpoint 和 resume。

每日时间盒：10 分钟闭卷写预期，50–60 分钟完成核心任务，15–20 分钟跑测试并记录；若有第 2 小时，只做当天标出的可选任务，不开启新方向。

## 1. 本周唯一目标

从公开 TinyStories 训练集开始，独立训练一个 10M–30M 参数 decoder-only Transformer，并完成：

1. 原始文本到 token ids；
2. causal self-attention 与 next-token loss；
3. AdamW、warmup + cosine schedule、gradient clipping；
4. 单卡 FP32 基准和 FP16 AMP；
5. checkpoint/resume 与固定 batch 等价性；
6. validation loss、perplexity、生成样例与错误分析。

本周不追求“生成效果像大模型”。通过标准是你能在白板上解释每个 tensor，并能从空目录重建训练入口。

## 2. 为什么选择 TinyStories

- 数据公开，官方 train/validation 文本可直接下载；
- 语言模式比 Shakespeare 更接近自然叙事，十几 M 参数也能看到学习现象；
- validation split 明确，适合建立可重复的 perplexity；
- 与 Stanford CS336 Assignment 1 的 tokenizer、Transformer、optimizer、训练闭环高度一致。

主要资源：

| 资源 | 用途 | 获取地址 |
|---|---|---|
| Stanford CS336 | 本周主课程框架；A1 要求实现 tokenizer、模型、优化器和最小 LM | https://cs336.stanford.edu/spring2025/ |
| CS336 A1 | 测试与作业说明；先自己实现，之后再接 adapters | https://github.com/stanford-cs336/assignment1-basics |
| Karpathy minBPE | 第 2 天完成后用于对照 BPE 结构 | https://github.com/karpathy/minbpe |
| nanoGPT | 第 5 天才阅读 train.py/model.py，对照你的实现 | https://github.com/karpathy/nanoGPT |
| LLMs from Scratch | 第二解释来源，适合逐章补概念 | https://github.com/rasbt/LLMs-from-scratch |
| TinyStories | 公开 train/validation 数据 | https://huggingface.co/datasets/roneneldan/TinyStories |

## 3. 网络受限环境的安装策略

不要先升级系统 CUDA、驱动或现有 torch。先复用机器上已经验证过的 PyTorch：

    mkdir -p v100-lab/week01-llm
    cd v100-lab/week01-llm
    python -m venv --system-site-packages .venv
    source .venv/bin/activate
    python -m pip install --upgrade pip
    python -m pip install pytest numpy pyyaml tqdm
    python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))"

如果 pip 只能通过公司镜像，沿用获批准的 index 配置；不要在命令或文档里写入账号、token。若 torch 无法 import，停止安装并请平台管理员给出与驱动匹配的 wheel/容器，不要自行安装最新版 CUDA 13。

先探测来源：

    git ls-remote https://github.com/stanford-cs336/assignment1-basics.git HEAD
    git ls-remote https://github.com/karpathy/minbpe.git HEAD
    python -m pip index versions pytest

克隆只作参考：

    mkdir -p third_party
    git clone --depth 1 https://github.com/stanford-cs336/assignment1-basics.git third_party/cs336-a1
    git clone --depth 1 https://github.com/karpathy/minbpe.git third_party/minbpe
    git clone --depth 1 https://github.com/karpathy/nanoGPT.git third_party/nanoGPT

数据下载优先使用 CS336 A1 README 给出的 TinyStories 文件。若 wget 被代理拦截，可用 huggingface_hub：

    python -m pip install huggingface_hub
    hf download roneneldan/TinyStories TinyStoriesV2-GPT4-train.txt --repo-type dataset --local-dir data/raw
    hf download roneneldan/TinyStories TinyStoriesV2-GPT4-valid.txt --repo-type dataset --local-dir data/raw

若 Hugging Face 被禁止，只能走公司批准的镜像或管理员导入。不要绕过代理。本周可先用你现场生成的 1–5MB 合成故事完成代码正确性，但最终周验收需要公开 validation 数据。

## 4. 你要建立的目录

    week01-llm/
      README_LOCAL.md
      requirements-freeze.txt
      configs/tiny_20m.yaml
      data/raw/
      data/processed/
      src/tokenizer.py
      src/data.py
      src/model.py
      src/optim.py
      src/train.py
      src/eval.py
      src/generate.py
      src/checkpoint.py
      tests/test_tokenizer.py
      tests/test_attention.py
      tests/test_loss.py
      tests/test_checkpoint.py
      runs/
      decision.md

核心文件必须由你手敲。可以阅读官方文档和向 ChatGPT 问概念，但不要让聊天模型一次性生成整个实现。推荐纪律：

- 每次只问一个局部问题；
- 先写自己的预期 shape 和伪代码；
- 最多粘贴 30–50 行纯公开/合成代码；
- 不粘贴公司路径、日志、机器指标、内部数据；
- 修好后关掉参考，从空白处重写关键函数。

## 5. 固定模型配置

先用字符 tokenizer 跑通，再用 BPE。建议最终配置：

| 项 | 值 |
|---|---:|
| vocabulary | 约 4k–8k BPE |
| context length | 256 |
| layers | 6 |
| hidden size | 384 |
| heads | 6 |
| MLP ratio | 4 |
| dropout | 0.0–0.1 |
| 参数量 | 约 15M–30M，以脚本实算为准 |
| optimizer | AdamW |
| peak LR | 3e-4 起步 |
| warmup | 总 step 的 2%–5% |
| precision | FP32 reference；FP16 AMP main |

必须写一个参数量函数，禁止凭印象写“约 20M”：

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

## 6. 每日安排

### 周一：数据、next-token 任务与最小字符模型

目标：今天只证明数据和监督信号正确。

10 分钟，闭卷写出：

- 给定 token 序列 t0...tN，输入和标签如何错一位；
- batch、sequence、vocab 三个维度分别是什么；
- causal mask 为什么必须遮住未来位置。

核心任务：

1. 读取 TinyStories train/valid，打印字节数、行数、空样本数；
2. 先用前 1–5MB，构造字符词表；
3. 实现 encode/decode，保证往返完全一致；
4. 实现随机连续窗口 dataset；
5. 写 bigram baseline，并在 128 个窗口上过拟合；
6. 建立 validation loss，不允许用 train loss 代替。

必须完成的测试：

    pytest -q tests/test_tokenizer.py tests/test_loss.py

预期：

- decode(encode(text)) 与原文本完全相同；
- 输入 shape 为 B×T，logits 为 B×T×V；
- labels 等于输入序列右移一位；
- 128 窗口 overfit loss 明显下降；
- validation 代码使用 model.eval() 和 no_grad。

当天记录：

| 指标 | 必填 |
|---|---|
| train/valid 字节数 | 是 |
| vocab size | 是 |
| 128 窗口初始/最终 loss | 是 |
| 随机抽 3 个 encode/decode round trip | PASS/FAIL |

可选第二小时：手写 cross-entropy 的 log-softmax 版本，与 torch.nn.functional.cross_entropy 对比，绝对误差小于 1e-5。

### 周二：亲手实现 BPE tokenizer

目标：理解 tokenizer 是一个训练得到的数据组件，而不是 AutoTokenizer 黑盒。

核心任务：

1. 用 bytes 作为初始符号；
2. 统计相邻 pair；
3. 每轮合并最高频 pair；
4. 训练 1k 词表作 correctness 版，再训练 4k–8k 词表；
5. 保存 vocab、merges、special-token 约定；
6. encode 不得依赖训练 corpus 的全量驻留；
7. 与字符 tokenizer 比较 token/character ratio。

单元测试：

- UTF-8 中文、英文、数字、换行都能 round trip；
- 空字符串、单字节、重复 pattern 不崩溃；
- special token 不被普通 merge 吞掉；
- save 后 load 的 token ids 完全一致；
- 相同 corpus 与 seed 生成同一 tokenizer hash。

完成自己的版本之后才阅读：

    sed -n '1,260p' third_party/minbpe/minbpe/base.py
    sed -n '1,320p' third_party/minbpe/minbpe/basic.py

写出三点差异：

1. 你的复杂度瓶颈；
2. minBPE 的编码/解码不变量；
3. 生产 tokenizer 还缺 normalization、pre-tokenization 或 regex 哪些部分。

考核指标：

- 10 个 round trip 全通过；
- tokenizer 文件可重载；
- valid OOV 为 0，因为 byte fallback；
- 能口头解释 vocab 扩大对序列长度、embedding 参数和 softmax 计算的影响。

可选第二小时：运行 CS336 A1 tokenizer tests，但不要复制 staff/社区解答。

### 周三：Transformer 从张量开始

目标：从零写出可训练 decoder-only Transformer。

实现顺序必须是：

1. token embedding；
2. positional embedding 或 RoPE 二选一；本周建议 learned positional embedding；
3. pre-norm causal self-attention；
4. MLP；
5. residual block；
6. final norm；
7. tied LM head；
8. next-token loss。

每个模块先在 CPU 上跑 shape test。attention 需要额外完成：

- mask 后未来位置概率为 0；
- 每一行 attention probability 和约等于 1；
- B=2、T=7、非整齐长度仍正确；
- Q/K/V reshape 与 transpose 后的维度能手写出来；
- eval 模式固定输入得到固定输出。

推荐 test oracle：做一个极小维度的 for-loop attention，与向量化实现对比，最大绝对误差小于 1e-5。

当天命令：

    pytest -q tests/test_attention.py tests/test_loss.py
    python -m src.train --config configs/tiny_20m.yaml --device cpu --max_steps 3 --dry_run
    python -m src.train --config configs/tiny_20m.yaml --device cuda --precision fp32 --max_steps 20

验收：

- 所有参数都有梯度，除非明确冻结；
- logits/loss 无 NaN/Inf；
- 参数量在 config 预期范围；
- 20 step loss 方向正确；
- 能解释 1 个 block 的参数量和 attention 的 O(T²) 来源。

可选第二小时：实现 RoPE，并用 rotation norm 不变性作单测；不要同时替换训练主配置。

### 周四：完整训练器、FP16、checkpoint/resume

目标：把“能反向传播”升级为可恢复的训练系统。

训练器必须包含：

- train/eval mode 切换；
- gradient accumulation；
- AdamW；
- warmup + cosine LR；
- autocast(float16)；
- GradScaler；
- unscale 后 gradient clipping；
- non-finite loss/grad 检查；
- 定期 validation；
- 保存 best 和 last；
- terminal JSON summary。

FP16 顺序：

    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        logits, loss = model(x, y)
        loss = loss / grad_acc_steps
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(optimizer)
    scaler.update()

checkpoint 至少保存：

- model；
- optimizer；
- scheduler 或其可重建状态；
- GradScaler；
- Python、NumPy、torch CPU、torch CUDA RNG；
- global step、tokens_seen；
- config；
- tokenizer hash 与数据 manifest。

测试流程：

1. 固定 seed，连续跑 40 step；
2. 同 seed 跑 20 step，保存，退出；
3. 从 step 20 恢复，再跑到 40；
4. 在 step 21 的固定 batch 比较 loss；
5. deterministic 条件满足时，目标相对误差小于 1%；完全确定性路径应更接近。

验收：

- FP16 200 step 无非有限参数；
- warmup 后 skipped step 比例低于 1%；
- GradScaler state 确实变化并被恢复；
- resume 后 LR、step、tokens_seen 连续；
- loss clip 发生在 unscale 之后。

可选第二小时：从零实现 AdamW 更新，用小张量和 torch.optim.AdamW 跑 5 step 对照。

### 周五：正式短训、评测、生成与闭卷重建

目标：形成完整项目结论，不以“代码跑通”收尾。

正式实验只跑两组：

| Run | tokenizer | precision | 目的 |
|---|---|---|---|
| A | char 或小 BPE | FP32，短参考 | 数值 reference |
| B | 4k–8k BPE | FP16 AMP | 主训练 |

建议主训练 2k–10k step，按当日可用时长设置 max_steps；可以下班后继续，但必须遵守公司调度规则并有资源上限。

评测：

1. 固定完整 validation 子集，报告平均 NLL；
2. perplexity = exp(NLL)，NLL 太大时先报告 NLL；
3. 报告 tokens/s、step p50、peak allocated/reserved；
4. 使用固定 10 个 prompt，每个 greedy 生成一次；
5. 使用 3 个随机 seed 的 sampling 只作质性观察；
6. 建错误桶：重复、语法破碎、角色漂移、提前终止、记忆式复述。

最后 30 分钟，关闭代码，手写以下伪代码：

- 一个 batch 如何生成；
- attention forward；
- loss mask；
- 一个 AMP optimizer step；
- checkpoint save/load。

之后才阅读 nanoGPT：

    sed -n '1,360p' third_party/nanoGPT/model.py
    sed -n '1,420p' third_party/nanoGPT/train.py

写一页对照：你遗漏了什么、nanoGPT 为性能做了什么、哪些设计你不同意以及原因。

## 7. 本周定量验收

| 类别 | PASS |
|---|---|
| 数据 | train/valid 独立；hash 记录；无空窗口 |
| tokenizer | round trip 100%；save/load 一致；byte fallback 无 OOV |
| 模型 | attention oracle 最大误差 <1e-5；所有预期参数有梯度 |
| overfit | 128–256 个固定窗口能明显过拟合；若不能不得长训 |
| FP16 | 200 step 无 NaN/Inf；warmup 后 skipped step <1% |
| resume | fixed next-batch loss 相对偏差 <1%；LR/step/scaler/RNG 连续 |
| 评测 | validation NLL/perplexity、吞吐、显存、生成错误桶齐全 |
| 掌控力 | 不看仓库能写出训练主循环与 tensor shapes |

本周状态只能取：

- PASS：全部核心门槛通过；
- FAIL-SYSTEM：环境、AMP、checkpoint 或数据程序有 bug；
- FAIL-MODEL：系统正确但短训效果不达预期；
- INCONCLUSIVE：公开 validation 未拿到或正式 run 未完成。

## 8. 常见故障判断

### loss 接近 log(vocab) 且不降

按顺序检查：

1. labels 是否真的右移一位；
2. causal mask 方向是否反；
3. optimizer 是否拿到全部参数；
4. loss 是否把 padding 算进去；
5. learning rate 是否为 0；
6. overfit 128 窗口是否成功。

### FP16 频繁 overflow

检查：

- 是否错误地 model.half() 后又做敏感 reduction；
- loss 是否除以 accumulation；
- clip 是否在 unscale 后；
- logits 极值、grad norm 首次异常模块；
- 用 FP32 fixed batch 是否正常。

不要只把初始 scale 调小后宣布解决。

### 数据下载失败

- 先保存完整错误类型：DNS、TLS、403、timeout；
- git 可用而 HF 不可用时，使用公司镜像/审批导入；
- 不安装未知代理工具；
- 使用合成数据继续单测，但周结论标记 INCONCLUSIVE。

## 9. 本周应留下的内部产物

    runs/week01/
      config.yaml
      env.txt
      tokenizer.json
      tokenizer.sha256
      data_manifest.json
      last.pt
      best.pt
      metrics.jsonl
      eval.json
      generations.txt
      decision.md

全部留在公司机器。向聊天窗口求助时，只粘贴你自己写的最小公开代码、tensor shape 和经过脱敏的异常类型，不粘贴真实路径、性能数据、拓扑日志或内部信息。

## 10. 本周闭卷口试

能连续回答以下问题才进入第 02 周：

1. 为什么 next-token loss 的 labels 必须 shift？
2. BPE 词表扩大怎样同时影响序列长度、embedding 和 LM head？
3. causal mask 在 softmax 前还是后施加？为什么？
4. pre-norm 与 post-norm 的差异是什么？
5. weight tying 节省了什么，又约束了什么？
6. AdamW 与 L2 regularization 为什么不完全等价？
7. gradient accumulation 如何改变 optimizer step 与 tokens seen？
8. autocast 与 GradScaler 分别解决什么问题？
9. checkpoint 为什么必须保存 RNG 和 scaler？
10. perplexity 为什么只能在相同 tokenizer/数据处理下公平比较？

## 11. 与后续路线的接口

第 02 周 VLM 会直接复用本周：

- language decoder；
- tokenizer 与 loss mask；
- Trainer、AMP、checkpoint、eval；
- 参数量、吞吐、显存记录；
- overfit-first 与 fixed-batch-resume 方法。

不要把本周代码丢掉后直接运行 nanoVLM。第 02 周的目标是理解“图像 token 如何进入你已经掌握的语言模型”。
