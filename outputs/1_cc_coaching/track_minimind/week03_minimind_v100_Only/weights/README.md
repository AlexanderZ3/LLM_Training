# `weights/` — 权重落地位置（`MM_WEIGHTS_ROOT`）

> 生成日期：2026-09-08 · 生成方：cc · **本目录不存放权重本体。**

V100 在内网，`huggingface_hub` 下不动东西。**所有权重都要在有外网的机器上下好，再拷进来。**

## 本周需要的

| 子目录                     | 来源                                                                      | 什么时候需要   | 大小       |
| -------------------------- | ------------------------------------------------------------------------- | -------------- | ---------- |
| `minimind_tokenizer/`    | MiniMind 仓库自带的`model/` 目录（`git clone` 就有，不用单独下）      | Day 1 起，每天 | < 5 MB     |
| `internlm2_1_8b_reward/` | 见`../datasets/README.md`，放在 `MM_DATA_ROOT` 下即可，不必重复放这里 | 可选，扩展模块 | 3.17 GiB   |
| `qat_base/`              | 你自己选的、要做 8-bit 量化的那个大模型                                   | QAT 扩展模块   | 取决于模型 |

## `qat_base/` 该放什么

QAT 扩展模块的实验设计成**与具体模型无关**：所有量化逻辑作用在 `torch.nn.Linear` 上，换模型只换一个路径。所以这里放什么都行，只要是 HuggingFace 标准布局（`config.json` + `*.safetensors` + tokenizer 文件）。

从小到大的建议顺序，先在小的上把链路跑通，再换大的：

1. 先用 MiniMind 自己训出来的 26M/64M checkpoint —— 不用下载，Day 1–3 就产出了，链路验证最快。
2. 再换一个 1–2 B 的公开模型，验证"per-channel 比 per-tensor 好多少"这类结论在真实权重分布上还成不成立。
3. 最后才是你真正要量化的那个超大模型。

**选基座时有一个 V100 特有的坑必须先查。** 现在公开的大模型（Qwen、Llama 系）绝大多数是 **bf16 预训练**的。bf16 和 fp16 的指数范围差很多：bf16 能表示的最大值和 fp32 一样是 3.4e38，fp16 只到 65504。torch 的 AMP 文档对此有明确警告——bf16 预训练的模型在 fp16 下会**梯度上溢**（overflow），而不是通常担心的下溢，`GradScaler` 会把 scale 一路往下压，压到 1 以下就等于 loss scaling 失效了。

V100 没有 bf16。所以拿一个 bf16 预训练模型到 V100 上做 QAT 或 RL 之前，先跑一次 fp16 前向的溢出统计，再决定要不要继续。这一步比装环境重要。

在外网机器上下载并拷贝：

```bash
# 外网机器。命令名取决于 huggingface_hub 版本：
#   hub >= 0.34：hf download <repo_id> --local-dir ./qat_base_tmp
#   hub 0.3x  ：huggingface-cli download <repo_id> --local-dir ./qat_base_tmp
# 两者等价；旧名在新版里仍可用但会提示改名。
hf download <repo_id> --local-dir ./qat_base_tmp
tar -cf qat_base.tar -C ./qat_base_tmp .
# 拷进内网后
mkdir -p $MM_WEIGHTS_ROOT/qat_base && tar -xf qat_base.tar -C $MM_WEIGHTS_ROOT/qat_base
```

`--local-dir` 会在目标目录下建一个 `.cache/huggingface/` 放元数据。拷贝之前可以删掉它，不影响加载。

拷完在 V100 上确认离线模式能加载得动：

```bash
HF_HUB_OFFLINE=1 python lab/scripts/check_weights_layout.py --dir $MM_WEIGHTS_ROOT/qat_base --tokenizer
```

**只需要 `HF_HUB_OFFLINE=1`。** `TRANSFORMERS_OFFLINE` 在 transformers 4.55 的 `utils/hub.py` 里已经不再被读取（离线判定统一走 `huggingface_hub.constants.HF_HUB_OFFLINE`），写了也不起作用，反而会让人以为多了一层保险。

## 目录里的东西不出公司

如果 `qat_base/` 里放的是公司自己的模型，那么它、以及它派生出的任何量化产物、误差统计、逐层敏感度表，都**留在 V100 上**。带出来的只有任务卡证据字段里的抽象值。
