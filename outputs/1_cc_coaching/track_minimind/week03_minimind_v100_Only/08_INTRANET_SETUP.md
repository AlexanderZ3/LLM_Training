# Week M03 附录 — 内网 V100 的环境配置

> 生成日期：2026-09-08 · 生成方：cc
> 适用：公司内网 8×V100，既有 Conda + PyTorch 2.1.0，`pip install` 与 `git clone` 可用，权重必须外网下好拷进来。
> **本周主线（Day 0–5）不需要装任何新包。** 这份文档是给两个扩展模块、以及你以后在内网配环境时用的。

## 0. 一句话

内网装包的失败几乎都是同一个原因：**某个包为了满足自己的依赖，把 torch 2.1.0 换掉了。** 所以这份文档的核心只有一条——用 constraints 文件把 torch 钉死，让冲突在解析阶段**可见地失败**，而不是静默地把环境改坏。

## 1. 版本相容矩阵（`已确认`，2026-09-08 核验）

以 torch 2.1.0 为固定点。每一行的依据是该版本在 PyPI 上声明的 `requires_dist`，或官方文档明写的门槛。

| 包 | 与 torch 2.1.0 相容的区间 | 卡在哪里 |
| --- | --- | --- |
| **numpy** | **`<2` 必须** | torch ≤2.1 的 `.numpy()` / `.from_numpy()` 在 numpy 2.x 下失效；numpy 1.x 编译的扩展与 2.x 二进制不兼容 |
| **transformers** | **`>=4.46.0, <4.56.0`** | 4.56.0 起 `setup.py` 声明 `torch>=2.2` |
| **tokenizers** | `>=0.21, <0.22` | 由 transformers 4.53–4.55 钉死 |
| **huggingface-hub** | `>=0.34.0, <1.0` | transformers 4.55 的要求；不要升 1.x |
| **safetensors** | `>=0.4.3` | 同上 |
| **accelerate** | `>=1.4.0`（1.x 全系列仍声明 `torch>=2.0.0`） | 需 Python ≥3.10 |
| **peft** | 宽松，`>=0.14.0` 即可 | 0.14.0 是 bnb 0.45 做 adapter merging 的下限 |
| **trl** | **`>=0.14.0, <=0.22.2`** | 0.14.0 是 `GRPOTrainer` 的起点；0.23.0 起要 transformers ≥4.56.1 |
| **datasets** | `>=3.0.0` | 不依赖 torch |
| **bitsandbytes** | **`<=0.45.5`** | 0.46.0 起 `torch>=2.2`，0.48.0 起 `torch>=2.4` |
| **deepspeed** | 声明层全系列都写 `torch>=2.0.0` 无上界；**实跑建议 `0.12.x`–`0.14.x`** | 声明宽松不等于能跑。它会 JIT 编译 CUDA 算子，唯一判据是现场 `ds_report` |

### 明确不能用的（不要浪费时间试）

| 包 | 原因 |
| --- | --- |
| `torchao` | 最低 torch 2.5。没有任何版本能配 2.1 |
| `vllm` | **每个版本都硬钉一个精确的 torch 版本**——连最老的 0.2.7 都要求 `torch==2.1.2` 而不是 2.1.0。且官方最低 compute capability 是 7.5，V100 是 7.0 |
| `verl` | 要求 CUDA ≥12.8 且 vllm ≥0.18 |
| `OpenRLHF` | 绑定 vllm 0.27.1 |

**这四个的共同点值得记住**：现代 RL 训练框架都把推理引擎当作硬依赖，而推理引擎是整个生态里对 GPU 架构和 torch 版本最挑剔的一环。在老卡的内网上做 RL，路线必须是"训练框架 + transformers 原生 generate"，而不是"训练框架 + vLLM"。`trl` 的 `GRPOConfig.use_vllm` 默认就是 `False`，这条路是官方支持的，不是将就。

## 2. 把 torch 钉死：constraints 文件

先在内网建一个文件，**以后每一条 pip 命令都带上它**：

```text
# constraints-v100.txt
torch==2.1.0
numpy<2
```

用法：

```bash
pip install -c constraints-v100.txt <包名>
```

constraints 文件只管"如果要装这个包，必须是这个版本"，不触发安装。任何试图升级 torch 的包会在**依赖解析阶段直接失败**，你会看到一条清楚的冲突信息，而不是装完之后发现 torch 变了。

**不要用 `--no-deps` 来保 torch。** 它确实能阻止 pip 动 torch，但它同时关掉了所有依赖检查，装完的环境可能缺包，而且要到 import 或运行时才炸。`--no-deps` 只在一种情况下合理：你明确知道依赖都已满足，只想把一个 wheel 放进去。那种情况下它是 `模板` 级命令，用之前先想清楚。

## 3. 离线 wheelhouse：在外网下好，整包搬进来

内网 pip 源不一定有你要的版本，有依赖的包尤其容易解析失败。稳妥做法是在外网机器上把整棵依赖树下成 wheel，打包搬进内网。

### 外网机器

```bash
python -m pip download \
  --only-binary=:all: \
  --platform manylinux2014_x86_64 \
  --python-version 310 \
  --implementation cp \
  --abi cp310 \
  --dest ./wheelhouse_v100 \
  -c constraints-v100.txt \
  "transformers>=4.55.0,<4.56.0" "tokenizers>=0.21,<0.22" "huggingface_hub>=0.34.0,<1.0" \
  "safetensors>=0.4.3" "accelerate>=1.4.0" "peft>=0.14.0" \
  "trl>=0.14.0,<=0.22.2" "datasets>=3.0.0" "bitsandbytes==0.45.5" "numpy<2"
```

`--platform` / `--python-version` / `--implementation` / `--abi` 这四个参数，pip 官方文档明确要求：

- 用了其中任何一个，就**必须**同时加 `--only-binary=:all:` 或 `--no-deps`；
- 它们各自默认取当前系统的值，而不是最宽松的值，所以**要给就四个一起给**；
- 通用 wheel（无平台/abi 限制的）即使你写了过度约束也照样会被匹配到。

**四个参数的值要对准 V100 机器上的解释器，不是外网机器的。** 先在 V100 上跑一次：

```bash
python -c "import sys, sysconfig; print(sys.version_info[:2], sysconfig.get_platform())"
```

拿到的 Python 版本填 `--python-version`（`3.10` 写成 `310`），平台填 `--platform`。填错的典型症状是：下载成功、拷进去装不上，报 `not a supported wheel on this platform`。

### 搬进内网之后

```bash
python -m pip install --no-index --find-links=./wheelhouse_v100 \
  -c constraints-v100.txt \
  "transformers>=4.55.0,<4.56.0" "trl>=0.14.0,<=0.22.2" "bitsandbytes==0.45.5"
```

`--no-index` 让 pip 完全不去连网络源，只从 `--find-links` 指的目录找。这样即使内网能连到某个 pip 镜像，也不会意外从那里拉到一个不同的版本。

### 下载不到 wheel 的包

`--only-binary=:all:` 遇到没有目标平台 wheel 的包会直接失败。这不是 bug，是它在告诉你"这个包必须在目标机上编译"。**deepspeed 是典型**：它要 nvcc 和 JIT 编译。真遇到时的分支是：

1. 先看公司 Conda 环境里是不是已经装过一个能用的版本（`ds_report`）；
2. 装不了就跳过——本周主线和两个扩展模块**都不需要 deepspeed**。

## 4. 权重：只能外网下好拷进来

```bash
# 外网机器。命令名取决于 huggingface_hub 版本：
#   hub >= 0.34：hf download ...
#   hub 0.3x  ：huggingface-cli download ...
hf download <repo_id> --revision <40 位 commit hash> --local-dir ./model_tmp
rm -rf ./model_tmp/.cache/huggingface      # 只是元数据，不用搬
tar -cf model.tar -C ./model_tmp .
```

**`--revision` 一定要写全长的 40 位 commit hash。** 内网发现版本不对时你没法重新下，只能重走一遍外网流程。

搬进内网后：

```bash
mkdir -p $MM_WEIGHTS_ROOT/<name> && tar -xf model.tar -C $MM_WEIGHTS_ROOT/<name>
HF_HUB_OFFLINE=1 python lab/scripts/check_weights_layout.py --name <name> --tokenizer
```

### 离线环境变量：只需要一个

```bash
export HF_HUB_OFFLINE=1
```

`HF_HUB_OFFLINE=1` 会让 huggingface_hub 完全不发 HTTP 请求，缓存里没有就直接报错——这正是内网想要的行为：**立刻失败，而不是挂在那里等超时**。

- `TRANSFORMERS_OFFLINE`：transformers 4.55 的 `utils/hub.py` 里已经不再读它，离线判定统一走 `HF_HUB_OFFLINE`。写了没用。
- `HF_DATASETS_OFFLINE`：datasets 里仍被读取，但只是 `HF_HUB_OFFLINE` 的别名。
- `HF_HOME`：token 与缓存的根目录，默认 `~/.cache/huggingface`。内网上如果 home 目录配额小，把它指到大盘。
- `HUGGINGFACE_HUB_CACHE` / `HUGGING_FACE_HUB_TOKEN`：已弃用，改用 `HF_HUB_CACHE` / `HF_TOKEN`。

## 5. 在拷进内网之前必须先验的一件事

`internlm2-1_8b-reward` 这类**带远程代码**的模型（加载时要 `trust_remote_code=True`），它的 `modeling_*.py` 是针对某个 transformers 版本写的。你的 V100 上 transformers 必须 ≤4.55.x，而这批远程代码能否在 4.55.x 上加载，目前 `未知`。

**在外网机器上先用 4.55.x 试加载一次再拷。** 内网发现不兼容时你没有退路。

```bash
# 外网机器，一个临时环境里
pip install "transformers>=4.55.0,<4.56.0"
python -c "
from transformers import AutoTokenizer, AutoConfig
d = './model_tmp'
print(AutoConfig.from_pretrained(d, trust_remote_code=True))
print(len(AutoTokenizer.from_pretrained(d, trust_remote_code=True)))
"
```

## 6. 一份可以直接抄的开工清单

每次登上 V100 先跑这几行。前四条是路径合同（见 `datasets/README.md`），后面是离线开关：

```bash
export MM_DATA_ROOT=/your/path/to/minimind/datasets
export MM_WEIGHTS_ROOT=/your/path/to/minimind/weights
export MM_RUNS_ROOT=/your/scratch/minimind/runs
export MINIMIND_ROOT=/your/path/to/MiniMind

export HF_HUB_OFFLINE=1
export HF_HOME=/your/scratch/hf            # 如果 home 盘小
export NCCL_DEBUG=WARN                     # 出问题时改成 INFO

python lab/scripts/check_data_layout.py    # 数据到位了吗
```

## 7. 内网踩坑速查

| 症状 | 多半是什么 | 先查哪里 |
| --- | --- | --- |
| `pip install` 之后 `import torch` 版本变了 | 某个包把 torch 升了 | 装的时候没带 `-c constraints-v100.txt`。重装 torch 2.1.0 并从头再来 |
| `not a supported wheel on this platform` | wheelhouse 的四元组对准了外网机器而不是 V100 | 在 V100 上跑 `sysconfig.get_platform()`，用那个值重下 |
| 加载模型时挂住不动，最后超时 | 代码在偷偷连 huggingface.co | 没设 `HF_HUB_OFFLINE=1`。设了之后会立刻报错并告诉你缺哪个文件 |
| `ImportError: cannot import name ...` from transformers | 装到了 ≥4.56，或者代码用了 4.56+ 的 API | `pip show transformers`；本周代码只能用 ≤4.55 的 API |
| numpy 相关的 `_ARRAY_API not found` 或二进制不兼容 | numpy 装成 2.x | `pip install "numpy<2" -c constraints-v100.txt` |
| 加载远程代码模型报找不到模块 | 那几个 `.py` 没拷全 | `python lab/scripts/check_weights_layout.py --dir <目录>`，它会列出 `auto_map` 引用了但不存在的文件 |
| `RuntimeError: Current CUDA Device does not support bfloat16` | 某个脚本的 dtype 默认值还是 bfloat16 | V100 只能 fp16。查命令行的 `--dtype` |
| 数据字节数对不上 | 传输时用了文本模式，换行被改了 | 用 tar / rsync / scp 二进制重传那几个文件 |

## 8. 来源

全部为 2026-09-08 联网核验的官方一手资料：

- pip：`pip download` 的平台参数约束、`--no-index` / `--find-links` / constraints 文件语义 —— https://pip.pypa.io/en/stable/cli/pip_download/ 、https://pip.pypa.io/en/stable/user_guide/ 、https://pip.pypa.io/en/stable/cli/pip_install/
- huggingface_hub 环境变量与 `hf download` 语义 —— https://huggingface.co/docs/huggingface_hub/package_reference/environment_variables 、https://huggingface.co/docs/huggingface_hub/en/guides/download
- transformers 4.55.0 的 `setup.py` 与 `utils/hub.py`（`TRANSFORMERS_OFFLINE` 不再被读取） —— https://raw.githubusercontent.com/huggingface/transformers/v4.55.0/setup.py 、https://raw.githubusercontent.com/huggingface/transformers/v4.55.0/src/transformers/utils/hub.py
- numpy 2.x 与 torch ≤2.1 的不兼容 —— https://github.com/pytorch/pytorch/issues/107302
- bitsandbytes 的 compute capability 门槛与 torch 依赖 —— https://huggingface.co/docs/bitsandbytes/main/en/installation 、https://pypi.org/pypi/bitsandbytes/0.45.5/json
- trl 的 GRPO 起点与依赖区间 —— https://github.com/huggingface/trl/releases/tag/v0.14.0 、https://huggingface.co/docs/trl/en/grpo_trainer
- vLLM 的 CC 门槛与 torch 钉死 —— https://docs.vllm.ai/en/latest/getting_started/installation/gpu.html 、https://pypi.org/pypi/vllm/0.2.7/json
- verl / OpenRLHF 的要求 —— https://verl.readthedocs.io/en/latest/start/install.html 、https://raw.githubusercontent.com/OpenRLHF/OpenRLHF/main/README.md
- torchao 的 torch 版本表 —— https://github.com/pytorch/ao/issues/2919
