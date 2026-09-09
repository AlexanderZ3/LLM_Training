# `datasets/` — 本仓库与 V100 服务器之间的路径合同

> 生成日期：2026-09-08 · 生成方：cc  
> **本目录不存放数据本体。** 它只存放目录名、文件清单和校验脚本，让这台 Windows 机器上写的代码里的每一条路径，和 V100 服务器上的实际路径**逐字对应**。

## 0. 为什么有这个目录

数据你已经在外网下好了，会自己拷到 V100。cc 这边没有数据、也不该有数据。但如果 cc 写代码时随口编一个路径，你拷过去之后每个脚本都要改一遍，改漏一个就是一次浪费的排队。

所以规则是：**代码里不出现任何字面路径。** 所有路径从 `lab/src/mm_v100/paths.py` 一个地方出，那个地方只认三个环境变量。你在 V100 上设一次环境变量，全部脚本同时对齐。

## 1. 三个根目录

| 环境变量 | 本仓库默认值 | V100 上应该设成什么 | 放什么 |
| --- | --- | --- | --- |
| `MM_DATA_ROOT` | 本目录 | 你放 `minimind_dataset/` 的父目录 | 只读数据；11 个 jsonl + 可选奖励模型 |
| `MM_WEIGHTS_ROOT` | `../weights` | 你放 tokenizer 与基座权重的目录 | 只读权重 |
| `MM_RUNS_ROOT` | `../runs` | 一个**大容量可写**分区 | checkpoint、日志、profile、eval 输出 |

在 V100 上（每次开工先跑，或写进你的 `~/.bashrc`）：

```bash
export MM_DATA_ROOT=/your/path/to/minimind/datasets
export MM_WEIGHTS_ROOT=/your/path/to/minimind/weights
export MM_RUNS_ROOT=/your/scratch/minimind/runs
export MINIMIND_ROOT=/your/path/to/MiniMind      # git clone 下来的仓库
```

这四条是**唯一**需要你按机器修改的东西。改完之后 `lab/` 里所有脚本都不用动。

## 2. 数据要拷成什么样子

`MM_DATA_ROOT` 下必须长这样。名字必须**逐字一致**，包括大小写和下划线：

```text
$MM_DATA_ROOT/
├── minimind_dataset/                  # 必需
│   ├── pretrain_t2t_mini.jsonl              1,241,043,656 B
│   ├── sft_t2t_mini.jsonl                   1,739,201,170 B
│   ├── dpo.jsonl                               53,653,322 B
│   ├── rlaif.jsonl                             23,754,740 B
│   ├── pretrain_t2t.jsonl                   8,275,074,893 B
│   ├── sft_t2t.jsonl                       14,096,018,369 B
│   ├── agent_rl.jsonl                          82,036,930 B
│   ├── agent_rl_math.jsonl                     18,372,683 B
│   ├── lora_medical.jsonl                      34,002,385 B
│   ├── lora_exam.jsonl                         24,650,716 B
│   └── lora_identity.jsonl                         22,789 B
└── internlm2_1_8b_reward/             # 可选，只在把规则奖励换成模型打分时才需要
    ├── model-00001-of-00002.safetensors     1,981,392,544 B
    ├── model-00002-of-00002.safetensors     1,417,790,344 B
    ├── model.safetensors.index.json                13,682 B
    ├── config.json                                    813 B
    ├── configuration_internlm2.py                   9,042 B
    ├── modeling_internlm2.py                       91,364 B
    ├── tokenization_internlm2.py                    8,806 B
    ├── tokenization_internlm2_fast.py               7,805 B
    ├── tokenizer.model                          1,477,754 B
    ├── tokenizer_config.json                        2,950 B
    └── special_tokens_map.json                        809 B
```

数据集 11 个文件合计 **25,587,831,653 B（23.83 GiB）**；奖励模型 11 个文件合计 **3,400,795,913 B（3.17 GiB）**。合计 28,988,627,566 B（27.00 GiB）。

字节数来自 2026-09-05 那次下载的实际校验日志（`已确认`）。机器可读版本在 `DATA_MANIFEST.json`，校验脚本读的是那个文件，不是这张表。

### 两个容易踩的点

1. **奖励模型的三个 `.py` 必须一起拷。** `modeling_internlm2.py`、`configuration_internlm2.py`、`tokenization_internlm2*.py` 是远程代码，加载时要 `trust_remote_code=True`。只拷 safetensors 会在加载时报找不到模块，而且报错信息不会告诉你缺的是这几个文件。
2. **别用会改字节的传输方式。** 文本模式的 FTP、某些同步工具的换行转换，会把 jsonl 的 `\n` 变成 `\r\n`，字节数对不上、`json.loads` 也可能出错。用二进制传输或打 tar 包。

## 3. 拷完之后先跑这一条

```bash
python lab/scripts/check_data_layout.py
```

它逐个文件比对存在性和**精确字节数**，输出 `OK / MISSING / SIZE-MISMATCH`，最后给一行总结和非零退出码。这是 Day 0 的第一个证据字段，也是后面每一天的隐含前提。

想连 sha256 一起验（慢，23 GB 大概几分钟）：

```bash
python lab/scripts/check_data_layout.py --sha256
```

清单里存的是下载日志给出的 **sha256 前 16 位**，脚本比对的也是前 16 位。要做全量比对，把外网那台机器上的 `SHA256SUMS.txt` 一起拷进 `$MM_DATA_ROOT`，脚本检测到就自动改用全量比对。

## 4. 数据不出公司这一条在这里怎么落地

- 这个目录、`DATA_MANIFEST.json`、校验脚本可以自由在两边流动，因为它们只包含**文件名和字节数**，不包含任何样本内容。
- `$MM_RUNS_ROOT` 下的东西（checkpoint、日志、profile、eval 输出）**留在 V100 上**。带出来的只有任务卡证据字段里那些抽象值：比值、是否收敛、第几步、通过还是没通过。
- 校验脚本**不打印任何样本内容**，只打印文件名和字节数，所以它的输出可以直接贴给 cc。

## 5. 许可

数据集卡片同时标了 `apache-2.0` 和 `cc-by-nc-2.0`，后者是**非商用**。个人学习和内部实验没问题；任何对外用途先自己去读数据集卡片。奖励模型许可标注是 `other`，用前读模型卡。cc 不替你做许可判断。
