# Week M03 Lab Guide — MiniMind 纯 V100 全链路训练系统

> 生成日期：2026-09-08 · 生成方：cc · 状态：**本机 CPU 门实测通过；所有 GPU 结论待公司 8×V100。**
> 本文件是 `lab/` 的总说明。安装、smoke 命令、目录速查在 `lab/README.md`；这里讲的是
> **代码从哪来、路径怎么对齐、哪个文件产出哪个证据、三本账的仪表在哪、坏了先查什么**。
>
> **标签约定**：`本机实测` = 在这台无 GPU 的 Windows 机器上真跑过并抄下了输出；
> `估算` = 公式/推导出来的数，没有在目标机上量过；`未验证` = 一行都没在 GPU 上跑过。
> 全文凡是涉及 V100 的数字，一律是 `估算` 或 `未验证`。

---

## 0. 代码来源与裁剪说明

这一节存在的理由：`lab/src/mm_v100/` 里的十六个模块**不是凭空写出来的**，其中十一个是从
前两周已经通过 CPU 单测的实现里复制并裁剪来的。不写清楚，这些代码看起来像新代码，
review 时会重复审已经审过的东西，而真正新写的那几个模块反而被略过。

### 0.1 逐文件出处

| 本周文件 | 来源 | 裁掉了什么 | 加了什么 / 为什么 |
| --- | --- | --- | --- |
| `lab/src/mm_v100/common.py` | week02 `lab/src/mm_dist/common.py` | MiniMind 仓库导入（`resolve_minimind_root` / `import_minimind` / `build_model`）、MoE 字段、`RealJsonlDataset` 与 `build_dataset` | `PurePythonScaler` + `ScalerAdapter`（CPU 上也能复现"跳步"这个不变量）、`grad_global_norm` / `param_global_norm`、`rng_state` / `set_rng_state`、`resolve_config`、`free_port` / `set_worker_env`、`TinyDataset.label_ignore_prefix`（模拟 SFT 的 prompt 屏蔽）；进程组超时默认从 1800 s 收到 600 s |
| `lab/src/mm_v100/probe.py` | week02 `lab/scripts/probe_company.py` | `nvidia-smi` 拓扑解析（本周不做专家并行，拓扑对结论无影响，少一个泄密面） | int8 能力探测、`GradScaler` 官方默认值读取、`torch.distributed.checkpoint` 的 API 名探测、GPU 独占性判断、磁盘余量**只报档位不报精确值** |
| `lab/src/mm_v100/compat_audit.py` | week02 `lab/scripts/audit_minimind_compat.py` | MoE / 专家并行相关条目 | 输出改成「四个计数 + 明细」的结构化字典，`make_evidence.py` 能直接取那四个数 |
| `lab/src/mm_v100/collectives.py` | week02 `lab/src/mm_dist/train_fsdp.py` 里的 `CollectiveCounter` | 与 FSDP 训练循环的耦合（独立成模块） | 每次调用的**字节量**与 dtype；新增手算期望表 `ddp_expected` / `fsdp_expected` / `compare_to_expected`，供 Day 3 闭卷对答案 |
| `lab/src/mm_v100/bounded_train.py` | week01 `lab/src/mm_probe/bounded_train.py` + week02 `lab/src/mm_dist/train_ddp.py` + `train_fsdp.py`（三合一） | MiniMind 依赖、MoE、activation checkpointing 扫描、profiler | 三种放置（`single` / `ddp` / `fsdp`）**共用一个训练循环**：只有数据切分、loss 归约、日志字段完全一致，两条曲线才可以逐步相减 |
| `lab/src/mm_v100/ckpt_reshard.py` | week02 `lab/src/mm_dist/ckpt_reshard.py` | 必须有 CUDA 的 FSDP 上下文（隔离进 `fsdp_sharded_save` / `fsdp_sharded_load` 两个函数，明确标注只能在 V100 上跑） | `FlatShardCheckpoint`：一个不依赖 FSDP、CPU 上能真跑的分片格式，让"N 卡写 → M 卡读"这条门在本机就是真测试 |
| `lab/src/mm_v100/faults.py` | week01 `lab/src/mm_probe/mask_fault.py` + week02 `lab/src/mm_dist/faults.py` | `kill_rank` / NCCL 超时环境这类**会报错或会 hang 的显性故障** | 重组成三档**静默**故障（`label_shift` / `scaler_stuck` / `ddp_no_sync`）+ 三个不变量 + 一张判别表 |
| `lab/src/mm_v100/data_contract.py` | week01 `lab/src/mm_probe/inspect_dataset.py` + `mask_fault.py` | 对 MiniMind tokenizer 的硬依赖、终端彩色表格 | 可插拔 tokenizer + 内置确定性 `SimpleTokenizer`（用 `crc32` 而不是 `hash()`，后者带进程级随机盐）、`to_minimind_pair()` 显式转换、toy fixture |
| `lab/src/mm_v100/numeric_ref.py` | week01 `lab/src/mm_probe/bounded_train.py` 的 dtype / scaler 部分 | MiniMind 依赖与 checkpoint 逻辑 | 只留「同一初始权重、同一批数据、只改精度」这一件事，四个证据字段 |
| `lab/src/mm_v100/eval_generate.py` | week01 `lab/src/mm_probe/eval_generate.py` | MiniMind 模型 / 权重加载 | 角色泄漏检测（`role_leak`）、`echo_ratio`、聚合指标；三个判据全部标成**定性**，不产生分数 |
| `lab/src/mm_v100/rl_numeric.py` | week01 `lab/src/mm_probe/dpo_check.py` + `grpo_stats.py` + `rule_reward.py` | MiniMind tokenizer / dataset 依赖、stdout 日志解析 | 只留 CPU 上能闭式验证的四条断言（D1 / D2 / G1 / G2），并补上 `ratio@step0` |
| `lab/tests/conftest.py` | week02 `lab/tests/conftest.py` | MiniMind 定位与 `SKIP_NO_MINIMIND` | `runs_root` fixture 把 `MM_RUNS_ROOT` 指到 `tmp_path`，测试永远不往周包的 `runs/` 里写文件 |

**本周新写、没有前身的**：`paths.py`（路径合同）、`console.py`（UTF-8 兜底）、`cli.py`（入口共用件）、
`model.py`（自带模型）、`ledger.py`（字节账手算 vs 实测）、`lab/scripts/run_day.py` 与六个 shell 包装、
以及两个扩展模块 `lab/src/mm_quant/`、`lab/src/mm_rl/` 全部十二个模块。

`lab/scripts/check_data_layout.py` 与 `lab/scripts/check_weights_layout.py` 是本周包早先一次交付的
产物，不来自前两周。

### 0.2 本周为什么自带模型，以及它不是什么

`lab/src/mm_v100/model.py` 的 `TinyCausalLM` 是一个**与 MiniMind 形状对齐的自带实现**，
不是 MiniMind 本体。这一句要读两遍，因为它同时是本周最大的结构性好处和最大的边界。

**它不是什么**：
- 它**不含 MiniMind 的权重**。MiniMind 训出来的 `.pth` 加载不进来，反过来也一样。
- 它**不含 MiniMind 的 tokenizer**。本 lab 用 `data_contract.SimpleTokenizer`（字符级、确定性、纯标准库）。
- 因此它**不能替 MiniMind 出任何"模型质量"结论**。它出的是结构性结论：字节账、通信账、
  数值链、故障判别——这些只依赖形状与算子，不依赖权重。

**它是什么，以及为什么值得**：形状与 MiniMind 逐项对齐（RMSNorm + RoPE + GQA + SwiGLU +
tied embedding + **QK-norm**，`intermediate_size = ceil(hidden × π / 64) × 64`），
所以 `v100_768` 配置下参数量精确等于 **63,912,192**，与 `01_FOUNDATIONS.md` 第 1 节
手算到的那个数一致（`本机实测`：`byte_ledger.py --config v100_768 --hand-only` 打印 `P = 63,912,192`）。

week02 的 lab 直接 import `MINIMIND_ROOT` 里的 `MiniMindForCausalLM`。代价是：cc 这台机器上
没有那个仓库，于是 **DDP 等价、字节账、故障注入、分片 checkpoint 这四条最需要提前验一遍的门
全部被 skip**，开环风险一路带到公司机器上才第一次暴露。自带模型把这四条门变成了**本机真测试**：

| 门 | week02（依赖外部仓库） | 本周（自带模型） |
| --- | --- | --- |
| 固定全局批 DDP 等价 | 无仓库则 skip | `本机实测` 2 进程 gloo，逐步 `\|Δloss\| = 0.000e+00` |
| 三档静默故障判别 | 无仓库则 skip | `本机实测` 四档各被唯一不变量抓住 |
| 字节账手算 vs 实测 | 无仓库则 skip | `本机实测` param / grad / optim 三项**逐字节相等** |
| 分片 checkpoint 重切 | 无仓库则 skip | `本机实测` 2 卡写 / 1 卡读，loss 连续性 `\|Δ\| = 0.000e+00` |

`MINIMIND_ROOT` 在本 lab 里只有一个用途：Day 0 的静态兼容审计（`compat_audit.py` 用正则扫它的
源码，不 import、不执行、不写回）。没设这个变量，Day 0 少一步，其余四天全部照跑。

---

## 1. 环境与安装

### 1.1 两条路径，一句话对照

| | 公司 8×V100（唯一执行环境） | 本机 Windows（写代码 + CPU 单测） |
| --- | --- | --- |
| 解释器 | 已有的那个（不新建 conda 环境） | conda `rfm`：`C:/Users/zzz_7893/miniconda3/envs/rfm/python.exe` |
| Python | `未知`，Day 0 探针读出来 | 3.11.16（`本机实测`） |
| torch | `2.1.0`，**不升级** | 2.14.0+cpu（`本机实测`） |
| numpy | 目标机上必须 `< 2`（torch ≤ 2.1 的 `.numpy()` 在 numpy 2.x 下失效） | 2.4.6（`本机实测`，新版 torch 不受这条约束） |
| GPU | 8×V100 32 GB，sm70 | **无**。`nvidia-smi` 不存在 |
| 精度 | 一律 `--dtype float16` + GradScaler；`bfloat16` 在参数解析阶段被拒 | fp16 是**软件模拟**，只验方法不验数值 |
| 能验什么 | 全部 | CPU 单测、静态检查、`spawn` 多进程 gloo |

### 1.2 主线零新依赖（这是设计目标，不是巧合）

`lab/requirements.txt` 里**没有一行是需要在公司机器上新装的**。实际的 `install_requires` 只有三行：

```text
torch>=2.1.0
numpy>=1.21
pytest>=7.0        # 只有本机跑单测要，V100 上不需要
```

其余全是注释，说明两条路径的 API 差异。**两个扩展模块也一样**：`mm_quant/` 与 `mm_rl/`
全部用纯 PyTorch 写，`bitsandbytes` 只作为可选对照出现在 `mm_quant/opt8bit.try_import_bnb()` 里，
装不上是预期结果，不是故障。

**明确不使用**：`bitsandbytes` / `deepspeed` / `trl` / `peft` / `vllm` / `flash-attn` /
`megablocks` / `DeepEP` / `apex` / `wandb`。理由分别在 `06_QUANT_LOWBIT.md` 第 1 节、
`07_RL_LOWPRECISION.md` 第 1 节、`08_INTRANET_SETUP.md` 第 1 节。

> **在公司机器上不要跑 `pip install -r lab/requirements.txt`。**
> `torch>=2.1.0` 这一行会让 pip 认为可以把 torch 升到最新版，而"某个包顺手把 torch 换掉"
> 正是内网装包失败的头号原因（`08_INTRANET_SETUP.md` 第 2 节）。这个文件是一张**声明表**，
> 用来读，不用来装。要确认环境满足条件就跑这一条（`可直接执行`）：
>
> ```bash
> python -c "import torch, numpy; print(torch.__version__, numpy.__version__)"
> ```
>
> 期望看到 `2.1.0` 和一个 `1.x` 的 numpy。不是这两个数就先停下来读 `08_INTRANET_SETUP.md`。

### 1.3 本机怎么跑

所有本机命令都用 `rfm` 那个解释器的绝对路径，不要用裸 `python`：

```bash
cd outputs/1_cc_coaching/track_minimind/week03_minimind_v100_Only
export PY="C:/Users/zzz_7893/miniconda3/envs/rfm/python.exe"
export PYTHONIOENCODING=utf-8            # Windows 控制台默认 cp1252，中文 print 会直接抛异常
export MM_RUNS_ROOT=/tmp/mm_runs         # 本机用一个临时目录，别写进仓库

"$PY" -m compileall -q lab/src lab/scripts
"$PY" -m pytest lab/tests -q -rs
```

`PYTHONIOENCODING` 不是可选的美化项。本机控制台代码页是 cp1252，脚本打第一行中文日志就会
`UnicodeEncodeError` 崩掉，而崩的位置看起来和业务逻辑毫无关系。`lab/src/mm_v100/console.py`
的 `use_utf8()` 在每个入口做了兜底，环境变量是第二道保险。

### 1.4 本机没有 torchrun，多进程走 spawn

本机没有 `torchrun`，所以"2 进程等价""故障 C""分片 checkpoint 重切"这三件事用
`torch.multiprocessing.spawn` 跑——各脚本的 `--launcher spawn` / `--nproc N` 就是干这件事的。
到了 V100 上换成 `--launcher torchrun`，同一套逻辑。

Windows 上 `spawn` 会重新 import 目标模块，所以每个 worker 函数都必须是**模块顶层的普通函数**。
这一条也是 `lab/scripts/run_day.py` 目前那个已知缺陷的根源，见第 5.4 节和第 7.4 节。

---

## 2. 路径合同

### 2.1 四个环境变量

代码里**一条字面路径都没有**。所有位置从 `lab/src/mm_v100/paths.py` 一个地方出，那个地方只认四个环境变量：

| 变量 | 含义 | 缺省回退 | 缺了会怎样 |
| --- | --- | --- | --- |
| `MM_DATA_ROOT` | 只读数据根目录 | 周包内 `datasets/` | 回退到周包内的空目录，`dataset_dir()` 报错并给出两条可执行的下一步 |
| `MM_WEIGHTS_ROOT` | 只读权重根目录 | 周包内 `weights/` | 同上 |
| `MM_RUNS_ROOT` | **可写**产物根目录 | 周包内 `runs/` | 产物会落进仓库目录，这在 V100 上等于把公司产物放进可同步的位置，**必须显式设成 scratch 分区** |
| `MINIMIND_ROOT` | MiniMind 仓库克隆位置 | **没有回退** | `minimind_root()` 直接抛 `PathContractError`，只影响 Day 0 的兼容审计一步 |

在 V100 上（`模板`，把右边换成你的实际路径）：

```bash
export MM_DATA_ROOT=/your/path/to/minimind/datasets
export MM_WEIGHTS_ROOT=/your/path/to/minimind/weights
export MM_RUNS_ROOT=/your/scratch/minimind/runs
export MINIMIND_ROOT=/your/path/to/MiniMind
```

### 2.2 `paths.py` 的三条设计约束

1. **不 import torch。** 它必须在任何环境里都能 import，包括只装了标准库的环境——
   否则 Day 0 的路径自检会因为 torch 装不上而无法运行，那是本末倒置。
2. **异常消息必须写清楚下一步做什么**，而不是只报一个 `FileNotFoundError`。
   代码在这台机器上写、在另一台机器上第一次运行，报错信息就是唯一的排错入口。
   例：字节数对不上时它会打出「期望 X B，实际 Y B（差 ±Z B）」外加"多半是传输被截断，
   或者用了文本模式传输把换行改了"。
3. **`run_dir(tag)` 固定建 `ckpt/` `logs/` `profiles/` 三个子目录**，脚本不自己拼路径。
   tag 用 `day0_probe`、`day1_pretrain` 这种稳定名字，重跑同一天就覆盖同一个目录，
   这样证据字段里的路径是稳定的。

这条约束有测试兜底：`lab/tests/test_scripts_cli.py::test_no_literal_absolute_paths` 会扫描
`lab/src` 与 `lab/scripts`，出现字面绝对路径就失败。

### 2.3 拷完数据先跑什么

顺序固定，四步（`可直接执行`）：

```bash
python -c "import sys; sys.path.insert(0,'lab/src'); from mm_v100 import paths; print(paths.describe())"
python lab/scripts/check_data_layout.py
python lab/scripts/check_weights_layout.py --name minimind_tokenizer
python lab/scripts/probe_env.py
```

第一条打印四个根目录的解析结果与存在性，**不打印任何数据内容**，所以它的输出可以直接贴出来。
第二条逐文件比对存在性和**精确字节数**（清单在 `datasets/DATA_MANIFEST.json`），
输出 `OK` / `MISSING` / `SIZE-MISMATCH` / `SHA-MISMATCH` 四种状态之一。
第三条堵内网上最常见的两种权重失败：拷漏 `trust_remote_code` 要用的那几个 `.py`、以及
代码在缓存未命中时偷偷连外网（用 `HF_HUB_OFFLINE=1` 让它**立刻失败**而不是等超时）。

### 2.4 什么留在公司里

`$MM_RUNS_ROOT` 下的 checkpoint、日志、profiler 产物全部留在公司内。唯一设计成可以带出来的是
`evidence.json`，由 `lab/scripts/make_evidence.py` 生成：它只做**白名单聚合**（比值、布尔、
计数、第几步），不复制任何原始数字序列（`losses` 数组、逐步 grad_norm、逐步显存），
键名里含 `path` / `dir` / `root` / `host` / `node` / `uuid` / `serial` / `topology` / `name` / `file`
的一律丢弃。显存只带出「相对第一步的倍数」，不带绝对值。

**脚本只做机械过滤，最后一道关是你自己**——生成后打开看一眼再带走。

---

## 3. 文件依赖图

### 3.1 分层结构

```text
lab/src/mm_v100/
  paths.py  console.py  cli.py          <- 基础层：路径、编码、入口共用件（不 import torch 的是 paths.py）
        |
  common.py   model.py                  <- 设施层：dtype 守卫 / Scaler / 分布式 / 数据切分 / 日志；自带模型
        |
  ├── probe.py          compat_audit.py         <- Day 0
  ├── data_contract.py  numeric_ref.py          <- Day 1
  ├── faults.py         collectives.py          <- Day 2 / 3 的仪表
  ├── bounded_train.py                          <- Day 1-5 的统一训练循环（single / ddp / fsdp）
  ├── ledger.py         ckpt_reshard.py         <- Day 3
  ├── eval_generate.py                          <- Day 4
  └── rl_numeric.py                             <- Day 5
```

`bounded_train.py` 是唯一被多个入口共用的训练循环：`wiring_smoke.py`、`equiv_check.py`、
`run_faults.py`、`resume_check.py`、`train_bounded.py` 都最终落到它。分成几个脚本各写一遍，
早晚会在某一处产生差异，而那个差异会伪装成"FSDP 的数值问题"。

### 3.2 入口脚本 → 依赖模块 → 产物

`$RUN` 表示 `$MM_RUNS_ROOT/<run-tag>`，各脚本的默认 `--run-tag` 写在第三列。

| Day | 入口脚本 | 依赖模块 | 产物（写在 `$RUN/` 下） | run-tag |
| --- | --- | --- | --- | --- |
| 0 | `lab/scripts/probe_env.py` | `probe.py` `paths.py` `cli.py` | `probe.json` | `day0_probe` |
| 0 | `lab/scripts/audit_compat.py` | `compat_audit.py` `paths.py` | `compat_audit.md` `compat_audit.json` | `day0_probe` |
| 0 | `lab/scripts/wiring_smoke.py` | 子进程调 `train_bounded.py` → `bounded_train.py` | `wiring_smoke.json` `smoke_ws<N>_rank<r>.jsonl` | `day0_probe` |
| 0 | `lab/scripts/check_data_layout.py` | `paths.py` + `datasets/DATA_MANIFEST.json` | 只打屏 / `--json` | — |
| 0 | `lab/scripts/check_weights_layout.py` | `paths.py` | 只打屏 | — |
| 1 | `lab/scripts/check_data_contract.py` | `data_contract.py` | `data_contract.json`（+ `--toy` 时的 fixture jsonl） | `day1_pretrain` |
| 1 | `lab/scripts/run_numeric_ref.py` | `numeric_ref.py` `model.py` `common.py` | `numeric_ref.json` | `day1_pretrain` |
| 2 | `lab/scripts/equiv_check.py` | spawn → `bounded_train.py` | `equiv_A_rank*.jsonl` `equiv_B_rank*.jsonl` `equiv_table.json` | `day2_sft` |
| 2 | `lab/scripts/run_faults.py` | spawn → `bounded_train.py` + `faults.py` | `fault_<mode>_rank*.jsonl` `fault_table.json` | `day2_sft` |
| 3 | `lab/scripts/byte_ledger.py --hand-only` | `ledger.py` `model.param_breakdown` `collectives.py` | `byte_ledger_hand.json` | `day3_dist` |
| 3 | `lab/scripts/byte_ledger.py` | 同上 + `ledger.measure` | `byte_ledger.md` `byte_ledger.json` | `day3_dist` |
| 3 | `lab/scripts/reshard_check.py` | spawn → `ckpt_reshard.py` | `ckpt/reshard/`（`meta.json` + `shard_*.pt`）`reshard_check.json` | `day3_dist` |
| 3 | `lab/scripts/resume_check.py` | `bounded_train.save_checkpoint` / `load_checkpoint` | `resume_continuous_rank0.jsonl` `resume_part2_rank0.jsonl` `resume_check.json` | `day3_dist` |
| 4 | `lab/scripts/eval_compare.py --tag before` | `eval_generate.py` `model.py` | `eval_before.jsonl` `eval_before_summary.json` | `day4_profile` |
| 4 | `lab/scripts/train_bounded.py --stage sft --save-final 1` | `bounded_train.py` | `ckpt/final.pt` `metrics_day4_rank0.jsonl` | `day4_profile` |
| 4 | `lab/scripts/eval_compare.py --tag after --checkpoint $RUN/ckpt/final.pt` | `eval_generate.py` | `eval_after.jsonl` `eval_after_summary.json` | `day4_profile` |
| 5 | `lab/scripts/rl_numeric_check.py` | `rl_numeric.py` `data_contract.py` `model.py` | `rl_numeric.json`（+ 无 `--dpo-file` 时的 `toy_dpo.jsonl`） | `day5_rl` |
| 每天最后 | `lab/scripts/make_evidence.py` | 读该目录下全部 `*.json` 与 `metrics_*_rank0.jsonl` | `evidence.json` | 由 `--run-tag` 指定 |
| 扩展 | `lab/scripts/probe_int8_caps.py` | 独立 | `int8_caps.json` | `ext_qat` |
| 扩展 | `lab/scripts/run_quant_suite.py` | `mm_quant/*` | `quant_suite.jsonl` `quant_suite_summary.json` | 配置里的 `tag` |
| 扩展 | `lab/scripts/run_rl_suite.py` | `mm_rl/*` | `rl_suite.jsonl` `rl_suite_summary.json` | 配置里的 `tag` |

`lab/scripts/run_day.py` 是这张表的编排器：`--day N` 按门的顺序依次调用当天的入口，
默认在第一个失败的步骤停下。六个 `lab/scripts/run_day*.sh` 每个只做一件事——用对解释器
调一次 `run_day.py`，不设 `-e`、不写循环、不对原生命令用输出重定向（`CLAUDE.md` 3.2 的教训）。
`run_day.py` 目前在需要 `spawn` 的步骤上有一个已知缺陷，见第 5.4 节。

### 3.3 Day N 的产物是 Day N+1 的什么

这条链上大部分是**判定依赖**（前一天的结论决定后一天能不能开跑），只有两处是**文件依赖**。

```text
Day 0  probe.json
         ├─ devices.bf16_supported = False       ──> Day 1-5 一律 --dtype float16（判定依赖）
         ├─ sdpa.functional.flash                ──> Day 1+ 的 --sdpa-backend 取值（判定依赖）
         ├─ devices.device_count                 ──> Day 2/3 的 --nproc 上限（判定依赖）
         └─ wiring_smoke.json 每档 world_size 跑满 2 步
                                                 ──> 才允许进 Day 2 的多进程实验（门）
Day 1  numeric_ref.json
         ├─ compare.first_diverge_step / max_dloss ──> Day 2 起 fp16 曲线可不可比（判定依赖）
         └─ data_contract.json I1 = 0            ──> Day 2 故障 A 的对照基线（判定依赖）
Day 2  equiv_table.json  max|delta| < tol        ──> Day 3 的多卡实测才有意义（门）
       fault_table.json  四档各被唯一不变量抓住   ──> Gate 的三行判别表（交付物）
Day 3  byte_ledger_hand.json（先闭卷手算）
         └─ 与 byte_ledger.json 的实测列并排      ──> Gate A0 的标准答案
       ckpt/reshard/                             ──> **文件依赖**：save 端写，load 端读（同一脚本内）
Day 4  ckpt/final.pt                             ──> **文件依赖**：eval_compare --tag after 读它
         └─ 也是 Day 5 可选的 policy 起点
Day 5  rl_numeric.json  D1/D2/G2 三条断言        ──> 管线接对了没有（不是效果好不好）
```

**唯一不可跳的顺序**：Day 0 的 wiring smoke。从 1 卡到 8 卡之间会坏的东西按出现顺序是
import、配置解析、模型构建、单卡前反向、进程组建立、NCCL 握手、集合通信、显存。
一次跑 8 卡失败时这八件事分不清是哪一件；按 1 → 2 → 8 走，失败点直接被夹在两档之间。

### 3.4 配置文件

| 配置 | 用途 | `hidden` / `layers` / `inter` | 备注 |
| --- | --- | --- | --- |
| `lab/configs/tiny_cpu.json` | CPU 单测与 dry-run（秒级） | 64 / 2 / 128 | 刻意的小玩具，**不镜像** MiniMind 形状 |
| `lab/configs/smoke_v100.json` | V100 上 2 步跑完的 wiring smoke | 256 / 4 / 832 | 只验接线，不看数值 |
| `lab/configs/v100_768.json` | **Gate 主配置**，字节账手算题用这一组 | 768 / 8 / 2432 | P = 63,912,192 |
| `lab/configs/v100_768_accum.json` | 同一全局批、`accum=4` 的显存对照组 | 768 / 8 / 2432 | activation 应降到约 1/4 |
| `lab/configs/quant_cpu_smoke.json` / `quant_v100.json` | Q1–Q6 | — | 合成权重；真结论要配 `--state-dict` |
| `lab/configs/rl_cpu_smoke.json` / `rl_v100.json` | R1–R6 | 64 / 2 / 128 与 512 / 8 / 1664 | 用 `mm_rl.toy.TinyCausalLM` |

---

## 4. 三本账的仪表在代码里各在哪

三本账是本周的主线：**字节账**决定它能不能跑，**通信账**决定它跑多快，**误差账**决定
学习信号还在不在。三个仪表各自独立，可以单独调用、单独出证据。

### 4.1 字节账 → `lab/src/mm_v100/ledger.py`

**测什么**：`param` / `grad` / `optim` / `activation` 四项，每卡各占多少字节。分手算与实测两列。

**手算：`hand_ledger(model_cfg, world_size, mode, batch_per_rank, seq_len, ...)`**
只吃配置，不建模型——所以它可以在没有 GPU 的地方先算好，这正是 Day 3 "先闭卷再对答案"的前提。

| 项 | 公式 | 为什么容易算错 |
| --- | --- | --- |
| `param_bytes` | `P × 4 / shard_param[mode]` | AMP 训练时参数留在 **fp32**（`autocast` 只影响算子的输入输出，不改 `nn.Parameter` 的 dtype），所以是 4 字节/元素，不是 2 |
| `grad_bytes` | `P × 4 / shard_grad[mode]` | 与参数同 dtype 同形状 |
| `optim_bytes` | `P × 4 × 2 / shard_grad[mode] + n_tensors × 4` | AdamW 除了 `exp_avg` / `exp_avg_sq`，还给**每个参数张量**存一个 fp32 的 `step` 标量。少了这一项，手算列和实测列会永远差那么一点，看起来像测量误差 |
| `activation_bytes` | `L·B·T·(8·H + 4·I)·2 + B·T·V·(2+4) + B·T·H·2` | **这一项是 `估算`**。autograd 到底存了哪些中间张量取决于算子实现（尤其 SDPA 走哪个后端）。可验证的是它的标度律：对 `B` 严格线性，对 `T` 至少线性 |

三种放置下的分母（`W` = world_size）：

```text
DDP / FSDP NO_SHARD    param = 4P      grad = 4P      optim = 8P
FSDP SHARD_GRAD_OP     param = 4P      grad = 4P/W    optim = 8P/W
FSDP FULL_SHARD        param = 4P/W    grad = 4P/W    optim = 8P/W
                       + 瞬时的 all_gather buffer = 单个 FSDP 单元的参数量 × 2（MixedPrecision 下）
```

FULL_SHARD 省的是常驻，付的是通信与那个瞬时 buffer。如果 auto-wrap 退化成"只有一个根单元"，
那个 buffer 就等于整份模型，FSDP 白做——所以 `n_units` 是手算表里的**显式输入**，不是隐含假设。

**实测：`measure(model, input_ids, labels, optimizer, ...)`** 顺序有讲究：
activation 只在 backward 之前存在，所以必须在 `saved_tensors_hooks` 里做 forward；
optim 只在 `step()` 之后存在（AdamW 的两个矩是惰性创建的），所以要最后测。
`SavedActivationMeter` 用 `torch.autograd.graph.saved_tensors_hooks` 统计，
按 `(data_ptr, nbytes)` 去重并排除参数自身的 storage——这是 CPU 上唯一靠得住的 activation 度量
（`torch.cuda.max_memory_allocated` 在没有 GPU 的机器上恒为 0）。

**怎么调**（`可直接执行`）：

```bash
# 只手算，闭卷之后用它对答案；不需要 GPU，也不建模型
python lab/scripts/byte_ledger.py --config v100_768 --hand-only \
    --world-size 8 --mode full_shard --batch-per-rank 8 --seq-len 512 --show-collectives

# 手算 + 实测（本机用 tiny_cpu；V100 上换 v100_768 --device cuda:0）
python lab/scripts/byte_ledger.py --config tiny_cpu --device cpu
```

**输出字段**：`params_total` `params_breakdown` `n_param_tensors` `param_bytes` `grad_bytes`
`optim_bytes` `activation_bytes` `activation_parts{layers,logits,embed}` `all_gather_buffer_bytes`
`total_bytes` `formulas`；实测侧多 `activation_tensors` `peak_mem_mb`；
对照表 `compare()` 给每项 `hand` / `measured` / `delta` / `ratio` / `ok`，
其中 **param / grad / optim 要求逐字节相等，activation 只看比值落在 0.25–4.0 之间**。
退出码 0 = 前三项相等；activation 不影响退出码。

**Gate 主配置下的手算结果**（`本机实测`的算术，`估算`的 activation；`W=8`, `B=8`, `T=512`）：

| 项 | DDP（每卡） | FSDP FULL_SHARD（每卡） |
| --- | ---: | ---: |
| `param_bytes` | 255,648,768 | 31,956,096 |
| `grad_bytes` | 255,648,768 | 31,956,096 |
| `optim_bytes` | 511,297,896 | 63,912,552 |
| `activation_bytes`（`估算`） | 1,203,765,248 | 1,203,765,248 |
| `all_gather_buffer` | 0 | 14,202,709 |
| **合计** | 2,226,360,680 B ≈ 2.07 GiB | 1,331,589,992 B ≈ 1.24 GiB |

读这张表要读出来的一句话：**分片把 1.02 GiB 的静态项压到 0.12 GiB，但 activation 那 1.12 GiB
一动不动。** 64M 模型上 activation 是主项，所以 FSDP 在这个规模上省不了多少——
真正能动 activation 的是 batch、seq 和 activation checkpointing，不是分片策略。

### 4.2 误差账 → `lab/src/mm_v100/numeric_ref.py`

**测什么**：fp16 相对 fp32 **从哪一步开始偏、偏多少、偏的方向**。

**方法上的三个约束，缺一个结论就不成立**：

1. 两条曲线的初始权重必须**逐位相同**——用 `copy.deepcopy`，不是同 seed 重建。
   同 seed 重建在 CUDA 上不保证逐位一致（cuDNN 算法选择、kernel 顺序）。
2. 两条曲线第 `s` 步消费的样本必须相同——用 `common.micro_batch_indices`。
3. loss 都在 fp32 里累加——否则比的是"求和的精度差"而不是"训练的精度差"。

**怎么调**（`可直接执行`）：

```bash
python lab/scripts/run_numeric_ref.py --config tiny_cpu --device cpu --steps 6
python lab/scripts/run_numeric_ref.py --config v100_768 --device cuda:0 --steps 20 --tol 1e-3
```

**四个证据字段**：

| 字段 | 含义 | 怎么读 |
| --- | --- | --- |
| `max_dloss` | 两条曲线 `\|loss_fp16 − loss_fp32\|` 的最大值 | 量级问题，配 `max_rel_dloss` 一起看 |
| `first_diverge_step` | 第一次超过 `--tol` 的 step，一直没超就是 `None` | **比 `max_dloss` 更有诊断价值**：第 1 步就偏多半是前向数值问题（RMSNorm / softmax 溢出）；第 5 步才偏多半是优化器状态在低精度梯度下慢慢跑偏 |
| `scale_trajectory` | GradScaler 的 scale 序列 | 健康形态是"起点 65536，前几步可能连掉几次，然后长期不变"。单调下降到底 = 每步都有 inf，那是故障 B |
| `skip_steps` / `skip_ratio` | 被 scaler 跳掉的 step 数与比例 | `> 0` 不一定是病（开头几步常见），但 `skip_ratio > 1/3` 时这条 loss 曲线**不能**用来和 fp32 比较 |

`verdict()` 把这四个折成 PASS/FAIL 并给出失败时的第一个检查点，例如
"fp16 曲线里出现了 nan/inf。首个检查：RMSNorm 有没有在 float32 里算平方和；再看
`scale_trajectory` 是不是一路下降。"

**本机跑出来长这样**（`本机实测`，`tiny_cpu`，6 步）：

```text
max_dloss=3.386e-05  max_rel=6.085e-06  first_diverge_step=None  tol=1.0e-03
scale: first=65536.0 last=65536.0  skip_steps=0  skip_ratio=0.00
verdict=PASS
```

**这组数字不能迁移到 V100。** 本机的 fp16 是 CPU 软件模拟，走的不是 Tensor Core。
本机跑通只证明**这套对照方法本身没写错**；`max_dloss` 的具体数值必须在 V100 上重取。

### 4.3 通信账 → `lab/src/mm_v100/collectives.py`

**测什么**：一个 step 内的集合通信，**类型、次数、每次多大**。

**做法**：`CollectiveMeter` 是一个上下文管理器，进入时把 `torch.distributed` 模块上的八个
公共函数名（`all_reduce` / `broadcast` / `all_gather` / `all_gather_into_tensor` /
`reduce_scatter` / `reduce_scatter_tensor` / `all_to_all_single` / `barrier`）临时换成带计数的
包装，退出时还原。torch 2.1 的 DDP reducer 与 FSDP 都是通过**模块属性**调用这些函数的，
所以替换模块属性能拦到。只包公共名，不包 `_all_gather_base` / `_reduce_scatter_base`——
后者在 2.1 里是前者的 deprecated 包装，两个都包会重复计数。

字节量取的是**本 rank 发出去的那一份**，因为通信账关心每张卡的出口带宽，不是聚合后的总量。

**一个诚实的限制**：DDP 的梯度 allreduce 有一部分在 C++ reducer 里直接走 ProcessGroup，
不经过 Python 的 `dist.all_reduce`，所以本计数器在 DDP 下**可能少数**。
因此通信账**以 FSDP 路径为主**（FSDP 的 all_gather / reduce_scatter 是 Python 侧调用，能数全），
**DDP 侧的判据是"跨 rank 梯度逐元素一致"（不变量 I3），不是次数**。

**手算表**：

```python
ddp_expected(n_params, grad_dtype_bytes=4, bucket_cap_mb=25.0)
# 次数 = ceil(梯度总字节 / bucket 字节)，每次一个 bucket
# 单次 allreduce 的线上流量 = 2*(W-1)/W * bytes（ring：reduce-scatter 一圈 + all-gather 一圈）

fsdp_expected(n_params, n_units, world_size, param_dtype_bytes=2, grad_dtype_bytes=2, sharding=...)
# FULL_SHARD    : 2*n_units 次 all_gather + n_units 次 reduce_scatter
# SHARD_GRAD_OP : 1*n_units 次 all_gather + n_units 次 reduce_scatter
# NO_SHARD      : 退化成 DDP，只剩梯度 allreduce
```

**怎么调**（`可直接执行`）：

```bash
# 手算表（不需要 GPU）
python lab/scripts/byte_ledger.py --config v100_768 --hand-only \
    --world-size 8 --mode full_shard --show-collectives

# 训练时实测（有 Python 侧开销，只在需要时开）
python lab/scripts/train_bounded.py --config v100_768 --placement fsdp \
    --count-collectives 1 --max-steps 4
```

**输出字段**：`counts{op: n}` `bytes{op: n}` `total_calls` `total_bytes` `total_mib`；
手算侧 `expected_*_calls` `*_payload_bytes_each` `expected_total_bytes`；
`compare_to_expected()` 给每个 op 的 `expected` / `measured` / `ratio` / `within_tol`，
默认容差 `±25%`（FSDP 会为 buffer 与根单元多发几次），**超出说明 wrap 粒度或 sharding 策略
与你以为的不一样**。开了 `--count-collectives 1` 时，每步的 `summary()` 会作为 `collectives`
字段写进 `metrics_*.jsonl`。

**Gate 主配置下的手算通信账**（`估算`，`W=8`，`P=63,912,192`）：

| | DDP（fp32 梯度） | FSDP FULL_SHARD（fp16 参数 + fp16 归约） |
| --- | --- | --- |
| 次数 | 10 次 `all_reduce`（25 MB bucket，`ceil(255,648,768 / 26,214,400)`） | 18 次 `all_gather` + 9 次 `reduce_scatter`（`n_units = 8 层 + 1 根`） |
| 单次本 rank 出口 | 一个 bucket ≈ 26 MB | all_gather 1,775,338 B；reduce_scatter 14,202,709 B |
| 每 step 每卡合计 | 255,648,768 B | 159,780,480 B |

注意 `n_units = 9`：8 个 `Block` 各一个单元，加一个根单元装 `embed_tokens` / `final_norm` / `lm_head`。
tied embedding 与 `lm_head` 必须落在**同一个** FSDP 单元里，按 `Block` 粒度 wrap 时它们都留在
根单元，天然合法。Day 3 的实测第一件事就是核对 `n_fsdp_units == layers + 1`——
不等于 9 就说明 auto-wrap 策略没生效，后面所有通信账都不用看了。

**仪表自己的通信不算进账**：不变量 I3 需要一次 `all_gather` 传梯度指纹，
`bounded_train.py` 特意把它放在 `CollectiveMeter` 的上下文**之外**，
否则"一个 step 有几次集合通信"这个结论会被仪表自己污染。

---

## 5. 观测与故障树

### 5.1 三档静默故障：不变量 → 最小检查 → 判据

本周练的不是"会报错"或"会 hang"的故障，是**不报错、不 hang、loss 曲线看起来还挺正常、
但学习信号已经断了**的那三种。

| 档 | 注入方式（`lab/src/mm_v100/faults.py`） | 不变量 | 判据 | loss 曲线长什么样 |
| --- | --- | --- | --- | --- |
| **A `label_shift`** | `labels` 整体 `roll(1)`，**有效位置集合逐位不变** | **I1** `label_align_violations`：`labels != -100` 的位置必须有 `labels[i] == input_ids[i]` | `I1 > 0` | 平滑单调下降，只是平台高一点——模型学会了"预测上一个 token"这个更简单的任务 |
| **B `scaler_stuck`** | `register_hook` 往某个参数的梯度里塞 `inf`，GradScaler 每步都跳过 `optimizer.step()` | **I2** `optimizer_step_ratio = applied / total` 应接近 1，且 `param_delta_norm > 0` | `I2 ratio < 0.5` 或 `param_delta_norm <= 0` | **平坦**，在中断时刻的值附近抖动（前向照做，`loss.item()` 照常打印，`lr` 照常衰减） |
| **C `ddp_no_sync`** | 所有 micro-batch 都在 `no_sync()` 里做，梯度从不 allreduce | **I3** `cross_rank_grad_delta`：同步后各 rank 的梯度应**逐元素**相同 | `I3 > 1e-6` | 平滑下降，rank 0 上看起来完全正常——它确实在正确地拟合它看到的那 1/W 数据 |

用 `roll` 而不是切片补 `-100` 是刻意的：切片会改变 `n_label_tokens`，那样第二个不变量就能抓到它，
故障也就不"静默"了。`roll` 保持计数不变，**只有 I1 能抓**——这才是要练的那种。

用 `register_hook` 而不是在 backward 之后直接改 `.grad` 也是刻意的：后者在 DDP 下会被
allreduce 之前的 bucket 复制绕过，注入不进去。

**I3 为什么比的是逐元素梯度指纹，不是梯度范数。** 这是构建期在本机做故障注入时发现的
（`本机实测`）：基于范数的定义在数学上没问题，但**抓不到故障 C**——不同 micro-batch 的梯度
**方向**不同，**范数**却往往几乎相等，小模型加随机数据上实测各 rank 只差 3.58e-07，
落在任何合理阈值之下。根源是**范数是一个把方向信息全部丢掉的标量，而故障 C 恰恰只改变方向**；
提高精度（fp32 → fp64）救不了这一点，因为问题不是数值噪声，是这个统计量本身对要检测的差异
不敏感。所以 `cross_rank_grad_delta()` 改成 `all_gather` 一段**梯度指纹**（前 8 个参数张量各取
前 1024 个元素）之后**逐元素**比，返回相对偏差 `max|g_i − g_0| / mean|g_0|`。
**健康 DDP 下它精确为 0.0**——因为 all_reduce 的契约就是归约后各 rank 拿到逐位相同的 buffer。
`world_size = 1` 时它返回 `None`，不能拿 `0.0` 冒充"通过"。

这里有一条比不变量本身更值得记的教训：**先问这个统计量对你要检测的差异敏不敏感，
再问它的数值精度够不够。** 一个不敏感的量，算得再精确也还是个假阴性机器。

### 5.2 判别表与最小检查命令

```text
mode           I1>0   I2 低   I3>0
------------------------------------
none           否      否      否
label_shift    是      否      否
scaler_stuck   否      是      否
ddp_no_sync    否      否      是

必要不充分：loss 在下降 —— 四种模式下都成立。这是本周的主命题。
充分：任一不变量单独越界即可定位到唯一一档。
```

**一条命令跑完四档**（`可直接执行`）：

```bash
# 本机：2 进程 gloo
python lab/scripts/run_faults.py --config tiny_cpu --nproc 2 --force-cpu \
    --dtype float16 --max-steps 3

# V100：8 卡
python lab/scripts/run_faults.py --config smoke_v100 --nproc 8 \
    --dtype float16 --max-steps 6 --launcher torchrun
```

退出码 0 = 四档各自被正确的不变量抓住；1 = 有一档没被抓住或被抓错。

**本机真实输出**（`本机实测`，`tiny_cpu`，2 进程 gloo，3 步）：

```text
mode           I1         I2_ratio       I2_delta       I3               verdict
none           0          1.000          3.724e-03      0.0              none
label_shift    96         1.000          4.003e-03      0.0              label_shift
scaler_stuck   0          0.000          0.000e+00      nan              scaler_stuck
ddp_no_sync    0          1.000          3.740e-03      12.579889555222003 ddp_no_sync
```

读这张表的三个要点：

1. **`none` 与 `label_shift` 的 I2/I3 完全一样**（step_ratio 都是 1.000，I3 都是精确 0.0）。
   只有 I1 从 0 跳到 96。这就是"唯一识别性"的样子。
2. **`scaler_stuck` 那一行的 I3 是 `nan`，不是数字。** 因为所有梯度都被塞了 `inf`，
   指纹相减得到 `nan`。分类器仍然判对，因为 `float('nan') > 1e-6` 是 `False`，
   I3 不会误触发。看到 `nan` 不要以为是 I3 坏了——它是故障 B 的**副作用**，不是判据。
3. **`ddp_no_sync` 的 I3 是 12.58，不是一个小数。** 相对偏差没有量纲，
   同一个阈值 `1e-6` 在 `tiny_cpu` 和 `v100_768` 上通用。

### 5.3 单档的最小检查

**A（不需要 GPU、不需要模型）**：

```bash
python lab/scripts/check_data_contract.py --toy sft          # 无数据时用内置 fixture
python lab/scripts/check_data_contract.py --file sft_t2t_mini.jsonl --stage sft
```

看最后三行：`I1 label_align_violations_total`（正常值 0）、
`I2 n_label_tokens_mean / n_nonpad_mean / ratio`、`verdict`。
`ratio == 1.0` 说明 prompt 也进了 loss——pretrain 正常，SFT 就是错的。
`n_label_tokens == 0` 说明没有任何 target，`cross_entropy` 会对 0 个元素取均值得到 `nan`。
**这条命令只打印计数和比例，不打印任何样本内容**，所以它的输出可以直接贴出来看。

**B**：

```bash
python lab/scripts/run_numeric_ref.py --config v100_768 --device cuda:0 --steps 20
```

看 `scale_trajectory` 与 `skip_ratio`。要分开两种来源：
- **缩放溢出**（`scale × max|g| > 65504`）：减半几次就解决，scale 稳定在某个 `2^k`。
  **这是健康的自适应，不是故障。**
- **前向本身就产生 inf/NaN**：不论 scale 多小 `found_inf` 恒真，scale 从 65536 一路单调减半永不恢复。
  16 步到 1，40 步到 `2^-24`。

所以 **"scale 下降过"是 B 的必要条件，不是充分条件**；把它当判据会在每一个正常 run 上误报。
充分的是这一对：**I2 落地率 ≈ 0 且参数一步没动**。

**C**：

```bash
python lab/scripts/train_bounded.py --launcher spawn --nproc 2 --placement ddp \
    --config tiny_cpu --force-cpu --backend gloo --dtype float32 --max-steps 3
```

看日志行里的 `I3=`。健康 DDP 下它精确为 `0.0`。
一个补充判据：如果 `I3 = 0` 但**各 rank 的 loss 也完全相同**，那说明各 rank 在看同一批数据
（采样器未分片），此时是否同步已经无从区分——那是第四种故障，要靠跨 rank loss 差异来发现。

### 5.4 常见报错 → 第一检查点

| 症状 / 报错 | 多半是什么 | 第一检查点 |
| --- | --- | --- |
| `RuntimeError: Current CUDA Device does not support bfloat16`，或 lab 自己的"拒绝 bfloat16"长消息 | 某个命令的 `--dtype` 还是 `bfloat16`（MiniMind 全部 9 个 trainer 的默认值） | 命令行里改成 `--dtype float16`。lab 在**参数解析阶段**就拒绝，不会浪费已经建好的 8 个进程 |
| `PathContractError: 环境变量 MINIMIND_ROOT 没设` | 只影响 Day 0 的兼容审计 | `export MINIMIND_ROOT=...`，或者跳过这一步，其余四天照跑 |
| `PathContractError: 数据目录不存在` | `MM_DATA_ROOT` 没设，或者目录名大小写/下划线不一致 | `python lab/scripts/check_data_layout.py`，它会列出缺哪些 |
| `PathContractError: 字节数对不上` | 传输时用了文本模式，换行被改了 | 用 tar / rsync / scp 二进制重传那几个文件，再跑一次 `check_data_layout.py` |
| `SystemExit: FSDP1 需要 CUDA 设备，纯 CPU 上跑不了` | 在本机跑了 `--placement fsdp` | 本机只能验 `single` 与 `ddp`；FSDP 的门只能在 V100 上跑。这不是版本问题，torch 2.1 与新版行为一致 |
| `SystemExit: global_batch=X 必须能被 world_size*accum 整除` | 换了 `--nproc` 忘了同步改 `--global-batch` | 调 `--global-batch` 或 `--accum`，让 micro 是整数 |
| `_pickle.PicklingError: Can't pickle <function worker ...>: import of module 'mm_v100_script_*' failed` | **`run_day.py` 的已知缺陷**（第 7.4 节） | 直接调那一步的入口脚本（`equiv_check.py` / `run_faults.py` / `reshard_check.py`），单独跑都是通的 |
| `UnicodeEncodeError` 在打第一行日志时 | Windows 控制台代码页 | `export PYTHONIOENCODING=utf-8` |
| `[W socket.cpp:764] [c10d] The client socket has failed to connect to [<主机名>]:<端口>` | 本机 gloo 用主机名解析回环地址失败 | **本机噪声，不影响结果**。`reshard_check.py` 在本机每次都打这两行，随后照常 PASS |
| `probe_env.py` 退出码 1，`blocking_failures` 里有 `eight_gpus` / `bf16_absent_as_expected` | 在没有 GPU 的机器上跑的 | 本机上这是**预期**，不是故障。这条命令的结论只在目标机上有意义 |
| `--sdpa-backend flash` 在 V100 上报错 | sm70 没有 flash 后端（torch 2.1 的 flash 只覆盖 sm75–sm90） | **这是预期结果，也是 Day 0 要拿到的一条证据。** 用 `auto` 或 `mem_efficient` |
| `n_fsdp_units` 不等于 `layers + 1` | `auto_wrap_policy` 没生效，或 `transformer_layer_cls` 写错了 | 先修这个再看通信账。退化成单根单元时 all_gather buffer = 整份模型，FSDP 白做 |
| FSDP 构造时报共享参数错误 | tied 的 `embed_tokens` 与 `lm_head` 被分到了不同 unit | 确认按 `Block` 粒度 wrap，两者都应留在根单元 |
| reshard 失败，`total_numel` 对不上 | 读写两端的模型配置不同 | 先查 `meta.json` 的 `total_numel` 与当前模型参数总数；再查 `shard_bounds` 的余数分配在两端是否一致 |
| resume 之后头几步 loss 明显抬高 | checkpoint 少存了 optimizer（Adam 的一二阶矩清零） | `resume_check.py --tol 1e-6`，判据是**逐步相等**，不是"趋势接近" |
| loss 曲线出现"重新变简单"的台阶 | 少存了数据游标，从头吃数据 | 看 checkpoint 里的 `data_cursor.consumed_samples` |
| `zero_std_group_ratio > 0.8` | 规则 reward 值域只有 `[-3, 3]`，分不开这一组生成 | **预注册出口**：判 `INCONCLUSIVE`，`failure_class` 写 `insufficient`，不判 FAIL。这是实验设计的局限，不是管线错误 |

---

## 6. 两个扩展模块的代码地图

两个模块在 Gate 之后做，不进 Gate。本周交付的是**完整规格 + 可运行代码 + CPU 单测**；
实验执行排在 Gate 通过之后，因为两者合计 17 小时，超过主线本身。

### 6.1 `lab/src/mm_quant/` → Q1–Q6（规格见 `06_QUANT_LOWBIT.md`）

| 模块 | 对应实验 | 一句话 |
| --- | --- | --- |
| `lab/src/mm_quant/quantizers.py` | 全部的底座 | 对称/非对称 × per-tensor / per-channel / per-group 的 INT8 量化器；**scale 永远在 fp32 域算**（fp16 的 `amax/127` 对小 amax 的层会直接下溢成 0，整块权重变 0 而 loss 前几十步看不出来） |
| `lab/src/mm_quant/error_budget.py` | **Q1 + Q2** | Q1 逐层"误差 vs 存储"表与 `outlier_ratio = max\|w\|/mean\|w\|`；Q2 截断阈值 alpha 扫描与 MSE 的 clipping/rounding 分解 |
| `lab/src/mm_quant/opt8bit.py` | **Q3** | block-wise 8-bit Adam 状态。两个要点：必须 per-block；per-block 还不够，`exp_avg_sq` 还必须用幂律映射（线性 8-bit 在块内把小值压成 0，本机实测 30 步的最小二乘 loss 从 0.002 涨到 3973） |
| `lab/src/mm_quant/weight_only.py` | **Q4** | W8A16 weight-only PTQ：int8 存权重、用前 dequant 回 fp16 再做 GEMM。字节账真实，**速度账在 V100 上是负的**（多一次 dequant，没有 INT8 Tensor Core） |
| `lab/src/mm_quant/qat.py` | **Q5** | per-channel 权重 fake-quant + STE，自写路径与 `torch.ao` 路径**逐位对照**。这是量化模块里唯一在 V100 上完整成立的一件事——fake-quant 有原生 CUDA 前反向 kernel |
| `lab/src/mm_quant/smooth.py` | **Q6** | 激活 outlier 统计、跨 batch 重合率、SmoothQuant 式幅度迁移的 alpha 扫描。核心不变量：`(X/s) @ (W*s).T == X @ W.T`，迁移在 fp32 下是**恒等变换** |

入口：`lab/scripts/probe_int8_caps.py`（Q0 能力探针，**信息收集器不是门**，任何一项失败都不会非零退出）
与 `lab/scripts/run_quant_suite.py`（`.sh` 只负责选解释器调一次）。跑 `quant_v100.json` 之前
先跑 Q0：它的 `fake_quant.on_cuda` 决定 Q5 能不能在 GPU 上做，
`int_mm_matches_fp32` 决定 `torch._int_mm` 的结果能不能信（pytorch#107671，能跑不等于算对）。

### 6.2 `lab/src/mm_rl/` → R1–R6（规格见 `07_RL_LOWPRECISION.md`）

| 模块 | 对应实验 | 一句话 |
| --- | --- | --- |
| `lab/src/mm_rl/logprob.py` | **R1** | 同一批 token 的 logprob 在 fp32 / fp16-compute / fp16-stored 三条路径下的差异，并归到 `IDENTICAL` / `MASK_MISMATCH` / `PRECISION` / `PATH_MISMATCH` 四类之一；顺带注入一次 mask 错位和一次路径错位，证明归因器**能把它们分开** |
| `lab/src/mm_rl/kl.py` | **R2** | 三个 KL 估计量 `k1 = -log_r`、`k2 = log_r²/2`、`k3 = expm1(log_r) - log_r` 在 fp64/fp32/fp16 下的均值、方差、负值比例。**k3 出现负值 = 实现写错了（该用 `expm1`），不是精度问题** |
| `lab/src/mm_rl/scaler_rules.py` | **R3** | GradScaler 三条硬规则的**会抛异常的纪律检查器**：多 loss 各自 scale、多 optimizer 各自 step、`update()` 每 iteration 只调一次且在所有 step 之后（推论：梯度累积期间 scale 必须不变） |
| `lab/src/mm_rl/advantage.py` | **R4** | 组内 advantage 归一化的灾难性抵消。结论不是"fp16 不能用"，而是"**advantage 归一化必须在 fp32 里做**"——它相对前向反向的计算量可以忽略，没有理由省 |
| `lab/src/mm_rl/ref_model.py` | **R5** | **两个模块的交点**：把冻结的 ref 模型 weight-only INT8。三个判据 `kl_rel_bias` / `sign_agreement` / `grad_cosine`，其中符号一致率比均值更硬——符号翻了，KL 惩罚的方向就反了 |
| `lab/src/mm_rl/lora_policy.py` | **R6** | LoRA policy 的四项字节账 + step-0 不变量。B 零初始化 ⇒ policy 输出与 ref **逐位相同** ⇒ 三个 KL 估计量**精确等于 0**。这是整个 lab 里唯一一个"应当精确为 0"的断言 |
| `lab/src/mm_rl/toy.py` | 单测与 smoke 的替身 | 与 MiniMind 同形的最小因果 LM（子模块命名一致，所以 LoRA 的 target 选择、量化的 skip 列表可以原样搬到真模型上）。**它不是 MiniMind，也不假装是** |

入口：`lab/scripts/run_rl_suite.py`（`.sh` 同样只选解释器）。

**不装 vLLM / verl / OpenRLHF**：三者的共同点值得记住——**现代 RL 框架都把推理引擎当硬依赖，
而推理引擎是整个生态里对 GPU 架构和 torch 版本最挑剔的一环。** 六个实验全部自包含，
连 `trl` 都不依赖。

---

## 7. 本机验证状态

本机 = Windows 11，conda `rfm`，`C:/Users/zzz_7893/miniconda3/envs/rfm/python.exe`，
Python 3.11.16，torch 2.14.0+cpu，numpy 2.4.6，pytest 9.1.1，**无 GPU**（`torch.cuda.is_available()` 为 `False`）。

### 7.1 pytest（`本机实测`，2026-09-08，模型改动之后重跑）

命令：

```bash
cd outputs/1_cc_coaching/track_minimind/week03_minimind_v100_Only
PYTHONIOENCODING=utf-8 "C:/Users/zzz_7893/miniconda3/envs/rfm/python.exe" -m pytest lab/tests -q -rs
```

尾部输出，原样抄录：

```text
=========================== short test summary info ===========================
SKIPPED [1] lab\tests\test_quant_qat.py:173: 本机没有 GPU；fake-quant 的 CUDA kernel（FakeQuantizeCore.cu / FusedObsFakeQuant.cu）与 AMP 的真实行为必须在公司 8xV100 上验证
SKIPPED [1] lab\tests\test_quant_qat.py:182: 本机没有 GPU；fake-quant 的 CUDA kernel（FakeQuantizeCore.cu / FusedObsFakeQuant.cu）与 AMP 的真实行为必须在公司 8xV100 上验证
SKIPPED [1] lab\tests\test_reshard_cpu.py:162: FSDP SHARDED_STATE_DICT 需要 CUDA；cc 的开发机没有 GPU，这条只能在公司 8xV100 上跑
SKIPPED [1] lab\tests\test_rl_ref_model.py:176: 显存与 fp16 计算路径的结论必须在公司 8xV100 上测，本机无 GPU
SKIPPED [1] lab\tests\test_rl_scaler_rules.py:238: torch 2.1 的 GradScaler 只有 CUDA 实现；没有 GPU 时 enabled=False、scale 恒为 1，测不到溢出/backoff/跳步的真实行为。必须在公司 8xV100 上验证
SKIPPED [1] lab\tests\test_rl_scaler_rules.py:258: torch 2.1 的 GradScaler 只有 CUDA 实现；没有 GPU 时 enabled=False、scale 恒为 1，测不到溢出/backoff/跳步的真实行为。必须在公司 8xV100 上验证
368 passed, 6 skipped, 1 warning in 413.91s (0:06:53)
```

退出码 0。那一条 warning 来自 `test_quant_opt8bit.py::test_adam8bit_rejects_sparse_gradients`，
是 torch 关于稀疏张量不变量检查默认关闭的提示，与被测逻辑无关。

**六条跳过的用例，全部是 `skipif(not torch.cuda.is_available())`**，逐条说明：

| 用例 | 跳过原因 | 到 V100 上必须先做什么 |
| --- | --- | --- |
| `test_quant_qat.py::test_fake_quant_cuda_forward_and_backward` | 需要 CUDA。fake-quant 的 CUDA kernel（`FakeQuantizeCore.cu` / `FusedObsFakeQuant.cu`）与自写 STE 的**逐位一致性**只能在 GPU 上验 | 先跑 `probe_int8_caps.py`，结果里 `fake_quant.on_cuda` 不是 ok 时 Q5 要退回 CPU 并在证据里写明 |
| `test_quant_qat.py::test_qat_under_amp_float16_on_cuda` | 需要 CUDA。`autocast(float16) + GradScaler` 下的 QAT 行为 | 同上，且 V100 上**不能**换成 `bfloat16` |
| `test_reshard_cpu.py::test_fsdp_sharded_save_load_placeholder` | FSDP `SHARDED_STATE_DICT` 需要 CUDA | 见 7.3 的特别提醒 |
| `test_rl_ref_model.py::test_int8_reference_saves_real_device_memory` | 显存是不是真省了，只有量 `torch.cuda.max_memory_allocated` 才算数 | R5 的显存结论完全待测 |
| `test_rl_scaler_rules.py::test_enabled_scaler_backs_off_on_inf_and_skips_the_step` | torch 2.1 的 GradScaler 只有 CUDA 实现；无 GPU 时 `enabled=False`、scale 恒为 1，测不到溢出/backoff/跳步 | 本机验的是**调用顺序的纪律**，scale 的真实动态（init 65536 / backoff 0.5 / growth_interval 2000）待测 |
| `test_rl_scaler_rules.py::test_rl_amp_iteration_under_real_autocast_float16` | 同上 | 同上 |

### 7.2 本机真跑过的入口命令（`本机实测`，逐条抄录判据）

| 命令 | 结果 |
| --- | --- |
| `python -m compileall -q lab/src lab/scripts` | 退出码 0 |
| `python lab/scripts/probe_env.py` | 退出码 1，`blocking_failures = torch_version_matches_2.1.0, sm70_in_arch_list, eight_gpus, bf16_absent_as_expected`。**在本机这是预期**：这条命令的结论只在目标机上有意义 |
| `python lab/scripts/wiring_smoke.py --sizes 1,2 --force-cpu --config tiny_cpu` | 退出码 0，两档各跑满 2 步（`ws=2` 用时 9.8 s） |
| `python lab/scripts/check_data_contract.py --toy sft` | 退出码 0，`I1 label_align_violations_total = 0`，`ratio = 0.389`，`verdict = PASS` |
| `python lab/scripts/run_numeric_ref.py --config tiny_cpu --device cpu --steps 6` | 退出码 0，`max_dloss = 3.386e-05`，`first_diverge_step = None`，`scale 65536 → 65536`，`skip_steps = 0` |
| `python lab/scripts/equiv_check.py --config tiny_cpu --force-cpu --global-batch 8 --a-nproc 1 --a-accum 2 --b-nproc 2 --b-accum 1 --max-steps 3` | 退出码 0，逐步 `\|delta\|` 全部 `0.000e+00`，`max\|delta\| = 0.000e+00` |
| `python lab/scripts/run_faults.py --config tiny_cpu --nproc 2 --force-cpu --dtype float16 --max-steps 3` | 退出码 0，四档各被唯一不变量抓住（表见 5.2） |
| `python lab/scripts/byte_ledger.py --config v100_768 --hand-only --world-size 8 --mode full_shard --batch-per-rank 8 --seq-len 512 --show-collectives` | 退出码 0，`P = 63,912,192`，`n_units = 9`，18 次 all_gather + 9 次 reduce_scatter |
| `python lab/scripts/byte_ledger.py --config tiny_cpu --device cpu` | 退出码 0，`param` / `grad` / `optim` 三项手算与实测**逐字节相等**（361,984 / 361,984 / 724,064），`activation` 比值 1.806（估算项，落在 0.25–4.0 内） |
| `python lab/scripts/reshard_check.py --config tiny_cpu --force-cpu --save-nproc 2 --load-nproc 1 --max-steps 3` | 退出码 0，`total_numel = 106880`，分片和等于总长，step 换算 3(ws=2) → 6(ws=1)，loss 连续性 `\|delta\| = 0.000e+00` |
| `python lab/scripts/resume_check.py --config tiny_cpu --force-cpu --total-steps 6 --break-at 3` | 退出码 0，三步逐步 `\|delta\|` 全部 `0.000e+00`（容差 1e-6） |
| `python lab/scripts/run_day.py --day 1 / --day 4 / --day 5 --config tiny_cpu --force-cpu` | 三天全部退出码 0，每步 `rc=0` |
| `python lab/scripts/make_evidence.py --day 3 --skill byte_ledger --run-tag day3_dist --dry-run` | 退出码 0，19 个白名单字段，无路径、无主机名 |

`--help` 与 `--dry-run` 不再单独手跑：`lab/tests/test_scripts_cli.py` 里的
`test_help_exits_zero_for_every_script` 与 `test_dry_run_exits_cleanly` 对 `lab/scripts/` 下
**每一个** `.py` 都做了这两件事，随上面那次 pytest 一起通过。

### 7.3 一条会让人误判的特别提醒

`lab/tests/test_reshard_cpu.py::test_fsdp_sharded_save_load_placeholder` 是一个
**占位测试**：它在无 CUDA 时被跳过，**在有 CUDA 的机器上会主动失败**，失败消息是
"到了有 GPU 的机器上请把这条改成真实的 FSDP 往返测试"。

所以在 V100 上第一次跑 `pytest lab/tests` 时，看到这一条 FAILED **不是回归**，
是它按设计在提醒你补一个真实的 FSDP `SHARDED_STATE_DICT` 往返测试。
把它换掉之后，V100 上的 pytest 才应该全绿。

### 7.4 本机发现的一个缺陷（未修，因为编排层不在本文件作者的写权限内）

`lab/scripts/run_day.py` 的 `--day 2` 与 `--day 3` 在本机**跑不完**：

```text
_pickle.PicklingError: Can't pickle <function worker at 0x...>:
    import of module 'mm_v100_script_equiv_check' failed
```

原因：`run_day.py` 的 `load_script()` 用 `importlib` 以一个合成模块名
（`mm_v100_script_<脚本名>`）加载兄弟脚本，而 `torch.multiprocessing.spawn` 按
「模块名 + 限定名」**pickle** worker 函数，那个合成模块名在子进程里 import 不到。
`equiv_check.py`（Day 2）、`run_faults.py`（Day 2）、`reshard_check.py`（Day 3）
三个脚本都用 `spawn`，所以都中招。`wiring_smoke.py` 用的是 `subprocess`，不受影响。

**这不是 Windows 特有的**：`torch.multiprocessing.spawn` 显式指定 `start_method="spawn"`，
Linux 上同样按名字 pickle，所以 V100 上大概率同样命中。

**绕过办法（`本机实测`都通过）**：直接调那一步的入口脚本，不经过 `run_day.py`。
第 3.2 节那张表里每一行都是可以单独执行的完整命令。Day 0、1、4、5 的 `run_day.py` 不受影响。

### 7.5 什么完全没有验证过

**GPU 上一行都没跑过。** 下面每一项都是 `未验证`，文档里凡是标 `估算` 的数字全部属于此类：

| 项 | 状态 | 第一次在 V100 上先做什么 |
| --- | --- | --- |
| NCCL 通信、`torchrun` 启动 | 未验证 | `wiring_smoke.py --sizes 1,2,8 --launcher torchrun`，失败即停在那一档 |
| FSDP1（`--placement fsdp`） | 未验证；CPU 上直接 `SystemExit` 并说明原因 | 先 1 卡 `--placement fsdp --max-steps 2`，确认 `n_fsdp_units == layers + 1` |
| FSDP 分片 checkpoint（`--format fsdp`） | 未验证 | 先把 `--format flat` 那条路径跑通（本机已验），再换 |
| fp16 的真实数值行为 | 未验证（本机 fp16 是 CPU 软件模拟） | `run_numeric_ref.py --device cuda:0`，先看 `scale_trajectory` 是否稳定 |
| 显存与吞吐数字 | 未验证 | 8 卡非独占时 `step_time_ratio` 一律标 `measurement` 类不确定，只保留 `mem_ratio` 与 `collectives_total` 作结论 |
| SDPA 后端实际命中 | 未验证 | `probe_env.py` 的 `sdpa.functional`，预期 `flash = False` |
| `torch._int_mm` 在 sm70 上 | 未验证 | `probe_env.py` 的 `int8.callable_on_device`，预期 `False`。**注意本机这一项打印的是 `True`**——那是新版 torch 的 CPU 实现，对 V100 没有任何预测力 |
| activation 字节的手算系数 | `估算` | `ACT_COEF_HIDDEN = 8.0` / `ACT_COEF_INTER = 4.0` 是逐行数出来的，换 SDPA 后端会变；实测列以 `saved_tensors_hooks` 为准 |
| 量化模块 Q1–Q6 的全部 GPU 侧结论 | 未验证 | 先跑 `probe_int8_caps.py` |
| RL 模块 R1–R6 的全部 GPU 侧结论 | 未验证 | 先跑 `run_rl_suite.py --dry-run` 确认配置，再上 `rl_v100.json` |

### 7.6 两条与代码无关但会影响结论的边界

- **数据许可**：数据集卡片同时标了 `apache-2.0` 与 `cc-by-nc-2.0`（后者**非商用**）。
  把这批数据放到公司机器上是否触及这条边界，**请按公司政策自行判断**——
  cc 只标出这一条，不做法律判断。
- **`zero_std_group_ratio` 可能高到让 GRPO 无结论**：本周用规则 reward，值域只有 `[-3, 3]`。
  Day 5 预注册了 `INCONCLUSIVE` 出口（`> 0.8` 判 INCONCLUSIVE，不判 FAIL），
  Gate 不依赖 GRPO 有正向结果。
