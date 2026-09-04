# Week M01 Lab Guide — MiniMind 单机全链路（5070 Ti）

> 代码基线：MiniMind commit `7a6fddd63a30c06b2fdd5fac4089922b29bc841b`
> 实验包：`lab/`（安装与目录说明见 `lab/README.md`）
> 理论对照：`01_FOUNDATIONS.md`；本周目标与逐日地图：`00_WEEK_CARD.md`
> 命令等级约定：`可直接执行` = 复制即可跑（前提已在该条注明）；`模板` = 必须替换尖括号里的值或环境变量后才能跑；`伪代码` = 描述逻辑，不能直接跑。
> 数值标注约定：`本机实测` = 2026-09-04 在 `ResearchAgentPy310`（CPU）上真实跑出来的输出；`估算` = 由已确认参数推出的量级，未在任何 GPU 上验证。

---

## 0. 平台与预算

### 0.1 两台机器各自负责什么

| 环境 | 本周负责的门 | 明确不负责的事 |
| --- | --- | --- |
| 个人 5070 Ti 16 GB（sm_120） | Day 0 GPU 探针、Day 2 有界 pretrain、Day 3 SFT + 故障注入、Day 4 DPO、Day 5 GRPO smoke 与 eval/resume | 多卡；任何需要 > 16 GB 的配置；不能把单卡结果说成多卡结果 |
| 本机 CPU（`ResearchAgentPy310`） | `py_compile`、`pytest lab/tests`、`tiny_cpu` 配置的 dry run、纯函数数值断言（label 规则、DPO 数学、GRPO advantage、日志解析） | 有界训练；bf16/fp16 行为；显存与吞吐；任何"这个配置在 GPU 上能跑"的结论 |

判据很简单：**凡是输出里带显存、tokens/s、bf16 的结论，本机一律不产出**。`bounded_train` 在本机的 `peak_mem_mb` 恒为 `null`，`probe_env.py` 的 `arch_list` 是空列表 `[]`（`本机实测`）。

### 0.2 每次运行的时间上限与止损

原仓库 `trainer/train_*.py` **没有步数上限参数**（只有 `--epochs`），一旦启动就是整轮。这是本周所有"有界"设计的起因。三种止损手段：

| 步骤 | 时间上限 | 怎么强制 | 止损动作 |
| --- | --- | --- | --- |
| Day 0 探针 | 2 分钟（不算 sha256）；含 4 个 jsonl 的 sha256 时 15 分钟 | `probe_env.py --skip-hash` 跳过哈希 | 超时说明磁盘在读 GB 级文件，先 `--skip-hash` 拿到 GPU 结论，哈希单独跑 |
| Day 1 数据检查 | 3 分钟 | `inspect_dataset --n 3`（默认只看 3 条） | 用 `--fixture` 完全绕开大文件 |
| Day 2/3/4 有界训练 | 每次 20 分钟 | `bounded_train --max-steps N`，先用 `--max-steps 20` 估单步耗时再定 N | 单步耗时 × N > 20 分钟就降 `--max-steps` 或降 `--batch-size`，不要"先跑着看看" |
| Day 5 GRPO smoke | 15 分钟 | `run_grpo_smoke.ps1 -MaxMinutes 15` / `MAX_MINUTES=15`，脚本超时 kill 进程 | 配 `--save_interval 5`，被 kill 也有 checkpoint；smoke 的目的是"跑通并产出组内 reward"，不是收敛 |
| 单次 eval | 5 分钟 | `eval_generate --max-new-tokens 64`（8 条固定 prompt） | 生成不停就是症状本身，见第 5 节 F5 |

单日总预算按 `00_WEEK_CARD.md` 的逐日估时（Day 0 60 min / Day 1 90 / Day 2 120 / Day 3 120 / Day 4 90 / Day 5 120）。**超过当日估时 1.5 倍仍未过门，停止调参，把现象写进证据并转入第 5 节故障树。**

### 0.3 显存预算（全部 `估算`，等 5070 Ti 实测覆盖）

`01_FOUNDATIONS.md` 第 5.1 节给出显存模型，第 5.2 节给出 GRPO 三模型同驻的分析。本 lab 的 4 档配置按该模型选参：`smoke_5070ti` 估算 < 4 GB、`pretrain_5070ti` 估算 < 6 GB、`sft_5070ti` 估算 < 8 GB。这三个数字**没有任何实测支撑**，Day 0 之后的第一件事就是用 `bounded_train` 日志里的 `peak_mem_mb` 把它们换成实测值。GRPO 的显存是本周唯一预注册 `INCONCLUSIVE` 出口的项（见第 5 节 F7、F8）。

---

## 1. 数据与模型来源

### 1.1 代码

| 项 | 值 |
| --- | --- |
| 仓库 | `https://github.com/jingyaogong/minimind` |
| revision | commit `7a6fddd63a30c06b2fdd5fac4089922b29bc841b` |
| license | Apache-2.0 |
| 固定方法 | `setup_minimind.*` 执行 `git checkout <commit>` 后立即 `git rev-parse HEAD` 校验，不等就报错退出 |
| 持续核验 | `probe_env.py` 每次都比对 HEAD 与 `mm_probe.minimind_env.PINNED_COMMIT`，不等则写入 `warnings` |
| 用途 | 模型类 `model/model_minimind.py`、数据类 `dataset/lm_dataset.py`、tokenizer `model/tokenizer.json`、原训练脚本 `trainer/train_*.py` |

tokenizer 随仓库一起固定，不单独下载：vocab 6400，`<|endoftext|>`=0（pad）、`<|im_start|>`=1（bos）、`<|im_end|>`=2（eos）。`probe_env.py` 的 `tokenizer_loads` 字段就是断言这四个值（`本机实测`：`true`）。

### 1.2 数据

四个文件都来自 HF 数据集 `jingyaogong/minimind_dataset`，license 见该数据集卡片。`setup_minimind.*` 用 `huggingface_hub.hf_hub_download(repo_id='jingyaogong/minimind_dataset', filename='pretrain_t2t_mini.jsonl', repo_type='dataset', local_dir=$MINIMIND_ROOT/dataset)` 逐个下载（四个文件名见下表）。

| 文件 | 周卡记录字节数 | 字段 | 用在哪 |
| --- | --- | --- | --- |
| `pretrain_t2t_mini.jsonl` | 1,241,043,656 | `{"text": str}` | Day 1-2 |
| `sft_t2t_mini.jsonl` | 1,739,201,170 | `{"conversations":[{"role","content"}]}` | Day 1、3 |
| `dpo.jsonl` | 53,653,322 | `{"chosen":[msgs],"rejected":[msgs]}` | Day 4 |
| `rlaif.jsonl` | 23,754,740 | `{"conversations":[msgs]}`（末条 assistant 被丢弃，只留 prompt） | Day 5 |

**hash 的诚实说明**：本周**没有**预先已知的官方 sha256。`setup_minimind.*` 在首次下载当天用 `Get-FileHash` / `sha256sum` 计算并写入 `$MINIMIND_ROOT/dataset/SHA256SUMS.txt`——这是**你自己的下载记录**，作用是让以后每次 `probe_env.py`（不加 `--skip-hash`）能证明文件没被改动或截断。字节数则有期望值：`mm_probe.minimind_env.DATA_FILES` 里写死了上表四个数，`probe_env.py` 的 `size_matches_card` 就是这个比对。**首次下载得到的 hash 不能证明文件是官方原版，只能证明后续没变。** 上表四个字节数本机也未复核（本机没下载数据）。

### 1.3 Reward 模型（Day 5，可选）

| 项 | 值 |
| --- | --- |
| ID | `internlm/internlm2-1_8b-reward` |
| 大小 | safetensors 约 3,399,182,888 B（约 3.4 GB） |
| 加载方式 | 需要 `trust_remote_code` |
| 下载 | `hf download internlm/internlm2-1_8b-reward --local-dir "<MINIMIND_ROOT 的同级目录>/internlm2-1_8b-reward"`（旧版 CLI 用 `huggingface-cli download`） |
| 16 GB 是否放得下 | `未知`，本周无实测 |

### 1.4 离线替代（无网络 / 不下载 GB 级数据时）

| 需要的东西 | 替代 | 在哪 |
| --- | --- | --- |
| 4 个 jsonl | 内置 toy fixture：4 条 pretrain、3 条 SFT、3 条 DPO、2 条 RLAIF，字段格式已按固定 commit 的 `dataset/lm_dataset.py` 逐行核对 | `lab/src/mm_probe/fixtures.py`；CLI 用 `--fixture`（`inspect_dataset`）或 `--data-path fixture`（`bounded_train`，也是 `tiny_cpu.json` 的默认） |
| 1.8B reward 模型 | `RuleRewardModel`：四条手写规则（长度区间、与问题的 2-gram 重叠、3-gram 重复率、空回答），值域 [-3, 3]，接口与 `LMForRewardModel` 相同 | `lab/src/mm_probe/rule_reward.py`，由 `patch_grpo_rule_reward.py` 注入 |
| 官方 checkpoint | 不需要：`bounded_train` 默认随机初始化，`--from-weight none` 之外的权重都是你自己上一步导出的 | — |

**fixture 的边界**：它保证字段格式与编码规则正确，**不代表真实数据分布**。fixture 上量到的截断率、平均长度、loss 起点都不可外推到 mini 数据集。

---

## 2. 文件依赖图

### 2.1 谁依赖谁

```text
                       [外部：网络]
                            |
                            v
              scripts/setup_minimind.ps1|.sh
                            |
        +-------------------+--------------------+
        v                                        v
  $MINIMIND_ROOT/                     $MINIMIND_ROOT/dataset/
    model/model_minimind.py             pretrain_t2t_mini.jsonl
    model/tokenizer.json                sft_t2t_mini.jsonl
    dataset/lm_dataset.py               dpo.jsonl
    trainer/train_*.py                  rlaif.jsonl
        |                               SHA256SUMS.txt
        |                                        |
        +--------------+-------------------------+
                       | （经环境变量 MINIMIND_ROOT）
                       v
        src/mm_probe/minimind_env.py   <-- 所有模块定位 MiniMind 的唯一入口
                       |
   +-------------+-----+------+--------------+---------------+
   v             v            v              v               v
scripts/     inspect_    bounded_train.py  dpo_check.py  eval_generate.py
probe_env.py dataset.py        |                |               |
   |             |             |                |               |
   |             |      +------+------+         |               |
   |             |      v             v         |               |
   |             |  hooks.py     mask_fault.py  |               |
   |             |      |             |         |               |
   |             +------+------+------+---------+               |
   |                           |                                |
   v                           v                                v
runs/probe.json     runs/<config>_<stage>/log_<stage>.jsonl  runs/eval_<tag>.jsonl
                              |                                 |
                              v                                 v
                        parse_log.py                    eval_generate.py --compare
                              |
                              v
               runs/<config>_<stage>/log_<stage>.csv|.png

  [MiniMind trainer/train_grpo.py --debug_mode 的 stdout]
                              |
                              v
                        grpo_stats.py  --> runs/grpo_smoke_<r>/grpo_stats.json|.csv
                        parse_log.py   --> runs/grpo_smoke_<r>/grpo_steps.csv|.png

  scripts/patch_grpo_rule_reward.py + src/mm_probe/rule_reward.py
                              |
                              v
              $MINIMIND_ROOT/trainer/train_grpo_rule.py
```

配置文件被 `--config` 按名字或路径加载（`minimind_env.load_config`）：`bounded_train`、`dpo_check`、`eval_generate` 三者都吃 `--config`。`configs/*.json` 不依赖任何其他文件。

`tests/conftest.py` 只依赖 `src/`；`MINIMIND_ROOT` 缺失时依赖 tokenizer/模型的 13 条用例 skip。

### 2.2 每个本地路径由哪一步创建

| 路径 | 由哪一步创建 | 备注 |
| --- | --- | --- |
| `$MINIMIND_ROOT/`（含 `model/`、`dataset/lm_dataset.py`、`trainer/`） | `setup_minimind.ps1` / `.sh`（`git clone` + `checkout`） | 加 `-SkipData` / `--skip-data` 时只创建这些 |
| `$MINIMIND_ROOT/dataset/pretrain_t2t_mini.jsonl` | `setup_minimind.*` 的下载段（`hf_hub_download`） | 不加 `-SkipData` 时 |
| `$MINIMIND_ROOT/dataset/sft_t2t_mini.jsonl` | 同上 | 同上 |
| `$MINIMIND_ROOT/dataset/dpo.jsonl` | 同上 | 同上 |
| `$MINIMIND_ROOT/dataset/rlaif.jsonl` | 同上 | 同上 |
| `$MINIMIND_ROOT/dataset/SHA256SUMS.txt` | `setup_minimind.*` 的哈希段 | 只对**已存在**的文件写行；`-SkipData` 且数据未下载时该文件为空或缺失 |
| `$MINIMIND_ROOT/trainer/train_grpo_rule.py` | `scripts/patch_grpo_rule_reward.py` | 幂等；`run_grpo_smoke.* -Reward rule` 会先自动调它 |
| `$MINIMIND_ROOT/out/pretrain_768.pth` | `run_pretrain_bounded.*`：`bounded` 模式经 `--export-pth`；`epoch` 模式由原 `train_pretrain.py` 写 | Day 3 的 `--from-weight` 输入 |
| `$MINIMIND_ROOT/out/full_sft_768.pth` | `run_sft_bounded.*`，**且仅当 `-MaskFault none` / `MASK_FAULT=none`** | 故障注入的运行**不**导出，避免污染下游；Day 4/5 的输入 |
| `$MINIMIND_ROOT/out/dpo_768.pth` | `run_dpo_bounded.*` | Day 4 产物 |
| `$MINIMIND_ROOT/out/`、`$MINIMIND_ROOT/checkpoints/` | `epoch` 模式下由 MiniMind 原脚本创建 | 原脚本还写 `{weight}_{hidden}_resume.pth` |
| `lab/runs/probe.json` | `scripts/probe_env.py`（`--out` 的默认值） | Day 0 证据 |
| `lab/runs/<config>_<stage>/` | `bounded_train`（`--save-dir` 的默认值） | 例：`runs/tiny_cpu_sft/` |
| `lab/runs/<config>_<stage>/log_<stage>.jsonl` | `bounded_train`（`--log-jsonl` 的默认值 = `<save-dir>/log_<stage>.jsonl`） | 每步一行 |
| `lab/runs/<config>_<stage>/fixture_<stage>.jsonl` | `bounded_train`，当数据路径解析为 `fixture` 时（`fixtures.write_fixture_jsonl`） | 让 MiniMind 的 dataset 类有真文件可读 |
| `lab/runs/<config>_<stage>/ckpt_step<N>.pt`、`latest.pt` | `bounded_train --save-every N` | 含 model/optimizer/scheduler/scaler/rng/step/cursor |
| `lab/runs/<config>_<stage>/log_<stage>.csv`、`.png` | `parse_log`（由各 `run_*` 脚本自动调用） | 无 matplotlib 时无 `.png`，改打 ASCII |
| `lab/runs/pretrain_<Config>/` | `run_pretrain_bounded.*` | |
| `lab/runs/sft_<Config>_<MaskFault>/` | `run_sft_bounded.*` | 三种 MaskFault 各一个目录，便于并排对照 |
| `lab/runs/dpo_<Config>/` + `dpo_check_before.json` / `dpo_check_after.json` | `run_dpo_bounded.*` | 训练前后各一次 |
| `lab/runs/grpo_smoke_<Reward>/grpo_stdout.log` | `run_grpo_smoke.*`（tee / 重定向 MiniMind stdout） | `grpo_stats` 与 `parse_log` 的输入 |
| `lab/runs/grpo_smoke_<Reward>/grpo_stats.json`、`grpo_groups.csv`、`grpo_steps.csv` | `run_grpo_smoke.*` 末尾的 `grpo_stats` 与 `parse_log` | Day 5 证据 |
| `lab/runs/eval_<tag>.jsonl` | `eval_generate`（`--out` 的默认值） | `run_*` 脚本会用 `--out` 指到各自 RunDir |

**`lab/runs/` 整个目录都是产物**，仓库里不预置任何内容，第一次运行时由脚本 `mkdir -p` 创建。

---

## 3. 逐门执行

七道门，前一道不过不进下一道。每条命令都标了等级与预期观测。

统一前缀（本机）：

```powershell
$Py = "D:/Software/Large/Anconda/envs/ResearchAgentPy310/python.exe"
```

5070 Ti 机器上把 `$Py` 换成那台机器上装了 cu128 torch 的解释器，并 `$env:MM_PYTHON = $Py` 让 `run_*` 脚本也用它。

### 门 0 — 环境与来源锁定（Day 0）

**0a. 克隆并固定 commit**　`命令等级：模板`（替换 `<clone-dir>`）

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File lab\scripts\setup_minimind.ps1 -Root <clone-dir> -Python python
$env:MINIMIND_ROOT = "<clone-dir>"
```

bash：

```bash
ROOT=$HOME/minimind PYTHON=python bash lab/scripts/setup_minimind.sh
export MINIMIND_ROOT=$HOME/minimind
```

预期观测：打印 `[setup] MiniMind at 7a6fddd63a30c06b2fdd5fac4089922b29bc841b`，随后每个文件一行 `sha256=<64 hex> size=<bytes>`。HEAD 不等于锁定值时脚本直接抛错退出。下载 4 个文件约 3 GB，耗时取决于带宽（`估算` 10-40 分钟）。

**0b. 探针**　`命令等级：可直接执行`（`MINIMIND_ROOT` 已设置）

```powershell
& $Py lab\scripts\probe_env.py --skip-hash
```

预期观测（5070 Ti，全部 `估算`）：`compute_capability: "12.0"`、`arch_list` 含 `"sm_120"`、`bf16_supported: true`、`cuda_matmul_ok: true`、`total_mem_mb` 约 16000、`warnings: []`。
预期观测（本机 CPU，`本机实测`）：`cuda_available: false`，`gpu_name` / `compute_capability` / `bf16_supported` 均为 `null`，`arch_list: []`，`commit_matches: true`，`tokenizer_loads: true`；未下载数据时 `warnings` 恰好 4 条"数据缺失"，退出码 0。

**过门判据**：`commit_matches` 与 `tokenizer_loads` 都为 `true`；5070 Ti 上另加 `cuda_matmul_ok: true`。`cuda_matmul_ok: false` 直接跳第 5 节 F1。

### 门 1 — CPU 单测（Day 0-1，本机即可）

`命令等级：可直接执行`（在 `week01_minimind_5070ti/` 目录下）

```powershell
& $Py -m pytest lab\tests -q
```

预期观测（`本机实测`）：`42 passed in 23.17s`。不设 `MINIMIND_ROOT` 时 `29 passed, 13 skipped in 13.54s`。

`命令等级：可直接执行`（语法门，23 个 `.py` 全过才算）

```bash
"D:/Software/Large/Anconda/envs/ResearchAgentPy310/python.exe" -m py_compile $(find lab -name "*.py" | sort | tr '\n' ' ')
```

预期观测（`本机实测`）：无输出、退出码 0。

**过门判据**：42 passed 且 0 failed。若报 `PermissionError` 指向 `AppData\Local\Temp\pytest-of-<user>`，那是 pytest 默认临时目录不可写，不是代码问题，见第 5 节 F10。

### 门 2 — 数据 → token/label 因果链（Day 1）

**2a. fixture 模式**　`命令等级：可直接执行`

```powershell
& $Py lab\scripts\mmp.py inspect_dataset --stage sft --fixture --n 1 --max-seq-len 64
```

预期观测（`本机实测`，fixture 第 0 条 `你好` / `你好！有什么可以帮你？`）：

```text
   0 | <|im_start|>       |     1 |   -100 | template
   4 | 你好                 |  1968 |   -100 | user
  11 | \n                 |   234 |   -100 | template
  12 | 你好                 |  1968 |   1968 | assistant
  19 | <|im_end|>         |     2 |      2 | template
  20 | \n                 |   234 |    234 | template
stats: {"n_nonpad": 21, "n_in_loss": 9, "loss_token_ratio": 0.42857142857142855, "first_loss_pos": 12,
        "eos_in_loss": true, "truncated": false, "raw_len": 21,
        "per_segment": {"template": {"tokens": 13, "in_loss": 2}, "user": {"tokens": 1, "in_loss": 0},
                        "assistant": {"tokens": 7, "in_loss": 7}, "pad": {"tokens": 43, "in_loss": 0}}}
```

三个必须成立的不变量：`first_loss_pos` 正好落在 `<|im_start|>assistant\n` 这 5 个 token 之后（这里 12 = 7 + 5）；`user` 段与 `pad` 段的 label 全是 -100；`eos_in_loss: true`（模型必须学会停）。

**2b. 真实数据**　`命令等级：可直接执行`（需已下载 `sft_t2t_mini.jsonl`）

```powershell
& $Py lab\scripts\mmp.py inspect_dataset --stage sft --n 5 --no-table --out lab\runs\inspect_sft.json
& $Py lab\scripts\mmp.py inspect_dataset --stage pretrain --n 5 --no-table --out lab\runs\inspect_pretrain.json
```

预期观测（`估算`）：`aggregate.mean_loss_token_ratio` 在 0.3-0.6 之间（SFT 只有 assistant 段进 loss）；`truncation_ratio` 在默认 `max_seq_len 768` 下应远小于 1。**若 `mean_loss_token_ratio` 接近 0 或接近 1，立刻停**，前者是 bos 序列匹配失败、后者是 mask 没生效，见第 5 节 F5。

**2c. 两个随机分支各看一次**　`命令等级：可直接执行`

```powershell
& $Py lab\scripts\mmp.py inspect_dataset --stage sft --fixture --n 1 --add-system --keep-empty-think
```

MiniMind 在 `__getitem__` 里以 20% 概率加 system prompt、80% 概率删空 `<think>\n\n</think>\n\n`。本模块把它们改成确定开关，否则同一条样本两次编码结果不同，没法对照。预期观测：加 `--add-system` 后开头多出一个 system 块且其 label 仍全为 -100；加 `--keep-empty-think` 后 assistant 段内多出 `<think>` / `</think>` token 且它们**进** loss。

**过门判据**：`tests/test_label_mask.py` 的 11 条全过（已在门 1 覆盖），且你能对一条**新的**多轮样本手写出 label 序列并被 `inspect_dataset` 证实。

### 门 3 — 单 batch 前反向（Day 2 起点，本机即可）

`命令等级：可直接执行`

```powershell
& $Py lab\scripts\mmp.py bounded_train --config tiny_cpu --stage sft --max-steps 1
```

预期观测（`本机实测`）：

```text
[bounded:sft] step 1/1 loss 8.7434 lr 1.00e-05 grad_norm 3.2004 scale - tok/s 1047 peak_mem -
```

随后 summary 里 `n_params: 532928`、`device: "cpu"`、`dtype: "float32"`。（`tok/s 1047` 含首步预热开销，低于稳态。）

**过门判据**：loss 有限且落在 ln(6400)=8.764 附近（随机初始化的理论值，见 `01_FOUNDATIONS.md` 第 2.1 节）；`grad_norm` 有限且非 0。loss 远离 8.76 就说明 label 或权重加载有问题。

### 门 4 — smoke（Day 2，5070 Ti）

`命令等级：模板`（需 `$env:MINIMIND_ROOT` 与 `$env:MM_PYTHON`）

```powershell
$env:MM_PYTHON = "<5070Ti 上的 python.exe>"
.\lab\scripts\run_pretrain_bounded.ps1 -MaxSteps 20 -Config smoke_5070ti -SaveEvery 0
```

bash：

```bash
MAX_STEPS=20 CONFIG=smoke_5070ti SAVE_EVERY=0 bash lab/scripts/run_pretrain_bounded.sh
```

预期观测（全部 `估算`）：20 行 `[bounded:pretrain]`，`peak_mem_mb` 稳定在 4000 以下、`scaler_scale` 为 `null`（bf16 不用 GradScaler）、`tokens_per_s` 在前 2-3 步偏低（CUDA 上下文与 kernel autotune）之后稳定。

**过门判据**：20 步全部 loss 有限，且 `peak_mem_mb` 有真实数值（不是 `null`）。这一步的产出就是把第 0.3 节的显存 `估算` 换成实测。

### 门 5 — 有界训练（Day 2-4，5070 Ti）

**5a. overfit 一小撮样本**　`命令等级：模板`

```powershell
.\lab\scripts\run_pretrain_bounded.ps1 -MaxSteps 300 -OverfitN 128 -Config pretrain_5070ti
```

预期观测（`估算`）：`--overfit-n 128` 让数据游标只在前 128 条里循环，loss 应显著下降（远低于 8.76）。**这是唯一能证明"梯度确实在改变模型"的廉价实验**：如果 128 条样本都 overfit 不动，问题不在数据量，在 label、lr 或权重加载。

**5b. 有界 pretrain**　`命令等级：模板`

```powershell
.\lab\scripts\run_pretrain_bounded.ps1 -MaxSteps 2000 -SaveEvery 500 -Config pretrain_5070ti
```

产出 `$MINIMIND_ROOT/out/pretrain_768.pth`（下一门的输入）与 `runs/pretrain_pretrain_5070ti/log_pretrain.jsonl|.csv|.png`。

**5c. 有界 SFT + 故障注入三连**　`命令等级：模板`

```powershell
.\lab\scripts\run_sft_bounded.ps1 -MaxSteps 300 -Config sft_5070ti -MaskFault none
.\lab\scripts\run_sft_bounded.ps1 -MaxSteps 300 -Config sft_5070ti -MaskFault user_in_loss
.\lab\scripts\run_sft_bounded.ps1 -MaxSteps 50  -Config sft_5070ti -MaskFault assistant_all_ignored
```

三次写到三个独立目录，只有第一次导出 `out/full_sft_768.pth`。对照生成结果：

```powershell
& $Py lab\scripts\mmp.py eval_generate --compare lab\runs\sft_sft_5070ti_none\eval.jsonl lab\runs\sft_sft_5070ti_user_in_loss\eval.jsonl
```

预期观测：`none` 的 loss 曲线正常下降；`user_in_loss` 的 loss **起点更低且下降更快**（模板 token 极易预测），生成里出现复述 user 文本或自己生成 `<|im_start|>user`；`assistant_all_ignored` 的 loss 立刻是 `nan`，`n_label_tokens` 为 0，`bounded_train` 退出码 2。第三种的 50 步足够，不用跑 300。

先看故障症状的廉价版（本机即可，`命令等级：可直接执行`）：

```powershell
& $Py lab\scripts\mmp.py mask_fault --mode user_in_loss --index 1 --max-seq-len 48
```

预期观测（`本机实测`）：`in-loss tokens: normal=6  faulty=40`，逐 token 表里 label 列显示 `-100->1`、`-100->832` 这样的变化。

**5d. 有界 DPO**　`命令等级：模板`

```powershell
.\lab\scripts\run_dpo_bounded.ps1 -MaxSteps 200 -SaveEvery 50 -Config sft_5070ti
```

脚本自动做三件事：训练前 `dpo_check --ref same --check-ref-frozen`、有界训练、训练后 `dpo_check --policy dpo_768.pth --ref full_sft_768.pth`。

单独跑 `dpo_check` 的廉价版（本机即可，`命令等级：可直接执行`）：

```powershell
& $Py lab\scripts\mmp.py dpo_check --config tiny_cpu --n 2
```

预期观测（`本机实测`）：`"dpo_loss": 0.6931471824645996`、`"ln2_reference": 0.6931471805599453`、`reward_margin: [0.0, 0.0]`、`logits_dpo: [0.0, 0.0]`。**policy = ref 时 loss 必须等于 ln2**，这是 DPO 唯一的解析检查点。训练后 `reward_margin` 应转正、`dpo_loss` 应小于 0.6931。

### 门 6 — GRPO smoke + eval（Day 5，5070 Ti）

**6a. GRPO smoke**　`命令等级：模板`

```powershell
.\lab\scripts\run_grpo_smoke.ps1 -Reward internlm -NumGenerations 2 -MaxMinutes 15
```

放不下 reward 模型时：

```powershell
.\lab\scripts\run_grpo_smoke.ps1 -Reward rule -NumGenerations 2 -MaxMinutes 15
```

预期观测（`估算`）：stdout 里每步一段 `[DEBUG] step=N, sample[0]` + 每个 generation 的 `RESPONSE_BEGIN` / `RESPONSE_END` / `reward=`，以及每步一条汇总行，格式为：

```text
Epoch:[1/1](3/500), Reward: 1.2345, KL_ref: 0.0012, Adv Std: 0.9000, Adv Mean: 0.0000, Actor Loss: -0.0123, Avg Response Len: 120.50, Learning Rate: 0.00000030
```

这正是 `parse_log --format minimind` 与 `grpo_stats --format debug-log` 解析的那一行（上面的数值是格式示例，不是实测值）。15 分钟到点被 kill 是**预期结果**，不是失败。

**6b. 读组内统计**　`命令等级：可直接执行`（`grpo_stdout.log` 已存在时；`run_grpo_smoke.*` 末尾已自动调用）

```powershell
& $Py lab\scripts\mmp.py grpo_stats --input lab\runs\grpo_smoke_rule\grpo_stdout.log --format debug-log --csv lab\runs\grpo_smoke_rule\grpo_groups.csv --json lab\runs\grpo_smoke_rule\grpo_stats.json
```

预期观测：先打印一个 JSON 头（`n_steps` / `n_groups` / `zero_std_group_ratio` / `mean_group_std` / `mean_reward` / `mean_len`），再打印逐步表。**`zero_std_group_ratio` 是本门的核心数字**，超过 0.5 见第 5 节 F6。

用合成输入验证这条链路（本机即可，`命令等级：可直接执行`）：

```bash
printf '{"step":1,"rewards":[[1.0,1.0],[0.0,2.0]]}\n{"step":2,"rewards":[[0.5,1.5]]}\n' > /tmp/r.jsonl
"D:/Software/Large/Anconda/envs/ResearchAgentPy310/python.exe" lab/scripts/mmp.py grpo_stats --input /tmp/r.jsonl
```

预期观测（`本机实测`）：`"zero_std_group_ratio": 0.3333333333333333`、`"mean_group_std": 0.5`、`"mean_reward": 1.0`；逐步表第 1 步 `zero_std=1`、`adv_std=0.7070`。

**6c. 跨阶段 eval**　`命令等级：模板`

```powershell
& $Py lab\scripts\mmp.py eval_generate --config sft_5070ti --checkpoint "$env:MINIMIND_ROOT\out\pretrain_768.pth" --mode raw --tag pretrain --out lab\runs\eval_pretrain.jsonl
& $Py lab\scripts\mmp.py eval_generate --config sft_5070ti --checkpoint "$env:MINIMIND_ROOT\out\full_sft_768.pth" --mode chat --tag sft --out lab\runs\eval_sft.jsonl
& $Py lab\scripts\mmp.py eval_generate --config sft_5070ti --checkpoint "$env:MINIMIND_ROOT\out\dpo_768.pth" --mode chat --tag dpo --out lab\runs\eval_dpo.jsonl
& $Py lab\scripts\mmp.py eval_generate --compare lab\runs\eval_sft.jsonl lab\runs\eval_dpo.jsonl
```

`--mode raw` 用于 pretrain 权重（直接续写），`--mode chat` 用于 SFT 之后（走 chat 模板）。预期观测（`估算`）：pretrain 权重续写通顺但不回答问题、`ended_with_eos` 多为 false；SFT 之后 `ended_with_eos` 应显著变多。

### 门 7 — resume 与故障恢复（Day 5）

`命令等级：可直接执行`（本机 tiny 版，验证机制本身）

```powershell
& $Py lab\scripts\mmp.py bounded_train --config tiny_cpu --stage sft --max-steps 4 --save-every 2 --save-dir lab\runs\res_a --quiet
& $Py lab\scripts\mmp.py bounded_train --config tiny_cpu --stage sft --max-steps 4 --resume lab\runs\res_a\ckpt_step2.pt --save-dir lab\runs\res_b
```

预期观测（`本机实测`）：第二条先打印 `[bounded] resumed from <ckpt 路径>: step=2 cursor={'epoch': 1, 'pos': 1, 'n': 3, 'batch_size': 2, 'seed': 42, 'overfit_n': 0}`，然后只跑 step 3 与 4。`tests/test_bounded_train.py::test_save_resume_equivalence` 断言恢复后的 step 3/4 的 `loss`、`lr`、`sample_indices` 与连续训练**逐位相等**（`abs=1e-6`），本机已通过。

`命令等级：模板`（5070 Ti 上的真实恢复）

```powershell
.\lab\scripts\run_pretrain_bounded.ps1 -MaxSteps 2000 -Config pretrain_5070ti -Resume "<lab>\runs\pretrain_pretrain_5070ti\ckpt_step500.pt"
```

预期观测（`估算`）：GPU + bf16 下 **不要期待 bitwise 相等**。原因见 `01_FOUNDATIONS.md` 第 6 节第 12 条：SDPA 反向非确定、以及 MiniMind 原格式 checkpoint 存的是 `half()` 后的权重。判据改成"恢复后 20 步的 loss 均值与中断前 20 步的均值差落在噪声范围内"。CPU + float32 下才应该逐位相等（本机已验证）。

**过门判据**：CPU 上逐位相等；GPU 上均值差在噪声范围内且曲线无台阶。出现台阶见第 5 节 F9。

---

## 4. 日志与曲线怎么读

### 4.1 `bounded_train` 的 JSONL（每步一行）

| 字段 | 含义 | 正常范围 | 异常信号 |
| --- | --- | --- | --- |
| `step` | 全局 micro-step，从 1 计；`--resume` 后从 `ckpt.step + 1` 继续 | 连续无跳号 | 跳号说明日志被两次运行覆盖写 |
| `stage` | `pretrain` / `sft` / `dpo` | 与 `--stage` 一致 | — |
| `epoch` / `cursor_pos` | 数据游标状态，`(epoch, pos)` 完全决定下一 batch | `pos` 在 `[0, len(ds))` 内循环 | resume 后 `pos` 不接续 = 游标没恢复 |
| `sample_indices` | 本步用到的样本下标 | `--overfit-n N` 时全部 < N | resume 前后同 step 的该字段必须相同 |
| `loss` | pretrain/sft 为 `res.loss + res.aux_loss`；dpo 为 DPO loss + `aux_loss` | 随机初始化起点 ≈ ln(6400) = 8.76（`本机实测` 8.74）；DPO 首步 = ln2 = 0.6931（`本机实测` 0.69314718） | `nan` → 第 5 节 F5；恒定不动 → F5 或 lr 为 0 |
| `lr` | `minimind_lr(step, max_steps, base_lr)`，逐字复现 `trainer_utils.get_lr`：`lr*(0.1+0.45*(1+cos(pi*step/total)))` | 从 `0.955*base_lr` 单调降到 `0.1*base_lr` | 恒定 = `--max-steps` 传错；注意 `total` 就是 `--max-steps`，改它会改整条 lr 曲线 |
| `grad_norm` | clip **之前**的全局二范数（`hooks.grad_global_norm`） | `本机实测` tiny_cpu 上 3.0-4.6；GPU 上量级 `估算` 0.1-10 | `nan` → fp16 溢出或 label 全 -100；持续贴着 `grad_clip` → lr 偏大 |
| `optimizer_step` | 本步是否真的调用了 `optimizer.step()` | `accumulation_steps=1` 时恒 `true` | `accum>1` 时每 `accum` 步一次 `true`；全 `false` 说明 accum 与 max_steps 配错 |
| `scaler_scale` | `GradScaler.get_scale()`，**只有 float16 才非 null** | float16 下从 65536 起，偶发减半 | 一路减半到 < 1 → 第 5 节 F3；bf16/float32 下为 `null` 是正确的 |
| `tokens` | 本步非 pad token 数（dpo 为 chosen+rejected 合计） | 接近 `batch_size × max_seq_len × (1 - pad率)` | 远低于预期 = 样本太短或截断设置有问题 |
| `tokens_per_s` | 本步吞吐（含 CUDA sync） | `本机实测` tiny_cpu 4200-7700；前 2-3 步偏低是正常预热 | 持续走低 → 第 5 节 F11 |
| `peak_mem_mb` | `torch.cuda.max_memory_allocated()`，**无 CUDA 时为 null** | 见第 0.3 节 `估算` | 逐步单调上升 = 有张量没释放 |
| `dtype` | `float32` / `float16` / `bfloat16` | 与 `--dtype` 一致 | 配了 cuda 但本机无 CUDA 时会被降级为 cpu，stderr 有提示 |
| `elapsed_s` | 自本次运行开始的墙钟秒 | 单调递增 | — |
| `n_label_tokens` | 进 loss 的位置数（pretrain/sft）；dpo 为 mask 之和 | SFT 上约占非 pad token 的 30%-60%（`估算`） | 0 → label 全 -100，第 5 节 F5；等于 `tokens` → mask 没生效 |
| `reward_margin` | **仅 dpo**：`beta * logits_dpo` 的 batch 均值 | 首步 0（policy = ref）；训练中应转正 | 长期为 0 → 第 5 节 F5 的 DPO 版（`mask.sum(1)` 为 0） |
| `dpo_accuracy` | **仅 dpo**：`logits_dpo > 0` 的比例 | 首步 0；训练中应升向 1 | — |

### 4.2 MiniMind 原脚本 stdout（`parse_log --format minimind`）

字段名被统一成小写下划线（`Adv Std` → `adv_std`，`Learning Rate` → `learning_rate`）。

| 来源脚本 | 解析出的字段 | 读法 |
| --- | --- | --- |
| `train_pretrain.py` / `train_full_sft.py` | `epoch`、`epochs`、`step`、`iters`、`loss`、`logits_loss`、`aux_loss`、`lr`、`epoch_time` | `loss = logits_loss + aux_loss`；dense 模型 `aux_loss` 恒 0，非 0 说明开了 MoE |
| `train_dpo.py` | `loss`、`dpo_loss`、`aux_loss`、`learning_rate`、`epoch_time` | `dpo_loss` 从 0.6931 出发；官方 lr 4e-8 极小，几百步内变化很小是正常的（原因见 `01_FOUNDATIONS.md` 第 4.4 节） |
| `train_grpo.py` | `reward`、`kl_ref`、`adv_std`、`adv_mean`、`actor_loss`、`avg_response_len`、`learning_rate` | `adv_mean` 按定义恒 ≈ 0（组内减均值），**不要**把它当学习信号；`adv_std` ≈ 0 才是警报 |

`parse_log` 的 ASCII 摘要每行给 `first / last / min / max` 加一条 sparkline，例（`本机实测`）：

```text
rows=3 step 1→3
field                    first        last         min         max  trend
loss                    8.7434      8.6395      8.6395      8.7481  %@
grad_norm               3.2004      4.0422      3.2004      4.5389   @+
```

3 步的趋势没有意义，只用来确认字段解析对了。

### 4.3 `grpo_stats` 的输出

| 字段 | 含义 | 正常范围 |
| --- | --- | --- |
| `n_groups` | 累计组数（每个 prompt 一组） | = 步数 × batch_size |
| `zero_std_group_ratio` | 组内 reward 完全相同的组占比 | `估算` `num_generations=2` 时偏高属正常；**> 0.5 就是警报** |
| `mean_group_std` | 组内 reward 标准差的均值（`unbiased=False`） | 越接近 0 越没信号 |
| `mean_reward` | 全部 reward 的均值 | 规则 reward 值域 [-3, 3] |
| `mean_len` | 平均回答长度（有 `lengths` 用 token/字符数，否则用日志里的 `Avg Response Len`） | 持续单调上升要警惕长度 hack |
| `per_step[].adv_mean` | 组内减均值后的 advantage 均值 | 按定义 ≈ 0 |
| `per_step[].adv_std` | advantage 的标准差 | ≈ 1（除以了 std）；≈ 0 说明该步全部组都是零方差 |

### 4.4 `dpo_check` 的输出

`policy_logp_chosen` / `policy_logp_rejected` / `ref_logp_chosen` / `ref_logp_rejected` 是 masked 求和后的**序列** log-prob，都是负数，绝对值随 `mask_tokens_*` 增大而增大——**跨样本比较它们没有意义**，要看的是差：

- `logp_diff_chosen(policy-ref)` 与 `logp_diff_rejected(policy-ref)`：训练目标是让前者上升、后者下降。
- `logits_dpo = (pi_c - pi_r) - (ref_c - ref_r)`，`reward_margin = beta * logits_dpo`。
- `dpo_loss` 与 `ln2_reference` 并排给出，方便一眼比对：policy = ref 时两者必须相等（`本机实测` 0.6931471824645996 vs 0.6931471805599453，差为 float32 精度）。
- `--check-ref-frozen` 额外给 `ref_frozen_check`：`ref_unchanged` 必须 `true`、`policy_changed` 必须 `true`。**这两个 hash 是"ref 到底冻没冻住"的唯一硬证据。**

---

## 5. 故障树

对照 `01_FOUNDATIONS.md` 第 6 节的 14 条失败模式。下表每条给"症状 → 首个检查 → 下一步"。

### F1 — sm_120 不被支持（对应 Day 0）

**症状**：`torch.cuda.is_available()` 返回 `True`，但第一次真正的 CUDA 计算报 `CUDA error: no kernel image is available for execution on the device`；或者显式提示 `sm_120 is not compatible with the current PyTorch installation`。

**首个检查**：

```powershell
& $Py lab\scripts\probe_env.py --skip-hash
```

看两个字段：`compute_capability` 是否 `"12.0"`，`arch_list` 里有没有 `"sm_120"`。`probe_env.py` 对这个组合有专门的 warning。

**下一步**：装 cu128 wheel（PyPI 默认源的 wheel 不含 sm_120 kernel）：

```powershell
pip install --index-url https://download.pytorch.org/whl/cu128 "torch>=2.7"
```

复验 `cuda_matmul_ok: true`。注意 `$MINIMIND_ROOT/requirements.txt` 把 torch 注释成 `# torch==2.6.0`，**2.6.0 没有 sm_120 kernel**，在这台机器上以 cu128 wheel 为准。这条不解决就是 `BLOCKED`，不要往下走。

### F2 — 训练 OOM（对应 Day 2-4）

**症状**：`torch.OutOfMemoryError: CUDA out of memory`，通常在第 1 步或某次序列变长时。

**首个检查**：看最后一条 JSONL 的 `peak_mem_mb` 与 `tokens`，确认是稳定占用还是某步突增。

**下一步**（按顺序，一次只改一个）：

1. 降 `--batch-size`，同时按比例升 `--accumulation-steps` 保持有效 batch 不变（有效 batch = `batch_size × accumulation_steps`，改它会改 lr 的合理值，见 `01_FOUNDATIONS.md` 第 4.3 节）。
2. 降 `--max-seq-len`。attention 的中间量随 T 平方增长，这一项通常比 batch 更有效。
3. 换 `smoke_5070ti` 配置先跑通，再逐步往 `pretrain_5070ti` 靠。
4. 仍不行且是 GRPO，直接跳 F7。

**不要做的事**：不要为了塞下而改 `pretrain_5070ti.json` 的官方默认值——那样曲线就没法与原脚本对照了。要改就新建一档配置。

### F3 — bf16 与 fp16 选错（对应 Day 2）

**症状 A（fp16 特有）**：loss 出现 `nan`，或 JSONL 里 `scaler_scale` 一路减半（65536 → 32768 → 一直往下），`grad_norm` 为 `nan`。
**症状 B（bf16 特有）**：loss 不 nan，但下降明显比 fp32 慢或停滞。

**首个检查**：`probe_env.py` 的 `bf16_supported`。5070 Ti 是 `true`（`估算`，待实测）。然后看 JSONL 的 `scaler_scale`：bf16 与 float32 下它应该是 `null`；如果不是 `null`，说明 `--dtype` 传成了 `float16`。

**下一步**：

- `bf16_supported: true`（5070 Ti 的情况）→ 用 `--dtype bfloat16`。bf16 与 fp32 指数位相同，不会溢出，**不需要 GradScaler**，`make_scaler` 会以 `enabled=False` 构造它。这是本 lab 所有 `*_5070ti.json` 的默认。
- `bf16_supported: false`（老卡）→ 只能 `--dtype float16`，此时 `bounded_train` 自动启用 GradScaler。偶发减半是正常跳步；持续减半是真溢出，先切 `--dtype float32` 确认是精度问题而非 lr 问题，再回 fp16 并降 lr。
- 排查期间的黄金对照：`--dtype float32` 跑同样步数。fp32 正常而 fp16 不正常 = 精度问题；两者都不正常 = 问题不在精度。

### F4 — tokenizer 加载失败（对应 Day 0-1）

**症状**：`OSError: Can't load tokenizer for '<path>/model'`、`MiniMindNotFound`、或 `probe_env.py` 的 `tokenizer_loads` 不是布尔值而是一段以 `error: ` 开头的字符串（`probe_minimind` 把异常文本原样填进该字段）。

**首个检查**：`find_minimind_root` 要求三个文件同时存在：`model/model_minimind.py`、`model/tokenizer.json`、`dataset/lm_dataset.py`。手工确认：

```powershell
dir "$env:MINIMIND_ROOT\model\tokenizer.json", "$env:MINIMIND_ROOT\model\model_minimind.py", "$env:MINIMIND_ROOT\dataset\lm_dataset.py"
```

**下一步**（按可能性排序）：

1. `MINIMIND_ROOT` 指到了子目录或父目录 → 它必须是**仓库根**（`.git` 所在那层）。
2. 目录存在但文件缺失 → `git -C $env:MINIMIND_ROOT checkout 7a6fddd63a30c06b2fdd5fac4089922b29bc841b` 重新签出。`setup_minimind.*` 用了 `--filter=blob:none`，网络中断可能留下不完整的工作区。
3. 加载成功但 `tokenizer_loads: false`（不是 error 字符串）→ 这是**值断言**失败：`len(tok)` 不是 6400，或 bos/eos/pad 不是 1/2/0。说明 commit 不对或 `model/` 被换过，回到第 2 条。
4. `transformers` 未安装或版本不符 → `pip install transformers==4.57.6`。
5. Windows 上打印中文 token 时报 `UnicodeEncodeError` → 这不是加载失败，是控制台 GBK 编码。所有 CLI 入口都调了 `utf8_stdout()`；若你自己写脚本调用本包函数，先调它。

### F5 — label 全 -100 导致 loss 异常（对应 Day 1、3）

**症状 A**：`loss` 是 `nan`，`grad_norm` 是 `nan` 或 0，权重不变。
**症状 B**：`loss` 有限但停在 ln(6400)=8.76 附近永不下降。
**症状 C**：`loss` 几步内掉到 < 0.5，且生成只输出 `<|im_end|>`。

**首个检查**：看 JSONL 的 `n_label_tokens`。

```powershell
& $Py lab\scripts\mmp.py parse_log --input lab\runs\<run>\log_sft.jsonl --fields loss,n_label_tokens,grad_norm
```

- `n_label_tokens == 0` → 症状 A 确诊。`F.cross_entropy(..., ignore_index=-100)` 对 0 个有效元素取均值就是 `nan`。
- `n_label_tokens == tokens` → mask 完全没生效，所有非 pad token 都进了 loss（这正是 `user_in_loss` 故障的签名）。
- `n_label_tokens` 很小但非 0 → 症状 C，bos 序列匹配上的次数太少。

**下一步**：

1. 用 `inspect_dataset` 在同一份数据上逐 token 看：

   ```powershell
   & $Py lab\scripts\mmp.py inspect_dataset --stage sft --n 3
   ```

   看 `first_loss_pos` 是否落在 `<|im_start|>assistant\n`（5 个 token）之后、`eos_in_loss` 是否为 `true`。
2. `first_loss_pos` 为 `null` → 整条样本没有任何 assistant 段被识别。核对 `special_sequences(tokenizer)` 返回的 `bos_seq` 是不是 `[1, 1388, 570, 811, 234]`。不是就回 F4（tokenizer 不对）。
3. `eos_in_loss: false` → 对应症状"生成永不停止"（`01_FOUNDATIONS.md` 第 6 节第 4 条）：label 里必须含 id 2。用 `inspect_dataset --n 50 --no-table` 看 `aggregate.eos_in_loss_ratio`，它应为 1.0。低于 1.0 说明部分样本在 `<|im_end|>` 之前就被 `max_seq_len` 截断了 → 升 `--max-seq-len` 或过滤超长样本。
4. 确认是你自己注入的故障 → 检查命令行有没有 `--mask-fault`。`assistant_all_ignored` 下 loss = `nan`、退出码 2 **是预期症状**，不是 bug。
5. DPO 版的同一问题：`reward_margin` 恒为 0 且 `n_label_tokens`（= mask 之和）为 0。用 `dpo_check --n 2` 看 `mask_tokens_chosen` / `mask_tokens_rejected`，它们必须 > 0。

### F6 — GRPO advantage 全 0（对应 Day 5）

**症状**：日志里 `Adv Std` 长期 ≈ 0，`Reward` 不动，`Actor Loss` ≈ 0，模型几乎不更新（只剩 KL 项在起作用）。

**首个检查**：

```powershell
& $Py lab\scripts\mmp.py grpo_stats --input lab\runs\grpo_smoke_<r>\grpo_stdout.log --format debug-log
```

看 `zero_std_group_ratio`。advantage 的定义是 `(r - mean) / (std + 1e-4)`：组内 reward 全相等时 `std = 0`，分子也是 0，**整组 advantage 精确为 0**，这一组对梯度没有任何贡献。

**下一步**（按代价从低到高）：

1. `zero_std_group_ratio > 0.5` → 升 `-NumGenerations`（`num_generations`）。G=2 时两个 generation 拿到相同分数的概率很高；G=4 显著降低，代价是生成时间与显存线性上升。
2. 用 `grpo_groups.csv` 逐组看 reward 向量。如果 reward 只取少数几个离散值（规则 reward 的典型表现，值域 [-3, 3] 且只有几个整数组合），那是 **reward 分辨率不够**，不是采样问题 → 换回 `-Reward internlm`，或接受它并明确记为"规则 reward 下的管线验证"。
3. 检查采样是否退化：`avg_len` 全组相同、生成文本完全一致 → 采样温度或 `do_sample` 有问题，不同 generation 实际是同一条。
4. 三条都排除后仍全 0，本周判 `INCONCLUSIVE` 并记录 `zero_std_group_ratio` 的实测值，不要盲目调参消耗预算。

### F7 — GRPO 显存爆炸（左 padding 落到手写 attention 分支）（对应 Day 5）

**症状**：GRPO 在训练前向（不是生成阶段）OOM，或显存占用比同 batch/seq 的 SFT 高一个数量级；`batch_size=1` 时反而正常，`batch_size>1` 就炸。

**根因（下面两段是固定 commit 的源码摘录，为聚焦分支条件省去了与本问题无关的行）**：`model/model_minimind.py` 的 `Attention.forward` 只有满足全部条件才走高效的 `F.scaled_dot_product_attention`：

```python
if self.flash and (seq_len > 1) and (not self.is_causal or past_key_value is None) \
        and (attention_mask is None or torch.all(attention_mask == 1)):
    output = F.scaled_dot_product_attention(xq, xk, xv, ..., is_causal=self.is_causal)
else:
    scores = (xq @ xk.transpose(-2, -1)) / math.sqrt(self.head_dim)
    ...
    output = self.attn_dropout(F.softmax(scores.float(), dim=-1).type_as(xq)) @ xv
```

而 `trainer/train_grpo.py` 对 prompt 批量编码时用的是：

```python
prompt_inputs = tokenizer(prompts, return_tensors="pt", padding=True, return_token_type_ids=False,
                          padding_side="left", add_special_tokens=False).to(args.device)
```

`padding=True` + `padding_side="left"`：只要同一批里的 prompt **长度不等**，`attention_mask` 就含 0，`torch.all(attention_mask == 1)` 为假，于是掉进 `else` 分支。该分支显式构造 `[B, n_heads, T, T]` 的 `scores`，并且 `F.softmax(scores.float(), ...)` 把它升到 **fp32**——SDPA 本来是不落盘这个矩阵的。这就是显存突增一个数量级的来源。`batch_size=1` 时所有 prompt 等长（只有一条），`attention_mask` 全 1，反而走了高效分支——这解释了"batch 1 正常、batch>1 就炸"。

**首个检查**：确认是不是这条路径——固定 `--batch_size 1` 重跑。1 正常而 >1 炸，基本可以确诊。

**下一步**：

1. 保持 `--batch_size 1`（`run_grpo_smoke.*` 的默认就是 1，正是为此）。这是本周推荐的走法。
2. 降 `--max_seq_len` 与 `--max_gen_len`（默认脚本用 256/256，远低于官方默认的 1024）。`scores` 是 T 的平方项，这一项最有效。
3. 需要 batch > 1 时，让同批 prompt 长度对齐（按长度分桶），使 `attention_mask` 全 1。这属于改 MiniMind 训练脚本，超出本周范围，记入下周议题。
4. 把 `peak_mem_mb` 的实测值写进证据——本周对 GRPO 显存**没有任何实测数字**，这一条是本周最有价值的产出之一。

### F8 — reward 模型放不下（对应 Day 5）

**症状**：加载 `internlm/internlm2-1_8b-reward` 时 OOM；或加载成功但一开始训练就 OOM（policy + ref + reward 三个模型同驻，见 `01_FOUNDATIONS.md` 第 5.2 节）；或报 `trust_remote_code` 相关错误。

**首个检查**：确认模型路径存在且完整（约 3.4 GB safetensors）。`run_grpo_smoke.*` 默认找 `$MINIMIND_ROOT` 的同级目录 `internlm2-1_8b-reward`，可用 `-RewardModelPath` / `REWARD_MODEL_PATH` 覆盖。

**下一步**：

1. 直接切规则 reward，**不需要下载任何模型**：

   ```powershell
   .\lab\scripts\run_grpo_smoke.ps1 -Reward rule -NumGenerations 2 -MaxMinutes 15
   ```

   脚本会先调 `patch_grpo_rule_reward.py` 生成 `$MINIMIND_ROOT/trainer/train_grpo_rule.py`。该脚本**不修改**原 `train_grpo.py`，只复制一份并替换 `reward_model = LMForRewardModel(...)` 那一行为 `RuleRewardModel(...)`；幂等，重复运行结果相同。若它报"未找到预期的 reward_model 实例化行"，说明 commit 不是锁定值，回 F4 第 2 条。
2. 明确记录代价：`rule_reward.py` 是四条手写规则的加总，值域 [-3, 3]。它能让管线跑通并产生非零 advantage，但 **reward 上升不代表生成质量上升**。用它得到的结论只能写成"GRPO 管线在规则 reward 下可运行"，不能写成"GRPO 提升了模型"。
3. 想保留真 reward 模型 → 只能降 policy 侧占用（`--batch_size 1`、`--max_gen_len` 更短），或把 reward 模型的 dtype 降到 fp16（原脚本已是 `dtype=torch.float16`）。16 GB 够不够本周无实测，跑出结果就记数字，跑不出就判 `INCONCLUSIVE`。

### F9 — resume 后 loss 跳变（对应 Day 5）

**症状**：从 checkpoint 恢复后，loss 曲线出现明显台阶（跳高或跳低），而不是平滑接续。

**首个检查**：先分清是**机制坏了**还是**精度噪声**。

```powershell
& $Py lab\scripts\mmp.py bounded_train --config tiny_cpu --stage sft --max-steps 4 --save-every 2 --save-dir lab\runs\res_a --quiet
& $Py lab\scripts\mmp.py bounded_train --config tiny_cpu --stage sft --max-steps 4 --resume lab\runs\res_a\ckpt_step2.pt --save-dir lab\runs\res_b
```

CPU + float32 下，step 3/4 的 loss 必须与连续训练**逐位相等**（`tests/test_bounded_train.py::test_save_resume_equivalence` 断言 `abs=1e-6`，本机已通过）。这里不相等 = 机制坏了；这里相等而 GPU 上跳变 = 精度/非确定性。

**下一步（机制坏了）**：checkpoint 必须包含七样东西，缺一样都会跳变。逐个核对：

```powershell
& $Py -c "import torch; ck = torch.load(r'lab\runs\res_a\ckpt_step2.pt', map_location='cpu', weights_only=False); print(sorted(ck.keys())); print(ck['step'], ck['cursor'])"
```

必须有 `model`、`optimizer`、`scheduler`、`scaler`、`rng`、`step`、`cursor`。最常见的三个漏项及其签名：

| 漏了什么 | 跳变的样子 |
| --- | --- |
| `optimizer`（Adam 的一阶/二阶动量） | 恢复后头几十步 loss 明显变差再慢慢恢复 |
| `cursor`（数据游标） | 恢复后 `sample_indices` 与连续训练不同，样本被重复或跳过 |
| `scheduler`（这里是 `total_steps` 与 `base_lr`） | `lr` 字段直接跳。注意 `--resume` 时 `--max-steps` **会覆盖** `scheduler.total_steps`：恢复时传的 `--max-steps` 必须与原始运行相同，否则整条 cos 曲线被重新拉伸，这是最容易踩的一个 |

**下一步（GPU 上的精度噪声）**：不要试图追求 bitwise 相等。原因（`01_FOUNDATIONS.md` 第 6 节第 12 条）：MiniMind 原格式 checkpoint 存的是 `half()` 后的权重（fp32 master 被舍入到 fp16）；SDPA 的反向本身非确定；数据增广用 `random.random()`（system prompt 20%、空思考块 80%），跳过 batch 后 RNG 流不同。判据改成：恢复后 20 步的 loss 均值 vs 中断前 20 步的均值，差落在同期波动范围内。超出范围再回上面的机制检查。

### F10 — pytest 报 `PermissionError` 指向临时目录（对应 Day 0-1，环境问题）

**症状**：`pytest lab/tests` 里所有用了 `tmp_path` fixture 的用例集体 ERROR，栈底是 `PermissionError: [WinError 5]` + `os.scandir` + 路径 `C:\Users\<user>\AppData\Local\Temp\pytest-of-<user>`。纯函数用例正常通过。

**首个检查**：这是 pytest 默认临时目录不可写，**不是 lab 代码问题**。确认方法是给一个你有权限的目录：

```powershell
& $Py -m pytest lab\tests -q --basetemp=D:\tmp\pt
```

**下一步**：通过就是环境问题，用 `--basetemp` 长期绕开（或清理/接管那个残留目录）。仍失败才去看具体用例。本机 2026-09-04 就命中过这条：默认目录下 11 个 ERROR，`--basetemp` 指到可写目录后 `42 passed`（`本机实测`，详见第 7 节）。

### F11 — 吞吐远低于预期（对应 Day 2，`01_FOUNDATIONS.md` 第 6 节第 13 条）

**症状**：`tokens_per_s` 远低于同配置预期，GPU 利用率低。

**首个检查**：看 `tokens_per_s` 是"一直低"还是"抖动"。抖动通常是数据加载在等。

**下一步**：MiniMind 的 DataLoader 在线 tokenize，Windows 上 worker 的启动与序列化开销显著。`run_*` 脚本的 `epoch` 模式用 `--num_workers 4`，`run_grpo_smoke.*` 用 `--num_workers 0`。做 0 / 2 / 4 三档对照，取实测最快的。注意 `bounded_train` 不用 DataLoader（按游标直接索引 dataset），所以它的吞吐与原脚本不可直接比较。

---

## 6. 证据清单

本周需提交的字段。每条注明来源文件与产生它的命令。缺字段就是没过门。

### 6.1 环境与来源（Day 0）

| 字段 | 来源 |
| --- | --- |
| `python`、`torch`、`torch_cuda_version`、`transformers`、`datasets` | `lab/runs/probe.json`（`probe_env.py`） |
| `gpu_name`、`compute_capability`、`total_mem_mb`、`arch_list`、`bf16_supported` | 同上 |
| `cuda_matmul_ok`（+ 失败时 `cuda_matmul_error`） | 同上。这是 sm_120 的**决定性**字段 |
| `minimind_commit`、`commit_matches` | 同上。必须等于 `7a6fddd63a30c06b2fdd5fac4089922b29bc841b` |
| `tokenizer_loads` | 同上。必须 `true` |
| 4 个数据文件的 `exists` / `size` / `size_matches_card` / `sha256` | 同上（不加 `--skip-hash` 时才有 sha256） |
| `warnings` 全文 | 同上。空列表也要提交 |

### 6.2 数据因果链（Day 1）

| 字段 | 来源 |
| --- | --- |
| 一条**新的**多轮 chat 样本的手写 token/label 序列 | 你自己写，然后用 `inspect_dataset` 核对 |
| `n_nonpad`、`n_in_loss`、`loss_token_ratio`、`first_loss_pos`、`eos_in_loss`、`per_segment` | `inspect_dataset --out <json>` 的 `samples[]` |
| `mean_loss_token_ratio`、`truncation_ratio`、`eos_in_loss_ratio`、`mean_raw_len` | 同上的 `aggregate` |
| pretrain 与 sft 两套规则各一份 | 同上，`--stage` 各跑一次 |

### 6.3 训练（Day 2-4）

| 字段 | 来源 |
| --- | --- |
| 每步 `step`、`loss`、`lr`、`grad_norm`、`scaler_scale`、`tokens_per_s`、`peak_mem_mb`、`n_label_tokens` | `runs/<config>_<stage>/log_<stage>.jsonl` |
| loss / lr / grad_norm 曲线 | `runs/<config>_<stage>/log_<stage>.csv` + `.png`（`parse_log`） |
| `peak_mem_mb` 的实测峰值（三档配置各一个） | 同上。用它替换第 0.3 节的 `估算` |
| overfit 128 样本的 loss 下降幅度 | `run_pretrain_bounded.* -OverfitN 128` 的日志首末 loss |
| 正常 vs `user_in_loss` 的 loss 曲线并排 | `runs/sft_<cfg>_none/` 与 `runs/sft_<cfg>_user_in_loss/` |
| 正常 vs `user_in_loss` 的生成对照（含 `identical` 标记） | `eval_generate --compare` 的输出 |
| `assistant_all_ignored` 下的 `loss=nan` 与 `n_label_tokens=0` | `runs/sft_<cfg>_assistant_all_ignored/log_sft.jsonl` |
| 一次 mask 错位 debug 的完整推理链（症状 → 检查 → 定位） | 你自己写，引用上面三行的具体数字 |

### 6.4 DPO（Day 4）

| 字段 | 来源 |
| --- | --- |
| 训练前的 `dpo_loss` 与 `ln2_reference` | `runs/dpo_<cfg>/dpo_check_before.json` |
| `ref_frozen_check.ref_unchanged`（须 `true`）、`policy_changed`（须 `true`） | 同上 |
| 训练后的 `reward_margin`、`logits_dpo`、`accuracy`、`dpo_loss` | `runs/dpo_<cfg>/dpo_check_after.json` |
| 每步 `reward_margin`、`dpo_accuracy` 曲线 | `runs/dpo_<cfg>/log_dpo.jsonl` + `.csv` |
| `mask_tokens_chosen` / `mask_tokens_rejected`（须 > 0） | `dpo_check_*.json` |

### 6.5 GRPO 与 eval（Day 5）

| 字段 | 来源 |
| --- | --- |
| `zero_std_group_ratio`、`mean_group_std`、`mean_reward`、`mean_len` | `runs/grpo_smoke_<r>/grpo_stats.json` |
| 逐步 `adv_std`、`zero_std_groups`、`avg_len` | `runs/grpo_smoke_<r>/grpo_groups.csv` |
| `reward`、`kl_ref`、`adv_std`、`actor_loss`、`avg_response_len` 曲线 | `runs/grpo_smoke_<r>/grpo_steps.csv` + `.png` |
| 用的是 `internlm` 还是 `rule` reward，以及为什么 | 你自己写。用 `rule` 时必须写明"reward 不代表质量" |
| GRPO 的 `peak_mem_mb` 或 OOM 现场（含 batch/seq 参数） | GPU 日志。OOM 也是有效证据 |
| 三个阶段（pretrain / sft / dpo）的固定 8 prompt 生成 | `runs/eval_*.jsonl`，每条含 `n_new_tokens`、`ended_with_eos`、`output_text` |
| resume 前后各 20 步的 loss 均值与差值 | 中断前后两段 `log_*.jsonl` |

### 6.6 本机验证（贯穿全周）

| 字段 | 来源 |
| --- | --- |
| `pytest lab/tests -q` 的原文结尾行 | 见第 7 节 |
| `py_compile` 覆盖的 `.py` 数量与结果 | 见第 7 节 |
| 跳过的测试条数与原因 | 见第 7 节 |
| 两个日期的证据（Gate 要求） | 至少两次不同日期的运行 |

---

## 7. 本机验证状态（诚实记录）

验证时间：2026-09-04。机器：Windows 11（10.0.26200），**无 GPU**。解释器：`D:/Software/Large/Anconda/envs/ResearchAgentPy310/python.exe`（Python 3.10.20，torch 2.14.0+cpu，transformers 4.57.6，datasets 3.6.0，numpy 2.2.6，pytest 9.1.1，matplotlib 3.10.9，huggingface_hub 0.36.2）。

MiniMind 克隆位置：会话 scratchpad 下的 `minimind`，`git rev-parse HEAD` = `7a6fddd63a30c06b2fdd5fac4089922b29bc841b`（等于锁定值）。**未下载任何数据文件**，4 个 jsonl 全部不存在。

### 7.1 pytest

命令（在 `week01_minimind_5070ti/` 下，`MINIMIND_ROOT` 指向上述克隆）：

```text
"D:/Software/Large/Anconda/envs/ResearchAgentPy310/python.exe" -m pytest lab/tests -q --basetemp=<可写目录>
```

原文结果：

```text
..........................................                               [100%]
42 passed in 23.17s
```

**0 failed，0 skipped，0 error。** 42 条按文件分布：`test_label_mask.py` 11、`test_mask_fault.py` 7、`test_dpo_math.py` 6、`test_parse_log.py` 6、`test_bounded_train.py` 5、`test_grpo_stats.py` 5、`test_env.py` 2。

**必须记录的环境坑**：第一次不带 `--basetemp` 运行时，结果是

```text
31 passed, 11 errors in 17.38s
```

11 个 error 全部是同一个根因：`PermissionError: [WinError 5]` 发生在 `os.scandir('C:\Users\13289\AppData\Local\Temp\pytest-of-13289')`，即 pytest 创建 `tmp_path` 时读不了自己的默认临时目录根。手工 `ls` 该目录同样 `Permission denied`。**这是本机环境权限问题，不是 lab 代码问题**：受影响的恰好是全部使用 `tmp_path` fixture 的用例，指定可写 `--basetemp` 后 42 条全过。已写进第 5 节 F10 与 `lab/README.md` 的 smoke 步骤②。

### 7.2 不设 `MINIMIND_ROOT` 时的 skip

```text
sssss.......s.........sssssss.............                               [100%]
29 passed, 13 skipped in 13.54s
```

13 条 skip 全部来自 `conftest.py` 的 `minimind_root` fixture，skip 原因文本统一为 `需要 MINIMIND_ROOT：MINIMIND_ROOT 未设置。先运行 lab/scripts/setup_minimind.(ps1|sh) 克隆 MiniMind`。分布：

| 文件 | skip 条数 | 为什么必须有 MiniMind |
| --- | --- | --- |
| `test_bounded_train.py` | 5（全部） | 需要 `MiniMindForCausalLM` 与 `PretrainDataset`/`SFTDataset`/`DPODataset` |
| `test_label_mask.py` | 7（11 条中的 7 条） | 需要真 tokenizer（`test_fixture_sft_mask_semantics` 参数化 3 条 + 另 4 条） |
| `test_env.py` | 1（2 条中的 1 条） | `test_probe_reports_commit_when_root_set` 要读 git HEAD |

剩下 29 条是纯函数用例（合成 id 的 label 规则、DPO 数学、GRPO advantage、日志解析、故障注入），不需要 MiniMind 也不需要 GPU。

**没有任何测试因为缺少 GPU 而 skip**——本 lab 的测试全部设计成 CPU 可跑。

### 7.3 py_compile

```text
"D:/Software/Large/Anconda/envs/ResearchAgentPy310/python.exe" -m py_compile $(find lab -name "*.py" | sort | tr '\n' ' ')
```

覆盖 `lab/` 下全部 **23 个 `.py`**（`scripts/` 3 个、`src/mm_probe/` 12 个、`tests/` 8 个含 `conftest.py`）。结果：**全部通过，无输出，退出码 0。**

另外单独验证了 `patch_grpo_rule_reward.py` 生成的产物：`$MINIMIND_ROOT/trainer/train_grpo_rule.py` 通过 `py_compile`（该文件在原脚本基础上插入 `sys.path.insert` 与 `RuleRewardModel` 导入，原脚本第 2 行已 `import sys`，所以插入的代码有效）。

### 7.4 本机实际执行过的 CLI（全部成功）

| 命令 | 结果 |
| --- | --- |
| `probe_env.py --skip-hash` | 退出码 0；`commit_matches: true`、`tokenizer_loads: true`、`cuda_available: false`、`arch_list: []`；`warnings` 4 条（4 个数据文件缺失） |
| `mmp.py --help` | 列出 7 个模块 |
| `mmp.py inspect_dataset --stage sft --fixture --n 1 --max-seq-len 64` | 逐 token 表 + `n_nonpad: 21, n_in_loss: 9, first_loss_pos: 12, eos_in_loss: true` |
| `mmp.py bounded_train --config tiny_cpu --stage sft --max-steps 3` | 3 步 loss 8.7434 / 8.7481 / 8.6395，`n_params: 532928`，`peak_mem` 为 `-` |
| `mmp.py bounded_train --config tiny_cpu --stage sft --max-steps 4 --save-every 2`，再以 `--resume <save-dir>/ckpt_step2.pt` 重跑 | 正确从 step 2 恢复（打印 `cursor={'epoch': 1, 'pos': 1, 'n': 3, 'batch_size': 2, 'seed': 42, 'overfit_n': 0}`），只跑 step 3 与 step 4 |
| `mmp.py parse_log --input <jsonl> --csv <csv>` | 写出 3 行 16 列 CSV + ASCII sparkline 摘要 |
| `mmp.py dpo_check --config tiny_cpu --n 2` | `dpo_loss: 0.6931471824645996` vs `ln2_reference: 0.6931471805599453`，`reward_margin: [0.0, 0.0]` |
| `mmp.py mask_fault --mode user_in_loss --index 1 --max-seq-len 48` | `in-loss tokens: normal=6  faulty=40` |
| `mmp.py grpo_stats --input <合成 jsonl>` | `zero_std_group_ratio: 0.3333`、`mean_group_std: 0.5`、逐步表正确 |
| `mmp.py eval_generate --config tiny_cpu --max-new-tokens 4` | 8 条 prompt 全部生成，写出 JSONL（随机权重下输出无意义，只验证管线） |
| `patch_grpo_rule_reward.py` | 写出 `train_grpo_rule.py`，产物可 `py_compile` |

### 7.5 本机**没有**验证、必须等 5070 Ti 的门

| 门 | 为什么本机做不了 | 需要什么才算通过 |
| --- | --- | --- |
| sm_120 kernel 可用性（第 5 节 F1） | 本机 torch 是 `+cpu` 构建，`get_arch_list()` 返回 `[]` | `probe_env.json` 里 `arch_list` 含 `sm_120` 且 `cuda_matmul_ok: true` |
| bf16 行为（F3） | 无 GPU，`is_bf16_supported()` 返回 `null` | `bf16_supported: true` 且 bf16 下 20 步 loss 有限 |
| fp16 + GradScaler 的真实跳步 | CPU 上 float16 路径虽被 `test_pretrain_overfit_float16_scaler` 覆盖（断言 `scaler_scale > 0`），但**没有真实的溢出与跳步** | GPU 上 fp16 训练里观察到 `scaler_scale` 偶发减半后恢复 |
| 显存峰值（第 0.3 节的三个 `估算`） | `peak_memory_mb` 无 CUDA 时恒返回 `None` | 三档配置各拿到一个 `peak_mem_mb` 实测值 |
| 吞吐 tokens/s（F11） | 本机 4200-7700 tok/s 是 tiny_cpu（53 万参数）的数字，**与 64M 模型在 GPU 上的吞吐无关**，不可外推 | GPU 上稳态 `tokens_per_s` |
| 有界 pretrain / SFT / DPO 的真实曲线 | 需要 GB 级数据与 GPU | 门 5 的全部产物 |
| overfit 128 样本的收敛 | 同上 | loss 显著低于 8.76 |
| GRPO smoke 与显存（F7） | 需要 GPU；且本机未下载 `rlaif.jsonl` | 门 6 的产物，或一次带完整参数的 OOM 现场 |
| reward 模型能否放进 16 GB（F8） | 未下载该模型，无 GPU | 实测数字，或明确判 `INCONCLUSIVE` |
| GPU 上 resume 的数值一致性（F9） | 本机只验证了 CPU+float32 的**逐位相等**；GPU+bf16 的非确定性无法在此复现 | 恢复前后 20 步 loss 均值差在噪声范围内 |
| 数据文件的 size 与 sha256（第 1.2 节） | 未下载 | `probe.json` 的 `size_matches_card` 全为 `true`，且 `SHA256SUMS.txt` 写出 |

### 7.6 本轮修复的实现问题

**没有。** 42 条测试在带 `--basetemp` 的首次完整运行中即全部通过，23 个 `.py` 全部 `py_compile` 通过，第 7.4 节列出的 11 条 CLI 全部按各自文档的行为成功执行。唯一遇到的失败是第 7.1 节记录的 pytest 临时目录 `PermissionError`，经确认为本机环境权限问题而非 lab 代码缺陷，因此未改动任何 `.py`，改为写进第 5 节 F10 与 `lab/README.md` 的 smoke 说明。
