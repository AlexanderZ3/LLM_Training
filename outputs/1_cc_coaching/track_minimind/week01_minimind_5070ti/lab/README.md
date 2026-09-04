# `mm_probe` — Week M01 实验包（MiniMind 单机全链路仪表）

## 这个 lab 是什么

一个**挂在 MiniMind 之上的仪表包**。它回答的问题是：一条 JSONL 样本怎么变成 token 与 label（谁进 loss、谁是 -100），走过 decoder-only 前向，再经 pretrain → SFT → DPO → GRPO 改变权重，而且每一环都能被你自己写的代码量出来。

具体做四件事：

1. **复现并断言 MiniMind 的数据规则**：`inspect_dataset.py` 逐字复现 `dataset/lm_dataset.py` 的 `PretrainDataset` / `SFTDataset` / `DPODataset` 编码与 label 规则，`tests/test_label_mask.py` 把结果与 MiniMind 原类逐位对比。
2. **补上原仓库没有的"有界"能力**：MiniMind 的 `trainer/train_*.py` 没有步数上限参数（只有 `--epochs`），CPU 上无法 3 步验证。`bounded_train.py` 用 MiniMind 的**原模型类与原 dataset 类**自写最小训练循环，加 `--max-steps` / `--overfit-n` / `--save-every` / `--resume` / `--mask-fault`。
3. **挂仪表**：每步 JSONL 记 step / loss / lr / grad_norm / scaler_scale / tokens_per_s / peak_mem_mb；`parse_log.py` 把它和 MiniMind 原脚本 stdout 都解析成 CSV 与曲线；`grpo_stats.py` 算组内 advantage 与 std=0 组占比；`dpo_check.py` 算序列 logp 与 reward margin。
4. **故障注入**：`mask_fault.py` 把 SFT 的 label mask 人为错位，让你只凭 loss 曲线与固定 prompt 生成反推故障点。

## 这个 lab 不是什么

- **不是 MiniMind 的副本。** 仓库里没有一行 MiniMind 源码。所有模型类、dataset 类、原训练脚本都通过环境变量 `MINIMIND_ROOT` 引用你自己的克隆，且锁定在 commit `7a6fddd63a30c06b2fdd5fac4089922b29bc841b`。`probe_env.py` 会核对 `git rev-parse HEAD` 是否等于这个值，不等就进 `warnings`。
- **不是 MiniMind 训练脚本的替代品。** `bounded_train.py` 只用于"能不能跑通、数值对不对、能不能恢复"这类有界验证。跑完整 epoch 请用 `run_*_bounded.*` 脚本的 `-Mode epoch` / `MODE=epoch` 分支，它调的是原仓库的 `trainer/train_*.py`。
- **不下载数据也能用。** 所有单测与 CPU dry run 走 `fixtures.py` 里的内置 toy 样本（4 条 pretrain、3 条 SFT、3 条 DPO、2 条 RLAIF），字段格式与官方数据一致。GB 级数据集只有真正做有界训练时才需要。

## 环境：两条路径

### 路径 A — 个人 5070 Ti 16 GB（有界训练主线）

`命令等级：模板`（先把 `<clone-dir>` 换成你的目录）

```powershell
# 1) torch：5070 Ti 是 sm_120，必须从 cu128 索引装 torch>=2.7，PyPI 默认 wheel 没有 sm_120 kernel
pip install --index-url https://download.pytorch.org/whl/cu128 "torch>=2.7"
pip install transformers==4.57.6 datasets==3.6.0 huggingface_hub matplotlib pytest

# 2) 克隆 MiniMind 到固定 commit 并下载 4 个 jsonl（约 3 GB，需要网络）
powershell -NoProfile -ExecutionPolicy Bypass -File lab\scripts\setup_minimind.ps1 -Root <clone-dir> -Python python

# 3) 只要代码不要数据时加 -SkipData
powershell -NoProfile -ExecutionPolicy Bypass -File lab\scripts\setup_minimind.ps1 -Root <clone-dir> -SkipData
```

装完立刻验：`arch_list` 必须含 `sm_120`，否则第一次 CUDA 矩阵乘会报 `no kernel image is available`。

```powershell
python lab\scripts\probe_env.py --skip-hash
```

`命令等级：可直接执行`（`MINIMIND_ROOT` 已设置时）。看两个字段：`cuda_matmul_ok` 为 `true`，`warnings` 里没有 sm_120 条目。

### 路径 B — 本机 CPU（课程构建与单测）

本机无 GPU，只跑单测、`py_compile` 和 `tiny_cpu` 配置的 dry run，**不做任何有界训练**。解释器固定用 conda env `ResearchAgentPy310`，禁止裸 `python`（那是 Windows 商店占位）。

`命令等级：可直接执行`

```powershell
$Py = "D:/Software/Large/Anconda/envs/ResearchAgentPy310/python.exe"
& $Py -c "import torch, transformers, datasets; print(torch.__version__, transformers.__version__, datasets.__version__)"
```

该环境 2026-09-04 实测已装：torch 2.14.0+cpu、transformers 4.57.6、datasets 3.6.0、numpy 2.2.6、pytest 9.1.1、matplotlib 3.10.9、huggingface_hub 0.36.2。版本清单见 `requirements.txt` 第二段。

这条路径**仍然需要 MiniMind 克隆**（模型类、dataset 类、tokenizer 都在里面），但**不需要数据文件**：42 条测试里 13 条依赖 `MINIMIND_ROOT`，0 条依赖 GB 级 jsonl。

## 设置 `MINIMIND_ROOT`

所有模块按 `--minimind-root` → 环境变量 `MINIMIND_ROOT` 的顺序解析。目录必须是 MiniMind 仓库根（`minimind_env.find_minimind_root` 会检查 `model/model_minimind.py`、`model/tokenizer.json`、`dataset/lm_dataset.py` 三个文件是否都在）。

PowerShell（当前会话）：

```powershell
$env:MINIMIND_ROOT = "D:\work\minimind"
```

PowerShell（持久化到用户环境变量，新开的终端才生效）：

```powershell
[Environment]::SetEnvironmentVariable("MINIMIND_ROOT", "D:\work\minimind", "User")
```

bash / WSL / Linux：

```bash
export MINIMIND_ROOT="$HOME/minimind"
```

不想设环境变量时，每条命令单独传：

```bash
python lab/scripts/mmp.py inspect_dataset --minimind-root /path/to/minimind --fixture
```

未设置且未传 `--minimind-root` 时，模块抛 `MiniMindNotFound` 并直接给出修复步骤；pytest 里则通过 `conftest.py` 的 `minimind_root` fixture 自动 skip 而不是失败。

## 5 分钟 smoke

在 `week01_minimind_5070ti/` 目录下执行。全部 `命令等级：可直接执行`（前提：`MINIMIND_ROOT` 已指向固定 commit 的克隆）。

```powershell
$Py = "D:/Software/Large/Anconda/envs/ResearchAgentPy310/python.exe"
```

**① 环境探针**（约 10 秒）

```powershell
& $Py lab\scripts\probe_env.py --skip-hash
```

无 GPU 时 `cuda_available` 为 `false`、`gpu_name` / `compute_capability` / `bf16_supported` 全为 `null`，退出码仍是 0。关键看 `commit_matches: true` 与 `tokenizer_loads: true`。数据文件没下载时 `warnings` 里会有 4 条"数据缺失"，这在 CPU 路径上是预期的。`--skip-hash` 跳过 GB 级文件的 sha256（不跳的话约 1-3 分钟）。

**② 单测**（约 25 秒）

```powershell
& $Py -m pytest lab\tests -q
```

预期 `42 passed`。若报 `PermissionError` 指向 `AppData\Local\Temp\pytest-of-<user>`，是 pytest 默认临时目录不可写，改用一个你有权限的目录：

```powershell
& $Py -m pytest lab\tests -q --basetemp=D:\tmp\pt
```

不设 `MINIMIND_ROOT` 时预期 `29 passed, 13 skipped`，跳过的都是需要 tokenizer 或 MiniMind 模型类的用例。

**③ 数据 → token/label 因果链（fixture 模式，不碰 GB 级数据）**（约 5 秒）

```powershell
& $Py lab\scripts\mmp.py inspect_dataset --stage sft --fixture --n 1 --max-seq-len 64
```

本机实测输出：逐 token 表里 idx 0-11 的 `<|im_start|>user\n你好<|im_end|>\n<|im_start|>assistant\n` 全部 label = -100，idx 12-18 的 assistant 内容 label = input_id，idx 19-20 的 `<|im_end|>` 与 `\n` 也进 loss；`stats` 为 `n_nonpad: 21, n_in_loss: 9, first_loss_pos: 12, eos_in_loss: true`。换 `--stage pretrain` 看另一套规则（bos 与 eos 都进 loss）。

**④ 有界训练 3 步（tiny_cpu）**（约 15 秒）

```powershell
& $Py lab\scripts\mmp.py bounded_train --config tiny_cpu --stage sft --max-steps 3
```

本机实测（随机初始化，`n_params: 532928`）：

```text
[bounded:sft] step 1/3 loss 8.7434 lr 7.75e-05 grad_norm 3.2004 scale - tok/s 4897 peak_mem -
[bounded:sft] step 2/3 loss 8.7481 lr 3.25e-05 grad_norm 4.5389 scale - tok/s 4227 peak_mem -
[bounded:sft] step 3/3 loss 8.6395 lr 1.00e-05 grad_norm 4.0422 scale - tok/s 7681 peak_mem -
```

loss 起点 8.74 就是 ln(6400)=8.764 附近，符合随机初始化；3 步太短，不要期待下降趋势。`scale` 与 `peak_mem` 为 `-` 是因为 float32（无 GradScaler）且无 CUDA。日志 JSONL 默认写到 `lab/runs/tiny_cpu_sft/log_sft.jsonl`。

**⑤ 把日志读成曲线**（可选，约 3 秒）

```powershell
& $Py lab\scripts\mmp.py parse_log --input lab\runs\tiny_cpu_sft\log_sft.jsonl --csv lab\runs\tiny_cpu_sft\log_sft.csv
```

没装 matplotlib 时 `--png` 自动降级为 ASCII sparkline，不报错。

## 目录说明

```text
lab/
├── README.md                 本文件
├── requirements.txt          两条安装路径的依赖（5070 Ti cu128 / 本机 CPU 实测版本）
├── configs/                  4 档配置，被 --config 按文件名或名字（如 tiny_cpu）加载
├── scripts/                  入口脚本
├── src/mm_probe/             可 import 的仪表包
└── tests/                    42 条 pytest
```

### `configs/`

| 文件 | 解决什么问题 |
| --- | --- |
| `tiny_cpu.json` | CPU 上 3 步跑通：hidden 64 / 2 层 / seq 64 / float32 / device cpu，数据全走 `fixture`。它不代表任何真实训练结果，只用来验证代码路径与数值不变量。 |
| `smoke_5070ti.json` | 5070 Ti 上"能不能跑"的第一道门：官方 64M 结构（768/8/8/4）但 seq 256、batch 8、bf16，估算显存 < 4 GB。 |
| `pretrain_5070ti.json` | 有界 pretrain：完全对齐 `train_pretrain.py` 官方默认（seq 340 / batch 32 / accum 8 / lr 5e-4 / bf16），这样曲线才能和原脚本对照。 |
| `sft_5070ti.json` | 有界 full SFT（seq 768 / lr 1e-5 / 物理 batch 8 × accum 2 = 官方有效 batch 16），并在 `stages.dpo` 段覆写 DPO 的官方默认（beta 0.15 / lr 4e-8 / batch 4 / seq 1024）。 |

### `scripts/`

| 文件 | 解决什么问题 |
| --- | --- |
| `setup_minimind.ps1` / `.sh` | 从零开始：克隆 MiniMind → `checkout` 锁定 commit（校验 HEAD）→ 用 `hf_hub_download` 拉 4 个 jsonl → 写 `dataset/SHA256SUMS.txt`。`-SkipData` / `--skip-data` 只要代码不要数据。 |
| `probe_env.py` | Day 0 的唯一产物 `probe.json`：torch/CUDA 版本、GPU 名、compute capability、`arch_list`、`is_bf16_supported`、SDPA 后端开关、一次真实 CUDA 矩阵乘（这才能抓到 sm_120 不被支持）、MiniMind commit 是否等于锁定值、4 个数据文件的存在性/大小/sha256。参数：`--out` `--minimind-root` `--skip-hash`。 |
| `mmp.py` | 统一入口，免设 `PYTHONPATH`。`python lab/scripts/mmp.py <module> [args]` 等价于 `PYTHONPATH=lab/src python -m mm_probe.<module> [args]`。`<module>` 取 `inspect_dataset`、`bounded_train`、`parse_log`、`mask_fault`、`dpo_check`、`grpo_stats`、`eval_generate`。 |
| `patch_grpo_rule_reward.py` | 16 GB 放不下 1.8B reward 模型时的出路：**不改**原 `train_grpo.py`，而是复制出 `$MINIMIND_ROOT/trainer/train_grpo_rule.py` 并只替换 `reward_model` 那一行为 `mm_probe.rule_reward.RuleRewardModel`。幂等，重复运行结果相同。 |
| `run_pretrain_bounded.ps1` / `.sh` | Day 2 的一条命令：有界 pretrain（`-Mode bounded`，默认）或原脚本整轮（`-Mode epoch`），结束自动 `parse_log` 出 CSV 与 PNG，并导出 `$MINIMIND_ROOT/out/pretrain_768.pth`。 |
| `run_sft_bounded.ps1` / `.sh` | Day 3：有界 full SFT，`-MaskFault` / `MASK_FAULT` 切正常与两种错位，各写独立 `runs/` 目录，结束都对 8 条固定 prompt 做贪心生成，供正常/错位并排对照。 |
| `run_dpo_bounded.ps1` / `.sh` | Day 4：训练**前后各做一次** `dpo_check`（前一次带 `--check-ref-frozen` 验证 ref 参数 hash 不变），中间跑有界 DPO，导出 `out/dpo_768.pth`。 |
| `run_grpo_smoke.ps1` / `.sh` | Day 5：GRPO smoke。原 `train_grpo.py` 没有步数上限，本脚本用 `-MaxMinutes` / `MAX_MINUTES` 超时终止（`--save_interval 5` 保证落盘）；`-Reward rule` 切规则 reward。结束用 `grpo_stats` 从 `--debug_mode` 日志里还原组内 reward。 |

### `src/mm_probe/`

| 文件 | 解决什么问题 |
| --- | --- |
| `__init__.py` | 声明包版本与 `PINNED_MINIMIND_COMMIT`，并写明哪些模块是纯函数（不需要 MiniMind）、哪些依赖 MiniMind。 |
| `minimind_env.py` | 所有"怎么找到 MiniMind"的逻辑集中在这里：`find_minimind_root`、`add_minimind_to_path`、`load_tokenizer`、`build_model`、`load_config`、`git_head`、`sha256_file`、`resolve_data_path`、`load_state_dict_any`（同时吃 MiniMind 的 `.pth` 与本包的 `.pt`）、`utf8_stdout`（Windows 控制台 GBK 打不出中文 token 的修复）。 |
| `fixtures.py` | 无网络、无 GB 级数据时的 toy 样本，字段格式与官方数据逐一对齐。`write_fixture_jsonl(stage, path)` 落盘给 MiniMind 的 dataset 类读。 |
| `inspect_dataset.py` | 数据 → tokenizer → token/label 的因果链检查。纯函数 `generate_labels` 逐字复现 MiniMind 的规则且不需要 tokenizer（可用合成 id 单测）；CLI 逐 token 打印 id/label/段落归属并给统计（`n_in_loss`、`first_loss_pos`、`eos_in_loss`、截断率）。把 MiniMind 的两个随机分支改成确定开关 `--add-system` / `--keep-empty-think`，否则没法对照。 |
| `bounded_train.py` | 原仓库缺的有界训练循环。`--max-steps` 让 CPU 3 步可跑；`--overfit-n` 固定前 N 条样本；`--dtype float16` 自动带 GradScaler；`--save-every` + `--resume` 保存并恢复 model/optimizer/scheduler/scaler/RNG/step/数据游标；`--mask-fault` 注入故障；`--export-pth` 导出 MiniMind 格式 half state_dict 供原脚本 `--from_weight` 接力。 |
| `hooks.py` | 三个仪表原语：`ThroughputMeter`（带 CUDA sync 的 tokens/s）、`peak_memory_mb`、`grad_global_norm`（与 `clip_grad_norm_` 同定义，在 clip **之前**取才是真实梯度大小）。纯 torch，不依赖 MiniMind。 |
| `parse_log.py` | 把日志变成能读的东西。同时吃 `bounded_train` 的 JSONL 和 MiniMind 三种 stdout 格式（pretrain/sft、dpo、grpo），字段名统一成小写下划线，输出 CSV + PNG；没有 matplotlib 时降级成 ASCII sparkline 而不是报错。 |
| `mask_fault.py` | Day 3 的故障注入。`apply_fault` 是纯函数（list 与 tensor 都吃），两种错位：`assistant_all_ignored`（全 -100，症状是 loss = nan）与 `user_in_loss`（所有非 pad token 进 loss，症状是 loss 起点偏低、生成复述 user）。 |
| `dpo_check.py` | DPO 的数值全在这里：`per_token_logps`、`sequence_logp`、`dpo_loss`、以及签名与 MiniMind `train_dpo.py` 完全一致的 `dpo_loss_minimind`。`param_hash` 让"ref 到底冻没冻住"变成可断言的事实。 |
| `grpo_stats.py` | GRPO 组内统计。`group_advantage` 用与 MiniMind 相同的 `(r - mean) / (pop_std + 1e-4)`；`summarize` 给出 std=0 组占比（这是"advantage 全 0、没有学习信号"的直接证据）；`parse_debug_log` 从 `--debug_mode` 的 stdout 里把每组 reward 抠出来。 |
| `eval_generate.py` | 8 条固定 prompt 的贪心生成，用于跨阶段与正常/错位对照。`--mode chat` 走 chat 模板，`--mode raw` 直接续写（适合 pretrain 权重）。`--compare A.jsonl B.jsonl` 并排打印两次运行并标 `identical`。 |
| `rule_reward.py` | 16 GB 放不下 1.8B reward 模型时的替代，接口与 MiniMind `LMForRewardModel` 相同（`get_score(messages, response)`）。分数是规则给的，只能用来跑通管线与观察 advantage 统计，**不能当模型质量信号**。 |

### `tests/`（42 条）

| 文件 | 条数 | 解决什么问题 |
| --- | --- | --- |
| `conftest.py` | — | 把 `lab/src` 加进 `sys.path`；`MINIMIND_ROOT` 缺失时让依赖 tokenizer/模型的用例 skip 而不是失败。 |
| `test_env.py` | 2 | `probe_env.py` 能跑通、退出码 0、`probe.json` 字段齐全；无 GPU 时 GPU 字段确实是 null；MiniMind HEAD 等于锁定 commit。 |
| `test_label_mask.py` | 11 | Day 1 的核心。前半用合成 id 断言纯规则（单轮、多轮、截断、无 assistant 三种情况）；后半用真 tokenizer 断言 fixture 样本的 user/system/pad 段必为 -100、assistant 段 label = input_id、`<|im_end|>\n` 归属正确，并与 MiniMind 原 `SFTDataset.generate_labels` / `PretrainDataset` **逐位**对比。 |
| `test_bounded_train.py` | 5 | tiny 配置 3 步 loss 有限且日志字段齐全；`--overfit-n 2` 确实只用前 2 条；float16 必须带 GradScaler；**save → resume 后下一步 loss 与连续训练逐位相等**（这是 resume 正确性的判据）；DPO 首步 loss = ln2；两种 mask 故障的症状。 |
| `test_dpo_math.py` | 6 | 手算小例子锁死 DPO 数学：`per_token_logps` 对 V=3 的解析值、`logits_dpo = 1` 时 loss = log(1+e^-0.15)、policy = ref 时 loss = ln2、`dpo_loss_minimind` 与 `dpo_loss` 等价、beta 单调性、`param_hash` 只在参数变化时改变。 |
| `test_mask_fault.py` | 7 | `apply_fault` 的 list 与 tensor 两种输入等价、不改原值、缺 `input_ids` 时报错、未知 mode 报错。 |
| `test_grpo_stats.py` | 5 | reward 全等 → advantage 全 0 且 `zero_std` 为真；与 torch 的 `std(unbiased=False)` 逐值一致；扁平输入缺 `num_generations` 时报错；`parse_debug_log` 能从一段真实格式的 GRPO stdout 里还原组结构。 |
| `test_parse_log.py` | 6 | MiniMind 三种 stdout 行格式各自解析正确、非日志行被忽略、CSV 往返、JSONL 自动识别。 |

## 已知限制

1. **本机无 GPU。** 所有 GPU 相关的门（bf16 前向、显存峰值、吞吐、sm_120 kernel 可用性）在本机**没有也无法**验证。`probe_env.py` 在本机输出 `cuda_available: false`、`arch_list: []`，`bounded_train` 的 `peak_mem_mb` 恒为 `null`。这些必须在 5070 Ti 上重跑，详见 `../02_LAB_GUIDE.md` 第 7 节。
2. **没有下载真实数据集。** 4 个 jsonl（合计约 3 GB）本机一个都没下。因此：`probe_env.py` 的 `data_files` 全部 `exists: false`；周卡记录的字节数与 sha256 **没有**在本机复核过，`setup_minimind.*` 写出的 `SHA256SUMS.txt` 是首次下载当天才产生的记录，不是预先已知的期望值。所有测试与 smoke 都走 `fixtures.py`，fixture 的字段格式已按固定 commit 的 `dataset/lm_dataset.py` 逐行核对，但**样本分布与真实数据无关**（例如 fixture 的截断率、平均长度不可外推）。
3. **GRPO 显存未实测。** 16 GB 上跑 GRPO + `internlm/internlm2-1_8b-reward`（约 3.4 GB safetensors）到底放不放得下，没有任何实测数字。`run_grpo_smoke.*` 的默认参数（batch 1 / `num_generations` 2 / seq 256 / gen_len 256）是按"最保守起步"选的，不是实测可行值。OOM 是预期结果之一，出路是 `-Reward rule` 走规则 reward。相关的显存爆炸机制（左 padding 让 attention 落到手写分支）见 `../02_LAB_GUIDE.md` 第 5 节。
4. **`bounded_train` 不等于 MiniMind 原脚本。** 它复用原模型类与原 dataset 类，lr 公式也逐字复现 `trainer_utils.get_lr`，但训练循环是本包自写的，没有 DDP、没有 wandb/swanlab、没有原脚本的 `_resume.pth` 格式。跨脚本接力只通过 `--export-pth` 导出的 half state_dict 进行。用它的曲线和原脚本曲线对照时，只有同配置同数据下才有可比性。
5. **`rule_reward.py` 不是 reward 模型。** 它是四条手写规则（长度区间、与问题的 2-gram 重叠、3-gram 重复率、空回答）的加总，值域 [-3, 3]。它能让 GRPO 管线跑起来并产生非零 advantage，但 reward 上升**不代表**生成质量上升。
