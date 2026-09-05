# Week M01 · Day 0 — 锁死环境、代码 commit 与数据指纹

| 字段 | 值 |
| --- | --- |
| 主要产物 | `lab/runs/probe.json`（含 `commit_matches: true` 与 `tokenizer_loads: true`），配 `pytest lab/tests -q` 的结尾行 |
| 估时 | `60 分钟`（步骤估时之和） |
| 环境 | `个人 5070 Ti`（步骤 1–2 的 GPU 字段需要它）+ `CPU 即可`（步骤 3–5 本机可完成） |
| AI 辅助等级要求 | `A2`（命令照抄，但 `probe.json` 每个字段的含义必须自己读出来并复述） |
| 前置 | 本周第一张卡，无证据前置。需要：能访问 GitHub 与 Hugging Face 的网络、约 4 GB 空闲磁盘 |
| 本卡对应门 | `静态检查`（`02_LAB_GUIDE.md` 的门 0 与门 1） |
| 可裁剪项 | 步骤 3（全量 sha256）可推迟到 Day 2 之前补齐；步骤 1、2、4 不可裁 |

## 为什么做

本周所有数字（loss 起点、显存峰值、label 位置）只有在"代码 commit 固定、tokenizer 固定、数据文件固定"的前提下才可比较。commit 不锁死，Day 1 手算的 token/label 序列就无法判定对错；数据指纹不记，Day 2 的 loss 曲线和别人的曲线不构成同一实验。5070 Ti 是 sm_120，PyTorch wheel 选错时 CUDA 矩阵乘会直接失败，这一条必须在跑任何训练之前查出来。

## 步骤

### 1. 克隆 MiniMind 并锁定 commit（20 分钟）· `模板`

先把 `D:\work\minimind` 换成你自己的克隆目录。

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File lab\scripts\setup_minimind.ps1 -Root D:\work\minimind -Python python
$env:MINIMIND_ROOT = "D:\work\minimind"
```

bash：

```bash
ROOT=$HOME/minimind PYTHON=python bash lab/scripts/setup_minimind.sh
export MINIMIND_ROOT=$HOME/minimind
```

- 预期：打印 `[setup] MiniMind at 7a6fddd63a30c06b2fdd5fac4089922b29bc841b`，随后每个数据文件一行 `sha256=<64 hex> size=<bytes>`，并写出 `<clone-dir>/dataset/SHA256SUMS.txt`。4 个 jsonl 合计约 3 GB，耗时 `估算` 10–40 分钟，取决于带宽。
- 带宽或磁盘不够时先做：加 `-SkipData`（bash 用 `--skip-data`）只拉代码，约 1 分钟；本卡其余步骤全部照常，数据留到 Day 2 之前补。
- **想一次把全部数据集和奖励模型下齐**（27.00 GB，适合整夜挂着）：改用 `lab\scripts\download_datasets.py --overnight`（或双击周目录下的 `下载数据集.bat`），用法、层级、续传与校验见 [`06_DATASET_DOWNLOAD.md`](../06_DATASET_DOWNLOAD.md)。它与 `setup_minimind` 不冲突：前者管代码与 commit，后者管数据；两边都按精确字节数校验。
- 不对时先查：`git -C D:\work\minimind rev-parse HEAD` 的输出是否逐字等于 `7a6fddd63a30c06b2fdd5fac4089922b29bc841b`。不等时脚本会抛错退出，说明 checkout 没落到锁定 commit。

### 2. 跑探针拿 `probe.json`（10 分钟）· `可直接执行`

```powershell
$Py = "D:/Software/Large/Anconda/envs/ResearchAgentPy310/python.exe"
& $Py lab\scripts\probe_env.py --skip-hash
```

在 5070 Ti 那台机器上，把 `$Py` 换成该机器上装了 cu128 torch 的解释器，并 `$env:MM_PYTHON = $Py`。

- 预期（5070 Ti，`估算`）：`compute_capability: "12.0"`、`arch_list` 含 `"sm_120"`、`bf16_supported: true`、`cuda_matmul_ok: true`、`total_mem_mb` 约 16000、`warnings: []`。
- 预期（本机 CPU，`本机实测`）：`cuda_available: false`，`gpu_name` / `compute_capability` / `bf16_supported` 均为 `null`，`arch_list: []`，`commit_matches: true`，`tokenizer_loads: true`；未下载数据时 `warnings` 恰好 4 条数据缺失，退出码 0。
- 没有 GPU 时先做：在本机跑同一条命令。GPU 三个字段为 `null` 是预期结果，本步在本机能拿到的过门字段是 `commit_matches` 与 `tokenizer_loads`；`cuda_matmul_ok` 等上 5070 Ti 再补。
- 不对时先查：`cuda_matmul_ok: false` 且 `cuda_matmul_error` 里含 `no kernel image is available` → wheel 里没有 sm_120 kernel，按 `02_LAB_GUIDE.md` 第 5 节 F1 重装 cu128 的 `torch>=2.7`；`tokenizer_loads: false` 见同节 F4。

### 3. 记录 4 个数据文件的字节数与 sha256（10 分钟）· `可直接执行`

```powershell
& $Py lab\scripts\probe_env.py --out lab\runs\probe_hash.json
```

- 预期：不加 `--skip-hash` 时每个数据文件多出 `sha256` 字段，耗时 `估算` 1–3 分钟；`pretrain_t2t_mini.jsonl` 的 `size` 为 1241043656、`sft_t2t_mini.jsonl` 为 1739201170、`dpo.jsonl` 为 53653322、`rlaif.jsonl` 为 23754740，四个 `size_matches_card` 全为 `true`。
- 不对时先查：`size_matches_card: false` 时先看 `exists`。文件存在但字节数不符，就把 `probe_hash.json` 里的 `sha256` 与 `<clone-dir>/dataset/SHA256SUMS.txt`（步骤 1 首次下载当天写下的值）逐字对比，两者不一致说明下载被截断，重跑步骤 1。

### 4. 跑 CPU 单测（10 分钟）· `可直接执行`

```powershell
& $Py -m pytest lab\tests -q --basetemp=D:\tmp\pt
```

- 预期（`本机实测`）：`42 passed`。若没设 `MINIMIND_ROOT`，预期 `29 passed, 13 skipped`，跳过的都是需要 tokenizer 或 MiniMind 模型类的用例。
- 不对时先查：若出现 `PermissionError: [WinError 5]` 且路径指向 `AppData\Local\Temp\pytest-of-<user>`，那是 pytest 默认临时目录不可写，不是 lab 代码问题；`--basetemp` 必须指到一个你有写权限的目录（见 `02_LAB_GUIDE.md` 第 5 节 F10，本机 2026-09-04 命中过这条）。

### 5. 单 batch 前反向（10 分钟）· `可直接执行`

```powershell
& $Py lab\scripts\mmp.py bounded_train --config tiny_cpu --stage sft --max-steps 1
```

- 预期（`本机实测`）：`[bounded:sft] step 1/1 loss 8.7434 lr 1.00e-05 grad_norm 3.2004 scale - tok/s 1047 peak_mem -`；summary 里 `n_params: 532928`、`device: "cpu"`、`dtype: "float32"`。loss 落在 ln(6400)=8.764 附近就是随机初始化的理论值。
- 不对时先查：loss 远离 8.76 或 `grad_norm` 为 0，先跑 `& $Py lab\scripts\mmp.py inspect_dataset --stage sft --fixture --n 1 --max-seq-len 64` 看 `first_loss_pos` 与 `n_in_loss` 是否非空（对应 `02_LAB_GUIDE.md` 第 5 节 F5）。

## 今日自测（闭卷，2 题，只记录能/不能）

1. `M01-R-03`：不查文档写出 MiniMind 默认配置的 12 个值与总参数量。
2. `M01-T-04`：bf16 与 fp16 的格式差别，以及为什么 bf16 下 `GradScaler(enabled=False)` 是安全的。

## 证据字段（填完即可归档到 memory/cc/evidence/）

```json
{
  "date": "YYYY-MM-DD",
  "week": 1,
  "day": 0,
  "card": "day0",
  "skill": "env_probe_source_lock",
  "env": "5070ti",
  "ai_level": "A2",
  "config_id": "minimind@7a6fddd + tiny_cpu.json",
  "primary_artifact": "lab/runs/probe.json",
  "observations": {
    "commit_matches": "true | false",
    "tokenizer_loads": "true | false",
    "cuda_matmul_ok": "true | false | 本机无 GPU 未测",
    "compute_capability": "12.0 | null",
    "bf16_supported": "true | false | null",
    "total_mem_mb": 0,
    "arch_list_has_sm120": "true | false",
    "warnings_count": 0,
    "data_size_matches_card": "4/4 | 见 probe_hash.json",
    "pytest_tail": "42 passed",
    "step1_loss": 8.7434
  },
  "status": "PASS | FAIL-SYSTEM | BLOCKED",
  "failure_class": "none | env | system",
  "self_check": {"q1": "能 | 不能", "q2": "能 | 不能"},
  "time_spent_min": 0,
  "notes": "一句话，不含敏感信息"
}
```
