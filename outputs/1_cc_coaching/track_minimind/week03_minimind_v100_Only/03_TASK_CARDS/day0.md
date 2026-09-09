# Week M03 · Day 0 — 把 V100 的真实能力、MiniMind 的兼容性、数据路径、torchrun 一次钉死

| 字段 | 值 |
| --- | --- |
| 主要产物 | `$MM_RUNS_ROOT/day0_probe/evidence.json`（由 `probe.json` + `compat_audit.md` + `wiring_smoke.json` 聚合而成） |
| 估时 | `100 分钟`（步骤估时之和） |
| 环境 | `公司 V100`（步骤 2 在本机 CPU 上做） |
| AI 辅助等级要求 | `A2`；步骤 5 判断"停在哪一档、为什么" 要求 `A1`（只查官方文档） |
| 前置 | 无。本周自包含。数据已按 `datasets/README.md` 第 2 节拷到 `$MM_DATA_ROOT` |
| 本卡对应门 | `静态检查` |
| 可裁剪项 | 步骤 4（MiniMind 兼容审计）——没有 clone 仓库时它本来就跳过。**步骤 5 wiring smoke 不可裁** |

所有命令在周包根目录 `week03_minimind_v100_Only/` 下执行。

## 为什么做

本周所有 GPU 代码在 cc 这边一行都没在 GPU 上跑过（`00_WEEK_CARD.md` 第 9 节）。Day 0 的四件事各堵一类"到 Day 3 才爆发"的风险：路径对不上（后面每天都隐含依赖）、torch wheel 不含 sm_70、MiniMind 原脚本默认 `bfloat16`、`torchrun` + NCCL 起不来。前三件是静态事实，第四件是全周唯一的早期闸门。

## 步骤

### 1. 路径合同：四个环境变量与数据字节数（15 分钟）· `模板`

先设四个环境变量（把右边换成你机器上的实际路径），再跑两条校验：

```bash
export MM_DATA_ROOT=/your/path/to/minimind/datasets
export MM_WEIGHTS_ROOT=/your/path/to/minimind/weights
export MM_RUNS_ROOT=/your/scratch/minimind/runs
export MINIMIND_ROOT=/your/path/to/MiniMind
export HF_HUB_OFFLINE=1

python -c "import sys; sys.path.insert(0,'lab/src'); from mm_v100 import paths; print(paths.describe())"
python lab/scripts/check_data_layout.py
```

- 预期：`describe()` 打出的三行里 `MM_DATA_ROOT` / `MM_WEIGHTS_ROOT` / `MM_RUNS_ROOT` 都是"环境变量 + 存在"；`check_data_layout.py` 的必需组是 11 个 jsonl，末尾两行是 `有问题 0` 与 `全部通过。这一行可以作为 Day 0 的证据字段 data_layout=PASS。`，退出码 0。
- 不对时先查：脚本打印的 `SIZE-MISMATCH` 那一行后面的 `差 ±N B`——非零就是传输被改过字节，用二进制方式重拷那一个文件。

### 2. 本机 CPU 门：全套单测与静态编译（15 分钟）· `可直接执行`

本机（Windows，conda 环境 `rfm`，无 GPU）。这一步与步骤 1 无依赖，等 V100 排队时先跑：

```powershell
conda activate rfm
python -m compileall -q lab/src lab/scripts
python -m pytest lab/tests -q
```

- 预期：`compileall` 无输出（全部可编译）；`pytest` 打出 `368 passed, 6 skipped`（本机实测，2026-09-08；跳过的 6 个都在 `reason` 里写明"需要 CUDA"）。整套约 10 分钟，慢在 `mp.spawn` 起进程。
- 不对时先查：第一个 `FAILED` 后面的用例名——若是 `test_scripts_cli.py` 里的用例，先看 `MM_RUNS_ROOT` 是否指到了一个不可写目录。

### 3. V100 能力探针（20 分钟）· `可直接执行`

```bash
python lab/scripts/probe_env.py
```

- 预期：摘要里 `device_count=8`、`bf16_supported=False`、`sdpa functional` 中 `"flash": false` 而 `"mem_efficient": true`、`int8 _int_mm callable=False`（这三条都是**预期结果**，不是故障）；七条 `[PASS]` 检查全过，`$MM_RUNS_ROOT/day0_probe/probe.json` 出现。
- 不对时先查：最后一行 `[probe] 未通过：` 里的第一个 id。`sm70_in_arch_list` 失败是阻断级——那个 wheel 里没有 V100 的 SASS，今天就停在这里，不要往下走。

### 4. MiniMind 对 torch 2.1 的静态兼容审计（15 分钟）· `可直接执行`

只读扫描 `$MINIMIND_ROOT` 下的 `trainer/` `model/` `dataset/`，不 import、不执行：

```bash
python lab/scripts/audit_compat.py --subdirs trainer,model,dataset
```

- 预期：最后打印一行 `[audit] n_default_bfloat16=… n_autocast_no_scaler=… n_torch_compile=… n_newer_api_hits=0`。四个数里只有末尾这个决定退出码，必须是 **0**；`n_default_bfloat16` 应当 ≥ 1（`估算`：9，来自 `00_WEEK_CARD.md` 3.1 的"全部 9 个 trainer 默认 `--dtype bfloat16`"）。`compat_audit.md` 与 `compat_audit.json` 出现。
- 不对时先查：`compat_audit.md` 里 newer-API 命中那一行的文件名与符号名——那是 torch 2.1 里不存在的 API，直接跑原脚本会 `AttributeError`。

没有 clone MiniMind 时这一步返回 2 并告诉你怎么设 `MINIMIND_ROOT`；跳过它，当天证据的 `failure_class` 记 `env`。

### 5. wiring smoke：world_size 1 → 2 → 8，各 2 步（20 分钟）· `可直接执行` · **不可裁**

```bash
python lab/scripts/wiring_smoke.py --sizes 1,2,8 --config smoke_v100 \
    --dtype float32 --steps 2 --global-batch 8 \
    --launcher torchrun --backend nccl
```

**为什么不可裁**：cc 这边没有 GPU，`torchrun` 从来没有在这里跑起来过；从 1 卡到 8 卡之间会坏的东西按出现顺序是 import、配置解析、模型构建、单卡前反向、进程组建立、NCCL 握手、集合通信、显存——一次直接跑 8 卡失败时这八件事分不开。这是全周唯一的早期闸门，跳过它风险会一路累积到 Day 3 才爆发。三档合计 2–3 分钟。

- 预期：三行 `[smoke ws=1] PASS  steps=2`、`[smoke ws=2] PASS  steps=2`、`[smoke ws=8] PASS  steps=2`；`wiring_smoke.json` 出现且 `all_pass` 为 true、`visible_gpus` 为 8。
- 不对时先查：脚本停在的那一档的 `error` 字段。`torchrun 退出码 N` 说明子进程自己崩了，去看它的 stderr；ws=2 过而 ws=8 不过，先查 8 张卡是不是都可见（步骤 3 的 `device_count`）。

### 6. 生成当天证据（15 分钟）· `可直接执行`

```bash
python lab/scripts/make_evidence.py --week 3 --day 0 --card day0 \
    --run-tag day0_probe --skill v100_probe --env v100 --ai-level A2 \
    --status PASS --failure-class none --time-spent 100 \
    --primary-artifact "Day 0 探针 + 兼容审计 + 三档 wiring smoke 的抽象值" \
    --config-id "smoke_v100 / torch2.1 / commit 7a6fddd"
```

- 预期：`[evidence] observations 共 N 个字段，全部来自白名单聚合。`，`$MM_RUNS_ROOT/day0_probe/evidence.json` 出现，里面有 `all_checks_pass`、`device_count`、`bf16_supported`、`n_newer_api_hits` 这几项。
- 不对时先查：报 `run 目录不存在` 时，核对 `--run-tag` 是不是 `day0_probe`（前面三个脚本的默认 run-tag 就是它）。

生成的 `evidence.json` 是本周唯一设计成可以带出公司的文件；带走之前自己再看一眼，脚本只做机械白名单过滤。

## 公司路径差异

- 本卡除步骤 2 外全部在公司 8×V100 上执行，一律 `--dtype float16`（步骤 5 用 `float32` 是因为它只验接线不验数值）；`bfloat16` 会在参数解析阶段被拒绝。
- 每次运行 ≤ 10 分钟。`$MM_RUNS_ROOT` 下的 `probe.json`、`compat_audit.md`、`wiring_smoke.json`、日志和 checkpoint 全部留在公司内；交流时只把下面证据字段里的抽象值（比值、布尔、计数）填好。

## 今日自测（闭卷，2 题，只记录能/不能）

1. （Explain）为什么 `torch.backends.cuda.flash_sdp_enabled()` 返回 True 并不能说明 V100 上能用 flash 后端？探针用什么办法把这件事从"推断"提到"已确认"？
2. （Design）如果只允许跑一次多卡实验，为什么仍然要按 1 → 2 → 8 三档来跑，而不是直接跑 8 卡？说出这三档之间各自被隔离掉的失败类别。

## 证据字段（填完即可归档到 `memory/cc/evidence/`）

```json
{
  "date": "2026-09-DD",
  "week": 3,
  "day": 0,
  "card": "day0",
  "skill": "v100_probe",
  "env": "v100",
  "ai_level": "A2",
  "config_id": "smoke_v100 / torch2.1.0 / minimind@7a6fddd",
  "primary_artifact": "day0_probe/evidence.json：探针 + 兼容审计 + 三档 wiring smoke",
  "observations": {
    "data_layout_pass": true,
    "cpu_tests_passed": 368,
    "all_checks_pass": true,
    "device_count": 8,
    "bf16_supported": false,
    "sdpa_flash_functional": false,
    "int8_callable_on_device": false,
    "n_default_bfloat16": 9,
    "n_newer_api_hits": 0,
    "wiring_smoke_max_world_size_passed": 8
  },
  "status": "PASS",
  "failure_class": "none",
  "self_check": {"q1": "能", "q2": "能"},
  "time_spent_min": 100,
  "notes": "三档 wiring smoke 全过；后面每天都以此为前提"
}
```
