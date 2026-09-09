# Week M03 · Day 2 — 让 2 卡/8 卡走出同一条曲线，再把它弄坏三次并只凭日志区分开

| 字段 | 值 |
| --- | --- |
| 主要产物 | `$MM_RUNS_ROOT/day2_sft/equiv_table.json` + 一张**手写**的三行判别表 `$MM_RUNS_ROOT/day2_sft/decision_table.md` |
| 估时 | `120 分钟`（步骤估时之和） |
| 环境 | `公司 V100`（步骤 1 在本机 CPU 上做） |
| AI 辅助等级要求 | `A1`（只查官方文档）；步骤 5 手写判别表要求 `A0`（闭卷、限时 15 分钟） |
| 前置 | Day 1 证据：`max_dloss` 已知、`first_diverge_step` 已知、`label_align_violations_total = 0` |
| 本卡对应门 | `2 卡等价` → `8 卡等价` → `故障注入` |
| 可裁剪项 | 步骤 2（2 卡等价）——8 卡等价过了就覆盖了它。**步骤 4 三档故障注入不可裁** |

所有命令在周包根目录 `week03_minimind_v100_Only/` 下执行。

## 为什么做

"loss 在下降"对正常训练和三档静默故障**同时**成立，所以它是必要条件不是充分条件。今天先用固定全局批把"多卡与单卡等价"这条基线立起来（否则后面任何异常都可以推给"多卡本来就不一样"），再注入三档不报错、不 hang、曲线还挺正常的故障，用三个不变量把它们分开。这张手写判别表是 Gate 的 debug 题唯一的依据。

## 步骤

### 1. 本机 CPU 门：2 进程 gloo 等价与四档故障（15 分钟）· `可直接执行`

```powershell
conda activate rfm
python -m pytest lab/tests/test_ddp_equiv_cpu.py lab/tests/test_faults.py -q
```

- 预期：一行 `N passed`，没有 `failed`。这两个文件里的用例是真起 2 个 gloo 进程跑的，不是纸面推导（本机实测 2026-09-08）。
- 不对时先查：第一个 `FAILED` 的用例名——若在 `test_faults.py::` 的 `ddp_no_sync` 相关用例，先看 `cross_rank_grad_delta` 是不是还在做逐元素指纹比对（改回比范数就抓不到这一档）。

### 2. 2 卡固定全局批等价（20 分钟）· `可直接执行`

A 侧 = 1 卡 accum 8，B 侧 = 2 卡 accum 4，全局批都是 64，两侧 micro 都是 8：

```bash
python lab/scripts/equiv_check.py --config v100_768 --stage sft \
    --global-batch 64 --a-nproc 1 --a-accum 8 --b-nproc 2 --b-accum 4 \
    --max-steps 6 --tol 1e-4 --backend nccl \
    --out-dir $MM_RUNS_ROOT/day2_sft_ws2
```

用单独的 `--out-dir` 是因为步骤 3 会往 `day2_sft` 写同名的 `equiv_table.json` 与同名日志，不分开就被覆盖。

- 预期：脚本先打 `A: world_size=1 accum=8 micro=8` / `B: world_size=2 accum=4 micro=8`（两侧 micro 必须相同，否则会额外打一条 `[warn]`）；逐步表 6 行，末尾 `max|delta|=…  tol=1.0e-04  -> PASS`（`估算`：`max|delta|` 落在 1e-7–1e-5，比 Day 1 的 `max_dloss` 小得多，因为两侧都是 fp32）。
- 不对时先查：第 1 步的 `|delta|` 就大于 1e-2 时，看脚本开头打印的两侧 `micro` 是否相同——不同就说明 batch 维度的归约顺序不同，那不是通信问题。

### 3. 8 卡固定全局批等价（25 分钟）· `可直接执行`

```bash
python lab/scripts/equiv_check.py --config v100_768 --stage sft \
    --global-batch 64 --a-nproc 1 --a-accum 8 --b-nproc 8 --b-accum 1 \
    --max-steps 6 --tol 1e-4 --backend nccl --run-tag day2_sft
```

- 预期：`A: world_size=1 accum=8 micro=8` / `B: world_size=8 accum=1 micro=8`；`max|delta|` < 1e-4 → `PASS`；`$MM_RUNS_ROOT/day2_sft/equiv_table.json` 出现，里面 `a_world_size=1`、`b_world_size=8`、`pass=true`。
- 不对时先查：`|delta|` 在 1e-4 和 1e-2 之间时，改跑同一条命令加 `--key grad_norm` 比梯度范数——梯度范数也偏，问题在数据切分；只有 loss 偏，问题在归约顺序。

### 4. 三档静默故障注入（35 分钟）· `可直接执行` · **不可裁**

四档一次跑完（`none` 是对照组），同一个脚本、同一个 seed、同一份数据：

```bash
python lab/scripts/run_faults.py --config v100_768 --placement ddp --stage sft \
    --dtype float16 --nproc 8 --accum 1 --global-batch 64 --max-steps 6 \
    --backend nccl --launcher spawn --run-tag day2_sft \
    --modes none,label_shift,scaler_stuck,ddp_no_sync
```

**为什么不可裁**：Gate 的 A1 与 debug 两题都直接考这四行的观测值，没有这一步就没有可复述的现场。三档的机理见 `01_FOUNDATIONS.md` §6.3–§6.5。

- 预期：末尾"判别表（实测）"四行，`none` 行是 `I1=0`、`I2_ratio≈1.000`、`I3=0.0`、`verdict=none`；`label_shift` 行 `I1>0`；`scaler_stuck` 行 `I2_ratio` 明显低于 1 且 `I2_delta` 为 0；`ddp_no_sync` 行 `I3` 大于 1e-6。`fault_table.json` 出现且 `all_detected` 为 true。
- 不对时先查：`ddp_no_sync` 行的 `I3` 显示 `None`——那说明这一档跑在了 `nproc<2` 的路径上（脚本会打 `[warn] nproc=1 时 ddp_no_sync 没有内容`），把 `--nproc` 调回 8。

注意 I3 是**梯度指纹的逐元素比对**（取前若干个参数张量的前若干个元素 all_gather 后逐元素比），健康 DDP 下恒等于 0.0。它不是"比较各 rank 的梯度范数"——范数把方向信息全丢了，实测下 `ddp_no_sync` 只让范数差 3.58e-07，任何合理阈值都抓不到。

### 5. 闭卷手写三行判别表，再对答案（15 分钟）· `伪代码`

先关掉终端，写文件 `$MM_RUNS_ROOT/day2_sft/decision_table.md`，内容是一张四行三列的表：

- 行：`none` / `label_shift` / `scaler_stuck` / `ddp_no_sync`
- 列：`I1>0` / `I2 低` / `I3>0`
- 每行末尾再写一句"这一档为什么用这条不变量是**充分**的"

写完再对答案：

```bash
python lab/scripts/run_faults.py --config v100_768 --dry-run
```

`--dry-run` 只打印官方判别表，不跑训练、不覆盖步骤 4 的 `fault_table.json`。

- 预期：自己写的四行三列与脚本打印的 `render_decision_table()` **逐格一致**；脚本末尾两行是 `必要不充分：loss 在下降 —— 四种模式下都成立。` 与 `充分：任一不变量单独越界即可定位到唯一一档。`
- 不对时先查：最常写反的是 `ddp_no_sync` 行的 `I3` 一格——回到 `01_FOUNDATIONS.md` §6.6 的"I3 ≠ 0 充分但不必要"那一行。

### 6. 生成当天证据（10 分钟）· `可直接执行`

```bash
python lab/scripts/make_evidence.py --week 3 --day 2 --card day2 \
    --run-tag day2_sft --skill fault_triage --env v100 --ai-level A1 \
    --status PASS --failure-class none --time-spent 120 \
    --primary-artifact "8 卡固定全局批等价表 + 四档故障的三不变量判别表" \
    --config-id "v100_768 / float16 / world_size=8 / global_batch=64"
```

- 预期：`evidence.json` 出现在 `$MM_RUNS_ROOT/day2_sft/`，`observations` 里有 `max_abs_delta`、`steps_compared`、`all_detected`，以及每个 `fault_*` 日志聚合出的 `I1_align_viol` / `I2_step_ratio` / `I3_grad_delta_is_zero`。
- 不对时先查：`all_detected` 没出现在 `observations` 里时，核对步骤 4 是不是在同一个 `--run-tag day2_sft` 下跑的。

## 公司路径差异

- 等价实验强制 fp32（`equiv_check.py` 自带 `--equiv-check`），故障实验用 fp16——`scaler_stuck` 这一档只有 fp16 才有内容。
- 8 卡非独占时步骤 3 的 `max|delta|` 仍然可信（它与调度无关），但任何时间类数字都不可信；Day 0 探针里的 `gpu_exclusive.looks_exclusive` 是判断依据。
- 逐步 loss 序列、`fault_table.json` 的原始数字、显存与吞吐留在公司内；只把下面证据字段里的比值、布尔和计数填好。

## 今日自测（闭卷，2 题，只记录能/不能）

1. （Recall）写出三档静默故障各自的"不变量 → 判据"三行，并标出哪一条是充分的、哪一条只是必要的。
2. （Trade-off）为什么 I3 用梯度指纹的逐元素比对而不是梯度范数？举一个范数相同而方向完全不同的具体情形。

## 证据字段（填完即可归档到 `memory/cc/evidence/`）

```json
{
  "date": "2026-09-DD",
  "week": 3,
  "day": 2,
  "card": "day2",
  "skill": "fault_triage",
  "env": "v100",
  "ai_level": "A1",
  "config_id": "v100_768 / float16 / world_size=8 / global_batch=64",
  "primary_artifact": "day2_sft/equiv_table.json + 手写的四行三列判别表",
  "observations": {
    "equiv_ws2_max_abs_delta": 0.0,
    "equiv_ws8_max_abs_delta": 0.0,
    "equiv_steps_compared": 6,
    "all_detected": true,
    "none_I3_is_zero": true,
    "label_shift_I1_gt_0": true,
    "scaler_stuck_I2_step_ratio": 0.0,
    "ddp_no_sync_I3_gt_tol": true,
    "hand_table_matches_reference": true
  },
  "status": "PASS",
  "failure_class": "none",
  "self_check": {"q1": "能", "q2": "能"},
  "time_spent_min": 120,
  "notes": "手写判别表与 render_decision_table() 逐格一致"
}
```
