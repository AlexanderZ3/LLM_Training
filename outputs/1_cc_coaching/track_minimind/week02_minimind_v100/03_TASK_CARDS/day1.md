# Week M02 · Day 1 — 单卡 FP16 跑通，再让 1 卡 / 2 卡 / 8 卡在固定全局批下走出同一条 loss 曲线

| 字段 | 值 |
| --- | --- |
| 主要产物 | 两条 `--compare-json` 判定行（`ws1 vs ws2`、`ws1 vs ws8`）的 `max|delta|` 与 `PASS/FAIL`，来自 `<公司内路径>/day1/equiv_ws{1,2,8}_rank0.jsonl` |
| 估时 | `90 分钟`（步骤估时之和） |
| 环境 | `公司 V100` |
| AI 辅助等级要求 | `A1`（命令可查本卡与 `02_LAB_GUIDE.md` 第 3 节，但"为什么三次的 `micro_batch` 必须相同"要能当场讲出来） |
| 前置 | Day 0 的证据：`bf16_supported = false`、`has_sm_70 = true`、`device_count = 8`、`wiring_smoke_three_runs_started = true` |
| 本卡对应门 | `FP32 ref` → `FP16` → `2 卡等价` → `多卡等价` |
| 可裁剪项 | 步骤 4（2 卡 smoke）；步骤 1–3、5–6 不可裁 |

## 为什么做（三句以内）

DDP 是本周四种放置方式的基准线，后面 FSDP、EP、GRPO 的每一个比值都要拿它当分母。如果 1 卡和 8 卡在固定全局批下算出的都不是同一个梯度估计量，那后面所有的 step time 与显存比较都在比两个不同的实验。今天用不变量 I1 把这条基准线钉住。

## 步骤

### 1. 单卡 FP32 reference 一步（10 分钟）· `模板`

```bash
export MINIMIND_ROOT=<MiniMind 仓库绝对路径>
export PYTHONPATH=$PWD/lab/src:$PYTHONPATH
python -m mm_dist.train_ddp --config lab/configs/v100_dense.json \
  --out-dir <公司内路径>/day1/gate2 --dtype float32 \
  --global-batch 8 --accum 1 --max-steps 1
```

- 预期：header 行里 `params_total` 约 63.9M、`world_size` 为 1、`dtype` 为 `float32`；第 1 步 `loss` 落在 `ln(6400) ≈ 8.76` 附近（随机初始化，`估算`）；`scaler_scale` 为 1.0。
- 不对时先查：`loss` 明显不在 8.7 量级 → 看 header 的 `params_total` 与 `use_moe`，`v100_dense.json` 应当是 dense、8 层、hidden 768。

### 2. 同一个 batch 换成 FP16（10 分钟）· `模板`

```bash
python -m mm_dist.train_ddp --config lab/configs/v100_dense.json \
  --out-dir <公司内路径>/day1/gate2 --dtype float16 \
  --global-batch 8 --accum 1 --max-steps 1 --log-name fp16_1step
```

- 预期：`fp16_1step_rank0.jsonl` 出现；它的 `loss` 与步骤 1 的相对差 < 1%（`估算`）；`scaler_scale` 起始为 65536；`skipped_steps` 为 0 或 1（第一步跳一次是正常的）。
- 不对时先查：`loss` 打印 `nan` → 先看 `grad_norm` 是不是 `inf`，再用 `--dtype float32` 跑同一配置；fp32 也发散就是模型/学习率问题，不是 fp16 问题（故障树 F8）。

### 3. 确认 bf16 被显式拒绝（5 分钟）· `模板`

```bash
python -m mm_dist.train_ddp --config lab/configs/v100_dense.json \
  --out-dir <公司内路径>/day1/gate2 --dtype bfloat16 --max-steps 1
```

- 预期：在建模型**之前**就抛 `RuntimeError`，消息里含 "V100"、"compute capability 7.0"、"请改用 --dtype float16"；不进训练循环，`<公司内路径>/day1/gate2` 下不新增日志文件。
- 不对时先查：没有报错反而开始训练 → `common.assert_dtype_supported` 没被调用，检查是不是绕过了 `train_ddp.train()` 的入口。

### 4. 2 卡 fp16 smoke（15 分钟）· `模板`

```bash
torchrun --nproc_per_node 2 -m mm_dist.train_ddp \
  --config lab/configs/v100_dense.json --out-dir <公司内路径>/day1/gate3 \
  --dtype float16 --global-batch 16 --accum 1 --max-steps 5 \
  --nccl-timeout-s 600 --log-name smoke_ws2
```

- 预期：5 步都有日志行；header 的 `world_size` 为 2、`backend` 为 `nccl`；**两个 rank 的 `loss` 字段完全相同**（它是 all_reduce 之后再除以 `world_size × accum` 的量）；`tokens_per_s` 有值。
- 不对时先查：两个 rank 的 `loss` 不同 → all_reduce 没生效，看 header 的 `--backend`；卡住不动 → 把 `--nccl-timeout-s` 压到 60 让它快点失败，再按故障树 F2 用 `py-spy dump` 看每个 rank 卡在哪个 collective。

### 5. 固定全局批跑三次：1 卡 / 2 卡 / 8 卡（30 分钟）· `模板`

```bash
OUT_DIR=<公司内路径>/day1/gate4 MAX_STEPS=20 bash lab/scripts/run_ddp_equiv.sh
```

脚本内部跑 `W=1/accum=8`、`W=2/accum=4`、`W=8/accum=1`，三次的 micro batch 都是 8；`--equiv-check` 会自动把 `dtype` 强制成 `float32` 并把 `seed_per_rank` 关掉，因为 fp16 下不同的归约顺序带来的差异比判定阈值还大。

- 预期：三次运行的 header 里 `micro_batch` 都是 8、`seed_per_rank` 都是 `false`、`dtype` 都是 `float32`；`<公司内路径>/day1/gate4/` 下出现 `equiv_ws1_rank0.jsonl`、`equiv_ws2_rank0.jsonl`、`equiv_ws8_rank0.jsonl`；总耗时 6–9 分钟（`估算`）。
- 不对时先查：这是本周第二次跑 `run_*` 脚本，若它在公司机器上第一次启动就失败，先用 `MAX_STEPS=2` 把线路跑通再回到 20 步；单次运行超过 12 分钟就 Ctrl-C，把 `MAX_STEPS` 减半重跑。

### 6. 逐步比对并给出判定（15 分钟）· `可直接执行`

脚本结尾已经跑了两条比对，把它们的输出抄下来；需要重跑时：

```bash
python -m mm_dist.train_ddp --compare-json \
  <公司内路径>/day1/gate4/equiv_ws1_rank0.jsonl \
  <公司内路径>/day1/gate4/equiv_ws8_rank0.jsonl --compare-tol 1e-4
```

- 预期：两条比对都给出 `max|delta| < 1e-4` 与 `PASS`；`steps_compared` 为 20。写课程那台机器上用 tiny 配置做过同类实验（`W=1/accum=4` vs `W=2/accum=2`，micro 都是 2），三步 loss **完全相等**（`本机实测`），所以公司机器上 fp32 下应当也是极小的差值。
- 不对时先查：第 1 步就差 > 1e-2 → 按故障树 F3 的四步走：先核对三次 header 的 `micro_batch` 是否相同，再确认 `seed_per_rank=false`、`dtype=float32`，最后打印每步各 rank 的样本索引并集与 `[(s-1)G, sG)` 对账。

### 7. 填证据字段（5 分钟）· `可直接执行`

- 预期：`02_LAB_GUIDE.md` 第 6.2 节的六个字段全部有值：`micro_batch_identical`、`max_abs_delta_ws1_ws2`、`max_abs_delta_ws1_ws8`、`tol`、`equiv_pass`、`steps_compared`；两个 delta 只报数量级即可。
- 不对时先查：想把 loss 序列一起记下来时停一下——只有 `max|delta|` 这个抽象量出公司，逐步 loss 留在公司内。

## 公司路径差异

- 全部步骤在公司 8×V100 上执行；`--out-dir` 指向公司内部路径。
- **fp16 不是 bf16**：步骤 2、4 的训练命令显式写 `--dtype float16`；步骤 3 用 `--dtype bfloat16` 只是为了看那条 `RuntimeError`，它不会进训练循环。步骤 5 的等价实验由 `--equiv-check` 强制走 fp32，这是不变量 I1 的判定条件。
- 日志、checkpoint、逐步 loss 序列、`tokens_per_s` 绝对值全部留在公司内；离开公司机器的只有下面证据字段里的抽象值。

## 今日自测（闭卷，2 题，只记录能/不能）

1. `M02-E-05`（fixed-global-batch 不变量 I1 的五个条件；`setup_seed(42+rank)` 为什么在当前默认配置下不破坏初值一致）
2. `M02-D-05`（fp16 的 scale 一路减半 + 两种 world_size 第 1 步就差 1e-2，两个异常分开定位）

## 证据字段（填完即可归档到 memory/cc/evidence/）

```json
{
  "date": "YYYY-MM-DD",
  "week": 2,
  "day": 1,
  "card": "day1",
  "skill": "ddp_fixed_global_batch_equivalence",
  "env": "v100",
  "ai_level": "A1",
  "config_id": "v100_dense.json + G=64 + equiv-check(fp32)",
  "primary_artifact": "compare_logs: equiv_ws1 vs equiv_ws2 / equiv_ws8 的 max|delta| 与判定",
  "observations": {
    "fp32_vs_fp16_first_step_rel_diff": "<1% | 其他",
    "scaler_scale_start": 65536,
    "skipped_steps_first_run": 0,
    "bf16_rejected_before_model_build": "true | false",
    "ws2_smoke_loss_identical_across_ranks": "true | false",
    "micro_batch_identical": "true | false",
    "max_abs_delta_ws1_ws2": "<数量级，如 1e-7>",
    "max_abs_delta_ws1_ws8": "<数量级>",
    "tol": "1e-4",
    "equiv_pass": "true | false",
    "steps_compared": 20
  },
  "status": "PASS | FAIL-MODEL | FAIL-SYSTEM | INCONCLUSIVE",
  "failure_class": "none | numeric | measurement | system | data_contract",
  "self_check": {"M02-E-05": "能 | 不能", "M02-D-05": "能 | 不能"},
  "time_spent_min": 0,
  "notes": "<一句话：判定结果与最可疑的一条差异来源，不含绝对 loss 与吞吐数字>"
}
```
