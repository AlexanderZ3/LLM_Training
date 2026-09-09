# Week M03 · Day 1 — 在单卡上同时立住数据契约和 fp32↔fp16 的误差账

| 字段 | 值 |
| --- | --- |
| 主要产物 | `$MM_RUNS_ROOT/day1_pretrain/numeric_ref.json`（fp32 参考曲线 vs fp16 AMP 曲线的四个字段） |
| 估时 | `110 分钟`（步骤估时之和） |
| 环境 | `公司 V100`（步骤 1 在本机 CPU 上做） |
| AI 辅助等级要求 | `A2`；步骤 5 对 `scale_trajectory` 与 `first_diverge_step` 的判读要求 `A1` |
| 前置 | Day 0 证据：`data_layout_pass = true`、`wiring_smoke_max_world_size_passed = 8`、`bf16_supported = false` |
| 本卡对应门 | `CPU 单测` → `单 batch 前反向` → `128 样本有界训练` → `FP32 ref` → `FP16 AMP` |
| 可裁剪项 | 步骤 4（128 样本有界训练）。步骤 2 与步骤 5 是主要产物的两条腿，不裁 |

所有命令在周包根目录 `week03_minimind_v100_Only/` 下执行。本周训练的是 `lab/src/mm_v100/model.py` 里自带的 `TinyCausalLM`——它与 MiniMind **形状对齐**（同样的 768/8/8/4/6400、`intermediate_size=2432`、tied embedding），但不含 MiniMind 的权重和 tokenizer。字节账、误差账、通信账的结论可迁移；MiniMind 自己的 `.pth` 不能加载进来。

## 为什么做

Day 2 要判"两条曲线是不是同一条"，Day 3 要判"恢复之后是不是同一次训练"，两者都需要一个**已知的数值噪声底**。今天先把这个底测出来：fp32 是参考，fp16 是被测对象，`max_dloss` 与 `first_diverge_step` 就是后面所有等价判据的量纲来源。同时把数据契约（D1/D3 断言）钉死——曲线好看只证明目标可优化，不证明目标正确。

## 步骤

### 1. 本机 CPU 门：数据契约、误差账、dtype 守卫（15 分钟）· `可直接执行`

```powershell
conda activate rfm
python -m pytest lab/tests/test_data_contract.py lab/tests/test_numeric_ref.py lab/tests/test_dtype_guard.py -q
```

- 预期：一行 `N passed`，没有 `failed`（本机实测 2026-09-08：整套 `lab/tests` 是 368 passed / 6 skipped）。
- 不对时先查：第一个 `FAILED` 后面的用例名——`test_dtype_guard.py` 里的失败说明 `assert_dtype_supported` 对 `bfloat16` 的拒绝逻辑被改动过，那会让 V100 上的守卫失效。

### 2. 数据契约：一条 JSONL 走到 (input_ids, labels)（20 分钟）· `可直接执行`

```bash
python lab/scripts/check_data_contract.py --file pretrain_t2t_mini.jsonl \
    --stage pretrain --max-len 512 --limit 8 --run-tag day1_pretrain
```

数据还没拷进来时把 `--file pretrain_t2t_mini.jsonl` 换成 `--toy pretrain`，链路照样验完，但证据里 `config_id` 要写 `toy`。

- 预期：`I1 label_align_violations_total = 0（正常值 0）`、`verdict = PASS`；逐样本表里 `label_ratio` 列在 0.9–1.0 之间（pretrain 下每条序列被计分的 token 是非 pad 数减 1）；`$MM_RUNS_ROOT/day1_pretrain/data_contract.json` 出现。
- 不对时先查：逐样本表里 `align_viol` 第一条非零的那一行——非零意味着 `labels` 在非 `-100` 处不等于 `input_ids`，也就是数据侧多做了一次位移（`01_FOUNDATIONS.md` §5.3 算例 A）。

### 3. 单 batch 前反向：第 0 步 loss 必须是 ln V（10 分钟）· `可直接执行`

```bash
python lab/scripts/train_bounded.py --config v100_768 --run-tag day1_pretrain \
    --placement single --stage pretrain --dtype float16 \
    --global-batch 8 --accum 1 --dry-run
```

- 预期：`[dry-run] forward ok  logits=(8, 512, 6400)  loss=8.7xxxx`。随机初始化下 loss 必须落在 `[8.71, 8.82]`（`计算`：ln 6400 = 8.76405，`01_FOUNDATIONS.md` §5.4）。
- 不对时先查：偏出这个区间时先看 `lab/configs/v100_768.json` 的 `vocab_size` 是不是 6400——loss 的理论值只由它决定，与掩码无关。

### 4. 128 样本上的有界训练（25 分钟）· `可直接执行`

```bash
python lab/scripts/train_bounded.py --config v100_768 --run-tag day1_pretrain \
    --placement single --stage pretrain --dtype float32 \
    --global-batch 8 --accum 1 --num-samples 128 --max-steps 16 \
    --log-interval 1 --log-name overfit128
```

`TinyDataset` 的样本由 `(seed, index)` 决定，`micro_batch_indices` 单调推进不重复，所以这 16 步恰好把 128 条样本过一遍。**它不是"反复喂同一批直到 loss 归零"的 overfit**——`01_FOUNDATIONS.md` §5.5 已经说明那种 overfit 对位移故障完全无效。这一步验的是"前反向 + 优化器 + 日志"这条链在 fp32 下是通的。

- 预期：`overfit128_rank0.jsonl` 出现，含 1 行 header 与 16 行 step；第 1 步 loss 在 `[8.71, 8.82]`，第 16 步低于第 1 步（`估算`：下降 0.05–1.0；合成 token 是均匀随机的，条件熵本身就是 ln V，降不多是正常的）；每行 `grad_norm` 有限且大于 0。
- 不对时先查：`overfit128_rank0.jsonl` 里 `grad_norm` 是否恒为 0——恒为 0 说明梯度没形成，回到步骤 3 的 dry run 看 loss 是不是真的带上了 `labels`。

### 5. 误差账：fp32 参考曲线 vs fp16 AMP 曲线（30 分钟）· `可直接执行`

```bash
python lab/scripts/run_numeric_ref.py --config v100_768_accum \
    --device cuda:0 --steps 20 --tol 1e-3 --low-dtype float16 \
    --sdpa-backend auto --run-tag day1_pretrain
```

用 `v100_768_accum` 而不是 `v100_768`：两者全局批都是 64，但前者 `accum=4` 把 micro-batch 压到 16。单卡上要同时放两份模型（fp32 参考与 fp16 被测），`v100_768` 的 micro=64 在纯 fp32 下按 `01_FOUNDATIONS.md` §2.4 的估算式激活约 16 GB，会顶到 32 GB 卡的上限。

- 预期：逐步对照表 20 行；`scale: first=65536.0`，`skip_ratio` < 0.34，`verdict=PASS`；`numeric_ref.json` 出现。`max_dloss` 与 `first_diverge_step` 记下来——它们是 Day 2 判等价、Day 3 判恢复的噪声底（`估算`：`max_dloss` 在 1e-4–1e-2 量级，V100 上第一次测才有真值）。
- 不对时先查：`scale: first=65536.0 last=…` 里的 `last`。`last` 比 `first` 小很多且 `scale_monotone_down` 为真，就是 Day 2 故障 B 的形态——先看前向 logits 是不是已经出现 inf，而不是去调 `init_scale`。

### 6. 生成当天证据（10 分钟）· `可直接执行`

```bash
python lab/scripts/make_evidence.py --week 3 --day 1 --card day1 \
    --run-tag day1_pretrain --skill amp_numeric_ref --env v100 --ai-level A2 \
    --status PASS --failure-class none --time-spent 110 \
    --primary-artifact "fp32 vs fp16 的逐步误差账与数据契约的两个不变量" \
    --config-id "v100_768_accum / float16 / steps=20"
```

- 预期：`evidence.json` 出现在 `$MM_RUNS_ROOT/day1_pretrain/`，`observations` 里能看到 `max_dloss`、`first_diverge_step`、`skip_ratio`、`scale_first`、`scale_last`、`label_align_violations_total`。
- 不对时先查：`observations` 里没有 `max_dloss` 时，核对步骤 5 是否真的写出了 `numeric_ref.json`（`make_evidence.py` 只扫 run 目录里已存在的 `*.json`）。

## 公司路径差异

- 一律 `--dtype float16` + `GradScaler`；`bfloat16` 在 V100 上会被 dtype 守卫在参数解析阶段拒绝（`00_WEEK_CARD.md` 3.1）。
- 步骤 5 的 `--sdpa-backend auto` 在 V100 上会落到 mem-efficient 或 math，没有 flash 后端；这会改变激活账的系数但不改变误差账的结论。
- 日志、`numeric_ref.json` 原始曲线、显存数字全部留在公司内；只把下面证据字段里的抽象值填好。

## 今日自测（闭卷，2 题，只记录能/不能）

1. （Explain）为什么 loss 标量必须留在 fp32？给出 8.764 附近 fp16 的 ULP 数值，并说明它和"要分辨 1e-4 量级的 loss 差"之间的矛盾。
2. （Debug）一条 loss 曲线平滑单调下降，但收敛平台比预期高约 1 nat。D1/D2 两个计数都正常。接下来做哪一个检查能区分"双重位移"和"模型太小"？

## 证据字段（填完即可归档到 `memory/cc/evidence/`）

```json
{
  "date": "2026-09-DD",
  "week": 3,
  "day": 1,
  "card": "day1",
  "skill": "amp_numeric_ref",
  "env": "v100",
  "ai_level": "A2",
  "config_id": "v100_768_accum / float16 / steps=20",
  "primary_artifact": "day1_pretrain/numeric_ref.json：fp32 vs fp16 的逐步误差账",
  "observations": {
    "label_align_violations_total": 0,
    "label_ratio_mean": 0.0,
    "step0_loss_in_lnV_band": true,
    "max_dloss": 0.0,
    "max_rel_dloss": 0.0,
    "first_diverge_step": 0,
    "skip_ratio": 0.0,
    "scale_first": 65536.0,
    "scale_last": 65536.0,
    "overfit128_loss_ratio_last_over_first": 0.0
  },
  "status": "PASS",
  "failure_class": "none",
  "self_check": {"q1": "能", "q2": "能"},
  "time_spent_min": 110,
  "notes": "max_dloss 与 first_diverge_step 作为 Day 2/3 等价判据的噪声底"
}
```
