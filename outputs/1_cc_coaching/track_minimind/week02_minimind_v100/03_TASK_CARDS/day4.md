# Week M02 · Day 4 — 人为造一个热点专家，再用 aux 损失系数把它压回去，量出"均衡"与"分工"这对权衡

| 字段 | 值 |
| --- | --- |
| 主要产物 | `<公司内路径>/day4/gate8/moe_sweep.csv` 汇成的 bias × aux_coef 二维抽象表（`max_load_ratio`、`router_entropy`、`logits_loss_ratio`、`wait_ms_ratio`）加一句结论 |
| 估时 | `90 分钟`（步骤估时之和） |
| 环境 | `公司 V100` |
| AI 辅助等级要求 | `A1`（扫描是脚本跑的；主指标为什么不能选 `aux_loss` 必须由你自己论证） |
| 前置 | Day 3 的证据：`split_invariant_holds = true`、`ep1_equiv_pass = true`（EP 流水可信，木桶效应才归因得了 router） |
| 本卡对应门 | `有界训练` → `消融` |
| 可裁剪项 | 步骤 3（单点热点复现的逐步曲线）；步骤 2 超预算时把 `BIAS_LIST` 砍到 `0,1,4`、`AUX_COEF_LIST` 砍到 `0,0.01` |

## 为什么做（三句以内）

Day 3 量到的木桶效应还只是一个现象：某个 rank 的 `expert_compute` 高、其余 rank 的 `wait` 高。今天要证明它由 router 的负载分布**因果地**决定——手动给一个专家的 logit 加常数就能造出来，加大 aux 系数就能压回去。同时要看清代价：把 router 强制推向均匀会让它失去分工，这正是 MoE 存在的理由。

## 步骤

### 1. 先用缩表 + `MAX_STEPS=2` 试线路（10 分钟）· `模板`

```bash
export MINIMIND_ROOT=<MiniMind 仓库绝对路径>
OUT_DIR=<公司内路径>/day4/wiring MAX_STEPS=2 \
  BIAS_LIST=0,4 AUX_COEF_LIST=0,0.01 bash lab/scripts/run_moe_sweep.sh
```

`run_moe_sweep.sh` 在写课程的那台机器上没有端到端跑过（无 GPU，且它的 torch wheel 缺 libuv 起不了 `torchrun`），所以第一次运行先压到 2×2 组合、2 步。

- 预期：`torchrun` 起来并写出 `<公司内路径>/day4/wiring/moe_sweep.csv`，CSV 有 4 行数据；接着单点复现那一段写出 `hot_expert_rank<r>.jsonl`。
- 不对时先查：CSV 行数少于组合数 → 某一格的 `torchrun` 中途失败了，看 stdout 里最后一格的 bias/coef 与它的退出码；`torchrun` 卡住 → 把 `NCCL_TIMEOUT_S` 压到 60 让它快点失败。

### 2. 全量 5×4 扫描（30 分钟）· `模板`

```bash
OUT_DIR=<公司内路径>/day4/gate8 MAX_STEPS=10 \
  BIAS_LIST=0,0.5,1,2,4 AUX_COEF_LIST=0,0.001,0.01,0.1 \
  bash lab/scripts/run_moe_sweep.sh
```

自变量：`--bias-list` 给 `--expert-id 0` 那个专家的 logit 加常数；`--aux-coef-list` 覆盖 config 里的 `router_aux_loss_coef`（MiniMind 默认是 `5e-4`，它不在这四个扫描点里，读表时要记得基线在哪）。脚本的 `DTYPE` 默认 `float16`，落到 `torchrun` 上就是 `--dtype float16`。

- 预期：`<公司内路径>/day4/gate8/moe_sweep.csv` 有 20 行数据，列是 `router_stats.CSV_COLUMNS`（`expert_fractions` 用 `|` 连接成一个字段）；总耗时 6–10 分钟（`估算`）。
- 不对时先查：单次运行超过 12 分钟就 Ctrl-C，先把 `BIAS_LIST` 砍到 `0,1,4` 再跑；某一格算出 `nan` → 看那一格的 `scaler_scale` 是不是一路减半（故障树 F8），fp16 发散和 router 坍缩是两回事。

### 3. 单点热点复现，看逐步曲线（15 分钟）· `模板`

脚本第二段已经用 `BIAS_LIST` 里最大的那一档跑了一次 `router_stats stats`，产出 `hot_expert_rank<r>.jsonl`。核对它：

- 预期：`bias=4` 那一档的 `max_load_ratio` 明显高于 `bias=0`；`router_entropy` 明显低于 `bias=0`；`wait_ms` 在各 rank 之间分化（最慢那张卡的 `wait_ms` ≈ 0，其余卡被它拖住）。写课程的机器上用 `bias=50` 做过极端验证：`expert_fractions[hot] > 0.99`、`max_load_ratio ≈ E`、`router_entropy < 0.05`（`本机实测`）。
- 不对时先查：`bias=4` 下 `max_load_ratio` 仍然 ≈1 → 看 `--expert-id` 是不是指到了一个本来就没人路由的专家，或者 `--aux-coef` 用的是 `stats` 段那一行写死的 `0.0005` 而 bias 被它压住了。

### 4. 统计口径自检（10 分钟）· `可直接执行`

在读任何趋势之前，先确认这张表的口径没写错。

- 预期：CSV 里每一格的 `mean_load_ratio` 恒等于 `1.0`，每一格的 `sum(expert_fractions)` == 1。另外两项**不在 CSV 里**（`CSV_COLUMNS` 只有 20 列，`extrasaction="ignore"` 会静默丢弃其余字段），要从步骤 3 产出的 `<公司内路径>/day4/gate8/hot_expert_rank<r>.jsonl` 逐行读：`expert_counts` 之和 == `token 数 × top_k × 层数`，`aux_loss_unit_coef` 落在 `[1, E]` 之间。
- 不对时先查：`mean_load_ratio` 不等于 1.0 → 统计口径写错了，这张表的所有趋势都不用看；先核对 `expert_fractions` 是不是对所有 MoE 层求和之后再归一化的。在 JSONL 里找不到 `expert_counts` → 步骤 3 那次是不是走了 `sweep` 子命令（它传 `jsonl=None`，不落逐步记录），只有 `stats` 子命令才写 JSONL。

### 5. 读二维表：均衡、熵、语言损失、等待（20 分钟）· `可直接执行`

对每一格算四个抽象量：`max_load_ratio`（落在 `[1, E]`）、`router_entropy`（落在 `[0, log E]`，E=4 时 `log E` = 1.386）、`logits_loss_ratio`（相对 `bias=0, coef=0` 那一格的比值）、`wait_ms_ratio`（相对同一基线的比值）。

- 预期（趋势为 `估算`）：`bias` 增大 → `max_load_ratio` 从 ≈1 升向 E、`router_entropy` 从 ≈`log E` 降向 0、`wait_ms_ratio` 上升；`aux_coef` 增大 → `max_load_ratio` 回落到 ≈1，但 `logits_loss_ratio` 可能变差且不收窄。
- 不对时先查：`aux_coef=0.1` 那一列的 `max_load_ratio` ≈1 而 `logits_loss_ratio` 明显 > 1 → 这正是故障树 F7 描述的形态，看 `router_entropy` 是不是被推到接近 `log E`——那意味着 router 被强制均匀分配，均衡是约束条件而不是优化目标。

### 6. 回填证据字段（5 分钟）· `可直接执行`

按 `02_LAB_GUIDE.md` 第 6.6 节把四张二维表压成证据字段。只写比值、布尔和计数，绝对 loss、绝对耗时和拓扑细节留在公司内。

- 预期：`max_load_ratio` / `router_entropy` / `logits_loss_ratio` / `wait_ms_ratio` 四张 5×4 表齐全，`mean_load_ratio_is_one` 为 true，`expert_fractions_sum_is_one` 为 true，`hot_expert_reproduced` 为 true，`sweep_cells_completed` == 20。

- 不对时先查：`sweep_cells_completed` < 20 → 回步骤 2 看哪一格的 `torchrun` 失败；`hot_expert_reproduced` 为 false → 回步骤 3 确认 `--expert-id` 指的是真的被路由到的专家。

### 7. 写出今天的一句话结论（5 分钟）· `模板`

按这个形式写一行，填进证据的 `notes`：

`在 bias=<值> 造出的热点下，aux_coef 从 <a> 提到 <b> 使 max_load_ratio 从 <x> 回落到 <y>，同时 logits_loss_ratio 变为 <z>；主指标是 logits_loss_ratio，判定为 <改善 | 白赚 | INCONCLUSIVE>。`

- 预期：这句话里每一个数都是比值或落在已知区间的无量纲量，没有绝对 loss、没有绝对毫秒。
- 不对时先查：如果你写不出主指标的取值，说明扫描里缺了 `bias=0, coef=0` 这个对照基线——没有它，`logits_loss_ratio` 的分母不存在，这一天只能判 `INCONCLUSIVE`。

## 公司路径差异

- 全部步骤在公司 8×V100 上执行；`--out-dir` 与 `--csv` 都指向公司内部路径。
- **fp16 不是 bf16**：`run_moe_sweep.sh` 的 `DTYPE` 默认 `float16`，落到两条 `torchrun` 命令上就是 `--dtype float16`；改成 `bfloat16` 会在建模型之前被守卫拒绝。
- 第一次跑本卡的 `run_*` 脚本时先用 `MAX_STEPS=2` 与缩短的两个 list 试线路（步骤 1），因为脚本里的 `torchrun` 命令在写课程的机器上无法端到端执行。
- CSV 原文、`hot_expert_rank<r>.jsonl` 原文、绝对 loss 与绝对 `wait_ms` 全部留在公司内；离开公司机器的只有下面证据字段里的二维比值表与布尔。

## 今日自测（闭卷，2 题，只记录能/不能）

1. `M02-D-03`（step time 变长、EP 组内一个 rank 计算高其余等待高——四步定位路径与"修复是不是白赚"的主指标）
2. `M02-P-01`（aux 系数消融的假设、主副指标、对照基线、失败边界）

## 证据字段（填完即可归档到 memory/cc/evidence/）

```json
{
  "date": "YYYY-MM-DD",
  "week": 2,
  "day": 4,
  "card": "day4",
  "skill": "router_load_balance_ablation",
  "env": "v100",
  "ai_level": "A1",
  "config_id": "v100_moe.json + bias-list 0,0.5,1,2,4 + aux-coef-list 0,0.001,0.01,0.1 + fp16",
  "primary_artifact": "<公司内路径>/day4/gate8/moe_sweep.csv 汇成的 bias × aux_coef 二维抽象表",
  "observations": {
    "mean_load_ratio_is_one": "true | false",
    "expert_fractions_sum_is_one": "true | false",
    "max_load_ratio_bias0_coef0": "<落在 [1,E]>",
    "max_load_ratio_bias4_coef0": "<落在 [1,E]>",
    "max_load_ratio_bias4_coef0.1": "<落在 [1,E]>",
    "router_entropy_bias0_coef0": "<落在 [0,log E]>",
    "router_entropy_bias4_coef0": "<落在 [0,log E]>",
    "router_entropy_bias4_coef0.1": "<落在 [0,log E]>",
    "logits_loss_ratio_bias4_coef0": "<相对基线的比值>",
    "logits_loss_ratio_bias4_coef0.1": "<相对基线的比值>",
    "wait_ms_ratio_bias4_coef0": "<相对基线的比值>",
    "hot_expert_reproduced": "true | false",
    "sweep_cells_completed": 20
  },
  "status": "PASS | FAIL-MODEL | INCONCLUSIVE",
  "failure_class": "none | model | measurement | numeric | insufficient",
  "self_check": {"M02-D-03": "能 | 不能", "M02-P-01": "能 | 不能"},
  "time_spent_min": 0,
  "notes": "<步骤 6 的那一句结论，只含比值与区间值>"
}
```
