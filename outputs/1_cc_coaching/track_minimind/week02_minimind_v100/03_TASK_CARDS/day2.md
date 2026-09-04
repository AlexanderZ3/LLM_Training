# Week M02 · Day 2 — 用同一个全局批量出 FSDP 三种分片策略对 DDP 的比值，再把 8 卡的分片 checkpoint 换到 4 卡上恢复

| 字段 | 值 |
| --- | --- |
| 主要产物 | 一张四行比值表（`ddp` / `full_shard` / `shard_grad_op` / `no_shard` 的 `step_time_ratio`、`mem_ratio`、`collectives_total`） |
| 次要产物 | `verify_8to4.json` 的 `pass` 与 `abs_delta`（门 6；时间不够时可推到下一次，主要产物不受影响） |
| 估时 | `120 分钟`（步骤估时之和） |
| 环境 | `公司 V100` |
| AI 辅助等级要求 | `A1`（比值可以照算，但"64M 上 FSDP 更慢"的因果链要能自己讲完，它是 Gate 的 A1 题） |
| 前置 | Day 1 的证据：`equiv_pass = true`、`micro_batch_identical = true`（DDP 基线成立，才有资格当分母） |
| 本卡对应门 | `有界训练` → `resume` |
| 可裁剪项 | 步骤 5（activation checkpointing 对照）与步骤 2 里的 `no_shard`（它只是对照的对照）；步骤 2、3、6 不可裁 |

## 为什么做（三句以内）

DDP 把完整参数、完整梯度、完整优化器状态放在每一张卡上，FSDP 把这三样都切成 1/8——今天要量的是这一步换来了什么、又付出了什么。同时，分片 checkpoint 能不能换一个 world_size 加载，决定了本周 Day 5 的故障恢复演练有没有地基。预期结论是 64M 这个尺寸上 FSDP 比 DDP 慢；**这是结论，不是失败**，写进证据时状态照样是 `PASS`。

## 步骤

### 1. 先用 `MAX_STEPS=2` 把 FSDP 脚本的线路跑通（10 分钟）· `模板`

`train_fsdp.py`、`ckpt_reshard.py` 的训练路径在写课程的那台机器上**跑不了**：FSDP1 需要 CUDA 设备，参数全在 CPU 上时 torch 的 `_init_device_handle` 会回落到 `torch.cuda.current_device()` 并抛 `FSDP needs a non-CPU accelerator device`（torch 2.1.0 与写课程机器上的版本行为一致）。所以本卡的两个 `run_*` 脚本都先压到 2 步：

```bash
export MINIMIND_ROOT=<MiniMind 仓库绝对路径>
OUT_DIR=<公司内路径>/day2/wiring MAX_STEPS=2 bash lab/scripts/run_fsdp_vs_ddp.sh
```

- 预期：四次 `torchrun`（DDP + 三种 sharding）都进入训练循环并各写 2 行；FSDP 的 header 里 `fsdp_units` == 9（8 个 `MiniMindBlock` + 根）。
- 不对时先查：`fsdp_units` == 1 → `auto_wrap_policy` 没切开任何 block，整个模型是一个 FSDP 单元，后面所有的数都不用看了（故障树 F9），先确认 `transformer_auto_wrap_policy` 里的类是 `common.import_minimind()` 拿到的同一个类对象。

### 2. DDP 基线 + FSDP 三种策略各跑 20 步（30 分钟）· `模板`

```bash
OUT_DIR=<公司内路径>/day2/gate5 MAX_STEPS=20 bash lab/scripts/run_fsdp_vs_ddp.sh
```

脚本的 `DTYPE` 默认就是 `float16`，它原样传给四条 `torchrun` 命令里的 `--dtype float16`；把它改成 `bfloat16` 会被守卫显式拒绝。

- 预期：`<公司内路径>/day2/gate5/` 下出现 `ddp_ws8_rank0.jsonl` 与 `fsdp_{full_shard,shard_grad_op,no_shard}_ws8_rank0.jsonl`；四次运行的 `global_batch` 都是 64、`accum` 都是 1；总耗时 8–12 分钟（`估算`）。
- 不对时先查：单次运行超过 12 分钟就 Ctrl-C，先把 `MAX_STEPS` 降到 10；仍然超时就去掉 `no_shard` 那一次。

### 3. 核对 `fsdp_units` 与 `collectives_total`（15 分钟）· `可直接执行`

从四条 JSONL 的 header 与 step 行里读出 `fsdp_units`、`collectives_total`、`collectives`（按类型的字典）。

- 预期（全部 `估算`，来自 `01_FOUNDATIONS.md` 3.3/3.4）：`fsdp_units` == 9；`collectives_total` DDP 约 11（`bucket_cap_mb=25`，255.6 MB 梯度）、`FULL_SHARD` 约 27（forward all_gather 9 + backward all_gather 9 + reduce_scatter 9）、`SHARD_GRAD_OP` 约 18（forward 后不 reshard，backward 前不再 all_gather）、`NO_SHARD` 与 DDP 同量级。
- 不对时先查：`collectives` 里出现明显翻倍的计数 → `CollectiveCounter` 只包公共名，`_all_gather_base` 在 torch 2.1 里是公共名的 deprecated 包装，两个都包会重复计数。

### 4. 算出比值表（15 分钟）· `可直接执行`

对每种 sharding 算 `step_time_ratio = fsdp.step_time_s / ddp.step_time_s` 与 `mem_ratio = fsdp.peak_mem_mb / ddp.peak_mem_mb`；`step_time_s` 从第 3 步起算平均（第 1 步含 CUDA 上下文与 cudnn 预热）。

- 预期（`估算`）：`mem_ratio_full_shard_vs_ddp` **< 1**；`step_time_ratio_full_shard_vs_ddp` **> 1**。后者不是字节量问题——每 rank 实际字节 FSDP 335 MB < DDP 447 MB——而是 27 次对 11 次的次数与延迟问题，且 all_gather 必须在层计算之前完成、掩盖不住。DDP 与 FULL_SHARD 的 loss 在 fp16 下用 `--compare-tol 1e-2` 比对，脚本已经跑了这一条。
- 不对时先查：`mem_ratio` ≥ 1 → 回到故障树 F9：先看 `fsdp_units`，再确认 `limit_all_gathers=True`，再用 `--sharding no_shard` 作对照；如果 `no_shard` 就已经比 DDP 高，问题不在分片策略。

### 5. activation checkpointing 开/关对照（15 分钟）· `模板`

```bash
OUT_DIR=<公司内路径>/day2/gate5_ac AC=1 MAX_STEPS=20 bash lab/scripts/run_fsdp_vs_ddp.sh
```

- 预期：header 里 `activation_checkpointed_blocks` == 8；同配置下 `ac_mem_ratio`（AC 开 / AC 关的 `peak_mem_mb`）< 1，`ac_step_time_ratio` > 1（多一次前向）。
- 不对时先查：`activation_checkpointed_blocks` 为 0 → `--activation-checkpointing` 传的是 `1` 而不是 `0`，脚本靠 `AC` 环境变量传这个值。

### 6. 8 卡保存分片 checkpoint，4 卡恢复并 verify（30 分钟）· `模板`

```bash
OUT_DIR=<公司内路径>/day2/gate6 SAVE_NPROC=8 LOAD_NPROC=4 TOL=1e-2 \
  bash lab/scripts/run_reshard.sh
# 回退路径也跑一次（FULL_STATE_DICT + rank0_only）
OUT_DIR=<公司内路径>/day2/gate6_full FORMAT=full TOL=1e-2 bash lab/scripts/run_reshard.sh
```

- 预期：`<公司内路径>/day2/gate6/verify_8to4.json` 里 `"pass": true`、`abs_delta < tol`、`saved_world_size` 8、`current_world_size` 4；日志里有一行 `[reshard] GPU 数量变化(8→4)，... step 20 -> 40`，核对 `40 == 20 * 8 // 4`。
- 不对时先查：加载报 `size mismatch` 或 `flat_param` 长度不匹配 → 按故障树 F4：先看 `meta.json` 的 `format` 与恢复时用的是否一致，再确认恢复时的 wrap 结构（`--sharding`、config、`--activation-checkpointing`）与保存时完全相同——分片是 `(world_size, wrap 粒度)` 的函数；优化器状态必须走 `FSDP.optim_state_dict_to_load`。

### 7. 填证据字段（5 分钟）· `可直接执行`

对着 `02_LAB_GUIDE.md` 第 6.3 与 6.4 节填：`fsdp_units`、四个 `collectives_total`、四个比值、`ac_blocks`、`ac_mem_ratio`、`ac_step_time_ratio`；以及 `saved_world_size`、`current_world_size`、`format`、`abs_delta`、`tol`、`verify_pass`、`within_tail_band`、`converted_step_correct`。

- 预期：所有性能字段都是**比值**，没有一个绝对秒数或绝对 MB。
- 不对时先查：`step_time_ratio > 1` 时别急着判 `FAIL`——64M 上 FSDP 更慢是本周的预期结论，只要 `mem_ratio < 1` 且 `fsdp_units == 9`，状态就是 `PASS`。

## 公司路径差异

- 全部步骤在公司 8×V100 上执行；FSDP 需要 CUDA，写课程的机器上**无法验证**这条链路的任何一段（只有 `torch.distributed.checkpoint` 的 2 进程写、1 进程读这个地基在那台机器上过了）。所以本卡的每一个数字都是这台机器上的第一次实测。
- **fp16 不是 bf16**：`run_fsdp_vs_ddp.sh` 与 `run_reshard.sh` 的 `DTYPE` 默认 `float16`，落到命令上就是 `--dtype float16`；`TOL=1e-2` 就是为 fp16 定的，换成 `DTYPE=float32` 时用 `1e-4`。
- 日志、checkpoint 文件、profiler 结果、绝对显存与绝对 step time 全部留在公司内；离开公司机器的只有下面证据字段里的比值、布尔与计数。

## 今日自测（闭卷，2 题，只记录能/不能）

1. `M02-D-02`（8→4 卡恢复的 `size mismatch` 与 `step` 换算两个失败；三种 `StateDictType` 的代价与选择）
2. `M02-T-03`（64M 上 FSDP 比 DDP 慢、27B 上分片必须——完整因果链与转折点估算）

## 证据字段（填完即可归档到 memory/cc/evidence/）

```json
{
  "date": "YYYY-MM-DD",
  "week": 2,
  "day": 2,
  "card": "day2",
  "skill": "fsdp_vs_ddp_and_checkpoint_reshard",
  "env": "v100",
  "ai_level": "A1",
  "config_id": "v100_dense.json + G=64 + fp16 + wrap=MiniMindBlock",
  "primary_artifact": "<公司内路径>/day2/gate6/verify_8to4.json + 四行 FSDP/DDP 比值表",
  "observations": {
    "fsdp_units": 9,
    "collectives_total_ddp": 0,
    "collectives_total_full_shard": 0,
    "collectives_total_shard_grad_op": 0,
    "collectives_total_no_shard": 0,
    "step_time_ratio_full_shard_vs_ddp": "<比值，预期 >1>",
    "step_time_ratio_shard_grad_op_vs_ddp": "<比值>",
    "mem_ratio_full_shard_vs_ddp": "<比值，预期 <1>",
    "mem_ratio_shard_grad_op_vs_ddp": "<比值>",
    "ac_blocks": 8,
    "ac_mem_ratio": "<比值，预期 <1>",
    "ac_step_time_ratio": "<比值，预期 >1>",
    "saved_world_size": 8,
    "current_world_size": 4,
    "format": "sharded | full（取自你传的 FORMAT=，cmd_verify 的结果字典里没有这一项）",
    "abs_delta": "<数量级>",
    "tol": "1e-2",
    "verify_pass": "true | false",
    "within_tail_band": "true | false",
    "converted_step_correct": "true | false"
  },
  "status": "PASS | FAIL-SYSTEM | INCONCLUSIVE",
  "failure_class": "none | system | measurement | numeric | insufficient",
  "self_check": {"M02-D-02": "能 | 不能", "M02-T-03": "能 | 不能"},
  "time_spent_min": 0,
  "notes": "<一句话：FSDP 更慢是否成立、慢在次数还是字节，不含绝对秒数与绝对 MB>"
}
```
