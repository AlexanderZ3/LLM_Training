# Week M03 · Day 3 — 先闭卷手算每卡四项字节账，再 FSDP 实测，最后 8→4 卡 reshard

| 字段 | 值 |
| --- | --- |
| 主要产物 | `$MM_RUNS_ROOT/day3_dist/byte_ledger.md`（手算列 + 实测列 + 偏差列） |
| 估时 | `120 分钟`（步骤估时之和） |
| 环境 | `公司 V100` |
| AI 辅助等级要求 | `A0`（步骤 1 闭卷手算，限时 25 分钟，不看代码不看文档）；步骤 2–7 是 `A2` |
| 前置 | Day 2 证据：`equiv_ws8_max_abs_delta < 1e-4`、`all_detected = true` |
| 本卡对应门 | `有界训练` → `checkpoint/resume 等价` |
| 可裁剪项 | 步骤 6（resume 等价，单进程 fp32，Day 5 还会再碰恢复）。**步骤 1 闭卷手算与步骤 5 的 8→4 reshard 都不可裁** |

所有命令在周包根目录 `week03_minimind_v100_Only/` 下执行。

## 为什么做

字节账是三本账里唯一决定"能不能跑"的一本，也是 Gate A0 唯一的闭卷题。**先算完再开机**：先看实测数再"手算"就变成了抄，而 Gate 考的正是没有机器时你能不能把这四项写出来。算完之后 FSDP 实测负责验证，8→4 reshard 负责回答"这份分片 checkpoint 换个卡数还认不认"——这是唯一能扩展到 7B/27B 的 checkpoint 路径。

## 步骤

### 1. 闭卷手算四项字节账（25 分钟）· `伪代码` · **不可裁，先算完再开机**

关掉终端，不看 `lab/` 的代码，不看 `01_FOUNDATIONS.md`。输入只有 `lab/configs/v100_768.json` 里这几个字段：`hidden_size=768`、`num_hidden_layers=8`、`num_attention_heads=8`、`num_key_value_heads=4`、`vocab_size=6400`、`intermediate_size=2432`、`tie_embeddings=true`；再加 `world_size=8`、`FULL_SHARD`、`batch_per_rank=8`、`seq_len=512`、参数常驻 fp32、autocast 计算 fp16、AdamW 两个矩、FSDP 单元数 = 层数 + 1。

写文件 `$MM_RUNS_ROOT/day3_dist/hand_calc.md`，五行，每行"一个数 + 一条公式"：

1. `P`（参数总量，记得 tied embedding 让 `lm_head` 计 0）
2. `param_bytes`（每卡）
3. `grad_bytes`（每卡）
4. `optim_bytes`（每卡；AdamW 除了两个矩，还给**每个参数张量**存一个 fp32 的 step 标量）
5. `activation_bytes`（每卡；这一项是估算，写出它对 `B`、`T`、`L`、`V` 的标度就算过）

- 预期：`hand_calc.md` 出现，五行都有数、都有公式，全程没有运行任何脚本。
- 不对时先查：写不出第 4 行时回到"静态字节 = 16P"这个分解（param 4 + grad 4 + Adam 两个矩 8），再单独补 per-tensor 的 step 标量那一项。

### 2. 对答案：`--hand-only` 手算表 + 通信账（15 分钟）· `可直接执行`

```bash
python lab/scripts/byte_ledger.py --config v100_768 --hand-only \
    --world-size 8 --mode full_shard --batch-per-rank 8 --seq-len 512 \
    --param-dtype float32 --compute-dtype float16 --show-collectives \
    --run-tag day3_dist
```

- 预期（这五个数是恒等式，不是估算，可以逐位核对）：`P = 63,912,192 参数`；`param_bytes 31,956,096 B`；`grad_bytes 31,956,096 B`；`optim_bytes 63,912,552 B`；`all_gather_buffer 14,202,709 B`。通信账那一段是 `expected_all_gather_calls: 18`、`expected_reduce_scatter_calls: 9`（合计 27 次/step）。`byte_ledger_hand.json` 出现。
- 不对时先查：自己那一列的 `optim_bytes` 与它差 **360 B** 时，就是漏了 AdamW 的 per-tensor step 标量——这个模型有 90 个参数张量，90 × 4 = 360。

### 3. FSDP 实测：8 卡 FULL_SHARD 有界训练（25 分钟）· `可直接执行`

```bash
torchrun --nproc_per_node 8 lab/scripts/train_bounded.py \
    --config v100_768 --run-tag day3_dist \
    --placement fsdp --sharding full_shard --dtype float16 \
    --global-batch 64 --accum 1 --max-steps 8 --log-interval 1 \
    --count-collectives 1 --seed-per-rank 0 --log-name fsdp_full_shard
```

fp16 + FSDP 时脚本会自动换成 `ShardedGradScaler`（普通 `GradScaler` 只看本 rank 那一片梯度，在 `FULL_SHARD` 下本身就是一个静默 bug）。

- 预期：`fsdp_full_shard_rank0.jsonl` 出现，header 行里 `n_fsdp_units` 为 **9**（8 个 block + 1 个 root）、`world_size` 为 8、`micro_batch` 为 8；8 行 step 记录，每行 `peak_mem_mb` 大于 0。
- 不对时先查：`n_fsdp_units` 不是 9 —— auto-wrap policy 没生效，FSDP 退化成单一 root unit，那时前向峰值等于整份模型，字节账全部作废（`01_FOUNDATIONS.md` §9.1 第 9 条）。

### 4. 字节账实测列，生成主要产物（15 分钟）· `可直接执行`

```bash
python lab/scripts/byte_ledger.py --config v100_768 --device cuda:0 \
    --batch-per-rank 8 --seq-len 512 \
    --param-dtype float32 --compute-dtype float16 --run-tag day3_dist
```

实测是单进程整份账，所以对照列用的是 `world_size=1` 的 DDP 手算（脚本自己会重算一份 `hand_single_rank`），不是步骤 2 那张 `FULL_SHARD` 表。

- 预期：`byte_ledger.md` 出现，四行表里 `param_bytes` = `grad_bytes` = `255,648,768`、`optim_bytes` = `511,297,896`，这三行判定都是"一致"；`activation_bytes` 那一行判定是"一致（估算项，只看量级）"，比值落在 0.25–4.0。退出码 0。
- 不对时先查：`param_bytes` 行不一致时先看 tie 是否生效——`lm_head.weight` 与 `embed_tokens.weight` 应当是同一个对象，去重后只算一次。

### 5. 8 卡写分片 checkpoint，4 卡读回来（20 分钟）· `可直接执行` · **不可裁**

```bash
python lab/scripts/reshard_check.py --config v100_768 --format flat \
    --save-nproc 8 --load-nproc 4 --max-steps 3 \
    --global-batch 16 --accum 1 --tol 1e-4 --run-tag day3_dist
```

这个脚本的写端固定用 gloo + CPU 起 8 个进程（读代码可见），所以它验的是**分片边界、余数分配、step 换算、loss 连续性**这四件事，不是 NCCL。把 `--global-batch` 压到 16 只为控制 CPU 时间，不影响分片逻辑。

- 预期：`saved_ws=8 -> load_ws=4`；`分片长度（新 world_size）= [...]  合计等于总长=True`；`step 换算：3 (ws=8) -> 6 (ws=4)`；末行 `loss 连续性：... |delta|=… tol=1.0e-04 -> PASS`；`reshard_check.json` 出现。
- 不对时先查：`ckpt/reshard/meta.json` 里的 `world_size` 与 `total_numel` 这两个字段——报 `缺少分片：shard_N.pt` 或 `扁平向量长度 … 与索引表要求的 … 不一致` 都从这里查起。

`step 换算` 那一行只保证"消费的样本数不变"。要同时保住全局批，`accum` 必须跟着翻倍（8→4 时 `A' = 2A`），否则 Adam 在同样的数据上做了两倍的更新——而 loss 曲线照样平滑下降，看不出来（`01_FOUNDATIONS.md` §8.2）。

### 6. resume 等价：连续训练 vs 断点恢复（10 分钟）· `可直接执行`

```bash
python lab/scripts/resume_check.py --config v100_768 --stage pretrain \
    --dtype float32 --total-steps 6 --break-at 3 \
    --global-batch 8 --accum 1 --tol 1e-6 --run-tag day3_dist
```

- 预期：逐步对照表里每一行 `|delta|` 都是 `0.000e+00`；末行 `compared_steps=6  max|delta|=0.000e+00  tol=1.0e-06 -> PASS`（本机实测 2026-09-08：单进程 fp32 下逐步 loss 差恰好 0.0）；`resume_check.json` 出现。
- 不对时先查：`|delta|` 在 1e-2 量级时，先看 checkpoint 里 `optimizer` / `scaler` / `rng` / `data_cursor` 四样是不是都存了（`save_checkpoint` 的 payload 里应当四样齐全）。

### 7. 生成当天证据（10 分钟）· `可直接执行`

```bash
python lab/scripts/make_evidence.py --week 3 --day 3 --card day3 \
    --run-tag day3_dist --skill byte_ledger --env v100 --ai-level A0 \
    --status PASS --failure-class none --time-spent 120 \
    --primary-artifact "每卡四项字节账的手算列/实测列/偏差列 + 8→4 reshard 的连续性判定" \
    --config-id "v100_768 / fsdp full_shard / float16 / world_size=8"
```

- 预期：`evidence.json` 出现在 `$MM_RUNS_ROOT/day3_dist/`，`observations` 里有 `params_total`、`n_fsdp_units`、`exact_items_match`、`activation_ratio`、`saved_world_size`、`load_world_size`、`converted_step`、`slices_sum_equals_total`。
- 不对时先查：`exact_items_match` 没出现时，核对步骤 4 是否真的写出了 `byte_ledger.json`（步骤 2 的 `--hand-only` 只写 `byte_ledger_hand.json`，不含对照）。

`hand_calc_done_before_run` 这个字段由你自己填，它记录的是"步骤 1 是否发生在任何运行之前"。填 `false` 不扣分，但那次 Gate 的 A0 项不成立。

## 公司路径差异

- 步骤 3 必须用 `torchrun`；单进程 FSDP 也要 `torchrun --nproc_per_node 1`，否则脚本会退出并告诉你"FSDP 需要已初始化的进程组"。
- 步骤 5 与步骤 6 在 CPU 上跑，不占 GPU 排队；步骤 3、4 占 GPU。
- 8 卡非独占时 `step_time_s` 与 `tokens_per_s` 全部标 `measurement` 类不确定，结论只用 `peak_mem_mb` 的比值与 `collectives_total`。
- `byte_ledger.md`、`fsdp_full_shard_rank0.jsonl`、分片 checkpoint 全部留在公司内；只把下面证据字段里的抽象值填好。

## 今日自测（闭卷，2 题，只记录能/不能）

1. （Apply）给定 `hidden=768, layers=8, heads=8, kv_heads=4, vocab=6400, intermediate=2432, tie=True, world_size=8`，30 分钟内写出 DDP 与 FSDP `FULL_SHARD` 下每卡的四项字节账，以及一个 step 内的集合通信序列（类型 + 次数 + 单次张量字节量）。
2. （Trade-off）在这个 64M 模型上 FSDP 相对 DDP 省下多少常驻字节？为什么"省显存"不是在这里用 FSDP 的理由？如果目标真的是省显存，杠杆应该按在哪里？

## 证据字段（填完即可归档到 `memory/cc/evidence/`）

```json
{
  "date": "2026-09-DD",
  "week": 3,
  "day": 3,
  "card": "day3",
  "skill": "byte_ledger",
  "env": "v100",
  "ai_level": "A0",
  "config_id": "v100_768 / fsdp full_shard / float16 / world_size=8",
  "primary_artifact": "day3_dist/byte_ledger.md：手算列 + 实测列 + 偏差列",
  "observations": {
    "hand_calc_done_before_run": true,
    "hand_params_total_correct": true,
    "hand_optim_bytes_correct": true,
    "params_total": 63912192,
    "n_fsdp_units": 9,
    "expected_collectives_per_step": 27,
    "exact_items_match": true,
    "activation_ratio": 0.0,
    "saved_world_size": 8,
    "load_world_size": 4,
    "converted_step": 6,
    "slices_sum_equals_total": true,
    "reshard_loss_continuity_pass": true,
    "resume_max_abs_delta": 0.0
  },
  "status": "PASS",
  "failure_class": "none",
  "self_check": {"q1": "能", "q2": "能"},
  "time_spent_min": 120,
  "notes": "闭卷手算在任何运行之前完成；手算与实测的 param/grad/optim 三项逐位相等"
}
```
