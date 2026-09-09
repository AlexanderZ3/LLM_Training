# Week M03 · Day 4 — 有界 pretrain → SFT，落盘 checkpoint，独立 eval（不看 loss）

| 字段 | 值 |
| --- | --- |
| 主要产物 | `$MM_RUNS_ROOT/day4_profile/eval_before.jsonl` + `eval_after.jsonl` 这一对并排对照，以及它们所依赖的 `$MM_RUNS_ROOT/day4_profile/ckpt/final.pt` |
| 估时 | `110 分钟`（步骤估时之和） |
| 环境 | `公司 V100`（步骤 1 在本机 CPU 上做） |
| AI 辅助等级要求 | `A2`；步骤 5 对三个定性观测的解读要求 `A1` |
| 前置 | Day 3 证据：`exact_items_match = true`、`reshard_loss_continuity_pass = true`、`n_fsdp_units = 9` |
| 本卡对应门 | `有界训练` → `独立 eval` |
| 可裁剪项 | 步骤 2（训练前 eval）——可以只做训练后的一次，但那样就失去了对照。**步骤 4 的 checkpoint 落盘不可裁**，Day 5 要用它 |

所有命令在周包根目录 `week03_minimind_v100_Only/` 下执行。今天训练和落盘的都是 `lab/src/mm_v100/model.py` 里自带的 `TinyCausalLM`，与 MiniMind 形状对齐但权重是它自己训出来的；`00_WEEK_CARD.md` 第 4 节里写的 `full_sft_768.pth` 是 MiniMind 的命名习惯，本 lab 的实际文件名是 `ckpt/final.pt`，`eval_compare.jsonl` 对应的是 `eval_before.jsonl` 与 `eval_after.jsonl` 这一对。

## 为什么做

前三天所有判据都是数字：loss 差、字节数、通信次数。它们能证明"管线在按你写的目标优化"，不能证明"这个目标是你想要的那个"（`01_FOUNDATIONS.md` §5.3）。今天引入一个与 loss 完全无关的判据：固定 8 条 prompt 的贪心生成，看 `eos_rate` / `role_leak_rate` / `echo_ratio` 三个定性观测在 SFT 前后怎么变。同时把 Day 5 的输入落盘。

## 步骤

### 1. 本机 CPU 门：eval 与数据契约（10 分钟）· `可直接执行`

```powershell
conda activate rfm
python -m pytest lab/tests/test_eval_generate.py lab/tests/test_data_contract.py -q
```

- 预期：一行 `N passed`，没有 `failed`。
- 不对时先查：第一个 `FAILED` 的用例名——若在 `test_eval_generate.py` 的 `role_leak` 相关用例，说明 ChatML 模板的角色标记被改过，那会让步骤 5 的 `role_leak_rate` 失去意义。

### 2. 训练前 eval：随机初始化的行为基线（15 分钟）· `可直接执行`

```bash
python lab/scripts/eval_compare.py --config v100_768 --tag before \
    --mode chat --max-new-tokens 32 --device cuda:0 --seed 42 \
    --run-tag day4_profile
```

- 预期：先打一行 `[eval] tokenizer=…`（说明用的是内置 `SimpleTokenizer` 还是 `$MM_WEIGHTS_ROOT/minimind_tokenizer`）；8 条 prompt 的生成结果；`eval_before.jsonl` 与 `eval_before_summary.json` 出现。随机初始化下 `eos_rate` 应当接近 0.000（`估算`：未训练模型基本不会生成结束符）。
- 不对时先查：`[eval] tokenizer=` 那一行。前后两次 eval 必须用同一个 tokenizer 来源，否则 `echo_ratio` 与 `role_leak_rate` 不可比。

### 3. 有界 pretrain：8 卡 DDP，20 步（25 分钟）· `可直接执行`

```bash
torchrun --nproc_per_node 8 lab/scripts/train_bounded.py \
    --config v100_768 --out-dir $MM_RUNS_ROOT/day4_pretrain \
    --placement ddp --stage pretrain --dtype float16 \
    --global-batch 64 --accum 1 --max-steps 20 --log-interval 1 \
    --seed-per-rank 0 --save-final 1 --log-name metrics_pretrain
```

写到独立的 `$MM_RUNS_ROOT/day4_pretrain`，是因为 `--save-final 1` 固定落在 `<out-dir>/ckpt/final.pt`，与步骤 4 的 SFT 产物同名。

- 预期：`[header]` 一行里 `world_size` 为 8、`micro_batch` 为 8、`dtype` 为 float16；20 行 step 记录；`$MM_RUNS_ROOT/day4_pretrain/ckpt/final.pt` 出现。
- 不对时先查：header 里的 `micro_batch` 不是 8 —— 它应当等于 `global_batch / (world_size × accum)` = 64/8/1；不是 8 说明 `--global-batch` 或 `--accum` 传错了。

### 4. 有界 SFT：从 pretrain 的权重接着训，并落盘（25 分钟）· `可直接执行` · **不可裁**

```bash
torchrun --nproc_per_node 8 lab/scripts/train_bounded.py \
    --config v100_768 --run-tag day4_profile \
    --placement ddp --stage sft --dtype float16 \
    --global-batch 64 --accum 1 --max-steps 40 --log-interval 1 \
    --seed-per-rank 0 --save-final 1 --log-name metrics_sft \
    --resume $MM_RUNS_ROOT/day4_pretrain/ckpt/final.pt
```

`--resume` 会把 model / optimizer / scaler / RNG 一起恢复，所以这一轮从第 21 步开始、跑到第 40 步；`--max-steps 40` 同时是 cosine 调度的分母，两轮必须用不同的 total 才能让 lr 相位连上。`--stage sft` 让前 128 个位置当 prompt 被屏蔽（`label_ignore_prefix`），这正是 `role_leak` 要检验的边界。

- 预期：先打 `[resume] 从 …/day4_pretrain/ckpt/final.pt 恢复，step=20 data_cursor={…}`；step 21 到 40 共 20 行；`$MM_RUNS_ROOT/day4_profile/ckpt/final.pt` 出现。
- 不对时先查：`[resume]` 那一行的 `step=` 不是 20 —— 说明加载的不是步骤 3 落盘的那个文件，核对 `--resume` 的路径。

### 5. 训练后 eval 与并排对照（20 分钟）· `可直接执行`

```bash
python lab/scripts/eval_compare.py --config v100_768 --tag after \
    --mode chat --max-new-tokens 32 --device cuda:0 --seed 42 \
    --run-tag day4_profile --checkpoint $MM_RUNS_ROOT/day4_profile/ckpt/final.pt

python lab/scripts/eval_compare.py \
    --compare $MM_RUNS_ROOT/day4_profile/eval_before.jsonl \
              $MM_RUNS_ROOT/day4_profile/eval_after.jsonl
```

- 预期：第一条命令打 `[eval] 载入 …/ckpt/final.pt`，`eval_after.jsonl` 与 `eval_after_summary.json` 出现；第二条命令末尾打一行 JSON，含 `identical_ratio`、`a_eos_rate`、`b_eos_rate`、`a_role_leak_rate`、`b_role_leak_rate`。`identical_ratio` 必须**小于 1.0**（40 步之后生成必须已经变了）。
- 不对时先查：`identical_ratio` 等于 1.0 时，看第一条命令有没有打出 `[eval] 载入` 那一行——没打就是 `--checkpoint` 没传到，评的还是随机初始化。

这三个数是**定性判据**，不是分数。64M + 40 步有界训练下不存在可比的质量分；把它们写成分数就是把 smoke 提升成业务结论。

### 6. 生成当天证据（15 分钟）· `可直接执行`

```bash
python lab/scripts/make_evidence.py --week 3 --day 4 --card day4 \
    --run-tag day4_profile --skill independent_eval --env v100 --ai-level A2 \
    --status PASS --failure-class none --time-spent 110 \
    --primary-artifact "SFT 前后 8 条固定 prompt 的并排生成对照，以及它依赖的 SFT 权重" \
    --config-id "v100_768 / ddp / float16 / pretrain20+sft20"
```

- 预期：`evidence.json` 出现在 `$MM_RUNS_ROOT/day4_profile/`，`observations` 里有 `eos_rate`、`role_leak_rate`、`mean_new_tokens`、`mean_echo_ratio`，以及 `metrics_sft_rank0` 聚合出的 `loss_ratio_last_over_first` 与 `skipped_steps`。
- 不对时先查：`eos_rate` 只出现一份（应当有 before 与 after 两份摘要 JSON 各一份）时，核对步骤 2 与步骤 5 是不是都写到了 `day4_profile`。

## 公司路径差异

- 两轮训练都用 `torchrun --nproc_per_node 8` 与 `--dtype float16`；每轮 20 步、全局批 64、序列 512，单轮远低于 10 分钟的运行上限。
- eval 在单卡上跑（`--device cuda:0`），不需要 `torchrun`。
- 生成出来的文本、`ckpt/final.pt`、逐步日志全部留在公司内。生成文本尤其不要带出——它可能包含训练数据的片段。只把下面证据字段里的比率、计数、布尔填好。

## 今日自测（闭卷，2 题，只记录能/不能）

1. （Explain）为什么 loss 下降不能证明 SFT 的 prompt 掩码是对的？`role_leak_rate` 这个观测抓的是哪一类错误，它和 D2 的 `ρ` 有什么分工？
2. （Debug）SFT 之后 `eos_rate` 仍然是 0，但 loss 曲线正常下降。列出两个候选原因，并各给一条能把它们分开的最小检查。

## 证据字段（填完即可归档到 `memory/cc/evidence/`）

```json
{
  "date": "2026-09-DD",
  "week": 3,
  "day": 4,
  "card": "day4",
  "skill": "independent_eval",
  "env": "v100",
  "ai_level": "A2",
  "config_id": "v100_768 / ddp / float16 / pretrain20+sft20",
  "primary_artifact": "day4_profile 的 eval_before.jsonl 与 eval_after.jsonl 并排对照",
  "observations": {
    "pretrain_steps": 20,
    "sft_resumed_from_step": 20,
    "sft_steps": 20,
    "final_ckpt_written": true,
    "before_eos_rate": 0.0,
    "after_eos_rate": 0.0,
    "before_role_leak_rate": 0.0,
    "after_role_leak_rate": 0.0,
    "before_mean_echo_ratio": 0.0,
    "after_mean_echo_ratio": 0.0,
    "identical_ratio": 0.0,
    "sft_loss_ratio_last_over_first": 0.0,
    "sft_skipped_steps": 0
  },
  "status": "PASS",
  "failure_class": "none",
  "self_check": {"q1": "能", "q2": "能"},
  "time_spent_min": 110,
  "notes": "三个 eval 观测是定性判据，不作为质量分；Day 5 用 day4_profile/ckpt/final.pt 作起点"
}
```
