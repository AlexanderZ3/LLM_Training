# Week M03 · Day 5 — 低精度下的 DPO/GRPO 数值链 + kill 一个 rank 从分片 checkpoint 恢复

| 字段 | 值 |
| --- | --- |
| 主要产物 | `$MM_RUNS_ROOT/day5_rl/rl_numeric.json`（D1/D2/G2 三条断言 + G1 的组内统计） |
| 估时 | `120 分钟`（步骤估时之和） |
| 环境 | `公司 V100`（步骤 1 在本机 CPU 上做） |
| AI 辅助等级要求 | `A1`（只查官方文档）；步骤 5 的失败分类要求 `A1` |
| 前置 | Day 4 证据：`final_ckpt_written = true`、`identical_ratio < 1.0`；Day 3 证据：`slices_sum_equals_total = true` |
| 本卡对应门 | `smoke` → `eval` → `恢复/故障测试` |
| 可裁剪项 | 步骤 4（用 Day 4 权重做 RL 起点的行为基线）。**步骤 5 的 kill-rank 恢复不可裁** |

所有命令在周包根目录 `week03_minimind_v100_Only/` 下执行。本周不装 `vLLM` / `verl` / `OpenRLHF`——三者都把推理引擎当硬依赖，而推理引擎是整个生态里对 GPU 架构和 torch 版本最挑剔的一环（`00_WEEK_CARD.md` 3.3）。今天用的全是纯 PyTorch 的解析断言。

## 为什么做

64M 模型 + 有界步数下，DPO/GRPO 的"效果好不好"没有信息量。有信息量的是那些**不管模型多小、跑多少步都必须成立的等式**：初始 loss 等于 ln2、ref 模型的梯度是 None（不是 0）、第 0 步的重要性采样比恒等于 1。它们一旦不成立，说明管线接错了，而不是"效果不好"。最后再把 Day 3 的 reshard 推到它真正的用途：一个 rank 死了，用剩下的卡从分片 checkpoint 接着走。

## 步骤

### 1. 本机 CPU 门：RL 数值链的四组单测（15 分钟）· `可直接执行`

```powershell
conda activate rfm
python -m pytest lab/tests/test_rl_numeric.py lab/tests/test_rl_logprob.py lab/tests/test_rl_kl.py lab/tests/test_rl_advantage.py -q
```

- 预期：一行 `N passed`，没有 `failed`。
- 不对时先查：第一个 `FAILED` 的用例名——若在 `test_rl_logprob.py`，说明 token 级 log-prob 的 mask 约定被改过，那会让步骤 3 的 G2 断言失去意义。

### 2. DPO 两条断言的 smoke（10 分钟）· `可直接执行`

先用内置 fixture 跑 `--dry-run`，只验 D1/D2，跳过 GRPO 采样、不写文件：

```bash
python lab/scripts/rl_numeric_check.py --config v100_768 --device cuda:0 \
    --n-pairs 2 --max-len 64 --beta 0.15 --tol 1e-5 --dry-run \
    --run-tag day5_rl
```

- 预期：两行 `[PASS]`：`D1_init_loss_ln2  init_loss=0.69314718 ln2=0.69314718 |delta|=…`（`计算`：ln 2 = 0.6931472，policy 与 ref 权重相同时 `logits_dpo` 恒为 0）和 `D2_ref_grads_none`；末行 `[dry-run] 跳过 GRPO 采样，未写文件`。
- 不对时先查：D1 的 `|delta|` 大于 1e-5 时，看 chosen 与 rejected 的 mask 是不是由同一个编码函数生成——两侧 mask 规则不一致会让 `logits_dpo` 不为 0。

### 3. 完整数值链：真 DPO 数据 + GRPO 组内统计（30 分钟）· `可直接执行`

```bash
python lab/scripts/rl_numeric_check.py --config v100_768 --device cuda:0 \
    --dpo-file dpo.jsonl --n-pairs 4 --max-len 128 --beta 0.15 --tol 1e-5 \
    --group-size 4 --n-groups 4 --max-new-tokens 16 --seed 42 \
    --run-tag day5_rl
```

- 预期：三行 `[PASS]`（`D1_init_loss_ln2` / `D2_ref_grads_none` / `G2_ratio_at_step0`，其中 G2 的 `max|ratio-1|` 应当小于 1e-6）；再一段 `G1 zero_std_group_ratio=0.xxx (n_groups=4, group_size=4)` 与 `mean_reward` / `adv_std`；`rl_numeric.json` 出现，`status` 是 `PASS` 或 `INCONCLUSIVE`。
- 不对时先查：G2 的 `max|ratio-1|` 大于 1e-6 时，先查 `old_logp` 是不是来自另一次前向——它必须是同一次前向的 detach 结果，否则第 0 步的比值不可能是 1。

**预注册的出口**：`zero_std_group_ratio > 0.8` 时本次 GRPO 判 `INCONCLUSIVE`，`failure_class` 记 `insufficient`，**不判 FAIL**。规则 reward 的值域只有 `[-3, 3]`，分辨率本来就差；组内全相等说明这批生成没有学习信号，那是实验设计的局限，不是管线错误。Gate 不依赖 GRPO 有正向结果。

### 4. 用 Day 4 的 SFT 权重做 RL 起点的行为基线（25 分钟）· `可直接执行`

```bash
python lab/scripts/eval_compare.py --config v100_768 --tag rl_base \
    --mode chat --max-new-tokens 32 --device cuda:0 --seed 42 \
    --run-tag day5_rl --checkpoint $MM_RUNS_ROOT/day4_profile/ckpt/final.pt

python lab/scripts/eval_compare.py \
    --compare $MM_RUNS_ROOT/day4_profile/eval_after.jsonl \
              $MM_RUNS_ROOT/day5_rl/eval_rl_base.jsonl
```

同一份权重、同一个 seed、同一组 prompt，两次生成必须完全一致——这是把"RL 起点"这个前提钉死的最便宜的检查。

- 预期：第一条命令打 `[eval] 载入 …/day4_profile/ckpt/final.pt`，`eval_rl_base.jsonl` 出现；第二条命令末尾 JSON 里 `identical_ratio` 等于 **1.0**。
- 不对时先查：`identical_ratio` 小于 1.0 时，核对两次 eval 的 `--seed` 与 `--max-new-tokens` 是否相同——贪心解码在同权重同输入下必须逐 token 复现。

### 5. kill 一个 rank：8 卡写分片，7 卡读回来（30 分钟）· `可直接执行` · **不可裁**

```bash
python lab/scripts/reshard_check.py --config v100_768 --format flat \
    --save-nproc 8 --load-nproc 7 --max-steps 3 \
    --global-batch 8 --accum 1 --tol 1e-4 --run-tag day5_rl
```

8 个进程各写自己那一片，然后按 7 个 rank 重新切分并验 loss 连续性。分片边界的余数是"均分、余数分给前面的 rank"，读写两端必须用同一条规则——不一致时拼回来的向量会错位，表现是 loss 突然跳回随机初始化水平，而不是报错。

- 预期：`saved_ws=8 -> load_ws=7`；`分片长度（新 world_size）= [...]  合计等于总长=True`；`step 换算：3 (ws=8) -> 3 (ws=7)`；末行 `loss 连续性：… |delta|=… tol=1.0e-04 -> PASS`；`$MM_RUNS_ROOT/day5_rl/reshard_check.json` 出现。
- 不对时先查：报 `缺少分片：shard_N.pt` 时，就是第 N 个 rank 在写盘前退出了——去看它的 stderr；本 lab 的写端是每 rank 各写各的，任何一个 rank 失败都会缺一片。

真实的 kill-rank 恢复还要一起恢复三样东西，缺一样就不是同一次训练：`GradScaler` 的 `scale` 与 `growth_tracker`、采样器的 epoch 与 epoch 内偏移、Adam 的两个矩以及每参数的 `step` 计数（偏差修正用的就是它）。

### 6. 生成当天证据（10 分钟）· `可直接执行`

```bash
python lab/scripts/make_evidence.py --week 3 --day 5 --card day5 \
    --run-tag day5_rl --skill rl_numeric_chain --env v100 --ai-level A1 \
    --status PASS --failure-class none --time-spent 120 \
    --primary-artifact "DPO/GRPO 的四条解析断言与 8→7 kill-rank 恢复的连续性判定" \
    --config-id "v100_768 / float16 / dpo.jsonl / group=4x4"
```

`zero_std_group_ratio > 0.8` 时把 `--status` 改成 `INCONCLUSIVE`、`--failure-class` 改成 `insufficient`。

- 预期：`evidence.json` 出现在 `$MM_RUNS_ROOT/day5_rl/`，`observations` 里有 `D1_init_loss_is_ln2`、`D2_ref_grads_all_none`、`G2_ratio_is_one`、`zero_std_group_ratio`、`adv_std`、`mean_reward`、`saved_world_size`、`load_world_size`、`slices_sum_equals_total`。
- 不对时先查：`zero_std_group_ratio` 没出现时，核对步骤 3 是不是带了 `--dry-run`（dry-run 会跳过 GRPO 采样并且不写文件）。

## 公司路径差异

- 步骤 2、3、4 在单卡上跑（`--device cuda:0`），不需要 `torchrun`；步骤 5 由脚本自己起 8 个 gloo + CPU 进程，不占 GPU 排队。
- 本周不安装任何新包。`trl` 的可用区间是 `0.14.0`–`0.22.2` 配 `transformers < 4.56`，本卡的四条断言不依赖它们。
- `rl_numeric.json` 里的原始生成文本、reward 序列、分片 checkpoint 全部留在公司内；只把下面证据字段里的布尔与比值填好。

## 今日自测（闭卷，2 题，只记录能/不能）

1. （Recall）为什么 DPO 在 policy 与 ref 权重相同时初始 loss 必须精确等于 ln2？为什么判据要求 ref 的梯度是 `None` 而不是 0？
2. （Design）8 卡训练中第 3 号卡掉了，只剩 7 卡。要保住"全局批不变"和"消费的样本数不变"这两件事，`step`、`accum`、`batch_per_rank` 各该怎么改？两件事能同时保住吗？

## 证据字段（填完即可归档到 `memory/cc/evidence/`）

```json
{
  "date": "2026-09-DD",
  "week": 3,
  "day": 5,
  "card": "day5",
  "skill": "rl_numeric_chain",
  "env": "v100",
  "ai_level": "A1",
  "config_id": "v100_768 / float16 / dpo.jsonl / group=4x4",
  "primary_artifact": "day5_rl/rl_numeric.json：D1/D2/G2 三条断言 + G1 组内统计",
  "observations": {
    "D1_init_loss_is_ln2": true,
    "abs_delta_to_ln2": 0.0,
    "D2_ref_grads_all_none": true,
    "G2_ratio_is_one": true,
    "ratio_max_abs_dev_from_1": 0.0,
    "zero_std_group_ratio": 0.0,
    "mean_reward": 0.0,
    "adv_std": 0.0,
    "rl_base_identical_ratio": 1.0,
    "saved_world_size": 8,
    "load_world_size": 7,
    "converted_step": 3,
    "slices_sum_equals_total": true,
    "killrank_loss_continuity_pass": true
  },
  "status": "PASS",
  "failure_class": "none",
  "self_check": {"q1": "能", "q2": "能"},
  "time_spent_min": 120,
  "notes": "GRPO 的 zero_std_group_ratio 高时按预注册出口记 INCONCLUSIVE / insufficient"
}
```
