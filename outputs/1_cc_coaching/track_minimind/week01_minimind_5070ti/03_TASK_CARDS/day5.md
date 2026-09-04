# Week M01 · Day 5 — GRPO smoke、跨阶段 eval 与 resume 等价

| 字段 | 值 |
| --- | --- |
| 主要产物 | `lab/runs/grpo_smoke_rule/grpo_stats.json`：`zero_std_group_ratio` / `mean_group_std` / `mean_reward` / `mean_len` 四个组内统计数字 |
| 估时 | `120 分钟`（步骤估时之和） |
| 环境 | `个人 5070 Ti`（步骤 2–5 需要 GPU；步骤 1 与步骤 5 的 tiny 部分 `CPU 即可`） |
| AI 辅助等级要求 | `A1`（`adv_std = 0` 与 `zero_std_group_ratio` 的含义必须自己解释；允许查 `01_FOUNDATIONS.md` 第 5 节） |
| 前置 | Day 3 的证据：`out/full_sft_768.pth` 已导出；Day 4 的证据：`out/dpo_768.pth` 已导出；Day 2 的 `ckpt_step500.pt` 仍在 |
| 本卡对应门 | `smoke` → `eval` → `resume`（`02_LAB_GUIDE.md` 的门 6 与门 7） |
| 可裁剪项 | 步骤 4（跨阶段 eval）可只跑 sft 与 dpo 两档；步骤 1、2、3、6 不可裁 |

## 为什么做

GRPO 是本周唯一一个"跑不出结论也算有效证据"的阶段：16 GB 上的显存没有任何实测，`zero_std_group_ratio` 很可能高到让学习信号消失。今天要拿到的不是"GRPO 提升了模型"，而是三个可归档的事实：组内统计的实测数字、三个阶段生成的横向对照、以及 resume 是否等价。`INCONCLUSIVE` 在本卡是预注册的合法出口。

**必须先记住的一条**：`--batch_size 1` 不只是省显存。MiniMind 的 `Attention.forward` 只在 `attention_mask` 全为 1 时才走 `F.scaled_dot_product_attention`；而 `train_grpo.py` 用 `padding=True, padding_side="left"` 批量编码 prompt，batch>1 且 prompt 长度不等时 `attention_mask` 就含 0，注意力掉进手写分支，显式构造 `[B, n_heads, T, T]` 的 `scores` 并用 `F.softmax(scores.float(), ...)` 升到 fp32——显存因此突增一个数量级。详见 `01_FOUNDATIONS.md` 第 5 节与 `02_LAB_GUIDE.md` 第 5 节 F7。

## 步骤

### 1. 本机用合成输入验证 `grpo_stats` 链路（10 分钟）· `可直接执行`

```powershell
$Py = "D:/Software/Large/Anconda/envs/ResearchAgentPy310/python.exe"
New-Item -ItemType Directory -Force lab\runs | Out-Null
'{"step":1,"rewards":[[1.0,1.0],[0.0,2.0]]}', '{"step":2,"rewards":[[0.5,1.5]]}' | Set-Content -Encoding utf8 lab\runs\grpo_synth.jsonl
& $Py lab\scripts\mmp.py grpo_stats --input lab\runs\grpo_synth.jsonl
```

- 预期（`本机实测`同样输入）：`"zero_std_group_ratio": 0.3333333333333333`、`"mean_group_std": 0.5`、`"mean_reward": 1.0`；逐步表第 1 步 `zero_std=1`、`adv_std=0.7070`。第一组 reward 全等所以整组 advantage 精确为 0（分子分母同时为 0，分母加了 1e-4 所以不是 nan）。
- 不对时先查：数字对不上先核对 `--eps` 是默认 1e-4，以及 `group_advantage` 用的是总体标准差（`unbiased=False`），与 MiniMind 的 `(r - mean) / (std + 1e-4)` 一致。

### 2. GRPO smoke 15 分钟（25 分钟）· `模板`

```powershell
$env:MINIMIND_ROOT = "D:\work\minimind"
$env:MM_PYTHON = "<5070Ti 上的 python.exe>"
.\lab\scripts\run_grpo_smoke.ps1 -Reward rule -NumGenerations 2 -MaxMinutes 15
```

脚本固定传 `--batch_size 1 --num_generations 2 --max_seq_len 256 --max_gen_len 256 --loss_type grpo --debug_mode --debug_interval 1`，并先调 `patch_grpo_rule_reward.py` 生成 `$MINIMIND_ROOT/trainer/train_grpo_rule.py`（**不修改**原 `train_grpo.py`，幂等）。

- 预期（`估算`）：stdout 每步一段 `[DEBUG] step=N, sample[0]` + 每个 generation 的 `RESPONSE_BEGIN` / `RESPONSE_END` / `reward=`，外加每步一条汇总行，字段依次是 `Epoch`、`Reward`、`KL_ref`、`Adv Std`、`Adv Mean`、`Actor Loss`、`Avg Response Len`、`Learning Rate`（完整格式示例见 `02_LAB_GUIDE.md` 门 6a，那里的数值只是格式示例不是实测值）；这正是 `parse_log --format minimind` 与 `grpo_stats --format debug-log` 解析的那一行。15 分钟到点被终止是**预期结果**，不是失败；`--save_interval 5` 保证每 5 步落盘。
- 没有 GPU 时先做：本步无法在本机替代（需要真实生成）。本机能做的只有步骤 1 的合成链路验证与 `& $Py -m pytest lab\tests\test_grpo_stats.py -q --basetemp=D:\tmp\pt`（预期 `5 passed`），把 GPU 部分整体推到 5070 Ti。
- 不对时先查：报 `torch.OutOfMemoryError` 且 traceback 落在 `F.softmax(scores.float(), dim=-1)` 这一行，就是上面说的手写 attention 分支——确认 `--batch_size` 确实是 1，再降 `--max_seq_len` / `--max_gen_len`（`scores` 是 T 的平方项），见 F7；reward 模型加载失败见 F8，`-Reward rule` 已经是那条出路。

### 3. 读组内统计并判定学习信号（20 分钟）· `可直接执行`

```powershell
& $Py lab\scripts\mmp.py grpo_stats --input lab\runs\grpo_smoke_rule\grpo_stdout.log --format debug-log --csv lab\runs\grpo_smoke_rule\grpo_groups.csv --json lab\runs\grpo_smoke_rule\grpo_stats.json
```

- 预期：先打印一个 JSON 头（`n_steps` / `n_groups` / `zero_std_group_ratio` / `mean_group_std` / `mean_reward` / `mean_len`），再打印逐步表。`zero_std_group_ratio` 是本门的核心数字：它就是"advantage 全 0、这一组对梯度没有任何贡献"的直接证据。
- 不对时先查：`zero_std_group_ratio > 0.5` 时按 `02_LAB_GUIDE.md` 第 5 节 F6 的顺序排查——先升 `-NumGenerations` 到 4（代价是生成时间与显存线性上升），再用 `grpo_groups.csv` 逐组看 reward 是不是只取少数几个离散值（规则 reward 的典型表现，值域 [-3, 3]，那是分辨率不够而不是采样问题），最后看 `avg_len` 是否全组相同（采样退化）。三条都排除仍全 0 就判 `INCONCLUSIVE` 并记下实测的 `zero_std_group_ratio`，不要盲目调参消耗预算。

### 4. 三个阶段的固定 prompt 生成对照（25 分钟）· `模板`

```powershell
& $Py lab\scripts\mmp.py eval_generate --config sft_5070ti --checkpoint "$env:MINIMIND_ROOT\out\pretrain_768.pth" --mode raw --tag pretrain --out lab\runs\eval_pretrain.jsonl
& $Py lab\scripts\mmp.py eval_generate --config sft_5070ti --checkpoint "$env:MINIMIND_ROOT\out\full_sft_768.pth" --mode chat --tag sft --out lab\runs\eval_sft.jsonl
& $Py lab\scripts\mmp.py eval_generate --config sft_5070ti --checkpoint "$env:MINIMIND_ROOT\out\dpo_768.pth" --mode chat --tag dpo --out lab\runs\eval_dpo.jsonl
& $Py lab\scripts\mmp.py eval_generate --compare lab\runs\eval_sft.jsonl lab\runs\eval_dpo.jsonl
```

- 预期（`估算`）：pretrain 权重用 `--mode raw` 直接续写，通顺但不回答问题，`ended_with_eos` 多为 `false`；SFT 之后用 `--mode chat` 走模板，`ended_with_eos` 应显著变多。`--compare` 并排打印并标 `identical`；DPO 只跑了 200 步且 lr 4e-8，`identical` 多为真是可以接受的观察结果。
- 时间不够时先做：只跑 sft 与 dpo 两条加 `--compare`，跳过 pretrain 那条；主要产物不受影响。
- 不对时先查：`--mode chat` 下输出里出现 `<|im_start|>user` 这类角色越界，回 Day 3 的 `user_in_loss` 对照——说明用的 checkpoint 来自故障注入那次运行；核对 `--checkpoint` 路径。

### 5. resume 等价：先 CPU 逐位、再 GPU 均值（25 分钟）· `可直接执行`

```powershell
& $Py lab\scripts\mmp.py bounded_train --config tiny_cpu --stage sft --max-steps 4 --save-every 2 --save-dir lab\runs\res_a --quiet
& $Py lab\scripts\mmp.py bounded_train --config tiny_cpu --stage sft --max-steps 4 --resume lab\runs\res_a\ckpt_step2.pt --save-dir lab\runs\res_b
```

GPU 侧接着跑（`命令等级：模板`，路径来自 Day 2 步骤 4 的 `-SaveEvery 500`）：

```powershell
.\lab\scripts\run_pretrain_bounded.ps1 -MaxSteps 2000 -Config pretrain_5070ti -Resume "lab\runs\pretrain_pretrain_5070ti\ckpt_step500.pt"
```

- 预期（CPU，`本机实测`）：第二条先打印 `[bounded] resumed from <ckpt 路径>: step=2 cursor={'epoch': 1, 'pos': 1, 'n': 3, 'batch_size': 2, 'seed': 42, 'overfit_n': 0}`，然后只跑 step 3 与 4，且这两步的 `loss` / `lr` / `sample_indices` 与连续训练**逐位相等**（`abs=1e-6`）。
- 预期（GPU，`估算`）：bf16 + SDPA 下**不要期待 bitwise 相等**；判据改成"恢复后 20 步的 loss 均值与中断前 20 步的均值差落在噪声范围内，且曲线无台阶"。
- 没有 GPU 时先做：只跑上面两条 CPU 命令，外加 `& $Py -m pytest lab\tests\test_bounded_train.py -q --basetemp=D:\tmp\pt`（预期 `5 passed`，其中一条就是 save→resume 逐位相等）；GPU 侧那条整体推迟。
- 不对时先查：曲线出现明显台阶而不是平滑接续，走 `02_LAB_GUIDE.md` 第 5 节 F9，先分清是机制坏了还是精度噪声——MiniMind 原格式 checkpoint 存的是 `half()` 后的权重，dataset 里还有基于 `random.random()` 的数据增广，这两条都会破坏 bitwise 复现但只造成一次性跳变后回落。

### 6. 写出本周 GRPO 的结论与可信度标注（15 分钟）· `模板`

把下面五格填成一段不超过 200 字的结论，直接进证据字段的 `notes` 与 `observations`：

- 用的是 `internlm` 还是 `rule` reward，以及为什么；用 `rule` 时必须写明"reward 上升不代表生成质量上升"；
- `zero_std_group_ratio` 的实测值，以及据此判定学习信号有/无；
- GRPO 的 `peak_mem_mb` 实测值，或 OOM 现场（含当时的 batch / seq / gen_len 参数）——OOM 也是有效证据；
- resume 的 CPU 逐位结果与 GPU 均值差；
- 本卡的 `status`：三条判据都拿到数字判 `PASS`；组内 std 恒为 0 且三条排查都排除后判 `INCONCLUSIVE`；显存放不下且规则 reward 也跑不起来判 `FAIL-SYSTEM`。
- 预期：五格全部有具体数字或明确的"未测"，没有一格是形容词。
- 不对时先查：写不出可信度标注时，回 `lab/README.md` 的"已知限制"第 3 条与第 5 条，它们已经写明 GRPO 显存无实测、规则 reward 不是质量信号。

## 今日自测（闭卷，2 题，只记录能/不能）

1. `M01-E-05`：组内 advantage 的完整公式、CISPO 与 GRPO 的 token 级 loss 差别、默认 torch rollout 下 ratio 等于多少。
2. `M01-D-04`：从 traceback 那一行读出走的是哪条 attention 路径，写出触发它的三个子条件，并解释 batch 2→1 时显存为什么不是线性下降。

## 证据字段（填完即可归档到 memory/cc/evidence/）

```json
{
  "date": "YYYY-MM-DD",
  "week": 1,
  "day": 5,
  "card": "day5",
  "skill": "grpo_group_stats_eval_resume",
  "env": "5070ti",
  "ai_level": "A1",
  "config_id": "train_grpo_rule.py batch1/gen2/seq256/genlen256 + pretrain_5070ti.json",
  "primary_artifact": "lab/runs/grpo_smoke_rule/grpo_stats.json",
  "observations": {
    "reward_source": "rule | internlm",
    "n_steps": 0,
    "n_groups": 0,
    "zero_std_group_ratio": 0.0,
    "mean_group_std": 0.0,
    "mean_reward": 0.0,
    "mean_len": 0.0,
    "grpo_peak_mem_mb": "数值 | OOM(batch/seq/gen_len)",
    "batch_size_used": 1,
    "eval_pretrain_ended_with_eos": "n/8",
    "eval_sft_ended_with_eos": "n/8",
    "eval_sft_vs_dpo_identical": "n/8",
    "resume_cpu_bitwise_equal": "true | false",
    "resume_gpu_mean_diff": 0.0
  },
  "status": "PASS | FAIL-MODEL | FAIL-SYSTEM | INCONCLUSIVE",
  "failure_class": "none | system | measurement | insufficient | model",
  "self_check": {"q1": "能 | 不能", "q2": "能 | 不能"},
  "time_spent_min": 0,
  "notes": "一句话，不含敏感信息；用 rule reward 时必须写明 reward 不代表质量"
}
```
