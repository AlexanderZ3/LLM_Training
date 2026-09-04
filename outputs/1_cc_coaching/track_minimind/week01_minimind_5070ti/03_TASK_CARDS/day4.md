# Week M01 · Day 4 — DPO：ln2 解析检查、ref 冻结与 reward margin

| 字段 | 值 |
| --- | --- |
| 主要产物 | `lab/runs/dpo_sft_5070ti/dpo_check_before.json` 与 `dpo_check_after.json`：训练前 `dpo_loss = ln2`、`ref_unchanged = true`，训练后 `reward_margin` 转正 |
| 估时 | `90 分钟`（步骤估时之和） |
| 环境 | `个人 5070 Ti`（步骤 2–5 需要 GPU；步骤 1、6 `CPU 即可`） |
| AI 辅助等级要求 | `A1`（`dpo_loss = ln2` 与 `reward_margin` 符号这两个判据必须自己推导出来，不看 `05_REFERENCE_ANSWERS.md`） |
| 前置 | Day 3 的证据：`out/full_sft_768.pth` 已由 `-MaskFault none` 那次运行导出 |
| 本卡对应门 | `FP32 ref` 意义上的解析参照点 + `有界训练`（`02_LAB_GUIDE.md` 的门 5d） |
| 可裁剪项 | 步骤 5（曲线）时间不够可只看 CSV 末行；步骤 1、2、4 不可裁 |

## 为什么做

DPO 是本周唯一有解析参照点的阶段：policy 与 ref 相同时 loss 必须精确等于 ln2 ≈ 0.6931，这让"实现对不对"变成一个可断言的事实而不是看曲线感觉。同时 ref 必须冻结——如果 ref 跟着更新，log-ratio 恒为 0，整个目标函数退化，而 loss 曲线看上去仍然"正常"。今天把这两条钉死，Day 5 的 GRPO 才有可信的 ref 概念可用。

## 步骤

### 1. 本机跑 ln2 解析检查（10 分钟）· `可直接执行`

```powershell
$Py = "D:/Software/Large/Anconda/envs/ResearchAgentPy310/python.exe"
& $Py lab\scripts\mmp.py dpo_check --config tiny_cpu --n 2
```

- 预期（`本机实测`）：`"dpo_loss": 0.6931471824645996`、`"ln2_reference": 0.6931471805599453`、`reward_margin: [0.0, 0.0]`、`logits_dpo: [0.0, 0.0]`，且 `mask_tokens_chosen` / `mask_tokens_rejected` 都大于 0。policy = ref 时 loss 必须等于 ln2，这是 DPO 唯一的解析检查点。
- 不对时先查：`reward_margin` 非 0 说明 policy 与 ref 不是同一份权重，确认 `--ref` 用的是默认值 `same`；`mask_tokens_chosen` 为 0 走 `02_LAB_GUIDE.md` 第 5 节 F5 第 5 条。

### 2. 训练前的 ref 冻结校验（15 分钟）· `可直接执行`

```powershell
$env:MINIMIND_ROOT = "D:\work\minimind"
& $Py lab\scripts\mmp.py dpo_check --config sft_5070ti --policy "$env:MINIMIND_ROOT\out\full_sft_768.pth" --ref same --n 2 --beta 0.15 --check-ref-frozen --out lab\runs\dpo_check_manual_before.json
```

- 预期：`--check-ref-frozen` 会对 policy 做一步更新（默认 `--step-lr 1e-4`）后比较 ref 的参数 hash，输出里 `ref_frozen_check.ref_unchanged` 必须为 `true`、`policy_changed` 必须为 `true`；`dpo_loss` 仍在 0.6931 附近（policy 与 ref 起点相同）。
- 没有 GPU 时先做：把 `--config sft_5070ti` 换成 `--config tiny_cpu` 跑同一条命令，验证的是 `param_hash` 机制本身；64M 模型上的数值等 5070 Ti 再取。
- 不对时先查：`ref_unchanged: false` 说明 ref 与 policy 共享了同一组张量（没有二次加载或没有 `requires_grad_(False)`），先在 `lab/src/mm_probe/dpo_check.py` 里找 `param_hash` 的调用点，确认 ref 是独立副本。

### 3. 有界 DPO 200 步（35 分钟）· `模板`

```powershell
$env:MM_PYTHON = "<5070Ti 上的 python.exe>"
.\lab\scripts\run_dpo_bounded.ps1 -MaxSteps 200 -SaveEvery 50 -Config sft_5070ti
```

- 预期：脚本自动做三件事——训练前 `dpo_check --ref same --check-ref-frozen` 写出 `lab\runs\dpo_sft_5070ti\dpo_check_before.json`、有界 DPO 200 步、训练后 `dpo_check --policy dpo_768.pth --ref full_sft_768.pth` 写出 `dpo_check_after.json`；同时产出 `log_dpo.jsonl|.csv|.png` 与 `$MINIMIND_ROOT\out\dpo_768.pth`。首步 loss 应为 ln2 ≈ 0.6931。
- 没有 GPU 时先做：本机 `& $Py lab\scripts\mmp.py bounded_train --config tiny_cpu --stage dpo --max-steps 5`，`tests/test_bounded_train.py` 已断言 DPO 首步 loss = ln2，本机能复现这一条。
- 不对时先查：脚本抛"缺少 out\full_sft_768.pth"就是 Day 3 步骤 2 没跑完或那次带了 `-MaskFault`；首步 loss 不等于 0.6931 说明 `--from-weight` 没加载成功，看 stdout 里 bounded_train 打印的权重来源。

### 4. 训练后对照 `dpo_check` 的四个数字（15 分钟）· `可直接执行`

```powershell
& $Py lab\scripts\mmp.py dpo_check --config sft_5070ti --policy "$env:MINIMIND_ROOT\out\dpo_768.pth" --ref "$env:MINIMIND_ROOT\out\full_sft_768.pth" --n 2 --beta 0.15 --out lab\runs\dpo_check_manual_after.json
```

- 预期：`reward_margin` 由 `[0.0, 0.0]` 转为正值、`logits_dpo` 转正、`accuracy` 大于 0、`dpo_loss` 小于 0.6931。把 before/after 的这四个数字并排记下来。lr 只有 4e-8，200 步的变化幅度可能很小，只要符号正确即算过门。
- 没有 GPU 时先做：这一步需要加载两份 64M 权重做前向，本机 CPU 上也能跑（慢），但 `--config sft_5070ti` 的 `max_seq_len` 较大，可加 `--max-seq-len 256` 缩短。
- 不对时先查：`reward_margin` 仍为 0，先确认 `--policy` 指的是 `dpo_768.pth` 而不是 `full_sft_768.pth`；两者相同时 margin 必然为 0，这不是训练失败。

### 5. 读 margin 与 accuracy 的逐步曲线（10 分钟）· `可直接执行`

```powershell
& $Py lab\scripts\mmp.py parse_log --input lab\runs\dpo_sft_5070ti\log_dpo.jsonl --csv lab\runs\dpo_sft_5070ti\log_dpo_check.csv --fields loss,reward_margin,dpo_accuracy,lr,grad_norm
```

- 预期：`loss` 从 0.6931 起微降、`reward_margin` 从 0 起微升、`dpo_accuracy` 从 0.5 附近起微升；三条曲线方向一致才说明训练有效。`lr` 起点约 4e-8。
- 不对时先查：`reward_margin` 抖动但无趋势，先看 `grad_norm` 是不是被 `grad_clip` 顶住；lr 4e-8 本身极小，方向对但幅度小是预期的（见 `01_FOUNDATIONS.md` 第 4.4 节）。

### 6. 跑 DPO 数学单测确认实现等价（5 分钟）· `可直接执行`

```powershell
& $Py -m pytest lab\tests\test_dpo_math.py -q --basetemp=D:\tmp\pt
```

- 预期：`6 passed`。这 6 条锁死了：`per_token_logps` 在 V=3 上的解析值、`logits_dpo = 1` 时 loss = log(1+e^-0.15)、policy = ref 时 loss = ln2、`dpo_loss_minimind` 与 `dpo_loss` 等价、beta 的单调性、`param_hash` 只在参数变化时改变。
- 不对时先查：出现 `skipped` 说明 `MINIMIND_ROOT` 没设；出现 `failed` 且是 beta 单调性那条，先核对 `--beta` 默认值 0.15 与 MiniMind `train_dpo.py` 一致。

## 今日自测（闭卷，2 题，只记录能/不能）

1. `M01-E-04`：DPO 为什么必须成对、ref 的状态与上下文、序列 logp 求和对不同长度回答的系统性影响、初始 loss 的推导。
2. `M01-R-05`：五个阶段的默认学习率、两类 checkpoint 的文件名格式与 `_resume.pth` 的键。

## 证据字段（填完即可归档到 memory/cc/evidence/）

```json
{
  "date": "YYYY-MM-DD",
  "week": 1,
  "day": 4,
  "card": "day4",
  "skill": "dpo_logratio_and_ref_freeze",
  "env": "5070ti",
  "ai_level": "A1",
  "config_id": "sft_5070ti.json stages.dpo (beta 0.15 / lr 4e-8)",
  "primary_artifact": "lab/runs/dpo_sft_5070ti/dpo_check_before.json + dpo_check_after.json",
  "observations": {
    "tiny_cpu_dpo_loss": 0.6931471824645996,
    "ln2_reference": 0.6931471805599453,
    "ref_unchanged": "true | false",
    "policy_changed": "true | false",
    "mask_tokens_chosen": 0,
    "mask_tokens_rejected": 0,
    "before_reward_margin": 0.0,
    "after_reward_margin": 0.0,
    "after_logits_dpo": 0.0,
    "after_accuracy": 0.0,
    "after_dpo_loss": 0.0,
    "dpo_peak_mem_mb": 0,
    "test_dpo_math_tail": "6 passed"
  },
  "status": "PASS | FAIL-MODEL | FAIL-SYSTEM | INCONCLUSIVE",
  "failure_class": "none | numeric | model | measurement | system",
  "self_check": {"q1": "能 | 不能", "q2": "能 | 不能"},
  "time_spent_min": 0,
  "notes": "一句话，不含敏感信息"
}
```
