# Week M01 · Day 3 — Full SFT 与 mask 错位故障的正常/错位对照

| 字段            | 值                                                                                                              |
| --------------- | --------------------------------------------------------------------------------------------------------------- |
| 主要产物        | 正常（`none`）与错位（`user_in_loss`）两条 loss 曲线的并排对照，加一段写下来的结论：症状 → 检查 → 定位    |
| 估时            | `120 分钟`（步骤估时之和）                                                                                    |
| 环境            | `个人 5070 Ti`（步骤 2–5 需要 GPU；步骤 1、6 `CPU 即可`）                                                  |
| AI 辅助等级要求 | `A1`（步骤 6 的结论必须闭卷先写，写完再对 `02_LAB_GUIDE.md` 第 5 节 F5；这是 Gate 第三题的预演）            |
| 前置            | Day 2 的证据：`exported_pth` = `out/pretrain_768.pth` 已生成；Day 1 的证据：`fixture_first_loss_pos = 12` |
| 本卡对应门      | `有界训练 + 故障注入`（`02_LAB_GUIDE.md` 的门 5c）                                                          |
| 可裁剪项        | 步骤 3 的`-MaxSteps 300` 可降到 150（对照仍成立，两条曲线步数必须相同）；步骤 2、5、6 不可裁                  |

## 为什么做

Gate 的核心题是"只凭 loss 曲线与固定 prompt 生成对照反推 mask 故障"。这要求你手里有一条已知正确的基线曲线和一条已知故障的曲线，并且知道故障的因果签名：`user_in_loss` 让模板 token 进 loss，模板极易预测，所以 loss 起点更低、曲线更平滑——这是必然结果不是巧合。今天把这三条曲线跑出来并写下判据，Day 5 之后就没有机会再补。

## 步骤

### 1. 本机先看故障签名的廉价版（10 分钟）· `可直接执行`

```powershell
$Py = "D:/Software/Large/Anconda/envs/ResearchAgentPy310/python.exe"
& $Py lab\scripts\mmp.py mask_fault --mode user_in_loss --index 1 --max-seq-len 48
& $Py lab\scripts\mmp.py mask_fault --mode assistant_all_ignored --index 1 --max-seq-len 48
```

- 预期（`本机实测`，`user_in_loss`）：`in-loss tokens: normal=6  faulty=40`，逐 token 表的 label 列显示 `-100->1`、`-100->832` 这类变化；`assistant_all_ignored` 下 faulty 的 in-loss 计数为 0。
- 不对时先查：命令报未知 mode，核对 `--mode` 只接受 `none` / `assistant_all_ignored` / `user_in_loss` 三个取值（`lab/src/mm_probe/mask_fault.py` 的 `FAULT_MODES`）。

### 2. 正常 SFT 300 步（35 分钟）· `模板`

- 预期：写到 `lab\runs\sft_sft_5070ti_none\`，产出 `log_sft.jsonl|.csv|.png`、`eval.jsonl`（8 条固定 prompt 的贪心生成），并导出 `$MINIMIND_ROOT\out\full_sft_768.pth`（Day 4 与 Day 5 的输入）。loss 曲线正常下降且有明显抖动，`n_label_tokens` 每步都远小于该步的非 pad token 总数。
- 没有 GPU 时先做：本机 `& $Py lab\scripts\mmp.py bounded_train --config tiny_cpu --stage sft --max-steps 20 --save-dir lab\runs\tiny_none`，只验证代码路径与日志字段；tiny 配置的曲线形状不能当作对照基线。
- 不对时先查：脚本抛"缺少 out\pretrain_768.pth"就是 Day 2 步骤 4 没跑完；`n_label_tokens` 等于非 pad 总数说明 mask 没生效（`02_LAB_GUIDE.md` 第 5 节 F5）。

### 3. 注入 `user_in_loss` 跑同样步数（35 分钟）· `模板`

```powershell
.\lab\scripts\run_sft_bounded.ps1 -MaxSteps 300 -Config sft_5070ti -MaskFault user_in_loss
```

- 预期：写到独立目录 `lab\runs\sft_sft_5070ti_user_in_loss\`，**不**导出 `.pth`。loss 起点明显低于步骤 2 的起点、下降更快、曲线更平滑；`n_label_tokens` 等于每步的非 pad token 总数——这是该故障的直接签名。
- 没有 GPU 时先做：本机 `& $Py lab\scripts\mmp.py bounded_train --config tiny_cpu --stage sft --max-steps 20 --mask-fault user_in_loss --save-dir lab\runs\tiny_user_in_loss`，对比 `n_label_tokens` 两次的差别。
- 不对时先查：两条曲线的起点差不出来，先确认两次运行的 `-MaxSteps`、`-Config`、`-Dtype` 完全一致，且 `bounded_train` 的 `--seed` 都是默认 42（否则比较的不是同一个变量）。

### 4. 注入 `assistant_all_ignored` 跑 50 步（10 分钟）· `模板`

```powershell
.\lab\scripts\run_sft_bounded.ps1 -MaxSteps 50 -Config sft_5070ti -MaskFault assistant_all_ignored
```

- 预期：loss 立刻是 `nan`，`n_label_tokens` 为 0，`bounded_train` 退出码 2，脚本打印退出码提示。这是**预期症状**而不是 bug：`F.cross_entropy(..., ignore_index=-100)` 对 0 个有效元素取均值就是 `nan`。50 步足够，不用跑 300。
- 没有 GPU 时先做：本机 `& $Py lab\scripts\mmp.py bounded_train --config tiny_cpu --stage sft --max-steps 3 --mask-fault assistant_all_ignored`，同样会看到 `nan` 与退出码 2。
- 不对时先查：如果 loss 不是 `nan` 而是有限值，说明 `--mask-fault` 没传进去；看 `log_sft.jsonl` 第一行有没有 `n_label_tokens: 0`。

### 5. 固定 prompt 生成并排对照（15 分钟）· `可直接执行`

```powershell
& $Py lab\scripts\mmp.py eval_generate --compare lab\runs\sft_sft_5070ti_none\eval.jsonl lab\runs\sft_sft_5070ti_user_in_loss\eval.jsonl
```

- 预期：并排打印两次运行的 8 条输出并标 `identical`。`user_in_loss` 一侧应出现角色越界：复述 user 的原文，或自己生成 `<|im_start|>user`；`none` 一侧不应出现这两种。
- 没有 GPU 时先做：这一步只读两个 `eval.jsonl` 文件、不加载模型，本机可跑，前提是两个文件已从 5070 Ti 上的运行目录里存在。
- 不对时先查：`identical` 全为真说明两次生成用的是同一个 checkpoint——`run_sft_bounded.ps1` 用的是各自 `$RunDir\latest.pt`，确认两个目录下的 `latest.pt` 修改时间不同。

### 6. 闭卷写下三段故障的判别结论（15 分钟）· `模板`

在证据字段的 `notes` 之外，另写一段不超过 200 字的结论，按下面四格填：

- 症状（只看 loss 曲线能说的）：`none` 的起点 `<数字>`、`user_in_loss` 的起点 `<数字>`、`assistant_all_ignored` 的 `nan`；
- 不变量断言（对单条样本可执行）：`0 < n_label_tokens < n_nonpad`，且 `labels[i] == -100` 对所有 user 段与模板段成立；
- 最小检查（脚本 + 看哪个数字）：`parse_log --fields loss,n_label_tokens,grad_norm` 看 `n_label_tokens` 落在 0 / 很小 / 等于 tokens 三档中的哪一档；
- 哪一对特征是充分的、哪一对只是必要的。
- 预期：三种故障各有一条只看 loss 的判别特征与一条只看生成的判别特征，且你能指出"loss 起点更低"单独出现时无法区分故障与"换了更容易的数据集"。
- 不对时先查：写不出区分实验时，回 `02_LAB_GUIDE.md` 第 5 节 F5 的三档 `n_label_tokens` 判据，它把三种症状映射到三个可执行的数字。

## 可选扩展（不计入本卡估时）

LoRA 是本周的可选扩展，本 lab 没有封装入口，走 MiniMind 原脚本 `trainer/train_lora.py`（rank 固定 16，无 `--lora_rank`，只作用于 `in_features == out_features` 的方阵 Linear）。`命令等级：模板`：

```powershell
Push-Location (Join-Path $env:MINIMIND_ROOT "trainer")
& $env:MM_PYTHON train_lora.py --epochs 1 --dtype bfloat16 --data_path ../dataset/sft_t2t_mini.jsonl
Pop-Location
```

时间不够时整段跳过，主要产物不受影响。

## 今日自测（闭卷，2 题，只记录能/不能）

1. `M01-D-02`：给定"loss 初值更低且更平滑 + 生成越界"的日志，写出被破坏的不变量与最小检查。
2. `M01-T-03`：16 GB 上全参 SFT 与 LoRA 的四项显存账，以及哪一项 LoRA 不省。

## 证据字段（填完即可归档到 memory/cc/evidence/）

```json
{
  "date": "YYYY-MM-DD",
  "week": 1,
  "day": 3,
  "card": "day3",
  "skill": "sft_label_mask_fault_injection",
  "env": "5070ti",
  "ai_level": "A1",
  "config_id": "sft_5070ti.json + mask_fault {none, user_in_loss, assistant_all_ignored}",
  "primary_artifact": "lab/runs/sft_sft_5070ti_none/log_sft.csv vs lab/runs/sft_sft_5070ti_user_in_loss/log_sft.csv",
  "observations": {
    "none_loss_first": 0.0,
    "none_loss_last": 0.0,
    "user_in_loss_loss_first": 0.0,
    "user_in_loss_loss_last": 0.0,
    "none_n_label_tokens": 0,
    "user_in_loss_n_label_tokens": 0,
    "assistant_all_ignored_loss": "nan",
    "assistant_all_ignored_n_label_tokens": 0,
    "bounded_train_exit_code": 2,
    "eval_compare_identical": "true | false",
    "role_leak_observed": "是 | 否",
    "peak_mem_mb_sft": 0,
    "conclusion_written": "是 | 否"
  },
  "status": "PASS | FAIL-MODEL | FAIL-SYSTEM | INCONCLUSIVE",
  "failure_class": "none | data_contract | model | measurement | system",
  "self_check": {"q1": "能 | 不能", "q2": "能 | 不能"},
  "time_spent_min": 0,
  "notes": "一句话，不含敏感信息"
}
```
