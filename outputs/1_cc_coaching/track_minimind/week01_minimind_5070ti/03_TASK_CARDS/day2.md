# Week M01 · Day 2 — 从单 batch 前反向走到有界 pretrain 并把曲线读出来

| 字段 | 值 |
| --- | --- |
| 主要产物 | `lab/runs/pretrain_pretrain_5070ti/log_pretrain.jsonl` 与 `parse_log` 出的 `log_pretrain.csv` + `log_pretrain.png` 曲线 |
| 估时 | `120 分钟`（步骤估时之和） |
| 环境 | `个人 5070 Ti`（步骤 2–6 需要 GPU；步骤 1 `CPU 即可`） |
| AI 辅助等级要求 | `A2`（命令是模板，但曲线的三个判据——loss 起点、overfit 幅度、`peak_mem_mb` 实测值——必须自己读出来并解释） |
| 前置 | Day 0 的证据：`cuda_matmul_ok = true`、`commit_matches = true`；Day 1 的证据：`mean_loss_token_ratio_sft` 已记录 |
| 本卡对应门 | `dry run` → `smoke` → `overfit`（`02_LAB_GUIDE.md` 的门 3、门 4、门 5a、门 5b） |
| 可裁剪项 | 步骤 4 的 `-MaxSteps 2000` 可降到 800（主要产物仍成立）；步骤 1、3、5 不可裁 |

## 为什么做

"loss 在降"什么都不证明，除非先证明梯度确实在改变模型。今天的顺序是固定的：先在 CPU 上确认单 batch 前反向的 loss 落在 ln(6400)=8.764 附近，再上 GPU 跑 20 步 smoke 拿到第一个真实 `peak_mem_mb`，再用 128 条样本 overfit 证明"128 条都拟合不动就不是数据量问题"，最后才跑有界 pretrain 并导出 `out/pretrain_768.pth`。这个 `.pth` 是 Day 3 SFT 的唯一输入，没有它后面三天全部无法开始。

## 步骤

### 1. CPU 单 batch 前反向（10 分钟）· `可直接执行`

```powershell
$Py = "D:/Software/Large/Anconda/envs/ResearchAgentPy310/python.exe"
& $Py lab\scripts\mmp.py bounded_train --config tiny_cpu --stage pretrain --max-steps 1
```

- 预期（`本机实测`同配置的 sft 版为 `loss 8.7434`）：一行 `[bounded:pretrain] step 1/1 loss <8.7 附近> lr 1.00e-05 grad_norm <有限且非 0>`，summary 里 `device: "cpu"`、`dtype: "float32"`、`peak_mem` 为 `-`（无 CUDA）。
- 不对时先查：loss 远离 8.76 说明 label 或权重加载有问题，先回 Day 1 步骤 4 用 `inspect_dataset --stage pretrain --fixture` 看是不是所有非 pad 位置都进了 loss。

### 2. 5070 Ti 上跑 20 步 smoke（20 分钟）· `模板`

先把 `<5070Ti 上的 python.exe>` 换成那台机器上装了 cu128 torch 的解释器路径。

```powershell
$env:MINIMIND_ROOT = "D:\work\minimind"
$env:MM_PYTHON = "<5070Ti 上的 python.exe>"
.\lab\scripts\run_pretrain_bounded.ps1 -MaxSteps 20 -Config smoke_5070ti -SaveEvery 0
```

bash：

```bash
MAX_STEPS=20 CONFIG=smoke_5070ti SAVE_EVERY=0 bash lab/scripts/run_pretrain_bounded.sh
```

- 预期（`估算`）：20 行 `[bounded:pretrain]`，`peak_mem_mb` 稳定在 4000 以下且是真实数值（不是 `null`），`scaler_scale` 为 `null`（bf16 不用 GradScaler），`tokens_per_s` 前 2–3 步偏低之后稳定。本步的产出就是把 `02_LAB_GUIDE.md` 第 0.3 节的显存 `估算` 换成实测。
- 没有 GPU 时先做：本机跑 `& $Py lab\scripts\mmp.py bounded_train --config tiny_cpu --stage pretrain --max-steps 20 --log-jsonl lab\runs\tiny_cpu_pretrain\log_pretrain.jsonl`，验证的是日志字段齐全与代码路径，`peak_mem_mb` 恒为 `null`，显存与吞吐必须等 5070 Ti 重跑。
- 不对时先查：报 `torch.OutOfMemoryError` 就先降 `smoke_5070ti.json` 的 `batch_size`，见 `02_LAB_GUIDE.md` 第 5 节 F2；`peak_mem_mb` 是 `null` 说明跑的其实是 CPU，回 Day 0 的 `probe.json` 看 `cuda_available`。

### 3. 128 条样本 overfit（30 分钟）· `模板`

```powershell
.\lab\scripts\run_pretrain_bounded.ps1 -MaxSteps 300 -OverfitN 128 -Config pretrain_5070ti
```

- 预期（`估算`）：`--overfit-n 128` 让数据游标只在前 128 条里循环，loss 从 8.76 附近显著下降到远低于 8.76。这是唯一能廉价证明"梯度确实在改变模型"的实验。
- 没有 GPU 时先做：本机 `& $Py lab\scripts\mmp.py bounded_train --config tiny_cpu --stage pretrain --max-steps 60 --overfit-n 2`，看 2 条样本上 loss 是否下降；tiny 配置只有 532928 个参数，下降幅度不能外推到 64M 模型。
- 不对时先查：300 步后 loss 仍贴着 8.76 不动，问题不在数据量，在 label、lr 或权重加载三者之一——先用 `& $Py lab\scripts\mmp.py parse_log --input lab\runs\pretrain_pretrain_5070ti\log_pretrain.jsonl --fields loss,n_label_tokens,grad_norm` 看 `n_label_tokens` 是不是 0（`02_LAB_GUIDE.md` 第 5 节 F5）。

### 4. 有界 pretrain 并导出 `pretrain_768.pth`（40 分钟）· `模板`

```powershell
.\lab\scripts\run_pretrain_bounded.ps1 -MaxSteps 2000 -SaveEvery 500 -Config pretrain_5070ti
```

- 预期：产出 `$MINIMIND_ROOT\out\pretrain_768.pth`（Day 3 的输入）、`lab\runs\pretrain_pretrain_5070ti\log_pretrain.jsonl|.csv|.png`，以及 `ckpt_step500.pt` / `ckpt_step1000.pt` / `ckpt_step1500.pt` / `latest.pt`（Day 5 resume 的输入）。2000 步全部 loss 有限。
- 时间不够时先做：把 `-MaxSteps` 降到 800、`-SaveEvery` 保持 500，`out/pretrain_768.pth` 与 `ckpt_step500.pt` 照常产生，后三天不受影响。
- 不对时先查：脚本抛 `bounded_train 退出码 2` 表示最后一步 loss 非有限；先看 JSONL 最后一行的 `loss` 与 `grad_norm` 是 `nan` 还是 `inf`，`nan` 走 F5，训练中途才转 `nan` 走 F3（dtype 选错）。

### 5. 把日志读成曲线并给出三个判据（15 分钟）· `可直接执行`

```powershell
& $Py lab\scripts\mmp.py parse_log --input lab\runs\pretrain_pretrain_5070ti\log_pretrain.jsonl --csv lab\runs\pretrain_pretrain_5070ti\log_pretrain.csv --png lab\runs\pretrain_pretrain_5070ti\log_pretrain.png --fields loss,lr,grad_norm
```

- 预期：CSV 每步一行，字段名统一成小写下划线；PNG 出三条曲线（没装 matplotlib 时自动降级成 ASCII sparkline，不报错）。写下三个数字：首步 loss、末步 loss、`grad_norm` 的量级范围。lr 曲线应符合 `get_lr(s, S, lr) = lr·(0.1 + 0.45·(1 + cos(π·s/S)))`，即起点约 `lr`、终点约 `0.1·lr`。
- 不对时先查：CSV 行数少于步数，说明 `--log-jsonl` 指到了上一次运行的文件被追加/覆盖；确认 `lab\runs\pretrain_pretrain_5070ti\` 目录下的文件修改时间。

### 6. 记录显存与吞吐的实测值（5 分钟）· `可直接执行`

```powershell
& $Py lab\scripts\mmp.py parse_log --input lab\runs\pretrain_pretrain_5070ti\log_pretrain.jsonl --fields peak_mem_mb,tokens_per_s,scaler_scale
```

- 预期：`peak_mem_mb` 是真实数值且在 16000 以下并趋于平稳；`tokens_per_s` 前几步偏低后稳定；bf16 下 `scaler_scale` 全为 `null`。把 smoke（`smoke_5070ti`）与有界（`pretrain_5070ti`）两档的 `peak_mem_mb` 峰值各记一个数。
- 不对时先查：`tokens_per_s` 一直很低而不是抖动，按 `02_LAB_GUIDE.md` 第 5 节 F11 处理；注意 `bounded_train` 不用 DataLoader（按游标直接索引 dataset），它的吞吐与原脚本不可直接比较。

## 今日自测（闭卷，2 题，只记录能/不能）

1. `M01-E-01`：`F.cross_entropy(..., ignore_index=-100)` 的分母到底是什么集合，以及随机初始化首步 loss 应该是多少。
2. `M01-A-03`：一个优化步吃多少条序列、`get_lr` 在三点的取值、以及累积窗口内哪类样本被高估。

## 证据字段（填完即可归档到 memory/cc/evidence/）

```json
{
  "date": "YYYY-MM-DD",
  "week": 1,
  "day": 2,
  "card": "day2",
  "skill": "bounded_pretrain_and_instrumentation",
  "env": "5070ti",
  "ai_level": "A2",
  "config_id": "tiny_cpu.json + smoke_5070ti.json + pretrain_5070ti.json",
  "primary_artifact": "lab/runs/pretrain_pretrain_5070ti/log_pretrain.jsonl (+ .csv/.png)",
  "observations": {
    "cpu_step1_loss": 0.0,
    "smoke_peak_mem_mb": 0,
    "smoke_scaler_scale": "null",
    "overfit128_loss_first": 0.0,
    "overfit128_loss_last": 0.0,
    "bounded_max_steps": 2000,
    "bounded_loss_first": 0.0,
    "bounded_loss_last": 0.0,
    "bounded_peak_mem_mb": 0,
    "tokens_per_s_steady": 0,
    "grad_norm_range": "min-max",
    "exported_pth": "out/pretrain_768.pth 是否生成"
  },
  "status": "PASS | FAIL-MODEL | FAIL-SYSTEM | INCONCLUSIVE",
  "failure_class": "none | numeric | system | model | measurement",
  "self_check": {"q1": "能 | 不能", "q2": "能 | 不能"},
  "time_spent_min": 0,
  "notes": "一句话，不含敏感信息"
}
```
