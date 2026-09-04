# 命令等级：模板 —— 需要 `$env:MINIMIND_ROOT`、`$env:MM_PYTHON`（缺省 python）、Day 3 的 out/full_sft_768.pth；
#   -Reward internlm（默认）还需要 reward 模型：
#     hf download internlm/internlm2-1_8b-reward --local-dir "$env:MINIMIND_ROOT\..\internlm2-1_8b-reward"   # ≈3.4 GB safetensors；旧版 CLI 用 huggingface-cli download
#   -Reward rule：16 GB 放不下（或不想下载）时，用 scripts/patch_grpo_rule_reward.py 生成 trainer/train_grpo_rule.py（规则 reward）。
# Day 5：GRPO smoke。原 train_grpo.py 没有步数上限，本脚本用 -MaxMinutes 超时终止（save_interval 5 → 每 5 步落盘），
# 参数按周卡：--batch_size 1 --num_generations 2 --max_seq_len 256 --max_gen_len 256 --loss_type grpo；--debug_mode --debug_interval 1
# 让每步的 reward 打印到 stdout，grpo_stats 从日志里读组内 reward 算 advantage / std=0 组占比 / 平均长度。
param(
    [ValidateSet("internlm", "rule")][string]$Reward = "internlm",
    [int]$NumGenerations = 2,
    [int]$MaxMinutes = 15,
    [string]$Dtype = "bfloat16",
    [string]$RewardModelPath = ""
)
$ErrorActionPreference = "Stop"
if (-not $env:MINIMIND_ROOT) { throw "请先设置 `$env:MINIMIND_ROOT" }
$Python = if ($env:MM_PYTHON) { $env:MM_PYTHON } else { "python" }
$Lab = Split-Path -Parent $PSScriptRoot
if (-not (Test-Path (Join-Path $env:MINIMIND_ROOT "out\full_sft_768.pth"))) { throw "缺少 out\full_sft_768.pth：先运行 run_sft_bounded.ps1" }
$RunDir = Join-Path $Lab "runs\grpo_smoke_$Reward"
New-Item -ItemType Directory -Force $RunDir | Out-Null
$Log = Join-Path $RunDir "grpo_stdout.log"
$ErrLog = Join-Path $RunDir "grpo_stderr.log"

if ($Reward -eq "rule") {
    & $Python "$Lab\scripts\patch_grpo_rule_reward.py"
    $Script = "train_grpo_rule.py"
    $RmPath = "unused"
} else {
    $Script = "train_grpo.py"
    $RmPath = if ($RewardModelPath) { $RewardModelPath } else { "../../internlm2-1_8b-reward" }
}
$argList = @($Script, "--batch_size", "1", "--num_generations", $NumGenerations, "--max_seq_len", "256", "--max_gen_len", "256",
             "--loss_type", "grpo", "--accumulation_steps", "1", "--dtype", $Dtype, "--epochs", "1",
             "--log_interval", "1", "--save_interval", "5", "--debug_mode", "--debug_interval", "1", "--num_workers", "0",
             "--from_weight", "full_sft", "--reward_model_path", $RmPath, "--data_path", "../dataset/rlaif.jsonl")
Write-Host "[grpo] $Python $($argList -join ' ')  (timeout $MaxMinutes min)"
$p = Start-Process -FilePath $Python -ArgumentList $argList -WorkingDirectory (Join-Path $env:MINIMIND_ROOT "trainer") `
        -NoNewWindow -PassThru -RedirectStandardOutput $Log -RedirectStandardError $ErrLog
if (-not $p.WaitForExit($MaxMinutes * 60 * 1000)) {
    $p.Kill()
    Write-Host "[grpo] 达到 $MaxMinutes 分钟上限，已终止（smoke 目的已达成；checkpoint 每 5 步在 ../out 与 ../checkpoints）"
} else {
    Write-Host "[grpo] 进程退出码 $($p.ExitCode)（非 0 时看 $ErrLog：OOM / reward 模型加载失败 见 02_LAB_GUIDE 第 5 节）"
}
& $Python "$Lab\scripts\mmp.py" grpo_stats --input $Log --format debug-log --csv "$RunDir\grpo_groups.csv" --json "$RunDir\grpo_stats.json"
& $Python "$Lab\scripts\mmp.py" parse_log --input $Log --format minimind --csv "$RunDir\grpo_steps.csv" --png "$RunDir\grpo_steps.png" --fields reward,kl_ref,adv_std,actor_loss,avg_response_len
