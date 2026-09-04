# 命令等级：模板 —— 需要先 `$env:MINIMIND_ROOT=<clone>`；`$env:MM_PYTHON` 指向 5070 Ti 上装好 cu128 torch 的解释器（缺省 python）。
# Day 2：有界 pretrain。
#   -Mode bounded（默认）：用 lab 的 bounded_train.py（原 train_pretrain.py 没有步数上限参数，只能整轮），
#                        结束时导出 $MINIMIND_ROOT/out/pretrain_768.pth 供后续 SFT（原脚本 --from_weight pretrain 也认这个路径）。
#   -Mode epoch        ：调用原仓库 trainer/train_pretrain.py 跑 1 个整 epoch（参数 = 官方默认，已核验 commit 7a6fddd），stdout tee 到 lab/runs/。
# 例：
#   .\run_pretrain_bounded.ps1 -MaxSteps 20 -Config smoke_5070ti                # smoke
#   .\run_pretrain_bounded.ps1 -MaxSteps 300 -OverfitN 128 -Config pretrain_5070ti   # 128 样本 overfit
#   .\run_pretrain_bounded.ps1 -MaxSteps 2000 -SaveEvery 500 -Config pretrain_5070ti  # 有界训练
param(
    [string]$Config = "pretrain_5070ti",
    [int]$MaxSteps = 200,
    [int]$OverfitN = 0,
    [int]$SaveEvery = 100,
    [string]$Dtype = "bfloat16",
    [ValidateSet("bounded", "epoch")][string]$Mode = "bounded",
    [string]$Resume = ""
)
$ErrorActionPreference = "Stop"
if (-not $env:MINIMIND_ROOT) { throw "请先设置 `$env:MINIMIND_ROOT" }
$Python = if ($env:MM_PYTHON) { $env:MM_PYTHON } else { "python" }
$Lab = Split-Path -Parent $PSScriptRoot
$RunDir = Join-Path $Lab "runs\pretrain_$Config"
New-Item -ItemType Directory -Force $RunDir | Out-Null
New-Item -ItemType Directory -Force (Join-Path $env:MINIMIND_ROOT "out") | Out-Null

if ($Mode -eq "bounded") {
    $argv = @("$Lab\scripts\mmp.py", "bounded_train", "--config", $Config, "--stage", "pretrain",
              "--max-steps", $MaxSteps, "--overfit-n", $OverfitN, "--save-every", $SaveEvery, "--dtype", $Dtype,
              "--save-dir", $RunDir, "--log-jsonl", "$RunDir\log_pretrain.jsonl",
              "--export-pth", (Join-Path $env:MINIMIND_ROOT "out\pretrain_768.pth"))
    if ($Resume) { $argv += @("--resume", $Resume) }
    & $Python @argv
    if ($LASTEXITCODE -ne 0) { throw "bounded_train 退出码 $LASTEXITCODE（2 = 最后一步 loss 非有限）" }
    & $Python "$Lab\scripts\mmp.py" parse_log --input "$RunDir\log_pretrain.jsonl" --csv "$RunDir\log_pretrain.csv" --png "$RunDir\log_pretrain.png"
} else {
    Push-Location (Join-Path $env:MINIMIND_ROOT "trainer")
    try {
        # 官方默认：batch 32, accumulation 8, max_seq_len 340, lr 5e-4, bf16；checkpoint 写到 ../out 与 ../checkpoints
        & $Python train_pretrain.py --epochs 1 --batch_size 32 --accumulation_steps 8 --max_seq_len 340 --dtype $Dtype `
            --log_interval 20 --save_interval 500 --num_workers 4 --data_path ../dataset/pretrain_t2t_mini.jsonl 2>&1 |
            Tee-Object -FilePath "$RunDir\minimind_pretrain_stdout.log"
    } finally { Pop-Location }
    & $Python "$Lab\scripts\mmp.py" parse_log --input "$RunDir\minimind_pretrain_stdout.log" --format minimind --csv "$RunDir\minimind_pretrain.csv" --png "$RunDir\minimind_pretrain.png"
}
