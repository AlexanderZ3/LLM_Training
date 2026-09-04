# 命令等级：模板 —— 需要 `$env:MINIMIND_ROOT`、`$env:MM_PYTHON`（缺省 python）、以及 Day 2 产出的 $MINIMIND_ROOT/out/pretrain_768.pth。
# Day 3：有界 full SFT + mask 故障注入对照。
#   -Mode bounded（默认）：lab bounded_train.py，--from-weight pretrain_768.pth，结束导出 out/full_sft_768.pth（-MaskFault none 时）。
#   -Mode epoch        ：原仓库 trainer/train_full_sft.py 跑 1 epoch（官方默认 seq 768 / lr 1e-5；物理 batch 8 × accum 2 = 官方有效 batch 16）。
#   -MaskFault assistant_all_ignored | user_in_loss ：故障注入版，结果写到独立目录 runs/sft_<Config>_<fault>，不导出 pth。
# 每次结束都对 8 条固定 prompt 做贪心生成（eval_generate），供正常/错位对照：
#   python scripts\mmp.py eval_generate --compare runs\sft_<Config>_none\eval.jsonl runs\sft_<Config>_user_in_loss\eval.jsonl
param(
    [string]$Config = "sft_5070ti",
    [int]$MaxSteps = 300,
    [int]$OverfitN = 0,
    [int]$SaveEvery = 100,
    [string]$Dtype = "bfloat16",
    [ValidateSet("bounded", "epoch")][string]$Mode = "bounded",
    [ValidateSet("none", "assistant_all_ignored", "user_in_loss")][string]$MaskFault = "none",
    [string]$Resume = ""
)
$ErrorActionPreference = "Stop"
if (-not $env:MINIMIND_ROOT) { throw "请先设置 `$env:MINIMIND_ROOT" }
$Python = if ($env:MM_PYTHON) { $env:MM_PYTHON } else { "python" }
$Lab = Split-Path -Parent $PSScriptRoot
$FromWeight = Join-Path $env:MINIMIND_ROOT "out\pretrain_768.pth"
if (-not (Test-Path $FromWeight)) { throw "缺少 $FromWeight：先运行 run_pretrain_bounded.ps1" }
$RunDir = Join-Path $Lab "runs\sft_${Config}_$MaskFault"
New-Item -ItemType Directory -Force $RunDir | Out-Null

if ($Mode -eq "bounded") {
    $argv = @("$Lab\scripts\mmp.py", "bounded_train", "--config", $Config, "--stage", "sft",
              "--max-steps", $MaxSteps, "--overfit-n", $OverfitN, "--save-every", $SaveEvery, "--dtype", $Dtype,
              "--from-weight", $FromWeight, "--mask-fault", $MaskFault,
              "--save-dir", $RunDir, "--log-jsonl", "$RunDir\log_sft.jsonl")
    if ($MaskFault -eq "none") { $argv += @("--export-pth", (Join-Path $env:MINIMIND_ROOT "out\full_sft_768.pth")) }
    if ($Resume) { $argv += @("--resume", $Resume) }
    & $Python @argv
    $code = $LASTEXITCODE
    & $Python "$Lab\scripts\mmp.py" parse_log --input "$RunDir\log_sft.jsonl" --csv "$RunDir\log_sft.csv" --png "$RunDir\log_sft.png"
    & $Python "$Lab\scripts\mmp.py" eval_generate --config $Config --checkpoint "$RunDir\latest.pt" --mode chat --max-new-tokens 64 --out "$RunDir\eval.jsonl" --tag "sft_$MaskFault"
    if ($code -ne 0) { Write-Host "bounded_train 退出码 $code（2 = 最后一步 loss 非有限；assistant_all_ignored 故障下这是预期症状）" }
} else {
    Push-Location (Join-Path $env:MINIMIND_ROOT "trainer")
    try {
        & $Python train_full_sft.py --epochs 1 --batch_size 8 --accumulation_steps 2 --max_seq_len 768 --learning_rate 1e-5 --dtype $Dtype `
            --from_weight pretrain --log_interval 20 --save_interval 500 --num_workers 4 --data_path ../dataset/sft_t2t_mini.jsonl 2>&1 |
            Tee-Object -FilePath "$RunDir\minimind_sft_stdout.log"
    } finally { Pop-Location }
    & $Python "$Lab\scripts\mmp.py" parse_log --input "$RunDir\minimind_sft_stdout.log" --format minimind --csv "$RunDir\minimind_sft.csv" --png "$RunDir\minimind_sft.png"
    & $Python "$Lab\scripts\mmp.py" eval_generate --config $Config --checkpoint (Join-Path $env:MINIMIND_ROOT "out\full_sft_768.pth") --mode chat --out "$RunDir\eval.jsonl" --tag "sft_epoch"
}
