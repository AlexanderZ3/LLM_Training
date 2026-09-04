# 命令等级：模板 —— 需要 `$env:MINIMIND_ROOT`、`$env:MM_PYTHON`（缺省 python）、Day 3 产出的 $MINIMIND_ROOT/out/full_sft_768.pth。
# Day 4：DPO。原 trainer/train_dpo.py 没有步数上限参数（只有 --epochs），所以：
#   -Mode bounded（默认）：lab bounded_train.py --stage dpo（镜像 train_dpo.py 的 loss：beta 0.15，ref = 初始权重的冻结副本），
#                        每步记录 loss / reward_margin / dpo_accuracy；结束导出 out/dpo_768.pth。
#   -Mode epoch        ：原脚本跑 1 个整 epoch（官方默认 batch 4, lr 4e-8, seq 1024, beta 0.15）。
# 训练前后各做一次 dpo_check（同一条 fixture 对：policy vs ref 的 logp 差、loss、margin；--check-ref-frozen 验证 ref hash 不变）。
param(
    [string]$Config = "sft_5070ti",
    [int]$MaxSteps = 200,
    [int]$SaveEvery = 50,
    [string]$Dtype = "bfloat16",
    [ValidateSet("bounded", "epoch")][string]$Mode = "bounded",
    [string]$Resume = ""
)
$ErrorActionPreference = "Stop"
if (-not $env:MINIMIND_ROOT) { throw "请先设置 `$env:MINIMIND_ROOT" }
$Python = if ($env:MM_PYTHON) { $env:MM_PYTHON } else { "python" }
$Lab = Split-Path -Parent $PSScriptRoot
$FromWeight = Join-Path $env:MINIMIND_ROOT "out\full_sft_768.pth"
if (-not (Test-Path $FromWeight)) { throw "缺少 $FromWeight：先运行 run_sft_bounded.ps1（-MaskFault none）" }
$RunDir = Join-Path $Lab "runs\dpo_$Config"
New-Item -ItemType Directory -Force $RunDir | Out-Null

& $Python "$Lab\scripts\mmp.py" dpo_check --config $Config --policy $FromWeight --ref same --n 2 --beta 0.15 --check-ref-frozen --out "$RunDir\dpo_check_before.json"

if ($Mode -eq "bounded") {
    $argv = @("$Lab\scripts\mmp.py", "bounded_train", "--config", $Config, "--stage", "dpo",
              "--max-steps", $MaxSteps, "--save-every", $SaveEvery, "--dtype", $Dtype,
              "--from-weight", $FromWeight, "--save-dir", $RunDir, "--log-jsonl", "$RunDir\log_dpo.jsonl",
              "--export-pth", (Join-Path $env:MINIMIND_ROOT "out\dpo_768.pth"))
    if ($Resume) { $argv += @("--resume", $Resume) }
    & $Python @argv
    if ($LASTEXITCODE -ne 0) { throw "bounded_train 退出码 $LASTEXITCODE" }
    & $Python "$Lab\scripts\mmp.py" parse_log --input "$RunDir\log_dpo.jsonl" --csv "$RunDir\log_dpo.csv" --png "$RunDir\log_dpo.png" --fields loss,reward_margin,dpo_accuracy,lr,grad_norm
    $After = Join-Path $env:MINIMIND_ROOT "out\dpo_768.pth"
} else {
    Push-Location (Join-Path $env:MINIMIND_ROOT "trainer")
    try {
        & $Python train_dpo.py --epochs 1 --batch_size 4 --learning_rate 4e-8 --max_seq_len 1024 --beta 0.15 --dtype $Dtype `
            --from_weight full_sft --log_interval 20 --save_interval 500 --num_workers 4 --data_path ../dataset/dpo.jsonl 2>&1 |
            Tee-Object -FilePath "$RunDir\minimind_dpo_stdout.log"
    } finally { Pop-Location }
    & $Python "$Lab\scripts\mmp.py" parse_log --input "$RunDir\minimind_dpo_stdout.log" --format minimind --csv "$RunDir\minimind_dpo.csv" --png "$RunDir\minimind_dpo.png"
    $After = Join-Path $env:MINIMIND_ROOT "out\dpo_768.pth"
}
# 训练后：policy = dpo 权重，ref = full_sft 权重 → margin 应为正、loss < ln2
& $Python "$Lab\scripts\mmp.py" dpo_check --config $Config --policy $After --ref $FromWeight --n 2 --beta 0.15 --out "$RunDir\dpo_check_after.json"
& $Python "$Lab\scripts\mmp.py" eval_generate --config $Config --checkpoint $After --mode chat --out "$RunDir\eval.jsonl" --tag dpo
