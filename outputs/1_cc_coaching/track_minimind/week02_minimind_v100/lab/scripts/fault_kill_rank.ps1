# 命令等级：模板（先替换 -OutDir / $env:MINIMIND_ROOT）
#
# Day 5 故障演练：训练到第 -AfterStep 步时 kill 掉 -KillRank，观察其余 rank 的行为。
# 跑两遍，只差一个环境变量：
#   A) async=1（默认）：watchdog 在 -NcclTimeoutS 秒后让其余 rank 崩溃，torchrun 退出非 0
#   B) async=0        ：其余 rank 静默阻塞到进程组 timeout
# 计时是主要观测：从被杀那一刻到 torchrun 退出的墙钟时间。
# -NcclTimeoutS 默认压到 60 秒，好让 B) 也能在 10 分钟预算内结束。
param(
    [string]$OutDir = "",
    [string]$Config = "",
    [int]$Nproc = 8,
    [int]$KillRank = 3,
    [int]$AfterStep = 5,
    [int]$MaxSteps = 20,
    [int]$GlobalBatch = 64,
    [string]$Dtype = "float16",
    [int]$NcclTimeoutS = 60,
    [string]$Python = "python"
)

$ErrorActionPreference = "Stop"
$LabDir = Split-Path -Parent $PSScriptRoot
if (-not $OutDir) { $OutDir = Join-Path $LabDir "runs\fault" }        # 公司环境改成公司内部路径
if (-not $Config) { $Config = Join-Path $LabDir "configs\v100_dense.json" }
if (-not $env:MINIMIND_ROOT) { throw "请先设置 `$env:MINIMIND_ROOT = '<MiniMind 仓库绝对路径>'" }
$env:PYTHONPATH = (Join-Path $LabDir "src") + ";" + $env:PYTHONPATH

New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
Write-Host "[fault_kill_rank] 先看一眼当前的 NCCL 相关环境变量与建议值"
& $Python -m mm_dist.faults env --nccl-timeout-s $NcclTimeoutS

function Invoke-Case([int]$Async) {
    Write-Host ""
    Write-Host "=== 用例：TORCH_NCCL_ASYNC_ERROR_HANDLING=$Async ==="
    $env:TORCH_NCCL_ASYNC_ERROR_HANDLING = "$Async"
    $env:NCCL_ASYNC_ERROR_HANDLING = "$Async"
    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    torchrun --nproc_per_node $Nproc -m mm_dist.faults demo `
        --config $Config --out-dir $OutDir --dtype $Dtype `
        --global-batch $GlobalBatch --accum 1 --max-steps $MaxSteps `
        --kill-rank $KillRank --after-step $AfterStep `
        --async-error-handling $Async --overwrite-env 1 `
        --nccl-timeout-s $NcclTimeoutS `
        --log-name "fault_async$Async"
    $rc = $LASTEXITCODE
    $sw.Stop()
    Write-Host "[fault_kill_rank] async=$Async torchrun 退出码=$rc 墙钟=$([int]$sw.Elapsed.TotalSeconds)s"
}

Invoke-Case 1
Invoke-Case 0

Write-Host ""
Write-Host "[fault_kill_rank] 证据字段（抽象值）："
Write-Host "  killed_at_step（应等于 AfterStep=$AfterStep）"
Write-Host "  两个用例的 torchrun 退出码"
Write-Host "  两个用例从被杀到退出的墙钟秒数之比"
Write-Host "  存活 rank 的日志最后一条 step（说明它们停在哪一步）"
Write-Host "恢复演练接 run_reshard.ps1：用 4 卡从最后一个 checkpoint 恢复并 verify。"
