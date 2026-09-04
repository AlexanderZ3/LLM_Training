# 命令等级：模板（先替换 -OutDir / $env:MINIMIND_ROOT，其余可直接跑）
#
# Day 1：fixed-global-batch 等价。同一个 -GlobalBatch，跑 1 / 2 / 8 卡三次，逐步比 loss。
# 判定：max|delta| < 1e-4（float32 等价模式）。
# 预算：tiny 配置每次 < 1 分钟；v100_dense + MaxSteps=20 估计每次 2-4 分钟（估算）。
param(
    [string]$OutDir = "",
    [string]$Config = "",
    [int]$GlobalBatch = 64,
    [int]$MaxSteps = 20,
    [int]$NcclTimeoutS = 600,
    [string]$Python = "python"
)

$ErrorActionPreference = "Stop"
$LabDir = Split-Path -Parent $PSScriptRoot
if (-not $OutDir) { $OutDir = Join-Path $LabDir "runs\ddp_equiv" }   # 公司环境请改成公司内部路径
if (-not $Config) { $Config = Join-Path $LabDir "configs\v100_dense.json" }

if (-not $env:MINIMIND_ROOT) { throw "请先设置 `$env:MINIMIND_ROOT = '<MiniMind 仓库绝对路径>'" }
$env:PYTHONPATH = (Join-Path $LabDir "src") + ";" + $env:PYTHONPATH
if (-not $env:TORCH_NCCL_ASYNC_ERROR_HANDLING) { $env:TORCH_NCCL_ASYNC_ERROR_HANDLING = "1" }
if (-not $env:NCCL_ASYNC_ERROR_HANDLING) { $env:NCCL_ASYNC_ERROR_HANDLING = "1" }

New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
Write-Host "[run_ddp_equiv] out=$OutDir config=$Config G=$GlobalBatch steps=$MaxSteps"

function Invoke-One([int]$Nproc, [int]$Accum) {
    $micro = $GlobalBatch / ($Nproc * $Accum)
    Write-Host "--- world_size=$Nproc accum=$Accum (micro=$micro)"
    torchrun --nproc_per_node $Nproc -m mm_dist.train_ddp `
        --config $Config --out-dir $OutDir `
        --global-batch $GlobalBatch --accum $Accum `
        --max-steps $MaxSteps --equiv-check `
        --nccl-timeout-s $NcclTimeoutS `
        --log-name "equiv_ws$Nproc"
    if ($LASTEXITCODE -ne 0) { throw "world_size=$Nproc 运行失败，退出码 $LASTEXITCODE" }
}

Invoke-One 1 8
Invoke-One 2 4
Invoke-One 8 1

Write-Host "--- 比对 ws1 vs ws2"
& $Python -m mm_dist.train_ddp --compare-json `
    (Join-Path $OutDir "equiv_ws1_rank0.jsonl") (Join-Path $OutDir "equiv_ws2_rank0.jsonl") `
    --compare-tol 1e-4
Write-Host "--- 比对 ws1 vs ws8"
& $Python -m mm_dist.train_ddp --compare-json `
    (Join-Path $OutDir "equiv_ws1_rank0.jsonl") (Join-Path $OutDir "equiv_ws8_rank0.jsonl") `
    --compare-tol 1e-4

Write-Host "[run_ddp_equiv] 证据字段：max|delta|、steps_compared、micro_batch（三次必须相同）"
