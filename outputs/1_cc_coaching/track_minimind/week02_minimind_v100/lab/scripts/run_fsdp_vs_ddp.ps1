# 命令等级：模板（先替换 -OutDir / $env:MINIMIND_ROOT；需要 8 张 V100，CPU 上跑不了 FSDP）
#
# Day 2：同 global batch 下 DDP 与 FSDP 三种策略的对照。
# 看四个数：step_time_s、peak_mem_mb、collectives_total、loss 是否与 DDP 一致。
# 预期（估算，非实测）：64M 模型上 FULL_SHARD 的 step time 高于 DDP（每层 all_gather
# 的延迟掩盖不住），显存则低于 DDP。
# 预算：4 次运行 × MaxSteps=20，估计合计 8-12 分钟；超时先把 MaxSteps 降到 10。
param(
    [string]$OutDir = "",
    [string]$Config = "",
    [int]$Nproc = 8,
    [int]$GlobalBatch = 64,
    [int]$MaxSteps = 20,
    [string]$Dtype = "float16",        # V100 没有 bf16；bfloat16 会被显式拒绝
    [int]$ActivationCheckpointing = 0,
    [int]$NcclTimeoutS = 600,
    [string]$Python = "python"
)

$ErrorActionPreference = "Stop"
$LabDir = Split-Path -Parent $PSScriptRoot
if (-not $OutDir) { $OutDir = Join-Path $LabDir "runs\fsdp_vs_ddp" }   # 公司环境改成公司内部路径
if (-not $Config) { $Config = Join-Path $LabDir "configs\v100_dense.json" }
if (-not $env:MINIMIND_ROOT) { throw "请先设置 `$env:MINIMIND_ROOT = '<MiniMind 仓库绝对路径>'" }
$env:PYTHONPATH = (Join-Path $LabDir "src") + ";" + $env:PYTHONPATH
if (-not $env:TORCH_NCCL_ASYNC_ERROR_HANDLING) { $env:TORCH_NCCL_ASYNC_ERROR_HANDLING = "1" }
if (-not $env:NCCL_ASYNC_ERROR_HANDLING) { $env:NCCL_ASYNC_ERROR_HANDLING = "1" }

New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
Write-Host "[run_fsdp_vs_ddp] nproc=$Nproc G=$GlobalBatch steps=$MaxSteps dtype=$Dtype ac=$ActivationCheckpointing"

Write-Host "--- 基线：DDP"
torchrun --nproc_per_node $Nproc -m mm_dist.train_ddp `
    --config $Config --out-dir $OutDir --dtype $Dtype `
    --global-batch $GlobalBatch --accum 1 --max-steps $MaxSteps `
    --nccl-timeout-s $NcclTimeoutS --log-name "ddp_ws$Nproc"
if ($LASTEXITCODE -ne 0) { throw "DDP 基线失败，退出码 $LASTEXITCODE" }

foreach ($Strategy in @("full_shard", "shard_grad_op", "no_shard")) {
    Write-Host "--- FSDP $Strategy"
    torchrun --nproc_per_node $Nproc -m mm_dist.train_fsdp `
        --config $Config --out-dir $OutDir --dtype $Dtype `
        --sharding $Strategy --activation-checkpointing $ActivationCheckpointing `
        --global-batch $GlobalBatch --accum 1 --max-steps $MaxSteps `
        --nccl-timeout-s $NcclTimeoutS --log-name "fsdp_${Strategy}_ws$Nproc"
    if ($LASTEXITCODE -ne 0) { throw "FSDP $Strategy 失败，退出码 $LASTEXITCODE" }
}

Write-Host "--- loss 一致性：DDP vs FULL_SHARD（fp16 下容差放宽到 1e-2）"
& $Python -m mm_dist.train_ddp --compare-json `
    (Join-Path $OutDir "ddp_ws${Nproc}_rank0.jsonl") `
    (Join-Path $OutDir "fsdp_full_shard_ws${Nproc}_rank0.jsonl") --compare-tol 1e-2

Write-Host "[run_fsdp_vs_ddp] 证据字段（只带出比值，不带出绝对数）："
Write-Host "  step_time_ratio = fsdp.step_time_s / ddp.step_time_s"
Write-Host "  mem_ratio       = fsdp.peak_mem_mb / ddp.peak_mem_mb"
Write-Host "  collectives_total（DDP 与三种 FSDP 各一个整数）、fsdp_units（应为层数+1）"
