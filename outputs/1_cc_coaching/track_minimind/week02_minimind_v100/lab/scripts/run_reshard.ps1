# 命令等级：模板（先替换 -OutDir / -CkptDir / $env:MINIMIND_ROOT；需要 GPU）
#
# Day 2 下半场：8 卡训练 → 保存分片 checkpoint → 4 卡恢复 → 验证 loss 连续。
# 判定：verify 输出的 abs_delta < -Tol，且 converted_step == step * saved_ws / current_ws。
# 预算：save 约 MaxSteps 步 + verify 一次前向，估计 3-6 分钟。
param(
    [string]$OutDir = "",
    [string]$CkptDir = "",
    [string]$Config = "",
    [int]$SaveNproc = 8,
    [int]$LoadNproc = 4,
    [int]$GlobalBatch = 64,
    [int]$MaxSteps = 20,
    [string]$Dtype = "float16",
    [ValidateSet("sharded", "full")][string]$Format = "sharded",
    [double]$Tol = 1e-2,               # fp16 用 1e-2；-Dtype float32 时用 1e-4
    [int]$NcclTimeoutS = 600
)

$ErrorActionPreference = "Stop"
$LabDir = Split-Path -Parent $PSScriptRoot
if (-not $OutDir) { $OutDir = Join-Path $LabDir "runs\reshard" }      # 公司环境改成公司内部路径
if (-not $CkptDir) { $CkptDir = Join-Path $OutDir "ckpt_sharded" }
if (-not $Config) { $Config = Join-Path $LabDir "configs\v100_dense.json" }
if (-not $env:MINIMIND_ROOT) { throw "请先设置 `$env:MINIMIND_ROOT = '<MiniMind 仓库绝对路径>'" }
$env:PYTHONPATH = (Join-Path $LabDir "src") + ";" + $env:PYTHONPATH
if (-not $env:TORCH_NCCL_ASYNC_ERROR_HANDLING) { $env:TORCH_NCCL_ASYNC_ERROR_HANDLING = "1" }
if (-not $env:NCCL_ASYNC_ERROR_HANDLING) { $env:NCCL_ASYNC_ERROR_HANDLING = "1" }

New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
Write-Host "[run_reshard] $SaveNproc 卡保存 -> $LoadNproc 卡恢复  format=$Format tol=$Tol"

Write-Host "--- 保存（world_size=$SaveNproc）"
torchrun --nproc_per_node $SaveNproc -m mm_dist.ckpt_reshard save `
    --config $Config --ckpt-dir $CkptDir --out-dir $OutDir `
    --dtype $Dtype --format $Format --sharding full_shard `
    --global-batch $GlobalBatch --accum 1 --max-steps $MaxSteps `
    --nccl-timeout-s $NcclTimeoutS
if ($LASTEXITCODE -ne 0) { throw "保存失败，退出码 $LASTEXITCODE" }

$Accum = [int]($SaveNproc / $LoadNproc)
$ResultJson = Join-Path $OutDir "verify_${SaveNproc}to${LoadNproc}.json"
Write-Host "--- 恢复并验证（world_size=$LoadNproc, accum=$Accum）"
torchrun --nproc_per_node $LoadNproc -m mm_dist.ckpt_reshard verify `
    --config $Config --ckpt-dir $CkptDir --out-dir $OutDir `
    --dtype $Dtype --format $Format --sharding full_shard `
    --global-batch $GlobalBatch --accum $Accum `
    --max-steps $MaxSteps --tol $Tol --result-json $ResultJson `
    --nccl-timeout-s $NcclTimeoutS
if ($LASTEXITCODE -ne 0) { Write-Host "[run_reshard] verify 判定为 FAIL（退出码 $LASTEXITCODE），见 $ResultJson" }

Write-Host "[run_reshard] 证据字段：pass、abs_delta、tol、saved_world_size、current_world_size"
Write-Host "              （只带出这五个抽象值，checkpoint 与日志留在公司内）"
