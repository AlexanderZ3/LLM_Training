# 命令等级：模板（先替换 -OutDir / $env:MINIMIND_ROOT）
#
# Day 4：热点专家（router bias）× aux loss 系数 的二维扫描，输出一张 CSV。
# 自变量：-BiasList（给 -ExpertId 那个专家的 logit 加常数）、-AuxCoefList
# 因变量：max_load_ratio、router_entropy、mean_logits_loss、mean_wait_ms
# 预期读法（估算）：bias 增大 -> max_load_ratio 上升、entropy 下降、wait_ms 分化；
# aux_coef 增大 -> max_load_ratio 回落到 ~1，但 mean_logits_loss 可能变差（失败模式 #7）。
# 预算：默认 5×4=20 组 × 10 步，估计 6-10 分钟；超预算先把 -BiasList 砍到 "0,1,4"。
param(
    [string]$OutDir = "",
    [string]$Config = "",
    [int]$Nproc = 8,
    [string]$BiasList = "0,0.5,1,2,4",
    [string]$AuxCoefList = "0,0.001,0.01,0.1",
    [int]$ExpertId = 0,
    [int]$GlobalBatch = 64,
    [int]$MaxSteps = 10,
    [string]$Dtype = "float16",
    [int]$NcclTimeoutS = 600
)

$ErrorActionPreference = "Stop"
$LabDir = Split-Path -Parent $PSScriptRoot
if (-not $OutDir) { $OutDir = Join-Path $LabDir "runs\moe_sweep" }    # 公司环境改成公司内部路径
if (-not $Config) { $Config = Join-Path $LabDir "configs\v100_moe.json" }
if (-not $env:MINIMIND_ROOT) { throw "请先设置 `$env:MINIMIND_ROOT = '<MiniMind 仓库绝对路径>'" }
$env:PYTHONPATH = (Join-Path $LabDir "src") + ";" + $env:PYTHONPATH
if (-not $env:TORCH_NCCL_ASYNC_ERROR_HANDLING) { $env:TORCH_NCCL_ASYNC_ERROR_HANDLING = "1" }
if (-not $env:NCCL_ASYNC_ERROR_HANDLING) { $env:NCCL_ASYNC_ERROR_HANDLING = "1" }

New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
$CsvPath = Join-Path $OutDir "moe_sweep.csv"
Write-Host "[run_moe_sweep] bias=$BiasList aux=$AuxCoefList expert=$ExpertId steps=$MaxSteps"

torchrun --nproc_per_node $Nproc -m mm_dist.router_stats sweep `
    --config $Config --out-dir $OutDir --dtype $Dtype `
    --expert-id $ExpertId --bias-list $BiasList --aux-coef-list $AuxCoefList `
    --global-batch $GlobalBatch --accum 1 --max-steps $MaxSteps `
    --csv $CsvPath --nccl-timeout-s $NcclTimeoutS
if ($LASTEXITCODE -ne 0) { throw "扫描失败，退出码 $LASTEXITCODE" }

$LastBias = ($BiasList -split ",")[-1]
Write-Host "--- 单点复现（bias=$LastBias，写逐步 JSONL 便于看曲线）"
torchrun --nproc_per_node $Nproc -m mm_dist.router_stats stats `
    --config $Config --out-dir $OutDir --dtype $Dtype `
    --expert-id $ExpertId --bias $LastBias --aux-coef 0.0005 `
    --global-batch $GlobalBatch --accum 1 --max-steps $MaxSteps `
    --log-name "hot_expert" --nccl-timeout-s $NcclTimeoutS
if ($LASTEXITCODE -ne 0) { throw "单点复现失败，退出码 $LASTEXITCODE" }

Write-Host "[run_moe_sweep] 证据字段（抽象值）：CSV 里的 bias、aux_coef、max_load_ratio、"
Write-Host "  router_entropy、mean_logits_loss、mean_wait_ms；不带出 CSV 原文与日志。"
Write-Host "[run_moe_sweep] CSV -> $CsvPath"
