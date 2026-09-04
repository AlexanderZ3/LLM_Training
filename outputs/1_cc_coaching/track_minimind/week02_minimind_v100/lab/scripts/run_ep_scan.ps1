# 命令等级：模板（先替换 -OutDir / $env:MINIMIND_ROOT）
#
# Day 3：EP=1/2/4/8 扫描。EP=8 需要 num_experts=8（约束 ep_size<=E 且整除），
# 脚本用 -NumExperts 覆盖 config，不改 config 文件本身。
# 先跑 equiv（单进程、无通信）确认六步流水数值上等于原始 MOEFeedForward，
# 再跑各 EP 规模的 -TimeBreakdown。
# 预算：equiv < 30 秒；每个 EP 规模 MaxSteps=10 估计 1-2 分钟，合计 < 10 分钟。
param(
    [string]$OutDir = "",
    [string]$Config = "",
    [int]$Nproc = 8,
    [int]$NumExperts = 8,
    [int]$Batch = 8,
    [int]$SeqLen = 512,
    [int]$MaxSteps = 10,
    [string]$Dtype = "float16",
    [double]$CapacityFactor = 0,       # 0 = dropless
    [int]$NcclTimeoutS = 600,
    [string]$Python = "python"
)

$ErrorActionPreference = "Stop"
$LabDir = Split-Path -Parent $PSScriptRoot
if (-not $OutDir) { $OutDir = Join-Path $LabDir "runs\ep_scan" }      # 公司环境改成公司内部路径
if (-not $Config) { $Config = Join-Path $LabDir "configs\v100_moe.json" }
if (-not $env:MINIMIND_ROOT) { throw "请先设置 `$env:MINIMIND_ROOT = '<MiniMind 仓库绝对路径>'" }
$env:PYTHONPATH = (Join-Path $LabDir "src") + ";" + $env:PYTHONPATH
if (-not $env:TORCH_NCCL_ASYNC_ERROR_HANDLING) { $env:TORCH_NCCL_ASYNC_ERROR_HANDLING = "1" }
if (-not $env:NCCL_ASYNC_ERROR_HANDLING) { $env:NCCL_ASYNC_ERROR_HANDLING = "1" }

New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
Write-Host "[run_ep_scan] nproc=$Nproc E=$NumExperts batch=$Batch seq=$SeqLen steps=$MaxSteps"

Write-Host "--- 门 1：EP=1 数值等价（单进程，无集合通信）"
& $Python -m mm_dist.ep_moe equiv `
    --config $Config --out-dir $OutDir --num-experts $NumExperts `
    --batch 2 --seq-len 64 --tol 1e-5 --force-cpu --dtype float32
if ($LASTEXITCODE -ne 0) { throw "EP=1 等价检查失败，先修 permute/unpermute 再往下走" }

foreach ($Ep in @(1, 2, 4, 8)) {
    if ($Ep -gt $NumExperts) {
        Write-Host "--- 跳过 EP=$Ep：ep_size 不能大于 num_experts=$NumExperts"
        continue
    }
    Write-Host "--- EP=$Ep"
    torchrun --nproc_per_node $Nproc -m mm_dist.ep_moe run `
        --config $Config --out-dir $OutDir --dtype $Dtype `
        --ep-size $Ep --num-experts $NumExperts `
        --capacity-factor $CapacityFactor --time-breakdown `
        --batch $Batch --seq-len $SeqLen --max-steps $MaxSteps `
        --nccl-timeout-s $NcclTimeoutS --log-name "ep$Ep"
    if ($LASTEXITCODE -ne 0) { throw "EP=$Ep 运行失败，退出码 $LASTEXITCODE" }
}

Write-Host "[run_ep_scan] 证据字段（抽象值）："
Write-Host "  equiv 的 max_abs_diff 与 pass"
Write-Host "  每个 EP：time_ms 里 dispatch_a2a / combine_a2a / expert_compute 的占比"
Write-Host "  wait_before_dispatch 在各 rank 之间的极差（木桶效应）"
Write-Host "  recv_tokens 的最大/最小比值（负载不均程度）"
