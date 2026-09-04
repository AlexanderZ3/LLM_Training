# 命令等级：模板（先替换 -OutDir / $env:MINIMIND_ROOT）
#
# Day 5：GRPO 三种角色放置的对照。
#   replicate    每卡全量（policy+rollout+reward），DDP 包 policy
#   policy_fsdp  policy 分片到 8 卡，rollout/reward 仍每卡一份（需要 GPU）
#   split_roles  4 卡 policy / 2 卡 rollout / 2 卡 reward，每轮 broadcast 同步权重
# 看三个数：peak_mem_mb、step_time_s、role_idle_ms；
# 权重同步正确性看 param_checksum（split_roles 下 broadcast 后各 rank 必须相同）。
# 预算：3 种放置 × MaxSteps=5 × 生成 64 token，估计 6-10 分钟。
# 时间不够时优先保 policy_fsdp（Gate 依赖它），split_roles 允许 INCONCLUSIVE。
param(
    [string]$OutDir = "",
    [string]$Config = "",
    [int]$Nproc = 8,
    [int]$NPolicy = 4,
    [int]$NRollout = 2,
    [int]$NReward = 2,
    [int]$NPrompts = 4,
    [int]$GroupSize = 4,
    [int]$PromptLen = 32,
    [int]$MaxGenLen = 64,
    [int]$MaxSteps = 5,
    [string]$Dtype = "float16",
    [ValidateSet("rule", "model")][string]$Reward = "rule",
    [string]$RewardPath = "",          # -Reward model 时指向公司内部已有的 HF 目录
    [int]$NcclTimeoutS = 600
)

$ErrorActionPreference = "Stop"
$LabDir = Split-Path -Parent $PSScriptRoot
if (-not $OutDir) { $OutDir = Join-Path $LabDir "runs\grpo_roles" }   # 公司环境改成公司内部路径
if (-not $Config) { $Config = Join-Path $LabDir "configs\v100_dense.json" }
if (-not $env:MINIMIND_ROOT) { throw "请先设置 `$env:MINIMIND_ROOT = '<MiniMind 仓库绝对路径>'" }
if ($Reward -eq "model" -and -not $RewardPath) { throw "-Reward model 需要 -RewardPath <公司内部 HF 模型目录>" }
$env:PYTHONPATH = (Join-Path $LabDir "src") + ";" + $env:PYTHONPATH
if (-not $env:TORCH_NCCL_ASYNC_ERROR_HANDLING) { $env:TORCH_NCCL_ASYNC_ERROR_HANDLING = "1" }
if (-not $env:NCCL_ASYNC_ERROR_HANDLING) { $env:NCCL_ASYNC_ERROR_HANDLING = "1" }

New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
Write-Host "[run_grpo_roles] nproc=$Nproc 角色=$NPolicy/$NRollout/$NReward prompts=${NPrompts}x$GroupSize gen=$MaxGenLen steps=$MaxSteps"

foreach ($Placement in @("replicate", "policy_fsdp", "split_roles")) {
    Write-Host "--- placement=$Placement"
    $RewardArgs = @("--reward", $Reward)
    if ($Reward -eq "model") { $RewardArgs += @("--reward-path", $RewardPath) }
    torchrun --nproc_per_node $Nproc -m mm_dist.grpo_roles `
        --config $Config --out-dir $OutDir --dtype $Dtype `
        --placement $Placement `
        --n-policy $NPolicy --n-rollout $NRollout --n-reward $NReward `
        --n-prompts $NPrompts --group-size $GroupSize `
        --prompt-len $PromptLen --max-gen-len $MaxGenLen `
        --max-steps $MaxSteps @RewardArgs `
        --nccl-timeout-s $NcclTimeoutS --log-name "grpo_$Placement"
    if ($LASTEXITCODE -ne 0) { throw "placement=$Placement 失败，退出码 $LASTEXITCODE" }
}

Write-Host "[run_grpo_roles] 证据字段（抽象值）："
Write-Host "  mem_ratio_fsdp_vs_replicate = policy_fsdp.peak_mem_mb / replicate.peak_mem_mb"
Write-Host "  step_time_ratio（三种放置两两比值）"
Write-Host "  split_roles 各角色的 role_idle_ms 占 step_time 的比例"
Write-Host "  broadcast 后各 rank 的 param_checksum 是否一致（布尔值）"
