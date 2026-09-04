# 命令等级：可直接执行（改 -Python 与 -Out 即可）
# 作用：整夜无人值守下载 MiniMind 全量数据集 + 奖励模型；断线自动重试，可随时 Ctrl-C 后重跑续传。
# 用法：
#   powershell -NoProfile -ExecutionPolicy Bypass -File lab\scripts\download_all.ps1
#   powershell -NoProfile -ExecutionPolicy Bypass -File lab\scripts\download_all.ps1 -Tier mini
#   powershell -NoProfile -ExecutionPolicy Bypass -File lab\scripts\download_all.ps1 -Endpoint https://hf-mirror.com
param(
    [ValidateSet("mini", "full", "reward", "all")][string]$Tier = "all",
    [string]$Out = "",
    [string]$Python = "D:\Software\Large\Anconda\envs\ResearchAgentPy310\python.exe",
    [string]$Endpoint = "",
    [int]$OuterRounds = 20,          # 外层重扫轮数：整夜跑给足重试机会
    [int]$SleepBetweenRounds = 60,   # 每轮之间等待秒数
    [switch]$NoHash,                 # 跳过 sha256（27 GB 全量哈希约十几分钟）
    [switch]$DryRun
)
$ErrorActionPreference = "Stop"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUNBUFFERED = "1"

$WeekDir = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
if ($Out -eq "") { $Out = Join-Path $WeekDir "datasets" }
$Script = Join-Path $PSScriptRoot "download_datasets.py"

if (-not (Test-Path $Python)) { throw "找不到解释器：$Python（用 -Python 指定）" }
if (-not (Test-Path $Script)) { throw "找不到 download_datasets.py：$Script" }

$argsBase = @($Script, "--tier", $Tier, "--out", $Out, "--retries", "6", "--rounds", "2")
if ($Endpoint -ne "") { $argsBase += @("--endpoint", $Endpoint) }
if ($NoHash) { $argsBase += "--no-hash" }

if ($DryRun) {
    & $Python @($argsBase + "--dry-run")
    exit $LASTEXITCODE
}

New-Item -ItemType Directory -Force $Out | Out-Null
$RunLog = Join-Path $Out ("download_run_{0}.log" -f (Get-Date -Format "yyyyMMdd_HHmmss"))
Write-Host "[download_all] tier=$Tier"
Write-Host "[download_all] out=$Out"
Write-Host "[download_all] 本轮日志=$RunLog"
Write-Host "[download_all] 累积日志=$(Join-Path $Out 'download.log')"
Write-Host "[download_all] 最多 $OuterRounds 轮，每轮间隔 ${SleepBetweenRounds}s。Ctrl-C 可随时停止，重跑本脚本续传。"
Write-Host ""

$exit = 1
for ($i = 1; $i -le $OuterRounds; $i++) {
    Write-Host "===== 外层第 $i/$OuterRounds 轮  $(Get-Date -Format 'HH:mm:ss') ====="
    "===== 外层第 $i/$OuterRounds 轮  $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') =====" |
        Out-File -Append -Encoding utf8 $RunLog
    & $Python @argsBase 2>&1 | Tee-Object -Append -FilePath $RunLog
    $exit = $LASTEXITCODE
    if ($exit -eq 0) {
        Write-Host ""
        Write-Host "[download_all] 全部完成（外层第 $i 轮）。"
        break
    }
    if ($exit -eq 2) {
        Write-Host "[download_all] 磁盘空间不足，停止。换 -Out 到别的盘，或用 -Tier mini。"
        break
    }
    if ($i -lt $OuterRounds) {
        Write-Host "[download_all] 本轮仍有未完成项，${SleepBetweenRounds}s 后再来一轮…"
        Start-Sleep -Seconds $SleepBetweenRounds
    }
}

Write-Host ""
if ($exit -eq 0) {
    Write-Host "[download_all] 校验清单：$(Join-Path $Out 'DOWNLOAD_MANIFEST.json')"
    if (-not $NoHash) { Write-Host "[download_all] 哈希清单：$(Join-Path $Out 'SHA256SUMS.txt')" }
} else {
    Write-Host "[download_all] 退出码 $exit —— 还有文件未就绪。重跑本脚本即可继续（已完成的会跳过）。"
}
exit $exit
