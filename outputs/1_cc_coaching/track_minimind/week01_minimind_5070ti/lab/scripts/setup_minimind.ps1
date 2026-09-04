# 命令等级：模板 —— 先改 -Root（克隆目录）与 -Python（5070 Ti 机器上装好 torch/huggingface_hub 的解释器）再运行。
# 作用：克隆 MiniMind → checkout 锁定 commit → 用 huggingface_hub 下载周卡列出的 4 个 jsonl 到 dataset/ → 写 dataset/SHA256SUMS.txt
# 用法：powershell -NoProfile -ExecutionPolicy Bypass -File lab/scripts/setup_minimind.ps1 -Root D:\work\minimind -Python python [-SkipData]
param(
    [string]$Root = "$HOME\minimind",
    [string]$Python = "python",
    [switch]$SkipData
)
$ErrorActionPreference = "Stop"
$Commit = "7a6fddd63a30c06b2fdd5fac4089922b29bc841b"
$Repo = "https://github.com/jingyaogong/minimind"
$Files = @("pretrain_t2t_mini.jsonl", "sft_t2t_mini.jsonl", "dpo.jsonl", "rlaif.jsonl")

if (-not (Test-Path (Join-Path $Root ".git"))) {
    Write-Host "[setup] cloning $Repo -> $Root"
    git clone --filter=blob:none $Repo $Root
}
git -C $Root fetch --all --quiet
git -C $Root checkout --quiet $Commit
$head = (git -C $Root rev-parse HEAD).Trim()
if ($head -ne $Commit) { throw "checkout 失败：HEAD=$head 期望=$Commit" }
Write-Host "[setup] MiniMind at $Commit"

$DataDir = Join-Path $Root "dataset"
if (-not $SkipData) {
    # 数据集：HF jingyaogong/minimind_dataset（约 3 GB；需要网络；国内可先 $env:HF_ENDPOINT="https://hf-mirror.com"）
    $py = @"
from huggingface_hub import hf_hub_download
import sys
files = sys.argv[2:]
for f in files:
    p = hf_hub_download(repo_id='jingyaogong/minimind_dataset', filename=f, repo_type='dataset', local_dir=sys.argv[1])
    print('downloaded', p)
"@
    & $Python -c $py $DataDir @Files
    if ($LASTEXITCODE -ne 0) { throw "数据下载失败（退出码 $LASTEXITCODE）" }
}

$sums = @()
foreach ($f in $Files) {
    $p = Join-Path $DataDir $f
    if (Test-Path $p) {
        $h = (Get-FileHash -Algorithm SHA256 $p).Hash.ToLower()
        $size = (Get-Item $p).Length
        $sums += "$h  $f  $size"
        Write-Host "[setup] $f sha256=$h size=$size"
    } else {
        Write-Host "[setup] 缺少 $p"
    }
}
if ($sums.Count -gt 0) {
    $sums | Set-Content -Encoding utf8 (Join-Path $DataDir "SHA256SUMS.txt")
    Write-Host "[setup] written $(Join-Path $DataDir 'SHA256SUMS.txt')"
}
Write-Host "[setup] done. 设置：`$env:MINIMIND_ROOT = '$Root'"
