[CmdletBinding()]
param(
    [string]$Root
)

$ErrorActionPreference = 'Stop'
if ([string]::IsNullOrWhiteSpace($Root)) {
    $scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
    $Root = Split-Path -Parent $scriptDir
}
$rootPath = [System.IO.Path]::GetFullPath($Root)
$courseRoot = Join-Path $rootPath 'outputs\0_basic_training'

$weeks = [ordered]@{
    1  = 'week01_tinystories_minigpt'
    2  = 'week02_nanovlm_vqa'
    3  = 'week03_v100_fp16_topology'
    4  = 'week04_tinyflowpolicy'
    5  = 'week05_data_contract_repro'
    6  = 'week06_single_gpu_profiling'
    7  = 'week07_ddp_scaling'
    8  = 'week08_fsdp_checkpoint'
    9  = 'week09_diffusion_vs_flow'
    10 = 'week10_smolvla'
    11 = 'week11_mini_wam'
    12 = 'week12_finexec_sft_cpt'
    13 = 'week13_finexec_scaling_peft_grpo'
    14 = 'week14_capstone'
}

$files = [ordered]@{
    '01_FOUNDATIONS.md'      = @{ Min = 5750; Terms = @('##', 'shape', '失败') }
    '02_LAB_GUIDE.md'        = @{ Min = 14000; Terms = @('验收', '证据', '预期') }
    '03_ORAL_EXAM.md'        = @{ Min = 1000; Terms = @('Recall', 'Debug', 'Trade-off') }
    '04_REFERENCE_ANSWERS.md'= @{ Min = 1800; Terms = @('评分', '误区') }
}

$errors = [System.Collections.Generic.List[string]]::new()
$warnings = [System.Collections.Generic.List[string]]::new()
$checked = 0
$idPattern = '(?i)(?<![A-Z0-9])(?:W(?:0[1-9]|1[0-4])-(?:Q|[A-Z]+)-?\d{1,3}|Q\d{2,3})(?![A-Z0-9])'
$linkPattern = '(?<!!)\[[^\]]+\]\(([^)]+)\)'

function Get-MarkdownProse([string]$Text) {
    $insideFence = $false
    $kept = [System.Collections.Generic.List[string]]::new()
    foreach ($line in ($Text -split "`r?`n")) {
        if ($line -match '^\s*(?:```|~~~)') {
            $insideFence = -not $insideFence
            continue
        }
        if ($insideFence -or $line -match '^(?: {4}|`t)') { continue }
        $kept.Add($line)
    }
    return ($kept -join "`n")
}

foreach ($entry in $weeks.GetEnumerator()) {
    $week = [int]$entry.Key
    $weekDir = Join-Path $courseRoot $entry.Value
    if (-not (Test-Path -LiteralPath $weekDir -PathType Container)) {
        $errors.Add("missing week directory: $weekDir")
        continue
    }

    $texts = @{}
    foreach ($fileEntry in $files.GetEnumerator()) {
        $path = Join-Path $weekDir $fileEntry.Key
        if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
            $errors.Add("missing file: $path")
            continue
        }
        $checked += 1
        $text = Get-Content -Raw -Encoding UTF8 -LiteralPath $path
        $texts[$fileEntry.Key] = $text

        if ($text.Trim().Length -lt $fileEntry.Value.Min) {
            $errors.Add("content too short ($($text.Trim().Length) < $($fileEntry.Value.Min)): $path")
        }
        foreach ($term in $fileEntry.Value.Terms) {
            if ($text.IndexOf($term, [System.StringComparison]::OrdinalIgnoreCase) -lt 0) {
                $errors.Add("missing required term '$term': $path")
            }
        }
        if ($fileEntry.Key -eq '01_FOUNDATIONS.md' -and $text -notmatch '(?i)失败|故障|风险|OOM|hang|NaN|不一致') {
            $errors.Add("foundations lacks failure/fault/risk discussion: $path")
        }
        if (-not $text.TrimStart().StartsWith('#')) {
            $errors.Add("missing Markdown title: $path")
        }

        if ($fileEntry.Key -eq '01_FOUNDATIONS.md') {
            $foundationGroups = @(
                @{ Name = 'data/model object'; Pattern = '(?i)数据|dataset|模型|model' },
                @{ Name = 'source/version policy'; Pattern = '(?i)官方|来源|revision|commit|tag|版本' },
                @{ Name = 'algorithm/formula'; Pattern = '(?i)公式|推导|算法|loss|目标函数' },
                @{ Name = 'shape/dtype worked example'; Pattern = '(?i)shape|dtype|float(?:16|32|64)|int(?:32|64)' },
                @{ Name = 'invariant/gate'; Pattern = '(?i)不变量|Gate|门槛|正确性' },
                @{ Name = 'infrastructure coupling'; Pattern = '(?i)infra|显存|吞吐|通信|I/O|数值' },
                @{ Name = 'self check'; Pattern = '(?i)自检|Teach-back|闭卷|思考题' }
            )
            foreach ($group in $foundationGroups) {
                if ($text -notmatch $group.Pattern) {
                    $errors.Add("foundations lacks $($group.Name): $path")
                }
            }
            $headingCount = [regex]::Matches($text, '(?m)^#{2,4}\s+').Count
            if ($headingCount -lt 8) {
                $errors.Add("foundations has only $headingCount substantive headings; need >= 8: $path")
            }
        }

        if ($fileEntry.Key -eq '02_LAB_GUIDE.md') {
            $labGroups = @(
                @{ Name = 'environment preflight'; Pattern = '(?i)环境|前置检查|preflight' },
                @{ Name = 'download or declared offline data'; Pattern = '(?i)下载|download|clone|合成数据|无需下载|不下载外部' },
                @{ Name = 'version pinning'; Pattern = '(?i)revision|commit|tag|版本|hash|SHA-256' },
                @{ Name = 'license/data authorization'; Pattern = '(?i)许可|license|授权|审批|批准|无需外部数据' },
                @{ Name = 'directory/files'; Pattern = '(?i)目录|文件' },
                @{ Name = 'implementation'; Pattern = '(?i)实现|代码' },
                @{ Name = 'training path'; Pattern = '(?i)train|训练' },
                @{ Name = 'incremental smoke path'; Pattern = '(?i)smoke|冒烟|probe|探针|最小.{0,8}运行|短.{0,8}运行|50-step' },
                @{ Name = 'independent evaluation/validation'; Pattern = '(?i)eval|评测|验证|validator|审计' },
                @{ Name = 'checkpoint/resume'; Pattern = '(?i)checkpoint|resume|恢复' },
                @{ Name = 'failure diagnosis'; Pattern = '(?i)若失败|故障|诊断|排查' },
                @{ Name = 'not-run evidence label'; Pattern = '(?i)待执行|不是实测|未实测|尚未.*运行' }
            )
            foreach ($group in $labGroups) {
                if ($text -notmatch $group.Pattern) {
                    $errors.Add("lab guide lacks $($group.Name): $path")
                }
            }
            $dayCount = [regex]::Matches($text, '(?im)^##\s+(?:Day|第\s*\d+\s*天)').Count
            if ($dayCount -lt 3) {
                $errors.Add("lab guide has only $dayCount day/phase sections; need >= 3: $path")
            }
            $actionCount = [regex]::Matches($text, '(?im)^\s*(?:python(?:3)?\s|CUDA_VISIBLE_DEVICES=|torchrun\s|git\s+(?:clone|ls-remote)|hf\s+download|nvidia-smi|nsys\s|ncu\s)').Count
            if ($actionCount -lt 8) {
                $errors.Add("lab guide has only $actionCount concrete command lines; need >= 8: $path")
            }
            if ($text -match '(?im)^\s*(?:TODO\b|pass\s*(?:#.*)?$)|此处省略|请自行补全|剩余代码自行') {
                $errors.Add("lab guide contains an unfinished implementation marker: $path")
            }
        }

        $markdownProse = Get-MarkdownProse $text
        foreach ($match in [regex]::Matches($markdownProse, $linkPattern)) {
            $rawTarget = $match.Groups[1].Value.Trim().Trim('<', '>')
            $target = ($rawTarget -split '[#?]', 2)[0]
            if (-not $target -or $target -match '^(?i:https?://|mailto:)') { continue }
            $candidate = [System.IO.Path]::GetFullPath((Join-Path (Split-Path -Parent $path) $target))
            if (-not (Test-Path -LiteralPath $candidate)) {
                $errors.Add("broken local link: $path -> $rawTarget")
            }
        }

        $lineNumber = 0
        foreach ($line in ($text -split "`r?`n")) {
            $lineNumber += 1
            if ($line -match '(?i)V100.*不支持.*FP16' -and $line -notmatch '笔误|错误|纠正|误解|误区|并非|不是|原句|输入') {
                $errors.Add("likely false V100/FP16 claim: ${path}:$lineNumber")
            }
            if ($line -match '已实测通过|已经实机通过|已在用户.*运行通过') {
                $warnings.Add("possible unsupported run claim: ${path}:$lineNumber")
            }
        }
    }

    if ($texts.ContainsKey('03_ORAL_EXAM.md') -and $texts.ContainsKey('04_REFERENCE_ANSWERS.md')) {
        $oralIds = @([regex]::Matches($texts['03_ORAL_EXAM.md'], $idPattern) | ForEach-Object { $_.Value.ToUpperInvariant() } | Sort-Object -Unique)
        $answerIds = @([regex]::Matches($texts['04_REFERENCE_ANSWERS.md'], $idPattern) | ForEach-Object { $_.Value.ToUpperInvariant() } | Sort-Object -Unique)
        if ($oralIds.Count -lt 20) {
            $errors.Add(('week {0:D2}: only {1} unique oral-exam IDs; need >= 20' -f $week, $oralIds.Count))
        }
        $missing = @($oralIds | Where-Object { $_ -notin $answerIds })
        $extra = @($answerIds | Where-Object { $_ -notin $oralIds })
        if ($missing.Count -gt 0) { $errors.Add(('week {0:D2}: questions without answers: {1}' -f $week, ($missing -join ', '))) }
        if ($extra.Count -gt 0) { $errors.Add(('week {0:D2}: answer IDs without questions: {1}' -f $week, ($extra -join ', '))) }
        if ($texts['03_ORAL_EXAM.md'] -match '参考答案|标准答案|答案[:：]') {
            $warnings.Add(('week {0:D2}: oral exam may leak an answer marker' -f $week))
        }
    }

    if ($texts.ContainsKey('02_LAB_GUIDE.md')) {
        $lab = $texts['02_LAB_GUIDE.md']
        if ($lab -notmatch '(?i)公司|V100') { $errors.Add(('week {0:D2}: lab guide lacks company/V100 boundary' -f $week)) }
        if ($lab -notmatch '(?i)5070\s*Ti|5070Ti') { $errors.Add(('week {0:D2}: lab guide lacks personal 5070 Ti lane' -f $week)) }
        if ($lab -notmatch '(?:不得|禁止|不能|严禁|不).{0,30}(?:导出|上传|外传|外带|带出)|(?:导出|上传|外传|外带|带出).{0,30}(?:不得|禁止|不能|严禁)|(?:公司|原始|所有|全部).{0,60}(?:留公司|留在公司|留.*内部|留.*本机|公司机器)') {
            $errors.Add(('week {0:D2}: lab guide lacks explicit company artifact boundary' -f $week))
        }
    }
}

Write-Output "Curriculum root: $rootPath"
Write-Output "Checked files: $checked/56"
foreach ($warning in $warnings) { Write-Output "WARN: $warning" }
foreach ($errorMessage in $errors) { Write-Output "ERROR: $errorMessage" }

if ($errors.Count -gt 0) {
    Write-Output "RESULT: FAIL ($($errors.Count) error(s), $($warnings.Count) warning(s))"
    exit 1
}

Write-Output "RESULT: PASS (0 errors, $($warnings.Count) warning(s))"
exit 0
