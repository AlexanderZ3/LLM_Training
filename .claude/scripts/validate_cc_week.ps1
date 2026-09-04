<#
.SYNOPSIS
  Static validator for one cc week package under outputs/1_cc_coaching/weekNN_<slug>/.
.DESCRIPTION
  Checks structure, required sections, task-card granularity contract, exam/answer ID mapping,
  banned placeholder words, export-risk phrases, and local markdown links.
  Read-only. Does not run training, does not access company assets, does not use the network.
  Exit code 0 = no errors, 1 = errors found, 2 = usage / package not found.
.EXAMPLE
  powershell -NoProfile -ExecutionPolicy Bypass -File .claude/scripts/validate_cc_week.ps1 -Week 01
#>
param(
    [Parameter(Mandatory = $true)][string]$Week,
    [string]$Root = "",
    [string]$Track = "",          # optional sub-track under outputs/1_cc_coaching, e.g. "track_minimind"
    [string]$IdPrefix = "W"       # exam ID prefix: W for the 14-week main track, M for minimind track, etc.
)

$ErrorActionPreference = "Stop"
if ($Root -eq "") { $Root = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path }
$Week = $Week.PadLeft(2, "0")

$script:Errors = New-Object System.Collections.Generic.List[string]
$script:Warnings = New-Object System.Collections.Generic.List[string]
function Add-Err([string]$m) { $script:Errors.Add($m) }
function Add-Warn([string]$m) { $script:Warnings.Add($m) }
function Read-Text([string]$p) { return [System.IO.File]::ReadAllText($p, [System.Text.Encoding]::UTF8) }

# ---------- locate package ----------
$base = Join-Path $Root "outputs\1_cc_coaching"
if ($Track -ne "") { $base = Join-Path $base $Track }
if (-not (Test-Path $base)) { Write-Host "ERROR: $base not found"; exit 2 }
$pkgDirs = @(Get-ChildItem -Path $base -Directory | Where-Object { $_.Name -match "^week$Week`_" })
if ($pkgDirs.Count -eq 0) { Write-Host "ERROR: no package matching week${Week}_* under $base"; exit 2 }
if ($pkgDirs.Count -gt 1) { Write-Host "ERROR: multiple packages for week $Week"; exit 2 }
$pkg = $pkgDirs[0].FullName
Write-Host "Package: $pkg"

# ---------- required files ----------
$requiredFiles = @(
    "00_WEEK_CARD.md", "01_FOUNDATIONS.md", "02_LAB_GUIDE.md",
    "03_TASK_CARDS\day0.md", "03_TASK_CARDS\day1.md", "03_TASK_CARDS\day2.md",
    "03_TASK_CARDS\day3.md", "03_TASK_CARDS\day4.md", "03_TASK_CARDS\day5.md",
    "04_ORAL_EXAM.md", "05_REFERENCE_ANSWERS.md",
    "lab\README.md", "lab\requirements.txt"
)
$requiredDirs = @("lab\src", "lab\tests", "lab\scripts", "lab\configs")
foreach ($f in $requiredFiles) {
    $p = Join-Path $pkg $f
    if (-not (Test-Path $p)) { Add-Err "missing file: $f" }
    elseif ((Get-Item $p).Length -lt 200) { Add-Err "file too small (<200 bytes): $f" }
}
foreach ($d in $requiredDirs) {
    if (-not (Test-Path (Join-Path $pkg $d))) { Add-Err "missing dir: $d" }
}

# ---------- numbered sections ----------
function Test-NumberedSections([string]$rel, [int]$maxIdx) {
    $p = Join-Path $pkg $rel
    if (-not (Test-Path $p)) { return }
    $t = Read-Text $p
    for ($i = 0; $i -le $maxIdx; $i++) {
        if ($t -notmatch "(?m)^## $i\. ") { Add-Err "$rel : missing section '## $i.'" }
    }
}
Test-NumberedSections "01_FOUNDATIONS.md" 9
Test-NumberedSections "02_LAB_GUIDE.md" 7

# ---------- task cards ----------
$cardRequired = @("| 主要产物 |", "| 估时 |", "| AI 辅助等级要求 |", "## 为什么做", "## 步骤", "## 证据字段")
for ($d = 0; $d -le 5; $d++) {
    $rel = "03_TASK_CARDS\day$d.md"
    $p = Join-Path $pkg $rel
    if (-not (Test-Path $p)) { continue }
    $t = Read-Text $p
    foreach ($h in $cardRequired) {
        if ($t.IndexOf($h) -lt 0) { Add-Err "$rel : missing '$h'" }
    }
    $steps = [regex]::Matches($t, "(?m)^### (\d+)\. ")
    $n = $steps.Count
    if ($n -eq 0) { Add-Err "$rel : no numbered steps '### N. '" }
    elseif ($n -gt 7) { Add-Err "$rel : $n steps (> 7)" }
    # each step block must contain expected + first-check
    $blocks = [regex]::Split($t, "(?m)^### \d+\. ")
    for ($b = 1; $b -lt $blocks.Count; $b++) {
        $blk = $blocks[$b]
        if ($blk.IndexOf("预期") -lt 0) { Add-Err "$rel : step $b lacks '预期'" }
        if ($blk.IndexOf("不对时先查") -lt 0) { Add-Err "$rel : step $b lacks '不对时先查'" }
    }
    # estimated minutes in step headers: (NN 分钟)
    $mins = [regex]::Matches($t, "(?m)^### \d+\. .*?（(\d+) 分钟）")
    if ($mins.Count -gt 0) {
        $sum = 0; foreach ($m in $mins) { $sum += [int]$m.Groups[1].Value }
        if ($sum -gt 120) { Add-Err "$rel : step minutes sum to $sum (> 120)" }
        if ($sum -lt 30) { Add-Warn "$rel : step minutes sum to $sum (< 30)" }
    } else { Add-Warn "$rel : no per-step minute estimates found" }
    # command grade labels
    if ($t -notmatch "可直接执行|模板|伪代码") { Add-Err "$rel : no command grade label" }
    # evidence json block must reference schema keys
    foreach ($k in @('"status"', '"ai_level"', '"primary_artifact"', '"failure_class"')) {
        if ($t.IndexOf($k) -lt 0) { Add-Err "$rel : evidence block lacks $k" }
    }
}

# ---------- exam / answers ----------
$examP = Join-Path $pkg "04_ORAL_EXAM.md"
$ansP = Join-Path $pkg "05_REFERENCE_ANSWERS.md"
if ((Test-Path $examP) -and (Test-Path $ansP)) {
    $idRe = "$IdPrefix$Week-[READPT]-\d{2}"
    $qIds = @([regex]::Matches((Read-Text $examP), $idRe) | ForEach-Object { $_.Value } | Sort-Object -Unique)
    $aIds = @([regex]::Matches((Read-Text $ansP), $idRe) | ForEach-Object { $_.Value } | Sort-Object -Unique)
    if ($qIds.Count -lt 20) { Add-Err "04_ORAL_EXAM.md : only $($qIds.Count) question IDs (< 20)" }
    $missingA = @($qIds | Where-Object { $aIds -notcontains $_ })
    $extraA = @($aIds | Where-Object { $qIds -notcontains $_ })
    if ($missingA.Count -gt 0) { Add-Err "answers missing for: $($missingA -join ', ')" }
    if ($extraA.Count -gt 0) { Add-Err "answers without question: $($extraA -join ', ')" }
    foreach ($layer in @("R", "E", "A", "D", "P", "T")) {
        if (-not ($qIds | Where-Object { $_ -match "-$layer-" })) { Add-Warn "04_ORAL_EXAM.md : no questions in layer $layer" }
    }
    Write-Host "Exam IDs: $($qIds.Count) questions / $($aIds.Count) answers"
}

# ---------- banned placeholders (markdown) ----------
$bannedMd = @("自行实现", "参考前文", "类似地处理", "根据情况调整", "TODO", "TBD", "FIXME", "NotImplementedError")
$exportRisk = @("发给我日志", "上传日志", "导出 trace", "导出trace", "导出 checkpoint", "导出checkpoint", "把截图发", "上传 checkpoint")
$mdFiles = Get-ChildItem -Path $pkg -Recurse -Filter *.md
foreach ($f in $mdFiles) {
    $t = Read-Text $f.FullName
    $rel = $f.FullName.Substring($pkg.Length + 1)
    foreach ($w in $bannedMd) {
        if ($t.IndexOf($w) -ge 0) { Add-Err "$rel : banned placeholder '$w'" }
    }
    foreach ($w in $exportRisk) {
        if ($t.IndexOf($w) -ge 0) { Add-Err "$rel : export-risk phrase '$w'" }
    }
    # local links -- strip fenced code blocks and inline code first, so log/command
    # samples such as "Epoch:[1/1](3/500)" are not mistaken for markdown links
    $tLinks = [regex]::Replace($t, '(?ms)^[ \t]*(```|~~~).*?^[ \t]*\1[ \t]*$', "`n")
    $tLinks = [regex]::Replace($tLinks, '`[^`\r\n]*`', ' ')
    foreach ($m in [regex]::Matches($tLinks, "\]\(([^)#\s]+)(#[^)]*)?\)")) {
        $target = $m.Groups[1].Value
        if ($target -match "^(https?:|mailto:)") { continue }
        $tp = Join-Path $f.DirectoryName $target
        if (-not (Test-Path $tp)) { Add-Err "$rel : broken local link '$target'" }
    }
}

# ---------- python placeholders ----------
$pyFiles = @(Get-ChildItem -Path (Join-Path $pkg "lab") -Recurse -Filter *.py -ErrorAction SilentlyContinue)
foreach ($f in $pyFiles) {
    $lines = [System.IO.File]::ReadAllLines($f.FullName, [System.Text.Encoding]::UTF8)
    $rel = $f.FullName.Substring($pkg.Length + 1)
    for ($i = 0; $i -lt $lines.Count; $i++) {
        $l = $lines[$i]
        # A 'pass' is a placeholder only when it is the whole body of a def/class.
        # 'except X: pass', 'with ...: pass', 'if ...: pass' are deliberate no-ops.
        if ($l -match "^(\s*)pass\s*$") {
            $indent = $Matches[1].Length
            for ($j = $i - 1; $j -ge 0; $j--) {
                $prev = $lines[$j]
                if ($prev -match "^\s*$" -or $prev -match "^\s*#") { continue }
                $pi = ($prev -replace "^(\s*).*$", '$1').Length
                if ($pi -lt $indent) {
                    if ($prev -match "^\s*(async\s+def|def|class)\s") {
                        Add-Err "$rel`:$($i+1) : empty def/class body ('pass' placeholder)"
                    }
                    break
                }
            }
        }
        if ($l -match "NotImplementedError|TODO|FIXME") { Add-Err "$rel`:$($i+1) : placeholder '$($l.Trim())'" }
        if ($l -match "^\s*\.\.\.\s*$") { Add-Err "$rel`:$($i+1) : ellipsis placeholder" }
    }
}
if ($pyFiles.Count -eq 0) { Add-Err "lab/ contains no .py files" }
# entry scripts should expose --help via argparse
$scriptPy = @(Get-ChildItem -Path (Join-Path $pkg "lab\scripts") -Filter *.py -ErrorAction SilentlyContinue)
foreach ($f in $scriptPy) {
    $t = Read-Text $f.FullName
    if ($t -notmatch 'argparse|click|typer|--help') { Add-Warn "lab/scripts/$($f.Name) : no argparse/click/typer and no manual --help handling" }
}

# ---------- runtime-claim guard ----------
$card = Join-Path $pkg "00_WEEK_CARD.md"
if (Test-Path $card) {
    $t = Read-Text $card
    if ($t -notmatch "本机验证状态") { Add-Err "00_WEEK_CARD.md : missing '本机验证状态' section" }
    if ($t -match "GATE-PASS" -and $t -notmatch "06_GATE_RECORD") { Add-Warn "00_WEEK_CARD.md : GATE-PASS without a gate record reference" }
}

# ---------- report ----------
Write-Host ""
Write-Host "Errors: $($script:Errors.Count)  Warnings: $($script:Warnings.Count)"
foreach ($e in $script:Errors) { Write-Host "  ERROR  $e" }
foreach ($w in $script:Warnings) { Write-Host "  WARN   $w" }
if ($script:Errors.Count -gt 0) { exit 1 } else { Write-Host "RESULT: PASS (static only; runtime unverified)"; exit 0 }
