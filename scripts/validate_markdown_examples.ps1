[CmdletBinding()]
param(
    [string]$Root
)

$ErrorActionPreference = 'Stop'
if ([string]::IsNullOrWhiteSpace($Root)) {
    $scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
    $Root = Split-Path -Parent $scriptDir
}
$courseRoot = Join-Path ([System.IO.Path]::GetFullPath($Root)) 'outputs\0_basic_training'
$errors = [System.Collections.Generic.List[string]]::new()
$warnings = [System.Collections.Generic.List[string]]::new()
$counts = @{ Files = 0; Fences = 0; Json = 0; PowerShell = 0; Bash = 0 }

$bashPath = $null
$bashCommand = Get-Command bash -ErrorAction SilentlyContinue
if ($bashCommand) {
    $bashPath = $bashCommand.Source
} else {
    foreach ($candidate in @(
        'C:\Program Files\Git\bin\bash.exe',
        'D:\Software\Little\Git\Git\bin\bash.exe'
    )) {
        if (Test-Path -LiteralPath $candidate -PathType Leaf) {
            $bashPath = $candidate
            break
        }
    }
}
if (-not $bashPath) {
    $warnings.Add('bash not found; bash blocks were not syntax-checked')
}

function Test-BashBlock([string]$Code) {
    $startInfo = [System.Diagnostics.ProcessStartInfo]::new()
    $startInfo.FileName = $bashPath
    $startInfo.Arguments = '-n -s'
    $startInfo.UseShellExecute = $false
    $startInfo.RedirectStandardInput = $true
    $startInfo.RedirectStandardError = $true
    $startInfo.CreateNoWindow = $true
    $process = [System.Diagnostics.Process]::new()
    $process.StartInfo = $startInfo
    [void]$process.Start()
    $process.StandardInput.Write($Code)
    $process.StandardInput.Close()
    $stderr = $process.StandardError.ReadToEnd()
    $process.WaitForExit()
    return @{ ExitCode = $process.ExitCode; Error = $stderr.Trim() }
}

foreach ($file in Get-ChildItem -LiteralPath $courseRoot -Recurse -File -Filter '*.md') {
    $counts.Files += 1
    $text = Get-Content -Raw -Encoding UTF8 -LiteralPath $file.FullName
    if ($text.Contains([char]0xFFFD)) {
        $errors.Add("replacement character found: $($file.FullName)")
    }
    $lines = @($text -split "`r?`n")
    $inside = $false
    $language = ''
    $startLine = 0
    $buffer = [System.Collections.Generic.List[string]]::new()
    for ($index = 0; $index -lt $lines.Count; $index += 1) {
        if ($lines[$index] -match '^\s*```\s*([^\s`]*)') {
            if (-not $inside) {
                $inside = $true
                $language = $Matches[1].ToLowerInvariant()
                $startLine = $index + 1
                $buffer.Clear()
            } else {
                $counts.Fences += 1
                $code = $buffer -join "`n"
                try {
                    if ($language -eq 'json') {
                        $counts.Json += 1
                        $code | ConvertFrom-Json | Out-Null
                    } elseif ($language -eq 'powershell') {
                        $counts.PowerShell += 1
                        [void][scriptblock]::Create($code)
                    } elseif ($language -eq 'bash') {
                        $counts.Bash += 1
                        if ($bashPath) {
                            $result = Test-BashBlock $code
                            if ($result.ExitCode -ne 0) {
                                $errors.Add("bash parse failed: $($file.FullName):$startLine :: $($result.Error)")
                            }
                        }
                    }
                } catch {
                    $errors.Add("$language parse failed: $($file.FullName):$startLine :: $($_.Exception.Message)")
                }
                $inside = $false
                $language = ''
            }
            continue
        }
        if ($inside) {
            $buffer.Add($lines[$index])
        }
    }
    if ($inside) {
        $errors.Add("unclosed fence: $($file.FullName):$startLine")
    }
}

Write-Output "Course root: $courseRoot"
Write-Output "Markdown files: $($counts.Files)"
Write-Output "Fenced blocks: $($counts.Fences) (json=$($counts.Json), powershell=$($counts.PowerShell), bash=$($counts.Bash))"
foreach ($warning in $warnings) { Write-Output "WARN: $warning" }
foreach ($errorMessage in $errors) { Write-Output "ERROR: $errorMessage" }
if ($errors.Count -gt 0) {
    Write-Output "RESULT: FAIL ($($errors.Count) error(s), $($warnings.Count) warning(s))"
    exit 1
}
Write-Output "RESULT: PASS (0 errors, $($warnings.Count) warning(s))"
exit 0
