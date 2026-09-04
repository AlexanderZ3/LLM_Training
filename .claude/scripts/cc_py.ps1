<#
.SYNOPSIS
  Run Python through the project's conda environment (never the Windows Store shim).
.DESCRIPTION
  Forwards all arguments to <conda root>\envs\<env>\python.exe.
  Environment selection: $env:CC_CONDA_ENV (default "ResearchAgentPy310"). Conda root: $env:CC_CONDA_ROOT
  (default "D:\Software\Large\Anconda"). Use "base" to run the root interpreter.
.EXAMPLE
  powershell -NoProfile -ExecutionPolicy Bypass -File .claude/scripts/cc_py.ps1 -m pytest lab/tests -q
  $env:CC_CONDA_ENV="parttime"; powershell -File .claude/scripts/cc_py.ps1 -c "import torch; print(torch.__version__)"
#>
$root = $env:CC_CONDA_ROOT
if ([string]::IsNullOrEmpty($root)) { $root = "D:\Software\Large\Anconda" }
$envName = $env:CC_CONDA_ENV
if ([string]::IsNullOrEmpty($envName)) { $envName = "ResearchAgentPy310" }

if ($envName -eq "base") { $py = Join-Path $root "python.exe" }
else { $py = Join-Path (Join-Path $root "envs") (Join-Path $envName "python.exe") }

if (-not (Test-Path $py)) {
    Write-Host "cc_py: interpreter not found: $py"
    Write-Host "cc_py: available envs:"
    Get-ChildItem (Join-Path $root "envs") -Directory | ForEach-Object { Write-Host "  - $($_.Name)" }
    exit 2
}

& $py @args
exit $LASTEXITCODE
