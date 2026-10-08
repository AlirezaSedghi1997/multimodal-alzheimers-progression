[CmdletBinding()]
param(
    [int]$Limit = 0
)

$ErrorActionPreference = 'Stop'
# PROJECT_RUNTIME_ON_H_V1
. 'H:\New project\scripts\project_runtime_env.ps1'

$ProjectRoot = 'H:\New project'
$Python = Join-Path $ProjectRoot '.venv\Scripts\python.exe'
$Script = Join-Path $ProjectRoot 'scripts\materialize_method_y_transforms_v2.py'
$OutputRoot = Join-Path $ProjectRoot 'results\method_y_revival_v2'

if (-not (Test-Path -LiteralPath $Python)) { throw "Python environment missing: $Python" }
if (-not (Test-Path -LiteralPath $Script)) { throw "Materializer missing: $Script" }

Set-Location -LiteralPath $ProjectRoot
if ($Limit -gt 0) {
    & $Python $Script --output-root $OutputRoot --limit $Limit
} else {
    & $Python $Script --output-root $OutputRoot
}
$code = $LASTEXITCODE
Write-Host "Method Y transform stage exited with code $code. Re-run this same command to resume from the last committed scan."
exit $code

