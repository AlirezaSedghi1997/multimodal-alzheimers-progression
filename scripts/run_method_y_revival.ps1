[CmdletBinding()]
param(
    [switch]$PreflightOnly
)

$ErrorActionPreference = "Stop"
$ProjectRoot = "H:\New project"
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$Script = Join-Path $ProjectRoot "scripts\run_method_y_revival.py"
$Config = Join-Path $ProjectRoot "configs\method_y_revival_v1.yaml"
$OutputRoot = Join-Path $ProjectRoot "results\method_y_revival_v1"

if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw "PROJECT_PYTHON_NOT_FOUND: $Python"
}

$Arguments = @(
    $Script,
    "--config", $Config,
    "--output-root", $OutputRoot
)
if ($PreflightOnly) {
    $Arguments += "--preflight-only"
}
else {
    $Arguments += "--build-tau-endpoint"
    $Arguments += "--train-tau-regional"
    $Arguments += "--build-amyloid-endpoint"
    $Arguments += "--train-amyloid-regional"
    $Arguments += "--build-scanner-audit"
}

& $Python @Arguments
exit $LASTEXITCODE
