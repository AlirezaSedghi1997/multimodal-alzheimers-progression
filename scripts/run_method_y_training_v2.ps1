[CmdletBinding()]
param(
    [switch]$ExecuteTraining,
    [string]$ProjectRoot = "H:\New project",
    [string]$PythonPath = "",
    [string]$PythonScript = "",
    [string]$ConfigPath = "",
    [string]$OutputRoot = "",
    [string]$Task4Root = "",
    [string]$BranchReadinessPath = "",
    [string]$FeatureValidationStatePath = "",
    [string]$PretrainingReadinessReceipt = ""
)

$ErrorActionPreference = "Stop"
$canonicalRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
$resolvedProject = [System.IO.Path]::GetFullPath($ProjectRoot)
if ($resolvedProject.TrimEnd("\") -ne $canonicalRoot.TrimEnd("\")) {
    throw "PROJECT_ROOT_MUST_BE_THIS_CHECKOUT"
}
if (-not $resolvedProject.StartsWith("H:\", [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "PROJECT_ROOT_MUST_BE_ON_H_DRIVE"
}
if ([string]::IsNullOrWhiteSpace($PythonPath)) {
    $PythonPath = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
}
if ([string]::IsNullOrWhiteSpace($PythonScript)) {
    $PythonScript = Join-Path $ProjectRoot "scripts\run_method_y_training_v2.py"
}
if ([string]::IsNullOrWhiteSpace($ConfigPath)) {
    $ConfigPath = Join-Path $ProjectRoot "configs\method_y_training_control_v2.yaml"
}
if ([string]::IsNullOrWhiteSpace($OutputRoot)) {
    $OutputRoot = Join-Path $ProjectRoot "results\method_y_revival_v2\training_control"
}
if ([string]::IsNullOrWhiteSpace($Task4Root)) {
    $Task4Root = Join-Path $ProjectRoot "results\method_y_revival_v2\pretraining_contracts"
}
if ([string]::IsNullOrWhiteSpace($BranchReadinessPath)) {
    $BranchReadinessPath = Join-Path $ProjectRoot "results\method_y_revival_v2\readiness\branch_readiness_v2.json"
}
if ([string]::IsNullOrWhiteSpace($FeatureValidationStatePath)) {
    $FeatureValidationStatePath = Join-Path $ProjectRoot "results\method_y_revival_v2\feature_contracts\source_regional_validation_state.json"
}
if ([string]::IsNullOrWhiteSpace($PretrainingReadinessReceipt)) {
    $PretrainingReadinessReceipt = Join-Path $ProjectRoot "results\method_y_revival_v2\readiness\PRETRAINING_READINESS_V2.json"
}
$RuntimeTemp = Join-Path $OutputRoot "_runtime_temp"

New-Item -ItemType Directory -Force -Path $RuntimeTemp | Out-Null
$env:TEMP = $RuntimeTemp
$env:TMP = $RuntimeTemp
$env:JOBLIB_TEMP_FOLDER = Join-Path $RuntimeTemp "joblib"
$env:PYTHONPYCACHEPREFIX = Join-Path $RuntimeTemp "pycache"
$env:PYTHONDONTWRITEBYTECODE = "1"
$env:MPLCONFIGDIR = Join-Path $RuntimeTemp "matplotlib"
$env:NUMBA_CACHE_DIR = Join-Path $RuntimeTemp "numba"
$env:TORCH_HOME = Join-Path $RuntimeTemp "torch"
$env:HF_HOME = Join-Path $RuntimeTemp "huggingface"

if (-not (Test-Path -LiteralPath $PythonPath -PathType Leaf)) {
    throw "PROJECT_PYTHON_NOT_FOUND: $PythonPath"
}

$Arguments = @(
    $PythonScript,
    "--project-root", $ProjectRoot,
    "--config", $ConfigPath,
    "--output-root", $OutputRoot,
    "--task4-root", $Task4Root,
    "--branch-readiness", $BranchReadinessPath,
    "--feature-validation-state", $FeatureValidationStatePath
)
if ($ExecuteTraining) {
    $defaultReadiness = Join-Path $ProjectRoot "results\method_y_revival_v2\readiness\PRETRAINING_READINESS_V2.json"
    $resolvedReceipt = [System.IO.Path]::GetFullPath($PretrainingReadinessReceipt)
    $resolvedDefault = [System.IO.Path]::GetFullPath($defaultReadiness)
    if ($resolvedReceipt -eq $resolvedDefault) {
        throw "EXECUTE_TRAINING_REQUIRES_TASK6_RECEIPT_NOT_PRETRAINING_READINESS"
    }
    $Arguments += "--execute-training"
    $Arguments += "--pretraining-readiness-receipt"
    $Arguments += $PretrainingReadinessReceipt
} else {
    $Arguments += "--preflight-only"
}

& $PythonPath @Arguments
exit $LASTEXITCODE
