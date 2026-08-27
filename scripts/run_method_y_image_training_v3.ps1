$ErrorActionPreference = 'Stop'
$ProjectRoot = 'H:\New project'
$RuntimeRoot = Join-Path $ProjectRoot '.runtime\method_y_image_v3'

$env:TMP = Join-Path $RuntimeRoot 'tmp'
$env:TEMP = $env:TMP
$env:TMPDIR = $env:TMP
$env:PYTHONPYCACHEPREFIX = Join-Path $RuntimeRoot 'pycache'
$env:TORCH_HOME = Join-Path $RuntimeRoot 'torch'
$env:HF_HOME = Join-Path $RuntimeRoot 'huggingface'
$env:MPLCONFIGDIR = Join-Path $RuntimeRoot 'matplotlib'
$env:XDG_CACHE_HOME = Join-Path $RuntimeRoot 'xdg'
$env:PIP_CACHE_DIR = 'H:\New project\.pip_cache'
$env:IMAGE_V3_PERF = '0'
$env:IMAGE_V3_CHANNELS_LAST = '0'
$env:CUBLAS_WORKSPACE_CONFIG = ':4096:8'

Set-Location -LiteralPath $ProjectRoot
$Python = 'H:\New project\.venv\Scripts\python.exe'
$Config = 'H:\New project\configs\method_y_image_training_v3_2.yaml'
$Adapter = 'H:\New project\scripts\method_y_image_resume_adapter.py'
$Launcher = 'H:\New project\scripts\run_method_y_image_training_v3.py'

$AdapterOutput = & $Python $Adapter --project-root $ProjectRoot --config $Config
$AdapterExitCode = $LASTEXITCODE
if ($AdapterExitCode -ne 0) {
    $AdapterOutput | Write-Output
    exit $AdapterExitCode
}

try {
    $ResumeEnvironment = $AdapterOutput | ConvertFrom-Json
} catch {
    throw "METHOD_Y_RESUME_ADAPTER_INVALID_JSON: $AdapterOutput"
}
if ($ResumeEnvironment.status -ne 'READY') {
    throw "METHOD_Y_RESUME_ADAPTER_NOT_READY: $AdapterOutput"
}
if ($ResumeEnvironment.environment.GIT_DIR) {
    $env:GIT_DIR = [string]$ResumeEnvironment.environment.GIT_DIR
    $env:GIT_WORK_TREE = [string]$ResumeEnvironment.environment.GIT_WORK_TREE
    $env:GIT_OPTIONAL_LOCKS = [string]$ResumeEnvironment.environment.GIT_OPTIONAL_LOCKS
}

& $Python $Launcher --config $Config --resume
exit $LASTEXITCODE
