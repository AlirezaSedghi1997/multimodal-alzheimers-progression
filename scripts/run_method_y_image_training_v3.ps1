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
$env:IMAGE_V3_PERF = '1'

Set-Location -LiteralPath $ProjectRoot
& 'H:\New project\.venv\Scripts\python.exe' 'H:\New project\scripts\run_method_y_image_training_v3.py' --config 'H:\New project\configs\method_y_image_training_v3_1.yaml' --resume
exit $LASTEXITCODE
