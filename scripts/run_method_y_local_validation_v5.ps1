param([ValidateSet('preflight', 'run', 'verify')][string]$Mode = 'run')
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $root
$env:TEMP = Join-Path $root 'tmp'
$env:TMP = $env:TEMP
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
New-Item -ItemType Directory -Force -Path $env:TEMP | Out-Null
& (Join-Path $root '.venv\Scripts\python.exe') -B -u (Join-Path $PSScriptRoot 'run_method_y_local_validation_v5.py') --mode $Mode
exit $LASTEXITCODE
