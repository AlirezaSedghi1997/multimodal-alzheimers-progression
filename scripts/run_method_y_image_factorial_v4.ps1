[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'
$ProjectRoot = 'H:\New project'
$OutputRoot = Join-Path $ProjectRoot 'results\method_y_revival_v4_image_factorial_v1'
$RuntimeRoot = Join-Path $OutputRoot 'runtime'
$TempRoot = Join-Path $OutputRoot 'tmp'

New-Item -ItemType Directory -Force -Path $RuntimeRoot, $TempRoot | Out-Null
$env:TEMP = $TempRoot
$env:TMP = $TempRoot
$env:PYTHONPYCACHEPREFIX = Join-Path $RuntimeRoot 'pycache'
$env:PYTHONHASHSEED = '20260714'
$env:CUBLAS_WORKSPACE_CONFIG = ':4096:8'

Set-Location -LiteralPath $ProjectRoot
& (Join-Path $ProjectRoot '.venv\Scripts\python.exe') `
  (Join-Path $ProjectRoot 'scripts\run_method_y_image_factorial_v4.py') `
  --config (Join-Path $ProjectRoot 'configs\method_y_image_factorial_v4.yaml') `
  --mode run `
  --resume

exit $LASTEXITCODE
