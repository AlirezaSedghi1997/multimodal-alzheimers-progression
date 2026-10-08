$ErrorActionPreference = 'Stop'
$project = 'H:\New project'
$env:TEMP = Join-Path $project 'tmp\method_y_amyloid_lmo_a4_v2'
$env:TMP = $env:TEMP
$env:PYTHONPYCACHEPREFIX = Join-Path $project 'tmp\method_y_amyloid_lmo_a4_v2\pycache'
$env:CUBLAS_WORKSPACE_CONFIG = ':4096:8'
New-Item -ItemType Directory -Force -Path $env:TEMP, $env:PYTHONPYCACHEPREFIX | Out-Null
Set-Location -LiteralPath $project
& (Join-Path $project '.venv\Scripts\python.exe') (Join-Path $project 'scripts\run_method_y_amyloid_lmo_a4_v2.py') --mode run
exit $LASTEXITCODE
