[CmdletBinding()]
param([ValidateSet('run','preflight','benchmark','smoke-preprocess','verify')][string]$Mode='run')
$ErrorActionPreference='Stop'
$Root=(Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
if ($Root -ne 'H:\New project') { throw 'Authoritative project must be H:\New project' }
. (Join-Path $PSScriptRoot 'project_runtime_env.ps1')
$env:PYTHONDONTWRITEBYTECODE='1'
$env:CUBLAS_WORKSPACE_CONFIG=':4096:8'
$env:OMP_NUM_THREADS='4';$env:MKL_NUM_THREADS='4';$env:OPENBLAS_NUM_THREADS='4'
$env:ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS='2'
Set-Location -LiteralPath $Root
& (Join-Path $Root '.venv\Scripts\python.exe') -B -u (Join-Path $PSScriptRoot 'run_xyz_mri_baseline_fast_v1.py') --mode $Mode
$Code=$LASTEXITCODE
if ($Code -ne 0) { Write-Host 'Stopped safely. Preserve the output folder; repeat the same command after resolving the reported error.' }
exit $Code
