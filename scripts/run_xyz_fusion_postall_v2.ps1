[CmdletBinding()]
param([ValidateSet('all','preflight','audit','statistics','tau','pacc','scanner','spatial','four_way','verify','promote')][string]$Mode='preflight')
$ErrorActionPreference='Stop'
$Root=(Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
if ($Root -ne 'H:\New project') { throw 'Authoritative project must be H:\New project' }
Set-Location -LiteralPath $Root
$Runtime=Join-Path $Root '.runtime\xyz_fusion_postall_v2'
New-Item -ItemType Directory -Force -Path $Runtime | Out-Null
foreach ($Key in @('TEMP','TMP','TMPDIR','MPLCONFIGDIR','TORCH_HOME','CUDA_CACHE_PATH','NUMBA_CACHE_DIR','JOBLIB_TEMP_FOLDER')) {
    $Directory=Join-Path $Runtime $Key
    New-Item -ItemType Directory -Force -Path $Directory | Out-Null
    [Environment]::SetEnvironmentVariable($Key,$Directory,'Process')
}
$env:PYTHONDONTWRITEBYTECODE='1'
$env:CUBLAS_WORKSPACE_CONFIG=':4096:8'
& (Join-Path $Root '.venv\Scripts\python.exe') -B -u (Join-Path $Root 'scripts\run_xyz_fusion_postall_v2.py') --mode $Mode
$Result=$LASTEXITCODE
if ($Result -eq 130) { Write-Host 'INTERRUPTED_RESUMABLE. Repeat the identical command; keep all output files.' }
elseif ($Result -ne 0) { Write-Host 'Stopped safely. Preserve evidence; resolve the reported failure before resuming.' }
exit $Result
