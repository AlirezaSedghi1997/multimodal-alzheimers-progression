[CmdletBinding()]
param([ValidateSet('all','preflight','verify')][string]$Mode='preflight')
$ErrorActionPreference='Stop'
$Root=(Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
if ($Root -ne 'H:\New project') { throw 'Authoritative project must be H:\New project' }
Set-Location -LiteralPath $Root
$Runtime=Join-Path $Root '.runtime\xyz_postfusion_final_v1'
New-Item -ItemType Directory -Force -Path $Runtime | Out-Null
foreach ($Key in @('TEMP','TMP','TMPDIR','MPLCONFIGDIR','TORCH_HOME','TORCHINDUCTOR_CACHE_DIR','CUDA_CACHE_PATH','NUMBA_CACHE_DIR','JOBLIB_TEMP_FOLDER')) {
    $Directory=Join-Path $Runtime $Key
    New-Item -ItemType Directory -Force -Path $Directory | Out-Null
    [Environment]::SetEnvironmentVariable($Key,$Directory,'Process')
}
$env:PYTHONDONTWRITEBYTECODE='1'
$env:CUBLAS_WORKSPACE_CONFIG=':4096:8'
& (Join-Path $Root '.venv\Scripts\python.exe') -B -u (Join-Path $Root 'scripts\run_xyz_postfusion_final_v1.py') --mode $Mode
$Result=$LASTEXITCODE
if ($Result -eq 0) { Write-Host 'Requested stages completed. Execution completion is not a positive scientific claim.' }
elseif ($Result -eq 130) { Write-Host 'INTERRUPTED_RESUMABLE. Preserve output and repeat the identical command.' }
else { Write-Host 'Stopped safely. Preserve evidence and resolve the reported error; completed work is retained.' }
exit $Result
