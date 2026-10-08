[CmdletBinding()]
param([ValidateSet('all','preflight','smoke','verify')][string]$Mode='all')
$ErrorActionPreference='Stop'
$Root=(Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
if ($Root -ne 'H:\New project') { throw 'Authoritative project must be H:\New project' }
Set-Location -LiteralPath $Root
$Runtime=Join-Path $Root '.runtime\xyz_fp32_spatial_v1'
New-Item -ItemType Directory -Force -Path $Runtime | Out-Null
foreach ($Key in @('TEMP','TMP','TMPDIR','MPLCONFIGDIR','TORCH_HOME','TORCHINDUCTOR_CACHE_DIR','CUDA_CACHE_PATH','NUMBA_CACHE_DIR','JOBLIB_TEMP_FOLDER')) {
    $Directory=Join-Path $Runtime $Key
    New-Item -ItemType Directory -Force -Path $Directory | Out-Null
    [Environment]::SetEnvironmentVariable($Key,$Directory,'Process')
}
$env:PYTHONDONTWRITEBYTECODE='1'
$env:CUBLAS_WORKSPACE_CONFIG=':4096:8'
$Logs=Join-Path $Runtime 'logs'
New-Item -ItemType Directory -Force -Path $Logs | Out-Null
$Log=Join-Path $Logs ((Get-Date -Format 'yyyyMMdd_HHmmss')+'_'+$Mode+'.log')
Start-Transcript -LiteralPath $Log | Out-Null
try {
    & (Join-Path $Root '.venv\Scripts\python.exe') -B -u (Join-Path $Root 'scripts\run_xyz_fp32_spatial_v1.py') --mode $Mode
    $Result=$LASTEXITCODE
} finally { Stop-Transcript | Out-Null }
if ($Result -eq 0) { Write-Host 'Requested scope completed. FP32 sensitivity is separate from AMP parent. Scientific map validity requires review.' }
elseif ($Result -eq 130) { Write-Host 'INTERRUPTED_RESUMABLE: preserve output and repeat the identical command.' }
else { Write-Host 'Stopped safely. Completed units are retained. Do not delete results or change the locked config.' }
exit $Result
