[CmdletBinding()]
param([ValidateSet('preflight','run','verify')][string]$Mode='run', [int]$LimitNew=0)
$ErrorActionPreference='Stop'
. 'H:\New project\scripts\project_runtime_env.ps1'
$env:PYTHONDONTWRITEBYTECODE='1'
$env:CUBLAS_WORKSPACE_CONFIG=':4096:8'
Set-Location -LiteralPath 'H:\New project'
if ($Mode -eq 'run') {
    # Check ExecutablePath as well as command text: relative Python commands
    # need not contain the project directory in their command line.
    $writers = @(Get-CimInstance Win32_Process | Where-Object {
        ($_.Name -match '^python(w)?\.exe$') -and
        (($_.ExecutablePath -like 'H:\New project\*') -or ($_.CommandLine -like '*New project*'))
    })
    if ($writers.Count -gt 0) { throw 'Another project Python is active. Run spatial inference separately; nothing was stopped.' }
    $gpuPids = @(& nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits)
    if ($LASTEXITCODE -ne 0) { throw 'GPU process state could not be verified.' }
    if (@($gpuPids | Where-Object { $_ -match '^\s*\d+\s*$' }).Count -gt 0) {
        throw 'GPU compute process active. Wait for it to finish; this launcher will not stop it.'
    }
}
& '.\.venv\Scripts\python.exe' -B 'scripts\run_xy_spatial_validation_v1.py' --mode $Mode --limit-new $LimitNew
$code=$LASTEXITCODE
if (($code -eq 0) -and ($Mode -ne 'preflight')) {
    & '.\.venv\Scripts\python.exe' -B 'scripts\summarize_xy_spatial_validation_v1.py'
    $code=$LASTEXITCODE
}
Write-Host 'Re-run this exact command to recover committed cases. An interrupted, uncommitted case is recalculated. No model is trained.'
exit $code
