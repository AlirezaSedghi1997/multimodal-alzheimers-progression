[CmdletBinding()]
param([ValidateSet('all','preflight','run','verify')][string]$Mode='all')
$ErrorActionPreference='Stop'
$Root=(Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
if ($Root -ne 'H:\New project') { throw 'Authoritative project must be H:\New project' }
Set-Location -LiteralPath $Root
$Runtime=Join-Path $Root '.runtime\xyz_fusion_v2'
New-Item -ItemType Directory -Force -Path $Runtime | Out-Null
$env:TEMP=$Runtime
$env:TMP=$Runtime
$env:TMPDIR=$Runtime
$env:PYTHONDONTWRITEBYTECODE='1'
$env:CUBLAS_WORKSPACE_CONFIG=':4096:8'
& (Join-Path $Root '.venv\Scripts\python.exe') -B -u (Join-Path $PSScriptRoot 'run_xyz_fusion_v2.py') --mode $Mode
$Result=$LASTEXITCODE
if ($Result -ne 0) { Write-Host 'Stopped safely. Preserve output and repeat the same command after resolving the reported error.' }
exit $Result
