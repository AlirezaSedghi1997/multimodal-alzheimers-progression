$ProjectRuntimeRoot = 'H:\New project\.runtime'
$ProjectTemp = Join-Path $ProjectRuntimeRoot 'tmp'
$ProjectCache = Join-Path $ProjectRuntimeRoot 'cache'
New-Item -ItemType Directory -Path $ProjectTemp, $ProjectCache -Force | Out-Null
$env:TEMP = $ProjectTemp
$env:TMP = $ProjectTemp
$env:TMPDIR = $ProjectTemp
$env:TORCH_HOME = Join-Path $ProjectCache 'torch'
$env:HF_HOME = Join-Path $ProjectCache 'huggingface'
$env:HUGGINGFACE_HUB_CACHE = Join-Path $env:HF_HOME 'hub'
$env:UV_CACHE_DIR = Join-Path $ProjectCache 'uv'
$env:PIP_CACHE_DIR = Join-Path $ProjectCache 'pip'
$env:MPLCONFIGDIR = Join-Path $ProjectCache 'matplotlib'
$env:NUMBA_CACHE_DIR = Join-Path $ProjectCache 'numba'
$env:JOBLIB_TEMP_FOLDER = Join-Path $ProjectTemp 'joblib'
foreach ($path in @(
    $env:TEMP, $env:TORCH_HOME, $env:HF_HOME, $env:HUGGINGFACE_HUB_CACHE,
    $env:UV_CACHE_DIR, $env:PIP_CACHE_DIR, $env:MPLCONFIGDIR,
    $env:NUMBA_CACHE_DIR, $env:JOBLIB_TEMP_FOLDER
)) {
    New-Item -ItemType Directory -Path $path -Force | Out-Null
}
