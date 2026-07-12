param(
    [int]$MaxRuntimeMinutes = 0,
    [switch]$PreflightOnly
)

$ErrorActionPreference = "Stop"

$ProjectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path
Set-Location -LiteralPath $ProjectRoot

function Invoke-PythonChecked {
    param(
        [string]$StepName,
        [string[]]$PythonArgs
    )

    & python @PythonArgs
    if ($LASTEXITCODE -ne 0) {
        throw "$StepName failed with exit code $LASTEXITCODE"
    }
}

Invoke-PythonChecked `
    -StepName "GPU preflight" `
    -PythonArgs @(
        "scripts\gpu_preflight_check.py",
        "--cache", "data\cached\t1_volume_cache_128.h5",
        "--require_gpu"
    )

if ($PreflightOnly) {
    Write-Host "Preflight complete. Training was not started."
    exit 0
}

$TrainingArgs = @(
    "scripts\train_3d_mri.py",
    "--cache", "data\cached\t1_volume_cache_128.h5",
    "--resolution", "128",
    "--phase", "final",
    "--backbone", "resnet18_3d",
    "--folds", "5",
    "--epochs", "100",
    "--use_hparam_manifest",
    "--amp",
    "--checkpoint",
    "--grad_accum", "8",
    "--device", "cuda",
    "--require_gpu",
    "--out_dir", "results\final_results\three_d_augmented_resnet18_128"
)
if ($MaxRuntimeMinutes -gt 0) {
    $TrainingArgs += @("--max_runtime_minutes", "$MaxRuntimeMinutes")
}

Invoke-PythonChecked `
    -StepName "Method X MRI final augmented training" `
    -PythonArgs $TrainingArgs
