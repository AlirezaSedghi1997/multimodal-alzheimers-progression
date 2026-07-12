param(
    [int]$MaxRuntimeMinutes = 0,
    [int]$TargetTotalRows = 4000,
    [string[]]$Backbones = @("resnet18_3d"),
    [switch]$AllBackbones,
    [switch]$BuildOnly,
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

if ($AllBackbones) {
    $Backbones = @("resnet18_3d", "densenet121_3d", "efficientnet3d_b0")
}

$CombinedCache = "data\cached\t1_volume_cache_128.h5"

if ($BuildOnly) {
    Write-Host "The validated materialized cache is already published at $CombinedCache. Training was not started."
    exit 0
}

Invoke-PythonChecked `
    -StepName "GPU preflight" `
    -PythonArgs @(
        "scripts\gpu_preflight_check.py",
        "--cache", $CombinedCache,
        "--require_gpu"
    )

if ($PreflightOnly) {
    Write-Host "Preflight complete. Training was not started."
    exit 0
}

foreach ($Backbone in $Backbones) {
    $OutDir = "results\final_results\three_d_foldlocal_augmented_$($TargetTotalRows)_$($Backbone)_128"
    $TrainingArgs = @(
        "scripts\train_3d_mri.py",
        "--cache", $CombinedCache,
        "--resolution", "128",
        "--phase", "final",
        "--backbone", $Backbone,
        "--folds", "5",
        "--epochs", "100",
        "--early_stopping_patience", "1000",
        "--use_hparam_manifest",
        "--amp",
        "--checkpoint",
        "--grad_accum", "8",
        "--device", "cuda",
        "--require_gpu",
        "--out_dir", $OutDir
    )
    if ($MaxRuntimeMinutes -gt 0) {
        $TrainingArgs += @("--max_runtime_minutes", "$MaxRuntimeMinutes")
    }
    Invoke-PythonChecked `
        -StepName "Method X MRI fold-local augmented training ($Backbone)" `
        -PythonArgs $TrainingArgs
}
