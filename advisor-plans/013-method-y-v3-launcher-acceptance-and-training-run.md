# Plan 013: launcher واحد را بساز، acceptance را پاس کن و image training را رزوم‌پذیر اجرا کن

> **دستور ایجنت مجری:** فقط پس از DONE شدن Planهای 009–012 اجرا شود. این Plan
> هم launcher را می‌سازد و هم پس از acceptance، فرمان production را ارائه/اجرا
> می‌کند. همان command باید cache، branch، fold، candidate، epoch، outer commit
> و reports را resume کند. هیچ timeout داخلی و هیچ `--no-resume` مجاز نیست.
>
> **Drift check:**
>
> ```powershell
> Set-Location -LiteralPath 'H:\New project'
> git diff --stat 5405716..HEAD -- configs src/method_y_revival scripts tests/method_y_revival docs
> Get-Content advisor-plans\README.md
> ```
>
> اگر Plan 009–012 در README همگی DONE نیستند، STOP.

## وضعیت

- **اولویت:** P0 برای شروع production training
- **اندازه:** XL (build + acceptance + runtime طولانی)
- **ریسک:** HIGH
- **وابسته به:** 009, 010, 011, 012
- **دسته:** DX / orchestration / tests / shipping
- **Planned at:** commit `5405716`, 2026-08-24

## چرا مهم است

در snapshot 2026-08-24 این فایل‌ها وجود ندارند:

```text
scripts/run_method_y_image_training_v3.py
scripts/run_method_y_image_training_v3.ps1
src/method_y_revival/image_training_orchestrator_v3.py  (تا Plan 011)
```

بدون launcher واحد، اجرای دستی چند script می‌تواند cache/config اشتباه، دو writer،
outer-test تکراری یا temp روی C ایجاد کند. acceptance باید قبل از آموزش واقعی
ثابت کند 8GB GPU قابل استفاده است و interruption synthetic بازیابی می‌شود.

## فایل‌های مجاز

```text
scripts/run_method_y_image_training_v3.ps1                   (new)
scripts/run_method_y_image_training_v3.py                    (Plan 011؛ تکمیل CLI/state)
src/method_y_revival/image_training_orchestrator_v3.py        (Plan 011؛ تکمیل state)
scripts/verify_method_y_image_training_v3.py                  (new)
configs/method_y_image_training_v3_1.yaml
tests/method_y_revival/test_image_training_launcher_v3.py      (new)
tests/method_y_revival/test_image_training_acceptance_v3.py    (new)
tests/method_y_revival/test_image_training_storage_v3.py       (new)
docs/METHOD_Y_IMAGE_TRAINING_V3_COMMAND.md                     (new)
advisor-plans/README.md                                        (status only)
```

خارج scope: data/raw، H:\A4 Learn mutation، v1/v2 overwrite، external DUA،
XAI، تغییر endpoint/split/features پس از lock.

## state machine نهایی

```text
INIT
→ PREFLIGHT_DONE
→ DUAL_GATE_CONTRACT_VERIFIED
→ READINESS_LOCKED
→ PATTERN_CACHES_COMPLETE
→ GPU_SMOKE_PASSED
→ AMYLOID_INNER_HPO
→ AMYLOID_MODELS_LOCKED
→ AMYLOID_OUTER_COMMITTED
→ TAU_INNER_HPO
→ TAU_MODELS_LOCKED
→ TAU_OUTER_COMMITTED
→ ROBUSTNESS_COMPLETE
→ REPORTS_COMPLETE
→ CONSISTENCY_PRIVACY_PASS
→ FINALIZED
```

ترتیب amyloid سپس tau به‌دلیل support قوی‌تر amyloid و امکان یافتن خطای runtime
زودتر قفل شود. این ترتیب claim یا selection را تغییر نمی‌دهد.

## مراحل

### مرحله 1: RED launcher tests

```text
test_ps1_sets_all_temp_and_cache_roots_to_h
test_ps1_uses_canonical_project_venv
test_ps1_has_no_timeout_and_no_no_resume
test_launcher_rejects_active_writer_same_root
test_launcher_rejects_contract_mismatch
test_launcher_skips_hash_verified_completed_stage
test_finalized_rerun_only_verifies_and_never_trains
test_external_pending_does_not_block_internal_run
test_outer_committed_stage_cannot_be_reentered
test_launcher_emits_epoch_progress_with_eta
```

### مرحله 2: PS1 storage environment

PS1 باید:

```powershell
$ErrorActionPreference = 'Stop'
$ProjectRoot = 'H:\New project'
$RuntimeRoot = Join-Path $ProjectRoot '.runtime\method_y_image_v3'

$env:TMP = Join-Path $RuntimeRoot 'tmp'
$env:TEMP = $env:TMP
$env:TMPDIR = $env:TMP
$env:PYTHONPYCACHEPREFIX = Join-Path $RuntimeRoot 'pycache'
$env:TORCH_HOME = Join-Path $RuntimeRoot 'torch'
$env:HF_HOME = Join-Path $RuntimeRoot 'huggingface'
$env:MPLCONFIGDIR = Join-Path $RuntimeRoot 'matplotlib'
$env:XDG_CACHE_HOME = Join-Path $RuntimeRoot 'xdg'
$env:PIP_CACHE_DIR = 'H:\New project\.pip_cache'

Set-Location -LiteralPath $ProjectRoot
& 'H:\New project\.venv\Scripts\python.exe' `
  'H:\New project\scripts\run_method_y_image_training_v3.py' `
  --config 'H:\New project\configs\method_y_image_training_v3_1.yaml' `
  --resume
exit $LASTEXITCODE
```

هیچ `Start-Process`، background job یا redirect به C استفاده نشود.

### مرحله 3: preflight

Python launcher قبل از هر write:

1. project/output/temp drive = H؛
2. process guard و WriterLock؛
3. git commit/dirty digest؛
4. config/readiness/cache/split/membership hashes؛
5. GPU name/VRAM/CUDA/Torch؛
6. H free space ≥20 GiB؛
7. v1/v2 read-only evidence hashes؛
8. no external requirement for internal execution؛
9. claim limitations ثبت شوند؛
10. orchestration state self-hash بررسی شود.

### مرحله 4: GPU smoke هشت‌گیگابایتی

قبل از HPO production، روی 4–8 scan development بدون outer-test:

```text
forward + backward
AMP
batch_size=2
gradient_accumulation=4
checkpoint write/read strict
peak VRAM recorded
finite loss/gradient
```

اگر OOM:

- همان config/root را تغییر نده؛
- `VRAM_SMOKE_BLOCKED.json` بنویس و STOP؛
- fallback از پیش‌تعریف‌شده فقط در نسخهٔ config/output جدید:

```yaml
batch_size: 1
gradient_accumulation_steps: 8
normalization: GroupNorm unchanged
all variants/candidates/branches use same fallback
```

قبل از fallback یک test باید effective-batch gradient را در tolerance قفل‌شده
مقایسه کند. به BatchNorm یا تغییر معماری روی نیاور.

### مرحله 5: CLI modes

Python CLI modes:

```text
--mode preflight
--mode smoke
--mode run       (default/resume all stages)
--mode status    (read-only)
--mode verify    (read-only)
```

`--resume` default true باشد؛ `--no-resume` اصلاً وجود نداشته باشد. branch
debug flag ممکن است برای test باشد، اما command رسمی هر دو شاخه را اجرا کند.

### مرحله 6: progress output

هر epoch یک JSON line flush کند:

```json
{
  "event": "EPOCH_COMMITTED",
  "branch": "amyloid",
  "variant": "scalar_tabular_plus_pattern_image",
  "outer_fold": 0,
  "stage": "inner_hpo",
  "candidate_id": "lr1e4_wd1e3",
  "inner_fold": 0,
  "epoch": 3,
  "target_epochs": 40,
  "train_loss": 0.0,
  "validation_loss": 0.0,
  "epoch_seconds": 0.0,
  "eta_seconds": 0.0,
  "checkpoint": "H:/.../epoch_checkpoint.pt"
}
```

در resume باید پیام `RESUMED_FROM_COMMITTED_EPOCH` با epoch قبلی چاپ شود.

### مرحله 7: synthetic full-pipeline acceptance

یک config 16³/CPU با 5x3 folds، دو variant و دو candidate:

1. run تا وسط epoch؛ interruption؛
2. همان command؛ resume؛
3. interruption بعد result commit؛
4. resume بدون outer call دوم؛
5. reports/privacy؛
6. finalized rerun فقط verify.

انتظار bitwise weights/history برای path uninterrupted vs resumed.

### مرحله 8: verification قبل از production

```powershell
& '.\.venv\Scripts\python.exe' -m pytest -p no:cacheprovider `
  tests\method_y_revival\test_image_training_policy_v3.py `
  tests\method_y_revival\test_image_training_manifest_v3.py `
  tests\method_y_revival\test_image_cache_v3.py `
  tests\method_y_revival\test_image_survival_v3.py `
  tests\method_y_revival\test_image_training_runtime_v3.py `
  tests\method_y_revival\test_image_modeling_v3.py `
  tests\method_y_revival\test_image_training_orchestrator_v3.py `
  tests\method_y_revival\test_image_outer_exactly_once_v3.py `
  tests\method_y_revival\test_image_robustness_v3.py `
  tests\method_y_revival\test_image_reporting_v3.py `
  tests\method_y_revival\test_image_privacy_claim_v3.py `
  tests\method_y_revival\test_image_training_launcher_v3.py `
  tests\method_y_revival\test_image_training_acceptance_v3.py -q

& '.\.venv\Scripts\python.exe' -m pytest -p no:cacheprovider `
  tests\method_y_revival -q

& '.\.venv\Scripts\python.exe' -m pytest -p no:cacheprovider tests -q
```

همه zero failure. عدد تست hard-code نشود. full suite تاریخی جای اجرای تازه را
نمی‌گیرد.

### مرحله 9: dry-run command

```powershell
powershell -NoProfile -ExecutionPolicy Bypass `
  -File 'H:\New project\scripts\run_method_y_image_training_v3.ps1' `
  -Mode preflight
```

Expected:

```text
READY_TO_RUN_FINAL_INTERNAL_EXPLORATORY_IMAGE_TRAINING
training_execution_allowed=true
positive_claim_allowed=false
external_claim_allowed=false
```

### مرحله 10: production command

پس از تمام معیارهای بالا:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass `
  -File 'H:\New project\scripts\run_method_y_image_training_v3.ps1'
```

اگر قطع شد، دقیقاً همان command را دوباره اجرا کن. هیچ پوشه‌ای حذف نشود.

### مرحله 11: نتیجهٔ terminal

Finalized فقط وقتی:

```text
all caches hash-verified
2 branches × 2 matched variants × 5 outer folds committed
all winners selected from inner IBS only
all checkpoints locked
robustness/calibration/subgroup/bootstrap reports complete or NOT_ESTIMABLE with reason
privacy PASS
claim consistency PASS
external status EXTERNAL_PENDING (nonblocking)
```

نتیجه می‌تواند science gate FAIL باشد و همچنان اجرای engineering کامل باشد.
نتیجهٔ منفی حذف یا rerun برای significance نشود.

## آزمون‌ها

مجموعهٔ آزمون‌های مرحلهٔ 8 و dry-run مرحلهٔ 9 test plan رسمی این Plan است.
هیچ عدد تاریخی جای اجرای تازه را نمی‌گیرد؛ نتیجه، command، exit code و زمان
اجرا در agent report ثبت شود.

## معیار پایان

- [ ] PS1/CLI واحد وجود و tests دارد.
- [ ] تمام temp/cache روی H است.
- [ ] GPU smoke پاس یا fallback نسخه‌دار و prespecified پاس شده است.
- [ ] synthetic interruption acceptance bitwise پاس است.
- [ ] full project suite صفر failure دارد.
- [ ] preflight واقعی READY است.
- [ ] command رسمی بدون timeout/resume-blocker است.
- [ ] پس از production، تمام outer foldها exactly-once committed هستند.
- [ ] claim status با limitations واقعی هم‌راستاست.

## STOP conditions

- هر Plan پیشین TODO/BLOCKED؛
- active writer همان output root؛
- run-contract mismatch؛
- source/cache/membership/split hash mismatch؛
- GPU smoke OOM بدون fallback نسخه‌دار؛
- H free <20 GiB؛
- هر outer access قبل lock یا ambiguous in-progress بدون result؛
- full suite failure؛
- privacy failure؛
- درخواست تغییر endpoint/split/feature بعد از outer access.

## نگهداری

این command فقط internal exploratory image extension است. external dataset،
quantitative equivalence و XAI در نسخه‌های جدا اضافه می‌شوند و نباید run
production حاضر را mutate کنند.
