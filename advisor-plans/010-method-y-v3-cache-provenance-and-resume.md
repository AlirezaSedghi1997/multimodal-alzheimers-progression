# Plan 010: cacheهای v3 را provenance-bound بساز و resume مرز terminal را کامل کن

> **دستور ایجنت مجری:** Plan 009 باید DONE باشد. ابتدا تست RED بنویس، سپس فقط
> فایل‌های scope را تغییر بده. ساخت کامل cache مجاز است؛ training/HPO/outer-test
> در این Plan ممنوع است. هر execution طولانی باید با همان command رزوم‌پذیر باشد.
>
> **Drift check:**
>
> ```powershell
> Set-Location -LiteralPath 'H:\New project'
> git diff --stat 5405716..HEAD -- src/method_y_revival/image_cache_v3.py `
>   src/method_y_revival/image_training_runtime_v3.py `
>   scripts/build_method_y_image_cache_v3.py tests/method_y_revival
> Get-FileHash -Algorithm SHA256 `
>   src\method_y_revival\image_cache_v3.py, `
>   src\method_y_revival\image_training_runtime_v3.py, `
>   scripts\build_method_y_image_cache_v3.py
> ```
>
> SHAهای مبنا در work log 20260824 هستند. تفاوت = code drift؛ ابتدا بازبینی کن.

## وضعیت

- **اولویت:** P0
- **اندازه:** L (پیاده‌سازی M + runtime I/O طولانی)
- **ریسک:** MED؛ خرابی قرارداد می‌تواند cache اشتباه را به مدل متصل کند
- **وابسته به:** Plan 009
- **دسته:** correctness / provenance / performance / tests
- **Planned at:** commit `5405716`, 2026-08-24

## چرا مهم است

`image_cache_v3.py:127-142` contract cache را از membership/template/
normalization می‌سازد، ولی hash کامل source cache و readiness run contract را
در آن bind نمی‌کند. برای HDF5 نیز `_load_source` در خطوط `60-78` group را
می‌خواند اما `source_sha256` attr آن را با membership تطبیق نمی‌دهد. اگر source
HDF5 پس از readiness تغییر کند، ساخت cache می‌تواند بدون fail یک ورودی جدید را
زیر contract قدیمی بخواند.

همچنین `image_training_runtime_v3.py:473-480` ابتدا terminal epoch checkpoint
را می‌نویسد و سپس در خطوط `511-535` locked checkpoint/summary را ایجاد می‌کند.
اگر process در این فاصله قطع شود، resume در خطوط `357-360` به‌جای تکمیل
finalization خطای `TERMINAL_CHECKPOINT_WITHOUT_LOCKED_ARTIFACT` می‌دهد. شرط
کاربر این است که هر interruption امن از آخرین epoch کامل ادامه/بازیابی شود.

## وضعیت فعلی

```text
derived amyloid cache: missing
derived tau cache:     missing
cache summary:         missing
requested membership:  amyloid 560 development + 335 transport in private file
training cache scope:  development only unless explicitly configured otherwise
tau development:       306
```

نکته: cache builder فعلی membership کامل amyloid را می‌خواند؛ قبل از اجرا باید
تصمیم قفل شود که cache شامل development+transport باشد یا development-only.
پیشنهاد این Plan: cache می‌تواند هر دو role را برای reuse بسازد، اما trainer
primary فقط development را مصرف کند و LEARN فقط پس از model lock ارزیابی شود.

## محدوده

**مجاز:**

```text
src/method_y_revival/image_cache_v3.py
src/method_y_revival/image_training_runtime_v3.py
scripts/build_method_y_image_cache_v3.py
configs/method_y_image_training_v3_1.yaml                 (یا نسخهٔ خروجی Plan 009)
tests/method_y_revival/test_image_cache_v3.py
tests/method_y_revival/test_image_training_runtime_v3.py
tests/method_y_revival/test_image_cache_launcher_v3.py     (new)
advisor-plans/README.md                                    (status only)
```

**ممنوع:** trainer high-level، outer test، تغییر feature/endpoint/split، raw data،
artifactهای v1/v2 و پاک‌کردن root superseded.

## مراحل

### مرحله 1: process/storage preflight

```powershell
Get-CimInstance Win32_Process |
  Where-Object {
    $_.Name -match 'python|powershell' -and
    $_.CommandLine -match 'method_y|PET|tau|amyloid'
  } | Select ProcessId,ParentProcessId,Name,CreationDate,CommandLine

Get-PSDrive H | Select Name,Free,Used
```

اگر training/PET writer فعال است، cache build را شروع نکن. حداقل 20 GiB free
قبل و بعد حفظ شود. temp env در اجرای واقعی فقط H باشد.

### مرحله 2: RED tests provenance

تست‌های زیر را قبل از تغییر اضافه کن:

```text
test_hdf5_group_source_sha_must_match_membership
test_source_cache_file_hash_change_breaks_cache_contract
test_readiness_contract_hash_is_part_of_cache_contract
test_completed_cache_rejects_membership_or_source_drift
test_cache_group_must_be_complete_and_finite
test_float16_roundtrip_error_is_below_locked_tolerance
```

fixture HDF5 باید group attr `source_sha256` داشته باشد. یک test attr را تغییر
دهد و انتظار fail بگذارد. هیچ secret/raw ID در fixture نباشد.

### مرحله 3: contract cache را گسترش بده

signature پیشنهادی:

```python
def build_pattern_cache(
    *,
    membership: pd.DataFrame,
    reference_template: Path,
    output_cache: Path,
    state_path: Path,
    normalization: Mapping[str, Any],
    branch: str,
    readiness_run_contract_sha256: str,
    source_artifact_sha256: str,
    progress_every: int = 5,
) -> dict[str, Any]: ...
```

contract payload باید شامل این‌ها باشد:

```python
{
  "membership_sha256": canonical_membership_digest,
  "private_membership_file_sha256": ...,  # توسط launcher تأیید شود
  "readiness_run_contract_sha256": ...,
  "source_artifact_sha256": ...,
  "reference_template_sha256": ...,
  "normalization": ...,
  "dtype": "float16",
  "compression": {"name": "gzip", "level": 4, "shuffle": True},
}
```

launcher باید source artifact را دوباره hash کند و با hash ثبت‌شده در
readiness `run_contract.json` تطبیق دهد. تغییر source =
`SOURCE_ARTIFACT_CHANGED_SINCE_READINESS`، نه بازسازی بی‌صدا.

### مرحله 4: HDF5 group verification

برای `HDF5_GROUP`:

```python
group = handle[group_name]
require(group.attrs["status"] == "COMPLETE")
require(group.attrs["source_sha256"] == row["source_image_sha256"])
require(group.attrs["tracer"].upper() == expected_tracer)
require(group["image"].shape == (1, *reference.shape))
require(group.attrs["finite_fraction"] == 1.0)
```

برای tau همان file-byte hash فعلی حفظ شود. هیچ intensity repair، imputation یا
auto-rescale اضافه نکن.

### مرحله 5: precision audit را قفل کن

قبل از full cache، 16 scan outcome-blind و hash-ordered را انتخاب کن:

```python
float32_normalized = robust_pattern_normalize(source)
float16_roundtrip = float32_normalized.astype(np.float16).astype(np.float32)
max_abs_error = max(abs(float32_normalized - float16_roundtrip))
mean_abs_error = mean(...)
```

threshold فنی از پیش تعیین‌شده:

```text
max_abs_error <= 0.005
all finite == true
shape identical == true
```

اگر fail شد، STOP و cache float32 نسخه‌دار پیشنهاد کن؛ threshold را بعد از دیدن
نتیجه بالا نبر.

### مرحله 6: RED tests resume terminal boundary

به runtime یک injection فقط-تست اضافه کن و موارد زیر را بساز:

```text
test_resume_recovers_terminal_epoch_before_locked_checkpoint
test_resume_recovers_locked_checkpoint_before_summary
test_recovered_locked_weights_equal_uninterrupted_weights
test_terminal_fold_is_not_retrained_after_recovery
test_checkpoint_contract_mismatch_still_fails_closed
```

### مرحله 7: finalization idempotent را پیاده کن

تابع مستقل پیشنهادی:

```python
def finalize_terminal_checkpoint(checkpoint_path, locked_path, summary_path):
    ckpt = load_and_validate(checkpoint_path)
    require(ckpt["fold_terminal"] is True)
    best_state = ckpt["best_model_state"]

    if locked_path.exists():
        verify_locked_contract_and_state_hash(locked_path, best_state)
    else:
        atomic_torch_save(locked_path, locked_payload(best_state, ckpt))

    if summary_path.exists():
        verify_summary_matches_checkpoint(summary_path, ckpt, locked_path)
    else:
        atomic_write_json(summary_path, summary_from(ckpt, locked_path))

    return summary
```

در resume، terminal epoch باید این function را صدا بزند و هیچ epoch تازه‌ای
اجرا نکند. locked checkpoint بهتر است `checkpoint_sha256` و
`best_model_state_sha256` داشته باشد؛ hash serialization باید deterministic یا
file-byte verification باشد.

### مرحله 8: full cache build رزوم‌پذیر

پس از GREEN:

```powershell
$env:TMP='H:\New project\tmp'
$env:TEMP='H:\New project\tmp'
$env:TMPDIR='H:\New project\tmp'
$env:PYTHONPYCACHEPREFIX='H:\New project\.runtime\pycache'

& 'H:\New project\.venv\Scripts\python.exe' `
  'H:\New project\scripts\build_method_y_image_cache_v3.py' `
  --config 'H:\New project\configs\method_y_image_training_v3_1.yaml' `
  --branch both
```

اگر قطع شد، دقیقاً همان command را دوباره اجرا کن. انتظار:

```text
PATTERN_CACHE_PROGRESS در console
status=COMPLETED برای هر شاخه
failure_count=0
quantitative_scale_claim_allowed=false
```

Count مورد انتظار را از membership همان config بخوان؛ hard-code نکن. در
snapshot فعلی amyloid cache ممکن است 895 row (development+transport) و tau
306 row داشته باشد. trainer بعدی فقط 560 A4 development را برای amyloid HPO
مصرف می‌کند.

### مرحله 9: full cache verification

تمام groupها را بدون بارگذاری یک‌جای تصاویر بررسی کن:

```python
for group in h5:
    assert status == COMPLETE
    assert shape == (1, 128, 128, 128)
    assert dtype == float16
    assert finite_fraction == 1.0
assert group_keys == expected_patient_hashes
```

گزارش فقط aggregate/hash-level باشد.

## آزمون‌ها

```powershell
& '.\.venv\Scripts\python.exe' -m pytest -p no:cacheprovider `
  tests\method_y_revival\test_image_cache_v3.py `
  tests\method_y_revival\test_image_cache_launcher_v3.py `
  tests\method_y_revival\test_image_training_runtime_v3.py -q
```

سپس کل Method Y. انتظار: zero failure. تست interruption باید weights و history
را با اجرای بدون interruption bitwise برابر کند.

## معیار پایان

- [ ] source/readiness/template/membership/dtype/compression همگی در contract هستند.
- [ ] HDF5 group attrs پیش از خواندن validate می‌شوند.
- [ ] terminal-boundary resume بدون retraining بازیابی می‌شود.
- [ ] precision audit پاس است.
- [ ] cacheهای هر دو شاخه کامل، finite، exact-membership و روی H هستند.
- [ ] training یا outer-test اجرا نشده است.
- [ ] تست‌های متمرکز و Method Y صفر failure دارند.

## STOP conditions

- source cache hash با readiness فرق دارد؛
- حتی یک group/scan غیرfinite، wrong tracer یا hash mismatch؛
- float16 tolerance fail؛
- H کمتر از 20 GiB free؛
- writer فعال؛
- count membership تغییر؛
- نیاز به repair عددی سه scan نامعتبر؛
- resume recovery به تغییر seed/split/config نیاز پیدا کند.

## نگهداری

هر config/cache جدید باید output root و contract جدید داشته باشد. cache ناقص را
پاک نکن؛ همان command باید per-subject commitهای معتبر را reuse کند.
