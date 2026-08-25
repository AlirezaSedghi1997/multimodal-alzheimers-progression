# Plan 012: robustness، calibration، subgroup و claim/privacy reports را کامل کن

> **دستور ایجنت مجری:** این Plan پس از GREEN شدن trainer Plan 011 پیاده می‌شود.
> برای تست می‌توان از OOF مصنوعی استفاده کرد؛ روی outer واقعی تا زمان اجرای
> launcher Plan 013 دسترسی نداشته باش. هیچ sensitivity analysis مجاز نیست
> endpoint، split، threshold یا primary winner را تغییر دهد.
>
> **Drift check:**
>
> ```powershell
> Set-Location -LiteralPath 'H:\New project'
> git diff --stat 5405716..HEAD -- src/method_y_revival scripts tests/method_y_revival
> rg -n "horizon_calibration_from_oof|subgroup_incremental_report|incremental_value_gate|run_leave_manufacturer" src\method_y_revival
> ```

## وضعیت

- **اولویت:** P1؛ لازم برای تفسیر نهایی و claim gate، نه برای ساخت cache
- **اندازه:** L
- **ریسک:** HIGH؛ گزارش غلط می‌تواند limitation را به ادعای مثبت تبدیل کند
- **وابسته به:** 009, 010, 011
- **دسته:** statistics / correctness / privacy / docs / tests
- **Planned at:** commit `5405716`, 2026-08-24

## چرا مهم است

config در `configs/method_y_image_training_v3.yaml:124-132` پنج robustness
analysis را وعده می‌دهد، ولی هنوز consumer ندارد. scanner balancing یا probe
نمی‌تواند domain shift را حذف کند؛ فقط حساسیت مدل را اندازه می‌گیرد. tau در
snapshot فعلی فقط Siemens را با ≥30 subject و ≥10 event دارد؛ GE و Philips
هرکدام 9 event دارند. بنابراین tau حتی با عملکرد خوب داخلی باید limitation
scanner را حفظ کند.

## reuse اجباری

- `oof_reporting.horizon_calibration_from_oof`؛
- `oof_reporting.build_subgroup_metadata`؛
- `oof_reporting.subgroup_incremental_report`؛
- `science_gates.cached_paired_oof_bootstrap`؛
- `science_gates.incremental_value_gate`؛
- `scanner_sensitivity.py` فقط پس از بررسی سازگاری API با image predictions؛
- `privacy.py` و الگوی `posttraining_finalize.py` برای public bundle.

منطق موجود را duplicate نکن؛ adapter image بساز.

## فایل‌های مجاز

```text
src/method_y_revival/image_robustness_v3.py                 (new)
src/method_y_revival/image_reporting_v3.py                  (new)
scripts/build_method_y_image_reports_v3.py                  (new)
configs/method_y_image_training_v3_1.yaml                   (فقط report contract)
tests/method_y_revival/test_image_robustness_v3.py           (new)
tests/method_y_revival/test_image_reporting_v3.py            (new)
tests/method_y_revival/test_image_privacy_claim_v3.py        (new)
advisor-plans/README.md                                      (status only)
```

خارج scope: تغییر primary model/HPO، external refit، XAI، official quantitative
claim، raw IDs در public output.

## مراحل

### مرحله 1: RED tests

```text
test_scanner_support_is_not_robustness_proof
test_scanner_weights_fit_from_outer_train_only
test_scanner_is_never_model_feature
test_scanner_balanced_challenger_reuses_locked_primary_config
test_leave_manufacturer_out_never_pools_ge_and_philips
test_unknown_scanner_remains_unknown
test_scanner_probe_uses_oof_embeddings_only
test_calibration_uses_locked_oof_without_refit
test_subgroup_small_cells_are_not_estimable
test_incremental_gate_stays_exploratory_when_scanner_gate_fails
test_quantitative_and_external_limitations_survive_positive_internal_delta
test_public_report_contains_no_patient_hash_path_embedding_or_prediction
```

### مرحله 2: scanner-balanced challenger

برای هر outer fold، weight فقط از outer-train manufacturer ساخته شود:

```python
counts = Counter(train.manufacturer)
w_i = 1 / counts[manufacturer_i]
w_i = w_i / mean(w_i)
```

قواعد:

- UNKNOWN یک category مستقل است؛ impute نشود.
- scanner هرگز predictor نیست.
- همان winner، fixed epochs، architecture، split و seed primary image استفاده
  شود؛ HPO تازه روی test ممنوع.
- این challenger sensitivity است و primary را جایگزین نمی‌کند.
- outer evaluation آن نیز controller/path مستقل و exactly-once دارد.

### مرحله 3: leave-manufacturer-out

Eligibility per manufacturer:

```text
held-out subjects >= 30
held-out events >= 10
remaining train contains >= 1 event
```

هر manufacturer مستقل بماند؛ GE+Philips هرگز `non-Siemens` نشوند. برای tau
در snapshot فعلی احتمالاً فقط Siemens eligibility دارد و training set بعد از
holdout باید دوباره event support را پاس کند. موارد ناکافی با
`NOT_ESTIMABLE_PRESPECIFIED_SUPPORT` ثبت شوند، نه zero performance.

این analysis از outer-CV primary برای انتخاب model استفاده نمی‌کند. یک protocol
سازگار:

```python
for eligible_manufacturer in prespecified_manufacturers:
    train = all development excluding manufacturer
    test = held manufacturer
    lock preprocessing + primary winner family
    fit without test access
    evaluate once
```

### مرحله 4: UNKNOWN-excluded sensitivity

membership/splitهای original عوض نشوند. فقط OOF rows دارای UNKNOWN از گزارش
sensitivity حذف و metrics paired/aggregate دوباره محاسبه شود. نام خروجی باید
`UNKNOWN_EXCLUDED_SENSITIVITY` باشد، نه primary.

### مرحله 5: OOF scanner probe

embedding فقط از outer-test locked model و پس از commit primary استخراج شود.
تمام subjectها دقیقاً یک OOF embedding دارند. Probe:

```text
multinomial/logistic probe
cross-validation grouped/stratified فقط روی OOF audit set
macro one-vs-rest AUC
per-class support
permutation/null check
```

اگر class support کافی نیست، `NOT_ESTIMABLE`. probe بالا diagnosis است و خودکار
model selection را عوض نمی‌کند. برای public report فقط aggregate AUC/support؛
embedding خصوصی بماند.

### مرحله 6: calibration و subgroup

Horizonها فقط اگر `valid_survival_horizons` آن‌ها را estimable بداند. گزارش:

```text
calibration intercept/slope + grouped plot values
sex
age prespecified bins
treatment arm
manufacturer
UNKNOWN scanner
```

حداقل per subgroup: 30 subject و 10 event. در غیر این صورت
`NOT_ESTIMABLE_LOW_SUPPORT`. هیچ subgroup positive claim مستقل صادر نشود.

### مرحله 7: paired incremental bootstrap

برای هر branch matched OOF base/augmented را با 2000 subject-level bootstrap
مقایسه کن:

```python
delta_ibs = IBS(augmented) - IBS(base)  # منفی بهتر
delta_cindex = C(augmented) - C(base)   # غیرمنفی مطلوب
```

positive internal image incremental gate فقط اگر:

```text
all 5 folds committed
upper CI(delta IBS) < 0
lower CI(delta C-index) >= 0
scanner sensitivity gate passed
```

حتی در این حالت final claim باید limitations کمی/خارجی را حفظ کند. عبارت مجاز:

```text
Pattern-normalized PET image features added internal prognostic information
beyond the prespecified scalar-plus-tabular baseline in A4.
```

عبارات official Centiloid/CenTauR، diagnosis، causal و external ممنوع‌اند.

### مرحله 8: claim consistency state machine

```python
training_execution_allowed = readiness.hard_blockers == []
internal_incremental_pass = paired_gate and scanner_gate
quantitative_standard_claim = internal_incremental_pass and quantitative_equivalence
external_claim = quantitative_standard_claim and locked_external_no_refit_pass
```

`scanner_support_available` هیچ‌گاه جای `scanner_gate` نباشد.

### مرحله 9: privacy-safe rendering

خصوصی:

```text
private/oof_predictions/**
private/oof_embeddings/**
private/subgroup_linkage/**
```

عمومی/aggregate:

```text
reports/<branch>_image_incremental.json
reports/<branch>_calibration.json
reports/<branch>_subgroups.json
reports/<branch>_scanner_sensitivity.json
reports/IMAGE_CLAIM_POLICY.json
```

privacy scan باید recursively این tokenها را fail کند:

```text
patient_hash
image_path_private
H:\
A4 Learn
individual_prediction
embedding vector
.pt / .ckpt / private parquet
```

خود نام کلید `patient_hash` در schema documentation می‌تواند false positive
باشد؛ scanner باید JSON values و artifact roles را context-aware بررسی کند یا
report directory را طوری بسازد که این کلید اصلاً وارد نشود.

## آزمون و verification

```powershell
& '.\.venv\Scripts\python.exe' -m pytest -p no:cacheprovider `
  tests\method_y_revival\test_image_robustness_v3.py `
  tests\method_y_revival\test_image_reporting_v3.py `
  tests\method_y_revival\test_image_privacy_claim_v3.py -q
```

سپس کل Method Y. تمام tests باید zero failure باشند.

## معیار پایان

- [ ] پنج analysis config واقعاً implementation/test دارند.
- [ ] scanner هیچ‌گاه predictor نیست.
- [ ] GE/Philips ادغام نمی‌شوند.
- [ ] UNKNOWN impute نمی‌شود.
- [ ] calibration/subgroups فقط OOF locked را مصرف می‌کنند.
- [ ] claim gate با quantitative/external limitations سازگار است.
- [ ] public reports هیچ row-level information ندارند.
- [ ] هیچ production outer-test در زمان build این Plan خوانده نشده است.

## STOP conditions

- OOF membership incomplete یا mismatched؛
- probe/sensitivity نیازمند بازکردن HPO یا تغییر primary winner؛
- scanner metadata مفقود را باید حدس زد؛
- bootstrap کمتر از 2000 iteration معتبر؛
- public privacy scan fail؛
- ادعای مثبت بدون هر پنج outer fold؛
- تغییر خارج scope.

## نگهداری

external validation بعداً output مستقل دارد و هرگز برای recalibration primary یا
feature selection استفاده نمی‌شود. XAI نیز plan جدا می‌خواهد و شرط شروع image
training نیست.
