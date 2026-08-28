# Plan 019: پس از readiness معتبر، preprocessing اصلی 1669 MRI را اجرا و resume کن

> **Executor instructions:** این کارت جایگزین بخش runtime کهنهٔ Plan 003 است.
> source/config/test را تغییر نده. فرمان را دوباره اجرا کن تا resume شود. هر
> integrity/QC defect = BLOCK و بازگشت به کارت توسعهٔ جدا، نه hotfix.

## Status

- **Priority:** P0/runtime
- **Effort:** XL (برآورد قبلی 45–75 ساعت؛ تضمین زمانی نیست)
- **Risk:** HIGH
- **Depends on:** Plan 018 و readiness معتبر
- **Category:** data preparation
- **Planned at:** commit `6b6f852`, 2026-08-28

## Preconditions

- readiness زیرین با validator live PASS؛ صرف وجود JSON کافی نیست.
- هیچ Method X/Method Y/BrainIAC/foreign CUDA writer فعال یا UNKNOWN نیست.
- source/config/policy/closure/profile/calibration/smoke/test/battery hashes با
  readiness برابرند.
- primary root صفر یا subset دقیقاً contract-identical دارد؛ invalid/quarantine
  ندارد.
- H free space و CUDA policy PASS.

## تنها فرمان مجاز

```powershell
Set-Location -LiteralPath 'H:\New project'

& 'H:\New project\.venvs\brainiac\Scripts\python.exe' `
  'H:\New project\scripts\research\prepare_brainiac_raw_t1_v2.py' `
  --mode full `
  --device cuda
```

همان فرمان برای start و resume است. هیچ flag دیگری اضافه نکن.

## رفتار الزامی

```text
valid committed sample -> verify + skip
missing sample -> process once
committed mismatch -> exit 78 / BLOCK, never overwrite
QC anomaly/quarantine -> exit 3 and aggregate blocked
SIGINT/SIGTERM -> drain <=2 in-flight, mirror journals, exit 75
rerun -> continue remaining sample only
completion -> 1669/1669 live-byte revalidated, exit 0
```

Counts هر 25 sample یا 30 دقیقه گزارش شوند؛ raw path/ID/label ممنوع.

## Final validation

```text
expected_n=1669
valid_committed_n=1669
unique_n=1669
missing_n=0
invalid_n=0
quarantined_n=0
temp_n=0
reparse_n=0
labels_read=0
pet_read=0
raw_path_persisted=0
all output shape 96x96x96 and finite
all receipts current contract and threshold binding
sensitivity 64 complete; mask median>=0.99, p05>=0.97
direct 128->96 REPORT_ONLY_NOT_PRIMARY
```

validator باید تمام 1669 output/transform/receipt/journal bytes را streaming
rehash کند. نمونه‌گیری جای full validation نیست.

## Preservation

پس از هر pause و در completion:

- Method X source digest دقیقاً `9b0f90a5...871b3`؛
- battery manifest و full finalized tree inventory ثابت؛
- outer loader call count صفر؛
- Plans 005–007 تا قبل از aggregate terminal شروع نشده‌اند.

## Done criteria

- [ ] 1669/1669 receipt و output معتبر.
- [ ] aggregate schema v2 live-byte verified و immutable.
- [ ] sensitivity 64 و mask stability PASS.
- [ ] pause/resume واقعی حداقل یک بار بدون rewrite اثبات شده.
- [ ] source/battery/outer preservation PASS.
- [ ] پس از completion، Planهای 005–008 فقط با drift review تازه unlock می‌شوند.

## STOP conditions

- writer/GPU/lock UNKNOWN؛
- readiness یا هر input binding drift؛
- quarantine/privacy/geometry/integrity failure؛
- resume committed file را بازنویسی کند؛
- نیاز به source edit؛
- external dataset یا label برای preprocessing درخواست شود.

## Downstream handoff

پس از DONE، قبل از اجرای Plans 005–008، هر فایل آن‌ها را نسبت به schemas/HEAD
زنده drift-review کن. ترتیب حفظ می‌شود:

```text
005 frozen feature preparation
→ 006 real six-arm CAFT trainer acceptance
→ 007 orchestrator/exact-resume/autosupervision
→ 008 final BrainIAC training readiness/traceability
```

Plan 006 تا آن زمان `CODE_ONLY_UNACCEPTED` و Plans 005/007 `NOT_EXECUTED` باقی
می‌مانند.
