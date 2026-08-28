# Plan 017: acceptance واقعی equivalence، calibration و smoke را اجرا کن؛ primary را شروع نکن

> **Executor instructions:** این کارت runtime است و source/config را تغییر
> نمی‌دهد. فقط وقتی Plans 014–016 DONE و Git closure clean است اجرا شود. اگر
> source change لازم شد، BLOCK کن و به Plan 016 برگردان؛ hotfix هنگام runtime
> ممنوع است.

## Status

- **Priority:** P0
- **Effort:** L/runtime (چند ساعت تا بیش از یک روز)
- **Risk:** HIGH — raw MRI preprocessing + GPU
- **Depends on:** Plans 014, 015, 016
- **Category:** acceptance, performance, recovery
- **Planned at:** commit `6b6f852`, 2026-08-28

## Preconditions

همه باید machine-PASS باشند:

```text
no Method X/Method Y/BrainIAC writer
no foreign/unknown CUDA compute PID
shared guard absent or owned by current acceptance process
dependency closure untracked_count=0
Method X source digest 9b0f90a5...871b3
battery manifest 40a4a009...db971d and full battery tree inventory unchanged
primary valid receipt count=0
equivalence_v2 roots absent or contract-identical resumable
legacy roots only under superseded with receipts
H free space >=40 GiB
CUDA total VRAM >=7.5 GiB
all paths non-reparse
```

هر process/lock/VRAM query UNKNOWN = STOP. process کاربر را kill نکن.

## Command

```powershell
Set-Location -LiteralPath 'H:\New project'

& 'H:\New project\.venvs\brainiac\Scripts\python.exe' `
  'H:\New project\scripts\research\prepare_brainiac_raw_t1_v2.py' `
  --mode acceptance `
  --device cuda
```

اجرای دوبارهٔ همین command باید از stage/sample معتبر ادامه دهد. `--no-resume`،
limit، workers، policy override یا sample override ممنوع است.

## Stage expectations

### A. preflight و reconciliation

Expected: همهٔ checks literal `true`؛ no sample write پیش از PASS. archive فقط
مطابق Plan 014 و با receipt. dependency/preflight report باید HEAD/config/policy/
closure/source/battery hashes را bind کند.

### B. three-leg equivalence نسل فعلی

roots:

```text
r1_primary_v2_smoke/equivalence_v2/sequential
r1_primary_v2_smoke/equivalence_v2/parallel_1
r1_primary_v2_smoke/equivalence_v2/parallel_2
```

هر leg دقیقاً 12 receipt، 12 output، 12 transform، journal terminal و zero
quarantine/temp/lock دارد. sample set یکی و label-blind است. validator تمام
bytes را rehash می‌کند.

تصمیم profile:

- exact + هر سه fresh + median time reduction >=15% → `PARALLEL_SAFE`؛
- exact ولی resumed/not-fresh یا speed <15% → `SEQUENTIAL_SAFE`؛
- هر mismatch/integrity/QC → BLOCK؛ profile PASS صادر نشود.

هیچ اصراری برای parallel وجود ندارد؛ correctness بر سرعت مقدم است.

### C. calibration 32

دقیقاً 32 lowest-hash sample، label-blind. threshold manifest باید به 32 sample
hash، 32 receipt hash، contract/policy/config/closure/runtime/template/weights
bind باشد. rerun همان bytes/hash و zero new receipt بدهد. MAD anomaly/hard QC
aggregate را BLOCK می‌کند؛ threshold تغییر نده.

### D. real 12-sample smoke

اگر sequential equivalence leg دقیقاً همان raw-T1 pipeline و contract را دارد،
validator می‌تواند آن را به‌عنوان smoke bind کند؛ صرف count کافی نیست. Expected:

```text
12/12 valid
finite 96x96x96
physical affine/QC valid
labels_read=false, pet_read=false
raw_path/raw_identifier absent recursively
zero quarantine/temp/reparse/unknown lock
```

### E. safe-stop/fault acceptance

روی fixture subprocess و یک smoke representative:

- SIGINT و SIGTERM در sequential و parallel؛
- hard kill در transaction boundaries؛
- corrupt receipt/output/transform؛
- journal truncation؛
- receipt-before-journal recovery؛
- stale/reparse/PID-reuse lock.

uninterrupted و resumed باید output/transform/receipt/aggregate canonical bytes
یکسان بسازند. هر failure = BLOCK، نه waiver.

### F. پایان acceptance

این command باید پیش از primary متوقف شود. Expected:

```text
exit 0
acceptance_status=PASS
primary_completed_rows=0
primary_sample_outputs_written=false
readiness_candidate=true
```

readiness نهایی در Plan 018 پس از traceability/full regression صادر می‌شود.

## Monitoring

فقط counts، elapsed و sample-hash prefix کوتاه گزارش کن؛ raw path/ID ممنوع.

```powershell
Get-ChildItem -LiteralPath `
  'H:\New project\data\derived\method_x_brainiac\r1_primary_v2_smoke\equivalence_v2' `
  -Recurse -Filter '*.json' | Measure-Object
```

Ctrl+C باید بعد از حداکثر دو in-flight transaction، exit 75 دهد. اگر terminal
برنگشت، process را force-kill نکن مگر operator صریحاً بخواهد؛ state را ثبت و
BLOCK کن.

## Done criteria

- [ ] current-generation three-leg profile live-byte PASS دارد.
- [ ] mode صادقانه PARALLEL_SAFE یا SEQUENTIAL_SAFE است.
- [ ] calibration 32/32 و threshold binding PASS است.
- [ ] real smoke 12/12 و privacy/geometry PASS است.
- [ ] stop/fault/corruption acceptance PASS است.
- [ ] primary هنوز 0/1669 است.
- [ ] source digest و full battery tree inventory قبل/بعد identical است.
- [ ] هیچ Plans 005–007 stage اجرا نشده است.

## STOP conditions

- writer/CUDA process فعال یا UNKNOWN؛
- gate/anchor/closure/hash drift؛
- هر QC/quarantine/privacy issue؛
- artifact نسل قدیمی در current root؛
- signal/restart inconsistency؛
- primary output قبل از acceptance terminal؛
- نیاز به source edit.
