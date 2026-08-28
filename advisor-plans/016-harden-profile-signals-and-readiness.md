# Plan 016: execution profile را به bytes زنده bind کن، توقف همهٔ stageها را ایمن کن و readiness builder بساز

> **Executor instructions:** ابتدا RED testها را اضافه کن. این کارت هیچ MRI واقعی
> جدیدی پردازش نمی‌کند؛ فقط code/fixture tests. broad refactor ممنوع است.

## Status

- **Priority:** P0
- **Effort:** L
- **Risk:** MED-HIGH — runtime state machine و scientific evidence
- **Depends on:** Plan 015
- **Category:** correctness, recovery, evidence integrity
- **Planned at:** commit `6b6f852`, 2026-08-28

## Current state

- `build_execution_profile`: `scripts/research/prepare_brainiac_raw_t1_v2.py:200-269`.
- `validate_execution_profile`: همان فایل `:272-309`.
- مقدار غایب `fresh_run` در line 248 به‌صورت `True` پذیرفته می‌شود.
- test helper در `tests/test_brainiac_preprocessing_speedup.py:96-129` یک JSON
  خودساخته را به‌عنوان profile معتبر می‌پذیرد.
- signal handler فقط flag را set می‌کند (`:132-144`)؛ equivalence/calibration/
  sensitivity آن را مصرف نمی‌کنند.
- `READY_REPORT_PATH` در `:89-91` تعریف شده و writer ندارد.
- config هنوز `sequential_vs_orchestrated` می‌گوید، درحالی‌که policy سه leg
  (1 sequential + 2 independent parallel) را الزام می‌کند.

## Scope

```text
scripts/research/prepare_brainiac_raw_t1_v2.py
src/research/brainiac_preprocessing_v2.py
src/research/brainiac_preprocessing_v2_schema.py
configs/research/brainiac_preprocessing_r1_primary_v2.yaml
tests/test_brainiac_preprocessing_speedup.py
tests/test_brainiac_preprocessing_final_repairs.py
tests/test_brainiac_preprocessing_v2_run.py
tests/fixtures/brainiac_transaction_worker.py
```

اگر extraction کوچک لازم شد، فقط یک ماژول
`src/research/brainiac_preprocessing_supervision.py` برای profile/readiness/state
بساز. logic علمی pipeline را جابه‌جا نکن.

## Steps

### 1. RED tests برای profile جعلی و evidence drift

موارد اجباری:

- `fresh_run` missing → reject، نه default true؛
- `fresh_run=false` → هرگز PARALLEL_SAFE؛
- profile self-hash صحیح ولی root/receipt غایب → reject؛
- receipt digest map صحیح‌نما ولی live receipt bytes متفاوت → reject؛
- scientific digest از field حذف‌شده/اضافی → reject؛
- سه leg با sample set یا contract متفاوت → reject؛
- partial/temp/quarantine/reparse/lock در هر leg → reject؛
- elapsed self-asserted بدون leg timing receipt → parallel promotion block؛
- profile contract قدیمی → reject؛
- `SEQUENTIAL_SAFE` فقط پس از سه leg کامل و integrity صحیح قابل صدور است.

### 2. profile schema v2 byte-bound بساز

هر leg باید این evidence را داشته باشد:

```text
role: sequential | parallel_1 | parallel_2
root_relative_path
root_tree_sha256
sample_hashes (exact 12, label-blind selection digest)
receipt_file_sha256_by_sample
output_sha256_by_sample
transform_sha256_by_sample
canonical_scientific_sha256_by_sample
contract_sha256
started_at_utc / finished_at_utc / elapsed_seconds
fresh_run (required boolean)
resume_event_count
integrity_status recomputed from live bytes
```

validator path را خودش از allowlisted roots resolve کند؛ caller حق arbitrary
root ندارد. تمام receiptها را با `validate_receipt` و live output/transform
rehash کند. embedded summary فقط cache است، authority نیست.

اگر leg interrupted/resumed بود، exact equivalence می‌تواند PASS شود ولی timing
promotion ممنوع و execution mode باید `SEQUENTIAL_SAFE` باشد. profile موجود با
schema/contract mismatch overwrite نشود؛ blocker + versioned supersession لازم
است.

### 3. stop flag را به تمام long-running stages عبور بده

امضاهای زیر `stop_requested` بگیرند:

```text
run_equivalence
_calibrate_if_needed
run_smoke / real 12-sample acceptance
run_sensitivity
run_primary (هم‌اکنون دارد)
run_orchestrated / run_orchestrated_batch
```

قاعده:

1. signal جدید submit نمی‌کند؛
2. حداکثر دو sample in-flight تا transaction commit/rollback امن drain می‌شوند؛
3. artifact terminal/profile/threshold فقط در stage کامل ساخته می‌شود؛
4. خروجی `PAUSED_RESUMABLE`, exit 75؛
5. rerun همان command committed valid را skip می‌کند؛
6. hard kill از آخرین journal معتبر ادامه می‌دهد.

در Windows تست واقعی subprocess برای SIGINT و SIGTERM/terminate در هر stage
نماینده اجرا کن؛ فقط monkeypatch کافی نیست.

### 4. config-policy contradiction را ببند

config باید دقیقاً بنویسد:

```text
equivalence_design: 12_LABEL_BLIND_ONE_SEQUENTIAL_TWO_INDEPENDENT_PARALLEL
parallel_repeat_count: 2
parallel_minimum_speedup_fraction: 0.15
execution_profile_authority: HASH_LOCKED_LIVE_BYTE_VERIFIED
```

`_config()` باید این fields را validate کند. هر تغییر config hash generation
جدید می‌سازد؛ legacy receipt reuse باید fail closed باشد.

### 5. acceptance mode بدون primary بساز

public mode جدید:

```text
--mode acceptance
```

ترتیب دقیق:

```text
preflight
→ safe reconciliation (Plan 014 rules)
→ current-generation three-leg equivalence
→ calibration 32
→ validated real 12-sample smoke (equivalence sequential leg may satisfy it
   only if exact same pipeline/contract and is explicitly cross-bound)
→ test/fault evidence check
→ readiness candidate
→ STOP BEFORE primary
```

`--mode full` باید ابتدا همین acceptance function را idempotently صدا بزند و
فقط بعد از readiness معتبر وارد primary شود. command رسمی full تغییر نمی‌کند.

### 6. readiness builder واقعی بساز

`READY_TO_START_OR_RESUME_BRAINIAC_PREPROCESSING.json` فقط وقتی صادر شود که:

```text
preflight PASS and no active/unknown writer
dependency closure tracked + hash-bound
source digest and full battery preservation PASS
execution profile live-byte validated
calibration 32/32 and threshold manifest bound
real smoke 12/12 physical QC PASS
transaction + SIGINT/SIGTERM + corruption tests PASS
primary has 0 valid receipts OR a contract-identical resumable subset
no invalid/quarantine/temp/reparse/unknown locks
Plans 005-007 remain hard-disabled until 1669/1669
claim ceiling POST_HOC_INTERNAL_TECHNICAL_EVIDENCE
```

artifact باید تمام input/report hashes، test manifest، current HEAD/dirty scoped
inventory و `completion=false`, `training_authorization=false` را bind کند.
موجودِ mismatch overwrite نشود. `READY_REPORT_PATH` باید توسط test پوشش داده شود.

## Verification

```powershell
& 'H:\New project\.venvs\brainiac\Scripts\python.exe' -m pytest `
  tests/test_brainiac_preprocessing_final_repairs.py `
  tests/test_brainiac_preprocessing_speedup.py `
  tests/test_brainiac_preprocessing_v2.py `
  tests/test_brainiac_preprocessing_v2_run.py `
  tests/test_brainiac_preprocessing_repair.py -q
```

Expected: zero failure؛ test count دقیق در report ثبت شود، حدس زده نشود.

## Done criteria

- [ ] profile schema v2 از live roots/receipts/bytes recompute می‌شود.
- [ ] missing/false fresh_run هرگز parallel را unlock نمی‌کند.
- [ ] تمام long stages signal-safe و same-command resumable هستند.
- [ ] acceptance mode primary را شروع نمی‌کند.
- [ ] readiness builder موجود است و fixture فقط با evidence کامل READY می‌شود.
- [ ] config و policy یک equivalence design دارند.
- [ ] no primary/calibration/live equivalence run در این کارت.
- [ ] source/battery preservation ثابت است.

## STOP conditions

- تغییر schema نیازمند reuse artifact نسل قدیمی؛
- signal test نمی‌تواند bounded drain/75 را ثابت کند؛
- profile validation بدون live root قابل انجام است؛
- readiness برای UNKNOWN/NOT_MEASURABLE صادر می‌شود؛
- source scope به trainer/battery finalized bytes کشیده می‌شود.
