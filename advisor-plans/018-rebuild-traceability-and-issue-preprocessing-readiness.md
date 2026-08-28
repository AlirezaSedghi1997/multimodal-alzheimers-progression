# Plan 018: traceability را از artifacts زنده بازسازی و تنها readiness پیش‌پردازش را صادر کن

> **Executor instructions:** این کارت پس از PASS واقعی Plan 017 اجرا می‌شود.
> status دستی را patch نکن؛ generator باید facts را از authorityهای byte-bound
> بخواند. primary/embedding/CAFT اجرا نمی‌شود.

## Status

- **Priority:** P0
- **Effort:** M
- **Risk:** LOW-MED — governance/traceability
- **Depends on:** Plan 017
- **Category:** docs, tests, release gate
- **Planned at:** commit `6b6f852`, 2026-08-28

## Defects to close

1. novelty matrix registry SHA قدیمی دارد و P08 builder fail می‌شود.
2. roadmap JSON هنوز R0 incomplete و R1 hard-disabled می‌نویسد.
3. `docs/PROJECT_MEMORY_INDEX.md` count/PID قدیمی را live معرفی می‌کند.
4. SDD progress هنوز statusهای قدیمی دارد.
5. full BrainIAC suite چهار failure دارد؛ دو test stateful، یک test battery-stale
   و یک linkage failure.
6. readiness path تا Plan 016 writer نداشت.

## Scope

```text
research/brainiac_caft_20260808/novelty_matrix*.json
scripts/build_brainiac_p08_status.py
results/method_x_registry/project_roadmap.json
docs/plans/method_x_project_roadmap.md
docs/PROJECT_MEMORY_INDEX.md
.superpowers/sdd/brainiac_caft_training_readiness_20260822/**
.superpowers/sdd/method_x_brainiac_caft_execution_tasks_20260808/**
advisor-plans/README.md
tests/test_brainiac_b1_action_readiness.py
tests/test_brainiac_preexecution_gate.py
results/method_x_registry/brainiac_r1/READY_TO_START_OR_RESUME_BRAINIAC_PREPROCESSING.json
results/method_x_registry/brainiac_r1/brainiac_preprocessing_test_manifest_v1.json
```

Old status/novelty artifacts حذف یا in-place جعل نشوند؛ نسخهٔ جدید باید
`supersedes` و hash predecessor داشته باشد.

## Steps

### 1. novelty linkage را بدون تغییر ادعای علمی بازسازی کن

registry زنده را audit کن و novelty matrix جدید را با SHA/size همان registry
بساز. counts، closest prior art، `prior_art_exact_match_found=false` و
`NOVELTY_CANDIDATE` باید از evidence موجود recompute شوند. اگر recompute با
matrix قبلی semantic mismatch دارد، STOP؛ hash را کورکورانه عوض نکن.

### 2. tests را hermetic و علمی اصلاح کن

- چون battery terminal است، test نباید
  `FIVE_CNN_BATTERY_NOT_TERMINAL` را الزام کند؛ باید terminal binding و blockerهای
  واقعی preprocessing/downstream را assert کند.
- gate tests باید temp authority/injected process+lock providers داشته باشند؛
  به lock زندهٔ workspace وابسته نباشند.
- یک integration test جدا باید live blocker را read-only گزارش کند، نه اینکه
  `allowed=true` را hard-code کند.
- novelty test باید current registry/matrix binding را verify کند.

test را برای سبزشدن با حذف fail-closed assertion ضعیف نکن.

### 3. full related verification را اجرا کن

```powershell
$files = Get-ChildItem -LiteralPath 'H:\New project\tests' -File `
  -Filter 'test_brainiac*.py' | Sort-Object Name | ForEach-Object FullName

& 'H:\New project\.venvs\brainiac\Scripts\python.exe' -m pytest @files -q
```

سپس dependencyهای shared مربوط به action guard/source isolation را اجرا کن.
test manifest باید command، Python executable، HEAD، file hashes، passed/failed/
skipped/warnings، start/end و exit code را ثبت کند. Expected: zero failure/error.

### 4. roadmap/memory/SDD را از generator واحد بساز

Expected facts پیش از primary:

```text
R0=R0_COMPLETE
CITATION_CORROBORATION_PARTIAL
NOVELTY_CANDIDATE
battery=VERIFIED_TERMINAL
Plan 002=ACCEPTED_READY_FOR_PRIMARY
Plan 003=NOT_STARTED_PRIMARY_0_OF_1669
Plans 005-007=NOT_EXECUTED / HARD_DISABLED_UNTIL_1669
Plan 006=CODE_ONLY_UNACCEPTED
external=OPTIONAL_UNAVAILABLE_NONBLOCKING
claim=POST_HOC_INTERNAL_TECHNICAL_EVIDENCE
```

memory نباید PID/count «زنده» ذخیره کند مگر با timestamp و artifact link؛ status
runtime باید از command `--mode status` خوانده شود.

### 5. readiness نهایی را recompute و atomic صادر کن

فقط artifact زیر:

```text
results/method_x_registry/brainiac_r1/
  READY_TO_START_OR_RESUME_BRAINIAC_PREPROCESSING.json
```

Required fields:

```text
schema_version
status=READY_TO_START_OR_RESUME_BRAINIAC_PREPROCESSING
completion=false
training_authorization=false
primary_expected_n=1669
primary_valid_n=0 (یا subset contract-identical برای resume)
preflight/profile/calibration/smoke/test/closure/traceability bindings
source_digest=9b0f90a5...871b3
battery_manifest_sha256=40a4a009...db971d
full_battery_tree_preservation=PASS
plans_005_006_007_allowed=false
human_approval_required=false
manual_override_allowed=false
claim_ceiling=POST_HOC_INTERNAL_TECHNICAL_EVIDENCE
external_status=ROBUSTNESS_EVIDENCE_NOT_EXTERNAL_VALIDATION
integrity_sha256
```

validator باید همهٔ زیرین‌ها را دوباره بخواند. وجود فایل readiness به‌تنهایی
authority نیست. rerun باید zero-write/already-ready باشد؛ mismatch باید blocker
نسخه‌دار بسازد و overwrite نکند.

### 6. preservation postflight

freeze Plan 014 را با live bytes مقایسه کن. Method X source، battery manifest،
تمام finalized checkpoint/state/outer-result hashes باید identical باشند.
هیچ outer loader call یا subject prediction جدید مجاز نیست.

## Done criteria

- [ ] full BrainIAC suite zero failure/error است.
- [ ] test manifest دقیق و hash-bound است.
- [ ] novelty/P08 linkage PASS است و claim همان NOVELTY_CANDIDATE مانده.
- [ ] roadmap/memory/SDD با live artifacts هم‌راستا هستند.
- [ ] فقط preprocessing readiness صادر شده؛ completion/training auth صادر نشده.
- [ ] primary 0/1669 و Plans 005–007 hard-disabled هستند.
- [ ] source و full battery tree preservation PASS است.

## STOP conditions

- هر test failure/error یا waiver بدون dependency proof؛
- novelty semantic drift؛
- readiness زیرین UNKNOWN/NOT_MEASURABLE؛
- status generator به self-asserted JSON اعتماد می‌کند؛
- outer/battery/source drift؛
- primary/embedding/training شروع شده است.
