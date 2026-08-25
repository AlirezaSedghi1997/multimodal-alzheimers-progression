# Plan 009: قرارداد dual-gate را رسمی و readiness را fail-closed کن

> **دستور ایجنت مجری:** این فایل را کامل بخوان، سپس دقیقاً به‌ترتیب مراحل
> پیش برو. هر تغییر با RED test آغاز شود. در صورت رخ‌دادن هر STOP condition
> متوقف شو و گزارش بده؛ config، seed، split یا output root را خودسرانه تغییر
> نده. در پایان row همین Plan در `advisor-plans/README.md` را به‌روزرسانی کن.
>
> **Drift check:**
>
> ```powershell
> Set-Location -LiteralPath 'H:\New project'
> git diff --stat 5405716..HEAD -- docs configs src/method_y_revival scripts tests/method_y_revival
> Get-FileHash -Algorithm SHA256 `
>   configs\method_y_image_training_v3.yaml, `
>   src\method_y_revival\image_training_policy.py, `
>   src\method_y_revival\image_training_manifest_v3.py, `
>   scripts\build_method_y_image_training_readiness_v3.py
> ```
>
> SHAهای مورد انتظار در `METHOD_Y_IMAGE_V3_TODAY_LOG_20260824.md` ثبت شده‌اند.
> اگر هرکدام فرق دارد، ابتدا diff را بخوان؛ هیچ overwrite کور انجام نده.

## وضعیت

- **اولویت:** P0؛ پیش‌نیاز تمام trainingهای image v3
- **اندازه:** M
- **ریسک:** MED؛ تغییر اشتباه می‌تواند claim gate را باز یا membership را عوض کند
- **وابسته به:** هیچ‌کدام
- **دسته:** correctness / docs / science governance / tests
- **Planned at:** commit `5405716`, 2026-08-24

## چرا مهم است

درخواست قفل‌شدهٔ جدید اجازه می‌دهد image training داخلیِ اکتشافی با ورودی
pattern-normalized اجرا شود، حتی وقتی equivalence کمی یا external validation
موجود نیست. اما منبع حقیقت قدیمی در
`docs/PROGRAM_DEFAULT_SOURCE_OF_TRUTH_2026-07-19.md:183` هنوز می‌گوید در این
شرایط «آموزش/ادعا شروع نمی‌شود». بدون ADR جدید، دو عامل می‌توانند دو تصمیم
متضاد بگیرند.

هم‌زمان، readiness فعلی در
`scripts/build_method_y_image_training_readiness_v3.py:404-415` وجود حداقل دو
manufacturer را به `scanner_robustness_proven` پاس می‌دهد. تعداد کافی فقط
feasibility تحلیل robustness است، نه اثبات آن. این خطا فعلاً training را خراب
نمی‌کند، ولی limitation amyloid را زودتر از موعد حذف می‌کند و در آینده می‌تواند
claim gate را اشتباه باز کند.

## وضعیت فعلی و شواهد

- ADR فعلی در `docs/decisions/ADR-2026-07-19-default-method-xy-program.md:23`
  برای pivot/تغییر معماری ADR جدید می‌خواهد.
- تفکیک gate در `src/method_y_revival/image_training_policy.py:36-42` پیاده شده است.
- hard execution blockers در همان فایل، خطوط `52-79`، tracer، endpoint، split،
  leakage، event support، finite و geometry را نگه می‌دارند.
- `image_training_manifest_v3.py:137-190` هر fold را بررسی می‌کند، اما صریحاً
  ثابت نمی‌کند مجموع outer-testها هر subject را دقیقاً یک بار پوشش می‌دهد یا
  `inner_train ∪ inner_validation == filtered_outer_train` است.
- snapshot زنده فعلی این invariantها را پاس می‌کند؛ نقص در validator است، نه
  اثبات خرابی membership فعلی.
- readiness canonical فقط `results/method_y_revival_v3_image_v1/` است. مسیر
  `results/method_y_revival_v3_image/` superseded است و نباید مصرف/حذف شود.

## فرمان‌های موردنیاز

| هدف | فرمان | انتظار |
|---|---|---|
| process guard | دستور مرحله 1 | هیچ writer هم‌مسیر |
| RED/GREEN متمرکز | `& '.\.venv\Scripts\python.exe' -m pytest -p no:cacheprovider tests\method_y_revival\test_image_training_policy_v3.py tests\method_y_revival\test_image_training_manifest_v3.py -q` | صفر failure |
| بازسازی readiness | `& '.\.venv\Scripts\python.exe' scripts\build_method_y_image_training_readiness_v3.py --branch both` | exit 0، status READY |
| suite Method Y | `& '.\.venv\Scripts\python.exe' -m pytest -p no:cacheprovider tests\method_y_revival -q` | صفر failure |

## محدوده

**فقط این فایل‌ها مجاز به تغییر/ایجادند:**

```text
docs/decisions/ADR-2026-08-24-method-y-image-dual-gate.md                 (new)
docs/PROGRAM_DEFAULT_SOURCE_OF_TRUTH_2026-07-19.md                       (append-only addendum/link)
configs/method_y_image_training_v3.yaml
src/method_y_revival/image_training_policy.py
src/method_y_revival/image_training_manifest_v3.py
scripts/build_method_y_image_training_readiness_v3.py
tests/method_y_revival/test_image_training_policy_v3.py
tests/method_y_revival/test_image_training_manifest_v3.py
tests/method_y_revival/test_image_training_readiness_v3.py               (new)
advisor-plans/README.md                                                   (status only)
```

**خارج از محدوده و ممنوع:**

- هر فایل v1/v2 در `results/`، `caches/` یا public bundle؛
- ساخت cache، HPO، training یا outer-test؛
- تغییر endpoint، split، seed، tracer membership یا feature list؛
- معرفی pattern-normalized PET به‌عنوان Centiloid/CenTauR؛
- هر human sign-off یا manual annotation gate.

## Git workflow

- branch جدید فقط در صورت دستور مالک؛ در worktree dirty فعلی branch switch نکن.
- `git reset/clean/checkout --/commit/push` ممنوع است.
- قبل و بعد، `git status --short -- <scope>` را ذخیره کن.

## مراحل

### مرحله 1: preflight و process guard

```powershell
Get-CimInstance Win32_Process |
  Where-Object {
    $_.Name -match 'python|powershell|matlab|spm' -and
    $_.CommandLine -match 'method_y_revival_v3_image_v1|method_y_image|PET|tau|amyloid'
  } |
  Select-Object ProcessId,ParentProcessId,Name,CreationDate,CommandLine
```

اگر writer همین output root را هدف گرفته، STOP. process فقط بر اساس نام
`python.exe` متوقف نشود.

### مرحله 2: RED tests برای semantics جدید

در `test_image_training_readiness_v3.py` ابتدا تست‌های زیر را اضافه کن و شکست
آن‌ها را ببین:

```text
test_scanner_support_does_not_equal_scanner_robustness_proof
test_amyloid_readiness_keeps_scanner_robustness_limitation_before_analysis
test_each_filtered_subject_appears_in_exactly_one_outer_test
test_inner_train_and_validation_exactly_partition_filtered_outer_train
test_duplicate_outer_test_membership_fails_closed
test_missing_inner_member_fails_closed
test_superseded_output_root_is_never_selected_from_config
```

انتظار RED: دست‌کم چهار تست invariant/semantics فعلی fail شوند.

### مرحله 3: ADR نسخه‌دار را ایجاد کن

ADR جدید باید این تصمیم‌ها را بدون ابهام ثبت کند:

```yaml
source_regional_v2_role: PRIMARY_LOCAL_INTERNAL_RESULT
image_v3_role: SECONDARY_FINAL_INTERNAL_EXPLORATORY_ANALYSIS
training_execution_requires:
  - exact tracer identity
  - future locked endpoint
  - locked subject-grouped 5x3 split
  - zero subject leakage
  - valid transform/geometry
  - finite voxels
  - adequate event support
claim_requires_additionally:
  - quantitative equivalence
  - actual scanner robustness analyses
  - OOF incremental value
external_claim_requires_additionally:
  - locked no-refit external evaluation
human_review_required: false
```

تصریح کن statistics جای provenance کمی یا cohort مستقل را نمی‌گیرد. ADR
2026-07-19 حذف نشود؛ وضعیت آن برای این موضوع `SUPERSEDED_IN_PART` و لینک ADR
جدید اضافه شود. به سند source-of-truth فقط یک addendum نسخه‌دار اضافه کن؛ متن
تاریخی را بازنویسی نکن.

**Verify:** `rg -n "dual-gate|SUPERSEDED_IN_PART|training_execution_allowed|positive_claim_allowed" docs` باید ADR و addendum را نشان دهد.

### مرحله 4: support و proof اسکنر را جدا کن

در readiness builder دو متغیر مستقل بساز:

```python
scanner_support_available = _scanner_support_available(audit)
scanner_robustness_proven = False  # فقط گزارش post-training hash-bound می‌تواند تغییر دهد

policy = assess_image_training_policy(
    ...,
    scanner_robustness_proven=scanner_robustness_proven,
)
```

نام output:

```json
{
  "scanner_support_available_for_prespecified_analysis": true,
  "scanner_robustness_proven": false
}
```

فیلد قدیمی `scanner_support_gate_passed` را یا حذف کن یا فقط با alias صریح
`scanner_support_available...` نگه دار؛ هرگز آن را proof ننام. readiness فعلی
هر دو شاخه باید limitation `SCANNER_ROBUSTNESS_NOT_PROVEN` داشته باشد.

### مرحله 5: split invariantها را fail-closed کن

در `audit_membership`:

```python
outer_test_occurrences = Counter()

for outer in outer_folds:
    require(train.isdisjoint(test))
    require(train | test == subject_set)
    outer_test_occurrences.update(test)

    for inner in outer["inner_folds"]:
        require(inner_train.isdisjoint(validation))
        require(inner_train | validation == train)

require(set(outer_test_occurrences) == subject_set)
require(all(count == 1 for count in outer_test_occurrences.values()))
```

هر mismatch باید exception یا hard blocker مشخص بدهد؛ silent intersection نباید
عضو گمشده را پنهان کند. برای split اصلی، ابتدا membership names را به set
تبدیل کن و membership خارج از filtered subset را مجاز بدان، اما partition
filtered subset باید دقیق باشد.

### مرحله 6: readiness را در همان root بازسازی کن

چون input contract semantics تغییر می‌کند، **نباید** artifact قبلی با hash
متفاوت را روی همان root به زور overwrite کنی. دو حالت مجاز:

1. اگر `run_contract.json` موجود با contract جدید mismatch داد: root نسخه‌دار
   جدید `results/method_y_revival_v3_image_v2/` و config نسخه‌دار جدید بساز؛
2. اگر contract byte-identical ماند و فقط report semantics عوض شد، یک report
   generation جدید با input hashهای قبلی بنویس و provenance آن را ثبت کن.

پیشنهاد امن: config جدید `method_y_image_training_v3_1.yaml` با output root
`method_y_revival_v3_image_v2`. artifact v1 حذف/دست‌کاری نشود.

### مرحله 7: GREEN و regression

فرمان‌های جدول بالا را اجرا کن. snapshot مورد انتظار membership نباید تغییر
کند:

```text
amyloid development: 560 subjects / 110 events
amyloid LEARN transport: 335 / 23
tau development: 306 / 75
```

اگر count تغییر کرد، STOP و diff membership را فقط aggregate/hash-level گزارش کن.

## آزمون‌ها

- تست‌های malformed split باید مصنوعی و بدون raw ID/path باشند.
- تست support/proof باید نشان دهد حتی با دو manufacturer کافی،
  `scanner_robustness_proven=false` است تا گزارش post-training معتبر وارد شود.
- تست ADR/claim consistency اضافه کن که policy قدیمی نتواند training را به‌خاطر
  quantitative equivalence ببندد یا claim را بدون آن باز کند.
- از ساختار تست‌های موجود در
  `tests/method_y_revival/test_image_training_policy_v3.py:25-43` و
  `test_image_training_manifest_v3.py:58-75` پیروی کن.

## معیار پایان

- [ ] ADR و addendum جدید وجود و hash دارند.
- [ ] support اسکنر با proof robustness یکی نیست.
- [ ] هر دو شاخه execution-ready و claim-limited باقی می‌مانند.
- [ ] outer-test exact-once membership و inner partition در validator enforce می‌شود.
- [ ] counts بالا بدون تغییر است.
- [ ] هیچ v1/v2/superseded artifact تغییر نکرده است.
- [ ] تست‌های متمرکز و کل Method Y صفر failure دارند.
- [ ] row Plan 009 در README به DONE تغییر کرده است.

## STOP conditions

- writer فعال همین root؛
- count/membership hash غیرمنتظره؛
- split اصلی invariant جدید را fail کند؛
- نیاز به تغییر endpoint، raw data یا config علمی فراتر از dual-gate؛
- هر تلاش برای تبدیل scanner support به claim proof؛
- هر تغییر خارج از scope.

## نگهداری

تا زمانی که Plan 012 گزارش hash-bound scanner sensitivity را نسازد،
`scanner_robustness_proven` باید false بماند. هیچ ایجنت بعدی حق ندارد این boolean
را از روی count یا AUC training تعیین کند.
