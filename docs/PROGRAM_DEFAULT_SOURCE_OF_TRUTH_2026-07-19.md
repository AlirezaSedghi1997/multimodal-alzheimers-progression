# برنامه‌ی پیش‌فرض و منبع حقیقت پروژه — Method X / Method Y

**نسخه:** `2026-07-19`  
**وضعیت:** `ACCEPTED / DEFAULT_EXECUTION_PROGRAM`  
**مالک مسیر اجرایی:** `H:\New project`  
**داده‌ی خام:** `H:\A4 Learn` (فقط خواندنی)  
**مرجع کامل تصمیم‌گیری:** `H:\New project\supervisor_synthesis_5agents\08_final_outputs\`

این سند خلاصه‌ی علمی نیست؛ قرارداد هدایت اجرای پروژه است. هر عامل یا اجرای بعدی باید این سند و فایل‌های مرجع درج‌شده در آن را بخواند و هیچ تصمیمی را بر اساس شمارش، گزارش یا state قدیمیِ خارج از این منابع اتخاذ نکند.

## ۱. ترتیب اجباری مطالعه و منبع حقیقت

پیش از هر تغییر، این منابع به ترتیب زیر خوانده و hash آن‌ها در snapshot ثبت می‌شود:

1. پرامپت نظارتی اصلی: `C:\Users\USER\Desktop\final_codex_supervisory_prompt_fa.md`
2. وضعیت زنده مخزن و processها: `H:\New project\supervisor_synthesis_5agents\08_final_outputs\01_CURRENT_PROJECT_STATE.md` و `H:\New project\supervisor_synthesis_5agents\05_project_audit\audit_snapshot.json`
3. خروجی قبلی Codex: `H:\New project\research_outputs\early_ad_innovation_20260718\`
4. خروجی‌های عامل‌ها: `C:\Users\USER\Desktop\خروجی ایجنت ها\`؛ فایل‌های duplicate/raw عامل ۴ فقط برای coverage هستند و شاهد مستقل محسوب نمی‌شوند.
5. تصمیم نهایی و مشخصات طرح منتخب:
   - `H:\New project\supervisor_synthesis_5agents\08_final_outputs\00_FINAL_EXECUTIVE_DECISION.md`
   - `H:\New project\supervisor_synthesis_5agents\08_final_outputs\07_SELECTED_PROJECT_SPECIFICATION.md`

فهرست کامل ۱۶ خروجی الزامی در `H:\New project\supervisor_synthesis_5agents\08_final_outputs\` قرار دارد و باید برای اجرای هر مرحله به‌عنوان قرارداد خوانده شود:

```text
00_FINAL_EXECUTIVE_DECISION.md
01_CURRENT_PROJECT_STATE.md
02_SOURCE_BY_SOURCE_EVALUATION.md
03_CONSENSUS_AND_CONTRADICTIONS.md
04_VERIFIED_EVIDENCE_SUMMARY.md
05_DATASET_AND_ENDPOINT_GATE.md
06_CANDIDATE_SCORECARD.csv
07_SELECTED_PROJECT_SPECIFICATION.md
08_CURRENT_REPO_IMPLEMENTATION_MAP.md
09_SIX_WEEK_EXECUTION_PLAN.md
10_EXPERIMENT_AND_ABLATION_PROTOCOL.md
11_LEAKAGE_CALIBRATION_AND_STATISTICS_CHECKLIST.md
12_CLAIM_BOUNDARIES.md
13_REJECTION_LOG.md
14_FALLBACK_AND_STOP_RULES.md
15_FINAL_RESPONSE_TO_USER.md
```

## ۲. تصمیم اجرایی قفل‌شده

مسیر اصلی `continuation_prespecified_method_y` است؛ Practical Continuation Score ثبت‌شده ۸۴٫۰ است. پروژه به‌سمت pivot پیچیده، raw 3D، MoE یا external-dependent تغییر نمی‌کند مگر آن‌که یک نسخه‌ی جدید این تصمیم را با evidence و ADR جدید supersede کند.

هسته‌ی نوآوری قابل‌دفاع، «پروتکل matched و leakage-closed برای ارزش افزوده‌ی PET نسبت به clinical+cognition، با outer-test دقیقاً یک‌بار» است. این عبارت novelty مکانیزمی یا causal نیست. discordance زیستی–عملکردی فقط extension مشروط است و از ابتدا نوآوری مفروض نیست.

تقسیم داده و ادعا:

- `A4`: development اصلی Method Y.
- `LEARN`: internal transport؛ external validation محسوب نمی‌شود.
- `ADNI/SCAN/OASIS-3`: فقط پس از DUA، mapping lock و ارزیابی بدون refit.
- endpoint محلی: `future sustained CDR progression`؛ `PACC decline` فقط secondary با قرارداد future-only.
- MCI→AD، تشخیص آلزایمر، causal effect و external generalization در داده‌ی محلی ممنوع.
- Method X PET-free باقی می‌ماند؛ confounding بین cohort و label همچنان مانع claim بالینی است.

Fallback قفل‌شده: اگر هر gate علمی/فنی شکست بخورد، نتیجه‌ی منفی/اکتشافی فعلی تکمیل و گزارش می‌شود؛ شواهد منفی حذف یا بازتفسیر مثبت نمی‌شوند.

## ۳. وضعیت و حفاظت از transform جاری Method Y

خروجی و cacheهای v1 هرگز بازنویسی یا حذف نمی‌شوند. مسیر transform جاری:

```text
H:\New project\results\method_y_revival_v2\transforms\
H:\New project\scripts\materialize_method_y_transforms_v2.py
H:\New project\scripts\run_method_y_transforms_manual.ps1
H:\New project\scripts\audit_method_y_transform_failures_v2.py
H:\New project\scripts\audit_method_y_transform_qc_v2.py
H:\New project\scripts\lock_method_y_transform_eligibility_v2.py
H:\New project\src\method_y_revival\transform_eligibility.py
```

سه failure با `T1_INTENSITY_SIGNAL_GATE_FAILED` به‌دلیل scaling/intercept نامعتبر منبع T1 ثبت شده‌اند. repair عددی یا حدس intercept ممنوع است. این سه مورد فقط با reason ثبت‌شده‌ی `SOURCE_T1_INVALID_NUMERIC_SCALING` از downstream membership کنار گذاشته می‌شوند؛ raw data تغییر نمی‌کند و انتخاب outcome-independent است. در صورت وجود lock معتبر، اجرای دوباره باید از آخرین scan/manifest commit ادامه یابد و QC خواندنی reuse شود.

قرارداد transform eligibility:

```text
source manifest: 1317
downstream eligible after approved source-QC exclusions: 1314
excluded: 3
automatic intensity repair: false
raw data mutation: false
```

این اعداد فقط snapshot هستند و باید قبل از گزارش نهایی دوباره از `eligibility_lock.json` و `qc_audit.json` خوانده شوند.

## ۴. state machine و resume

ممیزی پنج‌ایجنت:

```text
INIT
→ SAFETY_SNAPSHOTTED
→ INVENTORY_LOCKED
→ UNIQUE_CONTENT_READ
→ REGISTRY_LOCKED
→ PROJECT_AUDITED
→ DECISIVE_EVIDENCE_VERIFIED
→ CANDIDATES_SCORED
→ RED_TEAM_DONE
→ OUTPUTS_RENDERED
→ VERIFIED
```

آماده‌سازی Method Y:

```text
INIT → PREFLIGHT_DONE → ASSETS_LOCKED → LINKAGE_LOCKED
→ TRANSFORMS_VALIDATED → QUANTITATIVE_PET_VALIDATED
→ ENDPOINTS_LOCKED → SPLITS_LOCKED → INNER_SELECTION_DONE
→ MODELS_LOCKED → OUTER_EVAL_COMMITTED
→ EXTERNAL_EVAL_COMMITTED → XAI_VALIDATED → REPORTS_FINALIZED
```

هر stage فقط پس از hash verification stage قبلی شروع می‌شود. state و artifact به‌صورت atomic نوشته می‌شوند. اگر artifact قبل از state نوشته شده باشد، resume باید hash را recover کند و از اجرای دوباره outer-test جلوگیری کند.

برای هر run contract این موارد immutable هستند: commit، dirty digest، config hash، داده/manifest hash، split digest، seed، fold count، architecture، loss، optimizer، scheduler، augmentation، loader و checkpoint contract. تغییر هرکدام با همان output root باید `RUN_CONTRACT_MISMATCH` بدهد.

## ۵. مراحل اجرایی و فایل‌های لازم

### مرحله ۰ — safety/provenance

بررسی کن:

```text
H:\New project\supervisor_synthesis_5agents\tools\run_synthesis.py
H:\New project\supervisor_synthesis_5agents\tools\audit_project_state.py
H:\New project\supervisor_synthesis_5agents\00_admin\T0.json
H:\New project\supervisor_synthesis_5agents\00_admin\Tfinal.json
H:\New project\supervisor_synthesis_5agents\00_admin\run_contract.json
```

پیش از هر write روی MRI/PET/training/report:

```powershell
Get-CimInstance Win32_Process |
  Where-Object { $_.Name -match 'python|powershell|matlab|spm' } |
  Select-Object ProcessId,ParentProcessId,Name,CommandLine
```

writer فعال هرگز فقط بر اساس نام process متوقف نمی‌شود؛ اگر workspace ممیزی را هدف نگرفته است، فقط snapshot read-only گرفته می‌شود.

### مرحله ۱ — inventory/linkage/manifest

بررسی کن:

```text
H:\New project\supervisor_synthesis_5agents\tools\inventory.py
H:\New project\src\method_y_revival\metadata_linkage.py
H:\New project\results\method_y_revival_v2\manifests\
```

FBP فقط amyloid و FTP فقط tau است. linkage بر اساس UID/subject+visit/date دقیق انجام می‌شود؛ sidecar مبهم، nearest-date بدون حد ۱۸۳ روز، یا subject-only metadata پذیرفته نیست. duplicateها در سطح byte/voxel/semantic ثبت می‌شوند ولی subject مستقل شمرده نمی‌شوند. manifest بعد از hash lock immutable است.

### مرحله ۲ — PET geometry/QC

بررسی کن:

```text
H:\New project\src\method_y_revival\pet_geometry.py
H:\New project\src\method_y_revival\pet_preprocessing.py
H:\New project\src\method_y_revival\pet_qc.py
H:\New project\scripts\audit_method_y_transform_qc_v2.py
H:\New project\results\method_y_revival_v2\qc\
H:\New project\results\method_y_revival_v2\provenance\assets\
```

PET باید با زنجیره‌ی مستند `PET→T1→MNI` و template قفل‌شده پردازش شود؛ shape-only zoom ممنوع. FBP/FTP quantitative scale حفظ می‌شود، per-scan z-score برای claim-bearing ممنوع است، NaN/Inf داخل brain failure است و nonfinite background فقط با ثبت aggregate مجاز است. visual QC باید agent multi-pass باشد و human sign-off شرط completion نیست.

### مرحله ۳ — quantitative PET gate

بررسی کن:

```text
H:\New project\scripts\validate_method_y_quantitative_pet_v2.py
H:\New project\results\method_y_revival_v2\provenance\quantitative_pet_feature_validation_v2.json
H:\New project\results\method_y_revival_v2\provenance\quantitative_pet_feature_provenance_reference.json
H:\New project\results\method_y_revival_v2\provenance\roi_mapping_v2.json
```

Centiloid/CenTauR فقط با asset رسمی، license، space، label-table hash و calibration مستقل claim-bearing می‌شوند. cache قدیمی SUVR-preserving مرجع فنی است و `centiloid`/`amyloid_pos` مشتق از همان PET هرگز target مدل نیست. ROI سفارشی با fuzzy string mapping وارد claim نمی‌شود. تا وقتی `CENTILOID_CALIBRATION_REFERENCE_NOT_PROVIDED` یا `SCAN_LEVEL_QUANTITATIVE_VALIDATION_NOT_RECOMPUTED` برقرار است، آموزش/ادعا شروع نمی‌شود.

### مرحله ۴ — endpoint

بررسی کن:

```text
H:\New project\src\method_y_revival\endpoints.py
H:\New project\results\method_y_revival_v2\endpoints\
H:\New project\supervisor_synthesis_5agents\10_pre_transform_preparation_v1\02_contracts\endpoint_contract.json
```

baseline CDR باید صفر و قبل/هم‌زمان PET باشد. event فقط اولین worsening تأییدشده با follow-up بعدی است؛ last visit بدون confirmation event نیست. PACC فقط secondary و future-only است. outcome-derived imputation، circular baseline variables و post-origin PET ممنوع‌اند.

### مرحله ۵ — nested split/HPO/model

بررسی کن:

```text
H:\New project\src\method_y_revival\splits.py
H:\New project\src\method_y_revival\survival_models.py
H:\New project\src\method_y_revival\evaluation.py
H:\New project\supervisor_synthesis_5agents\08_final_outputs\10_EXPERIMENT_AND_ABLATION_PROTOCOL.md
H:\New project\supervisor_synthesis_5agents\08_final_outputs\11_LEAKAGE_CALIBRATION_AND_STATISTICS_CHECKLIST.md
```

split subject-grouped پنج outer و سه inner است. scaler/imputer/harmonizer/feature selection داخل fold fit می‌شود. CoxNet مدل اصلی و RSF/GBSA challenger با بودجه برابرند. winner فقط inner IBS و one-standard-error انتخاب می‌شود. outer-test دقیقاً یک‌بار، با censoring train-only، support از پیش تعیین‌شده و calibration/subgroup/scanner reports ارزیابی می‌شود. paired A/T با ۹۱ event فقط در صورت feasibility/power قفل‌شده extension است.

### مرحله ۶ — external validation

بررسی کن:

```text
H:\New project\configs\method_y_external_mapping_v1.yaml
H:\New project\src\method_y_revival\external_validation.py
H:\New project\results\method_y_revival_v2\external_validation\
```

ADNI/SCAN/OASIS-3 تا DUA، mapping lock و feature intersection lock موجود نیستند. external evaluation بدون refit/reselection/harmonization outcome-based است. نبود DUA مانع synthesis نیست ولی external claim را ممنوع می‌کند.

### مرحله ۷ — Method X و XAI

بررسی کن:

```text
H:\New project\scripts\run_method_x_mri_5fold.ps1
H:\New project\scripts\orchestrate_method_x_mri_5fold.py
H:\New project\scripts\spm_completion_guard.py
H:\New project\data\cached\t1_volume_cache_128.h5
H:\New project\results\method_x_5fold_corrected_v1\
H:\New project\supervisor_synthesis_5agents\10_pre_transform_preparation_v1\02_contracts\method_x_matched_benchmark_template.json
H:\New project\src\evaluation\
```

Method X PET-free است، اما Cramér’s V cohort/label برابر ۱ گزارش شده است؛ تا توسعه در cohort دارای variation درون-cohort، matched backbone benchmark و strict model/XAI contract انجام نشود، فقط technical benchmark است و claim تشخیصی/بالینی مجاز نیست. XAI باید checkpoint contract را strict load کند، synthetic atlas و `strict=False` ممنوع است، و attribution فقط با fidelity و randomized-model sanity منتشر می‌شود.

## ۶. گیت‌ها و claim policy

```text
artifact_status == COMPLETED
  به‌تنهایی claim را مجاز نمی‌کند.

science_gate_status != PASS
  → claim_status نمی‌تواند EXTERNALLY_VALIDATED باشد.

outer_folds_completed != required
  → overall status FAILED/INCOMPLETE؛ میانگین‌گیری از foldهای باقی‌مانده ممنوع.

scanner shortcut، leakage، invalid geometry، same-PET target،
missing calibration، ambiguous linkage یا missing DUA
  → claim مثبت را متوقف می‌کند.
```

claimهای مجاز:

- اگر gate داخلی پاس شود: «baseline PET burden در محیط داخلی A4 اطلاعات پیش‌آگهی افزوده ارائه کرد.»
- اگر gate شکست بخورد: «ارزش افزوده‌ی PET تحت پروتکل از پیش‌تعیین‌شده اثبات نشد؛ نتیجه exploratory/technical است.»

claimهای ممنوع: تشخیص آلزایمر، MCI→AD محلی، causal effect، external generalization و «ناحیه‌ای که آلزایمر از آن تشخیص داده شد».

## ۷. تست‌ها و معیار تحویل

حداقل تست‌ها:

```text
H:\New project\supervisor_synthesis_5agents\tools\tests\
H:\New project\tests\method_y_revival\test_transform_eligibility_v2.py
H:\New project\tests\method_y_revival\test_transform_qc_audit_v2.py
H:\New project\tests\test_method_y_quantitative_validator_v2.py
H:\New project\tests\method_y_revival\test_transform_materializer_v2.py
```

فرمان‌های پایه:

```powershell
Set-Location 'H:\New project'
& '.\.venv\Scripts\python.exe' -m pytest -q
& '.\.venv\Scripts\python.exe' 'supervisor_synthesis_5agents\tools\run_synthesis.py' verify --workspace 'H:\New project\supervisor_synthesis_5agents'
```

تحویل فقط وقتی معتبر است که canonical inputs read-complete باشند، duplicateها دوباره evidence نشوند، claimهای تصمیم‌ساز source/line/evidence-level داشته باشند، candidate با hard FAIL برنده نشود، ۱۶ خروجی و database و CSVها وجود داشته باشند، privacy scan پاس باشد، protected paths نسبت به T0 تغییر نکرده باشند، و resume در هر stage از آخرین commit ادامه دهد.

## ۸. قانون تغییر این برنامه

این فایل و ADR متناظر منبع پیش‌فرض تصمیم‌گیری هستند. تغییر اساسی فقط با:

1. ساخت نسخه‌ی جدید در `docs/decisions/`؛
2. ذکر فایل/گیت/دلیل supersede؛
3. حفظ این سند بدون حذف؛
4. اجرای consistency guard و ثبت snapshot جدید؛

مجاز است. خروجی‌های v1، داده‌ی خام، cacheهای canonical، transformهای جاری و artifactهای منفی تاریخی حذف یا overwrite نمی‌شوند.

## ۹. رجوع سریع به برنامه‌ی شش‌هفته‌ای

جزئیات زمان‌بندی و stop rule در این فایل‌هاست و باید عیناً خوانده شود:

```text
H:\New project\supervisor_synthesis_5agents\08_final_outputs\09_SIX_WEEK_EXECUTION_PLAN.md
H:\New project\supervisor_synthesis_5agents\08_final_outputs\14_FALLBACK_AND_STOP_RULES.md
H:\New project\supervisor_synthesis_5agents\08_final_outputs\12_CLAIM_BOUNDARIES.md
```

## ۱۰. پیوست ۲۰۲۶-۰۸-۲۴: تفکیک Dual-Gate برای تحلیل اکتشافی داخلی تصویر Method Y v3

مطابق با تصمیم قفل‌شده در [ADR-2026-08-24](decisions/ADR-2026-08-24-method-y-image-dual-gate.md)، خطوط مربوط به توقف آموزش در غیاب کالیبراسیون رسمی Centiloid/CenTauR (مانند بخش ۳، خط ۱۸۳) برای شاخهٔ ثانویه و اکتشافی تصویر Method Y به دو گیت مستقل تفکیک می‌گردند:
- `training_execution_allowed`: نیازمند سلامت فنی داده، عدم نشت فردی، هندسهٔ معتبر، مقادیر متناهی، ردیاب صحیح و کفایت رخداد.
- `positive_claim_allowed` و `external_claim_allowed`: همچنان نیازمند اثبات هم‌ارزی کمی، تحلیل استحکام اسکنر و داده‌های خارجی مستقل.
متن تاریخی فوق حفظ شده و این پیوست چارچوب ارزیابی مدل‌های تصویری v3 را رسمیت می‌بخشد.


