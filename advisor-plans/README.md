# بستهٔ تحویل اجرایی BrainIAC-CAFT

این پوشه با مهارت `improve` در تاریخ 2026-08-22 ساخته شده است. مخاطب آن
عامل‌های اجرایی با زمینهٔ صفر، از جمله Gemini و GLM، هستند. فایل‌ها باید به
ترتیب وابستگی خوانده و اجرا شوند؛ هر عامل فقط یک کارت را بر عهده بگیرد.

> مرجع قطعی پروژه فقط `H:\New project` است. این بسته برنامه و مستندات را
> تغییر می‌دهد؛ هیچ آموزش، preprocessing، embedding extraction یا outer-test
> در زمان تولید آن اجرا نشده است.

## حقیقت زنده در زمان برنامه‌ریزی

- Branch: `method-y-post-revival-sanitized-20260812`
- Commit: `54057162efe69556539d4c3e092b9e4871c4f161`
- Worktree: بسیار dirty؛ شمارش زنده `565` entry بود. هیچ `reset`، `clean`،
  branch switch یا commit جمعی مجاز نیست.
- Method X training source digest:
  `b73eef3d67b494986a1e2273a873f5663c874f34588a649bf708a508755a1be9`
- Battery manifest SHA-256:
  `61e0536877ec7ada8d8bcf9d46f77d7f7d121805fa59d2df2124801ca0b9b1a4`
- Retained battery lock SHA-256:
  `cd76c730bc264ac1051d16521f6085775c9d4404427e84d66539e11961be928c`
- **Amendment 2026-08-22T19:12Z:** در 17:06Z باتری به‌اشتباه توسط اپراتور
  restart شد و پروسه فوراً از رفت. baseline جدید preflight از این پس:
  manifest SHA `a568b01de7912adc4ed0cb32ebc5742aab24b4556578e98ea718f218e8664777`
  و lock SHA `7ca81523803af97236cb2323a73216db8c7be5a4b93b4e3e265e50ec7c247d08`.
  جزئیات و رأی اپراتور:
  [addendum work-log](000-20260822-work-log-and-live-state.md). مقادیر بالایی
  فقط برای بازبینی Plan 001 تاریخی‌اند.
- **Amendment 2026-08-25T05:30Z (fix terminal):** باتری پس از اجرای مشروع بعدی به terminal رسید:
  manifest SHA `40a4a009b3a2453fb953e2b2e9b9ec789bdf896741dc2278a898a3b38fdb971d`
  (5/5 effective `COMPLETED`) و `.battery.lock` حذف شد (no active writer؛ terminal). پیش از این
  `43b95fd299...` و lock `b11d5768...` بودند. همچنین `method_x_training_source_contract()` به‌دلیل
  dirty evolution و سازگار‌سازی minimal stubs از `b73eef3d...` به `8c02bd511e61caefe4b679e404049fcae45bec9dc21d3707d68ba43dd5b3bd21`
  تغییر کرد — این baseline جدید برای preflight‌های بعدی معتبر است؛ مقادیر `b73eef`/`61e053`/`cd76c73` فقط تاریخی‌اند.
- Freeze artifact SHA-256:
  `642909c42f1fde290f71425ddbf74793503980eb8ea9762afad0a18a7411a0f4`.
- باتری پنج‌مدلی اکنون **terminal** است (2026-08-24): ResNet10/18/34/DenseNet121/MedNeXt-S `COMPLETED`؛
  ResNet50/EfficientNet `PROTOCOL_EXCLUDED_NOT_STARTED`؛ هیچ writer فعال نیست.
- B0 کامل است. B1 پس از بازبینی مستقل هنوز
  `IN_PROGRESS / REQUEST_CHANGES` است. B2 تا B6 pending هستند.
- آخرین آزمون متمرکز پس از تغییرات نیمه‌تمام anchor:
  `94 passed in 70.22s`. آخرین full named BrainIAC run پیش از تغییرات anchor
  برابر `294 passed, 1497 deselected` بود؛ بنابراین نباید آن را verification
  تازهٔ bytes فعلی معرفی کرد.

شرح کامل شواهد در
[`000-20260822-work-log-and-live-state.md`](000-20260822-work-log-and-live-state.md)
است.

## قواعد مشترک همهٔ عامل‌ها

1. ابتدا فایل کارت را کامل بخوانید؛ سپس drift check و preconditionها را اجرا
   کنید. از روی حافظه یا پیام‌های قبلی عمل نکنید.
2. فقط در `H:\New project` کار کنید. استثنا: checkpoint فعال و موقت CAFT در
   allowlist صریح روی C که در Plan 007 تعریف شده است.
3. قبل از هر write، process/lock را read-only بررسی کنید. `UNKNOWN` و
   `NOT_MEASURABLE` هرگز PASS نیستند.
4. تغییر کد باید با RED test شروع و با GREEN test، regression و گزارش hash
   پایان یابد.
5. این فایل‌های باتری خارج از scope تمام کارت‌های BrainIAC هستند و نباید
   تغییر کنند:

   ```text
   scripts/train_3d_mri.py
   scripts/run_method_x_claim_benchmark.py
   src/models/**
   src/training/method_x_*.py
   results/final_results/method_x_model_battery_v3_depth_and_diversity_20260729/**
   ```

 6. پس از هر کارت، این digest باید دقیقاً ثابت بماند:

   ```powershell
   .\.venv\Scripts\python.exe -c "from scripts.train_3d_mri import method_x_training_source_contract; print(method_x_training_source_contract()[0])"
   ```

   خروجی مورد انتظار (baseline به‌روز 2026-08-25، پس از audit fail-closed):

   ```text
   9b0f90a5a14f8764397c6a15ff8016462f0872e2f827b2ec47fe54ab773871b3
   ```

   تاریخچه: `b73eef…` (فقط Plan 001) ← `8c02bd51…` (بازیابی importها) ←
   `9b0f90a5…` (فعال‌سازی مجدد A6 validate_claim/claim_from_three_d_aggregate،
   fit_source_counts، inspect/ensure_exact_run_contract، amuse_x_hybrid3d backbone).

7. outer loader تا قفل head/calibration در Plan 006 نباید فراخوانی شود.
   هیچ outer artifact موجود بازنویسی یا دوباره ارزیابی نمی‌شود.
8. Method X و CAFT در R1 کاملاً PET-free هستند. predictorها فقط این پنج موردند:

   ```text
   age, sex, education_years, apoe4_carrier, mmse_base
   ```

9. encoderهای BrainIAC و ResNet34 frozen می‌مانند؛ LoRA در R1 همیشه false است.
10. نبود dataset خارجی nonblocking است، اما برچسب خروجی باید
    `ROBUSTNESS_EVIDENCE_NOT_EXTERNAL_VALIDATION` و claim ceiling باید
    `POST_HOC_INTERNAL_TECHNICAL_EVIDENCE` باشد.
11. در failure، عامل حق تغییر seed، split، threshold، arm یا policy را ندارد.
    فقط blocker ماشین‌خوان بنویسد و متوقف شود.
12. هیچ عامل نباید `git reset`, `git clean`, `git checkout --`, branch switch،
    push یا commit جمعی انجام دهد.

## ترتیب اجرا و وضعیت

| Plan | عنوان | اولویت | اندازه | وابستگی | وضعیت |
|---:|---|---|---|---|---|
| 000 | گزارش کار امروز و حقیقت زنده | P0 | S | — | DONE |
| 001 | بستن B1 و anchor مستقل ضد rollback | P0 | L | 000 | DONE |
| 002 | ساخت preprocessing واقعی schema v2 | P0 | L | 001 | BLOCKED_REPAIR_REQUIRED |
| 003 | اجرای resumable preprocessing روی 1669 MRI | P0 | XL/runtime | 002 | NOT_STARTED_PRIMARY_0_OF_1669 |
| 004 | ساخت embedding/split/ResNet34 feature preparation | P0 | XL | 001, 002 contract | DONE |
| 005 | اجرای frozen feature preparation | P0 | L/runtime | 003, 004 | NOT_EXECUTED |
| 006 | ساخت trainer واقعی شش-arm CAFT | P0 | XL | 004 schemas | NOT_EXECUTED_CODE_ONLY_UNACCEPTED_SCAFFOLD |
| 007 | ساخت orchestrator، exact-resume و autosupervision | P0 | XL | 002, 004, 006 | NOT_EXECUTED |
| 008 | acceptance، traceability و readiness نهایی | P0 | L | 003, 005, 006, 007 | TODO |

مقادیر وضعیت مجاز: `TODO`, `IN_PROGRESS`, `DONE`, `BLOCKED: <reason>`,
`REJECTED: <reason>`.

## وابستگی‌ها و امکان هم‌پوشانی

- Plan 001 باید پیش از هر I/O واقعی B2/B3 تمام و مستقل re-review شود.
- Plan 002 کد را می‌سازد؛ Plan 003 اجرای طولانی آن است. این جداسازی برای این
  است که code readiness با completion داده اشتباه نشود.
- Plan 004 را می‌توان پس از تثبیت schema B2 ساخت، اما Plan 005 بدون
  `1669/1669` preprocessing مجاز نیست.
- Plan 006 را می‌توان با fixtureهای syntheticِ غیرعلمی برای آزمون کد ساخت،
  ولی آموزش واقعی فقط بعد از Plan 005 و terminal شدن باتری مجاز است.
- Plan 007 coordinator نهایی است. اجرای source-edit هم‌زمان توسط چند عامل در
  همین dirty worktree ممنوع است؛ عامل‌ها را sequential اجرا کنید.
- Plan 008 می‌تواند readiness نوع
  `READY_FOR_BRAINIAC_TRAINING_PENDING_BATTERY_TERMINAL` صادر کند. authorization
  آموزش واقعی فقط پس از اثبات terminal بودن پنج مدل ساخته می‌شود.

## چیزهایی که عمداً رد شدند

- فعال‌کردن extractor قدیمی cache128: از نظر علمی نامعتبر و hard-disabled است.
- مهاجرت یا overwrite سه receipt قدیمی schema v1: سابقه باید حفظ و نمونه‌ها در
  root نسخه‌دار v2 دوباره پردازش شوند.
- استفاده از JSON خوداظهاری برای readiness: هر نقش B2 تا B5 validator
  byte-bound مخصوص خود می‌خواهد.
- استفاده از outer test برای انتخاب C، early stopping، temperature، conformal
  یا promotion: leakage است.
- ادعای external validation بدون DUA و dataset واقعی: ممنوع است.
- LoRA یا fine-tune encoder در R1: خارج از scope و همواره locked=false.
- شروع BrainIAC هم‌زمان با CNN: guard مشترک باید یکی را fail-closed متوقف کند.

## فایل تسک‌های باقی‌مانده (پس از Plan 003)

برای عامل‌های با توانایی تفکر محدود، لیست گام‌به‌گام همه کارهای باقی‌مانده:

[`remaining-execution-tasks-20260823.md`](remaining-execution-tasks-20260823.md)

شامل: تکمیل اسکریپت‌های runtime (R1)، اجرای Plan 005 (R2)، ساخت orchestrator
Plan 007 (R3)، acceptance Plan 008 (R4)، تصمیم اپراتور split (R0)، تکمیل باتری
(R5)، و اجرای آموزش واقعی CAFT (R6).

## فایل پرامپت جامع

برای سپردن کار به Gemini/GLM، متن
[`GEMINI_GLM_MASTER_PROMPT_FA.md`](GEMINI_GLM_MASTER_PROMPT_FA.md) را بدون
خلاصه‌سازی کپی کنید و فقط مقدار `<ASSIGNED_PLAN>` را با شمارهٔ کارت جایگزین
کنید.

---

# بستهٔ تحویل Method Y image training v3 — 2026-08-24

این بخش مستقل از بستهٔ BrainIAC-CAFT بالاست. برای Method Y فقط فایل‌های این
بخش و Planهای 009–013 را مصرف کنید؛ شماره‌های 001–008 متعلق به Method X هستند.

## حقیقت زندهٔ Method Y image v3

- canonical output: `results/method_y_revival_v3_image_v1/`
- root بدون suffix یعنی `results/method_y_revival_v3_image/` superseded است.
- readiness: `READY_FOR_FINAL_INTERNAL_IMAGE_TRAINING`.
- execution: مجاز؛ positive/external claim: غیرمجاز.
- amyloid development: 560 subject / 110 event؛ LEARN transport: 335 / 23.
- tau development: 306 / 75.
- cacheهای pattern v3، high-level trainer و launcher production هنوز ساخته/
  اجرا نشده‌اند.
- verification تازه: `23 passed` متمرکز؛ `363 passed` کل Method Y.

گزارش کامل:
[`METHOD_Y_IMAGE_V3_TODAY_LOG_20260824.md`](METHOD_Y_IMAGE_V3_TODAY_LOG_20260824.md).

## ترتیب اجرا و وضعیت Method Y image v3

| Plan | عنوان | اولویت | اندازه | وابستگی | وضعیت |
|---:|---|---|---|---|---|
| 009 | dual-gate ADR + readiness hardening | P0 | M | - | VERIFIED_DONE |
| 010 | cache provenance + terminal resume + cache materialization | P0 | L | 009 | VERIFIED_DONE |
| 011 | nested image trainer + inner IBS + exactly-once outer | P0 | XL | 009, 010 | REVISE_REQUIRED (v1 INVALIDATED; repairs committed, v2 pending run) |
| 012 | scanner/calibration/subgroup/bootstrap/claim/privacy | P1 | L | 009-011 | REVISE_REQUIRED (executor wired; real-OOF pass awaits v2) |
| 013 | launcher + GPU smoke + acceptance + production run | P0 | XL | 009-012 | IN_PROGRESS_INVALIDATED_V1 (launcher v3_2/v2 READY; smoke PASSED) |

## وابستگی‌های Method Y

```text
009 → 010 → 011 → 012 → 013
```

- 009 باید تعارض source-of-truth و readiness scanner semantics را حل کند.
- 010 تنها مرحله‌ای است که پیش از trainer cache واقعی را materialize می‌کند.
- 011 trainer را با synthetic integration تست می‌کند، ولی production اجرا
  نمی‌کند.
- 012 sensitivity/reporting را می‌سازد؛ statistics جای external data یا
  quantitative provenance را نمی‌گیرد.
- 013 تنها مرحلهٔ مجاز برای شروع production image training است.

## موارد بررسی و ردشده

- **بازکردن claim صرفاً با اجازهٔ training:** رد؛ execution و claim دو gate
  مستقل‌اند.
- **معادل‌گرفتن دو manufacturer کافی با robustness اثبات‌شده:** رد؛ فقط
  feasibility است.
- **حذف سه scan scaling-invalid یا تعمیر intensity حدسی:** حذف از membership
  حفظ می‌شود، اما raw/artifact حذف یا repair نمی‌شود.
- **image-only به‌عنوان مدل اصلی:** رد؛ comparison اصلی scalar+tabular در برابر
  همان baseline + image است.
- **استفاده از LEARN به‌عنوان external:** رد؛ internal transport است.
- **تغییر split/HPO پس از دیدن outer result:** رد؛ leakage/optional stopping.
- **ساخت فایل‌های handoff روی C/Desktop:** انجام نشد؛ مهارت improve فقط اجازهٔ
  نوشتن در `advisor-plans/` را می‌دهد و همهٔ artifacts پروژه باید روی H بمانند.

## پرامپت Gemini/GLM

[`GEMINI_GLM_METHOD_Y_IMAGE_V3_MASTER_PROMPT_FA.md`](GEMINI_GLM_METHOD_Y_IMAGE_V3_MASTER_PROMPT_FA.md)
را کامل کپی کنید و `<ASSIGNED_PLAN>` را تعیین کنید. manifest hashها:
[`PACKAGE_MANIFEST_METHOD_Y_IMAGE_V3_20260824.md`](PACKAGE_MANIFEST_METHOD_Y_IMAGE_V3_20260824.md).

---

# متمم ناظر BrainIAC پیش از preprocessing — 2026-08-28

این متمم، وضعیت و دستور اجرای فعلی BrainIAC را از روی شواهد زنده بازتعریف
می‌کند و در هر تعارضی بر بخش BrainIAC مورخ 2026-08-22 اولویت دارد. بخش مستقل
Method Y بالا/پایین این فایل با این متمم تغییر نمی‌کند.

## منابع مرجع این دور

- [گزارش ممیزی ناظر](BRAINIAC_PREPROCESSING_SUPERVISOR_AUDIT_20260828.md)
- [ماتریس پذیرش](BRAINIAC_PREPROCESSING_ACCEPTANCE_MATRIX_20260828.md)
- [پرامپت جامع Gemini/GLM](BRAINIAC_PREPROCESSING_REPAIR_MASTER_PROMPT_20260828_FA.md)
- [manifest هش بسته](BRAINIAC_PREPROCESSING_REPAIR_PACKAGE_MANIFEST_20260828.md)

## ترتیب اجرای قطعی و وضعیت

هر عامل فقط یک کارت را، به‌ترتیب زیر و در همان worktree اجرا کند. آغاز کارت
بعدی بدون artifact پذیرش کارت قبلی ممنوع است.

| Plan | عنوان | وابستگی | وضعیت فعلی |
|---:|---|---|---|
| 014 | freeze زنده، کشف همهٔ writerها و reconciliation | — | DONE |
| 015 | بستن Git/dependency closure | 014 | DONE |
| 016 | سخت‌سازی execution profile، signal و readiness builder | 015 | TODO |
| 017 | پذیرش واقعی preprocessing بدون primary run | 014–016 | TODO |
| 018 | بازسازی traceability و صدور readiness preprocessing | 017 | TODO |
| 019 | اجرای primary روی 1669 MRI پس از readiness | 018 | BLOCKED_PENDING_READINESS |

## supersession و مرز اجرا

- Plan 002 از نظر قرارداد اولیه حفظ می‌شود، اما پذیرش و repair پیاده‌سازی آن
  فقط با Planهای 014 تا 018 تعیین می‌شود: `SUPERSEDED_FOR_REPAIR_BY_014_018`.
- بخش runtime در Plan 003 با Plan 019 جایگزین شده است؛ فرمان قدیمی Plan 003
  پیش از PASS شدن Plan 018 اجرا نشود.
- Planهای 005 تا 008 هنوز اجرا یا پذیرفته نشده‌اند. Plan 006 فقط scaffold
  آزمایش‌نشده است و عنوان «trainer آماده» ندارد.
- هیچ‌یک از کارت‌های 014 تا 018 اجازهٔ اجرای primary 1669، extraction نهایی،
  CAFT training، outer evaluation یا LoRA را نمی‌دهد.
- مسیر authoritative همین پوشه روی `H:\New project` است؛ نسخهٔ Desktop ساخته
  نشده است.

وضعیت شروع این دور: `NOT_READY_FOR_PRIMARY_PREPROCESSING`. صدور
`READY_TO_START_OR_RESUME_BRAINIAC_PREPROCESSING.json` فقط در پایان Plan 018
و پس از PASS ماشین‌خوان همهٔ gateهای 014 تا 017 مجاز است.
