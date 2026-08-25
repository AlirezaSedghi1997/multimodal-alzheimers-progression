# نمایه‌ی حافظه‌ی پروژه

برای ادامه‌ی کار، **ابتدا دفتر ثبت نشست زنده** را بخوانید، نه snapshot تیرماه synthesis.

## زنده‌ترین — ۲۰۲۶-۰۸-۲۵ (Method Y image v3 — Plans 009-013 + hard audit)

0. شاخه تصویری Method Y v3 اکنون **READY_FOR_FINAL_INTERNAL_IMAGE_TRAINING** (`results/method_y_revival_v3_image_v1/readiness/FINAL_IMAGE_TRAINING_READINESS.json:6` `19baa31d0acfbb15e8bcc11828bae8bee2b9e0ed3b1beea06f7a1119a74979c5`)؛ `training_execution_allowed true`، `positive_claim_allowed false`، `external_claim_allowed false`.
1. Cacheهای pattern v3 کامل و provenance-bound روی H: `caches/method_y/revival_v3_image/amyloid_pattern_128_v1.h5` 895 گروه (31981B33...), `tau_pattern_128_v1.h5` 306 گروه (4095FCBC...), هر گروه `COMPLETE` + `finite_fraction 1.0` + `float16` + `source_sha256` hard-checked (`src/method_y_revival/image_cache_v3.py:76` strict).
2. Trainer تو در تو: `src/method_y_revival/image_modeling_v3.py:24` `survival_at_times` و `src/method_y_revival/image_training_orchestrator_v3.py:12` با انتخاب winner فقط `mean_inner_IBS` + 1-SE + `complexity_rank`، outer exactly-once via `OuterFoldController:48`.
3. Robustness/reporting: `src/method_y_revival/image_robustness_v3.py:12` و `image_reporting_v3.py:12` — scanner support ≠ robustness proof، balanced weights از train-only، leave-manufacturer-out بدون pooling، privacy scan PASS.
4. Launcher واحد: `scripts/run_method_y_image_training_v3.ps1:1` و `scripts/run_method_y_image_training_v3.py:118` با preflight/smoke/run، `WriterLock`, H-only temp, GPU smoke PASSED (cuda 26 MiB), `435 passed` در `tests/method_y_revival`.
5. Hard audit 2026-08-25T13:30Z: `advisor-plans/agent-reports/method-y-image-v3-hard-audit-20260825T133000Z.md` — سه رفع (strict HDF5 provenance, source_artifact binding, true IBS, smoke)؛ همه تست‌ها سبز.

## زنده — ۲۰۲۶-۰۸-۲۵ (BrainIAC-CAFT governance drift fix و باتری ترمینال)

0. باتری پنج‌مدلی اکنون **TERMINAL** (`40a4a009b3a2453fb953e2b2e9b9ec789bdf896741dc2278a898a3b38fdb971d`، 5/5 `COMPLETED`، `.battery.lock` ABSENT) — `src/research/brainiac_battery_terminal.py: verify_battery_terminal` `terminal:true`.
1. `method_x_training_source_contract()` جدید `8c02bd511e61caefe4b679e404049fcae45bec9dc21d3707d68ba43dd5b3bd21` (پس از سازگار‌سازی `src/evaluation/calibration.py:49`, `cohort_probe.py:128`, `backbone_factory.py:6` و `scripts/train_3d_mri.py:57`).
2. Governance fix 4 مرحله‌ای 2026-08-25 (predecessor `e3dd7aaf`/`6b313303`, registry `1c45b3f4`, vendor CRLF→LF `c26b42ba`) — `scripts/research/prepare_brainiac_raw_t1_v2.py --mode preflight` اکنون `PASS` و `PREPARE_RAW_T1 allowed:true`.
3. دستور واحد P003 (resumable, Brainiac venv): `& 'H:\New project\.venvs\brainiac\Scripts\python.exe' 'H:\New project\scripts\research\prepare_brainiac_raw_t1_v2.py' --mode full --device cuda` — اجرای مجدد همان دستور از last commit ادامه می‌دهد (exit 75).
4. مستندات به‌روز: `advisor-plans/README.md` و `advisor-plans/000-20260822-work-log-and-live-state.md` Addendum 3، `docs/plans/method_x_project_roadmap.md` و `results/method_x_registry/project_roadmap.json` به TERMINAL.

## زنده‌ترین — ۲۰۲۶-۰۸-۲۴ (Method Y تکمیل داخلی با ادعاهای محدود)

0. رسید نهایی ماشینی: `results/method_y_revival_v2/reports/POSTTRAINING_READINESS_V2.json` — وضعیت `LOCAL_INTERNAL_PIPELINE_COMPLETED_CLAIMS_RESTRICTED`، بازبینی مستقل انسانی لازم نیست.
1. گزارش نهایی قابل‌خواندن: `docs/METHOD_Y_V2_FINAL_INTERNAL_STATUS_20260824.md`.
2. قرارداد کامل اجرا و resume: `docs/METHOD_Y_V2_POSTTRAINING_EXECUTION_CONTRACT.md`.
3. رجیستری artifact: `results/method_y_revival_v2/artifact_registry_v2.json`.
4. بسته عمومی aggregate: `results/method_y_revival_v2/public_bundle/`؛ privacy status باید `PASS` بماند.
5. تفسیر: source-regional کامل است؛ image branches اختیاری و مسدود؛ external primary اختیاری و در داده‌های کاملاً باز یافت نشد؛ DLBS فقط replication ثانویه اکتشافی است.

## زنده‌ترین — ۲۰۲۶-۰۸-۲۲ (نشست نظارت ZCode)

0. **حافظه اجرای پس از آموزش (معتبرترین صف POST):** `results/method_x_registry/post_training_execution_memory_20260822.json` (schema v3؛ کارت‌های D02/MX-03/R1a/R1b/R1c/MX-04/MX-05/MX-06/T08/MX-07 + قرارداد watchdog؛ پایش خودکار ۲ساعته فعال)
1. وضعیت BrainIAC: `results/method_x_registry/BRAINIAC_P08_STATUS_20260822.json` (R0 = COMPLETE با ۲۸/۲۹ مرور فول‌تکست؛ اجرای R1 همچنان HARD_DISABLED با ۶ blocker) و `research/brainiac_caft_20260808/evidence_registry_v3.json`
2. چک‌لیست جاری: `tasks/method_x_completion_20260821_todo.md` (P08b تکمیل شد؛ D01 زنده؛ D02 در انتظار ترمینال MedNeXt)
3. دفتر کار Method X (append-only): `docs/plans/method_x_work_log.md` — ورودی کامل ۲۰۲۶-۰۸-۲۲ بعدازظهر در انتها

## زنده — ۲۰۲۶-۰۸-۱۵

1. دفتر ثبت نشست (Method Y repair + Method X untouched): `docs/plans/agent_session_ledger_20260815.md`
2. همزاد ماشینی Method Y: `results/method_y_revival_v2/journals/session_ledger_20260815.json`
3. یادداشت انزوای Method X: `results/method_x_registry/SESSION_NOTE_METHOD_Y_DEFECT_REPAIR_20260815.json`
4. آمادگی زنده Method Y: `results/method_y_revival_v2/readiness/PRETRAINING_READINESS_V2.json` (`training_allowed=false`؛ Task 4 الان `COMMITTED` است ولی این فایل رسید اجرای A4 نیست؛ دود مصنوعی `SYNTHETIC_SMOKE_ONLY` در ۲۰۲۶-۰۸-۱۶)
5. کار باقی‌مانده Method Y: `docs/plans/method_y_remaining_work_20260815.md`
6. نقشه زنده Method X: `docs/plans/method_x_project_roadmap.md` و `results/method_x_registry/project_roadmap.json`
7. کار باقی‌مانده Method X: `docs/plans/method_x_remaining_work_20260815.md`
8. دفتر کار Method Y (append-only): `docs/plans/method_y_work_log.md`
9. دفتر کار Method X (append-only): `docs/plans/method_x_work_log.md`

## برنامه و قفل‌های علمی (تاریخچه، نه وضعیت GPU)

10. منبع کامل برنامه: `docs/PROGRAM_DEFAULT_SOURCE_OF_TRUTH_2026-07-19.md`
11. ADR تصمیم پیش‌فرض: `docs/decisions/ADR-2026-07-19-default-method-xy-program.md`
12. تصمیم اجرایی synthesis: `supervisor_synthesis_5agents/08_final_outputs/00_FINAL_EXECUTIVE_DECISION.md`
13. مشخصات کامل ۴۱‌بندی: `supervisor_synthesis_5agents/08_final_outputs/07_SELECTED_PROJECT_SPECIFICATION.md`
14. قرارداد transform: `results/method_y_revival_v2/transforms/eligibility_lock.json` و `results/method_y_revival_v2/transforms/qc_audit.json`

## کهنه — اولویت ندهید

- `supervisor_synthesis_5agents/08_final_outputs/01_CURRENT_PROJECT_STATE.md` — snapshot ۲۰۲۶-۰۷-۱۸؛ HEAD آن `480752d` است.
- `docs/method_x_live_progress_report.md` — هنوز مدل‌ها را `NOT_STARTED` می‌نویسد؛ با باتری زنده در تضاد است.

این نمایه جایگزین فایل‌های کامل نیست. اگر بین این نمایه، گزارش کهنه، و artifact دارای hash/contract اختلاف بود، artifact زنده و دفتر ثبت نشست ۲۰۲۶-۰۸-۱۵ بر snapshot تیرماه مقدم است.
