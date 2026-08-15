# نمایه‌ی حافظه‌ی پروژه

برای ادامه‌ی کار، **ابتدا دفتر ثبت نشست زنده** را بخوانید، نه snapshot تیرماه synthesis.

## زنده — ۲۰۲۶-۰۸-۱۵

1. دفتر ثبت نشست (Method Y repair + Method X untouched): `docs/plans/agent_session_ledger_20260815.md`
2. همزاد ماشینی Method Y: `results/method_y_revival_v2/journals/session_ledger_20260815.json`
3. یادداشت انزوای Method X: `results/method_x_registry/SESSION_NOTE_METHOD_Y_DEFECT_REPAIR_20260815.json`
4. آمادگی زنده Method Y: `results/method_y_revival_v2/readiness/PRETRAINING_READINESS_V2.json` (`training_allowed=false`؛ Task 4 هنوز `COMMITTED` نیست)
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
