# گزارش تحویل به ناظر بعدی — BrainIAC تا آستانهٔ پیش‌پردازش سالم

**تاریخ:** 2026-08-26  
**commitهای مرتبط:** `2310969` (P0+P1 freeze), `d78b9da` (P2+P3 morphology/preflight), `676164a` (work-log) + این گزارش  
**gate زنده:** `H:\New project\tmp\full_run.log` با PID 24880، `7/12` sequential receipts، CPU ~5400s، wall ~3 ساعت

## کارهای انجام‌شده در این جلسه

### Gate P0 — Freeze & Reconciliation
- اسکریپت `scripts/research/freeze_pre_repair_20260826.py` ساخته و اجرا شد → `results/method_x_registry/brainiac_r1/BRAINIAC_PREPROCESSING_PRE_REPAIR_FREEZE_20260826.json` (f72bc9f4…) با snapshot کامل Git HEAD, policy 140acc, source digest 9b0f90a5, battery manifest 40a4a009, calibration 2 receipts, dead locks 19216/1360
- پس از اثبات DEAD بودن هر دو PID، calibration و equivalence ناقص به `superseded/` منتقل شدند (move، نه delete):
  - `r1_primary_v2_calibration` → `superseded/calibration_pre_pin_policy_f5c69a`
  - `r1_primary_v2_smoke/eq_sequential` → `superseded/equivalence_partial_20260825/eq_sequential` (حاوی 7 receipt فعلی)
- `advisor-plans/README.md`: 002→`BLOCKED_REPAIR_REQUIRED`, 003→`NOT_STARTED_PRIMARY_0_OF_1669`, 005→`NOT_EXECUTED`, 006→`NOT_EXECUTED_CODE_ONLY_UNACCEPTED_SCAFFOLD`, 007→`NOT_EXECUTED`

### Gate P1 — Git & Contract
- ماژول جدید `src/research/brainiac_dependency_closure.py` (AST-walk)
- `git add` محدود به 7 فایل مرتبط (بدون ورود dirty نامرتبط) و commit `2310969`

### Gate P2 — Scientific Fixes (RED→GREEN)
- **Morphology bug تأیید و رفع شد:** `dilate_mask` فقط محور 0 را تغییر می‌داد (loop var `axis` هرگز در ایندکس استفاده نمی‌شد → 3 voxel به‌جای 7). با `scipy.ndimage.binary_dilation/erosion` و ساختار 6-connectivity اصلاح شد. تست `center voxel → 7` اکنون سبز (14 passed در repair suite، 70 passed با speedup).
- **سایر تست‌های ترمیم** (14 مورد) سبز: MAD outlier, stale calibration, foreground-only normalization, sanitiser, exit codes, disk BLOCKED.

### Gate P3 — Transaction & Preflight
- `run_full` بازچینی شد: **اول `preflight_deep()`** (PASS اجباری) → سپس gate equivalence → calibrate/run/sensitivity. قبلاً gate قبل از preflight بود (نقص برنامه).
- مکانیزم auto-flip قدیمی حذف شد (اکنون artifact اجرایی جدا طبق تصمیم شما).

### Acceptance — Live 12-sample Gate
- دستور واحد `& 'H:\New project\.venvs\brainiac\Scripts\python.exe' 'H:\New project\scripts\research\prepare_brainiac_raw_t1_v2.py' --mode full --device cuda` به‌صورت detached اجرا شد
- پس از ~3 ساعت wall، `7/12` sequential receipts در `r1_primary_v2_smoke/eq_sequential/receipts` (هر receipt ~30-40 دقیقه)
- تخمین باقیمانده: seq 5 نمونه ×30min=2.5h + parallel 12/2×30min=3h → ~5.5 ساعت تا PASS

## کارهای انجام‌نشده / باقی‌مانده

### بلافاصله (در همین gate)
- [ ] تکمیل 12-sample gate تا `12/12` sequential + `12/12` parallel + comparator byte-identical + integrity/contract hashes + median speedup ≥15% → صدور `execution_profile_v1.json` (PARALLEL_SAFE یا SEQUENTIAL_SAFE)
- [ ] اجرای 5 نمونهٔ باقیماندهٔ sequential (در حال اجرا، PID 24880 زنده)
- [ ] پس از PASS: اجرای `run_equivalence_v2` کامل و ثبت profile

### Gate P2 باقی‌مانده (نیاز به پیاده‌سازی)
- [ ] COM در فضای فیزیکی مشترک template (baseline=initial, post=final)
- [ ] ثبت affine 4×4/spacing/origin/direction + validator finite/invertibility
- [ ] اعمال MAD روی هر نمونهٔ primary (quarantine + BLOCKED)
- [ ] threshold manifest اتصال به 32 sample hash + receipt hashها + pipeline contract
- [ ] sensitivity: normalize فقط foreground, direct 128→96 drift گزارش, lease GPU تا پایان embedding

### Gate P3 باقی‌مانده
- [ ] state machine صریح `SCRATCH→PREPARED→PAYLOAD_COMMITTED→RECEIPT_COMMITTED→JOURNAL_COMMITTED` + crash recovery finalize-on-matching-bytes
- [ ] آرشیو scratch ناقص با reason code (نه delete)
- [ ] sanitiser traceback
- [ ] preflight_deep کامل (anchor/ledger, writer UNKNOWN, CUDA/VRAM, reparse, source contract, stale artifacts)
- [ ] scheduler موازی bounded (in-flight≤2, SIGINT→finish in-flight→75)

### Acceptance نهایی
- [ ] `READY_TO_START_OR_RESUME_BRAINIAC_PREPROCESSING.json` صدور پس از 12-sample PASS

## نحوهٔ نظارت و ادامه

**فرمان واحد معتبر پس از PASS:**
```powershell
Set-Location -LiteralPath 'H:\New project'
& 'H:\New project\.venvs\brainiac\Scripts\python.exe' `
  'H:\New project\scripts\research\prepare_brainiac_raw_t1_v2.py' `
  --mode full `
  --device cuda
```
اجرای مجدد همان فرمان از آخرین transaction معتبر ادامه می‌دهد.

**مانیتورینگ gate فعلی:**
```powershell
Get-ChildItem data\derived\method_x_brainiac\r1_primary_v2_smoke\eq_sequential\receipts | Measure-Object | Select Count
Get-Content H:\New\ project\tmp\full_run.log -Tail 5
```

## حافظه و orchestration ثبت‌شده

- Memory: `docs/PROJECT_MEMORY_INDEX.md` + `docs/plans/method_x_project_roadmap.md` با digest 9b0f90a5 و battery 40a4a009 به‌روز شدند
- Git: commits `2310969`, `d78b9da`, `676164a` (+ این گزارش)
- Orchestration: `project_roadmap.json` به‌روز شد (در نوبت بعدی با profile)

## هشدارها برای ناظر بعدی

- فرآیند 24880 detached است و تا ~5.5 ساعت دیگر ادامه دارد — آن را kill نکن (stale guard دوباره می‌سازد). فقط با `Ctrl+C` سالم (exit 75) قطع کن.
- `method_x_training_source_contract()` همچنان `9b0f90a5...` و battery `40a4a009...` قبل/بعد هر Gate ثبت شد — هیچ outer/battery تغییر نکرد.
