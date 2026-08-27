# گزارش تحویل به ناظر بعدی — BrainIAC P0-P3 (حالت زنده: gate 10/12)

**PID زنده:** 13276, CPU 10514s, wall ~5 ساعت از شروع full
**seq_receipts:** 10/12 در `r1_primary_v2_smoke/eq_sequential/receipts` (هر receipt ~40 دقیقه تک‌نخی)
**تخمین باقیمانده:** seq 2×40min=80min + par 12/2×40min=240min ≈ 5.3 ساعت تا PASS gate → سپس `execution_profile_v1.json` → calibrate 32 → primary 1669 → sensitivity 64

## کارهای انجام‌شده در این جلسه (P0-P3)

### P0 Freeze
- `BRAINIAC_PREPROCESSING_PRE_REPAIR_FREEZE_20260826.json` (f72bc9f4…) با snapshot کامل
- calibration 2 receipts + partial equivalence به `superseded/` منتقل شدند (پس از اثبات DEAD)

### P1 Git/Contract
- `brainiac_dependency_closure.py` جدید
- `git add` محدود 7 فایل + commit `2310969`

### P2 Scientific
- **Morphology bug تأیید شد:** `dilate_mask` فقط محور 0 را می‌گرفت (loop var استفاده نمی‌شد) → 3 به‌جای 7
- با `scipy.ndimage.binary_dilation` با ساختار 6-neighbour درست شد → 7 voxel
- تست‌های ترمیم 14 مورد سبز

### P3 Transaction/Preflight
- `run_full` بازچینی: preflight اول → gate equivalence → calibrate/run/sensitivity
- حذف auto-flip قدیمی
- 70 تست سبز

## کارهای انجام‌نشده

- تکمیل gate 12/12 → 10/12 فعلی
- COM فیزیکی، affine 4×4, MAD روی primary, sensitivity BG=0, direct drift, lease تا پایان embedding
- state machine 5 حالته, sanitiser, preflight_deep, scheduler موازی bounded
- smoke 12 واقعی + full 1669

## دستور نظارت

```powershell
Get-ChildItem data\derived\method_x_brainiac\r1_primary_v2_smoke\eq_sequential\receipts | Measure-Object | Select Count
Get-Content H:\New\ project\tmp\full_run.log -Tail 5
```

## دستور نهایی (پس از PASS تمام Gateها)

```powershell
Set-Location -LiteralPath 'H:\New project'
& 'H:\New project\.venvs\brainiac\Scripts\python.exe' `
  'H:\New project\scripts\research\prepare_brainiac_raw_t1_v2.py' `
  --mode full `
  --device cuda
```
