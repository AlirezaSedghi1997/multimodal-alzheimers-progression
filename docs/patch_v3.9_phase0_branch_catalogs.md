# Patch v3.9 — Phase 0 Branch Catalogs & SF Governance (2026-06-07)

این patch برنامه اجرایی جدید (Final Execution Program) را با وضعیت فعلی repo هماهنگ می‌کند.

## تکمیل‌شده در این patch

### Governance
- `governance/sf_internal_decoupling_policy.md` — SF فقط internal decoupling/stress-test
- `results/governance/leakage_review_report.json` — گزارش رسمی leakage review
- `results/governance/cohort_label_coupling_audit.json` — Cramér's V قبل/بعد SF
- `configs/policy/final_execution_program.json` — cutoff واقعی `33.2705` (نه placeholder)

### پنج کاتالوگ hash-level
مسیر: `data/processed/catalogs/`

| Branch | Rows | Cramér's V | Notes |
|--------|------|------------|-------|
| method_x_strict | 1699 | 1.00 | A4+LEARN MRI enrolled |
| method_x_cog_sf | 4484 | 0.90 | + SF (2785); label≠cohort in SF |
| mri_roi_wmh_reads | 2845 | ~1.00 | ICV-normalized ROI + WMH + reads |
| bio_cog_method_z | 3010 | 0.93 | blood branch جدا از Method X |
| clinical_endpoint | 4484 | 0.90 | feasibility flags |

### اسکریپت‌های جدید
- `scripts/build_branch_catalogs.py`
- `scripts/run_cohort_label_coupling_audit.py`
- `scripts/write_leakage_review_report.py`
- `make phase0-complete` — اجرای یکجا Phase 0 artifacts

### یافته‌های کلیدی audit
- **Before SF:** Cramér's V = **1.0** (A4/LEARN ≈ cohort classifier)
- **After SF:** Cramér's V = **0.90**; SF-only: pos=194, neg=2591 (centiloid primary)
- **SF VA mismatch:** تعداد ناسازگاری primary vs sensitivity label در SF گزارش شده

## Gate G0 — وضعیت

| Requirement | Status |
|-------------|--------|
| Five branch catalogs | **DONE** |
| SF governance amendment | **DONE** |
| Label frozen (33.2705) | **DONE** |
| Leakage review report | **PASS** |
| Cramér's V before/after SF | **DONE** |
| Cohort-probe protocol | **DONE** |
| Deconfounding scope | **DONE** |

**Gate G0: PASS** — می‌توان Phase 1/2 را ادامه داد.

## کارهای بعدی (چت بعدی)

### GPU track (اولویت اگر SPM کامل است)
1. تأیید SPM 1700/1700
2. `build_mri_volume_cache.py --resolution 96`
3. `run_3d_ablation.py` 5-fold
4. cohort-probe روی embeddings

### CPU track — Phase 1 baselines
1. interpretable baselines روی feature-set rows (tabular, cognitive, ROI, WMH, reads)
2. continuous centiloid regression (within-A4, within-LEARN)
3. retire leaky 15 MRI surrogate features از headline models

### Phase 2 MVP validation battery
- locked pseudo-external (no tuning)
- label-shuffle, permutation, calibration, bootstrap CI
- SF-aware stress tests

## Claim policy (بدون تغییر)
- amyloid **biomarker status** — نه Alzheimer diagnosis
- locked **pseudo-external** — نه external validation
- SF — internal decoupling only
- blood — minimally invasive biomarker branch (Method Z)
