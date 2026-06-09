# Program Alignment Audit — 2026-06-08

Mirror for desktop: update `C:\Users\USER\Desktop\پروژه\md_out\patch v3.11.md` if needed.

## Executive summary

| Phase | Progress | Gate |
|-------|----------|------|
| 0 — Governance & catalogs | **100%** | G0 **PASS** |
| 1 — Interpretable baselines + 3D | **100%** | G1 met |
| 2 — Minimum MVP | **100%** | G2 **MVP_COMPLETE** |
| 2 — Post-MVP | **~55%** | DCA + matched DONE; headline temporal reruns pending |
| 3 — Exploratory | **~45%** | Method Z + clinical baselines DONE; Method Y preprocess partial |
| 4 — Paper-grade | **~20%** | TRIPOD scaffold only |

**Overall weighted progress: ~68%**

Protected Method X-Strict path is scientifically sound with **claim blocking** where cohort-label coupling demands it (`CLAIM_MUST_BE_LIMITED_OR_REFRAMED`).

---

## Compliance with hard constraints

| Constraint | Status |
|------------|--------|
| PET = label only for Method X | **PASS** |
| No AD diagnosis claims | **PASS** |
| Pseudo-external wording | **PASS** |
| SF = internal decoupling only | **PASS** |
| Blood separate (Method Z) | **PASS** |
| Aggregate/hash-only outputs | **PASS** |
| Centiloid cutoff frozen (33.2705) | **PASS** |
| Leaky surrogates retired from headline | **PASS** (quarantined) |

---

## Issues found and disposition

### Scientific / logical

| ID | Issue | Severity | Action |
|----|-------|----------|--------|
| S1 | Cramér's V = 1.0 on Method X-Strict | Expected | Claim policy correctly limits interpretation |
| S2 | ROI nonlinear cohort-probe ~0.99 | Medium | Documented in `final_analysis_report.md` |
| S3 | Method Y class imbalance 168/23 (primary_strict) | High | Blocker until tier/weighting policy decided |
| S4 | 266 linked ≤90d vs 191 primary_strict tier name | Policy | Use `primary_feasible_90` or document tier choice |
| S5 | 3D `final_claim_allowed=false` | Expected | Not a training failure — validity gate |

### Technical / engineering

| ID | Issue | Severity | Fix (2026-06-08) |
|----|-------|----------|------------------|
| T1 | Stale `pet_preprocessing_started: false` | Low | **Fixed** — resume-aware status in design/summary builders |
| T2 | Preprocess summary lagged resume log | Low | **Fixed** — `pet_preprocess_progress` utility |
| T3 | SPM coreg NaN on native PET | High | **Fixed** — ANTs rigid + SPM MNI write hybrid |
| T4 | Resume UX / silent stdout | Medium | **Fixed** — `-u`, `--auto_batches`, status prints |
| T5 | Gap matrix missing post-MVP rows | Low | **Fixed** — DCA, matched, TRIPOD, Method Y prep, quarantine |
| T6 | Leaky `mri_features.csv` on disk | Medium | **Quarantined** — default path → `data/processed/quarantine/` |

### Deviations from 12-week program

| Item | Verdict |
|------|---------|
| Phase 3 started before all post-MVP complete | **Allowed** (CPU-only exploratory) |
| SSL / external embeddings not started | **Correct deferral** |
| Domain-ceiling citation list | **MISSING** — Phase 4 |
| Headline temporal-gap reruns | **PARTIAL** — catalogs exist, models not re-evaluated |

---

## Method Y PET preprocess (live)

- **Mode:** `spm_pet_t1_mni` (ANTs PET→T1 + SPM deformation + MNI write)
- **Resume log:** `results/logs/method_y_pet_preprocess_resume_fbp_spm_pet_t1_mni.json`
- **Command:** `python -u scripts/preprocess_pet_spatial.py ... --auto_batches --batch_size 10`
- **Training readiness:** `NOT_TRAINING_READY` until preprocess complete + class policy resolved

---

## Top priorities (next execution)

1. Complete SPM/MNI PET preprocess (resume-safe auto-batches)
2. Resolve Method Y class-imbalance / tier policy for training gate
3. Re-run temporal-gap sensitivity on headline models (tabular, 3D, ROI)
4. Phase 4: domain-ceiling citations, TRIPOD manuscript items, figure set
5. Refresh public bundle after above

---

## Evidence files

- `results/tables/implementation_gap_matrix.md`
- `results/tables/phase2_mvp/phase2_mvp_status.json`
- `results/tables/validity_audit/validity_audit_summary.json`
- `results/governance/phase0_governance_manifest.json`
- `docs/final_execution_program.md`
