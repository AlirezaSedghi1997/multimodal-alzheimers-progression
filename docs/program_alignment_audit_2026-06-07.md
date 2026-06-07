# Program Alignment Audit — 2026-06-07

Audit of the pasted **Final Execution Program** against repository state (`H:\New project`, data `H:\A4 Learn`).

## Executive verdict

| Phase | Program gate | Repo status | Verdict |
|-------|--------------|-------------|---------|
| **0** Governance & catalogs | G0 | Gate G0 **PASS** | **DONE** |
| **1** Interpretable baselines + 3D | G1 | Phase 1 baselines, centiloid, 3D 5-fold, OOF latents | **DONE** |
| **2** Validation battery (Minimum MVP) | G2 | `phase2_mvp_status.json` → **MVP_COMPLETE** | **DONE** |
| **2** Post-MVP refinements | optional | DCA, matched subsets, full temporal battery partially missing | **PARTIAL** |
| **3** Exploratory branches | G3 conditional | Method Z + clinical endpoint started; Method Y partial | **IN PROGRESS** |
| **4** Paper-grade bundle | G4 | Public bundle exists; TRIPOD/PROBAST/manuscript draft incomplete | **NOT STARTED** |

**Protected thesis object (Method X-Strict)** is preserved. Strong headline claims remain **blocked by design** (`CLAIM_MUST_BE_LIMITED_OR_REFRAMED`) because of cohort-label coupling — this is scientifically correct, not a pipeline failure.

---

## Terminology clarification (important)

| Term | Meaning in this program | **NOT** meaning |
|------|-------------------------|-----------------|
| `primary_strict` / 90 days | Max **\|PET scan day − T1 scan day\|** relative to study T0 (`abs_pet_t1_gap_days`) | Project execution duration, 12-week schedule, or GPU runtime |
| 12-week schedule | Engineering timeline for phases | Clinical pairing window |
| 191 linked PET samples | Catalog rows with PET–T1 gap in tier `primary_strict` (30–90 d) + successful PET/T1 link | Total PET files (5661) or all gap≤90 rows (266) |

Source: `scripts/build_master_catalog.py` computes gap; `configs/model/method_y_gap_cohort_eval_policy.yaml` defines tiers.

---

## Phase 0 — DONE (aligned)

| Deliverable | Artifact | Status |
|-------------|----------|--------|
| Five branch catalogs | `data/processed/catalogs/branch_catalog_manifest.json` | DONE |
| Feature allowlists | `results/governance/feature_allowlist_summary.json` | DONE |
| Label freeze | `results/governance/label_freeze.json` (cutoff **33.2705**) | DONE |
| Cramér's V before/after SF | `results/governance/cohort_label_coupling_audit.json` (V≈1.0 → 0.90) | DONE |
| Leakage review | `results/governance/leakage_review_report.json` | PASS |
| Cohort-probe protocol | `results/governance/cohort_probe_protocol.json` | DONE |
| Deconfounding scope | `results/governance/deconfounding_scope.json` | DONE |
| Method X PET-leakage matrix test | `results/governance/method_x_feature_matrix_pet_leakage_check.json` | DONE |
| SF governance | `governance/sf_internal_decoupling_policy.md` | DONE |

**Gap matrix:** 32 DONE / 1 PARTIAL / 0 MISSING / 1 SKIP.

---

## Phase 1 — DONE (aligned)

| Deliverable | Status | Notes |
|-------------|--------|-------|
| Interpretable baselines (tabular, cog, ROI, WMH, reads) | DONE | `results/final_results/phase1_baselines/` — **no leaky 15 surrogates in headline** |
| Continuous centiloid regression | DONE | pooled + within-A4 + within-LEARN |
| Incremental-value table | DONE | |
| 3D CNN 5-fold on 96³ cache | DONE | ROC-AUC ≈ 0.722 (ablation best) |
| OOF MRI latents + cohort-probe | DONE | 3D latent linear probe ≈ **0.50** (good); ROI nonlinear ≈ **0.99** (limitation) |
| Retire leaky MRI surrogates | DONE in headline paths | See **Issue I-1** below |

Blood feature-set rows correctly deferred to Phase 3 / Method Z per program.

---

## Phase 2 Minimum MVP — DONE (aligned)

`results/tables/phase2_mvp/phase2_mvp_status.json` → **MVP_COMPLETE** for:

- `method_x_strict_3d_cnn`
- `method_x_strict_tabular`
- `mri_roi_wmh_reads_branch`

Method X-Cog/SF stress tests: `results/final_results/method_x_cog_sf/sf_stress_tests_summary.json` (internal decoupling only).

Validity audit: `results/tables/validity_audit/validity_audit_summary.json` → label shuffle ≈ 0.51; substudy from MRI features ≈ 0.999 (cohort artifact flagged).

---

## Phase 2 post-MVP — PARTIAL (gaps vs full program §2.0)

| Item | Status | Notes |
|------|--------|-------|
| Temporal-gap sensitivity (≤60/90/120/all) | PARTIAL | `run_temporal_window_sensitivity.py` exists; not full battery on all headline models |
| Matched / propensity-weighted subsets | MISSING | Not found as completed artifact |
| DCA | MISSING | Required for clinical/blood if clinically interpreted |
| Pure random-feature baseline | UNCLEAR | Label-shuffle yes; random-feature dim-matched control not verified |
| Train-SF / test-enrolled & reverse | PARTIAL | SF stress tests include some arms; verify both directions complete |
| Leakage-inflation negative control (§2.C) | NOT RUN | Optional per program — OK to skip until MVP+ time |

---

## Phase 3 — IN PROGRESS (conditional, after G2)

| Branch | Status | Blockers / notes |
|--------|--------|------------------|
| **Method Z / Bio-Cog** | TRAINED (exploratory) | `bio_cog_method_z_summary.json`; SF coupling caveat; blood ablation AUC≈0.87 |
| **Clinical endpoint** | BASELINE DONE | Governance **FROZEN**; CDR/PACC baselines exploratory only |
| **Method Y PET/MM** | PARTIAL | Manifest **191** @ primary_strict; 2 QC-pass preprocess; **NOT_TRAINING_READY** (class imbalance 168/23) |
| **Multimodal fusion (Method Y)** | NOT TRAINED | Design only |
| **SSL** | NOT STARTED | Stretch-only — correct deferral |
| **External embedding sensitivity** | NOT STARTED | Optional — correct deferral |

---

## Phase 4 — NOT STARTED

| Item | Status |
|------|--------|
| TRIPOD+AI checklist (complete) | PARTIAL mention in governance only |
| PROBAST+AI self-assessment | NOT FOUND |
| Final Methods/Results/Limitations draft | PARTIAL (`thesis_grade_report.md` exists) |
| Domain-ceiling citation list | NOT FOUND as standalone artifact |
| 12-week progress table in final report | NOT FOUND |
| Required figure set (9–10 figures) | PARTIAL |

---

## Scientific / technical / logical issues found

### I-1 — Leaky 15 MRI surrogates still on disk (retired from headline, not deleted)

Validity audit still tests `mri_features_only` → AUC ≈ **0.999**. Headline Phase 1 excludes them, but `data/processed/mri_features.csv` persists. **Action:** keep as negative-control only; ensure no headline script imports them (audit allowlist).

### I-2 — Strong claim blocked despite complete 3D training (CORRECT)

`three_d_metrics.json`: `final_claim_allowed=false`, `validity_recommendation=CLAIM_MUST_BE_LIMITED_OR_REFRAMED`. Training contract satisfied; validity policy blocks strong claim due to cohort coupling. **Not a bug.**

### I-3 — ROI/WMH branch encodes cohort nonlinearly

Cohort-probe: 3D latent ≈ 0.50 (OK); ROI nonlinear ≈ 0.99. Must remain explicit limitation in Results — aligned with program §0.6 interpretation rules.

### I-4 — Method Y `primary_strict` tier excludes gold (≤30 d) rows

266 catalog rows have gap ≤90 d, but only **191** match tier name `primary_strict` (30–90 d). Gold tier (75 rows) excluded when `--tier primary_strict`. **Policy nuance, not PET shortage.** Consider `≤90` combined tier for feasibility reporting.

### I-5 — Temporal “90 days” confusion

Addressed in glossary above. Recommend adding to `docs/final_execution_program.md` and Method Y preregistration.

### I-6 — `train_3d` status logging inconsistency

PROGRESS.yaml shows both `FULL_3D_TRAINING` and `DEBUG_ONLY` events. Metrics artifact indicates final 5-fold contract met. **Action:** normalize stage runner status strings.

### I-7 — Phase 3 started before all Phase 2 post-MVP items

Allowed by program (“after Minimum MVP”). Method Z / clinical / Method Y CPU work is conditional and non-blocking — **aligned**, but post-MVP validation should not be forgotten.

### I-8 — Git / publication sync

PR #2 (`method-x-cog` → `main`) open; not merged. Public bundle on branch ahead of `main`.

---

## 12-week progress snapshot (program schedule)

| Weeks | Planned | Actual (2026-06-07) |
|-------|---------|---------------------|
| 1–2 | Phase 0 + GPU smoke | **DONE** (G0 PASS, SPM complete, 96³ cache) |
| 3–4 | Phase 1 baselines + 3D | **DONE** |
| 5–7 | Phase 2 MVP | **DONE** (MVP_COMPLETE) |
| 5–7 | Phase 2 post-MVP | **PARTIAL** |
| 8–10 | Phase 3 exploratory | **IN PROGRESS** (~40%) |
| 11–12 | Phase 4 paper bundle | **NOT STARTED** |

---

## Recommended next actions (for following chat)

**Priority A — close Phase 2 post-MVP gaps (CPU)**
1. Run/full-report temporal-gap sensitivity on headline models (≤60/90/120/all).
2. Matched/propensity-weighted subset analysis.
3. DCA for Method X-Cog and Method Z if reporting clinically.

**Priority B — Phase 3 continuation**
4. Method Y: policy decision on ≤90 d inclusive tier vs 191 strict; full preprocess if training approved.
5. Method Y fusion only after baselines + readiness gate.

**Priority C — Phase 4 prep**
6. TRIPOD+AI + PROBAST+AI artifacts.
7. Domain-ceiling citation list.
8. Merge PR #2; refresh public bundle on `main`.

**Priority D — documentation hygiene**
9. Normalize `train_3d` stage status in orchestrator.
10. Add temporal-gap glossary to preregistration docs (done in `final_execution_program.md` this session).

---

## Claim policy compliance — PASS

No forbidden Alzheimer-diagnosis claims detected in governance artifacts. Pseudo-external terminology preserved. SF marked internal only. PET label-only for Method X enforced.
