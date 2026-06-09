# md_out Alignment Index

**Purpose:** Map every document in `C:\Users\USER\Desktop\پروژه\md_out` to the live repo (`H:\New project`) without duplicating the 17k-line master execution program. Use this as the navigation layer between proposal/playbook sources and executable artifacts.

**Canonical unified agent prompt:** `docs/final_execution_program_unified_prompt.md` (v2.0 — 3-month full parallel, QC hardening)  
**Canonical human-readable program:** `docs/final_execution_program.md`  
**Live state:** `.agent/PROGRESS.yaml`

---

## 1. Source files in `md_out` (all 9)

| # | File | Role | Repo mirror / successor |
|---|------|------|-------------------------|
| 1 | `پروپوزال اصلی.md` | Scientific thesis proposal (OCR + الحاقیه 2026-05-25) | `docs/research_methods_limitations_ethics_statistics.md`, `docs/final_writeup_method_x_cog_method_y_pet_mm.md`, `governance/method_x_operating_contract.md` |
| 2 | `برنامه جامع اجرایی هوش مصنوعی.md` | Full v3.3 self-contained execution program (Prompts 0–13, Makefile, tests) | `docs/final_execution_program.md` + `docs/final_execution_program_unified_prompt.md` |
| 3 | `patch v3.4.md` | v3.4-PATCHED corrective pack on v3.3 | `governance/v3_4_patched_changelog.md`, `configs/policy/final_execution_program.json` |
| 4 | `patch v3.9.md` | Phase 0 branch catalogs + SF governance | `docs/patch_v3.9_phase0_branch_catalogs.md` |
| 5 | `patch v3.10.md` | Audit fixes + Phase 2 post-MVP | `docs/patch_v3.10_program_alignment.md`, `docs/program_alignment_audit_2026-06-08.md` |
| 6 | `وظایف من.md` | Human supervisor playbook + 12 AI prompts + Gates 1–4 | Partially implemented in `scripts/`, `tests/`, `Makefile`; gates in `results/tables/gate_matrix.md` |
| 7 | `plan execution.md` | Agent behavioral guidelines (simplicity, surgical edits) | Implicit in agent rules; no separate repo file |
| 8 | `Output Format Requirement.md` | Patch-only unified diff output | Agent session convention |
| 9 | `TOKEN-EFFICIENT MODE (MANDATORY).md` | Scope lock, one task per message, no raw data | Agent session convention |

---

## 2. Per-file content digest + repo mapping

### 2.1 `پروپوزال اصلی.md`

**Two layers:**

| Layer | Content | Still valid? | Repo mapping |
|-------|---------|--------------|--------------|
| **الحاقیه 2026-05-25** | Method X / X-Cog / Method Y split; 1700 catalog; Aβ± not AD; cohort coupling caveat | **Yes — authoritative** | `governance/method_x_operating_contract.md`, `governance/method_x_cog_operating_contract.md`, `governance/method_y_pet_exploratory_policy.md`, `configs/model/method_y_multimodal_plan.yaml` |
| **OCR original pages** | CN/MCI/AD classification; Hybrid EfficientNet3D+Swin; Gates 0.65/0.70; ADNI/OASIS/OpenNeuro | **Partially superseded** | See §5 Divergence table |

**Key scientific facts from الحاقیه:**

| Concept | md_out value | Repo artifact |
|-------|--------------|---------------|
| Catalog N | ~1700 | `data/processed/catalogs/method_x_strict_catalog.csv` (1699 rows) |
| Label distribution | 1163 Aβ+ / 537 Aβ− | `results/governance/cohort_label_coupling_audit.json` |
| Method X tabular | age, sex, education_years, apoe4_carrier, mmse_base | `scripts/build_method_x_strict_feature_matrix.py`, `configs/model/cognitive_feature_sets.yaml` (X-Cog only) |
| Method X-Cog best non-MRI | fixed_tabular+cognitive AUC ~0.785 | `results/final_results/method_x_cog/method_x_cog_summary.md` |
| Method Y status (design time) | DESIGN_READY_NOT_TRAINED | `results/tables/method_y_training_readiness.json` → `DESIGN_READY_EXECUTION_GATED` |
| Method Y scenarios | mri_only / pet_only / mri_pet / mri_pet_cognitive | `configs/model/method_y_multimodal_plan.yaml`, `results/tables/method_y_multimodal_design.md` |

**Proposal § themes → repo:**

| Proposal theme | Repo path |
|----------------|-----------|
| Patient-level split, pseudo-external lock | `splits/external_holdout.csv`, `splits/dev_split.csv`, `scripts/run_external_eval.py` |
| PET = label only (Method X) | `tests/test_method_x_pet_leakage*.py`, `results/governance/method_x_feature_matrix_pet_leakage_check.json` |
| Calibration in-fold only | `governance/calibration_policy.md`, `scripts/run_calibration.py` |
| SHAP tabular / Grad-CAM MRI | `scripts/run_xai_summary.py`, `results/final_results/xai/xai_summary.md` |
| Ablation matrix §7.2 | `scripts/run_3d_ablation.py`, `results/final_results/three_d_ablation/three_d_ablation_summary.md` |
| Tau regression exploratory | `scripts/run_tau_branch.py`, `results/final_results/tau/tau_regression_metrics.md` |
| TRIPOD / ethics | `docs/research_methods_limitations_ethics_statistics.md`, `scripts/write_tripod_*.py` |

---

### 2.2 `برنامه جامع اجرایی هوش مصنوعی.md`

**Structure (v3.3-REVISED):** Global header → Rules 1–15 → Repo skeleton → `.agent/` state → Prompts 0–13 → Gate matrix → Makefile → Publication bundle.

**Do not re-read end-to-end.** Use this index + `docs/final_execution_program_unified_prompt.md`.

| v3.3 section | Live repo equivalent |
|--------------|---------------------|
| Global header YAML (`method: X`, AB_POS vs AB_NEG) | `configs/config.yaml`, `configs/policy/final_execution_program.json` |
| Fast-track (64 dev / 96 final, 3/5 folds) | `scripts/train_3d_mri.py --phase dev|final`, `.agent/PROGRESS.yaml` |
| Rules 1–15 (state, atomic steps, 3-try) | `.agent/PROGRESS.yaml`, `scripts/run_project_stage.py`, `governance/agent_stage_execution_policy.md` |
| Prompt 0 bootstrap | Repo exists; `scripts/write_phase0_governance_artifacts.py` |
| Prompt 1-A skeleton | `src/`, `scripts/`, `tests/` tree |
| Prompt 1-B governance templates | `governance/*.md` |
| Prompt 2 label freeze | `scripts/freeze_label_definition.py` → `results/governance/label_freeze.json` (cutoff **33.2705**) |
| Prompt 3–6 preprocess/train | `scripts/preprocess_mri_spm.py`, `scripts/build_mri_volume_cache.py`, `scripts/train_3d_mri.py` |
| Prompt 7–9 ablation/XAI/external | `scripts/run_3d_ablation.py`, `scripts/run_xai_summary.py`, `scripts/run_external_eval.py` |
| Makefile targets | `Makefile` |
| Execution update block (Method X-Cog / Method Y) | `docs/patch_v3.9_phase0_branch_catalogs.md`, `docs/method_y_pet_assisted_exploratory_plan.md` |

**Codex execution update embedded in md_out (2026-05-25):**

| Stage | md_out name | Repo script / output |
|-------|-------------|----------------------|
| Cognitive features | `build_cognitive_features.py` | `scripts/build_cognitive_features.py` |
| Cognitive ablation | `run_cognitive_ablation.py` | `scripts/run_cognitive_ablation.py`, `scripts/run_cognitive_family_ablation.py` |
| Method Y inventory | aggregate PET counts | `scripts/method_y_pet_inventory.py` → `results/tables/method_y_pet_inventory.md` |
| Method Y design | 4 scenarios | `scripts/method_y_multimodal_design.py` → `results/tables/method_y_multimodal_design.md` |

---

### 2.3 `patch v3.4.md`

**Purpose:** Corrective pack on v3.3; preserves Method X contract.

| Patch item | Repo mapping |
|------------|--------------|
| Master catalog columns preserved | `data/processed/master_catalog.csv`, `scripts/build_master_catalog.py` |
| Non-negotiables unchanged | `governance/method_x_operating_contract.md` |
| v3.5 supplement (X-Cog + Method Y) | Same block as `patch v3.4.md` lines 3–65 → `docs/patch_notes/README_method_y_pet_patch.md` |
| Cognitive feature sets | `configs/model/cognitive_feature_sets.yaml` |
| Method X-Cog tests | `tests/test_method_x_cog_feature_contract.py`, `tests/test_cognitive_features_privacy.py` |
| Method Y design-ready | `docs/method_y_pet_assisted_exploratory_plan.md` |
| Forbidden-term / aggregate-only bundle checks | `scripts/build_public_bundle.py`, `src/utils/public_sanitize.py` |

---

### 2.4 `patch v3.9.md`

| Item | Repo mapping |
|------|--------------|
| Five hash-level branch catalogs | `data/processed/catalogs/*.csv`, `branch_catalog_manifest.json` |
| SF governance amendment | `governance/sf_internal_decoupling_policy.md` |
| Cramér's V before/after SF | `results/governance/cohort_label_coupling_audit.json` |
| Leakage review | `results/governance/leakage_review_report.md` |
| Centiloid cutoff frozen 33.2705 | `results/governance/label_freeze.json` |
| Gate G0 PASS | `.agent/PROGRESS.yaml` → `gate_h1_data`, `gate_h2_qc`, `gate_h3_split_lock` |

**Catalog files (six in unified program; five in v3.9 + Method Y tier in manifest):**

| Catalog | Path |
|---------|------|
| Method X-Strict | `data/processed/catalogs/method_x_strict_catalog.csv` |
| Method X-Cog/SF | `data/processed/catalogs/method_x_cog_sf_catalog.csv` |
| MRI ROI/WMH/Reads | `data/processed/catalogs/mri_roi_wmh_reads_catalog.csv` |
| Bio-Cog / Method Z | `data/processed/catalogs/bio_cog_method_z_catalog.csv` |
| Clinical Endpoint | `data/processed/catalogs/clinical_endpoint_catalog.csv` |
| Manifest | `data/processed/catalogs/branch_catalog_manifest.json` |

---

### 2.5 `patch v3.10.md`

| Fix / analysis | Repo mapping |
|----------------|--------------|
| Leaky MRI surrogates quarantined | `data/processed/quarantine/` |
| Method Y 266 linked tier | `scripts/method_y_gap_filter.py`, `results/tables/method_y_gap_filter_report.json` |
| ROI cohort-probe ~0.99 documented | `results/final_results/final_analysis/final_analysis_report.md`, `scripts/run_cohort_probe.py` |
| Temporal 90-day glossary | `docs/final_execution_program.md` § Temporal terminology |
| Matched propensity | `scripts/run_matched_propensity_analysis.py` → `results/tables/matched_propensity_subset.json` |
| DCA | `scripts/run_decision_curve_analysis.py` → `results/tables/decision_curve_analysis.json` |
| TRIPOD/PROBAST scaffold | `scripts/write_tripod_probast_checklists.py`, `results/governance/tripod_probast_checklists.json` |
| Still open: Method Y training | `results/tables/method_y_training_readiness.json` |
| Still open: Phase 4 manuscript | `results/governance/tripod_manuscript_scaffold.md` (PARTIAL) |

---

### 2.6 `وظایف من.md`

**Human supervisor playbook** — tasks AI cannot do (env, DUA, IRB, manual downloads) + staged AI prompts.

| md_out section | Intent | Repo / status |
|----------------|--------|---------------|
| §0 Environment (Win+WSL, GPU, credentials) | Human-only | N/A in repo |
| §0.7 DUA log | `governance/dua_log.csv` | Check if exists |
| §0.9 PROJECT_CHARTER, PRIVACY_RULES | Original CN/MCI/AD gates | Superseded by Aβ± governance; privacy in `governance/git_publish_safety_policy.md` |
| §0.10 Reporting contract (CI 1000, macro OvR) | **OvR forbidden in v3.3+** | Binary metrics only; `governance/calibration_policy.md` |
| Stage 1 Prompts 0–3 (Hydra/Lightning skeleton) | Bootstrap | Implemented without Hydra/Lightning — plain PyTorch |
| Stage 2 Prompt 4 (MONAI preprocess) | SPM + cache | `scripts/preprocess_mri_spm.py`, `scripts/build_mri_volume_cache.py` |
| Stage 3 Prompt 5 (ResNet18 baseline) | Tabular + 2D/3D baseline | `scripts/run_branch_baselines.py`, `scripts/train_3d_mri.py` |
| Stage 4 Prompt 6 (EfficientNet3D + fusion) | Hybrid backbone | **Deferred** — `Lightweight3DEncoder` primary; `src/models/backbones/backbone_factory.py` has EfficientNet optional |
| Stage 5–7 Prompts 7–9 (ablation, XAI, external) | Full pipeline | Matching scripts exist |
| Stage 8 Prompt 10 (tau regression) | Exploratory | `scripts/run_tau_branch.py` |
| Gates 1–4 (AUC 0.65/0.70 thresholds) | Original proposal gates | Repurposed; binary Aβ metrics in `results/tables/gate_matrix.md` |
| Gate 4 external Iranian cohort | True external | **Not implemented** — locked pseudo-external only |
| VRAM rules (<7.5GB, max 96³) | Hardware guard | `scripts/gpu_preflight_check.py`, `governance/compute_fallback_policy.md` |
| Never send patient data to AI | Privacy | All governance docs |

**12 prompt mapping (وظایف من → scripts):**

| Prompt # | Topic | Primary script(s) |
|----------|-------|-------------------|
| 0 | Bootstrap | `scripts/write_phase0_governance_artifacts.py` |
| 1 | Skeleton | repo structure (done) |
| 2 | Label/dictionary | `scripts/freeze_label_definition.py`, `scripts/build_master_catalog.py` |
| 3 | Preprocess | `scripts/preprocess_mri_spm.py`, `scripts/preprocess_mri_full.py` |
| 4 | Baseline | `scripts/run_branch_baselines.py` |
| 5 | 3D CNN | `scripts/train_3d_mri.py` |
| 6 | Ablation | `scripts/run_3d_ablation.py` |
| 7 | XAI | `scripts/run_xai_summary.py` |
| 8 | External eval | `scripts/run_external_eval.py` |
| 9 | Final analysis | `scripts/final_analysis.py` |
| 10 | Tau | `scripts/run_tau_branch.py` |
| 11–12 | Reporting/bundle | `scripts/build_public_bundle.py`, `scripts/build_thesis_grade_report.py` |

---

### 2.7 `plan execution.md`

Behavioral rules for coding agents:

| Rule | Application in repo |
|------|---------------------|
| Think before coding | Ask before changing governance |
| Simplicity first | Prefer `Lightweight3DEncoder` over unrequested EfficientNet switch |
| Surgical changes | Match existing script patterns |
| Goal-driven verification | Run `pytest` + stage scripts after edits |

No dedicated repo file — follow during all agent sessions.

---

### 2.8 `Output Format Requirement.md` + `TOKEN-EFFICIENT MODE (MANDATORY).md`

| Rule | When to use |
|------|-------------|
| Unified diff patch only | Small targeted fixes requested by user |
| File scope lock | User names explicit allowlist |
| One task per message | TOKEN-EFFICIENT sessions |
| No raw patient data | Always — entire project |

For full-program execution sessions, use `docs/final_execution_program_unified_prompt.md` instead of TOKEN-EFFICIENT mode.

---

## 3. Branch architecture map

| Branch | md_out source | Purpose | Key inputs | Key outputs |
|--------|---------------|---------|------------|-------------|
| **Method X-Strict** | پروپوزال + v3.3 | Protected thesis anchor | T1 96³ + 5 tabular | `results/final_results/three_d/`, `final_analysis_report.md` |
| **Method X-Cog/SF** | الحاقیه + patch v3.4 | Deconfounding / cognitive extension | tabular + cognitive families + SF | `results/final_results/method_x_cog/` |
| **MRI ROI/WMH/Reads** | unified program | Interpretable MRI (no leaky surrogates) | official ROI, WMH, reads | `scripts/extract_mri_features.py`, cohort-probe ~0.99 flagged |
| **Method Y-PET/MM** | الحاقیه + unified prompt | Co-primary PET-assisted science | PET volume + paired T1 | `results/tables/method_y_*`, `scripts/train_pet_method_y.py` |
| **Method Z / Bio-Cog** | unified program | Blood benchmark | pTau217, GFAP, etc. | `scripts/train_bio_cog_method_z.py` |
| **Clinical Endpoint** | unified program | CDR/PACC progression proxy | baseline predictors | `scripts/train_clinical_endpoint_baseline.py` |

---

## 4. Phase → script → artifact map

| Phase | Goal | Gate | Key scripts | Key artifacts |
|-------|------|------|-------------|---------------|
| **0** | Catalogs, label freeze, leakage | G0 | `build_branch_catalogs.py`, `write_phase0_governance_artifacts.py`, `write_leakage_review_report.py`, `run_cohort_label_coupling_audit.py` | `results/governance/*`, `data/processed/catalogs/*` |
| **1** | Interpretable baselines | G1 | `run_branch_baselines.py`, `run_centiloid_regression.py`, `build_incremental_value_table.py` | `results/final_results/baselines/`, incremental value table |
| **1B/2A** | 3D MRI + embeddings | — | `train_3d_mri.py`, `extract_3d_mri_embeddings_oof.py`, `run_3d_ablation.py` | `results/final_results/three_d_ablation/` |
| **2** | Validation battery | G2 MVP | `run_cohort_probe.py`, `run_calibration.py`, `run_validity_audit.py`, `run_matched_propensity_analysis.py`, `run_decision_curve_analysis.py`, `run_headline_temporal_gap_evaluation.py` | `results/final_results/final_analysis/`, thesis report |
| **3** | Exploratory branches | G3/G3Y | `preprocess_pet_spatial.py`, `build_pet_volume_cache.py`, `train_pet_method_y.py`, `train_bio_cog_method_z.py` | Method Y/Z summaries |
| **4** | Publication | G4 | `write_tripod_*`, `build_public_bundle.py`, `build_thesis_grade_report.py` | `public_bundle/`, TRIPOD scaffolds |
| **5** (stretch) | EfficientNet, KD, SSL | optional | `backbone_factory.py` (efficientnet3d_b0) | sensitivity supplements only |

---

## 5. Divergence: md_out original vs live repo

| Topic | md_out (original) | Live repo (A4/LEARN) | Action |
|-------|-------------------|----------------------|--------|
| Primary task | CN/MCI/AD 3-class | Aβ+ vs Aβ− biomarker status | Use live; OCR is historical |
| Cohort | ADNI/OASIS/OpenNeuro + Iranian external | A4+LEARN ~1700; pseudo-external ~340 | Never claim true external |
| Backbone | EfficientNet3D-S + Swin hybrid | Lightweight3D primary; EfficientNet deferred | Phase 5 ablation only |
| Gate 2 threshold | AUC ≥ 0.75 | Tabular ~0.73; 3D ~0.72; `final_claim_allowed=false` | Report honestly |
| Macro OvR metrics | Mentioned in وظایف من | **Forbidden** in v3.3+ | Binary metrics only |
| Method Y | Design-only in الحاقیه | Preprocess partial 110/266; training gated | Complete PET SPM + cache |
| Skull strip | Optional ablation in proposal | Not headline path | Optional if time |
| SF | Not in original OCR proposal | Internal decoupling instrument | Central to validity story |
| ROI high AUC | Expected biology signal | Nonlinear cohort-probe ~0.99 → artifact | Descriptive only |

**Unchanged non-negotiables (all sources agree):**

- PET forbidden as Method X input
- Patient-level grouped CV
- Pseudo-external locked before training
- Calibration fit in-fold only
- No PII in public outputs
- Modality-separated XAI (SHAP tabular, Grad-CAM MRI)

---

## 6. Method Y deep map (co-primary pillar)

| Concept | md_out | Repo |
|---------|--------|------|
| Scenarios | 4 modes in الحاقیه | `configs/model/method_y_multimodal_plan.yaml` |
| Fusion hierarchy | Added in unified prompt | `docs/method_y_phase2_modeling_design.md` |
| PET preprocess | Not started in الحاقیه | `scripts/preprocess_pet_spatial.py`, `scripts/spm_pet_preprocess_backend.py` |
| Readiness gate | DESIGN_READY_NOT_TRAINED | `scripts/check_method_y_training_readiness.py` |
| Class imbalance | — | `governance/method_y_class_imbalance_policy.md`, `configs/policy/method_y_class_imbalance_policy.json` |
| PET-only training script | — | `scripts/train_pet_method_y.py` |
| Y1–Y10 battery | unified prompt | To implement / orchestrate per `method_y_multimodal_design.md` |
| Safe PET command | — | See unified prompt §6.3 (no `--overwrite` on PASS) |

**Current status snapshot (2026-06-08):**

```
status: DESIGN_READY_EXECUTION_GATED
linked: 266 | preprocess done: ~110 | pending: ~156
execution_blockers: PET_PREPROCESS_INCOMPLETE, PET_CACHE_NOT_BUILT_YET
approved_tier: primary_feasible_90
forbidden_tier: gold_standard_30
```

---

## 7. Validation battery quick reference

| # | Test | Script |
|---|------|--------|
| 1 | Pseudo-external eval | `run_external_eval.py` |
| 2 | 5-fold grouped CV | all trainers |
| 3 | SF stress tests | `run_method_x_cog_sf_stress_tests.py` |
| 4 | Temporal gap | `run_headline_temporal_gap_evaluation.py` |
| 5 | Matched/propensity | `run_matched_propensity_analysis.py` |
| 6 | Cramér's V | `run_cohort_label_coupling_audit.py` |
| 7 | Cohort-probe | `run_cohort_probe.py` |
| 8–10 | Random/shuffle/permutation | `run_validity_audit.py`, ablation scripts |
| 11–12 | Ablation | `run_3d_ablation.py`, `run_cognitive_family_ablation.py` |
| 13 | Calibration | `run_calibration.py` |
| 14 | Bootstrap CI | embedded in metric builders |
| 15 | DCA | `run_decision_curve_analysis.py` |

Minimum Phase-2 MVP = items 1–10 in `docs/final_execution_program.md`.

---

## 8. Key config & governance index

| Topic | Path |
|-------|------|
| Root config | `configs/config.yaml` |
| Execution policy JSON | `configs/policy/final_execution_program.json` |
| Method X contract | `governance/method_x_operating_contract.md` |
| Method Y contract | `governance/method_y_multimodal_operating_contract.md` |
| SF policy | `governance/sf_internal_decoupling_policy.md` |
| Cognitive policy | `governance/cognitive_feature_policy.md` |
| Calibration | `governance/calibration_policy.md` |
| Label freeze | `results/governance/label_freeze.json` |
| Cohort-probe protocol | `results/governance/cohort_probe_protocol.json` |
| Deconfounding scope | `results/governance/deconfounding_scope.json` |
| Domain ceiling citations | `results/governance/domain_ceiling_citation_list.json` |
| Decisions log | `DECISIONS.md` |
| Progress state | `.agent/PROGRESS.yaml` |

---

## 9. Results & reporting index

| Deliverable | Path |
|-------------|------|
| Thesis-grade report | `results/final_results/thesis/thesis_grade_report.md` |
| Final analysis | `results/final_results/final_analysis/final_analysis_report.md` |
| 3D ablation | `results/final_results/three_d_ablation/three_d_ablation_summary.md` |
| Method X-Cog | `results/final_results/method_x_cog/method_x_cog_summary.md` |
| Method Y status | `docs/method_y_status_summary.md`, `results/final_results/method_y/phase_y_summary.md` |
| Implementation gap | `results/tables/implementation_gap_matrix.md` |
| Public bundle | `public_bundle/` (74 files, aggregate-only) |
| TRIPOD scaffold | `results/governance/tripod_manuscript_scaffold.md` |
| Expert audit | `docs/program_alignment_audit_2026-06-08.md` |

---

## 10. Literature / GitHub reference layer (methodology only)

Used in unified program §0.C4 — **never override repo governance.**

| Reference | Use for |
|-----------|---------|
| PubMed 40579043 / arXiv 2411.12061 | T1/multi-contrast amyloid prediction ceilings |
| Chiumento PET-guided KD repo | Optional Phase 5 distillation sensitivity |
| Aperture Neuro A4/LEARN data paper | PET protocol, scan counts |
| DEAL repo | Tabular+MRI fusion patterns |
| AmyPET | PET QC cross-check benchmark |
| TRIPOD+AI BMJ 2024 | Phase 4 reporting |
| Nature Med p-tau217 | Method Z benchmark context |

---

## 11. Which document to read for what

| I need to… | Read first |
|------------|------------|
| Execute the full program as AI agent | `docs/final_execution_program_unified_prompt.md` |
| Understand hard constraints quickly | `docs/final_execution_program.md` |
| Map md_out proposal to repo | **this file** |
| Method Y PET workflow | `docs/agent_method_y_pet_phase2_playbook.md` |
| Defense talking points | `docs/thesis_defense_talking_points.md` |
| Original 17k-line playbook detail | `md_out/برنامه جامع اجرایی هوش مصنوعی.md` (grep only) |
| Scientific scope amendment | `md_out/پروپوزال اصلی.md` الحاقیه block |
| Current gate status | `.agent/PROGRESS.yaml` |

---

## 12. Open work queue (from md_out + repo alignment)

| Priority | Item | Blocker |
|----------|------|---------|
| P0 | Complete PET SPM 266/266 | preprocess in progress |
| P0 | Preprocess QC audit gate | `run_preprocess_quality_audit.py` |
| P0 | Build PET volume cache | cache script after preprocess |
| P1 | Method Y battery Y1–Y10 | readiness + QC gate |
| P1 | Phase 4 TRIPOD + manuscript skeleton | manuscript PARTIAL |
| P1 | DCA + temporal gap + biomarker hierarchy figure | parallel CPU |
| P1 | EfficientNet3D ablation + PET-guided KD | parallel GPU (mandatory) |
| P1 | FLAIR sensitivity track | parallel after FLAIR index |
| P2 | Visual QC sign-off workflow | PET overlays |

---

## 13. Unused A4 Learn assets (read-only `H:\A4 Learn`)

**Purpose:** Document high-value dataset files not yet wired into training pipelines. Integrated into unified prompt §0.C6. Do not copy raw paths into repo artifacts.

### 13.1 Imaging modalities

| Modality / folder | Approx. count | Current repo use | 3-month plan |
|-------------------|---------------|------------------|--------------|
| T1 structural | ~1700 enrolled | **Used** — Method X | Complete SPM 96³ cache |
| FBP amyloid PET | ~266 paired | **Partial** — Method Y preprocess in progress | Complete + Y1–Y10 |
| FLAIR | ~14,000 | **Unused** | T1+FLAIR sensitivity (§5.3) |
| FTP tau PET | ~3,000 | **Unused** | Y8 tracer batch (subset) |
| T2 / T2* | large | **Unused** | Out of 3-month scope |
| DWI | large | **Unused** | Out of 3-month scope |
| fMRI rest / b0 | large | **Unused** | Out of 3-month scope |

### 13.2 Official CSV / clinical tables

| File | Use today | Planned use |
|------|-----------|-------------|
| `imaging_PET_VA.csv` | Label source (centiloid) | PET QC cross-check (`official_centiloid` in QC JSON) |
| `imaging_SUVR_amyloid.csv` | Partial | QC audit centiloid agreement |
| `imaging_SUVR_tau.csv` | None | Y8 tau sensitivity |
| `imaging_retinal.csv` | None | Method Z exploratory table |
| Plasma panel (pTau217, GFAP, Aβ, NfL, …) | Partial Method Z | Full Bio-Cog parallel track |
| SF screen-fail tables | X-Cog/SF stress | **Not** MRI external validation |

### 13.3 QC & audit hooks

| Script | Output |
|--------|--------|
| `scripts/run_preprocess_quality_audit.py` | `results/tables/preprocess_quality_audit.json` + `.md` |
| `scripts/build_visual_qc_manifest.py` | `results/qc/visual_review_manifest.json` |
| `scripts/sign_visual_qc.py` | Updates `visual_qc_signoff` on PET QC JSON |
| `scripts/run_method_y_battery.py` | `results/tables/method_y_battery_status.json` |
| `scripts/build_flair_index.py` | `results/tables/flair_volume_index.json` |
| `src/utils/preprocess_quality.py` | MRI PASS/HOLD/FAIL; official centiloid loader |
| `configs/model/method_y_spatial_preprocess_policy.yaml` | PET NCC fail/warn + centiloid delta gates |

**Command:**

```powershell
python scripts/run_preprocess_quality_audit.py --data_root "H:\A4 Learn"
```

### 13.5 Branch catalogs (6 branches)

| # | Branch | Repo artifact |
|---|--------|---------------|
| 1 | Method X-Strict | `data/processed/catalogs/method_x_strict_catalog.csv` |
| 2 | Method X-Cog | `data/processed/catalogs/method_x_cog_catalog.csv` |
| 3 | Method X-Cog/SF | `data/processed/catalogs/method_x_cog_sf_catalog.csv` |
| 4 | MRI ROI/WMH/reads | `data/processed/catalogs/method_mri_roi_wmh_reads_catalog.csv` |
| 5 | Method Z / Bio-Cog | blood catalog builders under `scripts/build_branch_catalogs.py` |
| 6 | Method Y PET paired | `data/preprocessed_pet/_private/manifests/fbp_private_manifest.csv` (not a sixth flat CSV in `catalogs/`) |

---

### 13.6 Tier naming (locked)

| Name | N | Role |
|------|---|------|
| `primary_feasible_90` | ~266 (232/34) | **Approved** Method Y training tier |
| `primary_strict` | ~191 | Gap-filtered; do not use as primary without governance bump |
| `gold_standard_30` | ~30 | **Forbidden** for training (imbalance policy) |

---

*Last aligned: 2026-06-08 (v2.0 QC + unused assets). Update when preprocess milestones or md_out patches change.*
