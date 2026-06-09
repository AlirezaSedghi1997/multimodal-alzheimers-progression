# Agent Memory Prompt — paste to restore project context

**Full comprehensive restore prompt:** `docs/AGENT_CONTEXT_RESTORE_PROMPT.md` ← use this when starting a new chat.

Copy the block below for a **short** restore, or open the full file above for complete critical path + QC + gaps.

---

```
CONTEXT RESTORE — amyloid-tau-prediction-a4learn-methodx

Repo: H:\New project
GitHub: github.com/AlirezaSedghi1997/amyloid-tau-prediction-a4learn-methodx
Data root (READ-ONLY): H:\A4 Learn

READ FIRST (in order):
1. docs/final_execution_program_unified_prompt.md  — canonical execution program
2. docs/md_out_alignment_index.md                    — md_out → repo mapping
3. .agent/PROGRESS.yaml                              — live gate/status
4. results/tables/method_y_training_readiness.json   — Method Y gate

ROLE: Senior ML engineer + neuroimaging methods scientist. Optimize VALIDITY and CLAIM CONTROL, not AUC.

DUAL-PILLAR SCIENCE:
• PILLAR 1 — Method X-Strict (protected thesis anchor): PET-FREE Aβ± prediction from T1 + fixed tabular (age, sex, education_years, apoe4_carrier, mmse_base). PET = label only. N≈1700 MRI enrolled.
• PILLAR 2 — Method Y-PET/MM (co-primary exploratory): PET-assisted modeling on ~266 PET–T1 linked hashes (232/34 imbalance). Headline = MRI incremental value over PET-only; Y1 PET-only AUC not for title/abstract.
• Method Z — blood benchmark (pTau217 etc.), separate from non-invasive claim.
• Method X-Cog/SF — deconfounding/cognitive extension; SF = internal stress-test only, NOT external.

HARD CONSTRAINTS (never violate):
- Target = "amyloid biomarker status" (Aβ+/Aβ−), NEVER Alzheimer diagnosis
- Holdout = "locked pseudo-external evaluation", NEVER true external validation
- Centiloid cutoff FROZEN at 33.2705 (results/governance/label_freeze.json)
- Cohort-label coupling Cramér's V ≈ 1.0 on enrolled A4/LEARN → pooled binary AUC is NOT biological proof
- Cohort-probe: linear + nonlinear + gap; never claim deconfounding from linear probe alone
- All preprocessing/calibration fit TRAIN FOLDS ONLY
- Public outputs: aggregate/hash only; no raw IDs, paths, images, HDF5, checkpoints
- Raw data READ-ONLY; never --overwrite completed PET preprocess PASS items

CURRENT STATUS (check PROGRESS.yaml for latest):
- Phase 0–2 MVP: COMPLETE (Method X baselines, 3D ablation, validation battery, XAI)
- 3D best CV AUC ~0.717; tabular descriptive ~0.73; final_claim_allowed=false for 3D headline
- 3D latent cohort-probe ~0.50 (good); ROI nonlinear probe ~0.99 (artifact — descriptive only)
- Method Y: DESIGN_READY_EXECUTION_GATED — PET preprocess ~230/266 (177 PASS + 53 SKIPPED_EXISTS); 36 pending on SPM host; QC audit gate PASS; visual QC reviewed (109 pass / 111 hold / 11 fail); PET cache n=109 built
- Phase 4 reporting: PARTIAL (TRIPOD scaffold + achieved-vs-cited table)
- Full 3-month scope retained (Y1–Y10, FLAIR, EfficientNet, KD) — parallel execution

METHOD Y SAFE PET COMMAND (background, do not stop; Windows — SPM required):
$env:SPM_DIR = "D:\Matlab & Tolboxes\spm_25.01.02\spm"
python -u scripts/preprocess_pet_spatial.py --data_root "H:\A4 Learn" --tracer fbp --tier primary_feasible_90 --spatial_mode spm_pet_t1_mni --execute --n_jobs 1 --auto_batches --batch_size 10 --matlab "C:\Program Files\MATLAB\R2024a\bin\matlab.exe" --spm_dir "D:\Matlab & Tolboxes\spm_25.01.02\spm" --overwrite_resume
(NO --overwrite)

QC BEFORE METHOD Y TRAINING:
python scripts/run_preprocess_quality_audit.py --data_root "H:\A4 Learn" --build_visual_manifest
python scripts/build_visual_qc_manifest.py
python scripts/batch_sign_visual_qc.py   # metrics-based review; use sign_visual_qc.py for manual overrides
python scripts/sign_visual_qc.py --sample_hash <hash> --decision pass|fail|hold
python scripts/check_method_y_training_readiness.py
python scripts/run_method_y_battery.py   # add --execute when gates clear

KEY SCRIPTS:
- Governance: write_phase0_governance_artifacts.py, check_method_y_training_readiness.py
- Method X: train_3d_mri.py, run_3d_ablation.py, run_cohort_probe.py, final_analysis.py
- Method Y: preprocess_pet_spatial.py, build_pet_volume_cache.py, train_pet_method_y.py
- Reporting: build_thesis_grade_report.py, build_public_bundle.py, write_tripod_*.py

WHEN CODING:
- Surgical minimal diffs; match repo conventions
- Run pytest after substantive edits
- Update .agent/PROGRESS.yaml after significant steps
- Do NOT commit unless user explicitly asks

SOURCE DOCS (external, do not duplicate):
- C:\Users\USER\Desktop\پروژه\md_out\ — 9 files; see docs/md_out_alignment_index.md

Execute per unified prompt phases. If blocked, state precise human_action_needed.
```

---

*Keep this file in sync when major milestones change (update CURRENT STATUS block manually or via agent).*
