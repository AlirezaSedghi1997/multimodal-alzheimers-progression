#!/usr/bin/env python3
"""Phase 4 TRIPOD+AI manuscript scaffold (aggregate governance text only)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.io_helpers import atomic_write_json, atomic_write_text
from src.utils.report_sources import fmt, load_json, nested
from src.utils.state_manager import StateManager


def main() -> int:
    freeze = load_json(Path("results/governance/label_freeze.json"))
    threshold = freeze.get("centiloid_threshold", "33.2705")
    phase1 = load_json(Path("results/final_results/phase1_baselines/phase1_baselines_summary.json"))
    cohort_probe = load_json(Path("results/tables/cohort_probe/cohort_probe_summary.json"))
    three_d = load_json(Path("results/final_results/three_d/three_d_metrics.json"))
    incr = load_json(Path("results/tables/phase1_incremental_value_table.json"))

    roi_probe = incr.get("cohort_probe_nonlinear_auc_roi_wmh_reads")
    tab_auc = nested(phase1, "results", "fixed_tabular", "roc_auc")
    three_d_auc = nested(three_d, "aggregate", "roc_auc")

    methods = f"""# Methods Scaffold (TRIPOD+AI-aligned)

## Title (draft)

Development and internal validation of PET-free models for amyloid biomarker status in an A4/LEARN-style cohort (not clinical Alzheimer disease diagnosis).

## Study design and setting

- Multicenter observational prevention-cohort style data (A4 + LEARN enrolled MRI sample).
- Model **development** with patient-grouped cross-validation and locked pseudo-external evaluation.
- Pre-specified validity analyses: cohort-probe, continuous centiloid regression, calibration, subgroup audit.

## Participants

- Enrolled MRI catalog: ~1699 hash-level rows with T1 and fixed tabular covariates.
- SF screening cohort used **only** for internal decoupling/stress tests — never external validation.
- Method Y exploratory subset: ~266 PET–T1 pairs with |gap|≤90 days (separate branch).

## Outcome

Primary binary endpoint: amyloid biomarker status from frozen centiloid threshold **{threshold}** (`results/governance/label_freeze.json`).
PET-derived centiloid is **label only** in Method X; never a predictor.

## Predictors by branch

### Method X-Strict (thesis anchor)

- T1 MRI 96³ cache + 3D CNN embedding fused with fixed tabular: age, sex, education_years, apoe4_carrier, mmse_base.
- No PET, blood, or extended cognitive features in strict claim path.

### MRI ROI/WMH/reads (interpretability extension)

- Official ICV-normalized ROI volumes; single ICV-normalized WMH volume; official MRI reads.
- Nonlinear cohort-probe on this block ≈ **{fmt(roi_probe)}** → interpret as sensitivity/descriptive, not independent biology.

### Method X-Cog / SF

- Cognitive summaries as exploratory extension; SF stress-test arms documented separately.

### Method Z (blood) and Method Y (PET)

- Separate branches with distinct claim boundaries; never merged into non-invasive Method X claim.

## Sample size and missing data

Report realized N per feature set after QC; imputation strategy documented in branch baselines.

## Model building

- Logistic regression for tabular/ROI baselines; 3D CNN + cross-attention fusion for Method X-Strict deep model.
- Hyperparameters locked per execution program; ablation variants: concat, cross-attention, latent alignment.

## Validation

1. 5-fold patient-grouped CV with OOF predictions.
2. Locked pseudo-external holdout (340 patients) — **not** independent external validation.
3. Cohort-probe: linear primary, nonlinear secondary (3D latent ≈0.50; ROI nonlinear ≈0.99).
4. Calibration (Brier, ECE) fit on train folds only.
5. Bootstrap 95% CIs on headline metrics.
6. Matched covariate subset and decision-curve sensitivity analyses.

## Performance measures

ROC-AUC, PR-AUC, Brier, balanced accuracy where class imbalance applies, bootstrap CIs.

## Software and reproducibility

Aggregate governance bundle (`public_bundle/`); no raw identifiers or imaging in public artifacts.

## Limitations (mandatory)

- Cramér's V≈1.0 cohort-label coupling on enrolled A4/LEARN binary label.
- Pooled binary AUC is descriptive; within-cohort and continuous centiloid analyses required.
- 3D CNN AUC ({fmt(three_d_auc)}) below tabular descriptive AUC ({fmt(tab_auc)}) with `final_claim_allowed=false`.
- Pseudo-external evaluation ≠ true external validation.
"""

    results = f"""# Results Scaffold

## Primary Method X-Strict results

- Fixed tabular 5-fold ROC-AUC ≈ **{fmt(tab_auc)}** (with CI in `phase1_baselines_summary`).
- 3D CNN + fusion ROC-AUC ≈ **{fmt(three_d_auc)}**; `final_claim_allowed=false`.
- Cohort-probe 3D latent linear ≈0.50 → embeddings do not encode cohort.

## Incremental value (with validity ranking)

- Highest **descriptive** pooled AUC: `{incr.get('best_descriptive_roc_auc_step')}` — **do not** present as best biological model.
- Validity-safe headline: `{incr.get('validity_safe_headline_step')}`.
- ROI+tabular high AUC coincides with ROI nonlinear cohort-probe ≈ **{fmt(roi_probe)}**.

## WMH / reads

- WMH-only and MRI-reads-only AUC ≈0.50; incremental signal driven by ROI volumes.

## Method X-Cog / SF stress tests

- Report SF-inclusive pooled metrics with coupling caveat.
- Emphasize train-SF→enrolled transfer collapse (~0.51 AUC).

## Continuous centiloid (target modeling)

- Pooled Spearman ~0.44 descriptive; within-A4 ~0.31; within-LEARN ~0.08.

## Post-MVP sensitivities

- Temporal-gap evaluation with bootstrap CIs per window.
- Matched 1:1 subset: full vs matched AUC difference ~0.007 with overlapping CIs.
- Decision curve: report threshold grid and net benefit arrays.

## Method Z and clinical endpoint (separate)

- Method Z blood AUC ~0.81–0.87 with CI; drop-one ablation highlights pTau217.
- Clinical endpoint CDR/PACC AUC ~0.63–0.66 — modest; not AD diagnosis claim.

## Method Y (exploratory)

- N=266 PET-eligible vs ~1700 MRI-enrolled; tier `primary_feasible_90`; class imbalance 232/34; gold_standard_30 forbidden for training.
- **Do not headline Y1 PET-only AUC** (tautology). Headline: MRI incremental value over PET-only (ΔAUC, DCA).
- Method Y framed as co-primary **exploratory** Results subsection only.

## Achieved vs literature contextual ceiling (mandatory table)

| Modality / branch | Literature contextual range (cited) | This study achieved | Notes |
|---|---|---|---|
| T1-only MRI amyloid status | AUC ~0.65–0.74 (ADNI/A4-style) | 3D CV ≈ **{fmt(three_d_auc)}** | `final_claim_allowed=false`; descriptive under coupling |
| Tabular fixed covariates | varies | ≈ **{fmt(tab_auc)}** | Descriptive pooled metric |
| Plasma p-tau217 vs amyloid PET | AUC ~0.95–0.97 (impaired cohorts) | Method Z separate table | Not Method X claim |
| PET-only on paired scans | high (label-related) | Y1 supplementary only | Tautology caveat |
| MRI incremental over PET | primary Method Y question | Y3/Y4 ΔAUC + DCA | Small N limits inference |

Every literature range must cite `results/governance/domain_ceiling_citation_list.json`; never present literature ranges as achieved targets.

## Discussion bullets

1. Validity controls (cohort-probe, claim policy) are central findings.
2. ROI branch is interpretability/sensitivity, not independent proof.
3. Extensions (blood, PET, clinical endpoint) must not upgrade Method X non-invasive claim.
"""

    doc = {
        "stage": "tripod_manuscript_scaffold",
        "status": "DRAFT_SCAFFOLD_EXPANDED",
        "tripod_ai_sections": {
            "title_status": "DRAFT_IN_METHODS",
            "abstract_status": "PENDING_MANUSCRIPT",
            "methods_scaffold_path": "results/governance/tripod_methods_scaffold.md",
            "results_scaffold_path": "results/governance/tripod_results_scaffold.md",
        },
        "claim_policy_refs": [
            "docs/final_execution_program.md",
            "governance/method_y_class_imbalance_policy.md",
            "results/governance/domain_ceiling_citation_list.json",
            "results/final_results/thesis/thesis_grade_report.md",
        ],
    }
    Path("results/governance").mkdir(parents=True, exist_ok=True)
    Path("results/governance/tripod_methods_scaffold.md").write_text(methods, encoding="utf-8")
    Path("results/governance/tripod_results_scaffold.md").write_text(results, encoding="utf-8")
    atomic_write_json("results/governance/tripod_manuscript_scaffold.json", doc)
    atomic_write_text(
        "results/governance/tripod_manuscript_scaffold.md",
        "# TRIPOD Manuscript Scaffold\n\n"
        "- status: `DRAFT_SCAFFOLD_EXPANDED`\n\n"
        "See `tripod_methods_scaffold.md` and `tripod_results_scaffold.md` for section-level draft text.\n",
    )
    StateManager().append_event("tripod_manuscript_scaffold", "COMPLETED", "Phase 4 TRIPOD manuscript scaffold expanded.")
    print(json.dumps({"status": doc["status"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
