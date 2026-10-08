"""Secondary diagnostics from committed predictions; no model reselection/refit."""
from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.xyz_fusion_v1.report import cell_frame, individual_bs


def standardized_difference(first, second):
    a = np.asarray(first, float); b = np.asarray(second, float)
    a = a[np.isfinite(a)]; b = b[np.isfinite(b)]
    if min(len(a), len(b)) < 2:
        return {"status": "NOT_ESTIMABLE", "smd": None}
    denominator = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    if denominator < 1e-12:
        return {"status": "NOT_ESTIMABLE_CONSTANT", "smd": None}
    return {"status": "DESCRIPTIVE", "smd": float((a.mean() - b.mean()) / denominator),
            "available_n_first": len(a), "available_n_second": len(b)}


def selection_diagnostics(frame, manifest, features):
    included = manifest.mri_sample_hash.isin(frame.mri_sample_hash)
    blood = frame.blood_ptau217.notna()
    groups = {"qc_included_vs_excluded": (manifest.loc[included], manifest.loc[~included]),
              "blood_available_vs_missing": (frame.loc[blood], frame.loc[~blood])}
    result = {}
    for name, (a, b) in groups.items():
        result[name] = {"n_first": len(a), "n_second": len(b),
                        "events_first": int(a.event.sum()), "events_second": int(b.event.sum()),
                        "features": {f: standardized_difference(a[f], b[f]) for f in features},
                        "interpretation": "DESCRIPTIVE_SELECTION_NOT_PROOF_OF_MISSING_AT_RANDOM"}
    return result


def statistical_audit(ctx):
    cells, report = ctx["cells"], ctx["report"]
    grid = ctx["config"]["fusion"]["ibs_grid_days"]
    maximum_error = 0.; checked = 0
    for fold in cells:
        for cell in fold["arms"].values():
            tr, te = cell_frame(cell)
            loss, _, _ = individual_bs(tr, te, np.asarray(cell["curves"]), grid)
            error = abs(float(loss.mean()) - cell["metrics"]["ibs"])
            if not np.isfinite(error) or error > 1e-10:
                raise ValueError("REPORTED_IBS_NOT_REPRODUCIBLE_FROM_COMMITTED_CURVES")
            maximum_error = max(maximum_error, error); checked += 1
    score_spread = []
    for outer in range(5):
        values = []
        for path in sorted((ctx['parent']/'private/jobs').glob(f'outer_predictions_{outer}_*.json')):
            import json
            values.extend(json.loads(path.read_text())['payload']['mri_risk'])
        score_spread.append({'fold': outer, 'n': len(values), 'mean': float(np.mean(values)),
                             'sd': float(np.std(values)), 'min': float(np.min(values)),
                             'max': float(np.max(values)), 'constant_score': bool(np.ptp(values) < 1e-8)})
    comparisons = []
    for item in report["paired_comparisons"]:
        ibs = item["delta_ibs_bonferroni_ci"]; uno = item["delta_uno_bonferroni_ci"]
        supported = bool(ibs and uno and ibs[1] < 0 and uno[0] >= 0)
        comparisons.append({"base": item["base"], "augmented": item["augmented"],
                            "simultaneous_ibs_ci": ibs, "simultaneous_uno_ci": uno,
                            "incremental_support_after_multiplicity": supported,
                            "positive_clinical_claim_allowed": False})
    manifest = pd.DataFrame(ctx["contract"]["manifest"])
    return {"status": "COMPLETED_FROM_COMMITTED_PREDICTIONS", "outer_cells_checked": checked,
            "maximum_ibs_reproduction_error": maximum_error, "comparisons": comparisons,
            "mri_score_distribution_by_fold": score_spread,
            "selection": selection_diagnostics(ctx["frame"], manifest,
                ctx["config"]["fusion"]["clinical_features"] + ["amyloid_centiloid"]),
            "claim_status": "POST_HOC_INTERNAL_EXPLORATORY",
            "bootstrap_scope": "CONDITIONAL_ON_FITTED_MODELS_NOT_TRAINING_OR_SELECTION_UNCERTAINTY",
            "unresolved": ["NO_EXTERNAL_DATA", "MRI_SITE_SCANNER_ROBUSTNESS_NOT_ESTABLISHED",
                           "FORMAL_PENALIZED_COX_PH_TEST_NOT_PERFORMED",
                           "MISSING_NOT_AT_RANDOM_CANNOT_BE_EXCLUDED"],
            "primary_training_repeated": False}
