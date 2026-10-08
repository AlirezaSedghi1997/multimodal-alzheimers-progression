"""Paired, patient-unit exploratory analysis of locked MX-05 add-on predictions.

Each bootstrap draw samples patients once and carries all seeds/arms for those
patients together.  Seed repetitions are averaged within a draw, never stacked
as independent observations.  The input panel is private; output is aggregate.
"""
from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping, Sequence

import numpy as np
from scipy.special import expit, logit
from scipy.stats import rankdata

SEEDS = (13, 29, 47)
ARMS = {
    "hybrid_selected": SEEDS,
    "hybrid_baseline": SEEDS,
    "hybrid_dense": SEEDS,
    "tabular_only": SEEDS,
    "prevalence": SEEDS,
    "mri_only": (13,),
    "random_label": (13,),
}
COMPARISONS = (
    ("hybrid_selected", "tabular_only", SEEDS),
    ("hybrid_selected", "prevalence", SEEDS),
    ("hybrid_selected", "mri_only", (13,)),
    ("hybrid_selected", "random_label", (13,)),
)
METRICS = ("auroc", "auprc", "brier", "calibration_intercept", "calibration_slope")


def validate_panel(rows: Sequence[Mapping], *, expected_n: int = 1336) -> list[dict]:
    if len(rows) != expected_n or expected_n < 2:
        raise ValueError("EXTENSION_PANEL_PATIENT_COUNT_INVALID")
    patients = set()
    normalized = []
    for row in rows:
        patient = str(row.get("patient_hash") or "")
        label, fold = row.get("label"), row.get("fold")
        gap = row.get("gap_days")
        if (not patient or patient in patients or type(label) is not int or label not in (0, 1)
            or type(fold) is not int or fold not in range(5)
            or type(gap) not in (int, float) or not math.isfinite(gap) or gap < 0):
            raise ValueError("EXTENSION_PANEL_ID_LABEL_FOLD_OR_GAP_INVALID")
        patients.add(patient)
        predictions = row.get("predictions")
        if not isinstance(predictions, Mapping) or set(predictions) != set(ARMS):
            raise ValueError("EXTENSION_PANEL_ARMS_INCOMPLETE")
        clean = {}
        for arm, seeds in ARMS.items():
            values = predictions[arm]
            if not isinstance(values, Mapping) or set(values) != {str(seed) for seed in seeds}:
                raise ValueError(f"EXTENSION_PANEL_SEEDS_INCOMPLETE:{arm}")
            clean[arm] = {}
            for seed in seeds:
                probability = values[str(seed)]
                if (type(probability) not in (int, float)
                    or not math.isfinite(probability) or not 0 <= probability <= 1):
                    raise ValueError(f"EXTENSION_PANEL_PROBABILITY_INVALID:{arm}:{seed}")
                clean[arm][seed] = float(probability)
        normalized.append({"patient_hash": patient, "label": label, "fold": fold,
                           "gap_days": float(gap), "predictions": clean})
    if {row["label"] for row in normalized} != {0, 1}:
        raise ValueError("EXTENSION_PANEL_ONE_CLASS")
    if {row["fold"] for row in normalized} != set(range(5)):
        raise ValueError("EXTENSION_PANEL_FOLD_COVERAGE_INVALID")
    return normalized


def calibration(y: np.ndarray, probability: np.ndarray) -> tuple[float, float] | None:
    """Unpenalized logistic recalibration intercept and slope, or not estimable."""
    if len(set(y.tolist())) != 2:
        return None
    z = logit(np.clip(probability, 1e-6, 1 - 1e-6))
    design = np.column_stack((np.ones(len(z)), z))
    if np.linalg.matrix_rank(design) != 2:
        return None
    coefficient = np.asarray([0.0, 1.0], dtype=np.float64)
    for _ in range(50):
        fitted = expit(design @ coefficient)
        weights = np.clip(fitted * (1 - fitted), 1e-12, None)
        hessian = design.T @ (design * weights[:, None])
        gradient = design.T @ (fitted - y)
        if not np.all(np.isfinite(hessian)) or np.linalg.cond(hessian) > 1e12:
            return None
        try:
            change = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:
            return None
        coefficient -= change
        if not np.all(np.isfinite(coefficient)) or np.max(np.abs(coefficient)) > 100:
            return None
        if np.max(np.abs(change)) < 1e-7:
            return float(coefficient[0]), float(coefficient[1])
    return None


def _metrics(y: np.ndarray, probability: np.ndarray) -> dict[str, float | None]:
    if len(set(y.tolist())) != 2:
        return {name: None for name in METRICS}
    positive_n = int(y.sum())
    negative_n = len(y) - positive_n
    ranks = rankdata(probability, method="average")
    auroc = float((ranks[y == 1].sum() - positive_n * (positive_n + 1) / 2)
                  / (positive_n * negative_n))
    order = np.argsort(-probability, kind="stable")
    sorted_p, sorted_y = probability[order], y[order]
    ends = np.r_[np.flatnonzero(np.diff(sorted_p) != 0), len(y) - 1]
    cumulative_positive = np.cumsum(sorted_y)[ends]
    group_positive = np.diff(np.r_[0, cumulative_positive])
    precision = cumulative_positive / (ends + 1)
    auprc = float(np.sum(group_positive * precision) / positive_n)
    calibrated = calibration(y, probability)
    return {
        "auroc": auroc,
        "auprc": auprc,
        "brier": float(np.mean(np.square(probability - y))),
        "calibration_intercept": calibrated[0] if calibrated else None,
        "calibration_slope": calibrated[1] if calibrated else None,
    }


def _seed_mean_metrics(y: np.ndarray, panel: list[dict], indices: np.ndarray,
                       arm: str, seeds: tuple[int, ...]) -> dict[str, float | None]:
    by_seed = []
    for seed in seeds:
        probability = np.asarray([panel[i]["predictions"][arm][seed] for i in indices],
                                 dtype=np.float64)
        by_seed.append(_metrics(y, probability))
    result = {}
    for name in METRICS:
        values = [row[name] for row in by_seed]
        result[name] = (float(np.mean(values)) if all(value is not None for value in values)
                        else None)
    return result


def _effect(y: np.ndarray, panel: list[dict], indices: np.ndarray,
            first: str, second: str, seeds: tuple[int, ...]) -> dict[str, float | None]:
    a = _seed_mean_metrics(y, panel, indices, first, seeds)
    b = _seed_mean_metrics(y, panel, indices, second, seeds)
    return {name: (a[name] - b[name] if a[name] is not None and b[name] is not None
                   else None) for name in METRICS}


def analyze_panel(rows: Sequence[Mapping], *, expected_n: int = 1336,
                  resamples: int = 2000, bootstrap_seed: int = 20260927) -> dict:
    if resamples < 1000 or type(bootstrap_seed) is not int:
        raise ValueError("EXTENSION_BOOTSTRAP_CONTRACT_INVALID")
    panel = validate_panel(rows, expected_n=expected_n)
    output = {"schema_version": 1, "status": "COMPLETED_INTERNAL_EXPLORATORY",
              "patient_unit": "patient_hash", "seed_repetitions_independent": False,
              "external_validation": False, "strong_disease_or_region_claim_allowed": False,
              "resamples_requested": resamples, "bootstrap_seed": bootstrap_seed,
              "all_patient_count": len(panel), "subsets": {}}
    for window in (None, 90, 180, 365):
        subset = [row for row in panel if window is None or row["gap_days"] <= window]
        label_counts = Counter(row["label"] for row in subset)
        key = "all" if window is None else str(window)
        report = {"n": len(subset), "positive_n": label_counts[1],
                  "negative_n": label_counts[0], "fold_counts": {
                      str(fold): count for fold, count in sorted(Counter(
                          row["fold"] for row in subset).items())}, "comparisons": {}}
        if min(label_counts[0], label_counts[1]) < 2:
            report["status"] = "NOT_ESTIMABLE_TWO_CLASSES_REQUIRED"
            output["subsets"][key] = report
            continue
        y_full = np.asarray([row["label"] for row in subset], dtype=np.int64)
        indices_full = np.arange(len(subset))
        point = {(a, b): _effect(y_full, subset, indices_full, a, b, seeds)
                 for a, b, seeds in COMPARISONS}
        report["arm_metrics"] = {
            arm: {"seed_set": list(seeds),
                  "metrics": _seed_mean_metrics(y_full, subset, indices_full, arm, seeds)}
            for arm, seeds in ARMS.items()
        }
        draws = {(a, b): {name: [] for name in METRICS} for a, b, _ in COMPARISONS}
        rng = np.random.default_rng(bootstrap_seed + (0 if window is None else window))
        for _ in range(resamples):
            indices = rng.integers(0, len(subset), len(subset))
            yy = y_full[indices]
            if len(set(yy.tolist())) != 2:
                continue
            for first, second, seeds in COMPARISONS:
                effect = _effect(yy, subset, indices, first, second, seeds)
                for name, value in effect.items():
                    if value is not None and math.isfinite(value):
                        draws[(first, second)][name].append(value)
        for first, second, seeds in COMPARISONS:
            metric_rows = {}
            for name in METRICS:
                values = draws[(first, second)][name]
                observed = point[(first, second)][name]
                metric_rows[name] = {
                    "selected_minus_comparator": observed,
                    "patient_bootstrap_ci95": ([float(v) for v in np.quantile(values, [0.025, 0.975])]
                                               if observed is not None and len(values) >= 0.9 * resamples
                                               else None),
                    "valid_resamples": len(values),
                    "status": ("ESTIMATED" if observed is not None
                               and len(values) >= 0.9 * resamples else "NOT_ESTIMABLE"),
                }
            report["comparisons"][f"{first}_minus_{second}"] = {
                "seed_set": list(seeds), "metrics": metric_rows}
        report["status"] = "DESCRIPTIVE_POST_HOC_INTERNAL"
        output["subsets"][key] = report
    return output
