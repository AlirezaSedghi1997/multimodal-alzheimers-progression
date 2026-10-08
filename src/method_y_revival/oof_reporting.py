from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .atomic_io import atomic_write_json
from .contracts import sha256_mapping
from .science_gates import ScienceGateError, paired_oof_bootstrap


class OOFReportingError(RuntimeError):
    """Raised when locked OOF evidence is incomplete or inconsistent."""


def _censoring_survival(reference: list[dict[str, Any]], time: float) -> float:
    times = np.asarray([float(row["time_days"]) for row in reference], dtype=float)
    censor_events = np.asarray([not bool(int(row["event"])) for row in reference])
    survival = 1.0
    for value in np.unique(times[(times <= float(time)) & censor_events]):
        at_risk = int(np.sum(times >= value))
        censored = int(np.sum((times == value) & censor_events))
        if at_risk > 0:
            survival *= 1.0 - censored / at_risk
    return max(float(survival), 1e-6)


def _weighted_logistic_calibration(
    predicted_risk: np.ndarray, outcome: np.ndarray, weights: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    clipped = np.clip(predicted_risk, 1e-6, 1.0 - 1e-6)
    linear = np.log(clipped / (1.0 - clipped))
    design = np.column_stack([np.ones(len(linear)), linear])
    beta = np.zeros(2, dtype=float)
    ridge = np.eye(2) * 1e-6
    for _ in range(100):
        eta = design @ beta
        probability = 1.0 / (1.0 + np.exp(-np.clip(eta, -30.0, 30.0)))
        variance = np.maximum(probability * (1.0 - probability), 1e-8)
        working = eta + (outcome - probability) / variance
        effective = weights * variance
        hessian = design.T @ (effective[:, None] * design) + ridge
        updated = np.linalg.solve(hessian, design.T @ (effective * working))
        if np.max(np.abs(updated - beta)) < 1e-9:
            beta = updated
            break
        beta = updated
    eta = design @ beta
    probability = 1.0 / (1.0 + np.exp(-np.clip(eta, -30.0, 30.0)))
    variance = np.maximum(probability * (1.0 - probability), 1e-8)
    hessian = design.T @ ((weights * variance)[:, None] * design) + ridge
    covariance = np.linalg.inv(hessian)
    return beta, np.sqrt(np.diag(covariance))


def horizon_calibration_from_oof(
    variant: dict[str, Any], *, horizon_days: float, minimum_effective_rows: int
) -> dict[str, Any]:
    predicted: list[float] = []
    outcomes: list[int] = []
    weights: list[float] = []
    censored_before_horizon = 0
    for fold in variant.get("folds", []):
        grid = np.asarray(fold["evaluation_grid_days"], dtype=float)
        reference = list(fold["censoring_reference"])
        if float(horizon_days) < float(grid.min()) or float(horizon_days) > float(grid.max()):
            continue
        for row in fold["oof_predictions"]:
            time = float(row["time_days"])
            event = bool(int(row["event"]))
            survival = float(
                np.interp(
                    float(horizon_days),
                    grid,
                    np.asarray(row["survival_probabilities"], dtype=float),
                )
            )
            if event and time <= float(horizon_days):
                outcome = 1
                weight = 1.0 / _censoring_survival(reference, time)
            elif time > float(horizon_days):
                outcome = 0
                weight = 1.0 / _censoring_survival(reference, float(horizon_days))
            else:
                censored_before_horizon += 1
                continue
            predicted.append(1.0 - survival)
            outcomes.append(outcome)
            weights.append(weight)
    if len(predicted) < int(minimum_effective_rows) or len(set(outcomes)) < 2:
        return {
            "status": "NOT_ESTIMABLE",
            "horizon_days": float(horizon_days),
            "effective_row_count": len(predicted),
            "censored_before_horizon_count": censored_before_horizon,
            "recalibration_applied": False,
        }
    risk = np.asarray(predicted, dtype=float)
    y = np.asarray(outcomes, dtype=float)
    weight = np.asarray(weights, dtype=float)
    beta, standard_error = _weighted_logistic_calibration(risk, y, weight)
    bins: list[dict[str, Any]] = []
    quantiles = np.unique(np.quantile(risk, np.linspace(0.0, 1.0, 11)))
    if len(quantiles) >= 3:
        bin_index = np.digitize(risk, quantiles[1:-1], right=True)
        for index in sorted(set(bin_index.tolist())):
            mask = bin_index == index
            bins.append(
                {
                    "bin": int(index),
                    "n": int(mask.sum()),
                    "mean_predicted_risk": float(np.average(risk[mask], weights=weight[mask])),
                    "weighted_observed_event_rate": float(
                        np.average(y[mask], weights=weight[mask])
                    ),
                }
            )
    return {
        "status": "ESTIMABLE",
        "method": "IPCW_WEIGHTED_LOGISTIC_CALIBRATION",
        "horizon_days": float(horizon_days),
        "effective_row_count": len(risk),
        "event_count": int(y.sum()),
        "censored_before_horizon_count": censored_before_horizon,
        "calibration_intercept": float(beta[0]),
        "calibration_intercept_95ci": [
            float(beta[0] - 1.959963984540054 * standard_error[0]),
            float(beta[0] + 1.959963984540054 * standard_error[0]),
        ],
        "calibration_slope": float(beta[1]),
        "calibration_slope_95ci": [
            float(beta[1] - 1.959963984540054 * standard_error[1]),
            float(beta[1] + 1.959963984540054 * standard_error[1]),
        ],
        "calibration_bins": bins,
        "recalibration_applied": False,
    }


def build_subgroup_metadata(
    *, endpoint: pd.DataFrame, scanner_linkage: pd.DataFrame
) -> pd.DataFrame:
    required = {"patient_hash", "age", "sex_binary", "treatment_active", "event"}
    missing = sorted(required - set(endpoint.columns))
    if missing:
        raise OOFReportingError(f"MISSING_SUBGROUP_COLUMNS:{missing}")
    scanner = scanner_linkage.loc[:, ["patient_hash", "manufacturer"]].drop_duplicates(
        "patient_hash"
    )
    result = endpoint.loc[:, sorted(required)].drop_duplicates("patient_hash").merge(
        scanner, on="patient_hash", how="left", validate="one_to_one"
    )
    result["manufacturer"] = result["manufacturer"].fillna("UNKNOWN").astype(str)
    result["age_group"] = pd.cut(
        result["age"].astype(float),
        bins=[-np.inf, 70.0, 75.0, np.inf],
        labels=["LT70", "70_TO_74", "GE75"],
        right=False,
    ).astype(str)
    result.attrs["unknown_scanner_imputed"] = False
    return result


def _filter_variant(variant: dict[str, Any], members: set[str]) -> dict[str, Any]:
    folds = []
    for fold in variant.get("folds", []):
        folds.append(
            {
                **fold,
                "oof_predictions": [
                    row
                    for row in fold["oof_predictions"]
                    if str(row["patient_hash"]) in members
                ],
            }
        )
    return {**variant, "folds": folds}


def subgroup_incremental_report(
    *,
    base_variant: dict[str, Any],
    augmented_variant: dict[str, Any],
    metadata: pd.DataFrame,
    subgroup_columns: list[str],
    minimum_subjects: int,
    minimum_events: int,
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    all_base_rows = {
        str(row["patient_hash"]): row
        for fold in base_variant.get("folds", [])
        for row in fold["oof_predictions"]
    }
    reports: dict[str, Any] = {}
    for column_index, column in enumerate(subgroup_columns):
        if column not in metadata.columns:
            reports[column] = {"status": "NOT_AVAILABLE"}
            continue
        column_reports: dict[str, Any] = {}
        for value_index, value in enumerate(sorted(metadata[column].dropna().astype(str).unique())):
            members = set(
                metadata.loc[metadata[column].astype(str) == value, "patient_hash"].astype(str)
            ) & set(all_base_rows)
            events = sum(int(all_base_rows[subject]["event"]) for subject in members)
            if len(members) < int(minimum_subjects) or events < int(minimum_events):
                column_reports[value] = {
                    "status": "NOT_ESTIMABLE_INSUFFICIENT_SUPPORT",
                    "subject_count": len(members),
                    "event_count": events,
                    "positive_subgroup_claim_allowed": False,
                }
                continue
            filtered_base = _filter_variant(base_variant, members)
            filtered_augmented = _filter_variant(augmented_variant, members)
            if any(len(fold["oof_predictions"]) < 2 for fold in filtered_base["folds"]):
                column_reports[value] = {
                    "status": "NOT_ESTIMABLE_EMPTY_OUTER_FOLD",
                    "subject_count": len(members),
                    "event_count": events,
                    "positive_subgroup_claim_allowed": False,
                }
                continue
            try:
                bootstrap = paired_oof_bootstrap(
                    filtered_base,
                    filtered_augmented,
                    iterations=int(iterations),
                    seed=int(seed) + column_index * 1000 + value_index,
                )
                column_reports[value] = {
                    "status": "ESTIMABLE",
                    "subject_count": len(members),
                    "event_count": events,
                    "delta_ibs_mean": bootstrap["delta_ibs_mean"],
                    "delta_ibs_95ci": bootstrap["delta_ibs_95ci"],
                    "delta_cindex_mean": bootstrap["delta_cindex_mean"],
                    "delta_cindex_95ci": bootstrap["delta_cindex_95ci"],
                    "positive_subgroup_claim_allowed": False,
                }
            except ScienceGateError as exc:
                column_reports[value] = {
                    "status": "NOT_ESTIMABLE_BOOTSTRAP_FAILURE",
                    "reason": str(exc),
                    "subject_count": len(members),
                    "event_count": events,
                    "positive_subgroup_claim_allowed": False,
                }
        reports[column] = column_reports
    return {
        "status": "COMPLETED_WITH_SUPPORT_GATES",
        "multiplicity_status": "DESCRIPTIVE_NO_POSITIVE_SUBGROUP_CLAIMS",
        "bootstrap_iterations": int(iterations),
        "subgroups": reports,
    }


def write_oof_report(path: str | Path, payload: dict[str, Any]) -> dict[str, Any]:
    report = dict(payload)
    report["report_sha256"] = sha256_mapping(report)
    atomic_write_json(path, report)
    return report
