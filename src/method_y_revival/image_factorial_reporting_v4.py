from __future__ import annotations

import math
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .atomic_io import atomic_write_json
from .image_reporting_v3 import privacy_scan_reports
from .image_robustness_v3 import leave_manufacturer_out_eligibility
from .oof_reporting import horizon_calibration_from_oof, subgroup_incremental_report
from .science_gates import cached_paired_oof_bootstrap


class ImageFactorialReportingV4Error(RuntimeError):
    pass


def _scaled_oof_scanner_probe(
    embeddings: np.ndarray, manufacturers: list[str], *, seed: int
) -> dict[str, Any]:
    """Fold-local scaled scanner audit; never a primary prediction model."""

    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import LabelEncoder, StandardScaler

    values = np.asarray(embeddings, dtype=float)
    if values.ndim != 2 or values.shape[0] != len(manufacturers):
        raise ImageFactorialReportingV4Error(
            "EMBEDDING_MANUFACTURER_LENGTH_MISMATCH"
        )
    if not np.isfinite(values).all():
        raise ImageFactorialReportingV4Error("SCANNER_PROBE_EMBEDDING_NONFINITE")
    encoder = LabelEncoder()
    labels = encoder.fit_transform(np.asarray(manufacturers, dtype=str))
    unique, counts = np.unique(labels, return_counts=True)
    if len(unique) < 2:
        return {"status": "NOT_ESTIMABLE", "reason": "INSUFFICIENT_CLASSES"}
    if int(counts.min()) < 5:
        return {
            "status": "NOT_ESTIMABLE",
            "reason": "INSUFFICIENT_PER_CLASS_SUPPORT",
            "per_class_support": dict(
                zip(encoder.classes_.tolist(), counts.astype(int).tolist(), strict=True)
            ),
        }
    n_splits = min(5, int(counts.min()))

    def evaluate(target: np.ndarray, random_state: int) -> tuple[list[float], int]:
        splitter = StratifiedKFold(
            n_splits=n_splits, shuffle=True, random_state=int(random_state)
        )
        scores: list[float] = []
        warning_count = 0
        for train_index, test_index in splitter.split(values, target):
            pipeline = make_pipeline(
                StandardScaler(),
                LogisticRegression(
                    max_iter=2000,
                    solver="lbfgs",
                    random_state=int(random_state),
                ),
            )
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", ConvergenceWarning)
                pipeline.fit(values[train_index], target[train_index])
            warning_count += sum(
                issubclass(item.category, ConvergenceWarning) for item in caught
            )
            probability = pipeline.predict_proba(values[test_index])
            if len(unique) == 2:
                score = roc_auc_score(target[test_index], probability[:, 1])
            else:
                score = roc_auc_score(
                    target[test_index],
                    probability,
                    labels=unique,
                    multi_class="ovr",
                    average="macro",
                )
            scores.append(float(score))
        return scores, warning_count

    scores, convergence_warnings = evaluate(labels, int(seed))
    rng = np.random.default_rng(int(seed))
    permutation_scores, permutation_warnings = evaluate(
        rng.permutation(labels), int(seed) + 1
    )
    if convergence_warnings or permutation_warnings:
        return {
            "status": "FAILED_CONVERGENCE",
            "convergence_warning_count": int(
                convergence_warnings + permutation_warnings
            ),
            "preprocessing": "STANDARD_SCALER_FIT_WITHIN_PROBE_FOLD",
        }
    if not scores or not np.isfinite(scores).all():
        return {"status": "NOT_ESTIMABLE", "reason": "NONFINITE_PROBE_AUC"}
    return {
        "status": "COMPLETED",
        "macro_auc": float(np.mean(scores)),
        "fold_aucs": [float(value) for value in scores],
        "permutation_null_auc": float(np.mean(permutation_scores)),
        "per_class_support": dict(
            zip(encoder.classes_.tolist(), counts.astype(int).tolist(), strict=True)
        ),
        "fold_count": int(n_splits),
        "preprocessing": "STANDARD_SCALER_FIT_WITHIN_PROBE_FOLD",
        "convergence_warning_count": 0,
        "embedding_source": "OOF_LOCKED_MODEL_ONLY",
        "scanner_is_primary_predictor": False,
    }


def _scanner_report_v4(
    *, variant: dict[str, Any], metadata: pd.DataFrame, seed: int
) -> dict[str, Any]:
    required = {"patient_hash", "manufacturer"}
    if not required.issubset(metadata.columns):
        raise ImageFactorialReportingV4Error("SCANNER_METADATA_COLUMNS_MISSING")
    normalized = metadata.loc[:, ["patient_hash", "manufacturer"]].copy()
    normalized["patient_hash"] = normalized["patient_hash"].astype(str)
    if normalized["patient_hash"].duplicated().any():
        raise ImageFactorialReportingV4Error("SCANNER_METADATA_PATIENT_DUPLICATE")
    normalized["manufacturer"] = (
        normalized["manufacturer"].fillna("UNKNOWN").astype(str).str.upper()
    )
    normalized = normalized.set_index("patient_hash")
    predictions = {
        str(row["patient_hash"]): row
        for fold in variant.get("folds", [])
        for row in fold.get("oof_predictions", [])
    }
    embeddings = {
        str(row["patient_hash"]): row.get("embedding", [])
        for fold in variant.get("folds", [])
        for row in fold.get("oof_embeddings", [])
    }
    common = sorted(set(predictions).intersection(normalized.index))
    support: dict[str, dict[str, int]] = {}
    for subject in common:
        manufacturer = str(normalized.loc[subject, "manufacturer"])
        cell = support.setdefault(manufacturer, {"subjects": 0, "events": 0})
        cell["subjects"] += 1
        cell["events"] += int(predictions[subject]["event"])
    embedding_common = sorted(set(embeddings).intersection(normalized.index))
    if embedding_common:
        probe = _scaled_oof_scanner_probe(
            np.asarray([embeddings[subject] for subject in embedding_common], dtype=float),
            [str(normalized.loc[subject, "manufacturer"]) for subject in embedding_common],
            seed=int(seed),
        )
    else:
        probe = {"status": "NOT_ESTIMABLE", "reason": "OOF_EMBEDDINGS_UNAVAILABLE"}
    unknown_count = int(support.get("UNKNOWN", {}).get("subjects", 0))
    return {
        "status": "COMPLETED_WITH_CLAIM_GATE_UNRESOLVED",
        "manufacturer_support": support,
        "leave_manufacturer_out_eligibility": leave_manufacturer_out_eligibility(
            support
        ),
        "leave_manufacturer_out_performance": {
            "status": "NOT_ESTIMABLE_NOT_EXECUTED_AS_LOCKED_TRAINING_DESIGN"
        },
        "oof_scanner_probe": probe,
        "unknown_scanner_excluded_sensitivity": {
            "status": "ELIGIBILITY_COUNTED",
            "unknown_excluded_count": unknown_count,
            "remaining_count": int(len(common) - unknown_count),
        },
        "scanner_robustness_passed": False,
        "terminal_reason": "LEAVE_MANUFACTURER_OUT_PERFORMANCE_NOT_ESTABLISHED",
    }


def factorial_pair_acceptance_gate(
    *,
    delta_ibs_ci: tuple[float, float],
    delta_uno_ci: tuple[float, float],
    fold_delta_ibs: list[float],
    calibration_not_worse: bool,
    selected_epoch_nll_gap_delta: float | None,
    scanner_probe_not_worse: bool,
    leave_manufacturer_out_performance_established: bool,
    margins: dict[str, Any],
) -> dict[str, Any]:
    """Prespecified internal operational gate; never upgrades scientific claims."""

    blockers: list[str] = []
    ibs_margin = float(margins["ibs_noninferiority_margin_absolute"])
    uno_margin = float(margins["uno_noninferiority_margin_absolute"])
    gap_margin = float(margins["selected_epoch_nll_gap_increase_tolerance"])
    if len(fold_delta_ibs) != 5:
        blockers.append("INCOMPLETE_FIVE_FOLD_COMPARISON")
    if not all(math.isfinite(float(value)) for value in (*delta_ibs_ci, *delta_uno_ci)):
        blockers.append("NONFINITE_PAIRED_CONFIDENCE_INTERVAL")
    elif float(delta_ibs_ci[1]) > ibs_margin:
        blockers.append("IBS_NOT_NONINFERIOR")
    if math.isfinite(float(delta_uno_ci[0])) and float(delta_uno_ci[0]) < -uno_margin:
        blockers.append("UNO_CINDEX_NOT_NONINFERIOR")
    if bool(margins.get("require_all_five_outer_folds_ibs_noninferior", True)) and (
        len(fold_delta_ibs) != 5
        or any(float(value) > ibs_margin for value in fold_delta_ibs)
    ):
        blockers.append("IBS_NONINFERIORITY_NOT_STABLE_IN_ALL_FOLDS")
    if not calibration_not_worse:
        blockers.append("CALIBRATION_NOT_WORSE_NOT_DEMONSTRATED")
    if selected_epoch_nll_gap_delta is None or not math.isfinite(
        float(selected_epoch_nll_gap_delta)
    ) or float(selected_epoch_nll_gap_delta) > gap_margin:
        blockers.append("TRAIN_VALIDATION_GAP_WORSENED")
    if not scanner_probe_not_worse:
        blockers.append("SCANNER_PROBE_NOT_WORSE_NOT_DEMONSTRATED")
    if bool(margins.get("require_leave_manufacturer_out_performance", True)) and not (
        leave_manufacturer_out_performance_established
    ):
        blockers.append("LEAVE_MANUFACTURER_OUT_PERFORMANCE_NOT_ESTABLISHED")
    return {
        "gate_role": "EXPLORATORY_OPERATIONAL_MODEL_COMPARISON",
        "exploratory_operational_acceptance": not blockers,
        "blockers": blockers,
        "delta_ibs_95ci": [float(value) for value in delta_ibs_ci],
        "delta_uno_cindex_95ci": [float(value) for value in delta_uno_ci],
        "fold_delta_ibs": [float(value) for value in fold_delta_ibs],
        "calibration_not_worse": bool(calibration_not_worse),
        "selected_epoch_nll_gap_delta": (
            float(selected_epoch_nll_gap_delta)
            if selected_epoch_nll_gap_delta is not None
            else None
        ),
        "scanner_probe_not_worse": bool(scanner_probe_not_worse),
        "leave_manufacturer_out_performance_established": bool(
            leave_manufacturer_out_performance_established
        ),
        "positive_claim_allowed": False,
        "external_claim_allowed": False,
        "claim_status": "POST_HOC_INTERNAL_EXPLORATORY",
    }


def _validate_matched_pair(
    reference: dict[str, Any], challenger: dict[str, Any], *, expected_folds: int
) -> None:
    expected = set(range(int(expected_folds)))
    reference_folds = {int(row.get("fold", -1)): row for row in reference.get("folds", [])}
    challenger_folds = {
        int(row.get("fold", -1)): row for row in challenger.get("folds", [])
    }
    if set(reference_folds) != expected or set(challenger_folds) != expected:
        raise ImageFactorialReportingV4Error("FAILED_INCOMPLETE_MATCHED_OUTER_CV")
    for fold in expected:
        reference_rows = {
            str(row.get("patient_hash")): row
            for row in reference_folds[fold].get("oof_predictions", [])
        }
        challenger_rows = {
            str(row.get("patient_hash")): row
            for row in challenger_folds[fold].get("oof_predictions", [])
        }
        if not reference_rows or set(reference_rows) != set(challenger_rows):
            raise ImageFactorialReportingV4Error("MATCHED_OOF_MEMBERSHIP_MISMATCH")
        for subject in reference_rows:
            left, right = reference_rows[subject], challenger_rows[subject]
            if int(left["event"]) != int(right["event"]) or float(
                left["time_days"]
            ) != float(right["time_days"]):
                raise ImageFactorialReportingV4Error("MATCHED_OOF_OUTCOME_MISMATCH")


def _calibration_comparison(
    *,
    reference_arm: str,
    challenger_arm: str,
    reference: dict[str, Any],
    challenger: dict[str, Any],
    horizons_days: list[float],
    tolerance: float,
) -> tuple[dict[str, Any], bool]:
    rows: dict[str, Any] = {}
    decisions: list[bool] = []
    for horizon in horizons_days:
        left = horizon_calibration_from_oof(
            reference, horizon_days=float(horizon), minimum_effective_rows=10
        )
        right = horizon_calibration_from_oof(
            challenger, horizon_days=float(horizon), minimum_effective_rows=10
        )
        row: dict[str, Any] = {
            "horizon_days": float(horizon),
            "reference_arm": reference_arm,
            "challenger_arm": challenger_arm,
            "reference": left,
            "challenger": right,
            "recalibration_applied": False,
        }
        if left.get("status") == "ESTIMABLE" and right.get("status") == "ESTIMABLE":
            reference_deviation = max(
                abs(float(left["calibration_intercept"])),
                abs(float(left["calibration_slope"]) - 1.0),
            )
            challenger_deviation = max(
                abs(float(right["calibration_intercept"])),
                abs(float(right["calibration_slope"]) - 1.0),
            )
            not_worse = challenger_deviation <= reference_deviation + float(tolerance)
            row.update(
                {
                    "status": "ESTIMABLE",
                    "reference_max_absolute_deviation": reference_deviation,
                    "challenger_max_absolute_deviation": challenger_deviation,
                    "challenger_not_worse": bool(not_worse),
                }
            )
            decisions.append(bool(not_worse))
        else:
            row.update(
                {
                    "status": "NOT_ESTIMABLE",
                    "challenger_not_worse": False,
                }
            )
        rows[str(float(horizon))] = row
    return (
        {
            "status": "COMPLETED_WITH_ESTIMABILITY_GATES",
            "comparison_semantics": "REFERENCE_ARM_VS_CHALLENGER_ARM",
            "horizons": rows,
        },
        bool(decisions) and all(decisions),
    )


def _scanner_probe_not_worse(
    reference_report: dict[str, Any],
    challenger_report: dict[str, Any],
    *,
    tolerance: float,
) -> bool:
    left = reference_report.get("oof_scanner_probe", {})
    right = challenger_report.get("oof_scanner_probe", {})
    if left.get("status") != "COMPLETED" or right.get("status") != "COMPLETED":
        return False
    left_auc = float(left.get("macro_auc", float("nan")))
    right_auc = float(right.get("macro_auc", float("nan")))
    return bool(
        math.isfinite(left_auc)
        and math.isfinite(right_auc)
        and right_auc <= left_auc + float(tolerance)
    )


def build_factorial_pair_reports(
    *,
    output_root: str | Path,
    branch: str,
    pair_kind: str,
    reference_arm: str,
    challenger_arm: str,
    reference: dict[str, Any],
    challenger: dict[str, Any],
    metadata: pd.DataFrame,
    horizons_days: list[float],
    bootstrap_iterations: int,
    seed: int,
    margins: dict[str, Any],
    expected_outer_folds: int = 5,
) -> dict[str, Any]:
    """Build honest generic arm-comparison reports from locked matched OOF rows."""

    _validate_matched_pair(reference, challenger, expected_folds=expected_outer_folds)
    reports = Path(output_root) / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    identity = {
        "schema_version": "4.0",
        "branch": branch,
        "pair_kind": pair_kind,
        "reference_arm": reference_arm,
        "challenger_arm": challenger_arm,
        "comparison_semantics": "MATCHED_IMAGE_PLUS_TABULAR_ARM_COMPARISON",
        "not_an_image_incremental_value_analysis": True,
        "analysis_role": "POST_HOC_INTERNAL_EXPLORATORY",
    }
    atomic_write_json(reports / f"{branch}_pair_identity.json", identity)

    calibration, calibration_not_worse = _calibration_comparison(
        reference_arm=reference_arm,
        challenger_arm=challenger_arm,
        reference=reference,
        challenger=challenger,
        horizons_days=horizons_days,
        tolerance=float(margins["calibration_deviation_tolerance_absolute"]),
    )
    atomic_write_json(reports / f"{branch}_calibration.json", calibration)
    subgroups = subgroup_incremental_report(
        base_variant=reference,
        augmented_variant=challenger,
        metadata=metadata,
        subgroup_columns=["sex_binary", "age_group", "treatment_active", "manufacturer"],
        minimum_subjects=30,
        minimum_events=10,
        iterations=int(bootstrap_iterations),
        seed=int(seed),
    )
    subgroups["comparison_semantics"] = "REFERENCE_ARM_VS_CHALLENGER_ARM"
    atomic_write_json(reports / f"{branch}_subgroups.json", subgroups)

    reference_scanner = _scanner_report_v4(
        variant=reference, metadata=metadata, seed=int(seed)
    )
    challenger_scanner = _scanner_report_v4(
        variant=challenger, metadata=metadata, seed=int(seed)
    )
    scanner_not_worse = _scanner_probe_not_worse(
        reference_scanner,
        challenger_scanner,
        tolerance=float(margins["scanner_probe_auc_increase_tolerance"]),
    )
    lmo_established = bool(
        reference_scanner.get("scanner_robustness_passed") is True
        and challenger_scanner.get("scanner_robustness_passed") is True
    )
    scanner = {
        "status": "COMPLETED_WITH_CLAIM_GATE_UNRESOLVED",
        "reference_arm": reference_arm,
        "challenger_arm": challenger_arm,
        "reference": reference_scanner,
        "challenger": challenger_scanner,
        "scanner_probe_not_worse": scanner_not_worse,
        "leave_manufacturer_out_performance_established": lmo_established,
    }
    atomic_write_json(reports / f"{branch}_scanner_comparison.json", scanner)

    bootstrap = cached_paired_oof_bootstrap(
        reference,
        challenger,
        output_path=reports / f"{branch}_paired_oof_bootstrap.json",
        iterations=int(bootstrap_iterations),
        seed=int(seed),
    )
    reference_folds = sorted(reference["folds"], key=lambda row: int(row["fold"]))
    challenger_folds = sorted(challenger["folds"], key=lambda row: int(row["fold"]))
    fold_delta_ibs = [
        float(right["integrated_brier_score"])
        - float(left["integrated_brier_score"])
        for left, right in zip(reference_folds, challenger_folds, strict=True)
    ]
    reference_gaps = np.asarray(
        reference.get("inner_selected_epoch_nll_gaps", []), dtype=float
    )
    challenger_gaps = np.asarray(
        challenger.get("inner_selected_epoch_nll_gaps", []), dtype=float
    )
    if (
        reference_gaps.size != 15
        or challenger_gaps.size != 15
        or not np.isfinite(reference_gaps).all()
        or not np.isfinite(challenger_gaps).all()
    ):
        nll_gap_delta = None
    else:
        nll_gap_delta = float(challenger_gaps.mean() - reference_gaps.mean())
    gate = factorial_pair_acceptance_gate(
        delta_ibs_ci=tuple(bootstrap["delta_ibs_95ci"]),
        delta_uno_ci=tuple(bootstrap["delta_cindex_95ci"]),
        fold_delta_ibs=fold_delta_ibs,
        calibration_not_worse=calibration_not_worse,
        selected_epoch_nll_gap_delta=nll_gap_delta,
        scanner_probe_not_worse=scanner_not_worse,
        leave_manufacturer_out_performance_established=lmo_established,
        margins=margins,
    )
    gate.update(
        {
            "branch": branch,
            "pair_kind": pair_kind,
            "reference_arm": reference_arm,
            "challenger_arm": challenger_arm,
            "quantitative_equivalence_proven": False,
            "locked_external_validation_passed": False,
        }
    )
    atomic_write_json(reports / f"{branch}_factorial_acceptance.json", gate)
    claim = {
        "schema_version": "4.0",
        "branch": branch,
        "pair_kind": pair_kind,
        "claim_status": "POST_HOC_INTERNAL_EXPLORATORY",
        "positive_claim_allowed": False,
        "external_claim_allowed": False,
        "official_centiloid_equivalence_claim_allowed": False,
        "official_centaur_equivalence_claim_allowed": False,
        "exploratory_operational_acceptance": gate[
            "exploratory_operational_acceptance"
        ],
        "hard_blockers": [
            "POST_HOC_OUTER_TEST_PREVIOUSLY_EXPOSED",
            "QUANTITATIVE_EQUIVALENCE_NOT_ESTABLISHED",
            "SCANNER_ROBUSTNESS_NOT_ESTABLISHED",
            "LOCKED_EXTERNAL_VALIDATION_MISSING",
        ],
    }
    atomic_write_json(reports / f"{branch}_CLAIM_POLICY.json", claim)
    privacy = privacy_scan_reports(reports)
    if privacy.get("status") != "PASS":
        raise ImageFactorialReportingV4Error("PUBLIC_REPORT_PRIVACY_FAILED")
    return {
        "status": "COMPLETED_FROM_MATCHED_LOCKED_OOF",
        "branch": branch,
        "pair_kind": pair_kind,
        "reference_arm": reference_arm,
        "challenger_arm": challenger_arm,
        "acceptance_gate": gate,
        "claim": claim,
        "privacy_scan": privacy,
        "public_reports": sorted(path.name for path in reports.glob("*.json")),
    }
