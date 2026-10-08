from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from sksurv.metrics import concordance_index_ipcw, integrated_brier_score
from sksurv.util import Surv

from .atomic_io import atomic_write_json, read_json
from .contracts import sha256_mapping


class ScienceGateError(ValueError):
    """Raised when paired outer-fold evidence is incomplete or inconsistent."""


def _survival_array(rows: list[dict[str, Any]]):
    return Surv.from_arrays(
        event=np.asarray([bool(int(row["event"])) for row in rows]),
        time=np.asarray([float(row["time_days"]) for row in rows]),
    )


def paired_oof_bootstrap(
    base_variant: dict[str, Any],
    augmented_variant: dict[str, Any],
    *,
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    if int(iterations) <= 0:
        raise ScienceGateError("POSITIVE_BOOTSTRAP_ITERATIONS_REQUIRED")
    base_folds = {int(fold["fold"]): fold for fold in base_variant.get("folds", [])}
    augmented_folds = {
        int(fold["fold"]): fold for fold in augmented_variant.get("folds", [])
    }
    if not base_folds or set(base_folds) != set(augmented_folds):
        raise ScienceGateError("PAIRED_OUTER_FOLD_MEMBERSHIP_MISMATCH")

    prepared: dict[int, dict[str, Any]] = {}
    subject_to_fold: dict[str, int] = {}
    for fold_id in sorted(base_folds):
        base_fold = base_folds[fold_id]
        augmented_fold = augmented_folds[fold_id]
        base_rows = {str(row["patient_hash"]): row for row in base_fold["oof_predictions"]}
        augmented_rows = {
            str(row["patient_hash"]): row for row in augmented_fold["oof_predictions"]
        }
        if set(base_rows) != set(augmented_rows):
            raise ScienceGateError("PAIRED_OOF_SUBJECT_MISMATCH")
        if list(base_fold["evaluation_grid_days"]) != list(
            augmented_fold["evaluation_grid_days"]
        ):
            raise ScienceGateError("PAIRED_OOF_EVALUATION_GRID_MISMATCH")
        for subject, base_row in base_rows.items():
            augmented_row = augmented_rows[subject]
            if (
                int(base_row["event"]) != int(augmented_row["event"])
                or float(base_row["time_days"]) != float(augmented_row["time_days"])
            ):
                raise ScienceGateError("PAIRED_OOF_OUTCOME_MISMATCH")
            if subject in subject_to_fold:
                raise ScienceGateError("DUPLICATE_OOF_SUBJECT")
            subject_to_fold[subject] = fold_id
        prepared[fold_id] = {
            "base": base_rows,
            "augmented": augmented_rows,
            "grid": np.asarray(base_fold["evaluation_grid_days"], dtype=float),
            "censoring_reference": list(base_fold["censoring_reference"]),
        }

    subjects = np.asarray(sorted(subject_to_fold), dtype=object)
    rng = np.random.default_rng(int(seed))
    delta_ibs: list[float] = []
    delta_cindex: list[float] = []
    failed_attempts = 0
    failure_reasons: Counter[str] = Counter()
    max_attempts = max(int(iterations) * 10, int(iterations) + 20)
    while len(delta_ibs) < int(iterations) and failed_attempts + len(delta_ibs) < max_attempts:
        sampled = rng.choice(subjects, size=len(subjects), replace=True).tolist()
        fold_ibs_base: list[float] = []
        fold_ibs_augmented: list[float] = []
        fold_c_base: list[float] = []
        fold_c_augmented: list[float] = []
        try:
            for fold_id, bundle in prepared.items():
                selected = [subject for subject in sampled if subject_to_fold[subject] == fold_id]
                if len(selected) < 2:
                    raise ValueError("BOOTSTRAP_FOLD_TOO_SMALL")
                base_rows = [bundle["base"][subject] for subject in selected]
                augmented_rows = [bundle["augmented"][subject] for subject in selected]
                y_train = _survival_array(bundle["censoring_reference"])
                y_test = _survival_array(base_rows)
                raw_grid = bundle["grid"]
                test_times = np.asarray([row["time_days"] for row in base_rows], dtype=float)
                train_times = np.asarray(
                    [row["time_days"] for row in bundle["censoring_reference"]], dtype=float
                )
                lower = max(float(test_times.min()), float(train_times.min()))
                upper = min(float(test_times.max()), float(train_times.max()))
                grid_mask = (raw_grid > lower) & (raw_grid < upper)
                if int(np.sum(grid_mask)) < 2:
                    raise ValueError("BOOTSTRAP_NO_COMMON_SURVIVAL_GRID")
                grid = raw_grid[grid_mask]
                base_survival = np.asarray(
                    [row["survival_probabilities"] for row in base_rows], dtype=float
                )[:, grid_mask]
                augmented_survival = np.asarray(
                    [row["survival_probabilities"] for row in augmented_rows], dtype=float
                )[:, grid_mask]
                base_risk = np.asarray([row["risk_score"] for row in base_rows], dtype=float)
                augmented_risk = np.asarray(
                    [row["risk_score"] for row in augmented_rows], dtype=float
                )
                fold_ibs_base.append(
                    float(integrated_brier_score(y_train, y_test, base_survival, grid))
                )
                fold_ibs_augmented.append(
                    float(integrated_brier_score(y_train, y_test, augmented_survival, grid))
                )
                fold_c_base.append(
                    float(concordance_index_ipcw(y_train, y_test, base_risk, tau=float(grid[-1]))[0])
                )
                fold_c_augmented.append(
                    float(
                        concordance_index_ipcw(
                            y_train, y_test, augmented_risk, tau=float(grid[-1])
                        )[0]
                    )
                )
            delta_ibs.append(float(np.mean(fold_ibs_augmented) - np.mean(fold_ibs_base)))
            delta_cindex.append(float(np.mean(fold_c_augmented) - np.mean(fold_c_base)))
        except (ValueError, ZeroDivisionError) as exc:
            failed_attempts += 1
            failure_reasons[f"{type(exc).__name__}:{exc}"] += 1
    if len(delta_ibs) != int(iterations):
        raise ScienceGateError(
            "INSUFFICIENT_VALID_BOOTSTRAP_ITERATIONS:"
            f"required={iterations}:completed={len(delta_ibs)}:reasons={dict(failure_reasons)}"
        )
    ibs_ci = np.percentile(np.asarray(delta_ibs), [2.5, 97.5])
    cindex_ci = np.percentile(np.asarray(delta_cindex), [2.5, 97.5])
    return {
        "status": "COMPLETED",
        "method": "paired_subject_bootstrap_over_locked_oof_predictions",
        "seed": int(seed),
        "requested_iterations": int(iterations),
        "successful_iterations": len(delta_ibs),
        "failed_attempts": failed_attempts,
        "failed_attempt_reasons": dict(failure_reasons),
        "delta_ibs_mean": float(np.mean(delta_ibs)),
        "delta_ibs_95ci": [float(value) for value in ibs_ci],
        "delta_cindex_mean": float(np.mean(delta_cindex)),
        "delta_cindex_95ci": [float(value) for value in cindex_ci],
        "subject_count": int(len(subjects)),
    }


def cached_paired_oof_bootstrap(
    base_variant: dict[str, Any],
    augmented_variant: dict[str, Any],
    *,
    output_path: str | Path,
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    evidence_digest = sha256_mapping(
        {
            "base_folds": base_variant.get("folds", []),
            "augmented_folds": augmented_variant.get("folds", []),
        }
    )
    target = Path(output_path)
    if target.is_file():
        existing = read_json(target)
        if existing.get("input_evidence_sha256") != evidence_digest:
            raise ScienceGateError("PAIRED_BOOTSTRAP_INPUT_EVIDENCE_MISMATCH")
        if int(existing.get("requested_iterations", -1)) != int(iterations):
            raise ScienceGateError("PAIRED_BOOTSTRAP_ITERATION_CONTRACT_MISMATCH")
        if int(existing.get("seed", -1)) != int(seed):
            raise ScienceGateError("PAIRED_BOOTSTRAP_SEED_CONTRACT_MISMATCH")
        if existing.get("status") != "COMPLETED":
            raise ScienceGateError("PAIRED_BOOTSTRAP_CACHE_NOT_COMPLETED")
        return existing
    result = paired_oof_bootstrap(
        base_variant,
        augmented_variant,
        iterations=int(iterations),
        seed=int(seed),
    )
    result["requested_iterations"] = int(iterations)
    result["seed"] = int(seed)
    result["input_evidence_sha256"] = evidence_digest
    result["cache_status"] = "COMPUTED_AND_LOCKED"
    atomic_write_json(target, result)
    return result


def incremental_value_gate(
    *,
    delta_ibs_ci: tuple[float, float],
    delta_cindex_ci: tuple[float, float],
    outer_folds_completed: int,
    expected_outer_folds: int,
    scanner_sensitivity_passed: bool,
) -> dict[str, Any]:
    blockers: list[str] = []
    if int(outer_folds_completed) != int(expected_outer_folds):
        blockers.append("INCOMPLETE_OUTER_CV")
    if float(delta_ibs_ci[1]) >= 0.0:
        blockers.append("IBS_IMPROVEMENT_CI_INCLUDES_ZERO")
    if float(delta_cindex_ci[0]) < 0.0:
        blockers.append("CINDEX_NOT_NONINFERIOR")
    if not scanner_sensitivity_passed:
        blockers.append("SCANNER_SHORTCUT_NOT_CONTROLLED")
    passed = not blockers
    return {
        "science_gate_status": "PASS" if passed else "FAIL",
        "claim_status": "INTERNAL_ONLY" if passed else "EXPLORATORY",
        "blockers": blockers,
        "delta_ibs_ci": list(delta_ibs_ci),
        "delta_cindex_ci": list(delta_cindex_ci),
    }


def assert_claim_consistency(record: dict[str, Any]) -> None:
    claim = record.get("claim_status")
    gate = record.get("science_gate_status")
    if claim in {"INTERNAL_ONLY", "EXTERNALLY_VALIDATED"} and gate != "PASS":
        raise ValueError("POSITIVE_CLAIM_WITHOUT_PASSED_SCIENCE_GATE")
