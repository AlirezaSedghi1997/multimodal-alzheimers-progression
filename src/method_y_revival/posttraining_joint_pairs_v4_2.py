"""Joint, fold-stratified uncertainty for the ten planned Method Y v4 pairs.

This consumes immutable outer predictions only. All five arms in a fold use
the SAME sampled subjects in every draw; this preserves pairing across the
planned contrasts. Intervals are conditional on fitted models and post-hoc
analysis. They exclude training and selection uncertainty and authorize no
positive, equivalence, non-inferiority, or external claim.
"""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any

import numpy as np

from .atomic_io import atomic_write_json
from .contracts import sha256_mapping
from .posttraining_bootstrap_v4_2 import _score_fold, _surv, fold_stratified_indices
from .posttraining_correction_v4_2 import (
    ARMS, BRANCHES, FOLDS, CorrectionV42Error, _verified_outer,
)
from .posttraining_pairwise_descriptive_v4_2 import PLANNED_PAIRS
from .posttraining_v3_incremental_v4_2 import _rows_on_grid, _support_stable_grid


def _require(ok: bool, code: str) -> None:
    if not ok:
        raise CorrectionV42Error(code)


def _load_receipt(path: Path, field: str) -> tuple[dict[str, Any], str]:
    _require(path.is_file(), f"JOINT_RECEIPT_MISSING:{path.name}")
    raw = path.read_bytes()
    try:
        value = json.loads(raw)
    except (ValueError, UnicodeDecodeError) as exc:
        raise CorrectionV42Error(f"JOINT_RECEIPT_UNREADABLE:{path.name}") from exc
    _require(isinstance(value, dict), f"JOINT_RECEIPT_NOT_OBJECT:{path.name}")
    _require(value.get(field) == sha256_mapping(
        {key: item for key, item in value.items() if key != field}
    ), f"JOINT_RECEIPT_HASH_MISMATCH:{path.name}")
    return value, hashlib.sha256(raw).hexdigest()


def prepare_joint_inputs(
    root: Path, correction: dict[str, Any], pairwise: dict[str, Any]
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, str]]:
    """Check each locked cell and reconstruct one matched array per branch/fold."""
    inputs: dict[str, str] = {}
    prepared: dict[str, list[dict[str, Any]]] = {}
    for branch in BRANCHES:
        prepared[branch] = []
        branch_subjects: set[str] = set()
        for fold in FOLDS:
            locked = {
                arm: _verified_outer(root, inputs, branch=branch, arm=arm, fold=fold)
                for arm in ARMS
            }
            reference = locked["A0"]
            membership = reference["keyed_outcomes"]
            _require(not (branch_subjects & set(membership)),
                     "JOINT_OUTER_SUBJECT_OVERLAP")
            branch_subjects.update(membership)
            reference_metrics = reference["metrics"]
            for arm in ARMS[1:]:
                _require(locked[arm]["keyed_outcomes"] == membership,
                         "JOINT_ARM_MEMBERSHIP_OR_OUTCOME_MISMATCH")
                _require(locked[arm]["metrics"]["evaluation_grid_days"]
                         == reference_metrics["evaluation_grid_days"],
                         "JOINT_ARM_EVALUATION_GRID_MISMATCH")
                _require(locked[arm]["metrics"]["censoring_reference"]
                         == reference_metrics["censoring_reference"],
                         "JOINT_ARM_CENSORING_REFERENCE_MISMATCH")
            grid = np.asarray(reference_metrics["evaluation_grid_days"], dtype=float)
            _require(len(grid) >= 2 and np.isfinite(grid).all()
                     and np.all(np.diff(grid) > 0), "JOINT_EVALUATION_GRID_INVALID")
            train = _surv(reference_metrics["censoring_reference"])
            order = sorted(membership)
            rows: dict[str, list[dict[str, Any]]] = {}
            points: dict[str, tuple[float, float]] = {}
            for arm in ARMS:
                metrics = locked[arm]["metrics"]
                by_subject = {str(row["patient_hash"]): row
                              for row in metrics["oof_predictions"]}
                _require(set(by_subject) == set(order), "JOINT_ARM_PREDICTION_COVERAGE_INVALID")
                rows[arm] = [by_subject[subject] for subject in order]
                points[arm] = _score_fold(train, rows[arm], grid)
                _require(
                    math.isclose(points[arm][0], float(metrics["integrated_brier_score"]),
                                 rel_tol=1e-8, abs_tol=1e-8)
                    and math.isclose(points[arm][1], float(metrics["uno_c_index"]),
                                     rel_tol=1e-8, abs_tol=1e-8),
                    "JOINT_LOCKED_METRIC_RECOMPUTATION_MISMATCH",
                )
            prepared[branch].append({
                "fold": fold, "train": train, "grid": grid,
                "rows": rows, "points": points, "size": len(order),
            })
        _require(len(branch_subjects) == correction["branches"][branch]["subject_count"],
                 "JOINT_BRANCH_SUBJECT_COUNT_MISMATCH")
    for name, digest in inputs.items():
        _require(correction["input_sha256"].get(name) == digest,
                 "JOINT_SOURCE_HASH_MISMATCH:" + name)
    for branch in BRANCHES:
        for first, second in PLANNED_PAIRS:
            expected = pairwise["branch_results"][branch]["planned_pairs"][f"{first}_vs_{second}"]
            delta_ibs = statistics.fmean(
                fold["points"][second][0] - fold["points"][first][0]
                for fold in prepared[branch]
            )
            delta_uno = statistics.fmean(
                fold["points"][second][1] - fold["points"][first][1]
                for fold in prepared[branch]
            )
            _require(
                math.isclose(delta_ibs, expected["fold_mean_delta_ibs"],
                             rel_tol=1e-8, abs_tol=1e-8)
                and math.isclose(delta_uno, expected["fold_mean_delta_uno"],
                                 rel_tol=1e-8, abs_tol=1e-8),
                "JOINT_PAIRWISE_POINT_MISMATCH",
            )
    return prepared, dict(sorted(inputs.items()))


def score_one_joint_draw(
    prepared: list[dict[str, Any]], sampled_indices: list[np.ndarray]
) -> dict[str, tuple[float, float]]:
    """Score all arms on the same sampled subject indices per outer fold."""
    _require(len(prepared) == len(sampled_indices) == 5, "JOINT_DRAW_FOLD_COVERAGE_INVALID")
    scores: dict[str, list[tuple[float, float]]] = {arm: [] for arm in ARMS}
    for fold, indices in zip(prepared, sampled_indices, strict=True):
        _require(len(indices) == fold["size"] and
                 all(0 <= int(i) < fold["size"] for i in indices),
                 "JOINT_DRAW_MEMBERSHIP_INVALID")
        for arm in ARMS:
            sample = [fold["rows"][arm][int(index)] for index in indices]
            scores[arm].append(_score_fold(fold["train"], sample, fold["grid"]))
    return {
        arm: (statistics.fmean(row[0] for row in scores[arm]),
              statistics.fmean(row[1] for row in scores[arm]))
        for arm in ARMS
    }


def restrict_to_outcome_supported_grid(prepared: list[dict[str, Any]]) -> dict[str, Any]:
    """Discard only an unsupported prefix, using follow-up times, not scores.

    Original locked scores were checked before this function is called. An IBS
    on the restricted grid is a *new post-hoc sensitivity estimand*, never a
    substitute for the source IBS. The final grid day (Uno's tau) is retained.
    """
    original = prepared[0]["grid"].tolist()
    stable = _support_stable_grid([
        {"grid": fold["grid"].tolist(), "baseline_rows": fold["rows"]["A0"]}
        for fold in prepared
    ])
    _require(stable[-1] == original[-1], "JOINT_UNO_TAU_CHANGED")
    if stable != original:
        for fold in prepared:
            old_grid = fold["grid"].tolist()
            fold["grid"] = np.asarray(stable, dtype=float)
            for arm in ARMS:
                fold["rows"][arm] = _rows_on_grid(fold["rows"][arm], old_grid, stable)
                restricted = _score_fold(fold["train"], fold["rows"][arm], fold["grid"])
                _require(math.isclose(restricted[1], fold["points"][arm][1],
                                      rel_tol=1e-8, abs_tol=1e-8),
                         "JOINT_UNO_POINT_CHANGED_ON_GRID_RESTRICTION")
                fold["points"][arm] = restricted
    return {
        "original_evaluation_grid_days": original,
        "support_stable_evaluation_grid_days": stable,
        "support_rule": "AT_LEAST_FIVE_TEST_SUBJECTS_BEFORE_FIRST_AND_AFTER_LAST_GRID_DAY_IN_EACH_FOLD",
        "ibs_estimand": (
            "SOURCE_GRID_REPRODUCED" if stable == original
            else "RESTRICTED_GRID_POST_HOC_SENSITIVITY_NOT_EQUIVALENT_TO_SOURCE_IBS"
        ),
        "uno_tau_preserved": True,
    }


def bootstrap_joint_branch(
    prepared: list[dict[str, Any]], *, iterations: int, seed: int
) -> dict[str, Any]:
    _require(int(iterations) >= 1000, "JOINT_BOOTSTRAP_REQUIRES_AT_LEAST_1000")
    _require([fold["fold"] for fold in prepared] == list(FOLDS),
             "JOINT_BOOTSTRAP_FOLD_COVERAGE_INVALID")
    rng = np.random.default_rng(int(seed))
    sizes = [fold["size"] for fold in prepared]
    draws: dict[str, dict[str, list[float]]] = {
        f"{first}_vs_{second}": {"delta_ibs": [], "delta_uno": []}
        for first, second in PLANNED_PAIRS
    }
    rejection_reasons: dict[str, int] = {}
    rejected = 0
    valid = 0
    max_attempts = 10 * int(iterations)
    while valid < int(iterations) and valid + rejected < max_attempts:
        indices = fold_stratified_indices(sizes, rng)
        try:
            scores = score_one_joint_draw(prepared, indices)
        except (ValueError, ZeroDivisionError, CorrectionV42Error) as exc:
            rejected += 1
            reason = f"{type(exc).__name__}:{exc}"
            rejection_reasons[reason] = rejection_reasons.get(reason, 0) + 1
            continue
        for first, second in PLANNED_PAIRS:
            pair = draws[f"{first}_vs_{second}"]
            pair["delta_ibs"].append(scores[second][0] - scores[first][0])
            pair["delta_uno"].append(scores[second][1] - scores[first][1])
        valid += 1
    _require(valid == int(iterations), "JOINT_BOOTSTRAP_VALID_DRAWS_INSUFFICIENT")
    if rejected / (valid + rejected) > 0.05:
        raise CorrectionV42Error(
            f"JOINT_BOOTSTRAP_REJECTION_RATE_EXCEEDS_FIVE_PERCENT:{rejected}/{valid + rejected}"
        )
    contrasts: dict[str, Any] = {}
    for first, second in PLANNED_PAIRS:
        key = f"{first}_vs_{second}"
        point_ibs = statistics.fmean(
            fold["points"][second][0] - fold["points"][first][0]
            for fold in prepared
        )
        point_uno = statistics.fmean(
            fold["points"][second][1] - fold["points"][first][1]
            for fold in prepared
        )
        contrasts[key] = {
            "point_delta_ibs": point_ibs,
            "point_delta_uno": point_uno,
            "delta_ibs_95ci": [float(x) for x in np.quantile(
                draws[key]["delta_ibs"], [0.025, 0.975])],
            "delta_uno_95ci": [float(x) for x in np.quantile(
                draws[key]["delta_uno"], [0.025, 0.975])],
            "status": "CONDITIONAL_EXPLORATORY_ONLY",
            "p_value": "NOT_COMPUTED",
            "familywise_claim": "NOT_EVALUATED",
        }
    return {
        "status": "COMPLETED_CONDITIONAL_EXPLORATORY",
        "method": "JOINT_FOLD_STRATIFIED_PAIRED_SUBJECT_BOOTSTRAP_FIXED_LOCKED_MODELS",
        "fold_sizes_preserved": sizes,
        "shared_subject_draws_across_arms": True,
        "valid_resamples": valid,
        "rejected_attempts": rejected,
        "rejected_attempt_reasons": rejection_reasons,
        "planned_contrasts": contrasts,
        "interpretation_limit": "CONDITIONAL_FIXED_MODELS_POST_HOC_NO_REFIT_OR_SELECTION_VARIANCE",
    }


def build_joint_report(
    root: str | Path, correction_path: str | Path, pairwise_path: str | Path,
    *, iterations: int = 2000, seed: int = 20260714,
) -> dict[str, Any]:
    root = Path(root).resolve()
    correction, correction_file_sha = _load_receipt(Path(correction_path), "report_sha256")
    pairwise, pairwise_file_sha = _load_receipt(Path(pairwise_path), "report_sha256")
    _require(correction.get("source_root") == str(root), "JOINT_SOURCE_ROOT_MISMATCH")
    _require(pairwise.get("correction_receipt_sha256") == correction_file_sha,
             "JOINT_PAIRWISE_PARENT_MISMATCH")
    _require(pairwise.get("source_run_contract_sha256")
             == correction.get("source_run_contract_sha256"),
             "JOINT_RUN_CONTRACT_MISMATCH")
    prepared, inputs = prepare_joint_inputs(root, correction, pairwise)
    branch_results: dict[str, Any] = {}
    for i, branch in enumerate(BRANCHES):
        support = restrict_to_outcome_supported_grid(prepared[branch])
        branch_results[branch] = {
            **support,
            **bootstrap_joint_branch(
                prepared[branch], iterations=int(iterations), seed=int(seed) + i
            ),
        }
    return {
        "schema_version": "4.2",
        "status": "COMPLETED_CONDITIONAL_EXPLORATORY",
        "science_gate_status": "BLOCKED",
        "claim_status": "POST_HOC_INTERNAL_EXPLORATORY",
        "positive_claim_allowed": False,
        "external_claim_allowed": False,
        "correction_receipt_sha256": correction_file_sha,
        "pairwise_receipt_sha256": pairwise_file_sha,
        "source_run_contract_sha256": correction["source_run_contract_sha256"],
        "iterations_per_branch": int(iterations),
        "seed": int(seed),
        "branch_results": branch_results,
        "multiplicity": "TEN_PLANNED_CONTRASTS_TWO_METRICS_NO_ADJUSTED_CLAIM_TEST",
        "input_sha256": inputs,
    }


def write_joint_report(
    root: str | Path, correction_path: str | Path, pairwise_path: str | Path,
    output_path: str | Path, *, iterations: int = 2000, seed: int = 20260714,
) -> dict[str, Any]:
    report = build_joint_report(
        root, correction_path, pairwise_path,
        iterations=iterations, seed=seed,
    )
    report["source_code_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    helper = Path(__file__).with_name("posttraining_bootstrap_v4_2.py")
    report["metric_helper_sha256"] = hashlib.sha256(helper.read_bytes()).hexdigest()
    support_helper = Path(__file__).with_name("posttraining_v3_incremental_v4_2.py")
    report["support_helper_sha256"] = hashlib.sha256(support_helper.read_bytes()).hexdigest()
    report["report_sha256"] = sha256_mapping(report)
    output = Path(output_path)
    if output.exists():
        _require(json.loads(output.read_text(encoding="utf-8")) == report,
                 "JOINT_OUTPUT_CONTRACT_MISMATCH")
        return report
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output, report)
    return report
