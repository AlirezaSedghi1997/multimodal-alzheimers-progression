"""Fold-stratified, paired uncertainty for a reconstructed Method Y policy.

This estimates variation in the locked out-of-fold predictions conditional on
the fitted models and the post-hoc policy.  It does not account for refitting
or for choosing the analysis after seeing outer results, so it cannot upgrade
a scientific claim.
"""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any

import numpy as np
from sksurv.metrics import concordance_index_ipcw, integrated_brier_score
from sksurv.util import Surv

from .atomic_io import atomic_write_json
from .contracts import sha256_mapping
from .posttraining_correction_v4_2 import (
    BRANCHES,
    CorrectionV42Error,
    _verified_outer,
)


def _surv(rows: list[dict[str, Any]]) -> np.ndarray:
    return Surv.from_arrays(
        event=np.asarray([bool(int(r["event"])) for r in rows]),
        time=np.asarray([float(r["time_days"]) for r in rows]),
    )


def fold_stratified_indices(sizes: list[int], rng: np.random.Generator) -> list[np.ndarray]:
    """Preserve the exact size of each outer fold in every bootstrap draw."""
    if len(sizes) != 5 or any(int(n) < 2 for n in sizes):
        raise CorrectionV42Error("BOOTSTRAP_FOLD_SIZES_INVALID")
    return [rng.integers(0, int(n), size=int(n)) for n in sizes]


def _score_fold(
    train: np.ndarray, rows: list[dict[str, Any]], grid: np.ndarray
) -> tuple[float, float]:
    test = _surv(rows)
    probabilities = np.asarray([r["survival_probabilities"] for r in rows], dtype=float)
    risk = np.asarray([r["risk_score"] for r in rows], dtype=float)
    if probabilities.shape != (len(rows), len(grid)) or not np.isfinite(probabilities).all():
        raise CorrectionV42Error("BOOTSTRAP_SURVIVAL_PROBABILITIES_INVALID")
    if not np.isfinite(risk).all():
        raise CorrectionV42Error("BOOTSTRAP_RISK_NONFINITE")
    ibs = float(integrated_brier_score(train, test, probabilities, grid))
    uno = float(concordance_index_ipcw(train, test, risk, tau=float(grid[-1]))[0])
    if not (math.isfinite(ibs) and 0 <= ibs <= 1 and math.isfinite(uno) and 0 <= uno <= 1):
        raise CorrectionV42Error("BOOTSTRAP_SCORE_OUT_OF_RANGE")
    return ibs, uno


def _prepare_branch(
    root: Path, correction: dict[str, Any], branch: str, inputs: dict[str, str]
) -> list[dict[str, Any]]:
    prepared: list[dict[str, Any]] = []
    for fold_row in correction["branches"][branch]["folds"]:
        fold = int(fold_row["outer_fold"])
        selected_arm = str(fold_row["selected_arm"])
        baseline = _verified_outer(root, inputs, branch=branch, arm="A0", fold=fold)
        selected = baseline if selected_arm == "A0" else _verified_outer(
            root, inputs, branch=branch, arm=selected_arm, fold=fold
        )
        if selected["private_sha256"] != fold_row["selected_outer_private_sha256"]:
            raise CorrectionV42Error("SELECTED_OUTER_ARTIFACT_CHANGED")
        if baseline["keyed_outcomes"] != selected["keyed_outcomes"]:
            raise CorrectionV42Error("BOOTSTRAP_PAIRED_OUTCOMES_MISMATCH")
        bm, sm = baseline["metrics"], selected["metrics"]
        if bm["evaluation_grid_days"] != sm["evaluation_grid_days"]:
            raise CorrectionV42Error("BOOTSTRAP_PAIRED_GRID_MISMATCH")
        if bm["censoring_reference"] != sm["censoring_reference"]:
            raise CorrectionV42Error("BOOTSTRAP_CENSORING_REFERENCE_MISMATCH")
        grid = np.asarray(bm["evaluation_grid_days"], dtype=float)
        if len(grid) < 2 or not np.all(np.diff(grid) > 0):
            raise CorrectionV42Error("BOOTSTRAP_GRID_INVALID")
        subject_order = sorted(baseline["keyed_outcomes"])
        base_rows = {str(r["patient_hash"]): r for r in bm["oof_predictions"]}
        selected_rows = {str(r["patient_hash"]): r for r in sm["oof_predictions"]}
        base_ordered = [base_rows[subject] for subject in subject_order]
        selected_ordered = [selected_rows[subject] for subject in subject_order]
        train = _surv(bm["censoring_reference"])
        original_base = _score_fold(train, base_ordered, grid)
        original_selected = _score_fold(train, selected_ordered, grid)
        for actual, stored in (
            (original_base[0], bm["integrated_brier_score"]),
            (original_base[1], bm["uno_c_index"]),
            (original_selected[0], sm["integrated_brier_score"]),
            (original_selected[1], sm["uno_c_index"]),
        ):
            if not math.isclose(actual, float(stored), rel_tol=1e-8, abs_tol=1e-8):
                raise CorrectionV42Error("LOCKED_OUTER_METRIC_RECOMPUTATION_MISMATCH")
        prepared.append({
            "fold": fold,
            "selected_arm": selected_arm,
            "train": train,
            "grid": grid,
            "baseline_rows": base_ordered,
            "selected_rows": selected_ordered,
            "point_delta_ibs": original_selected[0] - original_base[0],
            "point_delta_uno": original_selected[1] - original_base[1],
        })
    if [p["fold"] for p in prepared] != list(range(5)):
        raise CorrectionV42Error("BOOTSTRAP_FOLD_COVERAGE_INVALID")
    return prepared


def _bootstrap_branch(
    prepared: list[dict[str, Any]], *, iterations: int, seed: int
) -> dict[str, Any]:
    if int(iterations) < 1000:
        raise CorrectionV42Error("BOOTSTRAP_REQUIRES_AT_LEAST_1000_RESAMPLES")
    rng = np.random.default_rng(int(seed))
    sizes = [len(fold["baseline_rows"]) for fold in prepared]
    delta_ibs: list[float] = []
    delta_uno: list[float] = []
    failed = 0
    failure_reasons: dict[str, int] = {}
    max_attempts = 10 * int(iterations)
    while len(delta_ibs) < int(iterations) and failed + len(delta_ibs) < max_attempts:
        samples = fold_stratified_indices(sizes, rng)
        fold_dibs: list[float] = []
        fold_duno: list[float] = []
        try:
            for fold, index in zip(prepared, samples, strict=True):
                if fold["selected_arm"] == "A0":
                    fold_dibs.append(0.0)
                    fold_duno.append(0.0)
                    continue
                base_rows = [fold["baseline_rows"][int(i)] for i in index]
                selected_rows = [fold["selected_rows"][int(i)] for i in index]
                base_ibs, base_uno = _score_fold(fold["train"], base_rows, fold["grid"])
                selected_ibs, selected_uno = _score_fold(
                    fold["train"], selected_rows, fold["grid"]
                )
                fold_dibs.append(selected_ibs - base_ibs)
                fold_duno.append(selected_uno - base_uno)
            delta_ibs.append(statistics.fmean(fold_dibs))
            delta_uno.append(statistics.fmean(fold_duno))
        except (ValueError, ZeroDivisionError, CorrectionV42Error) as exc:
            failed += 1
            label = f"{type(exc).__name__}:{exc}"
            failure_reasons[label] = failure_reasons.get(label, 0) + 1
    if len(delta_ibs) != int(iterations):
        raise CorrectionV42Error("BOOTSTRAP_VALID_ITERATIONS_INSUFFICIENT")
    if failed / float(failed + len(delta_ibs)) > 0.05:
        raise CorrectionV42Error("BOOTSTRAP_REJECTION_RATE_EXCEEDS_FIVE_PERCENT")
    return {
        "status": "COMPLETED_CONDITIONAL_EXPLORATORY",
        "method": "FOLD_STRATIFIED_PAIRED_SUBJECT_BOOTSTRAP_FIXED_LOCKED_MODELS",
        "fold_sizes_preserved": sizes,
        "point_delta_ibs": statistics.fmean(p["point_delta_ibs"] for p in prepared),
        "point_delta_uno": statistics.fmean(p["point_delta_uno"] for p in prepared),
        "delta_ibs_95ci": [float(v) for v in np.quantile(delta_ibs, [0.025, 0.975])],
        "delta_uno_95ci": [float(v) for v in np.quantile(delta_uno, [0.025, 0.975])],
        "valid_resamples": len(delta_ibs),
        "rejected_attempts": failed,
        "rejected_attempt_reasons": failure_reasons,
        "interpretation_limit": (
            "CONDITIONAL_ON_FITTED_MODELS_AND_POST_HOC_SELECTION; "
            "DOES_NOT_INCLUDE_TRAINING_OR_SELECTION_VARIANCE"
        ),
    }


def build_bootstrap(
    root: str | Path, correction_path: str | Path, *, iterations: int = 2000,
    seed: int = 20260714,
) -> dict[str, Any]:
    root = Path(root).resolve()
    correction_path = Path(correction_path).resolve()
    raw = correction_path.read_bytes()
    correction = json.loads(raw)
    if correction.get("report_sha256") != sha256_mapping(
        {k: v for k, v in correction.items() if k != "report_sha256"}
    ):
        raise CorrectionV42Error("CORRECTION_RECEIPT_HASH_MISMATCH")
    correction_source = Path(__file__).with_name("posttraining_correction_v4_2.py")
    if correction.get("correction_source_sha256") != hashlib.sha256(
        correction_source.read_bytes()
    ).hexdigest():
        raise CorrectionV42Error("CORRECTION_SOURCE_CHANGED")
    if correction.get("source_root") != str(root):
        raise CorrectionV42Error("CORRECTION_SOURCE_ROOT_MISMATCH")
    inputs: dict[str, str] = {}
    branch_results: dict[str, Any] = {}
    for branch in BRANCHES:
        prepared = _prepare_branch(root, correction, branch, inputs)
        branch_results[branch] = _bootstrap_branch(
            prepared, iterations=int(iterations), seed=int(seed) + BRANCHES.index(branch)
        )
    for name, digest in inputs.items():
        if correction["input_sha256"].get(name) != digest:
            raise CorrectionV42Error("CORRECTION_INPUT_CHANGED:" + name)
    return {
        "schema_version": "4.2",
        "status": "COMPLETED_CONDITIONAL_EXPLORATORY",
        "science_gate_status": "BLOCKED",
        "claim_status": "POST_HOC_INTERNAL_EXPLORATORY",
        "positive_claim_allowed": False,
        "external_claim_allowed": False,
        "correction_receipt_sha256": hashlib.sha256(raw).hexdigest(),
        "source_run_contract_sha256": correction["source_run_contract_sha256"],
        "iterations": int(iterations),
        "seed": int(seed),
        "branches": branch_results,
        "input_sha256": dict(sorted(inputs.items())),
    }


def write_bootstrap(
    root: str | Path, correction_path: str | Path, output_path: str | Path,
    *, iterations: int = 2000, seed: int = 20260714,
) -> dict[str, Any]:
    report = build_bootstrap(root, correction_path, iterations=iterations, seed=seed)
    report["source_code_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    report["report_sha256"] = sha256_mapping(report)
    output = Path(output_path)
    if output.exists():
        existing = json.loads(output.read_text(encoding="utf-8"))
        if existing != report:
            raise CorrectionV42Error("BOOTSTRAP_OUTPUT_CONTRACT_MISMATCH")
        return report
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output, report)
    return report
