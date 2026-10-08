"""Reanalyse Method Y v3 scalar versus image using fixed outer-fold strata."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from .atomic_io import atomic_write_json
from .contracts import sha256_mapping
from .posttraining_bootstrap_v4_2 import _bootstrap_branch, _score_fold, _surv
from .posttraining_correction_v4_2 import (
    BRANCHES,
    CorrectionV42Error,
    _read_json,
    _self_hash,
)


VARIANTS = ("scalar_tabular", "scalar_tabular_plus_pattern_image")


def _support_stable_grid(folds: list[dict[str, Any]], *, minimum_tail: int = 5) -> list[float]:
    """Choose a common grid by time support alone, without inspecting scores.

    A bootstrap cannot estimate IBS at a grid time that lies outside the
    resampled test follow-up range.  At least five observations on either
    side of the grid in EVERY outer fold limits empty-tail draws.  Only a
    prefix may be dropped, so the last horizon and Uno's tau stay unchanged.
    This creates a clearly labeled sensitivity estimand when the source grid
    is too close to a fold's follow-up boundary.
    """
    if len(folds) != 5:
        raise CorrectionV42Error("V3_GRID_REQUIRES_FIVE_FOLDS")
    originals = [list(fold["grid"]) for fold in folds]
    if any(grid != originals[0] for grid in originals[1:]):
        raise CorrectionV42Error("V3_GRID_DIFFERS_BETWEEN_FOLDS")
    original = originals[0]
    for start in range(len(original) - 1):
        candidate = original[start:]
        if all(
            sum(float(row["time_days"]) < candidate[0] for row in fold["baseline_rows"])
            >= int(minimum_tail)
            and sum(float(row["time_days"]) > candidate[-1] for row in fold["baseline_rows"])
            >= int(minimum_tail)
            for fold in folds
        ):
            return candidate
    raise CorrectionV42Error("V3_NO_SUPPORT_STABLE_COMMON_GRID")


def _rows_on_grid(
    rows: list[dict[str, Any]], original_grid: list[float], stable_grid: list[float]
) -> list[dict[str, Any]]:
    positions = [original_grid.index(value) for value in stable_grid]
    return [
        {**row, "survival_probabilities": [row["survival_probabilities"][i] for i in positions]}
        for row in rows
    ]


def _verified_v3_outer(
    root: Path, inputs: dict[str, str], *, branch: str, variant: str, fold: int
) -> dict[str, Any]:
    base = root / "private" / "outer_evaluation" / branch / variant
    private = _read_json(base / f"fold_{fold}_private.json", inputs, root)
    state = _read_json(base / f"fold_{fold}" / "state.json", inputs, root)
    journal = _read_json(base / f"fold_{fold}" / "outer_access_journal.json", inputs, root)
    result = _read_json(base / f"fold_{fold}" / "outer_result.json", inputs, root)
    public = _read_json(root / "reports" / f"{branch}_{variant}_fold_{fold}_public.json", inputs, root)
    for payload, field in (
        (private, "private_record_sha256"), (state, "state_sha256"),
        (journal, "journal_sha256"), (result, "result_record_sha256"),
    ):
        _self_hash(payload, field, "V3_LOCKED_ARTIFACT_HASH_MISMATCH")
    if state.get("state") != "FINALIZED" or journal.get("access_count") != 1:
        raise CorrectionV42Error("V3_OUTER_NOT_FINALIZED_EXACTLY_ONCE")
    digest = private.get("run_contract_sha256")
    if not (
        digest == state.get("run_contract_sha256") == journal.get("run_contract_sha256")
        == result.get("run_contract_sha256") == public.get("run_contract")
    ):
        raise CorrectionV42Error("V3_OUTER_CONTRACT_MISMATCH")
    if not (
        private.get("branch") == branch and private.get("variant") == variant
        and private.get("outer_fold") == result.get("fold") == state.get("fold") == fold
    ):
        raise CorrectionV42Error("V3_OUTER_IDENTITY_MISMATCH")
    if not (
        state.get("result_record_sha256") == result.get("result_record_sha256")
        and state.get("checkpoint_sha256") == result.get("checkpoint_sha256")
        == journal.get("checkpoint_sha256")
    ):
        raise CorrectionV42Error("V3_OUTER_STATE_LINK_MISMATCH")
    metrics = private.get("metrics")
    if not isinstance(metrics, dict) or not (
        private.get("metrics_sha256") == result.get("metrics_sha256")
        == sha256_mapping(metrics)
        and result.get("metrics") == metrics
    ):
        raise CorrectionV42Error("V3_OUTER_METRICS_HASH_MISMATCH")
    for key in ("n_test", "test_events", "integrated_brier_score", "uno_c_index",
                "evaluation_grid_days", "administrative_censoring_cutoff_days"):
        if public.get("aggregate", {}).get(key) != metrics.get(key):
            raise CorrectionV42Error("V3_PUBLIC_PRIVATE_MISMATCH:" + key)
    rows = metrics.get("oof_predictions")
    if not isinstance(rows, list) or len(rows) != int(metrics["n_test"]):
        raise CorrectionV42Error("V3_OOF_COVERAGE_INVALID")
    keyed = {str(row["patient_hash"]): row for row in rows}
    if len(keyed) != len(rows):
        raise CorrectionV42Error("V3_OOF_DUPLICATE_SUBJECT")
    return {"metrics": metrics, "keyed_rows": keyed}


def build_v3_incremental(
    root: str | Path, *, iterations: int = 2000, seed: int = 20260714
) -> dict[str, Any]:
    root = Path(root).resolve()
    inputs: dict[str, str] = {}
    state = _read_json(root / "orchestration" / "launcher_state.json", inputs, root)
    contract = _read_json(root / "run_contract.json", inputs, root)
    _self_hash(state, "state_sha256", "V3_LAUNCHER_STATE_HASH_MISMATCH")
    if state.get("stage") != "FINALIZED" or state.get("facts", {}).get("final_cell_count") != 20:
        raise CorrectionV42Error("V3_LAUNCHER_NOT_FINALIZED")
    if state.get("run_contract_sha256") != contract.get("digest"):
        raise CorrectionV42Error("V3_MASTER_CONTRACT_MISMATCH")
    branches: dict[str, Any] = {}
    for branch in BRANCHES:
        original_folds: list[dict[str, Any]] = []
        seen: set[str] = set()
        for fold in range(5):
            base = _verified_v3_outer(root, inputs, branch=branch,
                                      variant=VARIANTS[0], fold=fold)
            image = _verified_v3_outer(root, inputs, branch=branch,
                                       variant=VARIANTS[1], fold=fold)
            bm, im = base["metrics"], image["metrics"]
            if set(base["keyed_rows"]) != set(image["keyed_rows"]):
                raise CorrectionV42Error("V3_PAIRED_MEMBERSHIP_MISMATCH")
            if seen & set(base["keyed_rows"]):
                raise CorrectionV42Error("V3_OUTER_FOLD_SUBJECT_OVERLAP")
            seen |= set(base["keyed_rows"])
            for subject in base["keyed_rows"]:
                left, right = base["keyed_rows"][subject], image["keyed_rows"][subject]
                if (int(left["event"]), float(left["time_days"])) != (
                    int(right["event"]), float(right["time_days"])
                ):
                    raise CorrectionV42Error("V3_PAIRED_OUTCOME_MISMATCH")
            if bm["evaluation_grid_days"] != im["evaluation_grid_days"] or (
                bm["censoring_reference"] != im["censoring_reference"]
            ):
                raise CorrectionV42Error("V3_PAIRED_EVALUATION_REFERENCE_MISMATCH")
            grid = np.asarray(bm["evaluation_grid_days"], dtype=float)
            ordered = sorted(base["keyed_rows"])
            base_rows = [base["keyed_rows"][s] for s in ordered]
            image_rows = [image["keyed_rows"][s] for s in ordered]
            train = _surv(bm["censoring_reference"])
            base_point = _score_fold(train, base_rows, grid)
            image_point = _score_fold(train, image_rows, grid)
            for actual, stored in (
                (base_point[0], bm["integrated_brier_score"]),
                (base_point[1], bm["uno_c_index"]),
                (image_point[0], im["integrated_brier_score"]),
                (image_point[1], im["uno_c_index"]),
            ):
                if not math.isclose(actual, float(stored), rel_tol=1e-8, abs_tol=1e-8):
                    raise CorrectionV42Error("V3_LOCKED_METRIC_RECOMPUTATION_MISMATCH")
            original_folds.append({
                "fold": fold,
                "train": train,
                "grid": grid.tolist(),
                "baseline_rows": base_rows,
                "selected_rows": image_rows,
            })
        stable_grid = _support_stable_grid(original_folds)
        prepared: list[dict[str, Any]] = []
        for fold in original_folds:
            grid = np.asarray(stable_grid, dtype=float)
            base_rows = _rows_on_grid(fold["baseline_rows"], fold["grid"], stable_grid)
            image_rows = _rows_on_grid(fold["selected_rows"], fold["grid"], stable_grid)
            base_point = _score_fold(fold["train"], base_rows, grid)
            image_point = _score_fold(fold["train"], image_rows, grid)
            prepared.append({
                "fold": fold["fold"],
                "selected_arm": VARIANTS[1],
                "train": fold["train"],
                "grid": grid,
                "baseline_rows": base_rows,
                "selected_rows": image_rows,
                "point_delta_ibs": image_point[0] - base_point[0],
                "point_delta_uno": image_point[1] - base_point[1],
            })
        corrected = _bootstrap_branch(
            prepared, iterations=int(iterations), seed=int(seed) + BRANCHES.index(branch)
        )
        old = _read_json(root / "reports" / f"{branch}_paired_bootstrap.json", inputs, root)
        if old.get("status") != "COMPLETED":
            raise CorrectionV42Error("V3_LEGACY_BOOTSTRAP_INCOMPLETE")
        branches[branch] = {
            "scalar_vs_image": corrected,
            "original_evaluation_grid_days": original_folds[0]["grid"],
            "support_stable_evaluation_grid_days": stable_grid,
            "support_rule": "AT_LEAST_FIVE_TEST_SUBJECTS_BEFORE_FIRST_AND_AFTER_LAST_GRID_DAY_IN_EACH_FOLD",
            "ibs_estimand": (
                "SOURCE_GRID_REPRODUCED" if stable_grid == original_folds[0]["grid"]
                else "RESTRICTED_GRID_POST_HOC_SENSITIVITY_NOT_EQUIVALENT_TO_SOURCE_IBS"
            ),
            "legacy_global_resampling": {
                "status": "SUPERSEDED_FOR_FOLD_STRATIFIED_UNCERTAINTY",
                "legacy_delta_ibs_95ci": old["delta_ibs_95ci"],
                "legacy_delta_uno_95ci": old["delta_cindex_95ci"],
                "legacy_report_sha256": inputs[
                    (root / "reports" / f"{branch}_paired_bootstrap.json").relative_to(root).as_posix()
                ],
            },
            "subject_count": len(seen),
        }
    return {
        "schema_version": "4.2",
        "status": "COMPLETED_CONDITIONAL_EXPLORATORY",
        "science_gate_status": "BLOCKED",
        "claim_status": "POST_HOC_INTERNAL_EXPLORATORY",
        "positive_claim_allowed": False,
        "external_claim_allowed": False,
        "source_run_contract_sha256": contract["digest"],
        "source_root": str(root),
        "branches": branches,
        "input_sha256": dict(sorted(inputs.items())),
    }


def write_v3_incremental(
    root: str | Path, output_path: str | Path,
    *, iterations: int = 2000, seed: int = 20260714,
) -> dict[str, Any]:
    report = build_v3_incremental(root, iterations=iterations, seed=seed)
    report["source_code_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    helper = Path(__file__).with_name("posttraining_bootstrap_v4_2.py")
    report["bootstrap_helper_sha256"] = hashlib.sha256(helper.read_bytes()).hexdigest()
    report["report_sha256"] = sha256_mapping(report)
    output = Path(output_path)
    if output.exists():
        if json.loads(output.read_text(encoding="utf-8")) != report:
            raise CorrectionV42Error("V3_INCREMENTAL_OUTPUT_CONTRACT_MISMATCH")
        return report
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output, report)
    return report
