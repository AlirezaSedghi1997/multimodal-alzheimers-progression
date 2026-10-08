"""Matched branch-specific v4 arm comparisons from committed fold metrics.

Point estimates are descriptive only. This deliberately does not reinterpret
raw risk-score differences as performance, use a pooled-branch Friedman test,
or promote legacy pairwise CIs to confirmatory inference.
"""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any

from .atomic_io import atomic_write_json
from .contracts import sha256_mapping
from .posttraining_correction_v4_2 import (
    ARMS,
    BRANCHES,
    FOLDS,
    CorrectionV42Error,
    _verified_outer,
)


PLANNED_PAIRS = (
    ("A0", "A1"),
    ("A0", "A2"),
    ("A1", "A3"),
    ("A2", "A3"),
    ("A3", "A4"),
)


def _require(ok: bool, reason: str) -> None:
    if not ok:
        raise CorrectionV42Error(reason)


def summarize_pairs(
    metrics: dict[str, dict[int, dict[str, dict[str, float]]]],
) -> dict[str, Any]:
    """Compute paired fold deltas only; never calculate inferential p-values."""
    _require(set(metrics) == set(BRANCHES), "PAIRWISE_BRANCH_COVERAGE_INVALID")
    result: dict[str, Any] = {}
    for branch in BRANCHES:
        _require(set(metrics[branch]) == set(FOLDS), "PAIRWISE_FOLD_COVERAGE_INVALID")
        per_arm: dict[str, Any] = {}
        for arm in ARMS:
            ibs: list[float] = []
            uno: list[float] = []
            for fold in FOLDS:
                _require(set(metrics[branch][fold]) == set(ARMS),
                         "PAIRWISE_ARM_COVERAGE_INVALID")
                one = metrics[branch][fold][arm]
                ibs.append(float(one["ibs"]))
                uno.append(float(one["uno"]))
            _require(all(math.isfinite(x) and 0 <= x <= 1 for x in ibs + uno),
                     "PAIRWISE_METRIC_INVALID")
            per_arm[arm] = {
                "fold_mean_ibs": statistics.fmean(ibs),
                "fold_mean_uno": statistics.fmean(uno),
            }
        pairs: dict[str, Any] = {}
        for first, second in PLANNED_PAIRS:
            delta_ibs = [metrics[branch][fold][second]["ibs"]
                         - metrics[branch][fold][first]["ibs"] for fold in FOLDS]
            delta_uno = [metrics[branch][fold][second]["uno"]
                         - metrics[branch][fold][first]["uno"] for fold in FOLDS]
            pairs[f"{first}_vs_{second}"] = {
                "first_arm": first,
                "second_arm": second,
                "fold_delta_ibs": delta_ibs,
                "fold_delta_uno": delta_uno,
                "fold_mean_delta_ibs": statistics.fmean(delta_ibs),
                "fold_mean_delta_uno": statistics.fmean(delta_uno),
                "status": "DESCRIPTIVE_MATCHED_FOLD_POINTS_ONLY",
                "confidence_interval": "NOT_CLAIM_VALID_FROM_LEGACY_RESAMPLING",
                "p_value": "NOT_COMPUTED",
                "noninferiority": "NOT_TESTED_ON_RAW_RISK_SCORE",
            }
        result[branch] = {"per_arm": per_arm, "planned_pairs": pairs}
    return result


def build_pairwise(root: str | Path, correction_path: str | Path) -> dict[str, Any]:
    root = Path(root).resolve()
    correction_file = Path(correction_path).resolve()
    raw = correction_file.read_bytes()
    correction = json.loads(raw)
    _require(correction.get("report_sha256") == sha256_mapping(
        {k: v for k, v in correction.items() if k != "report_sha256"}
    ), "PAIRWISE_CORRECTION_RECEIPT_HASH_MISMATCH")
    _require(correction.get("source_root") == str(root), "PAIRWISE_SOURCE_ROOT_MISMATCH")
    inputs: dict[str, str] = {}
    metrics: dict[str, dict[int, dict[str, dict[str, float]]]] = {}
    for branch in BRANCHES:
        metrics[branch] = {}
        seen: set[str] = set()
        for fold in FOLDS:
            metrics[branch][fold] = {}
            reference: dict[str, Any] | None = None
            for arm in ARMS:
                locked = _verified_outer(root, inputs, branch=branch, arm=arm, fold=fold)
                if reference is None:
                    reference = locked
                    overlap = seen & set(locked["keyed_outcomes"])
                    _require(not overlap, "PAIRWISE_OUTER_SUBJECT_OVERLAP")
                    seen |= set(locked["keyed_outcomes"])
                else:
                    _require(locked["keyed_outcomes"] == reference["keyed_outcomes"],
                             "PAIRWISE_OUTCOME_MISMATCH")
                    _require(locked["metrics"]["evaluation_grid_days"]
                             == reference["metrics"]["evaluation_grid_days"],
                             "PAIRWISE_EVALUATION_GRID_MISMATCH")
                    _require(locked["metrics"]["censoring_reference"]
                             == reference["metrics"]["censoring_reference"],
                             "PAIRWISE_CENSORING_REFERENCE_MISMATCH")
                metrics[branch][fold][arm] = {
                    "ibs": float(locked["metrics"]["integrated_brier_score"]),
                    "uno": float(locked["metrics"]["uno_c_index"]),
                }
        _require(len(seen) == int(correction["branches"][branch]["subject_count"]),
                 "PAIRWISE_BRANCH_SUBJECT_COUNT_MISMATCH")
    for path, digest in inputs.items():
        _require(correction["input_sha256"].get(path) == digest,
                 "PAIRWISE_SOURCE_CHANGED:" + path)
    report = {
        "schema_version": "4.2",
        "status": "COMPLETED_DESCRIPTIVE_ONLY",
        "science_gate_status": "BLOCKED",
        "claim_status": "POST_HOC_INTERNAL_EXPLORATORY",
        "positive_claim_allowed": False,
        "source_run_contract_sha256": correction["source_run_contract_sha256"],
        "correction_receipt_sha256": hashlib.sha256(raw).hexdigest(),
        "planned_pair_count_per_branch": len(PLANNED_PAIRS),
        "branch_results": summarize_pairs(metrics),
        "pooled_branch_friedman": "INVALID_NOT_REPEATED_MEASURES_OF_ONE_POPULATION",
        "selection_status": "POST_HOC_CORRECTED_FOLD_LOCAL_POLICY_SEPARATE_FROM_THIS_DESCRIPTIVE_BATTERY",
        "multiplicity_status": "NO_INFERENTIAL_HYPOTHESIS_TEST_PERFORMED",
        "input_sha256": dict(sorted(inputs.items())),
    }
    return report


def write_pairwise(
    root: str | Path, correction_path: str | Path, output_path: str | Path
) -> dict[str, Any]:
    report = build_pairwise(root, correction_path)
    report["source_code_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    report["report_sha256"] = sha256_mapping(report)
    output = Path(output_path)
    if output.exists():
        _require(json.loads(output.read_text(encoding="utf-8")) == report,
                 "PAIRWISE_OUTPUT_CONTRACT_MISMATCH")
        return report
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output, report)
    return report
