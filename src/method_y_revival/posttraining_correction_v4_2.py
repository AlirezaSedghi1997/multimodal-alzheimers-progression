"""Read-only correction of the completed Method Y v4 factorial evaluation.

The original global arm preference used inner scores from all five outer
partitions.  Since their training sets overlap, that preference cannot be
evaluated as a nested selection policy on the same outer folds.  This module
reconstructs an arm choice separately inside each outer fold and consumes
only its already committed result.  It never calls a model or materializes a
test dataset again.  All conclusions remain post-hoc and exploratory.
"""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .atomic_io import atomic_write_json
from .contracts import sha256_mapping


BRANCHES = ("amyloid", "tau")
ARMS = ("A0", "A1", "A2", "A3", "A4")
FOLDS = tuple(range(5))


class CorrectionV42Error(RuntimeError):
    """An immutable input or scientific invariant failed verification."""


def _require(condition: bool, code: str) -> None:
    if not condition:
        raise CorrectionV42Error(code)


def _read_json(path: Path, inputs: dict[str, str], root: Path) -> dict[str, Any]:
    _require(path.is_file(), f"INPUT_MISSING:{path.relative_to(root)}")
    raw = path.read_bytes()
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CorrectionV42Error(f"INPUT_INVALID_JSON:{path.relative_to(root)}") from exc
    _require(isinstance(payload, dict), f"INPUT_NOT_OBJECT:{path.relative_to(root)}")
    inputs[path.relative_to(root).as_posix()] = hashlib.sha256(raw).hexdigest()
    return payload


def _self_hash(payload: Mapping[str, Any], field: str, code: str) -> None:
    _require(
        payload.get(field) == sha256_mapping({k: v for k, v in payload.items() if k != field}),
        code,
    )


def select_arm_within_outer_fold(
    inner_by_arm: Mapping[str, list[float]],
    complexity_ranks: Mapping[str, int],
) -> dict[str, Any]:
    """One-SE selection from precisely three inner folds in ONE outer fold."""
    _require(set(inner_by_arm) == set(ARMS), "INNER_ARM_COVERAGE_INVALID")
    _require(set(complexity_ranks) == set(ARMS), "ARM_COMPLEXITY_COVERAGE_INVALID")
    _require(
        {int(complexity_ranks[arm]) for arm in ARMS} == set(range(5)),
        "ARM_COMPLEXITY_RANKS_INVALID",
    )
    summaries: dict[str, dict[str, float]] = {}
    for arm in ARMS:
        values = [float(v) for v in inner_by_arm[arm]]
        _require(
            len(values) == 3 and all(math.isfinite(v) and 0 <= v <= 1 for v in values),
            f"INNER_IBS_INVALID:{arm}",
        )
        summaries[arm] = {
            "mean_ibs": statistics.fmean(values),
            "standard_error": statistics.stdev(values) / math.sqrt(3),
        }
    best = min(ARMS, key=lambda arm: (summaries[arm]["mean_ibs"], arm))
    threshold = summaries[best]["mean_ibs"] + summaries[best]["standard_error"]
    eligible = [arm for arm in ARMS if summaries[arm]["mean_ibs"] <= threshold]
    selected = min(
        eligible,
        key=lambda arm: (int(complexity_ranks[arm]), summaries[arm]["mean_ibs"], arm),
    )
    return {
        "selected_arm": selected,
        "best_mean_arm": best,
        "one_standard_error_threshold": threshold,
        "eligible_within_one_standard_error": eligible,
        "arm_summaries": summaries,
        "selection_metric": "IPCW_INTEGRATED_BRIER_SCORE",
        "outer_test_consulted_for_this_fold_selection": False,
    }


def _verify_winner(
    payload: dict[str, Any], *, branch: str, arm: str, fold: int, master_digest: str
) -> list[float]:
    _self_hash(payload, "winner_record_sha256", "WINNER_RECORD_HASH_MISMATCH")
    _require(
        payload.get("branch") == branch
        and payload.get("variant") == arm
        and payload.get("outer_fold") == fold,
        "WINNER_IDENTITY_MISMATCH",
    )
    _require(payload.get("run_contract_sha256") == master_digest, "WINNER_CONTRACT_MISMATCH")
    winner = payload.get("winner")
    _require(isinstance(winner, dict) and winner.get("status") == "COMPLETED", "WINNER_INCOMPLETE")
    records = winner.get("inner_fold_metric_records")
    _require(isinstance(records, list) and len(records) == 3, "INNER_FOLD_COUNT_INVALID")
    _require({r.get("inner_fold") for r in records} == {0, 1, 2}, "INNER_FOLD_IDS_INVALID")
    values: list[float] = []
    for row in sorted(records, key=lambda item: item["inner_fold"]):
        _require(row.get("metric_name") == "IPCW_INTEGRATED_BRIER_SCORE", "INNER_METRIC_NAME_INVALID")
        checkpoint_sha = str(row.get("locked_checkpoint_sha256", ""))
        _require(len(checkpoint_sha) == 64 and all(c in "0123456789abcdef" for c in checkpoint_sha),
                 "INNER_CHECKPOINT_DIGEST_INVALID")
        values.append(float(row["metric_value"]))
    _require(
        all(math.isclose(a, b, abs_tol=1e-12) for a, b in zip(values, winner.get("inner_fold_ibs", [])))
        and len(winner.get("inner_fold_ibs", [])) == 3,
        "INNER_METRIC_RECORD_MISMATCH",
    )
    return values


def _verified_outer(
    root: Path, inputs: dict[str, str], *, branch: str, arm: str, fold: int
) -> dict[str, Any]:
    base = root / "private" / "outer_evaluation" / branch / arm
    private = _read_json(base / f"fold_{fold}_private.json", inputs, root)
    state = _read_json(base / f"fold_{fold}" / "state.json", inputs, root)
    journal = _read_json(base / f"fold_{fold}" / "outer_access_journal.json", inputs, root)
    result = _read_json(base / f"fold_{fold}" / "outer_result.json", inputs, root)
    public = _read_json(root / "reports" / f"{branch}_{arm}_fold_{fold}_public.json", inputs, root)
    _self_hash(private, "private_record_sha256", "PRIVATE_RECORD_HASH_MISMATCH")
    _self_hash(state, "state_sha256", "OUTER_STATE_HASH_MISMATCH")
    _self_hash(journal, "journal_sha256", "OUTER_ACCESS_JOURNAL_HASH_MISMATCH")
    _self_hash(result, "result_record_sha256", "OUTER_RESULT_HASH_MISMATCH")
    _require(state.get("state") == "FINALIZED" and journal.get("access_count") == 1,
             "OUTER_NOT_FINALIZED_EXACTLY_ONCE")
    _require(
        state.get("result_record_sha256") == result.get("result_record_sha256")
        and state.get("checkpoint_sha256") == result.get("checkpoint_sha256") == journal.get("checkpoint_sha256"),
        "OUTER_STATE_RESULT_LINK_MISMATCH",
    )
    _require(
        private.get("branch") == branch and private.get("variant") == arm
        and private.get("outer_fold") == fold and result.get("fold") == fold,
        "OUTER_IDENTITY_MISMATCH",
    )
    digest = private.get("run_contract_sha256")
    _require(
        digest == state.get("run_contract_sha256") == journal.get("run_contract_sha256")
        == result.get("run_contract_sha256") == public.get("run_contract_sha256"),
        "OUTER_CONTRACT_MISMATCH",
    )
    metrics = private.get("metrics")
    _require(isinstance(metrics, dict), "OUTER_METRICS_MISSING")
    metric_sha = sha256_mapping(metrics)
    _require(
        private.get("metrics_sha256") == result.get("metrics_sha256") == metric_sha
        and result.get("metrics") == metrics,
        "OUTER_METRICS_HASH_MISMATCH",
    )
    aggregate = public.get("aggregate")
    _require(isinstance(aggregate, dict), "PUBLIC_AGGREGATE_MISSING")
    for key in ("integrated_brier_score", "uno_c_index", "n_test", "test_events",
                "evaluation_grid_days", "administrative_censoring_cutoff_days"):
        _require(aggregate.get(key) == metrics.get(key), f"PUBLIC_PRIVATE_METRIC_MISMATCH:{key}")
    _require(metrics.get("outer_test_augmentation_applied") is False, "OUTER_TEST_AUGMENTED")
    rows = metrics.get("oof_predictions")
    _require(isinstance(rows, list) and len(rows) == int(metrics["n_test"]), "OOF_ROW_COUNT_MISMATCH")
    keyed: dict[str, tuple[int, float]] = {}
    for row in rows:
        subject = str(row["patient_hash"])
        _require(subject not in keyed, "OOF_DUPLICATE_SUBJECT")
        keyed[subject] = (int(row["event"]), float(row["time_days"]))
    _require(math.isfinite(float(metrics["integrated_brier_score"]))
             and 0 <= float(metrics["integrated_brier_score"]) <= 1,
             "OUTER_IBS_INVALID")
    _require(math.isfinite(float(metrics["uno_c_index"]))
             and 0 <= float(metrics["uno_c_index"]) <= 1,
             "OUTER_UNO_INVALID")
    return {"metrics": metrics, "keyed_outcomes": keyed, "metrics_sha256": metric_sha,
            "private_sha256": inputs[(base / f"fold_{fold}_private.json").relative_to(root).as_posix()]}


def _read_pair_and_calibration(root: Path, inputs: dict[str, str]) -> dict[str, Any]:
    """Correct point-estimate provenance; preserve old CI as non-confirmatory."""
    summary = _read_json(root / "reports" / "factorial_benchmark_summary.json", inputs, root)
    _require(summary.get("status") == "COMPLETED", "BENCHMARK_SUMMARY_INCOMPLETE")
    pairs = summary.get("pair_reports")
    _require(isinstance(pairs, dict) and len(pairs) == 10, "PAIR_REPORT_COVERAGE_INVALID")
    corrected: dict[str, Any] = {}
    calibration: dict[str, Any] = {}
    for key, report in sorted(pairs.items()):
        try:
            branch, pair = key.split(":", 1)
        except ValueError as exc:
            raise CorrectionV42Error("PAIR_KEY_INVALID") from exc
        _require(branch in BRANCHES and pair in {
            "A0_vs_A1", "A0_vs_A2", "A1_vs_A3", "A2_vs_A3", "A3_vs_A4"
        }, "PAIR_KEY_INVALID")
        base = root / "analysis_pairs" / pair / "reports"
        boot_path = base / f"{branch}_paired_oof_bootstrap.json"
        cal_path = base / f"{branch}_calibration.json"
        boot = _read_json(boot_path, inputs, root)
        cal = _read_json(cal_path, inputs, root)
        _require(boot.get("status") == "COMPLETED" and boot.get("successful_iterations") == 2000,
                 "LEGACY_BOOTSTRAP_INCOMPLETE")
        gate = report.get("acceptance_gate", {})
        _require(boot.get("delta_ibs_95ci") == gate.get("delta_ibs_95ci")
                 and boot.get("delta_cindex_95ci") == gate.get("delta_uno_cindex_95ci"),
                 "PAIR_BOOTSTRAP_SUMMARY_MISMATCH")
        corrected[key] = {
            "delta_ibs_point_from_legacy_bootstrap": float(boot["delta_ibs_mean"]),
            "delta_uno_point_from_legacy_bootstrap": float(boot["delta_cindex_mean"]),
            "delta_ibs_ci_legacy": boot["delta_ibs_95ci"],
            "delta_uno_ci_legacy": boot["delta_cindex_95ci"],
            "old_midpoint_was_not_point_estimate": True,
            "inference_status": "REANALYSIS_REQUIRED_FOLD_STRATIFIED_BOOTSTRAP",
            "bootstrap_source_sha256": inputs[boot_path.relative_to(root).as_posix()],
        }
        _require(cal.get("status") == "COMPLETED_WITH_ESTIMABILITY_GATES", "CALIBRATION_REPORT_INCOMPLETE")
        horizons = cal.get("horizons")
        _require(isinstance(horizons, dict) and set(horizons) == {"547.875", "1095.75", "1640.0"},
                 "CALIBRATION_HORIZONS_INVALID")
        horizon_status = {h: str(record.get("status")) for h, record in horizons.items()}
        _require(horizon_status["547.875"] == "ESTIMABLE"
                 and horizon_status["1095.75"] == "ESTIMABLE"
                 and horizon_status["1640.0"] == "NOT_ESTIMABLE",
                 "CALIBRATION_ESTIMABILITY_MISMATCH")
        calibration[key] = {
            "status": "COMPLETED_WITH_ESTIMABILITY_GATES",
            "horizon_status": horizon_status,
            "confidence_interval_note": "EXISTING_HESSIAN_INTERVALS_DO_NOT_ACCOUNT_FOR_FOLD_CLUSTERING",
            "source_sha256": inputs[cal_path.relative_to(root).as_posix()],
        }
    return {"contrasts": corrected, "calibration": calibration,
            "calibration_leg": "COMPLETED_WITH_ESTIMABILITY_GATES"}


def build_correction(root: str | Path) -> dict[str, Any]:
    """Verify locked artifacts and reconstruct a per-outer-fold arm policy."""
    root = Path(root).resolve()
    inputs: dict[str, str] = {}
    state = _read_json(root / "orchestration_state.json", inputs, root)
    contract = _read_json(root / "run_contract.json", inputs, root)
    _require(state.get("stage") == "FINALIZED" and state.get("outer_cells_finalized") == 50,
             "FACTORIAL_RUN_NOT_FINALIZED")
    _require(state.get("run_contract_sha256") == contract.get("digest"), "MASTER_CONTRACT_MISMATCH")
    branches: dict[str, Any] = {}
    for branch in BRANCHES:
        lock = _read_json(root / "orchestration" / branch / "inner_only_arm_selection.json", inputs, root)
        _self_hash(lock, "selection_record_sha256", "GLOBAL_SELECTION_RECORD_HASH_MISMATCH")
        _require(lock.get("branch") == branch and lock.get("locked_before_outer_evaluation") is True,
                 "GLOBAL_SELECTION_LOCK_INVALID")
        master_digest = str(lock.get("run_contract_sha256", ""))
        ranks = {
            arm: int(lock["selection"]["arm_summaries"][arm]["complexity_rank"])
            for arm in ARMS
        }
        fold_rows: list[dict[str, Any]] = []
        branch_subjects: set[str] = set()
        for fold in FOLDS:
            evidence: dict[str, list[float]] = {}
            outer: dict[str, dict[str, Any]] = {}
            for arm in ARMS:
                winner_path = root / "orchestration" / branch / f"winner_{arm}_outer_{fold}.json"
                winner = _read_json(winner_path, inputs, root)
                evidence[arm] = _verify_winner(
                    winner, branch=branch, arm=arm, fold=fold, master_digest=master_digest
                )
                outer[arm] = _verified_outer(root, inputs, branch=branch, arm=arm, fold=fold)
            reference = outer["A0"]
            for arm in ARMS[1:]:
                _require(outer[arm]["keyed_outcomes"] == reference["keyed_outcomes"],
                         "ARM_OUTER_MEMBERSHIP_OR_OUTCOME_MISMATCH")
                _require(outer[arm]["metrics"]["evaluation_grid_days"]
                         == reference["metrics"]["evaluation_grid_days"],
                         "ARM_OUTER_EVALUATION_GRID_MISMATCH")
            membership = set(reference["keyed_outcomes"])
            _require(not (membership & branch_subjects), "OUTER_FOLD_SUBJECT_OVERLAP")
            branch_subjects |= membership
            selection = select_arm_within_outer_fold(evidence, ranks)
            selected_arm = selection["selected_arm"]
            metrics = outer[selected_arm]["metrics"]
            fold_rows.append({
                "outer_fold": fold,
                "selected_arm": selected_arm,
                "selection": selection,
                "inner_winner_source_sha256": {
                    arm: inputs[(root / "orchestration" / branch
                                 / f"winner_{arm}_outer_{fold}.json").relative_to(root).as_posix()]
                    for arm in ARMS
                },
                "selected_outer_private_sha256": outer[selected_arm]["private_sha256"],
                "outer_test_subject_count": len(membership),
                "outer_test_event_count": int(metrics["test_events"]),
                "outer_ibs": float(metrics["integrated_brier_score"]),
                "outer_uno": float(metrics["uno_c_index"]),
                "reference_a0_ibs": float(reference["metrics"]["integrated_brier_score"]),
                "reference_a0_uno": float(reference["metrics"]["uno_c_index"]),
            })
        branches[branch] = {
            "original_global_arm": lock["selection"]["selected_arm"],
            "global_arm_internal_performance_role": "SELECTION_INFORMED_DESCRIPTIVE_ONLY",
            "strict_nested_policy_reconstruction": "POST_HOC_FROM_COMMITTED_OUTER_RESULTS",
            "folds": fold_rows,
            "subject_count": len(branch_subjects),
            "fold_mean_ibs": statistics.fmean(row["outer_ibs"] for row in fold_rows),
            "fold_mean_uno": statistics.fmean(row["outer_uno"] for row in fold_rows),
            "a0_fold_mean_ibs": statistics.fmean(row["reference_a0_ibs"] for row in fold_rows),
            "a0_fold_mean_uno": statistics.fmean(row["reference_a0_uno"] for row in fold_rows),
            "claim_status": "POST_HOC_INTERNAL_EXPLORATORY",
        }
    extended = _read_pair_and_calibration(root, inputs)
    return {
        "schema_version": "4.2",
        "status": "COMPLETED_FROM_HASH_VERIFIED_LOCKED_ARTIFACTS",
        "science_gate_status": "BLOCKED",
        "claim_status": "POST_HOC_INTERNAL_EXPLORATORY",
        "positive_claim_allowed": False,
        "external_claim_allowed": False,
        "source_run_contract_sha256": contract["digest"],
        "source_root": str(root),
        "design_correction": "ARM_CHOICE_IS_LOCAL_TO_EACH_OUTER_TRAIN_PARTITION",
        "interpretation_limit": "REANALYSIS_OCCURRED_AFTER_ALL_OUTER_ARMS_WERE_EXPOSED",
        "branches": branches,
        "legacy_post_eval_corrections": extended,
        "remaining_science_blockers": [
            "CORRECTED_FOLD_STRATIFIED_PAIRED_UNCERTAINTY_PENDING",
            "SCANNER_LEAVE_MANUFACTURER_OUT_NOT_ESTABLISHED",
            "QUANTITATIVE_PET_EQUIVALENCE_NOT_ESTABLISHED",
            "EXTERNAL_VALIDATION_PENDING",
        ],
        "input_sha256": dict(sorted(inputs.items())),
    }


def write_correction(root: str | Path, output_path: str | Path) -> dict[str, Any]:
    """Write a versioned receipt; refuse silent rebind in an existing path."""
    report = build_correction(root)
    source_file = Path(__file__).resolve()
    report["correction_source_sha256"] = hashlib.sha256(source_file.read_bytes()).hexdigest()
    report["report_sha256"] = sha256_mapping(report)
    output = Path(output_path)
    if output.exists():
        existing = json.loads(output.read_text(encoding="utf-8"))
        _require(existing == report, "CORRECTION_OUTPUT_CONTRACT_MISMATCH")
        return report
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output, report)
    return report
