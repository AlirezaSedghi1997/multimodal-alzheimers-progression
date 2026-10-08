from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.method_y_revival.atomic_io import read_json  # noqa: E402
from src.method_y_revival.contracts import sha256_mapping  # noqa: E402
from src.method_y_revival.image_factorial_v4 import matched_arm_differences  # noqa: E402
from src.method_y_revival.image_factorial_v4 import (  # noqa: E402
    select_inner_only_factorial_arm,
)
from src.method_y_revival.survival_models import select_inner_winner  # noqa: E402


def _self_hash_valid(payload: dict[str, Any], field: str) -> bool:
    claimed = payload.get(field)
    body = {key: value for key, value in payload.items() if key != field}
    return claimed == sha256_mapping(body)


def _privacy_violations(value: Any, path: str = "$") -> list[str]:
    forbidden = {
        "patient_hash",
        "source_subject_id",
        "raw_path",
        "image_path_private",
        "event",
        "time_days",
        "original_event",
        "original_time_days",
        "risk_score",
        "survival_probabilities",
        "embedding",
        "oof_predictions",
        "oof_embeddings",
        "censoring_reference",
    }
    failures: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            if str(key) in forbidden:
                failures.append(f"{path}.{key}")
            failures.extend(_privacy_violations(child, f"{path}.{key}"))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            failures.extend(_privacy_violations(child, f"{path}[{index}]"))
    return failures


def verify(*, config_path: Path, level: str) -> dict[str, Any]:
    from scripts.run_method_y_image_factorial_v4 import _preflight

    failures: list[str] = []
    try:
        preflight = _preflight(config_path)
    except Exception as exc:
        return {
            "status": "FAIL",
            "level": level,
            "failures": [f"PREFLIGHT:{type(exc).__name__}:{exc}"],
        }
    config = preflight["config"]
    output_root = Path(preflight["output_root"])
    if level == "readiness":
        return {
            "status": "PASS",
            "level": "readiness",
            "run_contract_sha256": preflight["run_contract_sha256"],
            "branch_summary": preflight["branch_summary"],
            "claim_status": "POST_HOC_INTERNAL_EXPLORATORY",
        }
    run_contract_path = output_root / "run_contract.json"
    if not run_contract_path.is_file():
        failures.append("RUN_CONTRACT_MISSING")
    else:
        run_contract = read_json(run_contract_path)
        if run_contract.get("digest") != preflight["run_contract_sha256"]:
            failures.append("RUN_CONTRACT_MISMATCH")
    state_path = output_root / "orchestration_state.json"
    if not state_path.is_file():
        failures.append("ORCHESTRATION_STATE_MISSING")
    else:
        state = read_json(state_path)
        if state.get("stage") != "FINALIZED" or int(state.get("outer_cells_finalized", -1)) != 50:
            failures.append("GLOBAL_FINALIZATION_INCOMPLETE")
        if state.get("claim_status") != "POST_HOC_INTERNAL_EXPLORATORY":
            failures.append("CLAIM_STATUS_INVALID")
        if state.get("positive_claim_allowed") is not False:
            failures.append("POSITIVE_CLAIM_MUST_REMAIN_FALSE")

    expected_cells = 0
    valid_cells = 0
    for branch in ("amyloid", "tau"):
        master_path = output_root / "orchestration" / branch / "master_contract.json"
        selection_path = (
            output_root
            / "orchestration"
            / branch
            / "inner_only_arm_selection.json"
        )
        if not master_path.is_file() or not selection_path.is_file():
            failures.append(f"INNER_ARM_SELECTION_LOCK_MISSING:{branch}")
        else:
            master = read_json(master_path)
            selection_record = read_json(selection_path)
            if not _self_hash_valid(selection_record, "selection_record_sha256"):
                failures.append(f"INNER_ARM_SELECTION_HASH_INVALID:{branch}")
            if (
                selection_record.get("run_contract_sha256")
                != master.get("master_contract_sha256")
            ):
                failures.append(f"INNER_ARM_SELECTION_CONTRACT_MISMATCH:{branch}")
            evidence_records = selection_record.get("inner_evidence", {})
            if selection_record.get("inner_evidence_sha256") != sha256_mapping(
                evidence_records
            ):
                failures.append(f"INNER_ARM_EVIDENCE_HASH_INVALID:{branch}")
            try:
                evidence: dict[str, list[float]] = {}
                for arm in ("A0", "A1", "A2", "A3", "A4"):
                    records = list(evidence_records[arm])
                    if len(records) != 15:
                        raise ValueError("EXPECTED_FIFTEEN_INNER_VALUES_PER_ARM")
                    grouped: dict[int, list[float]] = {}
                    for row in records:
                        grouped.setdefault(int(row["outer_fold"]), []).append(
                            float(row["metric_value"])
                        )
                    if set(grouped) != set(range(5)) or any(
                        len(values) != 3 for values in grouped.values()
                    ):
                        raise ValueError("INNER_EVIDENCE_NESTING_INVALID")
                    evidence[arm] = [
                        float(sum(grouped[fold]) / len(grouped[fold]))
                        for fold in range(5)
                    ]
                expected_selection = select_inner_only_factorial_arm(
                    evidence,
                    complexity_ranks={
                        str(key): int(value)
                        for key, value in config["execution"][
                            "inner_only_arm_complexity_ranks"
                        ].items()
                    },
                )
                if sha256_mapping(expected_selection) != sha256_mapping(
                    selection_record.get("selection", {})
                ):
                    failures.append(f"INNER_ARM_SELECTION_NOT_REPRODUCIBLE:{branch}")
            except Exception as exc:
                failures.append(
                    f"INNER_ARM_SELECTION_EVIDENCE_INVALID:{branch}:{type(exc).__name__}"
                )
            if selection_record.get("locked_before_outer_evaluation") is not True or (
                selection_record.get("outer_test_consulted") is not False
            ):
                failures.append(f"INNER_ARM_SELECTION_NOT_OUTER_BLIND:{branch}")
        for arm in ("A0", "A1", "A2", "A3", "A4"):
            for fold in range(5):
                expected_cells += 1
                winner_path = (
                    output_root
                    / "orchestration"
                    / branch
                    / f"winner_{arm}_outer_{fold}.json"
                )
                if not winner_path.is_file():
                    failures.append(f"WINNER_MISSING:{branch}:{arm}:{fold}")
                else:
                    winner_record = read_json(winner_path)
                    if not _self_hash_valid(winner_record, "winner_record_sha256"):
                        failures.append(f"WINNER_HASH_INVALID:{branch}:{arm}:{fold}")
                    candidates = winner_record.get("candidates", [])
                    expected_candidate_ids = {
                        str(row["candidate_id"])
                        for row in config["model"]["hpo_candidates"]
                    }
                    if {
                        str(row.get("candidate_id")) for row in candidates
                    } != expected_candidate_ids:
                        failures.append(
                            f"INNER_CANDIDATE_BUDGET_MISMATCH:{branch}:{arm}:{fold}"
                        )
                    for candidate in candidates:
                        if candidate.get("status") != "COMPLETED":
                            failures.append(f"INNER_CANDIDATE_INCOMPLETE:{branch}:{arm}:{fold}")
                        metric_records = candidate.get("inner_fold_metric_records", [])
                        if {
                            int(record.get("inner_fold", -1))
                            for record in metric_records
                        } != set(range(3)):
                            failures.append(
                                f"INNER_FOLD_EVIDENCE_INCOMPLETE:{branch}:{arm}:{fold}"
                            )
                        for record in metric_records:
                            value = record.get("metric_value")
                            if (
                                record.get("metric_name") != "IPCW_INTEGRATED_BRIER_SCORE"
                                or not isinstance(value, (int, float))
                                or not math.isfinite(float(value))
                                or not 0.0 <= float(value) <= 1.0
                            ):
                                failures.append(f"INNER_IBS_INVALID:{branch}:{arm}:{fold}")
                            if (
                                record.get("checkpoint_selection_metric_name")
                                != "IPCW_INTEGRATED_BRIER_SCORE"
                            ):
                                failures.append(
                                    f"INNER_CHECKPOINT_SELECTION_INVALID:{branch}:{arm}:{fold}"
                                )
                            gap = record.get(
                                "train_validation_nll_gap_at_selected_epoch"
                            )
                            if not isinstance(gap, (int, float)) or not math.isfinite(
                                float(gap)
                            ):
                                failures.append(
                                    f"INNER_OVERFIT_GAP_INVALID:{branch}:{arm}:{fold}"
                                )
                    try:
                        expected_winner = select_inner_winner(candidates)
                        if sha256_mapping(expected_winner) != sha256_mapping(
                            winner_record.get("winner", {})
                        ):
                            failures.append(
                                f"WINNER_NOT_REPRODUCIBLE_FROM_INNER:{branch}:{arm}:{fold}"
                            )
                    except Exception:
                        failures.append(
                            f"WINNER_SELECTION_EVIDENCE_INVALID:{branch}:{arm}:{fold}"
                        )
                cell = output_root / "private" / "outer_evaluation" / branch / arm / f"fold_{fold}"
                state_file = cell / "state.json"
                result_file = cell / "outer_result.json"
                journal_file = cell / "outer_access_journal.json"
                private_file = (
                    output_root
                    / "private"
                    / "outer_evaluation"
                    / branch
                    / arm
                    / f"fold_{fold}_private.json"
                )
                public_file = output_root / "reports" / f"{branch}_{arm}_fold_{fold}_public.json"
                if not all(path.is_file() for path in (state_file, result_file, journal_file, private_file, public_file)):
                    failures.append(f"OUTER_CELL_ARTIFACT_MISSING:{branch}:{arm}:{fold}")
                    continue
                outer_state = read_json(state_file)
                result = read_json(result_file)
                journal = read_json(journal_file)
                if not _self_hash_valid(outer_state, "state_sha256") or outer_state.get("state") != "FINALIZED":
                    failures.append(f"OUTER_STATE_INVALID:{branch}:{arm}:{fold}")
                    continue
                if not _self_hash_valid(result, "result_record_sha256"):
                    failures.append(f"OUTER_RESULT_HASH_INVALID:{branch}:{arm}:{fold}")
                    continue
                if not _self_hash_valid(journal, "journal_sha256") or journal.get("access_count") != 1:
                    failures.append(f"OUTER_ACCESS_NOT_EXACTLY_ONCE:{branch}:{arm}:{fold}")
                    continue
                metrics = result.get("metrics", {})
                ibs = metrics.get("integrated_brier_score")
                uno = metrics.get("uno_c_index")
                if not isinstance(ibs, (int, float)) or not 0.0 <= float(ibs) <= 1.0:
                    failures.append(f"OUTER_IBS_INVALID:{branch}:{arm}:{fold}")
                    continue
                if (
                    not isinstance(uno, (int, float))
                    or not math.isfinite(float(uno))
                    or not 0.0 <= float(uno) <= 1.0
                ):
                    failures.append(f"OUTER_UNO_INVALID:{branch}:{arm}:{fold}")
                    continue
                public_payload = read_json(public_file)
                if _privacy_violations(public_payload):
                    failures.append(f"PUBLIC_ROW_LEVEL_LEAK:{branch}:{arm}:{fold}")
                    continue
                valid_cells += 1

    if matched_arm_differences("A0", "A1") != ("augmentation_profile",):
        failures.append("A0_A1_NOT_MATCHED")
    if matched_arm_differences("A2", "A3") != ("augmentation_profile",):
        failures.append("A2_A3_NOT_MATCHED")
    if matched_arm_differences("A0", "A2") != ("encoder_family",):
        failures.append("A0_A2_NOT_MATCHED")
    summary_path = output_root / "reports" / "factorial_benchmark_summary.json"
    if not summary_path.is_file():
        failures.append("FACTORIAL_REPORT_MISSING")
    else:
        summary = read_json(summary_path)
        if summary.get("positive_claim_allowed") is not False:
            failures.append("FACTORIAL_REPORT_POSITIVE_CLAIM_INVALID")
        failures.extend(
            f"REPORT_PRIVACY:{value}"
            for value in _privacy_violations(summary)
        )
        if "scalar_tabular" in json.dumps(summary, sort_keys=True):
            failures.append("FACTORIAL_REPORT_SEMANTICALLY_MISLABELLED_AS_SCALAR_INCREMENT")
    for reference, challenger in (
        ("A0", "A1"),
        ("A2", "A3"),
        ("A0", "A2"),
        ("A1", "A3"),
        ("A3", "A4"),
    ):
        pair_root = output_root / "analysis_pairs" / f"{reference}_vs_{challenger}"
        for branch in ("amyloid", "tau"):
            identity_path = pair_root / "reports" / f"{branch}_pair_identity.json"
            acceptance_path = (
                pair_root / "reports" / f"{branch}_factorial_acceptance.json"
            )
            if not identity_path.is_file() or not acceptance_path.is_file():
                failures.append(
                    f"FACTORIAL_PAIR_REPORT_MISSING:{branch}:{reference}:{challenger}"
                )
                continue
            identity = read_json(identity_path)
            acceptance = read_json(acceptance_path)
            if (
                identity.get("reference_arm") != reference
                or identity.get("challenger_arm") != challenger
                or identity.get("not_an_image_incremental_value_analysis") is not True
            ):
                failures.append(
                    f"FACTORIAL_PAIR_IDENTITY_INVALID:{branch}:{reference}:{challenger}"
                )
            if acceptance.get("positive_claim_allowed") is not False:
                failures.append(
                    f"FACTORIAL_PAIR_POSITIVE_CLAIM_INVALID:{branch}:{reference}:{challenger}"
                )
    for report in output_root.glob("analysis_pairs/**/reports/*.json"):
        failures.extend(
            f"PAIR_REPORT_PRIVACY:{report.name}:{value}"
            for value in _privacy_violations(read_json(report))
        )
    return {
        "status": "PASS" if not failures else "FAIL",
        "level": "final",
        "expected_outer_cells": expected_cells,
        "valid_outer_cells": valid_cells,
        "failures": failures,
        "claim_status": "POST_HOC_INTERNAL_EXPLORATORY",
        "positive_claim_allowed": False,
        "external_claim_allowed": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify Method Y factorial v4")
    parser.add_argument(
        "--config", default=str(ROOT / "configs" / "method_y_image_factorial_v4.yaml")
    )
    parser.add_argument("--level", choices=("readiness", "final"), default="final")
    args = parser.parse_args()
    result = verify(config_path=Path(args.config), level=args.level)
    print(json.dumps(result, indent=2), flush=True)
    return 0 if result["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
