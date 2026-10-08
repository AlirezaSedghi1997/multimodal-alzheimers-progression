"""Independent, read-only release-integrity audit for the locked amyloid LMO run.

This checks that the stored public reports reproduce the already-committed
private outer results. It never evaluates images, refits a model, or upgrades
the post-hoc scientific claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from pathlib import Path
from typing import Any


class LMOReleaseAuditError(RuntimeError):
    pass


PUBLIC_METRIC_KEYS = frozenset(
    {
        "uno_c_index",
        "integrated_brier_score",
        "evaluation_grid_days",
        "administrative_censoring_cutoff_days",
        "n_train",
        "n_test",
        "train_events",
        "test_events",
        "test_rows_administratively_censored",
        "horizon_metrics",
        "evaluation_source",
    }
)
FORBIDDEN_KEYS = frozenset(
    {
        "patient_hash",
        "source_subject_id",
        "raw_path",
        "image_path_private",
        "event",
        "time_days",
        "original_event",
        "original_time_days",
        "oof_predictions",
        "oof_embeddings",
        "censoring_reference",
        "embedding",
    }
)
SUMMARY_KEYS = frozenset(
    {
        "schema_version", "status", "analysis_role", "outer_cells_finalized",
        "cells", "comparisons", "claim_status", "positive_claim_allowed",
        "external_validation_status", "source_space_status",
        "quantitative_equivalence_status", "limitations",
    }
)
LOCKED_LIMITATIONS = [
    "Only three local manufacturers; no independent cohort.",
    "Manufacturer-specific IBS grids differ and must not be averaged as identical estimands.",
    "FBP source-space and quantitative provenance remain blocked.",
    "The study was planned after observing the original v4 outputs and is exploratory.",
]
_DRIVE_PATH = re.compile(r"(?i)(?:^|[\s\"'])[a-z]:[\\/]")


def _load(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise LMOReleaseAuditError(f"JSON_UNREADABLE:{path}") from exc
    if not isinstance(value, dict):
        raise LMOReleaseAuditError(f"JSON_OBJECT_REQUIRED:{path}")
    return value


def _digest(value: dict[str, Any]) -> str:
    try:
        encoded = json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise LMOReleaseAuditError("NON_CANONICAL_JSON_VALUE") from exc
    return hashlib.sha256(encoded).hexdigest()


def _self_hashed(path: Path, field: str) -> dict[str, Any]:
    record = _load(path)
    body = {key: value for key, value in record.items() if key != field}
    if record.get(field) != _digest(body):
        raise LMOReleaseAuditError(f"SELF_HASH_MISMATCH:{path}")
    return body


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise LMOReleaseAuditError(f"CHECKPOINT_UNREADABLE:{path}") from exc
    return digest.hexdigest()


def _check_public(value: Any, patient_keys: set[str]) -> None:
    if isinstance(value, dict):
        if FORBIDDEN_KEYS.intersection(value):
            raise LMOReleaseAuditError("PUBLIC_ROW_LEVEL_KEY")
        for child in value.values():
            _check_public(child, patient_keys)
    elif isinstance(value, list):
        for child in value:
            _check_public(child, patient_keys)
    elif isinstance(value, str):
        if value in patient_keys:
            raise LMOReleaseAuditError("PUBLIC_SUBJECT_KEY")
        if _DRIVE_PATH.search(value) or "A4 Learn" in value:
            raise LMOReleaseAuditError("PUBLIC_LOCAL_PATH")


def compare_public_summary(
    summary: dict[str, Any], expected_cells: list[dict[str, Any]],
    manufacturer_order: list[str],
) -> None:
    if set(summary) != SUMMARY_KEYS or summary.get("schema_version") != "1.0":
        raise LMOReleaseAuditError("PUBLIC_SUMMARY_SCHEMA_INVALID")
    if summary.get("status") != "PASS_TECHNICAL_LOCAL_LMO_ONLY":
        raise LMOReleaseAuditError("SUMMARY_TECHNICAL_STATUS_INVALID")
    if summary.get("limitations") != LOCKED_LIMITATIONS:
        raise LMOReleaseAuditError("PUBLIC_LIMITATIONS_MISMATCH")
    required = {
        "analysis_role": "POST_HOC_INTERNAL_SCANNER_TRANSPORT_EXPLORATORY",
        "claim_status": "POST_HOC_INTERNAL_EXPLORATORY",
        "positive_claim_allowed": False,
        "external_validation_status": "EXTERNAL_PENDING",
        "source_space_status": "UNPROVEN_FOR_FBP_INPUTS",
        "quantitative_equivalence_status": "BLOCKED",
        "outer_cells_finalized": len(expected_cells),
    }
    for key, value in required.items():
        if summary.get(key) != value:
            raise LMOReleaseAuditError(f"SUMMARY_CLAIM_OR_STATUS_MISMATCH:{key}")
    if summary.get("cells") != expected_cells:
        raise LMOReleaseAuditError("PUBLIC_SUMMARY_DIFFERS_FROM_COMMITTED_RESULTS")
    expected_comparisons = []
    for manufacturer in manufacturer_order:
        pair = {
            cell["arm"]: cell
            for cell in expected_cells
            if cell["manufacturer"] == manufacturer
        }
        if set(pair) != {"A0", "A2"}:
            raise LMOReleaseAuditError("MATCHED_ARM_PAIR_MISSING")
        expected_comparisons.append(
            {
                "manufacturer": manufacturer,
                "a2_minus_a0_ibs": pair["A2"]["integrated_brier_score"]
                - pair["A0"]["integrated_brier_score"],
                "a2_minus_a0_uno": pair["A2"]["uno_c_index"]
                - pair["A0"]["uno_c_index"],
            }
        )
    if summary.get("comparisons") != expected_comparisons:
        raise LMOReleaseAuditError("PUBLIC_COMPARISON_MISMATCH")


def audit(root: Path) -> dict[str, Any]:
    root = root.resolve()
    plan = _load(root / "private" / "lmo_plan.json")
    contract = _load(root / "run_contract.json")
    state = _load(root / "orchestration_state.json")
    summary = _load(root / "reports" / "amyloid_lmo_public_summary.json")
    outer = plan.get("outer_folds")
    order = plan.get("manufacturer_order")
    if not isinstance(outer, list) or len(outer) != 3 or order != ["GE", "PHILIPS", "SIEMENS"]:
        raise LMOReleaseAuditError("THREE_PRESPECIFIED_MANUFACTURERS_REQUIRED")
    if plan.get("plan_sha256") != _digest({key: value for key, value in plan.items() if key != "plan_sha256"}):
        raise LMOReleaseAuditError("PLAN_HASH_MISMATCH")
    if contract.get("plan_sha256") != plan["plan_sha256"]:
        raise LMOReleaseAuditError("PLAN_CONTRACT_MISMATCH")
    if contract.get("digest") != _digest({key: value for key, value in contract.items() if key != "digest"}):
        raise LMOReleaseAuditError("ROOT_CONTRACT_DIGEST_MISMATCH")
    if contract.get("active_arms") != ["A0", "A2"]:
        raise LMOReleaseAuditError("A0_A2_CONTRACT_REQUIRED")
    if state.get("stage") != "FINALIZED" or state.get("outer_cells_finalized") != 6:
        raise LMOReleaseAuditError("SIX_CELLS_NOT_FINALIZED")
    if state.get("run_contract_sha256") != contract.get("digest"):
        raise LMOReleaseAuditError("ROOT_CONTRACT_MISMATCH")
    if state.get("positive_claim_allowed") is not False:
        raise LMOReleaseAuditError("UNAUTHORIZED_POSITIVE_CLAIM")

    patient_keys: set[str] = set()
    expected_cells: list[dict[str, Any]] = []
    for fold_row in outer:
        fold = int(fold_row["fold"])
        manufacturer = str(fold_row["heldout_manufacturer"])
        if manufacturer != order[fold]:
            raise LMOReleaseAuditError("FOLD_MANUFACTURER_ORDER_MISMATCH")
        patient_keys.update(str(key) for key in fold_row["test_subjects"])
        for arm in ("A0", "A2"):
            cell_dir = root / "private" / "outer_evaluation" / "amyloid" / arm / f"fold_{fold}"
            cell_state = _self_hashed(cell_dir / "state.json", "state_sha256")
            journal = _self_hashed(cell_dir / "outer_access_journal.json", "journal_sha256")
            result = _self_hashed(cell_dir / "outer_result.json", "result_record_sha256")
            checkpoint = root / "development" / "amyloid" / arm / f"outer_{fold}" / "final" / "locked_checkpoint.pt"
            checkpoint_sha = _file_sha256(checkpoint)
            run_sha = cell_state.get("run_contract_sha256")
            if (
                cell_state.get("state") != "FINALIZED"
                or cell_state.get("fold") != fold
                or cell_state.get("checkpoint_sha256") != checkpoint_sha
                or cell_state.get("result_record_sha256") != _digest(result)
                or journal.get("access_count") != 1
                or journal.get("authorization") != "TEST_MATERIALIZATION_AUTHORIZED_ONCE"
                or journal.get("checkpoint_sha256") != checkpoint_sha
                or result.get("checkpoint_sha256") != checkpoint_sha
                or any(record.get("fold") != fold or record.get("run_contract_sha256") != run_sha for record in (journal, result))
            ):
                raise LMOReleaseAuditError(f"OUTER_CELL_INTEGRITY_FAILED:{arm}:{fold}")
            metrics = result.get("metrics")
            if not isinstance(metrics, dict) or result.get("metrics_sha256") != _digest(metrics):
                raise LMOReleaseAuditError(f"METRICS_HASH_MISMATCH:{arm}:{fold}")
            ibs = float(metrics["integrated_brier_score"])
            uno = float(metrics["uno_c_index"])
            if not (math.isfinite(ibs) and 0 <= ibs <= 1 and math.isfinite(uno)):
                raise LMOReleaseAuditError(f"METRIC_INVALID:{arm}:{fold}")
            if (metrics.get("n_test"), metrics.get("test_events")) != (fold_row["test_count"], fold_row["test_events"]):
                raise LMOReleaseAuditError(f"OUTER_SUPPORT_MISMATCH:{arm}:{fold}")
            public = _load(root / "reports" / f"amyloid_{arm}_fold_{fold}_public.json")
            aggregate = {key: metrics[key] for key in PUBLIC_METRIC_KEYS if key in metrics}
            aggregate["outer_test_augmentation_applied"] = False
            if public != {
                "schema_version": "4.0",
                "aggregate": aggregate,
                "run_contract_sha256": run_sha,
                "claim_status": "POST_HOC_INTERNAL_EXPLORATORY",
            }:
                raise LMOReleaseAuditError(f"PUBLIC_FOLD_DIFFERS_FROM_PRIVATE:{arm}:{fold}")
            expected_cells.append(
                {
                    "manufacturer": manufacturer,
                    "arm": arm,
                    "n_test": int(metrics["n_test"]),
                    "test_events": int(metrics["test_events"]),
                    "uno_c_index": uno,
                    "integrated_brier_score": ibs,
                    "horizon_metrics": metrics.get("horizon_metrics", {}),
                    "evaluation_grid_days": metrics.get("evaluation_grid_days", []),
                    "checkpoint_sha256": checkpoint_sha,
                }
            )
    compare_public_summary(summary, expected_cells, order)
    for path in (root / "reports").rglob("*.json"):
        _check_public(_load(path), patient_keys)
    _check_public(summary, patient_keys)
    return {
        "status": "PASS_RELEASE_INTEGRITY_ONLY",
        "outer_cells_verified": len(expected_cells),
        "outer_access_count_each": 1,
        "public_private_metrics_equal": True,
        "public_privacy": "PASS",
        "science_gate_status": "NOT_EVALUATED_BY_THIS_AUDIT",
        "claim_status": "POST_HOC_INTERNAL_EXPLORATORY",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        report = audit(args.root)
    except (LMOReleaseAuditError, KeyError, TypeError, ValueError) as exc:
        print(json.dumps({"status": "FAIL", "reason": str(exc)}, ensure_ascii=False), flush=True)
        return 1
    print(json.dumps(report, ensure_ascii=False, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
