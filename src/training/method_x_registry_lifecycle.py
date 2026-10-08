"""Append-only lifecycle reconciliation for Method X registry records."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from src.training.method_x_registry import append_jsonl, sha256_file, utc_now
from src.utils.io_helpers import atomic_write_json


LIFECYCLE_NAME = "lifecycle.jsonl"
HISTORICAL_RECORDS = (
    {
        "run_id": "mx-historical-resnet10-64-3fold",
        "output_root": "results/final_results/backbone_ablation_battery/resnet10_3d",
        "status": "SUPERSEDED",
        "original_status": "COMPLETED_DEVELOPMENT_3FOLD",
        "phase": "historical_non_comparable",
        "reason": "NON_COMPARABLE_64_3FOLD",
        "backbone": "resnet10_3d",
        "resolution": 64,
        "folds": 3,
    },
    {
        "run_id": "mx-historical-hpo-corrected-v2",
        "output_root": "results/final_results/method_x_5fold_corrected_v2/hpo",
        "status": "SUPERSEDED",
        "original_status": "PAUSED_OR_INCOMPLETE",
        "phase": "historical_incomplete_hpo",
        "reason": "HPO_PAUSED_OR_INCOMPLETE_ZERO_COMPLETED_CANDIDATES",
        "backbone": "resnet10_3d",
        "resolution": 128,
        "folds": 4,
        "hpo_summary": {"accepted_candidates": 10, "completed_candidates": 0},
    },
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            rows.append(value)
    return rows


def _root_key(value: Any) -> str:
    return str(value or "").replace("\\", "/").casefold()


def choose_parent_run_id(
    records: Iterable[dict[str, Any]], *, current_run_id: str, current_output_root: str | None = None
) -> str | None:
    """Choose the latest compatible predecessor without rewriting history."""

    rows = [
        row
        for row in records
        if row.get("run_id") != current_run_id
        and str(row.get("status") or "").upper() not in {"SUPERSEDED", "INVALID", "ABORTED"}
        and not str(row.get("run_id") or "").startswith("mx-historical-")
    ]
    if not rows:
        return None
    current_key = _root_key(current_output_root)
    same_root = [row for row in rows if current_key and _root_key(row.get("output_root")) == current_key]
    if same_root:
        return same_root[-1].get("run_id")
    if "method_x_model_battery_v3" in current_key:
        v2_rows = [
            row
            for row in rows
            if "method_x_model_battery_v2" in _root_key(row.get("output_root"))
        ]
        if v2_rows:
            return v2_rows[-1].get("run_id")
    return rows[-1].get("run_id")


def _historical_record(root: Path, spec: dict[str, Any]) -> dict[str, Any]:
    evidence: list[dict[str, str]] = []
    path = root / spec["output_root"]
    if path.is_file():
        digest = sha256_file(path)
        if digest:
            evidence.append({"path": spec["output_root"], "sha256": digest})
    elif path.is_dir():
        for candidate in sorted(path.rglob("*.json"))[:8]:
            digest = sha256_file(candidate)
            if digest:
                evidence.append({"path": str(candidate.relative_to(root)).replace("\\", "/"), "sha256": digest})
    return {
        "schema_version": 1,
        "event_id": uuid.uuid4().hex,
        "run_id": spec["run_id"],
        "parent_run_id": None,
        "recorded_at_utc": utc_now(),
        "status": spec["status"],
        "original_status": spec["original_status"],
        "phase": spec["phase"],
        "method": "X",
        "output_root": spec["output_root"],
        "claim_boundary": "PET_FREE_INTERNAL_TECHNICAL_BENCHMARK_ONLY",
        "historical_comparability": {
            "status": "REJECTED",
            "reason": spec["reason"],
            "resolution": spec["resolution"],
            "folds": spec["folds"],
        },
        "models": [{"backbone": spec["backbone"], "status": "SUPERSEDED"}],
        "hpo_summary": spec.get("hpo_summary"),
        "final_claim_allowed": False,
        "autonomous_supervision": {
            "mode": "AUTONOMOUS_MULTI_LAYER_FAIL_CLOSED",
            "human_approval_required": False,
            "manual_override_allowed": False,
            "decision": "SUPERSEDED",
        },
        "artifacts": evidence,
    }


def reconcile_registry(
    *, root: Path, registry_root: Path, current_run_id: str | None = None, current_output_root: str | None = None
) -> dict[str, Any]:
    """Append parent, supersession and blocker-lifecycle events idempotently."""

    registry_root.mkdir(parents=True, exist_ok=True)
    runs_path = registry_root / "runs.jsonl"
    lifecycle_path = registry_root / LIFECYCLE_NAME
    records = read_jsonl(runs_path)
    if not records:
        raise RuntimeError("METHOD_X_REGISTRY_EMPTY")
    current = None
    if current_run_id:
        current = next((row for row in records if row.get("run_id") == current_run_id), None)
    if current is None and current_output_root:
        key = _root_key(current_output_root)
        current = next((row for row in reversed(records) if _root_key(row.get("output_root")) == key), None)
    if current is None:
        current = records[-1]
    current_id = str(current.get("run_id"))
    declared_parent = current.get("parent_run_id")
    valid_record_ids = {str(row.get("run_id")) for row in records if row.get("run_id")}
    parent_id = (
        str(declared_parent)
        if declared_parent and str(declared_parent) in valid_record_ids
        else choose_parent_run_id(
            records,
            current_run_id=current_id,
            current_output_root=str(current.get("output_root") or current_output_root or ""),
        )
    )
    existing_lifecycle = read_jsonl(lifecycle_path)
    existing_keys = {
        (row.get("event_type"), row.get("run_id"), row.get("parent_run_id"), row.get("code"))
        for row in existing_lifecycle
    }
    appended: list[dict[str, Any]] = []

    def event(event_type: str, **payload: Any) -> None:
        key = (event_type, payload.get("run_id"), payload.get("parent_run_id"), payload.get("code"))
        if key in existing_keys:
            return
        row = {
            "schema_version": 1,
            "event_id": uuid.uuid4().hex,
            "recorded_at_utc": utc_now(),
            "event_type": event_type,
            **payload,
        }
        append_jsonl(lifecycle_path, row)
        existing_keys.add(key)
        appended.append(row)

    if parent_id:
        event("PARENT_LINKED", run_id=current_id, parent_run_id=parent_id, relation="versioned_continuation")

    known_ids = {str(row.get("run_id")) for row in records}
    historical_added: list[str] = []
    historical_classified: list[str] = []
    for spec in HISTORICAL_RECORDS:
        if spec["run_id"] not in known_ids:
            record = _historical_record(root, spec)
            append_jsonl(runs_path, record)
            known_ids.add(spec["run_id"])
            historical_added.append(spec["run_id"])
            event(
                "RUN_SUPERSEDED",
                run_id=spec["run_id"],
                superseded_by=current_id,
                reason=spec["reason"],
            )
        event(
            "HISTORICAL_RUN_CLASSIFIED",
            run_id=spec["run_id"],
            code=spec["reason"],
            original_status=spec["original_status"],
            terminal_registry_status="SUPERSEDED",
            superseded_by=current_id,
            resolution=spec["resolution"],
            folds=spec["folds"],
            hpo_summary=spec.get("hpo_summary"),
            claim_status="NON_COMPARABLE_DO_NOT_USE_FOR_FINAL_MODEL_RANKING",
        )
        historical_classified.append(spec["run_id"])

    blockers = read_jsonl(registry_root / "blockers.jsonl")
    by_code: dict[str, list[dict[str, Any]]] = {}
    for blocker in blockers:
        code = str(blocker.get("code") or "UNKNOWN")
        by_code.setdefault(code, []).append(blocker)
    blocker_reconciled: list[str] = []
    for code, rows in by_code.items():
        run_ids = list(dict.fromkeys(str(row.get("run_id")) for row in rows if row.get("run_id")))
        if len(run_ids) <= 1:
            continue
        event(
            "BLOCKER_HISTORY_RECONCILED",
            run_id=current_id,
            code=code,
            historical_run_ids=run_ids[:-1],
            current_evidence_run_id=run_ids[-1],
            resolution_status="RETAINED_OPEN_IF_CURRENT_EVIDENCE_STILL_BLOCKS",
        )
        blocker_reconciled.append(code)
        duplicate_open_rows = [
            row for row in rows if str(row.get("status") or "OPEN").upper() == "OPEN"
        ]
        if len(duplicate_open_rows) > 1:
            event(
                "BLOCKER_DUPLICATE_OPEN_HISTORY_CLOSED",
                run_id=current_id,
                code=code,
                closed_duplicate_count=len(duplicate_open_rows) - 1,
                retained_event_id=duplicate_open_rows[0].get("event_id"),
                closed_event_ids=[row.get("event_id") for row in duplicate_open_rows[1:]],
                resolution_status="CLOSED_DUPLICATE_HISTORY_ONLY",
            )
    # Materialize the latest state without rewriting the append-only evidence.
    # A code absent from the current run is RESOLVED, never silently deleted.
    recorded_snapshot = current.get("active_blocker_codes")
    if isinstance(recorded_snapshot, list):
        active_current_codes = {str(code) for code in recorded_snapshot}
    else:
        active_current_codes = {
            str(row.get("code"))
            for row in blockers
            if str(row.get("run_id")) == current_id
            and str(row.get("status", "OPEN")).upper() in {"OPEN", "REOPENED"}
        }
    resolved_codes: list[str] = []
    closed_codes: list[str] = []
    blocker_state: dict[str, dict[str, Any]] = {}
    for code, rows in by_code.items():
        ordered = list(rows)
        latest = ordered[-1]
        latest_status = str(latest.get("status") or "OPEN").upper()
        state = "OPEN" if code in active_current_codes else "RESOLVED"
        blocker_state[code] = {
            "status": state,
            "first_run_id": str(ordered[0].get("run_id")) if ordered else None,
            "latest_evidence_run_id": current_id if state == "OPEN" else str(latest.get("run_id")),
            "transition_event_count": len(ordered),
            "legacy_occurrence_count": len(ordered),
            "latest_transition_status": latest_status,
        }
        if state == "RESOLVED":
            if latest_status == "CLOSED":
                event(
                    "BLOCKER_CLOSED",
                    run_id=current_id,
                    code=code,
                    closed_from_run_id=str(latest.get("run_id")),
                    resolution_status=str(latest.get("resolution_status") or "CLOSED"),
                )
                closed_codes.append(code)
            elif latest_status == "RESOLVED" or any(
                str(row.get("status", "OPEN")).upper() in {"OPEN", "REOPENED"}
                for row in ordered
            ):
                event(
                    "BLOCKER_RESOLVED",
                    run_id=current_id,
                    code=code,
                    resolved_from_run_id=str(
                        latest.get("resolved_from_run_id") or latest.get("run_id")
                    ),
                    resolution_status=str(
                        latest.get("resolution_status")
                        or "NO_LONGER_PRESENT_IN_CURRENT_RUN"
                    ),
                )
                resolved_codes.append(code)
    atomic_write_json(registry_root / "blocker_state.json", {
        "schema_version": 1,
        "updated_at_utc": utc_now(),
        "current_run_id": current_id,
        "states": blocker_state,
        "policy": "append_only_evidence_with_derived_latest_state",
    })
    event(
        "REGISTRY_RECONCILED",
        run_id=current_id,
        parent_run_id=parent_id,
        historical_records_added=historical_added,
        blocker_codes_reconciled=blocker_reconciled,
    )
    return {
        "status": "RECONCILED",
        "current_run_id": current_id,
        "parent_run_id": parent_id,
        "historical_records_added": historical_added,
        "historical_records_classified": historical_classified,
        "blocker_codes_reconciled": blocker_reconciled,
        "blocker_codes_resolved": sorted(resolved_codes),
        "blocker_codes_closed": sorted(closed_codes),
        "events_appended": len(appended),
        "lifecycle_path": str(lifecycle_path),
    }
