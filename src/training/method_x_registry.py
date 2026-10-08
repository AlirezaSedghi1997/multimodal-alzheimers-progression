"""Append-only Method X run registry helpers."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from src.training.method_x_spm_snapshot import DEFAULT_SNAPSHOT_REL, verify_spm_gate_snapshot
from src.training.method_x_stage_registry import collect_stage_readiness


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_snapshot(root: Path) -> dict[str, Any]:
    def run(*args: str) -> str:
        result = subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, check=False)
        return (result.stdout or "").strip()

    status = run("status", "--porcelain")
    return {
        "branch": run("branch", "--show-current"),
        "commit": run("rev-parse", "HEAD"),
        "dirty": bool(status),
        "status_line_count": len(status.splitlines()),
    }


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(line)
        handle.flush()
        os.fsync(handle.fileno())


def _read_jsonl_records(path: Path) -> list[dict[str, Any]]:
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


def relative_or_string(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def build_run_record(
    *,
    root: Path,
    output_root: Path,
    manifest: dict[str, Any],
    audit: dict[str, Any] | None,
    parent_run_id: str | None = None,
) -> dict[str, Any]:
    live_spm = verify_spm_gate_snapshot(
        root / DEFAULT_SNAPSHOT_REL,
        root=root,
        verify_sources=True,
    )
    manifest_spm = (
        manifest.get("spm_gate_snapshot")
        if isinstance(manifest.get("spm_gate_snapshot"), dict)
        else {}
    )
    expected_spm_digest = str(manifest_spm.get("snapshot_sha256") or "").strip().lower()
    observed_spm_digest = str(live_spm.get("snapshot_sha256") or "").strip().lower()
    digest_is_valid = lambda value: len(value) == 64 and all(
        char in "0123456789abcdef" for char in value
    )
    if (
        not live_spm.get("verified")
        or not live_spm.get("ready")
        or not digest_is_valid(expected_spm_digest)
        or not digest_is_valid(observed_spm_digest)
        or expected_spm_digest != observed_spm_digest
    ):
        raise RuntimeError(
            "REGISTRY_SPM_SNAPSHOT_BINDING_MISMATCH:"
            f"expected={expected_spm_digest or 'MISSING'}:"
            f"observed={observed_spm_digest or 'MISSING'}"
        )
    spm_snapshot_binding = {
        "status": "PASS",
        "manifest_sha256": expected_spm_digest,
        "verified_snapshot_sha256": observed_spm_digest,
        "verified": True,
        "ready": True,
    }
    run_id = f"mx-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:8]}"
    locked = manifest.get("locked_inputs", {})
    artifacts: list[dict[str, Any]] = []
    for relative in (
        "battery_manifest.json",
        "reports/state_reconciliation.json",
        "reports/matched_benchmark_preflight.json",
        "reports/model_comparison.json",
        "reports/mx03_validity_suite.json",
        "reports/mx03_validity_orchestrator.json",
        "reports/sanity_report.json",
        "reports/oof_comparison.json",
    ):
        path = output_root / relative
        digest = sha256_file(path)
        if digest:
            artifacts.append({"path": relative_or_string(path, root), "sha256": digest})
    # Stage reports are machine-produced gates.  Include their hashes in every
    # registry record so a later thesis report cannot silently mix readiness
    # state from another output root.
    stage_readiness = collect_stage_readiness(root=root, output_root=output_root)
    stage_paths = [
        output_root / "reports/mx03_validity_suite.json",
        output_root / "reports/mx04_finetune_readiness.json",
        output_root / "reports/mx05_hybrid_readiness.json",
        output_root / "reports/method_x_stage_readiness.json",
        output_root / "reports/method_x_autonomous_program.json",
        root / "results/final_results/method_x_anatomical_xai_v2/anatomical_xai_summary.json",
    ]
    known_artifact_paths = {str(item.get("path")) for item in artifacts}
    for path in stage_paths:
        digest = sha256_file(path)
        if digest:
            relative = relative_or_string(path, root)
            if relative not in known_artifact_paths:
                artifacts.append({"path": relative, "sha256": digest})
                known_artifact_paths.add(relative)
    autosupervision_path = root / "results/method_x_registry/autosupervision_events.jsonl"
    autosupervision_digest = sha256_file(autosupervision_path)
    if autosupervision_digest:
        artifacts.append(
            {
                "path": relative_or_string(autosupervision_path, root),
                "sha256": autosupervision_digest,
            }
        )
    snapshot_path = root / "results/method_x_registry/spm_gate_snapshot.json"
    snapshot_digest = sha256_file(snapshot_path)
    if snapshot_digest:
        artifacts.append({"path": relative_or_string(snapshot_path, root), "sha256": snapshot_digest})
    resume_path = root / "results/method_x_registry/resume_acceptance.json"
    resume_report: dict[str, Any] | None = None
    if resume_path.is_file():
        try:
            parsed_resume = json.loads(resume_path.read_text(encoding="utf-8"))
            if isinstance(parsed_resume, dict):
                resume_report = parsed_resume
        except (OSError, json.JSONDecodeError):
            resume_report = {"status": "INVALID_JSON"}
        resume_digest = sha256_file(resume_path)
        if resume_digest:
            artifacts.append({"path": relative_or_string(resume_path, root), "sha256": resume_digest})
    # Traceability, the final audit, and the delivery packet are derived from
    # this append-only record.  Embedding or hashing them here would create a
    # provenance cycle.  Stable path-only links retain discoverability while
    # the downstream delivery manifest binds their final hashes.
    downstream_artifacts = {
        "thesis_traceability": "results/method_x_registry/thesis_traceability.json",
        "final_completion_audit": "results/method_x_registry/final_completion_audit.json",
        "immutable_delivery_manifest": "results/method_x_registry/immutable_delivery_manifest.json",
    }
    data = audit or {}
    totals = data.get("totals", {}) if isinstance(data, dict) else {}

    model_records: list[dict[str, Any]] = []
    for row in manifest.get("models", []):
        model_record: dict[str, Any] = {
            "backbone": row.get("backbone"),
            "status": row.get("status"),
            "smoke_status": row.get("smoke", {}).get("status"),
            "training_status": row.get("training", {}).get("status"),
        }
        metrics_path = output_root / str(row.get("output_dir", "")) / "three_d_metrics.json"
        if row.get("status") == "COMPLETED" and metrics_path.is_file():
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            aggregate = metrics.get("aggregate", {}) if isinstance(metrics, dict) else {}
            model_record["aggregate_metrics"] = {
                key: aggregate.get(key)
                for key in (
                    "roc_auc",
                    "pr_auc",
                    "brier",
                    "ece",
                    "cohort_probe_auc",
                    "completed_folds",
                    "status",
                )
            }
            model_record["fold_metrics"] = [
                {
                    key: fold.get(key)
                    for key in (
                        "fold",
                        "roc_auc",
                        "pr_auc",
                        "brier",
                        "real_fit_n",
                        "augmented_fit_n",
                        "fit_exposure_n",
                    )
                }
                for fold in metrics.get("folds", [])
                if isinstance(fold, dict)
            ]
            metrics_sha = sha256_file(metrics_path)
            if metrics_sha:
                artifacts.append({"path": relative_or_string(metrics_path, root), "sha256": metrics_sha})
        model_records.append(model_record)
    validity_path = output_root / "reports/mx03_validity_suite.json"
    validity_report = None
    if validity_path.is_file():
        try:
            validity_report = json.loads(validity_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            validity_report = {"status": "INVALID_JSON"}
    program_path = output_root / "reports/method_x_autonomous_program.json"
    program_report: dict[str, Any] | None = None
    if program_path.is_file():
        try:
            parsed_program = json.loads(program_path.read_text(encoding="utf-8"))
            if isinstance(parsed_program, dict):
                program_report = parsed_program
        except (OSError, json.JSONDecodeError):
            program_report = {"status": "INVALID_JSON"}
    return {
        "schema_version": 1,
        "event_id": uuid.uuid4().hex,
        "run_id": run_id,
        "parent_run_id": parent_run_id,
        "recorded_at_utc": utc_now(),
        "status": manifest.get("status", "UNKNOWN"),
        "phase": "matched_battery_final" if manifest.get("status") in {"COMPLETED", "COMPLETED_WITH_BLOCKS"} else "matched_battery_reconciliation",
        "method": "X",
        "output_root": relative_or_string(output_root, root),
        "claim_boundary": "PET_FREE_INTERNAL_TECHNICAL_BENCHMARK_ONLY",
        "git": git_snapshot(root),
        "protocol": manifest.get("locked_protocol", {}),
        "locked_inputs": locked,
        "spm_gate_snapshot": manifest.get("spm_gate_snapshot", {}),
        "spm_snapshot_binding": spm_snapshot_binding,
        "autonomous_supervision": manifest.get(
            "autosupervision",
            {
                "mode": "AUTONOMOUS_MULTI_LAYER_FAIL_CLOSED",
                "human_approval_required": False,
                "manual_override_allowed": False,
            },
        ),
        "data": {
            "source_counts": data.get("source_counts", {}),
            "real_fit_exposures": totals.get("real_fit_exposures"),
            "augmented_fit_exposures": totals.get("augmented_fit_exposures"),
            "fit_exposures": totals.get("fit_exposures"),
            "folds_detail": data.get("folds_detail", []),
        },
        "models": model_records,
        "mx03_validity": validity_report or {"status": "NOT_RUN"},
        "stage_readiness": stage_readiness,
        "mx03_validity_orchestrator": stage_readiness.get("validity_orchestrator", {"status": "NOT_RUN"}),
        "mx04_finetune_readiness": stage_readiness.get("stages", {}).get("MX-04", {"status": "NOT_RUN"}),
        "mx05_hybrid_readiness": stage_readiness.get("stages", {}).get("MX-05", {"status": "NOT_RUN"}),
        "mx06_xai_readiness": stage_readiness.get("stages", {}).get("MX-06", {"status": "NOT_RUN"}),
        "autonomous_program": program_report or {"status": "NOT_RUN"},
        "resume_acceptance": resume_report or {"status": "NOT_RUN"},
        "downstream_artifacts": downstream_artifacts,
        "science_audit": manifest.get("science_audit", {}),
        "final_claim_allowed": False,
        "artifacts": artifacts,
    }


def append_run_bundle(
    *,
    root: Path,
    registry_root: Path,
    record: dict[str, Any],
    blockers: Iterable[dict[str, Any]] = (),
) -> None:
    blocker_rows = [dict(value) for value in blockers]
    current_by_code: dict[str, dict[str, Any]] = {}
    for blocker in blocker_rows:
        code = str(blocker.get("code") or "UNKNOWN")
        current_by_code.setdefault(code, blocker)
    record_to_write = {
        **record,
        "active_blocker_codes": sorted(current_by_code),
        "blocker_transition_policy": "OPEN_RESOLVED_REOPENED_NO_DUPLICATE_OPEN",
    }
    append_jsonl(registry_root / "runs.jsonl", record_to_write)
    append_jsonl(
        registry_root / "decisions.jsonl",
        {
            "schema_version": 1,
            "event_id": uuid.uuid4().hex,
            "run_id": record["run_id"],
            "recorded_at_utc": utc_now(),
            "decision": "RETAIN_INTERNAL_TECHNICAL_ONLY",
            "reason": "PET-free Method X remains cohort-confounded and lacks independent external validation.",
            "chapter_links": ["Method", "Results", "Discussion"],
        },
    )
    append_jsonl(
        registry_root / "interpretations.jsonl",
        {
            "schema_version": 1,
            "event_id": uuid.uuid4().hex,
            "run_id": record["run_id"],
            "recorded_at_utc": utc_now(),
            "observation": "Matched battery state and fold-local source counts were reconciled.",
            "evidence": record.get("artifacts", []),
            "statistical_meaning": "Counts describe exposure rows, not independent subjects.",
            "limitation": "Cohort-label confounding blocks diagnostic or external claims.",
            "allowed_wording": "internal PET-free technical benchmark",
            "chapter": "Results",
        },
    )
    blocker_path = registry_root / "blockers.jsonl"
    prior_events = _read_jsonl_records(blocker_path)
    latest_by_code: dict[str, dict[str, Any]] = {}
    for event in prior_events:
        latest_by_code[str(event.get("code") or "UNKNOWN")] = event
    active_statuses = {"OPEN", "REOPENED"}

    # Append only state transitions.  The per-run active snapshot above
    # preserves current evidence without growing one duplicate OPEN row per
    # audit/registry snapshot.
    for code, blocker in sorted(current_by_code.items()):
        previous = latest_by_code.get(code)
        previous_status = str((previous or {}).get("status") or "").upper()
        if previous_status in active_statuses:
            continue
        transition_status = "REOPENED" if previous is not None else "OPEN"
        append_jsonl(
            blocker_path,
            {
                "schema_version": 1,
                "event_id": uuid.uuid4().hex,
                "run_id": record["run_id"],
                "recorded_at_utc": utc_now(),
                **blocker,
                "code": code,
                "status": transition_status,
                "previous_event_id": (previous or {}).get("event_id"),
            },
        )
    for code, previous in sorted(latest_by_code.items()):
        previous_status = str(previous.get("status") or "").upper()
        if previous_status not in active_statuses or code in current_by_code:
            continue
        append_jsonl(
            blocker_path,
            {
                "schema_version": 1,
                "event_id": uuid.uuid4().hex,
                "run_id": record["run_id"],
                "recorded_at_utc": utc_now(),
                "status": "RESOLVED",
                "code": code,
                "severity": previous.get("severity"),
                "description": "Blocker is no longer present in the current run snapshot.",
                "resolved_from_event_id": previous.get("event_id"),
                "resolved_from_run_id": previous.get("run_id"),
                "resolution_status": "NO_LONGER_PRESENT_IN_CURRENT_RUN",
            },
        )
