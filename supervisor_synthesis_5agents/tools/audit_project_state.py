"""Read-only audit of the live repository and Method X/Y artifacts."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from core import (
    atomic_write_json,
    disk_snapshot,
    git_snapshot,
    path_fingerprint,
    process_snapshot,
    relevant_processes,
    utc_now,
)


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _count_manifest(path: Path) -> dict[str, Any]:
    payload = _read_json(path)
    if payload is None:
        return {"path": str(path), "status": "UNREADABLE_OR_MISSING"}
    rows = payload if isinstance(payload, list) else payload.get("rows", payload.get("items", []))
    if not isinstance(rows, list):
        return {"path": str(path), "status": "READ", "shape": type(payload).__name__}
    status_counts: dict[str, int] = {}
    for row in rows:
        if isinstance(row, dict):
            status = str(row.get("status", row.get("qc_status", row.get("transform_status", "UNKNOWN"))))
            status_counts[status] = status_counts.get(status, 0) + 1
    return {
        "path": str(path),
        "status": "READ",
        "row_count": len(rows),
        "status_counts": status_counts,
    }


def _artifact_summary(project_root: Path) -> dict[str, Any]:
    results = project_root / "results"
    candidates = [
        results / "method_y_revival_v2" / "orchestration_state.json",
        results / "method_y_revival_v2" / "reports" / "FINAL_STATUS.md",
        results / "method_y_revival_v2" / "reports" / "LIVE_STATUS_V2.json",
        results / "method_y_revival_v2" / "transforms" / "manifest.json",
        results / "method_y_revival_v2" / "external_validation" / "status.json",
        results / "method_y_revival_v2" / "public_bundle" / "manifest.json",
        results / "method_y_revival_v2" / "reports" / "longitudinal_pacc_status.json",
    ]
    summary: dict[str, Any] = {}
    for path in candidates:
        if path.suffix.lower() == ".json":
            summary[path.relative_to(project_root).as_posix()] = _read_json(path)
        else:
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
                summary[path.relative_to(project_root).as_posix()] = {
                    "sha256": __import__("hashlib").sha256(text.encode("utf-8")).hexdigest(),
                    "first_lines": text.splitlines()[:20],
                    "line_count": len(text.splitlines()),
                }
            except OSError:
                summary[path.relative_to(project_root).as_posix()] = None
    manifest = project_root / "results" / "method_y_revival_v2" / "transforms" / "manifest.json"
    summary["transform_manifest_summary"] = _count_manifest(manifest)
    return summary


def _component_reuse(project_root: Path) -> list[dict[str, Any]]:
    relative_paths = [
        "src/method_y_revival",
        "src",
        "scripts",
        "configs",
        "results/method_y_revival_v2",
        "results/method_y_execution",
        "results/method_y_tau_progression",
        "data/cached/t1_volume_cache_128.h5",
        "data/processed/mri_features.csv",
        "data/processed/clinical_features.csv",
        "data/processed/clinical_longitudinal.csv",
        "tests",
    ]
    rows: list[dict[str, Any]] = []
    for rel in relative_paths:
        path = project_root / rel
        exists = path.exists()
        file_count = 0
        if exists and path.is_dir():
            for child in path.rglob("*"):
                if child.is_file():
                    file_count += 1
                    if file_count >= 5000:
                        break
        rows.append(
            {
                "component": rel,
                "exists": exists,
                "file_count": file_count,
                "reuse_status": "AVAILABLE" if exists else "MISSING_OR_NOT_READY",
                "adaptation_days": 1 if exists else 5,
                "notes": "read-only inventory; no component was modified",
            }
        )
    return rows


def audit_live_state(project_root: Path, output_root: Path) -> dict[str, Any]:
    process = process_snapshot()
    audit = {
        "schema_version": "1.0",
        "captured_utc": utc_now(),
        "project_root": str(project_root),
        "git": git_snapshot(project_root),
        "process": process,
        "relevant_active_processes": relevant_processes(process),
        "disk": disk_snapshot(),
        "gpu": _gpu_snapshot(),
        "artifacts": _artifact_summary(project_root),
        "component_reuse": _component_reuse(project_root),
        "protected_paths": {
            rel: path_fingerprint(
                project_root / rel,
                max_files=5000,
                exclude_relative=("method_y_revival_v2/transforms",)
                if rel == "results"
                else (),
            )
            for rel in ("src", "scripts", "configs", "data", "results")
        },
        "active_writer_path": str(project_root / "results" / "method_y_revival_v2" / "transforms"),
        "external_data": {
            "DUA_status": "PENDING_DUA_ACCESS",
            "raw_external_manifest_available": False,
            "claim_status": "NOT_AVAILABLE",
        },
        "safety": {
            "active_transform_writer_observed": any(
                "method_y_transforms_manual" in str(row.get("CommandLine", "")).lower()
                for row in process.get("processes", [])
            ),
            "workspace_targeted_by_active_writer": any(
                "supervisor_synthesis_5agents" in str(row.get("CommandLine", "")).lower()
                and "run_synthesis.py" not in str(row.get("CommandLine", "")).lower()
                and "pytest" not in str(row.get("CommandLine", "")).lower()
                for row in process.get("processes", [])
            ),
            "writer_was_stopped": False,
            "writer_was_modified": False,
        },
    }
    output_root.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output_root / "audit_snapshot.json", audit)
    return audit


def _gpu_snapshot() -> dict[str, Any]:
    import subprocess

    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.used,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        return {"raw": result.stdout.strip(), "returncode": result.returncode}
    except OSError as exc:
        return {"raw": "", "error": str(exc)}


def write_safety_snapshot(project_root: Path, output_root: Path, label: str) -> dict[str, Any]:
    snapshot = audit_live_state(project_root, output_root)
    snapshot["snapshot_label"] = label
    atomic_write_json(output_root / f"{label}.json", snapshot)
    return snapshot
