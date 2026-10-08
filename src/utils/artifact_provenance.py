from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ARTIFACT_STATUSES = {"MISSING", "IMPLEMENTED", "EXECUTED", "COMPLETED"}
SCIENCE_GATE_STATUSES = {"NOT_EVALUATED", "BLOCKED", "PASS"}
CLAIM_STATUSES = {"NOT_ALLOWED", "LIMITED", "ALLOWED"}


def _truthy(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().casefold() in {"1", "true", "yes"}
    return bool(value)


def audit_cache_contract(
    attributes: dict[str, Any], *, expected_folds: int = 4
) -> dict[str, Any]:
    """Validate claim-relevant HDF5 attributes without reading image tensors."""

    blockers: list[str] = []
    if str(attributes.get("resolution")) != "128":
        blockers.append("CACHE_RESOLUTION_NOT_128")
    if str(attributes.get("split_folds")) != str(expected_folds):
        blockers.append(f"CACHE_SPLIT_FOLDS_NOT_{expected_folds}")
    if _truthy(attributes.get("contains_pet_input")):
        blockers.append("CACHE_CONTAINS_PET_INPUT")
    if _truthy(attributes.get("contains_raw_identifiers")):
        blockers.append("CACHE_CONTAINS_RAW_IDENTIFIERS")
    if attributes.get("source_geometry_audit_status") != "PASS":
        blockers.append("CACHE_SOURCE_GEOMETRY_NOT_PASS")
    return {
        "status": "PASS" if not blockers else "FAIL",
        "blockers": blockers,
        "resolution": attributes.get("resolution"),
        "split_folds": attributes.get("split_folds"),
        "contains_pet_input": _truthy(attributes.get("contains_pet_input")),
        "contains_raw_identifiers": _truthy(attributes.get("contains_raw_identifiers")),
        "source_geometry_audit_status": attributes.get("source_geometry_audit_status"),
        "expected_folds": expected_folds,
    }


def status_contract(
    *, artifact_status: str, science_gate_status: str, claim_status: str
) -> dict[str, str]:
    if artifact_status not in ARTIFACT_STATUSES:
        raise ValueError(f"INVALID_ARTIFACT_STATUS:{artifact_status}")
    if science_gate_status not in SCIENCE_GATE_STATUSES:
        raise ValueError(f"INVALID_SCIENCE_GATE_STATUS:{science_gate_status}")
    if claim_status not in CLAIM_STATUSES:
        raise ValueError(f"INVALID_CLAIM_STATUS:{claim_status}")
    if claim_status == "ALLOWED" and science_gate_status != "PASS":
        raise ValueError("CLAIM_REQUIRES_SCIENCE_GATE_PASS")
    return {
        "artifact_status": artifact_status,
        "science_gate_status": science_gate_status,
        "claim_status": claim_status,
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_value(repo_root: Path, *args: str) -> str | None:
    try:
        return subprocess.check_output(
            ["git", *args],
            cwd=repo_root,
            text=True,
            encoding="utf-8",
            errors="backslashreplace",
            stderr=subprocess.DEVNULL,
            timeout=20,
        ).strip()
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None


def _git_snapshot(repo_root: Path) -> dict[str, Any]:
    commit = _git_value(repo_root, "rev-parse", "HEAD")
    branch = _git_value(repo_root, "branch", "--show-current")
    status = _git_value(repo_root, "status", "--porcelain")
    status_text = status or ""
    tracked_diff = _git_value(repo_root, "diff", "--binary", "--no-ext-diff", "HEAD", "--") or ""
    untracked_text = _git_value(repo_root, "ls-files", "--others", "--exclude-standard") or ""
    untracked_rows: list[dict[str, Any]] = []
    for relative in sorted(line for line in untracked_text.splitlines() if line):
        path = repo_root / relative
        try:
            size = path.stat().st_size if path.is_file() else None
            content_sha256 = _sha256_file(path) if path.is_file() and size is not None and size <= 1024 * 1024 else None
        except OSError:
            size = None
            content_sha256 = None
        untracked_rows.append(
            {
                "relative_path": relative.replace("\\", "/"),
                "bytes": size,
                "sha256": content_sha256,
            }
        )
    status_digest = hashlib.sha256(status_text.encode("utf-8")).hexdigest()
    tracked_diff_digest = hashlib.sha256(tracked_diff.encode("utf-8")).hexdigest()
    untracked_digest = hashlib.sha256(
        json.dumps(untracked_rows, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    dirty_digest = hashlib.sha256(
        f"{status_digest}\0{tracked_diff_digest}\0{untracked_digest}".encode("utf-8")
    ).hexdigest()
    return {
        "commit": commit,
        "branch": branch,
        "dirty": bool(status_text),
        "dirty_tree_digest": dirty_digest,
        "status_digest": status_digest,
        "tracked_diff_digest": tracked_diff_digest,
        "untracked_manifest_digest": untracked_digest,
        "untracked_file_count": len(untracked_rows),
        "untracked_content_hash_limit_bytes": 1024 * 1024,
        "dirty_entry_count": len(status_text.splitlines()) if status_text else 0,
    }


def build_provenance(
    *,
    repo_root: Path,
    inputs: Iterable[Path] = (),
    parent_artifacts: Iterable[str] = (),
) -> dict[str, Any]:
    root = repo_root.resolve()
    input_rows: list[dict[str, Any]] = []
    for source in inputs:
        path = Path(source).resolve()
        try:
            label = path.relative_to(root).as_posix()
        except ValueError:
            label = path.name
        input_rows.append(
            {
                "path": label,
                "exists": path.exists(),
                "sha256": _sha256_file(path) if path.is_file() else None,
                "bytes": path.stat().st_size if path.is_file() else None,
            }
        )
    return {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git": _git_snapshot(root),
        "inputs": input_rows,
        "parent_artifacts": [str(item).replace("\\", "/") for item in parent_artifacts],
        "raw_identifiers_exported": False,
        "raw_paths_exported": False,
    }


def audit_claim_consistency(
    *, artifact: dict[str, Any], live_spm: dict[str, Any], hparam_manifest: dict[str, Any]
) -> dict[str, Any]:
    aggregate = artifact.get("aggregate") if isinstance(artifact.get("aggregate"), dict) else artifact
    blockers: list[str] = []
    live_spm_ready = bool(live_spm.get("ready"))
    artifact_spm_ready = bool(aggregate.get("spm_complete"))
    if artifact_spm_ready and not live_spm_ready:
        blockers.append("ARTIFACT_SPM_TRUE_LIVE_GUARD_FALSE")
    branches = hparam_manifest.get("branches")
    method_x_rows = [
        row
        for row in branches
        if isinstance(row, dict) and row.get("branch") == "method_x_mri"
    ] if isinstance(branches, list) else []
    method_x_hpo_done = (
        hparam_manifest.get("status") == "DONE"
        and (not method_x_rows or all(row.get("status") == "DONE" for row in method_x_rows))
    )
    if not method_x_hpo_done:
        blockers.append("HPO_NOT_FINAL")
    if bool(aggregate.get("final_claim_allowed")) and blockers:
        blockers.append("FINAL_CLAIM_WITH_UNPASSED_LIVE_GATES")
    return {
        "status": "PASS" if not blockers else "FAIL",
        "blockers": blockers,
        "artifact_status": aggregate.get("status", "UNKNOWN"),
        "live_spm_status": live_spm.get("status", "UNKNOWN"),
        "hparam_status": hparam_manifest.get("status", "UNKNOWN"),
        "final_claim_allowed": bool(aggregate.get("final_claim_allowed")) and not blockers,
        "status_contract": status_contract(
            artifact_status="COMPLETED" if aggregate else "MISSING",
            science_gate_status="PASS" if not blockers else "BLOCKED",
            claim_status="ALLOWED" if not blockers and bool(aggregate.get("final_claim_allowed")) else "NOT_ALLOWED",
        ),
    }
