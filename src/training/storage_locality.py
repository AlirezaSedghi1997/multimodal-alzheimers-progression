"""Fail-closed Method X storage-locality policy checks.

This module only inspects paths.  It never creates, moves, deletes, or scans
user Desktop content.  The sole C-drive exception is a narrowly shaped
active-resume checkpoint payload below the configured staging root.
"""

from __future__ import annotations

import fnmatch
import hashlib
import json
import os
import shutil
import stat
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, NamedTuple, Sequence

import yaml


STORAGE_LOCALITY_SCHEMA_VERSION = 1
_REPARSE_POINT_ATTRIBUTE = 0x0400

# Cadence-1 staging rotates ~400 MB files on C. A watch probe that collides
# with os.replace can see a transient Windows lock and report UNREADABLE.
# That is not scientific drift. Preflight still fail-closes on the same code.
WATCH_DEFERRED_INVENTORY_BLOCKERS = frozenset({"C_STAGING_INVENTORY_UNREADABLE"})
RESOLVED_PATH_STORAGE_REVALIDATION_FAILED = "RESOLVED_PATH_STORAGE_REVALIDATION_FAILED"


def _reason_tuple(values: Any) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        return ()
    return tuple(str(item) for item in values if str(item))


def nested_resolved_storage_blockers(report: Mapping[str, Any] | None) -> tuple[str, ...]:
    """Return blockers from a path_resolution revalidation of lexical storage."""

    if not isinstance(report, Mapping):
        return ()
    layers = report.get("layers")
    if not isinstance(layers, Sequence) or isinstance(layers, (str, bytes)):
        return ()
    ordered: list[str] = []
    seen: set[str] = set()
    for layer in layers:
        if not isinstance(layer, Mapping):
            continue
        if str(layer.get("name") or "") != "path_resolution":
            continue
        evidence = layer.get("evidence")
        if not isinstance(evidence, Mapping):
            continue
        resolved = evidence.get("resolved_storage")
        if not isinstance(resolved, Mapping):
            continue
        inner = resolved.get("evidence")
        candidates: list[str] = []
        if isinstance(inner, Mapping):
            candidates.extend(_reason_tuple(inner.get("blockers")))
        candidates.extend(_reason_tuple(resolved.get("findings")))
        for item in candidates:
            if item not in seen:
                seen.add(item)
                ordered.append(item)
    return tuple(ordered)


def effective_watch_inventory_blockers(
    blockers: Sequence[Any] | Mapping[str, Any] | None,
) -> tuple[str, ...]:
    """Return watch blockers, unwrapping inventory-only resolved-path failures.

    The first storage probe can PASS while the immediate resolved-path
    revalidation hits the same Windows lock and wraps it as
    ``RESOLVED_PATH_STORAGE_REVALIDATION_FAILED``. That wrapper is not itself
    a scientific or quota failure.
    """

    if isinstance(blockers, Mapping):
        top = _reason_tuple(blockers.get("blocking_reasons"))
        top_set = set(top)
        if top_set <= WATCH_DEFERRED_INVENTORY_BLOCKERS:
            return top
        if top_set == {RESOLVED_PATH_STORAGE_REVALIDATION_FAILED}:
            nested = nested_resolved_storage_blockers(blockers)
            if nested and set(nested) <= WATCH_DEFERRED_INVENTORY_BLOCKERS:
                return nested
        return top
    return _reason_tuple(blockers)


def watch_blockers_are_live_inventory_race(
    blockers: Sequence[Any] | Mapping[str, Any] | None,
) -> bool:
    """True when the only watch blocker is a transient C inventory read."""

    reasons = set(effective_watch_inventory_blockers(blockers))
    return bool(reasons) and reasons <= WATCH_DEFERRED_INVENTORY_BLOCKERS


class DiskUsage(NamedTuple):
    total: int
    used: int
    free: int


@dataclass(frozen=True)
class StoragePolicy:
    schema_version: int
    policy_id: str
    authoritative_project_root: Path
    checkpoint_staging_root: Path
    minimum_free_gb: int
    maximum_staging_gb: int
    permitted_basenames: tuple[str, ...]
    permitted_temp_patterns: tuple[str, ...]


def _absolute_lexical(path: Path) -> Path:
    """Make a path absolute without resolving away a reparse-point component."""

    return path if path.is_absolute() else path.absolute()


def load_storage_policy(path: Path) -> StoragePolicy:
    """Load the versioned policy and reject incomplete or malformed contracts."""

    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise ValueError("STORAGE_POLICY_MISSING_OR_INVALID") from exc
    if not isinstance(payload, dict):
        raise ValueError("STORAGE_POLICY_MISSING_OR_INVALID")
    staging = payload.get("checkpoint_staging")
    if not isinstance(staging, dict):
        raise ValueError("STORAGE_POLICY_CHECKPOINT_STAGING_MISSING")
    try:
        policy = StoragePolicy(
            schema_version=int(payload["schema_version"]),
            policy_id=str(payload["policy_id"]),
            authoritative_project_root=_absolute_lexical(Path(str(payload["authoritative_project_root"]))),
            checkpoint_staging_root=_absolute_lexical(Path(str(staging["root"]))),
            minimum_free_gb=int(staging["minimum_free_gb"]),
            maximum_staging_gb=int(staging["maximum_staging_gb"]),
            permitted_basenames=tuple(str(value) for value in staging["permitted_basenames"]),
            permitted_temp_patterns=tuple(str(value) for value in staging["permitted_temp_patterns"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("STORAGE_POLICY_FIELDS_INVALID") from exc
    if (
        policy.schema_version != STORAGE_LOCALITY_SCHEMA_VERSION
        or not policy.policy_id
        or policy.minimum_free_gb < 0
        or policy.maximum_staging_gb < 0
        or not policy.permitted_basenames
    ):
        raise ValueError("STORAGE_POLICY_FIELDS_INVALID")
    return policy


def _result(path: Path, *, allowed: bool, category: str, reason: str) -> dict[str, Any]:
    try:
        requested = str(path.resolve(strict=False))
    except OSError:
        requested = str(path)
    return {
        "allowed": allowed,
        "category": category,
        "reason": reason,
        "requested_path": requested,
    }


def _inspect_path_components(path: Path, *, lstat: Callable[[Path], os.stat_result]) -> str:
    """Return CLEAR, REPARSE_POINT, or UNREADABLE for existing components."""

    anchor = Path(path.anchor)
    current = anchor
    parts = path.parts[1:] if path.anchor else path.parts
    for part in parts:
        current = current / part
        try:
            metadata = lstat(current)
        except FileNotFoundError:
            continue
        except OSError:
            return "UNREADABLE"
        if stat.S_ISLNK(metadata.st_mode) or bool(getattr(metadata, "st_file_attributes", 0) & _REPARSE_POINT_ATTRIBUTE):
            return "REPARSE_POINT"
    return "CLEAR"


def _within(candidate: Path, root: Path) -> bool:
    try:
        candidate.relative_to(root)
    except ValueError:
        return False
    return True


def verify_path(
    path: Path,
    policy: StoragePolicy,
    *,
    lstat: Callable[[Path], os.stat_result] = os.lstat,
) -> dict[str, Any]:
    """Classify one absolute path without creating or scanning anything."""

    candidate = Path(path)
    if not candidate.is_absolute():
        return _result(candidate, allowed=False, category="FORBIDDEN", reason="RELATIVE_PATH_FORBIDDEN")
    inspection = _inspect_path_components(candidate, lstat=lstat)
    if inspection == "UNREADABLE":
        return _result(candidate, allowed=False, category="FORBIDDEN", reason="PATH_INSPECTION_UNREADABLE")
    if inspection == "REPARSE_POINT":
        return _result(candidate, allowed=False, category="FORBIDDEN", reason="REPARSE_POINT_FORBIDDEN")
    try:
        resolved = candidate.resolve(strict=False)
    except OSError:
        return _result(candidate, allowed=False, category="FORBIDDEN", reason="PATH_RESOLUTION_UNREADABLE")
    if _within(resolved, policy.authoritative_project_root):
        return _result(resolved, allowed=True, category="PROJECT_ARTIFACT", reason="H_PROJECT_ROOT")
    if not _within(resolved, policy.checkpoint_staging_root):
        return _result(resolved, allowed=False, category="FORBIDDEN", reason="PATH_OUTSIDE_ALLOWED_ROOTS")
    relative = resolved.relative_to(policy.checkpoint_staging_root)
    if len(relative.parts) != 4:
        return _result(resolved, allowed=False, category="FORBIDDEN", reason="C_CHECKPOINT_LAYOUT_FORBIDDEN")
    basename = relative.name
    permitted = basename in policy.permitted_basenames or any(
        fnmatch.fnmatchcase(basename, pattern) for pattern in policy.permitted_temp_patterns
    )
    if not permitted:
        return _result(resolved, allowed=False, category="FORBIDDEN", reason="C_CHECKPOINT_BASENAME_FORBIDDEN")
    try:
        metadata = lstat(candidate)
    except FileNotFoundError:
        metadata = None
    except OSError:
        return _result(candidate, allowed=False, category="FORBIDDEN", reason="PATH_INSPECTION_UNREADABLE")
    if metadata is not None and not stat.S_ISREG(metadata.st_mode):
        return _result(candidate, allowed=False, category="FORBIDDEN", reason="C_CHECKPOINT_NOT_REGULAR_FILE")
    return _result(resolved, allowed=True, category="CHECKPOINT_PAYLOAD", reason="C_CHECKPOINT_ALLOWLIST")


def _staging_inventory(
    policy: StoragePolicy,
    *,
    lstat: Callable[[Path], os.stat_result],
    walk: Callable[..., Any],
) -> tuple[list[dict[str, Any]], int, str, list[str]]:
    """Inventory only C checkpoint staging; user Desktop is never considered."""

    root = policy.checkpoint_staging_root
    rows: list[dict[str, Any]] = []
    total_bytes = 0
    findings: list[str] = []
    root_inspection = _inspect_path_components(root, lstat=lstat)
    if root_inspection != "CLEAR":
        return rows, total_bytes, root_inspection, findings
    try:
        root_metadata = lstat(root)
    except FileNotFoundError:
        return rows, total_bytes, "CLEAR", findings
    except OSError:
        return rows, total_bytes, "UNREADABLE", findings
    if not stat.S_ISDIR(root_metadata.st_mode):
        return rows, total_bytes, "CLEAR", ["C_STAGING_INVALID_LAYOUT"]
    walk_errors: list[OSError] = []

    def onerror(error: OSError) -> None:
        walk_errors.append(error)

    try:
        iterator = walk(root, topdown=True, followlinks=False, onerror=onerror)
        for directory, directories, filenames in iterator:
            base = Path(directory)
            try:
                depth = len(base.relative_to(root).parts)
            except ValueError:
                return rows, total_bytes, "UNREADABLE", findings
            if depth > 3:
                findings.append("C_STAGING_INVALID_LAYOUT")
                directories[:] = []
                continue
            safe_directories: list[str] = []
            for name in sorted(directories, key=str.casefold):
                child = base / name
                child_inspection = _inspect_path_components(child, lstat=lstat)
                if child_inspection == "UNREADABLE":
                    return rows, total_bytes, "UNREADABLE", findings
                if child_inspection == "REPARSE_POINT":
                    return rows, total_bytes, "REPARSE_POINT", findings
                forbidden_directory = (
                    name.casefold() in {"log", "logs", "config", "configs", "embedding", "embeddings"}
                    or name in policy.permitted_basenames
                    or any(fnmatch.fnmatchcase(name, pattern) for pattern in policy.permitted_temp_patterns)
                    or depth + 1 > 3
                )
                if forbidden_directory:
                    findings.append("C_STAGING_INVALID_LAYOUT")
                    continue
                safe_directories.append(name)
            directories[:] = safe_directories
            filenames = sorted(filenames, key=str.casefold)
            if depth < 3:
                if filenames or (depth > 0 and not directories):
                    findings.append("C_STAGING_INVALID_LAYOUT")
                continue
            valid_payloads = 0
            if directories:
                findings.append("C_STAGING_INVALID_LAYOUT")
            for name in filenames:
                path = base / name
                check = verify_path(path, policy, lstat=lstat)
                try:
                    metadata = lstat(path)
                except OSError:
                    return rows, total_bytes, "UNREADABLE", findings
                if not stat.S_ISREG(metadata.st_mode):
                    check = {**check, "allowed": False, "category": "FORBIDDEN", "reason": "C_CHECKPOINT_NOT_REGULAR_FILE"}
                size = int(metadata.st_size) if stat.S_ISREG(metadata.st_mode) else 0
                total_bytes += size
                rows.append({"path": check["requested_path"], "size_bytes": size, "check": check})
                if check["allowed"]:
                    valid_payloads += 1
            if valid_payloads == 0:
                findings.append("C_STAGING_INVALID_LAYOUT")
    except OSError:
        return rows, total_bytes, "UNREADABLE", findings
    if walk_errors:
        return rows, total_bytes, "UNREADABLE", findings
    return sorted(rows, key=lambda row: row["path"].casefold()), total_bytes, "CLEAR", findings


def _disk_usage(path: Path) -> DiskUsage:
    usage = shutil.disk_usage(path)
    return DiskUsage(total=int(usage.total), used=int(usage.used), free=int(usage.free))


def audit_storage(
    *,
    root: Path,
    output_root: Path,
    policy: StoragePolicy,
    disk_usage: Callable[[Path], DiskUsage] = _disk_usage,
    lstat: Callable[[Path], os.stat_result] = os.lstat,
    walk: Callable[..., Any] = os.walk,
) -> dict[str, Any]:
    """Return stable, read-only evidence for the project and C staging roots."""

    root_check = verify_path(root, policy, lstat=lstat)
    output_check = verify_path(output_root, policy, lstat=lstat)
    inventory, staging_bytes, inventory_state, inventory_findings = _staging_inventory(policy, lstat=lstat, walk=walk)
    try:
        usage_target = policy.checkpoint_staging_root if policy.checkpoint_staging_root.exists() else Path(policy.checkpoint_staging_root.anchor)
        usage = disk_usage(usage_target)
        free_bytes = int(usage.free)
    except OSError:
        free_bytes = -1
    gib = 1024**3
    minimum_free_bytes = policy.minimum_free_gb * gib
    maximum_staging_bytes = policy.maximum_staging_gb * gib
    free_space_status = "PASS" if free_bytes >= minimum_free_bytes else "BLOCK"
    quota_status = "PASS" if staging_bytes <= maximum_staging_bytes else "BLOCK"
    forbidden_files = [row["path"] for row in inventory if not row["check"]["allowed"]]
    blockers: list[str] = []
    if not root_check["allowed"]:
        blockers.append(f"PROJECT_ROOT_{root_check['reason']}")
    if not output_check["allowed"]:
        blockers.append(f"OUTPUT_ROOT_{output_check['reason']}")
    if free_space_status == "BLOCK":
        blockers.append("C_MINIMUM_FREE_SPACE_NOT_MET")
    if quota_status == "BLOCK":
        blockers.append("C_STAGING_QUOTA_EXCEEDED")
    if inventory_state == "REPARSE_POINT":
        blockers.append("C_STAGING_REPARSE_POINT_FORBIDDEN")
    if inventory_state == "UNREADABLE":
        blockers.append("C_STAGING_INVENTORY_UNREADABLE")
    blockers.extend(inventory_findings)
    if forbidden_files:
        blockers.append("C_STAGING_CONTAINS_FORBIDDEN_FILE")
    evidence = {
        "project_root": root_check,
        "output_root": output_check,
        "checkpoint_staging": {
            "root": str(policy.checkpoint_staging_root),
            "minimum_free_gb": policy.minimum_free_gb,
            "maximum_staging_gb": policy.maximum_staging_gb,
            "free_bytes": free_bytes,
            "free_space_status": free_space_status,
            "staging_bytes": staging_bytes,
            "quota_status": quota_status,
            "inspection_state": inventory_state,
            "reparse_point_status": (
                "PASS"
                if inventory_state == "CLEAR"
                else "BLOCK"
                if inventory_state == "REPARSE_POINT"
                else "UNKNOWN"
            ),
            "file_count": len(inventory),
            "files": inventory,
        },
    }
    policy_digest = hashlib.sha256(json.dumps(asdict(policy), default=str, sort_keys=True).encode("utf-8")).hexdigest()
    return {
        "schema_version": STORAGE_LOCALITY_SCHEMA_VERSION,
        "policy_id": policy.policy_id,
        "policy_sha256": policy_digest,
        "decision": "BLOCK" if blockers else "PASS",
        "blockers": sorted(blockers),
        "evidence": evidence,
    }
