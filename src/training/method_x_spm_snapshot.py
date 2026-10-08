"""Immutable, cheaply re-verifiable SPM gate snapshots for Method X."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from src.utils.io_helpers import atomic_write_json
from src.utils.reproducibility import canonical_digest


DEFAULT_SNAPSHOT_REL = Path("results/method_x_registry/spm_gate_snapshot.json")
DEFAULT_MANIFEST_REL = Path("results/logs/spm_mri_preprocess_v2_128_resume.json")
DEFAULT_CATALOG_REL = Path("data/processed/master_catalog.csv")
DEFAULT_PREPROCESSED_REL = Path("data/preprocessed_mri_128")
DEFAULT_GUARD_REL = Path("scripts/spm_completion_guard.py")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _relative(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _source_hashes(root: Path, paths: Iterable[Path]) -> dict[str, str]:
    return {_relative(path, root): sha256_file(path) for path in paths}


def _qc_bundle(root: Path, preprocessed_dir: Path) -> dict[str, Any]:
    qc_files = sorted((preprocessed_dir / "qc").glob("*.json"))
    rows = [(_relative(path, root), sha256_file(path)) for path in qc_files if path.is_file()]
    return {"count": len(rows), "digest": canonical_digest(rows)}


def _output_inventory(root: Path, preprocessed_dir: Path) -> dict[str, Any]:
    files = sorted(path for path in preprocessed_dir.glob("*.nii.gz") if path.is_file())
    rows = [(_relative(path, root), int(path.stat().st_size)) for path in files]
    return {"count": len(rows), "digest": canonical_digest(rows)}


def build_spm_gate_snapshot(
    *,
    root: Path,
    guard_status: dict[str, Any],
    manifest_path: Path | None = None,
    catalog_path: Path | None = None,
    preprocessed_dir: Path | None = None,
    guard_script: Path | None = None,
) -> dict[str, Any]:
    root = root.resolve()
    manifest_path = (manifest_path or root / DEFAULT_MANIFEST_REL).resolve()
    catalog_path = (catalog_path or root / DEFAULT_CATALOG_REL).resolve()
    preprocessed_dir = (preprocessed_dir or root / DEFAULT_PREPROCESSED_REL).resolve()
    guard_script = (guard_script or root / DEFAULT_GUARD_REL).resolve()
    required_sources = [manifest_path, catalog_path, guard_script]
    missing = [str(path) for path in required_sources if not path.is_file()]
    if missing:
        raise RuntimeError(f"SPM_SNAPSHOT_SOURCE_MISSING:{missing}")

    privacy = guard_status.get("privacy", {}) if isinstance(guard_status, dict) else {}
    if any(
        bool(privacy.get(field))
        for field in ("raw_identifiers_exported", "raw_paths_exported", "subject_level_outputs_exported")
    ):
        raise RuntimeError("SPM_SNAPSHOT_PRIVACY_CONTRACT_FAILED")

    body = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "VERIFIED_READY" if bool(guard_status.get("ready")) else "BLOCKED_NOT_READY",
        "ready": bool(guard_status.get("ready")),
        "guard_status": guard_status,
        "source_hashes": _source_hashes(root, required_sources),
        "qc_bundle": _qc_bundle(root, preprocessed_dir),
        "output_inventory": _output_inventory(root, preprocessed_dir),
        "privacy": {
            "raw_identifiers_exported": False,
            "raw_paths_exported": False,
            "subject_level_outputs_exported": False,
            "stored_subject_membership": False,
        },
    }
    return {**body, "snapshot_sha256": canonical_digest(body)}


def write_spm_gate_snapshot(path: Path, snapshot: dict[str, Any]) -> Path:
    atomic_write_json(path, snapshot)
    return path


def verify_spm_gate_snapshot(path: Path, *, root: Path, verify_sources: bool = True) -> dict[str, Any]:
    if not path.is_file():
        return {"verified": False, "ready": False, "status": "SNAPSHOT_MISSING", "path": str(path)}
    try:
        snapshot = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {
            "verified": False,
            "ready": False,
            "status": "SNAPSHOT_INVALID_JSON",
            "path": str(path),
            "error": str(exc),
        }
    stored_digest = snapshot.get("snapshot_sha256")
    body = {key: value for key, value in snapshot.items() if key != "snapshot_sha256"}
    checks: dict[str, bool] = {"snapshot_digest": stored_digest == canonical_digest(body)}
    if verify_sources:
        current_hashes: dict[str, str] = {}
        for relative in snapshot.get("source_hashes", {}):
            source = root / relative
            if source.is_file():
                current_hashes[relative] = sha256_file(source)
        checks["source_hashes"] = current_hashes == snapshot.get("source_hashes")
        preprocessed_dir = root / DEFAULT_PREPROCESSED_REL
        checks["qc_bundle"] = _qc_bundle(root, preprocessed_dir) == snapshot.get("qc_bundle")
        checks["output_inventory"] = _output_inventory(root, preprocessed_dir) == snapshot.get("output_inventory")
    verified = all(checks.values())
    return {
        "verified": verified,
        "ready": verified and bool(snapshot.get("ready")),
        "status": "VERIFIED_READY" if verified and bool(snapshot.get("ready")) else "SNAPSHOT_VERIFICATION_FAILED",
        "path": str(path),
        "snapshot_sha256": stored_digest,
        "guard_status": snapshot.get("guard_status", {}),
        "checks": checks,
    }


def bind_spm_snapshot_to_battery_manifest(
    manifest_path: Path,
    *,
    verification: dict[str, Any],
    root: Path,
) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not verification.get("verified") or not verification.get("ready"):
        raise RuntimeError("SPM_SNAPSHOT_NOT_VERIFIED_READY")
    snapshot_path = Path(str(verification["path"]))
    manifest["spm_gate_snapshot"] = {
        "status": verification["status"],
        "ready": True,
        "verified": True,
        "path": _relative(snapshot_path, root),
        "snapshot_sha256": verification["snapshot_sha256"],
        "guard_status": verification.get("guard_status", {}).get("status"),
    }
    atomic_write_json(manifest_path, manifest)
    return manifest
