"""Schema-v2 BrainIAC raw-T1 preprocessing runner (Plans 002 build + 003 run).

Modes:
  preflight   read-only evidence (no lease, no sample writes)
  calibrate   32 lowest-hash samples -> immutable MAD threshold manifest
              (idempotent: a valid existing manifest is reused, never rebuilt)
  run         full 1669 primary run, resumable: committed samples skip,
              Ctrl+C/Ctrl+Break stop at the next safe boundary (exit 75),
              re-running the SAME command continues from the last commit
  sensitivity 64 lowest-hash samples post-primary: mask-stability (blocker)
              + direct-128-to-96 drift (report-only, SENSITIVITY_ONLY role)
  full        the single operator command: preflight gate -> calibrate ->
              run -> sensitivity -> aggregate streaming revalidation
  smoke       bounded source-discipline smoke (sequential-only per policy)
  status      read-only completion counts

Exit codes: 0 completed/already-completed; 3 fail-closed blocker;
75 paused-resumable; 78 contract mismatch.  Raw paths/IDs never printed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import statistics
import subprocess
import sys
import threading
import time
from typing import Any, Callable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.research import brainiac_preprocessing_v2 as v2
from src.research.brainiac_preprocessing_v2_schema import (
    FIXED_QC_LIMITS,
    build_aggregate,
    canonical_sha,
    sha256_file,
    validate_receipt,
    validate_threshold_manifest,
)
from src.research.brainiac_dependency_closure import (
    build_execution_dependency_manifest,
    verify_manifest_tracking,
)
from src.research.brainiac_method_x_isolation import (
    verify_frozen_method_x_source_contract,
)

REGISTRY_ROOT = ROOT / "results" / "method_x_registry" / "brainiac_r1"
RUNTIME_LOCK = REGISTRY_ROOT / "brainiac_official_runtime_lock.json"
STRICT_RECEIPT = REGISTRY_ROOT / "brainiac_official_strict_load_receipt.json"
FORWARD_RECEIPT = REGISTRY_ROOT / "brainiac_official_forward_receipt.json"
TEMPLATE_PATH = (
    ROOT / "models" / "pretrained" / "brainiac" / "templates" / "spm25_avg152T1_2mm.nii"
)
TEMPLATE_LOCK = TEMPLATE_PATH.with_name("spm25_avg152T1_2mm.lock.json")
DERIVED_TEMPLATE = (
    ROOT / "data" / "derived" / "method_x_brainiac" / "templates" / "spm25_avg152T1_1mm.nii.gz"
)
DERIVED_TEMPLATE_LOCK = DERIVED_TEMPLATE.with_suffix(".lock.json")
PREFLIGHT_REPORT = REGISTRY_ROOT / "brainiac_preprocessing_v2_preflight.json"
THRESHOLD_MANIFEST_PATH = (
    ROOT / "data" / "derived" / "method_x_brainiac" / "r1_primary_v2" / "aggregate"
    / "qc_threshold_manifest.json"
)
CALIBRATION_ROOT = (
    ROOT / "data" / "derived" / "method_x_brainiac" / "r1_primary_v2_calibration"
)
PRIMARY_AGGREGATE_PATH = (
    ROOT / "data" / "derived" / "method_x_brainiac" / "r1_primary_v2" / "aggregate"
    / "brainiac_preprocessing_v2_manifest.json"
)
SENSITIVITY_REPORT_PATH = REGISTRY_ROOT / "brainiac_preprocessing_v2_sensitivity.json"
RUN_STATE_PATH = REGISTRY_ROOT / "brainiac_preprocessing_v2_run_state.json"
SMOKE_REPORT = REGISTRY_ROOT / "brainiac_preprocessing_v2_smoke.json"
EXECUTION_PROFILE_PATH = REGISTRY_ROOT / "brainiac_preprocessing_execution_profile_v2.json"
LEGACY_EXECUTION_PROFILE_PATH = (
    REGISTRY_ROOT / "brainiac_preprocessing_execution_profile_v1.json"
)
DEPENDENCY_MANIFEST_PATH = (
    REGISTRY_ROOT / "brainiac_preprocessing_dependency_closure_v1.json"
)
READY_REPORT_PATH = (
    REGISTRY_ROOT / "READY_TO_START_OR_RESUME_BRAINIAC_PREPROCESSING.json"
)
ACCEPTANCE_CANDIDATE_PATH = (
    REGISTRY_ROOT / "BRAINIAC_PREPROCESSING_ACCEPTANCE_CANDIDATE.json"
)
ACCEPTANCE_PREFLIGHT_SNAPSHOT_PATH = (
    REGISTRY_ROOT / "brainiac_preprocessing_acceptance_preflight_snapshot.json"
)
ACCEPTANCE_TEST_MANIFEST_PATH = (
    REGISTRY_ROOT / "brainiac_preprocessing_acceptance_test_manifest.json"
)
FULL_TEST_MANIFEST_PATH = (
    REGISTRY_ROOT / "brainiac_preprocessing_test_manifest_v1.json"
)
TRACEABILITY_MANIFEST_PATH = (
    REGISTRY_ROOT / "brainiac_preprocessing_traceability_manifest_v1.json"
)
SOURCE_ONLY_RESUME_MIGRATION_PATH = (
    REGISTRY_ROOT / "BRAINIAC_PLAN017_SOURCE_ONLY_RESUME_MIGRATION_V1.json"
)
POSTCALIBRATION_SOURCE_MIGRATION_PATH = (
    REGISTRY_ROOT
    / "BRAINIAC_PLAN017_POSTCALIBRATION_SOURCE_MIGRATION_V2.json"
)
LEGACY_SMOKE_RECONCILIATION_PATH = (
    REGISTRY_ROOT / "BRAINIAC_PLAN017_LEGACY_SMOKE_RECONCILIATION_V1.json"
)
PRIMARY_BINDING_SOURCE_COMPATIBILITY_PATH = (
    REGISTRY_ROOT
    / "BRAINIAC_PLAN017_PRIMARY_BINDING_SOURCE_COMPATIBILITY_V3.json"
)
CALIBRATION_RECONCILIATION_SOURCE_COMPATIBILITY_PATH = (
    REGISTRY_ROOT
    / "BRAINIAC_PLAN017_CALIBRATION_RECONCILIATION_SOURCE_COMPATIBILITY_V4.json"
)
FALSE_CALIBRATION_RECONCILIATION_PATH = (
    REGISTRY_ROOT
    / "superseded"
    / "20260831T140348Z-1788185028106323300-preprocessing-artifact-reconciliation.json"
)
PLAN017_LOCKED_EVIDENCE_DEPENDENCY_CLOSURE_SHA256 = (
    "93fbd6011cce4fd0f906c7e90f41fff44cffd30345ff57cfa877550d4d867ae0"
)
PLAN017_PRIMARY_BINDING_V3_DEPENDENCY_CLOSURE_SHA256 = (
    "d82d4f0b4061d5f92a9435eb85bb6b0970919baad3808773ef096e7d3017e946"
)
PLAN017_PRIMARY_BINDING_V3_RUNNER_SHA256 = (
    "e3fbb930e5a1ced765bcd17b0ab97dec4d07c57f534c233bd5c96849570add60"
)
P08_STATUS_PATH = (
    ROOT
    / "results"
    / "method_x_registry"
    / "BRAINIAC_P08_STATUS_20260828_V3.json"
)
NOVELTY_MATRIX_V3_PATH = (
    ROOT / "research" / "brainiac_caft_20260808" / "novelty_matrix_v3.json"
)
ACCEPTANCE_REQUIRED_CHECKS = frozenset(
    {
        "transaction_fault_injection",
        "sigint_sequential_pause_resume",
        "sigint_parallel_pause_resume",
        "sigterm_sequential_pause_resume",
        "sigterm_parallel_pause_resume",
        "hard_kill_pause_resume",
        "corrupt_receipt_rejected",
        "corrupt_output_transform_rejected",
        "journal_truncation_rejected",
        "receipt_before_journal_recovered",
        "reparse_lock_pid_reuse_rejected",
        "profile_live_byte_revalidation",
    }
)
PRE_REPAIR_LIVE_FREEZE_PATH = (
    REGISTRY_ROOT / "BRAINIAC_PRE_REPAIR_LIVE_FREEZE_SCHEMA2_20260828.json"
)
CHECKPOINT_STAGING_ROOT = Path(
    r"C:\Users\USER\AppData\Local\AMUSE-X\checkpoint_staging"
)
FREEZE_OPERATIONAL_UNTRACKED_PREFIXES = (
    ".runtime/",
    "tmp/",
    ".claude/",
    ".agents/",
    ".opencode/",
)
CONFIG_PATH = ROOT / "configs" / "research" / "brainiac_preprocessing_r1_primary_v2.yaml"
VENDOR_RUNTIME_CODE = {
    "official_dataset": v2.VENDOR_ROOT / "src" / "dataset.py",
    "official_model": v2.VENDOR_ROOT / "src" / "model.py",
    "preprocessing_package": v2.VENDOR_PREPROCESSING / "__init__.py",
    "hd_bet_config": v2.VENDOR_PREPROCESSING / "HD_BET" / "config.py",
    "hd_bet_data_loading": v2.VENDOR_PREPROCESSING / "HD_BET" / "data_loading.py",
    "hd_bet_entrypoint": v2.VENDOR_PREPROCESSING / "HD_BET" / "hd_bet.py",
    "hd_bet_network": v2.VENDOR_PREPROCESSING / "HD_BET" / "network_architecture.py",
    "hd_bet_paths": v2.VENDOR_PREPROCESSING / "HD_BET" / "paths.py",
    "hd_bet_prediction": v2.VENDOR_PREPROCESSING / "HD_BET" / "predict_case.py",
    "hd_bet_run": v2.VENDOR_PREPROCESSING / "HD_BET" / "run.py",
    "hd_bet_utils": v2.VENDOR_PREPROCESSING / "HD_BET" / "utils.py",
}
LINEAGE_PATH = ROOT / "results" / "method_x_registry" / "data_lineage_manifest.json"
BATTERY_MANIFEST_PATH = (
    ROOT
    / "results"
    / "final_results"
    / "method_x_model_battery_v3_depth_and_diversity_20260729"
    / "battery_manifest.json"
)
EXPECTED_METHOD_X_SOURCE_DIGEST = (
    "9b0f90a5a14f8764397c6a15ff8016462f0872e2f827b2ec47fe54ab773871b3"
)
EXPECTED_BATTERY_MANIFEST_SHA256 = (
    "40a4a009b3a2453fb953e2b2e9b9ec789bdf896741dc2278a898a3b38fdb971d"
)

MIN_FREE_DISK_GB = 40.0
MIN_CUDA_TOTAL_VRAM_GB = 7.5
PROGRESS_EVERY_N = 25


class _StopFlag:
    def __init__(self) -> None:
        self.requested = False
        self.reason = ""


def _install_signal_handlers(stop: _StopFlag) -> None:
    def _handler(signum, _frame):
        if not stop.requested:
            stop.reason = signal.Signals(signum).name
        stop.requested = True  # finish current sample, then exit 75

    for sig_name in ("SIGINT", "SIGTERM", "SIGBREAK"):
        number = getattr(signal, sig_name, None)
        if number is not None:
            try:
                signal.signal(number, _handler)
            except (ValueError, OSError):
                pass


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime())


def _exit_code_for(status: str) -> int:
    if status in {"PASS", "COMPLETED", "ALREADY_COMPLETED"}:
        return 0
    if status == "PAUSED_RESUMABLE":
        return 75
    if "CONTRACT_MISMATCH" in status:
        return 78
    return 3


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    encoded = (json.dumps(dict(payload), indent=2, sort_keys=True) + "\n").encode("utf-8")
    descriptor = os.open(
        temporary, os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_BINARY", 0)
    )
    try:
        os.write(descriptor, encoded)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.replace(temporary, path)


def _print_flush(message: str) -> None:
    print(message, flush=True)


def _sanitize_exception_message(error: BaseException) -> str:
    """Return a bounded reason with raw paths and sample hashes redacted."""

    message = str(error).replace(str(v2.RAW_DATA_ROOT), "<raw-redacted>")
    message = re.sub(r"(?i)H:\\A4 Learn(?:\\[^\s:;,]+)*", "<raw-redacted>", message)
    message = re.sub(r"\bs_[0-9a-f]{16,64}\b", "<sample-redacted>", message)
    return message[:500]


def build_pre_repair_freeze_artifact(
    *, snapshot: Mapping[str, Any], created_at_utc: str
) -> dict[str, Any]:
    """Seal live evidence while keeping the operational timestamp out of its digest."""

    body: dict[str, Any] = {
        "schema_version": 2,
        "artifact": "BRAINIAC_PRE_REPAIR_LIVE_FREEZE",
        "status": "FROZEN_BEFORE_ARTIFACT_RECONCILIATION",
        "created_at_utc": str(created_at_utc),
        "snapshot": dict(snapshot),
        "snapshot_sha256": canonical_sha(snapshot),
        "human_approval_required": False,
        "manual_override_allowed": False,
    }
    body["integrity_sha256"] = canonical_sha(body)
    return body


def write_pre_repair_freeze(
    path: Path,
    *,
    snapshot: Mapping[str, Any],
    created_at_utc: str | None = None,
) -> dict[str, Any]:
    """Create one immutable freeze, or verify an idempotent identical rerun."""

    target = Path(path)
    candidate = build_pre_repair_freeze_artifact(
        snapshot=snapshot,
        created_at_utc=created_at_utc or _utc_now(),
    )
    if target.exists():
        try:
            existing = json.loads(target.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise RuntimeError("BRAINIAC_PRE_REPAIR_FREEZE_INVALID") from exc
        if not isinstance(existing, dict):
            raise RuntimeError("BRAINIAC_PRE_REPAIR_FREEZE_INVALID")
        recorded_integrity = existing.get("integrity_sha256")
        unsealed = dict(existing)
        unsealed.pop("integrity_sha256", None)
        valid = (
            existing.get("schema_version") == 2
            and existing.get("artifact") == "BRAINIAC_PRE_REPAIR_LIVE_FREEZE"
            and recorded_integrity == canonical_sha(unsealed)
            and existing.get("snapshot_sha256")
            == canonical_sha(existing.get("snapshot") or {})
        )
        if not valid:
            raise RuntimeError("BRAINIAC_PRE_REPAIR_FREEZE_INVALID")
        if existing.get("snapshot_sha256") != candidate["snapshot_sha256"]:
            raise RuntimeError("BRAINIAC_PRE_REPAIR_FREEZE_DRIFT")
        return existing
    _atomic_write_json(target, candidate)
    return candidate


def archive_non_idempotent_pre_repair_freeze(
    path: Path,
    *,
    archive_root: Path,
    process_provider: Any | None = None,
    gpu_provider: Any | None = None,
    created_at_utc: str | None = None,
) -> dict[str, Any]:
    """Preserve a valid but operationally unstable freeze before rebuilding it."""

    from src.research.brainiac_readiness import inspect_writer_state, _writer_blockers

    source = Path(path)
    destination_root = Path(archive_root)
    if not source.is_file() or v2._is_reparse(source):
        raise RuntimeError("BRAINIAC_NONIDEMPOTENT_FREEZE_SOURCE_UNSAFE")
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("BRAINIAC_NONIDEMPOTENT_FREEZE_INVALID") from exc
    if not isinstance(payload, dict):
        raise RuntimeError("BRAINIAC_NONIDEMPOTENT_FREEZE_INVALID")
    unsealed = dict(payload)
    recorded_integrity = unsealed.pop("integrity_sha256", None)
    if (
        payload.get("schema_version") != 2
        or payload.get("artifact") != "BRAINIAC_PRE_REPAIR_LIVE_FREEZE"
        or recorded_integrity != canonical_sha(unsealed)
        or payload.get("snapshot_sha256") != canonical_sha(payload.get("snapshot") or {})
    ):
        raise RuntimeError("BRAINIAC_NONIDEMPOTENT_FREEZE_INVALID")
    battery = json.loads(BATTERY_MANIFEST_PATH.read_text(encoding="utf-8"))
    writer_state = inspect_writer_state(
        battery,
        process_provider=process_provider,
        gpu_provider=gpu_provider,
    )
    blockers = _writer_blockers(writer_state)
    if blockers:
        raise RuntimeError(
            "BRAINIAC_NONIDEMPOTENT_FREEZE_ARCHIVE_BLOCKED:" + ",".join(blockers)
        )
    raw = source.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    stamp = re.sub(r"[^0-9TZ]", "", created_at_utc or _utc_now())
    destination_root.mkdir(parents=True, exist_ok=True)
    target = destination_root / (
        f"BRAINIAC_PRE_REPAIR_LIVE_FREEZE_SCHEMA2_NONIDEMPOTENT-{stamp}-{digest[:12]}.json"
    )
    receipt_path = target.with_suffix(".receipt.json")
    if target.exists() or receipt_path.exists():
        raise RuntimeError("BRAINIAC_NONIDEMPOTENT_FREEZE_ARCHIVE_TARGET_EXISTS")
    os.replace(source, target)
    if sha256_file(target) != digest:
        raise RuntimeError("BRAINIAC_NONIDEMPOTENT_FREEZE_ARCHIVE_HASH_DRIFT")
    receipt: dict[str, Any] = {
        "schema_version": 1,
        "artifact": "BRAINIAC_NONIDEMPOTENT_FREEZE_ARCHIVE_RECEIPT",
        "status": "ARCHIVED_APPEND_ONLY",
        "created_at_utc": created_at_utc or _utc_now(),
        "reason": "DIRTY_INVENTORY_INCLUDED_UNTRACKED_TOOL_RUNTIME_CHURN",
        "source_name": source.name,
        "archive_name": target.name,
        "bytes": len(raw),
        "sha256": digest,
        "snapshot_sha256": payload["snapshot_sha256"],
        "writer_state": writer_state,
        "human_approval_required": False,
        "manual_override_allowed": False,
    }
    receipt["integrity_sha256"] = canonical_sha(receipt)
    _atomic_write_json(receipt_path, receipt)
    return receipt


def build_file_inventory(
    paths: list[Path] | tuple[Path, ...], *, project_root: Path = ROOT
) -> dict[str, Any]:
    """Hash a sorted set of regular, non-reparse files beneath one authority."""

    authority = Path(project_root).resolve()
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for supplied in sorted((Path(path) for path in paths), key=lambda item: str(item)):
        path = supplied.resolve(strict=True)
        try:
            relative = path.relative_to(authority).as_posix()
        except ValueError as exc:
            raise RuntimeError("BRAINIAC_FREEZE_PATH_OUTSIDE_AUTHORITY") from exc
        if relative in seen:
            raise RuntimeError("BRAINIAC_FREEZE_DUPLICATE_PATH")
        if not path.is_file() or v2._is_reparse(path):
            raise RuntimeError("BRAINIAC_FREEZE_FILE_UNSAFE")
        seen.add(relative)
        before = path.stat()
        digest = sha256_file(path)
        after = path.stat()
        if (
            int(before.st_size) != int(after.st_size)
            or int(before.st_mtime_ns) != int(after.st_mtime_ns)
        ):
            raise RuntimeError("BRAINIAC_FREEZE_FILE_DRIFT_DURING_HASH")
        rows.append(
            {
                "path": relative,
                "bytes": int(after.st_size),
                "sha256": digest,
            }
        )
    rows.sort(key=lambda row: row["path"])
    encoded = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {
        "files": rows,
        "file_count": len(rows),
        "total_bytes": sum(int(row["bytes"]) for row in rows),
        "inventory_sha256": hashlib.sha256(encoded).hexdigest(),
    }


def _validate_freeze_inventory(role: str, inventory: Mapping[str, Any]) -> None:
    files = inventory.get("files")
    if not isinstance(files, list):
        raise RuntimeError(f"BRAINIAC_FREEZE_INVENTORY_INVALID:{role}")
    for row in files:
        if (
            not isinstance(row, Mapping)
            or not isinstance(row.get("path"), str)
            or int(row.get("bytes", -1)) < 0
            or re.fullmatch(r"[0-9a-f]{64}", str(row.get("sha256") or ""))
            is None
        ):
            raise RuntimeError(f"BRAINIAC_FREEZE_INVENTORY_INVALID:{role}")
    encoded = json.dumps(files, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    try:
        file_count = int(inventory.get("file_count", -1))
        total_bytes = int(inventory.get("total_bytes", -1))
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"BRAINIAC_FREEZE_INVENTORY_INVALID:{role}") from exc
    if (
        file_count != len(files)
        or total_bytes != sum(int(row.get("bytes", -1)) for row in files)
        or inventory.get("inventory_sha256")
        != hashlib.sha256(encoded).hexdigest()
    ):
        raise RuntimeError(f"BRAINIAC_FREEZE_INVENTORY_INVALID:{role}")


def build_pre_repair_snapshot(
    *,
    branch: str,
    head: str,
    dirty_paths: list[str],
    writer_state: Mapping[str, Any],
    method_x_source_digest: str,
    battery_manifest_binding: Mapping[str, Any],
    battery_inventory: Mapping[str, Any],
    c_checkpoint_inventory: Mapping[str, Any],
    dependency_manifest: Mapping[str, Any],
    lock_inventory: list[Mapping[str, Any]],
    legacy_inventory: list[Mapping[str, Any]],
    primary_valid_n: int,
    predecessor_freeze_inventory: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate and assemble the immutable preservation snapshot."""

    from src.research.brainiac_readiness import _writer_blockers

    if re.fullmatch(r"[0-9a-f]{40}", str(head).casefold()) is None:
        raise RuntimeError("BRAINIAC_FREEZE_GIT_HEAD_INVALID")
    if method_x_source_digest != EXPECTED_METHOD_X_SOURCE_DIGEST:
        raise RuntimeError("BRAINIAC_FREEZE_SOURCE_DIGEST_DRIFT")
    if (
        battery_manifest_binding.get("sha256")
        != EXPECTED_BATTERY_MANIFEST_SHA256
    ):
        raise RuntimeError("BRAINIAC_FREEZE_BATTERY_MANIFEST_DRIFT")
    blockers = _writer_blockers(writer_state)
    if blockers:
        raise RuntimeError("BRAINIAC_FREEZE_WRITER_BLOCKED:" + ",".join(blockers))
    _validate_freeze_inventory("battery", battery_inventory)
    _validate_freeze_inventory("c_checkpoint", c_checkpoint_inventory)
    if predecessor_freeze_inventory is not None:
        _validate_freeze_inventory("predecessor_freezes", predecessor_freeze_inventory)
    closure_sha = str(dependency_manifest.get("closure_sha256") or "")
    if re.fullmatch(r"[0-9a-f]{64}", closure_sha.casefold()) is None:
        raise RuntimeError("BRAINIAC_FREEZE_DEPENDENCY_CLOSURE_INVALID")
    if int(primary_valid_n) != 0:
        raise RuntimeError("BRAINIAC_FREEZE_PRIMARY_ALREADY_STARTED")
    normalized_dirty = sorted(str(item) for item in dirty_paths)
    if any("\x00" in item for item in normalized_dirty):
        raise RuntimeError("BRAINIAC_FREEZE_DIRTY_INVENTORY_INVALID")
    dirty_encoded = json.dumps(
        normalized_dirty, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return {
        "freeze_phase": "LIVE_POST_AGENT_AUDIT_BEFORE_CORRECTIVE_RECONCILIATION",
        "branch": str(branch),
        "head": str(head).casefold(),
        "dirty_entry_tokens": [
            {
                "status": item[:2],
                "path_sha256": hashlib.sha256(item[3:].encode("utf-8")).hexdigest(),
            }
            for item in normalized_dirty
        ],
        "dirty_entry_count": len(normalized_dirty),
        "dirty_inventory_sha256": hashlib.sha256(dirty_encoded).hexdigest(),
        "dirty_inventory_scope": {
            "tracked_entries": "ALL",
            "untracked_entries": "EXCLUDE_TOOL_RUNTIME_PREFIXES_ONLY",
            "excluded_untracked_prefixes": [
                *FREEZE_OPERATIONAL_UNTRACKED_PREFIXES,
                ".tmp_*",
            ],
        },
        "writer_state": dict(writer_state),
        "method_x_source_digest": method_x_source_digest,
        "battery_manifest": dict(battery_manifest_binding),
        "battery_finalized_inventory": dict(battery_inventory),
        "c_checkpoint_inventory": dict(c_checkpoint_inventory),
        "dependency_manifest": dict(dependency_manifest),
        "locks": [dict(row) for row in lock_inventory],
        "legacy_artifacts": [dict(row) for row in legacy_inventory],
        "predecessor_freezes": (
            dict(predecessor_freeze_inventory)
            if predecessor_freeze_inventory is not None
            else None
        ),
        "primary_valid_n": 0,
        "outer_loader_invocation_count": 0,
        "finalized_artifact_rewrite_count": 0,
    }


def _git_live_state() -> tuple[str, str, list[str]]:
    def run(*arguments: str, binary: bool = False) -> bytes | str:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=ROOT,
            capture_output=True,
            text=not binary,
            check=False,
            timeout=60,
        )
        if completed.returncode != 0:
            raise RuntimeError("BRAINIAC_FREEZE_GIT_INSPECTION_FAILED")
        return completed.stdout

    branch = str(run("rev-parse", "--abbrev-ref", "HEAD")).strip()
    head = str(run("rev-parse", "HEAD")).strip()
    raw = run(
        "status", "--porcelain=v1", "-z", "--untracked-files=all", binary=True
    )
    if not isinstance(raw, bytes):
        raise RuntimeError("BRAINIAC_FREEZE_GIT_INSPECTION_INVALID")
    try:
        dirty = [item.decode("utf-8") for item in raw.split(b"\0") if item]
    except UnicodeError as exc:
        raise RuntimeError("BRAINIAC_FREEZE_GIT_INSPECTION_INVALID") from exc
    return branch, head, _filter_freeze_dirty_entries(dirty)


def _acceptance_scoped_git_state() -> tuple[str, str, list[str]]:
    """Inspect only execution/test sources; runtime data never invalidates resume."""

    dependency = _dependency_manifest()
    relative: set[str] = set(map(str, dict(dependency["python_closure"]).keys()))
    for role in ("dynamic_code_inputs", "static_inputs"):
        for binding in dict(dependency.get(role) or {}).values():
            if isinstance(binding, Mapping) and binding.get("path"):
                relative.add(str(binding["path"]))
    if ACCEPTANCE_TEST_MANIFEST_PATH.is_file():
        test_manifest = json.loads(
            ACCEPTANCE_TEST_MANIFEST_PATH.read_text(encoding="utf-8")
        )
        evidence = test_manifest.get("evidence_bindings")
        if isinstance(evidence, Mapping):
            for role in ("test_files", "source_files"):
                for binding in evidence.get(role, []):
                    if isinstance(binding, Mapping) and binding.get("path"):
                        relative.add(str(binding["path"]))
    for explicit in (
        ".gitattributes",
        "scripts/build_brainiac_p08_status.py",
        "scripts/research/build_brainiac_novelty_matrix_v3.py",
        "scripts/research/build_brainiac_preprocessing_acceptance_test_manifest.py",
        "scripts/research/build_brainiac_preprocessing_full_test_manifest.py",
        "scripts/research/build_brainiac_preprocessing_release.py",
    ):
        if (ROOT / explicit).is_file():
            relative.add(explicit)
    completed = subprocess.run(
        [
            "git",
            "status",
            "--porcelain=v1",
            "-z",
            "--",
            *sorted(relative),
        ],
        cwd=ROOT,
        capture_output=True,
        check=False,
        timeout=60,
    )
    if completed.returncode != 0:
        raise RuntimeError("BRAINIAC_ACCEPTANCE_SCOPED_GIT_INSPECTION_FAILED")
    try:
        dirty = [
            item.decode("utf-8")
            for item in completed.stdout.split(b"\0")
            if item
        ]
    except UnicodeError as exc:
        raise RuntimeError("BRAINIAC_ACCEPTANCE_SCOPED_GIT_INSPECTION_FAILED") from exc
    branch, head, _ = _git_live_state()
    return branch, head, sorted(dirty)


def _filter_freeze_dirty_entries(entries: list[str]) -> list[str]:
    """Drop only untracked tool-runtime churn; every tracked change remains bound."""

    retained: list[str] = []
    for raw_entry in entries:
        entry = str(raw_entry).replace("\\", "/")
        if len(entry) < 4:
            raise RuntimeError("BRAINIAC_FREEZE_GIT_INSPECTION_INVALID")
        status = entry[:2]
        path = entry[3:]
        if status != "??":
            retained.append(entry)
            continue
        path_folded = path.casefold()
        operational = path_folded.startswith(
            tuple(prefix.casefold() for prefix in FREEZE_OPERATIONAL_UNTRACKED_PREFIXES)
        ) or path_folded.startswith(".tmp_")
        if not operational:
            retained.append(entry)
    return sorted(retained)


def _battery_preservation_paths() -> list[Path]:
    battery_root = BATTERY_MANIFEST_PATH.parent.resolve()
    manifest = json.loads(BATTERY_MANIFEST_PATH.read_text(encoding="utf-8"))
    rows = manifest.get("models")
    if not isinstance(rows, list):
        raise RuntimeError("BRAINIAC_FREEZE_BATTERY_MODELS_INVALID")
    files: set[Path] = set()
    for child in battery_root.iterdir():
        if child.is_file() and child.name != ".method_x_reconciliation.lock":
            files.add(child)
    completed_names: set[str] = set()
    for row in rows:
        if not isinstance(row, Mapping) or row.get("status") != "COMPLETED":
            continue
        name = str(row.get("backbone") or "")
        output_dir = str(row.get("output_dir") or "")
        if not name or name in completed_names or not output_dir:
            raise RuntimeError("BRAINIAC_FREEZE_BATTERY_MODELS_INVALID")
        completed_names.add(name)
        model_root = (battery_root / output_dir).resolve(strict=True)
        try:
            model_root.relative_to(battery_root)
        except ValueError as exc:
            raise RuntimeError("BRAINIAC_FREEZE_MODEL_ROOT_OUTSIDE_BATTERY") from exc
        if not model_root.is_dir() or v2._is_reparse(model_root):
            raise RuntimeError("BRAINIAC_FREEZE_MODEL_ROOT_UNSAFE")
        for directory, directory_names, file_names in os.walk(model_root):
            base = Path(directory)
            for directory_name in directory_names:
                if v2._is_reparse(base / directory_name):
                    raise RuntimeError("BRAINIAC_FREEZE_MODEL_TREE_REPARSE")
            for file_name in file_names:
                files.add(base / file_name)
    expected = {
        "resnet10_3d",
        "resnet18_3d",
        "resnet34_3d",
        "densenet121_3d",
        "mednext_3d_s",
    }
    if completed_names != expected:
        raise RuntimeError("BRAINIAC_FREEZE_FIVE_MODEL_TERMINAL_SET_INVALID")
    return sorted(files, key=lambda path: path.relative_to(ROOT).as_posix())


def _checkpoint_staging_paths() -> list[Path]:
    if not CHECKPOINT_STAGING_ROOT.exists():
        return []
    if not CHECKPOINT_STAGING_ROOT.is_dir() or v2._is_reparse(CHECKPOINT_STAGING_ROOT):
        raise RuntimeError("BRAINIAC_FREEZE_CHECKPOINT_STAGING_UNSAFE")
    paths: list[Path] = []
    for path in CHECKPOINT_STAGING_ROOT.rglob("*"):
        if v2._is_reparse(path):
            raise RuntimeError("BRAINIAC_FREEZE_CHECKPOINT_STAGING_REPARSE")
        if path.is_file() and path.name in {"resume_current.pt", "resume_previous.pt"}:
            paths.append(path)
    return sorted(paths, key=lambda path: path.relative_to(CHECKPOINT_STAGING_ROOT).as_posix())


def _relevant_lock_inventory() -> list[dict[str, Any]]:
    from src.research import brainiac_action_guard as guard

    candidates = [
        guard.BATTERY_ROOT / guard.BATTERY_LOCK_NAME,
        guard.BATTERY_ROOT / guard.SHARED_GUARD_NAME,
    ]
    for root in (v2.PRIMARY_ROOT, CALIBRATION_ROOT, v2.SMOKE_ROOT):
        if root.exists():
            candidates.extend(sorted(root.rglob(".sample.lock")))
    rows: list[dict[str, Any]] = []
    for path in candidates:
        if not os.path.lexists(path):
            continue
        if not guard._regular_nonreparse(path):
            raise RuntimeError("BRAINIAC_FREEZE_LOCK_UNSAFE")
        raw = path.read_bytes()
        try:
            payload = json.loads(raw.decode("utf-8"))
            pid = int(payload.get("pid", -1))
        except (UnicodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
            raise RuntimeError("BRAINIAC_FREEZE_LOCK_INVALID") from exc
        live, creation = guard._pid_identity(pid)
        if live is None:
            raise RuntimeError("BRAINIAC_FREEZE_LOCK_LIVENESS_UNKNOWN")
        rows.append(
            {
                "path": path.relative_to(ROOT).as_posix(),
                "bytes": len(raw),
                "sha256": hashlib.sha256(raw).hexdigest(),
                "pid": pid,
                "liveness": "LIVE" if live else "DEAD",
                "recorded_process_create_time": payload.get("process_create_time"),
                "observed_process_create_time": creation,
            }
        )
    return sorted(rows, key=lambda row: row["path"])


def collect_pre_repair_live_snapshot(
    *, process_provider: Any | None = None, gpu_provider: Any | None = None
) -> dict[str, Any]:
    """Collect current byte authorities without mutating locks or artifacts."""

    from src.research.brainiac_readiness import inspect_writer_state

    branch, head, dirty = _git_live_state()
    battery = json.loads(BATTERY_MANIFEST_PATH.read_text(encoding="utf-8"))
    writer_state = inspect_writer_state(
        battery,
        process_provider=process_provider,
        gpu_provider=gpu_provider,
    )
    source = verify_frozen_method_x_source_contract(
        expected_digest=EXPECTED_METHOD_X_SOURCE_DIGEST
    )
    battery_inventory = build_file_inventory(
        _battery_preservation_paths(), project_root=ROOT
    )
    staging_paths = _checkpoint_staging_paths()
    c_inventory = build_file_inventory(
        staging_paths, project_root=CHECKPOINT_STAGING_ROOT
    )
    dependency_manifest = _dependency_manifest()
    legacy_inventory = [
        {
            "path": path.relative_to(ROOT).as_posix(),
            **_tree_evidence(path),
        }
        for path in _legacy_incompatible_artifacts()
    ]
    battery_binding = {
        "path": BATTERY_MANIFEST_PATH.relative_to(ROOT).as_posix(),
        "bytes": int(BATTERY_MANIFEST_PATH.stat().st_size),
        "sha256": sha256_file(BATTERY_MANIFEST_PATH),
    }
    primary_valid_n = (
        len(list((v2.PRIMARY_ROOT / "receipts").glob("*.json")))
        if (v2.PRIMARY_ROOT / "receipts").is_dir()
        else 0
    )
    predecessor_freeze_paths = [
        path
        for path in sorted(REGISTRY_ROOT.rglob("*FREEZE*.json"))
        if path.resolve() != PRE_REPAIR_LIVE_FREEZE_PATH.resolve()
    ]
    predecessor_freeze_inventory = build_file_inventory(
        predecessor_freeze_paths,
        project_root=ROOT,
    )
    return build_pre_repair_snapshot(
        branch=branch,
        head=head,
        dirty_paths=dirty,
        writer_state=writer_state,
        method_x_source_digest=str(source["source_contract_digest"]),
        battery_manifest_binding=battery_binding,
        battery_inventory=battery_inventory,
        c_checkpoint_inventory=c_inventory,
        dependency_manifest=dependency_manifest,
        lock_inventory=_relevant_lock_inventory(),
        legacy_inventory=legacy_inventory,
        primary_valid_n=primary_valid_n,
        predecessor_freeze_inventory=predecessor_freeze_inventory,
    )


def materialize_pre_repair_live_freeze() -> dict[str, Any]:
    snapshot = collect_pre_repair_live_snapshot()
    return write_pre_repair_freeze(
        PRE_REPAIR_LIVE_FREEZE_PATH,
        snapshot=snapshot,
    )


def build_preflight_decision(checks: Mapping[str, object]) -> dict[str, Any]:
    """UNKNOWN/NOT_MEASURABLE/False are blockers; only literal True passes."""

    blocking = sorted(str(name) for name, value in checks.items() if value is not True)
    return {
        "status": "PASS" if not blocking else "BLOCKED",
        "blocking_checks": blocking,
        "checks": dict(sorted((str(k), v) for k, v in checks.items())),
    }


EXECUTION_LEG_TIMING_FILENAME = "_execution_leg_timing_v2.json"
EXECUTION_PROFILE_SCIENTIFIC_FIELDS = (
    "output_sha256",
    "transform_sha256",
    "affine_before",
    "affine_after",
    "spacing_before",
    "spacing_after",
    "origin_before",
    "origin_after",
    "direction_before",
    "direction_after",
    "com_coordinate_frame",
    "center_of_mass_before",
    "center_of_mass_after",
    "registration_similarity_before",
    "registration_similarity_after",
    "foreground_fraction",
    "rigid_determinant",
    "orthogonality_error",
    "rotation_deg",
    "translation_norm_mm",
    "normalized_nonzero_mean",
    "normalized_nonzero_std",
)
EXECUTION_PROFILE_DIGEST_MAP_FIELDS = (
    "receipt_file_sha256_by_sample",
    "output_sha256_by_sample",
    "transform_sha256_by_sample",
    "canonical_scientific_sha256_by_sample",
)


def _execution_profile_roots() -> dict[str, Path]:
    base = v2.SMOKE_ROOT / "equivalence_v2"
    return {
        "sequential": base / "sequential",
        "parallel_1": base / "parallel_1",
        "parallel_2": base / "parallel_2",
    }


def _validate_execution_leg_summary(
    leg: Mapping[str, Any],
    *,
    expected_role: str,
    expected_contract_sha256: str,
    expected_samples: set[str],
) -> None:
    if "fresh_run" not in leg or not isinstance(leg.get("fresh_run"), bool):
        raise RuntimeError("BRAINIAC_EQUIVALENCE_FRESH_RUN_EVIDENCE_MISSING")
    required_live = (
        "role",
        "root_relative_path",
        "root_tree_sha256",
        "selection_sha256",
        "timing_receipt_sha256",
        "started_at_utc",
        "finished_at_utc",
        "resume_event_count",
        "live_evidence_status",
        *EXECUTION_PROFILE_DIGEST_MAP_FIELDS,
    )
    if any(field not in leg for field in required_live):
        raise RuntimeError("BRAINIAC_EQUIVALENCE_LIVE_EVIDENCE_MISSING")
    samples = list(map(str, leg.get("sample_hashes") or ()))
    if (
        leg.get("role") != expected_role
        or set(samples) != expected_samples
        or len(samples) != 12
        or leg.get("contract_sha256") != expected_contract_sha256
        or leg.get("integrity_status") != "PASS"
        or leg.get("live_evidence_status") != "PASS_RECOMPUTED_FROM_LIVE_BYTES"
        or not isinstance(leg.get("root_relative_path"), str)
        or not str(leg.get("started_at_utc") or "")
        or not str(leg.get("finished_at_utc") or "")
        or float(leg.get("elapsed_seconds", 0.0)) <= 0.0
        or not isinstance(leg.get("resume_event_count"), int)
        or int(leg["resume_event_count"]) < 0
        or leg.get("selection_sha256") != canonical_sha(sorted(samples))
    ):
        raise RuntimeError("BRAINIAC_EQUIVALENCE_LEG_INVALID")
    for field in ("root_tree_sha256", "timing_receipt_sha256"):
        if re.fullmatch(r"[0-9a-f]{64}", str(leg.get(field) or "")) is None:
            raise RuntimeError("BRAINIAC_EQUIVALENCE_LIVE_EVIDENCE_INVALID")
    for field in EXECUTION_PROFILE_DIGEST_MAP_FIELDS:
        digests = leg.get(field)
        if (
            not isinstance(digests, Mapping)
            or set(map(str, digests)) != expected_samples
            or any(
                re.fullmatch(r"[0-9a-f]{64}", str(value).casefold()) is None
                for value in digests.values()
            )
        ):
            raise RuntimeError("BRAINIAC_EQUIVALENCE_DIGEST_EVIDENCE_INVALID")


def build_execution_profile(
    *,
    sequential: Mapping[str, Any],
    parallel_runs: list[Mapping[str, Any]],
    expected_contract_sha256: str,
) -> dict[str, Any]:
    """Select parallel only from three complete live-byte-verified legs."""

    if len(parallel_runs) != 2:
        raise RuntimeError("BRAINIAC_EQUIVALENCE_PARALLEL_REPEAT_COUNT_INVALID")
    expected_samples = list(map(str, sequential.get("sample_hashes") or ()))
    if len(expected_samples) != 12 or len(set(expected_samples)) != 12:
        raise RuntimeError("BRAINIAC_EQUIVALENCE_SAMPLE_COUNT_INVALID")
    expected_set = set(expected_samples)
    roles_and_legs = [
        ("sequential", sequential),
        ("parallel_1", parallel_runs[0]),
        ("parallel_2", parallel_runs[1]),
    ]
    for role, leg in roles_and_legs:
        _validate_execution_leg_summary(
            leg,
            expected_role=role,
            expected_contract_sha256=expected_contract_sha256,
            expected_samples=expected_set,
        )
    exact = all(
        all(leg.get(field) == sequential.get(field) for field in EXECUTION_PROFILE_DIGEST_MAP_FIELDS)
        for leg in parallel_runs
    )
    sequential_seconds = float(sequential["elapsed_seconds"])
    parallel_median = float(
        statistics.median(float(leg["elapsed_seconds"]) for leg in parallel_runs)
    )
    speedup = 1.0 - parallel_median / sequential_seconds
    timing_fresh = all(
        leg.get("fresh_run") is True and int(leg.get("resume_event_count", -1)) == 0
        for _, leg in roles_and_legs
    )
    parallel_safe = exact and timing_fresh and speedup >= 0.15
    body: dict[str, Any] = {
        "schema_version": 2,
        "artifact": "BRAINIAC_PREPROCESSING_EXECUTION_PROFILE_V2",
        "status": "PASS",
        "execution_mode": "PARALLEL_SAFE" if parallel_safe else "SEQUENTIAL_SAFE",
        "sample_count": 12,
        "parallel_repeat_count": 2,
        "exact_artifact_and_scientific_equivalence": exact,
        "median_speedup_fraction": speedup,
        "minimum_speedup_fraction": 0.15,
        "timing_fresh": timing_fresh,
        "contract_sha256": expected_contract_sha256,
        "legs": {
            "sequential": dict(sequential),
            "parallel": [dict(leg) for leg in parallel_runs],
        },
        "manual_override_allowed": False,
    }
    body["integrity_sha256"] = canonical_sha(body)
    return body


def write_execution_leg_timing_receipt(
    root: Path,
    *,
    role: str,
    sample_hashes: list[str],
    contract_sha256: str,
    started_at_utc: str,
    finished_at_utc: str,
    elapsed_seconds: float,
    fresh_run: bool,
    resume_event_count: int,
) -> dict[str, Any]:
    target = Path(root) / EXECUTION_LEG_TIMING_FILENAME
    body: dict[str, Any] = {
        "schema_version": 2,
        "artifact": "BRAINIAC_EXECUTION_LEG_TIMING_V2",
        "role": role,
        "sample_hashes": sorted(map(str, sample_hashes)),
        "selection_sha256": canonical_sha(sorted(map(str, sample_hashes))),
        "contract_sha256": contract_sha256,
        "started_at_utc": str(started_at_utc),
        "finished_at_utc": str(finished_at_utc),
        "elapsed_seconds": float(elapsed_seconds),
        "fresh_run": bool(fresh_run),
        "resume_event_count": int(resume_event_count),
    }
    if len(body["sample_hashes"]) != 12 or float(elapsed_seconds) <= 0.0:
        raise RuntimeError("BRAINIAC_EQUIVALENCE_TIMING_RECEIPT_INVALID")
    body["integrity_sha256"] = canonical_sha(body)
    if target.exists():
        existing = json.loads(target.read_text(encoding="utf-8"))
        if existing != body:
            raise RuntimeError("BRAINIAC_EQUIVALENCE_TIMING_RECEIPT_IMMUTABLE_MISMATCH")
        return existing
    _atomic_write_json(target, body)
    return body


def recompute_execution_leg(
    *,
    role: str,
    expected_contract_sha256: str,
    expected_receipt_bindings: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    roots = _execution_profile_roots()
    if role not in roots:
        raise RuntimeError("BRAINIAC_EQUIVALENCE_ROLE_INVALID")
    root = roots[role]
    if not root.is_dir() or v2._is_reparse(root):
        raise RuntimeError("BRAINIAC_EQUIVALENCE_LIVE_ROOT_INVALID")
    for path in root.rglob("*"):
        if v2._is_reparse(path):
            raise RuntimeError("BRAINIAC_EQUIVALENCE_LIVE_ROOT_REPARSE")
        relative = path.relative_to(root).as_posix().casefold()
        if (
            path.name == ".sample.lock"
            or ".tmp" in path.name.casefold()
            or relative.startswith("quarantine/")
            or relative.startswith("scratch/") and path.is_file()
        ):
            raise RuntimeError("BRAINIAC_EQUIVALENCE_LIVE_ROOT_PARTIAL")
    timing_path = root / EXECUTION_LEG_TIMING_FILENAME
    if not timing_path.is_file() or v2._is_reparse(timing_path):
        raise RuntimeError("BRAINIAC_EQUIVALENCE_TIMING_RECEIPT_MISSING")
    timing = json.loads(timing_path.read_text(encoding="utf-8"))
    timing_body = dict(timing)
    timing_integrity = timing_body.pop("integrity_sha256", None)
    if (
        timing.get("schema_version") != 2
        or timing.get("artifact") != "BRAINIAC_EXECUTION_LEG_TIMING_V2"
        or timing.get("role") != role
        or timing_integrity != canonical_sha(timing_body)
        or timing.get("contract_sha256") != expected_contract_sha256
        or not isinstance(timing.get("fresh_run"), bool)
        or not isinstance(timing.get("resume_event_count"), int)
    ):
        raise RuntimeError("BRAINIAC_EQUIVALENCE_TIMING_RECEIPT_INVALID")
    receipts_dir = root / "receipts"
    receipt_paths = sorted(receipts_dir.glob("*.json")) if receipts_dir.is_dir() else []
    if len(receipt_paths) != 12:
        raise RuntimeError("BRAINIAC_EQUIVALENCE_SAMPLE_COUNT_INVALID")
    expected_bindings = dict(
        expected_receipt_bindings or _equivalence_receipt_bindings()
    )
    if canonical_sha(_threshold_contract_bindings(expected_bindings)) != expected_contract_sha256:
        raise RuntimeError("BRAINIAC_EQUIVALENCE_CONTRACT_BINDING_INVALID")
    receipts: dict[str, dict[str, Any]] = {}
    receipt_files: dict[str, str] = {}
    for receipt_path in receipt_paths:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        checked = validate_receipt(
            receipt,
            artifact_root=root,
            expected_bindings=expected_bindings,
        )
        sample_hash = str(checked["sample_hash"])
        if sample_hash in receipts or receipt_path.stem != sample_hash:
            raise RuntimeError("BRAINIAC_EQUIVALENCE_RECEIPT_IDENTITY_INVALID")
        receipts[sample_hash] = checked
        receipt_files[sample_hash] = sha256_file(receipt_path)
    sample_hashes = sorted(receipts)
    if (
        timing.get("sample_hashes") != sample_hashes
        or timing.get("selection_sha256") != canonical_sha(sample_hashes)
    ):
        raise RuntimeError("BRAINIAC_EQUIVALENCE_TIMING_SELECTION_DRIFT")
    tree = _tree_evidence(root)
    try:
        relative_root = root.resolve(strict=True).relative_to(ROOT.resolve()).as_posix()
    except (OSError, ValueError) as exc:
        raise RuntimeError("BRAINIAC_EQUIVALENCE_ROOT_OUTSIDE_PROJECT") from exc
    return {
        "role": role,
        "root_relative_path": relative_root,
        "root_tree_sha256": tree["tree_sha256"],
        "sample_hashes": sample_hashes,
        "selection_sha256": canonical_sha(sample_hashes),
        "receipt_file_sha256_by_sample": receipt_files,
        "output_sha256_by_sample": {
            sample: str(row["output_sha256"]) for sample, row in sorted(receipts.items())
        },
        "transform_sha256_by_sample": {
            sample: str(row["transform_sha256"]) for sample, row in sorted(receipts.items())
        },
        "canonical_scientific_sha256_by_sample": {
            sample: canonical_sha(
                {field: row[field] for field in EXECUTION_PROFILE_SCIENTIFIC_FIELDS}
            )
            for sample, row in sorted(receipts.items())
        },
        "contract_sha256": expected_contract_sha256,
        "started_at_utc": timing["started_at_utc"],
        "finished_at_utc": timing["finished_at_utc"],
        "elapsed_seconds": float(timing["elapsed_seconds"]),
        "fresh_run": timing["fresh_run"],
        "resume_event_count": int(timing["resume_event_count"]),
        "timing_receipt_sha256": sha256_file(timing_path),
        "integrity_status": "PASS",
        "live_evidence_status": "PASS_RECOMPUTED_FROM_LIVE_BYTES",
    }


def validate_execution_profile(
    profile: Mapping[str, Any],
    *,
    expected_contract_sha256: str,
    expected_receipt_bindings: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Rehash roots, receipts, outputs and transforms before trusting the profile."""

    body = dict(profile)
    recorded = body.pop("integrity_sha256", None)
    if recorded != canonical_sha(body):
        raise RuntimeError("BRAINIAC_EQUIVALENCE_PROFILE_INTEGRITY_INVALID")
    if (
        profile.get("schema_version") != 2
        or profile.get("artifact") != "BRAINIAC_PREPROCESSING_EXECUTION_PROFILE_V2"
    ):
        raise RuntimeError("BRAINIAC_EQUIVALENCE_PROFILE_SCHEMA_INVALID")
    legs = profile.get("legs")
    if not isinstance(legs, Mapping):
        raise RuntimeError("BRAINIAC_EQUIVALENCE_PROFILE_LEGS_MISSING")
    parallel = legs.get("parallel")
    if not isinstance(parallel, list) or len(parallel) != 2:
        raise RuntimeError("BRAINIAC_EQUIVALENCE_PROFILE_LEGS_INVALID")
    recomputed = {
        role: recompute_execution_leg(
            role=role,
            expected_contract_sha256=expected_contract_sha256,
            expected_receipt_bindings=expected_receipt_bindings,
        )
        for role in ("sequential", "parallel_1", "parallel_2")
    }
    rebuilt = build_execution_profile(
        sequential=recomputed["sequential"],
        parallel_runs=[recomputed["parallel_1"], recomputed["parallel_2"]],
        expected_contract_sha256=expected_contract_sha256,
    )
    for field in (
        "status",
        "execution_mode",
        "sample_count",
        "parallel_repeat_count",
        "exact_artifact_and_scientific_equivalence",
        "median_speedup_fraction",
        "minimum_speedup_fraction",
        "timing_fresh",
        "contract_sha256",
        "manual_override_allowed",
        "legs",
    ):
        if profile.get(field) != rebuilt.get(field):
            raise RuntimeError(
                f"BRAINIAC_EQUIVALENCE_PROFILE_RECOMPUTE_MISMATCH:{field}"
            )
    return dict(profile)


def _archive_proven_stale_shared_guard(
    *,
    process_provider: Any | None = None,
    gpu_provider: Any | None = None,
) -> dict[str, Any]:
    """Move a liveness-proven dead shared guard to append-only evidence."""

    from src.research import brainiac_action_guard as guard
    from src.research.brainiac_readiness import (
        _load_json_strict,
        _writer_blockers,
        inspect_writer_state,
    )

    battery, _ = _load_json_strict(guard.DEFAULT_BATTERY_MANIFEST)
    writer_state = inspect_writer_state(
        battery,
        process_provider=process_provider,
        gpu_provider=gpu_provider,
    )
    blockers = _writer_blockers(writer_state)
    if blockers:
        raise RuntimeError(
            "BRAINIAC_STALE_GUARD_ARCHIVE_BLOCKED:" + ",".join(blockers)
        )

    lock_path = guard.BATTERY_ROOT / guard.SHARED_GUARD_NAME
    status = guard._recorded_lock_status(lock_path, role="shared_guard")
    if status.get("present") is not True or status.get("liveness") != "DEAD":
        raise RuntimeError("BRAINIAC_STALE_GUARD_DEAD_PROOF_REQUIRED")
    if not guard._regular_nonreparse(lock_path):
        raise RuntimeError("BRAINIAC_STALE_GUARD_UNSAFE")
    raw = lock_path.read_bytes()
    try:
        lock_payload = json.loads(raw.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("BRAINIAC_STALE_GUARD_UNREADABLE") from exc
    digest = hashlib.sha256(raw).hexdigest()
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    archive_dir = REGISTRY_ROOT / "superseded" / "stale_shared_guards"
    archive_dir.mkdir(parents=True, exist_ok=True)
    archive_path = archive_dir / f"{stamp}-{time.time_ns()}-{digest}.lock"
    os.replace(lock_path, archive_path)
    if lock_path.exists() or not archive_path.is_file():
        raise RuntimeError("BRAINIAC_STALE_GUARD_ARCHIVE_FAILED")
    body: dict[str, Any] = {
        "schema_version": 1,
        "status": "ARCHIVED_PROVEN_DEAD",
        "role": "shared_guard",
        "dead_pid": status.get("pid"),
        "recorded_process_create_time": lock_payload.get("process_create_time"),
        "writer_state": writer_state,
        "source_sha256": digest,
        "archive_sha256": sha256_file(archive_path),
        "archive_path": str(archive_path),
        "created_at_utc": _utc_now(),
        "manual_override_allowed": False,
    }
    body["integrity_sha256"] = canonical_sha(body)
    receipt_path = archive_path.with_suffix(".receipt.json")
    _atomic_write_json(receipt_path, body)
    return {**body, "receipt_path": str(receipt_path)}


def _tree_evidence(path: Path) -> dict[str, Any]:
    """Hash every byte and relative path in a non-reparse file/tree."""

    candidate = Path(path)
    if not candidate.exists() or v2._is_reparse(candidate):
        raise RuntimeError("BRAINIAC_STALE_ARTIFACT_UNSAFE")
    files = [candidate] if candidate.is_file() else sorted(
        item for item in candidate.rglob("*") if item.is_file()
    )
    digest = hashlib.sha256()
    total = 0
    for item in files:
        if v2._is_reparse(item):
            raise RuntimeError("BRAINIAC_STALE_ARTIFACT_REPARSE_REJECTED")
        relative = item.name if candidate.is_file() else item.relative_to(candidate).as_posix()
        raw = item.read_bytes()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(hashlib.sha256(raw).digest())
        digest.update(b"\0")
        total += len(raw)
    return {
        "tree_sha256": digest.hexdigest(),
        "file_count": len(files),
        "bytes": total,
    }


def _calibration_generation_is_current() -> bool:
    """Return True only for a resumable calibration root under the live contract.

    A calibration directory is not legacy merely because it exists.  It is a
    sample-boundary checkpoint: every committed receipt must remain reusable
    after SIGINT, SIGTERM, or a later restart.  Pre-equivalence roots and roots
    whose committed receipts do not validate against the current scientific
    bindings are still classified as incompatible and archived append-only.
    """

    if not CALIBRATION_ROOT.exists():
        return True
    if not CALIBRATION_ROOT.is_dir() or v2._is_reparse(CALIBRATION_ROOT):
        return False
    _sequential_only, profile_valid = _parallel_unlock_state()
    if not profile_valid:
        return False
    allowed_top_level = {
        "outputs",
        "transforms",
        "receipts",
        "quarantine",
        "journals",
    }
    if any(path.name not in allowed_top_level for path in CALIBRATION_ROOT.iterdir()):
        return False
    membership = v2.load_locked_membership()
    selected = set(
        v2.select_calibration_samples(
            [str(row["sample_hash"]) for row in membership]
        )
    )
    # Calibration receipts are immutable pre-threshold evidence.  A later
    # source-only compatibility closure must not make those byte-valid receipts
    # look legacy: validate them against the same locked historical bindings
    # used by threshold reconstruction and equivalence replay.
    bindings = _equivalence_receipt_bindings()
    receipts_dir = CALIBRATION_ROOT / "receipts"
    for receipt_path in sorted(receipts_dir.glob("*.json")):
        sample_hash = receipt_path.stem
        if sample_hash not in selected:
            return False
        try:
            v2.load_committed_receipt(
                sample_hash,
                CALIBRATION_ROOT,
                expected_bindings=bindings,
            )
        except (OSError, RuntimeError, TypeError, ValueError, json.JSONDecodeError):
            return False
    return True


def _legacy_incompatible_artifacts() -> list[Path]:
    candidates = [
        v2.SMOKE_ROOT / "eq_sequential",
        v2.SMOKE_ROOT / "eq_parallel",
        v2.SMOKE_ROOT / "sequential",
        v2.SMOKE_ROOT / "orchestrated",
        LEGACY_EXECUTION_PROFILE_PATH,
    ]
    if CALIBRATION_ROOT.exists() and not _calibration_generation_is_current():
        candidates.append(CALIBRATION_ROOT)
    legacy_report = REGISTRY_ROOT / "brainiac_preprocessing_v2_equivalence.json"
    if legacy_report.is_file() and not EXECUTION_PROFILE_PATH.is_file():
        candidates.append(legacy_report)
    current_equivalence = v2.SMOKE_ROOT / "equivalence_v2"
    if current_equivalence.is_dir():
        receipt_paths = sorted(current_equivalence.glob("*/receipts/*.json"))
        expected_bindings = _equivalence_receipt_bindings()
        incompatible = False
        for receipt_path in receipt_paths:
            try:
                receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
                validate_receipt(
                    receipt,
                    artifact_root=receipt_path.parents[1],
                    expected_bindings=expected_bindings,
                )
            except (OSError, RuntimeError, TypeError, ValueError, json.JSONDecodeError):
                incompatible = True
                break
        if incompatible:
            candidates.append(current_equivalence)
            if EXECUTION_PROFILE_PATH.is_file():
                candidates.append(EXECUTION_PROFILE_PATH)
            if legacy_report.is_file():
                candidates.append(legacy_report)
    return [path for path in candidates if path.exists()]


def _assert_tree_locks_proven_dead(path: Path) -> None:
    locks = [path] if path.is_file() and path.name == ".sample.lock" else (
        list(path.rglob(".sample.lock")) if path.is_dir() else []
    )
    for lock in locks:
        try:
            payload = json.loads(lock.read_text(encoding="utf-8"))
            pid = int(payload.get("pid", -1))
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise RuntimeError("BRAINIAC_STALE_ARTIFACT_LOCK_INVALID") from exc
        live, _created = v2._pid_identity(pid)
        if live is not False:
            reason = "LIVE" if live is True else "UNKNOWN"
            raise RuntimeError(f"BRAINIAC_STALE_ARTIFACT_LOCK_{reason}")


def _archive_legacy_incompatible_artifacts() -> dict[str, Any] | None:
    """Move old-policy calibration/smoke evidence; never overwrite or delete it."""

    sources = _legacy_incompatible_artifacts()
    if not sources:
        return None
    for source in sources:
        _assert_tree_locks_proven_dead(source)
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    archive_root = (
        ROOT
        / "data"
        / "derived"
        / "method_x_brainiac"
        / "superseded"
        / f"preprocessing_contract_v2_reconciliation_{stamp}_{time.time_ns()}"
    )
    archive_root.mkdir(parents=True, exist_ok=False)
    rows: list[dict[str, Any]] = []
    for index, source in enumerate(sources):
        before = _tree_evidence(source)
        target = archive_root / f"{index:02d}_{source.name}"
        os.replace(source, target)
        after = _tree_evidence(target)
        if source.exists() or before != after:
            raise RuntimeError("BRAINIAC_STALE_ARTIFACT_ARCHIVE_VERIFY_FAILED")
        rows.append(
            {
                "source_role": source.name,
                "archive_relative_path": target.relative_to(ROOT).as_posix(),
                **after,
            }
        )
    body: dict[str, Any] = {
        "schema_version": 1,
        "status": "ARCHIVED_INCOMPATIBLE_PREPROCESSING_GENERATION",
        "artifact_count": len(rows),
        "artifacts": rows,
        "reason_code": "OLD_POLICY_OR_PRE_EQUIVALENCE_PROFILE",
        "created_at_utc": _utc_now(),
        "manual_override_allowed": False,
        "deleted": False,
    }
    body["integrity_sha256"] = canonical_sha(body)
    receipt = REGISTRY_ROOT / "superseded" / (
        f"{stamp}-{time.time_ns()}-preprocessing-artifact-reconciliation.json"
    )
    _atomic_write_json(receipt, body)
    return {**body, "receipt_path": str(receipt)}


def _inspect_sample_locks(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for lock in sorted(Path(root).glob("journals/*/.sample.lock")):
        try:
            payload = json.loads(lock.read_text(encoding="utf-8"))
            pid = int(payload.get("pid", -1))
            live, _created = v2._pid_identity(pid)
            liveness = "LIVE" if live is True else ("DEAD" if live is False else "UNKNOWN")
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            pid = None
            liveness = "UNKNOWN"
        rows.append(
            {
                "sample_hash_prefix": lock.parent.name[:10],
                "pid": pid,
                "liveness": liveness,
            }
        )
    return rows


def _inspect_primary_sample_locks() -> list[dict[str, Any]]:
    return _inspect_sample_locks(v2.PRIMARY_ROOT)


def _archive_proven_dead_sample_locks(root: Path) -> list[str]:
    archived: list[str] = []
    for lock in sorted(Path(root).glob("journals/*/.sample.lock")):
        sample_hash = lock.parent.name
        v2._reconcile_stale_sample_lock(
            lock, root=Path(root), sample_hash=sample_hash
        )
        archived.append(sample_hash[:10])
    return archived


def _archive_proven_dead_primary_sample_locks() -> list[str]:
    return _archive_proven_dead_sample_locks(v2.PRIMARY_ROOT)


def _pin_itk_threads() -> int:
    """Pin the measured byte-equivalent ITK thread budget (runtime policy v3)."""

    import SimpleITK as sitk

    threads = max(1, int(os.environ.get("BRAINIAC_V2_ITK_THREADS", "8")))
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(threads)
    return threads


def _scratch_paths(scratch: Path, sample_hash: str) -> tuple[Path, Path]:
    """Registered volume as UNCOMPRESSED .nii scratch; brain output stays .nii.gz."""

    return (
        scratch / f"{sample_hash}_registered.nii",
        scratch / f"{sample_hash}_brain.nii.gz",
    )


def _equivalence_mismatches(seq_receipts, par_receipts) -> list[str]:
    """Canonical scientific-field comparator between two execution legs."""

    fields = (
        "output_sha256",
        "transform_sha256",
        "foreground_fraction",
        "rigid_determinant",
        "orthogonality_error",
        "rotation_deg",
        "translation_norm_mm",
        "normalized_nonzero_std",
    )
    index = {str(r["sample_hash"]): r for r in par_receipts}
    out: list[str] = []
    for ref in seq_receipts:
        row = index.get(str(ref["sample_hash"]))
        if row is None:
            out.append(f"{ref['sample_hash'][:10]}:MISSING_IN_PARALLEL")
            continue
        for f in fields:
            if ref.get(f) != row.get(f):
                out.append(f"{ref['sample_hash'][:10]}:{f}")
    return out


def _load_integrity_json(path: Path) -> tuple[dict[str, Any], str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    body = dict(payload)
    recorded = body.pop("integrity_sha256", None)
    if recorded != canonical_sha(body):
        raise RuntimeError(f"BRAINIAC_V2_REQUIRED_RECEIPT_INTEGRITY_INVALID:{path.name}")
    return payload, sha256_file(path)


def _validate_runtime_receipts() -> dict[str, str]:
    runtime, runtime_sha = _load_integrity_json(RUNTIME_LOCK)
    strict, strict_sha = _load_integrity_json(STRICT_RECEIPT)
    forward, forward_sha = _load_integrity_json(FORWARD_RECEIPT)
    if (
        runtime.get("status") != "PASS"
        or strict.get("status") != "PASS"
        or strict.get("strict") is not True
        or forward.get("status") != "PASS"
        or forward.get("outer_test_touched") is not False
        or tuple(forward.get("input_shape", ())) != (1, 1, 96, 96, 96)
        or tuple(forward.get("output_shape", ())) != (1, 768)
        or strict.get("environment_lock_sha256") != runtime_sha
        or forward.get("strict_load_receipt_file_sha256") != strict_sha
    ):
        raise RuntimeError("BRAINIAC_V2_OFFICIAL_RUNTIME_CHAIN_INVALID")
    vendor_tree = str(strict.get("source_tree_sha256") or "").casefold()
    if len(vendor_tree) != 64:
        raise RuntimeError("BRAINIAC_V2_VENDOR_TREE_BINDING_INVALID")
    return {
        "runtime_lock_sha256": runtime_sha,
        "vendor_tree_sha256": vendor_tree,
    }


def _validate_template_assets() -> dict[str, str]:
    if not TEMPLATE_PATH.is_file() or not TEMPLATE_LOCK.is_file():
        raise RuntimeError("BRAINIAC_V2_TEMPLATE_ASSET_MISSING")
    if sha256_file(TEMPLATE_PATH) != v2.EXPECTED_TEMPLATE_SHA256:
        raise RuntimeError("BRAINIAC_V2_TEMPLATE_HASH_MISMATCH")
    lock = json.loads(TEMPLATE_LOCK.read_text(encoding="utf-8"))
    if lock.get("sha256") != v2.EXPECTED_TEMPLATE_SHA256:
        raise RuntimeError("BRAINIAC_V2_TEMPLATE_LOCK_INVALID")
    if not DERIVED_TEMPLATE.is_file() or not DERIVED_TEMPLATE_LOCK.is_file():
        raise RuntimeError("BRAINIAC_V2_DERIVED_TEMPLATE_MISSING")
    derived_lock = json.loads(DERIVED_TEMPLATE_LOCK.read_text(encoding="utf-8"))
    if (
        derived_lock.get("source_template_sha256") != v2.EXPECTED_TEMPLATE_SHA256
        or derived_lock.get("derived_sha256") != sha256_file(DERIVED_TEMPLATE)
        or derived_lock.get("output_spacing_mm") != [1.0, 1.0, 1.0]
    ):
        raise RuntimeError("BRAINIAC_V2_DERIVED_TEMPLATE_INVALID")
    return {
        "template_sha256": v2.EXPECTED_TEMPLATE_SHA256,
        "template_lock_sha256": sha256_file(TEMPLATE_LOCK),
    }


def _validate_hd_bet_weight() -> None:
    if (
        not v2.HD_BET_WEIGHT.is_file()
        or sha256_file(v2.HD_BET_WEIGHT) != v2.EXPECTED_HD_BET_WEIGHT_SHA256
    ):
        raise RuntimeError("BRAINIAC_V2_HD_BET_WEIGHT_INVALID")


def _policy() -> dict[str, Any]:
    payload = json.loads(Path(v2.POLICY_PATH).read_text(encoding="utf-8"))
    orchestration = payload.get("orchestration")
    execution = payload.get("execution")
    assets = payload.get("pinned_assets")
    storage = payload.get("storage_locality")
    performance = payload.get("performance_evidence")
    registration = payload.get("registration_contract")
    previous_performance_path = (
        ROOT / str(performance.get("previous_runtime_probe_relative_path"))
        if isinstance(performance, Mapping)
        else ROOT / "__missing_performance_evidence__"
    )
    repair_evidence_path = (
        ROOT / str(performance.get("determinism_repair_relative_path"))
        if isinstance(performance, Mapping)
        else ROOT / "__missing_determinism_repair_evidence__"
    )
    if (
        payload.get("schema_version") != 3
        or payload.get("immutable") is not True
        or not isinstance(orchestration, Mapping)
        or orchestration.get("runtime_policy_mutation_forbidden") is not True
        or int(orchestration.get("cpu_workers_max", -1)) != 2
        or int(orchestration.get("gpu_workers_max", -1)) != 1
        or int(orchestration.get("max_parallel_samples", -1)) != 2
        or int(orchestration.get("n4_samples_in_flight", -1)) != 2
        or int(orchestration.get("registration_workers_max", -1)) != 1
        or int(orchestration.get("itk_threads_default", -1)) != 8
        or orchestration.get("worker_model")
        != "PARALLEL_N4_SERIAL_REGISTRATION_SERIAL_GPU"
        or float(orchestration.get("parallel_minimum_speedup_fraction", -1.0))
        != 0.15
        or not isinstance(execution, Mapping)
        or execution.get("production_device") != "cuda"
        or execution.get("cpu_fallback") is not False
        or not isinstance(registration, Mapping)
        or registration.get("sampling_strategy") != "NONE_FULL"
        or int(registration.get("registration_threads", -1)) != 1
        or int(registration.get("registration_work_units", -1)) != 1
        or int(registration.get("parameter_quantization_decimals", -1))
        != RIGID_PARAMETER_QUANTIZATION_DECIMALS
        or int(registration.get("metric_quantization_decimals", -1))
        != REGISTRATION_METRIC_QUANTIZATION_DECIMALS
        or registration.get("quality_selection_or_fallback") is not False
        or not isinstance(assets, Mapping)
        or assets.get("vendor_commit") != v2.VENDOR_COMMIT
        or assets.get("hd_bet_weight_sha256")
        != v2.EXPECTED_HD_BET_WEIGHT_SHA256
        or assets.get("template_sha256") != v2.EXPECTED_TEMPLATE_SHA256
        or not isinstance(storage, Mapping)
        or storage.get("authoritative_artifacts") != "H_ONLY"
        or storage.get("local_scratch_allowlist") != str(v2.LOCAL_SCRATCH_ROOT)
        or int(storage.get("local_scratch_max_bytes", -1))
        != v2.LOCAL_SCRATCH_MAX_BYTES
        or int(storage.get("local_scratch_minimum_free_bytes", -1))
        != v2.LOCAL_SCRATCH_MIN_FREE_BYTES
        or storage.get("silent_fallback") is not False
        or not isinstance(performance, Mapping)
        or performance.get("scientific_pipeline_changed") is not True
        or performance.get("concurrent_n4_byte_equivalent_prior_probe") is not True
        or performance.get("seeded_random_registration_rejected") is not True
        or performance.get("final_exact_equivalence_pending_plan017") is not True
        or not previous_performance_path.is_file()
        or sha256_file(previous_performance_path)
        != performance.get("previous_runtime_probe_sha256")
        or not repair_evidence_path.is_file()
        or sha256_file(repair_evidence_path)
        != performance.get("determinism_repair_sha256")
    ):
        raise RuntimeError("BRAINIAC_V2_POLICY_CONTRACT_INVALID")
    return payload


def _config() -> dict[str, Any]:
    import yaml

    payload = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError("BRAINIAC_V2_CONFIG_INVALID")
    roots = payload.get("roots")
    membership = payload.get("membership")
    calibration = payload.get("calibration")
    sensitivity = payload.get("sensitivity")
    orchestration = payload.get("orchestration")
    registration = payload.get("registration")
    storage = payload.get("storage_locality")
    signal_contract = payload.get("signal")
    device = payload.get("device")
    if (
        payload.get("schema_version") != 2
        or not isinstance(roots, Mapping)
        or roots.get("primary") != "data/derived/method_x_brainiac/r1_primary_v2"
        or roots.get("raw_read_only") != str(v2.RAW_DATA_ROOT)
        or not isinstance(membership, Mapping)
        or int(membership.get("expected_real_rows", -1)) != v2.EXPECTED_REAL_ROWS
        or list(membership.get("fields") or ()) != ["sample_hash", "t1_file_hash"]
        or not isinstance(calibration, Mapping)
        or int(calibration.get("sample_count", -1)) != v2.CALIBRATION_SAMPLE_COUNT
        or calibration.get("threshold_rule") != "median_pm_6_mad"
        or not isinstance(sensitivity, Mapping)
        or int(sensitivity.get("sample_count", -1)) != v2.SENSITIVITY_SAMPLE_COUNT
        or sensitivity.get("direct_route_role") != "sensitivity_only"
        or not isinstance(orchestration, Mapping)
        or int(orchestration.get("cpu_workers_max", -1)) != 2
        or int(orchestration.get("gpu_workers_max", -1)) != 1
        or int(orchestration.get("n4_samples_in_flight", -1)) != 2
        or int(orchestration.get("registration_workers_max", -1)) != 1
        or int(orchestration.get("itk_threads_default", -1)) != 8
        or orchestration.get("worker_model")
        != "PARALLEL_N4_SERIAL_REGISTRATION_SERIAL_GPU"
        or orchestration.get("equivalence_required") is not True
        or orchestration.get("equivalence_design")
        != "12_LABEL_BLIND_ONE_SEQUENTIAL_TWO_INDEPENDENT_PARALLEL"
        or int(orchestration.get("parallel_repeat_count", -1)) != 2
        or float(orchestration.get("parallel_minimum_speedup_fraction", -1.0))
        != 0.15
        or orchestration.get("execution_profile_authority")
        != "HASH_LOCKED_LIVE_BYTE_VERIFIED"
        or not isinstance(registration, Mapping)
        or registration.get("transform") != "Euler3D_rigid"
        or registration.get("metric") != "MattesMutualInformation"
        or int(registration.get("histogram_bins", -1)) != 50
        or registration.get("sampling_strategy") != "NONE_FULL"
        or int(registration.get("threads", -1)) != 1
        or int(registration.get("work_units", -1)) != 1
        or int(registration.get("parameter_quantization_decimals", -1))
        != RIGID_PARAMETER_QUANTIZATION_DECIMALS
        or int(registration.get("metric_quantization_decimals", -1))
        != REGISTRATION_METRIC_QUANTIZATION_DECIMALS
        or not isinstance(storage, Mapping)
        or storage.get("authoritative_artifacts") != "H_ONLY"
        or storage.get("local_scratch") != str(v2.LOCAL_SCRATCH_ROOT)
        or int(storage.get("local_scratch_max_bytes", -1))
        != v2.LOCAL_SCRATCH_MAX_BYTES
        or int(storage.get("local_scratch_minimum_free_bytes", -1))
        != v2.LOCAL_SCRATCH_MIN_FREE_BYTES
        or storage.get("silent_fallback") is not False
        or not isinstance(signal_contract, Mapping)
        or int(signal_contract.get("clean_stop_exit_code", -1)) != 75
        or signal_contract.get("safe_boundary") != "after_current_sample_commit"
        or not isinstance(device, Mapping)
        or device.get("default") != "cuda"
        or device.get("fallback") != "none"
        or device.get("cpu_role") != "test_fixtures_only_not_production"
        or payload.get("claim_ceiling")
        != "POST_HOC_INTERNAL_TECHNICAL_EVIDENCE"
    ):
        raise RuntimeError("BRAINIAC_V2_CONFIG_CONTRACT_INVALID")
    return payload


def _lineage_binding() -> dict[str, str]:
    lineage = json.loads(LINEAGE_PATH.read_text(encoding="utf-8"))
    lineage_sha = sha256_file(LINEAGE_PATH)
    cache = v2.CACHE_PATH.stat()
    if (
        lineage.get("status") != "PASS"
        or Path(str(lineage.get("cache") or ""))
        != Path("data/cached/t1_volume_cache_128.h5")
        or int(lineage.get("cache_size", -1)) != cache.st_size
        or int(lineage.get("cache_mtime_ns", -1)) != cache.st_mtime_ns
        or len(str(lineage.get("cache_sha256") or "")) != 64
    ):
        raise RuntimeError("BRAINIAC_V2_DATA_LINEAGE_INVALID")
    return {
        "cache_sha256": str(lineage["cache_sha256"]).casefold(),
        "data_lineage_manifest_sha256": lineage_sha,
    }


def _dependency_manifest() -> dict[str, Any]:
    return build_execution_dependency_manifest(
        Path(__file__).resolve(),
        static_inputs={"policy": v2.POLICY_PATH, "config": CONFIG_PATH},
        dynamic_code_inputs=VENDOR_RUNTIME_CODE,
    )


def _pipeline_contract_sha256() -> str:
    policy = _policy()
    scientific = {
        "pipeline_contract": policy.get("pipeline_contract"),
        "fixed_qc_limits": policy.get("fixed_qc_limits"),
        "calibration": policy.get("calibration"),
        "quarantine": policy.get("quarantine"),
        "sensitivity": policy.get("sensitivity"),
        "registration_seed": v2.REGISTRATION_SEED,
    }
    return canonical_sha(scientific)


def _base_bindings() -> dict[str, str]:
    _config()
    membership = v2.load_locked_membership()
    dependencies = _dependency_manifest()
    bindings = {
        **_validate_runtime_receipts(),
        **_validate_template_assets(),
        **_lineage_binding(),
        "policy_sha256": v2.policy_sha256(),
        "config_sha256": sha256_file(CONFIG_PATH),
        "pipeline_contract_sha256": _pipeline_contract_sha256(),
        "dependency_closure_sha256": str(dependencies["closure_sha256"]),
        "membership_sha256": canonical_sha(membership),
        "catalog_sha256": sha256_file(v2.CATALOG_PATH),
    }
    _, transform_binding = v2.build_contract_transform()
    bindings["official_transform_source_sha256"] = transform_binding[
        "official_transform_source_sha256"
    ]
    if THRESHOLD_MANIFEST_PATH.is_file():
        bindings["qc_threshold_manifest_sha256"] = sha256_file(THRESHOLD_MANIFEST_PATH)
    else:
        bindings["qc_threshold_manifest_sha256"] = hashlib.sha256(
            b"NOT_YET_CALIBRATED_PLAN_003"
        ).hexdigest()
    return bindings


def _plan017_evidence_dependency_closure(fallback: str) -> str:
    """Resolve the closure that produced the locked Plan-017 evidence.

    Equivalence, calibration and threshold evidence are historical immutable
    artifacts.  Once their exact post-calibration migration has been sealed,
    later source-only changes must be proven by a compatibility receipt rather
    than silently pretending those bytes were generated by the latest source.
    Fresh workspaces without that migration simply use the current closure.
    """

    if not POSTCALIBRATION_SOURCE_MIGRATION_PATH.is_file():
        return str(fallback)
    report = json.loads(
        POSTCALIBRATION_SOURCE_MIGRATION_PATH.read_text(encoding="utf-8")
    )
    body = dict(report)
    recorded = body.pop("integrity_sha256", None)
    locked = str(report.get("new_dependency_closure_sha256") or "")
    if (
        report.get("status")
        != "PASS_POSTCALIBRATION_SOURCE_ONLY_MIGRATION"
        or recorded != canonical_sha(body)
        or locked != PLAN017_LOCKED_EVIDENCE_DEPENDENCY_CLOSURE_SHA256
    ):
        raise RuntimeError("BRAINIAC_PLAN017_LOCKED_EVIDENCE_CLOSURE_INVALID")
    return locked


def _equivalence_receipt_bindings() -> dict[str, str]:
    """Return the immutable pre-calibration bindings for equivalence evidence.

    The sequential/parallel equivalence legs are intentionally executed before
    the 32-sample threshold is locked.  Their provenance must therefore keep
    the pre-calibration sentinel even after the threshold file exists.  Using
    ``_base_bindings`` directly after calibration would both misrepresent that
    chronology and falsely invalidate byte-identical equivalence receipts.
    """

    bindings = _base_bindings()
    bindings["dependency_closure_sha256"] = _plan017_evidence_dependency_closure(
        bindings["dependency_closure_sha256"]
    )
    bindings["qc_threshold_manifest_sha256"] = hashlib.sha256(
        b"NOT_YET_CALIBRATED_PLAN_003"
    ).hexdigest()
    return bindings


def _threshold_evidence_bindings() -> dict[str, str]:
    """Bindings used to validate the already-locked 32-sample threshold."""

    bindings = _base_bindings()
    bindings["dependency_closure_sha256"] = _plan017_evidence_dependency_closure(
        bindings["dependency_closure_sha256"]
    )
    return bindings


def _threshold_contract_bindings(bindings: Mapping[str, str]) -> dict[str, str]:
    roles = (
        "policy_sha256",
        "config_sha256",
        "pipeline_contract_sha256",
        "dependency_closure_sha256",
        "membership_sha256",
        "cache_sha256",
        "catalog_sha256",
        "runtime_lock_sha256",
        "vendor_tree_sha256",
        "template_sha256",
        "template_lock_sha256",
        "official_transform_source_sha256",
    )
    selected = {role: str(bindings[role]) for role in roles}
    if any(len(value) != 64 for value in selected.values()):
        raise RuntimeError("BRAINIAC_V2_THRESHOLD_CONTRACT_BINDING_INVALID")
    return selected


def _load_bound_threshold_manifest(bindings: Mapping[str, str]) -> dict[str, Any]:
    if not THRESHOLD_MANIFEST_PATH.is_file():
        raise RuntimeError("BRAINIAC_V2_THRESHOLD_MANIFEST_REQUIRED_BEFORE_RUN")
    manifest = json.loads(THRESHOLD_MANIFEST_PATH.read_text(encoding="utf-8"))
    membership = v2.load_locked_membership()
    selected = v2.select_calibration_samples(
        [row["sample_hash"] for row in membership]
    )
    if str(bindings.get("qc_threshold_manifest_sha256") or "") != sha256_file(
        THRESHOLD_MANIFEST_PATH
    ):
        raise RuntimeError("BRAINIAC_V2_PRIMARY_THRESHOLD_BINDING_INVALID")
    threshold_evidence_bindings = _threshold_evidence_bindings()
    calibration_bindings = dict(threshold_evidence_bindings)
    calibration_bindings["qc_threshold_manifest_sha256"] = hashlib.sha256(
        b"NOT_YET_CALIBRATED_PLAN_003"
    ).hexdigest()
    receipt_hashes: dict[str, str] = {}
    for sample_hash in selected:
        receipt = v2.load_committed_receipt(
            sample_hash,
            CALIBRATION_ROOT,
            expected_bindings=calibration_bindings,
        )
        receipt_hashes[sample_hash] = str(receipt["integrity_sha256"])
    return validate_threshold_manifest(
        manifest,
        expected_sample_hashes=selected,
        expected_receipt_integrity_sha256s=receipt_hashes,
        expected_contract_bindings=_threshold_contract_bindings(
            threshold_evidence_bindings
        ),
    )


def _load_fixed_template() -> Any:
    import SimpleITK as sitk

    return sitk.ReadImage(str(DERIVED_TEMPLATE), sitk.sitkFloat32)


RIGID_PARAMETER_QUANTIZATION_DECIMALS = 4
REGISTRATION_METRIC_QUANTIZATION_DECIMALS = 7


def _canonical_rigid_parameters(parameters: Sequence[float]) -> tuple[float, ...]:
    """Remove scientifically irrelevant optimizer jitter before resampling.

    Full Mattes-MI evaluation on the pinned Windows/SimpleITK build still
    varied at roughly 1e-6 mm / 1e-9 rad between otherwise identical calls.
    Rounding all six Euler parameters to 1e-4 is far below the 1 mm template
    resolution while making the transform a stable, receipt-bindable object.
    """

    values = tuple(float(value) for value in parameters)
    if len(values) != 6 or not all(math.isfinite(value) for value in values):
        raise RuntimeError("BRAINIAC_V2_RIGID_PARAMETERS_INVALID")
    canonical = tuple(
        round(value, RIGID_PARAMETER_QUANTIZATION_DECIMALS) for value in values
    )
    return tuple(0.0 if value == 0.0 else value for value in canonical)


def _canonical_registration_metric(value: float) -> float:
    metric = float(value)
    if not math.isfinite(metric):
        raise RuntimeError("BRAINIAC_V2_RIGID_REGISTRATION_METRIC_NONFINITE")
    canonical = round(metric, REGISTRATION_METRIC_QUANTIZATION_DECIMALS)
    return 0.0 if canonical == 0.0 else canonical


def _canonicalize_rigid_transform(transform: Any, sitk: Any) -> Any:
    candidate = transform
    if hasattr(transform, "GetNumberOfTransforms"):
        count = int(transform.GetNumberOfTransforms())
        if count != 1:
            raise RuntimeError("BRAINIAC_V2_RIGID_TRANSFORM_COMPOSITION_INVALID")
        candidate = transform.GetBackTransform()
    try:
        canonical = sitk.Euler3DTransform(candidate)
    except RuntimeError as exc:
        raise RuntimeError("BRAINIAC_V2_RIGID_TRANSFORM_TYPE_INVALID") from exc
    fixed = tuple(float(value) for value in canonical.GetFixedParameters())
    if not fixed or not all(math.isfinite(value) for value in fixed):
        raise RuntimeError("BRAINIAC_V2_RIGID_FIXED_PARAMETERS_INVALID")
    canonical.SetParameters(_canonical_rigid_parameters(canonical.GetParameters()))
    return canonical


def _registration(fixed: Any, moving: Any, sitk: Any) -> tuple[Any, Any, float]:
    initial = sitk.CenteredTransformInitializer(
        fixed,
        moving,
        sitk.Euler3DTransform(),
        sitk.CenteredTransformInitializerFilter.GEOMETRY,
    )
    method = sitk.ImageRegistrationMethod()
    method.SetNumberOfThreads(1)
    method.SetNumberOfWorkUnits(1)
    method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    method.SetMetricSamplingStrategy(method.NONE)
    method.SetInterpolator(sitk.sitkLinear)
    method.SetOptimizerAsGradientDescentLineSearch(
        learningRate=0.5,
        numberOfIterations=10,
        convergenceMinimumValue=1e-6,
        convergenceWindowSize=10,
        lineSearchLowerLimit=0.0,
        lineSearchUpperLimit=2.0,
        lineSearchEpsilon=0.02,
        lineSearchMaximumIterations=5,
    )
    method.SetOptimizerScalesFromPhysicalShift()
    method.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
    method.SetSmoothingSigmasPerLevel(smoothingSigmas=[2, 1, 0])
    method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    method.SetInitialTransform(initial, inPlace=False)
    transform = _canonicalize_rigid_transform(method.Execute(fixed, moving), sitk)
    metric = _canonical_registration_metric(float(method.GetMetricValue()))
    return transform, initial, metric


def _metric_value(fixed: Any, moving: Any, sitk: Any) -> float:
    method = sitk.ImageRegistrationMethod()
    method.SetNumberOfThreads(1)
    method.SetNumberOfWorkUnits(1)
    method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    method.SetMetricSamplingStrategy(method.NONE)
    method.SetInterpolator(sitk.sitkLinear)
    return _canonical_registration_metric(float(method.MetricEvaluate(fixed, moving)))


def _transform_qc(transform: Any, sitk: Any, fixed_template: Any) -> dict[str, float]:
    import numpy as np

    size = fixed_template.GetSize()
    center = np.asarray(
        fixed_template.TransformContinuousIndexToPhysicalPoint(
            tuple((s - 1) / 2.0 for s in size)
        ),
        dtype=np.float64,
    )
    mapped_center = np.asarray(transform.TransformPoint(tuple(center)), dtype=np.float64)
    translation_mm = float(np.linalg.norm(mapped_center - center))
    columns = []
    for axis in range(3):
        probe = center.copy()
        probe[axis] += 1.0
        columns.append(
            np.asarray(transform.TransformPoint(tuple(probe)), dtype=np.float64)
            - mapped_center
        )
    jacobian = np.stack(columns, axis=1)
    determinant = float(np.linalg.det(jacobian))
    orthogonality = float(np.max(np.abs(jacobian.T @ jacobian - np.eye(3))))
    trace = float(
        np.clip(
            (jacobian[0, 0] + jacobian[1, 1] + jacobian[2, 2] - 1.0) / 2.0, -1.0, 1.0
        )
    )
    rotation_deg = float(math.degrees(math.acos(trace)))
    return {
        "rigid_determinant": determinant,
        "orthogonality_error": orthogonality,
        "rotation_deg": rotation_deg,
        "translation_norm_mm": translation_mm,
    }


def _image_geometry(image: Any) -> dict[str, list[float]]:
    import numpy as np

    spacing = np.asarray(image.GetSpacing(), dtype=np.float64)
    origin = np.asarray(image.GetOrigin(), dtype=np.float64)
    direction = np.asarray(image.GetDirection(), dtype=np.float64).reshape(3, 3)
    matrix = direction @ np.diag(spacing)
    affine = np.eye(4, dtype=np.float64)
    affine[:3, :3] = matrix
    affine[:3, 3] = origin
    if (
        not np.isfinite(affine).all()
        or np.any(spacing <= 0)
        or abs(float(np.linalg.det(matrix))) <= 1e-12
    ):
        raise RuntimeError("BRAINIAC_V2_IMAGE_GEOMETRY_INVALID")
    return {
        "affine_4x4": affine.reshape(-1).tolist(),
        "spacing": spacing.tolist(),
        "origin": origin.tolist(),
        "direction": direction.reshape(-1).tolist(),
    }


def _foreground_com_physical(image: Any) -> list[float]:
    from scipy.ndimage import center_of_mass
    import SimpleITK as sitk

    array = sitk.GetArrayFromImage(image)
    mask = array != 0
    if not mask.any():
        raise RuntimeError("BRAINIAC_V2_FOREGROUND_EMPTY")
    zyx = center_of_mass(mask)
    xyz = tuple(float(value) for value in reversed(zyx))
    return [float(value) for value in image.TransformContinuousIndexToPhysicalPoint(xyz)]


def _relative_com(com: Any, template_com: Any) -> list[float]:
    import numpy as np

    delta = np.asarray(com, dtype=np.float64) - np.asarray(template_com, dtype=np.float64)
    if delta.shape != (3,) or not np.isfinite(delta).all():
        raise RuntimeError("BRAINIAC_V2_COM_INVALID")
    return delta.tolist()


_REGISTRATION_LOCK = threading.Lock()
_GPU_LOCK = threading.Lock()


def _serialized_registration_stage(operation: Callable[[], Any]) -> Any:
    """Keep all random-sampled registration/QC in one deterministic lane."""

    with _REGISTRATION_LOCK:
        return operation()


def _run_hd_bet(registered: Path, brain_output: Path, *, device: str) -> None:
    if str(v2.VENDOR_PREPROCESSING) not in sys.path:
        sys.path.insert(0, str(v2.VENDOR_PREPROCESSING))
    if sha256_file(v2.HD_BET_WEIGHT) != v2.EXPECTED_HD_BET_WEIGHT_SHA256:
        raise RuntimeError("BRAINIAC_V2_HD_BET_WEIGHT_HASH_MISMATCH")
    from HD_BET.hd_bet import hd_bet

    hd_bet(
        str(registered),
        str(brain_output),
        mode="fast",
        device="0" if device == "cuda" else "cpu",
        tta=0,
        pp=1,
        save_mask=1,
        overwrite_existing=1,
    )
    mask = brain_output.with_name(brain_output.name[:-7] + "_mask.nii.gz")
    if not brain_output.is_file() or not mask.is_file():
        raise RuntimeError("BRAINIAC_V2_HD_BET_OUTPUT_MISSING")


class ProgressReporter:
    """Counts + short hash prefix + elapsed only; never raw paths/IDs."""

    def __init__(self, root: Path, total: int, started_at: float) -> None:
        self.root = root
        self.total = total
        self.started_at = started_at
        self.since_last = 0
        self._lock = threading.Lock()

    def committed(self) -> int:
        return len(list((self.root / "receipts").glob("*.json")))

    def tick(self, sample_hash: str) -> None:
        with self._lock:
            self.since_last += 1
            if self.since_last >= PROGRESS_EVERY_N:
                self.since_last = 0
                _print_flush(
                    json.dumps(
                        {
                            "progress": "PREPROCESSING",
                            "committed": self.committed(),
                            "total": self.total,
                            "current_sample_prefix": sample_hash[:10],
                            "elapsed_seconds": round(
                                time.monotonic() - self.started_at, 1
                            ),
                        }
                    )
                )


def make_real_pipeline(
    *,
    fixed_template: Any,
    device: str,
    sources: Mapping[str, Path],
    transform_contract: Any,
    extra_bindings: Mapping[str, str],
    progress: ProgressReporter | None = None,
) -> Callable[[Mapping[str, str], Path], dict[str, Any]]:
    """One sample: N4 -> rigid -> HD-BET -> official 96^3 -> nonzero normalize."""

    itk_threads = _pin_itk_threads()

    def pipeline(item: Mapping[str, str], scratch: Path) -> dict[str, Any]:
        import nibabel as nib
        import numpy as np
        import SimpleITK as sitk

        total_started = time.monotonic()
        sample_hash = str(item["sample_hash"])
        source_path = sources[sample_hash]  # in-memory only; never persisted
        source = sitk.ReadImage(str(source_path), sitk.sitkFloat32)
        if source.GetDimension() != 3 or min(source.GetSize()) <= 0:
            raise RuntimeError("BRAINIAC_V2_RAW_T1_GEOMETRY_INVALID")
        n4_started = time.monotonic()
        n4_filter = sitk.N4BiasFieldCorrectionImageFilter()
        n4_filter.SetNumberOfThreads(itk_threads)
        n4_filter.SetNumberOfWorkUnits(itk_threads)
        n4 = n4_filter.Execute(source)
        n4_seconds = time.monotonic() - n4_started

        registration_started = time.monotonic()

        def registration_stage() -> tuple[Any, Any, Any, float, float, list[float], list[float], dict[str, Any], dict[str, Any], dict[str, float]]:
            transform, initial_transform, _metric_after = _registration(
                fixed_template, n4, sitk
            )
            initial_registered = sitk.Resample(
                n4,
                fixed_template,
                initial_transform,
                sitk.sitkLinear,
                0.0,
                sitk.sitkFloat32,
            )
            similarity_before = _metric_value(fixed_template, initial_registered, sitk)
            registered = sitk.Resample(
                n4,
                fixed_template,
                transform,
                sitk.sitkLinear,
                0.0,
                sitk.sitkFloat32,
            )
            similarity_after = _metric_value(fixed_template, registered, sitk)
            template_com = _foreground_com_physical(fixed_template)
            com_before = _relative_com(
                _foreground_com_physical(initial_registered), template_com
            )
            com_after = _relative_com(
                _foreground_com_physical(registered), template_com
            )
            geometry_before = _image_geometry(initial_registered)
            geometry_after = _image_geometry(registered)
            qc_transform = _transform_qc(transform, sitk, fixed_template)
            return (
                transform,
                registered,
                initial_registered,
                similarity_before,
                similarity_after,
                com_before,
                com_after,
                geometry_before,
                geometry_after,
                qc_transform,
            )

        (
            transform,
            registered,
            _initial_registered,
            similarity_before,
            similarity_after,
            com_before,
            com_after,
            geometry_before,
            geometry_after,
            qc_transform,
        ) = _serialized_registration_stage(registration_stage)
        registration_seconds = time.monotonic() - registration_started

        registered_path, brain_path = _scratch_paths(scratch, sample_hash)
        sitk.WriteImage(registered, str(registered_path), False)
        hd_bet_started = time.monotonic()
        with _GPU_LOCK:
            _run_hd_bet(registered_path, brain_path, device=device)
        hd_bet_seconds = time.monotonic() - hd_bet_started

        tensor = transform_contract({"image": str(brain_path)})["image"]
        array = tensor.detach().cpu().numpy().astype("float32", copy=False)
        affine = np.asarray(tensor.affine.detach().cpu(), dtype=np.float64)
        if tuple(array.shape) != (1, 96, 96, 96) or not np.isfinite(array).all():
            raise RuntimeError("BRAINIAC_V2_OFFICIAL_96_TRANSFORM_INVALID")
        nonzero = array[array != 0]
        if nonzero.size == 0:
            raise RuntimeError("BRAINIAC_V2_FOREGROUND_EMPTY")
        foreground_fraction = float(nonzero.size / array.size)
        mean = float(nonzero.mean())
        std = float(nonzero.std())

        output_path = scratch / "output.nii.gz"
        nib.save(nib.Nifti1Image(array[0], affine), str(output_path))
        transform_path = scratch / "transform.tfm"
        sitk.WriteTransform(transform, str(transform_path))
        total_seconds = time.monotonic() - total_started
        if progress is not None:
            progress.tick(sample_hash)
        return {
            "output_path": output_path,
            "transform_path": transform_path,
            "source_sha256": sha256_file(source_path),
            "source_size": source_path.stat().st_size,
            "affine_before": geometry_before["affine_4x4"],
            "affine_after": geometry_after["affine_4x4"],
            "spacing_before": geometry_before["spacing"],
            "spacing_after": geometry_after["spacing"],
            "origin_before": geometry_before["origin"],
            "origin_after": geometry_after["origin"],
            "direction_before": geometry_before["direction"],
            "direction_after": geometry_after["direction"],
            "com_coordinate_frame": "TEMPLATE_PHYSICAL_MM_RELATIVE",
            **qc_transform,
            "foreground_fraction": foreground_fraction,
            "finite": True,
            "normalized_nonzero_mean": mean,
            "normalized_nonzero_std": std,
            "center_of_mass_before": com_before,
            "center_of_mass_after": com_after,
            "registration_similarity_before": similarity_before,
            "registration_similarity_after": similarity_after,
            "runtime_stage_seconds": {
                "n4": n4_seconds,
                "registration_serial": registration_seconds,
                "hd_bet_serial": hd_bet_seconds,
                "total": total_seconds,
            },
        }

    return pipeline


# ---------------------------------------------------------------------------
# modes
# ---------------------------------------------------------------------------


def preflight(*, device: str = "cuda") -> dict[str, Any]:
    """Recompute every preparation authority and fail closed on uncertainty."""

    import shutil

    checks: dict[str, object] = {}
    evidence: dict[str, Any] = {}

    def probe(name: str, operation: Callable[[], Any], predicate: Callable[[Any], bool]) -> Any:
        try:
            value = operation()
            passed = bool(predicate(value))
            checks[name] = True if passed else "FAILED"
            return value
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            checks[name] = "BLOCKED:" + _sanitize_exception_message(exc)
            return None

    _pin_itk_threads()
    config = probe("scientific_config", _config, lambda value: isinstance(value, dict))
    policy = probe("scientific_policy", _policy, lambda value: isinstance(value, dict))

    from src.research.brainiac_execution_gate import assess_brainiac_execution_gate
    from src.research.brainiac_action_guard import inspect_action_exclusion

    action = probe(
        "readiness_anchor_ledger",
        lambda: assess_brainiac_execution_gate("PREPARE_RAW_T1"),
        lambda value: value.get("allowed") is True and value.get("status") == "PASS",
    )
    exclusion = probe(
        "writer_and_process_exclusion",
        inspect_action_exclusion,
        lambda value: value.get("status") == "PASS" and not value.get("blockers"),
    )
    probe("cnn_writer_scan", lambda: (v2.assert_no_cnn_writer(), True)[1], bool)
    runtime = probe(
        "official_runtime_chain",
        _validate_runtime_receipts,
        lambda value: all(len(str(item)) == 64 for item in value.values()),
    )
    templates = probe(
        "template_assets",
        _validate_template_assets,
        lambda value: all(len(str(item)) == 64 for item in value.values()),
    )
    probe("hd_bet_weight", lambda: (_validate_hd_bet_weight(), True)[1], bool)
    transform_binding = probe(
        "official_transform",
        lambda: v2.build_contract_transform()[1],
        lambda value: len(str(value.get("official_transform_source_sha256") or "")) == 64,
    )
    membership = probe(
        "locked_membership",
        v2.load_locked_membership,
        lambda value: len(value) == v2.EXPECTED_REAL_ROWS,
    )
    raw_index = probe("raw_t1_index", v2.build_raw_index, lambda value: len(value) > 0)
    if membership is not None and raw_index is not None:
        resolved = probe(
            "raw_t1_resolution",
            lambda: v2.resolve_sources(membership, index=raw_index),
            lambda value: len(value) == v2.EXPECTED_REAL_ROWS,
        )
    else:
        resolved = None
        checks["raw_t1_resolution"] = "BLOCKED:UPSTREAM_UNKNOWN"

    dependencies = probe(
        "dependency_closure",
        _dependency_manifest,
        lambda value: not verify_manifest_tracking(value),
    )
    untracked_dependencies = (
        verify_manifest_tracking(dependencies) if dependencies is not None else []
    )
    if dependencies is not None and not untracked_dependencies:
        _atomic_write_json(DEPENDENCY_MANIFEST_PATH, dependencies)
    lineage = probe(
        "data_lineage_binding",
        _lineage_binding,
        lambda value: len(str(value.get("cache_sha256") or "")) == 64,
    )
    bindings = probe(
        "complete_receipt_contract",
        _base_bindings,
        lambda value: all(len(str(item)) == 64 for item in value.values()),
    )

    usage = shutil.disk_usage(ROOT)
    free_disk_gb = usage.free / (1024**3)
    checks["storage_free_space"] = True if free_disk_gb >= MIN_FREE_DISK_GB else False
    roots_to_check = [ROOT, REGISTRY_ROOT, v2.PRIMARY_ROOT.parent]
    checks["storage_nonreparse"] = True if all(
        not v2._is_reparse(path) for path in roots_to_check if path.exists()
    ) else False
    local_scratch = probe(
        "local_scratch_storage",
        lambda: v2.inspect_local_scratch_storage(create=False),
        lambda value: value.get("status") == "PASS"
        and int(value.get("used_bytes", -1)) <= v2.LOCAL_SCRATCH_MAX_BYTES
        and int(value.get("free_bytes", -1)) >= v2.LOCAL_SCRATCH_MIN_FREE_BYTES,
    )

    if device == "cuda":
        try:
            import torch

            cuda_ok = torch.cuda.is_available() and torch.cuda.device_count() >= 1
            checks["cuda_runtime"] = True if cuda_ok else False
            evidence["cuda_device_count"] = int(torch.cuda.device_count())
            if cuda_ok:
                total_vram_gb = float(
                    torch.cuda.get_device_properties(0).total_memory / (1024**3)
                )
                checks["cuda_total_vram"] = (
                    True if total_vram_gb >= MIN_CUDA_TOTAL_VRAM_GB else False
                )
                evidence["cuda_total_vram_gb"] = round(total_vram_gb, 3)
                evidence["cuda_minimum_total_vram_gb"] = MIN_CUDA_TOTAL_VRAM_GB
            else:
                checks["cuda_total_vram"] = "BLOCKED:CUDA_UNAVAILABLE"
        except Exception as exc:  # CUDA import/driver failures are runtime blockers.
            checks["cuda_runtime"] = "BLOCKED:" + _sanitize_exception_message(exc)
            checks["cuda_total_vram"] = "BLOCKED:CUDA_RUNTIME_UNKNOWN"
    else:
        checks["cuda_runtime"] = "BLOCKED:CUDA_REQUIRED_FOR_PRODUCTION_PREPROCESSING"
        checks["cuda_total_vram"] = "BLOCKED:CUDA_REQUIRED_FOR_PRODUCTION_PREPROCESSING"

    try:
        source_isolation = verify_frozen_method_x_source_contract(
            expected_digest=EXPECTED_METHOD_X_SOURCE_DIGEST
        )
        live_source_digest = str(source_isolation["source_contract_digest"])
        checks["method_x_source_contract"] = True
    except (ImportError, OSError, RuntimeError, TypeError, ValueError) as exc:
        source_isolation = None
        live_source_digest = "UNKNOWN"
        checks["method_x_source_contract"] = "BLOCKED:" + _sanitize_exception_message(exc)
    live_battery_sha = sha256_file(BATTERY_MANIFEST_PATH)
    checks["battery_manifest_immutable"] = (
        True if live_battery_sha == EXPECTED_BATTERY_MANIFEST_SHA256 else False
    )

    sample_locks = _inspect_primary_sample_locks()
    checks["primary_sample_locks_absent"] = True if not sample_locks else (
        "BLOCKED:PRIMARY_SAMPLE_LOCKS_PRESENT"
    )
    calibration_sample_locks = _inspect_sample_locks(CALIBRATION_ROOT)
    checks["calibration_sample_locks_absent"] = (
        True
        if not calibration_sample_locks
        else "BLOCKED:CALIBRATION_SAMPLE_LOCKS_PRESENT"
    )
    legacy_artifacts = _legacy_incompatible_artifacts()
    checks["legacy_incompatible_artifacts_absent"] = (
        True if not legacy_artifacts else "BLOCKED:LEGACY_GENERATION_REQUIRES_ARCHIVE"
    )
    decision = build_preflight_decision(checks)
    calibration = (
        v2.select_calibration_samples([m["sample_hash"] for m in membership])
        if membership is not None
        else []
    )
    sensitivity = (
        v2.select_sensitivity_samples([m["sample_hash"] for m in membership])
        if membership is not None
        else []
    )
    body: dict[str, Any] = {
        "schema_version": 3,
        "status": decision["status"],
        "blocking_checks": decision["blocking_checks"],
        "checks": decision["checks"],
        "created_at_utc": _utc_now(),
        "action": "PREPARE_RAW_T1",
        "expected_samples": len(membership or ()),
        "membership_sha256": canonical_sha(membership) if membership is not None else None,
        "raw_index_size": len(raw_index or ()),
        "resolved_source_count": len(resolved or ()),
        "free_disk_gb": round(free_disk_gb, 1),
        "free_disk_ok": free_disk_gb >= MIN_FREE_DISK_GB,
        "local_scratch_storage": local_scratch,
        "threshold_manifest_present": THRESHOLD_MANIFEST_PATH.is_file(),
        "calibration_selection_rule": f"{len(calibration)}_lowest_sha256_of_sample_hash",
        "sensitivity_selection_rule": f"{len(sensitivity)}_lowest_sample_hash",
        "policy_sha256": v2.policy_sha256() if policy is not None else None,
        "config_sha256": sha256_file(CONFIG_PATH) if config is not None else None,
        "dependency_closure_sha256": (
            dependencies.get("closure_sha256") if dependencies is not None else None
        ),
        "dependency_manifest_path": (
            DEPENDENCY_MANIFEST_PATH.relative_to(ROOT).as_posix()
            if dependencies is not None and not untracked_dependencies
            else None
        ),
        "dependency_untracked_count": len(untracked_dependencies),
        "official_transform": transform_binding,
        "readiness_gate": action,
        "writer_exclusion": exclusion,
        "runtime_binding": runtime,
        "template_binding": templates,
        "lineage_binding": lineage,
        "receipt_contract_complete": bindings is not None,
        "method_x_source_digest": live_source_digest,
        "method_x_source_isolation": source_isolation,
        "battery_manifest_sha256": live_battery_sha,
        "runtime_evidence": evidence,
        "primary_sample_locks": sample_locks,
        "calibration_sample_locks": calibration_sample_locks,
        "legacy_incompatible_artifacts": [
            path.relative_to(ROOT).as_posix() for path in legacy_artifacts
        ],
        "raw_paths_persisted": False,
        "labels_read": False,
        "pet_read": False,
        "sample_outputs_written": False,
        "claim_ceiling": "POST_HOC_INTERNAL_TECHNICAL_EVIDENCE",
    }
    body["integrity_sha256"] = canonical_sha(
        {k: v for k, v in body.items() if k != "integrity_sha256"}
    )
    _atomic_write_json(PREFLIGHT_REPORT, body)
    return body


def _calibrate_if_needed(
    *, device: str, stop: _StopFlag | None = None
) -> dict[str, Any]:
    """Plan 003 Step 1: threshold manifest built exactly once, then reused."""

    stop = stop or _StopFlag()
    if stop.requested:
        return {"status": "PAUSED_RESUMABLE", "exit_code": 75}
    membership = v2.load_locked_membership()
    selected = v2.select_calibration_samples([m["sample_hash"] for m in membership])
    items = [m for m in membership if m["sample_hash"] in set(selected)]
    current_bindings = _base_bindings()
    evidence_bindings = (
        _threshold_evidence_bindings()
        if THRESHOLD_MANIFEST_PATH.is_file()
        else current_bindings
    )
    threshold_contract = _threshold_contract_bindings(evidence_bindings)
    calibration_bindings = dict(evidence_bindings)
    calibration_bindings["qc_threshold_manifest_sha256"] = hashlib.sha256(
        b"NOT_YET_CALIBRATED_PLAN_003"
    ).hexdigest()
    if THRESHOLD_MANIFEST_PATH.is_file():
        manifest = json.loads(THRESHOLD_MANIFEST_PATH.read_text(encoding="utf-8"))
        receipt_hashes: dict[str, str] = {}
        for sample_hash in selected:
            receipt = v2.load_committed_receipt(
                sample_hash,
                CALIBRATION_ROOT,
                expected_bindings=calibration_bindings,
            )
            receipt_hashes[sample_hash] = str(receipt["integrity_sha256"])
        try:
            validate_threshold_manifest(
                manifest,
                expected_sample_hashes=selected,
                expected_receipt_integrity_sha256s=receipt_hashes,
                expected_contract_bindings=threshold_contract,
            )
        except RuntimeError as exc:
            raise RuntimeError(
                "BRAINIAC_V2_EXISTING_THRESHOLD_MANIFEST_INVALID_NO_OVERWRITE"
            ) from exc
        return {
            "status": "ALREADY_COMPLETED",
            "state": "LOCKED",
            "threshold_manifest_sha256": sha256_file(THRESHOLD_MANIFEST_PATH),
            "selected_n": manifest.get("sample_count"),
            "selection_label_blind": True,
            "rule": "MEDIAN_PLUS_MINUS_6_MAD",
            "all_thresholds_finite": True,
        }
    fixed = _load_fixed_template()
    bindings = calibration_bindings
    transform_contract, _ = v2.build_contract_transform()
    sources = v2.resolve_sources(items, index=v2.build_raw_index())
    pipeline = make_real_pipeline(
        fixed_template=fixed,
        device=device,
        sources=sources,
        transform_contract=transform_contract,
        extra_bindings=bindings,
        progress=ProgressReporter(CALIBRATION_ROOT, len(items), time.monotonic()),
    )
    report = v2.execute(
        mode="smoke",
        limit=len(items),
        root=CALIBRATION_ROOT,
        device=device,
        pipeline_factory=lambda: pipeline,
        bindings=bindings,
        membership=items,
        stop_requested=lambda: stop.requested,
    )
    if report["status"] == "PAUSED_RESUMABLE":
        return report
    if report["status"] != "COMPLETED":
        return report
    metrics: dict[str, dict[str, float]] = {}
    for receipt in report["receipts"]:
        metrics[receipt["sample_hash"]] = {
            "foreground_fraction": float(receipt["foreground_fraction"]),
            "normalized_nonzero_std": float(receipt["normalized_nonzero_std"]),
            "rotation_deg": float(receipt["rotation_deg"]),
            "translation_norm_mm": float(receipt["translation_norm_mm"]),
        }
    manifest = v2.build_threshold_manifest(
        metrics,
        receipt_integrity_sha256s={
            str(receipt["sample_hash"]): str(receipt["integrity_sha256"])
            for receipt in report["receipts"]
        },
        contract_bindings=threshold_contract,
        fixed_bounds={
            "foreground_fraction": FIXED_QC_LIMITS["foreground_fraction"],
            "normalized_nonzero_std": FIXED_QC_LIMITS["normalized_nonzero_std"],
        },
    )
    _atomic_write_json(THRESHOLD_MANIFEST_PATH, manifest)
    return {
        "status": "PASS",
        "state": "LOCKED",
        "selected_n": len(items),
        "selection_label_blind": True,
        "rule": "MEDIAN_PLUS_MINUS_6_MAD",
        "all_thresholds_finite": True,
        "threshold_manifest_sha256": sha256_file(THRESHOLD_MANIFEST_PATH),
    }


def _streaming_validate_primary() -> dict[str, Any]:
    """Plan 003 Step 6: rehash every receipt and live byte; trust no counts."""

    root = v2.PRIMARY_ROOT
    membership = v2.load_locked_membership()
    expected = {m["sample_hash"] for m in membership}
    receipts_dir = root / "receipts"
    seen: dict[str, str] = {}
    invalid: list[str] = []
    expected_bindings = _base_bindings()
    for path in sorted(receipts_dir.glob("*.json")):
        try:
            receipt = json.loads(path.read_text(encoding="utf-8"))
            validate_receipt(
                receipt,
                artifact_root=root,
                expected_bindings=expected_bindings,
            )
            if receipt["sample_hash"] not in expected:
                raise RuntimeError("BRAINIAC_V2_RECEIPT_OUTSIDE_MEMBERSHIP")
            seen[receipt["sample_hash"]] = receipt["integrity_sha256"]
        except (RuntimeError, json.JSONDecodeError, OSError, ValueError):
            invalid.append(path.stem[:10])
    quarantined = (
        len(list((root / "quarantine").glob("*"))) if (root / "quarantine").exists() else 0
    )
    missing = sorted(expected - set(seen))
    receipts = [
        json.loads((receipts_dir / f"{h}.json").read_text(encoding="utf-8"))
        for h in sorted(seen)
    ]
    complete = (
        not missing and not invalid and quarantined == 0 and len(seen) == len(expected)
    )
    aggregate = (
        build_aggregate(
            receipts=receipts,
            expected_sample_hashes=sorted(expected),
            artifact_root=root,
        )
        if complete
        else None
    )
    return {
        "schema_version": 2,
        "status": "COMPLETED" if complete else "BLOCKED",
        "expected_n": len(expected),
        "valid_committed_n": len(seen),
        "unique_sample_hash_n": len(seen),
        "missing_n": len(missing),
        "invalid_n": len(invalid),
        "quarantined_n": quarantined,
        "label_read_count": 0,
        "pet_read_count": 0,
        "raw_path_persisted_count": 0,
        "output_shape": [96, 96, 96],
        "all_finite": True,
        "aggregate": aggregate,
        "aggregate_integrity_sha256": (
            aggregate["integrity_sha256"] if aggregate is not None else None
        ),
    }


def run_primary(*, device: str, stop: _StopFlag | None = None) -> dict[str, Any]:
    """Plan 003 Step 3: the single resumable full-1669 command."""

    stop = stop or _StopFlag()
    sequential_only, profile_valid = _parallel_unlock_state()
    if not profile_valid:
        return {
            "status": "BLOCKED",
            "reason_code": "BRAINIAC_V2_EXECUTION_PROFILE_REQUIRED",
        }
    membership = v2.load_locked_membership()
    expected = [m["sample_hash"] for m in membership]
    committed = len(list((v2.PRIMARY_ROOT / "receipts").glob("*.json")))
    if committed == len(expected):
        validation = _streaming_validate_primary()
        if validation["status"] == "COMPLETED":
            _atomic_write_json(PRIMARY_AGGREGATE_PATH, validation["aggregate"])
            return {
                "status": "ALREADY_COMPLETED",
                "valid_committed_n": validation["valid_committed_n"],
                "zero_writes": True,
                "aggregate_integrity_sha256": validation["aggregate_integrity_sha256"],
            }
        return {
            "status": "BLOCKED",
            "reason_code": "BRAINIAC_V2_AGGREGATE_INVALID",
            "detail": {
                k: validation[k] for k in ("missing_n", "invalid_n", "quarantined_n")
            },
        }
    if not THRESHOLD_MANIFEST_PATH.is_file():
        return {
            "status": "BLOCKED",
            "reason_code": "BRAINIAC_V2_THRESHOLD_MANIFEST_REQUIRED_BEFORE_RUN",
        }
    # Primary receipts must bind the locked threshold file itself.  Only the
    # pre-calibration equivalence/calibration evidence uses the sentinel.
    bindings = _base_bindings()
    threshold_manifest = _load_bound_threshold_manifest(bindings)
    fixed = _load_fixed_template()
    transform_contract, _ = v2.build_contract_transform()
    sources = v2.resolve_sources(membership, index=v2.build_raw_index())
    progress = ProgressReporter(v2.PRIMARY_ROOT, len(expected), time.monotonic())
    pipeline = make_real_pipeline(
        fixed_template=fixed,
        device=device,
        sources=sources,
        transform_contract=transform_contract,
        extra_bindings=bindings,
        progress=progress,
    )
    started = time.monotonic()
    report = v2.execute(
        mode="run",
        limit=0,
        root=v2.PRIMARY_ROOT,
        device=device,
        pipeline_factory=lambda: pipeline,
        bindings=bindings,
        membership=membership,
        stop_requested=(lambda: stop.requested) if stop is not None else None,
        threshold_manifest=threshold_manifest,
        parallel=not sequential_only,
    )
    elapsed = round(time.monotonic() - started, 1)
    if report["status"] == "PAUSED_RESUMABLE":
        _atomic_write_json(
            RUN_STATE_PATH,
            {
                "status": "PAUSED_RESUMABLE",
                "signal": stop.reason,
                "completed": report["completed"],
                "selected": report["selected"],
                "elapsed_seconds": elapsed,
                "resume_semantics": "RERUN_SAME_COMMAND_SKIPS_COMMITTED",
                "updated_at_utc": _utc_now(),
            },
        )
        return {
            "status": "PAUSED_RESUMABLE",
            "exit_code": 75,
            "completed": report["completed"],
            "selected": report["selected"],
            "signal": stop.reason,
        }
    if report["status"] == "BLOCKED_BY_QUARANTINE":
        return {
            "status": "BLOCKED",
            "reason_code": "BRAINIAC_V2_QUARANTINE_BLOCKS_AGGREGATE",
            "quarantined": report["quarantined"],
        }
    if report["status"] != "COMPLETED":
        return {"status": "BLOCKED", "detail": str(report.get("status"))}
    validation = _streaming_validate_primary()
    if validation["status"] != "COMPLETED":
        return {
            "status": "BLOCKED",
            "reason_code": "BRAINIAC_V2_AGGREGATE_INVALID",
            "detail": {
                k: validation[k] for k in ("missing_n", "invalid_n", "quarantined_n")
            },
        }
    _atomic_write_json(PRIMARY_AGGREGATE_PATH, validation["aggregate"])
    return {
        "status": "COMPLETED",
        "valid_committed_n": validation["valid_committed_n"],
        "aggregate_integrity_sha256": validation["aggregate_integrity_sha256"],
        "elapsed_seconds": elapsed,
    }


def _embed_96_array(array: Any, *, device: str, encoder: Any):
    import numpy as np
    import torch

    tensor = torch.from_numpy(np.ascontiguousarray(array))[None, None].to(device)
    with torch.inference_mode():
        return encoder(tensor).detach().cpu().numpy().astype("float64")


def _load_direct_cache_128(sample_hashes: list[str]) -> dict[str, Any]:
    """Read only real image/source/status/sample datasets for report-only drift."""

    import h5py

    wanted = set(sample_hashes)
    found: dict[str, Any] = {}
    with h5py.File(v2.CACHE_PATH, "r") as handle:
        for index, (sample, source, status) in enumerate(
            zip(
                handle["sample_hash"][:],
                handle["source"][:],
                handle["status"][:],
                strict=True,
            )
        ):
            decoded = v2._decode(sample)
            if decoded not in wanted:
                continue
            if v2._decode(source) != "real" or v2._decode(status).upper() != "OK":
                raise RuntimeError("BRAINIAC_V2_DIRECT_ROUTE_REAL_ONLY_VIOLATION")
            found[decoded] = handle["image"][index, 0].astype("float32")
    if set(found) != wanted:
        raise RuntimeError("BRAINIAC_V2_DIRECT_ROUTE_SAMPLE_MISSING")
    return found


def _direct_128_to_96(volume: Any) -> Any:
    import numpy as np
    import torch
    import torch.nn.functional as functional

    array = np.asarray(volume, dtype=np.float32)
    if array.shape != (128, 128, 128) or not np.isfinite(array).all():
        raise RuntimeError("BRAINIAC_V2_DIRECT_ROUTE_INPUT_INVALID")
    tensor = torch.from_numpy(np.ascontiguousarray(array))[None, None]
    resized = functional.interpolate(
        tensor,
        size=(96, 96, 96),
        mode="trilinear",
        align_corners=False,
    )[0, 0].numpy()
    return v2.normalize_foreground(resized, resized != 0)


def run_sensitivity(
    *, device: str, stop: _StopFlag | None = None
) -> dict[str, Any]:
    """Plan 003 Step 5: mask stability is a blocker; direct drift is report-only."""

    stop = stop or _StopFlag()
    if stop.requested:
        return {"status": "PAUSED_RESUMABLE", "exit_code": 75}
    import nibabel as nib
    import numpy as np

    expected_bindings = _base_bindings()
    previous = (
        json.loads(SENSITIVITY_REPORT_PATH.read_text(encoding="utf-8"))
        if SENSITIVITY_REPORT_PATH.is_file()
        else None
    )

    membership = v2.load_locked_membership()
    selected = v2.select_sensitivity_samples([m["sample_hash"] for m in membership])
    root = v2.PRIMARY_ROOT
    receipts: dict[str, dict[str, Any]] = {}
    invalid: list[str] = []
    for sample_hash in selected:
        try:
            receipts[sample_hash] = v2.load_committed_receipt(
                sample_hash,
                root,
                expected_bindings=expected_bindings,
            )
        except (RuntimeError, OSError, ValueError, json.JSONDecodeError):
            invalid.append(sample_hash[:10])
    if invalid:
        return {
            "status": "BLOCKED",
            "reason_code": "BRAINIAC_V2_SENSITIVITY_REQUIRES_VALID_PRIMARY",
            "invalid_n": len(invalid),
        }

    receipt_bindings = {
        sample_hash: str(receipt["integrity_sha256"])
        for sample_hash, receipt in sorted(receipts.items())
    }
    if previous is not None:
        body = dict(previous)
        recorded = body.pop("integrity_sha256", None)
        if (
            previous.get("status") == "PASS"
            and recorded == canonical_sha(body)
            and previous.get("contract_bindings") == expected_bindings
            and previous.get("primary_receipt_integrity_sha256s")
            == receipt_bindings
            and previous.get("selected_sample_hashes") == selected
        ):
            return {"status": "ALREADY_COMPLETED"}

    from src.research.brainiac_model_loader import load_brainiac_encoder
    from src.research.brainiac_execution_gate import brainiac_action_lease

    direct_volumes = _load_direct_cache_128(selected)
    cosines: list[float] = []
    direct_cosines: list[float] = []
    with brainiac_action_lease("EXTRACT_FROZEN_BRAINIAC_EMBEDDINGS"):
        encoder = load_brainiac_encoder(device=device)
        for sample_hash in selected:
            if stop.requested:
                return {"status": "PAUSED_RESUMABLE", "exit_code": 75}
            receipt = receipts[sample_hash]
            base = nib.load(str(root / receipt["output_relative_path"])).get_fdata(
                dtype="float32"
            )
            mask = base != 0
            base_embedding = _embed_96_array(base, device=device, encoder=encoder)
            for morph in (v2.erode_mask, v2.dilate_mask):
                perturbed_mask = morph(mask, iterations=1)
                volume = v2.normalize_foreground(
                    np.where(perturbed_mask, base, 0.0), perturbed_mask
                )
                embedding = _embed_96_array(volume, device=device, encoder=encoder)
                cosine = float(
                    np.dot(base_embedding, embedding)
                    / (np.linalg.norm(base_embedding) * np.linalg.norm(embedding) + 1e-12)
                )
                if not math.isfinite(cosine):
                    raise RuntimeError("BRAINIAC_V2_SENSITIVITY_NONFINITE")
                cosines.append(cosine)
            direct = _direct_128_to_96(direct_volumes[sample_hash])
            direct_embedding = _embed_96_array(direct, device=device, encoder=encoder)
            direct_cosine = float(
                np.dot(base_embedding, direct_embedding)
                / (np.linalg.norm(base_embedding) * np.linalg.norm(direct_embedding) + 1e-12)
            )
            if not math.isfinite(direct_cosine):
                raise RuntimeError("BRAINIAC_V2_DIRECT_ROUTE_NONFINITE")
            direct_cosines.append(direct_cosine)
    median_cosine = float(np.median(cosines))
    p05_cosine = float(np.percentile(cosines, 5))
    body: dict[str, Any] = {
        "schema_version": 2,
        "status": "PASS" if (median_cosine >= 0.99 and p05_cosine >= 0.97) else "BLOCKED",
        "reason_code": None if (median_cosine >= 0.99 and p05_cosine >= 0.97)
        else "BRAINIAC_V2_MASK_STABILITY_FAILED",
        "selection_n": len(selected),
        "selection_label_blind": True,
        "mask_median_cosine": median_cosine,
        "mask_p05_cosine": p05_cosine,
        "mask_median_threshold": 0.99,
        "mask_p05_threshold": 0.97,
        "no_wraparound": True,
        "direct_128_to_96_role": "SENSITIVITY_ONLY",
        "direct_route_drift": "REPORT_ONLY_NOT_PRIMARY",
        "direct_route_median_cosine": float(np.median(direct_cosines)),
        "direct_route_p05_cosine": float(np.percentile(direct_cosines, 5)),
        "pipeline_contract_sha256": expected_bindings["pipeline_contract_sha256"],
        "contract_bindings": expected_bindings,
        "selected_sample_hashes": selected,
        "primary_receipt_integrity_sha256s": receipt_bindings,
        "embeddings_from": "primary_schema_v2_outputs_only",
        "labels_read": False,
        "raw_paths_persisted": False,
        "created_at_utc": _utc_now(),
    }
    body["integrity_sha256"] = canonical_sha(
        {k: v for k, v in body.items() if k != "integrity_sha256"}
    )
    _atomic_write_json(SENSITIVITY_REPORT_PATH, body)
    return body


def run_equivalence(
    *,
    limit: int,
    device: str,
    workers: int = 2,
    stop: _StopFlag | None = None,
) -> dict[str, Any]:
    """Plan-002 revision gate: sequential vs deterministic-parallel on N samples.

    Both legs run with BRAINIAC_V2_ITK_THREADS pinned (default 1) so that N4 and
    rigid registration are bit-deterministic; the parallel leg only changes
    SAMPLE-level scheduling (GPU HD-BET stays serialized by _GPU_LOCK).
    PASS is required before policy may unlock parallel_samples for the full run.
    """

    if int(limit) != 12:
        raise RuntimeError("BRAINIAC_EQUIVALENCE_SAMPLE_COUNT_INVALID")
    membership = v2.load_locked_membership()
    selected = set(
        v2.select_calibration_samples(
            [row["sample_hash"] for row in membership], count=12
        )
    )
    items = [row for row in membership if row["sample_hash"] in selected]
    fixed = _load_fixed_template()
    bindings = _equivalence_receipt_bindings()
    transform_contract, _ = v2.build_contract_transform()
    index = v2.build_raw_index()

    def make(root: Path):
        sources = v2.resolve_sources(items, index=index)
        return make_real_pipeline(
            fixed_template=fixed,
            device=device,
            sources=sources,
            transform_contract=transform_contract,
            extra_bindings=bindings,
            progress=ProgressReporter(root, len(items), time.monotonic()),
        )

    stop = stop or _StopFlag()
    contract_sha = canonical_sha(_threshold_contract_bindings(bindings))
    if EXECUTION_PROFILE_PATH.is_file():
        existing = json.loads(EXECUTION_PROFILE_PATH.read_text(encoding="utf-8"))
        return validate_execution_profile(
            existing,
            expected_contract_sha256=contract_sha,
            expected_receipt_bindings=bindings,
        )
    roots = _execution_profile_roots()
    seq_root = roots["sequential"]
    par_roles = ("parallel_1", "parallel_2")
    par_roots = [roots[role] for role in par_roles]
    if stop.requested:
        return {"status": "PAUSED_RESUMABLE", "exit_code": 75, "completed_legs": 0}
    seq_fresh = not seq_root.exists() or not any(seq_root.rglob("*"))
    seq_resume_events = 0 if seq_fresh else 1
    seq_started_utc = _utc_now()
    started = time.monotonic()
    seq_report = v2.execute(
        mode="smoke",
        limit=len(items),
        root=seq_root,
        device=device,
        pipeline_factory=lambda: make(seq_root),
        bindings=bindings,
        membership=items,
        stop_requested=lambda: stop.requested,
    )
    seq_elapsed = time.monotonic() - started
    if seq_report["status"] == "PAUSED_RESUMABLE":
        return {"status": "PAUSED_RESUMABLE", "exit_code": 75, "completed_legs": 0}
    if seq_report["status"] != "COMPLETED":
        body = {"status": "FAIL", "reason": "SEQUENTIAL_LEG_FAILED",
                "detail": str(seq_report.get("status"))}
        body["integrity_sha256"] = canonical_sha(
            {k: v for k, v in body.items() if k != "integrity_sha256"}
        )
        _atomic_write_json(REGISTRY_ROOT / "brainiac_preprocessing_v2_equivalence.json", body)
        return body
    write_execution_leg_timing_receipt(
        seq_root,
        role="sequential",
        sample_hashes=sorted(str(row["sample_hash"]) for row in seq_report["receipts"]),
        contract_sha256=contract_sha,
        started_at_utc=seq_started_utc,
        finished_at_utc=_utc_now(),
        elapsed_seconds=max(seq_elapsed, 1e-9),
        fresh_run=seq_fresh,
        resume_event_count=seq_resume_events,
    )
    if stop.requested:
        return {"status": "PAUSED_RESUMABLE", "exit_code": 75, "completed_legs": 1}
    from src.research.brainiac_execution_gate import brainiac_action_lease

    completed_parallel = 0
    for role, par_root in zip(par_roles, par_roots, strict=True):
        if stop.requested:
            return {
                "status": "PAUSED_RESUMABLE",
                "exit_code": 75,
                "completed_legs": 1 + completed_parallel,
            }
        par_fresh = not par_root.exists() or not any(par_root.rglob("*"))
        par_resume_events = 0 if par_fresh else 1
        par_started_utc = _utc_now()
        started = time.monotonic()
        with brainiac_action_lease("PREPARE_RAW_T1"):
            par_report = v2.run_orchestrated_batch(
                items,
                par_root,
                pipeline_factory=lambda root=par_root: make(root),
                bindings=bindings,
                cpu_workers=min(2, max(1, int(workers))),
                gpu_workers=1,
                stop_requested=lambda: stop.requested,
                scratch_base=v2.LOCAL_SCRATCH_ROOT,
            )
        par_elapsed = time.monotonic() - started
        if par_report["status"] == "PAUSED_RESUMABLE":
            return {
                "status": "PAUSED_RESUMABLE",
                "exit_code": 75,
                "completed_legs": 1 + completed_parallel,
            }
        if par_report["status"] != "COMPLETED":
            return {
                "status": "BLOCKED",
                "reason_code": "BRAINIAC_EQUIVALENCE_PARALLEL_LEG_FAILED",
                "role": role,
            }
        write_execution_leg_timing_receipt(
            par_root,
            role=role,
            sample_hashes=sorted(
                str(row["sample_hash"]) for row in par_report["receipts"]
            ),
            contract_sha256=contract_sha,
            started_at_utc=par_started_utc,
            finished_at_utc=_utc_now(),
            elapsed_seconds=max(par_elapsed, 1e-9),
            fresh_run=par_fresh,
            resume_event_count=par_resume_events,
        )
        completed_parallel += 1

    recomputed = {
        role: recompute_execution_leg(
            role=role,
            expected_contract_sha256=contract_sha,
            expected_receipt_bindings=bindings,
        )
        for role in ("sequential", "parallel_1", "parallel_2")
    }
    profile = build_execution_profile(
        sequential=recomputed["sequential"],
        parallel_runs=[recomputed["parallel_1"], recomputed["parallel_2"]],
        expected_contract_sha256=contract_sha,
    )
    profile["created_at_utc"] = _utc_now()
    profile["claim_ceiling"] = "POST_HOC_INTERNAL_TECHNICAL_EVIDENCE"
    profile["integrity_sha256"] = canonical_sha(
        {key: value for key, value in profile.items() if key != "integrity_sha256"}
    )
    if EXECUTION_PROFILE_PATH.exists():
        existing = json.loads(EXECUTION_PROFILE_PATH.read_text(encoding="utf-8"))
        if existing != profile:
            raise RuntimeError("BRAINIAC_EQUIVALENCE_PROFILE_IMMUTABLE_MISMATCH")
    else:
        _atomic_write_json(EXECUTION_PROFILE_PATH, profile)
    _atomic_write_json(
        REGISTRY_ROOT / "brainiac_preprocessing_v2_equivalence.json", profile
    )
    return profile


def _policy_path() -> Path:
    return Path(v2.POLICY_PATH)


def _current_equivalence_contract_sha256() -> str:
    return canonical_sha(
        _threshold_contract_bindings(_equivalence_receipt_bindings())
    )


def _parallel_unlock_state() -> tuple[bool, bool]:
    """Return (sequential_only, valid profile recorded) without policy edits."""

    if not EXECUTION_PROFILE_PATH.is_file():
        return True, False
    try:
        profile = json.loads(EXECUTION_PROFILE_PATH.read_text(encoding="utf-8"))
        checked = validate_execution_profile(
            profile,
            expected_contract_sha256=_current_equivalence_contract_sha256(),
            expected_receipt_bindings=_equivalence_receipt_bindings(),
        )
        return checked["execution_mode"] != "PARALLEL_SAFE", True
    except (OSError, RuntimeError, ValueError, TypeError, json.JSONDecodeError):
        return True, False


def _preflight_and_reconcile(*, device: str) -> tuple[dict[str, Any], int]:
    """Run the mutation-minimal reconciliation shared by acceptance and full."""

    pre = preflight(device=device)
    shared_guard = (pre.get("writer_exclusion") or {}).get("shared_guard") or {}
    if shared_guard.get("liveness") == "DEAD":
        reconciliation = _archive_proven_stale_shared_guard()
        _print_flush(
            json.dumps(
                {
                    "phase": "reconciliation",
                    "status": reconciliation["status"],
                    "source_sha256": reconciliation["source_sha256"],
                }
            )
        )
        pre = preflight(device=device)
    exclusion = pre.get("writer_exclusion") or {}
    writer_clear = (
        exclusion.get("status") == "PASS"
        and not exclusion.get("blockers")
        and pre.get("checks", {}).get("cnn_writer_scan") is True
    )
    reconciled = False
    if writer_clear and pre.get("legacy_incompatible_artifacts"):
        receipt = _archive_legacy_incompatible_artifacts()
        if receipt is not None:
            reconciled = True
            _print_flush(
                json.dumps(
                    {
                        "phase": "reconciliation",
                        "status": receipt["status"],
                        "artifact_count": receipt["artifact_count"],
                    }
                )
            )
    locks = list(pre.get("primary_sample_locks") or ())
    if writer_clear and locks and all(row.get("liveness") == "DEAD" for row in locks):
        archived = _archive_proven_dead_primary_sample_locks()
        reconciled = True
        _print_flush(
            json.dumps(
                {
                    "phase": "reconciliation",
                    "status": "ARCHIVED_PROVEN_DEAD_PRIMARY_SAMPLE_LOCKS",
                    "count": len(archived),
                }
            )
        )
    calibration_locks = list(pre.get("calibration_sample_locks") or ())
    if (
        writer_clear
        and calibration_locks
        and all(row.get("liveness") == "DEAD" for row in calibration_locks)
    ):
        archived = _archive_proven_dead_sample_locks(CALIBRATION_ROOT)
        reconciled = True
        _print_flush(
            json.dumps(
                {
                    "phase": "reconciliation",
                    "status": "ARCHIVED_PROVEN_DEAD_CALIBRATION_SAMPLE_LOCKS",
                    "count": len(archived),
                }
            )
        )
    if reconciled:
        pre = preflight(device=device)
    if pre["status"] != "PASS" or not pre.get("free_disk_ok", False):
        return pre, 3
    return pre, 0


def _equivalence_smoke_evidence(profile: Mapping[str, Any]) -> dict[str, Any]:
    """Cross-bind the validated sequential equivalence leg as the real smoke arm."""

    legs = profile.get("legs")
    sequential = legs.get("sequential") if isinstance(legs, Mapping) else None
    if (
        not isinstance(sequential, Mapping)
        or len(sequential.get("sample_hashes") or ()) != 12
        or sequential.get("live_evidence_status")
        != "PASS_RECOMPUTED_FROM_LIVE_BYTES"
    ):
        raise RuntimeError("BRAINIAC_ACCEPTANCE_SMOKE_CROSS_BIND_INVALID")
    scientific: dict[str, Any] = {
        "schema_version": 3,
        "artifact": "BRAINIAC_REAL_12_SAMPLE_SMOKE_CROSS_BIND",
        "status": "PASS",
        "rows": 12,
        "source_role": "EQUIVALENCE_SEQUENTIAL_LEG",
        "root_relative_path": sequential["root_relative_path"],
        "root_tree_sha256": sequential["root_tree_sha256"],
        "selection_sha256": sequential["selection_sha256"],
        "contract_sha256": profile["contract_sha256"],
        "execution_profile_integrity_sha256": profile["integrity_sha256"],
        "physical_qc_validated_from_live_receipts": True,
        "label_blind": True,
        "claim_ceiling": "POST_HOC_INTERNAL_TECHNICAL_EVIDENCE",
    }
    scientific_sha = canonical_sha(scientific)
    if SMOKE_REPORT.is_file():
        existing = json.loads(SMOKE_REPORT.read_text(encoding="utf-8"))
        existing_body = dict(existing)
        recorded = existing_body.pop("integrity_sha256", None)
        if recorded != canonical_sha(existing_body):
            raise RuntimeError("BRAINIAC_ACCEPTANCE_SMOKE_EXISTING_INVALID")
        existing_scientific = dict(existing_body)
        existing_scientific.pop("created_at_utc", None)
        if canonical_sha(existing_scientific) != scientific_sha:
            raise RuntimeError("BRAINIAC_ACCEPTANCE_SMOKE_IMMUTABLE_MISMATCH")
        return existing
    body = {**scientific, "created_at_utc": _utc_now()}
    body["integrity_sha256"] = canonical_sha(body)
    _atomic_write_json(SMOKE_REPORT, body)
    return body


def _validated_acceptance_test_manifest() -> dict[str, Any]:
    if not ACCEPTANCE_TEST_MANIFEST_PATH.is_file():
        raise RuntimeError("BRAINIAC_ACCEPTANCE_TEST_MANIFEST_MISSING")
    report = json.loads(ACCEPTANCE_TEST_MANIFEST_PATH.read_text(encoding="utf-8"))
    body = dict(report)
    recorded = body.pop("integrity_sha256", None)
    required = report.get("required_checks")
    summary = report.get("summary")
    evidence = report.get("evidence_bindings")
    if (
        report.get("schema_version") != 2
        or report.get("artifact")
        != "BRAINIAC_PREPROCESSING_ACCEPTANCE_TEST_MANIFEST"
        or report.get("status") != "PASS"
        or recorded != canonical_sha(body)
        or not isinstance(required, Mapping)
        or set(map(str, required)) != ACCEPTANCE_REQUIRED_CHECKS
        or any(
            not isinstance(value, Mapping)
            or value.get("status") != "PASS"
            or not isinstance(value.get("nodeids"), list)
            or not value.get("nodeids")
            for value in required.values()
        )
        or not isinstance(summary, Mapping)
        or int(summary.get("passed", -1)) < len(ACCEPTANCE_REQUIRED_CHECKS)
        or int(summary.get("failed", -1)) != 0
        or int(summary.get("errors", -1)) != 0
        or int(report.get("exit_code", -1)) != 0
        or str(report.get("platform") or "").casefold() != "windows"
        or not isinstance(evidence, Mapping)
        or set(evidence) != {"junit", "test_files", "source_files"}
        or not isinstance(evidence.get("junit"), Mapping)
        or not isinstance(evidence.get("test_files"), list)
        or not evidence.get("test_files")
        or not isinstance(evidence.get("source_files"), list)
        or not evidence.get("source_files")
    ):
        raise RuntimeError("BRAINIAC_ACCEPTANCE_TEST_MANIFEST_INVALID")
    _validate_artifact_binding(evidence["junit"])
    for role in ("test_files", "source_files"):
        for binding in evidence[role]:
            if not isinstance(binding, Mapping):
                raise RuntimeError("BRAINIAC_ACCEPTANCE_TEST_MANIFEST_INVALID")
            _validate_artifact_binding(binding)
    _, head, _ = _git_live_state()
    if report.get("git_head") != head.casefold():
        raise RuntimeError("BRAINIAC_ACCEPTANCE_TEST_MANIFEST_GIT_DRIFT")
    dependencies = _dependency_manifest()
    if report.get("dependency_closure_sha256") != dependencies.get(
        "closure_sha256"
    ):
        raise RuntimeError("BRAINIAC_ACCEPTANCE_TEST_MANIFEST_CLOSURE_DRIFT")
    return report


def _validated_full_test_manifest() -> dict[str, Any]:
    if not FULL_TEST_MANIFEST_PATH.is_file():
        raise RuntimeError("BRAINIAC_FULL_TEST_MANIFEST_MISSING")
    report = json.loads(FULL_TEST_MANIFEST_PATH.read_text(encoding="utf-8"))
    body = dict(report)
    recorded = body.pop("integrity_sha256", None)
    summary = report.get("summary")
    test_bindings = report.get("test_files")
    source_bindings = report.get("source_files")
    if (
        report.get("schema_version") != 1
        or report.get("artifact") != "BRAINIAC_PREPROCESSING_FULL_TEST_MANIFEST"
        or report.get("status") != "PASS"
        or recorded != canonical_sha(body)
        or int(report.get("exit_code", -1)) != 0
        or str(report.get("platform") or "").casefold() != "windows"
        or not isinstance(summary, Mapping)
        or int(summary.get("passed", -1)) <= 0
        or int(summary.get("failed", -1)) != 0
        or int(summary.get("errors", -1)) != 0
        or not isinstance(test_bindings, list)
        or len(test_bindings) < 20
        or not isinstance(source_bindings, list)
        or not source_bindings
    ):
        raise RuntimeError("BRAINIAC_FULL_TEST_MANIFEST_INVALID")
    bound_tests = {
        _validate_artifact_binding(binding)
        for binding in test_bindings
        if isinstance(binding, Mapping)
    }
    expected_tests = set((ROOT / "tests").glob("test_brainiac*.py"))
    expected_tests = {path.resolve(strict=True) for path in expected_tests}
    if bound_tests != expected_tests:
        raise RuntimeError("BRAINIAC_FULL_TEST_FILE_SET_DRIFT")
    for binding in source_bindings:
        if not isinstance(binding, Mapping):
            raise RuntimeError("BRAINIAC_FULL_TEST_MANIFEST_INVALID")
        _validate_artifact_binding(binding)
    _, head, _ = _git_live_state()
    dependency = _dependency_manifest()
    if (
        report.get("git_head") != head.casefold()
        or report.get("dependency_closure_sha256")
        != dependency.get("closure_sha256")
    ):
        raise RuntimeError("BRAINIAC_FULL_TEST_MANIFEST_SOURCE_DRIFT")
    return report


def _validated_traceability_manifest() -> dict[str, Any]:
    if not TRACEABILITY_MANIFEST_PATH.is_file():
        raise RuntimeError("BRAINIAC_TRACEABILITY_MANIFEST_MISSING")
    report = json.loads(TRACEABILITY_MANIFEST_PATH.read_text(encoding="utf-8"))
    body = dict(report)
    recorded = body.pop("integrity_sha256", None)
    facts = report.get("facts")
    bindings = report.get("evidence_bindings")
    documentation = report.get("documentation_bindings")
    preservation = report.get("full_battery_tree_preservation_pre_traceability")
    if (
        report.get("schema_version") != 1
        or report.get("artifact") != "BRAINIAC_PREPROCESSING_TRACEABILITY_MANIFEST"
        or report.get("status") != "PASS"
        or recorded != canonical_sha(body)
        or not isinstance(facts, Mapping)
        or facts.get("r0") != "R0_COMPLETE"
        or facts.get("citation_corroboration")
        != "CITATION_CORROBORATION_PARTIAL"
        or facts.get("novelty") != "NOVELTY_CANDIDATE"
        or facts.get("battery") != "VERIFIED_TERMINAL"
        or facts.get("plan_002") != "ACCEPTED_READY_FOR_PRIMARY"
        or facts.get("plan_003") != "NOT_STARTED_PRIMARY_0_OF_1669"
        or facts.get("plans_005_006_007_allowed") is not False
        or facts.get("claim_ceiling")
        != "POST_HOC_INTERNAL_TECHNICAL_EVIDENCE"
        or not isinstance(bindings, Mapping)
        or not isinstance(documentation, list)
        or len(documentation) != 6
        or not isinstance(preservation, Mapping)
        or preservation.get("status") != "PASS"
        or preservation.get("method_x_source_digest")
        != EXPECTED_METHOD_X_SOURCE_DIGEST
        or preservation.get("battery_manifest_sha256")
        != EXPECTED_BATTERY_MANIFEST_SHA256
        or int(preservation.get("outer_loader_invocation_count", -1)) != 0
        or int(preservation.get("finalized_artifact_rewrite_count", -1)) != 0
    ):
        raise RuntimeError("BRAINIAC_TRACEABILITY_MANIFEST_INVALID")
    expected = {
        "acceptance_candidate": ACCEPTANCE_CANDIDATE_PATH,
        "full_test_manifest": FULL_TEST_MANIFEST_PATH,
        "p08_status": P08_STATUS_PATH,
        "novelty_matrix": NOVELTY_MATRIX_V3_PATH,
        "dependency_manifest": DEPENDENCY_MANIFEST_PATH,
        "source_only_resume_migration": SOURCE_ONLY_RESUME_MIGRATION_PATH,
        "postcalibration_source_migration": POSTCALIBRATION_SOURCE_MIGRATION_PATH,
        "primary_binding_source_compatibility": PRIMARY_BINDING_SOURCE_COMPATIBILITY_PATH,
        "calibration_reconciliation_source_compatibility": CALIBRATION_RECONCILIATION_SOURCE_COMPATIBILITY_PATH,
        "legacy_smoke_reconciliation": LEGACY_SMOKE_RECONCILIATION_PATH,
    }
    if set(bindings) != set(expected):
        raise RuntimeError("BRAINIAC_TRACEABILITY_BINDINGS_INVALID")
    for role, expected_path in expected.items():
        binding = bindings[role]
        if (
            not isinstance(binding, Mapping)
            or _validate_artifact_binding(binding)
            != expected_path.resolve(strict=True)
        ):
            raise RuntimeError("BRAINIAC_TRACEABILITY_BINDINGS_INVALID")
    for binding in documentation:
        if not isinstance(binding, Mapping):
            raise RuntimeError("BRAINIAC_TRACEABILITY_DOCUMENT_BINDING_INVALID")
        _validate_artifact_binding(binding)
    _validated_source_only_resume_migration()
    _validated_postcalibration_source_migration()
    _validated_primary_binding_source_compatibility()
    _validated_calibration_reconciliation_source_compatibility()
    _validated_legacy_smoke_reconciliation()
    return report


def _validated_source_only_resume_migration() -> dict[str, Any]:
    if not SOURCE_ONLY_RESUME_MIGRATION_PATH.is_file():
        raise RuntimeError("BRAINIAC_SOURCE_ONLY_RESUME_MIGRATION_MISSING")
    report = json.loads(
        SOURCE_ONLY_RESUME_MIGRATION_PATH.read_text(encoding="utf-8")
    )
    body = dict(report)
    recorded = body.pop("integrity_sha256", None)
    dependency = _dependency_manifest()
    if (
        report.get("schema_version") != 1
        or report.get("artifact")
        != "BRAINIAC_PLAN017_SOURCE_ONLY_RESUME_MIGRATION_V1"
        or report.get("status")
        != "PASS_SOURCE_ONLY_RUNTIME_RESUME_MIGRATION"
        or recorded != canonical_sha(body)
        or report.get("new_dependency_closure_sha256")
        != "83a2af4cc37e960cb8463e752c2dec673107e54523cf4214c31dd78a5d60c4bf"
        or int(report.get("scientific_input_drift_count", -1)) != 0
        or int(report.get("output_byte_drift_count", -1)) != 0
        or int(report.get("transform_byte_drift_count", -1)) != 0
        or int(report.get("scientific_receipt_field_drift_count", -1)) != 0
        or report.get("preprocessing_reexecuted") is not False
        or int(report.get("outer_loader_invocation_count", -1)) != 0
        or int(report.get("migrated_equivalence_receipts", -1)) != 36
        or int(report.get("migrated_calibration_receipts", -1)) != 4
        or report.get("primary_valid_n") != 0
        or report.get("plans_005_006_007_allowed") is not False
        or report.get("manual_override_allowed") is not False
        or report.get("claim_ceiling")
        != "POST_HOC_INTERNAL_TECHNICAL_EVIDENCE"
    ):
        raise RuntimeError("BRAINIAC_SOURCE_ONLY_RESUME_MIGRATION_INVALID")
    return report


def _validated_postcalibration_source_migration() -> dict[str, Any]:
    if not POSTCALIBRATION_SOURCE_MIGRATION_PATH.is_file():
        raise RuntimeError("BRAINIAC_POSTCALIBRATION_SOURCE_MIGRATION_MISSING")
    report = json.loads(
        POSTCALIBRATION_SOURCE_MIGRATION_PATH.read_text(encoding="utf-8")
    )
    body = dict(report)
    recorded = body.pop("integrity_sha256", None)
    parent = report.get("parent_migration")
    if (
        report.get("schema_version") != 1
        or report.get("artifact")
        != "BRAINIAC_PLAN017_POSTCALIBRATION_SOURCE_MIGRATION_V2"
        or report.get("status")
        != "PASS_POSTCALIBRATION_SOURCE_ONLY_MIGRATION"
        or recorded != canonical_sha(body)
        or report.get("old_dependency_closure_sha256")
        != "83a2af4cc37e960cb8463e752c2dec673107e54523cf4214c31dd78a5d60c4bf"
        or report.get("new_dependency_closure_sha256")
        != PLAN017_LOCKED_EVIDENCE_DEPENDENCY_CLOSURE_SHA256
        or not isinstance(parent, Mapping)
        or _validate_artifact_binding(parent)
        != SOURCE_ONLY_RESUME_MIGRATION_PATH.resolve(strict=True)
        or int(report.get("scientific_input_drift_count", -1)) != 0
        or int(report.get("output_byte_drift_count", -1)) != 0
        or int(report.get("transform_byte_drift_count", -1)) != 0
        or int(report.get("scientific_receipt_field_drift_count", -1)) != 0
        or int(report.get("threshold_numeric_drift_count", -1)) != 0
        or report.get("preprocessing_reexecuted") is not False
        or int(report.get("outer_loader_invocation_count", -1)) != 0
        or int(report.get("migrated_equivalence_receipts", -1)) != 36
        or int(report.get("migrated_calibration_receipts", -1)) != 32
        or report.get("threshold_manifest_migrated") is not True
        or report.get("primary_valid_n") != 0
        or report.get("plans_005_006_007_allowed") is not False
        or report.get("manual_override_allowed") is not False
        or report.get("claim_ceiling")
        != "POST_HOC_INTERNAL_TECHNICAL_EVIDENCE"
    ):
        raise RuntimeError("BRAINIAC_POSTCALIBRATION_SOURCE_MIGRATION_INVALID")
    return report


def _validated_primary_binding_source_compatibility() -> dict[str, Any]:
    """Validate the immutable V3 link, independent of later live source drift."""

    if not PRIMARY_BINDING_SOURCE_COMPATIBILITY_PATH.is_file():
        raise RuntimeError("BRAINIAC_PRIMARY_BINDING_SOURCE_COMPATIBILITY_MISSING")
    report = json.loads(
        PRIMARY_BINDING_SOURCE_COMPATIBILITY_PATH.read_text(encoding="utf-8")
    )
    body = dict(report)
    recorded = body.pop("integrity_sha256", None)
    parent = report.get("parent_migration")
    transition = report.get("allowed_source_transition")
    if (
        report.get("schema_version") != 1
        or report.get("artifact")
        != "BRAINIAC_PLAN017_PRIMARY_BINDING_SOURCE_COMPATIBILITY_V3"
        or report.get("status")
        != "PASS_SOURCE_ONLY_PRIMARY_BINDING_COMPATIBILITY"
        or recorded != canonical_sha(body)
        or report.get("old_dependency_closure_sha256")
        != PLAN017_LOCKED_EVIDENCE_DEPENDENCY_CLOSURE_SHA256
        or report.get("new_dependency_closure_sha256")
        != PLAN017_PRIMARY_BINDING_V3_DEPENDENCY_CLOSURE_SHA256
        or not isinstance(parent, Mapping)
        or _validate_artifact_binding(parent)
        != POSTCALIBRATION_SOURCE_MIGRATION_PATH.resolve(strict=True)
        or not isinstance(transition, Mapping)
        or transition.get("path")
        != "scripts/research/prepare_brainiac_raw_t1_v2.py"
        or transition.get("old_sha256")
        != "3084e52fccec3e19bcd77331c324ee0c7ea606f142d1bd5e9f45fc5984b26c63"
        or transition.get("new_sha256")
        != PLAN017_PRIMARY_BINDING_V3_RUNNER_SHA256
        or int(report.get("scientific_preprocessing_function_drift_count", -1))
        != 0
        or int(report.get("output_byte_drift_count", -1)) != 0
        or int(report.get("transform_byte_drift_count", -1)) != 0
        or int(report.get("threshold_numeric_drift_count", -1)) != 0
        or report.get("primary_binding_uses_current_dependency_closure") is not True
        or report.get("primary_binding_uses_locked_threshold_sha256") is not True
        or report.get("historical_evidence_uses_locked_dependency_closure") is not True
        or report.get("preprocessing_reexecuted") is not False
        or int(report.get("outer_loader_invocation_count", -1)) != 0
        or report.get("primary_valid_n") != 0
        or report.get("plans_005_006_007_allowed") is not False
        or report.get("manual_override_allowed") is not False
        or report.get("claim_ceiling")
        != "POST_HOC_INTERNAL_TECHNICAL_EVIDENCE"
    ):
        raise RuntimeError("BRAINIAC_PRIMARY_BINDING_SOURCE_COMPATIBILITY_INVALID")
    return report


def _validated_calibration_reconciliation_source_compatibility() -> dict[str, Any]:
    """Validate the append-only V4 repair of the false calibration archive."""

    if not CALIBRATION_RECONCILIATION_SOURCE_COMPATIBILITY_PATH.is_file():
        raise RuntimeError(
            "BRAINIAC_CALIBRATION_RECONCILIATION_SOURCE_COMPATIBILITY_MISSING"
        )
    report = json.loads(
        CALIBRATION_RECONCILIATION_SOURCE_COMPATIBILITY_PATH.read_text(
            encoding="utf-8"
        )
    )
    body = dict(report)
    recorded = body.pop("integrity_sha256", None)
    parent = report.get("parent_v3")
    reconciliation = report.get("false_reconciliation")
    transition = report.get("allowed_source_transition")
    current_dependency = report.get("current_dependency_manifest")
    archived_dependency = report.get("archived_dependency_manifest")
    live_dependency = _dependency_manifest()
    if (
        report.get("schema_version") != 1
        or report.get("artifact")
        != "BRAINIAC_PLAN017_CALIBRATION_RECONCILIATION_SOURCE_COMPATIBILITY_V4"
        or report.get("status")
        != "PASS_FALSE_CALIBRATION_RECONCILIATION_REPAIRED"
        or recorded != canonical_sha(body)
        or not isinstance(parent, Mapping)
        or _validate_artifact_binding(parent)
        != PRIMARY_BINDING_SOURCE_COMPATIBILITY_PATH.resolve(strict=True)
        or not isinstance(reconciliation, Mapping)
        or _validate_artifact_binding(reconciliation)
        != FALSE_CALIBRATION_RECONCILIATION_PATH.resolve(strict=True)
        or not isinstance(transition, Mapping)
        or transition.get("path")
        != "scripts/research/prepare_brainiac_raw_t1_v2.py"
        or transition.get("old_sha256")
        != PLAN017_PRIMARY_BINDING_V3_RUNNER_SHA256
        or transition.get("new_sha256")
        != sha256_file(ROOT / "scripts/research/prepare_brainiac_raw_t1_v2.py")
        or report.get("old_dependency_closure_sha256")
        != PLAN017_PRIMARY_BINDING_V3_DEPENDENCY_CLOSURE_SHA256
        or report.get("new_dependency_closure_sha256")
        != live_dependency.get("closure_sha256")
        or not isinstance(current_dependency, Mapping)
        or _validate_artifact_binding(current_dependency)
        != DEPENDENCY_MANIFEST_PATH.resolve(strict=True)
        or not isinstance(archived_dependency, Mapping)
        or not _validate_artifact_binding(archived_dependency).is_file()
        or int(report.get("scientific_preprocessing_function_drift_count", -1))
        != 0
        or int(report.get("output_byte_drift_count", -1)) != 0
        or int(report.get("transform_byte_drift_count", -1)) != 0
        or int(report.get("threshold_numeric_drift_count", -1)) != 0
        or report.get("calibration_restored_from_exact_archive") is not True
        or int(report.get("historical_calibration_receipt_count", -1)) != 32
        or report.get("preprocessing_reexecuted") is not False
        or int(report.get("outer_loader_invocation_count", -1)) != 0
        or report.get("primary_valid_n") != 0
        or report.get("plans_005_006_007_allowed") is not False
        or report.get("manual_override_allowed") is not False
        or report.get("claim_ceiling")
        != "POST_HOC_INTERNAL_TECHNICAL_EVIDENCE"
    ):
        raise RuntimeError(
            "BRAINIAC_CALIBRATION_RECONCILIATION_SOURCE_COMPATIBILITY_INVALID"
        )
    _validated_primary_binding_source_compatibility()
    reconciliation_report = json.loads(
        FALSE_CALIBRATION_RECONCILIATION_PATH.read_text(encoding="utf-8")
    )
    reconciliation_body = dict(reconciliation_report)
    reconciliation_integrity = reconciliation_body.pop("integrity_sha256", None)
    rows = reconciliation_report.get("artifacts")
    if (
        reconciliation_report.get("status")
        != "ARCHIVED_INCOMPATIBLE_PREPROCESSING_GENERATION"
        or reconciliation_integrity != canonical_sha(reconciliation_body)
        or not isinstance(rows, list)
        or len(rows) != 1
        or rows[0].get("source_role") != "r1_primary_v2_calibration"
    ):
        raise RuntimeError("BRAINIAC_FALSE_CALIBRATION_RECONCILIATION_INVALID")
    restored_tree = report.get("restored_calibration_tree")
    if (
        not isinstance(restored_tree, Mapping)
        or restored_tree != _tree_evidence(CALIBRATION_ROOT)
        or restored_tree.get("tree_sha256") != rows[0].get("tree_sha256")
        or restored_tree.get("file_count") != rows[0].get("file_count")
        or restored_tree.get("bytes") != rows[0].get("bytes")
    ):
        raise RuntimeError("BRAINIAC_RESTORED_CALIBRATION_TREE_INVALID")
    _load_bound_threshold_manifest(_threshold_evidence_bindings())
    receipts = CALIBRATION_ROOT / "receipts"
    if not receipts.is_dir() or len(list(receipts.glob("*.json"))) != 32:
        raise RuntimeError("BRAINIAC_RESTORED_CALIBRATION_RECEIPTS_INVALID")
    primary_receipts = v2.PRIMARY_ROOT / "receipts"
    if primary_receipts.is_dir() and any(primary_receipts.glob("*.json")):
        raise RuntimeError("BRAINIAC_PRIMARY_STARTED_BEFORE_V4_COMPATIBILITY")
    return report


def _validated_legacy_smoke_reconciliation() -> dict[str, Any]:
    if not LEGACY_SMOKE_RECONCILIATION_PATH.is_file():
        raise RuntimeError("BRAINIAC_LEGACY_SMOKE_RECONCILIATION_MISSING")
    report = json.loads(
        LEGACY_SMOKE_RECONCILIATION_PATH.read_text(encoding="utf-8")
    )
    body = dict(report)
    recorded = body.pop("integrity_sha256", None)
    archived = report.get("archived_smoke")
    reason = report.get("reason_receipt")
    if (
        report.get("schema_version") != 1
        or report.get("artifact")
        != "BRAINIAC_PLAN017_LEGACY_SMOKE_RECONCILIATION"
        or report.get("status") != "PASS"
        or report.get("action")
        != "ARCHIVED_EXACT_LEGACY_SCHEMA2_SMOKE"
        or recorded != canonical_sha(body)
        or report.get("legacy_file_sha256")
        != "c132a7ccdfe88673d592d6f37128133d3a345a5df750bb2c3f64238bd30a01a3"
        or not isinstance(archived, Mapping)
        or not isinstance(reason, Mapping)
        or report.get("preprocessing_reexecuted") is not False
        or int(report.get("outer_loader_invocation_count", -1)) != 0
        or report.get("primary_valid_n") != 0
        or report.get("calibration_valid_n") != 32
        or report.get("plans_005_006_007_allowed") is not False
        or report.get("training_authorization") is not False
        or report.get("manual_override_allowed") is not False
        or report.get("claim_ceiling")
        != "POST_HOC_INTERNAL_TECHNICAL_EVIDENCE"
    ):
        raise RuntimeError("BRAINIAC_LEGACY_SMOKE_RECONCILIATION_INVALID")
    _validate_artifact_binding(archived)
    _validate_artifact_binding(reason)
    return report


def _artifact_binding(path: Path) -> dict[str, Any]:
    resolved = Path(path).resolve(strict=True)
    try:
        relative = resolved.relative_to(ROOT.resolve()).as_posix()
    except ValueError as exc:
        raise RuntimeError("BRAINIAC_ACCEPTANCE_EVIDENCE_OUTSIDE_PROJECT") from exc
    if not resolved.is_file() or v2._is_reparse(resolved):
        raise RuntimeError("BRAINIAC_ACCEPTANCE_EVIDENCE_UNSAFE")
    before = resolved.stat()
    digest = sha256_file(resolved)
    after = resolved.stat()
    if (
        before.st_size != after.st_size
        or before.st_mtime_ns != after.st_mtime_ns
    ):
        raise RuntimeError("BRAINIAC_ACCEPTANCE_EVIDENCE_DRIFT")
    return {"path": relative, "bytes": int(after.st_size), "sha256": digest}


def _validate_artifact_binding(binding: Mapping[str, Any]) -> Path:
    path = (ROOT / str(binding.get("path") or "")).resolve(strict=True)
    try:
        path.relative_to(ROOT.resolve())
    except ValueError as exc:
        raise RuntimeError("BRAINIAC_ACCEPTANCE_EVIDENCE_OUTSIDE_PROJECT") from exc
    if (
        not path.is_file()
        or v2._is_reparse(path)
        or int(binding.get("bytes", -1)) != path.stat().st_size
        or str(binding.get("sha256") or "").casefold() != sha256_file(path)
    ):
        raise RuntimeError("BRAINIAC_ACCEPTANCE_EVIDENCE_BINDING_DRIFT")
    return path


def validate_acceptance_candidate(
    path: Path = ACCEPTANCE_CANDIDATE_PATH,
) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    body = dict(payload)
    recorded = body.pop("integrity_sha256", None)
    if (
        payload.get("schema_version") != 1
        or payload.get("artifact")
        != "BRAINIAC_PREPROCESSING_ACCEPTANCE_CANDIDATE"
        or payload.get("status") != "ACCEPTANCE_CANDIDATE_PASS"
        or payload.get("readiness_candidate") is not True
        or payload.get("completion") is not False
        or payload.get("training_authorization") is not False
        or payload.get("manual_override_allowed") is not False
        or payload.get("human_approval_required") is not False
        or recorded != canonical_sha(body)
    ):
        raise RuntimeError("BRAINIAC_ACCEPTANCE_CANDIDATE_INVALID")
    bindings = payload.get("evidence_bindings")
    if not isinstance(bindings, Mapping) or set(bindings) != {
        "preflight",
        "dependency_manifest",
        "execution_profile",
        "threshold_manifest",
        "smoke_report",
        "acceptance_test_manifest",
        "pre_repair_freeze",
    }:
        raise RuntimeError("BRAINIAC_ACCEPTANCE_CANDIDATE_BINDINGS_INVALID")
    expected_paths = {
        "preflight": ACCEPTANCE_PREFLIGHT_SNAPSHOT_PATH,
        "dependency_manifest": DEPENDENCY_MANIFEST_PATH,
        "execution_profile": EXECUTION_PROFILE_PATH,
        "threshold_manifest": THRESHOLD_MANIFEST_PATH,
        "smoke_report": SMOKE_REPORT,
        "acceptance_test_manifest": ACCEPTANCE_TEST_MANIFEST_PATH,
        "pre_repair_freeze": PRE_REPAIR_LIVE_FREEZE_PATH,
    }
    for role, binding in bindings.items():
        if not isinstance(binding, Mapping):
            raise RuntimeError("BRAINIAC_ACCEPTANCE_CANDIDATE_BINDINGS_INVALID")
        bound_path = _validate_artifact_binding(binding)
        if bound_path != expected_paths[role].resolve(strict=True):
            raise RuntimeError("BRAINIAC_ACCEPTANCE_CANDIDATE_BINDINGS_INVALID")
    preflight = json.loads(
        ACCEPTANCE_PREFLIGHT_SNAPSHOT_PATH.read_text(encoding="utf-8")
    )
    preflight_body = dict(preflight)
    preflight_integrity = preflight_body.pop("integrity_sha256", None)
    if (
        preflight.get("status") != "PASS"
        or preflight_integrity != canonical_sha(preflight_body)
        or preflight.get("dependency_closure_sha256")
        != _dependency_manifest().get("closure_sha256")
        or int(preflight.get("dependency_untracked_count", -1)) != 0
        or preflight.get("method_x_source_digest")
        != EXPECTED_METHOD_X_SOURCE_DIGEST
        or preflight.get("battery_manifest_sha256")
        != EXPECTED_BATTERY_MANIFEST_SHA256
    ):
        raise RuntimeError("BRAINIAC_ACCEPTANCE_CANDIDATE_PREFLIGHT_DRIFT")
    profile = json.loads(EXECUTION_PROFILE_PATH.read_text(encoding="utf-8"))
    validate_execution_profile(
        profile,
        expected_contract_sha256=_current_equivalence_contract_sha256(),
        expected_receipt_bindings=_equivalence_receipt_bindings(),
    )
    _validated_acceptance_test_manifest()
    branch, head, dirty = _acceptance_scoped_git_state()
    if dirty:
        raise RuntimeError("BRAINIAC_ACCEPTANCE_SCOPED_GIT_DIRTY")
    dirty_encoded = json.dumps(
        sorted(dirty), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    git_state = payload.get("git_state")
    if (
        not isinstance(git_state, Mapping)
        or git_state.get("branch") != branch
        or git_state.get("head") != head.casefold()
        or int(git_state.get("dirty_entry_count", -1)) != len(dirty)
        or git_state.get("dirty_inventory_sha256")
        != hashlib.sha256(dirty_encoded).hexdigest()
    ):
        raise RuntimeError("BRAINIAC_ACCEPTANCE_CANDIDATE_GIT_DRIFT")
    if (
        payload.get("method_x_source_digest") != EXPECTED_METHOD_X_SOURCE_DIGEST
        or payload.get("battery_manifest_sha256")
        != EXPECTED_BATTERY_MANIFEST_SHA256
        or sha256_file(BATTERY_MANIFEST_PATH)
        != EXPECTED_BATTERY_MANIFEST_SHA256
    ):
        raise RuntimeError("BRAINIAC_ACCEPTANCE_CANDIDATE_PRESERVATION_DRIFT")
    verify_frozen_method_x_source_contract(
        expected_digest=EXPECTED_METHOD_X_SOURCE_DIGEST
    )
    return payload


def validate_final_preprocessing_readiness(
    path: Path = READY_REPORT_PATH,
) -> dict[str, Any]:
    """Fail closed until Plan 018 has issued and bound final readiness."""

    if not Path(path).is_file():
        raise RuntimeError("BRAINIAC_FINAL_PREPROCESSING_READINESS_MISSING")
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    body = dict(payload)
    recorded = body.pop("integrity_sha256", None)
    if (
        payload.get("schema_version") != 1
        or payload.get("artifact")
        != "READY_TO_START_OR_RESUME_BRAINIAC_PREPROCESSING"
        or payload.get("status")
        != "READY_TO_START_OR_RESUME_BRAINIAC_PREPROCESSING"
        or payload.get("completion") is not False
        or payload.get("training_authorization") is not False
        or payload.get("plans_005_006_007_allowed") is not False
        or payload.get("human_approval_required") is not False
        or payload.get("manual_override_allowed") is not False
        or payload.get("unknown_is_pass") is not False
        or payload.get("primary_expected_n") != 1669
        or payload.get("claim_ceiling")
        != "POST_HOC_INTERNAL_TECHNICAL_EVIDENCE"
        or payload.get("external_status")
        != "ROBUSTNESS_EVIDENCE_NOT_EXTERNAL_VALIDATION"
        or payload.get("method_x_source_digest")
        != EXPECTED_METHOD_X_SOURCE_DIGEST
        or payload.get("battery_manifest_sha256")
        != EXPECTED_BATTERY_MANIFEST_SHA256
        or recorded != canonical_sha(body)
    ):
        raise RuntimeError("BRAINIAC_FINAL_PREPROCESSING_READINESS_INVALID")
    bindings = payload.get("evidence_bindings")
    expected_paths = {
        "acceptance_candidate": ACCEPTANCE_CANDIDATE_PATH,
        "full_test_manifest": FULL_TEST_MANIFEST_PATH,
        "traceability_manifest": TRACEABILITY_MANIFEST_PATH,
        "p08_status": P08_STATUS_PATH,
        "novelty_matrix": NOVELTY_MATRIX_V3_PATH,
        "dependency_manifest": DEPENDENCY_MANIFEST_PATH,
        "preflight": ACCEPTANCE_PREFLIGHT_SNAPSHOT_PATH,
        "execution_profile": EXECUTION_PROFILE_PATH,
        "threshold_manifest": THRESHOLD_MANIFEST_PATH,
        "smoke_report": SMOKE_REPORT,
        "pre_repair_freeze": PRE_REPAIR_LIVE_FREEZE_PATH,
        "battery_manifest": BATTERY_MANIFEST_PATH,
        "source_only_resume_migration": SOURCE_ONLY_RESUME_MIGRATION_PATH,
        "postcalibration_source_migration": POSTCALIBRATION_SOURCE_MIGRATION_PATH,
        "primary_binding_source_compatibility": PRIMARY_BINDING_SOURCE_COMPATIBILITY_PATH,
        "calibration_reconciliation_source_compatibility": CALIBRATION_RECONCILIATION_SOURCE_COMPATIBILITY_PATH,
        "legacy_smoke_reconciliation": LEGACY_SMOKE_RECONCILIATION_PATH,
    }
    if not isinstance(bindings, Mapping) or set(bindings) != set(expected_paths):
        raise RuntimeError("BRAINIAC_FINAL_PREPROCESSING_READINESS_INVALID")
    for role, expected_path in expected_paths.items():
        binding = bindings[role]
        if (
            not isinstance(binding, Mapping)
            or _validate_artifact_binding(binding)
            != expected_path.resolve(strict=True)
        ):
            raise RuntimeError("BRAINIAC_FINAL_PREPROCESSING_READINESS_BINDING_DRIFT")
    candidate_binding = bindings["acceptance_candidate"]
    if payload.get("acceptance_candidate") != candidate_binding:
        raise RuntimeError("BRAINIAC_FINAL_PREPROCESSING_READINESS_INVALID")
    candidate_path = _validate_artifact_binding(candidate_binding)
    validate_acceptance_candidate(candidate_path)
    _validated_full_test_manifest()
    _validated_traceability_manifest()
    _validated_source_only_resume_migration()
    _validated_postcalibration_source_migration()
    _validated_primary_binding_source_compatibility()
    _validated_calibration_reconciliation_source_compatibility()
    _validated_legacy_smoke_reconciliation()
    for evidence_path, expected_status in (
        (P08_STATUS_PATH, "R0_COMPLETE_PREPARATION_PARTIAL_NONBLOCKING_FOR_MAIN_CNN"),
        (NOVELTY_MATRIX_V3_PATH, "R0_COMPLETE_NOVELTY_CANDIDATE"),
    ):
        report = json.loads(evidence_path.read_text(encoding="utf-8"))
        report_body = dict(report)
        report_integrity = report_body.pop("integrity_sha256", None)
        if (
            report.get("status") != expected_status
            or report_integrity != canonical_sha(report_body)
        ):
            raise RuntimeError("BRAINIAC_FINAL_PREPROCESSING_READINESS_LINKAGE_INVALID")
    source = verify_frozen_method_x_source_contract(
        expected_digest=EXPECTED_METHOD_X_SOURCE_DIGEST
    )
    if (
        source.get("source_contract_digest") != EXPECTED_METHOD_X_SOURCE_DIGEST
        or sha256_file(BATTERY_MANIFEST_PATH) != EXPECTED_BATTERY_MANIFEST_SHA256
    ):
        raise RuntimeError("BRAINIAC_FINAL_PREPROCESSING_READINESS_PRESERVATION_DRIFT")
    freeze = json.loads(PRE_REPAIR_LIVE_FREEZE_PATH.read_text(encoding="utf-8"))
    frozen_inventory = freeze.get("snapshot", {}).get("battery_finalized_inventory")
    live_inventory = build_file_inventory(
        _battery_preservation_paths(), project_root=ROOT
    )
    preservation = payload.get("full_battery_tree_preservation")
    if (
        not isinstance(frozen_inventory, Mapping)
        or live_inventory != frozen_inventory
        or not isinstance(preservation, Mapping)
        or preservation.get("status") != "PASS"
        or preservation.get("battery_inventory_sha256")
        != live_inventory.get("inventory_sha256")
        or int(preservation.get("outer_loader_invocation_count", -1)) != 0
        or int(preservation.get("finalized_artifact_rewrite_count", -1)) != 0
    ):
        raise RuntimeError("BRAINIAC_FINAL_PREPROCESSING_READINESS_PRESERVATION_DRIFT")
    live = run_status()
    if (
        live.get("status") != "PASS"
        or int(live.get("expected_rows", -1)) != 1669
        or int(live.get("completed_rows", -1)) < int(
            payload.get("primary_valid_n_at_issuance", -1)
        )
        or int(live.get("completed_rows", -1)) > 1669
        or live.get("invalid_rows")
    ):
        raise RuntimeError("BRAINIAC_FINAL_PREPROCESSING_READINESS_RESUME_SUBSET_INVALID")
    if v2.PRIMARY_ROOT.exists():
        for candidate in v2.PRIMARY_ROOT.rglob("*"):
            relative = candidate.relative_to(v2.PRIMARY_ROOT).as_posix().casefold()
            if (
                v2._is_reparse(candidate)
                or candidate.name == ".sample.lock"
                or ".tmp" in candidate.name.casefold()
                or (relative.startswith("quarantine/") and candidate.is_file())
            ):
                raise RuntimeError("BRAINIAC_FINAL_PREPROCESSING_RUNTIME_TREE_UNSAFE")
    return payload


def build_preprocessing_readiness_candidate(
    *,
    preflight_report: Mapping[str, Any],
    execution_profile: Mapping[str, Any],
    calibration_report: Mapping[str, Any],
    smoke_report: Mapping[str, Any],
) -> dict[str, Any]:
    """Issue readiness only from complete, current-generation live evidence."""

    if ACCEPTANCE_CANDIDATE_PATH.is_file():
        return validate_acceptance_candidate(ACCEPTANCE_CANDIDATE_PATH)

    blockers: list[str] = []
    if preflight_report.get("status") != "PASS":
        blockers.append("PREFLIGHT_NOT_PASS")
    try:
        checked_profile = validate_execution_profile(
            execution_profile,
            expected_contract_sha256=_current_equivalence_contract_sha256(),
            expected_receipt_bindings=_equivalence_receipt_bindings(),
        )
    except (OSError, RuntimeError, ValueError, TypeError, json.JSONDecodeError):
        checked_profile = {}
        blockers.append("EXECUTION_PROFILE_NOT_LIVE_BYTE_VALID")
    if (
        calibration_report.get("status") not in {"PASS", "ALREADY_COMPLETED"}
        or int(calibration_report.get("selected_n", -1)) != 32
        or not THRESHOLD_MANIFEST_PATH.is_file()
    ):
        blockers.append("CALIBRATION_32_NOT_LOCKED")
    if (
        smoke_report.get("status") != "PASS"
        or int(smoke_report.get("rows", -1)) != 12
        or smoke_report.get("physical_qc_validated_from_live_receipts") is not True
    ):
        blockers.append("REAL_12_SAMPLE_SMOKE_NOT_VALID")
    try:
        tests = _validated_acceptance_test_manifest()
    except (OSError, RuntimeError, ValueError, TypeError, json.JSONDecodeError):
        tests = {}
        blockers.append("ACCEPTANCE_TEST_EVIDENCE_NOT_VALID")
    membership = v2.load_locked_membership()
    expected_bindings = _base_bindings()
    receipt_paths = (
        sorted((v2.PRIMARY_ROOT / "receipts").glob("*.json"))
        if (v2.PRIMARY_ROOT / "receipts").is_dir()
        else []
    )
    if len(receipt_paths) > len(membership):
        blockers.append("PRIMARY_RECEIPT_COUNT_INVALID")
    else:
        for path in receipt_paths:
            try:
                receipt = json.loads(path.read_text(encoding="utf-8"))
                validate_receipt(
                    receipt,
                    artifact_root=v2.PRIMARY_ROOT,
                    expected_bindings=expected_bindings,
                )
            except (OSError, RuntimeError, ValueError, TypeError, json.JSONDecodeError):
                blockers.append("PRIMARY_RESUME_SUBSET_INVALID")
                break
    if v2.PRIMARY_ROOT.exists():
        for path in v2.PRIMARY_ROOT.rglob("*"):
            relative = path.relative_to(v2.PRIMARY_ROOT).as_posix().casefold()
            if (
                v2._is_reparse(path)
                or path.name == ".sample.lock"
                or ".tmp" in path.name.casefold()
                or relative.startswith("quarantine/") and path.is_file()
            ):
                blockers.append("PRIMARY_RUNTIME_TREE_UNSAFE")
                break
    if blockers:
        return {
            "schema_version": 1,
            "status": "BLOCKED",
            "blocking_reasons": sorted(set(blockers)),
            "completion": False,
            "training_authorization": False,
            "claim_ceiling": "POST_HOC_INTERNAL_TECHNICAL_EVIDENCE",
        }
    preflight_snapshot = dict(preflight_report)
    if ACCEPTANCE_PREFLIGHT_SNAPSHOT_PATH.exists():
        existing_preflight = json.loads(
            ACCEPTANCE_PREFLIGHT_SNAPSHOT_PATH.read_text(encoding="utf-8")
        )
        if existing_preflight != preflight_snapshot:
            raise RuntimeError("BRAINIAC_ACCEPTANCE_PREFLIGHT_SNAPSHOT_MISMATCH")
    else:
        _atomic_write_json(ACCEPTANCE_PREFLIGHT_SNAPSHOT_PATH, preflight_snapshot)
    bindings = {
        "preflight": _artifact_binding(ACCEPTANCE_PREFLIGHT_SNAPSHOT_PATH),
        "dependency_manifest": _artifact_binding(DEPENDENCY_MANIFEST_PATH),
        "execution_profile": _artifact_binding(EXECUTION_PROFILE_PATH),
        "threshold_manifest": _artifact_binding(THRESHOLD_MANIFEST_PATH),
        "smoke_report": _artifact_binding(SMOKE_REPORT),
        "acceptance_test_manifest": _artifact_binding(
            ACCEPTANCE_TEST_MANIFEST_PATH
        ),
        "pre_repair_freeze": _artifact_binding(PRE_REPAIR_LIVE_FREEZE_PATH),
    }
    branch, head, dirty = _acceptance_scoped_git_state()
    if dirty:
        raise RuntimeError("BRAINIAC_ACCEPTANCE_SCOPED_GIT_DIRTY")
    dirty_encoded = json.dumps(
        sorted(dirty), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    source = verify_frozen_method_x_source_contract(
        expected_digest=EXPECTED_METHOD_X_SOURCE_DIGEST
    )
    if sha256_file(BATTERY_MANIFEST_PATH) != EXPECTED_BATTERY_MANIFEST_SHA256:
        raise RuntimeError("BRAINIAC_ACCEPTANCE_BATTERY_MANIFEST_DRIFT")
    body: dict[str, Any] = {
        "schema_version": 1,
        "artifact": "BRAINIAC_PREPROCESSING_ACCEPTANCE_CANDIDATE",
        "status": "ACCEPTANCE_CANDIDATE_PASS",
        "created_at_utc": _utc_now(),
        "readiness_candidate": True,
        "completion": False,
        "training_authorization": False,
        "primary_valid_receipts": len(receipt_paths),
        "primary_expected_receipts": len(membership),
        "execution_mode": checked_profile["execution_mode"],
        "evidence_bindings": bindings,
        "test_manifest_integrity_sha256": tests["integrity_sha256"],
        "git_state": {
            "branch": branch,
            "head": head.casefold(),
            "dirty_entry_count": len(dirty),
            "dirty_inventory_sha256": hashlib.sha256(dirty_encoded).hexdigest(),
        },
        "method_x_source_digest": str(source["source_contract_digest"]),
        "battery_manifest_sha256": EXPECTED_BATTERY_MANIFEST_SHA256,
        "plans_005_006_007_allowed": False,
        "human_approval_required": False,
        "manual_override_allowed": False,
        "unknown_is_pass": False,
        "claim_ceiling": "POST_HOC_INTERNAL_TECHNICAL_EVIDENCE",
    }
    body["integrity_sha256"] = canonical_sha(body)
    if ACCEPTANCE_CANDIDATE_PATH.exists():
        existing = json.loads(ACCEPTANCE_CANDIDATE_PATH.read_text(encoding="utf-8"))
        existing_body = dict(existing)
        recorded = existing_body.pop("integrity_sha256", None)
        candidate_body = dict(body)
        candidate_body.pop("integrity_sha256", None)
        existing_semantic = dict(existing_body)
        candidate_semantic = dict(candidate_body)
        existing_semantic.pop("created_at_utc", None)
        candidate_semantic.pop("created_at_utc", None)
        if (
            recorded != canonical_sha(existing_body)
            or existing_semantic != candidate_semantic
        ):
            raise RuntimeError("BRAINIAC_PREPROCESSING_READINESS_IMMUTABLE_MISMATCH")
        return existing
    _atomic_write_json(ACCEPTANCE_CANDIDATE_PATH, body)
    return body


def run_acceptance(
    *, device: str, stop: _StopFlag | None = None
) -> dict[str, Any]:
    """Complete all real acceptance gates and stop before primary processing."""

    stop = stop or _StopFlag()
    pre, code = _preflight_and_reconcile(device=device)
    if code != 0:
        return {"status": "BLOCKED", "exit_code": code, "readiness_valid": False}
    if stop.requested:
        return {"status": "PAUSED_RESUMABLE", "exit_code": 75}
    # Acceptance must prove the complete append-only source/reconciliation
    # chain before it is allowed to reuse calibration or issue a candidate.
    _validated_calibration_reconciliation_source_compatibility()
    locked, passed = _parallel_unlock_state()
    if locked and not passed:
        gate = run_equivalence(limit=12, device=device, stop=stop)
        if gate.get("status") == "PAUSED_RESUMABLE":
            return {**gate, "readiness_valid": False}
        if gate.get("status") != "PASS":
            return {"status": "BLOCKED", "exit_code": 3, "readiness_valid": False}
    profile = json.loads(EXECUTION_PROFILE_PATH.read_text(encoding="utf-8"))
    profile = validate_execution_profile(
        profile,
        expected_contract_sha256=_current_equivalence_contract_sha256(),
        expected_receipt_bindings=_equivalence_receipt_bindings(),
    )
    calibration = _calibrate_if_needed(device=device, stop=stop)
    if calibration.get("status") == "PAUSED_RESUMABLE":
        return {**calibration, "readiness_valid": False}
    if calibration["status"] not in {"PASS", "ALREADY_COMPLETED"}:
        return {"status": "BLOCKED", "exit_code": 3, "readiness_valid": False}
    smoke = _equivalence_smoke_evidence(profile)
    readiness = build_preprocessing_readiness_candidate(
        preflight_report=pre,
        execution_profile=profile,
        calibration_report=calibration,
        smoke_report=smoke,
    )
    ready = readiness.get("status") == "ACCEPTANCE_CANDIDATE_PASS"
    return {
        "status": "PASS" if ready else "BLOCKED",
        "exit_code": 0 if ready else 3,
        "readiness_candidate": ready,
        "readiness_valid": False,
        "candidate": readiness,
    }


def run_full(*, device: str) -> int:
    """Run acceptance first; enter primary only with immutable readiness."""

    stop = _StopFlag()
    _install_signal_handlers(stop)
    acceptance = run_acceptance(device=device, stop=stop)
    _print_flush(
        json.dumps(
            {
                "phase": "acceptance",
                "status": acceptance.get("status"),
                "readiness_candidate": acceptance.get("readiness_candidate", False),
            }
        )
    )
    if acceptance.get("status") == "PAUSED_RESUMABLE":
        return 75
    if acceptance.get("readiness_candidate") is not True:
        return 3
    try:
        validate_final_preprocessing_readiness()
    except (OSError, RuntimeError, ValueError, TypeError, json.JSONDecodeError) as exc:
        _print_flush(
            json.dumps(
                {
                    "phase": "final_readiness",
                    "status": "BLOCKED",
                    "reason_code": _sanitize_exception_message(exc),
                }
            )
        )
        return 3

    primary = run_primary(device=device, stop=stop)
    _print_flush(
        json.dumps(
            {"phase": "primary",
             **{k: primary[k] for k in
                ("status", "completed", "valid_committed_n") if k in primary}},
            default=str,
        )
    )
    code = _exit_code_for(str(primary["status"]))
    if code != 0:
        return code

    sensitivity = run_sensitivity(device=device, stop=stop)
    _print_flush(
        json.dumps(
            {"phase": "sensitivity",
             **{k: sensitivity[k] for k in
                ("status", "mask_median_cosine", "mask_p05_cosine")
                if k in sensitivity}}
        )
    )
    code = _exit_code_for(str(sensitivity["status"]))
    if code != 0:
        return code

    validation = _streaming_validate_primary()
    _print_flush(
        json.dumps(
            {
                "phase": "final_validation",
                "status": validation["status"],
                "valid_committed_n": validation["valid_committed_n"],
                "aggregate_integrity_sha256": validation["aggregate_integrity_sha256"],
            }
        )
    )
    return _exit_code_for(str(validation["status"]))


def run_smoke(
    *, limit: int, device: str, stop: _StopFlag | None = None
) -> dict[str, Any]:
    stop = stop or _StopFlag()
    if stop.requested:
        return {"status": "PAUSED_RESUMABLE", "exit_code": 75}
    membership = v2.load_locked_membership()
    items = membership[: max(1, int(limit))]
    fixed = _load_fixed_template()
    bindings = _base_bindings()
    transform_contract, _ = v2.build_contract_transform()
    index = v2.build_raw_index()
    sources = v2.resolve_sources(items, index=index)
    pipeline = make_real_pipeline(
        fixed_template=fixed,
        device=device,
        sources=sources,
        transform_contract=transform_contract,
        extra_bindings=bindings,
    )
    sequential_root = v2.SMOKE_ROOT / "sequential"
    orchestrated_root = v2.SMOKE_ROOT / "orchestrated"
    sequential = v2.execute(
        mode="smoke",
        limit=limit,
        root=sequential_root,
        device=device,
        pipeline_factory=lambda: pipeline,
        bindings=bindings,
        membership=items,
        stop_requested=lambda: stop.requested,
    )
    if sequential["status"] != "COMPLETED":
        return sequential
    if bool(_policy().get("orchestration", {}).get("sequential_only")):
        body: dict[str, Any] = {
            "schema_version": 2,
            "status": "PASS",
            "mode": "smoke",
            "rows": len(items),
            "sequential_status": sequential["status"],
            "equivalence": "SKIPPED_SEQUENTIAL_ONLY_LOCKED",
            "claim_ceiling": "POST_HOC_INTERNAL_TECHNICAL_EVIDENCE",
            "created_at_utc": _utc_now(),
        }
        body["integrity_sha256"] = canonical_sha(
            {k: v for k, v in body.items() if k != "integrity_sha256"}
        )
        _atomic_write_json(SMOKE_REPORT, body)
        return body
    from src.research.brainiac_execution_gate import brainiac_action_lease

    with brainiac_action_lease("PREPARE_RAW_T1"):
        orch_receipts = v2.run_orchestrated(
            items,
            orchestrated_root,
            pipeline_factory=lambda: pipeline,
            bindings=bindings,
            cpu_workers=2,
            gpu_workers=1,
            stop_requested=lambda: stop.requested,
            scratch_base=v2.LOCAL_SCRATCH_ROOT,
        )
    seq_by_sample = {r["sample_hash"]: r for r in sequential["receipts"]}
    mismatched: list[str] = []
    for receipt in orch_receipts:
        reference = seq_by_sample[receipt["sample_hash"]]
        for field in (
            "output_sha256",
            "transform_sha256",
            "foreground_fraction",
            "rigid_determinant",
            "normalized_nonzero_std",
        ):
            if receipt[field] != reference[field]:
                mismatched.append(f"{receipt['sample_hash']}:{field}")
    body = {
        "schema_version": 2,
        "status": "PASS" if not mismatched else "EQUIVALENCE_FAILED",
        "mode": "smoke",
        "rows": len(items),
        "sequential_status": sequential["status"],
        "equivalence_mismatched": mismatched,
        "claim_ceiling": "POST_HOC_INTERNAL_TECHNICAL_EVIDENCE",
        "created_at_utc": _utc_now(),
    }
    body["integrity_sha256"] = canonical_sha(
        {k: v for k, v in body.items() if k != "integrity_sha256"}
    )
    _atomic_write_json(SMOKE_REPORT, body)
    return body


def run_status() -> dict[str, Any]:
    membership = v2.load_locked_membership()
    root = v2.PRIMARY_ROOT
    expected_bindings = _base_bindings()
    completed = 0
    invalid: list[str] = []
    for member in membership:
        try:
            v2.load_committed_receipt(
                member["sample_hash"], root, expected_bindings=expected_bindings
            )
            completed += 1
        except RuntimeError as exc:
            if "NOT_COMMITTED" not in str(exc):
                invalid.append(member["sample_hash"][:10])
    return {
        "status": "PASS" if not invalid else "BLOCKED",
        "expected_rows": len(membership),
        "completed_rows": completed,
        "invalid_rows": invalid,
        "remaining_rows": len(membership) - completed,
        "threshold_manifest_present": THRESHOLD_MANIFEST_PATH.is_file(),
        "aggregate_present": PRIMARY_AGGREGATE_PATH.is_file(),
        "raw_paths_persisted": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=(
            "preflight",
            "acceptance",
            "calibrate",
            "run",
            "sensitivity",
            "full",
            "smoke",
            "status",
            "equivalence",
        ),
        required=True,
    )
    parser.add_argument("--limit", type=int, default=v2.SMOKE_LIMIT)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()

    try:
        if args.mode == "full":
            return run_full(device=args.device)

        stop = _StopFlag()
        _install_signal_handlers(stop)
        if args.mode == "acceptance":
            result = run_acceptance(device=args.device, stop=stop)
            _print_flush(json.dumps(result, default=str))
            return int(result.get("exit_code", 3))
        if args.mode in {"calibrate", "run", "sensitivity", "smoke", "equivalence"}:
            gate = preflight(device=args.device)
            if gate.get("status") != "PASS":
                _print_flush(
                    json.dumps(
                        {
                            "status": "BLOCKED",
                            "reason_code": "BRAINIAC_V2_PREFLIGHT_REQUIRED",
                            "blocking_checks": gate.get("blocking_checks", []),
                            "exit_code": 3,
                        }
                    )
                )
                return 3
        if args.mode == "equivalence":
            result = run_equivalence(limit=args.limit, device=args.device, stop=stop)
        elif args.mode == "preflight":
            result = preflight(device=args.device)
        elif args.mode == "calibrate":
            result = _calibrate_if_needed(device=args.device, stop=stop)
        elif args.mode == "run":
            result = run_primary(device=args.device, stop=stop)
        elif args.mode == "sensitivity":
            result = run_sensitivity(device=args.device, stop=stop)
        elif args.mode == "smoke":
            result = run_smoke(limit=args.limit, device=args.device, stop=stop)
        else:
            result = run_status()
        _print_flush(json.dumps(result, indent=2, sort_keys=True, default=str))
        return _exit_code_for(str(result.get("status", "BLOCKED")))
    except KeyboardInterrupt:
        _print_flush(json.dumps({"status": "PAUSED_RESUMABLE", "exit_code": 75}))
        return 75
    except Exception as exc:
        reason = _sanitize_exception_message(exc)
        status = "CONTRACT_MISMATCH" if "CONTRACT" in reason else "BLOCKED"
        code = 78 if status == "CONTRACT_MISMATCH" else 3
        _print_flush(json.dumps({"status": status, "reason_code": reason, "exit_code": code}))
        return code


if __name__ == "__main__":
    raise SystemExit(main())
