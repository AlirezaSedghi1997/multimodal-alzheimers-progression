from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import re
import statistics
import stat
import subprocess
import sys
import tempfile
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT_STAGING_ROOT = (
    r"C:\Users\USER\AppData\Local\AMUSE-X\checkpoint_staging"
)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.matched_benchmark import validate_matched_backbone_grid
from src.training.method_x_autosupervisor import (
    REQUIRED_SUPERVISION_LAYERS,
    assert_path_beneath_root_without_reparse,
    discover_method_x_processes,
    enforce_autonomous_supervision,
    evaluate_autonomous_supervision,
    load_supervision_policy,
    resolve_paused_progress_checkpoint,
    run_supervised_process,
    validate_resumable_checkpoint_evidence,
)
from src.training.method_x_spm_snapshot import DEFAULT_SNAPSHOT_REL, verify_spm_gate_snapshot
from src.training.storage_locality import watch_blockers_are_live_inventory_race


CACHE_REL = Path("data/cached/t1_volume_cache_128.h5")
SPLITS_REL = Path("splits")
DEVICE_POLICY_REL = Path("configs/compute/device_policy.yaml")
AUGMENTATION_SUMMARY_REL = Path("results/tables/fold_local_mri_augmentation_summary.json")
CACHE_VERIFICATION_REL = Path("results/tables/final_mri_augmented_cache_verification.json")
# Launch-time checkpoint staging overlay. Set once from the CLI in main() and
# appended only when the child is spawned. It is never stored in the manifest
# command and never enters the retry-authorization digest, so toggling staging
# cannot be mistaken for a scientific change on an in-flight row.
_RUNTIME_STAGING_OVERLAY: list[str] = []

MANIFEST_NAME = "battery_manifest.json"
RECONCILIATION_GUARD_NAME = ".method_x_reconciliation.lock"
COMPARISON_JSON = Path("reports/model_comparison.json")
COMPARISON_MD = Path("reports/model_comparison.md")
LEGACY_UNVERSIONED_OUTPUT_ROOT = Path(
    "results/final_results/method_x_model_battery_v3_depth_and_diversity"
)
ADVANCEABLE_MODEL_STATUSES = frozenset(
    {"COMPLETED", "PROTOCOL_EXCLUDED_NOT_STARTED"}
)
# Backwards-compatible public name; "terminal" now means safe to pass in the
# scientific serial schedule, not merely that a child process stopped.
TERMINAL_MODEL_STATUSES = ADVANCEABLE_MODEL_STATUSES
FIVE_MODEL_AMENDMENT = Path(
    "amendments/method_x_battery_v3_five_model_amendment.v1.json"
)
FIVE_MODEL_ORDER = (
    "resnet10_3d",
    "resnet18_3d",
    "resnet34_3d",
    "densenet121_3d",
    "mednext_3d_s",
)
PROTOCOL_EXCLUDED_BACKBONES = ("resnet50_3d", "efficientnet3d_b0")
MAX_TRANSIENT_IO_RETRIES = 2
ORPHAN_RUNNING_ROW_STATUSES = frozenset(
    {
        "TRAINING_RUNNING",
        "BLOCKED_STALE_RUNNING",
        "BLOCKED_TRAINING_EXECUTION",
    }
)
ORPHAN_RUNNING_TRAINING_STATUSES = frozenset(
    {
        "RUNNING",
        "BLOCKED_STALE_RUNNING",
        "TRAINING_RUNNING",
        "BLOCKED_TRAINING_EXECUTION",
    }
)
LOCKED_FOLD_COUNT = 5


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_file_local(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_payload_sha256(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def archive_superseded_preflight_blocker(
    output_root: Path, *, preflight: dict[str, Any]
) -> dict[str, Any] | None:
    """Move a stale preflight blocker to immutable evidence after a safe preflight."""

    if preflight.get("status") != "PASS" or preflight.get("training_started") is not False:
        raise RuntimeError("PREFLIGHT_BLOCKER_ARCHIVE_SUCCESS_EVIDENCE_INVALID")
    output_root = Path(output_root)
    source = assert_path_beneath_root_without_reparse(
        output_root,
        output_root / "reports/BLOCKED_PREFLIGHT.json",
        outside_error="PREFLIGHT_BLOCKER_ARCHIVE_PATH_ESCAPE",
        reparse_error="PREFLIGHT_BLOCKER_ARCHIVE_REPARSE_REJECTED",
    )
    if not os.path.lexists(source):
        return None
    source = assert_path_beneath_root_without_reparse(
        output_root,
        source,
        outside_error="PREFLIGHT_BLOCKER_ARCHIVE_PATH_ESCAPE",
        reparse_error="PREFLIGHT_BLOCKER_ARCHIVE_REPARSE_REJECTED",
        require_exists=True,
        require_regular_file=True,
    )
    source_metadata = os.lstat(source)
    source_sha256 = _sha256_file_local(source)
    archive_rel = Path("archive/preflight_blockers") / f"{source_sha256}-BLOCKED_PREFLIGHT.json"
    receipt_rel = Path("reports/preflight_blocker_archives") / f"{source_sha256}.json"
    archive = assert_path_beneath_root_without_reparse(
        output_root,
        output_root / archive_rel,
        outside_error="PREFLIGHT_BLOCKER_ARCHIVE_PATH_ESCAPE",
        reparse_error="PREFLIGHT_BLOCKER_ARCHIVE_REPARSE_REJECTED",
    )
    receipt_path = assert_path_beneath_root_without_reparse(
        output_root,
        output_root / receipt_rel,
        outside_error="PREFLIGHT_BLOCKER_ARCHIVE_PATH_ESCAPE",
        reparse_error="PREFLIGHT_BLOCKER_ARCHIVE_REPARSE_REJECTED",
    )
    archive.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    assert_path_beneath_root_without_reparse(
        output_root,
        archive.parent,
        outside_error="PREFLIGHT_BLOCKER_ARCHIVE_PATH_ESCAPE",
        reparse_error="PREFLIGHT_BLOCKER_ARCHIVE_REPARSE_REJECTED",
        require_exists=True,
    )
    assert_path_beneath_root_without_reparse(
        output_root,
        receipt_path.parent,
        outside_error="PREFLIGHT_BLOCKER_ARCHIVE_PATH_ESCAPE",
        reparse_error="PREFLIGHT_BLOCKER_ARCHIVE_REPARSE_REJECTED",
        require_exists=True,
    )
    if os.path.lexists(archive):
        archive = assert_path_beneath_root_without_reparse(
            output_root,
            archive,
            outside_error="PREFLIGHT_BLOCKER_ARCHIVE_PATH_ESCAPE",
            reparse_error="PREFLIGHT_BLOCKER_ARCHIVE_REPARSE_REJECTED",
            require_exists=True,
            require_regular_file=True,
        )
        if (
            _sha256_file_local(archive) != source_sha256
            or archive.stat().st_size != source_metadata.st_size
        ):
            raise RuntimeError("PREFLIGHT_BLOCKER_ARCHIVE_CONFLICT")
    os.replace(source, archive)
    if (
        _sha256_file_local(archive) != source_sha256
        or archive.stat().st_size != source_metadata.st_size
    ):
        raise RuntimeError("PREFLIGHT_BLOCKER_ARCHIVE_EVIDENCE_MISMATCH")
    body = {
        "schema_version": 1,
        "gate": "SUPERSEDED_PREFLIGHT_BLOCKER_ARCHIVE",
        "status": "PASS",
        "original_path": "reports/BLOCKED_PREFLIGHT.json",
        "archive_path": archive_rel.as_posix(),
        "sha256": source_sha256,
        "size_bytes": int(source_metadata.st_size),
        "successful_preflight_sha256": _canonical_payload_sha256(preflight),
        "training_started": False,
        "outer_test_accessed": False,
        "deleted_count": 0,
    }
    receipt = {**body, "integrity_sha256": _canonical_payload_sha256(body)}
    if receipt_path.exists():
        try:
            existing = json.loads(receipt_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise RuntimeError("PREFLIGHT_BLOCKER_ARCHIVE_RECEIPT_INVALID") from exc
        if existing != receipt:
            raise RuntimeError("PREFLIGHT_BLOCKER_ARCHIVE_RECEIPT_CONFLICT")
    else:
        atomic_write_json(receipt_path, receipt)
    return {**receipt, "receipt_path": receipt_rel.as_posix()}


def _pid_is_active_battery(pid: int) -> bool:
    try:
        import psutil

        command = " ".join(psutil.Process(pid).cmdline()).casefold()
        return "run_method_x_claim_benchmark.py" in command
    except ImportError:
        pass
    except Exception:
        return False
    if os.name != "nt":
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False
    query = (
        f"$p=Get-CimInstance Win32_Process -Filter 'ProcessId={int(pid)}' -ErrorAction SilentlyContinue; "
        "if($p){$p.CommandLine}"
    )
    try:
        result = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", query],
            capture_output=True,
            text=True,
            check=False,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return "run_method_x_claim_benchmark.py" in (result.stdout or "").casefold()


def _pid_is_live(pid: int) -> bool | None:
    try:
        import psutil

        process = psutil.Process(pid)
        return process.is_running() and process.status() != psutil.STATUS_ZOMBIE
    except ImportError:
        pass
    except psutil.NoSuchProcess:
        return False
    except Exception:
        return None
    if os.name != "nt":
        try:
            os.kill(pid, 0)
            return True
        except PermissionError:
            return True
        except OSError:
            return False
    query = (
        f"$p=Get-CimInstance Win32_Process -Filter 'ProcessId={int(pid)}' -ErrorAction SilentlyContinue; "
        "if($p){$p.CommandLine}"
    )
    try:
        result = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", query],
            capture_output=True,
            text=True,
            check=False,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return bool((result.stdout or "").strip())


def _pid_create_time(pid: int) -> float | None:
    try:
        import psutil

        return float(psutil.Process(pid).create_time())
    except Exception:
        return None


def _archive_stale_runtime_lock(output_root: Path, path: Path, stem: str) -> Path:
    archive_dir = (
        output_root
        / "archive"
        / "stale_locks"
        / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    )
    archive_dir.mkdir(parents=True, exist_ok=True)
    archived = archive_dir / f"{stem}.{uuid.uuid4().hex}.json"
    os.replace(path, archived)
    return archived


def _runtime_lock_is_regular(path: Path) -> bool:
    try:
        metadata = os.lstat(path)
    except FileNotFoundError:
        return False
    except OSError as exc:
        raise RuntimeError("METHOD_X_RUNTIME_LOCK_INSPECTION_FAILED") from exc
    return stat.S_ISREG(metadata.st_mode) and not bool(
        getattr(metadata, "st_file_attributes", 0) & 0x0400
    )


def assert_no_active_reconciliation(output_root: Path) -> None:
    """Fail closed or archive a readable stale reconciliation guard."""

    path = output_root / RECONCILIATION_GUARD_NAME
    if not os.path.lexists(path):
        return
    if not _runtime_lock_is_regular(path):
        raise RuntimeError("METHOD_X_RECONCILIATION_GUARD_UNSAFE")
    try:
        owner = read_json(path)
        pid = int(owner.get("pid", -1))
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
        raise RuntimeError("METHOD_X_RECONCILIATION_GUARD_UNREADABLE") from exc
    if owner.get("role") != "method_x_reconciliation" or pid <= 0:
        raise RuntimeError("METHOD_X_RECONCILIATION_GUARD_INVALID")
    liveness = _pid_is_live(pid)
    if liveness is None:
        raise RuntimeError(f"METHOD_X_RECONCILIATION_LIVENESS_UNKNOWN:pid={pid}")
    if liveness:
        recorded_start = owner.get("process_create_time")
        current_start = _pid_create_time(pid)
        try:
            same_process_instance = (
                recorded_start is None
                or current_start is None
                or abs(float(recorded_start) - current_start) < 0.01
            )
        except (TypeError, ValueError) as exc:
            raise RuntimeError("METHOD_X_RECONCILIATION_GUARD_INVALID") from exc
        if same_process_instance:
            raise RuntimeError(f"METHOD_X_RECONCILIATION_ACTIVE:pid={pid}")
    try:
        _archive_stale_runtime_lock(output_root, path, "reconciliation.lock")
    except FileNotFoundError:
        # A concurrent owner transition is ambiguous; never continue through it.
        raise RuntimeError("METHOD_X_RECONCILIATION_GUARD_RACE")


def acquire_reconciliation_guard(output_root: Path) -> Path:
    """Atomically exclude battery launches for the whole reconciliation."""

    output_root.mkdir(parents=True, exist_ok=True)
    path = output_root / RECONCILIATION_GUARD_NAME
    for _ in range(2):
        try:
            descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            assert_no_active_reconciliation(output_root)
            continue
        try:
            payload = {
                "pid": os.getpid(),
                "role": "method_x_reconciliation",
                "created_at_utc": utc_now(),
            }
            process_create_time = _pid_create_time(os.getpid())
            if process_create_time is not None:
                payload["process_create_time"] = process_create_time
            os.write(descriptor, json.dumps(payload, sort_keys=True).encode("utf-8"))
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        return path
    raise RuntimeError("METHOD_X_RECONCILIATION_GUARD_ACQUISITION_RACE")


def release_reconciliation_guard(path: Path | None) -> None:
    if path is None or not os.path.lexists(path):
        return
    if not _runtime_lock_is_regular(path):
        return
    try:
        owner = read_json(path)
    except (OSError, RuntimeError, json.JSONDecodeError):
        return
    if (
        int(owner.get("pid", -1)) == os.getpid()
        and owner.get("role") == "method_x_reconciliation"
    ):
        path.unlink(missing_ok=True)


def acquire_battery_lock(output_root: Path) -> Path:
    output_root = assert_path_beneath_root_without_reparse(
        ROOT,
        output_root,
        outside_error="OUTPUT_ROOT_OUTSIDE_H_PROJECT",
        reparse_error="OUTPUT_ROOT_REPARSE_POINT_REJECTED",
    )
    output_root.mkdir(parents=True, exist_ok=True)
    output_root = assert_path_beneath_root_without_reparse(
        ROOT,
        output_root,
        outside_error="OUTPUT_ROOT_OUTSIDE_H_PROJECT",
        reparse_error="OUTPUT_ROOT_REPARSE_POINT_REJECTED",
        require_exists=True,
    )
    assert_no_active_reconciliation(output_root)
    path = output_root / ".battery.lock"
    for _ in range(2):
        try:
            descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            if not _runtime_lock_is_regular(path):
                raise RuntimeError("BATTERY_LOCK_UNSAFE")
            try:
                owner = read_json(path)
                pid = int(owner.get("pid", -1))
            except (OSError, ValueError, RuntimeError, json.JSONDecodeError):
                pid = -1
            if pid > 0 and _pid_is_active_battery(pid):
                raise RuntimeError(f"BATTERY_RUNNER_ALREADY_ACTIVE:pid={pid}")
            # A stale lock is evidence, not disposable scratch.  Archive it so
            # a later audit can distinguish a clean release from a forced stop.
            try:
                _archive_stale_runtime_lock(output_root, path, "battery.lock")
            except FileNotFoundError:
                pass
            continue
        try:
            os.write(
                descriptor,
                json.dumps({"pid": os.getpid(), "created_at_utc": utc_now()}, sort_keys=True).encode("utf-8"),
            )
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        try:
            assert_no_active_reconciliation(output_root)
        except BaseException:
            release_battery_lock(path)
            raise
        return path
    raise RuntimeError("BATTERY_LOCK_ACQUISITION_RACE")


def release_battery_lock(path: Path | None) -> None:
    if path is None or not os.path.lexists(path):
        return
    if not _runtime_lock_is_regular(path):
        return
    try:
        owner = read_json(path)
    except (OSError, RuntimeError, json.JSONDecodeError):
        return
    if int(owner.get("pid", -1)) == os.getpid():
        path.unlink(missing_ok=True)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"REQUIRED_JSON_NOT_OBJECT:{path.name}")
    return value


def initialize_manifest(
    *,
    config: dict[str, Any],
    output_root: Path,
    cache_sha256: str,
    split_sha256: str,
    config_sha256: str,
    device_policy_sha256: str,
    cache_characterization: dict[str, Any],
) -> dict[str, Any]:
    """Create the immutable, PET-free battery contract before any model starts."""

    return {
        "schema_version": 1,
        "protocol_name": config.get("protocol_name", "method_x_model_battery_v2_resnet34_resnet50"),
        "created_at_utc": utc_now(),
        "updated_at_utc": utc_now(),
        "status": "PREFLIGHT_PASSED",
        "method_x_pet_free": True,
        "final_claim_allowed": False,
        "autosupervision": {
            "mode": "AUTONOMOUS_MULTI_LAYER_FAIL_CLOSED",
            "human_approval_required": False,
            "manual_override_allowed": False,
            "decision": "PENDING",
        },
        "science_gate_status": "BLOCKED_PENDING_INDEPENDENT_SCIENCE_GATE",
        "locked_protocol": {
            "resolution": int(config["resolution"]),
            "folds": int(config["folds"]),
            "epochs": int(config["epochs"]),
            "seed": int(config["seed"]),
            "common": dict(config["common"]),
        },
        "locked_inputs": {
            "cache_sha256": cache_sha256,
            "split_sha256": split_sha256,
            "config_sha256": config_sha256,
            "device_policy_sha256": device_policy_sha256,
        },
        "cache_characterization": dict(cache_characterization),
        "models": [
            {
                "sequence": index,
                "backbone": backbone,
                "status": "PENDING",
                "output_dir": (Path("models") / backbone).as_posix(),
                "smoke": {"status": "PENDING", "official_128_vram_gate": False},
                "training": {"status": "NOT_STARTED"},
            }
            for index, backbone in enumerate(config["backbones"])
        ],
        "output_layout": {
            "root_name": output_root.name,
            "per_model": "models/<backbone>",
            "comparison": COMPARISON_JSON.as_posix(),
        },
    }


def assert_manifest_compatible(
    manifest: dict[str, Any],
    *,
    cache_sha256: str,
    split_sha256: str,
    config_sha256: str,
    device_policy_sha256: str,
) -> None:
    expected = {
        "cache_sha256": cache_sha256,
        "split_sha256": split_sha256,
        "config_sha256": config_sha256,
        "device_policy_sha256": device_policy_sha256,
    }
    if manifest.get("locked_inputs") != expected:
        raise RuntimeError("BATTERY_LOCKED_INPUT_DRIFT")


def _valid_sha256(value: Any) -> bool:
    text = str(value or "").strip().lower()
    return len(text) == 64 and all(char in "0123456789abcdef" for char in text)


def assert_spm_manifest_binding(
    manifest: dict[str, Any], verification: dict[str, Any]
) -> str:
    """Lock one verified SPM snapshot to a battery for its entire lifecycle."""

    live_digest = str(verification.get("snapshot_sha256") or "").strip().lower()
    if (
        not verification.get("verified")
        or not verification.get("ready")
        or not _valid_sha256(live_digest)
    ):
        raise RuntimeError("SPM_GATE_SNAPSHOT_NOT_VERIFIED")
    binding = manifest.get("spm_gate_snapshot")
    bound_digest = (
        str(binding.get("snapshot_sha256") or "").strip().lower()
        if isinstance(binding, dict)
        else ""
    )
    training_started = bool(manifest.get("training_started")) or any(
        isinstance(row, dict)
        and str(row.get("training", {}).get("status", "")).upper()
        not in {"", "NOT_STARTED"}
        for row in manifest.get("models", [])
    )
    if bound_digest and bound_digest != live_digest:
        raise RuntimeError(
            f"BATTERY_SPM_SNAPSHOT_DRIFT:bound={bound_digest}:live={live_digest}"
        )
    if not bound_digest and training_started:
        raise RuntimeError("BATTERY_SPM_SNAPSHOT_BINDING_MISSING_AFTER_TRAINING_START")
    return live_digest


def build_training_command(
    *,
    python: str,
    config: dict[str, Any],
    backbone: str,
    output_dir: Path,
    cache_sha256: str,
    spm_snapshot_sha256: str,
    phase: str = "final",
    extra_args: Sequence[str] = (),
) -> list[str]:
    common = config["common"]
    command = [
        python,
        "-u",
        str(ROOT / "scripts" / "train_3d_mri.py"),
        "--cache", CACHE_REL.as_posix(),
        "--cache_sha256", cache_sha256,
        "--spm_snapshot_sha256", spm_snapshot_sha256,
        "--require_exact_resume_contract",
        "--splits", SPLITS_REL.as_posix(),
        "--out_dir", str(output_dir),
        "--phase", str(phase),
        "--backbone", backbone,
        "--fusion", str(common["fusion"]),
        "--task_loss", str(common["task_loss"]),
        "--lr", str(common["learning_rate"]),
        "--scheduler", str(common["scheduler"]),
        "--dropout", str(common["dropout"]),
        "--resolution", str(config["resolution"]),
        "--folds", str(config["folds"]),
        "--epochs", str(config["epochs"]),
        "--seed", str(config["seed"]),
        "--batch_size", str(common["batch_size"]),
        "--grad_accum", str(common["grad_accum"]),
        "--device", "cuda",
        "--require_gpu",
        "--restore_best_at_end",
    ]
    if common.get("amp") is True:
        command.append("--amp")
    if common.get("gradient_checkpointing") is True:
        command.append("--checkpoint")
    if common.get("online_augmentation") is True:
        command.append("--monai_augment")
    # Throughput-only loader knobs; indices, transforms, seeds and fold
    # membership remain locked. Windows RTX 4060 defaults are deliberately
    # conservative to avoid worker oversubscription.
    command.extend([
        "--num_workers", str(common.get("num_workers", 2)),
        "--pin_memory" if common.get("pin_memory", True) else "--no-pin_memory",
        "--persistent_workers" if common.get("persistent_workers", False) else "--no-persistent_workers",
        "--prefetch_factor", str(common.get("prefetch_factor", 2)),
    ])
    command.extend(str(value) for value in extra_args)
    return command


def assert_training_sequence_ready(rows: list[dict[str, Any]], index: int) -> None:
    incomplete = [
        row["backbone"]
        for row in rows[:index]
        if row.get("status") not in ADVANCEABLE_MODEL_STATUSES
    ]
    if incomplete:
        raise RuntimeError(f"PREVIOUS_MODEL_NOT_ADVANCEABLE:{incomplete}")


def later_models_need_execution(rows: Sequence[Mapping[str, Any]], index: int) -> bool:
    """True when a later effective model is not yet safe to skip."""

    return any(
        str(row.get("status") or "") not in ADVANCEABLE_MODEL_STATUSES
        for row in rows[index + 1 :]
    )


def effective_model_rows(
    manifest: dict[str, Any], *, output_root: Path
) -> list[dict[str, Any]]:
    """Return the immutable effective order, validating any live amendment.

    A five-row future battery needs no amendment.  The reconciled seven-row v3
    battery does: its append-only amendment is the sole authority that removes
    the two never-started models from execution.
    """

    rows = manifest.get("models")
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        raise RuntimeError("BATTERY_MODEL_ROWS_INVALID")
    by_name: dict[str, dict[str, Any]] = {}
    for row in rows:
        name = str(row.get("backbone") or "")
        if not name or name in by_name:
            raise RuntimeError("BATTERY_MODEL_ROWS_INVALID")
        by_name[name] = row

    amendment_path = output_root / FIVE_MODEL_AMENDMENT
    if not amendment_path.exists():
        seven_model_identity = set(FIVE_MODEL_ORDER) | set(PROTOCOL_EXCLUDED_BACKBONES)
        if set(by_name) == seven_model_identity:
            raise RuntimeError("FIVE_MODEL_AMENDMENT_MISSING")
        if set(by_name) != set(FIVE_MODEL_ORDER):
            raise RuntimeError("FIVE_MODEL_MANIFEST_MEMBERSHIP_INVALID")
        if [str(row.get("backbone")) for row in rows] != list(FIVE_MODEL_ORDER):
            raise RuntimeError("FIVE_MODEL_MANIFEST_ORDER_INVALID")
        return [by_name[name] for name in FIVE_MODEL_ORDER]
    if amendment_path.is_symlink() or not amendment_path.is_file():
        raise RuntimeError("FIVE_MODEL_AMENDMENT_PATH_INVALID")
    amendment = read_json(amendment_path)
    unsigned = dict(amendment)
    recorded_digest = unsigned.pop("scientific_payload_sha256", None)
    expected_proof = {
        name: "SMOKE_PASS_ONLY_NO_TRAINING_CHECKPOINT_OR_METRICS_ARTIFACTS"
        for name in PROTOCOL_EXCLUDED_BACKBONES
    }
    locked_inputs = manifest.get("locked_inputs")
    if (
        amendment.get("schema_version") != 1
        or amendment.get("amendment_id")
        != "method_x_battery_v3_five_model_amendment.v1"
        or amendment.get("append_only") is not True
        or recorded_digest != _canonical_payload_sha256(unsigned)
        or amendment.get("excluded_backbones")
        != list(PROTOCOL_EXCLUDED_BACKBONES)
        or amendment.get("effective_order") != list(FIVE_MODEL_ORDER)
        or amendment.get("not_started_proof") != expected_proof
        or not isinstance(locked_inputs, dict)
        or amendment.get("locked_config_sha256")
        != locked_inputs.get("config_sha256")
    ):
        raise RuntimeError("FIVE_MODEL_AMENDMENT_INVALID")
    if set(by_name) != set(FIVE_MODEL_ORDER) | set(PROTOCOL_EXCLUDED_BACKBONES):
        raise RuntimeError("FIVE_MODEL_MANIFEST_MEMBERSHIP_INVALID")
    for name in PROTOCOL_EXCLUDED_BACKBONES:
        row = by_name[name]
        if (
            row.get("status") != "PROTOCOL_EXCLUDED_NOT_STARTED"
            or not isinstance(row.get("training"), dict)
            or row["training"].get("status")
            != "PROTOCOL_EXCLUDED_NOT_STARTED"
        ):
            raise RuntimeError(f"FIVE_MODEL_EXCLUSION_STATUS_INVALID:{name}")
    for name in FIVE_MODEL_ORDER:
        if by_name[name].get("status") == "PROTOCOL_EXCLUDED_NOT_STARTED":
            raise RuntimeError(f"FIVE_MODEL_EFFECTIVE_ROW_EXCLUDED:{name}")
    return [by_name[name] for name in FIVE_MODEL_ORDER]


def _is_paused_resumable_status(value: Any) -> bool:
    status = str(value or "").upper()
    return status in {"PAUSED_RESUMABLE", "TRAINING_PAUSED_RESUMABLE"} or status.startswith(
        "PAUSED_RESUMABLE_"
    )


def _command_option(command: Sequence[str], option: str) -> str:
    positions = [index for index, value in enumerate(command) if value == option]
    if len(positions) != 1 or positions[0] + 1 >= len(command):
        raise RuntimeError(f"STORED_TRAINING_COMMAND_OPTION_INVALID:{option}")
    return str(command[positions[0] + 1])


def validated_stored_training_command(
    row: dict[str, Any],
    *,
    output_root: Path,
    expected_command: Sequence[str] | None = None,
) -> list[str]:
    """Return the byte-for-byte manifest command after semantic safety checks."""

    training = row.get("training")
    command = training.get("command") if isinstance(training, dict) else None
    if (
        not isinstance(command, list)
        or len(command) < 7
        or not all(isinstance(value, str) and value for value in command)
    ):
        raise RuntimeError("TARGET_STORED_COMMAND_MISSING")
    if command[1] != "-u":
        raise RuntimeError("STORED_TRAINING_COMMAND_INVALID")
    executable = Path(command[0])
    executable = executable if executable.is_absolute() else ROOT / executable
    trainer = Path(command[2])
    trainer = trainer if trainer.is_absolute() else ROOT / trainer
    try:
        executable_resolved = executable.resolve(strict=True)
        trainer_resolved = trainer.resolve(strict=True)
        trainer_resolved.relative_to(ROOT.resolve())
    except (OSError, RuntimeError, ValueError) as exc:
        raise RuntimeError("STORED_TRAINING_COMMAND_PATH_INVALID") from exc
    if (
        executable_resolved != Path(sys.executable).resolve()
        or trainer_resolved != (ROOT / "scripts" / "train_3d_mri.py").resolve()
    ):
        raise RuntimeError("STORED_TRAINING_COMMAND_INVALID")
    if any(
        forbidden in command
        for forbidden in ("--no_resume", "--max_runtime_minutes")
    ) or "--require_exact_resume_contract" not in command:
        raise RuntimeError("STORED_TRAINING_COMMAND_RESUME_POLICY_INVALID")
    backbone = str(row.get("backbone") or "")
    if _command_option(command, "--backbone") != backbone:
        raise RuntimeError("STORED_TRAINING_COMMAND_BACKBONE_MISMATCH")
    expected_output = (output_root / str(row.get("output_dir") or "")).resolve()
    recorded_output = Path(_command_option(command, "--out_dir"))
    recorded_output = (
        recorded_output if recorded_output.is_absolute() else ROOT / recorded_output
    ).resolve()
    try:
        expected_output.relative_to(output_root.resolve())
    except ValueError as exc:
        raise RuntimeError("STORED_TRAINING_COMMAND_OUTPUT_INVALID") from exc
    if recorded_output != expected_output:
        raise RuntimeError("STORED_TRAINING_COMMAND_OUTPUT_MISMATCH")
    if expected_command is not None and list(command) != [
        str(value) for value in expected_command
    ]:
        raise RuntimeError("STORED_TRAINING_COMMAND_SCIENTIFIC_DRIFT")
    return list(command)


def validate_continuation_recovery_evidence(
    *, output_root: Path, row: Mapping[str, Any]
) -> dict[str, Any]:
    """Re-prove the one receipt-bound C-staging continuation candidate."""

    from src.program_governance.method_x_live_reconciliation import (
        _completed_receipt,
        _default_recovery_provider,
        _rows_by_name,
        _validate_recovery,
    )

    manifest = read_json(output_root / MANIFEST_NAME)
    rows = _rows_by_name(manifest)
    backbone = str(row.get("backbone") or "")
    canonical_row = rows.get(backbone)
    if backbone != "mednext_3d_s" or canonical_row is None:
        raise RuntimeError("CONTINUATION_RECOVERY_ROW_INVALID")
    if (
        canonical_row.get("status") != row.get("status")
        or canonical_row.get("output_dir") != row.get("output_dir")
    ):
        raise RuntimeError("CONTINUATION_RECOVERY_ROW_INVALID")
    receipt = _completed_receipt(output_root, manifest)
    if receipt is None or receipt.get("status") != "ALREADY_RECONCILED":
        raise RuntimeError("CONTINUATION_RECOVERY_RECEIPT_INVALID")
    live = _validate_recovery(
        _default_recovery_provider(
            output_root=output_root, row=canonical_row, manifest=manifest
        )
    )
    training = (
        canonical_row.get("training")
        if isinstance(canonical_row.get("training"), dict)
        else {}
    )
    stored = training.get("exact_recovery")
    if not isinstance(stored, dict):
        raise RuntimeError("CONTINUATION_RECOVERY_MANIFEST_EVIDENCE_MISSING")
    bindings = {
        "run_id": "run_id",
        "fold": "current_fold",
        "source": "source",
        "payload_path": "payload_path",
        "payload_sha256": "payload_sha256",
        "payload_size_bytes": "payload_size_bytes",
        "optimizer_step": "resume_from_step",
        "target_step": "target_step",
        "replay_steps": "replay_steps",
        "generation": "generation",
        "journal_record_sha256": "journal_record_sha256",
        "sampler_cursor": "sampler_cursor",
    }
    if any(stored.get(old) != live.get(new) for old, new in bindings.items()):
        raise RuntimeError("CONTINUATION_RECOVERY_LIVE_EVIDENCE_DRIFT")
    return {
        "evidence_source": "CONTINUATION_RECONCILIATION_RECEIPT_V1",
        "backbone": backbone,
        "current_fold": live["current_fold"],
        "finalized_folds": live["finalized_folds"],
        "transaction_id": receipt["transaction_id"],
        "receipt_path": Path(str(receipt["receipt_path"])).relative_to(
            output_root
        ).as_posix(),
        "exact_recovery": live,
    }


def apply_continuation_recovery_compatibility(
    *, report: Mapping[str, Any], output_root: Path
) -> dict[str, Any]:
    """Replace only the known stale H proof with a fresh receipt-bound C proof."""

    expected_finding = (
        "RESUMABLE_EVIDENCE_INVALID:mednext_3d_s:TASK2_PROGRESS_BYTES_MISMATCH"
    )
    materialized = copy.deepcopy(dict(report))
    if (
        materialized.get("decision") != "BLOCK"
        or set(materialized.get("blocking_reasons") or []) != {expected_finding}
        or materialized.get("processes")
    ):
        return materialized
    layers = materialized.get("layers")
    if not isinstance(layers, list):
        return materialized
    recovery_layers = [
        layer
        for layer in layers
        if isinstance(layer, dict) and layer.get("name") == "recovery_resume"
    ]
    other_layers = [
        layer
        for layer in layers
        if isinstance(layer, dict) and layer.get("name") != "recovery_resume"
    ]
    if (
        len(recovery_layers) != 1
        or recovery_layers[0].get("status") != "BLOCK"
        or set(recovery_layers[0].get("findings") or []) != {expected_finding}
        or len(other_layers) != len(REQUIRED_SUPERVISION_LAYERS) - 1
        or any(layer.get("status") not in {"PASS", "WARN"} for layer in other_layers)
    ):
        return materialized
    try:
        manifest = read_json(output_root / MANIFEST_NAME)
        matches = [
            row
            for row in (manifest.get("models") or [])
            if isinstance(row, dict) and row.get("backbone") == "mednext_3d_s"
        ]
        if len(matches) != 1:
            return materialized
        candidate = validate_continuation_recovery_evidence(
            output_root=output_root, row=matches[0]
        )
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError):
        return materialized
    recovery = recovery_layers[0]
    recovery["status"] = "PAUSE"
    recovery["findings"] = ["RESUMABLE_PAUSED_ROW"]
    evidence = recovery.get("evidence")
    evidence = dict(evidence) if isinstance(evidence, dict) else {}
    evidence["resumable"] = [candidate]
    evidence["compatibility_source"] = candidate["evidence_source"]
    recovery["evidence"] = evidence
    materialized["decision"] = "PAUSE_RESUMABLE"
    materialized["blocking_reasons"] = ["RESUMABLE_PAUSED_ROW"]
    automation = materialized.get("automation")
    automation = dict(automation) if isinstance(automation, dict) else {}
    automation["next_action"] = "RESUME_EXACT_CONTRACT"
    materialized["automation"] = automation
    return materialized


def exact_resume_gate_authorizes_row(
    report: dict[str, Any], row: dict[str, Any], *, output_root: Path
) -> bool:
    """Allow only the paused model named by a fresh exact-resume decision.

    ``PAUSE_RESUMABLE`` means that a general caller must not advance the
    battery.  The battery scheduler itself is also the component that carries
    out the supervisor's ``RESUME_EXACT_CONTRACT`` next action, so it may launch
    the current paused row only when every other layer passed and the reported
    checkpoint evidence still matches the filesystem.
    """

    if report.get("decision") != "PAUSE_RESUMABLE":
        return False
    if report.get("automation", {}).get("next_action") != "RESUME_EXACT_CONTRACT":
        return False
    if set(report.get("blocking_reasons") or []) != {"RESUMABLE_PAUSED_ROW"}:
        return False
    if report.get("processes"):
        return False
    if not _is_paused_resumable_status(row.get("status")):
        return False
    training = row.get("training") if isinstance(row.get("training"), dict) else {}
    if not _is_paused_resumable_status(training.get("status")):
        return False

    layers = report.get("layers")
    if not isinstance(layers, list):
        return False
    layer_names = [
        layer.get("name") for layer in layers if isinstance(layer, dict)
    ]
    if any(layer_names.count(name) != 1 for name in REQUIRED_SUPERVISION_LAYERS):
        return False
    recovery_layers = [
        layer for layer in layers
        if isinstance(layer, dict) and layer.get("name") == "recovery_resume"
    ]
    if len(recovery_layers) != 1:
        return False
    if any(
        layer.get("status") not in {"PASS", "WARN"}
        for layer in layers
        if isinstance(layer, dict) and layer.get("name") != "recovery_resume"
    ):
        return False
    recovery = recovery_layers[0]
    if recovery.get("status") != "PAUSE":
        return False
    if set(recovery.get("findings") or []) != {"RESUMABLE_PAUSED_ROW"}:
        return False
    resumable = recovery.get("evidence", {}).get("resumable")
    if not isinstance(resumable, list) or not resumable:
        return False
    if not all(isinstance(candidate, dict) for candidate in resumable):
        return False
    candidate_names = [str(candidate.get("backbone") or "") for candidate in resumable]
    if len(set(candidate_names)) != len(candidate_names):
        return False
    matching = [
        candidate
        for candidate in resumable
        if candidate.get("backbone") == row.get("backbone")
    ]
    if len(matching) != 1:
        return False
    candidate = matching[0]

    # Re-read and re-hash immediately before launch to close the
    # report-to-child gap. Legacy H proofs and the one receipt-bound C proof
    # have separate validators; neither is accepted by filename alone.
    try:
        if candidate.get("evidence_source") == (
            "CONTINUATION_RECONCILIATION_RECEIPT_V1"
        ):
            expected = validate_continuation_recovery_evidence(
                output_root=output_root, row=row
            )
        else:
            manifest = read_json(output_root / MANIFEST_NAME)
            expected = validate_resumable_checkpoint_evidence(
                output_root=output_root, manifest=manifest, row=row
            )
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError):
        return False
    return candidate == expected


def classify_training_returncode(returncode: int, observed_status: str | None = None) -> str:
    """Map a child exit to the scheduler-safe Method X lifecycle.

    Only the reserved ``75`` exit (or an explicit resumable metrics marker)
    is a pause.  Every other non-zero exit is a hard execution block; silently
    converting arbitrary failures into ``PAUSED_RESUMABLE`` can make the next
    model start against an invalid or half-written predecessor.
    """

    code = int(returncode)
    marker = str(observed_status or "").upper()
    if code == 0 and marker in {"PAUSED_RESUMABLE", "TRAINING_PAUSED_RESUMABLE"}:
        return "PAUSE_RESUMABLE"
    if code == 0:
        return "COMPLETE"
    if code == 75:
        return "PAUSE_RESUMABLE"
    if code == 78:
        return "BLOCK_RUN_CONTRACT"
    return "BLOCK_EXECUTION"


def _merge_reconciled_count(
    destination: dict[str, Any], field: str, value: int, *, scope: str
) -> None:
    if field in destination and int(destination[field]) != int(value):
        raise RuntimeError(f"RECONCILED_METRICS_COUNT_CONFLICT:{scope}.{field}")
    destination[field] = int(value)


def load_metrics_with_reconciliation_sidecar(path: Path) -> dict[str, Any]:
    """Load metrics and apply a hash-bound count repair only in memory."""

    payload = read_json(path)
    sidecar_path = path.with_name("three_d_metrics.reconciled.v1.json")
    if not sidecar_path.exists():
        return payload
    if sidecar_path.is_symlink() or not sidecar_path.is_file():
        raise RuntimeError("RECONCILED_METRICS_SIDECAR_PATH_INVALID")
    sidecar = read_json(sidecar_path)
    if (
        sidecar.get("schema_version") != 1
        or sidecar.get("artifact") != "resnet10_3d_validation_repair_sidecar"
        or sidecar.get("original_metrics_path") != path.name
    ):
        raise RuntimeError("RECONCILED_METRICS_SIDECAR_INVALID")
    if sidecar.get("original_metrics_sha256") != _sha256_file_local(path):
        raise RuntimeError("RECONCILED_METRICS_ORIGINAL_HASH_MISMATCH")
    aggregate_counts = sidecar.get("aggregate")
    fold_evidence = sidecar.get("fold_local_evidence")
    folds = payload.get("folds")
    aggregate = payload.get("aggregate")
    if (
        not isinstance(aggregate_counts, dict)
        or set(aggregate_counts)
        != {"real_fit_n", "augmented_fit_n", "fit_exposure_n"}
        or not isinstance(fold_evidence, list)
        or not isinstance(folds, list)
        or not isinstance(aggregate, dict)
        or aggregate.get("backbone") != "resnet10_3d"
        or len(fold_evidence) != len(folds)
    ):
        raise RuntimeError("RECONCILED_METRICS_SIDECAR_INVALID")
    evidence_by_fold: dict[int, dict[str, Any]] = {}
    totals = {"real_fit_n": 0, "augmented_fit_n": 0, "fit_exposure_n": 0}
    for evidence in fold_evidence:
        if not isinstance(evidence, dict) or set(evidence) != {
            "fold",
            "real_fit_n",
            "augmented_fit_n",
            "fit_exposure_n",
        }:
            raise RuntimeError("RECONCILED_METRICS_FOLD_EVIDENCE_INVALID")
        fold = int(evidence["fold"])
        if fold in evidence_by_fold:
            raise RuntimeError("RECONCILED_METRICS_FOLD_EVIDENCE_INVALID")
        evidence_by_fold[fold] = evidence
        real = int(evidence["real_fit_n"])
        augmented = int(evidence["augmented_fit_n"])
        exposure = int(evidence["fit_exposure_n"])
        if min(real, augmented, exposure) < 0 or exposure != real + augmented:
            raise RuntimeError("RECONCILED_METRICS_FOLD_EVIDENCE_INVALID")
        for field in totals:
            totals[field] += int(evidence[field])
    expected_aggregate = {field: int(aggregate_counts[field]) for field in totals}
    if totals != expected_aggregate:
        raise RuntimeError("RECONCILED_METRICS_AGGREGATE_MISMATCH")

    enriched = copy.deepcopy(payload)
    enriched_aggregate = enriched["aggregate"]
    for field, value in expected_aggregate.items():
        _merge_reconciled_count(enriched_aggregate, field, value, scope="aggregate")
    augmentation = enriched_aggregate.setdefault("augmentation", {})
    if not isinstance(augmentation, dict):
        raise RuntimeError("RECONCILED_METRICS_AUGMENTATION_CONFLICT")
    for field, value in expected_aggregate.items():
        _merge_reconciled_count(augmentation, field, value, scope="augmentation")
    source = augmentation.get("source_of_truth")
    if source is not None and (not isinstance(source, str) or not source.strip()):
        raise RuntimeError("RECONCILED_METRICS_SOURCE_CONFLICT")
    if source is None:
        augmentation["source_of_truth"] = sidecar_path.name
    augmentation["reconciliation_sidecar"] = sidecar_path.name
    augmentation["reconciliation_sidecar_sha256"] = _sha256_file_local(
        sidecar_path
    )
    for fold_payload in enriched["folds"]:
        if not isinstance(fold_payload, dict):
            raise RuntimeError("RECONCILED_METRICS_FOLD_SCHEMA_INVALID")
        fold = int(fold_payload.get("fold", -1))
        evidence = evidence_by_fold.get(fold)
        if evidence is None or int(fold_payload.get("outer_test_evaluation_count", -1)) != 1:
            raise RuntimeError("RECONCILED_METRICS_FOLD_EVIDENCE_INVALID")
        for field in totals:
            _merge_reconciled_count(
                fold_payload, field, int(evidence[field]), scope=f"fold_{fold}"
            )
    return enriched


def validate_completed_metrics(
    path: Path,
    *,
    backbone: str,
    config: dict[str, Any],
    cache_sha256: str | None = None,
    split_sha256: str | None = None,
    expected_training_split_digest: str | None = None,
    expected_spm_snapshot_sha256: str | None = None,
    strict_contract: bool = False,
) -> dict[str, Any]:
    payload = load_metrics_with_reconciliation_sidecar(path)
    aggregate = payload.get("aggregate")
    folds = payload.get("folds")
    if not isinstance(aggregate, dict) or aggregate.get("status") != "COMPLETED":
        raise RuntimeError("BATTERY_METRICS_NOT_COMPLETED")
    if aggregate.get("backbone") != backbone:
        raise RuntimeError("BATTERY_METRICS_BACKBONE_MISMATCH")
    if not isinstance(folds, list) or len(folds) != int(config["folds"]):
        raise RuntimeError("BATTERY_METRICS_FOLD_COUNT_INVALID")
    if int(aggregate.get("folds", -1)) != int(config["folds"]):
        raise RuntimeError("BATTERY_METRICS_FOLD_COUNT_INVALID")
    if int(aggregate.get("epochs", -1)) != int(config["epochs"]):
        raise RuntimeError("BATTERY_METRICS_EPOCH_BUDGET_MISMATCH")
    if int(aggregate.get("requested_resolution", -1)) != int(config["resolution"]):
        raise RuntimeError("BATTERY_METRICS_RESOLUTION_MISMATCH")
    if int(aggregate.get("cache_resolution", -1)) != int(config["resolution"]):
        raise RuntimeError("BATTERY_METRICS_CACHE_RESOLUTION_MISMATCH")
    common = config.get("common", {})
    expected_common = {
        "fusion": common.get("fusion"),
        "task_loss": common.get("task_loss"),
        "scheduler": common.get("scheduler"),
        "batch_size": common.get("batch_size"),
        "lr": common.get("learning_rate"),
        "dropout": common.get("dropout"),
        "grad_accum": common.get("grad_accum"),
    }
    for field, expected_value in expected_common.items():
        if expected_value is None:
            continue
        observed = aggregate.get(field)
        # Historical fixtures and pre-hardening metrics may not contain every
        # field; validate those once they are present, while preserving the
        # existing hash/fold contract checks.
        if observed is None:
            continue
        if field in {"lr", "dropout"}:
            if observed is None or not math.isclose(float(observed), float(expected_value), rel_tol=1e-9, abs_tol=1e-12):
                raise RuntimeError(f"BATTERY_METRICS_COMMON_HPARAM_MISMATCH:{field}")
        elif observed != expected_value:
            raise RuntimeError(f"BATTERY_METRICS_COMMON_HPARAM_MISMATCH:{field}")
    if strict_contract:
        # A production battery must carry the complete matched-protocol
        # decomposition.  Missing fields are not treated as historical
        # compatibility; they make the artifact ineligible for reuse.
        strict_expected = {
            "amp_enabled": common.get("amp"),
            "include_supplementary_train": common.get("include_supplementary_train", True),
            "include_global_supplementary_train": common.get("include_global_supplementary_train", False),
            "hpo_selection_only": False,
            "uses_pet_as_input": False,
            "uses_raw_identifiers": False,
            "uses_raw_paths": False,
        }
        for field, expected_value in strict_expected.items():
            if expected_value is None:
                continue
            if field not in aggregate:
                raise RuntimeError(f"BATTERY_METRICS_COMMON_HPARAM_MISSING:{field}")
            observed = aggregate.get(field)
            if observed != expected_value:
                raise RuntimeError(f"BATTERY_METRICS_COMMON_HPARAM_MISMATCH:{field}")
        policy = aggregate.get("outer_test_evaluation_policy")
        if not isinstance(policy, dict):
            raise RuntimeError("BATTERY_METRICS_COMMON_HPARAM_MISSING:outer_test_evaluation_policy")
        for field, expected_value in {
            "during_training_epochs": False,
            "after_checkpoint_lock_only": True,
            "max_evaluations_per_fold": 1,
        }.items():
            if field not in policy:
                raise RuntimeError(f"BATTERY_METRICS_COMMON_HPARAM_MISSING:outer_test_evaluation_policy.{field}")
            if policy[field] != expected_value:
                raise RuntimeError(f"BATTERY_METRICS_COMMON_HPARAM_MISMATCH:outer_test_evaluation_policy.{field}")
        augmentation = aggregate.get("augmentation")
        if not isinstance(augmentation, dict):
            raise RuntimeError("BATTERY_METRICS_COMMON_HPARAM_MISSING:augmentation")
        for field in ("real_fit_n", "augmented_fit_n", "fit_exposure_n", "source_of_truth"):
            if field not in augmentation:
                raise RuntimeError(f"BATTERY_METRICS_COMMON_HPARAM_MISSING:augmentation.{field}")
        aggregate_counts = {}
        for field in ("real_fit_n", "augmented_fit_n", "fit_exposure_n"):
            if field not in aggregate:
                raise RuntimeError(f"BATTERY_METRICS_COMMON_HPARAM_MISSING:{field}")
            aggregate_counts[field] = int(aggregate[field])
            if int(augmentation[field]) != aggregate_counts[field]:
                raise RuntimeError(f"BATTERY_METRICS_COMMON_HPARAM_MISMATCH:augmentation.{field}")
        if aggregate_counts["fit_exposure_n"] != aggregate_counts["real_fit_n"] + aggregate_counts["augmented_fit_n"]:
            raise RuntimeError("BATTERY_METRICS_AUGMENTATION_COUNT_INCONSISTENT")
        fold_counts = {"real_fit_n": 0, "augmented_fit_n": 0, "fit_exposure_n": 0}
        for fold in folds:
            if not isinstance(fold, dict):
                raise RuntimeError("BATTERY_METRICS_FOLD_SCHEMA_INVALID")
            for field in fold_counts:
                if field not in fold:
                    raise RuntimeError(f"BATTERY_METRICS_FOLD_FIELD_MISSING:{field}")
                fold_counts[field] += int(fold[field])
            if int(fold.get("outer_test_evaluation_count", -1)) != 1:
                raise RuntimeError("BATTERY_METRICS_OUTER_TEST_POLICY_INVALID")
            if bool(fold.get("hpo_selection_only", True)):
                raise RuntimeError("BATTERY_METRICS_HPO_SELECTION_ONLY_INVALID")
        if fold_counts != aggregate_counts:
            raise RuntimeError("BATTERY_METRICS_AUGMENTATION_COUNT_MISMATCH")
    contract_path = path.parent / "run_contract.json"
    if not contract_path.is_file():
        raise RuntimeError("BATTERY_RUN_CONTRACT_MISSING")
    contract_payload = read_json(contract_path)
    contract = contract_payload.get("run_contract")
    if not isinstance(contract, dict):
        raise RuntimeError("BATTERY_RUN_CONTRACT_MISSING")
    parameters = contract.get("parameters") if isinstance(contract.get("parameters"), dict) else {}
    if (
        contract.get("folds") != int(config["folds"])
        or contract.get("epochs") != int(config["epochs"])
        or contract.get("seed") != int(config["seed"])
        or parameters.get("backbone") not in (None, backbone)
        or (cache_sha256 is not None and contract.get("cache_sha256") != cache_sha256)
        or (
            expected_spm_snapshot_sha256 is not None
            and contract.get("spm_snapshot_sha256") != expected_spm_snapshot_sha256
        )
        or (
            expected_training_split_digest is not None
            and contract.get("split_digest") != expected_training_split_digest
        )
        or (
            expected_training_split_digest is None
            and split_sha256 is not None
            and contract.get("split_digest") != split_sha256
        )
    ):
        raise RuntimeError("BATTERY_RUN_CONTRACT_MISMATCH")
    return payload


def validate_model_metrics_for_row(
    path: Path,
    *,
    backbone: str,
    config: dict[str, Any],
    cache_sha256: str | None = None,
    split_sha256: str | None = None,
    expected_training_split_digest: str | None = None,
    expected_spm_snapshot_sha256: str | None = None,
) -> dict[str, Any]:
    """Validate a model row with every immutable scientific binding."""

    if not expected_spm_snapshot_sha256:
        raise RuntimeError("BATTERY_SPM_SNAPSHOT_DIGEST_REQUIRED")
    return validate_completed_metrics(
        path,
        backbone=backbone,
        config=config,
        cache_sha256=cache_sha256,
        split_sha256=split_sha256,
        expected_training_split_digest=expected_training_split_digest,
        expected_spm_snapshot_sha256=expected_spm_snapshot_sha256,
        strict_contract=True,
    )


def _fold_mean_ci(rows: list[dict[str, Any]], metric: str) -> list[float | None]:
    values = [float(row[metric]) for row in rows if row.get(metric) is not None and math.isfinite(float(row[metric]))]
    if not values:
        return [None, None]
    mean = statistics.fmean(values)
    if len(values) < 2:
        return [mean, mean]
    half_width = 1.96 * statistics.stdev(values) / math.sqrt(len(values))
    return [mean - half_width, mean + half_width]


def build_comparison_report(
    manifest: dict[str, Any], metrics_by_model: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    locked = manifest["locked_inputs"]
    rows: list[dict[str, Any]] = []
    for model in manifest["models"]:
        backbone = model["backbone"]
        training = model.get("training", {})
        if model.get("status") == "COMPLETED":
            if (
                training.get("cache_sha256") != locked["cache_sha256"]
                or training.get("split_sha256") != locked["split_sha256"]
            ):
                raise RuntimeError("BATTERY_METRIC_HASH_MISMATCH")
            payload = metrics_by_model[backbone]
            aggregate = payload["aggregate"]
            folds = payload["folds"]
            rows.append(
                {
                    "backbone": backbone,
                    "status": "COMPLETED",
                    "roc_auc": aggregate.get("roc_auc"),
                    "pr_auc": aggregate.get("pr_auc"),
                    "brier": aggregate.get("brier"),
                    "per_fold": folds,
                    "fold_mean_95ci": {
                        metric: _fold_mean_ci(folds, metric) for metric in ("roc_auc", "pr_auc", "brier")
                    },
                }
            )
        else:
            rows.append({"backbone": backbone, "status": model.get("status", "UNKNOWN")})
    by_name = {row["backbone"]: row for row in rows if row["status"] == "COMPLETED"}
    delta = None
    if "resnet34_3d" in by_name and "resnet50_3d" in by_name:
        delta = {
            "roc_auc_delta": float(by_name["resnet50_3d"]["roc_auc"]) - float(by_name["resnet34_3d"]["roc_auc"]),
            "direction": "resnet50_minus_resnet34",
            "source": "matched_battery_only",
        }
    terminal = all(row["status"] in TERMINAL_MODEL_STATUSES for row in manifest["models"])
    return {
        "schema_version": 1,
        "status": "COMPLETED" if terminal else "IN_PROGRESS",
        "matched_hashes": True,
        "locked_inputs": locked,
        "models": rows,
        "resnet34_to_resnet50": delta,
        "ci_method": "fold_normal_approximation_descriptive_only",
        "promotion_ci_required": "paired_subject_level_bootstrap_min_1000_resamples",
        "promotion_ci_status": "PENDING_UNTIL_PAIRED_OOF_PREDICTIONS_ARE_FINALIZED",
        "final_claim_allowed": False,
        "science_gate_status": manifest.get(
            "science_gate_status", "BLOCKED_PENDING_INDEPENDENT_SCIENCE_GATE"
        ),
        "generated_at_utc": utc_now(),
    }


def smoke_backbone(
    backbone: str,
    *,
    device_name: str,
    resolution: int,
    hard_stop_vram_gb: float,
) -> dict[str, Any]:
    """Build and forward the requested backbone without any architecture fallback."""

    started = utc_now()
    try:
        import torch

        from src.models.backbones.backbone_factory import build_3d_backbone

        if device_name == "cuda" and not torch.cuda.is_available():
            return {
                "status": "BLOCKED_DEPENDENCY",
                "blocker": "CUDA_UNAVAILABLE",
                "official_128_vram_gate": False,
                "backbone": backbone,
                "created_at_utc": started,
            }
        device = torch.device(device_name)
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
        model = build_3d_backbone(backbone, in_channels=1, out_dim=8).to(device).eval()
        input_shape = [1, 1, int(resolution), int(resolution), int(resolution)]
        with torch.inference_mode():
            output = model(torch.zeros(input_shape, dtype=torch.float32, device=device))
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            allocated = float(torch.cuda.max_memory_allocated(device) / (1024**3))
            reserved = float(torch.cuda.max_memory_reserved(device) / (1024**3))
            peak = max(allocated, reserved)
        else:
            allocated = reserved = peak = None
        if tuple(output.shape) != (1, 8):
            raise RuntimeError(f"SMOKE_OUTPUT_SHAPE_INVALID:{tuple(output.shape)}")
        official = device.type == "cuda" and int(resolution) == 128
        status = "PASS" if official else "STATIC_PASS"
        if official and peak is not None and peak > float(hard_stop_vram_gb):
            status = "BLOCKED_VRAM_CAP"
        return {
            "status": status,
            "backbone": backbone,
            "device": str(device),
            "input_shape": input_shape,
            "output_shape": list(output.shape),
            "peak_vram_gb": peak,
            "max_memory_allocated_gb": allocated,
            "max_memory_reserved_gb": reserved,
            "hard_stop_vram_gb": float(hard_stop_vram_gb),
            "official_128_vram_gate": status == "PASS" and official,
            "fallback_backbone_used": False,
            "created_at_utc": started,
        }
    except ImportError as exc:
        return {
            "status": "BLOCKED_DEPENDENCY",
            "blocker": type(exc).__name__,
            "official_128_vram_gate": False,
            "backbone": backbone,
            "created_at_utc": started,
        }
    except RuntimeError as exc:
        if "out of memory" in str(exc).casefold():
            return {
                "status": "BLOCKED_VRAM_CAP",
                "blocker": "CUDA_OUT_OF_MEMORY",
                "official_128_vram_gate": False,
                "backbone": backbone,
                "hard_stop_vram_gb": float(hard_stop_vram_gb),
                "created_at_utc": started,
            }
        return {
            "status": "BLOCKED_SMOKE_FAILED",
            "blocker": type(exc).__name__,
            "official_128_vram_gate": False,
            "backbone": backbone,
            "created_at_utc": started,
        }


def load_cache_characterization() -> dict[str, Any]:
    augmentation = read_json(ROOT / AUGMENTATION_SUMMARY_REL)
    verification = read_json(ROOT / CACHE_VERIFICATION_REL)
    expected = {
        "total_rows": 6213,
        "eligible_real_rows": 1669,
        "materialized_augmented_rows": 4544,
        "folds": 5,
        "seed": 13,
        "resolution": 128,
    }
    observed = {
        "total_rows": int(augmentation.get("total_rows", -1)),
        "eligible_real_rows": int(augmentation.get("eligible_real_rows", -1)),
        "materialized_augmented_rows": int(augmentation.get("materialized_augmented_rows", -1)),
        "folds": int(augmentation.get("split_folds", -1)),
        "seed": int(augmentation.get("training_seed", -1)),
        "resolution": 128,
    }
    if observed != expected or augmentation.get("status") != "PUBLISHED_CANONICAL" or verification.get("status") != "PASS":
        raise RuntimeError("BATTERY_CACHE_CHARACTERIZATION_DRIFT")
    return {
        **observed,
        "status": "PASS",
        "canonical_cache": CACHE_REL.as_posix(),
        "augmentation_training_scope": "fold_local_inner_fit_only",
        "source_cache_manifest_sha256": augmentation.get("source_cache_manifest_sha256"),
        "source_split_contract_sha256": augmentation.get("split_contract_sha256"),
    }


def load_cached_lineage_membership(
    *,
    cache_path: Path,
    split_sha256: str,
    cache_sha256: str,
) -> dict[str, Any] | None:
    """Reuse a previously verified lineage audit only when its inputs are bound.

    The canonical cache is multi-gigabyte.  Re-reading every HDF5 lineage
    vector on every resume preflight is unnecessary once a lock records the
    exact cache size/mtime, cache digest, split digest and audit digest.  Any
    mismatch falls back to the full validation path; no stale audit is trusted.
    """

    audit_path = ROOT / "results/method_x_registry/augmentation_count_audit.json"
    lock_path = ROOT / "results/method_x_registry/data_lineage_manifest.json"
    if not audit_path.is_file() or not lock_path.is_file():
        return None
    try:
        audit = read_json(audit_path)
        lock = read_json(lock_path)
        stat = cache_path.stat()
        audit_digest = _sha256_file_local(audit_path)
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError):
        return None
    expected_counts = {
        "real": 1669,
        "materialized_masked_augmentation": 4544,
        "total": 6213,
    }
    if (
        lock.get("status") != "PASS"
        or lock.get("cache_sha256") != cache_sha256
        or lock.get("split_sha256") != split_sha256
        or lock.get("audit_sha256") != audit_digest
        or int(lock.get("cache_size", -1)) != int(stat.st_size)
        or int(lock.get("cache_mtime_ns", -1)) != int(stat.st_mtime_ns)
        or audit.get("status") != "PASS"
        or audit.get("observed_source_counts") != expected_counts
        or audit.get("expected_source_counts") != expected_counts
        or int(audit.get("folds", -1)) != 5
        or int(audit.get("seed", -1)) != 13
    ):
        return None
    return {
        "status": "PASS_REUSED_LOCKED_LINEAGE",
        "source": "results/method_x_registry/data_lineage_manifest.json",
        "cache_sha256": cache_sha256,
        "split_sha256": split_sha256,
        "audit_sha256": audit_digest,
        "source_counts": expected_counts,
        "folds_detail": audit.get("folds_detail", []),
        "full_cache_validation_status": "PASS_LOCKED_AUDIT",
    }


def _output_root(config: dict[str, Any], requested: Path | None) -> Path:
    if not isinstance(config, dict) or not isinstance(config.get("output_root"), str):
        raise RuntimeError("CONFIG_OUTPUT_ROOT_INVALID")
    configured = Path(config["output_root"])
    if requested is None and configured == LEGACY_UNVERSIONED_OUTPUT_ROOT:
        raise RuntimeError(
            "EXPLICIT_VERSIONED_OUTPUT_ROOT_REQUIRED:"
            "pass --output-root so corrected results cannot recreate the archived legacy root"
        )
    value = requested or configured
    value = value if value.is_absolute() else ROOT / value
    return assert_path_beneath_root_without_reparse(
        ROOT,
        value,
        outside_error="OUTPUT_ROOT_OUTSIDE_H_PROJECT",
        reparse_error="OUTPUT_ROOT_REPARSE_POINT_REJECTED",
    )


def _load_cli_config(path: Path) -> tuple[Path, dict[str, Any]]:
    candidate = path if path.is_absolute() else ROOT / path
    candidate = assert_path_beneath_root_without_reparse(
        ROOT,
        candidate,
        outside_error="CONFIG_PATH_OUTSIDE_H_PROJECT",
        reparse_error="CONFIG_PATH_REPARSE_POINT_REJECTED",
        require_exists=True,
        require_regular_file=True,
    )
    try:
        before = os.lstat(candidate)
        with candidate.open("r", encoding="utf-8") as handle:
            opened = os.fstat(handle.fileno())
            text = handle.read()
        after = os.lstat(candidate)
        if (
            (opened.st_dev, opened.st_ino) != (after.st_dev, after.st_ino)
            or (before.st_dev, before.st_ino) != (after.st_dev, after.st_ino)
            or before.st_size != after.st_size
            or before.st_mtime_ns != after.st_mtime_ns
        ):
            raise RuntimeError("CONFIG_PATH_CHANGED_DURING_READ")
        payload = yaml.safe_load(text)
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise RuntimeError(f"CONFIG_LOAD_FAILED:{type(exc).__name__}:{exc}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError("CONFIG_SCHEMA_INVALID:top-level mapping required")
    return candidate, payload


def _safe_command(command: list[str]) -> list[str]:
    """Canonicalize only paths contained by the repository for stable storage.

    A stored command may predate canonicalization and therefore contain the
    absolute H-side interpreter path.  Treating that spelling difference as a
    scientific-command change blocks an otherwise identical exact resume.
    Paths outside the repository and all non-path tokens remain byte-for-byte.
    """

    root = ROOT.resolve()
    canonical: list[str] = []
    for raw in command:
        value = str(raw)
        candidate = Path(value)
        if candidate.is_absolute():
            try:
                relative = candidate.resolve(strict=False).relative_to(root)
            except (OSError, RuntimeError, ValueError):
                pass
            else:
                value = str(relative)
        canonical.append(value)
    return canonical


def _child_log_paths(
    output_root: Path, *, backbone: str, attempt: int
) -> tuple[Path, Path]:
    root = output_root.resolve()
    try:
        root.relative_to(ROOT.resolve())
    except ValueError as exc:
        raise RuntimeError("CHILD_LOG_ROOT_OUTSIDE_H_PROJECT") from exc
    token = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    stem = f"{token}-{uuid.uuid4().hex}-attempt-{int(attempt)}"
    log_dir = root / "logs" / "children" / backbone
    return log_dir / f"{stem}.stdout.log", log_dir / f"{stem}.stderr.log"


def _log_sha256(path: Path) -> str | None:
    return _sha256_file_local(path) if path.is_file() else None


def _read_log_tail(path: Path, limit: int = 131072) -> str:
    if not path.is_file():
        return ""
    with path.open("rb") as handle:
        size = path.stat().st_size
        if size > limit:
            handle.seek(size - limit)
        return handle.read(limit).decode("utf-8", errors="replace")


WATCH_INVENTORY_RACE_RETRY_SECONDS = 0.5
MANIFEST_IO_RETRY_ATTEMPTS = 8
MANIFEST_IO_RETRY_BASE_SECONDS = 0.2
_SUPERVISOR_INFRASTRUCTURE_REASONS = frozenset(
    {"SUPERVISOR_START_HOOK_ERROR", "SUPERVISOR_PROBE_ERROR"}
)
_WINDOWS_FILE_LOCK_TOKENS = (
    "winerror 5",
    "winerror 32",
    "access is denied",
    "sharing violation",
)
_WINDOWS_FILE_LOCK_WINERRORS = frozenset({5, 32})


def _supervised_watch_report(supervised: Mapping[str, Any] | None) -> Mapping[str, Any] | None:
    if not isinstance(supervised, Mapping):
        return None
    last_report = supervised.get("last_report")
    if isinstance(last_report, Mapping):
        return last_report
    return supervised


def defer_watch_inventory_race(watch_report: Mapping[str, Any]) -> dict[str, Any]:
    """Keep a healthy trainer alive across a transient C inventory lock."""

    report = dict(watch_report)
    report["decision"] = "CONTINUE"
    report["watch_inventory_race_deferred"] = True
    automation = dict(report.get("automation") or {})
    automation["next_action"] = "RUN_NEXT_LOCKED_STAGE"
    report["automation"] = automation
    return report


def resolve_watch_inventory_race(
    *,
    first_report: Mapping[str, Any],
    retry_probe: Callable[[], Mapping[str, Any]],
) -> tuple[dict[str, Any], bool]:
    """Return ``(report, should_enforce)``. Inventory-only watch BLOCK is not a kill."""

    first = dict(first_report)
    if first.get("decision") != "BLOCK":
        return first, False
    if not watch_blockers_are_live_inventory_race(first):
        return first, True
    retry = dict(retry_probe())
    if retry.get("decision") != "BLOCK":
        return retry, False
    if watch_blockers_are_live_inventory_race(retry):
        return defer_watch_inventory_race(retry), False
    return retry, True


def is_windows_file_lock(value: BaseException | str) -> bool:
    """True for the Windows sharing/access-denied class that USB H replaces hit."""

    if isinstance(value, BaseException):
        winerror = getattr(value, "winerror", None)
        if winerror in _WINDOWS_FILE_LOCK_WINERRORS:
            return True
    text = str(value).casefold()
    return any(token in text for token in _WINDOWS_FILE_LOCK_TOKENS)


def call_with_windows_file_lock_retry(
    operation: Callable[[], Any],
    *,
    attempts: int = MANIFEST_IO_RETRY_ATTEMPTS,
    sleep: Callable[[float], None] = time.sleep,
    base_seconds: float = MANIFEST_IO_RETRY_BASE_SECONDS,
) -> Any:
    """Retry a filesystem replace that lost a Windows sharing race."""

    last_error: OSError | None = None
    total = max(1, int(attempts))
    for index in range(total):
        try:
            return operation()
        except OSError as exc:
            if not is_windows_file_lock(exc):
                raise
            last_error = exc
            if index + 1 >= total:
                break
            sleep(float(base_seconds) * (index + 1))
    assert last_error is not None
    raise last_error


def supervisor_infrastructure_reason(supervised: Mapping[str, Any] | None) -> str:
    if not isinstance(supervised, Mapping):
        return ""
    return str(supervised.get("reason") or "")


def resolve_child_lifecycle(
    *,
    returncode: int,
    observed_status: str | None,
    failure: Mapping[str, Any],
) -> str:
    """Map a classified child failure onto the scheduler lifecycle.

    The supervisor reuses exit 78 for infrastructure kills. Only an explicit
    scientific-contract classification may become ``BLOCK_RUN_CONTRACT``.
    """

    category = str(failure.get("category") or "")
    if category == "SCIENTIFIC_CONTRACT_MISMATCH":
        return "BLOCK_RUN_CONTRACT"
    if category == "STAGING_INVENTORY_WATCH_RACE":
        return "PAUSE_RESUMABLE"
    lifecycle = classify_training_returncode(returncode, observed_status)
    if lifecycle == "BLOCK_RUN_CONTRACT":
        return "BLOCK_EXECUTION"
    return lifecycle


def classify_child_failure(
    *,
    returncode: int,
    stdout_log_path: Path,
    stderr_log_path: Path,
    supervised: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Classify a child failure with contract precedence over I/O retry."""

    code = int(returncode)
    stdout_tail = _read_log_tail(stdout_log_path)
    stderr_tail = _read_log_tail(stderr_log_path)
    combined = (stderr_tail + "\n" + stdout_tail).casefold()
    contract_markers = (
        "run_contract_mismatch",
        "contract_mismatch",
        "battery_locked_input_drift",
        "scientific_contract",
        "spm_snapshot_drift",
        "split_digest_mismatch",
        "cache_sha256_mismatch",
    )
    transient_markers = (
        "winerror 32",
        "sharing violation",
        "resource temporarily unavailable",
        "temporarily unavailable",
        "stale file handle",
        "connection reset by peer",
        "errno 11",
    )
    watch_report = _supervised_watch_report(supervised)
    inventory_race = watch_blockers_are_live_inventory_race(watch_report)
    infra_reason = supervisor_infrastructure_reason(supervised)
    probe_error = (
        ""
        if not isinstance(supervised, Mapping)
        else str(supervised.get("probe_error") or "")
    )
    if inventory_race:
        category, retryable = "STAGING_INVENTORY_WATCH_RACE", False
    elif infra_reason in _SUPERVISOR_INFRASTRUCTURE_REASONS:
        if is_windows_file_lock(probe_error):
            category, retryable = "TRANSIENT_IO", True
        else:
            category, retryable = "NON_RETRYABLE_EXECUTION", False
    elif (
        isinstance(supervised, Mapping)
        and supervised.get("terminated_by_supervisor")
        and not any(marker in combined for marker in contract_markers)
    ):
        category, retryable = "NON_RETRYABLE_EXECUTION", False
    elif code == 0:
        category, retryable = "NONE", False
    elif code == 75:
        category, retryable = "RESUMABLE_STOP", False
    elif code == 78 or any(marker in combined for marker in contract_markers):
        category, retryable = "SCIENTIFIC_CONTRACT_MISMATCH", False
    elif any(marker in combined for marker in transient_markers):
        category, retryable = "TRANSIENT_IO", True
    else:
        category, retryable = "NON_RETRYABLE_EXECUTION", False
    normalized = re.sub(r"0x[0-9a-f]+", "0x#", combined)
    fingerprint = hashlib.sha256(
        f"{category}\n{returncode}\n{normalized}".encode("utf-8")
    ).hexdigest()
    return {
        "category": category,
        "retryable": retryable,
        "fingerprint_sha256": fingerprint,
        "traceback_present": "traceback (most recent call last)" in combined,
        "stdout_sha256": _log_sha256(stdout_log_path),
        "stderr_sha256": _log_sha256(stderr_log_path),
        "watch_inventory_race": inventory_race,
    }


def run_child_with_bounded_io_retry(
    command: Sequence[str],
    *,
    backbone: str,
    output_root: Path,
    attempt_runner: Callable[..., dict[str, Any]],
    attempt_observer: Callable[[dict[str, Any], dict[str, Any]], None] | None = None,
    max_transient_retries: int = MAX_TRANSIENT_IO_RETRIES,
) -> dict[str, Any]:
    """Run one immutable command, retrying only explicit transient I/O."""

    immutable_command = tuple(str(value) for value in command)
    if not immutable_command:
        raise RuntimeError("EMPTY_CHILD_COMMAND")
    if int(max_transient_retries) < 0 or int(max_transient_retries) > 2:
        raise RuntimeError("TRANSIENT_IO_RETRY_BUDGET_INVALID")
    attempts: list[dict[str, Any]] = []
    last_supervised: dict[str, Any] = {}
    last_failure: dict[str, Any] = {}
    for attempt in range(1, int(max_transient_retries) + 2):
        stdout_log, stderr_log = _child_log_paths(
            output_root, backbone=backbone, attempt=attempt
        )
        last_supervised = attempt_runner(
            command=list(immutable_command),
            stdout_log_path=stdout_log,
            stderr_log_path=stderr_log,
            attempt=attempt,
        )
        if not isinstance(last_supervised, dict) or "returncode" not in last_supervised:
            raise RuntimeError("CHILD_ATTEMPT_RESULT_INVALID")
        returncode = int(last_supervised["returncode"])
        last_failure = classify_child_failure(
            returncode=returncode,
            stdout_log_path=stdout_log,
            stderr_log_path=stderr_log,
            supervised=last_supervised,
        )
        attempt_record = {
            "attempt": attempt,
            "returncode": returncode,
            "stdout_log": stdout_log.relative_to(ROOT).as_posix(),
            "stderr_log": stderr_log.relative_to(ROOT).as_posix(),
            "failure": last_failure,
        }
        attempts.append(attempt_record)
        if attempt_observer is not None:
            attempt_observer(copy.deepcopy(attempt_record), last_supervised)
        should_retry = (
            last_failure["retryable"]
            and attempt <= int(max_transient_retries)
        )
        if not should_retry:
            retry_count = attempt - 1
            return {
                "returncode": returncode,
                "supervised": last_supervised,
                "attempts": attempts,
                "failure": last_failure,
                "retry_count": retry_count,
                "retry_exhausted": bool(
                    last_failure["retryable"]
                    and retry_count == int(max_transient_retries)
                ),
            }
    raise AssertionError("bounded retry loop exhausted without a terminal result")


def infer_paused_fold_layout(*, output_root: Path, row: Mapping[str, Any]) -> dict[str, Any]:
    """Read fold-state files and the live H progress checkpoint for one row.

    Absence of a consecutive TRAINING fold plus a non-empty progress file is
    ``ORPHAN_*``, never a fabricated pause.
    """

    backbone = str(row.get("backbone") or "")
    output_dir = str(row.get("output_dir") or "").replace("\\", "/")
    expected_output_dir = f"models/{backbone}"
    if not backbone or output_dir != expected_output_dir:
        raise RuntimeError("ORPHAN_MODEL_OUTPUT_DIR_INVALID")
    model_dir = assert_path_beneath_root_without_reparse(
        output_root,
        output_root / Path(output_dir),
        outside_error="ORPHAN_MODEL_DIR_OUTSIDE_OUTPUT_ROOT",
        reparse_error="ORPHAN_MODEL_DIR_REPARSE_POINT_REJECTED",
        require_exists=True,
    )
    finalized: list[int] = []
    training_fold: int | None = None
    live_digest: str | None = None
    for fold in range(LOCKED_FOLD_COUNT):
        state_path = model_dir / f"method_x_3d_fold_{fold}_state.json"
        if not state_path.is_file():
            break
        state_path = assert_path_beneath_root_without_reparse(
            output_root,
            state_path,
            outside_error="ORPHAN_FOLD_STATE_OUTSIDE_OUTPUT_ROOT",
            reparse_error="ORPHAN_FOLD_STATE_REPARSE_POINT_REJECTED",
            require_exists=True,
            require_regular_file=True,
        )
        state = read_json(state_path)
        digest = str(state.get("contract_digest") or "")
        if (
            not isinstance(state, dict)
            or state.get("fold") != fold
            or not _valid_sha256(digest)
        ):
            raise RuntimeError(f"ORPHAN_FOLD_STATE_INVALID:{fold}")
        stage = str(state.get("stage") or "")
        if stage == "FINALIZED":
            if training_fold is not None or fold != len(finalized):
                raise RuntimeError(f"ORPHAN_FOLD_ORDER_INVALID:{fold}")
            finalized.append(fold)
            if live_digest is None:
                live_digest = digest
        elif stage == "TRAINING":
            if training_fold is not None or fold != len(finalized):
                raise RuntimeError(f"ORPHAN_TRAINING_FOLD_NOT_NEXT:{fold}")
            training_fold = fold
            live_digest = digest
        else:
            raise RuntimeError(f"ORPHAN_FOLD_STAGE_INVALID:{fold}")
    if training_fold is None or live_digest is None:
        raise RuntimeError("ORPHAN_NO_TRAINING_FOLD")
    return {
        "current_fold": training_fold,
        "finalized_folds": finalized,
        "state_contract_digest": live_digest,
        "progress_checkpoint": resolve_paused_progress_checkpoint(
            output_root=output_root,
            output_dir=expected_output_dir,
            fold=training_fold,
        ),
    }


def _upsert_resumable_evidence(
    manifest: dict[str, Any],
    *,
    backbone: str,
    layout: Mapping[str, Any],
) -> dict[str, Any]:
    reconciliation = manifest.setdefault(
        "state_reconciliation",
        {"schema_version": 1, "status": "RECONCILED", "resumable_evidence": []},
    )
    if not isinstance(reconciliation, dict):
        raise RuntimeError("ORPHAN_RECONCILIATION_BLOCK_INVALID")
    rows = reconciliation.get("resumable_evidence")
    if not isinstance(rows, list):
        rows = []
        reconciliation["resumable_evidence"] = rows
    existing: dict[str, Any] | None = None
    for item in rows:
        if isinstance(item, dict) and item.get("backbone") == backbone:
            existing = item
            break
    payload = dict(existing) if existing is not None else {}
    payload["backbone"] = backbone
    payload["current_fold"] = int(layout["current_fold"])
    payload["finalized_folds"] = list(layout["finalized_folds"])
    payload["state_contract_digest"] = str(layout["state_contract_digest"])
    payload["progress_checkpoint"] = dict(layout["progress_checkpoint"])
    if existing is None:
        rows.append(payload)
    else:
        existing.update(payload)
        payload = existing
    reconciliation["status"] = "RECONCILED"
    reconciliation["schema_version"] = 1
    return payload


def _row_looks_orphaned_running(row: Mapping[str, Any]) -> bool:
    status = str(row.get("status") or "")
    training = row.get("training") if isinstance(row.get("training"), dict) else {}
    training_status = str(training.get("status") or "")
    return (
        status in ORPHAN_RUNNING_ROW_STATUSES
        or training_status in ORPHAN_RUNNING_TRAINING_STATUSES
    )


def _stored_command_forbids_resume(command: Any) -> bool:
    if not isinstance(command, list) or not all(isinstance(item, str) for item in command):
        return True
    return any(item in {"--no_resume", "--max_runtime_minutes"} for item in command)


def reconcile_orphaned_training_rows(
    manifest: dict[str, Any],
    *,
    output_root: Path,
    live_writer_pids: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Reclassify dead RUNNING / STALE rows to proof-bearing paused statuses.

    This is not the one-shot battery reconciler. It never discards a stored
    command, output directory, or locked input hash. A live writer PID, a
    ``--no_resume`` command, or missing fold/progress proof stays blocked.
    """

    live_pids = {int(pid) for pid in (live_writer_pids or ())}
    reconciled: list[dict[str, Any]] = []
    blocked: list[dict[str, Any]] = []
    models = manifest.get("models")
    if not isinstance(models, list):
        return {
            "schema_version": 1,
            "reason": "STALE_RUNNING_RECONCILED_NO_LIVE_WRITER",
            "reconciled": reconciled,
            "blocked": blocked,
        }
    for row in models:
        if not isinstance(row, dict) or not _row_looks_orphaned_running(row):
            continue
        backbone = str(row.get("backbone") or "")
        training = row.get("training")
        if not isinstance(training, dict):
            training = {}
            row["training"] = training
        previous_status = str(row.get("status") or "")
        previous_training_status = str(training.get("status") or "")
        stored_command = (
            list(training["command"]) if isinstance(training.get("command"), list) else None
        )
        try:
            recorded_pid = int(training["pid"]) if training.get("pid") is not None else None
        except (TypeError, ValueError):
            recorded_pid = None
        if recorded_pid is not None and recorded_pid in live_pids:
            row["status"] = "BLOCKED_STALE_RUNNING"
            training.update(
                status="BLOCKED_STALE_RUNNING",
                reason="LIVE_WRITER_PID_STILL_ACTIVE",
                updated_at_utc=utc_now(),
            )
            blocked.append(
                {
                    "backbone": backbone,
                    "reason": "LIVE_WRITER_PID_STILL_ACTIVE",
                    "pid": recorded_pid,
                }
            )
            continue
        if _stored_command_forbids_resume(training.get("command")):
            row["status"] = "BLOCKED_STALE_RUNNING"
            training.update(
                status="BLOCKED_STALE_RUNNING",
                reason="ORPHAN_STORED_COMMAND_RESUME_POLICY_INVALID",
                updated_at_utc=utc_now(),
            )
            if stored_command is not None:
                training["command"] = stored_command
            blocked.append(
                {
                    "backbone": backbone,
                    "reason": "ORPHAN_STORED_COMMAND_RESUME_POLICY_INVALID",
                }
            )
            continue
        try:
            layout = infer_paused_fold_layout(output_root=output_root, row=row)
            paused_status = (
                f"PAUSED_RESUMABLE_{int(layout['current_fold'])}_OF_{LOCKED_FOLD_COUNT}"
            )
            row["status"] = paused_status
            training["status"] = paused_status
            _upsert_resumable_evidence(manifest, backbone=backbone, layout=layout)
            validate_resumable_checkpoint_evidence(
                output_root=output_root, manifest=manifest, row=row
            )
        except (OSError, RuntimeError, ValueError, TypeError, json.JSONDecodeError) as exc:
            row["status"] = "BLOCKED_STALE_RUNNING"
            training.update(
                status="BLOCKED_STALE_RUNNING",
                reason=str(exc).split(":", 1)[0],
                updated_at_utc=utc_now(),
            )
            if stored_command is not None:
                training["command"] = stored_command
            blocked.append({"backbone": backbone, "reason": str(exc).split(":", 1)[0]})
            continue
        if stored_command is not None:
            training["command"] = stored_command
        training.update(
            status=paused_status,
            reason="STALE_RUNNING_RECONCILED_NO_LIVE_WRITER",
            previous_status=previous_status,
            previous_training_status=previous_training_status,
            reconciled_at_utc=utc_now(),
            updated_at_utc=utc_now(),
        )
        reconciled.append(
            {
                "backbone": backbone,
                "status": paused_status,
                "current_fold": int(layout["current_fold"]),
                "previous_status": previous_status,
            }
        )
    if reconciled:
        manifest["status"] = "PAUSED_RESUMABLE"
        journal = manifest.setdefault(
            "orphan_running_reconciliation", {"schema_version": 1, "events": []}
        )
        if isinstance(journal, dict):
            events = journal.setdefault("events", [])
            if isinstance(events, list):
                events.append(
                    {
                        "created_at_utc": utc_now(),
                        "reason": "STALE_RUNNING_RECONCILED_NO_LIVE_WRITER",
                        "reconciled": reconciled,
                        "blocked": blocked,
                    }
                )
    return {
        "schema_version": 1,
        "reason": "STALE_RUNNING_RECONCILED_NO_LIVE_WRITER",
        "reconciled": reconciled,
        "blocked": blocked,
    }


def normalize_fail_closed_supervision_report(
    report: Mapping[str, Any],
) -> dict[str, Any]:
    """Turn malformed, unknown, or internally inconsistent supervision into BLOCK."""

    materialized = copy.deepcopy(dict(report))
    layers = materialized.get("layers")
    automation = materialized.get("automation")
    unsafe: set[str] = set()
    if not isinstance(layers, list) or not isinstance(automation, dict):
        unsafe.add("SUPERVISION_REPORT_SCHEMA_INVALID")
        layers = layers if isinstance(layers, list) else []
        automation = automation if isinstance(automation, dict) else {}
    names: list[str] = []
    observed_block = False
    observed_pause = False
    allowed_statuses = {"PASS", "WARN", "PAUSE", "BLOCK"}
    for layer in layers:
        if not isinstance(layer, dict):
            unsafe.add("SUPERVISION_LAYER_SCHEMA_INVALID")
            continue
        name = str(layer.get("name") or "UNNAMED")
        status = str(layer.get("status") or "")
        findings = layer.get("findings")
        names.append(name)
        if status not in allowed_statuses:
            unsafe.add(
                f"SUPERVISION_LAYER_STATUS_NOT_SAFE:{name}:{status or 'EMPTY'}"
            )
            continue
        if status in {"BLOCK", "PAUSE"} and not (
            isinstance(findings, list) and findings
        ):
            unsafe.add(f"SUPERVISION_LAYER_{status}_WITHOUT_REASON:{name}")
        observed_block = observed_block or status == "BLOCK"
        observed_pause = observed_pause or status == "PAUSE"
    if len(names) != len(set(names)):
        unsafe.add("SUPERVISION_LAYER_NAME_DUPLICATED")
    required = set(REQUIRED_SUPERVISION_LAYERS)
    if set(names) != required:
        unsafe.add("REQUIRED_SUPERVISION_LAYER_SET_INVALID")
    if (
        automation.get("human_approval_required") is not False
        or automation.get("manual_override_allowed") is not False
        or automation.get("fail_closed") is not True
        or set(automation.get("required_layers") or []) != required
    ):
        unsafe.add("AUTONOMOUS_SUPERVISION_POLICY_INVALID")
    decision = str(materialized.get("decision") or "")
    if decision not in {"CONTINUE", "PAUSE_RESUMABLE", "BLOCK", "COMPLETE"}:
        unsafe.add("SUPERVISION_DECISION_INVALID")
    if observed_block and decision != "BLOCK":
        unsafe.add("SUPERVISION_BLOCK_DECISION_MISMATCH")
    if observed_pause and not observed_block and decision != "PAUSE_RESUMABLE":
        unsafe.add("SUPERVISION_PAUSE_DECISION_MISMATCH")
    if unsafe:
        existing = materialized.get("blocking_reasons")
        reasons = {
            str(value)
            for value in (existing if isinstance(existing, list) else [])
        }
        reasons.update(unsafe)
        materialized["decision"] = "BLOCK"
        materialized["blocking_reasons"] = sorted(reasons)
        automation["next_action"] = "WRITE_BLOCKER_AND_STOP"
        materialized["automation"] = automation
    return materialized


def _autonomous_gate(
    *,
    output_root: Path,
    config: dict[str, Any],
    phase: str,
    expected_pid: int | None = None,
) -> dict[str, Any]:
    """Run the mandatory machine supervisor and persist its decision.

    There is intentionally no bypass flag.  A BLOCK is raised to the caller;
    a PAUSE is returned so the existing resumable return-code contract can
    preserve the last committed optimizer step.
    """

    evaluated = evaluate_autonomous_supervision(
        root=ROOT,
        output_root=output_root,
        phase=phase,
        config=config,
        expected_pid=expected_pid,
    )
    report = normalize_fail_closed_supervision_report(
        apply_continuation_recovery_compatibility(
            report=evaluated, output_root=output_root
        )
    )
    enforce_autonomous_supervision(root=ROOT, output_root=output_root, report=report)
    if report.get("decision") == "BLOCK":
        raise RuntimeError(
            "AUTOSUPERVISOR_BLOCKED:" + json.dumps(report.get("blocking_reasons", []), sort_keys=True)
        )
    return report


def _is_receipt_bound_paused_manifest(
    *, output_root: Path, manifest: Mapping[str, Any]
) -> bool:
    """Return true only for the byte-exact reconciled continuation manifest.

    The external receipt binds the entire manifest. Preflight may validate this
    state and write separate reports, but it must not refresh telemetry inside
    the bound file before the exact-resume gate consumes the receipt.
    """

    from src.program_governance.method_x_live_reconciliation import (
        _completed_receipt,
    )

    receipt = _completed_receipt(output_root, manifest)
    return receipt is not None and receipt.get("status") == "ALREADY_RECONCILED"


def prepare_manifest(
    *, config: dict[str, Any], config_path: Path, output_root: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    from scripts.orchestrate_method_x_mri_5fold import (
        active_method_x_processes,
        cached_file_sha256,
        sha256_file,
        validate_cache_contract,
        validate_cache_split_membership,
        validate_split_contract,
    )

    active = active_method_x_processes()
    if active:
        raise RuntimeError(f"ACTIVE_METHOD_X_PROCESS_BLOCKS_LAUNCH:{active}")
    discovered = discover_method_x_processes()
    if discovered:
        raise RuntimeError(f"ACTIVE_METHOD_X_PROCESS_BLOCKS_LAUNCH:{discovered}")
    config_report = validate_matched_backbone_grid(config)
    cache_report = validate_cache_contract(ROOT / CACHE_REL)
    split_report = validate_split_contract(ROOT / SPLITS_REL)
    device_policy = yaml.safe_load((ROOT / DEVICE_POLICY_REL).read_text(encoding="utf-8"))
    hard_stop = float(device_policy["vram"]["hard_stop_gb"])
    if hard_stop != float(config["hardware"]["hard_stop_vram_gb"]):
        raise RuntimeError("BATTERY_DEVICE_POLICY_HARD_STOP_DRIFT")
    output_root.joinpath("locked").mkdir(parents=True, exist_ok=True)
    locked = {
        "cache_sha256": cached_file_sha256(
            ROOT / CACHE_REL, output_root / "locked" / "cache_sha256_registry.json"
        ),
        "split_sha256": str(split_report["sha256"]),
        "config_sha256": sha256_file(config_path),
        "device_policy_sha256": sha256_file(ROOT / DEVICE_POLICY_REL),
    }
    membership = load_cached_lineage_membership(
        cache_path=ROOT / CACHE_REL,
        split_sha256=str(locked["split_sha256"]),
        cache_sha256=str(locked["cache_sha256"]),
    )
    if membership is None:
        membership = validate_cache_split_membership(ROOT / CACHE_REL, ROOT / SPLITS_REL)
    manifest_path = output_root / MANIFEST_NAME
    receipt_bound = False
    if manifest_path.is_file():
        manifest = read_json(manifest_path)
        assert_manifest_compatible(manifest, **locked)
        receipt_bound = _is_receipt_bound_paused_manifest(
            output_root=output_root, manifest=manifest
        )
    else:
        manifest = initialize_manifest(
            config=config,
            output_root=output_root,
            cache_characterization=load_cache_characterization(),
            **locked,
        )
    working_manifest = copy.deepcopy(manifest) if receipt_bound else manifest
    if membership.get("source"):
        working_manifest["lineage_lock"] = {
            "status": membership.get("status"),
            "source": membership.get("source"),
            "audit_sha256": membership.get("audit_sha256"),
            "cache_sha256": membership.get("cache_sha256"),
            "split_sha256": membership.get("split_sha256"),
        }
        if not receipt_bound:
            atomic_write_json(manifest_path, working_manifest)
    # A forced stop, supervisor kill, or reboot can leave RUNNING / STALE
    # labels even though no writer is alive. Reclassify from fold-state proof;
    # never discard the stored command, output directory, or run contract.
    orphan_report = reconcile_orphaned_training_rows(
        working_manifest,
        output_root=output_root,
        live_writer_pids=(),
    )
    cohort_confound = bool(split_report.get("cohort_label_confound", False))
    spm_snapshot = verify_spm_gate_snapshot(
        ROOT / DEFAULT_SNAPSHOT_REL,
        root=ROOT,
        verify_sources=True,
    )
    if not spm_snapshot.get("verified") or not spm_snapshot.get("ready"):
        raise RuntimeError(
            "SPM_GATE_SNAPSHOT_NOT_VERIFIED:" + json.dumps(spm_snapshot, sort_keys=True)
        )
    spm_snapshot_sha256 = assert_spm_manifest_binding(working_manifest, spm_snapshot)
    working_manifest["spm_gate_snapshot"] = {
        "status": spm_snapshot.get("status"),
        "ready": True,
        "verified": True,
        "path": str((ROOT / DEFAULT_SNAPSHOT_REL).relative_to(ROOT)),
        "snapshot_sha256": spm_snapshot_sha256,
        "guard_status": (spm_snapshot.get("guard_status") or {}).get("status"),
    }
    working_manifest["science_gate_status"] = (
        "BLOCKED_COHORT_LABEL_CONFOUND_DIAGNOSTIC_CLAIMS"
        if cohort_confound
        else "BLOCKED_PENDING_INDEPENDENT_SCIENCE_GATE"
    )
    working_manifest["science_audit"] = {
        "cohort_label_confound": cohort_confound,
        "diagnostic_claim_allowed": False,
        "technical_benchmark_allowed": True,
        "policy": "retain_internal_technical_comparison_only; do_not_interpret_as_diagnostic_gain",
    }
    if receipt_bound:
        changed = sorted(
            str(key)
            for key in set(manifest) | set(working_manifest)
            if manifest.get(key) != working_manifest.get(key)
        )
        if changed:
            raise RuntimeError(
                "RECEIPT_BOUND_MANIFEST_STATIC_DRIFT:" + ",".join(changed)
            )
    else:
        working_manifest["updated_at_utc"] = utc_now()
        atomic_write_json(manifest_path, working_manifest)
    autosupervision = _autonomous_gate(
        output_root=output_root,
        config=config,
        phase="preflight",
    )
    if not receipt_bound:
        working_manifest["autosupervision"] = {
            "mode": autosupervision["automation"]["mode"],
            "human_approval_required": False,
            "manual_override_allowed": False,
            "decision": autosupervision["decision"],
            "event_id": autosupervision["event_id"],
        }
        working_manifest["updated_at_utc"] = utc_now()
        atomic_write_json(manifest_path, working_manifest)
    manifest = working_manifest
    report = {
        "schema_version": 1,
        "status": "PASS",
        "training_started": False,
        "process_guard": {"status": "PASS", "active": []},
        "matched_config": config_report,
        "cache_contract": cache_report,
        "split_contract": split_report,
        "cache_split_membership": membership,
        "locked_inputs": locked,
        "cache_characterization": manifest["cache_characterization"],
        "lineage_lock": manifest.get("lineage_lock"),
        "spm_gate_snapshot": manifest["spm_gate_snapshot"],
        "autosupervision": autosupervision,
        "hard_stop_vram_gb": hard_stop,
        "final_claim_allowed": False,
        "science_audit": manifest["science_audit"],
        "orphan_running_reconciliation": orphan_report,
        "stale_training_rows_blocked": [
            item.get("backbone") for item in orphan_report.get("blocked", [])
        ],
    }
    atomic_write_json(output_root / "reports" / "matched_benchmark_preflight.json", report)
    return manifest, report


def run_smoke_sequence(
    manifest: dict[str, Any],
    *,
    output_root: Path,
    device_name: str,
    resolution: int,
    hard_stop_vram_gb: float,
) -> None:
    allowed_previous = {"STATIC_PASS", "PASS", "BLOCKED_VRAM_CAP", "BLOCKED_SMOKE_FAILED", "BLOCKED_DEPENDENCY"}
    rows = manifest["models"]
    for index, row in enumerate(rows):
        if any(previous.get("smoke", {}).get("status") not in allowed_previous for previous in rows[:index]):
            raise RuntimeError("PREVIOUS_MODEL_SMOKE_NOT_TERMINAL")
        existing = row.get("smoke", {})
        official_requested = device_name == "cuda" and resolution == 128
        official_memory_instrumented = (
            existing.get("max_memory_allocated_gb") is not None
            and existing.get("max_memory_reserved_gb") is not None
        )
        if existing.get("status") in allowed_previous and (
            not official_requested
            or existing.get("status", "").startswith("BLOCKED")
            or (
                existing.get("official_128_vram_gate") is True
                and official_memory_instrumented
            )
        ):
            continue
        row["status"] = "SMOKE_RUNNING"
        row["smoke"] = {"status": "RUNNING", "official_128_vram_gate": False}
        manifest["updated_at_utc"] = utc_now()
        atomic_write_json(output_root / MANIFEST_NAME, manifest)
        report = smoke_backbone(
            row["backbone"],
            device_name=device_name,
            resolution=resolution,
            hard_stop_vram_gb=hard_stop_vram_gb,
        )
        row["smoke"] = report
        if report["status"] == "STATIC_PASS":
            row["status"] = "SMOKE_STATIC_PASS"
        elif report["status"] == "PASS":
            row["status"] = "SMOKE_PASS"
        else:
            row["status"] = report["status"]
        atomic_write_json(output_root / row["output_dir"] / "smoke.json", report)
        manifest["updated_at_utc"] = utc_now()
        atomic_write_json(output_root / MANIFEST_NAME, manifest)


def write_comparison(output_root: Path, manifest: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    from scripts.train_3d_mri import method_x_split_contract_digest

    metrics: dict[str, dict[str, Any]] = {}
    locked = manifest["locked_inputs"]
    expected_training_split_digest = method_x_split_contract_digest(
        ROOT / SPLITS_REL, int(config["folds"]), selection_only=False
    )
    for row in manifest["models"]:
        if row.get("status") == "COMPLETED":
            path = output_root / row["output_dir"] / "three_d_metrics.json"
            metrics[row["backbone"]] = validate_model_metrics_for_row(
                path,
                backbone=row["backbone"],
                config=config,
                cache_sha256=locked["cache_sha256"],
                split_sha256=locked["split_sha256"],
                expected_training_split_digest=expected_training_split_digest,
                expected_spm_snapshot_sha256=str(
                    manifest.get("spm_gate_snapshot", {}).get("snapshot_sha256") or ""
                ),
            )
    report = build_comparison_report(manifest, metrics)
    atomic_write_json(output_root / COMPARISON_JSON, report)
    lines = [
        "# Method X matched model battery\n",
        f"- Status: `{report['status']}`\n",
        "- Scope: PET-free internal technical benchmark\n",
        "- Final claim allowed: `false`\n",
        f"- CI method: `{report['ci_method']}`\n",
        "\n| Backbone | Status | ROC-AUC | PR-AUC | Brier |\n",
        "|---|---|---:|---:|---:|\n",
    ]
    for row in report["models"]:
        lines.append(
            f"| {row['backbone']} | {row['status']} | {row.get('roc_auc', '')} | {row.get('pr_auc', '')} | {row.get('brier', '')} |\n"
        )
    path = output_root / COMPARISON_MD
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(lines), encoding="utf-8", newline="\n")
    return report


def _validation_context(
    manifest: dict[str, Any], config: dict[str, Any]
) -> dict[str, str]:
    from scripts.train_3d_mri import method_x_split_contract_digest

    locked = manifest.get("locked_inputs")
    if not isinstance(locked, dict):
        raise RuntimeError("BATTERY_LOCKED_INPUTS_MISSING")
    spm = manifest.get("spm_gate_snapshot")
    if not isinstance(spm, dict):
        raise RuntimeError("BATTERY_SPM_SNAPSHOT_BINDING_MISSING")
    return {
        "cache_sha256": str(locked.get("cache_sha256") or ""),
        "split_sha256": str(locked.get("split_sha256") or ""),
        "training_split_digest": method_x_split_contract_digest(
            ROOT / SPLITS_REL, int(config["folds"]), selection_only=False
        ),
        "spm_snapshot_sha256": str(spm.get("snapshot_sha256") or ""),
    }


def _validate_completed_row(
    *,
    row: dict[str, Any],
    output_root: Path,
    config: dict[str, Any],
    context: dict[str, str],
) -> dict[str, Any]:
    metrics_path = output_root / str(row.get("output_dir") or "") / "three_d_metrics.json"
    return validate_model_metrics_for_row(
        metrics_path,
        backbone=str(row.get("backbone") or ""),
        config=config,
        cache_sha256=context["cache_sha256"],
        split_sha256=context["split_sha256"],
        expected_training_split_digest=context["training_split_digest"],
        expected_spm_snapshot_sha256=context["spm_snapshot_sha256"],
    )


def _write_scheduler_manifest(output_root: Path, manifest: dict[str, Any]) -> None:
    manifest["updated_at_utc"] = utc_now()
    path = output_root / MANIFEST_NAME
    call_with_windows_file_lock_retry(lambda: atomic_write_json(path, manifest))


def _execute_model_row(
    *,
    manifest: dict[str, Any],
    row: dict[str, Any],
    output_root: Path,
    config: dict[str, Any],
    context: dict[str, str],
    command: Sequence[str],
) -> int:
    """Execute exactly one row; every non-completion result stops the caller."""

    from scripts.orchestrate_method_x_mri_5fold import active_method_x_processes

    hard_stop = float(config["hardware"]["hard_stop_vram_gb"])
    before_model_gate = _autonomous_gate(
        output_root=output_root, config=config, phase="before_model"
    )
    if (
        before_model_gate.get("decision") == "PAUSE_RESUMABLE"
        and not exact_resume_gate_authorizes_row(
            before_model_gate, row, output_root=output_root
        )
    ):
        return 75
    if active_method_x_processes():
        raise RuntimeError("ACTIVE_METHOD_X_PROCESS_BLOCKS_LAUNCH")
    if row.get("smoke", {}).get("official_128_vram_gate") is not True:
        smoke = smoke_backbone(
            str(row["backbone"]),
            device_name="cuda",
            resolution=128,
            hard_stop_vram_gb=hard_stop,
        )
        row["smoke"] = smoke
        atomic_write_json(output_root / str(row["output_dir"]) / "smoke.json", smoke)
        if smoke.get("status") != "PASS":
            row["status"] = str(smoke.get("status") or "BLOCKED_SMOKE_FAILED")
            row.setdefault("training", {})["status"] = "NOT_STARTED"
            manifest["status"] = "COMPLETED_WITH_BLOCKS"
            _write_scheduler_manifest(output_root, manifest)
            return 3

    training = row.get("training")
    if not isinstance(training, dict):
        training = {}
        row["training"] = training
    stored = training.get("command")
    safe_cmd = _safe_command(list(command))
    if stored is None:
        training["command"] = safe_cmd
    elif (
        not isinstance(stored, list)
        or not all(isinstance(value, str) for value in stored)
        or _safe_command(list(stored)) != list(safe_cmd)
    ):
        raise RuntimeError("STORED_TRAINING_COMMAND_DRIFT")
    history = training.setdefault("attempts", [])
    if not isinstance(history, list):
        raise RuntimeError("CHILD_ATTEMPT_HISTORY_INVALID")
    row["status"] = "TRAINING_RUNNING"
    manifest["training_started"] = True
    training.update(
        status="RUNNING",
        cache_sha256=context["cache_sha256"],
        split_sha256=context["split_sha256"],
        spm_snapshot_sha256=context["spm_snapshot_sha256"],
        started_at_utc=training.get("started_at_utc") or utc_now(),
        latest_attempt_started_at_utc=utc_now(),
    )
    _write_scheduler_manifest(output_root, manifest)
    supervision_policy = load_supervision_policy(ROOT)

    def _watch_child(child_pid: int) -> dict[str, Any]:
        first_report = evaluate_autonomous_supervision(
            root=ROOT,
            output_root=output_root,
            phase="watch",
            config=config,
            expected_pid=child_pid,
            policy=supervision_policy,
        )

        def _retry_watch() -> dict[str, Any]:
            time.sleep(WATCH_INVENTORY_RACE_RETRY_SECONDS)
            return evaluate_autonomous_supervision(
                root=ROOT,
                output_root=output_root,
                phase="watch",
                config=config,
                expected_pid=child_pid,
                policy=supervision_policy,
            )

        watch_report, should_enforce = resolve_watch_inventory_race(
            first_report=first_report,
            retry_probe=_retry_watch,
        )
        if should_enforce:
            enforce_autonomous_supervision(
                root=ROOT, output_root=output_root, report=watch_report
            )
        return watch_report

    def _record_child_pid(child_pid: int) -> None:
        authorization = training.get("retry_authorization")
        if isinstance(authorization, dict) and authorization.get("status") == "PENDING":
            authorization["status"] = "CONSUMED"
            authorization["consumed_at_utc"] = utc_now()
            authorization["consumed_by_pid"] = int(child_pid)
        row["status"] = "TRAINING_RUNNING"
        training["status"] = "RUNNING"
        training["pid"] = int(child_pid)
        training["owner_recorded_at_utc"] = utc_now()
        _write_scheduler_manifest(output_root, manifest)

    def _retry_authorization_valid(attempt: int, retry_command: Sequence[str]) -> bool:
        authorization = training.get("retry_authorization")
        if not isinstance(authorization, dict) or not history:
            return False
        previous = history[-1]
        previous_failure = previous.get("failure") if isinstance(previous, dict) else None
        return bool(
            authorization.get("status") == "PENDING"
            and authorization.get("category") == "TRANSIENT_IO"
            and authorization.get("next_attempt") == int(attempt)
            and authorization.get("command_sha256")
            == _canonical_payload_sha256([str(value) for value in retry_command])
            and isinstance(previous_failure, dict)
            and previous_failure.get("retryable") is True
            and previous_failure.get("category") == "TRANSIENT_IO"
            and authorization.get("failure_fingerprint_sha256")
            == previous_failure.get("fingerprint_sha256")
        )

    def _attempt_runner(
        *,
        command: list[str],
        stdout_log_path: Path,
        stderr_log_path: Path,
        attempt: int,
    ) -> dict[str, Any]:
        if attempt > 1:
            if not _retry_authorization_valid(attempt, command):
                raise RuntimeError("TRANSIENT_IO_RETRY_AUTHORIZATION_INVALID")
            retry_gate = _autonomous_gate(
                output_root=output_root, config=config, phase="before_retry"
            )
            pause_reasons = set(retry_gate.get("blocking_reasons") or [])
            if retry_gate.get("decision") == "PAUSE_RESUMABLE" and pause_reasons != {
                "RESUMABLE_PAUSED_ROW"
            }:
                return {
                    "returncode": 75,
                    "decision": "PAUSE_RESUMABLE",
                    "reason": "AUTOSUPERVISOR_PAUSED_BEFORE_RETRY",
                }
            if active_method_x_processes():
                raise RuntimeError("ACTIVE_METHOD_X_PROCESS_BLOCKS_RETRY")
        training["retry_attempt"] = attempt - 1
        training["latest_attempt_started_at_utc"] = utc_now()
        return run_supervised_process(
            [*command, *_RUNTIME_STAGING_OVERLAY],
            cwd=ROOT,
            output_root=output_root,
            interval_seconds=float(supervision_policy.get("poll_seconds", 15)),
            max_idle_seconds=float(
                supervision_policy.get("max_idle_seconds", 3600)
            ),
            check=_watch_child,
            on_start=_record_child_pid,
            stdout_log_path=stdout_log_path,
            stderr_log_path=stderr_log_path,
        )

    def _observe_attempt(
        attempt_record: dict[str, Any], supervised: dict[str, Any]
    ) -> None:
        attempt_record["completed_at_utc"] = utc_now()
        history.append(attempt_record)
        training["last_attempt"] = attempt_record
        training["last_attempt_supervision"] = supervised
        failure_record = attempt_record.get("failure")
        attempt_number = int(attempt_record.get("attempt", 0))
        if (
            isinstance(failure_record, dict)
            and failure_record.get("category") == "TRANSIENT_IO"
            and failure_record.get("retryable") is True
            and attempt_number <= MAX_TRANSIENT_IO_RETRIES
        ):
            row["status"] = "TRANSIENT_IO_RETRY_PENDING"
            training["status"] = "TRANSIENT_IO_RETRY_PENDING"
            training["retry_authorization"] = {
                "schema_version": 1,
                "status": "PENDING",
                "category": "TRANSIENT_IO",
                "next_attempt": attempt_number + 1,
                "command_sha256": _canonical_payload_sha256(
                    [str(value) for value in command]
                ),
                "failure_fingerprint_sha256": failure_record.get(
                    "fingerprint_sha256"
                ),
                "authorized_at_utc": utc_now(),
            }
        _write_scheduler_manifest(output_root, manifest)

    execution = run_child_with_bounded_io_retry(
        command,
        backbone=str(row["backbone"]),
        output_root=output_root,
        attempt_runner=_attempt_runner,
        attempt_observer=_observe_attempt,
    )
    returncode = int(execution["returncode"])
    supervised = execution["supervised"]
    failure = execution["failure"]
    training["autosupervision"] = supervised
    training["last_failure"] = failure
    training["retry_count"] = int(execution["retry_count"])
    training["retry_exhausted"] = bool(execution["retry_exhausted"])
    metrics_path = output_root / str(row["output_dir"]) / "three_d_metrics.json"
    observed_status = None
    if returncode == 0 and metrics_path.is_file():
        try:
            observed_status = load_metrics_with_reconciliation_sidecar(metrics_path).get(
                "aggregate", {}
            ).get("status")
        except (OSError, ValueError, RuntimeError, json.JSONDecodeError):
            observed_status = None
        if classify_training_returncode(returncode, observed_status) == "PAUSE_RESUMABLE":
            returncode = 75
    lifecycle = resolve_child_lifecycle(
        returncode=returncode,
        observed_status=observed_status,
        failure=failure,
    )

    if lifecycle == "BLOCK_RUN_CONTRACT":
        row["status"] = "BLOCKED_RUN_CONTRACT_MISMATCH"
        training.update(
            status="BLOCKED_RUN_CONTRACT_MISMATCH",
            returncode=returncode,
            phase="resume_contract_validation",
            updated_at_utc=utc_now(),
        )
        manifest["status"] = "COMPLETED_WITH_BLOCKS"
        _write_scheduler_manifest(output_root, manifest)
        atomic_write_json(
            output_root / str(row["output_dir"]) / "BLOCKED_RUN_CONTRACT_MISMATCH.json",
            {
                "status": "BLOCKED_RUN_CONTRACT_MISMATCH",
                "backbone": row["backbone"],
                "returncode": returncode,
                "failure": failure,
                "attempts": execution["attempts"],
                "action": "Run scripts/diff_method_x_contract.py; migrate source-only changes or create a new versioned run.",
            },
        )
        return 3
    if lifecycle == "PAUSE_RESUMABLE":
        row["status"] = "TRAINING_PAUSED_RESUMABLE"
        training.update(
            status="PAUSED_RESUMABLE",
            returncode=returncode,
            phase="training",
            updated_at_utc=utc_now(),
        )
        _write_scheduler_manifest(output_root, manifest)
        return 75
    if lifecycle == "BLOCK_EXECUTION":
        row["status"] = "BLOCKED_TRAINING_EXECUTION"
        reason = (
            "TRANSIENT_IO_RETRY_EXHAUSTED"
            if execution["retry_exhausted"]
            else supervised.get("reason") or "NONZERO_CHILD_EXIT"
        )
        training.update(
            status="BLOCKED_TRAINING_EXECUTION",
            returncode=returncode,
            phase="training",
            decision=supervised.get("decision"),
            reason=reason,
            updated_at_utc=utc_now(),
        )
        manifest["status"] = "COMPLETED_WITH_BLOCKS"
        _write_scheduler_manifest(output_root, manifest)
        atomic_write_json(
            output_root / str(row["output_dir"]) / "BLOCKED_TRAINING_EXECUTION.json",
            {
                "status": "BLOCKED_TRAINING_EXECUTION",
                "backbone": row["backbone"],
                "returncode": returncode,
                "decision": supervised.get("decision"),
                "reason": reason,
                "failure": failure,
                "attempts": execution["attempts"],
                "action": "Inspect the H-side child logs and resume only with the exact unchanged run contract.",
            },
        )
        return 3
    try:
        _validate_completed_row(
            row=row,
            output_root=output_root,
            config=config,
            context=context,
        )
    except (OSError, ValueError, RuntimeError, KeyError) as exc:
        row["status"] = "BLOCKED_POST_TRAINING_VALIDATION"
        training.update(
            status="BLOCKED_POST_TRAINING_VALIDATION",
            phase="post_training_validation",
            blocker=str(exc),
            updated_at_utc=utc_now(),
        )
        manifest["status"] = "COMPLETED_WITH_BLOCKS"
        _write_scheduler_manifest(output_root, manifest)
        atomic_write_json(
            output_root / str(row["output_dir"]) / "BLOCKED_POST_TRAINING_VALIDATION.json",
            {
                "status": "BLOCKED_POST_TRAINING_VALIDATION",
                "backbone": row["backbone"],
                "blocker": str(exc),
            },
        )
        return 3
    row["status"] = "COMPLETED"
    training.update(
        status="COMPLETED",
        completed_at_utc=utc_now(),
        metrics=metrics_path.name,
    )
    _write_scheduler_manifest(output_root, manifest)
    after_model_gate = _autonomous_gate(
        output_root=output_root, config=config, phase="after_model"
    )
    training["after_model_supervision"] = {
        "decision": after_model_gate["decision"],
        "event_id": after_model_gate.get("event_id"),
    }
    _write_scheduler_manifest(output_root, manifest)
    return 0


def run_targeted_resume(
    manifest: dict[str, Any],
    *,
    output_root: Path,
    config: dict[str, Any],
    backbone: str,
) -> tuple[int, dict[str, Any]]:
    rows = effective_model_rows(manifest, output_root=output_root)
    positions = {str(row.get("backbone")): index for index, row in enumerate(rows)}
    all_rows = {
        str(row.get("backbone")): row
        for row in manifest.get("models", [])
        if isinstance(row, dict)
    }
    if backbone in PROTOCOL_EXCLUDED_BACKBONES:
        raise RuntimeError(f"TARGET_BACKBONE_PROTOCOL_EXCLUDED:{backbone}")
    if backbone not in positions or backbone not in all_rows:
        raise RuntimeError(f"TARGET_BACKBONE_NOT_EFFECTIVE:{backbone}")
    index = positions[backbone]
    assert_training_sequence_ready(rows, index)
    context = _validation_context(manifest, config)
    for previous in rows[:index]:
        if previous.get("status") == "COMPLETED":
            _validate_completed_row(
                row=previous,
                output_root=output_root,
                config=config,
                context=context,
            )
    row = rows[index]
    if row.get("status") == "COMPLETED":
        _validate_completed_row(
            row=row, output_root=output_root, config=config, context=context
        )
        if later_models_need_execution(rows, index):
            sequence_code = run_training_sequence(
                manifest, output_root=output_root, config=config
            )
            return sequence_code, {
                "status": "ALREADY_COMPLETED",
                "backbone": backbone,
                "sequence_continued": True,
            }
        return 0, {"status": "ALREADY_COMPLETED", "backbone": backbone}
    if not _is_paused_resumable_status(row.get("status")):
        raise RuntimeError(f"TARGET_BACKBONE_NOT_RESUMABLE:{backbone}:{row.get('status')}")
    expected_command = _safe_command(
        build_training_command(
            python=sys.executable,
            config=config,
            backbone=str(row["backbone"]),
            output_dir=output_root / str(row["output_dir"]),
            cache_sha256=context["cache_sha256"],
            spm_snapshot_sha256=context["spm_snapshot_sha256"],
        )
    )
    command = validated_stored_training_command(
        row, output_root=output_root, expected_command=expected_command
    )
    code = _execute_model_row(
        manifest=manifest,
        row=row,
        output_root=output_root,
        config=config,
        context=context,
        command=command,
    )
    if code == 0 and row.get("status") != "COMPLETED":
        raise RuntimeError("MODEL_EXECUTOR_ZERO_WITHOUT_COMPLETION")
    if code == 0 and later_models_need_execution(rows, index):
        sequence_code = run_training_sequence(
            manifest, output_root=output_root, config=config
        )
        result_status = (
            "COMPLETED"
            if sequence_code == 0
            else "PAUSED_RESUMABLE"
            if sequence_code == 75
            else str(row.get("status") or "BLOCKED")
        )
        return sequence_code, {
            "status": result_status,
            "backbone": backbone,
            "target_completed": True,
            "sequence_continued": True,
        }
    result_status = (
        "COMPLETED"
        if code == 0
        else "PAUSED_RESUMABLE"
        if code == 75
        else str(row.get("status") or "BLOCKED")
    )
    return code, {"status": result_status, "backbone": backbone}


def run_training_sequence(
    manifest: dict[str, Any], *, output_root: Path, config: dict[str, Any]
) -> int:
    rows = effective_model_rows(manifest, output_root=output_root)
    context = _validation_context(manifest, config)
    for index, row in enumerate(rows):
        assert_training_sequence_ready(rows, index)
        status = str(row.get("status") or "")
        if status == "COMPLETED":
            _validate_completed_row(
                row=row,
                output_root=output_root,
                config=config,
                context=context,
            )
            continue
        if status == "PROTOCOL_EXCLUDED_NOT_STARTED":
            continue
        if status.startswith("BLOCKED") or status == "TRAINING_RUNNING":
            return 3
        if _is_paused_resumable_status(status):
            expected_command = _safe_command(
                build_training_command(
                    python=sys.executable,
                    config=config,
                    backbone=str(row["backbone"]),
                    output_dir=output_root / str(row["output_dir"]),
                    cache_sha256=context["cache_sha256"],
                    spm_snapshot_sha256=context["spm_snapshot_sha256"],
                )
            )
            command = validated_stored_training_command(
                row,
                output_root=output_root,
                expected_command=expected_command,
            )
        elif status in {"PENDING", "SMOKE_PASS", "NOT_STARTED"}:
            command = build_training_command(
                python=sys.executable,
                config=config,
                backbone=str(row["backbone"]),
                output_dir=output_root / str(row["output_dir"]),
                cache_sha256=context["cache_sha256"],
                spm_snapshot_sha256=context["spm_snapshot_sha256"],
            )
        else:
            raise RuntimeError(f"MODEL_STATUS_NOT_SCHEDULABLE:{row.get('backbone')}:{status}")
        code = _execute_model_row(
            manifest=manifest,
            row=row,
            output_root=output_root,
            config=config,
            context=context,
            command=command,
        )
        if code != 0:
            return code
        if row.get("status") != "COMPLETED":
            raise RuntimeError("MODEL_EXECUTOR_ZERO_WITHOUT_COMPLETION")
    if any(
        row.get("status") not in ADVANCEABLE_MODEL_STATUSES
        for row in manifest.get("models", [])
        if isinstance(row, dict)
    ):
        raise RuntimeError("BATTERY_NONADVANCEABLE_ROW_AFTER_SEQUENCE")
    manifest["status"] = "COMPLETED"
    _write_scheduler_manifest(output_root, manifest)
    write_comparison(output_root, manifest, config)
    _autonomous_gate(output_root=output_root, config=config, phase="postflight")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs/experiments/method_x_claim_benchmark_v3.yaml"
    )
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument(
        "--mode",
        choices=["preflight", "smoke", "run", "resume-backbone"],
        default="preflight",
    )
    parser.add_argument("--backbone", default=None)
    parser.add_argument("--smoke-device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--smoke-resolution", type=int, default=32)
    # Checkpoint staging is a runtime overlay, never part of the scientific
    # command. These are applied at launch and are deliberately excluded from
    # the command stored in the manifest, so enabling or disabling staging can
    # never read as scientific drift on a resumed row.
    parser.add_argument(
        "--checkpoint-staging-root",
        default=DEFAULT_CHECKPOINT_STAGING_ROOT,
        help=(
            "Policy-bound NVMe staging root for active resume payloads. The "
            "official command enables it automatically."
        ),
    )
    parser.add_argument(
        "--checkpoint-full-cadence",
        type=int,
        default=1,
        help=(
            "Full staged checkpoints per optimizer step. Cadence 1 is the locked "
            "fallback until deterministic replay equivalence is proven."
        ),
    )
    parser.add_argument(
        "--checkpoint-h-mirror",
        choices=["epoch", "fold", "epoch_and_fold"],
        default="epoch",
        help="When a staged payload is mirrored to the authoritative volume.",
    )
    parser.add_argument(
        "--resume-replay-policy",
        choices=["deterministic"],
        default="deterministic",
        help="Recovery policy for uncommitted steps. Only deterministic replay is permitted.",
    )
    return parser


def staging_overlay_args(args: argparse.Namespace) -> list[str]:
    """Translate the runner's staging flags into trainer arguments.

    Returns an empty list unless a staging root was requested, so the default
    invocation produces a byte-identical command to the one used before Task 4.
    """

    staging_root = str(
        getattr(args, "checkpoint_staging_root", "")
        or DEFAULT_CHECKPOINT_STAGING_ROOT
    ).strip()
    cadence = int(getattr(args, "checkpoint_full_cadence", 1))
    if cadence < 1:
        raise RuntimeError("INVALID_CHECKPOINT_FULL_CADENCE")
    if cadence != 1:
        raise RuntimeError("CHECKPOINT_REPLAY_NOT_AUTHORIZED")
    return [
        "--checkpoint_staging_root", staging_root,
        "--checkpoint_full_cadence", str(cadence),
        "--checkpoint_h_mirror", str(getattr(args, "checkpoint_h_mirror", "epoch")),
        "--resume_replay_policy", str(getattr(args, "resume_replay_policy", "deterministic")),
    ]


def main() -> int:
    args = build_parser().parse_args()
    if (args.mode == "resume-backbone") != bool(args.backbone):
        print(
            json.dumps(
                {
                    "status": "BLOCKED",
                    "blocker": "BACKBONE_ARGUMENT_REQUIRES_RESUME_BACKBONE_MODE",
                },
                indent=2,
            ),
            file=sys.stderr,
        )
        return 3
    try:
        global _RUNTIME_STAGING_OVERLAY
        _RUNTIME_STAGING_OVERLAY = staging_overlay_args(args)
    except RuntimeError as exc:
        print(
            json.dumps({"status": "BLOCKED", "blocker": str(exc)}, indent=2),
            file=sys.stderr,
        )
        return 3
    try:
        config_path, config = _load_cli_config(args.config)
        output_root = _output_root(config, args.output_root)
    except (OSError, RuntimeError, ValueError, KeyError, yaml.YAMLError) as exc:
        print(
            json.dumps(
                {
                    "schema_version": 1,
                    "status": "BLOCKED",
                    "phase": "cli_prewrite_validation",
                    "blocker": str(exc),
                },
                indent=2,
            ),
            file=sys.stderr,
        )
        return 3
    try:
        lock_path = acquire_battery_lock(output_root)
    except RuntimeError as exc:
        print(json.dumps({"status": "BLOCKED", "blocker": str(exc)}, indent=2), file=sys.stderr)
        return 3
    manifest: dict[str, Any] = {}
    try:
        manifest, preflight = prepare_manifest(
            config=config, config_path=config_path, output_root=output_root
        )
        if args.mode == "preflight":
            archive_superseded_preflight_blocker(output_root, preflight=preflight)
            print(json.dumps(preflight, indent=2, sort_keys=True))
            return 0
        if args.mode == "smoke":
            run_smoke_sequence(
                manifest,
                output_root=output_root,
                device_name=args.smoke_device,
                resolution=args.smoke_resolution,
                hard_stop_vram_gb=float(config["hardware"]["hard_stop_vram_gb"]),
            )
            comparison = write_comparison(output_root, manifest, config)
            print(json.dumps({"status": "SMOKE_SEQUENCE_COMPLETE", "comparison": comparison["status"]}, indent=2))
            return 0
        if args.mode == "resume-backbone":
            code, result = run_targeted_resume(
                manifest,
                output_root=output_root,
                config=config,
                backbone=str(args.backbone),
            )
            print(json.dumps(result, indent=2, sort_keys=True))
            return code
        return run_training_sequence(manifest, output_root=output_root, config=config)
    except (OSError, ValueError, RuntimeError, KeyError) as exc:
        blocker = {
            "schema_version": 1,
            "status": "BLOCKED",
            "blocker": str(exc),
            "training_started": any(row.get("training", {}).get("status") in {"RUNNING", "PAUSED_RESUMABLE", "COMPLETED"} for row in manifest.get("models", [])) if isinstance(manifest, dict) else False,
            "phase": "post_training_or_preflight",
            "created_at_utc": utc_now(),
        }
        atomic_write_json(output_root / "reports" / "BLOCKED_PREFLIGHT.json", blocker)
        print(json.dumps(blocker, indent=2), file=sys.stderr)
        return 3
    finally:
        release_battery_lock(lock_path)


if __name__ == "__main__":
    raise SystemExit(main())
