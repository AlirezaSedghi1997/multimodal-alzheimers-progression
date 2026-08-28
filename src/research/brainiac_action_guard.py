"""Shared fail-closed lease excluding Method X CNN and BrainIAC writers."""

from __future__ import annotations

import contextvars
import json
import os
import stat
import subprocess
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from src.research import brainiac_readiness as _readiness
from src.research.brainiac_readiness import (
    DEFAULT_BATTERY_MANIFEST,
    PROJECT_ROOT,
    _load_json_strict,
    _writer_blockers,
    bind_file,
    inspect_writer_state,
)


BATTERY_ROOT = DEFAULT_BATTERY_MANIFEST.parent
SHARED_GUARD_NAME = ".method_x_reconciliation.lock"
BATTERY_LOCK_NAME = ".battery.lock"
_O_BINARY = int(getattr(os, "O_BINARY", 0))


@dataclass(frozen=True)
class BrainIACActionLease:
    action: str
    guard_path: Path
    pid: int
    process_create_time: str
    nonce: str
    ownership_token: str
    raw: bytes


_ACTIVE_LEASE: contextvars.ContextVar[BrainIACActionLease | None] = (
    contextvars.ContextVar("brainiac_active_action_lease", default=None)
)


def _default_process_provider() -> list[dict[str, Any]]:
    """Local indirection keeps writer inspection testable and fail-closed."""

    return _readiness._default_process_provider()


def _default_gpu_process_provider() -> list[dict[str, Any]]:
    """Dynamic indirection preserves both production inspection and test injection."""

    return _readiness._default_gpu_process_provider()


def _regular_nonreparse(path: Path) -> bool:
    try:
        metadata = os.lstat(path)
    except FileNotFoundError:
        return False
    except OSError as exc:
        raise RuntimeError("BRAINIAC_RUNTIME_LOCK_INSPECTION_UNKNOWN") from exc
    return stat.S_ISREG(metadata.st_mode) and not bool(
        int(getattr(metadata, "st_file_attributes", 0))
        & int(getattr(os, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400))
    )


def _pid_identity(pid: int) -> tuple[bool | None, str | None]:
    if pid <= 0:
        return False, None
    try:
        import psutil

        process = psutil.Process(pid)
        live = process.is_running() and process.status() != psutil.STATUS_ZOMBIE
        return live, f"{float(process.create_time()):.6f}" if live else None
    except ImportError:
        pass
    except psutil.NoSuchProcess:
        return False, None
    except Exception:
        return None, None
    if os.name != "nt":
        try:
            os.kill(pid, 0)
            return True, "POSIX_CREATE_TIME_NOT_MEASURABLE"
        except PermissionError:
            return True, "POSIX_CREATE_TIME_NOT_MEASURABLE"
        except ProcessLookupError:
            return False, None
        except OSError:
            return None, None
    query = (
        f"$p=Get-CimInstance Win32_Process -Filter 'ProcessId={int(pid)}' "
        "-ErrorAction Stop; if($null -eq $p){'NOT_FOUND'}else{"
        "$p | Select-Object ProcessId,CreationDate | ConvertTo-Json -Compress}"
    )
    try:
        completed = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", query],
            capture_output=True,
            text=True,
            check=False,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError):
        return None, None
    if completed.returncode != 0:
        return None, None
    raw = completed.stdout.strip()
    if raw == "NOT_FOUND" or not raw:
        return False, None
    try:
        payload = json.loads(raw)
        creation = str(payload.get("CreationDate") or "")
    except (AttributeError, json.JSONDecodeError, TypeError, ValueError):
        return None, None
    return (True, creation) if creation else (None, None)


def _read_lock(path: Path) -> tuple[dict[str, Any], bytes]:
    if not _regular_nonreparse(path):
        raise RuntimeError("BRAINIAC_RUNTIME_LOCK_UNSAFE")
    try:
        raw = path.read_bytes()
        payload = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("BRAINIAC_RUNTIME_LOCK_UNREADABLE") from exc
    if not isinstance(payload, dict):
        raise RuntimeError("BRAINIAC_RUNTIME_LOCK_INVALID")
    return payload, raw


def _recorded_lock_status(path: Path, *, role: str) -> dict[str, Any]:
    if not os.path.lexists(path):
        return {"role": role, "present": False, "liveness": "ABSENT"}
    payload, _ = _read_lock(path)
    try:
        pid = int(payload.get("pid", -1))
    except (TypeError, ValueError) as exc:
        raise RuntimeError("BRAINIAC_RUNTIME_LOCK_INVALID") from exc
    live, creation = _pid_identity(pid)
    if live is None:
        raise RuntimeError("BRAINIAC_RUNTIME_LOCK_LIVENESS_UNKNOWN")
    if live:
        recorded = payload.get("process_create_time")
        if recorded is not None and str(recorded) != str(creation):
            raise RuntimeError("BRAINIAC_RUNTIME_LOCK_PID_REUSE")
    return {
        "role": role,
        "present": True,
        "pid": pid,
        "liveness": "LIVE" if live else "DEAD",
        "process_create_time": creation,
    }


def inspect_action_exclusion(
    *,
    process_provider: Any | None = None,
    gpu_provider: Any | None = None,
) -> dict[str, Any]:
    """Read-only writer/lock inspection. Definitively stale battery lock is retained."""

    if process_provider is None:
        process_provider = _default_process_provider
    if gpu_provider is None:
        gpu_provider = _default_gpu_process_provider
    manifest_binding = bind_file(DEFAULT_BATTERY_MANIFEST, project_root=PROJECT_ROOT)
    battery, _ = _load_json_strict(Path(manifest_binding["path"]))
    writer_state = inspect_writer_state(
        battery,
        process_provider=process_provider,
        gpu_provider=gpu_provider,
    )
    blockers = list(_writer_blockers(writer_state))
    battery_lock = _recorded_lock_status(
        BATTERY_ROOT / BATTERY_LOCK_NAME, role="battery_lock"
    )
    guard = _recorded_lock_status(
        BATTERY_ROOT / SHARED_GUARD_NAME, role="shared_guard"
    )
    if battery_lock.get("liveness") == "LIVE":
        blockers.append("ACTIVE_BATTERY_LOCK_OWNER")
    if guard.get("liveness") == "LIVE":
        blockers.append("ACTIVE_SHARED_GUARD_OWNER")
    if guard.get("present") and guard.get("liveness") == "DEAD":
        blockers.append("STALE_SHARED_GUARD_REQUIRES_RECONCILIATION")
    return {
        "status": "PASS" if not blockers else "BLOCKED",
        "blockers": list(dict.fromkeys(blockers)),
        "writer_state": writer_state,
        "battery_lock": battery_lock,
        "shared_guard": guard,
        "battery_manifest_binding": manifest_binding,
    }


def acquire_action_guard(action: str) -> BrainIACActionLease:
    """Atomically acquire the guard recognized by the Method X battery runner."""

    before = inspect_action_exclusion()
    if before["blockers"]:
        raise RuntimeError("BRAINIAC_ACTION_EXCLUSION_BLOCKED:" + ",".join(before["blockers"]))
    live, creation = _pid_identity(os.getpid())
    if live is not True or not creation:
        raise RuntimeError("BRAINIAC_OWN_PROCESS_IDENTITY_UNKNOWN")
    nonce = uuid.uuid4().hex
    token = uuid.uuid4().hex + uuid.uuid4().hex
    payload = {
        "schema_version": 1,
        "role": "method_x_reconciliation",
        "purpose": "brainiac_action_exclusion",
        "action": str(action),
        "pid": os.getpid(),
        "process_create_time": creation,
        "nonce": nonce,
        "ownership_token": token,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    raw = (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode(
        "utf-8"
    )
    path = BATTERY_ROOT / SHARED_GUARD_NAME
    try:
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY | _O_BINARY)
    except FileExistsError as exc:
        raise RuntimeError("BRAINIAC_ACTION_GUARD_ACQUISITION_RACE") from exc
    try:
        os.write(descriptor, raw)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    lease = BrainIACActionLease(
        action=str(action),
        guard_path=path,
        pid=os.getpid(),
        process_create_time=creation,
        nonce=nonce,
        ownership_token=token,
        raw=raw,
    )
    try:
        # Ignore our own guard, but repeat all process and battery-owner checks.
        payload_now, raw_now = _read_lock(path)
        if raw_now != raw or payload_now.get("ownership_token") != token:
            raise RuntimeError("BRAINIAC_ACTION_GUARD_OWNERSHIP_LOST")
        manifest_binding = bind_file(DEFAULT_BATTERY_MANIFEST, project_root=PROJECT_ROOT)
        battery, _ = _load_json_strict(Path(manifest_binding["path"]))
        writer_state = inspect_writer_state(
            battery,
            process_provider=_default_process_provider,
            gpu_provider=_default_gpu_process_provider,
        )
        blockers = _writer_blockers(writer_state)
        battery_lock = _recorded_lock_status(
            BATTERY_ROOT / BATTERY_LOCK_NAME, role="battery_lock"
        )
        if battery_lock.get("liveness") == "LIVE":
            blockers.append("ACTIVE_BATTERY_LOCK_OWNER")
        if blockers:
            raise RuntimeError(
                "BRAINIAC_ACTION_EXCLUSION_POST_ACQUIRE_BLOCKED:"
                + ",".join(blockers)
            )
        return lease
    except BaseException:
        release_action_guard(lease)
        raise


def release_action_guard(lease: BrainIACActionLease) -> None:
    path = lease.guard_path
    if not os.path.lexists(path):
        raise RuntimeError("BRAINIAC_ACTION_GUARD_RELEASE_EVIDENCE_MISSING")
    if not _regular_nonreparse(path):
        raise RuntimeError("BRAINIAC_ACTION_GUARD_RELEASE_UNSAFE")
    try:
        payload, raw = _read_lock(path)
    except RuntimeError as exc:
        raise RuntimeError("BRAINIAC_ACTION_GUARD_RELEASE_UNREADABLE") from exc
    try:
        owner_matches = (
            raw == lease.raw
            and int(payload.get("pid", -1)) == lease.pid
            and payload.get("nonce") == lease.nonce
            and payload.get("ownership_token") == lease.ownership_token
        )
    except (TypeError, ValueError) as exc:
        raise RuntimeError("BRAINIAC_ACTION_GUARD_RELEASE_OWNERSHIP_INVALID") from exc
    if not owner_matches:
        raise RuntimeError("BRAINIAC_ACTION_GUARD_RELEASE_OWNERSHIP_LOST")
    try:
        path.unlink(missing_ok=True)
    except OSError as exc:
        raise RuntimeError("BRAINIAC_ACTION_GUARD_RELEASE_UNLINK_FAILED") from exc
    if os.path.lexists(path):
        raise RuntimeError("BRAINIAC_ACTION_GUARD_RELEASE_NOT_CONFIRMED")


def activate_lease(lease: BrainIACActionLease) -> contextvars.Token:
    return _ACTIVE_LEASE.set(lease)


def deactivate_lease(token: contextvars.Token) -> None:
    _ACTIVE_LEASE.reset(token)


def require_active_lease(action: str) -> BrainIACActionLease:
    lease = _ACTIVE_LEASE.get()
    if lease is None or lease.action != str(action):
        raise RuntimeError("BRAINIAC_ACTION_LEASE_REQUIRED")
    payload, raw = _read_lock(lease.guard_path)
    if raw != lease.raw or payload.get("ownership_token") != lease.ownership_token:
        raise RuntimeError("BRAINIAC_ACTION_LEASE_INVALID")
    live, creation = _pid_identity(lease.pid)
    if live is not True or str(creation) != lease.process_create_time:
        raise RuntimeError("BRAINIAC_ACTION_LEASE_PROCESS_IDENTITY_INVALID")
    return lease
