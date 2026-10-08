"""Persistent, fail-closed loop protection for autonomous Method X stages.

The guard is intentionally small and dependency-free.  A retry is allowed
only for the same immutable contract and the same normalized failure
signature, and only twice.  A third identical failure becomes a machine
block; callers must create a new versioned run instead of retrying forever.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.utils.io_helpers import atomic_write_json


DEFAULT_MAX_SAME_FAILURE_ATTEMPTS = 2


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"schema_version": 1, "entries": {}, "events": []}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        # A corrupt guard is itself unsafe.  Do not silently reset it.
        raise RuntimeError("METHOD_X_LOOP_GUARD_STATE_INVALID")
    if not isinstance(value, dict) or not isinstance(value.get("entries", {}), dict):
        raise RuntimeError("METHOD_X_LOOP_GUARD_STATE_INVALID")
    value.setdefault("schema_version", 1)
    value.setdefault("events", [])
    return value


def _key(*, stage: str, contract_digest: str, failure_signature: str) -> str:
    stage_text = str(stage).strip()
    contract_text = str(contract_digest).strip().lower()
    signature_text = " ".join(str(failure_signature).strip().split())
    if not stage_text or not contract_text or not signature_text:
        raise ValueError("METHOD_X_LOOP_GUARD_KEY_INCOMPLETE")
    return hashlib.sha256(
        f"{stage_text}|{contract_text}|{signature_text}".encode("utf-8", "replace")
    ).hexdigest()


def record_failure(
    path: str | Path,
    *,
    stage: str,
    contract_digest: str,
    failure_signature: str,
    max_same_failure_attempts: int = DEFAULT_MAX_SAME_FAILURE_ATTEMPTS,
) -> dict[str, Any]:
    """Record one failure and return ``RETRY`` or ``BLOCK_LOOP_GUARD``.

    ``max_same_failure_attempts=2`` means the first and second observations
    are retry-eligible; the third identical observation is blocked.
    """

    maximum = int(max_same_failure_attempts)
    if maximum < 1:
        raise ValueError("METHOD_X_LOOP_GUARD_MAX_ATTEMPTS_INVALID")
    state_path = Path(path)
    state = _read(state_path)
    key = _key(stage=stage, contract_digest=contract_digest, failure_signature=failure_signature)
    previous = state["entries"].get(key) if isinstance(state["entries"].get(key), dict) else {}
    count = int(previous.get("same_failure_count", 0)) + 1
    decision = "RETRY" if count <= maximum else "BLOCK_LOOP_GUARD"
    event = {
        "stage": str(stage),
        "contract_digest": str(contract_digest).lower(),
        "failure_signature": " ".join(str(failure_signature).strip().split()),
        "same_failure_count": count,
        "decision": decision,
        "created_at_utc": _now(),
    }
    state["entries"][key] = event
    events = state.get("events") if isinstance(state.get("events"), list) else []
    events.append({"key": key, **event})
    state["events"] = events
    state["updated_at_utc"] = event["created_at_utc"]
    state["max_same_failure_attempts"] = maximum
    atomic_write_json(state_path, state)
    result = {"schema_version": 1, "key": key, **event}
    if decision == "BLOCK_LOOP_GUARD":
        blocker = state_path.parent / f"BLOCKED_{stage}_LOOP_GUARD.json"
        atomic_write_json(
            blocker,
            {
                "schema_version": 1,
                "status": "BLOCKED_LOOP_GUARD",
                "stage": str(stage),
                "contract_digest": str(contract_digest).lower(),
                "failure_signature": event["failure_signature"],
                "same_failure_count": count,
                "max_same_failure_attempts": maximum,
                "next_action": "CREATE_NEW_VERSIONED_RUN_OR_REPAIR_CONTRACT",
            },
        )
        result["blocker_path"] = str(blocker)
    return result


def record_success(path: str | Path, *, stage: str, contract_digest: str) -> dict[str, Any]:
    """Record a terminal success without deleting historical failure events."""

    state_path = Path(path)
    state = _read(state_path)
    event = {
        "stage": str(stage),
        "contract_digest": str(contract_digest).lower(),
        "decision": "COMPLETE",
        "created_at_utc": _now(),
    }
    events = state.get("events") if isinstance(state.get("events"), list) else []
    events.append(event)
    state["events"] = events
    state["updated_at_utc"] = event["created_at_utc"]
    atomic_write_json(state_path, state)
    return {"schema_version": 1, **event}

