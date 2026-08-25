"""Append-only, hash-chained BrainIAC readiness transitions.

The ledger replaces a mutable code-side hash for evolving B2--B6 readiness.
Static policy bytes remain pinned in code; dynamic state advances only through
validated, immutable events beneath the H project root.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping

from src.research.brainiac_readiness import (
    PROJECT_ROOT,
    _assert_readiness_privacy,
    _beneath,
    _canonical_json_bytes,
    _is_reparse,
    _sha256_bytes,
)

_O_BINARY = int(getattr(os, "O_BINARY", 0))
PINNED_B1_GENESIS_EVENT_SHA256 = (
    "fb382cee0140f446be0b2770c5799deb8b63ed1b546e7419485bdf67dd7535b4"
)
PRODUCTION_READINESS_LEDGER = (
    PROJECT_ROOT
    / "results"
    / "method_x_registry"
    / "brainiac_r1"
    / "readiness_transitions_v1"
)


_TRANSITION_FIELDS = (
    "schema_version",
    "artifact_type",
    "r0",
    "input_bindings",
    "derived_evidence",
    "battery_terminal_report",
    "pretraining_overlap_status",
    "claim_limits",
    "lora_unlocked",
    "outer_test_touched",
    "human_approval_required",
    "manual_override_allowed",
    "unknown_is_pass",
    "supersedes",
)


def readiness_transition_projection(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Return stable scientific evidence, excluding live process decisions."""

    projected = {key: payload.get(key) for key in _TRANSITION_FIELDS}
    # Keep ledger events small: hash the large per-fold model evidence
    # (5 backbones × 5 folds) which would otherwise push the OS anchor
    # message over the Windows Event Log per-message limit (~32k).
    # Previous non-terminal events had no `models` array, so this is
    # backwards-compatible.
    btr = projected.get("battery_terminal_report")
    if isinstance(btr, Mapping) and isinstance(btr.get("models"), list) and btr.get("models"):
        slim = {k: v for k, v in btr.items() if k != "models"}
        slim["models_digest"] = _sha256_bytes(_canonical_json_bytes(btr["models"]))
        slim["models_count"] = len(btr["models"])
        projected["battery_terminal_report"] = slim
    return json.loads(_canonical_json_bytes(projected).decode("utf-8"))


def _event_body(
    *, sequence: int, previous: str | None, payload: Mapping[str, Any]
) -> dict[str, Any]:
    canonical_payload = json.loads(_canonical_json_bytes(payload).decode("utf-8"))
    _assert_readiness_privacy(canonical_payload)
    return {
        "schema_version": 1,
        "artifact_type": "BRAINIAC_READINESS_TRANSITION",
        "sequence": int(sequence),
        "previous_event_sha256": previous,
        "readiness_payload_sha256": _sha256_bytes(
            _canonical_json_bytes(canonical_payload)
        ),
        "readiness_payload": canonical_payload,
    }


def _decode_event(path: Path) -> tuple[dict[str, Any], bytes]:
    try:
        raw = path.read_bytes()
        value = json.loads(raw.decode("utf-8"), parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)))
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"BRAINIAC_READINESS_LEDGER_EVENT_INVALID:{path.name}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"BRAINIAC_READINESS_LEDGER_EVENT_INVALID:{path.name}")
    return value, raw


def _validate_readiness_ledger(
    ledger_dir: str | Path,
    *,
    project_root: str | Path,
    owned_append_lock_pid: int | None,
    expected_genesis_sha256: str | None,
) -> dict[str, Any]:
    root = Path(project_root)
    ledger = Path(ledger_dir)
    if not os.path.lexists(ledger):
        parent = _beneath(ledger.parent, root, must_exist=True)
        if _is_reparse(parent):
            raise RuntimeError("BRAINIAC_READINESS_LEDGER_REPARSE_FORBIDDEN")
        return {"status": "EMPTY", "event_count": 0, "head_sha256": None}
    resolved = _beneath(ledger, root, must_exist=True)
    if not resolved.is_dir() or _is_reparse(resolved):
        raise RuntimeError("BRAINIAC_READINESS_LEDGER_DIRECTORY_INVALID")
    entries = sorted(resolved.iterdir(), key=lambda path: path.name)
    append_locks = [entry for entry in entries if entry.name == ".append.lock"]
    if append_locks:
        append_lock = append_locks[0]
        if (
            owned_append_lock_pid is None
            or _is_reparse(append_lock)
            or not append_lock.is_file()
            or append_lock.read_text(encoding="ascii") != str(owned_append_lock_pid)
        ):
            raise RuntimeError("BRAINIAC_READINESS_LEDGER_APPEND_IN_PROGRESS")
        entries.remove(append_lock)
    if any(not entry.is_file() or _is_reparse(entry) for entry in entries):
        raise RuntimeError("BRAINIAC_READINESS_LEDGER_ENTRY_INVALID")
    events = [entry for entry in entries if entry.suffix.casefold() == ".json"]
    if len(events) != len(entries):
        raise RuntimeError("BRAINIAC_READINESS_LEDGER_UNEXPECTED_ENTRY")

    previous: str | None = None
    genesis: str | None = None
    head_payload: dict[str, Any] | None = None
    for sequence, path in enumerate(events, start=1):
        event, raw = _decode_event(path)
        recorded_sha = str(event.get("event_sha256") or "").casefold()
        body = dict(event)
        body.pop("event_sha256", None)
        computed_sha = _sha256_bytes(_canonical_json_bytes(body))
        expected_name = f"{sequence:08d}-{computed_sha}.json"
        if path.name != expected_name or recorded_sha != computed_sha:
            raise RuntimeError("BRAINIAC_READINESS_LEDGER_EVENT_HASH_MISMATCH")
        if raw != _canonical_json_bytes(event):
            raise RuntimeError("BRAINIAC_READINESS_LEDGER_NONCANONICAL_EVENT")
        if event.get("sequence") != sequence:
            raise RuntimeError("BRAINIAC_READINESS_LEDGER_SEQUENCE_INVALID")
        if event.get("previous_event_sha256") != previous:
            raise RuntimeError("BRAINIAC_READINESS_LEDGER_CHAIN_BROKEN")
        payload = event.get("readiness_payload")
        if not isinstance(payload, Mapping):
            raise RuntimeError("BRAINIAC_READINESS_LEDGER_PAYLOAD_INVALID")
        _assert_readiness_privacy(payload)
        if event.get("readiness_payload_sha256") != _sha256_bytes(
            _canonical_json_bytes(payload)
        ):
            raise RuntimeError("BRAINIAC_READINESS_LEDGER_PAYLOAD_HASH_MISMATCH")
        previous = computed_sha
        if genesis is None:
            genesis = computed_sha
        head_payload = dict(payload)
    if (
        expected_genesis_sha256 is not None
        and genesis != str(expected_genesis_sha256).casefold()
    ):
        raise RuntimeError("BRAINIAC_READINESS_LEDGER_GENESIS_MISMATCH")
    return {
        "status": "PASS" if events else "EMPTY",
        "event_count": len(events),
        "head_sha256": previous,
        "head_payload": head_payload,
    }


def validate_readiness_ledger(
    ledger_dir: str | Path,
    *,
    project_root: str | Path = PROJECT_ROOT,
    expected_genesis_sha256: str | None = None,
) -> dict[str, Any]:
    """Validate sequence, exact filenames, payload hashes and hash continuity."""

    return _validate_readiness_ledger(
        ledger_dir,
        project_root=project_root,
        owned_append_lock_pid=None,
        expected_genesis_sha256=expected_genesis_sha256,
    )


def _is_production_ledger(ledger_dir: str | Path, project_root: str | Path) -> bool:
    try:
        return (
            Path(project_root).resolve() == PROJECT_ROOT.resolve()
            and Path(ledger_dir).resolve() == PRODUCTION_READINESS_LEDGER.resolve()
        )
    except OSError:
        return False


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except (OSError, NotImplementedError):
        return
    try:
        os.fsync(descriptor)
    except (OSError, NotImplementedError):
        pass
    finally:
        os.close(descriptor)


def _commit_event_exclusive(resolved: Path, name: str, raw: bytes) -> None:
    """Create one ledger event file exclusively; never overwrite existing bytes."""

    target = resolved / name
    try:
        descriptor = os.open(
            target, os.O_CREAT | os.O_EXCL | os.O_WRONLY | _O_BINARY
        )
    except FileExistsError as exc:
        raise RuntimeError("BRAINIAC_READINESS_LEDGER_FORK_OR_REPLAY") from exc
    except OSError as exc:
        raise RuntimeError("BRAINIAC_READINESS_LEDGER_EVENT_COMMIT_FAILED") from exc
    try:
        os.write(descriptor, raw)
        os.fsync(descriptor)
    except OSError as exc:
        raise RuntimeError("BRAINIAC_READINESS_LEDGER_EVENT_COMMIT_FAILED") from exc
    finally:
        os.close(descriptor)
    _fsync_directory(resolved)


def append_readiness_transition(
    ledger_dir: str | Path,
    payload: Mapping[str, Any],
    *,
    project_root: str | Path = PROJECT_ROOT,
    expected_genesis_sha256: str | None = None,
) -> dict[str, Any]:
    """Append one transition under an exclusive local lock; identical replay is a no-op.

    The production ledger may only grow through
    :func:`append_anchored_readiness_transition`; direct unanchored appends
    against it fail closed.
    """

    if _is_production_ledger(ledger_dir, project_root):
        raise RuntimeError("BRAINIAC_READINESS_LEDGER_UNANCHORED_APPEND_FORBIDDEN")

    root = Path(project_root)
    ledger = Path(ledger_dir)
    _beneath(ledger.parent, root, must_exist=True)
    ledger.mkdir(parents=False, exist_ok=True)
    resolved = _beneath(ledger, root, must_exist=True)
    lock = resolved / ".append.lock"
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY | _O_BINARY)
    except FileExistsError as exc:
        raise RuntimeError("BRAINIAC_READINESS_LEDGER_CONCURRENT_APPEND") from exc
    try:
        os.write(descriptor, str(os.getpid()).encode("ascii"))
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    try:
        report = _validate_readiness_ledger(
            resolved,
            project_root=root,
            owned_append_lock_pid=os.getpid(),
            expected_genesis_sha256=expected_genesis_sha256,
        )
        canonical_payload = json.loads(_canonical_json_bytes(payload).decode("utf-8"))
        _assert_readiness_privacy(canonical_payload)
        if report.get("head_payload") == canonical_payload:
            head = str(report["head_sha256"])
            event_path = resolved / f"{int(report['event_count']):08d}-{head}.json"
            event, _ = _decode_event(event_path)
            return event
        body = _event_body(
            sequence=int(report["event_count"]) + 1,
            previous=report.get("head_sha256"),
            payload=canonical_payload,
        )
        event_sha = _sha256_bytes(_canonical_json_bytes(body))
        if (
            int(report["event_count"]) == 0
            and expected_genesis_sha256 is not None
            and event_sha != str(expected_genesis_sha256).casefold()
        ):
            raise RuntimeError("BRAINIAC_READINESS_LEDGER_GENESIS_MISMATCH")
        event = {**body, "event_sha256": event_sha}
        raw = _canonical_json_bytes(event)
        _commit_event_exclusive(resolved, f"{body['sequence']:08d}-{event_sha}.json", raw)
        return event
    finally:
        if os.path.lexists(lock) and not _is_reparse(lock):
            lock.unlink(missing_ok=True)


def append_anchored_readiness_transition(
    ledger_dir: str | Path,
    payload: Mapping[str, Any],
    *,
    project_root: str | Path = PROJECT_ROOT,
    expected_genesis_sha256: str | None = None,
    anchor_reader: Any = None,
    anchor_writer: Any = None,
    anchor_key: bytes | None = None,
    anchor_pins: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Sole authority for growing the readiness ledger: OS anchor first, H second.

    Order of operations: validate H ledger and anchor chain alignment, recover
    any pending OS-first crash by writing exactly the authenticated pending
    bytes, derive the next event deterministically, write and authenticate the
    OS anchor successor, atomically commit the identical bytes to H, then
    revalidate both sides.  A caller can never append H bytes that the OS
    anchor chain does not already carry.
    """

    from src.research.brainiac_readiness_anchor import (
        PINNED_B1_BASELINE_EVENT_COUNT,
        PINNED_B1_BASELINE_HEAD_SHA256,
        PINNED_B1_BASELINE_MESSAGE_SHA256,
        PINNED_B1_BASELINE_RECORD_ID,
        build_anchor_message,
        load_production_anchor_key,
        read_production_anchor_records,
        validate_anchor_chain,
        write_production_anchor_message,
    )
    import uuid

    reader = anchor_reader or read_production_anchor_records
    writer = anchor_writer or write_production_anchor_message
    key = anchor_key if anchor_key is not None else load_production_anchor_key()
    pins = dict(
        anchor_pins
        or {
            "baseline_event_count": PINNED_B1_BASELINE_EVENT_COUNT,
            "baseline_head_sha256": PINNED_B1_BASELINE_HEAD_SHA256,
            "baseline_record_id": PINNED_B1_BASELINE_RECORD_ID,
            "baseline_message_sha256": PINNED_B1_BASELINE_MESSAGE_SHA256,
        }
    )

    def _chain(report: Mapping[str, Any], *, allow_pending: bool) -> dict[str, Any]:
        return validate_anchor_chain(
            ledger_dir=ledger,
            ledger_report=report,
            records=reader(),
            key=key,
            baseline_event_count=int(pins["baseline_event_count"]),
            baseline_head_sha256=str(pins["baseline_head_sha256"]),
            baseline_record_id=int(pins["baseline_record_id"]),
            baseline_message_sha256=str(pins["baseline_message_sha256"]),
            allow_pending=allow_pending,
        )

    root = Path(project_root)
    ledger = Path(ledger_dir)
    _beneath(ledger.parent, root, must_exist=True)
    ledger.mkdir(parents=False, exist_ok=True)
    resolved = _beneath(ledger, root, must_exist=True)

    def _commit(name: str, raw: bytes) -> None:
        try:
            _commit_event_exclusive(resolved, name, raw)
        except OSError as exc:
            raise RuntimeError("BRAINIAC_READINESS_LEDGER_EVENT_COMMIT_FAILED") from exc

    lock = resolved / ".append.lock"
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY | _O_BINARY)
    except FileExistsError as exc:
        raise RuntimeError("BRAINIAC_READINESS_LEDGER_CONCURRENT_APPEND") from exc
    try:
        os.write(descriptor, str(os.getpid()).encode("ascii"))
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    try:
        report = _validate_readiness_ledger(
            resolved,
            project_root=root,
            owned_append_lock_pid=os.getpid(),
            expected_genesis_sha256=expected_genesis_sha256,
        )
        chain = _chain(report, allow_pending=True)
        if chain.get("status") == "PENDING_LEDGER_WRITE":
            pending_raw = chain["pending_event_raw"]
            pending_event = json.loads(pending_raw.decode("utf-8"))
            _commit(
                f"{int(pending_event['sequence']):08d}-{pending_event['event_sha256']}.json",
                pending_raw,
            )
            report = _validate_readiness_ledger(
                resolved,
                project_root=root,
                owned_append_lock_pid=os.getpid(),
                expected_genesis_sha256=expected_genesis_sha256,
            )
            chain = _chain(report, allow_pending=True)
        if chain.get("status") != "PASS":
            raise RuntimeError("BRAINIAC_READINESS_ANCHOR_LEDGER_DIVERGENCE")

        canonical_payload = json.loads(_canonical_json_bytes(payload).decode("utf-8"))
        _assert_readiness_privacy(canonical_payload)
        if report.get("head_payload") == canonical_payload:
            head = str(report["head_sha256"])
            event, _ = _decode_event(
                resolved / f"{int(report['event_count']):08d}-{head}.json"
            )
            return {
                **event,
                "anchored_append": False,
                "anchor_record_id": int(chain["anchor_record_id"]),
            }

        sequence = int(report["event_count"]) + 1
        body = _event_body(
            sequence=sequence,
            previous=report.get("head_sha256"),
            payload=canonical_payload,
        )
        event_sha = _sha256_bytes(_canonical_json_bytes(body))
        if (
            int(report["event_count"]) == 0
            and expected_genesis_sha256 is not None
            and event_sha != str(expected_genesis_sha256).casefold()
        ):
            raise RuntimeError("BRAINIAC_READINESS_LEDGER_GENESIS_MISMATCH")
        event = {**body, "event_sha256": event_sha}
        raw = _canonical_json_bytes(event)
        message = build_anchor_message(
            event_raw=raw,
            sequence=sequence,
            event_sha256=event_sha,
            previous_event_sha256=str(report["head_sha256"]),
            previous_anchor_record_id=int(chain["anchor_record_id"]),
            key=key,
            nonce=uuid.uuid4().hex,
            baseline=False,
        )
        record_id = int(writer(message))
        if record_id <= 0:
            raise RuntimeError("BRAINIAC_LEDGER_ANCHOR_EVENTLOG_WRITE_UNCONFIRMED")
        confirmed = _chain(report, allow_pending=True)
        if (
            confirmed.get("status") != "PENDING_LEDGER_WRITE"
            or confirmed.get("pending_event_raw") != raw
            or int(confirmed.get("anchor_record_id", -1)) != record_id
        ):
            raise RuntimeError("BRAINIAC_LEDGER_ANCHOR_EVENTLOG_WRITE_UNCONFIRMED")
        _commit(f"{sequence:08d}-{event_sha}.json", raw)
        final_report = _validate_readiness_ledger(
            resolved,
            project_root=root,
            owned_append_lock_pid=os.getpid(),
            expected_genesis_sha256=expected_genesis_sha256,
        )
        final_chain = _chain(final_report, allow_pending=False)
        if (
            final_chain.get("status") != "PASS"
            or final_chain.get("head_sha256") != event_sha
        ):
            raise RuntimeError("BRAINIAC_READINESS_ANCHOR_LEDGER_DIVERGENCE")
        return {
            **event,
            "anchored_append": True,
            "anchor_record_id": record_id,
            "anchor_message_sha256": _sha256_bytes(message.encode("utf-8")),
        }
    finally:
        if os.path.lexists(lock) and not _is_reparse(lock):
            lock.unlink(missing_ok=True)
