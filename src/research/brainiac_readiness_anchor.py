"""OS-backed anti-rollback anchor for the BrainIAC readiness ledger.

Ledger JSON remains on H.  Each post-B1 transition is independently recorded
in the Windows Application event log and authenticated with a DPAPI-protected
HMAC key.  The event log supplies a monotonic record id that a rewritten H-side
tail cannot erase or roll back without being detected.
"""

from __future__ import annotations

import base64
import ctypes
from ctypes import wintypes
import hashlib
import hmac
import json
import os
from pathlib import Path
import subprocess
from typing import Any, Mapping, Sequence

from src.research.brainiac_readiness import PROJECT_ROOT, _canonical_json_bytes


ANCHOR_PREFIX = "AMUSE-X_BRAINIAC_READINESS_ANCHOR_V1:"
ANCHOR_ARTIFACT = "BRAINIAC_READINESS_LEDGER_OS_ANCHOR"
ANCHOR_EVENT_ID = 6102
ANCHOR_EVENT_PROVIDER = "Application"
ANCHOR_KEY_PATH = (
    PROJECT_ROOT / "governance" / "brainiac_readiness_anchor_key.v1.dpapi.json"
)
PINNED_ANCHOR_KEY_FILE_SHA256 = (
    "44127d5546187f442a43f7eab6f49917cc7edab4683d48ad6b86626793210a27"
)
PINNED_B1_BASELINE_EVENT_COUNT = 3
PINNED_B1_BASELINE_HEAD_SHA256 = (
    "105e381a79723b9b0fddfc752fa3631a4aea2514542bd55808e7e3e1a44ea8cd"
)
# Filled only after the one-time OS anchor is emitted and read back.  A zero
# value is deliberately non-authorizing during provisioning.
PINNED_B1_BASELINE_RECORD_ID = 92912
PINNED_B1_BASELINE_MESSAGE_SHA256 = (
    "3b4825685aba279f6e841376cee2ea99a068f052affbde29d47bd91fa0ce08a9"
)
_DPAPI_ENTROPY = b"AMUSE-X BrainIAC readiness ledger anchor v1"


def _canonical_json(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _event_sha(raw: bytes) -> tuple[dict[str, Any], str]:
    try:
        event = json.loads(raw.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise RuntimeError("BRAINIAC_LEDGER_ANCHOR_EVENT_INVALID") from exc
    if not isinstance(event, dict):
        raise RuntimeError("BRAINIAC_LEDGER_ANCHOR_EVENT_INVALID")
    recorded = str(event.get("event_sha256") or "").casefold()
    body = dict(event)
    body.pop("event_sha256", None)
    computed = _sha256(_canonical_json_bytes(body))
    if recorded != computed:
        raise RuntimeError("BRAINIAC_LEDGER_ANCHOR_EVENT_HASH_MISMATCH")
    return event, computed


def build_anchor_message(
    *,
    event_raw: bytes,
    sequence: int,
    event_sha256: str,
    previous_event_sha256: str | None,
    previous_anchor_record_id: int | None,
    key: bytes,
    nonce: str,
    baseline: bool,
) -> str:
    """Build a deterministic authenticated event-log payload."""

    if len(key) < 32:
        raise RuntimeError("BRAINIAC_LEDGER_ANCHOR_KEY_INVALID")
    event, computed = _event_sha(event_raw)
    if (
        int(sequence) < 1
        or event.get("sequence") != int(sequence)
        or computed != str(event_sha256).casefold()
        or event.get("previous_event_sha256") != previous_event_sha256
        or not str(nonce)
    ):
        raise RuntimeError("BRAINIAC_LEDGER_ANCHOR_EVENT_INVALID")
    body = {
        "schema_version": 1,
        "artifact_type": ANCHOR_ARTIFACT,
        "baseline": bool(baseline),
        "sequence": int(sequence),
        "event_sha256": computed,
        "previous_event_sha256": previous_event_sha256,
        "previous_anchor_record_id": previous_anchor_record_id,
        "event_bytes_sha256": _sha256(event_raw),
        "event_bytes_b64": base64.b64encode(event_raw).decode("ascii"),
        "nonce": str(nonce),
    }
    signature = hmac.new(key, _canonical_json(body), hashlib.sha256).hexdigest()
    payload = {**body, "hmac_sha256": signature}
    return ANCHOR_PREFIX + base64.b64encode(_canonical_json(payload)).decode("ascii")


def _parse_anchor_message(message: str, *, key: bytes) -> dict[str, Any]:
    if not isinstance(message, str) or not message.startswith(ANCHOR_PREFIX):
        raise RuntimeError("BRAINIAC_LEDGER_ANCHOR_MESSAGE_INVALID")
    try:
        raw = base64.b64decode(message[len(ANCHOR_PREFIX) :], validate=True)
        payload = json.loads(raw.decode("utf-8"))
    except (ValueError, UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("BRAINIAC_LEDGER_ANCHOR_MESSAGE_INVALID") from exc
    if not isinstance(payload, dict) or raw != _canonical_json(payload):
        raise RuntimeError("BRAINIAC_LEDGER_ANCHOR_MESSAGE_INVALID")
    body = dict(payload)
    signature = body.pop("hmac_sha256", None)
    expected = hmac.new(key, _canonical_json(body), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(str(signature or ""), expected):
        raise RuntimeError("BRAINIAC_LEDGER_ANCHOR_HMAC_INVALID")
    try:
        event_raw = base64.b64decode(str(body["event_bytes_b64"]), validate=True)
    except (KeyError, ValueError) as exc:
        raise RuntimeError("BRAINIAC_LEDGER_ANCHOR_MESSAGE_INVALID") from exc
    event, computed = _event_sha(event_raw)
    if (
        body.get("schema_version") != 1
        or body.get("artifact_type") != ANCHOR_ARTIFACT
        or type(body.get("baseline")) is not bool
        or type(body.get("sequence")) is not int
        or body.get("sequence") != event.get("sequence")
        or body.get("event_sha256") != computed
        or body.get("previous_event_sha256")
        != event.get("previous_event_sha256")
        or body.get("event_bytes_sha256") != _sha256(event_raw)
        or not isinstance(body.get("nonce"), str)
        or not body["nonce"]
    ):
        raise RuntimeError("BRAINIAC_LEDGER_ANCHOR_MESSAGE_INVALID")
    return {**body, "event_raw": event_raw, "message_sha256": _sha256(message.encode())}


def validate_anchor_chain(
    *,
    ledger_dir: str | Path,
    ledger_report: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    key: bytes,
    baseline_event_count: int,
    baseline_head_sha256: str,
    baseline_record_id: int,
    baseline_message_sha256: str,
    allow_pending: bool = False,
) -> dict[str, Any]:
    """Validate the unique OS anchor chain against the exact H ledger tail."""

    ledger = Path(ledger_dir)
    parsed: list[dict[str, Any]] = []
    seen_record_ids: set[int] = set()
    for record in records:
        try:
            record_id = int(record.get("record_id", 0))
        except (TypeError, ValueError) as exc:
            raise RuntimeError("BRAINIAC_LEDGER_ANCHOR_RECORD_INVALID") from exc
        if record_id <= 0 or record_id in seen_record_ids:
            raise RuntimeError("BRAINIAC_LEDGER_ANCHOR_RECORD_INVALID")
        seen_record_ids.add(record_id)
        parsed.append(
            {
                **_parse_anchor_message(str(record.get("message") or ""), key=key),
                "record_id": record_id,
            }
        )
    baselines = [row for row in parsed if row["record_id"] == int(baseline_record_id)]
    if len(baselines) != 1:
        raise RuntimeError("BRAINIAC_LEDGER_ANCHOR_BASELINE_MISSING")
    baseline = baselines[0]
    if (
        baseline.get("baseline") is not True
        or baseline.get("sequence") != int(baseline_event_count)
        or baseline.get("event_sha256") != str(baseline_head_sha256).casefold()
        or baseline.get("previous_anchor_record_id") is not None
        or baseline.get("message_sha256")
        != str(baseline_message_sha256).casefold()
    ):
        raise RuntimeError("BRAINIAC_LEDGER_ANCHOR_BASELINE_INVALID")

    consumed = {int(baseline_record_id)}
    current = baseline
    while True:
        candidates = [
            row
            for row in parsed
            if row["record_id"] not in consumed
            and row.get("previous_anchor_record_id") == current["record_id"]
            and row.get("sequence") == int(current["sequence"]) + 1
            and row.get("previous_event_sha256") == current["event_sha256"]
        ]
        if len(candidates) > 1:
            raise RuntimeError("BRAINIAC_LEDGER_ANCHOR_FORK")
        if not candidates:
            break
        successor = candidates[0]
        if successor.get("baseline") is not False or successor["record_id"] <= current["record_id"]:
            raise RuntimeError("BRAINIAC_LEDGER_ANCHOR_CHAIN_INVALID")
        consumed.add(successor["record_id"])
        current = successor
    if len(consumed) != len(parsed):
        raise RuntimeError("BRAINIAC_LEDGER_ANCHOR_FORK")

    anchor_count = int(current["sequence"])
    ledger_count = int(ledger_report.get("event_count", -1))
    if ledger_count > anchor_count:
        raise RuntimeError("BRAINIAC_READINESS_LEDGER_UNANCHORED_TAIL")
    pending = bool(
        allow_pending
        and ledger_count == anchor_count - 1
        and str(ledger_report.get("head_sha256") or "")
        == str(current.get("previous_event_sha256") or "")
    )
    if ledger_count < anchor_count and not pending:
        raise RuntimeError("BRAINIAC_READINESS_ANCHOR_LEDGER_DIVERGENCE")
    if pending:
        return {
            "status": "PENDING_LEDGER_WRITE",
            "event_count": anchor_count,
            "head_sha256": current["event_sha256"],
            "anchor_record_id": current["record_id"],
            "baseline_record_id": int(baseline_record_id),
            "anchor_backend": "WINDOWS_APPLICATION_EVENT_LOG_HMAC_DPAPI",
            "pending_event_raw": current["event_raw"],
        }
    if str(ledger_report.get("head_sha256") or "") != current["event_sha256"]:
        raise RuntimeError("BRAINIAC_READINESS_ANCHOR_LEDGER_DIVERGENCE")

    for row in parsed:
        sequence = int(row["sequence"])
        if sequence < int(baseline_event_count):
            raise RuntimeError("BRAINIAC_LEDGER_ANCHOR_CHAIN_INVALID")
        event_path = ledger / f"{sequence:08d}-{row['event_sha256']}.json"
        try:
            actual = event_path.read_bytes()
        except OSError as exc:
            raise RuntimeError("BRAINIAC_READINESS_ANCHOR_LEDGER_DIVERGENCE") from exc
        if actual != row["event_raw"]:
            raise RuntimeError("BRAINIAC_READINESS_ANCHOR_LEDGER_DIVERGENCE")
    return {
        "status": "PASS",
        "event_count": anchor_count,
        "head_sha256": current["event_sha256"],
        "anchor_record_id": current["record_id"],
        "baseline_record_id": int(baseline_record_id),
        "anchor_backend": "WINDOWS_APPLICATION_EVENT_LOG_HMAC_DPAPI",
    }


class _DataBlob(ctypes.Structure):
    _fields_ = [("cbData", wintypes.DWORD), ("pbData", ctypes.POINTER(ctypes.c_byte))]


def _blob(raw: bytes) -> tuple[_DataBlob, Any]:
    buffer = ctypes.create_string_buffer(raw)
    return (
        _DataBlob(len(raw), ctypes.cast(buffer, ctypes.POINTER(ctypes.c_byte))),
        buffer,
    )


def _unprotect_dpapi(raw: bytes) -> bytes:
    if os.name != "nt":
        raise RuntimeError("BRAINIAC_LEDGER_ANCHOR_WINDOWS_REQUIRED")
    encrypted, encrypted_buffer = _blob(raw)
    entropy, entropy_buffer = _blob(_DPAPI_ENTROPY)
    output = _DataBlob()
    description = wintypes.LPWSTR()
    crypt32 = ctypes.windll.crypt32
    kernel32 = ctypes.windll.kernel32
    ok = crypt32.CryptUnprotectData(
        ctypes.byref(encrypted),
        ctypes.byref(description),
        ctypes.byref(entropy),
        None,
        None,
        0x1,
        ctypes.byref(output),
    )
    del encrypted_buffer, entropy_buffer
    if not ok:
        raise RuntimeError("BRAINIAC_LEDGER_ANCHOR_DPAPI_UNPROTECT_FAILED")
    try:
        return ctypes.string_at(output.pbData, output.cbData)
    finally:
        kernel32.LocalFree(output.pbData)
        if description:
            kernel32.LocalFree(description)


def load_production_anchor_key() -> bytes:
    try:
        raw = ANCHOR_KEY_PATH.read_bytes()
    except OSError as exc:
        raise RuntimeError("BRAINIAC_LEDGER_ANCHOR_KEY_MISSING") from exc
    if _sha256(raw) != PINNED_ANCHOR_KEY_FILE_SHA256:
        raise RuntimeError("BRAINIAC_LEDGER_ANCHOR_KEY_FILE_TAMPERED")
    try:
        payload = json.loads(raw.decode("utf-8"))
        protected = base64.b64decode(payload["protected_key_b64"], validate=True)
    except (KeyError, ValueError, UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("BRAINIAC_LEDGER_ANCHOR_KEY_FILE_INVALID") from exc
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != 1
        or payload.get("artifact") != "brainiac_readiness_ledger_dpapi_key"
        or payload.get("dpapi_scope") != "CURRENT_WINDOWS_USER"
        or payload.get("key_bytes") != 32
    ):
        raise RuntimeError("BRAINIAC_LEDGER_ANCHOR_KEY_FILE_INVALID")
    key = _unprotect_dpapi(protected)
    if len(key) != 32:
        raise RuntimeError("BRAINIAC_LEDGER_ANCHOR_KEY_INVALID")
    return key


_NO_MATCH_ERROR_ID = "NoMatchingEventsFound,Microsoft.PowerShell.Commands.GetWinEventCommand"


def read_production_anchor_records() -> list[dict[str, Any]]:
    if os.name != "nt":
        raise RuntimeError("BRAINIAC_LEDGER_ANCHOR_WINDOWS_REQUIRED")
    prefix = ANCHOR_PREFIX.replace("'", "''")
    script = (
        "$ErrorActionPreference='Stop'; "
        f"$p='{prefix}'; $rows=@(); "
        "try { "
        "$rows = Get-WinEvent -FilterHashtable "
        f"@{{LogName='Application';ProviderName='{ANCHOR_EVENT_PROVIDER}';Id={ANCHOR_EVENT_ID}}} "
        "-ErrorAction Stop | "
        "ForEach-Object {$d=if($_.Properties.Count -gt 0){[string]$_.Properties[0].Value}else{''}; "
        "if($d.StartsWith($p)){[pscustomobject]@{record_id=[long]$_.RecordId;message=$d}}} "
        "} catch { "
        "if ($_.FullyQualifiedErrorId -ne '" + _NO_MATCH_ERROR_ID + "') { "
        "[Console]::Error.WriteLine($_.FullyQualifiedErrorId); exit 3 } "
        "} "
        "if($null -ne $rows -and @($rows).Count -gt 0)"
        "{$rows | Sort-Object record_id | ConvertTo-Json -Compress -Depth 3}; "
        "exit 0"
    )
    try:
        completed = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", script],
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError("BRAINIAC_LEDGER_ANCHOR_EVENTLOG_READ_UNKNOWN") from exc
    if completed.returncode != 0:
        raise RuntimeError("BRAINIAC_LEDGER_ANCHOR_EVENTLOG_READ_UNKNOWN")
    text = completed.stdout.strip()
    if not text:
        return []
    try:
        rows = json.loads(text)
    except json.JSONDecodeError as exc:
        raise RuntimeError("BRAINIAC_LEDGER_ANCHOR_EVENTLOG_READ_INVALID") from exc
    if isinstance(rows, Mapping):
        rows = [rows]
    if not isinstance(rows, list) or any(not isinstance(row, Mapping) for row in rows):
        raise RuntimeError("BRAINIAC_LEDGER_ANCHOR_EVENTLOG_READ_INVALID")
    seen_record_ids: set[int] = set()
    parsed: list[dict[str, Any]] = []
    for row in rows:
        try:
            record_id = int(row.get("record_id", 0))
        except (TypeError, ValueError) as exc:
            raise RuntimeError("BRAINIAC_LEDGER_ANCHOR_EVENTLOG_READ_INVALID") from exc
        if record_id <= 0 or record_id in seen_record_ids:
            raise RuntimeError("BRAINIAC_LEDGER_ANCHOR_EVENTLOG_READ_INVALID")
        seen_record_ids.add(record_id)
        parsed.append({"record_id": record_id, "message": str(row.get("message") or "")})
    return parsed


def write_production_anchor_message(message: str) -> int:
    if os.name != "nt" or not message.startswith(ANCHOR_PREFIX):
        raise RuntimeError("BRAINIAC_LEDGER_ANCHOR_MESSAGE_INVALID")
    script = (
        "$m=[Console]::In.ReadToEnd(); "
        f"Write-EventLog -LogName Application -Source '{ANCHOR_EVENT_PROVIDER}' "
        f"-EventId {ANCHOR_EVENT_ID} -EntryType Information -Message $m -ErrorAction Stop; "
        "exit 0"
    )
    try:
        completed = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", script],
            input=message,
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError("BRAINIAC_LEDGER_ANCHOR_EVENTLOG_WRITE_FAILED") from exc
    if completed.returncode != 0:
        raise RuntimeError("BRAINIAC_LEDGER_ANCHOR_EVENTLOG_WRITE_FAILED")
    matches = [
        int(row["record_id"])
        for row in read_production_anchor_records()
        if row.get("message") == message
    ]
    if len(matches) != 1:
        raise RuntimeError("BRAINIAC_LEDGER_ANCHOR_EVENTLOG_WRITE_UNCONFIRMED")
    return matches[0]


def authenticate_matching_baseline_records(
    *,
    records: Sequence[Mapping[str, Any]],
    key: bytes,
    event_raw: bytes,
    sequence: int,
    event_sha256: str,
) -> list[dict[str, Any]]:
    """Return authenticated baseline anchors binding exactly these ledger bytes.

    Only HMAC-valid messages count as authority; unauthenticated prefixed noise
    in the shared OS log is ignored.  Any *valid* baseline-shaped message that
    binds different bytes, or any valid successor, is a fork at provisioning
    time and fails closed.
    """

    matches: list[dict[str, Any]] = []
    conflicts = 0
    for record in records:
        try:
            row = _parse_anchor_message(str(record.get("message") or ""), key=key)
        except RuntimeError:
            continue
        try:
            record_id = int(record.get("record_id", 0))
        except (TypeError, ValueError) as exc:
            raise RuntimeError("BRAINIAC_LEDGER_ANCHOR_RECORD_INVALID") from exc
        if row.get("baseline") is not True or row.get("sequence") != int(sequence):
            conflicts += 1
            continue
        if (
            row.get("event_sha256") == str(event_sha256).casefold()
            and row.get("previous_anchor_record_id") is None
            and row.get("event_raw") == event_raw
        ):
            matches.append({**row, "record_id": record_id})
        else:
            conflicts += 1
    if conflicts:
        raise RuntimeError("BRAINIAC_LEDGER_ANCHOR_BASELINE_FORK")
    return matches


def discover_or_provision_baseline(
    *,
    records: Sequence[Mapping[str, Any]],
    key: bytes,
    event_raw: bytes,
    sequence: int,
    event_sha256: str,
    write_message: Any,
) -> dict[str, Any]:
    """Exactly-once baseline provisioning with idempotent rediscovery."""

    matches = authenticate_matching_baseline_records(
        records=records,
        key=key,
        event_raw=event_raw,
        sequence=sequence,
        event_sha256=event_sha256,
    )
    if len(matches) > 1:
        raise RuntimeError("BRAINIAC_LEDGER_ANCHOR_BASELINE_FORK")
    if len(matches) == 1:
        return {
            "record_id": int(matches[0]["record_id"]),
            "message_sha256": str(matches[0]["message_sha256"]),
            "created": False,
        }

    import uuid

    message = build_anchor_message(
        event_raw=event_raw,
        sequence=int(sequence),
        event_sha256=str(event_sha256),
        previous_event_sha256=json.loads(event_raw.decode("utf-8"))[
            "previous_event_sha256"
        ],
        previous_anchor_record_id=None,
        key=key,
        nonce=uuid.uuid4().hex,
        baseline=True,
    )
    record_id = int(write_message(message))
    if record_id <= 0:
        raise RuntimeError("BRAINIAC_LEDGER_ANCHOR_EVENTLOG_WRITE_UNCONFIRMED")
    confirmed = authenticate_matching_baseline_records(
        records=[{"record_id": record_id, "message": message}],
        key=key,
        event_raw=event_raw,
        sequence=sequence,
        event_sha256=event_sha256,
    )
    if len(confirmed) != 1 or int(confirmed[0]["record_id"]) != record_id:
        raise RuntimeError("BRAINIAC_LEDGER_ANCHOR_EVENTLOG_WRITE_UNCONFIRMED")
    return {
        "record_id": record_id,
        "message_sha256": _sha256(message.encode("utf-8")),
        "created": True,
    }


def validate_production_anchor(
    *, ledger_dir: str | Path, ledger_report: Mapping[str, Any], allow_pending: bool = False
) -> dict[str, Any]:
    if PINNED_B1_BASELINE_RECORD_ID <= 0 or len(PINNED_B1_BASELINE_MESSAGE_SHA256) != 64:
        raise RuntimeError("BRAINIAC_LEDGER_ANCHOR_BASELINE_NOT_PINNED")
    return validate_anchor_chain(
        ledger_dir=ledger_dir,
        ledger_report=ledger_report,
        records=read_production_anchor_records(),
        key=load_production_anchor_key(),
        baseline_event_count=PINNED_B1_BASELINE_EVENT_COUNT,
        baseline_head_sha256=PINNED_B1_BASELINE_HEAD_SHA256,
        baseline_record_id=PINNED_B1_BASELINE_RECORD_ID,
        baseline_message_sha256=PINNED_B1_BASELINE_MESSAGE_SHA256,
        allow_pending=allow_pending,
    )


def write_production_anchor_for_event(
    *, event_raw: bytes, sequence: int, event_sha256: str, previous_event_sha256: str,
    previous_anchor_record_id: int
) -> dict[str, Any]:
    import uuid

    message = build_anchor_message(
        event_raw=event_raw,
        sequence=sequence,
        event_sha256=event_sha256,
        previous_event_sha256=previous_event_sha256,
        previous_anchor_record_id=previous_anchor_record_id,
        key=load_production_anchor_key(),
        nonce=uuid.uuid4().hex,
        baseline=False,
    )
    record_id = write_production_anchor_message(message)
    return {"record_id": record_id, "message_sha256": _sha256(message.encode("utf-8"))}
