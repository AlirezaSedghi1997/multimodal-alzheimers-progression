from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from src.research import brainiac_readiness_anchor as _anchor_module


def _ledger(tmp_path: Path, payloads: list[dict]) -> tuple[Path, dict]:
    from src.research.brainiac_readiness_ledger import (
        append_readiness_transition,
        validate_readiness_ledger,
    )

    ledger = tmp_path / "ledger"
    for payload in payloads:
        append_readiness_transition(ledger, payload, project_root=tmp_path)
    return ledger, validate_readiness_ledger(ledger, project_root=tmp_path)


def _event_raw(ledger: Path, sequence: int) -> bytes:
    return next(ledger.glob(f"{sequence:08d}-*.json")).read_bytes()


def _anchor_fixture(tmp_path: Path) -> tuple[Path, dict, bytes, list[dict], str]:
    from src.research.brainiac_readiness_anchor import build_anchor_message

    ledger, report = _ledger(
        tmp_path,
        [{"stage": "B1", "value": 1}, {"stage": "B1", "value": 2}, {"stage": "B1", "value": 3}],
    )
    key = b"k" * 32
    baseline_message = build_anchor_message(
        event_raw=_event_raw(ledger, 3),
        sequence=3,
        event_sha256=str(report["head_sha256"]),
        previous_event_sha256=json.loads(_event_raw(ledger, 3))["previous_event_sha256"],
        previous_anchor_record_id=None,
        key=key,
        nonce="baseline-nonce",
        baseline=True,
    )
    records = [{"record_id": 100, "message": baseline_message}]
    return ledger, report, key, records, hashlib.sha256(baseline_message.encode()).hexdigest()


def test_os_anchor_accepts_a_legitimate_monotonic_tail(tmp_path: Path) -> None:
    from src.research.brainiac_readiness_anchor import (
        build_anchor_message,
        validate_anchor_chain,
    )
    from src.research.brainiac_readiness_ledger import (
        append_readiness_transition,
        validate_readiness_ledger,
    )

    ledger, baseline, key, records, baseline_sha = _anchor_fixture(tmp_path)
    append_readiness_transition(
        ledger, {"stage": "B2", "value": 4}, project_root=tmp_path
    )
    report = validate_readiness_ledger(ledger, project_root=tmp_path)
    message = build_anchor_message(
        event_raw=_event_raw(ledger, 4),
        sequence=4,
        event_sha256=str(report["head_sha256"]),
        previous_event_sha256=str(baseline["head_sha256"]),
        previous_anchor_record_id=100,
        key=key,
        nonce="tail-nonce",
        baseline=False,
    )
    records.append({"record_id": 101, "message": message})
    anchored = validate_anchor_chain(
        ledger_dir=ledger,
        ledger_report=report,
        records=records,
        key=key,
        baseline_event_count=3,
        baseline_head_sha256=str(baseline["head_sha256"]),
        baseline_record_id=100,
        baseline_message_sha256=baseline_sha,
    )
    assert anchored["status"] == "PASS"
    assert anchored["event_count"] == 4
    assert anchored["anchor_record_id"] == 101


def test_os_anchor_rejects_tail_rollback_and_regeneration(tmp_path: Path) -> None:
    from src.research.brainiac_readiness_anchor import (
        build_anchor_message,
        validate_anchor_chain,
    )
    from src.research.brainiac_readiness_ledger import (
        append_readiness_transition,
        validate_readiness_ledger,
    )

    ledger, baseline, key, records, baseline_sha = _anchor_fixture(tmp_path)
    append_readiness_transition(ledger, {"stage": "B2", "value": 4}, project_root=tmp_path)
    original = validate_readiness_ledger(ledger, project_root=tmp_path)
    records.append(
        {
            "record_id": 101,
            "message": build_anchor_message(
                event_raw=_event_raw(ledger, 4),
                sequence=4,
                event_sha256=str(original["head_sha256"]),
                previous_event_sha256=str(baseline["head_sha256"]),
                previous_anchor_record_id=100,
                key=key,
                nonce="original-tail",
                baseline=False,
            ),
        }
    )
    next(ledger.glob("00000004-*.json")).unlink()
    append_readiness_transition(ledger, {"stage": "B2", "value": 999}, project_root=tmp_path)
    rewritten = validate_readiness_ledger(ledger, project_root=tmp_path)
    with pytest.raises(RuntimeError, match="ANCHOR_LEDGER_DIVERGENCE"):
        validate_anchor_chain(
            ledger_dir=ledger,
            ledger_report=rewritten,
            records=records,
            key=key,
            baseline_event_count=3,
            baseline_head_sha256=str(baseline["head_sha256"]),
            baseline_record_id=100,
            baseline_message_sha256=baseline_sha,
        )


def test_os_anchor_rejects_competing_same_sequence_fork(tmp_path: Path) -> None:
    from src.research.brainiac_readiness_anchor import (
        build_anchor_message,
        validate_anchor_chain,
    )
    from src.research.brainiac_readiness_ledger import (
        append_readiness_transition,
        validate_readiness_ledger,
    )

    ledger, baseline, key, records, baseline_sha = _anchor_fixture(tmp_path)
    append_readiness_transition(ledger, {"stage": "B2", "value": 4}, project_root=tmp_path)
    report = validate_readiness_ledger(ledger, project_root=tmp_path)
    raw = _event_raw(ledger, 4)
    for record_id, nonce in ((101, "fork-a"), (102, "fork-b")):
        records.append(
            {
                "record_id": record_id,
                "message": build_anchor_message(
                    event_raw=raw,
                    sequence=4,
                    event_sha256=str(report["head_sha256"]),
                    previous_event_sha256=str(baseline["head_sha256"]),
                    previous_anchor_record_id=100,
                    key=key,
                    nonce=nonce,
                    baseline=False,
                ),
            }
        )
    with pytest.raises(RuntimeError, match="ANCHOR_FORK"):
        validate_anchor_chain(
            ledger_dir=ledger,
            ledger_report=report,
            records=records,
            key=key,
            baseline_event_count=3,
            baseline_head_sha256=str(baseline["head_sha256"]),
            baseline_record_id=100,
            baseline_message_sha256=baseline_sha,
        )


def test_os_anchor_rejects_restored_old_head(tmp_path: Path) -> None:
    from src.research.brainiac_readiness_anchor import validate_anchor_chain
    from src.research.brainiac_readiness_ledger import (
        append_readiness_transition,
        validate_readiness_ledger,
    )

    ledger, baseline, key, records, baseline_sha = _anchor_fixture(tmp_path)
    append_readiness_transition(ledger, {"stage": "B2", "value": 4}, project_root=tmp_path)
    report = validate_readiness_ledger(ledger, project_root=tmp_path)
    with pytest.raises(RuntimeError, match="LEDGER_UNANCHORED_TAIL"):
        validate_anchor_chain(
            ledger_dir=ledger,
            ledger_report=report,
            records=records,
            key=key,
            baseline_event_count=3,
            baseline_head_sha256=str(baseline["head_sha256"]),
            baseline_record_id=100,
            baseline_message_sha256=baseline_sha,
        )


# --- Plan 001: production EventLog reader, provisioning, anchored append ---


class _FakeAnchorEventLog:
    """Injectable Windows Application-log backend (test authority only)."""

    def __init__(self, records: list[dict] | None = None) -> None:
        self._records: list[dict] = [dict(row) for row in (records or [])]
        self.writes: list[str] = []

    def read(self) -> list[dict]:
        return [dict(row) for row in self._records]

    def seed(self, message: str) -> int:
        record_id = 9000 + len(self._records)
        self._records.append({"record_id": record_id, "message": message})
        return record_id

    def write(self, message: str) -> int:
        record_id = self.seed(message)
        self.writes.append(message)
        return record_id


def _anchored_fixture(tmp_path: Path) -> tuple[Path, dict, bytes, _FakeAnchorEventLog, dict]:
    from src.research.brainiac_readiness_anchor import build_anchor_message
    from src.research.brainiac_readiness_ledger import (
        append_readiness_transition,
        validate_readiness_ledger,
    )

    ledger = tmp_path / "ledger"
    for value in (1, 2, 3):
        append_readiness_transition(
            ledger, {"stage": "B1", "value": value}, project_root=tmp_path
        )
    report = validate_readiness_ledger(ledger, project_root=tmp_path)
    event3_raw = next(ledger.glob("00000003-*.json")).read_bytes()
    key = b"k" * 32
    baseline_message = build_anchor_message(
        event_raw=event3_raw,
        sequence=3,
        event_sha256=str(report["head_sha256"]),
        previous_event_sha256=json.loads(event3_raw)["previous_event_sha256"],
        previous_anchor_record_id=None,
        key=key,
        nonce="baseline-nonce",
        baseline=True,
    )
    log = _FakeAnchorEventLog()
    baseline_record_id = log.seed(baseline_message)
    pins = {
        "baseline_event_count": 3,
        "baseline_head_sha256": str(report["head_sha256"]),
        "baseline_record_id": baseline_record_id,
        "baseline_message_sha256": hashlib.sha256(
            baseline_message.encode("utf-8")
        ).hexdigest(),
    }
    return ledger, report, key, log, pins


def _anchored_append(
    ledger: Path, payload: dict, *, tmp_path: Path, key: bytes, log: _FakeAnchorEventLog, pins: dict
) -> dict:
    from src.research.brainiac_readiness_ledger import (
        append_anchored_readiness_transition,
    )

    return append_anchored_readiness_transition(
        ledger,
        payload,
        project_root=tmp_path,
        anchor_reader=log.read,
        anchor_writer=log.write,
        anchor_key=key,
        anchor_pins=pins,
    )


def _embedded_event_bytes(message: str, key: bytes) -> bytes:
    return _anchor_module._parse_anchor_message(message, key=key)["event_raw"]


def test_eventlog_no_match_is_empty_not_unknown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Zero matching events must be a distinguishable empty result, not UNKNOWN."""

    def fake_run(args, **kwargs):
        script = args[-1]
        if "NoMatchingEventsFound" in script:
            # The query script itself discriminates the typed no-match case.
            return subprocess.CompletedProcess(args, 0, stdout="", stderr="")
        return subprocess.CompletedProcess(
            args,
            1,
            stdout="",
            stderr="NoMatchingEventsFound,Microsoft.PowerShell.Commands.GetWinEventCommand",
        )

    monkeypatch.setattr(_anchor_module.subprocess, "run", fake_run)
    assert _anchor_module.read_production_anchor_records() == []


def test_eventlog_access_denied_is_unknown_and_blocks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_run(args, **kwargs):
        return subprocess.CompletedProcess(args, 3, stdout="", stderr="Access denied")

    monkeypatch.setattr(_anchor_module.subprocess, "run", fake_run)
    with pytest.raises(
        RuntimeError, match="BRAINIAC_LEDGER_ANCHOR_EVENTLOG_READ_UNKNOWN"
    ):
        _anchor_module.read_production_anchor_records()


def test_eventlog_timeout_is_unknown(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(args, **kwargs):
        raise subprocess.TimeoutExpired(cmd=args, timeout=30)

    monkeypatch.setattr(_anchor_module.subprocess, "run", fake_run)
    with pytest.raises(
        RuntimeError, match="BRAINIAC_LEDGER_ANCHOR_EVENTLOG_READ_UNKNOWN"
    ):
        _anchor_module.read_production_anchor_records()


def test_eventlog_reader_rejects_malformed_output_and_bad_record_ids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    emitted = {"stdout": "not-json"}

    def fake_run(args, **kwargs):
        return subprocess.CompletedProcess(args, 0, stdout=emitted["stdout"], stderr="")

    monkeypatch.setattr(_anchor_module.subprocess, "run", fake_run)
    with pytest.raises(
        RuntimeError, match="BRAINIAC_LEDGER_ANCHOR_EVENTLOG_READ_INVALID"
    ):
        _anchor_module.read_production_anchor_records()
    emitted["stdout"] = json.dumps(
        [{"record_id": 5, "message": "a"}, {"record_id": 5, "message": "b"}]
    )
    with pytest.raises(
        RuntimeError, match="BRAINIAC_LEDGER_ANCHOR_EVENTLOG_READ_INVALID"
    ):
        _anchor_module.read_production_anchor_records()
    emitted["stdout"] = json.dumps([{"record_id": 0, "message": "a"}])
    with pytest.raises(
        RuntimeError, match="BRAINIAC_LEDGER_ANCHOR_EVENTLOG_READ_INVALID"
    ):
        _anchor_module.read_production_anchor_records()


def test_provision_baseline_is_idempotent_after_write_before_pin(
    tmp_path: Path,
) -> None:
    from src.research.brainiac_readiness_anchor import discover_or_provision_baseline
    from src.research.brainiac_readiness_ledger import (
        append_readiness_transition,
        validate_readiness_ledger,
    )

    ledger = tmp_path / "ledger"
    for value in (1, 2, 3):
        append_readiness_transition(
            ledger, {"stage": "B1", "value": value}, project_root=tmp_path
        )
    report = validate_readiness_ledger(ledger, project_root=tmp_path)
    event3_raw = next(ledger.glob("00000003-*.json")).read_bytes()
    key = b"k" * 32
    log = _FakeAnchorEventLog()

    first = discover_or_provision_baseline(
        records=log.read(),
        key=key,
        event_raw=event3_raw,
        sequence=3,
        event_sha256=str(report["head_sha256"]),
        write_message=log.write,
    )
    second = discover_or_provision_baseline(
        records=log.read(),
        key=key,
        event_raw=event3_raw,
        sequence=3,
        event_sha256=str(report["head_sha256"]),
        write_message=log.write,
    )
    assert first["created"] is True
    assert second["created"] is False
    assert first["record_id"] == second["record_id"] > 0
    assert first["message_sha256"] == second["message_sha256"]
    assert len(log.writes) == 1


def test_duplicate_valid_baseline_anchor_is_fork_at_provisioning(
    tmp_path: Path,
) -> None:
    from src.research.brainiac_readiness_anchor import (
        build_anchor_message,
        discover_or_provision_baseline,
    )
    from src.research.brainiac_readiness_ledger import (
        append_readiness_transition,
        validate_readiness_ledger,
    )

    ledger = tmp_path / "ledger"
    for value in (1, 2, 3):
        append_readiness_transition(
            ledger, {"stage": "B1", "value": value}, project_root=tmp_path
        )
    report = validate_readiness_ledger(ledger, project_root=tmp_path)
    event3_raw = next(ledger.glob("00000003-*.json")).read_bytes()
    key = b"k" * 32
    log = _FakeAnchorEventLog()
    for nonce in ("duplicate-a", "duplicate-b"):
        log.seed(
            build_anchor_message(
                event_raw=event3_raw,
                sequence=3,
                event_sha256=str(report["head_sha256"]),
                previous_event_sha256=json.loads(event3_raw)[
                    "previous_event_sha256"
                ],
                previous_anchor_record_id=None,
                key=key,
                nonce=nonce,
                baseline=True,
            )
        )
    with pytest.raises(RuntimeError, match="BRAINIAC_LEDGER_ANCHOR_BASELINE_FORK"):
        discover_or_provision_baseline(
            records=log.read(),
            key=key,
            event_raw=event3_raw,
            sequence=3,
            event_sha256=str(report["head_sha256"]),
            write_message=log.write,
        )
    assert log.writes == []


def test_legal_append_creates_exactly_one_os_event_and_one_h_event(
    tmp_path: Path,
) -> None:
    from src.research.brainiac_readiness_anchor import validate_anchor_chain
    from src.research.brainiac_readiness_ledger import validate_readiness_ledger

    ledger, _baseline, key, log, pins = _anchored_fixture(tmp_path)
    result = _anchored_append(
        ledger, {"stage": "B2", "value": 4}, tmp_path=tmp_path, key=key, log=log, pins=pins
    )
    assert result["anchored_append"] is True
    assert len(log.writes) == 1
    files = sorted(path.name for path in ledger.glob("*.json"))
    assert len(files) == 4
    assert files[-1].startswith("00000004-")
    grown = validate_readiness_ledger(ledger, project_root=tmp_path)
    chain = validate_anchor_chain(
        ledger_dir=ledger,
        ledger_report=grown,
        records=log.read(),
        key=key,
        **pins,
    )
    assert chain["status"] == "PASS"
    assert chain["anchor_record_id"] == result["anchor_record_id"]
    assert chain["head_sha256"] == grown["head_sha256"]


def test_unchanged_projection_writes_no_new_anchor(tmp_path: Path) -> None:
    ledger, baseline, key, log, pins = _anchored_fixture(tmp_path)
    result = _anchored_append(
        ledger, {"stage": "B1", "value": 3}, tmp_path=tmp_path, key=key, log=log, pins=pins
    )
    assert result["anchored_append"] is False
    assert result["event_sha256"] == str(baseline["head_sha256"])
    assert log.writes == []
    assert len(list(ledger.glob("*.json"))) == 3


def test_anchor_first_crash_recovers_exact_event_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import src.research.brainiac_readiness_ledger as ledger_module

    ledger, _baseline, key, log, pins = _anchored_fixture(tmp_path)
    original_commit = ledger_module._commit_event_exclusive

    def crashing_commit(directory, name, raw):
        raise OSError("simulated crash between OS anchor write and H write")

    monkeypatch.setattr(ledger_module, "_commit_event_exclusive", crashing_commit)
    with pytest.raises(RuntimeError):
        _anchored_append(
            ledger,
            {"stage": "B2", "value": 4},
            tmp_path=tmp_path,
            key=key,
            log=log,
            pins=pins,
        )
    monkeypatch.setattr(ledger_module, "_commit_event_exclusive", original_commit)
    assert len(log.writes) == 1
    assert len(list(ledger.glob("*.json"))) == 3

    result = _anchored_append(
        ledger, {"stage": "B2", "value": 4}, tmp_path=tmp_path, key=key, log=log, pins=pins
    )
    assert len(log.writes) == 1  # recovery reused the pending anchor, no new OS write
    assert result["anchored_append"] is False  # replayed payload is now the head
    recovered = next(ledger.glob("00000004-*.json")).read_bytes()
    assert recovered == _embedded_event_bytes(log.writes[0], key)


def test_h_ahead_unanchored_tail_is_never_laundered(tmp_path: Path) -> None:
    from src.research.brainiac_readiness_ledger import append_readiness_transition

    ledger, _baseline, key, log, pins = _anchored_fixture(tmp_path)
    append_readiness_transition(
        ledger, {"stage": "B2", "value": 4}, project_root=tmp_path
    )
    with pytest.raises(
        RuntimeError, match="BRAINIAC_READINESS_LEDGER_UNANCHORED_TAIL"
    ):
        _anchored_append(
            ledger,
            {"stage": "B2", "value": 5},
            tmp_path=tmp_path,
            key=key,
            log=log,
            pins=pins,
        )
    assert log.writes == []
    assert len(list(ledger.glob("*.json"))) == 4


def test_same_sequence_anchor_fork_is_rejected_during_append(
    tmp_path: Path,
) -> None:
    from src.research.brainiac_readiness_anchor import build_anchor_message
    from src.research.brainiac_readiness_ledger import (
        append_readiness_transition,
        validate_readiness_ledger,
    )

    ledger, baseline, key, log, pins = _anchored_fixture(tmp_path)
    append_readiness_transition(
        ledger, {"stage": "B2", "value": 4}, project_root=tmp_path
    )
    grown = validate_readiness_ledger(ledger, project_root=tmp_path)
    raw4 = next(ledger.glob("00000004-*.json")).read_bytes()
    for nonce in ("fork-a", "fork-b"):
        log.seed(
            build_anchor_message(
                event_raw=raw4,
                sequence=4,
                event_sha256=str(grown["head_sha256"]),
                previous_event_sha256=str(baseline["head_sha256"]),
                previous_anchor_record_id=int(pins["baseline_record_id"]),
                key=key,
                nonce=nonce,
                baseline=False,
            )
        )
    with pytest.raises(RuntimeError, match="BRAINIAC_LEDGER_ANCHOR_FORK"):
        _anchored_append(
            ledger,
            {"stage": "B2", "value": 5},
            tmp_path=tmp_path,
            key=key,
            log=log,
            pins=pins,
        )
    assert log.writes == []


def test_dpapi_key_is_never_regenerated_on_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_path = _anchor_module.ANCHOR_KEY_PATH
    missing = tmp_path / "absent-key.dpapi.json"
    monkeypatch.setattr(_anchor_module, "ANCHOR_KEY_PATH", missing)
    with pytest.raises(RuntimeError, match="BRAINIAC_LEDGER_ANCHOR_KEY_MISSING"):
        _anchor_module.load_production_anchor_key()
    assert not missing.exists()

    tampered = tmp_path / "tampered-key.dpapi.json"
    tampered.write_bytes(b"tampered key bytes")
    monkeypatch.setattr(_anchor_module, "ANCHOR_KEY_PATH", tampered)
    with pytest.raises(
        RuntimeError, match="BRAINIAC_LEDGER_ANCHOR_KEY_FILE_TAMPERED"
    ):
        _anchor_module.load_production_anchor_key()
    assert tampered.read_bytes() == b"tampered key bytes"

    def _unprotect_failure(raw: bytes) -> bytes:
        raise RuntimeError("BRAINIAC_LEDGER_ANCHOR_DPAPI_UNPROTECT_FAILED")

    monkeypatch.setattr(_anchor_module, "_unprotect_dpapi", _unprotect_failure)
    monkeypatch.setattr(_anchor_module, "ANCHOR_KEY_PATH", real_path)
    with pytest.raises(
        RuntimeError, match="BRAINIAC_LEDGER_ANCHOR_DPAPI_UNPROTECT_FAILED"
    ):
        _anchor_module.load_production_anchor_key()
    assert real_path.is_file()


def test_live_b1_three_event_baseline_validates() -> None:
    """Live production chain: the pinned B1 baseline anchors every later head.

    The baseline pins the 3-event head at provisioning time (sequence and head
    SHA are enforced inside validate_anchor_chain); later cards grow the ledger
    only through anchored appends, so the live invariant is a PASS chain whose
    tip covers the entire ledger.
    """

    root = Path(__file__).resolve().parents[1]
    from src.research.brainiac_execution_gate import DEFAULT_READINESS_LEDGER
    from src.research.brainiac_readiness_anchor import (
        PINNED_B1_BASELINE_EVENT_COUNT,
        PINNED_B1_BASELINE_HEAD_SHA256,
        PINNED_B1_BASELINE_RECORD_ID,
        validate_production_anchor,
    )
    from src.research.brainiac_readiness_ledger import (
        PINNED_B1_GENESIS_EVENT_SHA256,
        validate_readiness_ledger,
    )

    report = validate_readiness_ledger(
        DEFAULT_READINESS_LEDGER,
        project_root=root,
        expected_genesis_sha256=PINNED_B1_GENESIS_EVENT_SHA256,
    )
    assert int(report["event_count"]) >= PINNED_B1_BASELINE_EVENT_COUNT
    anchored = validate_production_anchor(
        ledger_dir=DEFAULT_READINESS_LEDGER, ledger_report=report
    )
    assert anchored["status"] == "PASS"
    assert PINNED_B1_BASELINE_RECORD_ID > 0
    assert anchored["baseline_record_id"] == PINNED_B1_BASELINE_RECORD_ID
    # The baseline itself still binds the exact pinned 3-event head.
    assert PINNED_B1_BASELINE_HEAD_SHA256 == (
        "105e381a79723b9b0fddfc752fa3631a4aea2514542bd55808e7e3e1a44ea8cd"
    )
    # The chain tip covers the whole live ledger.
    assert anchored["event_count"] == int(report["event_count"])
    assert anchored["head_sha256"] == report["head_sha256"]
