"""One-time OS baseline anchor provisioning for the B1 readiness ledger.

Modes:
  --mode preflight          read-only precondition evidence, no writes
  --mode provision-baseline discover the unique authenticated baseline anchor,
                            writing it exactly once if absent

The script never prints the anchor message, the DPAPI key, or any protected
blob; only the OS record id and the message SHA-256 are reported.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

PROJECT_ROOT_BOOTSTRAP = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT_BOOTSTRAP) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_BOOTSTRAP))

from src.research.brainiac_action_guard import inspect_action_exclusion  # noqa: E402
from src.research.brainiac_readiness import PROJECT_ROOT, bind_file  # noqa: E402
from src.research.brainiac_readiness_anchor import (  # noqa: E402
    PINNED_B1_BASELINE_EVENT_COUNT,
    PINNED_B1_BASELINE_HEAD_SHA256,
    PINNED_B1_BASELINE_MESSAGE_SHA256,
    PINNED_B1_BASELINE_RECORD_ID,
    authenticate_matching_baseline_records,
    discover_or_provision_baseline,
    load_production_anchor_key,
    read_production_anchor_records,
    write_production_anchor_message,
)
from src.research.brainiac_readiness_ledger import (  # noqa: E402
    PINNED_B1_GENESIS_EVENT_SHA256,
    PRODUCTION_READINESS_LEDGER,
    validate_readiness_ledger,
)


def _ledger_state() -> dict[str, Any]:
    report = validate_readiness_ledger(
        PRODUCTION_READINESS_LEDGER,
        project_root=PROJECT_ROOT,
        expected_genesis_sha256=PINNED_B1_GENESIS_EVENT_SHA256,
    )
    return {
        "status": report.get("status"),
        "event_count": report.get("event_count"),
        "head_sha256": report.get("head_sha256"),
    }


def _baseline_event_binding() -> dict[str, Any]:
    event_path = (
        PRODUCTION_READINESS_LEDGER
        / f"{PINNED_B1_BASELINE_EVENT_COUNT:08d}-{PINNED_B1_BASELINE_HEAD_SHA256}.json"
    )
    binding = bind_file(event_path, project_root=PROJECT_ROOT)
    return {
        "path": binding["path"],
        "bytes": binding["bytes"],
        "sha256": binding["sha256"],
        "raw": Path(binding["path"]).read_bytes(),
    }


def preflight() -> dict[str, Any]:
    report: dict[str, Any] = {
        "schema_version": 1,
        "artifact": "BRAINIAC_B1_ANCHOR_PROVISION_PREFLIGHT",
        "writes_performed": False,
    }
    exclusion = inspect_action_exclusion()
    report["writer_state"] = {
        "status": exclusion.get("status"),
        "blockers": exclusion.get("blockers"),
    }
    if exclusion.get("blockers"):
        report["preflight"] = "BLOCKED_ACTIVE_OR_UNKNOWN_WRITER"
        return report

    ledger = _ledger_state()
    report["ledger"] = ledger
    if (
        int(ledger.get("event_count", -1)) != PINNED_B1_BASELINE_EVENT_COUNT
        or str(ledger.get("head_sha256")).casefold()
        != PINNED_B1_BASELINE_HEAD_SHA256.casefold()
    ):
        report["preflight"] = "BLOCKED_LEDGER_DRIFT"
        return report

    try:
        binding = _baseline_event_binding()
    except (OSError, RuntimeError) as exc:
        report["preflight"] = f"BLOCKED_BASELINE_EVENT_INVALID:{exc}"
        return report
    report["baseline_event_binding"] = {
        "bytes": binding["bytes"],
        "sha256": binding["sha256"],
    }

    try:
        key = load_production_anchor_key()
        records = read_production_anchor_records()
        matches = authenticate_matching_baseline_records(
            records=records,
            key=key,
            event_raw=binding["raw"],
            sequence=PINNED_B1_BASELINE_EVENT_COUNT,
            event_sha256=PINNED_B1_BASELINE_HEAD_SHA256,
        )
    except RuntimeError as exc:
        report["preflight"] = f"BLOCKED_ANCHOR_AUTHORITY_INVALID:{exc}"
        return report
    report["anchor_key_status"] = "PASS"
    report["matching_baseline_anchors"] = len(matches)
    if len(matches) > 1:
        report["preflight"] = "BLOCKED_BASELINE_FORK"
        return report
    report["pins"] = {
        "baseline_record_id": PINNED_B1_BASELINE_RECORD_ID,
        "baseline_message_sha256_is_pinned": len(PINNED_B1_BASELINE_MESSAGE_SHA256)
        == 64,
    }
    report["preflight"] = "PASS"
    return report


def provision_baseline() -> dict[str, Any]:
    pre = preflight()
    if pre.get("preflight") != "PASS":
        return {
            "schema_version": 1,
            "artifact": "BRAINIAC_B1_ANCHOR_PROVISION",
            "status": "BLOCKED",
            "writes_performed": False,
            "preflight": pre,
        }
    binding = _baseline_event_binding()
    receipt = discover_or_provision_baseline(
        records=read_production_anchor_records(),
        key=load_production_anchor_key(),
        event_raw=binding["raw"],
        sequence=PINNED_B1_BASELINE_EVENT_COUNT,
        event_sha256=PINNED_B1_BASELINE_HEAD_SHA256,
        write_message=write_production_anchor_message,
    )
    matches = authenticate_matching_baseline_records(
        records=read_production_anchor_records(),
        key=load_production_anchor_key(),
        event_raw=binding["raw"],
        sequence=PINNED_B1_BASELINE_EVENT_COUNT,
        event_sha256=PINNED_B1_BASELINE_HEAD_SHA256,
    )
    if len(matches) != 1:
        return {
            "schema_version": 1,
            "artifact": "BRAINIAC_B1_ANCHOR_PROVISION",
            "status": "BLOCKED",
            "writes_performed": receipt["created"],
            "reason_code": "BRAINIAC_LEDGER_ANCHOR_BASELINE_NOT_UNIQUE_AFTER_WRITE",
            "matching_baseline_anchors": len(matches),
        }
    return {
        "schema_version": 1,
        "artifact": "BRAINIAC_B1_ANCHOR_PROVISION",
        "status": "PROVISIONED" if receipt["created"] else "REUSED",
        "writes_performed": receipt["created"],
        "record_id": receipt["record_id"],
        "message_sha256": receipt["message_sha256"],
        "matching_baseline_anchors": 1,
        "ledger": _ledger_state(),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=("preflight", "provision-baseline"),
        default="preflight",
    )
    args = parser.parse_args(argv)
    if args.mode == "preflight":
        report: dict[str, Any] = preflight()
    else:
        report = provision_baseline()
    print(json.dumps(report, indent=2, sort_keys=True, default=str))
    return 0 if report.get("status", report.get("preflight")) in {
        "PASS",
        "PROVISIONED",
        "REUSED",
    } else 3


if __name__ == "__main__":
    raise SystemExit(main())
