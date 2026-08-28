"""Production BrainIAC authorization with a canonical, exclusive action lease.

Static governance is byte-pinned by :mod:`brainiac_readiness`. Dynamic
readiness is recomputed from canonical H paths and matched to an append-only
hash chain. Public APIs expose no path, SHA, process-provider or root override.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator, Mapping

from src.research.brainiac_action_guard import (
    BrainIACActionLease,
    acquire_action_guard,
    activate_lease,
    deactivate_lease,
    inspect_action_exclusion,
    release_action_guard,
    require_active_lease,
)
from src.research.brainiac_readiness import (
    ACTIONS,
    PINNED_POLICY_SHA256,
    PINNED_SCOPE_AMENDMENT_SHA256,
    PROJECT_ROOT,
    build_live_readiness,
)
from src.research.brainiac_readiness_anchor import validate_production_anchor
from src.research.brainiac_readiness_ledger import (
    PINNED_B1_GENESIS_EVENT_SHA256,
    PRODUCTION_READINESS_LEDGER,
    readiness_transition_projection,
    validate_readiness_ledger,
)


DEFAULT_R1_READINESS = (
    PROJECT_ROOT
    / "research"
    / "brainiac_caft_20260808"
    / "r1_readiness_status_schema4_20260822.json"
)
DEFAULT_READINESS_LEDGER = PRODUCTION_READINESS_LEDGER


class BrainIACExecutionBlocked(RuntimeError):
    """Raised before protected I/O when autonomous evidence is incomplete."""


def _blocked(code: str, exc: BaseException | None = None) -> BrainIACExecutionBlocked:
    error = BrainIACExecutionBlocked(f"BRAINIAC_EXECUTION_HARD_DISABLED:{code}")
    if exc is not None:
        error.__cause__ = exc
    return error


def _validate_transition_authority(live: Mapping[str, Any]) -> dict[str, Any]:
    report = validate_readiness_ledger(
        DEFAULT_READINESS_LEDGER,
        project_root=PROJECT_ROOT,
        expected_genesis_sha256=PINNED_B1_GENESIS_EVENT_SHA256,
    )
    if report.get("event_count", 0) < 1:
        raise _blocked("READINESS_LEDGER_NOT_PUBLISHED")
    if report.get("head_payload") != readiness_transition_projection(live):
        raise _blocked("READINESS_LEDGER_HEAD_DRIFT")
    try:
        anchor = validate_production_anchor(
            ledger_dir=DEFAULT_READINESS_LEDGER, ledger_report=report
        )
    except RuntimeError as exc:
        raise _blocked(f"READINESS_LEDGER_ANCHOR_INVALID:{exc}") from exc
    if anchor.get("status") != "PASS":
        raise _blocked("READINESS_LEDGER_ANCHOR_INVALID:STATUS_NOT_PASS")
    return {**report, "os_anchor": anchor}


def _assess_canonical(
    action: str, *, exclusion: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    requested = str(action).upper()
    if requested not in ACTIONS:
        raise _blocked("ACTION_INVALID")
    try:
        live = build_live_readiness(project_root=PROJECT_ROOT)
        ledger = _validate_transition_authority(live)
        exclusion_report = dict(exclusion or inspect_action_exclusion())
    except BrainIACExecutionBlocked:
        raise
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise _blocked(f"READINESS_RECOMPUTE_FAILED:{exc}", exc)

    decision = live.get("actions", {}).get(requested)
    if not isinstance(decision, Mapping):
        raise _blocked("ACTION_DECISION_MISSING")
    blockers = list(decision.get("blockers") or ())
    blockers.extend(str(item) for item in exclusion_report.get("blockers") or ())
    blockers = list(dict.fromkeys(blockers))
    return {
        "schema_version": 3,
        "action": requested,
        "allowed": not blockers,
        "status": "PASS" if not blockers else "BLOCKED",
        "blockers": blockers,
        "readiness_ledger_head_sha256": ledger["head_sha256"],
        "readiness_ledger_event_count": ledger["event_count"],
        "readiness_os_anchor_status": ledger["os_anchor"]["status"],
        "readiness_os_anchor_record_id": ledger["os_anchor"]["anchor_record_id"],
        "readiness_os_anchor_baseline_record_id": ledger["os_anchor"][
            "baseline_record_id"
        ],
        "policy_sha256": PINNED_POLICY_SHA256,
        "scope_amendment_sha256": PINNED_SCOPE_AMENDMENT_SHA256,
        "outer_test_touched": False,
        "lora_unlocked": False,
        "human_approval_required": False,
        "manual_override_allowed": False,
        "main_method_x_battery_affected": False,
    }


def assess_brainiac_execution_gate(action: str) -> dict[str, Any]:
    """Read-only canonical status; this report alone is not an I/O capability."""

    return _assess_canonical(action)


def assert_brainiac_execution_allowed(action: str) -> dict[str, Any]:
    """Require current readiness plus an active lease for the exact action."""

    try:
        lease = require_active_lease(str(action).upper())
    except RuntimeError as exc:
        raise _blocked(str(exc), exc)
    report = _assess_canonical(
        str(action).upper(), exclusion={"status": "PASS", "blockers": []}
    )
    if not report["allowed"]:
        raise _blocked(",".join(report["blockers"]))
    return {**report, "lease_pid": lease.pid, "lease_nonce": lease.nonce}


@contextmanager
def brainiac_action_lease(action: str) -> Iterator[dict[str, Any]]:
    """Hold the runner-recognized guard throughout one protected action."""

    requested = str(action).upper()
    before = assess_brainiac_execution_gate(requested)
    if not before["allowed"]:
        raise _blocked(",".join(before["blockers"]))
    try:
        lease: BrainIACActionLease = acquire_action_guard(requested)
    except RuntimeError as exc:
        raise _blocked(str(exc), exc)
    token = activate_lease(lease)
    try:
        under_guard = _assess_canonical(
            requested,
            exclusion={"status": "PASS", "blockers": []},
        )
        if not under_guard["allowed"]:
            raise _blocked(",".join(under_guard["blockers"]))
        yield {
            **under_guard,
            "lease_pid": lease.pid,
            "lease_nonce": lease.nonce,
        }
    finally:
        deactivate_lease(token)
        release_action_guard(lease)
