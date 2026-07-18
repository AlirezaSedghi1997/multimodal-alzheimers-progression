from __future__ import annotations

from typing import Any

from preflight import compare_transform_status


def evaluate_post_transform_gate(
    transform: dict[str, Any],
    live_status: dict[str, Any] | None,
) -> dict[str, Any]:
    if transform.get("active_writer"):
        return {
            "status": "BLOCKED_ACTIVE_WRITER",
            "training_allowed": False,
            "outer_test_allowed": False,
            "claim_allowed": False,
            "reason": "transform writer is still active",
        }
    consistency = compare_transform_status(transform, live_status, active_writer=False)
    if consistency["status"] != "PASS":
        return {
            "status": "BLOCKED_STATUS_RECONCILIATION",
            "training_allowed": False,
            "outer_test_allowed": False,
            "claim_allowed": False,
            "reason": consistency,
        }
    if live_status and live_status.get("status") in {"RUNNING", "IN_PROGRESS"}:
        return {
            "status": "BLOCKED_NONTERMINAL_STATUS",
            "training_allowed": False,
            "outer_test_allowed": False,
            "claim_allowed": False,
            "reason": "live status remains non-terminal",
        }
    return {
        "status": "READY_FOR_POST_TRANSFORM_QC",
        "training_allowed": False,
        "outer_test_allowed": False,
        "claim_allowed": False,
        "reason": "pre-QC gate passed; quantitative validation still required",
    }
