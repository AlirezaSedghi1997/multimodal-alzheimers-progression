"""Machine-readable lifecycle state for the Method X transformation stages.

The stage registry is deliberately derived from immutable-ish JSON reports and
never asks an operator to classify a stage.  A missing report is represented as
``NOT_RUN`` and a malformed report as ``INVALID``; neither state can silently
promote a run to a thesis-grade result.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from src.utils.io_helpers import atomic_write_json


STAGE_REPORTS: dict[str, str] = {
    "MX-03": "reports/mx03_validity_orchestrator.json",
    "MX-04": "reports/mx04_finetune_orchestrator.json",
    "MX-05": "reports/mx05_hybrid_orchestrator.json",
}
XAI_REPORT = "results/final_results/method_x_anatomical_xai_v3/anatomical_xai_summary.json"
VALIDITY_ORCHESTRATOR_REPORT = "reports/mx03_validity_orchestrator.json"


def _sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_report(root: Path, relative_path: str) -> dict[str, Any]:
    path = root / relative_path
    digest = _sha256(path)
    if not path.is_file():
        return {
            "status": "NOT_RUN",
            "report_path": relative_path,
            "report_sha256": None,
            "blockers": ["STAGE_REPORT_MISSING"],
        }
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return {
            "status": "INVALID",
            "report_path": relative_path,
            "report_sha256": digest,
            "blockers": ["STAGE_REPORT_INVALID_JSON"],
        }
    if not isinstance(payload, dict):
        return {
            "status": "INVALID",
            "report_path": relative_path,
            "report_sha256": digest,
            "blockers": ["STAGE_REPORT_NOT_OBJECT"],
        }
    status = str(payload.get("status") or "UNKNOWN").upper()
    blockers = payload.get("blockers")
    if not isinstance(blockers, list):
        blockers = []
    if status.startswith("BLOCKED") and not blockers:
        # Guarded/legacy reports often encode the blocker in ``status`` only.
        # Promote it into the common blocker channel rather than treating a
        # blocked artifact as merely informational.
        blockers = [status]
    supervision = payload.get("autonomous_supervision")
    # Older stage reports used only supervision_decision.  Normalize both
    # forms so registry consumers have one stable schema.
    decision = payload.get("supervision_decision")
    if isinstance(supervision, Mapping):
        decision = supervision.get("decision", decision)
    return {
        "status": status,
        "stage": payload.get("stage"),
        "report_path": relative_path,
        "report_sha256": digest,
        "blockers": [str(item) for item in blockers],
        "next_automatic_action": payload.get("next_automatic_action"),
        "supervision_decision": decision,
        "training_started": bool(payload.get("training_started", False)),
    }


def collect_stage_readiness(*, root: Path, output_root: Path) -> dict[str, Any]:
    """Collect MX-03..06 state without changing any training artifact."""

    root = root.resolve()
    output_root = output_root.resolve()
    stages: dict[str, dict[str, Any]] = {}
    for stage, relative in STAGE_REPORTS.items():
        # MX-03..05 reports live under the selected battery output root.
        stages[stage] = _load_report(output_root, relative)
    stages["MX-06"] = _load_report(root, XAI_REPORT)
    validity_orchestrator = _load_report(output_root, VALIDITY_ORCHESTRATOR_REPORT)

    blocking = [
        f"{stage}:{code}"
        for stage, report in stages.items()
        for code in report.get("blockers", [])
        if str(report.get("status", "")).startswith("BLOCKED")
        or report.get("status") in {"INVALID", "NOT_RUN"}
    ]
    if validity_orchestrator.get("status") not in {"NOT_RUN", "READY", "COMPLETED", "PASS"}:
        blocking.extend(
            f"MX-03_ORCHESTRATOR:{code}" for code in validity_orchestrator.get("blockers", [])
        )
    statuses = {stage: report.get("status", "UNKNOWN") for stage, report in stages.items()}
    if any(str(value).startswith("BLOCKED") or value == "INVALID" for value in statuses.values()):
        overall = "BLOCKED"
    elif any(value == "NOT_RUN" for value in statuses.values()):
        overall = "INCOMPLETE"
    elif all(value in {"READY", "COMPLETED", "PASS"} for value in statuses.values()):
        overall = "READY"
    else:
        overall = "INCOMPLETE"
    return {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "output_root": str(output_root),
        "status": overall,
        "stages": stages,
        "validity_orchestrator": validity_orchestrator,
        "blocking_reasons": sorted(set(blocking)),
        "autonomous_supervision": {
            "mode": "AUTONOMOUS_MULTI_LAYER_FAIL_CLOSED",
            "human_approval_required": False,
            "manual_override_allowed": False,
            "fail_closed": True,
        },
        "training_started": any(report.get("training_started") for report in stages.values()),
    }


def write_stage_readiness(*, root: Path, output_root: Path) -> tuple[Path, dict[str, Any]]:
    """Write the aggregate stage report and return its path and payload."""

    report = collect_stage_readiness(root=root, output_root=output_root)
    path = output_root / "reports/method_x_stage_readiness.json"
    atomic_write_json(path, report)
    return path, report
