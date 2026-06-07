from __future__ import annotations

import json
from pathlib import Path
from typing import Any


VALIDITY_AUDIT_PATH = Path("results/tables/validity_audit/validity_audit_summary.json")


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def spm_progress() -> dict[str, Any]:
    try:
        from scripts.spm_completion_guard import spm_completion_status

        return spm_completion_status()
    except Exception as exc:
        return {
            "ready": False,
            "status": "SPM_STATUS_UNAVAILABLE",
            "error": str(exc),
            "privacy": {
                "raw_identifiers_exported": False,
                "raw_paths_exported": False,
                "subject_level_outputs_exported": False,
            },
        }


def spm_is_complete() -> bool:
    return bool(spm_progress().get("ready"))


def validity_recommendation() -> str:
    audit = _read_json(VALIDITY_AUDIT_PATH)
    if not audit:
        return "PENDING_VALIDITY_AUDIT"
    verdict = audit.get("verdict")
    if not isinstance(verdict, dict):
        return "PENDING_VALIDITY_AUDIT"
    return str(verdict.get("final_claim_recommendation") or "PENDING_VALIDITY_AUDIT")


def validity_allows_strong_claim(recommendation: str | None = None) -> bool:
    rec = recommendation or validity_recommendation()
    return rec in {
        "CLAIM_ALLOWED",
        "CLAIM_ALLOWED_WITH_LIMITATIONS",
        "CLAIM_LIMITED_AFTER_REVIEW",
    }


def three_d_gate_status(three_d_metrics: dict[str, Any] | None) -> str:
    if not three_d_metrics:
        return "PENDING"
    aggregate = three_d_metrics.get("aggregate", {})
    if aggregate.get("final_claim_allowed") is True:
        return "PASSED"
    if aggregate.get("mode") == "FULL_3D_TRAINING":
        return "PARTIAL"
    if aggregate.get("mode") == "CPU_DEBUG_ONLY":
        return "DEBUG_ONLY"
    return "PENDING"


def evaluate_three_d_final_claim(
    *,
    mode: str,
    phase: str,
    resolution: int,
    folds: int,
    cache_resolution: int | None,
) -> dict[str, Any]:
    rec = validity_recommendation()
    spm_complete = spm_is_complete()
    blockers: list[str] = []
    if mode != "FULL_3D_TRAINING":
        blockers.append("NOT_FULL_3D_TRAINING")
    if phase != "final":
        blockers.append("NOT_FINAL_PHASE")
    if int(resolution) != 96:
        blockers.append("FINAL_REQUIRES_96_RESOLUTION")
    if cache_resolution is not None and int(cache_resolution) != 96:
        blockers.append("CACHE_RESOLUTION_NOT_96")
    if int(folds) != 5:
        blockers.append("FINAL_REQUIRES_5_FOLDS")
    if not spm_complete:
        blockers.append("SPM_NOT_COMPLETE")
    if not validity_allows_strong_claim(rec):
        blockers.append("VALIDITY_AUDIT_LIMITS_OR_BLOCKS_STRONG_CLAIM")

    final_claim_allowed = not blockers
    return {
        "final_claim_allowed": final_claim_allowed,
        "reporting_mode": "FINAL_STRONG_CLAIM_ALLOWED" if final_claim_allowed else "LIMITED_OR_DEBUG_REPORTING_ONLY",
        "validity_recommendation": rec,
        "spm_complete": spm_complete,
        "claim_blockers": blockers,
        "requirements": {
            "mode": "FULL_3D_TRAINING",
            "phase": "final",
            "resolution": 96,
            "folds": 5,
            "cache_resolution": 96,
            "spm_complete": True,
            "validity_allows_strong_claim": True,
        },
        "privacy": {
            "pet_as_input_allowed": False,
            "raw_identifiers_exported": False,
            "raw_paths_exported": False,
        },
    }
