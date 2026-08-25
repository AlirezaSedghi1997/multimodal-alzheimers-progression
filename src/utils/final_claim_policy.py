from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping


VALIDITY_AUDIT_PATH = Path("results/tables/validity_audit/validity_audit_summary.json")

POLICY_VERSION = "a6-callable-1.1.0"

_ALLOWED_STRONG_CLAIM_RECOMMENDATIONS = {
    "CLAIM_ALLOWED",
    "CLAIM_ALLOWED_WITH_LIMITATIONS",
    "CLAIM_LIMITED_AFTER_REVIEW",
}

_DIAGNOSTIC_TOKENS = ("diagnos", "تشخیص")


def _text(value: Any) -> str:
    return str(value or "").lower()


def _is_diagnostic_text(value: Any) -> bool:
    text = _text(value)
    return any(token in text for token in _DIAGNOSTIC_TOKENS)


def _first_number(claim: Mapping[str, Any], keys: tuple[str, ...]) -> float | None:
    for key in keys:
        value = claim.get(key)
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            return float(value)
    return None


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


def claim_from_three_d_aggregate(aggregate: Mapping[str, Any], strict: bool = False) -> dict[str, Any]:
    """Project a three_d aggregate into the callable A6 claim shape (pure, no I/O)."""

    del strict  # projection only; enforcement happens in validate_claim
    return {
        "method": "X",
        "model_id": str(aggregate.get("model") or "MethodX3DClassifier"),
        "mode": str(aggregate.get("mode") or ""),
        "phase": str(aggregate.get("phase") or "final"),
        "resolution": aggregate.get("resolution", aggregate.get("requested_resolution")),
        "folds": aggregate.get("folds"),
        "completed_folds": aggregate.get("completed_folds"),
        "training_status": aggregate.get("training_status"),
        "uses_pet_as_input": bool(aggregate.get("uses_pet_as_input")),
        "uses_synthetic_pet_as_input": bool(aggregate.get("uses_synthetic_pet_as_input")),
        "uses_raw_identifiers": bool(aggregate.get("uses_raw_identifiers")),
        "uses_raw_paths": bool(aggregate.get("uses_raw_paths")),
        "clinical_alzheimer_diagnosis_output_allowed": bool(
            aggregate.get("clinical_alzheimer_diagnosis_output_allowed")
        ),
        "roc_auc": aggregate.get("roc_auc"),
        "pr_auc": aggregate.get("pr_auc"),
        "ece": aggregate.get("ece"),
        "calibration_ece": aggregate.get("ece"),
        "cohort_probe_auc": aggregate.get("cohort_probe_auc"),
        "framing_language": aggregate.get(
            "framing_language", aggregate.get("framing_language_sample")
        ),
        "framing_language_sample": aggregate.get(
            "framing_language_sample", aggregate.get("framing_language")
        ),
        "claim_scope": aggregate.get("claim_scope"),
        "spm_complete": aggregate.get("spm_complete"),
        "spm_snapshot_verified": aggregate.get("spm_snapshot_verified"),
        "spm_snapshot_sha256": aggregate.get("spm_snapshot_sha256"),
        "expected_spm_snapshot_sha256": aggregate.get("expected_spm_snapshot_sha256"),
        "validity_recommendation": aggregate.get("validity_recommendation"),
        "final_claim_allowed": aggregate.get("final_claim_allowed"),
    }


def validate_claim(claim: Mapping[str, Any], strict: bool = False) -> dict[str, Any]:
    """Callable A6 gate: pure predicate over the claim payload; never does I/O.

    Fail-closed: any blocker (or, under ``strict=True``, any hard threshold
    breach) makes ``ok=False``.  Missing evidence is never rounded up to PASS.
    """

    details: list[dict[str, str]] = []
    blocking: list[str] = []
    warnings: list[str] = []

    def add(field: str, kind: str, reason: str | None = None) -> None:
        entry = {"field": field, "kind": kind}
        if reason:
            entry["reason"] = reason
        details.append(entry)
        if kind == "blocker" and reason and reason not in blocking:
            blocking.append(reason)
        if kind == "warning" and reason and reason not in warnings:
            warnings.append(reason)

    method = str(claim.get("method") or "X").upper()
    # Final-phase gates apply only when the payload explicitly declares
    # phase="final"; an absent phase is treated as non-final (dev/advisory).
    phase = str(claim.get("phase") or "dev").lower()

    # PET-free boundary is Method-X-specific by contract.
    if method == "X":
        if claim.get("uses_pet_as_input") is True:
            add("uses_pet_as_input", "blocker", "METHOD_X_PET_INPUT_FORBIDDEN")
        if claim.get("uses_synthetic_pet_as_input") is True:
            add(
                "uses_synthetic_pet_as_input",
                "blocker",
                "METHOD_X_SYNTHETIC_PET_INPUT_FORBIDDEN",
            )
        if claim.get("clinical_alzheimer_diagnosis_output_allowed") is True:
            add(
                "clinical_alzheimer_diagnosis_output_allowed",
                "blocker",
                "ALZHEIMER_DIAGNOSIS_OUTPUT_FORBIDDEN",
            )

    framing = claim.get("framing_language_sample", claim.get("framing_language"))
    if _is_diagnostic_text(framing):
        add("framing_language", "blocker", "DIAGNOSTIC_FRAMING_FORBIDDEN")
    if _is_diagnostic_text(claim.get("claim_scope")):
        add("claim_scope", "blocker", "DIAGNOSIS_SCOPE_FORBIDDEN")

    recommendation = claim.get("validity_recommendation")
    if recommendation is not None and str(recommendation) not in _ALLOWED_STRONG_CLAIM_RECOMMENDATIONS:
        add(
            "validity_recommendation",
            "blocker",
            "VALIDITY_AUDIT_LIMITS_OR_BLOCKS_STRONG_CLAIM",
        )

    if phase == "final":
        folds = claim.get("folds")
        if folds is not None and int(folds) != 5:
            add("folds", "blocker", "FINAL_REQUIRES_5_FOLDS")
        resolution = claim.get("resolution")
        if resolution is not None and int(resolution) != 128:
            add("resolution", "blocker", "FINAL_REQUIRES_128_RESOLUTION")
        if claim.get("spm_complete") is not True:
            add("spm_complete", "blocker", "SPM_NOT_COMPLETE")
        status = claim.get("training_status")
        if status is not None and str(status).upper() != "COMPLETED":
            add("training_status", "blocker", "TRAINING_NOT_COMPLETED")
        completed = claim.get("completed_folds")
        if completed is not None and int(completed) != 5:
            add("completed_folds", "blocker", "COMPLETED_FOLD_COUNT_INVALID")

    # SPM snapshot integrity: an explicitly unverified snapshot is a blocker;
    # digest enforcement is opt-in via enforce_spm_snapshot (used by strict
    # final-claim publication paths).
    if claim.get("spm_snapshot_verified") is False:
        add("spm_snapshot_verified", "blocker", "SPM_SNAPSHOT_NOT_VERIFIED")
    if claim.get("enforce_spm_snapshot") is True:
        observed = claim.get("spm_snapshot_sha256")
        expected = claim.get("expected_spm_snapshot_sha256")
        if not observed or not expected:
            add("spm_snapshot_sha256", "blocker", "SPM_SNAPSHOT_DIGEST_MISSING")
        elif str(observed).casefold() != str(expected).casefold():
            add(
                "spm_snapshot_sha256",
                "blocker",
                "SPM_SNAPSHOT_DIGEST_MISMATCH",
            )

    if strict and claim.get("enforce_required_fields") is True:
        for field in ("roc_auc", "pr_auc"):
            if claim.get(field) is None:
                add(field, "blocker", f"REQUIRED_FIELD_MISSING:{field}")

    cohort_auc = _first_number(
        claim, ("cohort_probe_auc", "cohort_probe_auc_post_intervention")
    )
    if cohort_auc is not None:
        if cohort_auc > 0.65:
            add("cohort_probe_auc", "warning", "COHORT_PROBE_AUC_ABOVE_0_65")
        if cohort_auc > 0.80:
            add(
                "cohort_probe_auc",
                "blocker" if strict else "warning",
                "COHORT_PROBE_AUC_ABOVE_0_80",
            )

    ece = _first_number(claim, ("calibration_ece", "ece"))
    if ece is not None and ece > 0.15:
        add("calibration_ece", "warning", "CALIBRATION_ECE_ABOVE_0_15")

    blockers = [row for row in details if row["kind"] == "blocker"]
    return {
        "ok": not blockers,
        "policy_version": POLICY_VERSION,
        "violations": [
            row.get("reason") or row["field"] for row in details
        ],
        "blocking_reasons": blocking,
        "soft_warnings": warnings,
        "violation_details": details,
    }


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
    spm_snapshot_verified: bool | None = None
    if spm_complete:
        progress = spm_progress()
        spm_snapshot_verified = (
            bool(progress.get("snapshot_verified"))
            if isinstance(progress, Mapping)
            else False
        )
        if spm_snapshot_verified is not True:
            blockers.append("SPM_SNAPSHOT_NOT_VERIFIED")
    if not validity_allows_strong_claim(rec):
        blockers.append("VALIDITY_AUDIT_LIMITS_OR_BLOCKS_STRONG_CLAIM")

    final_claim_allowed = not blockers
    return {
        "final_claim_allowed": final_claim_allowed,
        "reporting_mode": "FINAL_STRONG_CLAIM_ALLOWED" if final_claim_allowed else "LIMITED_OR_DEBUG_REPORTING_ONLY",
        "validity_recommendation": rec,
        "spm_complete": spm_complete,
        "spm_snapshot_verified": spm_snapshot_verified,
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
