from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[2]

VALIDITY_AUDIT_PATH = PROJECT_ROOT / "results/tables/validity_audit/validity_audit_summary.json"

# The SPM guard reads these with CWD-relative defaults.  They are re-anchored to
# the repository root here so a fail-closed gate can never silently report
# "incomplete" for a genuinely complete run merely because the process was
# started from another directory.  ``scripts/spm_completion_guard.py`` itself is
# left byte-identical: it is attested by the SPM gate snapshot, so editing it
# would invalidate that attestation.
SPM_GUARD_MANIFEST_PATH = PROJECT_ROOT / "results/logs/spm_mri_preprocess_v2_128_resume.json"
SPM_GUARD_PREPROCESSED_DIR = PROJECT_ROOT / "data/preprocessed_mri_128"
SPM_GUARD_CATALOG_PATH = PROJECT_ROOT / "data/processed/master_catalog.csv"
SPM_GATE_SNAPSHOT_PATH = PROJECT_ROOT / "results/method_x_registry/spm_gate_snapshot.json"

# Official Method-X final protocol (128^3 / 1.5mm, 5 folds).  The 96^3 legacy
# policy is retired: SPM preprocessing, the volume cache, and both claim gates
# are 128-based.  ``tests/test_proposal_extension_modules.py`` locks this.
OFFICIAL_RESOLUTION = 128
OFFICIAL_FINAL_FOLDS = 5
SUPPORTED_FINAL_RESOLUTIONS = frozenset({OFFICIAL_RESOLUTION})

POLICY_VERSION = "a6-callable-1.2.0"

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


def spm_snapshot_verification() -> bool:
    """Verify the SPM gate snapshot attestation.

    The previous implementation read a ``snapshot_verified`` key that the guard
    never emitted, so this check could never succeed and permanently blocked
    every strong final claim.  The real verifier is used instead.  Fail-closed:
    a missing snapshot, an unreadable file, or any failed check reports False.
    """

    try:
        from src.training.method_x_spm_snapshot import verify_spm_gate_snapshot

        result = verify_spm_gate_snapshot(SPM_GATE_SNAPSHOT_PATH, root=PROJECT_ROOT)
    except Exception:
        return False
    return result.get("verified") is True


def spm_progress() -> dict[str, Any]:
    try:
        from scripts.spm_completion_guard import spm_completion_status

        status = spm_completion_status(
            manifest_path=SPM_GUARD_MANIFEST_PATH,
            preprocessed_dir=SPM_GUARD_PREPROCESSED_DIR,
            catalog_path=SPM_GUARD_CATALOG_PATH,
        )
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
    if status.get("ready") is True and status.get("snapshot_verified") is None:
        # Attested by the guard snapshot rather than by the guard itself.
        status = {**status, "snapshot_verified": spm_snapshot_verification()}
    return status


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


def _aggregate_claims_strong_final_claim(aggregate: Mapping[str, Any]) -> bool:
    """Whether this run asserts a strong final claim.

    A selection-only / inner-validity arm is a private hyperparameter-selection
    measurement: it never accesses the outer test, it is explicitly reported as
    ``INNER_SELECTION_ONLY_NOT_FINAL_REPORTING`` and its ``final_claim_allowed``
    is already False.  Such a run must not be terminated for failing to qualify
    as a final claim, but it is still subject to every boundary and structural
    gate.  Any other run (absent/False selection flags) asserts the strong claim
    and keeps the full eligibility gate.
    """

    if aggregate.get("hpo_selection_only") is True:
        return False
    if aggregate.get("validity_inner_only") is True:
        return False
    return True


def claim_from_three_d_aggregate(aggregate: Mapping[str, Any], strict: bool = False) -> dict[str, Any]:
    """Project a three_d aggregate into the callable A6 claim shape (pure, no I/O).

    The aggregate carries the authoritative claim verdict inside its nested
    ``claim_policy`` object.  Reading only flat keys would silently drop it and
    make ``validate_claim`` judge a *missing* field rather than the real rule,
    so the nested object is the fallback source for every projected field.
    """

    del strict  # projection only; enforcement happens in validate_claim
    policy = aggregate.get("claim_policy")
    if not isinstance(policy, Mapping):
        policy = {}

    def field(key: str, default: Any = None) -> Any:
        """Flat aggregate value first, then the authoritative nested verdict."""

        if key in aggregate:
            return aggregate[key]
        return policy.get(key, default)

    return {
        "method": "X",
        "model_id": str(aggregate.get("model") or "MethodX3DClassifier"),
        "mode": str(aggregate.get("mode") or ""),
        "phase": str(aggregate.get("phase") or "final"),
        "resolution": aggregate.get("resolution", aggregate.get("requested_resolution")),
        "folds": aggregate.get("folds"),
        "completed_folds": aggregate.get("completed_folds"),
        "training_status": field("training_status"),
        # Preserve missing declarations as UNKNOWN.  bool(None) would silently
        # turn absent boundary evidence into a reassuring False.
        "uses_pet_as_input": aggregate.get("uses_pet_as_input"),
        "uses_synthetic_pet_as_input": aggregate.get("uses_synthetic_pet_as_input"),
        "uses_raw_identifiers": aggregate.get("uses_raw_identifiers"),
        "uses_raw_paths": aggregate.get("uses_raw_paths"),
        "clinical_alzheimer_diagnosis_output_allowed": aggregate.get(
            "clinical_alzheimer_diagnosis_output_allowed"
        ),
        "roc_auc": aggregate.get("roc_auc"),
        "pr_auc": aggregate.get("pr_auc"),
        "ece": field("ece"),
        "calibration_ece": field("ece"),
        "cohort_probe_auc": aggregate.get("cohort_probe_auc"),
        "framing_language": aggregate.get(
            "framing_language", aggregate.get("framing_language_sample")
        ),
        "framing_language_sample": aggregate.get(
            "framing_language_sample", aggregate.get("framing_language")
        ),
        "claim_scope": aggregate.get("claim_scope"),
        "spm_complete": field("spm_complete"),
        "spm_snapshot_verified": field("spm_snapshot_verified"),
        "spm_snapshot_sha256": field("spm_snapshot_sha256"),
        "expected_spm_snapshot_sha256": field("expected_spm_snapshot_sha256"),
        "validity_recommendation": field("validity_recommendation"),
        "final_claim_allowed": aggregate.get("final_claim_allowed"),
        "reporting_mode": field("reporting_mode"),
        "hpo_selection_only": aggregate.get("hpo_selection_only"),
        "validity_inner_only": aggregate.get("validity_inner_only"),
        "evaluation_scope": aggregate.get("evaluation_scope"),
        "outer_test_access_count": aggregate.get("outer_test_access_count"),
        "asserts_strong_final_claim": _aggregate_claims_strong_final_claim(aggregate),
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
    # Strong-claim *eligibility* (validity-audit verdict, snapshot attestation)
    # applies only to a payload that actually asserts a strong final claim.  An
    # inner/selection-only run asserts none, yet is still held to every
    # boundary clause and every structural protocol gate below, and the
    # authoritative eligibility verdict is persisted separately by
    # evaluate_three_d_final_claim (``final_claim_allowed`` /
    # ``reporting_mode``), so nothing is rounded up to PASS.  The key is absent
    # for every other payload, which preserves the historic fail-closed
    # behaviour; absence of evidence is never treated as "no strong claim".
    asserts_strong_final_claim = claim.get("asserts_strong_final_claim") is not False
    full_final = phase == "final" and claim.get("mode") == "FULL_3D_TRAINING"
    selection_only = (
        claim.get("hpo_selection_only") is True
        or claim.get("validity_inner_only") is True
    )

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
        if claim.get("uses_raw_identifiers") is True:
            add("uses_raw_identifiers", "blocker", "RAW_IDENTIFIERS_EXPORTED")
        if claim.get("uses_raw_paths") is True:
            add("uses_raw_paths", "blocker", "RAW_PATHS_EXPORTED")
        if full_final:
            for field in (
                "uses_pet_as_input", "uses_synthetic_pet_as_input",
                "uses_raw_identifiers", "uses_raw_paths",
                "clinical_alzheimer_diagnosis_output_allowed",
            ):
                if type(claim.get(field)) is not bool:
                    add(field, "blocker", f"BOUNDARY_DECLARATION_MISSING:{field}")

    if full_final and selection_only:
        if (type(claim.get("outer_test_access_count")) is not int
            or claim.get("outer_test_access_count") != 0):
            add("outer_test_access_count", "blocker", "SELECTION_ONLY_OUTER_ACCESS_INVALID")
        if claim.get("evaluation_scope") != "inner_validation_only":
            add("evaluation_scope", "blocker", "SELECTION_ONLY_SCOPE_INVALID")
        if claim.get("final_claim_allowed") is not False:
            add("final_claim_allowed", "blocker", "SELECTION_ONLY_FINAL_CLAIM_FORBIDDEN")
        if claim.get("reporting_mode") != "INNER_SELECTION_ONLY_NOT_FINAL_REPORTING":
            add("reporting_mode", "blocker", "SELECTION_ONLY_REPORTING_MODE_INVALID")

    framing = claim.get("framing_language_sample", claim.get("framing_language"))
    if _is_diagnostic_text(framing):
        add("framing_language", "blocker", "DIAGNOSTIC_FRAMING_FORBIDDEN")
    if _is_diagnostic_text(claim.get("claim_scope")):
        add("claim_scope", "blocker", "DIAGNOSIS_SCOPE_FORBIDDEN")

    recommendation = claim.get("validity_recommendation")
    if full_final and asserts_strong_final_claim and recommendation is None:
        add("validity_recommendation", "blocker", "VALIDITY_AUDIT_VERDICT_MISSING")
    elif (
        asserts_strong_final_claim
        and recommendation is not None
        and str(recommendation) not in _ALLOWED_STRONG_CLAIM_RECOMMENDATIONS
    ):
        add(
            "validity_recommendation",
            "blocker",
            "VALIDITY_AUDIT_LIMITS_OR_BLOCKS_STRONG_CLAIM",
        )

    if phase == "final":
        def is_expected_integer(value: Any, expected: int) -> bool:
            if type(value) is bool:
                return False
            try:
                return int(value) == expected
            except (TypeError, ValueError, OverflowError):
                return False

        folds = claim.get("folds")
        if (full_final and type(folds) is not int) or (
            folds is not None and not is_expected_integer(folds, 5)
        ):
            add("folds", "blocker", "FINAL_REQUIRES_5_FOLDS")
        resolution = claim.get("resolution")
        if (full_final and type(resolution) is not int) or (
            resolution is not None and not is_expected_integer(resolution, 128)
        ):
            add("resolution", "blocker", "FINAL_REQUIRES_128_RESOLUTION")
        if claim.get("spm_complete") is not True:
            add("spm_complete", "blocker", "SPM_NOT_COMPLETE")
        status = claim.get("training_status")
        if (full_final and status is None) or (status is not None and str(status).upper() != "COMPLETED"):
            add("training_status", "blocker", "TRAINING_NOT_COMPLETED")
        completed = claim.get("completed_folds")
        if (full_final and type(completed) is not int) or (
            completed is not None and not is_expected_integer(completed, 5)
        ):
            add("completed_folds", "blocker", "COMPLETED_FOLD_COUNT_INVALID")

    # SPM snapshot integrity: an explicitly unverified snapshot is a blocker;
    # digest enforcement is opt-in via enforce_spm_snapshot (used by strict
    # final-claim publication paths).  Like the validity verdict this attests a
    # released claim, so it is scoped to a payload that asserts one.
    if (full_final and asserts_strong_final_claim
        and claim.get("spm_snapshot_verified") is not True) or (
        asserts_strong_final_claim and claim.get("spm_snapshot_verified") is False
    ):
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
    # One guard evaluation per claim evaluation: the previous implementation
    # scanned every preprocessed volume twice, doubling the cost of a gate that
    # runs after each fold.
    progress = spm_progress()
    spm_complete = bool(progress.get("ready"))
    blockers: list[str] = []
    if mode != "FULL_3D_TRAINING":
        blockers.append("NOT_FULL_3D_TRAINING")
    if phase != "final":
        blockers.append("NOT_FINAL_PHASE")
    if int(resolution) != OFFICIAL_RESOLUTION:
        blockers.append(f"FINAL_REQUIRES_{OFFICIAL_RESOLUTION}_RESOLUTION")
    if cache_resolution is not None and int(cache_resolution) != OFFICIAL_RESOLUTION:
        blockers.append(f"CACHE_RESOLUTION_NOT_{OFFICIAL_RESOLUTION}")
    if int(folds) != OFFICIAL_FINAL_FOLDS:
        blockers.append("FINAL_REQUIRES_5_FOLDS")
    if not spm_complete:
        blockers.append("SPM_NOT_COMPLETE")
    spm_snapshot_verified: bool | None = None
    if spm_complete:
        spm_snapshot_verified = (
            progress.get("snapshot_verified") if isinstance(progress, Mapping) else None
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
            "resolution": OFFICIAL_RESOLUTION,
            "folds": OFFICIAL_FINAL_FOLDS,
            "cache_resolution": OFFICIAL_RESOLUTION,
            "spm_complete": True,
            "validity_allows_strong_claim": True,
        },
        "privacy": {
            "pet_as_input_allowed": False,
            "raw_identifiers_exported": False,
            "raw_paths_exported": False,
        },
    }
