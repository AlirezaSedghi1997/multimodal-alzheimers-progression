from __future__ import annotations

from collections.abc import Mapping
from typing import Any


class ImageTrainingPolicyError(ValueError):
    """Raised when a readiness payload is malformed rather than scientifically limited."""


_BRANCH_TRACER = {"amyloid": "FBP", "tau": "FTP"}


def _bool(mapping: Mapping[str, Any], key: str) -> bool:
    value = mapping.get(key)
    if not isinstance(value, bool):
        raise ImageTrainingPolicyError(f"BOOLEAN_FIELD_REQUIRED:{key}")
    return value


def _integer(mapping: Mapping[str, Any], key: str) -> int:
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ImageTrainingPolicyError(f"INTEGER_FIELD_REQUIRED:{key}")
    return int(value)


def assess_image_training_policy(
    *,
    integrity: Mapping[str, Any],
    quantitative_equivalence_proven: bool,
    scanner_robustness_proven: bool,
    external_validation_available: bool,
    incremental_value_demonstrated: bool = False,
) -> dict[str, Any]:
    """Separate technical execution eligibility from scientific claim eligibility.

    Quantitative calibration, scanner transport and external replication are
    claim gates.  They must not be silently repaired with statistics, but their
    absence does not prohibit a prespecified internal exploratory model.  In
    contrast, tracer identity, future-outcome temporality, geometry, finite
    voxels, membership isolation and event support remain hard execution gates.
    """

    if not isinstance(integrity, Mapping):
        raise ImageTrainingPolicyError("INTEGRITY_MAPPING_REQUIRED")
    branch = str(integrity.get("branch", "")).strip().lower()
    if branch not in _BRANCH_TRACER:
        raise ImageTrainingPolicyError(f"UNSUPPORTED_IMAGE_BRANCH:{branch or 'MISSING'}")
    tracer = str(integrity.get("tracer", "")).strip().upper()

    hard_blockers: list[str] = []
    expected_tracer = _BRANCH_TRACER[branch]
    if tracer != expected_tracer:
        hard_blockers.append(f"{branch.upper()}_REQUIRES_{expected_tracer}_ONLY")
    if not _bool(integrity, "endpoint_locked"):
        hard_blockers.append("ENDPOINT_CONTRACT_NOT_LOCKED")
    if not _bool(integrity, "endpoint_is_future"):
        hard_blockers.append("FUTURE_ENDPOINT_REQUIRED")
    if not _bool(integrity, "split_locked"):
        hard_blockers.append("SPLIT_CONTRACT_NOT_LOCKED")
    if _integer(integrity, "outer_fold_count") != 5:
        hard_blockers.append("FIVE_OUTER_FOLDS_REQUIRED")
    if _integer(integrity, "subject_overlap_count") != 0:
        hard_blockers.append("SUBJECT_LEAKAGE_DETECTED")
    if _integer(integrity, "eligible_subject_count") <= 0:
        hard_blockers.append("NO_ELIGIBLE_IMAGE_SUBJECTS")
    if _integer(integrity, "eligible_event_count") <= 0:
        hard_blockers.append("NO_ELIGIBLE_EVENTS")
    if _integer(integrity, "invalid_images_included") != 0:
        hard_blockers.append("INVALID_IMAGE_INCLUDED")
    if _integer(integrity, "minimum_outer_test_events") <= 0:
        hard_blockers.append("OUTER_FOLD_WITHOUT_EVENT")
    if _integer(integrity, "minimum_inner_validation_events") <= 0:
        hard_blockers.append("INNER_FOLD_WITHOUT_EVENT")
    if not _bool(integrity, "all_images_finite"):
        hard_blockers.append("NONFINITE_IMAGE_INCLUDED")
    if not _bool(integrity, "all_images_geometry_valid"):
        hard_blockers.append("INVALID_IMAGE_GEOMETRY_INCLUDED")

    limitations: list[str] = []
    if not bool(quantitative_equivalence_proven):
        limitations.append("QUANTITATIVE_EQUIVALENCE_NOT_PROVEN")
    if not bool(scanner_robustness_proven):
        limitations.append("SCANNER_ROBUSTNESS_NOT_PROVEN")
    if not bool(external_validation_available):
        limitations.append("EXTERNAL_VALIDATION_UNAVAILABLE")
    if not bool(incremental_value_demonstrated):
        limitations.append("INCREMENTAL_VALUE_NOT_YET_DEMONSTRATED")

    execution_allowed = not hard_blockers
    positive_claim_allowed = execution_allowed and not limitations
    return {
        "schema_version": "3.0",
        "branch": branch,
        "tracer": tracer,
        "training_role": "FINAL_INTERNAL_EXPLORATORY_IMAGE_MODEL",
        "input_interpretation": (
            "PATTERN_NORMALIZED_FBP_NOT_OFFICIAL_CENTILOID"
            if branch == "amyloid"
            else "PATTERN_NORMALIZED_FTP_NOT_CENTAUR_QUANTITATIVE"
        ),
        "training_execution_allowed": execution_allowed,
        "internal_outer_evaluation_allowed_after_lock": execution_allowed,
        "positive_claim_allowed": positive_claim_allowed,
        "external_claim_allowed": (
            positive_claim_allowed and bool(external_validation_available)
        ),
        "hard_blockers": hard_blockers,
        "claim_limitations": limitations,
        "invalid_samples_must_be_excluded_not_imputed": True,
        "quantitative_scale_may_not_be_claimed": not bool(
            quantitative_equivalence_proven
        ),
        "outer_test_policy": "EXACTLY_ONCE_AFTER_INNER_SELECTION_AND_CHECKPOINT_LOCK",
    }

