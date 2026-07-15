from __future__ import annotations

from typing import Iterable


class RegionalFeatureError(ValueError):
    """Raised if an outcome, identifier, or future measurement enters a predictor set."""


FORBIDDEN_FEATURE_TOKENS = (
    "event",
    "time_days",
    "followup",
    "future",
    "patient_hash",
    "subject_id",
    "raw_path",
)


def select_tau_regions(columns: Iterable[str], *, prefix: str = "tau_persi__") -> list[str]:
    selected = sorted(
        str(column)
        for column in columns
        if str(column).startswith(prefix)
        and not any(token in str(column).casefold() for token in FORBIDDEN_FEATURE_TOKENS)
    )
    if not selected:
        raise RegionalFeatureError("NO_DOCUMENTED_TAU_REGIONAL_FEATURES")
    return selected


def build_matched_feature_sets(
    *, base_features: Iterable[str], pet_features: Iterable[str]
) -> dict[str, list[str]]:
    base = list(dict.fromkeys(str(value) for value in base_features))
    pet = list(dict.fromkeys(str(value) for value in pet_features))
    for feature in [*base, *pet]:
        if any(token in feature.casefold() for token in FORBIDDEN_FEATURE_TOKENS):
            raise RegionalFeatureError(f"FORBIDDEN_MODEL_FEATURE:{feature}")
    overlap = set(base).intersection(pet)
    if overlap:
        raise RegionalFeatureError(f"PET_BASE_FEATURE_OVERLAP:{sorted(overlap)}")
    return {
        "clinical_cognition": base,
        "clinical_cognition_pet": [*base, *pet],
    }
