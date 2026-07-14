from __future__ import annotations


class PreprocessingContractError(ValueError):
    """Raised when preprocessing would erase quantitative meaning or physical geometry."""


def validate_intensity_contract(
    *, branch: str, intensity_mode: str, affine_reference_sha256: str | None
) -> None:
    if not affine_reference_sha256 or len(affine_reference_sha256) != 64:
        raise PreprocessingContractError("AFFINE_REFERENCE_REQUIRED")
    if branch == "amyloid" and intensity_mode in {
        "zscore",
        "per_scan_zscore",
        "foreground_zscore",
        "high_quality_foreground_zscore",
    }:
        raise PreprocessingContractError("AMYLOID_PER_SCAN_ZSCORE_FORBIDDEN")
    allowed = {
        "suvr_preserving",
        "regional_suvr",
        "centiloid",
        "exploratory_foreground_zscore",
    }
    if intensity_mode not in allowed and intensity_mode != "per_scan_zscore":
        raise PreprocessingContractError(f"UNSUPPORTED_INTENSITY_MODE:{intensity_mode}")
    if branch not in {"amyloid", "tau"}:
        raise PreprocessingContractError(f"UNSUPPORTED_BRANCH:{branch}")
