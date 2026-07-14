from __future__ import annotations

import numpy as np
import pytest


def test_affine_aware_resampling_matches_the_locked_reference_grid():
    import nibabel as nib

    from src.method_y_revival.pet_geometry import resample_to_reference

    source_data = np.zeros((8, 8, 8), dtype=np.float32)
    source_data[4, 4, 4] = 1.0
    source = nib.Nifti1Image(source_data, np.diag([2.0, 2.0, 2.0, 1.0]))
    reference = nib.Nifti1Image(
        np.zeros((16, 16, 16), dtype=np.float32), np.diag([1.0, 1.0, 1.0, 1.0])
    )

    result = resample_to_reference(source, reference)

    assert result.shape == reference.shape
    np.testing.assert_allclose(result.affine, reference.affine)
    peak = np.unravel_index(np.argmax(result.get_fdata()), result.shape)
    assert peak == (8, 8, 8)


def test_shape_only_array_resampling_is_rejected_for_claim_bearing_pet():
    from src.method_y_revival.pet_preprocessing import PreprocessingContractError, validate_intensity_contract

    with pytest.raises(PreprocessingContractError, match="AFFINE_REFERENCE_REQUIRED"):
        validate_intensity_contract(
            branch="amyloid",
            intensity_mode="suvr_preserving",
            affine_reference_sha256=None,
        )


def test_per_scan_zscore_is_rejected_for_claim_bearing_amyloid():
    from src.method_y_revival.pet_preprocessing import PreprocessingContractError, validate_intensity_contract

    with pytest.raises(PreprocessingContractError, match="AMYLOID_PER_SCAN_ZSCORE_FORBIDDEN"):
        validate_intensity_contract(
            branch="amyloid",
            intensity_mode="per_scan_zscore",
            affine_reference_sha256="a" * 64,
        )


def test_sustained_progression_uses_only_post_pet_visits_and_first_confirmed_increase():
    from src.method_y_revival.endpoints import build_sustained_progression

    endpoint = build_sustained_progression(
        pet_index_day=100.0,
        baseline_cdr=0.0,
        visits=[
            {"day": 90.0, "cdr_global": 0.5},
            {"day": 150.0, "cdr_global": 0.5},
            {"day": 200.0, "cdr_global": 0.5},
        ],
        max_followup_days=1640.0,
    )

    assert endpoint.event == 1
    assert endpoint.time_days == 50.0
    assert endpoint.confirmation_day == 200.0


def test_unconfirmed_last_visit_increase_is_right_censored_not_an_event():
    from src.method_y_revival.endpoints import build_sustained_progression

    endpoint = build_sustained_progression(
        pet_index_day=0.0,
        baseline_cdr=0.0,
        visits=[{"day": 300.0, "cdr_global": 0.0}, {"day": 600.0, "cdr_global": 0.5}],
        max_followup_days=1640.0,
    )

    assert endpoint.event == 0
    assert endpoint.time_days == 600.0
    assert endpoint.terminal_reason == "RIGHT_CENSORED_UNCONFIRMED_INCREASE"


def test_same_pet_derived_primary_target_is_rejected():
    from src.method_y_revival.endpoints import EndpointContractError, validate_target_contract

    with pytest.raises(EndpointContractError, match="SAME_PET_DERIVED_TARGET_FORBIDDEN"):
        validate_target_contract(
            branch="amyloid",
            target_name="amyloid_pos",
            target_source="same_input_pet",
            claim_bearing=True,
        )
