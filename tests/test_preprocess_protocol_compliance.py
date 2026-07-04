from __future__ import annotations

import pytest


def test_method_y_stage_blocks_legacy_pet_zscore_cache():
    from scripts.method_y._stage_common import validate_method_y_pet_cache_protocol

    with pytest.raises(SystemExit) as exc:
        validate_method_y_pet_cache_protocol(
            {"inputs": {"pet_h5": "caches/method_y/pet_mni_128.h5"}},
            stage="S1",
        )

    assert "BLOCKED_LEGACY_PET_ZSCORE_CACHE_NOT_PROTOCOL_COMPLIANT" in str(exc.value)


def test_method_y_default_pet_cache_is_suvr_preserving():
    from scripts.method_y._parallel_run import selected_input_paths
    from scripts.method_y._stage_common import DEFAULT_METHOD_Y_PET_CACHE, validate_method_y_pet_cache_protocol

    validate_method_y_pet_cache_protocol({"inputs": {}}, stage="S1")
    paths = selected_input_paths("S1", {"inputs": {}, "folds": {}}, "configs/method_y_stage_s1.yaml")

    assert DEFAULT_METHOD_Y_PET_CACHE.endswith("pet_mni_128_broad_s1_fbp_only_hq_v1.h5")
    assert paths["pet_h5"] == DEFAULT_METHOD_Y_PET_CACHE


def test_protocol_audit_finds_no_unsafe_legacy_pet_defaults():
    from scripts.audit_mri_pet_protocol_compliance import code_legacy_default_references

    assert code_legacy_default_references() == []


def test_method_y_stage_allows_explicit_legacy_exploratory_override():
    from scripts.method_y._stage_common import validate_method_y_pet_cache_protocol

    validate_method_y_pet_cache_protocol(
        {
            "inputs": {
                "pet_h5": "caches/method_y/pet_mni_128.h5",
                "allow_legacy_zscore_cache_for_exploratory_only": True,
            }
        },
        stage="S1",
    )


def test_pet_intensity_augment_is_very_light_and_nonnegative():
    torch = pytest.importorskip("torch")
    from src.models.method_y.petnet_lite import pet_intensity_augment

    image = torch.ones(2, 1, 8, 8, 8) * 1.5
    out = pet_intensity_augment(image)

    assert out.shape == image.shape
    assert torch.isfinite(out).all()
    assert float(out.min()) >= 0.0
    assert abs(float(out.mean()) - 1.5) < 0.02
