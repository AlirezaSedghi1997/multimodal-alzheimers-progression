from __future__ import annotations

import dataclasses

import numpy as np
import pytest
import torch

from src.method_y_revival.image_factorial_v4 import (
    FACTORIAL_ARMS,
    build_factorial_image_model,
    matched_arm_differences,
    select_inner_only_factorial_arm,
)
from src.method_y_revival.pet_augmentation_v4 import (
    PetAugmentationV4Error,
    PetSafeAugmentationConfig,
    apply_pet_safe_augmentation,
    apply_pet_safe_augmentation_batch,
    deterministic_augmentation_seed,
)
from src.method_y_revival.pet_augmentation_qc_v4 import select_largest_passing_envelope


def _safe_config(**overrides) -> PetSafeAugmentationConfig:
    values = {
        "enabled": True,
        "probability": 1.0,
        "max_rotation_degrees": 1.0,
        "max_translation_voxels": 0.5,
        "scale_min": 1.0,
        "scale_max": 1.0,
        "psf_smoothing_enabled": False,
        "count_noise_enabled": False,
        "qc_contract_sha256": "a" * 64,
    }
    values.update(overrides)
    return PetSafeAugmentationConfig(**values)


def _phantom() -> tuple[torch.Tensor, torch.Tensor]:
    mask = torch.zeros((1, 16, 16, 16), dtype=torch.float32)
    mask[:, 3:13, 4:12, 2:14] = 1.0
    grid = torch.linspace(0.0, 1.0, 16, dtype=torch.float32)
    image = (
        grid[:, None, None]
        + 2.0 * grid[None, :, None]
        + 3.0 * grid[None, None, :]
    ).unsqueeze(0)
    image = image * mask
    return image, mask


def test_augmentation_seed_is_subject_fold_epoch_deterministic() -> None:
    kwargs = dict(
        base_seed=20260714,
        branch="tau",
        subject_key="pt_hash",
        outer_fold=2,
        inner_fold=1,
        epoch=7,
    )
    assert deterministic_augmentation_seed(**kwargs) == deterministic_augmentation_seed(**kwargs)
    assert deterministic_augmentation_seed(**kwargs) != deterministic_augmentation_seed(
        **{**kwargs, "epoch": 8}
    )


def test_pet_safe_augmentation_is_repeatable_and_changes_by_epoch() -> None:
    image, mask = _phantom()
    config = _safe_config()
    common = dict(
        image=image,
        mask=mask,
        config=config,
        split_role="inner_train",
        branch="amyloid",
        subject_key="pt_hash",
        outer_fold=1,
        inner_fold=0,
        base_seed=20260714,
    )
    first = apply_pet_safe_augmentation(**common, epoch=3)
    repeated = apply_pet_safe_augmentation(**common, epoch=3)
    next_epoch = apply_pet_safe_augmentation(**common, epoch=4)

    assert torch.equal(first.image, repeated.image)
    assert torch.equal(first.mask, repeated.mask)
    assert first.parameters == repeated.parameters
    assert not torch.equal(first.image, next_epoch.image)
    assert torch.isfinite(first.image).all()
    assert torch.all(first.image[first.mask == 0] == 0)


def test_vectorized_batch_augmentation_matches_single_subject_calls() -> None:
    image, mask = _phantom()
    config = _safe_config()
    subjects = ["pt_a", "pt_b"]
    batch = apply_pet_safe_augmentation_batch(
        images=torch.stack((image, image * 2.0)),
        masks=torch.stack((mask, mask)),
        config=config,
        split_role="outer_train",
        branch="tau",
        subject_keys=subjects,
        outer_fold=2,
        inner_fold=None,
        epoch=5,
        base_seed=20260714,
    )
    singles = [
        apply_pet_safe_augmentation(
            image=image * float(index + 1),
            mask=mask,
            config=config,
            split_role="outer_train",
            branch="tau",
            subject_key=subject,
            outer_fold=2,
            inner_fold=None,
            epoch=5,
            base_seed=20260714,
        )
        for index, subject in enumerate(subjects)
    ]
    assert torch.allclose(batch.images, torch.stack([row.image for row in singles]), atol=1e-6)
    assert torch.equal(batch.masks, torch.stack([row.mask for row in singles]))
    assert batch.parameters == [row.parameters for row in singles]


@pytest.mark.parametrize("split_role", ["inner_validation", "outer_test", "external_test"])
def test_validation_and_test_images_are_bit_identical(split_role: str) -> None:
    image, mask = _phantom()
    result = apply_pet_safe_augmentation(
        image=image,
        mask=mask,
        config=_safe_config(),
        split_role=split_role,
        branch="tau",
        subject_key="pt_hash",
        outer_fold=0,
        inner_fold=0,
        epoch=99,
        base_seed=20260714,
    )
    assert result.applied is False
    assert result.parameters["reason"] == "NON_TRAIN_SPLIT"
    assert torch.equal(result.image, image)
    assert torch.equal(result.mask, mask)


@pytest.mark.parametrize(
    ("overrides", "reason"),
    [
        ({"max_rotation_degrees": 2.01}, "ROTATION_EXCEEDS_LOCKED_LIMIT"),
        ({"max_translation_voxels": 1.01}, "TRANSLATION_EXCEEDS_LOCKED_LIMIT"),
        ({"scale_min": 0.98}, "SCALE_EXCEEDS_LOCKED_LIMIT"),
        ({"scale_max": 1.02}, "SCALE_EXCEEDS_LOCKED_LIMIT"),
        ({"qc_contract_sha256": ""}, "QC_CONTRACT_SHA256_REQUIRED"),
    ],
)
def test_geometric_augmentation_contract_fails_closed(overrides: dict, reason: str) -> None:
    with pytest.raises(PetAugmentationV4Error, match=reason):
        _safe_config(**overrides)


def test_psf_and_count_noise_require_calibrated_physics_contracts() -> None:
    with pytest.raises(PetAugmentationV4Error, match="PSF_REGISTRY_REQUIRED"):
        _safe_config(psf_smoothing_enabled=True)
    with pytest.raises(PetAugmentationV4Error, match="CALIBRATED_COUNT_NOISE_MODEL_REQUIRED"):
        _safe_config(count_noise_enabled=True)


def test_factorial_arms_are_locked_and_matched() -> None:
    assert tuple(FACTORIAL_ARMS) == ("A0", "A1", "A2", "A3", "A4")
    assert matched_arm_differences("A0", "A1") == ("augmentation_profile",)
    assert matched_arm_differences("A2", "A3") == ("augmentation_profile",)
    assert matched_arm_differences("A0", "A2") == ("encoder_family",)
    assert matched_arm_differences("A1", "A3") == ("encoder_family",)
    assert FACTORIAL_ARMS["A4"].analysis_role == "SECONDARY_ARCHITECTURE_CHALLENGER"
    assert FACTORIAL_ARMS["A4"].allowed_width_multipliers == (0.25, 0.5)


@pytest.mark.parametrize("arm_id", ["A0", "A1", "A2", "A3", "A4"])
def test_all_image_arms_use_groupnorm_and_common_survival_interface(arm_id: str) -> None:
    model = build_factorial_image_model(
        arm_id,
        {
            "base_channels": 8,
            "embedding_dim": 32,
            "tabular_hidden_dim": 16,
            "dropout": 0.1,
            "resnet_width_multiplier": 0.25,
        },
        tabular_dim=11,
        hazard_bins=5,
    )
    assert not any(isinstance(module, torch.nn.modules.batchnorm._BatchNorm) for module in model.modules())
    assert any(isinstance(module, torch.nn.GroupNorm) for module in model.modules())
    with torch.no_grad():
        output = model(torch.zeros((2, 1, 32, 32, 32)), torch.zeros((2, 11)))
    assert tuple(output.shape) == (2, 5)


def test_factorial_arm_records_are_immutable() -> None:
    with pytest.raises(dataclasses.FrozenInstanceError):
        FACTORIAL_ARMS["A0"].encoder_family = "changed"  # type: ignore[misc]


def test_qc_selects_largest_passing_envelope_without_outcomes() -> None:
    candidates = [
        {"candidate_id": "identity", "rotation": 0.0, "translation": 0.0, "status": "PASS"},
        {"candidate_id": "primary", "rotation": 1.0, "translation": 0.5, "status": "PASS"},
        {"candidate_id": "upper", "rotation": 2.0, "translation": 1.0, "status": "FAIL"},
    ]
    assert select_largest_passing_envelope(candidates)["candidate_id"] == "primary"


def test_factorial_family_selection_is_inner_only_one_standard_error() -> None:
    evidence = {
        "A0": [0.151, 0.149, 0.150, 0.152, 0.148],
        "A1": [0.149, 0.148, 0.150, 0.151, 0.147],
        "A2": [0.147, 0.146, 0.148, 0.145, 0.149],
        "A3": [0.145, 0.143, 0.147, 0.144, 0.146],
        "A4": [0.146, 0.145, 0.148, 0.144, 0.147],
    }
    result = select_inner_only_factorial_arm(
        evidence,
        complexity_ranks={"A0": 0, "A1": 1, "A2": 2, "A3": 3, "A4": 4},
    )
    assert result["selection_metric"] == "IPCW_INTEGRATED_BRIER_SCORE"
    assert result["selection_rule"] == "ONE_STANDARD_ERROR_THEN_LOWEST_COMPLEXITY"
    assert result["selected_arm"] in result["eligible_within_one_standard_error"]
    assert result["selected_arm"] == min(
        result["eligible_within_one_standard_error"],
        key={"A0": 0, "A1": 1, "A2": 2, "A3": 3, "A4": 4}.get,
    )


def test_factorial_family_selection_rejects_incomplete_or_nonfinite_evidence() -> None:
    with pytest.raises(ValueError, match="ALL_FIVE_FACTORIAL_ARMS_REQUIRED"):
        select_inner_only_factorial_arm(
            {"A0": [0.1, 0.2]}, complexity_ranks={"A0": 0}
        )
    evidence = {arm: [0.1, 0.2, 0.3] for arm in FACTORIAL_ARMS}
    evidence["A3"][1] = float("nan")
    with pytest.raises(ValueError, match="INNER_ARM_EVIDENCE_INVALID"):
        select_inner_only_factorial_arm(
            evidence, complexity_ranks={arm: index for index, arm in enumerate(FACTORIAL_ARMS)}
        )
