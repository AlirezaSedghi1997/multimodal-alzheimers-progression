from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any, Mapping


class PetAugmentationV4Error(ValueError):
    """Raised when PET augmentation would violate the locked scientific contract."""


_TRAIN_SPLITS = frozenset({"inner_train", "outer_train"})
_ALL_SPLITS = _TRAIN_SPLITS | frozenset(
    {"inner_validation", "outer_test", "external_test"}
)


def _is_sha256(value: str | None) -> bool:
    if value is None or len(str(value)) != 64:
        return False
    try:
        int(str(value), 16)
    except ValueError:
        return False
    return True


@dataclass(frozen=True)
class PetSafeAugmentationConfig:
    """Locked, spatial-only PET augmentation policy.

    The primary v4 policy deliberately excludes MRI-style intensity transforms.
    Optional PSF and count-noise perturbations are only reachable when an
    independently hashed physics contract is supplied.
    """

    enabled: bool
    probability: float
    max_rotation_degrees: float
    max_translation_voxels: float
    scale_min: float
    scale_max: float
    psf_smoothing_enabled: bool = False
    count_noise_enabled: bool = False
    qc_contract_sha256: str | None = None
    psf_registry_sha256: str | None = None
    psf_fwhm_mm_by_scanner: Mapping[str, float] | None = None
    calibrated_count_noise_model_sha256: str | None = None
    count_noise_relative_std: float | None = None

    def __post_init__(self) -> None:
        if not 0.0 <= float(self.probability) <= 1.0:
            raise PetAugmentationV4Error("AUGMENTATION_PROBABILITY_INVALID")
        if not 0.0 <= float(self.max_rotation_degrees) <= 2.0:
            raise PetAugmentationV4Error("ROTATION_EXCEEDS_LOCKED_LIMIT")
        if not 0.0 <= float(self.max_translation_voxels) <= 1.0:
            raise PetAugmentationV4Error("TRANSLATION_EXCEEDS_LOCKED_LIMIT")
        if not (
            0.99 <= float(self.scale_min) <= float(self.scale_max) <= 1.01
        ):
            raise PetAugmentationV4Error("SCALE_EXCEEDS_LOCKED_LIMIT")
        if self.enabled and not _is_sha256(self.qc_contract_sha256):
            raise PetAugmentationV4Error("QC_CONTRACT_SHA256_REQUIRED")
        if self.psf_smoothing_enabled:
            if not _is_sha256(self.psf_registry_sha256) or not self.psf_fwhm_mm_by_scanner:
                raise PetAugmentationV4Error("PSF_REGISTRY_REQUIRED")
            if any(float(value) <= 0 for value in self.psf_fwhm_mm_by_scanner.values()):
                raise PetAugmentationV4Error("PSF_FWHM_MUST_BE_POSITIVE")
        if self.count_noise_enabled:
            if not _is_sha256(self.calibrated_count_noise_model_sha256):
                raise PetAugmentationV4Error("CALIBRATED_COUNT_NOISE_MODEL_REQUIRED")
            if self.count_noise_relative_std is None or not (
                0.0 < float(self.count_noise_relative_std) <= 0.25
            ):
                raise PetAugmentationV4Error("CALIBRATED_COUNT_NOISE_SCALE_REQUIRED")

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "PetSafeAugmentationConfig":
        return cls(**dict(payload))

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": bool(self.enabled),
            "probability": float(self.probability),
            "max_rotation_degrees": float(self.max_rotation_degrees),
            "max_translation_voxels": float(self.max_translation_voxels),
            "scale_min": float(self.scale_min),
            "scale_max": float(self.scale_max),
            "psf_smoothing_enabled": bool(self.psf_smoothing_enabled),
            "count_noise_enabled": bool(self.count_noise_enabled),
            "qc_contract_sha256": self.qc_contract_sha256,
            "psf_registry_sha256": self.psf_registry_sha256,
            "psf_fwhm_mm_by_scanner": (
                dict(self.psf_fwhm_mm_by_scanner)
                if self.psf_fwhm_mm_by_scanner is not None
                else None
            ),
            "calibrated_count_noise_model_sha256": (
                self.calibrated_count_noise_model_sha256
            ),
            "count_noise_relative_std": self.count_noise_relative_std,
        }


@dataclass(frozen=True)
class PetAugmentationResult:
    image: Any
    mask: Any
    applied: bool
    parameters: dict[str, Any]


@dataclass(frozen=True)
class PetBatchAugmentationResult:
    images: Any
    masks: Any
    applied: tuple[bool, ...]
    parameters: list[dict[str, Any]]


def deterministic_augmentation_seed(
    *,
    base_seed: int,
    branch: str,
    subject_key: str,
    outer_fold: int,
    inner_fold: int | None,
    epoch: int,
) -> int:
    material = "|".join(
        (
            "METHOD_Y_PET_AUGMENTATION_V4",
            str(int(base_seed)),
            str(branch),
            str(subject_key),
            str(int(outer_fold)),
            "FINAL" if inner_fold is None else str(int(inner_fold)),
            str(int(epoch)),
        )
    )
    digest = hashlib.sha256(material.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big", signed=False) % (2**63 - 1)


def _as_4d_float_tensor(value: Any, *, name: str):
    import torch

    tensor = torch.as_tensor(value, dtype=torch.float32)
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 4:
        raise PetAugmentationV4Error(f"{name.upper()}_MUST_BE_C_D_H_W")
    if not torch.isfinite(tensor).all():
        raise PetAugmentationV4Error(f"{name.upper()}_NONFINITE")
    return tensor


def _rotation_matrix_xyz(rx: float, ry: float, rz: float):
    import torch

    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)
    rot_x = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]],
        dtype=torch.float32,
    )
    rot_y = torch.tensor(
        [[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]],
        dtype=torch.float32,
    )
    rot_z = torch.tensor(
        [[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]],
        dtype=torch.float32,
    )
    return rot_z @ rot_y @ rot_x


def _sample_parameters(
    *,
    config: PetSafeAugmentationConfig,
    branch: str,
    subject_key: str,
    outer_fold: int,
    inner_fold: int | None,
    epoch: int,
    base_seed: int,
) -> dict[str, Any]:
    import numpy as np

    seed = deterministic_augmentation_seed(
        base_seed=base_seed,
        branch=branch,
        subject_key=subject_key,
        outer_fold=outer_fold,
        inner_fold=inner_fold,
        epoch=epoch,
    )
    rng = np.random.default_rng(seed)
    if float(rng.uniform()) >= float(config.probability):
        return {"applied": False, "reason": "PROBABILITY_DRAW", "seed": int(seed)}
    rotations = rng.uniform(
        -float(config.max_rotation_degrees),
        float(config.max_rotation_degrees),
        size=3,
    )
    translations = rng.uniform(
        -float(config.max_translation_voxels),
        float(config.max_translation_voxels),
        size=3,
    )
    scale = float(rng.uniform(float(config.scale_min), float(config.scale_max)))
    return {
        "applied": True,
        "seed": int(seed),
        "rotation_degrees_xyz": [float(value) for value in rotations],
        "translation_voxels_dhw": [float(value) for value in translations],
        "isotropic_scale": scale,
    }


def _theta_from_parameters(parameters: dict[str, Any], shape_dhw: tuple[int, int, int]):
    import numpy as np
    import torch

    if not parameters.get("applied"):
        return torch.eye(3, 4, dtype=torch.float32)
    depth, height, width = shape_dhw
    rotation = _rotation_matrix_xyz(
        *np.deg2rad(parameters["rotation_degrees_xyz"]).tolist()
    ) * float(parameters["isotropic_scale"])
    translation = parameters["translation_voxels_dhw"]
    normalized = torch.tensor(
        [
            2.0 * float(translation[2]) / max(1, width - 1),
            2.0 * float(translation[1]) / max(1, height - 1),
            2.0 * float(translation[0]) / max(1, depth - 1),
        ],
        dtype=torch.float32,
    )
    return torch.cat((rotation, normalized[:, None]), dim=1)


def _gaussian_blur_3d(image, sigma_voxels: float):
    import torch
    import torch.nn.functional as functional

    sigma = float(sigma_voxels)
    if sigma <= 0:
        return image
    radius = max(1, int(math.ceil(3.0 * sigma)))
    axis = torch.arange(-radius, radius + 1, dtype=image.dtype, device=image.device)
    kernel = torch.exp(-(axis**2) / (2.0 * sigma**2))
    kernel = kernel / kernel.sum()
    result = image.unsqueeze(0)
    channels = int(result.shape[1])
    for dimension in range(3):
        shape = [1, 1, 1]
        shape[dimension] = kernel.numel()
        weight = kernel.reshape(1, 1, *shape).repeat(channels, 1, 1, 1, 1)
        padding = [0, 0, 0]
        padding[dimension] = radius
        result = functional.conv3d(result, weight, padding=tuple(padding), groups=channels)
    return result.squeeze(0)


def apply_pet_safe_augmentation(
    *,
    image: Any,
    mask: Any,
    config: PetSafeAugmentationConfig,
    split_role: str,
    branch: str,
    subject_key: str,
    outer_fold: int,
    inner_fold: int | None,
    epoch: int,
    base_seed: int,
    scanner_model: str | None = None,
    voxel_spacing_mm: tuple[float, float, float] = (1.5, 1.5, 1.5),
) -> PetAugmentationResult:
    """Apply deterministic train-only PET perturbation to image and brain mask."""

    import torch
    import torch.nn.functional as functional

    if split_role not in _ALL_SPLITS:
        raise PetAugmentationV4Error(f"UNKNOWN_SPLIT_ROLE:{split_role}")
    image_tensor = _as_4d_float_tensor(image, name="image")
    mask_tensor = (_as_4d_float_tensor(mask, name="mask") > 0.5).to(torch.float32)
    if image_tensor.shape[-3:] != mask_tensor.shape[-3:]:
        raise PetAugmentationV4Error("IMAGE_MASK_SHAPE_MISMATCH")
    if split_role not in _TRAIN_SPLITS:
        return PetAugmentationResult(
            image=image_tensor.clone(),
            mask=mask_tensor.clone(),
            applied=False,
            parameters={"reason": "NON_TRAIN_SPLIT", "split_role": split_role},
        )
    if not config.enabled or float(config.probability) == 0.0:
        return PetAugmentationResult(
            image=image_tensor.clone(),
            mask=mask_tensor.clone(),
            applied=False,
            parameters={"reason": "AUGMENTATION_DISABLED", "split_role": split_role},
        )

    sampled = _sample_parameters(
        config=config,
        branch=branch,
        subject_key=subject_key,
        outer_fold=outer_fold,
        inner_fold=inner_fold,
        epoch=epoch,
        base_seed=base_seed,
    )
    if not sampled["applied"]:
        return PetAugmentationResult(
            image=image_tensor.clone(),
            mask=mask_tensor.clone(),
            applied=False,
            parameters={"reason": sampled["reason"], "seed": int(sampled["seed"])},
        )
    depth, height, width = (int(value) for value in image_tensor.shape[-3:])
    theta = _theta_from_parameters(sampled, (depth, height, width)).unsqueeze(0)
    grid = functional.affine_grid(
        theta,
        size=(1, int(image_tensor.shape[0]), depth, height, width),
        align_corners=True,
    )
    warped_image = functional.grid_sample(
        image_tensor.unsqueeze(0),
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    ).squeeze(0)
    warped_mask = functional.grid_sample(
        mask_tensor.unsqueeze(0),
        grid,
        mode="nearest",
        padding_mode="zeros",
        align_corners=True,
    ).squeeze(0)
    warped_mask = (warped_mask > 0.5).to(torch.float32)
    warped_image = warped_image * warped_mask

    psf_fwhm = None
    if config.psf_smoothing_enabled:
        if scanner_model is None or scanner_model not in config.psf_fwhm_mm_by_scanner:
            raise PetAugmentationV4Error("SCANNER_PSF_ENTRY_REQUIRED")
        psf_fwhm = float(config.psf_fwhm_mm_by_scanner[scanner_model])
        if len(set(float(v) for v in voxel_spacing_mm)) != 1:
            raise PetAugmentationV4Error("ANISOTROPIC_PSF_SMOOTHING_NOT_SUPPORTED")
        sigma_voxels = psf_fwhm / (2.354820045 * float(voxel_spacing_mm[0]))
        warped_image = _gaussian_blur_3d(warped_image, sigma_voxels) * warped_mask

    if config.count_noise_enabled:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(sampled["seed"] ^ 0x5A17))
        noise = torch.randn(
            warped_image.shape,
            generator=generator,
            dtype=warped_image.dtype,
            device="cpu",
        ).to(warped_image.device)
        signal_scale = torch.quantile(torch.abs(warped_image[warped_mask > 0]), 0.75)
        warped_image = (
            warped_image
            + noise
            * signal_scale
            * float(config.count_noise_relative_std)
            * warped_mask
        )

    if not torch.isfinite(warped_image).all():
        raise PetAugmentationV4Error("AUGMENTED_IMAGE_NONFINITE")
    return PetAugmentationResult(
        image=warped_image,
        mask=warped_mask,
        applied=True,
        parameters={
            "seed": int(sampled["seed"]),
            "rotation_degrees_xyz": list(sampled["rotation_degrees_xyz"]),
            "translation_voxels_dhw": list(sampled["translation_voxels_dhw"]),
            "isotropic_scale": float(sampled["isotropic_scale"]),
            "psf_fwhm_mm": psf_fwhm,
            "count_noise_applied": bool(config.count_noise_enabled),
            "split_role": split_role,
        },
    )


def apply_pet_safe_augmentation_batch(
    *,
    images: Any,
    masks: Any,
    config: PetSafeAugmentationConfig,
    split_role: str,
    branch: str,
    subject_keys: list[str] | tuple[str, ...],
    outer_fold: int,
    inner_fold: int | None,
    epoch: int,
    base_seed: int,
) -> PetBatchAugmentationResult:
    """Vectorized geometric augmentation for one training batch."""

    import torch
    import torch.nn.functional as functional

    image_tensor = torch.as_tensor(images, dtype=torch.float32)
    mask_tensor = (torch.as_tensor(masks, dtype=torch.float32) > 0.5).to(torch.float32)
    if image_tensor.ndim != 5 or mask_tensor.shape != image_tensor.shape:
        raise PetAugmentationV4Error("BATCH_IMAGE_MASK_MUST_BE_B_C_D_H_W")
    if len(subject_keys) != int(image_tensor.shape[0]):
        raise PetAugmentationV4Error("BATCH_SUBJECT_COUNT_MISMATCH")
    if split_role not in _ALL_SPLITS:
        raise PetAugmentationV4Error(f"UNKNOWN_SPLIT_ROLE:{split_role}")
    if split_role not in _TRAIN_SPLITS or not config.enabled:
        reason = "NON_TRAIN_SPLIT" if split_role not in _TRAIN_SPLITS else "AUGMENTATION_DISABLED"
        return PetBatchAugmentationResult(
            images=image_tensor.clone(),
            masks=mask_tensor.clone(),
            applied=tuple(False for _ in subject_keys),
            parameters=[{"reason": reason, "split_role": split_role} for _ in subject_keys],
        )
    if config.psf_smoothing_enabled or config.count_noise_enabled:
        raise PetAugmentationV4Error("BATCH_PHYSICS_AUGMENTATION_NOT_IMPLEMENTED")
    depth, height, width = (int(value) for value in image_tensor.shape[-3:])
    sampled = [
        _sample_parameters(
            config=config,
            branch=branch,
            subject_key=str(subject),
            outer_fold=outer_fold,
            inner_fold=inner_fold,
            epoch=epoch,
            base_seed=base_seed,
        )
        for subject in subject_keys
    ]
    theta = torch.stack(
        [_theta_from_parameters(row, (depth, height, width)) for row in sampled]
    ).to(device=image_tensor.device, dtype=image_tensor.dtype)
    grid = functional.affine_grid(theta, size=image_tensor.shape, align_corners=True)
    warped_images = functional.grid_sample(
        image_tensor,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )
    warped_masks = functional.grid_sample(
        mask_tensor,
        grid,
        mode="nearest",
        padding_mode="zeros",
        align_corners=True,
    )
    warped_masks = (warped_masks > 0.5).to(torch.float32)
    applied = torch.tensor(
        [bool(row["applied"]) for row in sampled],
        dtype=torch.bool,
        device=image_tensor.device,
    ).reshape(-1, 1, 1, 1, 1)
    output_images = torch.where(applied, warped_images * warped_masks, image_tensor)
    output_masks = torch.where(applied, warped_masks, mask_tensor)
    parameters = []
    for row in sampled:
        if not row["applied"]:
            parameters.append({"reason": row["reason"], "seed": int(row["seed"])})
        else:
            parameters.append(
                {
                    "seed": int(row["seed"]),
                    "rotation_degrees_xyz": list(row["rotation_degrees_xyz"]),
                    "translation_voxels_dhw": list(row["translation_voxels_dhw"]),
                    "isotropic_scale": float(row["isotropic_scale"]),
                    "psf_fwhm_mm": None,
                    "count_noise_applied": False,
                    "split_role": split_role,
                }
            )
    if not torch.isfinite(output_images).all():
        raise PetAugmentationV4Error("AUGMENTED_BATCH_NONFINITE")
    return PetBatchAugmentationResult(
        images=output_images,
        masks=output_masks,
        applied=tuple(bool(row["applied"]) for row in sampled),
        parameters=parameters,
    )
