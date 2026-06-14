from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence


def build_monai_volume_transform(*, split: str, enabled: bool, prob: float = 0.2):
    """Return a MONAI train-only 3D augmentation transform.

    Validation, test, and holdout data intentionally return ``None`` so
    reported evaluation metrics stay on real, unaugmented images.
    """
    split_name = split.lower().strip()
    if not enabled or split_name not in {"train", "fit", "training"}:
        return None
    try:
        from monai.transforms import (
            Compose,
            RandAdjustContrast,
            RandAffine,
            RandBiasField,
            RandFlip,
            RandGaussianNoise,
            RandGaussianSmooth,
        )
    except Exception as exc:  # pragma: no cover - environment guard
        raise RuntimeError("MONAI is required for volume augmentation; install monai or disable augmentation.") from exc
    return Compose(
        [
            RandFlip(prob=0.5, spatial_axis=0),
            RandFlip(prob=0.5, spatial_axis=1),
            RandAffine(
                prob=0.3,
                rotate_range=(0.1, 0.1, 0.1),
                translate_range=(5, 5, 5),
                scale_range=(0.05, 0.05, 0.05),
                padding_mode="border",
            ),
            RandAdjustContrast(prob=0.3, gamma=(0.8, 1.2)),
            RandGaussianSmooth(prob=0.2, sigma_x=(0.5, 1.0)),
            RandGaussianNoise(prob=0.3, mean=0.0, std=0.02),
            RandBiasField(prob=0.2),
        ]
    )


def build_augmentation_policy(*, enabled: bool, real_train_n: int, exposures_per_real: int = 4) -> dict:
    exposures = int(exposures_per_real if enabled else 1)
    real_n = int(real_train_n)
    return {
        "library": "MONAI",
        "enabled": bool(enabled),
        "train_transform_enabled": bool(enabled),
        "validation_transform_enabled": False,
        "test_transform_enabled": False,
        "holdout_transform_enabled": False,
        "real_train_n": real_n,
        "exposures_per_real": exposures,
        "augmented_training_exposures": real_n * exposures,
        "real_n_unchanged": True,
        "synthetic_or_augmented_counted_as_real_subjects": False,
        "selection_scope": "train_fold_only",
    }


@dataclass
class TransformingDataset:
    """Thin dataset wrapper that applies transforms only to image fields."""

    base_dataset: object
    transform: object | None
    image_positions: Sequence[int] = (0,)

    def __len__(self) -> int:
        return len(self.base_dataset)  # type: ignore[arg-type]

    def __getitem__(self, item: int):
        values = list(self.base_dataset[item])  # type: ignore[index]
        if self.transform is None:
            return tuple(values)
        for pos in self.image_positions:
            values[pos] = self.transform(values[pos])
        return tuple(values)


def maybe_wrap_train_dataset(base_dataset, *, enabled: bool, image_positions: Iterable[int]):
    transform = build_monai_volume_transform(split="train", enabled=enabled)
    if transform is None:
        return base_dataset
    return TransformingDataset(base_dataset, transform, tuple(image_positions))
