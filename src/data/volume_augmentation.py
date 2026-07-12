from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence


_DEFAULT_BASE_PROB = 0.2


def _scaled_probability(base: float, prob: float) -> float:
    scale = max(float(prob), 0.0) / _DEFAULT_BASE_PROB
    return min(max(float(base) * scale, 0.0), 1.0)


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
            RandAffine,
            RandBiasField,
        )
    except Exception as exc:  # pragma: no cover - environment guard
        raise RuntimeError("MONAI is required for volume augmentation; install monai or disable augmentation.") from exc
    # These volumes are affine-free, signed z-score HDF5 tensors.  We cannot
    # prove which tensor axis is left-right here, so flips are intentionally
    # excluded.  Global noise/offset/contrast transforms are also excluded:
    # they can turn the zero background into nonzero voxels.  Materialized
    # augmentation uses the mask-aware implementation in the cache builder.
    medium_prob = _scaled_probability(0.25, prob)
    light_prob = _scaled_probability(0.15, prob)
    return Compose(
        [
            RandAffine(
                prob=medium_prob,
                rotate_range=(math.radians(3.0),) * 3,
                translate_range=(2, 2, 2),
                scale_range=(0.025, 0.025, 0.025),
                padding_mode="zeros",
            ),
            RandBiasField(prob=light_prob, coeff_range=(0.0, 0.05)),
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


def maybe_wrap_train_dataset(base_dataset, *, enabled: bool, image_positions: Iterable[int], prob: float = 0.2):
    transform = build_monai_volume_transform(split="train", enabled=enabled, prob=prob)
    if transform is None:
        return base_dataset
    return TransformingDataset(base_dataset, transform, tuple(image_positions))
