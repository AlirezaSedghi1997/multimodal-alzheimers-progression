import pytest


def _transform_probabilities(transform):
    return [float(getattr(item, "prob")) for item in transform.transforms if hasattr(item, "prob")]


def test_monai_volume_transform_prob_zero_disables_random_ops():
    pytest.importorskip("monai")
    from src.data.volume_augmentation import build_monai_volume_transform

    transform = build_monai_volume_transform(split="train", enabled=True, prob=0.0)

    assert transform is not None
    assert _transform_probabilities(transform)
    assert all(prob == 0.0 for prob in _transform_probabilities(transform))


def test_monai_volume_transform_default_prob_uses_conservative_mask_safe_schedule():
    pytest.importorskip("monai")
    from src.data.volume_augmentation import build_monai_volume_transform

    transform = build_monai_volume_transform(split="train", enabled=True, prob=0.2)

    assert _transform_probabilities(transform) == pytest.approx([0.25, 0.15])
