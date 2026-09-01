from __future__ import annotations

import hashlib
import os
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from src.method_y_revival.image_factorial_v4 import build_factorial_image_model
from src.method_y_revival.image_training_runtime_v3 import PlannedTrainingInterruption
from src.method_y_revival.image_training_runtime_v4 import (
    FactorialH5SurvivalDataset,
    _ipcw_validation_ibs,
    fit_survival_model_v4,
)
from src.method_y_revival.pet_augmentation_v4 import PetSafeAugmentationConfig


def _write_caches(root: Path, subjects: list[str]) -> tuple[Path, Path]:
    root.mkdir(parents=True, exist_ok=True)
    image_path = root / "images.h5"
    mask_path = root / "masks.h5"
    with h5py.File(image_path, "w") as images, h5py.File(mask_path, "w") as masks:
        for index, subject in enumerate(subjects):
            image = np.zeros((1, 16, 16, 16), dtype=np.float16)
            image[:, 2:14, 2:14, 2:14] = np.float16(index + 1) / 8.0
            mask = np.zeros_like(image, dtype=np.uint8)
            mask[:, 2:14, 2:14, 2:14] = 1
            images.create_group(subject).create_dataset("image", data=image)
            masks.create_group(subject).create_dataset("mask", data=mask)
    return image_path, mask_path


def _augmentation() -> PetSafeAugmentationConfig:
    return PetSafeAugmentationConfig(
        enabled=True,
        probability=1.0,
        max_rotation_degrees=1.0,
        max_translation_voxels=0.5,
        scale_min=1.0,
        scale_max=1.0,
        qc_contract_sha256="b" * 64,
    )


def _dataset(
    root: Path,
    *,
    split_role: str,
    arm_id: str = "A1",
    defer_batch_augmentation: bool = False,
) -> FactorialH5SurvivalDataset:
    subjects = [f"pt_{index}" for index in range(6)]
    image_path, mask_path = _write_caches(root, subjects)
    return FactorialH5SurvivalDataset(
        cache_path=image_path,
        mask_cache_path=mask_path,
        patient_hashes=subjects,
        tabular=np.arange(18, dtype=np.float32).reshape(6, 3) / 10.0,
        event=np.asarray([0, 1, 0, 1, 0, 1], dtype=np.float32),
        time_days=np.asarray([500, 600, 700, 800, 900, 1000], dtype=np.float32),
        arm_id=arm_id,
        augmentation_config=_augmentation(),
        split_role=split_role,
        branch="tau",
        outer_fold=1,
        inner_fold=0,
        base_seed=20260714,
        ram_cache_max_bytes=64 * 1024**2,
        defer_batch_augmentation=defer_batch_augmentation,
    )


def test_dataset_applies_augmentation_only_to_training_splits(tmp_path: Path) -> None:
    train = _dataset(tmp_path / "train", split_role="inner_train")
    validation = _dataset(tmp_path / "validation", split_role="inner_validation")
    train.set_epoch(3)
    validation.set_epoch(3)
    raw = train.raw_image(0)
    assert not torch.equal(train[0]["image"], raw)
    assert torch.equal(validation[0]["image"], validation.raw_image(0))


def test_dataset_epoch_seed_is_exactly_reproducible(tmp_path: Path) -> None:
    first = _dataset(tmp_path / "first", split_role="outer_train")
    second = _dataset(tmp_path / "second", split_role="outer_train")
    first.set_epoch(4)
    second.set_epoch(4)
    assert torch.equal(first[2]["image"], second[2]["image"])
    first.set_epoch(5)
    assert not torch.equal(first[2]["image"], second[2]["image"])


def test_augmented_arm_requires_mask_cache(tmp_path: Path) -> None:
    subjects = ["pt_0"]
    image_path, _ = _write_caches(tmp_path, subjects)
    with pytest.raises(RuntimeError, match="MASK_CACHE_REQUIRED_FOR_AUGMENTED_ARM"):
        FactorialH5SurvivalDataset(
            cache_path=image_path,
            mask_cache_path=None,
            patient_hashes=subjects,
            tabular=np.zeros((1, 1), dtype=np.float32),
            event=np.zeros(1, dtype=np.float32),
            time_days=np.ones(1, dtype=np.float32),
            arm_id="A1",
            augmentation_config=_augmentation(),
            split_role="inner_train",
            branch="amyloid",
            outer_fold=0,
            inner_fold=0,
            base_seed=1,
        )


def _train(
    dataset: FactorialH5SurvivalDataset,
    output: Path,
    *,
    interrupt_after_epoch: int | None,
    device: str = "cpu",
    amp: bool = False,
) -> dict:
    torch.manual_seed(77)
    model = build_factorial_image_model(
        "A1",
        {
            "base_channels": 2,
            "embedding_dim": 8,
            "tabular_hidden_dim": 4,
            "dropout": 0.0,
        },
        tabular_dim=3,
        hazard_bins=2,
    )
    return fit_survival_model_v4(
        model=model,
        train_dataset=dataset,
        validation_dataset=None,
        bin_edges=[0.0, 750.0, 1200.0],
        output_dir=output,
        run_contract_digest=hashlib.sha256(b"runtime-v4-test").hexdigest(),
        seed=77,
        max_epochs=2,
        patience=10,
        learning_rate=1e-3,
        weight_decay=0.0,
        batch_size=2,
        gradient_accumulation_steps=1,
        device=device,
        amp=amp,
        fixed_epochs=2,
        interrupt_after_epoch=interrupt_after_epoch,
    )


def test_interruption_resume_matches_uninterrupted_with_epoch_augmentation(tmp_path: Path) -> None:
    uninterrupted_ds = _dataset(
        tmp_path / "uninterrupted_data",
        split_role="outer_train",
        defer_batch_augmentation=True,
    )
    interrupted_ds = _dataset(
        tmp_path / "interrupted_data",
        split_role="outer_train",
        defer_batch_augmentation=True,
    )
    uninterrupted = _train(uninterrupted_ds, tmp_path / "uninterrupted", interrupt_after_epoch=None)
    with pytest.raises(PlannedTrainingInterruption):
        _train(interrupted_ds, tmp_path / "resumed", interrupt_after_epoch=1)
    resumed = _train(interrupted_ds, tmp_path / "resumed", interrupt_after_epoch=None)

    assert uninterrupted["history"] == resumed["history"]
    left = torch.load(
        tmp_path / "uninterrupted" / "locked_checkpoint.pt",
        map_location="cpu",
        weights_only=False,
    )["model_state"]
    right = torch.load(
        tmp_path / "resumed" / "locked_checkpoint.pt",
        map_location="cpu",
        weights_only=False,
    )["model_state"]
    assert left.keys() == right.keys()
    assert all(torch.equal(left[key], right[key]) for key in left)


def test_ipcw_validation_metric_is_finite_bounded_and_fail_closed() -> None:
    logits = torch.zeros((6, 3), dtype=torch.float32)
    metric = _ipcw_validation_ibs(
        validation_logits=logits,
        train_event=np.asarray([0, 1, 0, 1, 0, 1]),
        train_time_days=np.asarray([500, 600, 700, 800, 900, 1000]),
        validation_event=np.asarray([0, 1, 0, 1, 0, 1]),
        validation_time_days=np.asarray([510, 610, 710, 810, 910, 990]),
        bin_edges=[0.0, 550.0, 750.0, 1100.0],
    )
    assert metric["metric_name"] == "IPCW_INTEGRATED_BRIER_SCORE"
    assert 0.0 <= metric["metric_value"] <= 1.0
    assert len(metric["evaluation_grid_days"]) >= 2

    with pytest.raises(RuntimeError, match="INNER_IBS_NOT_ESTIMABLE"):
        _ipcw_validation_ibs(
            validation_logits=torch.zeros((2, 2)),
            train_event=np.asarray([0, 1]),
            train_time_days=np.asarray([1.0, 2.0]),
            validation_event=np.asarray([0, 1]),
            validation_time_days=np.asarray([3.0, 4.0]),
            bin_edges=[0.0, 2.0, 5.0],
        )


def test_inner_checkpoint_selection_uses_ipcw_ibs_not_nll(tmp_path: Path) -> None:
    train = _dataset(
        tmp_path / "metric_train",
        split_role="inner_train",
        arm_id="A0",
        defer_batch_augmentation=True,
    )
    validation = _dataset(
        tmp_path / "metric_validation",
        split_role="inner_validation",
        arm_id="A0",
        defer_batch_augmentation=True,
    )
    torch.manual_seed(91)
    model = build_factorial_image_model(
        "A0",
        {
            "base_channels": 2,
            "embedding_dim": 8,
            "tabular_hidden_dim": 4,
            "dropout": 0.0,
        },
        tabular_dim=3,
        hazard_bins=3,
    )
    summary = fit_survival_model_v4(
        model=model,
        train_dataset=train,
        validation_dataset=validation,
        bin_edges=[0.0, 550.0, 750.0, 1100.0],
        output_dir=tmp_path / "metric_output",
        run_contract_digest=hashlib.sha256(b"ipcw-selection").hexdigest(),
        seed=91,
        max_epochs=1,
        patience=1,
        learning_rate=1e-3,
        weight_decay=0.0,
        batch_size=2,
        gradient_accumulation_steps=1,
        device="cpu",
        amp=False,
    )
    row = summary["history"][0]
    assert row["selection_metric_name"] == "IPCW_INTEGRATED_BRIER_SCORE"
    assert row["validation_ipcw_ibs"] == summary["best_validation"]
    assert "train_validation_nll_gap" in row


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA_NOT_AVAILABLE")
def test_cuda_interruption_resume_is_bitwise_exact_with_pet_augmentation(
    tmp_path: Path,
) -> None:
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    prior_benchmark = torch.backends.cudnn.benchmark
    prior_deterministic = torch.backends.cudnn.deterministic
    prior_tf32_matmul = torch.backends.cuda.matmul.allow_tf32
    prior_tf32_cudnn = torch.backends.cudnn.allow_tf32
    prior_algorithms = torch.are_deterministic_algorithms_enabled()
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    try:
        uninterrupted_ds = _dataset(
            tmp_path / "cuda_uninterrupted_data",
            split_role="outer_train",
            defer_batch_augmentation=True,
        )
        interrupted_ds = _dataset(
            tmp_path / "cuda_interrupted_data",
            split_role="outer_train",
            defer_batch_augmentation=True,
        )
        uninterrupted = _train(
            uninterrupted_ds,
            tmp_path / "cuda_uninterrupted",
            interrupt_after_epoch=None,
            device="cuda",
            amp=True,
        )
        with pytest.raises(PlannedTrainingInterruption):
            _train(
                interrupted_ds,
                tmp_path / "cuda_resumed",
                interrupt_after_epoch=1,
                device="cuda",
                amp=True,
            )
        resumed = _train(
            interrupted_ds,
            tmp_path / "cuda_resumed",
            interrupt_after_epoch=None,
            device="cuda",
            amp=True,
        )
        assert uninterrupted["history"] == resumed["history"]
        left = torch.load(
            tmp_path / "cuda_uninterrupted" / "locked_checkpoint.pt",
            map_location="cpu",
            weights_only=False,
        )["model_state"]
        right = torch.load(
            tmp_path / "cuda_resumed" / "locked_checkpoint.pt",
            map_location="cpu",
            weights_only=False,
        )["model_state"]
        assert left.keys() == right.keys()
        assert all(torch.equal(left[key], right[key]) for key in left)
    finally:
        torch.use_deterministic_algorithms(prior_algorithms)
        torch.backends.cudnn.benchmark = prior_benchmark
        torch.backends.cudnn.deterministic = prior_deterministic
        torch.backends.cuda.matmul.allow_tf32 = prior_tf32_matmul
        torch.backends.cudnn.allow_tf32 = prior_tf32_cudnn


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA_NOT_AVAILABLE")
def test_cuda_rerun_of_terminal_fold_with_existing_lock_is_device_safe(
    tmp_path: Path,
) -> None:
    """A completed fold must be idempotent when its lock was written on CPU.

    Production checkpoints are intentionally portable CPU state, while the
    resumed trainer runs on CUDA.  Re-entering a terminal fold therefore must
    compare state dictionaries after normalising devices, without retraining or
    rewriting the locked artifact.
    """
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    prior_benchmark = torch.backends.cudnn.benchmark
    prior_deterministic = torch.backends.cudnn.deterministic
    prior_tf32_matmul = torch.backends.cuda.matmul.allow_tf32
    prior_tf32_cudnn = torch.backends.cudnn.allow_tf32
    prior_algorithms = torch.are_deterministic_algorithms_enabled()
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    try:
        dataset = _dataset(
            tmp_path / "cuda_terminal_data",
            split_role="outer_train",
            defer_batch_augmentation=True,
        )
        output = tmp_path / "cuda_terminal_output"
        first = _train(
            dataset,
            output,
            interrupt_after_epoch=None,
            device="cuda",
            amp=True,
        )
        assert first["status"] == "CHECKPOINT_LOCKED"
        locked_path = output / "locked_checkpoint.pt"
        before = locked_path.read_bytes()

        rerun = _train(
            dataset,
            output,
            interrupt_after_epoch=None,
            device="cuda",
            amp=True,
        )

        assert rerun["status"] == "CHECKPOINT_LOCKED"
        assert rerun["resumed_from_epoch"] == first["epoch_completed"]
        assert locked_path.read_bytes() == before
    finally:
        torch.use_deterministic_algorithms(prior_algorithms)
        torch.backends.cudnn.benchmark = prior_benchmark
        torch.backends.cudnn.deterministic = prior_deterministic
        torch.backends.cuda.matmul.allow_tf32 = prior_tf32_matmul
        torch.backends.cudnn.allow_tf32 = prior_tf32_cudnn
