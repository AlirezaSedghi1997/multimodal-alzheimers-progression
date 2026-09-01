from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Iterable

import h5py
import numpy as np

from .atomic_io import atomic_write_json
from .image_factorial_v4 import FACTORIAL_ARMS
from .image_modeling_v3 import survival_at_times
from .image_survival_v3 import discrete_time_survival_nll
from .image_training_runtime_v3 import (
    H5PatternSurvivalDataset,
    PlannedTrainingInterruption,
    TrainingRuntimeV3Error,
    _atomic_torch_save,
    _cpu_state,
    _forward,
    _loader,
    _restore_rng,
    _rng_state,
    seed_everything,
)
from .pet_augmentation_v4 import (
    PetSafeAugmentationConfig,
    apply_pet_safe_augmentation,
    apply_pet_safe_augmentation_batch,
)


class TrainingRuntimeV4Error(TrainingRuntimeV3Error):
    """Raised when v4 factorial training cannot preserve its contract."""


class FactorialH5SurvivalDataset(H5PatternSurvivalDataset):
    """Epoch-aware H5 dataset with deterministic train-only PET augmentation."""

    def __init__(
        self,
        *,
        cache_path: str | Path,
        mask_cache_path: str | Path | None,
        patient_hashes: Iterable[str],
        tabular: np.ndarray,
        event: np.ndarray,
        time_days: np.ndarray,
        arm_id: str,
        augmentation_config: PetSafeAugmentationConfig,
        split_role: str,
        branch: str,
        outer_fold: int,
        inner_fold: int | None,
        base_seed: int,
        sample_weight: np.ndarray | None = None,
        ram_cache_max_bytes: int | None = None,
        defer_batch_augmentation: bool = False,
    ) -> None:
        if arm_id not in FACTORIAL_ARMS:
            raise TrainingRuntimeV4Error(f"UNKNOWN_FACTORIAL_ARM:{arm_id}")
        super().__init__(
            cache_path=cache_path,
            patient_hashes=patient_hashes,
            tabular=tabular,
            event=event,
            time_days=time_days,
            include_image=True,
            sample_weight=sample_weight,
            ram_cache_max_bytes=ram_cache_max_bytes,
        )
        self.arm_id = str(arm_id)
        self.augmentation_config = augmentation_config
        self.split_role = str(split_role)
        self.branch = str(branch)
        self.outer_fold = int(outer_fold)
        self.inner_fold = int(inner_fold) if inner_fold is not None else None
        self.base_seed = int(base_seed)
        self.epoch = 0
        self.defer_batch_augmentation = bool(defer_batch_augmentation)
        self._mask_handle: h5py.File | None = None
        self._mask_ram_cache: dict[str, np.ndarray] = {}
        self._common_mask: np.ndarray | None = None
        profile = FACTORIAL_ARMS[self.arm_id].augmentation_profile
        self.augmentation_enabled = profile != "NONE"
        self.mask_cache_path = Path(mask_cache_path) if mask_cache_path is not None else None
        if self.augmentation_enabled and (
            self.mask_cache_path is None or not self.mask_cache_path.is_file()
        ):
            raise TrainingRuntimeV4Error("MASK_CACHE_REQUIRED_FOR_AUGMENTED_ARM")

    def set_epoch(self, epoch: int) -> None:
        if int(epoch) < 0:
            raise TrainingRuntimeV4Error("DATASET_EPOCH_MUST_BE_NONNEGATIVE")
        self.epoch = int(epoch)

    def _mask_h5(self) -> h5py.File:
        if self.mask_cache_path is None:
            raise TrainingRuntimeV4Error("MASK_CACHE_REQUIRED_FOR_AUGMENTED_ARM")
        if self._mask_handle is None:
            self._mask_handle = h5py.File(self.mask_cache_path, "r")
        return self._mask_handle

    def _uses_common_nifti_mask(self) -> bool:
        if self.mask_cache_path is None:
            return False
        name = self.mask_cache_path.name.lower()
        return name.endswith(".nii") or name.endswith(".nii.gz")

    def _read_mask(self, patient: str) -> np.ndarray:
        if self._uses_common_nifti_mask():
            if self._common_mask is None:
                import nibabel as nib

                data = np.asarray(nib.load(self.mask_cache_path).dataobj)
                if data.ndim != 3:
                    raise TrainingRuntimeV4Error("COMMON_MASK_MUST_BE_3D")
                self._common_mask = (data > 0).astype(np.uint8)[None]
            mask = self._common_mask
            if mask.shape != self._read_image(patient).shape:
                raise TrainingRuntimeV4Error(f"MASK_IMAGE_SHAPE_MISMATCH:{patient}")
            return mask
        cached = self._mask_ram_cache.get(patient)
        if cached is not None:
            return cached
        handle = self._mask_h5()
        if patient not in handle or "mask" not in handle[patient]:
            raise TrainingRuntimeV4Error(f"MASK_CACHE_SUBJECT_MISSING:{patient}")
        mask = np.asarray(handle[patient]["mask"], dtype=np.uint8)
        if mask.shape != self._read_image(patient).shape:
            raise TrainingRuntimeV4Error(f"MASK_IMAGE_SHAPE_MISMATCH:{patient}")
        if not np.any(mask):
            raise TrainingRuntimeV4Error(f"MASK_EMPTY:{patient}")
        self._mask_ram_cache[patient] = mask
        return mask

    def raw_image(self, index: int):
        import torch

        patient = self.patient_hashes[index]
        return torch.from_numpy(self._read_image(patient).astype(np.float32))

    def __getitem__(self, index: int) -> dict[str, Any]:
        import torch

        patient = self.patient_hashes[index]
        raw_image = self.raw_image(index)
        row: dict[str, Any] = {
            "tabular": torch.from_numpy(self.tabular[index]),
            "event": torch.tensor(self.event[index], dtype=torch.float32),
            "time_days": torch.tensor(self.time_days[index], dtype=torch.float32),
            "patient_hash": patient,
        }
        if self.augmentation_enabled and not self.defer_batch_augmentation:
            mask = torch.from_numpy(self._read_mask(patient).astype(np.float32))
            result = apply_pet_safe_augmentation(
                image=raw_image,
                mask=mask,
                config=self.augmentation_config,
                split_role=self.split_role,
                branch=self.branch,
                subject_key=patient,
                outer_fold=self.outer_fold,
                inner_fold=self.inner_fold,
                epoch=self.epoch,
                base_seed=self.base_seed,
            )
            row["image"] = result.image
        else:
            row["image"] = raw_image
        if self.sample_weight is not None:
            row["sample_weight"] = torch.tensor(
                self.sample_weight[index], dtype=torch.float32
            )
        return row

    def masks_for_subjects(self, subjects: list[str] | tuple[str, ...]):
        import torch

        return torch.stack(
            [torch.from_numpy(self._read_mask(str(subject)).astype(np.float32)) for subject in subjects]
        )

    def __getstate__(self):
        state = super().__getstate__()
        state["_mask_handle"] = None
        return state

    def close(self) -> None:
        super().close()
        if self._mask_handle is not None:
            self._mask_handle.close()
            self._mask_handle = None


def _set_dataset_epoch(dataset: Any, epoch: int) -> None:
    setter = getattr(dataset, "set_epoch", None)
    if callable(setter):
        setter(int(epoch))


def _apply_deferred_batch_augmentation(dataset: Any, batch: dict[str, Any], device):
    if not bool(getattr(dataset, "defer_batch_augmentation", False)):
        return batch
    if not bool(getattr(dataset, "augmentation_enabled", False)):
        return batch
    subjects = [str(value) for value in batch["patient_hash"]]
    masks = dataset.masks_for_subjects(subjects).to(device, non_blocking=True)
    images = batch["image"].to(device, non_blocking=True)
    result = apply_pet_safe_augmentation_batch(
        images=images,
        masks=masks,
        config=dataset.augmentation_config,
        split_role=dataset.split_role,
        branch=dataset.branch,
        subject_keys=subjects,
        outer_fold=dataset.outer_fold,
        inner_fold=dataset.inner_fold,
        epoch=dataset.epoch,
        base_seed=dataset.base_seed,
    )
    updated = dict(batch)
    updated["image"] = result.images
    return updated


def _ipcw_validation_ibs(
    *,
    validation_logits,
    train_event: np.ndarray,
    train_time_days: np.ndarray,
    validation_event: np.ndarray,
    validation_time_days: np.ndarray,
    bin_edges: Iterable[float],
) -> dict[str, Any]:
    """Compute the sole inner checkpoint-selection metric, without fallback."""

    import pandas as pd
    from sksurv.metrics import integrated_brier_score
    from sksurv.util import Surv

    from .image_outer_eval_v3 import _administratively_censor, _evaluation_grid

    try:
        train = pd.DataFrame(
            {
                "event": np.asarray(train_event, dtype=int),
                "time_days": np.asarray(train_time_days, dtype=float),
            }
        )
        validation = pd.DataFrame(
            {
                "event": np.asarray(validation_event, dtype=int),
                "time_days": np.asarray(validation_time_days, dtype=float),
            }
        )
        if train.empty or validation.empty or int(validation_logits.shape[0]) != len(validation):
            raise ValueError("INNER_METRIC_ROW_COUNT_INVALID")
        edges = [float(value) for value in bin_edges]
        grid = _evaluation_grid(
            train["time_days"].to_numpy(),
            validation["time_days"].to_numpy(),
            edges,
        )
        cutoff = min(
            float(train["time_days"].max()),
            float(validation["time_days"].max()),
        )
        cutoff -= max(abs(cutoff) * 1e-9, 1e-9)
        eval_train = _administratively_censor(train, cutoff)
        eval_validation = _administratively_censor(validation, cutoff)
        survival = survival_at_times(
            validation_logits.detach().float().cpu(), edges, grid.tolist()
        ).numpy()
        if not np.isfinite(survival).all() or np.any(
            np.diff(survival, axis=1) > 1e-7
        ):
            raise ValueError("SURVIVAL_PROBABILITY_INVALID")
        y_train = Surv.from_arrays(
            event=eval_train["event"].astype(bool).to_numpy(),
            time=eval_train["time_days"].astype(float).to_numpy(),
        )
        y_validation = Surv.from_arrays(
            event=eval_validation["event"].astype(bool).to_numpy(),
            time=eval_validation["time_days"].astype(float).to_numpy(),
        )
        value = float(
            integrated_brier_score(y_train, y_validation, survival, grid)
        )
        if not np.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"IBS_OUT_OF_RANGE:{value}")
        return {
            "metric_name": "IPCW_INTEGRATED_BRIER_SCORE",
            "metric_value": value,
            "evaluation_grid_days": [float(item) for item in grid],
            "administrative_censoring_cutoff_days": float(cutoff),
            "censoring_distribution_source": "INNER_TRAIN_ONLY",
        }
    except Exception as exc:
        if isinstance(exc, TrainingRuntimeV4Error):
            raise
        raise TrainingRuntimeV4Error(
            f"INNER_IBS_NOT_ESTIMABLE:{type(exc).__name__}:{exc}"
        ) from exc


def fit_survival_model_v4(
    *,
    model,
    train_dataset,
    validation_dataset=None,
    bin_edges: Iterable[float],
    output_dir: str | Path,
    run_contract_digest: str,
    seed: int,
    max_epochs: int,
    patience: int,
    learning_rate: float,
    weight_decay: float,
    batch_size: int,
    gradient_accumulation_steps: int,
    device: str,
    amp: bool,
    amp_dtype: str = "bfloat16",
    amp_initial_scale: float = 256.0,
    amp_growth_interval: int = 2000,
    fixed_epochs: int | None = None,
    interrupt_after_epoch: int | None = None,
    progress_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Exact-resume v4 trainer; augmentation epoch is committed with the model."""

    import torch

    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    checkpoint_path = directory / "epoch_checkpoint.pt"
    locked_path = directory / "locked_checkpoint.pt"
    summary_path = directory / "training_summary.json"
    if len(str(run_contract_digest)) != 64:
        raise TrainingRuntimeV4Error("RUN_CONTRACT_SHA256_REQUIRED")
    target_epochs = int(fixed_epochs if fixed_epochs is not None else max_epochs)
    if target_epochs <= 0 or int(gradient_accumulation_steps) <= 0:
        raise TrainingRuntimeV4Error("TRAINING_EPOCH_OR_ACCUMULATION_INVALID")
    if fixed_epochs is None and validation_dataset is None:
        raise TrainingRuntimeV4Error("INNER_VALIDATION_REQUIRED_FOR_IPCW_SELECTION")

    device_object = torch.device(device)
    seed_everything(seed)
    model = model.to(device_object)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(learning_rate), weight_decay=float(weight_decay)
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, target_epochs)
    )
    amp_enabled = bool(amp and device_object.type == "cuda")
    requested_amp_dtype = str(amp_dtype).lower()
    if requested_amp_dtype not in {"bfloat16", "float16"}:
        raise TrainingRuntimeV4Error(f"UNSUPPORTED_AMP_DTYPE:{amp_dtype}")
    if amp_enabled and requested_amp_dtype == "bfloat16":
        if not torch.cuda.is_bf16_supported():
            raise TrainingRuntimeV4Error("BFLOAT16_NOT_SUPPORTED_BY_DEVICE")
        autocast_dtype = torch.bfloat16
    else:
        autocast_dtype = torch.float16
    scaler_enabled = bool(amp_enabled and autocast_dtype == torch.float16)
    try:
        scaler = torch.amp.GradScaler(
            "cuda",
            enabled=scaler_enabled,
            init_scale=float(amp_initial_scale),
            growth_interval=int(amp_growth_interval),
        )
    except (AttributeError, TypeError):  # pragma: no cover
        scaler = torch.cuda.amp.GradScaler(
            enabled=scaler_enabled,
            init_scale=float(amp_initial_scale),
            growth_interval=int(amp_growth_interval),
        )
    edge_values = [float(value) for value in bin_edges]
    edges = torch.as_tensor(edge_values, dtype=torch.float32, device=device_object)
    history: list[dict[str, Any]] = []
    best_validation = float("inf")
    best_epoch = 0
    best_state = _cpu_state(model)
    patience_counter = 0
    start_epoch = 1
    resumed_from_epoch: int | None = None

    def _lock_from_terminal(checkpoint: dict[str, Any]) -> dict[str, Any]:
        if checkpoint.get("run_contract_digest") != run_contract_digest:
            raise TrainingRuntimeV4Error("TRAINING_CHECKPOINT_CONTRACT_MISMATCH")
        if checkpoint.get("fold_terminal") is not True:
            raise TrainingRuntimeV4Error("FINALIZE_REQUIRES_TERMINAL_CHECKPOINT")
        best = checkpoint.get("best_model_state")
        if best is None:
            raise TrainingRuntimeV4Error("FINALIZE_MISSING_BEST_STATE")
        if locked_path.is_file():
            existing = torch.load(locked_path, map_location="cpu", weights_only=False)
            if existing.get("run_contract_digest") != run_contract_digest:
                raise TrainingRuntimeV4Error("LOCKED_CHECKPOINT_CONTRACT_MISMATCH")
            existing_state = existing.get("model_state", {})
            if set(existing_state) != set(best) or any(
                not torch.equal(existing_state[key], best[key]) for key in best
            ):
                raise TrainingRuntimeV4Error("LOCKED_CHECKPOINT_STATE_MISMATCH")
        else:
            _atomic_torch_save(
                locked_path,
                {
                    "schema_version": "4.0",
                    "run_contract_digest": run_contract_digest,
                    "model_state": best,
                    "best_epoch": int(checkpoint.get("best_epoch", 0)),
                    "terminal_reason": checkpoint.get("terminal_reason"),
                    "selection_metric_name": checkpoint.get(
                        "selection_metric_name"
                    ),
                },
            )
        payload = {
            "schema_version": "4.0",
            "status": "CHECKPOINT_LOCKED",
            "run_contract_digest": run_contract_digest,
            "start_epoch": int(checkpoint.get("epoch_completed", 0)) + 1,
            "resumed_from_epoch": int(checkpoint.get("epoch_completed", 0)),
            "epoch_completed": int(checkpoint.get("epoch_completed", 0)),
            "best_epoch": int(checkpoint.get("best_epoch", 0)),
            "best_validation": float(checkpoint.get("best_validation", 0.0)),
            "selection_metric_name": checkpoint.get("selection_metric_name"),
            "terminal_reason": str(checkpoint.get("terminal_reason")),
            "history": list(checkpoint.get("history", [])),
            "locked_checkpoint_path": str(locked_path),
        }
        if summary_path.is_file():
            existing_summary = json.loads(summary_path.read_text(encoding="utf-8"))
            if existing_summary.get("run_contract_digest") != run_contract_digest:
                raise TrainingRuntimeV4Error("SUMMARY_CONTRACT_MISMATCH")
            existing_summary["start_epoch"] = payload["start_epoch"]
            existing_summary["resumed_from_epoch"] = payload["resumed_from_epoch"]
            return existing_summary
        atomic_write_json(summary_path, payload)
        return payload

    if checkpoint_path.is_file():
        checkpoint = torch.load(checkpoint_path, map_location=device_object, weights_only=False)
        if checkpoint.get("run_contract_digest") != run_contract_digest:
            raise TrainingRuntimeV4Error("TRAINING_CHECKPOINT_CONTRACT_MISMATCH")
        model.load_state_dict(checkpoint["model_state"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        scaler.load_state_dict(checkpoint["scaler_state"])
        history = list(checkpoint["history"])
        best_validation = float(checkpoint["best_validation"])
        best_epoch = int(checkpoint["best_epoch"])
        best_state = checkpoint["best_model_state"]
        patience_counter = int(checkpoint["patience_counter"])
        resumed_from_epoch = int(checkpoint["epoch_completed"])
        start_epoch = resumed_from_epoch + 1
        _restore_rng(checkpoint["rng_state"])
        if checkpoint.get("fold_terminal") is True:
            return _lock_from_terminal(checkpoint)

    wall_start = time.perf_counter()
    terminal_reason = "MAX_EPOCHS_COMPLETED"
    last_epoch = start_epoch - 1
    for epoch in range(start_epoch, target_epochs + 1):
        epoch_start = time.perf_counter()
        _set_dataset_epoch(train_dataset, epoch)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        train_loader = _loader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            seed=int(seed) + epoch,
            pin_memory=device_object.type == "cuda",
        )
        train_sum = 0.0
        train_rows = 0
        optimizer_steps = 0
        for step, batch in enumerate(train_loader, start=1):
            batch = _apply_deferred_batch_augmentation(
                train_dataset, batch, device_object
            )
            with torch.autocast(
                device_type=device_object.type,
                dtype=autocast_dtype,
                enabled=amp_enabled,
            ):
                logits = _forward(model, batch, device_object)
            loss = discrete_time_survival_nll(
                logits.float(),
                batch["event"].to(device_object),
                batch["time_days"].to(device_object),
                edges.float(),
                batch.get("sample_weight").to(device_object)
                if batch.get("sample_weight") is not None
                else None,
            )
            scaler.scale(loss / int(gradient_accumulation_steps)).backward()
            if step % int(gradient_accumulation_steps) == 0 or step == len(train_loader):
                scale_before = float(scaler.get_scale())
                scaler.step(optimizer)
                scaler.update()
                scale_after = float(scaler.get_scale())
                if not scaler_enabled or scale_after >= scale_before:
                    optimizer_steps += 1
                optimizer.zero_grad(set_to_none=True)
            rows = int(batch["event"].numel())
            train_sum += float(loss.detach().cpu()) * rows
            train_rows += rows
        if optimizer_steps <= 0:
            raise TrainingRuntimeV4Error(f"NO_FINITE_OPTIMIZER_STEP_IN_EPOCH:{epoch}")

        train_loss = train_sum / max(1, train_rows)
        model.eval()
        if validation_dataset is None:
            validation_loss = train_loss
            validation_metric = None
            selection_value = float(train_loss)
            selection_metric_name = "FIXED_EPOCHS_NO_INNER_SELECTION"
        else:
            _set_dataset_epoch(validation_dataset, epoch)
            validation_sum = 0.0
            validation_rows = 0
            validation_logits_rows = []
            validation_loader = _loader(
                validation_dataset,
                batch_size=batch_size,
                shuffle=False,
                seed=int(seed),
                pin_memory=device_object.type == "cuda",
            )
            with torch.no_grad():
                for batch in validation_loader:
                    with torch.autocast(
                        device_type=device_object.type,
                        dtype=autocast_dtype,
                        enabled=amp_enabled,
                    ):
                        logits = _forward(model, batch, device_object)
                    validation_logits_rows.append(logits.detach().float().cpu())
                    loss = discrete_time_survival_nll(
                        logits.float(),
                        batch["event"].to(device_object),
                        batch["time_days"].to(device_object),
                        edges.float(),
                    )
                    rows = int(batch["event"].numel())
                    validation_sum += float(loss.detach().cpu()) * rows
                    validation_rows += rows
            validation_loss = validation_sum / max(1, validation_rows)
            if not validation_logits_rows:
                raise TrainingRuntimeV4Error("INNER_VALIDATION_LOGITS_MISSING")
            validation_metric = _ipcw_validation_ibs(
                validation_logits=torch.cat(validation_logits_rows, dim=0),
                train_event=np.asarray(train_dataset.event),
                train_time_days=np.asarray(train_dataset.time_days),
                validation_event=np.asarray(validation_dataset.event),
                validation_time_days=np.asarray(validation_dataset.time_days),
                bin_edges=edge_values,
            )
            selection_value = float(validation_metric["metric_value"])
            selection_metric_name = str(validation_metric["metric_name"])

        scheduler.step()
        if fixed_epochs is None and selection_value < best_validation - 1e-8:
            best_validation = selection_value
            best_epoch = epoch
            best_state = _cpu_state(model)
            patience_counter = 0
        elif fixed_epochs is None:
            patience_counter += 1
        else:
            best_validation = selection_value
            best_epoch = epoch
            best_state = _cpu_state(model)
            patience_counter = 0
        history.append(
            {
                "epoch": int(epoch),
                "train_loss": float(train_loss),
                "validation_loss": float(validation_loss),
                "train_validation_nll_gap": float(validation_loss - train_loss),
                "selection_metric_name": selection_metric_name,
                "validation_ipcw_ibs": (
                    float(validation_metric["metric_value"])
                    if validation_metric is not None
                    else None
                ),
                "selection_metric_grid_days": (
                    validation_metric["evaluation_grid_days"]
                    if validation_metric is not None
                    else []
                ),
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "optimizer_steps": int(optimizer_steps),
                "augmentation_epoch": int(epoch),
            }
        )
        last_epoch = epoch
        early_terminal = fixed_epochs is None and patience_counter >= int(patience)
        checkpoint = {
            "schema_version": "4.0",
            "run_contract_digest": run_contract_digest,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "scaler_state": scaler.state_dict(),
            "best_model_state": best_state,
            "epoch_completed": int(epoch),
            "best_epoch": int(best_epoch),
            "best_validation": float(best_validation),
            "selection_metric_name": selection_metric_name,
            "patience_counter": int(patience_counter),
            "history": history,
            "rng_state": _rng_state(),
            "fold_terminal": bool(early_terminal or epoch == target_epochs),
            "terminal_reason": (
                "EARLY_STOPPING" if early_terminal else "MAX_EPOCHS_COMPLETED"
            )
            if early_terminal or epoch == target_epochs
            else None,
        }
        _atomic_torch_save(checkpoint_path, checkpoint)
        elapsed = time.perf_counter() - epoch_start
        average = (time.perf_counter() - wall_start) / max(1, epoch - start_epoch + 1)
        print(
            json.dumps(
                {
                    "event": "EPOCH_COMMITTED",
                    **(progress_context or {}),
                    "epoch": epoch,
                    "target_epochs": target_epochs,
                    "train_loss": train_loss,
                    "validation_loss": validation_loss,
                    "train_validation_nll_gap": validation_loss - train_loss,
                    "selection_metric_name": selection_metric_name,
                    "validation_ipcw_ibs": (
                        float(validation_metric["metric_value"])
                        if validation_metric is not None
                        else None
                    ),
                    "epoch_seconds": elapsed,
                    "eta_seconds": max(0.0, (target_epochs - epoch) * average),
                    "checkpoint": str(checkpoint_path),
                }
            ),
            flush=True,
        )
        if interrupt_after_epoch is not None and epoch == int(interrupt_after_epoch):
            checkpoint["fold_terminal"] = False
            checkpoint["terminal_reason"] = None
            _atomic_torch_save(checkpoint_path, checkpoint)
            raise PlannedTrainingInterruption(f"INTERRUPTED_AFTER_EPOCH_{epoch}")
        if early_terminal:
            terminal_reason = "EARLY_STOPPING"
            break

    model.load_state_dict(best_state, strict=True)
    _atomic_torch_save(
        locked_path,
        {
            "schema_version": "4.0",
            "run_contract_digest": run_contract_digest,
            "model_state": best_state,
            "best_epoch": int(best_epoch),
            "terminal_reason": terminal_reason,
            "selection_metric_name": (
                "IPCW_INTEGRATED_BRIER_SCORE"
                if fixed_epochs is None
                else "FIXED_EPOCHS_NO_INNER_SELECTION"
            ),
        },
    )
    summary = {
        "schema_version": "4.0",
        "status": "CHECKPOINT_LOCKED",
        "run_contract_digest": run_contract_digest,
        "start_epoch": int(start_epoch),
        "resumed_from_epoch": resumed_from_epoch,
        "epoch_completed": int(last_epoch),
        "best_epoch": int(best_epoch),
        "best_validation": float(best_validation),
        "selection_metric_name": (
            "IPCW_INTEGRATED_BRIER_SCORE"
            if fixed_epochs is None
            else "FIXED_EPOCHS_NO_INNER_SELECTION"
        ),
        "terminal_reason": terminal_reason,
        "history": history,
        "locked_checkpoint_path": str(locked_path),
    }
    atomic_write_json(summary_path, summary)
    return summary
