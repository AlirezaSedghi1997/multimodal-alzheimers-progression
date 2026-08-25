from __future__ import annotations

import copy
import json
import os
import random
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import h5py
import numpy as np

from .atomic_io import atomic_write_json
from .image_survival_v3 import discrete_time_survival_nll


class TrainingRuntimeV3Error(RuntimeError):
    """Raised when exact-resume image training cannot preserve its contract."""


class PlannedTrainingInterruption(RuntimeError):
    """Test-only interruption after an atomically committed epoch."""


def seed_everything(seed: int) -> None:
    import torch

    numeric = int(seed)
    os.environ["PYTHONHASHSEED"] = str(numeric)
    random.seed(numeric)
    np.random.seed(numeric)
    torch.manual_seed(numeric)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(numeric)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass


class MemorySurvivalDataset:
    def __init__(
        self,
        *,
        tabular: np.ndarray,
        event: np.ndarray,
        time_days: np.ndarray,
        images: np.ndarray | None = None,
        patient_hashes: Iterable[str] | None = None,
        sample_weight: np.ndarray | None = None,
    ) -> None:
        import torch

        self.tabular = torch.as_tensor(tabular, dtype=torch.float32)
        self.event = torch.as_tensor(event, dtype=torch.float32)
        self.time_days = torch.as_tensor(time_days, dtype=torch.float32)
        self.images = (
            torch.as_tensor(images, dtype=torch.float32) if images is not None else None
        )
        self.sample_weight = (
            torch.as_tensor(sample_weight, dtype=torch.float32)
            if sample_weight is not None
            else None
        )
        if not (len(self.tabular) == len(self.event) == len(self.time_days)):
            raise TrainingRuntimeV3Error("MEMORY_DATASET_LENGTH_MISMATCH")
        if self.images is not None and len(self.images) != len(self.tabular):
            raise TrainingRuntimeV3Error("MEMORY_IMAGE_LENGTH_MISMATCH")
        if self.sample_weight is not None and len(self.sample_weight) != len(self.tabular):
            raise TrainingRuntimeV3Error("MEMORY_SAMPLE_WEIGHT_LENGTH_MISMATCH")
        self.patient_hashes = list(patient_hashes or [f"row_{i}" for i in range(len(self.tabular))])

    def __len__(self) -> int:
        return len(self.tabular)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = {
            "tabular": self.tabular[index],
            "event": self.event[index],
            "time_days": self.time_days[index],
            "patient_hash": self.patient_hashes[index],
        }
        if self.images is not None:
            row["image"] = self.images[index]
        if self.sample_weight is not None:
            row["sample_weight"] = self.sample_weight[index]
        return row


class H5PatternSurvivalDataset:
    def __init__(
        self,
        *,
        cache_path: str | Path,
        patient_hashes: Iterable[str],
        tabular: np.ndarray,
        event: np.ndarray,
        time_days: np.ndarray,
        include_image: bool,
        sample_weight: np.ndarray | None = None,
    ) -> None:
        self.cache_path = Path(cache_path)
        self.patient_hashes = tuple(str(value) for value in patient_hashes)
        self.tabular = np.asarray(tabular, dtype=np.float32)
        self.event = np.asarray(event, dtype=np.float32)
        self.time_days = np.asarray(time_days, dtype=np.float32)
        self.include_image = bool(include_image)
        self.sample_weight = (
            np.asarray(sample_weight, dtype=np.float32)
            if sample_weight is not None
            else None
        )
        self._handle: h5py.File | None = None
        if not (
            len(self.patient_hashes)
            == len(self.tabular)
            == len(self.event)
            == len(self.time_days)
        ):
            raise TrainingRuntimeV3Error("H5_DATASET_LENGTH_MISMATCH")
        if self.sample_weight is not None and len(self.sample_weight) != len(self.patient_hashes):
            raise TrainingRuntimeV3Error("H5_SAMPLE_WEIGHT_LENGTH_MISMATCH")

    def __len__(self) -> int:
        return len(self.patient_hashes)

    def _h5(self) -> h5py.File:
        if self._handle is None:
            self._handle = h5py.File(self.cache_path, "r")
        return self._handle

    def __getitem__(self, index: int) -> dict[str, Any]:
        import torch

        patient = self.patient_hashes[index]
        row = {
            "tabular": torch.from_numpy(self.tabular[index]),
            "event": torch.tensor(self.event[index], dtype=torch.float32),
            "time_days": torch.tensor(self.time_days[index], dtype=torch.float32),
            "patient_hash": patient,
        }
        if self.include_image:
            handle = self._h5()
            if patient not in handle or "image" not in handle[patient]:
                raise TrainingRuntimeV3Error(f"PATTERN_CACHE_SUBJECT_MISSING:{patient}")
            image = np.asarray(handle[patient]["image"], dtype=np.float32)
            if not np.isfinite(image).all():
                raise TrainingRuntimeV3Error("PATTERN_CACHE_NONFINITE")
            row["image"] = torch.from_numpy(image)
        if self.sample_weight is not None:
            row["sample_weight"] = torch.tensor(
                self.sample_weight[index], dtype=torch.float32
            )
        return row

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_handle"] = None
        return state

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None

    def __del__(self):  # pragma: no cover - best-effort Windows handle cleanup
        try:
            self.close()
        except Exception:
            pass


@dataclass(frozen=True)
class FoldTabularTransformer:
    columns: tuple[str, ...]
    medians: tuple[float, ...]
    means: tuple[float, ...]
    scales: tuple[float, ...]

    @classmethod
    def fit(cls, frame, columns: Iterable[str]) -> "FoldTabularTransformer":
        names = tuple(str(value) for value in columns)
        values = frame.loc[:, names].apply(lambda column: np.asarray(column, dtype=float))
        array = values.to_numpy(dtype=float)
        medians = np.nanmedian(array, axis=0)
        if not np.isfinite(medians).all():
            raise TrainingRuntimeV3Error("TABULAR_TRAIN_COLUMN_ALL_MISSING")
        filled = np.where(np.isfinite(array), array, medians)
        means = filled.mean(axis=0)
        scales = filled.std(axis=0)
        scales = np.where(scales > 1e-8, scales, 1.0)
        return cls(names, tuple(medians), tuple(means), tuple(scales))

    def transform(self, frame) -> np.ndarray:
        array = frame.loc[:, self.columns].to_numpy(dtype=float)
        medians = np.asarray(self.medians)
        filled = np.where(np.isfinite(array), array, medians)
        transformed = (filled - np.asarray(self.means)) / np.asarray(self.scales)
        if not np.isfinite(transformed).all():
            raise TrainingRuntimeV3Error("TABULAR_TRANSFORM_NONFINITE")
        return transformed.astype(np.float32)

    def to_dict(self) -> dict[str, Any]:
        return {
            "columns": list(self.columns),
            "medians": list(self.medians),
            "means": list(self.means),
            "scales": list(self.scales),
            "fit_scope": "TRAIN_PARTITION_ONLY",
        }


def _atomic_torch_save(path: Path, payload: dict[str, Any]) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        torch.save(payload, temporary)
        with temporary.open("r+b") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _cpu_state(model) -> dict[str, Any]:
    return {
        key: value.detach().cpu().clone() for key, value in model.state_dict().items()
    }


def _rng_state() -> dict[str, Any]:
    import torch

    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def _restore_rng(state: dict[str, Any]) -> None:
    import torch

    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    cpu_state = state["torch_cpu"]
    if isinstance(cpu_state, torch.Tensor) and cpu_state.device.type != "cpu":
        cpu_state = cpu_state.cpu()
    torch.set_rng_state(cpu_state)
    if torch.cuda.is_available() and state.get("torch_cuda"):
        cuda_states = state["torch_cuda"]
        # Ensure CPU tensors for set_rng_state_all (handles map_location=cuda case)
        fixed = []
        for s in cuda_states:
            if isinstance(s, torch.Tensor) and s.device.type != "cpu":
                fixed.append(s.cpu())
            else:
                fixed.append(s)
        torch.cuda.set_rng_state_all(fixed)


def _forward(model, batch: dict[str, Any], device):
    tabular = batch["tabular"].to(device, non_blocking=True)
    if "image" in batch:
        return model(batch["image"].to(device, non_blocking=True), tabular)
    return model(tabular)


def _loader(dataset, *, batch_size: int, shuffle: bool, seed: int, pin_memory: bool):
    import torch

    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        num_workers=0,
        pin_memory=bool(pin_memory),
        generator=generator,
        drop_last=False,
    )


def fit_survival_model(
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
    fixed_epochs: int | None = None,
    interrupt_after_epoch: int | None = None,
    progress_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    import torch

    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    checkpoint_path = directory / "epoch_checkpoint.pt"
    locked_path = directory / "locked_checkpoint.pt"
    summary_path = directory / "training_summary.json"
    if len(str(run_contract_digest)) != 64:
        raise TrainingRuntimeV3Error("RUN_CONTRACT_SHA256_REQUIRED")
    target_epochs = int(fixed_epochs if fixed_epochs is not None else max_epochs)
    if target_epochs <= 0 or int(gradient_accumulation_steps) <= 0:
        raise TrainingRuntimeV3Error("TRAINING_EPOCH_OR_ACCUMULATION_INVALID")
    device_object = torch.device(device)
    model = model.to(device_object)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(learning_rate), weight_decay=float(weight_decay)
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, target_epochs)
    )
    amp_enabled = bool(amp and device_object.type == "cuda")
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    except (AttributeError, TypeError):  # pragma: no cover - older pinned torch
        scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    edges = torch.as_tensor(list(bin_edges), dtype=torch.float32, device=device_object)
    history: list[dict[str, float | int]] = []
    best_validation = float("inf")
    best_epoch = 0
    best_state = _cpu_state(model)
    patience_counter = 0
    start_epoch = 1
    resumed_from_epoch: int | None = None

    def _finalize_terminal_checkpoint() -> dict[str, Any]:
        # Idempotent finalization: create locked checkpoint and summary from terminal epoch checkpoint
        ckpt = torch.load(checkpoint_path, map_location=device_object, weights_only=False)
        if ckpt.get("run_contract_digest") != run_contract_digest:
            raise TrainingRuntimeV3Error("TRAINING_CHECKPOINT_CONTRACT_MISMATCH")
        if ckpt.get("fold_terminal") is not True:
            raise TrainingRuntimeV3Error("FINALIZE_REQUIRES_TERMINAL_CHECKPOINT")
        best = ckpt.get("best_model_state")
        if best is None:
            raise TrainingRuntimeV3Error("FINALIZE_MISSING_BEST_STATE")
        # Locked checkpoint: atomic, verify if exists
        if locked_path.is_file():
            existing = torch.load(locked_path, map_location=device_object, weights_only=False)
            if existing.get("run_contract_digest") != run_contract_digest:
                raise TrainingRuntimeV3Error("LOCKED_CHECKPOINT_CONTRACT_MISMATCH")
            # Verify state hash if present (compare tensors properly)
            existing_state = existing.get("model_state") or {}
            mismatch = False
            if set(existing_state.keys()) != set(best.keys()):
                mismatch = True
            else:
                for k in best:
                    try:
                        if not torch.equal(existing_state[k], best[k]):
                            mismatch = True
                            break
                    except Exception:
                        mismatch = True
                        break
            if mismatch:
                raise TrainingRuntimeV3Error("LOCKED_CHECKPOINT_STATE_MISMATCH")
            # If already locked, return existing summary with resumed info
            if summary_path.is_file():
                try:
                    existing_summary = json.loads(summary_path.read_text(encoding="utf-8"))
                    # Update resumed info for second run
                    existing_summary["resumed_from_epoch"] = int(ckpt.get("epoch_completed", 1))
                    existing_summary["start_epoch"] = int(ckpt.get("epoch_completed", 1)) + 1
                    return existing_summary
                except (OSError, json.JSONDecodeError):
                    pass
        else:
            _atomic_torch_save(
                locked_path,
                {
                    "schema_version": "3.0",
                    "run_contract_digest": run_contract_digest,
                    "model_state": best,
                    "best_epoch": int(ckpt.get("best_epoch", 0)),
                    "terminal_reason": str(ckpt.get("terminal_reason", "MAX_EPOCHS_COMPLETED")),
                },
            )
        # Summary: atomic, verify if exists
        if summary_path.is_file():
            try:
                existing_summary = json.loads(summary_path.read_text(encoding="utf-8"))
                if existing_summary.get("run_contract_digest") != run_contract_digest:
                    raise TrainingRuntimeV3Error("SUMMARY_CONTRACT_MISMATCH")
            except (OSError, json.JSONDecodeError) as exc:
                raise TrainingRuntimeV3Error("SUMMARY_READ_FAILED") from exc
        else:
            # Reconstruct minimal summary from checkpoint
            hist = list(ckpt.get("history", []))
            summary_payload = {
                "schema_version": "3.0",
                "status": "CHECKPOINT_LOCKED",
                "run_contract_digest": run_contract_digest,
                "start_epoch": int(ckpt.get("epoch_completed", 0)),
                "resumed_from_epoch": int(ckpt.get("epoch_completed", 0)) - 1 if hist else None,
                "epoch_completed": int(ckpt.get("epoch_completed", 0)),
                "best_epoch": int(ckpt.get("best_epoch", 0)),
                "best_validation": float(ckpt.get("best_validation", 0.0)),
                "terminal_reason": str(ckpt.get("terminal_reason", "MAX_EPOCHS_COMPLETED")),
                "history": hist,
                "locked_checkpoint_path": str(locked_path),
            }
            atomic_write_json(summary_path, summary_payload)
        return json.loads(summary_path.read_text(encoding="utf-8"))

    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device_object, weights_only=False)
        if checkpoint.get("run_contract_digest") != run_contract_digest:
            raise TrainingRuntimeV3Error("TRAINING_CHECKPOINT_CONTRACT_MISMATCH")
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
            return _finalize_terminal_checkpoint()

    wall_start = time.perf_counter()
    terminal_reason = "MAX_EPOCHS_COMPLETED"
    last_epoch = start_epoch - 1
    for epoch in range(start_epoch, target_epochs + 1):
        epoch_start = time.perf_counter()
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
        for step, batch in enumerate(train_loader, start=1):
            with torch.autocast(
                device_type=device_object.type,
                dtype=torch.float16,
                enabled=amp_enabled,
            ):
                logits = _forward(model, batch, device_object)
                loss = discrete_time_survival_nll(
                    logits,
                    batch["event"].to(device_object),
                    batch["time_days"].to(device_object),
                    edges,
                    batch.get("sample_weight", None).to(device_object)
                    if batch.get("sample_weight", None) is not None
                    else None,
                )
                scaled_loss = loss / int(gradient_accumulation_steps)
            scaler.scale(scaled_loss).backward()
            if step % int(gradient_accumulation_steps) == 0 or step == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            rows = int(batch["event"].numel())
            train_sum += float(loss.detach().cpu()) * rows
            train_rows += rows

        model.eval()
        train_loss = train_sum / max(1, train_rows)
        if validation_dataset is None:
            validation_loss = train_loss
        else:
            validation_sum = 0.0
            validation_rows = 0
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
                        dtype=torch.float16,
                        enabled=amp_enabled,
                    ):
                        logits = _forward(model, batch, device_object)
                        loss = discrete_time_survival_nll(
                            logits,
                            batch["event"].to(device_object),
                            batch["time_days"].to(device_object),
                            edges,
                        )
                    rows = int(batch["event"].numel())
                    validation_sum += float(loss.detach().cpu()) * rows
                    validation_rows += rows
            validation_loss = validation_sum / max(1, validation_rows)
        scheduler.step()
        if fixed_epochs is None and validation_loss < best_validation - 1e-8:
            best_validation = validation_loss
            best_epoch = epoch
            best_state = _cpu_state(model)
            patience_counter = 0
        elif fixed_epochs is None:
            patience_counter += 1
        else:
            best_validation = validation_loss
            best_epoch = epoch
            best_state = _cpu_state(model)
            patience_counter = 0
        history.append(
            {
                "epoch": int(epoch),
                "train_loss": float(train_loss),
                "validation_loss": float(validation_loss),
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
            }
        )
        last_epoch = epoch
        early_terminal = fixed_epochs is None and patience_counter >= int(patience)
        checkpoint = {
            "schema_version": "3.0",
            "run_contract_digest": run_contract_digest,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "scaler_state": scaler.state_dict(),
            "best_model_state": best_state,
            "epoch_completed": int(epoch),
            "best_epoch": int(best_epoch),
            "best_validation": float(best_validation),
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
        completed_here = epoch - start_epoch + 1
        average = (time.perf_counter() - wall_start) / max(1, completed_here)
        eta = max(0.0, (target_epochs - epoch) * average)
        print(
            json.dumps(
                {
                    "event": "EPOCH_COMMITTED",
                    **(progress_context or {}),
                    "epoch": epoch,
                    "target_epochs": target_epochs,
                    "train_loss": train_loss,
                    "validation_loss": validation_loss,
                    "epoch_seconds": elapsed,
                    "eta_seconds": eta,
                    "checkpoint": str(checkpoint_path),
                }
            ),
            flush=True,
        )
        if interrupt_after_epoch is not None and epoch == int(interrupt_after_epoch):
            # Test-only: force nonterminal state after the same committed epoch.
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
            "schema_version": "3.0",
            "run_contract_digest": run_contract_digest,
            "model_state": best_state,
            "best_epoch": int(best_epoch),
            "terminal_reason": terminal_reason,
        },
    )
    summary = {
        "schema_version": "3.0",
        "status": "CHECKPOINT_LOCKED",
        "run_contract_digest": run_contract_digest,
        "start_epoch": int(start_epoch),
        "resumed_from_epoch": resumed_from_epoch,
        "epoch_completed": int(last_epoch),
        "best_epoch": int(best_epoch),
        "best_validation": float(best_validation),
        "terminal_reason": terminal_reason,
        "history": history,
        "locked_checkpoint_path": str(locked_path),
    }
    atomic_write_json(summary_path, summary)
    return summary
