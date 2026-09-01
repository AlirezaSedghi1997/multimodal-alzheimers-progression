from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from .atomic_io import atomic_write_json, read_json
from .contracts import sha256_mapping
from .image_factorial_v4 import FACTORIAL_ARMS, build_factorial_image_model
from .image_training_orchestrator_v3 import (
    PUBLIC_OUTER_METRIC_KEYS,
    load_locked_state_for_eval,
    read_verified_winner_record,
    write_private_outer_record,
    write_winner_record,
)
from .image_training_runtime_v3 import FoldTabularTransformer, seed_everything
from .image_training_runtime_v4 import (
    FactorialH5SurvivalDataset,
    _ipcw_validation_ibs,
    fit_survival_model_v4,
)
from .outer_evaluation_control import OuterFoldController
from .pet_augmentation_v4 import PetSafeAugmentationConfig
from .survival_models import select_inner_winner


class ImageFactorialOrchestratorV4Error(RuntimeError):
    pass


def _sha256_file(path: str | Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolved_arm_model_config(config: dict[str, Any], arm_id: str) -> dict[str, Any]:
    if arm_id not in FACTORIAL_ARMS:
        raise ImageFactorialOrchestratorV4Error(f"UNKNOWN_FACTORIAL_ARM:{arm_id}")
    resolved = dict(config.get("model", {}))
    resolved.pop("hpo_candidates", None)
    resolved.update(config.get("factorial_arms", {}).get(arm_id, {}).get("model_overrides", {}))
    return resolved


def build_factorial_cell_contract(
    *,
    config: dict[str, Any],
    branch: str,
    arm_id: str,
    outer_fold: int,
    candidate_id: str,
    membership_sha256: str,
    cache_sha256: str,
    mask_sha256: str,
    split_sha256: str,
    augmentation_qc_sha256: str,
    stage: str = "UNSPECIFIED",
    inner_fold: int | None = None,
    partition_contract: dict[str, Any] | None = None,
    training_parameters: dict[str, Any] | None = None,
) -> str:
    payload = {
        "schema_version": "4.0",
        "branch": str(branch),
        "arm_id": str(arm_id),
        "arm_contract": FACTORIAL_ARMS[arm_id].__dict__,
        "outer_fold": int(outer_fold),
        "candidate_id": str(candidate_id),
        "stage": str(stage),
        "inner_fold": int(inner_fold) if inner_fold is not None else None,
        "partition_contract": dict(partition_contract or {}),
        "training_parameters": dict(training_parameters or {}),
        "resolved_model_config": resolved_arm_model_config(config, arm_id),
        "hpo_candidates": config.get("model", {}).get("hpo_candidates", []),
        "hazard_bin_edges": config.get("shared", {}).get("hazard_bin_edges_days", []),
        "feature_columns": config.get("branches", {}).get(branch, {}).get("model_features", []),
        "membership_sha256": membership_sha256,
        "cache_sha256": cache_sha256,
        "mask_sha256": mask_sha256,
        "split_sha256": split_sha256,
        "augmentation_qc_sha256": augmentation_qc_sha256,
        "augmentation": config.get("augmentation", {}),
        "seed": int(config.get("seed", 0)),
        "runtime": config.get("runtime", {}),
        "software_runtime_contract": config.get("software_runtime_contract", {}),
        "training_source_sha256": config.get("training_source_sha256"),
    }
    return sha256_mapping(payload)


class ImageFactorialOrchestratorV4:
    def __init__(
        self,
        *,
        config: dict[str, Any],
        output_root: str | Path,
        branch: str,
        membership_sha256: str,
        cache_sha256: str,
        mask_sha256: str,
        split_sha256: str,
        augmentation_qc_sha256: str,
        h5_cache_path: str | Path,
        mask_path: str | Path,
        augmentation_qc: dict[str, Any],
    ) -> None:
        self.config = config
        self.output_root = Path(output_root)
        self.branch = str(branch).lower()
        if self.branch not in {"amyloid", "tau"}:
            raise ImageFactorialOrchestratorV4Error(f"UNSUPPORTED_BRANCH:{branch}")
        self.arms = tuple(config.get("execution", {}).get("arms", FACTORIAL_ARMS.keys()))
        if set(self.arms) != set(FACTORIAL_ARMS):
            raise ImageFactorialOrchestratorV4Error("ALL_FIVE_FACTORIAL_ARMS_REQUIRED")
        self.seed = int(config.get("seed", 0))
        self.hpo_candidates = list(config.get("model", {}).get("hpo_candidates", []))
        if not self.hpo_candidates or any(
            "complexity_rank" not in candidate for candidate in self.hpo_candidates
        ):
            raise ImageFactorialOrchestratorV4Error("VALID_HPO_CANDIDATES_REQUIRED")
        self.membership_sha256 = membership_sha256
        self.cache_sha256 = cache_sha256
        self.mask_sha256 = mask_sha256
        self.split_sha256 = split_sha256
        self.augmentation_qc_sha256 = augmentation_qc_sha256
        self.h5_cache_path = Path(h5_cache_path)
        self.mask_path = Path(mask_path)
        self.augmentation_qc = augmentation_qc
        self.state_root = self.output_root / "orchestration" / self.branch
        self.private_root = self.output_root / "private" / "outer_evaluation" / self.branch
        self.public_root = self.output_root / "reports"
        for path in (self.state_root, self.private_root, self.public_root):
            path.mkdir(parents=True, exist_ok=True)
        self.master_contract_sha256 = self._master_contract_digest()
        self._verify_or_initialize_master_contract()

    def _master_contract_digest(self) -> str:
        return sha256_mapping(
            {
                "schema_version": "4.0",
                "branch": self.branch,
                "arms": list(self.arms),
                "seed": self.seed,
                "hpo_candidates": self.hpo_candidates,
                "membership_sha256": self.membership_sha256,
                "cache_sha256": self.cache_sha256,
                "mask_sha256": self.mask_sha256,
                "split_sha256": self.split_sha256,
                "augmentation_qc_sha256": self.augmentation_qc_sha256,
                "model": self.config.get("model", {}),
                "factorial_arms": self.config.get("factorial_arms", {}),
                "runtime": self.config.get("runtime", {}),
                "software_runtime_contract": self.config.get("software_runtime_contract", {}),
            }
        )

    def _verify_or_initialize_master_contract(self) -> None:
        path = self.state_root / "master_contract.json"
        if path.is_file():
            payload = read_json(path)
            if payload.get("master_contract_sha256") != self.master_contract_sha256:
                raise ImageFactorialOrchestratorV4Error("RUN_CONTRACT_MISMATCH:MASTER")
            return
        atomic_write_json(
            path,
            {
                "schema_version": "4.0",
                "branch": self.branch,
                "master_contract_sha256": self.master_contract_sha256,
                "claim_status": "POST_HOC_INTERNAL_EXPLORATORY",
            },
        )

    def verify_inputs(self) -> None:
        for value, name in (
            (self.membership_sha256, "membership"),
            (self.cache_sha256, "cache"),
            (self.mask_sha256, "mask"),
            (self.split_sha256, "split"),
            (self.augmentation_qc_sha256, "augmentation_qc"),
        ):
            if len(str(value)) != 64:
                raise ImageFactorialOrchestratorV4Error(f"SHA256_REQUIRED:{name}")
        if not self.h5_cache_path.is_file() or not self.mask_path.is_file():
            raise ImageFactorialOrchestratorV4Error("CACHE_OR_MASK_MISSING")
        selected = self.augmentation_qc.get("selected_envelope", {})
        expected = self.config.get("augmentation", {})
        for key in (
            "max_rotation_degrees",
            "max_translation_voxels",
            "scale_min",
            "scale_max",
        ):
            if float(selected.get(key, float("nan"))) != float(expected.get(key, float("nan"))):
                raise ImageFactorialOrchestratorV4Error(
                    f"AUGMENTATION_QC_CONFIG_MISMATCH:{key}"
                )
        if expected.get("psf_smoothing_enabled") or expected.get("count_noise_enabled"):
            raise ImageFactorialOrchestratorV4Error("UNSUPPORTED_PHYSICS_AUGMENTATION_ENABLED")

    def _augmentation_config(self) -> PetSafeAugmentationConfig:
        policy = self.config["augmentation"]
        return PetSafeAugmentationConfig(
            enabled=True,
            probability=float(policy["probability"]),
            max_rotation_degrees=float(policy["max_rotation_degrees"]),
            max_translation_voxels=float(policy["max_translation_voxels"]),
            scale_min=float(policy["scale_min"]),
            scale_max=float(policy["scale_max"]),
            psf_smoothing_enabled=False,
            count_noise_enabled=False,
            qc_contract_sha256=self.augmentation_qc_sha256,
        )

    def _cell_contract(
        self,
        arm_id: str,
        outer_fold: int,
        candidate_id: str,
        *,
        stage: str,
        inner_fold: int | None,
        partition_contract: dict[str, Any],
        training_parameters: dict[str, Any],
    ) -> str:
        return build_factorial_cell_contract(
            config=self.config,
            branch=self.branch,
            arm_id=arm_id,
            outer_fold=outer_fold,
            candidate_id=candidate_id,
            membership_sha256=self.membership_sha256,
            cache_sha256=self.cache_sha256,
            mask_sha256=self.mask_sha256,
            split_sha256=self.split_sha256,
            augmentation_qc_sha256=self.augmentation_qc_sha256,
            stage=stage,
            inner_fold=inner_fold,
            partition_contract=partition_contract,
            training_parameters=training_parameters,
        )

    def _dataset(
        self,
        *,
        frame: pd.DataFrame,
        tabular: np.ndarray,
        arm_id: str,
        split_role: str,
        outer_fold: int,
        inner_fold: int | None,
    ) -> FactorialH5SurvivalDataset:
        return FactorialH5SurvivalDataset(
            cache_path=self.h5_cache_path,
            mask_cache_path=self.mask_path,
            patient_hashes=frame["patient_hash"].astype(str).tolist(),
            tabular=tabular,
            event=frame["event"].to_numpy(),
            time_days=frame["time_days"].to_numpy(),
            arm_id=arm_id,
            augmentation_config=self._augmentation_config(),
            split_role=split_role,
            branch=self.branch,
            outer_fold=outer_fold,
            inner_fold=inner_fold,
            base_seed=self.seed,
            defer_batch_augmentation=True,
        )

    def winner_path(self, arm_id: str, outer_fold: int) -> Path:
        return self.state_root / f"winner_{arm_id}_outer_{outer_fold}.json"

    def read_winner(self, arm_id: str, outer_fold: int) -> dict[str, Any]:
        return read_verified_winner_record(
            self.winner_path(arm_id, outer_fold),
            expected_branch=self.branch,
            expected_variant=arm_id,
            expected_outer_fold=outer_fold,
            expected_master_contract_sha256=self.master_contract_sha256,
        )["winner"]

    def run_inner_hpo(
        self,
        *,
        outer_fold: int,
        arm_id: str,
        outer_train_frame: pd.DataFrame,
        inner_splits: list[dict[str, Any]],
        tabular_columns: list[str],
        hazard_bin_edges: list[float],
        device: str,
    ) -> dict[str, Any]:
        if arm_id not in self.arms:
            raise ImageFactorialOrchestratorV4Error(f"UNKNOWN_FACTORIAL_ARM:{arm_id}")
        candidate_records: list[dict[str, Any]] = []
        outer_keys = set(outer_train_frame["patient_hash"].astype(str))
        for candidate in self.hpo_candidates:
            candidate_id = str(candidate["candidate_id"])
            scores: list[float] = []
            epochs: list[int] = []
            metric_records: list[dict[str, Any]] = []
            for split in inner_splits:
                inner_fold = int(split.get("fold", 0))
                train_keys = set(split.get("train_subjects") or split.get("train_subject_keys") or [])
                val_keys = set(
                    split.get("validation_subjects")
                    or split.get("validation_subject_keys")
                    or split.get("val_subjects")
                    or []
                )
                if not train_keys or not val_keys or train_keys & val_keys:
                    raise ImageFactorialOrchestratorV4Error("INNER_PARTITION_EMPTY_OR_LEAKED")
                if train_keys | val_keys != outer_keys:
                    raise ImageFactorialOrchestratorV4Error("INNER_PARTITION_INCOMPLETE")
                train = outer_train_frame[
                    outer_train_frame["patient_hash"].astype(str).isin(train_keys)
                ].copy()
                validation = outer_train_frame[
                    outer_train_frame["patient_hash"].astype(str).isin(val_keys)
                ].copy()
                transformer = FoldTabularTransformer.fit(train, tabular_columns)
                train_ds = self._dataset(
                    frame=train,
                    tabular=transformer.transform(train),
                    arm_id=arm_id,
                    split_role="inner_train",
                    outer_fold=outer_fold,
                    inner_fold=inner_fold,
                )
                val_ds = self._dataset(
                    frame=validation,
                    tabular=transformer.transform(validation),
                    arm_id=arm_id,
                    split_role="inner_validation",
                    outer_fold=outer_fold,
                    inner_fold=inner_fold,
                )
                training_seed = self.seed + outer_fold * 1000 + inner_fold * 10
                seed_everything(training_seed)
                model_config = resolved_arm_model_config(self.config, arm_id)
                model = build_factorial_image_model(
                    arm_id,
                    model_config,
                    tabular_dim=len(tabular_columns),
                    hazard_bins=len(hazard_bin_edges) - 1,
                )
                output = (
                    self.output_root
                    / "development"
                    / self.branch
                    / arm_id
                    / f"outer_{outer_fold}"
                    / "inner"
                    / candidate_id
                    / f"fold_{inner_fold}"
                )
                contract = self._cell_contract(
                    arm_id,
                    outer_fold,
                    candidate_id,
                    stage="INNER_HPO",
                    inner_fold=inner_fold,
                    partition_contract={
                        "train_subjects_sha256": sha256_mapping(
                            {"subjects": sorted(train_keys)}
                        ),
                        "validation_subjects_sha256": sha256_mapping(
                            {"subjects": sorted(val_keys)}
                        ),
                        "train_count": len(train_keys),
                        "validation_count": len(val_keys),
                        "tabular_transformer": transformer.to_dict(),
                    },
                    training_parameters={
                        "learning_rate": float(candidate["learning_rate"]),
                        "weight_decay": float(candidate["weight_decay"]),
                        "max_epochs": int(self.config["model"]["max_epochs"]),
                        "patience": int(self.config["model"]["early_stopping_patience"]),
                    },
                )
                summary = fit_survival_model_v4(
                    model=model,
                    train_dataset=train_ds,
                    validation_dataset=val_ds,
                    bin_edges=hazard_bin_edges,
                    output_dir=output,
                    run_contract_digest=contract,
                    seed=training_seed,
                    max_epochs=int(self.config["model"]["max_epochs"]),
                    patience=int(self.config["model"]["early_stopping_patience"]),
                    learning_rate=float(candidate["learning_rate"]),
                    weight_decay=float(candidate["weight_decay"]),
                    batch_size=int(self.config["model"]["batch_size"]),
                    gradient_accumulation_steps=int(
                        self.config["model"]["gradient_accumulation_steps"]
                    ),
                    device=device,
                    amp=bool(self.config["model"]["amp"] and device.startswith("cuda")),
                    amp_dtype=str(self.config["model"].get("amp_dtype", "bfloat16")),
                    progress_context={
                        "branch": self.branch,
                        "arm_id": arm_id,
                        "outer_fold": outer_fold,
                        "inner_fold": inner_fold,
                        "candidate_id": candidate_id,
                    },
                )
                if summary.get("selection_metric_name") != "IPCW_INTEGRATED_BRIER_SCORE":
                    raise ImageFactorialOrchestratorV4Error(
                        "INNER_CHECKPOINT_WAS_NOT_SELECTED_BY_IPCW_IBS"
                    )
                locked = output / "locked_checkpoint.pt"
                model.load_state_dict(load_locked_state_for_eval(locked, contract), strict=True)
                model = model.to(device).eval()
                try:
                    import torch
                    from .image_training_runtime_v3 import _forward

                    val_ds.set_epoch(0)
                    loader = torch.utils.data.DataLoader(
                        val_ds,
                        batch_size=int(self.config["model"]["batch_size"]),
                        shuffle=False,
                        num_workers=0,
                    )
                    rows = []
                    device_object = torch.device(device)
                    amp_enabled = bool(
                        self.config["model"]["amp"] and device_object.type == "cuda"
                    )
                    amp_dtype = (
                        torch.bfloat16
                        if str(self.config["model"].get("amp_dtype", "bfloat16"))
                        == "bfloat16"
                        else torch.float16
                    )
                    with torch.no_grad():
                        for batch in loader:
                            with torch.autocast(
                                device_type=device_object.type,
                                dtype=amp_dtype,
                                enabled=amp_enabled,
                            ):
                                batch_logits = _forward(model, batch, device_object)
                            rows.append(batch_logits.detach().float().cpu())
                    logits = torch.cat(rows, dim=0)
                    verified_metric = _ipcw_validation_ibs(
                        validation_logits=logits,
                        train_event=train["event"].to_numpy(),
                        train_time_days=train["time_days"].to_numpy(),
                        validation_event=validation["event"].to_numpy(),
                        validation_time_days=validation["time_days"].to_numpy(),
                        bin_edges=hazard_bin_edges,
                    )
                    ibs = float(verified_metric["metric_value"])
                    if abs(ibs - float(summary["best_validation"])) > 1e-6:
                        raise ValueError("LOCKED_CHECKPOINT_IBS_MISMATCH")
                except Exception as exc:
                    raise ImageFactorialOrchestratorV4Error(
                        f"INNER_IBS_NOT_ESTIMABLE:{type(exc).__name__}:{exc}"
                    ) from exc
                scores.append(ibs)
                epochs.append(int(summary["best_epoch"]))
                selected_history = next(
                    row
                    for row in summary["history"]
                    if int(row["epoch"]) == int(summary["best_epoch"])
                )
                metric_records.append(
                    {
                        "inner_fold": inner_fold,
                        "metric_name": "IPCW_INTEGRATED_BRIER_SCORE",
                        "metric_value": ibs,
                        "evaluation_grid_days": verified_metric[
                            "evaluation_grid_days"
                        ],
                        "administrative_censoring_cutoff_days": verified_metric[
                            "administrative_censoring_cutoff_days"
                        ],
                        "censoring_distribution_source": "INNER_TRAIN_ONLY",
                        "checkpoint_selection_metric_name": summary[
                            "selection_metric_name"
                        ],
                        "train_validation_nll_gap_at_selected_epoch": float(
                            selected_history["train_validation_nll_gap"]
                        ),
                        "inner_train_count": int(len(train)),
                        "inner_validation_count": int(len(validation)),
                        "locked_checkpoint_sha256": _sha256_file(locked),
                    }
                )
            candidate_records.append(
                {
                    "candidate_id": candidate_id,
                    "mean_ibs": float(np.mean(scores)),
                    "standard_error": float(np.std(scores, ddof=1) / np.sqrt(len(scores)))
                    if len(scores) > 1
                    else 0.0,
                    "complexity_rank": int(candidate["complexity_rank"]),
                    "learning_rate": float(candidate["learning_rate"]),
                    "weight_decay": float(candidate["weight_decay"]),
                    "inner_fold_ibs": scores,
                    "inner_fold_metric_records": metric_records,
                    "inner_best_epochs": epochs,
                    "status": "COMPLETED",
                }
            )
        winner = select_inner_winner(candidate_records)
        write_winner_record(
            self.winner_path(arm_id, outer_fold),
            branch=self.branch,
            variant=arm_id,
            outer_fold=outer_fold,
            master_contract_sha256=self.master_contract_sha256,
            winner=winner,
            candidates=candidate_records,
        )
        return {"winner": winner, "candidates": candidate_records}

    def run_final_training(
        self,
        *,
        outer_fold: int,
        arm_id: str,
        outer_train_frame: pd.DataFrame,
        tabular_columns: list[str],
        hazard_bin_edges: list[float],
        winner: dict[str, Any],
        device: str,
    ) -> dict[str, Any]:
        epochs = int(max(1, round(float(np.median(winner.get("inner_best_epochs") or [1])))))
        transformer = FoldTabularTransformer.fit(outer_train_frame, tabular_columns)
        dataset = self._dataset(
            frame=outer_train_frame,
            tabular=transformer.transform(outer_train_frame),
            arm_id=arm_id,
            split_role="outer_train",
            outer_fold=outer_fold,
            inner_fold=None,
        )
        training_seed = self.seed + outer_fold * 1000
        seed_everything(training_seed)
        model_config = resolved_arm_model_config(self.config, arm_id)
        model = build_factorial_image_model(
            arm_id,
            model_config,
            tabular_dim=len(tabular_columns),
            hazard_bins=len(hazard_bin_edges) - 1,
        )
        output = (
            self.output_root
            / "development"
            / self.branch
            / arm_id
            / f"outer_{outer_fold}"
            / "final"
        )
        outer_subjects = sorted(outer_train_frame["patient_hash"].astype(str).tolist())
        contract = self._cell_contract(
            arm_id,
            outer_fold,
            str(winner.get("candidate_id", "final")),
            stage="FINAL_OUTER_TRAIN",
            inner_fold=None,
            partition_contract={
                "outer_train_subjects_sha256": sha256_mapping(
                    {"subjects": outer_subjects}
                ),
                "outer_train_count": len(outer_subjects),
                "tabular_transformer": transformer.to_dict(),
            },
            training_parameters={
                "learning_rate": float(winner["learning_rate"]),
                "weight_decay": float(winner["weight_decay"]),
                "fixed_epochs": epochs,
            },
        )
        fit_survival_model_v4(
            model=model,
            train_dataset=dataset,
            validation_dataset=None,
            bin_edges=hazard_bin_edges,
            output_dir=output,
            run_contract_digest=contract,
            seed=training_seed,
            max_epochs=epochs,
            patience=epochs + 1,
            learning_rate=float(winner["learning_rate"]),
            weight_decay=float(winner["weight_decay"]),
            batch_size=int(self.config["model"]["batch_size"]),
            gradient_accumulation_steps=int(
                self.config["model"]["gradient_accumulation_steps"]
            ),
            device=device,
            amp=bool(self.config["model"]["amp"] and device.startswith("cuda")),
            amp_dtype=str(self.config["model"].get("amp_dtype", "bfloat16")),
            fixed_epochs=epochs,
            progress_context={
                "branch": self.branch,
                "arm_id": arm_id,
                "outer_fold": outer_fold,
                "stage": "FINAL_OUTER_TRAIN",
            },
        )
        locked = output / "locked_checkpoint.pt"
        checkpoint_sha = _sha256_file(locked)
        controller = OuterFoldController(
            self.output_root / "private" / "outer_evaluation" / self.branch / arm_id,
            fold=outer_fold,
            run_contract_sha256=contract,
        )
        controller.lock_checkpoint(checkpoint_sha)
        return {
            "locked_checkpoint_path": str(locked),
            "locked_checkpoint_sha256": checkpoint_sha,
            "run_contract_sha256": contract,
            "resolved_model_config": model_config,
        }

    def outer_evaluate(
        self,
        *,
        outer_fold: int,
        arm_id: str,
        run_contract_sha256: str,
        evaluator: Callable[[], dict[str, Any]],
    ) -> dict[str, Any]:
        controller = OuterFoldController(
            self.output_root / "private" / "outer_evaluation" / self.branch / arm_id,
            fold=outer_fold,
            run_contract_sha256=run_contract_sha256,
        )
        metrics = controller.evaluate_once(evaluator)
        controller.finalize()
        private_path = self.private_root / arm_id / f"fold_{outer_fold}_private.json"
        private_path.parent.mkdir(parents=True, exist_ok=True)
        write_private_outer_record(
            private_path,
            branch=self.branch,
            variant=arm_id,
            outer_fold=outer_fold,
            run_contract_sha256=run_contract_sha256,
            metrics=metrics,
        )
        public_metrics = {
            key: value for key, value in metrics.items() if key in PUBLIC_OUTER_METRIC_KEYS
        }
        public_metrics["outer_test_augmentation_applied"] = False
        atomic_write_json(
            self.public_root / f"{self.branch}_{arm_id}_fold_{outer_fold}_public.json",
            {
                "schema_version": "4.0",
                "aggregate": public_metrics,
                "run_contract_sha256": run_contract_sha256,
                "claim_status": "POST_HOC_INTERNAL_EXPLORATORY",
            },
        )
        return metrics

    def finalize_branch(self, expected_folds: int = 5) -> None:
        for arm_id in self.arms:
            root = self.output_root / "private" / "outer_evaluation" / self.branch / arm_id
            for fold in range(int(expected_folds)):
                state_path = root / f"fold_{fold}" / "state.json"
                if not state_path.is_file():
                    raise ImageFactorialOrchestratorV4Error(
                        f"FAILED_INCOMPLETE_OUTER_CV:{arm_id}:fold_{fold}:STATE_MISSING"
                    )
                raw = read_json(state_path)
                contract = raw.get("run_contract_sha256")
                controller = OuterFoldController(root, fold=fold, run_contract_sha256=contract)
                state = controller._load_state()
                checkpoint = state.get("checkpoint_sha256")
                result = controller._load_result(checkpoint) if checkpoint else None
                if state.get("state") != "FINALIZED" or result is None:
                    raise ImageFactorialOrchestratorV4Error(
                        f"FAILED_INCOMPLETE_OUTER_CV:{arm_id}:fold_{fold}"
                    )
        atomic_write_json(
            self.state_root / "branch_complete.json",
            {
                "schema_version": "4.0",
                "status": "BRANCH_OUTER_CV_COMPLETE",
                "branch": self.branch,
                "arms": list(self.arms),
                "outer_folds": int(expected_folds),
                "outer_cells": len(self.arms) * int(expected_folds),
                "claim_status": "POST_HOC_INTERNAL_EXPLORATORY",
            },
        )
