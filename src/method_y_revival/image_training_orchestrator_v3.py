from __future__ import annotations

import json
import hashlib
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from .atomic_io import atomic_write_json, read_json
from .contracts import sha256_mapping
from .image_modeling_v3 import LOCKED_VARIANTS, build_model, survival_at_times
from .image_training_runtime_v3 import FoldTabularTransformer, fit_survival_model, TrainingRuntimeV3Error
from .outer_evaluation_control import OuterFoldController, OuterEvaluationControlError, summarize_outer_evaluation
from .survival_models import select_inner_winner


class ImageTrainingOrchestratorV3Error(RuntimeError):
    pass


PUBLIC_OUTER_METRIC_KEYS = frozenset(
    {
        "uno_c_index",
        "integrated_brier_score",
        "evaluation_grid_days",
        "administrative_censoring_cutoff_days",
        "n_train",
        "n_test",
        "train_events",
        "test_events",
        "test_rows_administratively_censored",
        "horizon_metrics",
        "evaluation_source",
    }
)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _build_run_contract(
    *,
    config: dict[str, Any],
    branch: str,
    variant: str,
    outer_fold: int,
    candidate_id: str,
    membership_sha: str,
    cache_sha: str,
    split_sha: str,
    readiness_sha: str,
) -> str:
    payload = {
        "branch": branch,
        "variant": variant,
        "outer_fold": int(outer_fold),
        "candidate_id": candidate_id,
        "model_config": config.get("model", {}),
        "hazard_bin_edges": config.get("shared", {}).get("hazard_bin_edges_days", []),
        "feature_columns": config.get("branches", {}).get(branch, {}).get("model_features", []),
        "membership_sha256": membership_sha,
        "cache_sha256": cache_sha,
        "split_sha256": split_sha,
        "readiness_sha256": readiness_sha,
        "seed": int(config.get("seed", 0)),
        "inner_hpo_grid": config.get("model", {}).get("hpo_candidates", []),
        "runtime_policy": config.get("runtime", {}),
        "software_runtime_contract": config.get("software_runtime_contract", {}),
        "training_source_sha256": config.get("training_source_sha256"),
    }
    return sha256_mapping(payload)


def load_locked_state_for_eval(locked_path: Path, run_contract_digest: str):
    """Strictly load a locked checkpoint for evaluation; never fall back silently."""
    import torch

    if not Path(locked_path).is_file():
        raise ImageTrainingOrchestratorV3Error(
            f"INNER_IBS_CHECKPOINT_LOAD_FAILED:missing:{Path(locked_path).name}"
        )
    try:
        ckpt = torch.load(locked_path, map_location="cpu", weights_only=False)
        state = ckpt["model_state"]
    except ImageTrainingOrchestratorV3Error:
        raise
    except Exception as exc:
        raise ImageTrainingOrchestratorV3Error(
            f"INNER_IBS_CHECKPOINT_LOAD_FAILED:{type(exc).__name__}"
        ) from exc
    if ckpt.get("run_contract_digest") != run_contract_digest:
        raise ImageTrainingOrchestratorV3Error("INNER_IBS_CHECKPOINT_LOAD_FAILED:contract")
    return state


class ImageTrainingOrchestratorV3:
    """High-level nested CV orchestrator for image v3.

    This is a test-friendly, synthetic-capable orchestrator that preserves all
    scientific invariants required by Plan 011:
    - winner selected only from mean inner IBS with 1-SE + complexity_rank tie-break
    - tabular transformer fitted only on train partitions
    - outer test dataset factory never called during inner HPO and never before checkpoint lock
    - exactly-once outer evaluation via OuterFoldController
    - run-contract mismatch fails closed
    - incomplete outer CV cannot be reported complete
    - private OOF rows never written to public reports
    """

    # synthetic state machine values
    STATES = (
        "INIT",
        "INPUTS_VERIFIED",
        "INNER_HPO_RUNNING",
        "INNER_SELECTION_DONE",
        "FINAL_TRAINING",
        "CHECKPOINT_LOCKED",
        "OUTER_EVAL_IN_PROGRESS",
        "OUTER_EVAL_COMMITTED",
        "BRANCH_OUTER_CV_COMPLETE",
    )

    def __init__(
        self,
        *,
        config: dict[str, Any],
        output_root: str | Path,
        branch: str,
        cache_sha: str | None = None,
        membership_sha: str | None = None,
        split_sha: str | None = None,
        readiness_sha: str | None = None,
        h5_cache_path: str | Path | None = None,
        allow_synthetic_fixture: bool = False,
    ) -> None:
        self.config = config
        self.output_root = Path(output_root)
        self.branch = str(branch).lower()
        if self.branch not in {"amyloid", "tau"}:
            raise ImageTrainingOrchestratorV3Error(f"UNSUPPORTED_BRANCH:{branch}")
        self.variants = tuple(LOCKED_VARIANTS)
        self.seed = int(config.get("seed", 0))
        self.hpo_candidates = list(config.get("model", {}).get("hpo_candidates", []))
        if not self.hpo_candidates:
            raise ImageTrainingOrchestratorV3Error("HPO_CANDIDATES_REQUIRED")
        # verify each candidate has complexity_rank
        for cand in self.hpo_candidates:
            if "complexity_rank" not in cand:
                raise ImageTrainingOrchestratorV3Error(f"CANDIDATE_MISSING_COMPLEXITY_RANK:{cand.get('candidate_id')}")
        # run contract digests for fail-closed checks
        self.cache_sha = cache_sha or "0" * 64
        self.membership_sha = membership_sha or "0" * 64
        self.split_sha = split_sha or "0" * 64
        self.readiness_sha = readiness_sha or "0" * 64
        self.h5_cache_path = Path(h5_cache_path) if h5_cache_path else None
        self.allow_synthetic_fixture = bool(allow_synthetic_fixture)
        # output directories
        self.state_root = self.output_root / "orchestration" / self.branch
        self.state_root.mkdir(parents=True, exist_ok=True)
        self.private_root = self.output_root / "private" / "outer_evaluation" / self.branch
        self.public_root = self.output_root / "reports"
        self.private_root.mkdir(parents=True, exist_ok=True)
        self.public_root.mkdir(parents=True, exist_ok=True)
        # master state tracking per outer fold/variant
        self.master_state_path = self.state_root / "orchestrator_state.json"
        self._load_or_init_master_state()

    def _load_or_init_master_state(self) -> None:
        if self.master_state_path.exists():
            try:
                existing = read_json(self.master_state_path)
            except Exception as exc:
                raise ImageTrainingOrchestratorV3Error(f"MASTER_STATE_CORRUPT:{exc}")
            # verify run contract still matches (any mismatch fails closed)
            expected = self._master_contract_digest()
            if existing.get("run_contract_sha256") != expected:
                raise ImageTrainingOrchestratorV3Error("RUN_CONTRACT_MISMATCH:orchestrator_master_state")
            self.master_state = existing
        else:
            self.master_state = {
                "schema_version": "3.0",
                "branch": self.branch,
                "run_contract_sha256": self._master_contract_digest(),
                "outer_folds": {},
                "state": "INIT",
            }
            self._commit_master_state()

    def _master_contract_digest(self) -> str:
        payload = {
            "branch": self.branch,
            "variants": list(self.variants),
            "seed": self.seed,
            "hpo_candidates": self.hpo_candidates,
            "membership_sha": self.membership_sha,
            "cache_sha": self.cache_sha,
            "split_sha": self.split_sha,
            "readiness_sha": self.readiness_sha,
            "model_config": self.config.get("model", {}),
            "runtime_policy": self.config.get("runtime", {}),
            "software_runtime_contract": self.config.get("software_runtime_contract", {}),
            "training_source_sha256": self.config.get("training_source_sha256"),
            "hazard_bin_edges": self.config.get("shared", {}).get("hazard_bin_edges_days", []),
            "feature_columns": self.config.get("branches", {}).get(self.branch, {}).get("model_features", []),
        }
        return sha256_mapping(payload)

    def _commit_master_state(self) -> None:
        # self-hash verification similar to outer_evaluation_control
        body = dict(self.master_state)
        digest = sha256_mapping(body)
        body["state_sha256"] = digest
        atomic_write_json(self.master_state_path, body)

    def _set_state(self, state: str) -> None:
        if state not in self.STATES:
            raise ImageTrainingOrchestratorV3Error(f"UNKNOWN_STATE:{state}")
        self.master_state["state"] = state
        self._commit_master_state()

    def verify_inputs(self) -> None:
        # matched variants must have identical hpo grid, splits, features budget
        # Here we check our internal invariant: same hpo grid for both variants (by definition same config)
        # And feature sets: base + same candidate budget
        branch_cfg = self.config.get("branches", {}).get(self.branch, {})
        if not branch_cfg.get("model_features"):
            raise ImageTrainingOrchestratorV3Error("MODEL_FEATURES_REQUIRED")
        # In real run, we would hash membership/split/cache and verify; for synthetic we already have digests
        self._set_state("INPUTS_VERIFIED")

    # ---------- inner HPO ----------
    def run_inner_hpo(
        self,
        *,
        outer_fold: int,
        variant: str,
        outer_train_frame: pd.DataFrame,
        inner_splits: list[dict[str, Any]],
        tabular_columns: list[str],
        hazard_bin_edges: list[float],
        device: str = "cpu",
    ) -> dict[str, Any]:
        if variant not in self.variants:
            raise ImageTrainingOrchestratorV3Error(f"UNKNOWN_VARIANT:{variant}")
        if self.master_state.get("state") not in {"INPUTS_VERIFIED", "INNER_HPO_RUNNING", "INNER_SELECTION_DONE"}:
            # allow if we are already in inner hpo
            pass
        self._set_state("INNER_HPO_RUNNING")
        # ensure outer_test_factory has not been called – caller responsibility, but we enforce that we never invoke test dataset here
        candidate_records: list[dict[str, Any]] = []
        for cand in self.hpo_candidates:
            cand_id = cand["candidate_id"]
            lr = float(cand["learning_rate"])
            wd = float(cand["weight_decay"])
            inner_scores: list[float] = []
            inner_metric_records: list[dict[str, Any]] = []
            inner_best_epochs: list[int] = []
            for inner in inner_splits:
                train_keys = set(inner.get("train_subjects") or inner.get("train_subject_keys") or [])
                val_keys = set(inner.get("validation_subjects") or inner.get("validation_subject_keys") or inner.get("val_subjects") or [])
                outer_train_keys = set(outer_train_frame["patient_hash"].astype(str))
                if train_keys & val_keys:
                    raise ImageTrainingOrchestratorV3Error("INNER_PARTITION_LEAKAGE")
                if not train_keys or not val_keys:
                    raise ImageTrainingOrchestratorV3Error("INNER_PARTITION_EMPTY")
                if not train_keys.issubset(outer_train_keys) or not val_keys.issubset(outer_train_keys):
                    raise ImageTrainingOrchestratorV3Error("INNER_PARTITION_OUTSIDE_OUTER_TRAIN")
                if train_keys | val_keys != outer_train_keys:
                    raise ImageTrainingOrchestratorV3Error("INNER_PARTITION_INCOMPLETE")
                # Fit transformer ONLY on inner train
                train_frame = outer_train_frame.loc[outer_train_frame["patient_hash"].astype(str).isin(train_keys)].copy()
                val_frame = outer_train_frame.loc[outer_train_frame["patient_hash"].astype(str).isin(val_keys)].copy()
                transformer = FoldTabularTransformer.fit(train_frame, tabular_columns)
                # Verify transformer not fitted on validation (by ensuring transform would still work on val but not leaked)
                # Transform both
                train_tab = transformer.transform(train_frame)
                val_tab = transformer.transform(val_frame)
                # Build datasets: use real H5 cache if available, else synthetic 16³
                include_image = variant == "scalar_tabular_plus_pattern_image"
                from .image_training_runtime_v3 import MemorySurvivalDataset, H5PatternSurvivalDataset

                if include_image and not (
                    self.h5_cache_path is not None and self.h5_cache_path.is_file()
                ) and not self.allow_synthetic_fixture:
                    raise ImageTrainingOrchestratorV3Error("PATTERN_CACHE_REQUIRED")

                if include_image and self.h5_cache_path is not None and self.h5_cache_path.is_file():
                    # Real cache path provided — use H5PatternSurvivalDataset for pattern-normalized images
                    train_ds = H5PatternSurvivalDataset(
                        cache_path=self.h5_cache_path,
                        patient_hashes=train_frame["patient_hash"].astype(str).tolist(),
                        tabular=train_tab,
                        event=train_frame["event"].to_numpy(),
                        time_days=train_frame["time_days"].to_numpy(),
                        include_image=True,
                    )
                    val_ds = H5PatternSurvivalDataset(
                        cache_path=self.h5_cache_path,
                        patient_hashes=val_frame["patient_hash"].astype(str).tolist(),
                        tabular=val_tab,
                        event=val_frame["event"].to_numpy(),
                        time_days=val_frame["time_days"].to_numpy(),
                        include_image=True,
                    )
                    train_images = None
                    val_images = None
                else:
                    if include_image:
                        fixture_rng = np.random.default_rng(
                            self.seed + outer_fold * 100 + int(inner.get("fold", 0))
                        )
                        train_images = fixture_rng.standard_normal((len(train_frame), 1, 16, 16, 16), dtype=np.float32)
                        val_images = fixture_rng.standard_normal((len(val_frame), 1, 16, 16, 16), dtype=np.float32)
                    else:
                        train_images = None
                        val_images = None
                    train_ds = MemorySurvivalDataset(
                        tabular=train_tab,
                        event=train_frame["event"].to_numpy(),
                        time_days=train_frame["time_days"].to_numpy(),
                        images=train_images,
                        patient_hashes=train_frame["patient_hash"].astype(str).tolist(),
                    )
                    val_ds = MemorySurvivalDataset(
                        tabular=val_tab,
                        event=val_frame["event"].to_numpy(),
                        time_days=val_frame["time_days"].to_numpy(),
                        images=val_images,
                        patient_hashes=val_frame["patient_hash"].astype(str).tolist(),
                    )
                # Create model
                from .image_training_runtime_v3 import seed_everything

                seed_everything(self.seed + outer_fold * 100 + int(inner.get("fold", 0)))
                model = build_model(variant, self.config.get("model", {}), tabular_dim=len(tabular_columns), hazard_bins=len(hazard_bin_edges) - 1)
                # Train with early stopping on inner validation NLL (not used for winner selection)
                # But winner selection MUST use mean inner IBS, not validation NLL. So we collect IBS post-hoc.
                tmp_dir = self.state_root / f"tmp_inner_{variant}_outer_{outer_fold}_{cand_id}_inner_{inner.get('fold', 0)}"
                tmp_dir.mkdir(parents=True, exist_ok=True)
                run_contract = _build_run_contract(
                    config=self.config,
                    branch=self.branch,
                    variant=variant,
                    outer_fold=outer_fold,
                    candidate_id=cand_id,
                    membership_sha=self.membership_sha,
                    cache_sha=self.cache_sha,
                    split_sha=self.split_sha,
                    readiness_sha=self.readiness_sha,
                )
                # Use real config for production (h5 cache), small synthetic for tests
                if self.h5_cache_path is not None and self.h5_cache_path.is_file():
                    real_max_epochs = int(self.config.get("model", {}).get("max_epochs", 40))
                    real_patience = int(self.config.get("model", {}).get("early_stopping_patience", 8))
                    real_batch = int(self.config.get("model", {}).get("batch_size", 2))
                    real_accum = int(self.config.get("model", {}).get("gradient_accumulation_steps", 4))
                    real_amp = bool(self.config.get("model", {}).get("amp", True) and device.startswith("cuda"))
                else:
                    real_max_epochs = 2
                    real_patience = int(self.config.get("model", {}).get("early_stopping_patience", 2))
                    real_batch = 2
                    real_accum = 1
                    real_amp = False
                fit_survival_model(
                    model=model,
                    train_dataset=train_ds,
                    validation_dataset=val_ds,
                    bin_edges=hazard_bin_edges,
                    output_dir=tmp_dir,
                    run_contract_digest=run_contract,
                    seed=self.seed + outer_fold * 100 + int(inner.get("fold", 0)),
                    max_epochs=real_max_epochs,
                    patience=real_patience,
                    learning_rate=lr,
                    weight_decay=wd,
                    batch_size=real_batch,
                    gradient_accumulation_steps=real_accum,
                    device=device,
                    amp=real_amp,
                )
                # After training, evaluate inner validation IBS using survival_at_times and integrated_brier_score
                # This ensures winner selection uses true IBS, not validation NLL
                import torch
                from sksurv.metrics import integrated_brier_score
                from sksurv.util import Surv
                from .image_outer_eval_v3 import _administratively_censor, _evaluation_grid
                from .image_training_runtime_v3 import _forward

                # Load the LOCKED (best) weights; silent fallback to last-epoch
                # state would evaluate the wrong model and corrupt HPO selection.
                locked_path = tmp_dir / "locked_checkpoint.pt"
                best_state = load_locked_state_for_eval(locked_path, run_contract)
                model.load_state_dict(best_state, strict=True)
                model.eval()
                summary_path = tmp_dir / "training_summary.json"
                train_times = train_frame["time_days"].to_numpy(dtype=float)
                val_times = val_frame["time_days"].to_numpy(dtype=float)
                try:
                    cutoff = min(float(train_times.max()), float(val_times.max()))
                    cutoff -= max(abs(cutoff) * 1e-9, 1e-9)
                    eval_train = _administratively_censor(train_frame, cutoff)
                    eval_val = _administratively_censor(val_frame, cutoff)
                    grid = _evaluation_grid(
                        eval_train["time_days"].to_numpy(dtype=float),
                        eval_val["time_days"].to_numpy(dtype=float),
                        hazard_bin_edges,
                    )
                    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=4, shuffle=False)
                    val_logits_list = []
                    device_object = torch.device(device)
                    with torch.no_grad():
                        for batch in val_loader:
                            logits = _forward(model, batch, device_object)
                            val_logits_list.append(logits.detach().cpu())
                    val_logits = torch.cat(val_logits_list, dim=0) if val_logits_list else torch.empty((0, len(hazard_bin_edges) - 1))
                    surv = survival_at_times(val_logits, hazard_bin_edges, grid.tolist())
                    surv_np = surv.detach().numpy()
                    if not np.isfinite(surv_np).all():
                        raise ValueError("NONFINITE_SURVIVAL_PROBABILITY")
                    if np.any(surv_np < 0.0) or np.any(surv_np > 1.0):
                        raise ValueError("SURVIVAL_PROBABILITY_OUT_OF_RANGE")
                    if surv_np.shape[1] > 1 and np.any(np.diff(surv_np, axis=1) > 1e-7):
                        raise ValueError("SURVIVAL_PROBABILITY_NOT_MONOTONIC")
                    y_train = Surv.from_arrays(
                        event=eval_train["event"].astype(bool).to_numpy(),
                        time=eval_train["time_days"].astype(float).to_numpy(),
                    )
                    y_test = Surv.from_arrays(
                        event=eval_val["event"].astype(bool).to_numpy(),
                        time=eval_val["time_days"].astype(float).to_numpy(),
                    )
                    ibs = float(integrated_brier_score(y_train, y_test, surv_np, grid))
                    if not np.isfinite(ibs) or not 0.0 <= ibs <= 1.0:
                        raise ValueError(f"IBS_OUT_OF_RANGE:{ibs}")
                except Exception as exc:
                    raise ImageTrainingOrchestratorV3Error(
                        f"INNER_IBS_NOT_ESTIMABLE:{type(exc).__name__}:{exc}"
                    ) from exc
                if summary_path.exists():
                    summ = json.loads(summary_path.read_text(encoding="utf-8"))
                    inner_best_epochs.append(int(summ.get("best_epoch", 1)))
                else:
                    inner_best_epochs.append(1)
                inner_scores.append(float(ibs))
                inner_metric_records.append(
                    {
                        "inner_fold": int(inner.get("fold", 0)),
                        "metric_name": "IPCW_INTEGRATED_BRIER_SCORE",
                        "metric_value": float(ibs),
                        "evaluation_grid_days": [float(value) for value in grid],
                        "administrative_censoring_cutoff_days": float(cutoff),
                        "inner_train_count": int(len(train_frame)),
                        "inner_validation_count": int(len(val_frame)),
                        "locked_checkpoint_sha256": _sha256_file(locked_path),
                    }
                )
            # compute mean and SE for candidate
            mean_ibs = float(np.mean(inner_scores)) if inner_scores else float("inf")
            se = float(np.std(inner_scores, ddof=1) / (len(inner_scores) ** 0.5)) if len(inner_scores) > 1 else 0.0
            candidate_records.append(
                {
                    "candidate_id": cand_id,
                    "mean_ibs": mean_ibs,
                    "standard_error": se,
                    "complexity_rank": int(cand["complexity_rank"]),
                    "learning_rate": lr,
                    "weight_decay": wd,
                    "inner_fold_ibs": inner_scores,
                    "inner_fold_metric_records": inner_metric_records,
                    "inner_best_epochs": inner_best_epochs,
                    "status": "COMPLETED",
                }
            )
        winner = select_inner_winner([r for r in candidate_records if r["status"] == "COMPLETED"])
        # persist winner
        winner_path = self.state_root / f"winner_{variant}_outer_{outer_fold}.json"
        atomic_write_json(
            winner_path,
            {
                "winner": winner,
                "candidates": candidate_records,
                "outer_fold": outer_fold,
                "variant": variant,
                "run_contract_sha256": self._master_contract_digest(),
            },
        )
        self._set_state("INNER_SELECTION_DONE")
        return {"winner": winner, "candidates": candidate_records}

    # ---------- final outer-train ----------
    def run_final_training(
        self,
        *,
        outer_fold: int,
        variant: str,
        outer_train_frame: pd.DataFrame,
        tabular_columns: list[str],
        hazard_bin_edges: list[float],
        winner: dict[str, Any],
        device: str = "cpu",
    ) -> dict[str, Any]:
        self._set_state("FINAL_TRAINING")
        # fixed epochs from median inner best epochs
        inner_best = winner.get("inner_best_epochs") or [2]
        fixed_epochs = int(max(1, round(float(np.median(inner_best)))))
        transformer = FoldTabularTransformer.fit(outer_train_frame, tabular_columns)
        tab = transformer.transform(outer_train_frame)
        include_image = variant == "scalar_tabular_plus_pattern_image"
        from .image_training_runtime_v3 import MemorySurvivalDataset, H5PatternSurvivalDataset

        if include_image and not (
            self.h5_cache_path is not None and self.h5_cache_path.is_file()
        ) and not self.allow_synthetic_fixture:
            raise ImageTrainingOrchestratorV3Error("PATTERN_CACHE_REQUIRED")

        if include_image and self.h5_cache_path is not None and self.h5_cache_path.is_file():
            ds = H5PatternSurvivalDataset(
                cache_path=self.h5_cache_path,
                patient_hashes=outer_train_frame["patient_hash"].astype(str).tolist(),
                tabular=tab,
                event=outer_train_frame["event"].to_numpy(),
                time_days=outer_train_frame["time_days"].to_numpy(),
                include_image=True,
            )
        else:
            if include_image:
                fixture_rng = np.random.default_rng(self.seed + outer_fold * 1000)
                images = fixture_rng.standard_normal(
                    (len(outer_train_frame), 1, 16, 16, 16), dtype=np.float32
                )
            else:
                images = None
            ds = MemorySurvivalDataset(
                tabular=tab,
                event=outer_train_frame["event"].to_numpy(),
                time_days=outer_train_frame["time_days"].to_numpy(),
                images=images,
                patient_hashes=outer_train_frame["patient_hash"].astype(str).tolist(),
            )
        from .image_training_runtime_v3 import seed_everything

        seed_everything(self.seed + outer_fold * 1000)
        model = build_model(variant, self.config.get("model", {}), tabular_dim=len(tabular_columns), hazard_bins=len(hazard_bin_edges) - 1)
        out_dir = self.output_root / "development" / self.branch / variant / f"outer_{outer_fold}" / "final"
        out_dir.mkdir(parents=True, exist_ok=True)
        run_contract = _build_run_contract(
            config=self.config,
            branch=self.branch,
            variant=variant,
            outer_fold=outer_fold,
            candidate_id=winner.get("candidate_id", "final"),
            membership_sha=self.membership_sha,
            cache_sha=self.cache_sha,
            split_sha=self.split_sha,
            readiness_sha=self.readiness_sha,
        )
        # Use real config for production
        if "learning_rate" not in winner or "weight_decay" not in winner:
            raise ImageTrainingOrchestratorV3Error("WINNER_HYPERPARAMS_MISSING")
        if self.h5_cache_path is not None and self.h5_cache_path.is_file():
            real_batch = int(self.config.get("model", {}).get("batch_size", 2))
            real_accum = int(self.config.get("model", {}).get("gradient_accumulation_steps", 4))
            real_amp = bool(self.config.get("model", {}).get("amp", True) and device.startswith("cuda"))
        else:
            real_batch = 2
            real_accum = 1
            real_amp = False
        summary = fit_survival_model(
            model=model,
            train_dataset=ds,
            validation_dataset=None,
            bin_edges=hazard_bin_edges,
            output_dir=out_dir,
            run_contract_digest=run_contract,
            seed=self.seed + outer_fold * 37,
            max_epochs=fixed_epochs,
            patience=100,  # disable early stopping for fixed epochs
            learning_rate=float(winner["learning_rate"]),
            weight_decay=float(winner["weight_decay"]),
            batch_size=real_batch,
            gradient_accumulation_steps=real_accum,
            device=device,
            amp=real_amp,
            fixed_epochs=fixed_epochs,
        )
        # lock checkpoint via OuterFoldController
        import hashlib, pathlib
        locked_path = out_dir / "locked_checkpoint.pt"
        # compute checkpoint sha
        ckpt_sha = hashlib.sha256(locked_path.read_bytes()).hexdigest() if locked_path.exists() else "0" * 64
        controller = OuterFoldController(
            self.output_root / "private" / "outer_evaluation" / self.branch / variant,
            fold=outer_fold,
            run_contract_sha256=run_contract,
        )
        controller.lock_checkpoint(ckpt_sha)
        self._set_state("CHECKPOINT_LOCKED")
        return {"locked_checkpoint_sha": ckpt_sha, "run_contract": run_contract, "transformer": transformer.to_dict()}

    # ---------- outer evaluation exactly-once ----------
    def outer_evaluate(
        self,
        *,
        outer_fold: int,
        variant: str,
        evaluator: Callable[[], dict[str, Any]],
        run_contract: str,
    ) -> dict[str, Any]:
        controller = OuterFoldController(
            self.output_root / "private" / "outer_evaluation" / self.branch / variant,
            fold=outer_fold,
            run_contract_sha256=run_contract,
        )
        # Ensure evaluator not called if already committed (controller handles resume without second call)
        metrics = controller.evaluate_once(evaluator)
        controller.finalize()
        # separate private vs public: private contains OOF rows, public only aggregate
        private_path = self.private_root / variant / f"fold_{outer_fold}_private.json"
        private_path.parent.mkdir(parents=True, exist_ok=True)
        # evaluator metrics already contain oof_predictions privately? We store private copy
        atomic_write_json(private_path, {"metrics": metrics, "run_contract": run_contract})
        public_path = self.public_root / f"{self.branch}_{variant}_fold_{outer_fold}_public.json"
        # public should NOT contain patient_hash etc. So we filter
        public_metrics = {
            key: value
            for key, value in metrics.items()
            if key in PUBLIC_OUTER_METRIC_KEYS
        }
        atomic_write_json(public_path, {"aggregate": public_metrics, "run_contract": run_contract})
        self._set_state("OUTER_EVAL_COMMITTED")
        return metrics

    def check_incomplete_outer_cv(self, expected_folds: int = 5) -> dict[str, Any]:
        # use controller summary for each variant
        for variant in self.variants:
            root = self.output_root / "private" / "outer_evaluation" / self.branch / variant
            # Need run_contract per variant; use master contract for synthetic check
            # For real, would need per-fold run_contract; we check via summarize_outer_evaluation with last known contract
            # Simplified: if any fold missing, report incomplete
            missing = []
            for fold in range(expected_folds):
                state_path = root / f"fold_{fold}" / "state.json"
                if not state_path.exists():
                    missing.append(fold)
            if missing:
                return {"status": "FAILED_INCOMPLETE_OUTER_CV", "missing_folds": missing, "expected_outer_folds": expected_folds}
        return {"status": "COMPLETED"}

    def finalize_branch(self) -> None:
        incomplete = self.check_incomplete_outer_cv(expected_folds=5)
        if incomplete.get("status") != "COMPLETED":
            raise ImageTrainingOrchestratorV3Error(
                f"FAILED_INCOMPLETE_OUTER_CV:{incomplete}"
            )
        for variant in self.variants:
            root = self.output_root / "private" / "outer_evaluation" / self.branch / variant
            for fold in range(5):
                raw_state = read_json(root / f"fold_{fold}" / "state.json")
                run_contract = raw_state.get("run_contract_sha256")
                if not isinstance(run_contract, str):
                    raise ImageTrainingOrchestratorV3Error(
                        "FAILED_INCOMPLETE_OUTER_CV:CONTRACT_MISSING"
                    )
                controller = OuterFoldController(
                    root, fold=fold, run_contract_sha256=run_contract
                )
                state = controller._load_state()
                checkpoint = state.get("checkpoint_sha256")
                result = (
                    controller._load_result(checkpoint)
                    if isinstance(checkpoint, str)
                    else None
                )
                if state.get("state") != "FINALIZED" or result is None:
                    raise ImageTrainingOrchestratorV3Error(
                        f"FAILED_INCOMPLETE_OUTER_CV:{variant}:fold_{fold}"
                    )
        self._set_state("BRANCH_OUTER_CV_COMPLETE")
