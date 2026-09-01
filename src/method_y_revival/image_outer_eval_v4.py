from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .image_factorial_v4 import build_factorial_image_model
from .image_modeling_v3 import survival_at_times
from .image_outer_eval_v3 import _administratively_censor, _evaluation_grid


class ImageOuterEvalV4Error(RuntimeError):
    pass


def evaluate_locked_outer_v4(
    *,
    locked_checkpoint_path: str | Path,
    run_contract_digest: str,
    train_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    h5_cache_path: str | Path,
    tabular_columns: list[str],
    arm_id: str,
    hazard_bin_edges: list[float],
    model_config: dict[str, Any],
    device: str = "cpu",
    amp: bool = False,
    batch_size: int = 4,
    requested_horizons_days: list[float] | None = None,
    minimum_at_risk: int = 20,
) -> dict[str, Any]:
    """Evaluate a locked v4 model once; test images are never augmented."""

    import torch
    from sksurv.metrics import (
        concordance_index_ipcw,
        cumulative_dynamic_auc,
        integrated_brier_score,
    )
    from sksurv.util import Surv

    from .image_training_runtime_v3 import FoldTabularTransformer, H5PatternSurvivalDataset

    checkpoint_path = Path(locked_checkpoint_path)
    cache_path = Path(h5_cache_path)
    if not checkpoint_path.is_file():
        raise ImageOuterEvalV4Error("LOCKED_CHECKPOINT_MISSING")
    if not cache_path.is_file():
        raise ImageOuterEvalV4Error("PATTERN_CACHE_REQUIRED")
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if payload.get("run_contract_digest") != run_contract_digest:
        raise ImageOuterEvalV4Error("LOCKED_CHECKPOINT_CONTRACT_MISMATCH")
    if test_frame.empty:
        raise ImageOuterEvalV4Error("EMPTY_OUTER_TEST_FRAME")
    required = {"patient_hash", "event", "time_days", *tabular_columns}
    missing = sorted(required.difference(test_frame.columns))
    if missing:
        raise ImageOuterEvalV4Error(f"OUTER_TEST_COLUMNS_MISSING:{missing}")

    transformer = FoldTabularTransformer.fit(train_frame, tabular_columns)
    tabular = transformer.transform(test_frame)
    dataset = H5PatternSurvivalDataset(
        cache_path=cache_path,
        patient_hashes=test_frame["patient_hash"].astype(str).tolist(),
        tabular=tabular,
        event=test_frame["event"].to_numpy(),
        time_days=test_frame["time_days"].to_numpy(),
        include_image=True,
    )
    model = build_factorial_image_model(
        arm_id,
        model_config,
        tabular_dim=len(tabular_columns),
        hazard_bins=len(hazard_bin_edges) - 1,
    )
    state = payload.get("model_state")
    if state is None:
        raise ImageOuterEvalV4Error("LOCKED_CHECKPOINT_MODEL_STATE_MISSING")
    model.load_state_dict(state, strict=True)
    device_object = torch.device(device)
    model = model.to(device_object).eval()
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=int(batch_size), shuffle=False, num_workers=0
    )
    logits_rows: list[torch.Tensor] = []
    embedding_rows: list[torch.Tensor] = []
    with torch.no_grad():
        for batch in loader:
            with torch.autocast(
                device_type=device_object.type,
                dtype=torch.bfloat16 if str(model_config.get("amp_dtype")) == "bfloat16" else torch.float16,
                enabled=bool(amp and device_object.type == "cuda"),
            ):
                logits, embedding = model(
                    batch["image"].to(device_object, non_blocking=True),
                    batch["tabular"].to(device_object, non_blocking=True),
                    return_embedding=True,
                )
            logits_rows.append(logits.detach().float().cpu())
            embedding_rows.append(embedding.detach().float().cpu())
    if not logits_rows:
        raise ImageOuterEvalV4Error("NO_LOGITS_FROM_LOCKED_MODEL")
    logits = torch.cat(logits_rows, dim=0)
    embeddings = torch.cat(embedding_rows, dim=0).numpy()
    grid = _evaluation_grid(
        train_frame["time_days"].to_numpy(dtype=float),
        test_frame["time_days"].to_numpy(dtype=float),
        hazard_bin_edges,
    )
    survival = survival_at_times(logits, hazard_bin_edges, grid.tolist()).numpy()
    if not np.isfinite(survival).all() or np.any(np.diff(survival, axis=1) > 1e-7):
        raise ImageOuterEvalV4Error("SURVIVAL_PROBABILITY_INVALID")
    survival = np.clip(survival, 0.0, 1.0)
    risk = 1.0 - survival[:, -1]
    cutoff = min(
        float(train_frame["time_days"].max()), float(test_frame["time_days"].max())
    )
    cutoff -= max(abs(cutoff) * 1e-9, 1e-9)
    eval_train = _administratively_censor(train_frame, cutoff)
    eval_test = _administratively_censor(test_frame, cutoff)
    y_train = Surv.from_arrays(
        event=eval_train["event"].astype(bool).to_numpy(),
        time=eval_train["time_days"].astype(float).to_numpy(),
    )
    y_test = Surv.from_arrays(
        event=eval_test["event"].astype(bool).to_numpy(),
        time=eval_test["time_days"].astype(float).to_numpy(),
    )
    ibs = float(integrated_brier_score(y_train, y_test, survival, grid))
    uno = float(
        concordance_index_ipcw(y_train, y_test, risk, tau=float(grid[-1]))[0]
    )
    if not np.isfinite(ibs) or not 0.0 <= ibs <= 1.0:
        raise ImageOuterEvalV4Error(f"OUTER_IBS_NOT_ESTIMABLE:{ibs}")
    if not np.isfinite(uno):
        raise ImageOuterEvalV4Error("OUTER_UNO_NOT_ESTIMABLE")

    support_lower = max(
        float(eval_train["time_days"].min()), float(eval_test["time_days"].min())
    )
    support_upper = min(
        float(eval_train["time_days"].max()), float(eval_test["time_days"].max())
    )
    horizon_metrics: dict[str, dict[str, Any]] = {}
    for requested in requested_horizons_days or []:
        horizon = float(requested)
        at_risk = int(np.sum(eval_test["time_days"].to_numpy(dtype=float) >= horizon))
        if not support_lower < horizon < support_upper or at_risk < int(minimum_at_risk):
            horizon_metrics[str(horizon)] = {
                "status": "NOT_ESTIMABLE",
                "at_risk": at_risk,
                "reason": "OUTSIDE_COMMON_SUPPORT_OR_INSUFFICIENT_AT_RISK",
            }
            continue
        try:
            horizon_survival = survival_at_times(
                logits, hazard_bin_edges, [horizon]
            ).numpy()[:, 0]
            auc_values, _ = cumulative_dynamic_auc(
                y_train, y_test, 1.0 - horizon_survival, np.asarray([horizon])
            )
            auc = float(np.asarray(auc_values).reshape(-1)[0])
            if not np.isfinite(auc):
                raise ValueError("NONFINITE_DYNAMIC_AUC")
            horizon_metrics[str(horizon)] = {
                "status": "ESTIMABLE",
                "dynamic_auc": auc,
                "at_risk": at_risk,
            }
        except Exception as exc:
            horizon_metrics[str(horizon)] = {
                "status": "NOT_ESTIMABLE",
                "at_risk": at_risk,
                "reason": f"DYNAMIC_AUC_ERROR:{type(exc).__name__}",
            }

    hashes = test_frame["patient_hash"].astype(str).tolist()
    return {
        "uno_c_index": uno,
        "integrated_brier_score": ibs,
        "evaluation_grid_days": [float(value) for value in grid],
        "administrative_censoring_cutoff_days": float(cutoff),
        "n_train": int(len(train_frame)),
        "n_test": int(len(test_frame)),
        "train_events": int(train_frame["event"].sum()),
        "test_events": int(test_frame["event"].sum()),
        "horizon_metrics": horizon_metrics,
        "oof_predictions": [
            {
                "patient_hash": hashes[index],
                "event": int(eval_test["event"].iloc[index]),
                "time_days": float(eval_test["time_days"].iloc[index]),
                "original_event": int(test_frame["event"].iloc[index]),
                "original_time_days": float(test_frame["time_days"].iloc[index]),
                "risk_score": float(risk[index]),
                "survival_probabilities": [float(value) for value in survival[index]],
            }
            for index in range(len(test_frame))
        ],
        "oof_embeddings": [
            {
                "patient_hash": hashes[index],
                "embedding": [float(value) for value in embeddings[index]],
            }
            for index in range(len(hashes))
        ],
        "censoring_reference": [
            {"event": int(row.event), "time_days": float(row.time_days)}
            for row in eval_train[["event", "time_days"]].itertuples(index=False)
        ],
        "evaluation_source": "LOCKED_V4_CHECKPOINT_WEIGHTS_UNAUGMENTED_OUTER_TEST",
        "outer_test_augmentation_applied": False,
    }
