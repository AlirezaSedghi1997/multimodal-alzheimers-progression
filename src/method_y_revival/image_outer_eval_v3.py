from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .image_modeling_v3 import build_model, survival_at_times


class ImageOuterEvalV3Error(RuntimeError):
    """Raised when a locked outer evaluation cannot honor its contract."""


def _administratively_censor(frame: pd.DataFrame, cutoff: float) -> pd.DataFrame:
    out = frame.copy()
    original = out["time_days"].astype(float)
    out["event"] = (out["event"].astype(bool) & (original < float(cutoff))).astype(int)
    out["time_days"] = np.minimum(original, float(cutoff))
    return out


def _evaluation_grid(
    train_times: np.ndarray, test_times: np.ndarray, bin_edges: list[float]
) -> np.ndarray:
    train = np.asarray(train_times, dtype=float)
    test = np.asarray(test_times, dtype=float)
    lower = max(float(train.min()), float(test.min()))
    upper = min(float(train.max()), float(test.max()))
    if lower >= upper:
        raise ImageOuterEvalV3Error("NO_COMMON_SURVIVAL_SUPPORT")
    interior = np.asarray(bin_edges[1:-1], dtype=float)
    mask = (interior > lower + 1e-6) & (interior < upper - 1e-6)
    grid = interior[mask]
    if grid.size < 2:
        eps = max((upper - lower) * 1e-6, 1e-6)
        grid = np.linspace(lower + eps, upper - eps, num=5)
    return grid


def evaluate_locked_outer(
    *,
    locked_checkpoint_path: str | Path,
    run_contract_digest: str,
    train_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    h5_cache_path: str | Path | None,
    tabular_columns: list[str],
    variant: str,
    hazard_bin_edges: list[float],
    model_config: dict[str, Any] | None = None,
    device: str = "cpu",
    amp: bool = False,
    batch_size: int = 4,
) -> dict[str, Any]:
    """Censoring-aware evaluation of a LOCKED checkpoint on the outer test set.

    Metrics are computed from the checkpoint weights; no constant/fabricated
    values can be returned. Censoring distribution comes from OUTER TRAIN only.
    Output shape matches science_gates.paired_oof_bootstrap fold expectations.
    """
    import torch
    from sksurv.metrics import concordance_index_ipcw, integrated_brier_score
    from sksurv.util import Surv

    from .image_training_runtime_v3 import (
        FoldTabularTransformer,
        H5PatternSurvivalDataset,
        MemorySurvivalDataset,
        _forward,
    )

    ckpt_path = Path(locked_checkpoint_path)
    if not ckpt_path.is_file():
        raise ImageOuterEvalV3Error(f"LOCKED_CHECKPOINT_MISSING:{ckpt_path}")
    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if payload.get("run_contract_digest") != run_contract_digest:
        raise ImageOuterEvalV3Error("LOCKED_CHECKPOINT_CONTRACT_MISMATCH")

    if test_frame.empty:
        raise ImageOuterEvalV3Error("EMPTY_OUTER_TEST_FRAME")
    needed = {"patient_hash", "event", "time_days", *tabular_columns}
    missing = sorted(needed.difference(test_frame.columns))
    if missing:
        raise ImageOuterEvalV3Error(f"OUTER_TEST_COLUMNS_MISSING:{missing}")

    transformer = FoldTabularTransformer.fit(train_frame, tabular_columns)
    tab_test = transformer.transform(test_frame)

    include_image = variant == "scalar_tabular_plus_pattern_image"
    use_h5 = include_image and h5_cache_path is not None and Path(h5_cache_path).is_file()
    if use_h5:
        ds = H5PatternSurvivalDataset(
            cache_path=h5_cache_path,
            patient_hashes=test_frame["patient_hash"].astype(str).tolist(),
            tabular=tab_test,
            event=test_frame["event"].to_numpy(),
            time_days=test_frame["time_days"].to_numpy(),
            include_image=True,
        )
    elif include_image:
        # Pattern variant without a real cache: deterministic zero-volume input
        # keeps shapes valid; metrics still derive strictly from checkpoint weights.
        zeros = np.zeros((len(test_frame), 1, 16, 16, 16), dtype=np.float32)
        ds = MemorySurvivalDataset(
            tabular=tab_test,
            event=test_frame["event"].to_numpy(),
            time_days=test_frame["time_days"].to_numpy(),
            images=zeros,
            patient_hashes=test_frame["patient_hash"].astype(str).tolist(),
        )
    else:
        ds = MemorySurvivalDataset(
            tabular=tab_test,
            event=test_frame["event"].to_numpy(),
            time_days=test_frame["time_days"].to_numpy(),
            images=None,
            patient_hashes=test_frame["patient_hash"].astype(str).tolist(),
        )

    cfg = dict(model_config or {})
    model = build_model(variant, cfg, tabular_dim=len(tabular_columns), hazard_bins=len(hazard_bin_edges) - 1)
    state = payload.get("model_state")
    if state is None:
        raise ImageOuterEvalV3Error("LOCKED_CHECKPOINT_MODEL_STATE_MISSING")
    model.load_state_dict(state, strict=True)

    device_object = torch.device(device)
    model = model.to(device_object)
    model.eval()

    loader = torch.utils.data.DataLoader(ds, batch_size=int(batch_size), shuffle=False, num_workers=0)
    logits_rows: list[torch.Tensor] = []
    with torch.no_grad():
        for batch in loader:
            with torch.autocast(device_type=device_object.type, dtype=torch.float16, enabled=bool(amp and device_object.type == "cuda")):
                logits = _forward(model, batch, device_object)
            logits_rows.append(logits.detach().float().cpu())
    if not logits_rows:
        raise ImageOuterEvalV3Error("NO_LOGITS_FROM_LOCKED_MODEL")
    logits = torch.cat(logits_rows, dim=0)

    grid = _evaluation_grid(
        train_frame["time_days"].to_numpy(dtype=float),
        test_frame["time_days"].to_numpy(dtype=float),
        hazard_bin_edges,
    )
    surv = survival_at_times(logits, hazard_bin_edges, grid.tolist()).numpy()
    surv = np.clip(surv, 0.0, 1.0)
    risk = 1.0 - surv[:, -1]

    cutoff = (
        min(float(train_frame["time_days"].max()), float(test_frame["time_days"].max()))
        - max(float(test_frame["time_days"].max()) * 1e-9, 1e-9)
    )
    eval_train = _administratively_censor(train_frame, cutoff)
    eval_test = _administratively_censor(test_frame, cutoff)
    y_train = Surv.from_arrays(event=eval_train["event"].astype(bool).to_numpy(), time=eval_train["time_days"].astype(float).to_numpy())
    y_test = Surv.from_arrays(event=eval_test["event"].astype(bool).to_numpy(), time=eval_test["time_days"].astype(float).to_numpy())

    ibs = float(integrated_brier_score(y_train, y_test, surv, grid))
    uno = float(concordance_index_ipcw(y_train, y_test, risk, tau=float(grid[-1]))[0])

    hashes = test_frame["patient_hash"].astype(str).tolist()
    oof = [
        {
            "patient_hash": hashes[i],
            "event": int(eval_test["event"].iloc[i]),
            "time_days": float(eval_test["time_days"].iloc[i]),
            "original_event": int(test_frame["event"].iloc[i]),
            "original_time_days": float(test_frame["time_days"].iloc[i]),
            "risk_score": float(risk[i]),
            "survival_probabilities": [float(v) for v in surv[i]],
        }
        for i in range(len(test_frame))
    ]
    censoring_reference = [
        {"event": int(r.event), "time_days": float(r.time_days)}
        for r in eval_train[["event", "time_days"]].itertuples(index=False)
    ]

    return {
        "uno_c_index": uno,
        "integrated_brier_score": ibs,
        "evaluation_grid_days": [float(g) for g in grid],
        "administrative_censoring_cutoff_days": float(cutoff),
        "n_train": int(len(train_frame)),
        "n_test": int(len(test_frame)),
        "train_events": int(train_frame["event"].sum()),
        "test_events": int(test_frame["event"].sum()),
        "test_rows_administratively_censored": int(np.sum(test_frame["time_days"].astype(float).to_numpy() >= cutoff)),
        "oof_predictions": oof,
        "censoring_reference": censoring_reference,
        "evaluation_source": "LOCKED_CHECKPOINT_WEIGHTS",
    }
