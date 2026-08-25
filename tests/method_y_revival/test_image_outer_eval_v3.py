from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import yaml


CONFIG = yaml.safe_load(Path("configs/method_y_image_training_v3_1.yaml").read_text(encoding="utf-8"))
COLS = CONFIG["branches"]["amyloid"]["model_features"]
EDGES = CONFIG["shared"]["hazard_bin_edges_days"]


def _frame(n=24, seed=7):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "patient_hash": [f"ev_{i:03d}_{seed}" for i in range(n)],
            "event": rng.integers(0, 2, size=n),
            "time_days": rng.integers(300, 1500, size=n).astype(float),
        }
    )
    for c, default in [("age", 70.0), ("sex_binary", 0), ("education_years", 12), ("apoe4_carrier", 0), ("baseline_pacc_raw", 0.0), ("baseline_fctotal96", 0.0), ("baseline_ldeltotal", 0.0), ("baseline_digittotal", 0.0), ("baseline_mmscore", 28), ("treatment_active", 0), ("baseline_amyloid_centiloid", 0.0)]:
        df[c] = default
    if df["event"].sum() < 4:
        df.loc[:4, "event"] = 1
    return df


def _train_locked(out_dir, seed):
    from src.method_y_revival.image_modeling_v3 import build_model
    from src.method_y_revival.image_training_runtime_v3 import (
        FoldTabularTransformer,
        MemorySurvivalDataset,
        fit_survival_model,
        seed_everything,
    )

    df = _frame(seed=seed)
    tr = FoldTabularTransformer.fit(df, COLS)
    tab = tr.transform(df)
    seed_everything(seed)
    model = build_model("scalar_tabular_plus_pattern_image", CONFIG["model"], tabular_dim=len(COLS), hazard_bins=len(EDGES) - 1)
    zeros = np.zeros((len(df), 1, 16, 16, 16), dtype=np.float32)
    ds = MemorySurvivalDataset(tabular=tab, event=df["event"].to_numpy(), time_days=df["time_days"].to_numpy(), images=zeros)
    fit_survival_model(
        model=model, train_dataset=ds, validation_dataset=None, bin_edges=EDGES,
        output_dir=out_dir, run_contract_digest="b" * 64, seed=seed, max_epochs=1,
        patience=10, learning_rate=0.001, weight_decay=0.0001, batch_size=4,
        gradient_accumulation_steps=1, device="cpu", amp=False, fixed_epochs=1,
    )
    return out_dir / "locked_checkpoint.pt"


def test_real_outer_evaluator_metrics_depend_on_weights_not_constant(tmp_path):
    """ANTI-PLACEHOLDER: two differently-trained locked models must yield different metrics."""
    from src.method_y_revival.image_outer_eval_v3 import evaluate_locked_outer

    ck_a = _train_locked(tmp_path / "a", seed=11)
    ck_b = _train_locked(tmp_path / "b", seed=22)
    df = _frame()
    split = len(df) // 2
    train_df, test_df = df.iloc[:split], df.iloc[split:]
    kw = dict(
        run_contract_digest="b" * 64, train_frame=train_df, test_frame=test_df,
        h5_cache_path=None, tabular_columns=COLS,
        variant="scalar_tabular_plus_pattern_image", hazard_bin_edges=EDGES,
        device="cpu", amp=False,
    )
    m_a = evaluate_locked_outer(locked_checkpoint_path=ck_a, **kw)
    m_b = evaluate_locked_outer(locked_checkpoint_path=ck_b, **kw)
    for m in (m_a, m_b):
        assert isinstance(m["integrated_brier_score"], float) and np.isfinite(m["integrated_brier_score"])
        assert isinstance(m["uno_c_index"], float) and np.isfinite(m["uno_c_index"])
        assert len(m["oof_predictions"]) == len(test_df)
        assert len(m["evaluation_grid_days"]) >= 2
        for row in m["oof_predictions"]:
            assert 0.0 <= row["risk_score"] <= 1.0
            assert len(row["survival_probabilities"]) == len(m["evaluation_grid_days"])
    assert abs(m_a["integrated_brier_score"] - m_b["integrated_brier_score"]) > 1e-9


def test_real_outer_evaluator_rejects_contract_mismatch(tmp_path):
    from src.method_y_revival.image_outer_eval_v3 import ImageOuterEvalV3Error, evaluate_locked_outer

    ck = _train_locked(tmp_path / "c", seed=33)
    df = _frame()
    with pytest.raises(ImageOuterEvalV3Error, match="CONTRACT_MISMATCH"):
        evaluate_locked_outer(
            locked_checkpoint_path=ck, run_contract_digest="d" * 64,
            train_frame=df, test_frame=df, h5_cache_path=None,
            tabular_columns=COLS, variant="scalar_tabular_plus_pattern_image",
            hazard_bin_edges=EDGES, device="cpu", amp=False,
        )


def test_launcher_source_contains_no_fabricated_metric_literals():
    src = Path("scripts/run_method_y_image_training_v3.py").read_text(encoding="utf-8")
    for bad in ['0.62', '0.61', '_real_evaluator']:
        assert bad not in src, f"fabricated/placeholder artifact in launcher: {bad}"


def test_winner_record_carries_learning_rate_and_weight_decay(tmp_path):
    from src.method_y_revival.image_training_orchestrator_v3 import ImageTrainingOrchestratorV3

    cfg = {**CONFIG, "model": {**CONFIG["model"], "hpo_candidates": [
        {"candidate_id": "lo", "learning_rate": 0.0001, "weight_decay": 0.001, "complexity_rank": 0},
        {"candidate_id": "hi", "learning_rate": 0.0003, "weight_decay": 0.0001, "complexity_rank": 1},
    ]}}
    df = _frame(n=30, seed=5)
    hashes = df["patient_hash"].tolist()
    idx = np.arange(len(hashes))
    np.random.default_rng(0).shuffle(idx)
    folds = np.array_split(idx, 5)
    train_hashes = [hashes[i] for i in np.concatenate([f for j, f in enumerate(folds) if j != 0])]
    t_idx = np.arange(len(train_hashes))
    np.random.default_rng(1).shuffle(t_idx)
    parts = np.array_split(t_idx, 3)
    inner = []
    for k, v_idx in enumerate(parts):
        val = [train_hashes[i] for i in v_idx]
        trn = [train_hashes[i] for i in np.concatenate([s for j, s in enumerate(parts) if j != k])]
        inner.append({"fold": k, "train_subjects": trn, "validation_subjects": val})
    outer_train = df[df["patient_hash"].isin(train_hashes)]
    orch = ImageTrainingOrchestratorV3(
        config=cfg, output_root=tmp_path / "out", branch="amyloid",
        cache_sha="a" * 64, membership_sha="b" * 64, split_sha="c" * 64, readiness_sha="d" * 64,
    )
    res = orch.run_inner_hpo(
        outer_fold=0, variant="scalar_tabular", outer_train_frame=outer_train,
        inner_splits=inner, tabular_columns=COLS, hazard_bin_edges=EDGES, device="cpu",
    )
    for rec in res["candidates"]:
        assert "learning_rate" in rec and "weight_decay" in rec
    assert res["winner"]["learning_rate"] in (0.0001, 0.0003)


def test_run_final_training_requires_winner_hyperparams(tmp_path):
    from src.method_y_revival.image_training_orchestrator_v3 import (
        ImageTrainingOrchestratorV3Error,
        ImageTrainingOrchestratorV3,
    )

    orch = ImageTrainingOrchestratorV3(
        config=CONFIG, output_root=tmp_path / "out2", branch="amyloid",
        cache_sha="a" * 64, membership_sha="b" * 64, split_sha="c" * 64, readiness_sha="d" * 64,
    )
    df = _frame(n=18, seed=9)
    stripped = {"candidate_id": "x", "mean_ibs": 0.2, "standard_error": 0.01, "complexity_rank": 0}
    with pytest.raises(ImageTrainingOrchestratorV3Error, match="WINNER_HYPERPARAMS_MISSING"):
        orch.run_final_training(
            outer_fold=0, variant="scalar_tabular", outer_train_frame=df,
            tabular_columns=COLS, hazard_bin_edges=EDGES, winner=stripped, device="cpu",
        )
