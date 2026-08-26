from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pathlib import Path
import yaml
import hashlib
import json

from src.method_y_revival.image_training_runtime_v3 import MemorySurvivalDataset, FoldTabularTransformer, fit_survival_model
from src.method_y_revival.image_modeling_v3 import build_model
from src.method_y_revival.outer_evaluation_control import OuterFoldController


def _synthetic_frame(n=32, seed=0):
    rng = np.random.default_rng(seed)
    hashes = [f"pt_{i:06d}_{seed}" for i in range(n)]
    events = rng.integers(0, 2, size=n)
    if events.sum() < 5:
        events[:5] = 1
    times = rng.integers(200, 1500, size=n).astype(float)
    df = pd.DataFrame(
        {
            "patient_hash": hashes,
            "event": events,
            "time_days": times,
            "age": rng.normal(70, 5, size=n),
            "sex_binary": rng.integers(0, 2, size=n),
            "education_years": rng.integers(10, 20, size=n),
            "apoe4_carrier": rng.integers(0, 2, size=n),
            "baseline_pacc_raw": rng.normal(0, 1, size=n),
            "baseline_fctotal96": rng.normal(0, 1, size=n),
            "baseline_ldeltotal": rng.normal(0, 1, size=n),
            "baseline_digittotal": rng.normal(0, 1, size=n),
            "baseline_mmscore": rng.integers(24, 30, size=n),
            "treatment_active": rng.integers(0, 2, size=n),
            "baseline_amyloid_centiloid": rng.normal(20, 30, size=n),
        }
    )
    return df


def _splits_5x3(frame, seed=0):
    hashes = frame["patient_hash"].tolist()
    rng = np.random.default_rng(seed)
    idx = np.arange(len(hashes))
    rng.shuffle(idx)
    folds = np.array_split(idx, 5)
    outer = []
    for fid, test_idx in enumerate(folds):
        test_hashes = [hashes[i] for i in test_idx]
        train_idx = np.concatenate([f for i, f in enumerate(folds) if i != fid])
        train_hashes = [hashes[i] for i in train_idx]
        # inner 3
        rng2 = np.random.default_rng(seed + fid + 1000)
        t_idx = np.arange(len(train_hashes))
        rng2.shuffle(t_idx)
        inner_splits = np.array_split(t_idx, 3)
        inner_folds = []
        for iid, val_idx in enumerate(inner_splits):
            val_hashes = [train_hashes[i] for i in val_idx]
            train_inner_idx = np.concatenate([s for j, s in enumerate(inner_splits) if j != iid])
            train_inner_hashes = [train_hashes[i] for i in train_inner_idx]
            inner_folds.append({"fold": iid, "train_subjects": train_inner_hashes, "validation_subjects": val_hashes})
        outer.append({"fold": fid, "train_subjects": train_hashes, "test_subjects": test_hashes, "inner_folds": inner_folds})
    return outer


def test_synthetic_full_pipeline_interruption_and_exactly_once(tmp_path):
    """Integration: 24-40 subjects, 16^3, 5x3, 2 candidates, interruption, bitwise equality, outer access exactly 1."""
    config = yaml.safe_load(Path("configs/method_y_image_training_v3_1.yaml").read_text(encoding="utf-8"))
    # Use 32 subjects synthetic
    frame = _synthetic_frame(32, seed=123)
    splits = _splits_5x3(frame, seed=123)
    tabular_cols = config["branches"]["amyloid"]["model_features"]
    bin_edges = config["shared"]["hazard_bin_edges_days"]
    # Track outer access counts per variant/fold
    access_counts = {}

    def make_outer_dataset_factory(test_hashes):
        def factory():
            key = tuple(sorted(test_hashes))
            access_counts[key] = access_counts.get(key, 0) + 1
            sub = frame[frame["patient_hash"].isin(test_hashes)]
            # include_image variant will need image loading; we use MemorySurvivalDataset directly without H5
            return sub

        return factory

    # Choose 2 candidates for test (subset of 3)
    candidates = [
        {"candidate_id": "lr1e4_wd1e3", "learning_rate": 0.0001, "weight_decay": 0.001, "complexity_rank": 0},
        {"candidate_id": "lr1e4_wd1e4", "learning_rate": 0.0001, "weight_decay": 0.0001, "complexity_rank": 1},
    ]
    # Simulate nested HPO for one outer fold with synthetic 16^3, then final fit and outer eval with interruption
    from src.method_y_revival.image_training_orchestrator_v3 import ImageTrainingOrchestratorV3

    output_root = tmp_path / "synthetic_run"
    orch = ImageTrainingOrchestratorV3(
        config={**config, "model": {**config["model"], "hpo_candidates": candidates}},
        output_root=output_root,
        branch="amyloid",
        cache_sha="a" * 64,
        membership_sha="b" * 64,
        split_sha="c" * 64,
        readiness_sha="d" * 64,
        allow_synthetic_fixture=True,
    )
    orch.verify_inputs()
    outer_fold = 0
    outer_train_frame = frame[frame["patient_hash"].isin(splits[outer_fold]["train_subjects"])]
    inner_splits = splits[outer_fold]["inner_folds"]

    # ---- run inner HPO without interruption ----
    result_no_interrupt = orch.run_inner_hpo(
        outer_fold=outer_fold,
        variant="scalar_tabular",
        outer_train_frame=outer_train_frame,
        inner_splits=inner_splits,
        tabular_columns=tabular_cols,
        hazard_bin_edges=bin_edges,
        device="cpu",
    )
    winner = result_no_interrupt["winner"]
    # Check winner uses mean IBS (already via select_inner_winner)
    assert "mean_ibs" in winner

    # Now test final training with interruption after epoch 1 and resume
    # Use fit_survival_model directly to test interruption resumption bitwise
    # Create small datasets
    # deterministic images for bitwise comparison
    rng = np.random.default_rng(999)
    transformer = FoldTabularTransformer.fit(outer_train_frame, tabular_cols)
    tab = transformer.transform(outer_train_frame)
    images = rng.standard_normal(size=(len(outer_train_frame), 1, 16, 16, 16)).astype(np.float32)
    from src.method_y_revival.image_training_runtime_v3 import MemorySurvivalDataset, seed_everything

    model_cfg = config["model"]

    def run_training(interrupt_after=None, out_dir=None):
        # ensure deterministic model init via seeding
        seed_everything(999)
        torch_model = build_model("scalar_tabular_plus_pattern_image", model_cfg, tabular_dim=len(tabular_cols), hazard_bins=len(bin_edges) - 1)
        ds = MemorySurvivalDataset(tabular=tab, event=outer_train_frame["event"].to_numpy(), time_days=outer_train_frame["time_days"].to_numpy(), images=images, patient_hashes=outer_train_frame["patient_hash"].tolist())
        run_contract = "e" * 64
        # Use fixed_epochs for deterministic
        try:
            fit_survival_model(
                model=torch_model,
                train_dataset=ds,
                validation_dataset=None,
                bin_edges=bin_edges,
                output_dir=out_dir,
                run_contract_digest=run_contract,
                seed=999,
                max_epochs=3,
                patience=10,
                learning_rate=0.0001,
                weight_decay=0.0001,
                batch_size=2,
                gradient_accumulation_steps=1,
                device="cpu",
                amp=False,
                fixed_epochs=3,
                interrupt_after_epoch=interrupt_after,
            )
        except Exception as e:
            if "INTERRUPTED_AFTER_EPOCH" in str(e):
                raise
            raise

    import torch, tempfile, pathlib

    # Uninterrupted run
    out_uninterrupted = tmp_path / "uninterrupted"
    run_training(interrupt_after=None, out_dir=out_uninterrupted)
    ckpt_uninterrupted = torch.load(out_uninterrupted / "locked_checkpoint.pt", map_location="cpu", weights_only=False)
    # Interrupted run: first attempt with interrupt after 1
    out_resumed = tmp_path / "resumed"
    try:
        run_training(interrupt_after=1, out_dir=out_resumed)
        assert False, "should have interrupted"
    except Exception as e:
        assert "INTERRUPTED_AFTER_EPOCH_1" in str(e)
    # Resume same command without interrupt
    run_training(interrupt_after=None, out_dir=out_resumed)
    ckpt_resumed = torch.load(out_resumed / "locked_checkpoint.pt", map_location="cpu", weights_only=False)
    # Check bitwise weights equal
    for k in ckpt_uninterrupted["model_state"]:
        assert torch.equal(ckpt_uninterrupted["model_state"][k], ckpt_resumed["model_state"][k]), f"mismatch {k}"

    # Now test outer evaluation exactly-once with access counter
    # Use orchestrator's outer_evaluate
    access_counts.clear()
    factory = make_outer_dataset_factory(splits[outer_fold]["test_subjects"])
    # first, run final training via orchestrator to get locked checkpoint
    final = orch.run_final_training(
        outer_fold=outer_fold,
        variant="scalar_tabular_plus_pattern_image",
        outer_train_frame=outer_train_frame,
        tabular_columns=tabular_cols,
        hazard_bin_edges=bin_edges,
        winner=winner,
        device="cpu",
    )
    # evaluator that increments counter and returns dummy metrics with OOF
    def evaluator():
        # simulate dataset creation – this is the outer test dataset factory invocation
        ds_frame = factory()
        # return metrics
        return {"c_index": 0.71, "ibs": 0.11, "oof_predictions": [{"patient_hash": h, "event": 1, "time_days": 100} for h in ds_frame["patient_hash"][:2]]}

    metrics = orch.outer_evaluate(outer_fold=outer_fold, variant="scalar_tabular_plus_pattern_image", evaluator=evaluator, run_contract=final["run_contract"])
    assert metrics["c_index"] == 0.71
    assert access_counts[tuple(sorted(splits[outer_fold]["test_subjects"]))] == 1
    # Second call should recover without second access (resume)
    def failing_evaluator():
        assert False, "should not be called second time"

    metrics2 = orch.outer_evaluate(outer_fold=outer_fold, variant="scalar_tabular_plus_pattern_image", evaluator=failing_evaluator, run_contract=final["run_contract"])
    assert metrics2 == metrics
    assert access_counts[tuple(sorted(splits[outer_fold]["test_subjects"]))] == 1

    # Contract mismatch should fail closed
    with pytest.raises(Exception, match="RUN_CONTRACT_MISMATCH"):
        orch2 = ImageTrainingOrchestratorV3(
            config={**config, "model": {**config["model"], "hpo_candidates": candidates}},
            output_root=output_root,
            branch="amyloid",
            cache_sha="z" * 64,
            membership_sha="b" * 64,
            split_sha="c" * 64,
            readiness_sha="d" * 64,
        )
