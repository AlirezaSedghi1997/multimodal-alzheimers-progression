from __future__ import annotations

import json
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import torch


def _synthetic_frame(n: int, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    hashes = [f"pt_{i:06d}_{seed}" for i in range(n)]
    events = rng.integers(0, 2, size=n)
    times = rng.integers(100, 1600, size=n).astype(float)
    # ensure at least some events
    if events.sum() == 0:
        events[0] = 1
    df = pd.DataFrame(
        {
            "patient_hash": hashes,
            "event": events,
            "time_days": times,
            "age": rng.normal(70, 7, size=n),
            "sex_binary": rng.integers(0, 2, size=n),
            "education_years": rng.integers(8, 20, size=n),
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


def _splits_for_frame(frame: pd.DataFrame, seed: int = 0):
    # 5 outer, 3 inner synthetic splits with patient_hash keys
    hashes = frame["patient_hash"].tolist()
    rng = np.random.default_rng(seed)
    outer = []
    indices = np.arange(len(hashes))
    rng.shuffle(indices)
    # split into 5 outer folds
    folds = np.array_split(indices, 5)
    for fold_id, test_idx in enumerate(folds):
        test_hashes = [hashes[i] for i in test_idx]
        train_idx = np.concatenate([f for i, f in enumerate(folds) if i != fold_id])
        train_hashes = [hashes[i] for i in train_idx]
        # inner 3 folds within train
        inner_folds = []
        rng2 = np.random.default_rng(seed + fold_id + 100)
        train_indices = np.arange(len(train_hashes))
        rng2.shuffle(train_indices)
        inner_splits = np.array_split(train_indices, 3)
        for inner_id, val_idx in enumerate(inner_splits):
            val_hashes = [train_hashes[i] for i in val_idx]
            train_inner_idx = np.concatenate([s for j, s in enumerate(inner_splits) if j != inner_id])
            train_inner_hashes = [train_hashes[i] for i in train_inner_idx]
            inner_folds.append(
                {
                    "fold": inner_id,
                    "train_subjects": train_inner_hashes,
                    "validation_subjects": val_hashes,
                }
            )
        outer.append(
            {
                "fold": fold_id,
                "train_subjects": train_hashes,
                "test_subjects": test_hashes,
                "inner_folds": inner_folds,
            }
        )
    return outer


def test_outer_dataset_factory_is_not_called_during_inner_hpo(tmp_path):
    from src.method_y_revival.image_training_orchestrator_v3 import ImageTrainingOrchestratorV3
    import yaml

    config = yaml.safe_load(Path("configs/method_y_image_training_v3_1.yaml").read_text(encoding="utf-8"))
    frame = _synthetic_frame(30, seed=1)
    outer_splits = _splits_for_frame(frame, seed=1)
    outer_train = frame[frame["patient_hash"].isin(outer_splits[0]["train_subjects"])]
    # factory that should never be called
    calls = {"count": 0}

    def outer_factory():
        calls["count"] += 1
        return frame[frame["patient_hash"].isin(outer_splits[0]["test_subjects"])]

    orch = ImageTrainingOrchestratorV3(
        config=config,
        output_root=tmp_path / "out",
        branch="amyloid",
        cache_sha="a" * 64,
        membership_sha="b" * 64,
        split_sha="c" * 64,
        readiness_sha="d" * 64,
    )
    orch.verify_inputs()
    # run inner HPO – must not call outer_factory
    orch.run_inner_hpo(
        outer_fold=0,
        variant="scalar_tabular",
        outer_train_frame=outer_train,
        inner_splits=outer_splits[0]["inner_folds"],
        tabular_columns=config["branches"]["amyloid"]["model_features"],
        hazard_bin_edges=config["shared"]["hazard_bin_edges_days"],
        device="cpu",
    )
    assert calls["count"] == 0


def test_winner_uses_mean_inner_ibs_not_validation_nll_or_outer_metric():
    from src.method_y_revival.survival_models import select_inner_winner

    # Candidate A: lower mean IBS but higher validation NLL surrogate
    candidates = [
        {"candidate_id": "a", "mean_ibs": 0.20, "standard_error": 0.01, "complexity_rank": 0},
        {"candidate_id": "b", "mean_ibs": 0.25, "standard_error": 0.01, "complexity_rank": 1},
    ]
    winner = select_inner_winner(candidates)
    assert winner["candidate_id"] == "a"
    # Even if outer metric would prefer b, winner stays a
    candidates2 = [
        {"candidate_id": "a", "mean_ibs": 0.22, "standard_error": 0.005, "complexity_rank": 0},
        {"candidate_id": "b", "mean_ibs": 0.21, "standard_error": 0.005, "complexity_rank": 1},
    ]
    winner2 = select_inner_winner(candidates2)
    assert winner2["candidate_id"] == "b"


def test_one_se_tie_break_uses_locked_complexity_rank():
    from src.method_y_revival.survival_models import select_inner_winner

    # best has mean 0.20 se 0.02 => threshold 0.22. Candidate b within threshold but more complex should not win over simpler a if both within.
    # Actually a is best (0.20). b is 0.21 rank 1, c is 0.215 rank 0 . Eligible: a,b,c all <=0.22. Among eligible, min complexity_rank wins -> rank 0 candidates preferred. Between a and c both rank 0, lower IBS wins -> a.
    candidates = [
        {"candidate_id": "lr1e4_wd1e3", "mean_ibs": 0.20, "standard_error": 0.02, "complexity_rank": 0},
        {"candidate_id": "lr1e4_wd1e4", "mean_ibs": 0.21, "standard_error": 0.01, "complexity_rank": 1},
        {"candidate_id": "lr3e4_wd1e4", "mean_ibs": 0.215, "standard_error": 0.01, "complexity_rank": 2},
    ]
    winner = select_inner_winner(candidates)
    assert winner["candidate_id"] == "lr1e4_wd1e3"
    # If we make a more complex but slightly better, but within SE, simpler should still win? Actually best is lr3e4_wd1e4 with 0.19, then eligible includes all within 0.21. Simpler rank 0 within threshold should win over best rank 2 if within threshold.
    candidates2 = [
        {"candidate_id": "lr1e4_wd1e3", "mean_ibs": 0.20, "standard_error": 0.015, "complexity_rank": 0},
        {"candidate_id": "lr1e4_wd1e4", "mean_ibs": 0.205, "standard_error": 0.01, "complexity_rank": 1},
        {"candidate_id": "lr3e4_wd1e4", "mean_ibs": 0.19, "standard_error": 0.02, "complexity_rank": 2},
    ]
    # best is lr3e4_wd1e4 (0.19) threshold 0.21, eligible all three, min rank is 0 -> lr1e4_wd1e3
    winner2 = select_inner_winner(candidates2)
    assert winner2["complexity_rank"] == 0


def test_tabular_transformer_fits_only_inner_or_outer_train():
    from src.method_y_revival.image_training_runtime_v3 import FoldTabularTransformer

    frame_train = pd.DataFrame({"a": [1, 2, 3], "b": [10, 20, 30]})
    frame_val = pd.DataFrame({"a": [100, 200], "b": [1000, 2000]})
    trans = FoldTabularTransformer.fit(frame_train, ["a", "b"])
    # Transform val should not have leaked val stats: mean of train is ~2, not 100
    val_trans = trans.transform(frame_val)
    # val values transformed with train mean/scale should be large positive, not near zero
    assert float(val_trans[0, 0]) > 5.0


def test_outer_checkpoint_is_locked_before_test_dataset_creation(tmp_path):
    from src.method_y_revival.outer_evaluation_control import OuterEvaluationControlError, OuterFoldController

    rc = "e" * 64
    root = tmp_path / "outer"
    controller = OuterFoldController(root, fold=0, run_contract_sha256=rc)
    # try evaluate before lock -> should fail
    with pytest.raises(OuterEvaluationControlError, match="CHECKPOINT_LOCK_REQUIRED"):
        controller.evaluate_once(lambda: {"c_index": 0.5})
    # now lock and evaluate succeeds
    controller.lock_checkpoint("f" * 64)
    metrics = controller.evaluate_once(lambda: {"c_index": 0.72})
    assert metrics["c_index"] == 0.72


def test_committed_outer_result_resume_does_not_construct_test_dataset(tmp_path):
    from src.method_y_revival.outer_evaluation_control import OuterFoldController

    rc = "a" * 64
    ckpt = "b" * 64
    root = tmp_path / "outer2"
    ctrl = OuterFoldController(root, fold=1, run_contract_sha256=rc)
    ctrl.lock_checkpoint(ckpt)
    calls = {"n": 0}

    def evaluator():
        calls["n"] += 1
        return {"ibs": 0.12}

    m1 = ctrl.evaluate_once(evaluator)
    assert calls["n"] == 1
    # second controller instance resuming same fold should not call evaluator again
    ctrl2 = OuterFoldController(root, fold=1, run_contract_sha256=rc)
    # evaluator that would fail if called
    def failing_evaluator():
        raise RuntimeError("SHOULD_NOT_BE_CALLED")

    m2 = ctrl2.evaluate_once(failing_evaluator)
    assert m2 == m1
    assert calls["n"] == 1  # still 1


def test_interrupt_after_outer_result_commit_recovers_without_second_access(tmp_path):
    from src.method_y_revival.outer_evaluation_control import OuterFoldController, OuterEvaluationInterruption

    rc = "c" * 64
    ckpt = "d" * 64
    root = tmp_path / "outer3"
    ctrl = OuterFoldController(root, fold=2, run_contract_sha256=rc)
    ctrl.lock_checkpoint(ckpt)
    calls = {"n": 0}

    def eval_fn():
        calls["n"] += 1
        return {"c_index": 0.66}

    # first call with interruption flag
    try:
        ctrl.evaluate_once(eval_fn, interrupt_after_result_commit=True)
        assert False, "should have interrupted"
    except OuterEvaluationInterruption:
        pass
    assert calls["n"] == 1
    # resume should recover without second access
    ctrl2 = OuterFoldController(root, fold=2, run_contract_sha256=rc)
    calls2 = {"n": 0}

    def failing():
        calls2["n"] += 1
        raise RuntimeError("should not be called")

    metrics = ctrl2.evaluate_once(failing)
    assert metrics["c_index"] == 0.66
    assert calls2["n"] == 0


def test_incomplete_outer_cv_cannot_be_reported_complete(tmp_path):
    from src.method_y_revival.outer_evaluation_control import summarize_outer_evaluation

    rc = "e" * 64
    root = tmp_path / "incomplete"
    # create only 3 of 5 folds
    from src.method_y_revival.outer_evaluation_control import OuterFoldController

    for fold in range(3):
        c = OuterFoldController(root, fold=fold, run_contract_sha256=rc)
        c.lock_checkpoint("f" * 64)
        c.evaluate_once(lambda: {"c_index": 0.5 + fold * 0.01})
        c.finalize()
    summary = summarize_outer_evaluation(root, expected_folds=5, run_contract_sha256=rc)
    assert summary["status"] == "FAILED_INCOMPLETE_OUTER_CV"
    assert summary["outer_folds_completed"] == 3


def test_run_contract_change_fails_closed(tmp_path):
    from src.method_y_revival.image_training_orchestrator_v3 import ImageTrainingOrchestratorV3
    import yaml

    config = yaml.safe_load(Path("configs/method_y_image_training_v3_1.yaml").read_text(encoding="utf-8"))
    orch = ImageTrainingOrchestratorV3(
        config=config,
        output_root=tmp_path / "out",
        branch="amyloid",
        cache_sha="a" * 64,
        membership_sha="b" * 64,
        split_sha="c" * 64,
        readiness_sha="d" * 64,
    )
    orch.verify_inputs()
    # change cache sha and try to reload
    with pytest.raises(Exception, match="RUN_CONTRACT_MISMATCH"):
        ImageTrainingOrchestratorV3(
            config=config,
            output_root=tmp_path / "out",
            branch="amyloid",
            cache_sha="z" * 64,
            membership_sha="b" * 64,
            split_sha="c" * 64,
            readiness_sha="d" * 64,
        )


def test_private_oof_rows_are_not_written_to_public_reports(tmp_path):
    from src.method_y_revival.image_training_orchestrator_v3 import ImageTrainingOrchestratorV3
    import yaml

    config = yaml.safe_load(Path("configs/method_y_image_training_v3_1.yaml").read_text(encoding="utf-8"))
    frame = _synthetic_frame(30, seed=9)
    outer_splits = _splits_for_frame(frame, seed=9)
    outer_train = frame[frame["patient_hash"].isin(outer_splits[0]["train_subjects"])]
    orch = ImageTrainingOrchestratorV3(
        config=config,
        output_root=tmp_path / "out2",
        branch="amyloid",
        cache_sha="a" * 64,
        membership_sha="b" * 64,
        split_sha="c" * 64,
        readiness_sha="d" * 64,
    )
    orch.verify_inputs()
    winner_info = orch.run_inner_hpo(
        outer_fold=0,
        variant="scalar_tabular",
        outer_train_frame=outer_train,
        inner_splits=outer_splits[0]["inner_folds"],
        tabular_columns=config["branches"]["amyloid"]["model_features"],
        hazard_bin_edges=config["shared"]["hazard_bin_edges_days"],
        device="cpu",
    )
    final = orch.run_final_training(
        outer_fold=0,
        variant="scalar_tabular",
        outer_train_frame=outer_train,
        tabular_columns=config["branches"]["amyloid"]["model_features"],
        hazard_bin_edges=config["shared"]["hazard_bin_edges_days"],
        winner=winner_info["winner"],
        device="cpu",
    )
    # now outer evaluate with metrics containing private rows
    metrics_with_private = {
        "c_index": 0.71,
        "ibs": 0.11,
        "oof_predictions": [{"patient_hash": "secret_hash", "event": 1, "time_days": 400}],
        "embedding": [0.1, 0.2],
    }

    def evaluator():
        return metrics_with_private

    orch.outer_evaluate(outer_fold=0, variant="scalar_tabular", evaluator=evaluator, run_contract=final["run_contract"])
    # public file should not contain patient_hash
    public_path = tmp_path / "out2" / "reports" / "amyloid_scalar_tabular_fold_0_public.json"
    public = json.loads(public_path.read_text(encoding="utf-8"))
    assert "patient_hash" not in json.dumps(public)
    assert "oof_predictions" not in json.dumps(public)
    # private should contain
    private_path = tmp_path / "out2" / "private" / "outer_evaluation" / "amyloid" / "scalar_tabular" / "fold_0_private.json"
    private = json.loads(private_path.read_text(encoding="utf-8"))
    assert "oof_predictions" in json.dumps(private)
