from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import yaml


def _config() -> dict:
    config = yaml.safe_load(
        Path("configs/method_y_image_training_v3_1.yaml").read_text(encoding="utf-8")
    )
    config = copy.deepcopy(config)
    config["model"]["hpo_candidates"] = [
        {
            "candidate_id": "only",
            "learning_rate": 1e-4,
            "weight_decay": 1e-4,
            "complexity_rank": 0,
        }
    ]
    return config


def _frame(n: int = 30) -> pd.DataFrame:
    rng = np.random.default_rng(41)
    config = _config()
    data: dict[str, object] = {
        "patient_hash": [f"pt_{i:04d}" for i in range(n)],
        "event": np.asarray(([0, 1] * ((n + 1) // 2))[:n], dtype=int),
        "time_days": np.linspace(150.0, 1500.0, n),
    }
    for column in config["branches"]["amyloid"]["model_features"]:
        data[column] = rng.normal(size=n)
    return pd.DataFrame(data)


def test_runtime_and_source_policy_change_the_training_contract() -> None:
    from src.method_y_revival.image_training_orchestrator_v3 import _build_run_contract

    base = _config()
    base["runtime"]["deterministic_cuda"] = True
    base["training_source_sha256"] = "a" * 64
    kwargs = dict(
        branch="amyloid",
        variant="scalar_tabular",
        outer_fold=0,
        candidate_id="only",
        membership_sha="b" * 64,
        cache_sha="c" * 64,
        split_sha="d" * 64,
        readiness_sha="e" * 64,
    )
    digest_a = _build_run_contract(config=base, **kwargs)

    runtime_changed = copy.deepcopy(base)
    runtime_changed["runtime"]["deterministic_cuda"] = False
    digest_runtime = _build_run_contract(config=runtime_changed, **kwargs)

    source_changed = copy.deepcopy(base)
    source_changed["training_source_sha256"] = "f" * 64
    digest_source = _build_run_contract(config=source_changed, **kwargs)

    assert digest_runtime != digest_a
    assert digest_source != digest_a


def test_incomplete_inner_partition_fails_before_training(tmp_path, monkeypatch) -> None:
    import src.method_y_revival.image_training_orchestrator_v3 as module

    frame = _frame(18)
    keys = frame["patient_hash"].tolist()
    config = _config()
    orchestrator = module.ImageTrainingOrchestratorV3(
        config=config,
        output_root=tmp_path / "out",
        branch="amyloid",
        cache_sha="a" * 64,
        membership_sha="b" * 64,
        split_sha="c" * 64,
        readiness_sha="d" * 64,
    )
    orchestrator.verify_inputs()

    def training_must_not_start(**_kwargs):
        raise AssertionError("TRAINING_STARTED_BEFORE_PARTITION_VALIDATION")

    monkeypatch.setattr(module, "fit_survival_model", training_must_not_start)
    bad_inner = [
        {
            "fold": 0,
            "train_subjects": keys[:8],
            "validation_subjects": keys[8:12],
        }
    ]
    with pytest.raises(module.ImageTrainingOrchestratorV3Error, match="INNER_PARTITION_INCOMPLETE"):
        orchestrator.run_inner_hpo(
            outer_fold=0,
            variant="scalar_tabular",
            outer_train_frame=frame,
            inner_splits=bad_inner,
            tabular_columns=config["branches"]["amyloid"]["model_features"],
            hazard_bin_edges=config["shared"]["hazard_bin_edges_days"],
            device="cpu",
        )


def test_inner_ibs_failure_never_falls_back_to_validation_nll(tmp_path, monkeypatch) -> None:
    import sksurv.metrics
    from src.method_y_revival.image_training_orchestrator_v3 import (
        ImageTrainingOrchestratorV3,
        ImageTrainingOrchestratorV3Error,
    )

    frame = _frame(18)
    keys = frame["patient_hash"].tolist()
    thirds = [keys[:6], keys[6:12], keys[12:]]
    inner = []
    for fold, validation in enumerate(thirds):
        train = [key for key in keys if key not in set(validation)]
        inner.append(
            {
                "fold": fold,
                "train_subjects": train,
                "validation_subjects": validation,
            }
        )
    config = _config()
    orchestrator = ImageTrainingOrchestratorV3(
        config=config,
        output_root=tmp_path / "out",
        branch="amyloid",
        cache_sha="a" * 64,
        membership_sha="b" * 64,
        split_sha="c" * 64,
        readiness_sha="d" * 64,
    )
    orchestrator.verify_inputs()

    def forced_metric_failure(*_args, **_kwargs):
        raise ValueError("forced support failure")

    monkeypatch.setattr(sksurv.metrics, "integrated_brier_score", forced_metric_failure)
    with pytest.raises(ImageTrainingOrchestratorV3Error, match="INNER_IBS_NOT_ESTIMABLE"):
        orchestrator.run_inner_hpo(
            outer_fold=0,
            variant="scalar_tabular",
            outer_train_frame=frame,
            inner_splits=inner,
            tabular_columns=config["branches"]["amyloid"]["model_features"],
            hazard_bin_edges=config["shared"]["hazard_bin_edges_days"],
            device="cpu",
        )


def test_production_image_evaluation_rejects_missing_cache(tmp_path) -> None:
    from src.method_y_revival.image_modeling_v3 import build_model
    from src.method_y_revival.image_outer_eval_v3 import (
        ImageOuterEvalV3Error,
        evaluate_locked_outer,
    )

    config = _config()
    columns = config["branches"]["amyloid"]["model_features"]
    edges = config["shared"]["hazard_bin_edges_days"]
    contract = "a" * 64
    model = build_model(
        "scalar_tabular_plus_pattern_image",
        config["model"],
        tabular_dim=len(columns),
        hazard_bins=len(edges) - 1,
    )
    checkpoint = tmp_path / "locked.pt"
    torch.save(
        {"run_contract_digest": contract, "model_state": model.state_dict()}, checkpoint
    )
    frame = _frame(30)
    with pytest.raises(ImageOuterEvalV3Error, match="PATTERN_CACHE_REQUIRED"):
        evaluate_locked_outer(
            locked_checkpoint_path=checkpoint,
            run_contract_digest=contract,
            train_frame=frame.iloc[:20].copy(),
            test_frame=frame.iloc[20:].copy(),
            h5_cache_path=tmp_path / "missing.h5",
            tabular_columns=columns,
            variant="scalar_tabular_plus_pattern_image",
            hazard_bin_edges=edges,
            model_config=config["model"],
        )


def test_incomplete_branch_cannot_finalize(tmp_path) -> None:
    from src.method_y_revival.image_training_orchestrator_v3 import (
        ImageTrainingOrchestratorV3,
        ImageTrainingOrchestratorV3Error,
    )

    config = _config()
    orchestrator = ImageTrainingOrchestratorV3(
        config=config,
        output_root=tmp_path / "out",
        branch="amyloid",
        cache_sha="a" * 64,
        membership_sha="b" * 64,
        split_sha="c" * 64,
        readiness_sha="d" * 64,
    )
    with pytest.raises(ImageTrainingOrchestratorV3Error, match="FAILED_INCOMPLETE_OUTER_CV"):
        orchestrator.finalize_branch()


def test_public_privacy_scan_rejects_row_level_censoring_reference(tmp_path) -> None:
    from src.method_y_revival.image_reporting_v3 import privacy_scan_reports

    reports = tmp_path / "reports"
    reports.mkdir()
    (reports / "leaky.json").write_text(
        json.dumps(
            {
                "aggregate": {
                    "censoring_reference": [
                        {"event": 1, "time_days": 400.0},
                        {"event": 0, "time_days": 1200.0},
                    ]
                }
            }
        ),
        encoding="utf-8",
    )
    result = privacy_scan_reports(reports)
    assert result["status"] == "FAIL"
    assert any("row_level" in violation for violation in result["violations"])

