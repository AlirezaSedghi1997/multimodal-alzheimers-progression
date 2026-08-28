"""Fail-open hardening tests for the CAFT trainer (Plan 006 Step 1)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.research import brainiac_caft as caft
from src.research import brainiac_caft_training as ct
from src.research import brainiac_caft_evaluation as ce


def test_local_dim_512_is_rejected():
    import dataclasses

    with pytest.raises(RuntimeError, match="EXACTLY_128"):
        dataclasses.replace(caft.CaftConfig(), local_embedding_dim=512).validate()


def test_legacy_simulated_entrypoint_stays_blocked():
    from scripts.research import train_brainiac_caft_benchmark as script

    with pytest.raises(RuntimeError, match="BRAINIAC_SIMULATED_BENCHMARK_FORBIDDEN"):
        script.run_caft_ablation_benchmark()


def test_lora_never_unlocks_in_r1():
    with pytest.raises(RuntimeError, match="LORA_PROTOCOL_NOT_REGISTERED"):
        caft.lora_config(frozen_stage_promoted=True)  # run_version defaults to r1c
    assert ce.evaluate_technical_promotion(
        delta_auc=0.9, bootstrap_lower_ci=0.5, delta_brier=0.0,
        delta_ece=0.0, confound_worsened=False, sanity_passed=True,
        measurable=True,
    )["lora_unlocked"] is False


def test_nonfinite_everything_fails_closed():
    roles = {
        "head_fit": _bad_view(nan=True, tag="hf"),
        "model_selection": _bad_view(nan=False, tag="ms"),
        "probability_calibration": _bad_view(nan=False, tag="pc"),
        "conformal_calibration": _bad_view(nan=False, tag="cc"),
    }
    with pytest.raises(RuntimeError, match="NOT_FINITE"):
        ct.fit_logistic_arm(roles, use_tabular=False, c_grid=(1.0,), seed=13)
    labels = np.array([0, 1, 0, 1])
    logits = np.array([0.1, np.nan, -0.2, 0.3])
    with pytest.raises(RuntimeError, match="NOT_FINITE"):
        ce.compute_metrics(labels, logits)


def _bad_view(*, nan: bool, tag: str = "x"):
    n = 12
    rng = np.random.default_rng(9)
    logits = rng.normal(size=(n, 1)).astype("float32")
    if nan:
        logits[0, 0] = np.nan
    return {
        "subject_hashes": [f"s_{tag}_{i:04d}" for i in range(n)],
        "locked_logit": logits,
        "local_embedding": rng.normal(size=(n, 128)).astype("float32"),
        "brainiac_embedding": rng.normal(size=(n, 768)).astype("float32"),
        "tabular": rng.normal(size=(n, 5)).astype("float32"),
        "class_values": (rng.random(n) < 0.5).astype("int64"),
    }


def test_audit_fields_never_enter_predictors():
    with pytest.raises(RuntimeError, match="AUDIT_FIELD"):
        caft.CaftConfig(audit_fields_used=("site",)).validate()


def test_outer_count_zero_in_every_training_state():
    schema = ct.state_contract_schema()
    assert schema["outer_access_count"]["type"] == "int"
    assert schema["outer_access_count"]["during_fit"] == 0
