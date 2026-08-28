"""Role-safe training library tests for the six-arm CAFT trainer (Plan 006)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.research import brainiac_caft_training as ct


def _make_view(n: int, *, seed: int = 0, positive_rate: float = 0.4):
    rng = np.random.default_rng(seed)
    return {
        "subject_hashes": [f"s_{seed}_{i:04d}" for i in range(n)],
        "locked_logit": rng.normal(size=(n, 1)).astype("float32"),
        "local_embedding": rng.normal(size=(n, 128)).astype("float32"),
        "brainiac_embedding": rng.normal(size=(n, 768)).astype("float32"),
        "tabular": rng.normal(size=(n, 5)).astype("float32"),
        "class_values": (rng.random(n) < positive_rate).astype("int64"),
    }


def _roles(n_per_role: int = 40):
    return {
        "head_fit": _make_view(n_per_role, seed=1),
        "model_selection": _make_view(n_per_role, seed=2),
        "probability_calibration": _make_view(n_per_role, seed=3),
        "conformal_calibration": _make_view(n_per_role, seed=4),
    }


# ---------------------------------------------------------------------------
# typed role views — no caller strings, no all_data
# ---------------------------------------------------------------------------


def test_public_api_has_no_all_data_or_role_strings() -> None:
    import inspect

    for name, member in inspect.getmembers(ct, inspect.isclass):
        if member.__module__ != ct.__name__:
            continue
        for method in ("fit_all", "all_data"):
            assert not hasattr(member, method)
    signature = inspect.signature(ct.fit_logistic_arm)
    assert "role" not in signature.parameters


def test_head_fit_view_required_for_tabular_fit() -> None:
    preprocessor = ct.HeadFitTabularPreprocessor()
    with pytest.raises(RuntimeError, match="FIT_ROLE_FORBIDDEN"):
        preprocessor.fit(_make_view(20)["tabular"], role="model_selection")
    preprocessor.fit(_roles()["head_fit"]["tabular"], role="head_fit")
    receipt = preprocessor.receipt()
    assert receipt["fit_role"] == "head_fit"
    assert receipt["locked"] is True and receipt["finite_output"] is True


def test_role_overlap_is_rejected() -> None:
    roles = _roles()
    shared = "s_1_0000"
    roles["model_selection"]["subject_hashes"][0] = shared
    roles["head_fit"]["subject_hashes"][1] = shared
    with pytest.raises(RuntimeError, match="ROLE_OVERLAP"):
        ct.assert_roles_disjoint(roles)


def test_outer_role_unavailable_before_authorization() -> None:
    locked = ct.LockedOuterView()
    with pytest.raises(RuntimeError, match="OUTER_LOCKED"):
        locked.subject_hashes()
    with pytest.raises(RuntimeError, match="OUTER_LOCKED"):
        locked.subject_hashes(authorized=False)
    assert locked.call_count() == 0


# ---------------------------------------------------------------------------
# logistic arms / C selection
# ---------------------------------------------------------------------------


def test_logistic_c_selected_only_on_model_selection() -> None:
    roles = _roles()
    result = ct.fit_logistic_arm(
        roles, use_tabular=False, c_grid=(0.01, 0.1, 1.0, 10.0), seed=13
    )
    assert result["selected_c"] in (0.01, 0.1, 1.0, 10.0)
    assert result["selection_role"] == "model_selection"
    assert result["fit_role"] == "head_fit"
    assert np.isfinite(result["coefficients"]).all()


def test_logistic_arms_fit_once_deterministic() -> None:
    roles = _roles()
    first = ct.fit_logistic_arm(roles, use_tabular=True, c_grid=(0.1, 1.0), seed=13)
    second = ct.fit_logistic_arm(roles, use_tabular=True, c_grid=(0.1, 1.0), seed=13)
    assert np.allclose(first["coefficients"], second["coefficients"])


# ---------------------------------------------------------------------------
# neural arms / equal budget / frozen encoders
# ---------------------------------------------------------------------------


def test_neural_optimizer_contains_no_encoder_parameters() -> None:
    arm = ct.NeuralArm(arm_id=6, config=None, seed=13)
    head_param_ids = {id(p) for p in arm.module.parameters() if p.requires_grad}
    optimizer_ids = {
        id(p) for group in arm.optimizer.param_groups for p in group["params"]
    }
    # Every optimizer parameter belongs to the trainable head; encoders are
    # external to the arm (their outputs arrive detached), so nothing outside
    # the head can ever appear here.
    assert optimizer_ids <= head_param_ids
    assert optimizer_ids == head_param_ids


def test_neural_budget_is_equal_across_neural_arms() -> None:
    budget = ct.NeuralBudget()
    for arm_id in (4, 5, 6):
        spec = budget.for_arm(arm_id)
        assert spec["optimizer"] == "AdamW"
        assert spec["lr"] == 1e-3 and spec["weight_decay"] == 1e-4
        assert spec["batch_size"] == 64 and spec["max_epochs"] == 100
        assert spec["warmup_epochs"] == 5 and spec["patience"] == 10
    assert budget.for_arm(1)["optimizer"] == "NONE"


def test_training_seed_reproducibility() -> None:
    roles = _roles(n_per_role=24)
    first = ct.train_neural_arm(
        roles, arm_id=5, seed=13, max_epochs=2, batch_size=8, progress=False
    )
    second = ct.train_neural_arm(
        roles, arm_id=5, seed=13, max_epochs=2, batch_size=8, progress=False
    )
    assert np.allclose(first["final_weights"], second["final_weights"])


# ---------------------------------------------------------------------------
# state contract / resume equivalence
# ---------------------------------------------------------------------------


def test_serialize_resume_reproduces_training_exactly(tmp_path):
    roles = _roles(n_per_role=24)
    uninterrupted = ct.train_neural_arm(
        roles, arm_id=6, seed=29, max_epochs=3, batch_size=8, progress=False
    )
    resumed = ct.train_neural_arm(
        roles, arm_id=6, seed=29, max_epochs=3, batch_size=8,
        resume_after_epoch=1, progress=False
    )
    assert np.allclose(uninterrupted["final_weights"], resumed["final_weights"])
    state_schema = ct.state_contract_schema()
    for field_name in (
        "run_id", "fold", "arm", "seed", "epoch", "optimizer_step",
        "batch_cursor", "rng_python", "rng_numpy", "rng_torch",
        "best_selection", "outer_access_count",
    ):
        assert field_name in state_schema


def test_best_selection_is_finite_and_ranged():
    roles = _roles(n_per_role=20)
    result = ct.train_neural_arm(
        roles, arm_id=6, seed=13, max_epochs=2, batch_size=8, progress=False
    )
    assert 0.0 <= result["best_selection"]["roc_auc"] <= 1.0
    assert result["best_selection"]["epoch"] >= 0


def test_state_contract_outer_access_count_zero_during_fit():
    roles = _roles(n_per_role=16)
    result = ct.train_neural_arm(
        roles, arm_id=4, seed=47, max_epochs=1, batch_size=8, progress=False
    )
    assert result["state"]["outer_access_count"] == 0
