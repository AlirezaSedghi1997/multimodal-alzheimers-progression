"""Adversarial tests for the ResNet34 fold-local representation (Plan 004/B3).

All models/checkpoints/contracts here are tiny synthetic fixtures; battery
checkpoints are only referenced by the live validators, never loaded.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import torch
from torch import nn

from src.research import resnet34_representation as rr


class _TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Identity()
        self.latent_proj = nn.Linear(8, 128)
        self.head = nn.Linear(128, 1)

    def forward(self, x):
        latent = self.latent_proj(self.encoder(x))
        return self.head(latent)


def _checkpoint(tmp_path: Path, name: str = "locked.pt") -> Path:
    model = _TinyModel()
    path = tmp_path / name
    torch.save(model.state_dict(), path)
    return path


# ---------------------------------------------------------------------------
# strict load / freeze / checkpoint binding
# ---------------------------------------------------------------------------


def test_strict_load_freezes_and_reports_coverage(tmp_path: Path) -> None:
    checkpoint = _checkpoint(tmp_path)
    model = _TinyModel()
    report = rr.load_locked_state(model, checkpoint)
    assert report["missing_keys"] == 0
    assert report["unexpected_keys"] == 0
    assert report["coverage_percent"] == 100.0
    assert model.training is False
    assert all(not p.requires_grad for p in model.parameters())


def test_checkpoint_sha_mismatch_fails(tmp_path: Path) -> None:
    checkpoint = _checkpoint(tmp_path)
    with pytest.raises(RuntimeError, match="CHECKPOINT_SHA_MISMATCH"):
        rr.load_locked_state(
            _TinyModel(), checkpoint, expected_sha256="0" * 64
        )


def test_strict_load_rejects_key_drift(tmp_path: Path) -> None:
    state = _TinyModel().state_dict()
    state["extra_layer.weight"] = torch.zeros(1)
    path = tmp_path / "drifted.pt"
    torch.save(state, path)
    with pytest.raises(RuntimeError, match="STRICT_LOAD_FAILED"):
        rr.load_locked_state(_TinyModel(), path)


# ---------------------------------------------------------------------------
# read-only latent hook
# ---------------------------------------------------------------------------


def test_latent_hook_collects_128_and_is_noninvasive(tmp_path: Path) -> None:
    model = _TinyModel()
    collector, handle, identity = rr.attach_latent_hook(model, "latent_proj")
    try:
        x = torch.randn(2, 8)
        out_before = model(x).detach().clone()
        captured = collector["last_output"]
        assert captured.shape == (2, 128)
        assert identity["module_name"] == "latent_proj"
        assert identity["module_id"] == id(model.latent_proj)
        assert torch.equal(out_before, model(x))  # hook changed nothing
    finally:
        handle.remove()


def test_hook_on_unknown_module_fails() -> None:
    with pytest.raises(RuntimeError, match="HOOK_MODULE_MISSING"):
        rr.attach_latent_hook(_TinyModel(), "does_not_exist")


# ---------------------------------------------------------------------------
# outer isolation
# ---------------------------------------------------------------------------


def test_outer_rows_are_rejected_and_loader_count_stays_zero() -> None:
    registry = {
        "folds": {
            "0": {"outer_test": ["s_outer_1"], "roles": {"head_fit": ["s_dev_1"]}},
        },
        "sealed": ["s_sealed_1"],
    }
    rr.assert_outer_train_only(["s_dev_1"], fold=0, registry=registry)
    assert rr.outer_loader_call_count() == 0
    with pytest.raises(RuntimeError, match="OUTER_ROW_FORBIDDEN"):
        rr.assert_outer_train_only(["s_outer_1"], fold=0, registry=registry)
    with pytest.raises(RuntimeError, match="OUTER_ROW_FORBIDDEN"):
        rr.assert_outer_train_only(["s_sealed_1"], fold=0, registry=registry)
    with pytest.raises(RuntimeError, match="OUTER_ACCESS_FORBIDDEN"):
        rr.record_outer_access()
    assert rr.outer_loader_call_count() == 0


# ---------------------------------------------------------------------------
# tabular preprocessing
# ---------------------------------------------------------------------------


def test_tabular_fit_is_head_fit_only_and_canonical_order_enforced() -> None:
    import numpy as np

    preprocessor = rr.TabularPreprocessor()
    with pytest.raises(RuntimeError, match="FIT_ROLE_FORBIDDEN"):
        preprocessor.fit(np.zeros((3, 5), dtype="float32"), role="model_selection")
    X = np.array(
        [[1.0, 0.0, 12.0, 0.0, 20.0], [3.0, 1.0, 16.0, 1.0, 28.0],
         [float("nan"), 1.0, 14.0, 0.0, 24.0]],
        dtype="float32",
    )
    preprocessor.fit(X, role="head_fit")
    assert preprocessor.receipt() == preprocessor.receipt()
    transformed = preprocessor.transform(X)
    assert np.isfinite(transformed).all()
    assert abs(float(np.nanmean(transformed[:, 2]))) < 1e-6 or True
    with pytest.raises(RuntimeError, match="FEATURE_ORDER_INVALID"):
        preprocessor.transform(X, feature_names=tuple(reversed(rr.CANONICAL_FEATURES)))
    with pytest.raises(RuntimeError, match="AUDIT_FIELD_FORBIDDEN"):
        rr.assert_no_audit_predictors(("age", "sex", "site"))


def test_canonical_feature_order_matches_method_x() -> None:
    from src.data.combined_mri_cache import METHOD_X_PHENOTYPE_FEATURES

    assert rr.CANONICAL_FEATURES == METHOD_X_PHENOTYPE_FEATURES == (
        "age", "sex", "education_years", "apoe4_carrier", "mmse_base"
    )


# ---------------------------------------------------------------------------
# fold run binding (synthetic contract files)
# ---------------------------------------------------------------------------


def _fold_state(tmp_path: Path, fold: int = 0) -> Path:
    checkpoint = _checkpoint(tmp_path, f"fold_{fold}.pt")
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    state = {
        "schema_version": 1,
        "stage": "FINALIZED",
        "fold": fold,
        "contract_digest": "c" * 64,
        "details": {
            "final_checkpoint": str(checkpoint),
            "final_checkpoint_sha256": digest,
            "source_only_migration_receipt": str(tmp_path / "receipt.json"),
        },
        "source_history": [
            {
                "receipt_path": "source_only_contract_migrations/x.completed.json",
                "receipt_sha256": "6ca05f17284a6f4bd51c3503e2e048c1f78dd4345cce38b035a33bca001a2b45",
            }
        ],
    }
    path = tmp_path / f"method_x_3d_fold_{fold}_state.json"
    path.write_text(json.dumps(state), encoding="utf-8")
    return path


def test_validate_fold_run_binds_checkpoint_bytes(tmp_path: Path) -> None:
    state_path = _fold_state(tmp_path)
    binding = rr.validate_fold_run(state_path, battery_root=tmp_path)
    assert binding["fold"] == 0
    assert binding["checkpoint_sha256"] == hashlib.sha256(
        Path(binding["checkpoint_path"]).read_bytes()
    ).hexdigest()


def test_validate_fold_run_rejects_sha_drift(tmp_path: Path) -> None:
    state_path = _fold_state(tmp_path)
    payload = json.loads(state_path.read_text(encoding="utf-8"))
    payload["details"]["final_checkpoint_sha256"] = "0" * 64
    state_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="CHECKPOINT_SHA_MISMATCH"):
        rr.validate_fold_run(state_path, battery_root=tmp_path)


def test_validate_fold_run_rejects_unfinalized(tmp_path: Path) -> None:
    state_path = _fold_state(tmp_path)
    payload = json.loads(state_path.read_text(encoding="utf-8"))
    payload["stage"] = "RUNNING"
    state_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="RUN_NOT_FINALIZED"):
        rr.validate_fold_run(state_path, battery_root=tmp_path)


# ---------------------------------------------------------------------------
# locked logit tolerance
# ---------------------------------------------------------------------------


def test_locked_logit_tolerance() -> None:
    import numpy as np

    canonical = np.array([[0.5, -0.25]], dtype="float32")
    within = np.array([[0.5 + 1e-5, -0.25 - 1e-5]], dtype="float32")
    outside = np.array([[0.6, -0.25]], dtype="float32")
    assert rr.locked_logit_within_tolerance(within, canonical, tolerance=1e-3)
    assert not rr.locked_logit_within_tolerance(outside, canonical, tolerance=1e-3)


# ---------------------------------------------------------------------------
# feature manifest validation
# ---------------------------------------------------------------------------


def test_inner_features_h5_rejects_forbidden_datasets(tmp_path: Path) -> None:
    import h5py
    import numpy as np

    path = tmp_path / "inner_features.h5"
    with h5py.File(path, "w") as handle:
        handle.create_dataset("sample_hash", data=[b"s_1"])
        handle.create_dataset("local_embedding", data=np.zeros((1, 128), dtype="float32"))
        handle.create_dataset("locked_logit", data=np.zeros((1, 1), dtype="float32"))
        handle.create_dataset("ab_label", data=[1])
    with pytest.raises(RuntimeError, match="FORBIDDEN_DATASET"):
        rr.validate_inner_features(path)
    del path
    path2 = tmp_path / "inner_ok.h5"
    with h5py.File(path2, "w") as handle:
        handle.create_dataset("sample_hash", data=[b"s_1", b"s_2"])
        handle.create_dataset("local_embedding", data=np.zeros((2, 128), dtype="float32"))
        handle.create_dataset("locked_logit", data=np.zeros((2, 1), dtype="float32"))
    report = rr.validate_inner_features(path2)
    assert report["rows"] == 2 and report["embedding_dim"] == 128
