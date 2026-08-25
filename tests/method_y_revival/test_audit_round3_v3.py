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


def _frame(n=24, seed=3):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "patient_hash": [f"r3_{i:03d}" for i in range(n)],
        "event": rng.integers(0, 2, size=n),
        "time_days": rng.integers(300, 1500, size=n).astype(float),
    })
    for c, v in [("age", 70.0), ("sex_binary", 0), ("education_years", 12), ("apoe4_carrier", 0),
                 ("baseline_pacc_raw", 0.0), ("baseline_fctotal96", 0.0), ("baseline_ldeltotal", 0.0),
                 ("baseline_digittotal", 0.0), ("baseline_mmscore", 28), ("treatment_active", 0),
                 ("baseline_amyloid_centiloid", 0.0)]:
        df[c] = v
    if df["event"].sum() < 4:
        df.loc[:4, "event"] = 1
    return df


def test_h5_dataset_lazy_cache_returns_identical_volumes(tmp_path):
    """RAM cache must return byte-identical volumes and actually cache."""
    import h5py, nibabel as nib
    from src.method_y_revival.image_training_runtime_v3 import H5PatternSurvivalDataset

    ref = tmp_path / "ref.nii.gz"
    nib.save(nib.Nifti1Image(np.ones((8, 8, 8), dtype=np.float32), np.eye(4)), ref)
    src = tmp_path / "src.h5"
    with h5py.File(src, "w") as h:
        for i in range(2):
            g = h.create_group(f"pt_{i}")
            g.create_dataset("image", data=(np.full((1, 8, 8, 8), i + 1, dtype=np.float16)))
            g.attrs["status"] = "COMPLETE"
            g.attrs["finite_fraction"] = 1.0
    tab = np.zeros((2, len(COLS)), dtype=np.float32)
    ds = H5PatternSurvivalDataset(
        cache_path=src, patient_hashes=["pt_0", "pt_1"], tabular=tab,
        event=np.array([1, 0]), time_days=np.array([500.0, 700.0]), include_image=True,
    )
    a1 = ds[0]["image"].clone()
    a2 = ds[0]["image"].clone()
    assert torch.equal(a1, a2)
    assert float(a1.max()) == 1.0
    assert len(getattr(ds, "_ram_cache", {})) == 1


def test_h5_dataset_cache_respects_byte_cap(tmp_path):
    """A tiny cap must degrade gracefully to disk reads without error."""
    import h5py, nibabel as nib
    from src.method_y_revival.image_training_runtime_v3 import H5PatternSurvivalDataset

    ref = tmp_path / "r.nii.gz"
    nib.save(nib.Nifti1Image(np.ones((4, 4, 4), dtype=np.float32), np.eye(4)), ref)
    src = tmp_path / "s.h5"
    with h5py.File(src, "w") as h:
        g = h.create_group("p0")
        g.create_dataset("image", data=np.ones((1, 4, 4, 4), dtype=np.float16))
        g.attrs["status"] = "COMPLETE"
    ds = H5PatternSurvivalDataset(
        cache_path=src, patient_hashes=["p0"], tabular=np.zeros((1, len(COLS)), dtype=np.float32),
        event=np.array([0]), time_days=np.array([400.0]), include_image=True,
        ram_cache_max_bytes=8,
    )
    _ = ds[0]["image"]
    assert len(ds._ram_cache) == 0  # cap exceeded -> not cached, still served from disk


def test_inner_ibs_must_load_locked_best_weights_strictly(tmp_path):
    """Silent fallback to last-epoch weights is forbidden: corrupt locked ckpt must raise."""
    from src.method_y_revival.image_training_orchestrator_v3 import (
        ImageTrainingOrchestratorV3Error,
        load_locked_state_for_eval,
    )

    victim = tmp_path / "locked_checkpoint.pt"
    victim.write_bytes(b"not-a-real-checkpoint")
    with pytest.raises(ImageTrainingOrchestratorV3Error, match="INNER_IBS_CHECKPOINT_LOAD_FAILED"):
        load_locked_state_for_eval(victim, "a" * 64)
    with pytest.raises(ImageTrainingOrchestratorV3Error, match="INNER_IBS_CHECKPOINT_LOAD_FAILED"):
        load_locked_state_for_eval(tmp_path / "missing.pt", "a" * 64)

    # Contract mismatch must also fail closed.
    import torch as _t
    real = tmp_path / "real.pt"
    _t.save({"run_contract_digest": "b" * 64, "model_state": {}}, real)
    with pytest.raises(ImageTrainingOrchestratorV3Error, match="INNER_IBS_CHECKPOINT_LOAD_FAILED"):
        load_locked_state_for_eval(real, "a" * 64)


def test_real_oof_report_executor_writes_public_reports_and_gate(tmp_path):
    """Plan 012 executor must consume real OOF folds and emit public aggregates."""
    from src.method_y_revival.image_reporting_v3 import build_real_oof_reports

    rng = np.random.default_rng(0)
    grid = np.linspace(200, 1400, 12).tolist()

    # Matched pair: identical subjects per fold (real outer split guarantee);
    # only predictions differ between variants.
    def mk_variant(tag):
        folds = []
        for k in range(5):
            rows = []
            for i in range(30):
                surv = np.linspace(0.95, 0.05, 12) * (1 - 0.02 * k) + rng.normal(0, 0.005, 12).clip(-0.01, 0.01)
                rows.append({
                    "patient_hash": f"shared_{k}_{i}",
                    "event": int(rng.integers(0, 2)),
                    "time_days": float(rng.integers(350, 1350)),
                    "risk_score": float(rng.random()),
                    "survival_probabilities": [float(x) for x in np.clip(surv, 0, 1)],
                })
            if sum(r["event"] for r in rows) < 3:
                rows[0]["event"] = 1
            folds.append({"fold": k, "evaluation_grid_days": grid,
                          "censoring_reference": [{"event": 0, "time_days": 1400.0}, {"event": 1, "time_days": 400.0}],
                          "oof_predictions": rows})
        return {"folds": folds}

    base = mk_variant("b")
    aug = mk_variant("a")
    # Enforce paired outcomes: same event/time per shared subject.
    base_rows = {f["fold"]: {r["patient_hash"]: r for r in f["oof_predictions"]} for f in base["folds"]}
    for f in aug["folds"]:
        ref = base_rows[f["fold"]]
        for r in f["oof_predictions"]:
            r["event"] = ref[r["patient_hash"]]["event"]
            r["time_days"] = ref[r["patient_hash"]]["time_days"]
    out = tmp_path / "rep"
    summary = build_real_oof_reports(
        output_root=out, branch="amyloid",
        variants={"scalar_tabular": base, "scalar_tabular_plus_pattern_image": aug},
        metadata=None, horizons_days=[365.0], bootstrap_iterations=2000, seed=1,
        scanner_sensitivity_passed=False, quantitative_equivalence_proven=False,
        external_validation_pass=False,
    )
    assert summary["status"] in {"COMPLETED_FROM_REAL_OOF", "PARTIAL"}
    rep_dir = out / "reports"
    names = {p.name for p in rep_dir.glob("*.json")}
    assert "amyloid_image_incremental.json" in names
    assert "amyloid_calibration.json" in names
    assert "amyloid_subgroups.json" in names
    assert "IMAGE_CLAIM_POLICY.json" in names
    policy = json.loads((rep_dir / "IMAGE_CLAIM_POLICY.json").read_text(encoding="utf-8"))
    assert policy["positive_claim_allowed"] is False
    incremental = json.loads((rep_dir / "amyloid_image_incremental.json").read_text(encoding="utf-8"))
    assert incremental["science_gate"]["claim_status"] in {"EXPLORATORY", "INTERNAL_ONLY"}


def test_launcher_stage_completion_requires_executor_status_not_file_presence():
    src = Path("scripts/run_method_y_image_training_v3.py").read_text(encoding="utf-8")
    assert "build_real_oof_reports" in src
    assert 'state.set_stage("ROBUSTNESS_COMPLETE", status="COMPLETED_FROM_REAL_OOF")' not in src.split("build_real_oof_reports")[0]
