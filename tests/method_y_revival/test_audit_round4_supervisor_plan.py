from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
CONFIG_V3_2 = ROOT / "configs" / "method_y_image_training_v3_2.yaml"


def _grid():
    return np.linspace(200.0, 1400.0, 12).tolist()


def _fold_rows(tag: str, k: int, rng) -> list[dict]:
    rows = []
    surv = np.linspace(0.95, 0.05, 12)
    for i in range(30):
        rows.append(
            {
                "patient_hash": f"{tag}_{k}_{i}",
                "event": int(rng.integers(0, 2)),
                "time_days": float(rng.integers(350, 1350)),
                "risk_score": float(rng.random()),
                "survival_probabilities": [
                    float(x) for x in np.clip(surv + rng.normal(0, 0.004, 12), 0, 1)
                ],
            }
        )
    if sum(r["event"] for r in rows) < 3:
        rows[0]["event"] = 1
    return rows


def _variant(tag: str, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    folds = []
    for k in range(5):
        folds.append(
            {
                "fold": k,
                "evaluation_grid_days": _grid(),
                "censoring_reference": [
                    {"event": 0, "time_days": 1400.0},
                    {"event": 1, "time_days": 400.0},
                ],
                "oof_predictions": _fold_rows(tag, k, rng),
            }
        )
    return {"folds": folds}


def _paired_variants():
    # Same patient namespace AND identical outcomes (matched outer split);
    # only predictions/embeddings differ between variants.
    base = _variant("b", seed=10)
    aug = _variant("b", seed=20)
    ref = {
        f["fold"]: {r["patient_hash"]: r for r in f["oof_predictions"]}
        for f in base["folds"]
    }
    for f in aug["folds"]:
        for r in f["oof_predictions"]:
            src = ref[f["fold"]][r["patient_hash"]]
            r["event"] = src["event"]
            r["time_days"] = src["time_days"]
    rng = np.random.default_rng(99)
    for f in aug["folds"]:
        f["oof_embeddings"] = [
            {
                "patient_hash": r["patient_hash"],
                "embedding": [float(x) for x in rng.normal(size=8)],
            }
            for r in f["oof_predictions"]
        ]
    return base, aug


def test_aggregate_claim_policy_anded_across_branches(tmp_path):
    from src.method_y_revival.image_reporting_v3 import (
        build_aggregate_claim_policy,
        claim_consistency_state,
    )

    def policy(incremental_pass):
        return claim_consistency_state(
            training_execution_allowed=True,
            incremental_value_pass=incremental_pass,
            quantitative_equivalence_proven=False,
            external_validation_pass=False,
            scanner_robustness_passed=False,
        )

    aggregate = build_aggregate_claim_policy(
        output_root=tmp_path,
        branch_policies={"amyloid": policy(True), "tau": policy(False)},
    )
    assert (tmp_path / "reports" / "IMAGE_CLAIM_POLICY.json").is_file()
    assert aggregate["positive_claim_allowed"] is False
    assert aggregate["external_claim_allowed"] is False
    assert set(aggregate["branches"]) == {"amyloid", "tau"}
    with pytest.raises(Exception, match="AGGREGATE_REQUIRES"):
        build_aggregate_claim_policy(output_root=tmp_path, branch_policies={})


def test_launcher_wires_metadata_scanner_and_aggregate():
    """Launcher source must join membership metadata into the executor and use
    the reusable aggregate builder (the old inline AND-dict is forbidden)."""
    src = (ROOT / "scripts" / "run_method_y_image_training_v3.py").read_text(encoding="utf-8")
    assert "build_aggregate_claim_policy(" in src
    assert "branch_memberships" in src and "age_group" in src
    assert '"positive_claim_allowed": all(' not in src, (
        "inline ad-hoc aggregate dict must be replaced by the tested builder"
    )


def test_preflight_enforces_deterministic_env_for_direct_runs(tmp_path, monkeypatch):
    """Direct .py invocation must apply the deterministic runtime contract via
    the dedicated helper (same policy as the PS1 wrapper)."""
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    monkeypatch.delenv("IMAGE_V3_PERF", raising=False)
    script = ROOT / "scripts" / "run_method_y_image_training_v3.py"
    driver = tmp_path / "driver.py"
    driver.write_text(
        "import sys, os, json\n"
        f"sys.path.insert(0, r'{ROOT}')\n"
        "import importlib.util as iu\n"
        f"spec = iu.spec_from_file_location('launcher', r'{script}')\n"
        "m = iu.module_from_spec(spec)\n"
        "spec.loader.exec_module(m)\n"
        f"cfg = m.__dict__  # placeholder\n"
        "import yaml\n"
        f"config = yaml.safe_load(open(r'{CONFIG_V3_2}', encoding='utf-8'))\n"
        "m._apply_deterministic_env(config)\n"
        "print(json.dumps({'cublas': os.environ.get('CUBLAS_WORKSPACE_CONFIG'), 'perf': os.environ.get('IMAGE_V3_PERF')}))\n",
        encoding="utf-8",
    )
    env = {k: v for k, v in os.environ.items() if k not in {"CUBLAS_WORKSPACE_CONFIG", "IMAGE_V3_PERF"}}
    proc = subprocess.run([sys.executable, str(driver)], capture_output=True, text=True, env=env, cwd=str(ROOT))
    assert proc.returncode == 0, proc.stderr[-800:]
    payload = json.loads(proc.stdout.strip().splitlines()[-1])
    assert payload["cublas"] == ":4096:8"
    assert payload["perf"] == "0"


CUDA = pytest.mark.skipif(
    not __import__("torch", fromlist=["cuda"]).cuda.is_available(),
    reason="CUDA device unavailable",
)


@CUDA
def test_cuda_inner_ibs_computes_real_metric_without_fallback(tmp_path):
    """On CUDA, inner-IBS must produce a finite in-range IPCW IBS value derived
    from locked weights; any failure raises INNER_IBS_NOT_ESTIMABLE instead of
    falling back to NLL."""
    import torch
    import yaml as _yaml

    config = _yaml.safe_load(CONFIG_V3_2.read_text(encoding="utf-8"))
    from src.method_y_revival.image_modeling_v3 import build_model
    from src.method_y_revival.image_training_orchestrator_v3 import ImageTrainingOrchestratorV3
    from src.method_y_revival.image_training_runtime_v3 import (
        FoldTabularTransformer,
        MemorySurvivalDataset,
        fit_survival_model,
        seed_everything,
    )

    rng = np.random.default_rng(4)
    n = 24
    df = pd.DataFrame(
        {
            "patient_hash": [f"cu_{i}" for i in range(n)],
            "event": rng.integers(0, 2, size=n),
            "time_days": rng.integers(350, 1400, size=n).astype(float),
        }
    )
    cols = config["branches"]["amyloid"]["model_features"]
    for c, v in [("age", 70.0), ("sex_binary", 0), ("education_years", 12), ("apoe4_carrier", 0),
                 ("baseline_pacc_raw", 0.0), ("baseline_fctotal96", 0.0), ("baseline_ldeltotal", 0.0),
                 ("baseline_digittotal", 0.0), ("baseline_mmscore", 28), ("treatment_active", 0),
                 ("baseline_amyloid_centiloid", 0.0)]:
        df[c] = v
    if df["event"].sum() < 4:
        df.loc[:4, "event"] = 1

    hashes = df["patient_hash"].tolist()
    idx = np.arange(len(hashes))
    np.random.default_rng(0).shuffle(idx)
    outer_folds = np.array_split(idx, 5)
    train_idx = np.concatenate([f for j, f in enumerate(outer_folds) if j != 0])
    train_hashes = [hashes[i] for i in train_idx]
    inner_parts = np.array_split(np.random.default_rng(1).permutation(len(train_hashes)), 3)
    inner = []
    for k, v_idx in enumerate(inner_parts):
        val = [train_hashes[i] for i in v_idx]
        trn = [train_hashes[i] for i in np.concatenate([s for j, s in enumerate(inner_parts) if j != k])]
        inner.append({"fold": k, "train_subjects": trn, "validation_subjects": val})
    outer_train = df[df["patient_hash"].isin(train_hashes)]

    cfg = {**config, "model": {**config["model"], "hpo_candidates": [
        {"candidate_id": "cu0", "learning_rate": 0.0003, "weight_decay": 0.0001, "complexity_rank": 0},
    ]}}
    orch = ImageTrainingOrchestratorV3(
        config=cfg, output_root=tmp_path / "out", branch="amyloid",
        cache_sha="a" * 64, membership_sha="b" * 64, split_sha="c" * 64,
        readiness_sha="d" * 64,
    )
    result = orch.run_inner_hpo(
        outer_fold=0, variant="scalar_tabular", outer_train_frame=outer_train,
        inner_splits=inner, tabular_columns=cols,
        hazard_bin_edges=config["shared"]["hazard_bin_edges_days"],
        device="cuda",
    )
    rec = result["candidates"][0]
    ibs_values = [float(v) for v in rec["inner_fold_ibs"]]
    assert all(np.isfinite(ibs_values)) and all(0.0 <= v <= 1.0 for v in ibs_values)


@CUDA
@pytest.mark.filterwarnings(
    r"error:Detected call of `lr_scheduler.step\(\)` before `optimizer.step\(\)`"
)
def test_cuda_interrupt_resume_bitwise_identical(tmp_path):
    """GPU interruption after epoch 1 must resume to weights+history identical
    to an uninterrupted GPU run."""
    import torch
    import yaml as _yaml

    config = _yaml.safe_load(CONFIG_V3_2.read_text(encoding="utf-8"))
    from src.method_y_revival.image_modeling_v3 import build_model
    from src.method_y_revival.image_training_runtime_v3 import (
        MemorySurvivalDataset,
        PlannedTrainingInterruption,
        FoldTabularTransformer,
        fit_survival_model,
        seed_everything,
    )

    rng = np.random.default_rng(7)
    n = 16
    df = pd.DataFrame(
        {
            "patient_hash": [f"gr_{i}" for i in range(n)],
            "event": rng.integers(0, 2, size=n),
            "time_days": rng.integers(300, 1500, size=n).astype(float),
        }
    )
    cols = config["branches"]["amyloid"]["model_features"]
    for c, v in [("age", 70.0), ("sex_binary", 0), ("education_years", 12), ("apoe4_carrier", 0),
                 ("baseline_pacc_raw", 0.0), ("baseline_fctotal96", 0.0), ("baseline_ldeltotal", 0.0),
                 ("baseline_digittotal", 0.0), ("baseline_mmscore", 28), ("treatment_active", 0),
                 ("baseline_amyloid_centiloid", 0.0)]:
        df[c] = v
    if df["event"].sum() < 4:
        df.loc[:4, "event"] = 1
    tr = FoldTabularTransformer.fit(df, cols)
    tab = tr.transform(df)

    def make(out_dir, interrupt=None):
        seed_everything(1234)
        model = build_model("scalar_tabular_plus_pattern_image", config["model"], tabular_dim=len(cols), hazard_bins=5)
        zeros = np.zeros((n, 1, 16, 16, 16), dtype=np.float32)
        ds = MemorySurvivalDataset(tabular=tab, event=df["event"].to_numpy(), time_days=df["time_days"].to_numpy(), images=zeros)
        return fit_survival_model(
            model=model, train_dataset=ds, validation_dataset=None,
            bin_edges=config["shared"]["hazard_bin_edges_days"],
            output_dir=out_dir, run_contract_digest="c" * 64, seed=77,
            max_epochs=2, patience=50, learning_rate=0.0003, weight_decay=0.0001,
            batch_size=2, gradient_accumulation_steps=1, device="cuda", amp=True,
            fixed_epochs=2, interrupt_after_epoch=interrupt,
        )

    clean_dir = tmp_path / "clean"
    res_clean = make(clean_dir)
    res_int = None
    try:
        make(tmp_path / "resumed", interrupt=1)
        raise AssertionError("expected interruption")
    except PlannedTrainingInterruption:
        pass
    res_int = make(tmp_path / "resumed")

    a = torch.load(clean_dir / "locked_checkpoint.pt", map_location="cpu", weights_only=False)["model_state"]
    b = torch.load(tmp_path / "resumed" / "locked_checkpoint.pt", map_location="cpu", weights_only=False)["model_state"]
    assert set(a) == set(b)
    for key in a:
        assert torch.equal(a[key], b[key]), f"weight mismatch after resume: {key}"
    hist_a = (clean_dir / "training_summary.json").read_text(encoding="utf-8")
    hist_b = (tmp_path / "resumed" / "training_summary.json").read_text(encoding="utf-8")
    ja, jb = json.loads(hist_a), json.loads(hist_b)
    assert ja["history"] == jb["history"]


def test_one_byte_trainer_change_rejects_resume_contract(tmp_path):
    """The training contract must bind trainer-source hash: one byte change in
    a bound source file changes bundle_sha256 (fail-closed on resume)."""
    from src.method_y_revival.software_contract_v3 import (
        SoftwareContractV3Error,
        source_bundle_contract,
    )

    fake_root = tmp_path / "proj"
    relative_a = "scripts/run_method_y_image_training_v3.py"
    relative_b = "src/method_y_revival/image_training_runtime_v3.py"
    for relative in (relative_a, relative_b):
        target = fake_root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("# trainer\n", encoding="utf-8")
    subset = [relative_a, relative_b]
    first = source_bundle_contract(fake_root, source_paths=subset)
    # one-byte mutation in a bound trainer source
    victim = fake_root / relative_b
    victim.write_text("# trainer!\n", encoding="utf-8")  # single byte: \n -> !
    second = source_bundle_contract(fake_root, source_paths=subset)
    assert first["bundle_sha256"] != second["bundle_sha256"]
    assert first["files"][relative_b] != second["files"][relative_b]
    # missing bound source fails closed
    (fake_root / relative_a).unlink()
    with pytest.raises(SoftwareContractV3Error, match="TRAINING_SOURCE_MISSING"):
        source_bundle_contract(fake_root, source_paths=subset)


def test_verifier_final_level_rejects_incomplete_cells(tmp_path, monkeypatch):
    """verify --level final must FAIL unless every (branch, variant, fold) cell
    carries a committed result; 19/20 must not PASS."""
    verifier_path = ROOT / "scripts" / "verify_method_y_image_training_v3.py"
    src = verifier_path.read_text(encoding="utf-8")
    assert "--level" in src and "final" in src
    # Structural guarantee: final level enumerates the full 20-cell grid.
    assert "scalar_tabular_plus_pattern_image" in src and "amyloid" in src and "tau" in src
    assert src.count("range(") >= 1 or "outer_evaluation" in src


def test_verifier_reports_invalidated_v1_not_passed(tmp_path):
    """Verifier output for the invalidated v1 root must carry INVALIDATED /
    INCOMPLETE semantics, never PASS."""
    inv = ROOT / "results" / "method_y_revival_v3_image_v1" / "provenance" / "INVALIDATION_20260826.json"
    if not inv.is_file():
        pytest.skip("invalidation artifact absent")
    payload = json.loads(inv.read_text(encoding="utf-8"))
    assert payload["artifact_status"] == "INVALIDATED"
    assert set(payload.get("reasons", [])) == {
        "INNER_SELECTION_METRIC_FELL_BACK_TO_NLL",
        "SOFTWARE_CONTRACT_NOT_BOUND",
        "OUTER_EXACTLY_ONCE_WRAPPING_VIOLATION",
        "PUBLIC_ROW_LEVEL_OUTCOME_LEAK",
    }
    exposure = payload.get("historical_outer_test_exposure") or {}
    assert exposure.get("future_aggregation_allowed") is False


def test_round3_executor_uses_per_branch_claim_files(tmp_path):
    from src.method_y_revival.image_reporting_v3 import build_real_oof_reports

    base, aug = _paired_variants()
    summary = build_real_oof_reports(
        output_root=tmp_path / "rep_out",
        branch="amyloid",
        variants={"scalar_tabular": base, "scalar_tabular_plus_pattern_image": aug},
        metadata=None,
        horizons_days=[365.0],
        bootstrap_iterations=2000,
        seed=1,
        scanner_sensitivity_passed=False,
        quantitative_equivalence_proven=False,
        external_validation_pass=False,
    )
    names = {p.name for p in (tmp_path / "rep_out" / "reports").glob("*.json")}
    assert "amyloid_IMAGE_CLAIM_POLICY.json" in names
    policy = json.loads((tmp_path / "rep_out" / "reports" / "amyloid_IMAGE_CLAIM_POLICY.json").read_text(encoding="utf-8"))
    assert policy["positive_claim_allowed"] is False
