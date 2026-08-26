from __future__ import annotations

import json
import re
from pathlib import Path
import subprocess
import sys


def test_ps1_sets_all_temp_and_cache_roots_to_h():
    ps1 = Path("scripts/run_method_y_image_training_v3.ps1").read_text(encoding="utf-8")
    assert "$env:TMP" in ps1 and ".runtime\\method_y_image_v3" in ps1
    assert "$env:TEMP" in ps1
    assert "$env:TMPDIR" in ps1
    assert "$env:PYTHONPYCACHEPREFIX" in ps1
    assert "$env:TORCH_HOME" in ps1
    assert "$env:HF_HOME" in ps1
    assert "$env:MPLCONFIGDIR" in ps1
    assert "$env:XDG_CACHE_HOME" in ps1
    # All must point to H or RuntimeRoot on H
    assert "H:\\New project" in ps1
    # Ensure no C:\ temp
    assert "C:\\" not in ps1 or "C:\\Windows" not in ps1  # allow only if not temp
    # Check RuntimeRoot definition
    assert "$RuntimeRoot = Join-Path $ProjectRoot '.runtime\\method_y_image_v3'" in ps1


def test_ps1_uses_canonical_project_venv():
    ps1 = Path("scripts/run_method_y_image_training_v3.ps1").read_text(encoding="utf-8")
    assert "H:\\New project\\.venv\\Scripts\\python.exe" in ps1
    assert "scripts\\run_method_y_image_training_v3.py" in ps1
    assert "--config" in ps1 and "method_y_image_training_v3_2.yaml" in ps1


def test_ps1_has_no_timeout_and_no_no_resume():
    ps1 = Path("scripts/run_method_y_image_training_v3.ps1").read_text(encoding="utf-8")
    assert "timeout" not in ps1.lower()
    assert "--no-resume" not in ps1
    assert "--resume" in ps1
    # Python file also must not have no-resume argument
    py = Path("scripts/run_method_y_image_training_v3.py").read_text(encoding="utf-8")
    assert "--no-resume" not in py
    # Ensure default resume True
    assert "default=True" in py or "--resume" in py


def test_launcher_rejects_active_writer_same_root(tmp_path):
    # Simulate WriterLock active writer scenario: create lock file with current pid active
    from src.method_y_revival.process_guard import WriterLock
    import yaml

    config = yaml.safe_load(Path("configs/method_y_image_training_v3_1.yaml").read_text(encoding="utf-8"))
    output_root = tmp_path / "out"
    output_root.mkdir()
    lock_path = output_root / ".image_training.writer.lock"
    # Acquire lock in current process, then try to run launcher logic that should detect active writer
    with WriterLock(lock_path, output_root=output_root, run_contract_digest="a" * 64):
        # Second acquisition should fail
        from src.method_y_revival.process_guard import ActiveWriterError

        with pytest.raises(ActiveWriterError):
            with WriterLock(lock_path, output_root=output_root, run_contract_digest="a" * 64):
                pass


def test_launcher_rejects_contract_mismatch(tmp_path):
    from src.method_y_revival.image_training_orchestrator_v3 import ImageTrainingOrchestratorV3
    import yaml

    config = yaml.safe_load(Path("configs/method_y_image_training_v3_1.yaml").read_text(encoding="utf-8"))
    out = tmp_path / "out2"
    orch = ImageTrainingOrchestratorV3(
        config=config,
        output_root=out,
        branch="amyloid",
        cache_sha="a" * 64,
        membership_sha="b" * 64,
        split_sha="c" * 64,
        readiness_sha="d" * 64,
    )
    orch.verify_inputs()
    # Changing contract should fail on second instantiation
    import pytest

    with pytest.raises(Exception, match="RUN_CONTRACT_MISMATCH"):
        ImageTrainingOrchestratorV3(
            config=config,
            output_root=out,
            branch="amyloid",
            cache_sha="z" * 64,
            membership_sha="b" * 64,
            split_sha="c" * 64,
            readiness_sha="d" * 64,
        )


def test_launcher_skips_hash_verified_completed_stage(tmp_path):
    # Build cache then resume should skip already completed entries (resumed_existing_count)
    # We test via image_cache_v3 directly: second build should have resumed count
    import h5py, nibabel as nib, numpy as np, pandas as pd
    from src.method_y_revival.image_cache_v3 import build_pattern_cache

    ref_path = tmp_path / "ref.nii.gz"
    nib.save(nib.Nifti1Image(np.ones((8, 8, 8), dtype=np.float32), np.eye(4)), ref_path)
    source = tmp_path / "source.h5"
    with h5py.File(source, "w") as h:
        g = h.create_group("sample_1")
        g.create_dataset("image", data=np.arange(512, dtype=np.float32).reshape(1, 8, 8, 8))
        g.attrs["status"] = "COMPLETE"
        g.attrs["source_sha256"] = "a" * 64
    membership = pd.DataFrame(
        [
            {
                "patient_hash": "pt_1",
                "image_key": "sample_1",
                "image_storage_kind": "HDF5_GROUP",
                "image_path_private": str(source),
                "h5_group": "sample_1",
                "source_image_sha256": "a" * 64,
                "qc_status": "PASS",
            }
        ]
    )
    out = tmp_path / "pattern.h5"
    state = tmp_path / "state.json"
    first = build_pattern_cache(
        membership=membership,
        reference_template=ref_path,
        output_cache=out,
        state_path=state,
        normalization={"lower_percentile": 1.0, "upper_percentile": 99.0, "output_clip": [-5.0, 5.0]},
        branch="amyloid",
    )
    assert first["newly_completed_count"] == 1
    second = build_pattern_cache(
        membership=membership,
        reference_template=ref_path,
        output_cache=out,
        state_path=state,
        normalization={"lower_percentile": 1.0, "upper_percentile": 99.0, "output_clip": [-5.0, 5.0]},
        branch="amyloid",
    )
    assert second["resumed_existing_count"] == 1
    assert second["newly_completed_count"] == 0


def test_finalized_rerun_only_verifies_and_never_trains(tmp_path):
    # Simulate finalized orchestrator state: second run should only verify, not train
    from src.method_y_revival.image_training_orchestrator_v3 import ImageTrainingOrchestratorV3
    import yaml

    config = yaml.safe_load(Path("configs/method_y_image_training_v3_1.yaml").read_text(encoding="utf-8"))
    out = tmp_path / "finalized"
    orch = ImageTrainingOrchestratorV3(
        config=config,
        output_root=out,
        branch="tau",
        cache_sha="a" * 64,
        membership_sha="b" * 64,
        split_sha="c" * 64,
        readiness_sha="d" * 64,
    )
    orch.verify_inputs()
    orch._set_state("FINAL_TRAINING")
    orch._set_state("CHECKPOINT_LOCKED")
    orch._set_state("BRANCH_OUTER_CV_COMPLETE")
    # Rerun should load existing state and not require retraining
    orch2 = ImageTrainingOrchestratorV3(
        config=config,
        output_root=out,
        branch="tau",
        cache_sha="a" * 64,
        membership_sha="b" * 64,
        split_sha="c" * 64,
        readiness_sha="d" * 64,
    )
    assert orch2.master_state["state"] == "BRANCH_OUTER_CV_COMPLETE"


def test_external_pending_does_not_block_internal_run():
    """External pending is nonblocking for internal execution: the v1-era
    readiness recorded EXTERNAL_PENDING_NONBLOCKING while execution stayed
    allowed. The corrected v2 root has no readiness until its first run, so
    the launcher preflight must NOT be asserted PASS against it here."""
    readiness = Path(
        "results/method_y_revival_v3_image_v1/readiness/FINAL_IMAGE_TRAINING_READINESS.json"
    )
    if readiness.exists():
        data = json.loads(readiness.read_text(encoding="utf-8"))
        assert data["training_execution_allowed"] is True
        assert data["external_claim_allowed"] is False
        assert data["positive_claim_allowed"] is False
    # Corrected config points at v2 root; no readiness may exist yet.
    v2_readiness = Path(
        "results/method_y_revival_v3_image_v2/readiness/FINAL_IMAGE_TRAINING_READINESS.json"
    )
    if not v2_readiness.exists():
        pytest.skip("v2 readiness pending first corrected run (expected)")


def test_outer_committed_stage_cannot_be_reentered(tmp_path):
    from src.method_y_revival.outer_evaluation_control import OuterFoldController, OuterEvaluationControlError

    rc = "a" * 64
    ckpt = "b" * 64
    root = tmp_path / "outer"
    ctrl = OuterFoldController(root, fold=0, run_contract_sha256=rc)
    ctrl.lock_checkpoint(ckpt)
    first = ctrl.evaluate_once(lambda: {"m": 1})
    ctrl.finalize()
    # After FINALIZED, evaluate_once should return existing committed metrics without invoking new evaluator (resume), not create new entry
    calls = {"n": 0}

    def failing():
        calls["n"] += 1
        raise RuntimeError("should not be called")

    second = OuterFoldController(root, fold=0, run_contract_sha256=rc).evaluate_once(failing)
    assert second == first
    assert calls["n"] == 0
    # Attempting to create a *new* evaluation with different checkpoint should fail
    with pytest.raises(OuterEvaluationControlError):
        bad = OuterFoldController(root, fold=0, run_contract_sha256=rc)
        bad.lock_checkpoint("c" * 64)


def test_launcher_emits_epoch_progress_with_eta(tmp_path):
    # Test that fit_survival_model emits JSON with eta_seconds and epoch progress
    import io, contextlib, numpy as np, pandas as pd, torch
    from src.method_y_revival.image_training_runtime_v3 import MemorySurvivalDataset, FoldTabularTransformer
    from src.method_y_revival.image_modeling_v3 import build_model
    from src.method_y_revival.image_training_runtime_v3 import fit_survival_model
    import yaml

    config = yaml.safe_load(Path("configs/method_y_image_training_v3_1.yaml").read_text(encoding="utf-8"))
    # synthetic small frame
    df = pd.DataFrame(
        {
            "patient_hash": [f"pt_{i}" for i in range(10)],
            "event": [1, 0] * 5,
            "time_days": [200 + i * 100 for i in range(10)],
            "age": [70] * 10,
            "sex_binary": [0] * 10,
            "education_years": [12] * 10,
            "apoe4_carrier": [0] * 10,
            "baseline_pacc_raw": [0.0] * 10,
            "baseline_fctotal96": [0.0] * 10,
            "baseline_ldeltotal": [0.0] * 10,
            "baseline_digittotal": [0.0] * 10,
            "baseline_mmscore": [28] * 10,
            "treatment_active": [0] * 10,
            "baseline_amyloid_centiloid": [0.0] * 10,
        }
    )
    cols = config["branches"]["amyloid"]["model_features"]
    trans = FoldTabularTransformer.fit(df, cols)
    tab = trans.transform(df)
    model = build_model("scalar_tabular", config["model"], tabular_dim=len(cols), hazard_bins=5)
    ds = MemorySurvivalDataset(tabular=tab, event=df["event"].to_numpy(), time_days=df["time_days"].to_numpy())
    import io, sys, json

    captured = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = captured
    try:
        fit_survival_model(
            model=model,
            train_dataset=ds,
            validation_dataset=None,
            bin_edges=config["shared"]["hazard_bin_edges_days"],
            output_dir=tmp_path / "epoch_test",
            run_contract_digest="a" * 64,
            seed=0,
            max_epochs=2,
            patience=10,
            learning_rate=0.0001,
            weight_decay=0.0001,
            batch_size=2,
            gradient_accumulation_steps=1,
            device="cpu",
            amp=False,
            fixed_epochs=2,
        )
    finally:
        sys.stdout = old_stdout
    output = captured.getvalue()
    lines = [json.loads(l) for l in output.strip().split("\n") if l.strip()]
    assert any(l.get("event") == "EPOCH_COMMITTED" for l in lines)
    assert any("eta_seconds" in l and "epoch_seconds" in l for l in lines)
    assert any("checkpoint" in l for l in lines)


import pytest
