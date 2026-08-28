"""Plan 003 orchestration tests: resume, signal-safety, idempotence, exits.

The real pipeline is monkeypatched with a deterministic synthetic stage so the
full-run control logic is verified hermetically (no GPU, no study data).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import scripts.research.prepare_brainiac_raw_t1_v2 as runner
from src.research import brainiac_preprocessing_v2 as v2
from src.research.brainiac_preprocessing_v2_schema import validate_receipt


@pytest.fixture(autouse=True)
def _isolated(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    monkeypatch.setattr(v2, "assert_no_cnn_writer", lambda: None)

    members = [
        {"sample_hash": f"s_{i:06d}" + hashlib.sha256(str(i).encode()).hexdigest()[:10],
         "t1_file_hash": f"t1{i}"}
        for i in range(12)
    ]
    monkeypatch.setattr(v2, "load_locked_membership", lambda: members)

    # redirect every writable root to the test sandbox
    sandbox = tmp_path / "sandbox"
    monkeypatch.setattr(v2, "PRIMARY_ROOT", sandbox / "primary")
    monkeypatch.setattr(runner, "CALIBRATION_ROOT", sandbox / "calibration")
    monkeypatch.setattr(runner, "THRESHOLD_MANIFEST_PATH",
                        sandbox / "aggregate" / "qc_threshold_manifest.json")
    monkeypatch.setattr(runner, "PRIMARY_AGGREGATE_PATH",
                        sandbox / "aggregate" / "primary_manifest.json")
    monkeypatch.setattr(runner, "RUN_STATE_PATH", sandbox / "run_state.json")
    monkeypatch.setattr(runner, "PREFLIGHT_REPORT", sandbox / "preflight.json")
    monkeypatch.setattr(runner, "SENSITIVITY_REPORT_PATH", sandbox / "sensitivity.json")
    monkeypatch.setattr(runner, "_parallel_unlock_state", lambda: (True, True))
    # heavy/venv-bound preflight pieces are not under test here
    monkeypatch.setattr(runner, "_load_fixed_template", lambda: object())
    monkeypatch.setattr(runner, "_base_bindings", lambda: {
        "policy_sha256": "1" * 64, "config_sha256": "7" * 64,
        "pipeline_contract_sha256": "8" * 64,
        "dependency_closure_sha256": "9" * 64,
        "membership_sha256": "a" * 64, "cache_sha256": "b" * 64,
        "catalog_sha256": "c" * 64, "runtime_lock_sha256": "2" * 64,
        "vendor_tree_sha256": "3" * 64, "template_lock_sha256": "4" * 64,
        "template_sha256": "d" * 64,
        "official_transform_source_sha256": "5" * 64,
        "qc_threshold_manifest_sha256": "6" * 64,
    })
    monkeypatch.setattr(
        runner,
        "_load_bound_threshold_manifest",
        lambda _bindings: {"thresholds": {"foreground_fraction": {"low": 0.01, "high": 0.50}}},
    )
    monkeypatch.setattr(v2, "build_raw_index", lambda **kwargs: {})
    monkeypatch.setattr(v2, "resolve_sources", lambda membership, **kwargs: {})
    monkeypatch.setattr(v2, "build_contract_transform", lambda: (
        object(), {"official_transform_source_sha256": "5" * 64}))
    return sandbox


def _bindings() -> dict[str, str]:
    return {
        "policy_sha256": v2.policy_sha256(),
        "config_sha256": "7" * 64,
        "pipeline_contract_sha256": "8" * 64,
        "dependency_closure_sha256": "9" * 64,
        "membership_sha256": "a" * 64,
        "cache_sha256": "b" * 64,
        "catalog_sha256": "c" * 64,
        "runtime_lock_sha256": "2" * 64,
        "vendor_tree_sha256": "3" * 64,
        "template_lock_sha256": "4" * 64,
        "official_transform_source_sha256": "5" * 64,
        "qc_threshold_manifest_sha256": "6" * 64,
    }


def _write_threshold_manifest():
    from src.research.brainiac_preprocessing_v2_schema import FIXED_QC_LIMITS
    metrics = {f"s_{i}": {"foreground_fraction": 0.2} for i in range(32)}
    manifest = v2.build_threshold_manifest(
        metrics,
        receipt_integrity_sha256s={key: hashlib.sha256(key.encode()).hexdigest() for key in metrics},
        contract_bindings={"policy_sha256": "1" * 64},
        fixed_bounds={"foreground_fraction": FIXED_QC_LIMITS["foreground_fraction"]},
    )
    runner.THRESHOLD_MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    runner.THRESHOLD_MANIFEST_PATH.write_text(json.dumps(manifest), encoding="utf-8")


def _synthetic_pipeline(fail_on: set[str] | None = None):
    import numpy as np

    def pipeline(item, scratch: Path):
        sample_hash = item["sample_hash"]
        if fail_on and sample_hash in fail_on:
            raise RuntimeError("SYNTHETIC_PIPELINE_FAIL")
        output = scratch / "output.nii.gz"
        import nibabel as nib

        array = np.zeros((96, 96, 96), dtype="float32")
        array[32:64, 32:64, 32:64] = 1.0 + (int(sample_hash[2:6], 16) % 7)
        nib.save(nib.Nifti1Image(array, np.eye(4)), str(output))
        (scratch / "transform.tfm").write_bytes(b"#tfm\n")
        return {
            "output_path": output,
            "transform_path": scratch / "transform.tfm",
            "source_sha256": hashlib.sha256(sample_hash.encode()).hexdigest(),
            "source_size": 64,
            "affine_before": [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            "affine_after": [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            "spacing_before": [1.0, 1.0, 1.0],
            "spacing_after": [1.0, 1.0, 1.0],
            "origin_before": [0.0, 0.0, 0.0],
            "origin_after": [0.0, 0.0, 0.0],
            "direction_before": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            "direction_after": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            "com_coordinate_frame": "TEMPLATE_PHYSICAL_MM_RELATIVE",
            "rigid_determinant": 1.0,
            "orthogonality_error": 0.0,
            "rotation_deg": 0.0,
            "translation_norm_mm": 0.0,
            "foreground_fraction": 0.033,
            "finite": True,
            "normalized_nonzero_mean": 0.0,
            "normalized_nonzero_std": 1.0,
            "center_of_mass_before": [48.0, 48.0, 48.0],
            "center_of_mass_after": [48.0, 48.0, 48.0],
            "registration_similarity_before": -1.0,
            "registration_similarity_after": -2.0,
        }
    # fixed QC lower bound is 0.05; patch the fixture to stay inside bounds
    return pipeline


def test_full_run_resumes_from_last_commit(monkeypatch, tmp_path):
    from src.research import brainiac_preprocessing_v2_schema as schema
    import numpy as np
    import nibabel as nib

    # keep synthetic foreground inside the fixed QC window
    monkeypatch.setattr(schema, "FIXED_QC_LIMITS", {
        "rigid_determinant": (0.999, 1.001),
        "foreground_fraction": (0.01, 0.50),
        "normalized_nonzero_std": (0.95, 1.05),
    })

    calls = {"n": 0}

    def pipeline(item, scratch):
        calls["n"] += 1
        return _synthetic_pipeline()(item, scratch)

    _write_threshold_manifest()
    # bypass lease + heavy preflight pieces
    monkeypatch.setattr(v2, "execute", lambda **kwargs: _fake_execute(pipeline, **kwargs))

    stop = runner._StopFlag()
    first = runner.run_primary(device="cpu", stop=stop)
    assert first["status"] == "COMPLETED"
    assert calls["n"] == 12

    calls2 = {"n": 0}

    def pipeline2(item, scratch):
        calls2["n"] += 1
        return _synthetic_pipeline()(item, scratch)

    monkeypatch.setattr(v2, "execute", lambda **kwargs: _fake_execute(pipeline2, **kwargs))
    second = runner.run_primary(device="cpu")
    assert second["status"] == "ALREADY_COMPLETED"
    assert calls2["n"] == 0  # rerun does zero pipeline work


def _fake_execute(pipeline, *, mode, limit, root, device, pipeline_factory,
                  bindings, membership, stop_requested=None,
                  threshold_manifest=None, parallel=False):
    from src.research.brainiac_preprocessing_v2 import run_batch
    return run_batch(
        membership, root,
        pipeline_factory=lambda: pipeline,
        bindings=bindings,
        stop_requested=stop_requested,
        threshold_manifest=threshold_manifest,
    )


def test_signal_stop_finishes_sample_then_exits_75(monkeypatch, tmp_path):
    from src.research import brainiac_preprocessing_v2_schema as schema
    monkeypatch.setattr(schema, "FIXED_QC_LIMITS", {
        "rigid_determinant": (0.999, 1.001),
        "foreground_fraction": (0.01, 0.50),
        "normalized_nonzero_std": (0.95, 1.05),
    })
    _write_threshold_manifest()
    stop = runner._StopFlag()

    def pipeline(item, scratch):
        result = _synthetic_pipeline()(item, scratch)
        stop.requested = True  # signal arrives mid-sample
        return result

    monkeypatch.setattr(v2, "execute", lambda **kwargs: _fake_execute(pipeline, **kwargs))
    report = runner.run_primary(device="cpu", stop=stop)
    assert report["status"] == "PAUSED_RESUMABLE"
    assert report["completed"] == 1  # current sample finished at safe boundary
    state = json.loads(runner.RUN_STATE_PATH.read_text(encoding="utf-8"))
    assert state["resume_semantics"] == "RERUN_SAME_COMMAND_SKIPS_COMMITTED"


def test_threshold_manifest_gate_blocks_run_before_calibration(monkeypatch, tmp_path):
    monkeypatch.setattr(v2, "execute", lambda **kwargs: (_ for _ in ()).throw(
        AssertionError("execute must not be reached")))
    report = runner.run_primary(device="cpu")
    assert report["status"] == "BLOCKED"
    assert report["reason_code"] == "BRAINIAC_V2_THRESHOLD_MANIFEST_REQUIRED_BEFORE_RUN"


def test_calibrate_manifest_without_bound_receipts_is_rejected(monkeypatch, tmp_path):
    monkeypatch.setattr(v2, "select_calibration_samples", lambda values: list(values))
    monkeypatch.setattr(v2, "execute", lambda **kwargs: _fake_execute(
        _synthetic_pipeline(), **kwargs))
    from src.research.brainiac_preprocessing_v2_schema import FIXED_QC_LIMITS
    import tests.test_brainiac_preprocessing_v2 as _t  # noqa: F401
    # build a manifest through the module directly
    metrics = {f"s_{i}": {"foreground_fraction": 0.2} for i in range(32)}
    manifest = v2.build_threshold_manifest(
        metrics,
        receipt_integrity_sha256s={key: hashlib.sha256(key.encode()).hexdigest() for key in metrics},
        contract_bindings={"policy_sha256": "1" * 64},
        fixed_bounds={"foreground_fraction": FIXED_QC_LIMITS["foreground_fraction"]},
    )
    runner.THRESHOLD_MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    runner.THRESHOLD_MANIFEST_PATH.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(RuntimeError, match="NOT_COMMITTED"):
        runner._calibrate_if_needed(device="cpu")


def test_exit_code_mapping():
    assert runner._exit_code_for("COMPLETED") == 0
    assert runner._exit_code_for("ALREADY_COMPLETED") == 0
    assert runner._exit_code_for("PASS") == 0
    assert runner._exit_code_for("PAUSED_RESUMABLE") == 75
    assert runner._exit_code_for("BLOCKED") == 3
    assert runner._exit_code_for("X_CONTRACT_MISMATCH") == 78
