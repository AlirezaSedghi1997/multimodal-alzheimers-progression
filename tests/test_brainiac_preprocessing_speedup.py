from __future__ import annotations

import json

import pytest


def test_pin_itk_threads_respects_env(monkeypatch):
    from scripts.research.prepare_brainiac_raw_t1_v2 import _pin_itk_threads

    calls: list[int] = []

    class FakeSitk:
        class ProcessObject:
            @staticmethod
            def SetGlobalDefaultNumberOfThreads(n: int) -> None:
                calls.append(int(n))

    import sys
    import types

    fake = types.ModuleType("SimpleITK")
    fake.ProcessObject = FakeSitk.ProcessObject
    monkeypatch.setitem(sys.modules, "SimpleITK", fake)
    monkeypatch.setenv("BRAINIAC_V2_ITK_THREADS", "1")
    assert _pin_itk_threads() == 1
    monkeypatch.setenv("BRAINIAC_V2_ITK_THREADS", "0")
    assert _pin_itk_threads() == 1  # clamped to >=1
    assert calls == [1, 1]


def test_scratch_registered_volume_is_uncompressed_nii(tmp_path):
    from scripts.research.prepare_brainiac_raw_t1_v2 import _scratch_paths

    reg, brain = _scratch_paths(tmp_path, "abc")
    assert reg.suffix == ".nii" and not reg.name.endswith(".gz")
    assert brain.name.endswith(".nii.gz")


def test_equivalence_comparator_flags_scientific_field_drift():
    from scripts.research.prepare_brainiac_raw_t1_v2 import _equivalence_mismatches

    base = {
        "sample_hash": "a" * 64,
        "output_sha256": "x",
        "transform_sha256": "y",
        "foreground_fraction": 0.2,
        "rigid_determinant": 1.0,
        "orthogonality_error": 0.0,
        "rotation_deg": 3.0,
        "translation_norm_mm": 10.0,
        "normalized_nonzero_std": 1.0,
    }
    assert _equivalence_mismatches([base], [dict(base)]) == []
    drifted = dict(base)
    drifted["rotation_deg"] = 4.2
    out = _equivalence_mismatches([base], [drifted])
    assert out == [f"{'a' * 10}:rotation_deg"]


def test_parallel_runner_returns_identical_receipts(monkeypatch):
    from src.research import brainiac_preprocessing_v2 as v2

    seen: list[str] = []

    def fake_tx(item, root, *, pipeline, bindings):
        seen.append(item["sample_hash"])
        return {"sample_hash": item["sample_hash"], "ok": True}

    monkeypatch.setattr(v2, "run_sample_transaction", fake_tx)
    items = [{"sample_hash": f"{i:064x}"} for i in range(6)]
    par = v2.run_orchestrated(
        items, root=None, pipeline_factory=lambda: None,
        bindings={}, cpu_workers=2, gpu_workers=1,
    )
    par_seen = list(seen)
    seen.clear()
    seq = v2.run_batch(items, root=None, pipeline_factory=lambda: None, bindings={})
    assert sorted(par_seen) == sorted(i["sample_hash"] for i in items)
    assert [r["sample_hash"] for r in par] == [r["sample_hash"] for r in seq["receipts"]]


def test_policy_keeps_sequential_only_until_equivalence_pass():
    from src.research.brainiac_preprocessing_v2 import POLICY_PATH

    policy = json.loads(POLICY_PATH.read_text(encoding="utf-8"))
    orch = policy["orchestration"]
    assert orch["sequential_only"] is True
    assert orch["parallel_activation"].startswith("REQUIRES_12_SAMPLE_EQUIVALENCE_PASS")
    assert int(orch.get("itk_threads_default", 0)) == 1


def _write_pass_record(registry_root, samples=2):
    import json
    from pathlib import Path

    registry_root = Path(registry_root)
    registry_root.mkdir(parents=True, exist_ok=True)
    body = {"status": "PASS", "samples": samples}
    body["integrity_sha256"] = "z" * 64
    (registry_root / "brainiac_preprocessing_v2_equivalence.json").write_text(
        json.dumps(body), encoding="utf-8"
    )


def test_unlock_requires_equivalence_pass(tmp_path, monkeypatch):
    import scripts.research.prepare_brainiac_raw_t1_v2 as prep

    policy = tmp_path / "policy.json"
    policy.write_text(json.dumps({"orchestration": {"sequential_only": True}}), encoding="utf-8")
    monkeypatch.setattr(prep, "_policy_path", lambda: policy)
    monkeypatch.setattr(prep, "REGISTRY_ROOT", tmp_path)

    with pytest.raises(RuntimeError, match="BRAINIAC_V2_UNLOCK_REQUIRES_EQUIVALENCE_PASS"):
        prep._atomic_flip_sequential_only(False)

    _write_pass_record(tmp_path, samples=12)
    out = prep._atomic_flip_sequential_only(False)
    assert out["sequential_only"] is False and out["unlocked_samples"] == 12
    flipped = json.loads(policy.read_text(encoding="utf-8"))
    assert flipped["orchestration"]["sequential_only"] is False
    assert flipped["orchestration"]["unlocked_by"] == "AUTO_EQUIVALENCE_PASS"


def test_parallel_unlock_state_reads_policy_and_record(tmp_path, monkeypatch):
    import scripts.research.prepare_brainiac_raw_t1_v2 as prep

    policy = tmp_path / "policy.json"
    policy.write_text(json.dumps({"orchestration": {"sequential_only": True}}), encoding="utf-8")
    monkeypatch.setattr(prep, "_policy_path", lambda: policy)
    monkeypatch.setattr(prep, "REGISTRY_ROOT", tmp_path)
    locked, passed = prep._parallel_unlock_state()
    assert locked is True and passed is False
    _write_pass_record(tmp_path, samples=12)
    locked, passed = prep._parallel_unlock_state()
    assert locked is True and passed is True


def test_archive_moves_manifest_and_calibration_root(tmp_path, monkeypatch):
    import scripts.research.prepare_brainiac_raw_t1_v2 as prep

    manifest_dir = tmp_path / "aggregate"
    manifest_dir.mkdir()
    manifest = manifest_dir / "qc_threshold_manifest.json"
    manifest.write_text("{}", encoding="utf-8")
    calib = tmp_path / "calibration"
    calib.mkdir()

    monkeypatch.setattr(prep, "THRESHOLD_MANIFEST_PATH", manifest)
    monkeypatch.setattr(prep, "CALIBRATION_ROOT", calib)
    moved = prep._archive_pre_pin_artifacts()
    assert set(moved) == {"threshold_manifest", "calibration_root"}
    assert not manifest.exists() and not calib.exists()
    assert list(manifest_dir.glob("qc_threshold_manifest.json.pre_pin_*"))
    assert list(tmp_path.glob("calibration.pre_pin_*"))


def test_run_full_gate_fail_closed(monkeypatch):
    import scripts.research.prepare_brainiac_raw_t1_v2 as prep

    def forbidden(*_a, **_k):  # must never be reached on FAIL
        raise AssertionError("calibration reached despite gate FAIL")

    monkeypatch.setattr(prep, "preflight", lambda: {"status": "PASS", "free_disk_gb": 400.0, "free_disk_ok": True})
    monkeypatch.setattr(prep, "_parallel_unlock_state", lambda: (True, False))
    monkeypatch.setattr(prep, "run_equivalence", lambda *, limit, device: {"status": "FAIL", "mismatch_count": 3})
    monkeypatch.setattr(prep, "_calibrate_if_needed", forbidden)
    rc = prep.run_full(device="cuda")
    assert rc == 3


def test_run_full_pass_gate_unlocks_and_completes(monkeypatch):
    import scripts.research.prepare_brainiac_raw_t1_v2 as prep

    observed: dict[str, object] = {}
    monkeypatch.setattr(prep, "preflight", lambda: {"status": "PASS", "free_disk_gb": 400.0, "free_disk_ok": True})
    monkeypatch.setattr(prep, "_parallel_unlock_state", lambda: (True, False))
    monkeypatch.setattr(
        prep, "run_equivalence",
        lambda *, limit, device: (observed.__setitem__("limit", limit), {"status": "PASS"})[1],
    )
    monkeypatch.setattr(
        prep, "_archive_pre_pin_artifacts", lambda: observed.__setitem__("archived", True) or {}
    )
    monkeypatch.setattr(prep, "_calibrate_if_needed", lambda *, device: {"status": "PASS"})
    monkeypatch.setattr(prep, "run_primary", lambda *, device, stop=None: {
        "status": "COMPLETED", "valid_committed_n": 1669})
    monkeypatch.setattr(prep, "run_sensitivity", lambda *, device: {"status": "PASS"})
    monkeypatch.setattr(prep, "_streaming_validate_primary", lambda: {
        "status": "COMPLETED", "valid_committed_n": 1669,
        "aggregate_integrity_sha256": "a" * 64})
    rc = prep.run_full(device="cuda")
    assert rc == 0 and observed["limit"] == 12


def test_run_full_skips_gate_when_already_unlocked(monkeypatch):
    import scripts.research.prepare_brainiac_raw_t1_v2 as prep

    monkeypatch.setattr(prep, "_parallel_unlock_state", lambda: (False, True))

    def forbidden(**_k):
        raise AssertionError("gate ran although already unlocked")

    monkeypatch.setattr(prep, "run_equivalence", forbidden)
    monkeypatch.setattr(prep, "preflight", lambda: {"status": "PASS", "free_disk_gb": 400.0, "free_disk_ok": True})
    monkeypatch.setattr(prep, "_calibrate_if_needed", lambda *, device: {"status": "ALREADY_COMPLETED"})
    monkeypatch.setattr(prep, "run_primary", lambda *, device, stop=None: {
        "status": "COMPLETED", "valid_committed_n": 1669})
    monkeypatch.setattr(prep, "run_sensitivity", lambda *, device: {"status": "PASS"})
    monkeypatch.setattr(prep, "_streaming_validate_primary", lambda: {
        "status": "COMPLETED", "valid_committed_n": 1669,
        "aggregate_integrity_sha256": "b" * 64})
    assert prep.run_full(device="cuda") == 0
