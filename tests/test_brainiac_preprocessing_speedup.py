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

    def fake_tx(item, root, *, pipeline, bindings, **_kwargs):
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


def test_policy_defers_execution_mode_to_immutable_equivalence_profile():
    from src.research.brainiac_preprocessing_v2 import POLICY_PATH

    policy = json.loads(POLICY_PATH.read_text(encoding="utf-8"))
    orch = policy["orchestration"]
    assert orch["runtime_policy_mutation_forbidden"] is True
    assert orch["selection_authority"] == "HASH_LOCKED_EXECUTION_PROFILE_ONLY"
    assert orch["equivalence_design"] == (
        "12_LABEL_BLIND_ONE_SEQUENTIAL_TWO_INDEPENDENT_PARALLEL"
    )
    assert int(orch.get("itk_threads_default", 0)) == 8


def _write_execution_profile(path, mode="SEQUENTIAL_SAFE", contract="c" * 64):
    import scripts.research.prepare_brainiac_raw_t1_v2 as prep

    samples = [f"s_{index:064x}" for index in range(12)]

    def leg(seconds, role):
        receipt_map = {sample: ("a" * 63 + str(index % 10)) for index, sample in enumerate(samples)}
        scientific_map = {sample: ("b" * 63 + str(index % 10)) for index, sample in enumerate(samples)}
        output_map = {sample: ("c" * 63 + str(index % 10)) for index, sample in enumerate(samples)}
        transform_map = {sample: ("d" * 63 + str(index % 10)) for index, sample in enumerate(samples)}
        return {
            "role": role,
            "root_relative_path": f"missing-live-root/{role}",
            "root_tree_sha256": "e" * 64,
            "sample_hashes": samples,
            "selection_sha256": prep.canonical_sha(sorted(samples)),
            "receipt_file_sha256_by_sample": receipt_map,
            "output_sha256_by_sample": output_map,
            "transform_sha256_by_sample": transform_map,
            "canonical_scientific_sha256_by_sample": scientific_map,
            "elapsed_seconds": seconds,
            "contract_sha256": contract,
            "integrity_status": "PASS",
            "live_evidence_status": "PASS_RECOMPUTED_FROM_LIVE_BYTES",
            "fresh_run": True,
            "resume_event_count": 0,
            "started_at_utc": "2026-08-27T00:00:00+00:00",
            "finished_at_utc": "2026-08-27T00:02:00+00:00",
            "timing_receipt_sha256": "f" * 64,
        }

    if mode == "PARALLEL_SAFE":
        parallel_seconds = (70.0, 72.0)
    else:
        parallel_seconds = (95.0, 96.0)
    body = prep.build_execution_profile(
        sequential=leg(100.0, "sequential"),
        parallel_runs=[
            leg(parallel_seconds[0], "parallel_1"),
            leg(parallel_seconds[1], "parallel_2"),
        ],
        expected_contract_sha256=contract,
    )
    assert body["execution_mode"] == mode
    body.update({
        "created_at_utc": "2026-08-27T00:00:00+00:00",
        "claim_ceiling": "POST_HOC_INTERNAL_TECHNICAL_EVIDENCE",
    })
    body["integrity_sha256"] = prep.canonical_sha(
        {key: value for key, value in body.items() if key != "integrity_sha256"}
    )
    path.write_text(json.dumps(body), encoding="utf-8")


def test_parallel_unlock_state_rejects_profile_without_live_roots(tmp_path, monkeypatch):
    import scripts.research.prepare_brainiac_raw_t1_v2 as prep

    profile = tmp_path / "execution_profile.json"
    monkeypatch.setattr(prep, "EXECUTION_PROFILE_PATH", profile)
    monkeypatch.setattr(prep, "_current_equivalence_contract_sha256", lambda: "c" * 64)
    locked, passed = prep._parallel_unlock_state()
    assert locked is True and passed is False
    _write_execution_profile(profile, "SEQUENTIAL_SAFE")
    assert prep._parallel_unlock_state() == (True, False)
    _write_execution_profile(profile, "PARALLEL_SAFE")
    assert prep._parallel_unlock_state() == (True, False)


def test_execution_profile_rejects_fake_sample_count_integrity_and_old_contract(
    tmp_path, monkeypatch
):
    import scripts.research.prepare_brainiac_raw_t1_v2 as prep

    profile = tmp_path / "execution_profile.json"
    monkeypatch.setattr(prep, "EXECUTION_PROFILE_PATH", profile)
    monkeypatch.setattr(prep, "_current_equivalence_contract_sha256", lambda: "c" * 64)
    _write_execution_profile(profile, "SEQUENTIAL_SAFE")
    valid = json.loads(profile.read_text(encoding="utf-8"))

    fake_two = dict(valid)
    fake_two["legs"] = dict(valid["legs"])
    fake_two["legs"]["sequential"] = dict(valid["legs"]["sequential"])
    fake_two["legs"]["sequential"]["sample_hashes"] = fake_two["legs"]["sequential"]["sample_hashes"][:2]
    fake_two["integrity_sha256"] = prep.canonical_sha(
        {key: value for key, value in fake_two.items() if key != "integrity_sha256"}
    )
    profile.write_text(json.dumps(fake_two), encoding="utf-8")
    assert prep._parallel_unlock_state() == (True, False)

    corrupted = dict(valid)
    corrupted["integrity_sha256"] = "0" * 64
    profile.write_text(json.dumps(corrupted), encoding="utf-8")
    assert prep._parallel_unlock_state() == (True, False)

    _write_execution_profile(profile, "SEQUENTIAL_SAFE", contract="d" * 64)
    assert prep._parallel_unlock_state() == (True, False)


def test_runtime_never_mutates_scientific_policy():
    import scripts.research.prepare_brainiac_raw_t1_v2 as prep

    assert not hasattr(prep, "_atomic_flip_sequential_only")


def test_acceptance_pauses_before_calibration_and_never_starts_primary(monkeypatch):
    import scripts.research.prepare_brainiac_raw_t1_v2 as prep

    stop = prep._StopFlag()
    observed: dict[str, object] = {}
    monkeypatch.setattr(
        prep,
        "_preflight_and_reconcile",
        lambda *, device: ({"status": "PASS", "free_disk_ok": True}, 0),
        raising=False,
    )
    monkeypatch.setattr(prep, "_parallel_unlock_state", lambda: (True, False))

    def paused_equivalence(*, limit, device, stop=None, **_kwargs):
        observed["stop"] = stop
        return {"status": "PAUSED_RESUMABLE", "exit_code": 75}

    monkeypatch.setattr(prep, "run_equivalence", paused_equivalence)
    monkeypatch.setattr(
        prep,
        "_calibrate_if_needed",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("calibration started after equivalence pause")
        ),
    )
    monkeypatch.setattr(
        prep,
        "run_primary",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("primary must never start in acceptance")
        ),
    )

    result = prep.run_acceptance(device="cuda", stop=stop)
    assert result["status"] == "PAUSED_RESUMABLE"
    assert result["exit_code"] == 75
    assert observed["stop"] is stop


def test_acceptance_passes_same_stop_flag_to_calibration(monkeypatch, tmp_path):
    import scripts.research.prepare_brainiac_raw_t1_v2 as prep

    stop = prep._StopFlag()
    observed: dict[str, object] = {}
    profile_path = tmp_path / "profile.json"
    profile_path.write_text(json.dumps({"status": "PASS"}), encoding="utf-8")
    monkeypatch.setattr(prep, "EXECUTION_PROFILE_PATH", profile_path)
    monkeypatch.setattr(
        prep,
        "_preflight_and_reconcile",
        lambda *, device: ({"status": "PASS", "free_disk_ok": True}, 0),
    )
    monkeypatch.setattr(prep, "_parallel_unlock_state", lambda: (False, True))
    monkeypatch.setattr(
        prep,
        "validate_execution_profile",
        lambda *args, **kwargs: {"status": "PASS"},
    )
    monkeypatch.setattr(prep, "_current_equivalence_contract_sha256", lambda: "c" * 64)
    monkeypatch.setattr(prep, "_base_bindings", lambda: {})

    def paused_calibration(*, device, stop=None):
        observed["stop"] = stop
        return {"status": "PAUSED_RESUMABLE", "exit_code": 75}

    monkeypatch.setattr(prep, "_calibrate_if_needed", paused_calibration)
    result = prep.run_acceptance(device="cuda", stop=stop)
    assert result["status"] == "PAUSED_RESUMABLE"
    assert observed["stop"] is stop


def test_sensitivity_honors_preexisting_stop_before_any_io():
    import scripts.research.prepare_brainiac_raw_t1_v2 as prep

    stop = prep._StopFlag()
    stop.requested = True
    assert prep.run_sensitivity(device="cuda", stop=stop) == {
        "status": "PAUSED_RESUMABLE",
        "exit_code": 75,
    }


def test_full_never_starts_primary_when_acceptance_is_not_ready(monkeypatch):
    import scripts.research.prepare_brainiac_raw_t1_v2 as prep

    monkeypatch.setattr(
        prep,
        "run_acceptance",
        lambda *, device, stop=None: {
            "status": "BLOCKED",
            "exit_code": 3,
            "readiness_valid": False,
        },
        raising=False,
    )
    monkeypatch.setattr(
        prep,
        "run_primary",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("primary started without readiness")
        ),
    )
    assert prep.run_full(device="cuda") == 3


def test_acceptance_candidate_cannot_create_final_readiness(monkeypatch, tmp_path):
    import scripts.research.prepare_brainiac_raw_t1_v2 as prep

    candidate = tmp_path / "acceptance-candidate.json"
    final = tmp_path / "final-readiness.json"
    monkeypatch.setattr(prep, "ACCEPTANCE_CANDIDATE_PATH", candidate, raising=False)
    monkeypatch.setattr(prep, "READY_REPORT_PATH", final)
    monkeypatch.setattr(
        prep,
        "ACCEPTANCE_PREFLIGHT_SNAPSHOT_PATH",
        tmp_path / "preflight-snapshot.json",
    )
    monkeypatch.setattr(prep, "THRESHOLD_MANIFEST_PATH", tmp_path / "threshold.json")
    monkeypatch.setattr(prep, "DEPENDENCY_MANIFEST_PATH", tmp_path / "dependency.json")
    monkeypatch.setattr(prep, "EXECUTION_PROFILE_PATH", tmp_path / "profile.json")
    monkeypatch.setattr(prep, "SMOKE_REPORT", tmp_path / "smoke.json")
    monkeypatch.setattr(prep, "ACCEPTANCE_TEST_MANIFEST_PATH", tmp_path / "tests.json")
    monkeypatch.setattr(prep, "PRE_REPAIR_LIVE_FREEZE_PATH", tmp_path / "freeze.json")
    for path in (
        prep.THRESHOLD_MANIFEST_PATH,
        prep.DEPENDENCY_MANIFEST_PATH,
        prep.EXECUTION_PROFILE_PATH,
        prep.SMOKE_REPORT,
        prep.ACCEPTANCE_TEST_MANIFEST_PATH,
        prep.PRE_REPAIR_LIVE_FREEZE_PATH,
    ):
        path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(prep.v2, "load_locked_membership", lambda: [])
    monkeypatch.setattr(prep, "_base_bindings", lambda: {})
    monkeypatch.setattr(
        prep, "_current_equivalence_contract_sha256", lambda: "c" * 64
    )
    monkeypatch.setattr(
        prep,
        "validate_execution_profile",
        lambda *_a, **_k: {"execution_mode": "SEQUENTIAL_SAFE"},
    )
    monkeypatch.setattr(
        prep,
        "_validated_acceptance_test_manifest",
        lambda: {"integrity_sha256": "t" * 64},
    )
    monkeypatch.setattr(prep.v2, "PRIMARY_ROOT", tmp_path / "primary")
    monkeypatch.setattr(
        prep,
        "_acceptance_scoped_git_state",
        lambda: ("branch", "a" * 40, []),
    )

    report = prep.build_preprocessing_readiness_candidate(
        preflight_report={"status": "PASS"},
        execution_profile={},
        calibration_report={"status": "PASS", "selected_n": 32},
        smoke_report={
            "status": "PASS",
            "rows": 12,
            "physical_qc_validated_from_live_receipts": True,
        },
    )
    assert report["status"] == "ACCEPTANCE_CANDIDATE_PASS"
    assert report["readiness_candidate"] is True
    assert candidate.is_file()
    assert not final.exists()


def test_full_requires_final_readiness_after_acceptance_candidate(monkeypatch):
    import scripts.research.prepare_brainiac_raw_t1_v2 as prep

    monkeypatch.setattr(
        prep,
        "run_acceptance",
        lambda *, device, stop=None: {
            "status": "PASS",
            "readiness_candidate": True,
            "exit_code": 0,
        },
    )
    monkeypatch.setattr(
        prep,
        "validate_final_preprocessing_readiness",
        lambda: (_ for _ in ()).throw(RuntimeError("FINAL_READINESS_MISSING")),
        raising=False,
    )
    monkeypatch.setattr(
        prep,
        "run_primary",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("primary started from acceptance candidate")
        ),
    )
    assert prep.run_full(device="cuda") == 3


def test_run_full_gate_fail_closed(monkeypatch):
    import scripts.research.prepare_brainiac_raw_t1_v2 as prep

    def forbidden(*_a, **_k):  # must never be reached on FAIL
        raise AssertionError("calibration reached despite gate FAIL")

    monkeypatch.setattr(
        prep,
        "run_acceptance",
        lambda *, device, stop=None: {
            "status": "BLOCKED",
            "readiness_valid": False,
            "exit_code": 3,
        },
    )
    monkeypatch.setattr(prep, "run_primary", forbidden)
    rc = prep.run_full(device="cuda")
    assert rc == 3


def test_run_full_pass_gate_unlocks_and_completes(monkeypatch):
    import scripts.research.prepare_brainiac_raw_t1_v2 as prep

    observed: dict[str, object] = {}
    monkeypatch.setattr(
        prep,
        "run_acceptance",
        lambda *, device, stop=None: {
            "status": "PASS",
            "readiness_candidate": True,
            "exit_code": 0,
        },
    )
    monkeypatch.setattr(
        prep, "validate_final_preprocessing_readiness", lambda: {"status": "PASS"}
    )
    monkeypatch.setattr(prep, "run_primary", lambda *, device, stop=None: (
        observed.__setitem__("primary", True),
        {"status": "COMPLETED", "valid_committed_n": 1669},
    )[1])
    monkeypatch.setattr(prep, "run_sensitivity", lambda *, device, stop=None: {"status": "PASS"})
    monkeypatch.setattr(prep, "_streaming_validate_primary", lambda: {
        "status": "COMPLETED", "valid_committed_n": 1669,
        "aggregate_integrity_sha256": "a" * 64})
    rc = prep.run_full(device="cuda")
    assert rc == 0 and observed["primary"] is True


def test_run_full_skips_gate_when_already_unlocked(monkeypatch):
    import scripts.research.prepare_brainiac_raw_t1_v2 as prep

    def forbidden(**_k):
        raise AssertionError("gate ran although already unlocked")

    monkeypatch.setattr(prep, "run_equivalence", forbidden)
    monkeypatch.setattr(
        prep,
        "run_acceptance",
        lambda *, device, stop=None: {
            "status": "PASS",
            "readiness_candidate": True,
            "exit_code": 0,
        },
    )
    monkeypatch.setattr(
        prep, "validate_final_preprocessing_readiness", lambda: {"status": "PASS"}
    )
    monkeypatch.setattr(prep, "run_primary", lambda *, device, stop=None: {
        "status": "COMPLETED", "valid_committed_n": 1669})
    monkeypatch.setattr(prep, "run_sensitivity", lambda *, device, stop=None: {"status": "PASS"})
    monkeypatch.setattr(prep, "_streaming_validate_primary", lambda: {
        "status": "COMPLETED", "valid_committed_n": 1669,
        "aggregate_integrity_sha256": "b" * 64})
    assert prep.run_full(device="cuda") == 0
