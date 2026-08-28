"""Regression tests for Plan 016 execution-profile freshness."""

from pathlib import Path

import pytest


def _leg(role: str, *, fresh_run: bool | None, elapsed: float) -> dict:
    from scripts.research import prepare_brainiac_raw_t1_v2 as runner

    samples = [f"{index:064x}" for index in range(12)]
    digest_map = {sample: "a" * 64 for sample in samples}
    leg = {
        "role": role,
        "root_relative_path": f"data/derived/{role}",
        "root_tree_sha256": "b" * 64,
        "sample_hashes": samples,
        "selection_sha256": runner.canonical_sha(samples),
        "receipt_file_sha256_by_sample": digest_map,
        "output_sha256_by_sample": digest_map,
        "transform_sha256_by_sample": digest_map,
        "canonical_scientific_sha256_by_sample": digest_map,
        "contract_sha256": "c" * 64,
        "started_at_utc": "2026-08-28T00:00:00+00:00",
        "finished_at_utc": "2026-08-28T00:01:00+00:00",
        "elapsed_seconds": elapsed,
        "resume_event_count": 0,
        "timing_receipt_sha256": "d" * 64,
        "integrity_status": "PASS",
        "live_evidence_status": "PASS_RECOMPUTED_FROM_LIVE_BYTES",
    }
    if fresh_run is not None:
        leg["fresh_run"] = fresh_run
    return leg


def test_fresh_run_missing_rejects():
    from scripts.research import prepare_brainiac_raw_t1_v2 as runner

    sequential = _leg("sequential", fresh_run=None, elapsed=10.0)
    with pytest.raises(RuntimeError, match="FRESH_RUN_EVIDENCE_MISSING"):
        runner.build_execution_profile(
            sequential=sequential,
            parallel_runs=[
                _leg("parallel_1", fresh_run=True, elapsed=5.0),
                _leg("parallel_2", fresh_run=True, elapsed=5.0),
            ],
            expected_contract_sha256="c" * 64,
        )


def test_fresh_run_false_never_parallel_safe():
    from scripts.research import prepare_brainiac_raw_t1_v2 as runner

    profile = runner.build_execution_profile(
        sequential=_leg("sequential", fresh_run=True, elapsed=10.0),
        parallel_runs=[
            _leg("parallel_1", fresh_run=False, elapsed=5.0),
            _leg("parallel_2", fresh_run=True, elapsed=5.0),
        ],
        expected_contract_sha256="c" * 64,
    )
    assert profile["execution_mode"] == "SEQUENTIAL_SAFE"
    assert profile["timing_fresh"] is False


def test_profile_root_missing_rejects(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from scripts.research import prepare_brainiac_raw_t1_v2 as runner

    monkeypatch.setattr(
        runner,
        "_execution_profile_roots",
        lambda: {"sequential": tmp_path / "missing"},
    )
    with pytest.raises(RuntimeError, match="LIVE_ROOT_INVALID"):
        runner.recompute_execution_leg(
            role="sequential", expected_contract_sha256="c" * 64
        )


def test_stop_flag_propagates_to_all_stages():
    import inspect
    from scripts.research.prepare_brainiac_raw_t1_v2 import run_equivalence, _calibrate_if_needed, run_sensitivity

    for func in [run_equivalence, _calibrate_if_needed, run_sensitivity]:
        sig = inspect.signature(func)
        assert "stop_requested" in sig.parameters or "stop" in str(sig), f"{func.__name__} missing stop_requested"
