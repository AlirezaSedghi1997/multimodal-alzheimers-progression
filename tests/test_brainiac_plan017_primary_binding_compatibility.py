from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path

import pytest

from scripts.research import prepare_brainiac_raw_t1_v2 as runner
from scripts.research import (
    seal_brainiac_plan017_primary_binding_compatibility_v3 as compatibility,
)
from scripts.research import (
    seal_brainiac_plan017_calibration_reconciliation_v4 as reconciliation_v4,
)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def test_primary_and_historical_evidence_use_distinct_provenance_bindings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    threshold = tmp_path / "threshold.json"
    threshold.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(runner, "THRESHOLD_MANIFEST_PATH", threshold)
    current = {
        "dependency_closure_sha256": "c" * 64,
        "qc_threshold_manifest_sha256": runner.sha256_file(threshold),
    }
    monkeypatch.setattr(runner, "_base_bindings", lambda: dict(current))
    monkeypatch.setattr(
        runner,
        "_plan017_evidence_dependency_closure",
        lambda fallback: runner.PLAN017_LOCKED_EVIDENCE_DEPENDENCY_CLOSURE_SHA256,
    )

    historical = runner._equivalence_receipt_bindings()
    threshold_evidence = runner._threshold_evidence_bindings()

    assert current["dependency_closure_sha256"] == "c" * 64
    assert historical["dependency_closure_sha256"] == (
        runner.PLAN017_LOCKED_EVIDENCE_DEPENDENCY_CLOSURE_SHA256
    )
    assert threshold_evidence["dependency_closure_sha256"] == (
        runner.PLAN017_LOCKED_EVIDENCE_DEPENDENCY_CLOSURE_SHA256
    )
    assert threshold_evidence["qc_threshold_manifest_sha256"] == runner.sha256_file(
        threshold
    )
    assert historical["qc_threshold_manifest_sha256"] != runner.sha256_file(
        threshold
    )


def test_compatibility_rejects_unapproved_dependency_or_scientific_drift() -> None:
    old = {
        "closure_sha256": compatibility.OLD_DEPENDENCY_CLOSURE_SHA256,
        "python_closure": {
            compatibility.RUNNER_RELATIVE_PATH: compatibility.OLD_RUNNER_SHA256,
            "unchanged.py": _sha("unchanged"),
        },
        "dynamic_code_inputs": {"vendor": {"sha256": _sha("vendor")}},
        "static_inputs": {"policy": {"sha256": _sha("policy")}},
    }
    new = {
        **old,
        "closure_sha256": compatibility.NEW_DEPENDENCY_CLOSURE_SHA256,
        "python_closure": {
            compatibility.RUNNER_RELATIVE_PATH: compatibility.NEW_RUNNER_SHA256,
            "unchanged.py": _sha("unchanged"),
        },
    }
    compatibility.assert_dependency_transition(old, new)
    new["static_inputs"] = {"policy": {"sha256": _sha("changed-policy")}}
    with pytest.raises(RuntimeError, match="SCIENTIFIC_INPUT_DRIFT"):
        compatibility.assert_dependency_transition(old, new)


def test_critical_preprocessing_function_hashes_ignore_binding_only_change() -> None:
    old = """
def make_real_pipeline(value):
    return value + 1

def run_primary():
    bindings = old_bindings()
"""
    new = old.replace("old_bindings", "new_bindings")
    compatibility.assert_critical_preprocessing_functions_unchanged(old, new)

    changed = new.replace("return value + 1", "return value + 2")
    with pytest.raises(RuntimeError, match="PREPROCESSING_FUNCTION_DRIFT"):
        compatibility.assert_critical_preprocessing_functions_unchanged(old, changed)


def test_v3_compatibility_is_an_immutable_historical_link() -> None:
    assert runner.PLAN017_PRIMARY_BINDING_V3_RUNNER_SHA256 == (
        compatibility.NEW_RUNNER_SHA256
    )
    assert runner.PLAN017_PRIMARY_BINDING_V3_DEPENDENCY_CLOSURE_SHA256 == (
        compatibility.NEW_DEPENDENCY_CLOSURE_SHA256
    )


def test_v4_dependency_transition_allows_only_the_runner_source() -> None:
    old = {
        "closure_sha256": reconciliation_v4.OLD_DEPENDENCY_CLOSURE_SHA256,
        "python_closure": {
            reconciliation_v4.RUNNER_RELATIVE_PATH: reconciliation_v4.OLD_RUNNER_SHA256,
            "unchanged.py": _sha("unchanged"),
        },
        "dynamic_code_inputs": {"vendor": {"sha256": _sha("vendor")}},
        "static_inputs": {"policy": {"sha256": _sha("policy")}},
    }
    new = {
        **old,
        "closure_sha256": _sha("new-closure"),
        "python_closure": {
            reconciliation_v4.RUNNER_RELATIVE_PATH: _sha("new-runner"),
            "unchanged.py": _sha("unchanged"),
        },
    }
    monkey = pytest.MonkeyPatch()
    monkey.setattr(
        reconciliation_v4.runner,
        "sha256_file",
        lambda path: _sha("new-runner")
        if Path(path).as_posix().endswith(reconciliation_v4.RUNNER_RELATIVE_PATH)
        else _sha("other"),
    )
    try:
        reconciliation_v4.assert_dependency_transition(old, new)
        new["python_closure"]["unchanged.py"] = _sha("drift")
        with pytest.raises(RuntimeError, match="UNAPPROVED_SOURCE_DRIFT"):
            reconciliation_v4.assert_dependency_transition(old, new)
    finally:
        monkey.undo()


def test_acceptance_requires_v4_compatibility_before_calibration() -> None:
    source = inspect.getsource(runner.run_acceptance)
    compatibility_call = (
        "_validated_calibration_reconciliation_source_compatibility()"
    )
    assert compatibility_call in source
    assert source.index(compatibility_call) < source.index("_calibrate_if_needed")


def test_v4_revalidates_threshold_with_explicit_historical_bindings() -> None:
    validator_source = inspect.getsource(
        runner._validated_calibration_reconciliation_source_compatibility
    )
    sealer_source = inspect.getsource(reconciliation_v4._restore_calibration_tree)
    expected = "_load_bound_threshold_manifest(_threshold_evidence_bindings())"
    assert expected in validator_source.replace("runner.", "")
    assert expected in sealer_source.replace("runner.", "")
