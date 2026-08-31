from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.research import prepare_brainiac_raw_t1_v2 as runner
from scripts.research import (
    seal_brainiac_plan017_primary_binding_compatibility_v3 as compatibility,
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


def test_compatibility_live_runner_matches_exact_allowlist() -> None:
    assert compatibility.runner.sha256_file(
        compatibility.ROOT / compatibility.RUNNER_RELATIVE_PATH
    ) == compatibility.NEW_RUNNER_SHA256
