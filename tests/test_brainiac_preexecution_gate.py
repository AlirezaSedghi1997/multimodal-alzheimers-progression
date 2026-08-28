from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def test_novelty_matrix_v3_recomputes_live_registry_without_claim_widening() -> None:
    from scripts.research.build_brainiac_novelty_matrix_v3 import build_matrix

    report = build_matrix()
    assert report["status"] == "R0_COMPLETE_NOVELTY_CANDIDATE"
    assert report["registry_sha256"] == (
        "e397b9dd03653bf7d4c3b80e470b8ffa9389b40e04a2208530bb5eb5c0a7cd8f"
    )
    assert report["total_sources_evaluated"] == 29
    assert report["peer_reviewed_sources_count"] == 13
    assert report["full_text_reviewed_count"] == 28
    assert report["prior_art_exact_match_found"] is False
    assert report["claim_ceiling"] == "NOVELTY_CANDIDATE"


@pytest.mark.parametrize(
    "action",
    [
        "CAFT_TRAINING",
        "OUTER_FEATURE_SNAPSHOT",
        "PROMOTION",
        "EXTERNAL_EVALUATION",
    ],
)
def test_live_brainiac_downstream_execution_remains_hard_disabled(
    action: str,
) -> None:
    from src.research.brainiac_execution_gate import (
        BrainIACExecutionBlocked,
        assert_brainiac_execution_allowed,
    )

    with pytest.raises(
        BrainIACExecutionBlocked,
        match="BRAINIAC_EXECUTION_HARD_DISABLED",
    ):
        assert_brainiac_execution_allowed(action)


def test_embedding_script_checks_gate_before_source_or_output_io(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scripts.research import extract_brainiac_embeddings as script

    monkeypatch.setattr(script, "CACHE_PATH", tmp_path / "missing-cache.h5")
    monkeypatch.setattr(script, "DERIVED_DIR", tmp_path / "must-not-exist")
    monkeypatch.setattr(script, "OUT_H5_PATH", tmp_path / "must-not-exist" / "out.h5")
    monkeypatch.setattr(
        script,
        "MANIFEST_PATH",
        tmp_path / "must-not-exist" / "manifest.json",
    )

    with pytest.raises(
        RuntimeError,
        match="BRAINIAC_LEGACY_CACHE128_EXTRACTOR_PERMANENTLY_DISABLED",
    ):
        script.extract_all_embeddings(limit=1)
    assert not (tmp_path / "must-not-exist").exists()


def test_simulated_caft_benchmark_is_permanently_invalidated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scripts.research import train_brainiac_caft_benchmark as script

    target = tmp_path / "result.json"
    monkeypatch.setattr(script, "RESULTS_PATH", target)
    with pytest.raises(RuntimeError, match="BRAINIAC_SIMULATED_BENCHMARK_FORBIDDEN"):
        script.run_caft_ablation_benchmark()
    assert not target.exists()


def test_model_loader_requires_execution_gate_before_checkpoint_io(
    tmp_path: Path,
) -> None:
    from src.research.brainiac_model_loader import load_brainiac_encoder

    with pytest.raises(RuntimeError, match="BRAINIAC_EXECUTION_HARD_DISABLED"):
        load_brainiac_encoder(device="cpu")


def test_invalidation_receipt_binds_both_existing_non_scientific_artifacts() -> None:
    receipt_path = (
        ROOT
        / "results"
        / "method_x_registry"
        / "BRAINIAC_PREEXECUTION_INVALIDATION_20260821.json"
    )
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["status"] == "INVALID_NON_SCIENTIFIC_ARTIFACTS_RETAINED"
    assert receipt["training_performed"] is False
    assert receipt["outer_test_touched"] is False
    assert len(receipt["artifacts"]) == 3
    for row in receipt["artifacts"]:
        path = ROOT / row["path"]
        assert path.is_file()
        assert path.stat().st_size == row["bytes"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == row["sha256"]
        assert row["scientific_use_allowed"] is False


def test_p08_status_separates_r0_completion_from_training_readiness(
    tmp_path: Path, monkeypatch,
) -> None:
    """R0 completion enables bounded preparation, never the five-CNN training gate."""

    from scripts import build_brainiac_p08_status as p08
    from scripts.research.build_brainiac_novelty_matrix_v3 import build_matrix

    novelty = tmp_path / "novelty_matrix_v3.json"
    novelty.write_text(
        json.dumps(build_matrix(), sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    monkeypatch.setattr(p08, "NOVELTY", novelty)
    report = p08.build_status()
    assert report["status"] == "R0_COMPLETE_PREPARATION_PARTIAL_NONBLOCKING_FOR_MAIN_CNN"
    assert report["battery_facing_preexecution_gate"] == "PASS"
    assert report["main_cnn_battery_blocked_by_brainiac"] is False
    assert report["brainiac_execution_allowed"] is True
    assert report["brainiac_training_allowed"] is False
    assert report["r0"]["r0_status"] == "R0_COMPLETE"
    assert report["r0"]["citation_corroboration"] == "CITATION_CORROBORATION_PARTIAL"
    assert report["r0"]["full_text_reviewed"] == 28
    assert report["r0"]["fully_verified"] is False
    assert report["r1"]["status"] == "BLOCKED"
    assert report["r1"]["authorized_actions"] == ["PREPARE_RAW_T1"]
    assert report["training_performed"] is False
    assert report["outer_test_touched"] is False


def test_future_ready_state_requires_hash_chained_readiness_and_bound_inputs(
) -> None:
    """Dynamic readiness comes from the canonical append-only ledger head."""

    from src.research.brainiac_execution_gate import (
        DEFAULT_READINESS_LEDGER,
        assess_brainiac_execution_gate,
    )
    from src.research.brainiac_readiness_ledger import validate_readiness_ledger

    report = assess_brainiac_execution_gate("PREPARE_RAW_T1")
    ledger = validate_readiness_ledger(DEFAULT_READINESS_LEDGER, project_root=ROOT)
    assert report["allowed"] is True
    assert report["blockers"] == []
    assert report["readiness_ledger_head_sha256"] == ledger["head_sha256"]


def test_public_gate_exposes_no_dynamic_authority_override() -> None:
    import inspect

    from src.research.brainiac_execution_gate import assess_brainiac_execution_gate

    assert tuple(inspect.signature(assess_brainiac_execution_gate).parameters) == (
        "action",
    )


@pytest.mark.parametrize(
    "anchor_code",
    [
        "BRAINIAC_LEDGER_ANCHOR_BASELINE_NOT_PINNED",
        "BRAINIAC_LEDGER_ANCHOR_BASELINE_MISSING",
        "BRAINIAC_LEDGER_ANCHOR_HMAC_INVALID",
        "BRAINIAC_LEDGER_ANCHOR_FORK",
    ],
)
def test_gate_blocks_when_os_anchor_is_unpinned_missing_or_tampered(
    monkeypatch: pytest.MonkeyPatch, anchor_code: str
) -> None:
    """A valid H hash chain alone must never authorize protected action I/O."""

    from src.research import brainiac_execution_gate as gate

    def _failing_anchor(**kwargs):
        raise RuntimeError(anchor_code)

    monkeypatch.setattr(gate, "validate_production_anchor", _failing_anchor)
    with pytest.raises(
        gate.BrainIACExecutionBlocked,
        match="READINESS_LEDGER_ANCHOR_INVALID",
    ):
        gate.assess_brainiac_execution_gate("PREPARE_RAW_T1")


@pytest.mark.parametrize("bound_name", ["predecessor_scope", "battery_manifest"])
def test_ready_state_fails_closed_when_a_bound_authority_changes(
    tmp_path: Path,
    bound_name: str,
) -> None:
    """A READY decision cannot survive drift in either authority it evaluated."""

    from src.research.brainiac_execution_gate import DEFAULT_R1_READINESS
    from src.research.brainiac_readiness import (
        validate_materialized_readiness,
        write_readiness_artifact,
    )

    payload = json.loads(DEFAULT_R1_READINESS.read_text(encoding="utf-8"))
    payload["input_bindings"][bound_name]["sha256"] = "0" * 64
    readiness = tmp_path / "readiness.json"
    receipt = write_readiness_artifact(readiness, payload)
    with pytest.raises(RuntimeError, match="READINESS_INPUT_BINDING_MISMATCH"):
        validate_materialized_readiness(
            readiness,
            project_root=ROOT,
            expected_sha256=receipt["sha256"],
        )
