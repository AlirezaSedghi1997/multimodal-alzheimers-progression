from __future__ import annotations

import copy
from pathlib import Path

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[2]


def _config() -> dict:
    config = yaml.safe_load(
        (ROOT / "configs" / "method_y_image_training_v3_2.yaml").read_text(
            encoding="utf-8"
        )
    )
    config = copy.deepcopy(config)
    config["model"]["hpo_candidates"] = [
        {
            "candidate_id": "candidate_0",
            "learning_rate": 1e-4,
            "weight_decay": 1e-4,
            "complexity_rank": 0,
        }
    ]
    config["training_source_sha256"] = "1" * 64
    config["software_runtime_contract"] = {"contract_sha256": "2" * 64}
    return config


def _orchestrator(tmp_path):
    from src.method_y_revival.image_training_orchestrator_v3 import (
        ImageTrainingOrchestratorV3,
    )

    return ImageTrainingOrchestratorV3(
        config=_config(),
        output_root=tmp_path / "out",
        branch="amyloid",
        cache_sha="a" * 64,
        membership_sha="b" * 64,
        split_sha="c" * 64,
        readiness_sha="d" * 64,
        allow_synthetic_fixture=True,
    )


def test_source_bundle_binds_every_executable_training_dependency() -> None:
    from src.method_y_revival.software_contract_v3 import SOURCE_PATHS

    required = {
        "scripts/build_method_y_image_training_readiness_v3.py",
        "src/method_y_revival/image_launcher_state_v3.py",
        "src/method_y_revival/survival_models.py",
        "src/method_y_revival/science_gates.py",
        "src/method_y_revival/oof_reporting.py",
        "src/method_y_revival/contracts.py",
        "src/method_y_revival/atomic_io.py",
        "src/method_y_revival/process_guard.py",
    }
    assert required.issubset(set(SOURCE_PATHS))


def test_corrupt_master_state_fails_closed_on_resume(tmp_path) -> None:
    from src.method_y_revival.atomic_io import read_json, atomic_write_json
    from src.method_y_revival.image_training_orchestrator_v3 import (
        ImageTrainingOrchestratorV3,
        ImageTrainingOrchestratorV3Error,
    )

    first = _orchestrator(tmp_path)
    state = read_json(first.master_state_path)
    state["state"] = "BRANCH_OUTER_CV_COMPLETE"
    atomic_write_json(first.master_state_path, state)

    with pytest.raises(ImageTrainingOrchestratorV3Error, match="MASTER_STATE_HASH_MISMATCH"):
        ImageTrainingOrchestratorV3(
            config=_config(),
            output_root=tmp_path / "out",
            branch="amyloid",
            cache_sha="a" * 64,
            membership_sha="b" * 64,
            split_sha="c" * 64,
            readiness_sha="d" * 64,
            allow_synthetic_fixture=True,
        )


def test_existing_winner_contract_mismatch_is_terminal(tmp_path) -> None:
    from src.method_y_revival.atomic_io import atomic_write_json
    from src.method_y_revival.image_training_orchestrator_v3 import (
        ImageTrainingOrchestratorV3Error,
        read_verified_winner_record,
        write_winner_record,
    )

    candidates = [
        {
            "candidate_id": "candidate_0",
            "mean_ibs": 0.2,
            "standard_error": 0.01,
            "complexity_rank": 0,
            "status": "COMPLETED",
        }
    ]
    path = tmp_path / "winner.json"
    write_winner_record(
        path,
        branch="amyloid",
        variant="scalar_tabular",
        outer_fold=0,
        master_contract_sha256="a" * 64,
        winner=candidates[0],
        candidates=candidates,
    )
    with pytest.raises(ImageTrainingOrchestratorV3Error, match="RUN_CONTRACT_MISMATCH:winner"):
        read_verified_winner_record(
            path,
            expected_branch="amyloid",
            expected_variant="scalar_tabular",
            expected_outer_fold=0,
            expected_master_contract_sha256="b" * 64,
        )

    payload = __import__("json").loads(path.read_text(encoding="utf-8"))
    payload["winner"]["candidate_id"] = "tampered"
    atomic_write_json(path, payload)
    with pytest.raises(ImageTrainingOrchestratorV3Error, match="WINNER_RECORD_HASH_MISMATCH"):
        read_verified_winner_record(
            path,
            expected_branch="amyloid",
            expected_variant="scalar_tabular",
            expected_outer_fold=0,
            expected_master_contract_sha256="a" * 64,
        )


def test_committed_outer_result_recovers_hashed_private_mirror_without_reaccess(
    tmp_path,
) -> None:
    from src.method_y_revival.image_training_orchestrator_v3 import (
        read_verified_private_outer_record,
    )
    from src.method_y_revival.outer_evaluation_control import OuterFoldController

    orchestrator = _orchestrator(tmp_path)
    contract = "e" * 64
    outer_root = (
        orchestrator.output_root
        / "private"
        / "outer_evaluation"
        / "amyloid"
        / "scalar_tabular"
    )
    controller = OuterFoldController(outer_root, fold=0, run_contract_sha256=contract)
    controller.lock_checkpoint("f" * 64)
    expected = {
        "integrated_brier_score": 0.2,
        "uno_c_index": 0.61,
        "oof_predictions": [],
    }
    assert controller.evaluate_once(lambda: expected) == expected
    controller.finalize()

    calls = 0

    def must_not_reaccess():
        nonlocal calls
        calls += 1
        raise AssertionError("outer test was accessed twice")

    recovered = orchestrator.outer_evaluate(
        outer_fold=0,
        variant="scalar_tabular",
        evaluator=must_not_reaccess,
        run_contract=contract,
    )
    assert recovered == expected
    assert calls == 0
    mirror = orchestrator.private_root / "scalar_tabular" / "fold_0_private.json"
    record = read_verified_private_outer_record(
        mirror,
        expected_branch="amyloid",
        expected_variant="scalar_tabular",
        expected_outer_fold=0,
        expected_run_contract_sha256=contract,
    )
    assert record["metrics"] == expected


def test_private_outer_mirror_tampering_is_rejected(tmp_path) -> None:
    from src.method_y_revival.atomic_io import atomic_write_json, read_json
    from src.method_y_revival.image_training_orchestrator_v3 import (
        ImageTrainingOrchestratorV3Error,
        read_verified_private_outer_record,
        write_private_outer_record,
    )

    path = tmp_path / "fold_0_private.json"
    write_private_outer_record(
        path,
        branch="tau",
        variant="scalar_tabular_plus_pattern_image",
        outer_fold=0,
        run_contract_sha256="a" * 64,
        metrics={"integrated_brier_score": 0.3},
    )
    payload = read_json(path)
    payload["metrics"]["integrated_brier_score"] = 0.01
    atomic_write_json(path, payload)
    with pytest.raises(ImageTrainingOrchestratorV3Error, match="PRIVATE_OUTER_RECORD_HASH_MISMATCH"):
        read_verified_private_outer_record(
            path,
            expected_branch="tau",
            expected_variant="scalar_tabular_plus_pattern_image",
            expected_outer_fold=0,
            expected_run_contract_sha256="a" * 64,
        )


def test_launcher_has_no_committed_state_short_circuit_before_mirror_recovery() -> None:
    source = (ROOT / "scripts" / "run_method_y_image_training_v3.py").read_text(
        encoding="utf-8"
    )
    assert "SKIPPED_ALREADY_COMMITTED" not in source


def test_final_checkpoint_verification_binds_bytes_and_internal_contract(tmp_path) -> None:
    import torch
    from scripts.verify_method_y_image_training_v3 import (
        FinalVerificationV3Error,
        _verify_locked_checkpoint,
    )

    checkpoint = tmp_path / "locked_checkpoint.pt"
    torch.save(
        {
            "run_contract_digest": "a" * 64,
            "model_state": {"weight": torch.tensor([1.0])},
        },
        checkpoint,
    )
    import hashlib

    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    _verify_locked_checkpoint(
        checkpoint,
        expected_file_sha256=digest,
        expected_run_contract_sha256="a" * 64,
    )
    with pytest.raises(FinalVerificationV3Error, match="CHECKPOINT_INTERNAL_CONTRACT_MISMATCH"):
        _verify_locked_checkpoint(
            checkpoint,
            expected_file_sha256=digest,
            expected_run_contract_sha256="b" * 64,
        )
