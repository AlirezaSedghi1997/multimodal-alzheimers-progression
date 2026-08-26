from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from src.method_y_revival.contracts import sha256_mapping
from src.method_y_revival.review_receipt import (
    EVIDENCE_SCHEMA_VERSION,
    synthetic_layer_result,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _pin_canonical_project_root(monkeypatch, tmp_path: Path) -> None:
    import scripts.run_method_y_training_v2 as launcher

    monkeypatch.setattr(launcher, "SCRIPT_PROJECT_ROOT", tmp_path.resolve())


def _write_locked_inputs(root: Path) -> tuple[Path, Path, Path]:
    task4 = root / "pretraining_contracts"
    task4.mkdir(parents=True)
    reviewed_task4_names = (
        "PRETRAINING_CONTRACTS_TASK4.md",
        "endpoint_verification_v2.json",
        "external_intake_contract_v2.json",
        "predictor_temporality_contract_v2.json",
        "scanner_feasibility_and_sensitivity_contract_v2.json",
        "split_verification_v2.json",
    )
    for name in reviewed_task4_names:
        (task4 / name).write_text(f"locked:{name}\n", encoding="utf-8")
    reviewed_hashes = {name: _sha(task4 / name) for name in reviewed_task4_names}
    implementation_commit = "a" * 40
    run_contract_sha256 = "b" * 64
    evidence_root = task4 / "review_evidence"
    evidence_root.mkdir()
    layer_rows = []
    layer_specs = {
        "INVARIANT_TEST_SUITE": {
            "relative_path": "review_evidence/invariant_test_suite_v2.json",
        },
        "DETERMINISTIC_REBUILD": {
            "relative_path": "review_evidence/deterministic_rebuild_v2.json",
        },
        "ADVERSARIAL_AGENT_REVIEW": {
            "relative_path": "review_evidence/adversarial_agent_review_v2.json",
        },
    }
    for layer_id, spec in layer_specs.items():
        evidence = {
            "schema_version": EVIDENCE_SCHEMA_VERSION,
            "layer_id": layer_id,
            "status": "PASS",
            "implementation_commit": implementation_commit,
            "run_contract_sha256": run_contract_sha256,
            "reviewed_artifact_sha256": reviewed_hashes,
            **synthetic_layer_result(layer_id, reviewed_hashes),
        }
        evidence["evidence_contract_sha256"] = sha256_mapping(evidence)
        evidence_path = task4 / spec["relative_path"]
        evidence_path.write_text(json.dumps(evidence, sort_keys=True), encoding="utf-8")
        layer_rows.append(
            {
                "layer_id": layer_id,
                "status": "PASS",
                "evidence_relative_path": spec["relative_path"],
                "evidence_sha256": _sha(evidence_path),
                "evidence_contract_sha256": evidence["evidence_contract_sha256"],
            }
        )
    receipt = {
        "schema_version": "2.0",
        "status": "PASS",
        "review_type": "LAYERED_AGENT_MULTI_PASS_NO_HUMAN_SIGNOFF",
        "implementation_commit": implementation_commit,
        "run_contract_sha256": run_contract_sha256,
        "reviewed_artifact_sha256": reviewed_hashes,
        "layers": sorted(layer_rows, key=lambda row: row["layer_id"]),
        "training_started": False,
        "outer_evaluation_started": False,
        "external_outcomes_accessed": False,
    }
    receipt["receipt_sha256"] = sha256_mapping(receipt)
    receipt_name = "agent_multi_pass_review_receipt_v2.json"
    (task4 / receipt_name).write_text(
        json.dumps(receipt, sort_keys=True), encoding="utf-8"
    )
    state = {
        "schema_version": "2.0",
        "status": "COMMITTED",
        "stage": "ENDPOINT_SPLIT_SCANNER_EXTERNAL_VERIFIED",
        "training_started": False,
        "outer_evaluation_started": False,
        "external_outcomes_accessed": False,
        "agent_multi_pass_qc_status": "AGENT_MULTI_PASS_QC_PASS",
        "implementation_commit": implementation_commit,
        "run_contract_sha256": run_contract_sha256,
        "agent_multi_pass_review_receipt_sha256": receipt["receipt_sha256"],
        "artifact_sha256": {
            **reviewed_hashes,
            receipt_name: _sha(task4 / receipt_name),
        },
    }
    (task4 / "pretraining_validation_state.json").write_text(
        json.dumps(state, sort_keys=True), encoding="utf-8"
    )
    readiness = root / "readiness" / "branch_readiness_v2.json"
    readiness.parent.mkdir(parents=True)
    readiness.write_text(
        json.dumps(
            {
                "schema_version": "1.1",
                "branches": {
                    branch: {
                        "development_allowed": True,
                        "outer_evaluation_allowed": False,
                        "claim_allowed": False,
                        "external_claim_allowed": False,
                    }
                    for branch in (
                        "amyloid_source_regional",
                        "tau_source_regional",
                        "paired_source_regional",
                    )
                },
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    feature_state = root / "feature_contracts" / "source_regional_validation_state.json"
    feature_state.parent.mkdir(parents=True)
    feature_names = (
        "amyloid_source_regional_feature_contract.json",
        "paired_source_regional_feature_contract.json",
        "tau_source_regional_feature_contract.json",
    )
    for name in feature_names:
        (feature_state.parent / name).write_text(f"locked:{name}\n", encoding="utf-8")
    feature_state.write_text(
        json.dumps(
            {
                "status": "COMMITTED",
                "development_allowed": True,
                "artifact_sha256": {
                    **{
                        name: _sha(feature_state.parent / name)
                        for name in feature_names
                    },
                    "branch_readiness_v2.json": _sha(readiness),
                },
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return task4, readiness, feature_state


def _config(path: Path) -> Path:
    path.write_text(
        yaml.safe_dump(
            {
                "schema_version": "2.0",
                "seed": 20260714,
                "outer_folds": 5,
                "inner_folds": 3,
                "budget_per_family": 3,
                "execution_mode": "synthetic_smoke_only",
                "branches": [
                    "amyloid_source_regional",
                    "tau_source_regional",
                    "paired_source_regional",
                ],
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return path


def test_training_v2_launcher_defaults_to_preflight_and_creates_no_training_state(tmp_path, monkeypatch):
    from scripts.run_method_y_training_v2 import main

    _pin_canonical_project_root(monkeypatch, tmp_path)

    inputs = tmp_path / "inputs"
    task4, readiness, feature_state = _write_locked_inputs(inputs)
    output = tmp_path / "training-control"
    exit_code = main(
        [
            "--config",
            str(_config(tmp_path / "config.yaml")),
            "--project-root",
            str(tmp_path),
            "--output-root",
            str(output),
            "--task4-root",
            str(task4),
            "--branch-readiness",
            str(readiness),
            "--feature-validation-state",
            str(feature_state),
            "--preflight-only",
        ]
    )

    assert exit_code == 0
    report = json.loads((output / "TRAINING_CONTROL_PREFLIGHT_V2.json").read_text("utf-8"))
    assert report["status"] == "PREFLIGHT_ONLY_TRAINING_NOT_STARTED"
    assert report["training_started"] is False
    assert report["inner_hpo_started"] is False
    assert report["outer_evaluation_started"] is False
    assert report["model_families"] == ["coxnet", "gbsa", "rsf"]
    assert report["hpo_budget_per_family"] == 3
    assert not list(output.rglob("inner_hpo_state.json"))
    assert not list(output.rglob("outer_result.json"))


def test_explicit_training_flag_still_requires_task6_readiness_receipt(tmp_path, monkeypatch):
    from scripts.run_method_y_training_v2 import TrainingLaunchError, main

    _pin_canonical_project_root(monkeypatch, tmp_path)

    inputs = tmp_path / "inputs"
    task4, readiness, feature_state = _write_locked_inputs(inputs)
    output = tmp_path / "training-control"
    with pytest.raises(TrainingLaunchError, match="PRETRAINING_READINESS_RECEIPT_REQUIRED"):
        main(
            [
                "--config",
                str(_config(tmp_path / "config.yaml")),
                "--project-root",
                str(tmp_path),
                "--output-root",
                str(output),
                "--task4-root",
                str(task4),
                "--branch-readiness",
                str(readiness),
                "--feature-validation-state",
                str(feature_state),
                "--execute-training",
            ]
        )
    assert not list(output.rglob("inner_hpo_state.json"))


def test_valid_task6_receipt_dispatches_hash_bound_resumable_executor_once(tmp_path, monkeypatch):
    """Regression: ExecuteTraining must reach a guarded runtime, not a dead end."""

    from scripts.run_method_y_training_v2 import main

    _pin_canonical_project_root(monkeypatch, tmp_path)

    inputs = tmp_path / "inputs"
    task4, readiness, feature_state = _write_locked_inputs(inputs)
    output = tmp_path / "training-control"
    config = _config(tmp_path / "config.yaml")
    base_args = [
        "--config",
        str(config),
        "--project-root",
        str(tmp_path),
        "--output-root",
        str(output),
        "--task4-root",
        str(task4),
        "--branch-readiness",
        str(readiness),
        "--feature-validation-state",
        str(feature_state),
    ]
    assert main(base_args) == 0
    preflight = json.loads(
        (output / "TRAINING_CONTROL_PREFLIGHT_V2.json").read_text("utf-8")
    )

    class SyntheticResumableExecutor:
        executor_id = "synthetic_resumable_executor_v1"
        implementation_sha256 = "e" * 64
        resume_safe = True
        test_only = True

        def __init__(self):
            self.calls = 0

        def run(self, context):
            self.calls += 1
            assert context["training_control_contract_sha256"] == preflight[
                "training_control_contract_sha256"
            ]
            return {
                "status": "SYNTHETIC_TRAINING_RUNTIME_COMPLETED",
                "completed_units": 45,
            }

    executor = SyntheticResumableExecutor()
    receipt_body = {
        "schema_version": "2.0",
        "status": "READY_FOR_SOURCE_REGIONAL_TRAINING",
        "training_allowed": True,
        "training_control_contract_sha256": preflight[
            "training_control_contract_sha256"
        ],
        "training_executor": {
            "executor_id": executor.executor_id,
            "implementation_sha256": executor.implementation_sha256,
            "resume_safe": True,
            "test_only": True,
        },
    }
    receipt_body["receipt_sha256"] = sha256_mapping(receipt_body)
    receipt = tmp_path / "task6_receipt.json"
    receipt.write_text(json.dumps(receipt_body, sort_keys=True), encoding="utf-8")
    execute_args = [
        *base_args,
        "--pretraining-readiness-receipt",
        str(receipt),
        "--execute-training",
    ]

    assert main(execute_args, training_executor=executor) == 0
    assert main(execute_args, training_executor=executor) == 0
    assert executor.calls == 1
    runtime_state = json.loads(
        (output / "training_runtime_state.json").read_text("utf-8")
    )
    assert runtime_state["status"] == "TRAINING_RUNTIME_COMMITTED"
    orchestration = json.loads(
        (output / "orchestration_state.json").read_text("utf-8")
    )
    assert orchestration["stage"] == "TRAINING_RUNTIME_COMMITTED"
    assert orchestration["training_started"] is True


def test_task4_review_receipt_self_hash_and_layers_are_verified(tmp_path, monkeypatch):
    """Regression: a seventh file name is insufficient without receipt semantics."""

    from scripts.run_method_y_training_v2 import TrainingLaunchError, main

    _pin_canonical_project_root(monkeypatch, tmp_path)

    inputs = tmp_path / "inputs"
    task4, readiness, feature_state = _write_locked_inputs(inputs)
    receipt_path = task4 / "agent_multi_pass_review_receipt_v2.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["layers"] = receipt["layers"][:2]
    receipt["receipt_sha256"] = sha256_mapping(
        {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    )
    receipt_path.write_text(json.dumps(receipt, sort_keys=True), encoding="utf-8")
    state_path = task4 / "pretraining_validation_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["artifact_sha256"][receipt_path.name] = _sha(receipt_path)
    state["agent_multi_pass_review_receipt_sha256"] = receipt["receipt_sha256"]
    state_path.write_text(json.dumps(state, sort_keys=True), encoding="utf-8")

    with pytest.raises(TrainingLaunchError, match="TASK4_REVIEW_LAYER_SET_INVALID"):
        main(
            [
                "--config",
                str(_config(tmp_path / "config.yaml")),
                "--project-root",
                str(tmp_path),
                "--output-root",
                str(tmp_path / "training-control"),
                "--task4-root",
                str(task4),
                "--branch-readiness",
                str(readiness),
                "--feature-validation-state",
                str(feature_state),
            ]
        )


def test_task4_review_evidence_is_rehashed_and_semantically_revalidated(tmp_path, monkeypatch):
    from scripts.run_method_y_training_v2 import TrainingLaunchError, main

    _pin_canonical_project_root(monkeypatch, tmp_path)

    inputs = tmp_path / "inputs"
    task4, readiness, feature_state = _write_locked_inputs(inputs)
    receipt_path = task4 / "agent_multi_pass_review_receipt_v2.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    layer = next(
        item
        for item in receipt["layers"]
        if item["layer_id"] == "INVARIANT_TEST_SUITE"
    )
    evidence_path = task4 / layer["evidence_relative_path"]
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    evidence["zero_failures"] = False
    evidence["evidence_contract_sha256"] = sha256_mapping(
        {
            key: value
            for key, value in evidence.items()
            if key != "evidence_contract_sha256"
        }
    )
    evidence_path.write_text(json.dumps(evidence, sort_keys=True), encoding="utf-8")
    layer["evidence_sha256"] = _sha(evidence_path)
    layer["evidence_contract_sha256"] = evidence["evidence_contract_sha256"]
    receipt["receipt_sha256"] = sha256_mapping(
        {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    )
    receipt_path.write_text(json.dumps(receipt, sort_keys=True), encoding="utf-8")
    state_path = task4 / "pretraining_validation_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["artifact_sha256"][receipt_path.name] = _sha(receipt_path)
    state["agent_multi_pass_review_receipt_sha256"] = receipt["receipt_sha256"]
    state_path.write_text(json.dumps(state, sort_keys=True), encoding="utf-8")

    with pytest.raises(
        TrainingLaunchError,
        match="TASK4_REVIEW_EVIDENCE_INVALID:REVIEW_EVIDENCE_RESULT_INVALID",
    ):
        main(
            [
                "--config",
                str(_config(tmp_path / "config.yaml")),
                "--project-root",
                str(tmp_path),
                "--output-root",
                str(tmp_path / "training-control"),
                "--task4-root",
                str(task4),
                "--branch-readiness",
                str(readiness),
                "--feature-validation-state",
                str(feature_state),
            ]
        )


def test_guarded_runtime_rejects_private_or_uncontracted_executor_result_fields(tmp_path):
    from src.method_y_revival.training_runtime_control import (
        TrainingRuntimeControlError,
        dispatch_training_runtime,
    )

    class LeakyExecutor:
        executor_id = "synthetic_leaky_executor_v1"
        implementation_sha256 = "f" * 64
        resume_safe = True
        test_only = True

        def run(self, context):
            return {
                "status": "SYNTHETIC_COMPLETED",
                "completed_units": 1,
                "raw_path": r"H:\A4 Learn\Clinical\BID123.csv",
                "source_subject_id": "BID123",
            }

    executor = LeakyExecutor()
    descriptor = {
        "executor_id": executor.executor_id,
        "implementation_sha256": executor.implementation_sha256,
        "resume_safe": True,
        "test_only": True,
    }
    with pytest.raises(
        TrainingRuntimeControlError,
        match="TRAINING_EXECUTOR_RESULT_SCHEMA_INVALID",
    ):
        dispatch_training_runtime(
            output_root=tmp_path,
            training_control_contract_sha256="1" * 64,
            receipt_sha256="2" * 64,
            receipt_executor_contract=descriptor,
            executor=executor,
        )
    assert not (tmp_path / "training_runtime_result.json").exists()


def test_launcher_rejects_task4_artifact_hash_tampering(tmp_path, monkeypatch):
    from scripts.run_method_y_training_v2 import TrainingLaunchError, main

    _pin_canonical_project_root(monkeypatch, tmp_path)

    inputs = tmp_path / "inputs"
    task4, readiness, feature_state = _write_locked_inputs(inputs)
    (task4 / "endpoint_verification_v2.json").write_text(
        '{"status":"TAMPERED"}\n', encoding="utf-8"
    )
    with pytest.raises(TrainingLaunchError, match="TASK4_ARTIFACT_HASH_MISMATCH"):
        main(
            [
                "--config",
                str(_config(tmp_path / "config.yaml")),
                "--project-root",
                str(tmp_path),
                "--output-root",
                str(tmp_path / "training-control"),
                "--task4-root",
                str(task4),
                "--branch-readiness",
                str(readiness),
                "--feature-validation-state",
                str(feature_state),
            ]
        )


def test_launcher_rejects_an_incomplete_task4_artifact_set(tmp_path, monkeypatch):
    from scripts.run_method_y_training_v2 import TrainingLaunchError, main

    _pin_canonical_project_root(monkeypatch, tmp_path)

    inputs = tmp_path / "inputs"
    task4, readiness, feature_state = _write_locked_inputs(inputs)
    state_path = task4 / "pretraining_validation_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["artifact_sha256"].pop("external_intake_contract_v2.json")
    state_path.write_text(json.dumps(state, sort_keys=True), encoding="utf-8")

    with pytest.raises(TrainingLaunchError, match="TASK4_ARTIFACT_SET_MISMATCH"):
        main(
            [
                "--config",
                str(_config(tmp_path / "config.yaml")),
                "--project-root",
                str(tmp_path),
                "--output-root",
                str(tmp_path / "training-control"),
                "--task4-root",
                str(task4),
                "--branch-readiness",
                str(readiness),
                "--feature-validation-state",
                str(feature_state),
            ]
        )


def test_launcher_rejects_an_incomplete_feature_artifact_set(tmp_path, monkeypatch):
    from scripts.run_method_y_training_v2 import TrainingLaunchError, main

    _pin_canonical_project_root(monkeypatch, tmp_path)

    inputs = tmp_path / "inputs"
    task4, readiness, feature_state = _write_locked_inputs(inputs)
    state = json.loads(feature_state.read_text(encoding="utf-8"))
    state["artifact_sha256"].pop("tau_source_regional_feature_contract.json")
    feature_state.write_text(json.dumps(state, sort_keys=True), encoding="utf-8")

    with pytest.raises(TrainingLaunchError, match="FEATURE_ARTIFACT_SET_MISMATCH"):
        main(
            [
                "--config",
                str(_config(tmp_path / "config.yaml")),
                "--project-root",
                str(tmp_path),
                "--output-root",
                str(tmp_path / "training-control"),
                "--task4-root",
                str(task4),
                "--branch-readiness",
                str(readiness),
                "--feature-validation-state",
                str(feature_state),
            ]
        )


def test_noncanonical_h_project_root_is_rejected():
    from scripts.run_method_y_training_v2 import (
        SCRIPT_PROJECT_ROOT,
        TrainingLaunchError,
        main,
    )

    other = (SCRIPT_PROJECT_ROOT / "results").resolve()
    assert other.drive.upper() == "H:"
    assert other != SCRIPT_PROJECT_ROOT.resolve()
    with pytest.raises(TrainingLaunchError, match="PROJECT_ROOT_MUST_BE_THIS_CHECKOUT"):
        main(["--project-root", str(other), "--preflight-only"])


def test_powershell_training_v2_launcher_rejects_a_different_h_project_root(tmp_path):
    inputs = tmp_path / "inputs"
    task4, readiness, feature_state = _write_locked_inputs(inputs)
    output = tmp_path / "training-control"
    completed = subprocess.run(
        [
            "powershell",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(Path("scripts/run_method_y_training_v2.ps1").resolve()),
            "-ProjectRoot",
            str(tmp_path),
            "-PythonPath",
            sys.executable,
            "-PythonScript",
            str(Path("scripts/run_method_y_training_v2.py").resolve()),
            "-ConfigPath",
            str(_config(tmp_path / "config.yaml")),
            "-OutputRoot",
            str(output),
            "-Task4Root",
            str(task4),
            "-BranchReadinessPath",
            str(readiness),
            "-FeatureValidationStatePath",
            str(feature_state),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=90,
    )

    assert completed.returncode != 0
    assert "PROJECT_ROOT_MUST_BE_THIS_CHECKOUT" in (completed.stderr or "") + (
        completed.stdout or ""
    )
    assert not list(output.rglob("inner_hpo_state.json"))
