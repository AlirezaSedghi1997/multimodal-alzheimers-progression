from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import yaml

SCRIPT_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPT_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_PROJECT_ROOT))

from src.method_y_revival.atomic_io import atomic_write_json, read_json
from src.method_y_revival.contracts import sha256_mapping
from src.method_y_revival.model_family_adapters import (
    build_matched_candidate_registry,
)
from src.method_y_revival.process_guard import WriterLock
from src.method_y_revival.review_receipt import (
    AgentReviewReceiptError,
    REQUIRED_TASK4_REVIEW_LAYERS,
    validate_task4_review_evidence,
)
from src.method_y_revival.training_runtime_control import (
    TrainingRuntimeControlError,
    dispatch_training_runtime,
)


class TrainingLaunchError(ValueError):
    """Raised before training when a pretraining receipt or hash is not locked."""


_SOURCE_BRANCHES = (
    "amyloid_source_regional",
    "tau_source_regional",
    "paired_source_regional",
)
_TASK4_REVIEWED_ARTIFACTS = {
    "PRETRAINING_CONTRACTS_TASK4.md",
    "endpoint_verification_v2.json",
    "external_intake_contract_v2.json",
    "predictor_temporality_contract_v2.json",
    "scanner_feasibility_and_sensitivity_contract_v2.json",
    "split_verification_v2.json",
}
_TASK4_REVIEW_RECEIPT = "agent_multi_pass_review_receipt_v2.json"
_TASK4_ARTIFACTS = _TASK4_REVIEWED_ARTIFACTS | {_TASK4_REVIEW_RECEIPT}
_FEATURE_ARTIFACTS = {
    "amyloid_source_regional_feature_contract.json",
    "branch_readiness_v2.json",
    "paired_source_regional_feature_contract.json",
    "tau_source_regional_feature_contract.json",
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_canonical_project_root(project_root: Path) -> Path:
    """Reject any tree other than this checkout, including other paths on H:."""

    resolved = project_root.resolve()
    canonical = SCRIPT_PROJECT_ROOT.resolve()
    if resolved.drive.upper() != "H:":
        raise TrainingLaunchError("PROJECT_ROOT_MUST_BE_ON_H_DRIVE")
    if resolved != canonical:
        raise TrainingLaunchError("PROJECT_ROOT_MUST_BE_THIS_CHECKOUT")
    return resolved


def _resolved_under(path: Path, parent: Path, label: str) -> Path:
    resolved = path.resolve()
    parent_resolved = parent.resolve()
    try:
        resolved.relative_to(parent_resolved)
    except ValueError as exc:
        raise TrainingLaunchError(f"{label.upper()}_OUTSIDE_PROJECT_ROOT") from exc
    if resolved.drive.upper() != "H:":
        raise TrainingLaunchError(f"{label.upper()}_MUST_BE_ON_H_DRIVE")
    return resolved


def _verify_task4_bundle(task4_root: Path) -> dict[str, Any]:
    state_path = task4_root / "pretraining_validation_state.json"
    if not state_path.is_file():
        raise TrainingLaunchError("TASK4_VALIDATION_STATE_REQUIRED")
    state = read_json(state_path)
    required = {
        "status": "COMMITTED",
        "stage": "ENDPOINT_SPLIT_SCANNER_EXTERNAL_VERIFIED",
        "training_started": False,
        "outer_evaluation_started": False,
        "external_outcomes_accessed": False,
        "agent_multi_pass_qc_status": "AGENT_MULTI_PASS_QC_PASS",
    }
    if any(state.get(key) != value for key, value in required.items()):
        raise TrainingLaunchError("TASK4_STATE_NOT_FAIL_CLOSED_COMMITTED")
    hashes = state.get("artifact_sha256")
    if not isinstance(hashes, dict) or not hashes:
        raise TrainingLaunchError("TASK4_ARTIFACT_HASHES_REQUIRED")
    if set(hashes) != _TASK4_ARTIFACTS:
        raise TrainingLaunchError("TASK4_ARTIFACT_SET_MISMATCH")
    for name, expected in hashes.items():
        if Path(str(name)).name != str(name):
            raise TrainingLaunchError("TASK4_ARTIFACT_NAME_MUST_BE_BASENAME")
        path = task4_root / str(name)
        if not path.is_file() or _sha256_file(path) != expected:
            raise TrainingLaunchError(f"TASK4_ARTIFACT_HASH_MISMATCH:{name}")
    receipt = read_json(task4_root / _TASK4_REVIEW_RECEIPT)
    claimed_receipt_sha = receipt.get("receipt_sha256")
    receipt_body = {
        key: value for key, value in receipt.items() if key != "receipt_sha256"
    }
    if claimed_receipt_sha != sha256_mapping(receipt_body):
        raise TrainingLaunchError("TASK4_REVIEW_RECEIPT_SELF_HASH_MISMATCH")
    if state.get("agent_multi_pass_review_receipt_sha256") != claimed_receipt_sha:
        raise TrainingLaunchError("TASK4_REVIEW_RECEIPT_STATE_HASH_MISMATCH")
    reviewed = receipt.get("reviewed_artifact_sha256")
    expected_reviewed = {
        name: hashes[name] for name in sorted(_TASK4_REVIEWED_ARTIFACTS)
    }
    if reviewed != expected_reviewed:
        raise TrainingLaunchError("TASK4_REVIEWED_ARTIFACT_SET_MISMATCH")
    receipt_required = {
        "schema_version": "2.0",
        "status": "PASS",
        "review_type": "LAYERED_AGENT_MULTI_PASS_NO_HUMAN_SIGNOFF",
        "training_started": False,
        "outer_evaluation_started": False,
        "external_outcomes_accessed": False,
    }
    if any(receipt.get(key) != value for key, value in receipt_required.items()):
        raise TrainingLaunchError("TASK4_REVIEW_RECEIPT_STATUS_INVALID")
    if (
        receipt.get("implementation_commit") != state.get("implementation_commit")
        or receipt.get("run_contract_sha256") != state.get("run_contract_sha256")
    ):
        raise TrainingLaunchError("TASK4_REVIEW_RECEIPT_BINDING_MISMATCH")
    layers = receipt.get("layers")
    if not isinstance(layers, list) or len(layers) != len(
        REQUIRED_TASK4_REVIEW_LAYERS
    ):
        raise TrainingLaunchError("TASK4_REVIEW_LAYER_SET_INVALID")
    if {
        layer.get("layer_id") for layer in layers if isinstance(layer, dict)
    } != set(REQUIRED_TASK4_REVIEW_LAYERS):
        raise TrainingLaunchError("TASK4_REVIEW_LAYER_SET_INVALID")
    evidence_paths = [
        str(layer.get("evidence_relative_path"))
        for layer in layers
        if isinstance(layer, dict)
    ]
    try:
        independently_validated = validate_task4_review_evidence(
            task4_root,
            evidence_relative_paths=evidence_paths,
            implementation_commit=receipt.get("implementation_commit"),
            run_contract_sha256=receipt.get("run_contract_sha256"),
            reviewed_artifact_sha256=expected_reviewed,
        )
    except AgentReviewReceiptError as exc:
        raise TrainingLaunchError(f"TASK4_REVIEW_EVIDENCE_INVALID:{exc}") from exc
    if layers != independently_validated:
        raise TrainingLaunchError("TASK4_REVIEW_RECEIPT_EVIDENCE_MISMATCH")
    return state


def _blocked_task4_preflight_report(task4_root: Path) -> dict[str, Any]:
    task4_status = "UNKNOWN"
    qc_status = "UNKNOWN"
    quality_verdict = "UNKNOWN"
    try:
        state = read_json(task4_root / "pretraining_validation_state.json")
        if isinstance(state, dict):
            task4_status = str(state.get("status") or task4_status)
            qc_status = str(state.get("agent_multi_pass_qc_status") or qc_status)
    except (OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError):
        pass
    try:
        adversarial = read_json(
            task4_root / "review_evidence" / "adversarial_agent_review_v2.json"
        )
        if isinstance(adversarial, dict):
            quality_verdict = str(adversarial.get("quality_verdict") or quality_verdict)
    except (OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError):
        pass
    return {
        "schema_version": "2.0",
        "status": "PREFLIGHT_BLOCKED_TASK4_NOT_COMMITTED",
        "training_started": False,
        "inner_hpo_started": False,
        "outer_evaluation_started": False,
        "execute_training": False,
        "reason": "TASK4_STATE_NOT_FAIL_CLOSED_COMMITTED",
        "task4_status": task4_status,
        "task4_qc_status": qc_status,
        "adversarial_quality_verdict": quality_verdict,
        "command": ".venv/Scripts/python.exe scripts/run_method_y_training_v2.py --preflight-only",
    }


def _verify_source_readiness(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise TrainingLaunchError("BRANCH_READINESS_REQUIRED")
    payload = read_json(path)
    branches = payload.get("branches")
    if not isinstance(branches, dict):
        raise TrainingLaunchError("MALFORMED_BRANCH_READINESS")
    for branch in _SOURCE_BRANCHES:
        record = branches.get(branch)
        if not isinstance(record, dict) or record.get("development_allowed") is not True:
            raise TrainingLaunchError(f"SOURCE_DEVELOPMENT_NOT_ALLOWED:{branch}")
        if any(
            record.get(permission) is not False
            for permission in (
                "outer_evaluation_allowed",
                "claim_allowed",
                "external_claim_allowed",
            )
        ):
            raise TrainingLaunchError(f"PREMATURE_BRANCH_PERMISSION:{branch}")
    return payload


def _verify_feature_state(path: Path, readiness_path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise TrainingLaunchError("FEATURE_VALIDATION_STATE_REQUIRED")
    payload = read_json(path)
    if payload.get("status") != "COMMITTED" or payload.get("development_allowed") is not True:
        raise TrainingLaunchError("FEATURE_VALIDATION_NOT_COMMITTED")
    hashes = payload.get("artifact_sha256")
    if not isinstance(hashes, dict) or not hashes:
        raise TrainingLaunchError("FEATURE_ARTIFACT_HASHES_REQUIRED")
    if set(hashes) != _FEATURE_ARTIFACTS:
        raise TrainingLaunchError("FEATURE_ARTIFACT_SET_MISMATCH")
    for name in sorted(_FEATURE_ARTIFACTS - {"branch_readiness_v2.json"}):
        artifact = path.parent / name
        if not artifact.is_file() or hashes[name] != _sha256_file(artifact):
            raise TrainingLaunchError(f"FEATURE_ARTIFACT_HASH_MISMATCH:{name}")
    expected_readiness = hashes.get("branch_readiness_v2.json")
    if expected_readiness != _sha256_file(readiness_path):
        raise TrainingLaunchError("FEATURE_READINESS_HASH_MISMATCH")
    return payload


def _load_config(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise TrainingLaunchError("TRAINING_CONTROL_CONFIG_REQUIRED")
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or config.get("schema_version") != "2.0":
        raise TrainingLaunchError("TRAINING_CONTROL_CONFIG_SCHEMA_MISMATCH")
    if int(config.get("outer_folds", 0)) != 5 or int(config.get("inner_folds", 0)) != 3:
        raise TrainingLaunchError("FIVE_OUTER_THREE_INNER_REQUIRED")
    if config.get("execution_mode") != "synthetic_smoke_only":
        raise TrainingLaunchError("SYNTHETIC_SMOKE_ONLY_REQUIRED")
    if set(config.get("branches", [])) != set(_SOURCE_BRANCHES):
        raise TrainingLaunchError("EXACT_SOURCE_BRANCH_SET_REQUIRED")
    return config


def run_preflight(
    *,
    config_path: Path,
    project_root: Path,
    output_root: Path,
    task4_root: Path,
    branch_readiness_path: Path,
    feature_validation_state_path: Path,
) -> dict[str, Any]:
    config = _load_config(config_path)
    task4 = _verify_task4_bundle(task4_root)
    readiness = _verify_source_readiness(branch_readiness_path)
    feature_state = _verify_feature_state(
        feature_validation_state_path, branch_readiness_path
    )
    registry = build_matched_candidate_registry(
        seed=int(config["seed"]),
        budget_per_family=int(config["budget_per_family"]),
    )
    inputs = {
        "config": _sha256_file(config_path),
        "task4_validation_state": _sha256_file(
            task4_root / "pretraining_validation_state.json"
        ),
        "branch_readiness": _sha256_file(branch_readiness_path),
        "feature_validation_state": _sha256_file(feature_validation_state_path),
    }
    contract_payload = {
        "schema_version": "2.0",
        "seed": int(config["seed"]),
        "outer_folds": 5,
        "inner_folds": 3,
        "branches": sorted(_SOURCE_BRANCHES),
        "candidate_registry": registry,
        "input_sha256": inputs,
        "execution_default": "PREFLIGHT_ONLY",
        "task6_readiness_receipt_required_for_training": True,
    }
    report = {
        "schema_version": "2.0",
        "status": "PREFLIGHT_ONLY_TRAINING_NOT_STARTED",
        "training_started": False,
        "inner_hpo_started": False,
        "outer_evaluation_started": False,
        "external_outcomes_accessed": False,
        "model_families": sorted(registry["families"]),
        "hpo_budget_per_family": registry["budget_per_family"],
        "selection_metric": registry["selection_metric"],
        "selection_rule": registry["selection_rule"],
        "input_sha256": inputs,
        "task4_state_sha256": sha256_mapping(task4),
        "readiness_snapshot_sha256": sha256_mapping(readiness),
        "feature_state_sha256": sha256_mapping(feature_state),
        "training_control_contract": contract_payload,
        "training_control_contract_sha256": sha256_mapping(contract_payload),
    }
    atomic_write_json(output_root / "TRAINING_CONTROL_PREFLIGHT_V2.json", report)
    atomic_write_json(
        output_root / "orchestration_state.json",
        {
            "schema_version": "2.0",
            "stage": "TRAINING_CONTROL_PREFLIGHT_DONE",
            "training_started": False,
            "inner_hpo_started": False,
            "outer_evaluation_started": False,
            "training_control_contract_sha256": report[
                "training_control_contract_sha256"
            ],
        },
    )
    return report


def _require_task6_receipt(path: Path | None, preflight: dict[str, Any]) -> dict[str, Any]:
    if path is None or not path.is_file():
        raise TrainingLaunchError("PRETRAINING_READINESS_RECEIPT_REQUIRED")
    receipt = read_json(path)
    claimed = receipt.get("receipt_sha256")
    body = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    if claimed != sha256_mapping(body):
        raise TrainingLaunchError("PRETRAINING_READINESS_RECEIPT_SELF_HASH_MISMATCH")
    if (
        receipt.get("schema_version") != "2.0"
        or receipt.get("status") != "READY_FOR_SOURCE_REGIONAL_TRAINING"
        or receipt.get("training_allowed") is not True
        or receipt.get("training_control_contract_sha256")
        != preflight["training_control_contract_sha256"]
    ):
        raise TrainingLaunchError("PRETRAINING_READINESS_RECEIPT_MISMATCH")
    executor = receipt.get("training_executor")
    if not isinstance(executor, dict):
        raise TrainingLaunchError("TRAINING_EXECUTOR_RECEIPT_CONTRACT_REQUIRED")
    required_executor = (
        "executor_id",
        "implementation_sha256",
        "resume_safe",
        "test_only",
    )
    if set(required_executor) - set(executor):
        raise TrainingLaunchError("TRAINING_EXECUTOR_RECEIPT_CONTRACT_REQUIRED")
    if (
        not isinstance(executor.get("executor_id"), str)
        or not str(executor.get("executor_id")).strip()
        or executor.get("resume_safe") is not True
        or executor.get("test_only") is not True
    ):
        raise TrainingLaunchError("SYNTHETIC_SMOKE_ONLY_EXECUTOR_REQUIRED")
    return receipt


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fail-closed Method Y v2 training-control preflight"
    )
    parser.add_argument(
        "--config",
        default=str(SCRIPT_PROJECT_ROOT / "configs" / "method_y_training_control_v2.yaml"),
    )
    parser.add_argument("--project-root", default=str(SCRIPT_PROJECT_ROOT))
    parser.add_argument("--output-root")
    parser.add_argument("--task4-root")
    parser.add_argument("--branch-readiness")
    parser.add_argument("--feature-validation-state")
    parser.add_argument("--pretraining-readiness-receipt")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--preflight-only",
        action="store_true",
        help="Explicit preflight (default). Does not start training.",
    )
    mode.add_argument("--execute-training", action="store_true")
    return parser


def main(argv: list[str] | None = None, *, training_executor=None) -> int:
    args = build_parser().parse_args(argv)
    project_root = require_canonical_project_root(Path(args.project_root))
    output_root = _resolved_under(
        Path(args.output_root)
        if args.output_root
        else project_root / "results" / "method_y_revival_v2" / "training_control",
        project_root,
        "output_root",
    )
    task4_root = _resolved_under(
        Path(args.task4_root)
        if args.task4_root
        else project_root / "results" / "method_y_revival_v2" / "pretraining_contracts",
        project_root,
        "task4_root",
    )
    readiness = _resolved_under(
        Path(args.branch_readiness)
        if args.branch_readiness
        else project_root
        / "results"
        / "method_y_revival_v2"
        / "readiness"
        / "branch_readiness_v2.json",
        project_root,
        "branch_readiness",
    )
    feature_state = _resolved_under(
        Path(args.feature_validation_state)
        if args.feature_validation_state
        else project_root
        / "results"
        / "method_y_revival_v2"
        / "feature_contracts"
        / "source_regional_validation_state.json",
        project_root,
        "feature_validation_state",
    )
    config_path = _resolved_under(Path(args.config), project_root, "config")
    receipt = (
        _resolved_under(Path(args.pretraining_readiness_receipt), project_root, "receipt")
        if args.pretraining_readiness_receipt
        else None
    )
    lock = WriterLock(output_root / ".writer.lock", output_root=output_root)
    with lock:
        try:
            report = run_preflight(
                config_path=config_path,
                project_root=project_root,
                output_root=output_root,
                task4_root=task4_root,
                branch_readiness_path=readiness,
                feature_validation_state_path=feature_state,
            )
        except TrainingLaunchError as exc:
            if str(exc) != "TASK4_STATE_NOT_FAIL_CLOSED_COMMITTED":
                raise
            report = _blocked_task4_preflight_report(task4_root)
            atomic_write_json(output_root / "TRAINING_CONTROL_PREFLIGHT_V2.json", report)
            print(json.dumps(report, ensure_ascii=False, sort_keys=True), flush=True)
            return 2
        if args.execute_training:
            task6 = _require_task6_receipt(receipt, report)
            try:
                runtime_result = dispatch_training_runtime(
                    output_root=output_root,
                    training_control_contract_sha256=report[
                        "training_control_contract_sha256"
                    ],
                    receipt_sha256=task6["receipt_sha256"],
                    receipt_executor_contract=task6["training_executor"],
                    executor=training_executor,
                )
            except TrainingRuntimeControlError as exc:
                raise TrainingLaunchError(str(exc)) from exc
            atomic_write_json(
                output_root / "orchestration_state.json",
                {
                    "schema_version": "2.0",
                    "stage": "TRAINING_RUNTIME_COMMITTED",
                    "training_started": True,
                    "outer_evaluation_started": False,
                    "training_control_contract_sha256": report[
                        "training_control_contract_sha256"
                    ],
                    "pretraining_readiness_receipt_sha256": task6[
                        "receipt_sha256"
                    ],
                    "training_runtime_result_sha256": sha256_mapping(
                        runtime_result
                    ),
                },
            )
    print(json.dumps(report, ensure_ascii=False, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
