"""Atomic layered-agent review receipts for Method Y pretraining gates."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any, Iterable, Mapping

from .atomic_io import atomic_write_json, read_json
from .contracts import sha256_mapping
from .endpoint_builder import sha256_file


class AgentReviewReceiptError(ValueError):
    """Raised when a layered review cannot be proven from locked evidence."""


REQUIRED_TASK4_REVIEW_LAYERS = frozenset(
    {
        "INVARIANT_TEST_SUITE",
        "DETERMINISTIC_REBUILD",
        "ADVERSARIAL_AGENT_REVIEW",
    }
)
TASK4_REVIEW_EVIDENCE_PATHS = {
    "INVARIANT_TEST_SUITE": "review_evidence/invariant_test_suite_v2.json",
    "DETERMINISTIC_REBUILD": "review_evidence/deterministic_rebuild_v2.json",
    "ADVERSARIAL_AGENT_REVIEW": "review_evidence/adversarial_agent_review_v2.json",
}
TASK4_IMPLEMENTATION_RELATIVE_PATHS = (
    "configs/method_y_revival_v1.yaml",
    "scripts/build_method_y_pretraining_contracts_v2.py",
    "scripts/build_method_y_tau_progression.py",
    "src/method_y_revival/atomic_io.py",
    "src/method_y_revival/branch_readiness.py",
    "src/method_y_revival/contracts.py",
    "src/method_y_revival/endpoint_builder.py",
    "src/method_y_revival/pretraining_contracts.py",
    "src/method_y_revival/process_guard.py",
    "src/method_y_revival/review_receipt.py",
    "src/method_y_revival/splits.py",
)
_RUN_CONTRACT_KEYS = {
    "schema_version",
    "seed",
    "split_seed",
    "clinical_source_snapshot_sha256",
    "pacc_source_snapshot_sha256",
    "pacc_legacy_input_contract_sha256",
    "implementation_file_sha256",
    "endpoint_contract_sha256",
    "implementation_commit",
    "artifact_sha256",
    "reconstruction_mode",
    "release_eligible",
}
_EVIDENCE_COMMON_KEYS = {
    "schema_version",
    "layer_id",
    "status",
    "implementation_commit",
    "run_contract_sha256",
    "reviewed_artifact_sha256",
    "evidence_contract_sha256",
}
_EVIDENCE_RESULT_KEYS = {
    "INVARIANT_TEST_SUITE": {"zero_failures", "test_count"},
    "DETERMINISTIC_REBUILD": {"changed_count", "artifact_count"},
    "ADVERSARIAL_AGENT_REVIEW": {
        "spec_verdict",
        "quality_verdict",
        "blocking_findings",
    },
}


def _is_sha256(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _is_commit(value: Any) -> bool:
    if not isinstance(value, str) or len(value) not in {40, 64}:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _validate_layer_result(
    layer_id: str, payload: Mapping[str, Any], artifact_count: int
) -> None:
    if layer_id == "INVARIANT_TEST_SUITE":
        valid = (
            payload.get("zero_failures") is True
            and isinstance(payload.get("test_count"), int)
            and not isinstance(payload.get("test_count"), bool)
            and payload["test_count"] > 0
        )
    elif layer_id == "DETERMINISTIC_REBUILD":
        valid = (
            payload.get("changed_count") == 0
            and payload.get("artifact_count") == artifact_count
        )
    else:
        valid = (
            payload.get("spec_verdict") == "PASS"
            and payload.get("quality_verdict") == "PASS"
            and payload.get("blocking_findings") == 0
        )
    if not valid:
        raise AgentReviewReceiptError("REVIEW_EVIDENCE_RESULT_INVALID")


def validate_task4_run_contract(
    project_root: str | Path,
    output_root: str | Path,
    *,
    state: Mapping[str, Any],
    reviewed_artifact_sha256: Mapping[str, str],
) -> None:
    """Bind the declared contract to the real repository and locked artifacts."""

    project = Path(project_root).resolve()
    root = Path(output_root).resolve()
    run_contract = state.get("run_contract")
    commit = state.get("implementation_commit")
    digest = state.get("run_contract_sha256")
    reviewed = dict(sorted(reviewed_artifact_sha256.items()))
    if not (
        isinstance(run_contract, Mapping)
        and set(run_contract) == _RUN_CONTRACT_KEYS
        and _is_commit(commit)
        and run_contract.get("implementation_commit") == commit
        and _is_sha256(digest)
        and digest == sha256_mapping(run_contract)
        and run_contract.get("schema_version") == "2.0"
        and run_contract.get("reconstruction_mode")
        == "LOCKED_RAW_SOURCE_INDEPENDENT_REBUILD"
        and run_contract.get("release_eligible") is True
        and state.get("release_eligible") is True
        and run_contract.get("artifact_sha256") == reviewed
        and isinstance(run_contract.get("seed"), int)
        and not isinstance(run_contract.get("seed"), bool)
        and run_contract.get("seed") == run_contract.get("split_seed")
    ):
        raise AgentReviewReceiptError("TASK4_RUN_CONTRACT_SEMANTICS_INVALID")
    implementation = run_contract.get("implementation_file_sha256")
    if not (
        isinstance(implementation, Mapping)
        and set(implementation) == set(TASK4_IMPLEMENTATION_RELATIVE_PATHS)
        and all(_is_sha256(value) for value in implementation.values())
    ):
        raise AgentReviewReceiptError("TASK4_IMPLEMENTATION_SNAPSHOT_INVALID")
    for relative in TASK4_IMPLEMENTATION_RELATIVE_PATHS:
        path = (project / relative).resolve()
        try:
            path.relative_to(project)
        except ValueError as exc:  # pragma: no cover - constants are locked
            raise AgentReviewReceiptError(
                "TASK4_IMPLEMENTATION_PATH_OUTSIDE_PROJECT"
            ) from exc
        if not path.is_file() or sha256_file(path) != implementation[relative]:
            raise AgentReviewReceiptError("TASK4_IMPLEMENTATION_SNAPSHOT_MISMATCH")
    try:
        subprocess.run(
            ["git", "-C", str(project), "cat-file", "-e", f"{commit}^{{commit}}"],
            check=True,
            capture_output=True,
        )
        comparison = subprocess.run(
            [
                "git",
                "-C",
                str(project),
                "diff",
                "--quiet",
                commit,
                "--",
                *TASK4_IMPLEMENTATION_RELATIVE_PATHS,
            ],
            check=False,
            capture_output=True,
        )
    except OSError as exc:
        raise AgentReviewReceiptError("TASK4_IMPLEMENTATION_COMMIT_INVALID") from exc
    if comparison.returncode not in {0, 1}:
        raise AgentReviewReceiptError("TASK4_IMPLEMENTATION_COMMIT_INVALID")
    if comparison.returncode == 1:
        raise AgentReviewReceiptError("TASK4_IMPLEMENTATION_COMMIT_MISMATCH")
    split = read_json(root / "split_verification_v2.json")
    endpoint = read_json(root / "endpoint_verification_v2.json")
    primary = endpoint.get("primary_endpoints")
    endpoint_contracts = run_contract.get("endpoint_contract_sha256")
    if not (
        split.get("seed") == run_contract["seed"]
        and isinstance(primary, Mapping)
        and isinstance(endpoint_contracts, Mapping)
        and set(endpoint_contracts) == {"amyloid", "tau", "paired_at"}
        and all(
            isinstance(primary.get(branch), Mapping)
            and endpoint_contracts.get(branch)
            == primary[branch].get("contract_sha256")
            for branch in ("amyloid", "tau", "paired_at")
        )
    ):
        raise AgentReviewReceiptError("TASK4_RUN_CONTRACT_ARTIFACT_BINDING_INVALID")


def validate_task4_review_evidence(
    output_root: str | Path,
    *,
    evidence_relative_paths: Iterable[str],
    implementation_commit: str,
    run_contract_sha256: str,
    reviewed_artifact_sha256: Mapping[str, str],
) -> list[dict[str, str]]:
    """Rehash and validate the exact three evidence files used by Task 4."""

    if not _is_commit(implementation_commit) or not _is_sha256(
        run_contract_sha256
    ):
        raise AgentReviewReceiptError("REVIEW_EVIDENCE_BINDING_INVALID")
    reviewed = {
        str(name): str(digest).lower()
        for name, digest in reviewed_artifact_sha256.items()
    }
    if not reviewed or any(
        Path(name).name != name or not _is_sha256(digest)
        for name, digest in reviewed.items()
    ):
        raise AgentReviewReceiptError("REVIEW_EVIDENCE_BINDING_INVALID")
    paths = [str(value).replace("\\", "/") for value in evidence_relative_paths]
    if len(paths) != 3 or len(set(paths)) != 3 or set(paths) != set(
        TASK4_REVIEW_EVIDENCE_PATHS.values()
    ):
        raise AgentReviewReceiptError("REVIEW_EVIDENCE_PATH_SET_INVALID")
    root = Path(output_root).resolve()
    evidence_root = (root / "review_evidence").resolve()
    normalized: list[dict[str, str]] = []
    for relative in sorted(paths):
        layer_id = next(
            key
            for key, expected in TASK4_REVIEW_EVIDENCE_PATHS.items()
            if expected == relative
        )
        path = (root / Path(relative)).resolve()
        try:
            path.relative_to(evidence_root)
        except ValueError as exc:
            raise AgentReviewReceiptError("REVIEW_EVIDENCE_PATH_OUTSIDE_ROOT") from exc
        if not path.is_file():
            raise AgentReviewReceiptError("REVIEW_EVIDENCE_FILE_MISSING")
        payload = read_json(path)
        from .pretraining_contracts import assert_public_contract_privacy

        try:
            assert_public_contract_privacy(payload)
        except ValueError as exc:
            raise AgentReviewReceiptError("REVIEW_EVIDENCE_PRIVACY_INVALID") from exc
        unsigned = {
            key: value
            for key, value in payload.items()
            if key != "evidence_contract_sha256"
        }
        if not (
            set(payload)
            == _EVIDENCE_COMMON_KEYS | _EVIDENCE_RESULT_KEYS[layer_id]
            and payload.get("schema_version") == "2.0"
            and payload.get("layer_id") == layer_id
            and payload.get("status") == "PASS"
            and payload.get("implementation_commit") == implementation_commit
            and payload.get("run_contract_sha256") == run_contract_sha256
            and payload.get("reviewed_artifact_sha256") == reviewed
            and _is_sha256(payload.get("evidence_contract_sha256"))
            and payload.get("evidence_contract_sha256")
            == sha256_mapping(unsigned)
        ):
            raise AgentReviewReceiptError("REVIEW_EVIDENCE_BINDING_INVALID")
        _validate_layer_result(layer_id, payload, len(reviewed))
        normalized.append(
            {
                "layer_id": layer_id,
                "status": "PASS",
                "evidence_relative_path": relative,
                "evidence_sha256": sha256_file(path),
                "evidence_contract_sha256": payload[
                    "evidence_contract_sha256"
                ],
            }
        )
    return sorted(normalized, key=lambda row: row["layer_id"])


def finalize_task4_agent_review(
    output_root: str | Path,
    *,
    project_root: str | Path,
    evidence_relative_paths: Iterable[str],
) -> dict[str, Any]:
    """Promote a pending Task-4 bundle only after three hashed agent layers pass."""

    root = Path(output_root)
    state_path = root / "pretraining_validation_state.json"
    state = read_json(state_path)
    existing_hashes = state.get("artifact_sha256")
    if not isinstance(existing_hashes, Mapping) or not existing_hashes:
        raise AgentReviewReceiptError("PENDING_ARTIFACT_HASHES_MISSING")
    if not (
        state.get("status") == "AWAITING_AGENT_MULTI_PASS_QC"
        and state.get("stage") == "ENDPOINT_SPLIT_SCANNER_EXTERNAL_VERIFIED"
        and state.get("agent_multi_pass_qc_status")
        == "AGENT_MULTI_PASS_QC_PENDING"
        and state.get("training_started") is False
        and state.get("outer_evaluation_started") is False
        and state.get("external_outcomes_accessed") is False
    ):
        raise AgentReviewReceiptError("TASK4_STATE_NOT_PENDING_REVIEW")
    reviewed_hashes = {str(name): str(digest) for name, digest in existing_hashes.items()}
    for basename, expected in reviewed_hashes.items():
        if Path(basename).name != basename or not _is_sha256(expected):
            raise AgentReviewReceiptError("PENDING_ARTIFACT_HASH_INVALID")
        path = root / basename
        if not path.is_file() or sha256_file(path) != expected:
            raise AgentReviewReceiptError("PENDING_ARTIFACT_HASH_MISMATCH")
    implementation_commit = state.get("implementation_commit")
    run_contract_sha256 = state.get("run_contract_sha256")
    validate_task4_run_contract(
        project_root,
        root,
        state=state,
        reviewed_artifact_sha256=reviewed_hashes,
    )
    layers = validate_task4_review_evidence(
        root,
        evidence_relative_paths=evidence_relative_paths,
        implementation_commit=implementation_commit,
        run_contract_sha256=run_contract_sha256,
        reviewed_artifact_sha256=reviewed_hashes,
    )
    receipt = {
        "schema_version": "2.0",
        "status": "PASS",
        "review_type": "LAYERED_AGENT_MULTI_PASS_NO_HUMAN_SIGNOFF",
        "implementation_commit": implementation_commit,
        "run_contract_sha256": run_contract_sha256,
        "reviewed_artifact_sha256": dict(sorted(reviewed_hashes.items())),
        "layers": layers,
        "training_started": False,
        "outer_evaluation_started": False,
        "external_outcomes_accessed": False,
    }
    receipt["receipt_sha256"] = sha256_mapping(receipt)
    receipt_name = "agent_multi_pass_review_receipt_v2.json"
    receipt_path = root / receipt_name
    atomic_write_json(receipt_path, receipt)
    committed_hashes = {
        **reviewed_hashes,
        receipt_name: sha256_file(receipt_path),
    }
    committed = {
        **state,
        "status": "COMMITTED",
        "artifact_sha256": dict(sorted(committed_hashes.items())),
        "agent_multi_pass_qc_status": "AGENT_MULTI_PASS_QC_PASS",
        "agent_multi_pass_review_receipt_sha256": receipt["receipt_sha256"],
    }
    atomic_write_json(state_path, committed)
    return committed
