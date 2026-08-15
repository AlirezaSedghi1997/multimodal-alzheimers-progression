"""Fail-closed, branch-local Method Y v2 readiness records.

This module deliberately separates *permission to develop a source-regional
model* from image validation, completed outer evaluation, and claim gates.
It consumes aggregate contract evidence only.  Callers must provide evidence
explicitly; no raw images, external outcomes, identifiers, or private tables
are read by :func:`build_branch_readiness`.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable, Mapping

from .atomic_io import atomic_write_bytes, atomic_write_json
from .contracts import sha256_mapping
from .review_receipt import (
    AgentReviewReceiptError,
    REQUIRED_TASK4_REVIEW_LAYERS,
    validate_task4_review_evidence,
    validate_task4_run_contract,
)


BRANCH_NAMES = (
    "amyloid_source_regional",
    "amyloid_image",
    "tau_source_regional",
    "tau_image",
    "paired_source_regional",
    "external_transport",
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
SOURCE_ROLES = frozenset(
    {
        "caller_supplied_aggregate_contracts",
        "read_only_project_aggregate_contracts",
    }
)


class BranchReadinessError(ValueError):
    """Raised when a caller attempts to publish an inconsistent readiness record."""


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and _SHA256.fullmatch(value.lower()) is not None


def _locked_contract(payload: Mapping[str, Any]) -> bool:
    return payload.get("locked") is True and any(
        _is_sha256(payload.get(name))
        for name in ("feature_digest", "contract_sha256", "run_contract_sha256")
    )


def valid_locked_split_contract(payload: Mapping[str, Any]) -> bool:
    """Accept only the prespecified 5x3, subject-grouped, hash-locked split."""

    return (
        payload.get("group_unit") == "subject_key"
        and payload.get("outer_fold_count") == 5
        and payload.get("inner_fold_count") == 3
        and _is_sha256(payload.get("split_digest"))
    )


def _declared_contract_digest(payload: Mapping[str, Any]) -> str | None:
    for name in ("contract_sha256", "run_contract_sha256"):
        value = payload.get(name)
        if _is_sha256(value):
            return str(value).lower()
    return None


def _complete_internal_outer_evaluation(
    payload: Mapping[str, Any], *, split_digest: str | None, model_contract_digest: str | None
) -> bool:
    """Require exactly-once completion of every locked outer fold and bindings."""

    return (
        payload.get("status") == "COMMITTED"
        and payload.get("exactly_once") is True
        and payload.get("outer_fold_count") == 5
        and payload.get("outer_folds_completed") == 5
        and _is_sha256(payload.get("split_digest"))
        and _is_sha256(payload.get("model_contract_sha256"))
        and split_digest is not None
        and model_contract_digest is not None
        and payload.get("split_digest", "").lower() == split_digest
        and payload.get("model_contract_sha256", "").lower() == model_contract_digest
    )


def _valid_evidence(name: str, value: Any) -> dict[str, str] | None:
    if not isinstance(value, Mapping):
        return None
    artifact_id = value.get("artifact_id")
    status = value.get("status")
    digest = value.get("sha256")
    if (
        not isinstance(artifact_id, str)
        or not artifact_id.strip()
        or artifact_id != name
        or not isinstance(status, str)
        or status != "PASS"
        or not _is_sha256(digest)
    ):
        return None
    return {"artifact_id": artifact_id, "status": "PASS", "sha256": digest.lower()}


def _evidence_for(
    evidence: Mapping[str, Any], names: Iterable[str]
) -> tuple[dict[str, dict[str, str]], list[str]]:
    accepted: dict[str, dict[str, str]] = {}
    blockers: list[str] = []
    for name in names:
        if name not in evidence:
            blockers.append(f"MISSING_EVIDENCE:{name}")
            continue
        item = _valid_evidence(name, evidence[name])
        if item is None:
            blockers.append(f"INVALID_EVIDENCE:{name}")
        else:
            accepted[name] = item
    return accepted, blockers


def _public_evidence(items: Mapping[str, dict[str, str]]) -> list[dict[str, str]]:
    return [items[name] for name in sorted(items)]


def _branch_names(branch: str) -> dict[str, str]:
    return {
        "feature": f"{branch}_feature_contract",
        "endpoint": f"{branch}_endpoint_contract",
        "split": f"{branch}_split_contract",
        "temporality": "method_y_predictor_temporality_contract",
        "model": f"{branch}_model_contract",
        "outer": f"{branch}_outer_evaluation",
        "science": f"{branch}_science_gate",
    }


def _scanner_claim_blocker(branch: str) -> str:
    if branch.startswith("amyloid"):
        return "AMYLOID_SCANNER_SENSITIVITY_NOT_PASSED"
    return "TAU_SECOND_MANUFACTURER_EVENT_SUPPORT_INSUFFICIENT"


def _source_record(
    branch: str,
    evidence: Mapping[str, Any],
) -> dict[str, Any]:
    names = _branch_names(branch)
    development_required = (
        names["feature"],
        names["endpoint"],
        names["split"],
        names["temporality"],
    )
    dev_evidence, blockers = _evidence_for(evidence, development_required)
    development_allowed = not blockers

    model_evidence, outer_blockers = _evidence_for(evidence, (names["model"],))
    outer_evaluation_allowed = development_allowed and not outer_blockers
    if not development_allowed:
        outer_blockers = ["DEVELOPMENT_CONTRACTS_NOT_LOCKED", *outer_blockers]

    claim_required = (names["outer"], names["science"])
    claim_evidence, claim_blockers = _evidence_for(evidence, claim_required)
    if not outer_evaluation_allowed:
        claim_blockers.insert(0, "OUTER_EVALUATION_NOT_ALLOWED")
    scanner_evidence, scanner_blockers = _evidence_for(
        evidence, (f"{branch}_scanner_generalization",)
    )
    claim_evidence.update(scanner_evidence)
    if scanner_blockers:
        claim_blockers.append(_scanner_claim_blocker(branch))
    claim_allowed = not claim_blockers

    return {
        "branch": branch,
        "scope": "source_regional",
        "development_allowed": development_allowed,
        "outer_evaluation_allowed": outer_evaluation_allowed,
        "claim_allowed": claim_allowed,
        "external_claim_allowed": False,
        "blockers": blockers,
        "outer_evaluation_blockers": outer_blockers,
        "claim_blockers": claim_blockers,
        "external_claim_blockers": ["EXTERNAL_TRANSPORT_NOT_COMMITTED"],
        "evidence": _public_evidence({**dev_evidence, **model_evidence, **claim_evidence}),
    }


def _image_record(branch: str, evidence: Mapping[str, Any]) -> dict[str, Any]:
    names = _branch_names(branch)
    prefix = "AMYLOID" if branch.startswith("amyloid") else "TAU"
    development_required = (
        names["feature"],
        names["endpoint"],
        names["split"],
        f"{branch}_identity_gate",
        f"{branch}_calibration_gate",
    )
    accepted, raw_blockers = _evidence_for(evidence, development_required)
    blockers: list[str] = []
    for blocker in raw_blockers:
        name = blocker.rsplit(":", 1)[-1]
        if name.endswith("_identity_gate"):
            blockers.append(f"{prefix}_IMAGE_IDENTITY_NOT_LOCKED")
        elif name.endswith("_calibration_gate"):
            blockers.append(f"{prefix}_IMAGE_CALIBRATION_NOT_LOCKED")
        else:
            blockers.append(blocker)
    development_allowed = not blockers

    model_evidence, outer_blockers = _evidence_for(evidence, (names["model"],))
    outer_evaluation_allowed = development_allowed and not outer_blockers
    if not development_allowed:
        outer_blockers = ["DEVELOPMENT_CONTRACTS_NOT_LOCKED", *outer_blockers]
    claim_evidence, claim_blockers = _evidence_for(
        evidence, (names["outer"], names["science"])
    )
    scanner_evidence, scanner_blockers = _evidence_for(
        evidence, (f"{branch}_scanner_generalization",)
    )
    claim_evidence.update(scanner_evidence)
    if scanner_blockers:
        claim_blockers.append(_scanner_claim_blocker(branch))
    if not outer_evaluation_allowed:
        claim_blockers.insert(0, "OUTER_EVALUATION_NOT_ALLOWED")
    claim_allowed = not claim_blockers
    return {
        "branch": branch,
        "scope": "image",
        "development_allowed": development_allowed,
        "outer_evaluation_allowed": outer_evaluation_allowed,
        "claim_allowed": claim_allowed,
        "external_claim_allowed": False,
        "blockers": blockers,
        "outer_evaluation_blockers": outer_blockers,
        "claim_blockers": claim_blockers,
        "external_claim_blockers": ["EXTERNAL_TRANSPORT_NOT_COMMITTED"],
        "evidence": _public_evidence({**accepted, **model_evidence, **claim_evidence}),
    }


def _external_record(evidence: Mapping[str, Any]) -> dict[str, Any]:
    development_required = (
        "external_dua",
        "external_data",
        "external_mapping_contract",
        "external_locked_model_contract",
    )
    accepted, blockers = _evidence_for(evidence, development_required)
    development_allowed = not blockers
    outer_evaluation_allowed = development_allowed
    claim_evidence, claim_blockers = _evidence_for(
        evidence, ("external_evaluation", "external_science_gate")
    )
    claim_blockers = [
        "EXTERNAL_EVALUATION_NOT_COMMITTED"
        if blocker == "MISSING_EVIDENCE:external_evaluation"
        else blocker
        for blocker in claim_blockers
    ]
    if not outer_evaluation_allowed:
        claim_blockers.insert(0, "EXTERNAL_EVALUATION_NOT_ALLOWED")
    external_claim_allowed = not claim_blockers
    return {
        "branch": "external_transport",
        "scope": "external_transport",
        "development_allowed": development_allowed,
        "outer_evaluation_allowed": outer_evaluation_allowed,
        "claim_allowed": external_claim_allowed,
        "external_claim_allowed": external_claim_allowed,
        "blockers": blockers,
        "outer_evaluation_blockers": [] if outer_evaluation_allowed else list(blockers),
        "claim_blockers": [] if external_claim_allowed else ["EXTERNAL_TRANSPORT_CLAIM_NOT_ALLOWED"],
        "external_claim_blockers": claim_blockers,
        "evidence": _public_evidence({**accepted, **claim_evidence}),
    }


def assert_branch_readiness_invariants(report: Mapping[str, Any]) -> None:
    snapshot = report.get("evidence_snapshot")
    if not isinstance(snapshot, Mapping):
        raise BranchReadinessError("EVIDENCE_SNAPSHOT_MISSING")
    if snapshot.get("source_role") not in SOURCE_ROLES:
        raise BranchReadinessError("INVALID_EVIDENCE_SOURCE_ROLE")
    if not _is_sha256(snapshot.get("sha256")):
        raise BranchReadinessError("EVIDENCE_SNAPSHOT_DIGEST_INVALID")
    if not isinstance(snapshot.get("evidence_item_count"), int) or snapshot["evidence_item_count"] < 0:
        raise BranchReadinessError("EVIDENCE_SNAPSHOT_COUNT_INVALID")
    branches = report.get("branches")
    if not isinstance(branches, Mapping) or set(branches) != set(BRANCH_NAMES):
        raise BranchReadinessError("BRANCH_SET_INVALID")
    for name, record in branches.items():
        if not isinstance(record, Mapping):
            raise BranchReadinessError(f"BRANCH_RECORD_INVALID:{name}")
        if record.get("claim_allowed") and not record.get("outer_evaluation_allowed"):
            raise BranchReadinessError(f"CLAIM_WITHOUT_OUTER_PERMISSION:{name}")
        if record.get("external_claim_allowed") and not record.get("claim_allowed"):
            raise BranchReadinessError(f"EXTERNAL_CLAIM_WITHOUT_CLAIM_PERMISSION:{name}")
        if record.get("external_claim_allowed") and name == "external_transport":
            required = {"external_dua", "external_data", "external_mapping_contract", "external_locked_model_contract", "external_evaluation", "external_science_gate"}
            present = {item["artifact_id"] for item in record.get("evidence", [])}
            if not required.issubset(present):
                raise BranchReadinessError("EXTERNAL_CLAIM_MISSING_MAPPING_MODEL_OR_EVALUATION")


def _evidence_snapshot(evidence: Mapping[str, Any], *, source_role: str) -> dict[str, Any]:
    if source_role not in SOURCE_ROLES:
        raise BranchReadinessError("INVALID_EVIDENCE_SOURCE_ROLE")
    accepted = {
        name: item
        for name, value in evidence.items()
        if isinstance(name, str) and (item := _valid_evidence(name, value)) is not None
    }
    canonical = json.dumps(accepted, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return {
        "source_role": source_role,
        "evidence_item_count": len(accepted),
        "sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
    }


def build_branch_readiness(
    evidence: Mapping[str, Any], *, source_role: str = "caller_supplied_aggregate_contracts"
) -> dict[str, Any]:
    """Build branch-local readiness from explicit, aggregate evidence only."""

    if not isinstance(evidence, Mapping):
        evidence = {}
    branches = {
        "amyloid_source_regional": _source_record("amyloid_source_regional", evidence),
        "amyloid_image": _image_record("amyloid_image", evidence),
        "tau_source_regional": _source_record("tau_source_regional", evidence),
        "tau_image": _image_record("tau_image", evidence),
        "paired_source_regional": _source_record("paired_source_regional", evidence),
        "external_transport": _external_record(evidence),
    }
    report = {
        "schema_version": "1.1",
        "evidence_snapshot": _evidence_snapshot(evidence, source_role=source_role),
        "branches": branches,
    }
    assert_branch_readiness_invariants(report)
    return report


def _json_snapshot(path: Path) -> tuple[Mapping[str, Any], str] | None:
    """Read, parse and hash exactly the same immutable byte snapshot once."""

    if not path.is_file():
        return None
    try:
        raw = path.read_bytes()
        payload = json.loads(raw.decode("utf-8-sig"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(payload, Mapping):
        return None
    return payload, hashlib.sha256(raw).hexdigest()


def _locked_json_evidence(path: Path, artifact_id: str, *, predicate) -> dict[str, str] | None:
    snapshot = _json_snapshot(path)
    if snapshot is None:
        return None
    payload, digest = snapshot
    try:
        accepted = bool(predicate(payload))
    except (AttributeError, KeyError, TypeError, ValueError):
        return None
    if not accepted:
        return None
    return {"artifact_id": artifact_id, "status": "PASS", "sha256": digest}


def collect_project_evidence(project_root: str | Path, *, branches: Iterable[str] = BRANCH_NAMES) -> dict[str, dict[str, str]]:
    """Hash only the aggregate contracts required by selected readiness branches.

    Selecting a source-regional branch never opens an image-cache voxel file or
    an external outcome file.  Missing/unlocked artifacts are intentionally
    omitted so that :func:`build_branch_readiness` reports a fail-closed gate.
    """

    requested = set(branches)
    unknown = requested.difference(BRANCH_NAMES)
    if unknown:
        raise BranchReadinessError(f"UNKNOWN_READINESS_BRANCH:{','.join(sorted(unknown))}")
    root = Path(project_root)
    result_root = root / "results" / "method_y_revival_v2"
    collected: dict[str, dict[str, str]] = {}
    snapshot_cache: dict[Path, tuple[Mapping[str, Any], str] | None] = {}
    state_snapshot = _json_snapshot(result_root / "feature_contracts" / "source_regional_validation_state.json")
    task4_root = result_root / "pretraining_contracts"
    task4_state_snapshot = _json_snapshot(
        task4_root / "pretraining_validation_state.json"
    )
    task4_artifacts = frozenset(
        {
            "endpoint_verification_v2.json",
            "predictor_temporality_contract_v2.json",
            "split_verification_v2.json",
            "scanner_feasibility_and_sensitivity_contract_v2.json",
            "external_intake_contract_v2.json",
            "PRETRAINING_CONTRACTS_TASK4.md",
            "agent_multi_pass_review_receipt_v2.json",
        }
    )

    def source_state_allows(path: Path, digest: str) -> bool:
        if state_snapshot is None:
            return False
        state, _ = state_snapshot
        hashes = state.get("artifact_sha256")
        return state.get("status") == "COMMITTED" and state.get("development_allowed") is True and isinstance(hashes, Mapping) and _is_sha256(hashes.get(path.name)) and hashes.get(path.name) == digest

    def snapshot_for(path: Path) -> tuple[Mapping[str, Any], str] | None:
        key = path.resolve()
        if key not in snapshot_cache:
            snapshot_cache[key] = _json_snapshot(key)
        return snapshot_cache[key]

    def task4_bundle_valid() -> bool:
        if task4_state_snapshot is None:
            return False
        state, _ = task4_state_snapshot
        hashes = state.get("artifact_sha256")
        run_contract = state.get("run_contract")
        if not (
            state.get("status") == "COMMITTED"
            and state.get("stage")
            == "ENDPOINT_SPLIT_SCANNER_EXTERNAL_VERIFIED"
            and state.get("training_started") is False
            and state.get("outer_evaluation_started") is False
            and state.get("external_outcomes_accessed") is False
            and state.get("agent_multi_pass_qc_status")
            == "AGENT_MULTI_PASS_QC_PASS"
            and state.get("release_eligible") is True
            and isinstance(run_contract, Mapping)
            and run_contract.get("release_eligible") is True
            and run_contract.get("reconstruction_mode")
            == "LOCKED_RAW_SOURCE_INDEPENDENT_REBUILD"
            and state.get("implementation_commit")
            == run_contract.get("implementation_commit")
            and _is_sha256(state.get("run_contract_sha256"))
            and state.get("run_contract_sha256")
            == sha256_mapping(run_contract)
            and isinstance(hashes, Mapping)
            and set(hashes) == task4_artifacts
        ):
            return False
        for basename in task4_artifacts:
            if Path(basename).name != basename or not _is_sha256(hashes.get(basename)):
                return False
            path = task4_root / basename
            try:
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
            except OSError:
                return False
            if digest != hashes.get(basename):
                return False
        receipt_snapshot = _json_snapshot(
            task4_root / "agent_multi_pass_review_receipt_v2.json"
        )
        if receipt_snapshot is None:
            return False
        receipt, _ = receipt_snapshot
        unsigned_receipt = {
            key: value for key, value in receipt.items() if key != "receipt_sha256"
        }
        layers = receipt.get("layers")
        reviewed = receipt.get("reviewed_artifact_sha256")
        reviewed_expected = {
            name: digest
            for name, digest in hashes.items()
            if name != "agent_multi_pass_review_receipt_v2.json"
        }
        try:
            validate_task4_run_contract(
                root,
                task4_root,
                state=state,
                reviewed_artifact_sha256=reviewed_expected,
            )
            validated_layers = validate_task4_review_evidence(
                task4_root,
                evidence_relative_paths=[
                    row.get("evidence_relative_path", "")
                    for row in layers
                    if isinstance(row, Mapping)
                ]
                if isinstance(layers, list)
                else [],
                implementation_commit=state.get("implementation_commit"),
                run_contract_sha256=state.get("run_contract_sha256"),
                reviewed_artifact_sha256=reviewed_expected,
            )
        except AgentReviewReceiptError:
            return False
        if not (
            receipt.get("status") == "PASS"
            and receipt.get("review_type")
            == "LAYERED_AGENT_MULTI_PASS_NO_HUMAN_SIGNOFF"
            and _is_sha256(receipt.get("receipt_sha256"))
            and receipt.get("receipt_sha256") == sha256_mapping(unsigned_receipt)
            and state.get("agent_multi_pass_review_receipt_sha256")
            == receipt.get("receipt_sha256")
            and receipt.get("implementation_commit")
            == state.get("implementation_commit")
            and receipt.get("run_contract_sha256")
            == state.get("run_contract_sha256")
            and isinstance(reviewed, Mapping)
            and dict(reviewed) == reviewed_expected
            and isinstance(layers, list)
            and {row.get("layer_id") for row in layers if isinstance(row, Mapping)}
            == REQUIRED_TASK4_REVIEW_LAYERS
            and all(
                isinstance(row, Mapping)
                and row.get("status") == "PASS"
                and _is_sha256(row.get("evidence_sha256"))
                for row in layers
            )
            and validated_layers == layers
            and run_contract.get("artifact_sha256") == reviewed_expected
        ):
            return False
        return True

    task4_bundle_is_valid = task4_bundle_valid()

    def task4_endpoint_record(
        payload: Mapping[str, Any], branch: str
    ) -> Mapping[str, Any] | None:
        stem = "paired_at" if branch == "paired_source_regional" else branch.split("_", 1)[0]
        records = payload.get("primary_endpoints")
        if not isinstance(records, Mapping):
            return None
        record = records.get(stem)
        if not isinstance(record, Mapping):
            return None
        rebuild = record.get("independent_rebuild")
        if not isinstance(rebuild, Mapping):
            return None
        if not (
            record.get("status") == "PASS"
            and record.get("locked") is True
            and record.get("future_only") is True
            and record.get("pre_pet_outcome_used") is False
            and record.get("sustained_confirmation_required") is True
            and record.get("terminal_unconfirmed_increase_is_event") is False
            and isinstance(record.get("row_count"), int)
            and record["row_count"] > 0
            and isinstance(record.get("event_count"), int)
            and record["event_count"] > 0
            and _is_sha256(record.get("contract_sha256"))
            and _is_sha256(record.get("endpoint_artifact_sha256"))
            and rebuild.get("performed") is True
            and rebuild.get("row_digest_match") is True
            and rebuild.get("source_definition_binding_match") is True
        ):
            return None
        return record

    def task4_split_record(
        payload: Mapping[str, Any], branch: str
    ) -> Mapping[str, Any] | None:
        branches_payload = payload.get("branches")
        if not isinstance(branches_payload, Mapping):
            return None
        record = branches_payload.get(branch)
        if not isinstance(record, Mapping):
            return None
        expected_role = (
            "PRIMARY_DEVELOPMENT_A4"
            if branch == "amyloid_source_regional"
            else "A4_EXPLORATORY_DEVELOPMENT"
        )
        if not (
            record.get("status") == "PASS"
            and record.get("locked") is True
            and record.get("analysis_role") == expected_role
            and record.get("development_cohort") == "A4"
            and record.get("outer_fold_count") == 5
            and record.get("inner_fold_count") == 3
            and record.get("complete_outer_test_partition") is True
            and record.get("subject_overlap_detected") is False
            and record.get("all_inner_and_outer_partitions_have_both_event_classes")
            is True
            and _is_sha256(record.get("split_digest"))
            and _is_sha256(record.get("membership_digest"))
            and _is_sha256(record.get("private_artifact_sha256"))
        ):
            return None
        basename = record.get("private_artifact_basename")
        if not isinstance(basename, str) or Path(basename).name != basename:
            return None
        private_path = task4_root / "_private" / basename
        try:
            private_raw = private_path.read_bytes()
            private = json.loads(private_raw.decode("utf-8-sig"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            return None
        if not isinstance(private, Mapping):
            return None
        if hashlib.sha256(private_raw).hexdigest() != record.get(
            "private_artifact_sha256"
        ):
            return None
        split = private.get("split")
        if not (
            private.get("locked") is True
            and private.get("branch") == branch
            and private.get("endpoint_membership_digest")
            == record.get("membership_digest")
            and isinstance(split, Mapping)
            and split.get("split_digest") == record.get("split_digest")
        ):
            return None
        return record

    def task4_temporality_record(
        payload: Mapping[str, Any], branch: str
    ) -> Mapping[str, Any] | None:
        declared = payload.get("contract_sha256")
        unsigned = {
            key: value for key, value in payload.items() if key != "contract_sha256"
        }
        allowed = payload.get("allowed_predictors")
        allowed_digest = payload.get("allowed_predictor_digest")
        bindings = payload.get("source_endpoint_contract_sha256")
        if not (
            payload.get("status") == "PASS"
            and payload.get("locked") is True
            and _is_sha256(declared)
            and declared == sha256_mapping(unsigned)
            and isinstance(allowed, list)
            and _is_sha256(allowed_digest)
            and allowed_digest
            == sha256_mapping({"allowed_predictors": allowed})
            and isinstance(bindings, Mapping)
        ):
            return None
        endpoint_snapshot = snapshot_for(
            task4_root / "endpoint_verification_v2.json"
        )
        if endpoint_snapshot is None:
            return None
        endpoint_record = task4_endpoint_record(endpoint_snapshot[0], branch)
        if endpoint_record is None:
            return None
        stem = (
            "paired_at"
            if branch == "paired_source_regional"
            else branch.split("_", 1)[0]
        )
        if bindings.get(stem) != endpoint_record.get("contract_sha256"):
            return None
        return payload

    def add(path: Path, artifact_id: str, predicate) -> Mapping[str, Any] | None:
        snapshot = snapshot_for(path)
        if snapshot is None:
            return None
        payload, digest = snapshot
        if artifact_id.endswith("_source_regional_feature_contract") and not source_state_allows(path, digest):
            return None
        try:
            accepted = bool(predicate(payload))
        except (AttributeError, KeyError, TypeError, ValueError):
            return None
        if not accepted:
            return None
        collected[artifact_id] = {
            "artifact_id": artifact_id,
            "status": "PASS",
            "sha256": digest,
        }
        return payload

    for branch, _stem in (
        ("amyloid_source_regional", "amyloid"),
        ("tau_source_regional", "tau"),
        ("paired_source_regional", "paired_at"),
    ):
        if branch not in requested:
            continue
        add(
            result_root / "feature_contracts" / f"{branch}_feature_contract.json",
            f"{branch}_feature_contract",
            _locked_contract,
        )
        endpoint_verification = task4_root / "endpoint_verification_v2.json"
        if task4_bundle_is_valid:
            add(
                endpoint_verification,
                f"{branch}_endpoint_contract",
                lambda payload, selected=branch: task4_endpoint_record(
                    payload, selected
                )
                is not None,
            )
            add(
                task4_root / "predictor_temporality_contract_v2.json",
                "method_y_predictor_temporality_contract",
                lambda payload, selected=branch: task4_temporality_record(
                    payload, selected
                )
                is not None,
            )
        split_verification = task4_root / "split_verification_v2.json"
        task4_split_payload: Mapping[str, Any] | None = None
        if task4_bundle_is_valid:
            aggregate_split = add(
                split_verification,
                f"{branch}_split_contract",
                lambda payload, selected=branch: task4_split_record(payload, selected)
                is not None,
            )
            if aggregate_split is not None:
                task4_split_payload = task4_split_record(aggregate_split, branch)
        split_payload = task4_split_payload
        model_payload = add(
            result_root / "locked" / f"{branch}_model_contract.json",
            f"{branch}_model_contract",
            _locked_contract,
        )
        add(
            result_root / "outer_evaluation" / branch / "committed.json",
            f"{branch}_outer_evaluation",
            lambda payload: _complete_internal_outer_evaluation(
                payload,
                split_digest=(
                    str(split_payload.get("split_digest")).lower()
                    if split_payload is not None
                    else None
                ),
                model_contract_digest=(
                    _declared_contract_digest(model_payload)
                    if model_payload is not None
                    else None
                ),
            ),
        )
        add(
            result_root / "reports" / f"{branch}_science_gate.json",
            f"{branch}_science_gate",
            lambda payload: payload.get("science_gate_status") == "PASS",
        )
        scanner_kind = "amyloid" if branch == "amyloid_source_regional" else "tau"
        add(
            result_root / "reports" / f"scanner_generalization_{scanner_kind}.json",
            f"{branch}_scanner_generalization",
            lambda payload: payload.get("science_gate_status") == "PASS",
        )

    for branch, aggregate in (
        ("amyloid_image", "scan_level_amyloid_image_regional_validation.json"),
        ("tau_image", "scan_level_tau_quantitative_validation.json"),
    ):
        if branch not in requested:
            continue
        add(
            result_root / "feature_contracts" / f"{branch}_feature_contract.json",
            f"{branch}_feature_contract",
            _locked_contract,
        )
        endpoint_stem = "amyloid" if branch.startswith("amyloid") else "tau"
        add(
            result_root / "endpoints" / f"{endpoint_stem}_endpoint_contract.json",
            f"{branch}_endpoint_contract",
            _locked_contract,
        )
        split_payload = add(
            result_root / "splits" / f"{endpoint_stem}_nested_splits.json",
            f"{branch}_split_contract",
            valid_locked_split_contract,
        )
        add(
            result_root / "provenance" / aggregate,
            f"{branch}_identity_gate",
            lambda payload: payload.get("science_gate_status") == "PASS",
        )
        add(
            result_root / "provenance" / "quantitative_pet_feature_validation_v2.json",
            f"{branch}_calibration_gate",
            lambda payload: payload.get("science_gate_status") == "PASS",
        )
        model_payload = add(
            result_root / "locked" / f"{branch}_model_contract.json",
            f"{branch}_model_contract",
            _locked_contract,
        )
        add(
            result_root / "outer_evaluation" / branch / "committed.json",
            f"{branch}_outer_evaluation",
            lambda payload: _complete_internal_outer_evaluation(
                payload,
                split_digest=(
                    str(split_payload.get("split_digest")).lower()
                    if split_payload is not None
                    else None
                ),
                model_contract_digest=(
                    _declared_contract_digest(model_payload)
                    if model_payload is not None
                    else None
                ),
            ),
        )
        add(
            result_root / "reports" / f"{branch}_science_gate.json",
            f"{branch}_science_gate",
            lambda payload: payload.get("science_gate_status") == "PASS",
        )
        scanner_kind = "amyloid" if branch == "amyloid_image" else "tau"
        add(
            result_root / "reports" / f"scanner_generalization_{scanner_kind}.json",
            f"{branch}_scanner_generalization",
            lambda payload: payload.get("science_gate_status") == "PASS",
        )

    if "external_transport" in requested:
        add(
            result_root / "external_validation" / "readiness.json",
            "external_dua",
            lambda payload: payload.get("dua_status") == "APPROVED",
        )
        add(
            result_root / "external_validation" / "readiness.json",
            "external_data",
            lambda payload: payload.get("data_status") == "AVAILABLE",
        )
        add(
            result_root / "external_validation" / "mapping_contract.json",
            "external_mapping_contract",
            _locked_contract,
        )
        add(
            result_root / "locked" / "external_transport_model_contract.json",
            "external_locked_model_contract",
            _locked_contract,
        )
        add(
            result_root / "external_validation" / "committed.json",
            "external_evaluation",
            lambda payload: payload.get("status") == "COMMITTED" and payload.get("exactly_once") is True,
        )
        add(
            result_root / "external_validation" / "science_gate.json",
            "external_science_gate",
            lambda payload: payload.get("science_gate_status") == "PASS",
        )
    return collected


def render_branch_readiness_markdown(report: Mapping[str, Any]) -> str:
    assert_branch_readiness_invariants(report)
    snapshot = report["evidence_snapshot"]
    lines = [
        "# Method Y v2 branch readiness",
        "",
        "All permissions are fail-closed and branch-local.",
        f"- Evidence source role: `{snapshot['source_role']}`.",
        f"- Evidence snapshot SHA-256: `{snapshot['sha256']}`; items: `{snapshot['evidence_item_count']}`.",
        "- Regenerate with: `python scripts/build_method_y_branch_readiness_v2.py --project-root $PROJECT_ROOT --output-root results/method_y_revival_v2/readiness`.",
        "",
    ]
    for branch in BRANCH_NAMES:
        record = report["branches"][branch]
        lines.extend(
            [
                f"## {branch}",
                "",
                f"- development_allowed: `{str(record['development_allowed']).lower()}`",
                f"- outer_evaluation_allowed: `{str(record['outer_evaluation_allowed']).lower()}`",
                f"- claim_allowed: `{str(record['claim_allowed']).lower()}`",
                f"- external_claim_allowed: `{str(record['external_claim_allowed']).lower()}`",
                f"- blockers: `{', '.join(record['blockers']) or 'none'}`",
                f"- outer evaluation blockers: `{', '.join(record['outer_evaluation_blockers']) or 'none'}`",
                f"- claim blockers: `{', '.join(record['claim_blockers']) or 'none'}`",
                f"- external claim blockers: `{', '.join(record['external_claim_blockers']) or 'none'}`",
                "- aggregate evidence:",
            ]
        )
        for item in record["evidence"]:
            lines.append(f"  - `{item['artifact_id']}`: SHA-256 `{item['sha256']}`")
        lines.append("")
    return "\n".join(lines)


def write_branch_readiness(report: Mapping[str, Any], output_root: str | Path) -> tuple[Path, Path]:
    """Atomically publish aggregate/hash-level JSON and Markdown readiness artifacts."""

    assert_branch_readiness_invariants(report)
    root = Path(output_root)
    json_path = root / "branch_readiness_v2.json"
    markdown_path = root / "branch_readiness_v2.md"
    atomic_write_bytes(markdown_path, render_branch_readiness_markdown(report).encode("utf-8"))
    atomic_write_json(json_path, report)
    return json_path, markdown_path
