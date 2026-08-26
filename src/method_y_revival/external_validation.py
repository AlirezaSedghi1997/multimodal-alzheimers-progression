from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Callable, Mapping

import yaml


class ExternalValidationError(ValueError):
    """Raised if external outcomes influence model fitting or selection."""


def external_claim_status(
    *, internal_gate_passed: bool, external_gate_passed: bool, refit_used: bool
) -> str:
    if refit_used:
        raise ExternalValidationError("EXTERNAL_REFIT_FORBIDDEN")
    if internal_gate_passed and external_gate_passed:
        return "EXTERNALLY_VALIDATED"
    if internal_gate_passed:
        return "INTERNAL_ONLY"
    return "EXPLORATORY"


def validate_external_policy(
    *, refit_used: bool, feature_reselection_used: bool, recalibration_used: bool
) -> None:
    if refit_used:
        raise ExternalValidationError("EXTERNAL_REFIT_FORBIDDEN")
    if feature_reselection_used:
        raise ExternalValidationError("EXTERNAL_FEATURE_RESELECTION_FORBIDDEN")
    if recalibration_used:
        raise ExternalValidationError("PRIMARY_EXTERNAL_RECALIBRATION_FORBIDDEN")


def _canonical_digest(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(
            payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")
    ).hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    os.replace(temporary, path)


def _request_spec(mapping_path: Path) -> dict[str, Any]:
    candidate = mapping_path.resolve().parents[1] / "method_y_external_data_request_v2.yaml"
    if not candidate.is_file():
        candidate = Path(__file__).resolve().parents[2] / "configs" / "method_y_external_data_request_v2.yaml"
    if not candidate.is_file():
        return {
            "status": "REQUEST_SPEC_NOT_AVAILABLE",
            "sha256": "0" * 64,
        }
    return {
        "status": "REQUEST_SPEC_ONLY_NO_DATA_RECEIVED",
        "path": candidate.as_posix(),
        "sha256": hashlib.sha256(candidate.read_bytes()).hexdigest(),
    }


def build_external_readiness(
    *, mapping_path: str | Path, output_path: str | Path
) -> dict[str, Any]:
    """Build a non-blocking internal / fail-closed external readiness record."""

    source = Path(mapping_path)
    try:
        mapping = yaml.safe_load(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise ExternalValidationError("EXTERNAL_MAPPING_INVALID") from exc
    if not isinstance(mapping, dict) or not isinstance(mapping.get("cohorts"), dict):
        raise ExternalValidationError("EXTERNAL_MAPPING_INVALID")
    policy = mapping.get("policy") if isinstance(mapping.get("policy"), dict) else {}
    validate_external_policy(
        refit_used=bool(policy.get("refit", False)),
        feature_reselection_used=bool(policy.get("feature_reselection", False)),
        recalibration_used=bool(policy.get("primary_recalibration", False)),
    )
    if policy.get("external_outcome_accessed") is True:
        raise ExternalValidationError("EXTERNAL_OUTCOME_ACCESS_BEFORE_MAPPING_LOCK")
    blockers: list[str] = []
    available = False
    mapping_keys = (
        "outcome_mapping",
        "amyloid_mapping",
        "tau_mapping",
        "region_mapping",
    )
    for cohort, raw in sorted(mapping["cohorts"].items()):
        if not isinstance(raw, dict):
            raise ExternalValidationError("EXTERNAL_COHORT_MAPPING_INVALID")
        access = str(raw.get("access_status") or "")
        dictionary = str(raw.get("data_dictionary_status") or "")
        cohort_available = access in {"APPROVED", "DUA_APPROVED", "GRANTED"}
        available = available or cohort_available
        resolved = True
        for key in mapping_keys:
            item = raw.get(key)
            if not isinstance(item, dict):
                raise ExternalValidationError("EXTERNAL_COHORT_MAPPING_INVALID")
            status = str(item.get("status") or "")
            if status in {"RESOLVED", "LOCKED", "PASS"}:
                fields = item.get("fields")
                if not isinstance(fields, list) or not fields or not all(
                    isinstance(value, str) and value for value in fields
                ):
                    raise ExternalValidationError("RESOLVED_MAPPING_FIELDS_REQUIRED")
            else:
                resolved = False
        dictionary_ready = dictionary in {"AVAILABLE", "LOCKED", "HASH_LOCKED"}
        if not cohort_available or not dictionary_ready or not resolved:
            blockers.append(f"{str(cohort).upper()}_DUA_AND_DICTIONARY_PENDING")
    external_state = (
        "EXTERNAL_AVAILABLE_PENDING_ELIGIBILITY"
        if available
        else "EXTERNAL_UNAVAILABLE_NONBLOCKING"
    )
    report = {
        "schema_version": 2,
        "status": "EXTERNAL_PENDING" if blockers else "EXTERNAL_ELIGIBLE_LOCKED_NOT_RUN",
        "external_state": external_state if blockers else "EXTERNAL_ELIGIBLE_LOCKED_NOT_RUN",
        "science_gate_status": "BLOCKED" if blockers else "PASS",
        "claim_status": "INTERNAL_ONLY_MAXIMUM_IF_INTERNAL_GATE_PASSES",
        "external_claim_allowed": False,
        "required_for_internal_completion": False,
        "internal_execution_blocked": False,
        "external_outcome_accessed": False,
        "model_refit_allowed": False,
        "feature_reselection_allowed": False,
        "primary_recalibration_allowed": False,
        "blockers": blockers,
        "blocker_scope": "EXTERNAL_CLAIM_ONLY",
        "unavailable_branch": {
            "status": "ROBUSTNESS_EVIDENCE_NOT_EXTERNAL_VALIDATION",
            "required_analyses": [
                "OOF_PREDICTIONS",
                "PAIRED_SUBJECT_BOOTSTRAP",
                "CALIBRATION",
                "REPEATED_SEEDS",
                "NEGATIVE_CONTROLS",
                "SUBGROUP_AND_COHORT_STRESS_WHERE_ESTIMABLE",
                "DOMAIN_SHIFT_STRESS",
            ],
            "external_generalization_claim_allowed": False,
        },
        "request_spec": _request_spec(source),
    }
    _atomic_json(Path(output_path), report)
    return report


def lock_external_transport_contract(
    *,
    root: str | Path,
    cohort: str,
    mapping_digest: str,
    feature_intersection: list[str],
    model_digest: str,
    preprocessing_digest: str,
) -> dict[str, Any]:
    digests = (mapping_digest, model_digest, preprocessing_digest)
    if any(
        len(value) != 64 or any(char not in "0123456789abcdef" for char in value)
        for value in digests
    ):
        raise ExternalValidationError("EXTERNAL_CONTRACT_DIGEST_INVALID")
    features = list(dict.fromkeys(str(value) for value in feature_intersection if str(value)))
    if not features:
        raise ExternalValidationError("EXTERNAL_FEATURE_INTERSECTION_EMPTY")
    body = {
        "schema_version": 1,
        "artifact": "method_y_external_transport_contract",
        "cohort": str(cohort).upper(),
        "mapping_digest": mapping_digest,
        "feature_intersection": features,
        "model_digest": model_digest,
        "preprocessing_digest": preprocessing_digest,
        "refit_allowed": False,
        "feature_reselection_allowed": False,
        "primary_recalibration_allowed": False,
        "outer_selection_allowed": False,
    }
    contract = {**body, "contract_digest": _canonical_digest(body)}
    path = Path(root) / "external_validation" / str(cohort).lower() / "transport_contract.json"
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != contract:
            raise ExternalValidationError("EXTERNAL_CONTRACT_ALREADY_LOCKED")
    else:
        _atomic_json(path, contract)
    return contract


class ExternalEvaluationStore:
    """Exactly-once external evaluation store; never an adaptation interface."""

    def __init__(self, root: str | Path, *, cohort: str, contract_digest: str):
        self.root = Path(root)
        self.cohort = str(cohort).upper()
        self.contract_digest = str(contract_digest)
        directory = self.root / "external_validation" / self.cohort.lower()
        self.state_path = directory / "external_evaluation_state.json"
        self.result_path = directory / "external_evaluation_result.json"

    def evaluate_once(
        self,
        *,
        evaluator: Callable[[], Mapping[str, Any]],
        refit_used: bool = False,
        feature_reselection_used: bool = False,
        recalibration_used: bool = False,
        threshold_reselection_used: bool = False,
    ) -> dict[str, Any]:
        validate_external_policy(
            refit_used=refit_used,
            feature_reselection_used=feature_reselection_used,
            recalibration_used=recalibration_used,
        )
        if threshold_reselection_used:
            raise ExternalValidationError("EXTERNAL_THRESHOLD_RESELECTION_FORBIDDEN")
        if self.state_path.exists():
            state = json.loads(self.state_path.read_text(encoding="utf-8"))
            if (
                state.get("state") != "EXTERNAL_EVAL_COMMITTED"
                or state.get("contract_digest") != self.contract_digest
                or not self.result_path.is_file()
                or state.get("result_sha256")
                != hashlib.sha256(self.result_path.read_bytes()).hexdigest()
            ):
                raise ExternalValidationError("EXTERNAL_EVALUATION_STATE_INVALID")
            result = json.loads(self.result_path.read_text(encoding="utf-8"))
            if not isinstance(result, dict):
                raise ExternalValidationError("EXTERNAL_EVALUATION_RESULT_INVALID")
            return result
        if self.result_path.exists():
            raise ExternalValidationError("EXTERNAL_EVALUATION_AMBIGUOUS_COMMIT")
        result = dict(evaluator())
        if not result or any(
            isinstance(value, float) and not math.isfinite(value)
            for value in result.values()
        ):
            raise ExternalValidationError("EXTERNAL_EVALUATION_RESULT_INVALID")
        _atomic_json(self.result_path, result)
        state = {
            "schema_version": 1,
            "state": "EXTERNAL_EVAL_COMMITTED",
            "cohort": self.cohort,
            "contract_digest": self.contract_digest,
            "evaluation_count": 1,
            "result_path": self.result_path.name,
            "result_sha256": hashlib.sha256(self.result_path.read_bytes()).hexdigest(),
            "refit_used": False,
            "feature_reselection_used": False,
            "primary_recalibration_used": False,
            "threshold_reselection_used": False,
        }
        _atomic_json(self.state_path, state)
        return result
