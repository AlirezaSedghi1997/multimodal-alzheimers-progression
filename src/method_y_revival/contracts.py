from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Mapping

from .constants import ARTIFACT_STATUSES, BRANCHES, CLAIM_STATUSES, SCIENCE_GATE_STATUSES


class ContractError(ValueError):
    """Raised when a claim or artifact violates the immutable workflow contract."""


class RunContractMismatch(ContractError):
    """Raised instead of resuming into an output root produced by another contract."""


def canonical_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_mapping(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class RunContract:
    payload: dict[str, Any]
    digest: str

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "RunContract":
        normalized = json.loads(canonical_json(payload))
        return cls(payload=normalized, digest=sha256_mapping(normalized))

    def require_match(self, other: "RunContract") -> None:
        if self.digest != other.digest:
            raise RunContractMismatch(
                f"RUN_CONTRACT_MISMATCH:existing={self.digest}:current={other.digest}"
            )


@dataclass
class ArtifactRecord:
    artifact_id: str
    stage: str
    branch: str
    artifact_status: str
    science_gate_status: str
    claim_status: str
    terminal_reason: str | None = None
    schema_version: str = "1.0"
    created_utc: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    git_commit: str | None = None
    dirty_tree_digest: str | None = None
    config_sha256: str | None = None
    run_contract_sha256: str | None = None
    input_artifact_ids: list[str] = field(default_factory=list)
    input_sha256: dict[str, str] = field(default_factory=dict)
    output_sha256: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.artifact_id.strip():
            raise ContractError("ARTIFACT_ID_REQUIRED")
        if self.branch not in BRANCHES:
            raise ContractError(f"INVALID_BRANCH:{self.branch}")
        if self.artifact_status not in ARTIFACT_STATUSES:
            raise ContractError(f"INVALID_ARTIFACT_STATUS:{self.artifact_status}")
        if self.science_gate_status not in SCIENCE_GATE_STATUSES:
            raise ContractError(f"INVALID_SCIENCE_GATE_STATUS:{self.science_gate_status}")
        if self.claim_status not in CLAIM_STATUSES:
            raise ContractError(f"INVALID_CLAIM_STATUS:{self.claim_status}")
        if self.claim_status == "EXTERNALLY_VALIDATED" and self.science_gate_status != "PASS":
            raise ContractError("EXTERNAL_CLAIM_REQUIRES_SCIENCE_GATE_PASS")
        if self.claim_status == "INTERNAL_ONLY" and self.science_gate_status != "PASS":
            raise ContractError("INTERNAL_CLAIM_REQUIRES_SCIENCE_GATE_PASS")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
