"""Resolve immutable finalized Method X artifacts across source-only changes."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from src.training.method_x_outer_compatibility import (
    HISTORICAL_COMPATIBILITY_RECEIPT,
    HistoricalCompatibilityAuthority,
    load_historical_compatibility_entry,
)
from src.training.method_x_resume import scientific_contract_digest
from src.utils.reproducibility import canonical_digest


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(char in "0123456789abcdef" for char in value.lower())
    )


def _receipt_integrity(payload: Mapping[str, Any]) -> str:
    unsigned = dict(payload)
    unsigned.pop("integrity_sha256", None)
    return canonical_digest(unsigned)


@dataclass(frozen=True, slots=True)
class FinalizedSourceCompatibility:
    artifact_contract_digest: str
    target_contract_digest: str
    transition_count: int
    historical_authority: HistoricalCompatibilityAuthority | None


def _resolve_receipt_path(model_dir: Path, relative: Any) -> Path:
    if not isinstance(relative, str):
        raise RuntimeError("FINALIZED_SOURCE_COMPATIBILITY_RECEIPT_INVALID")
    relative_path = Path(relative)
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise RuntimeError("FINALIZED_SOURCE_COMPATIBILITY_RECEIPT_INVALID")
    path = model_dir / relative_path
    try:
        path.resolve().relative_to(model_dir.resolve())
    except (OSError, ValueError) as exc:
        raise RuntimeError("FINALIZED_SOURCE_COMPATIBILITY_RECEIPT_INVALID") from exc
    if path.is_symlink() or not path.is_file():
        raise RuntimeError("FINALIZED_SOURCE_COMPATIBILITY_RECEIPT_INVALID")
    return path


def _advance_source_chain(
    *,
    model_dir: Path,
    history: list[Any],
    artifact_digest: str,
    target_digest: str,
    fold: int,
    checkpoint_sha256: str,
    outer_result_digest: str,
) -> int:
    if artifact_digest == target_digest:
        return 0
    current = artifact_digest
    transitions = 0
    used_paths: set[str] = set()
    while current != target_digest:
        candidates = [
            link
            for link in history
            if isinstance(link, dict)
            and link.get("from_run_contract_digest") == current
        ]
        if len(candidates) != 1:
            raise RuntimeError("FINALIZED_SOURCE_COMPATIBILITY_CHAIN_INVALID")
        link = candidates[0]
        relative = link.get("receipt_path")
        expected_receipt_sha256 = link.get("receipt_sha256")
        target = link.get("to_run_contract_digest")
        if (
            not isinstance(relative, str)
            or relative in used_paths
            or not _is_sha256(expected_receipt_sha256)
            or not _is_sha256(target)
        ):
            raise RuntimeError("FINALIZED_SOURCE_COMPATIBILITY_CHAIN_INVALID")
        used_paths.add(relative)
        receipt_path = _resolve_receipt_path(model_dir, relative)
        if _sha256_file(receipt_path) != expected_receipt_sha256:
            raise RuntimeError("FINALIZED_SOURCE_COMPATIBILITY_RECEIPT_TAMPERED")
        try:
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise RuntimeError("FINALIZED_SOURCE_COMPATIBILITY_RECEIPT_INVALID") from exc
        folds = receipt.get("folds") if isinstance(receipt, dict) else None
        matches = (
            [row for row in folds if isinstance(row, dict) and row.get("fold") == fold]
            if isinstance(folds, list)
            else []
        )
        if (
            not isinstance(receipt, dict)
            or receipt.get("schema_version") != 1
            or receipt.get("artifact")
            != "method_x_source_only_contract_migration_receipt"
            or receipt.get("append_only") is not True
            or receipt.get("integrity_sha256") != _receipt_integrity(receipt)
            or receipt.get("old_run_contract_digest") != current
            or receipt.get("new_run_contract_digest") != target
            or len(matches) != 1
        ):
            raise RuntimeError("FINALIZED_SOURCE_COMPATIBILITY_RECEIPT_INVALID")
        row = matches[0]
        if (
            row.get("pre_checkpoint_sha256") != checkpoint_sha256
            or row.get("post_checkpoint_sha256") != checkpoint_sha256
            or row.get("pre_outer_result_digest") != outer_result_digest
            or row.get("post_outer_result_digest") != outer_result_digest
        ):
            raise RuntimeError("FINALIZED_SOURCE_COMPATIBILITY_ARTIFACT_MUTATION")
        current = str(target)
        transitions += 1
        if transitions > len(history):
            raise RuntimeError("FINALIZED_SOURCE_COMPATIBILITY_CHAIN_INVALID")
    return transitions


def resolve_finalized_source_compatibility(
    *,
    model_dir: str | Path,
    state: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
    locked_outer: Mapping[str, Any],
    expected_contract: Mapping[str, Any],
    expected_fold: int,
    checkpoint_sha256: str,
    outer_file_sha256: str,
) -> FinalizedSourceCompatibility:
    """Validate artifact-at-creation contract, then prove a chain to current."""

    directory = Path(model_dir)
    artifact_contract = checkpoint.get("run_contract")
    artifact_digest = checkpoint.get("run_contract_digest")
    target_digest = expected_contract.get("digest")
    outer_digest = locked_outer.get("result_digest")
    details = state.get("details") if isinstance(state.get("details"), Mapping) else None
    if (
        state.get("stage") != "FINALIZED"
        or state.get("fold") != int(expected_fold)
        or state.get("contract_digest") != target_digest
        or checkpoint.get("fold") != int(expected_fold)
        or not isinstance(artifact_contract, Mapping)
        or not _is_sha256(artifact_digest)
        or artifact_contract.get("digest") != artifact_digest
        or canonical_digest(
            {key: value for key, value in artifact_contract.items() if key != "digest"}
        )
        != artifact_digest
        or not _is_sha256(target_digest)
        or canonical_digest(
            {key: value for key, value in expected_contract.items() if key != "digest"}
        )
        != target_digest
        or scientific_contract_digest(artifact_contract)
        != scientific_contract_digest(expected_contract)
        or locked_outer.get("fold") != int(expected_fold)
        or locked_outer.get("evaluation_count") != 1
        or locked_outer.get("run_contract_digest") != artifact_digest
        or locked_outer.get("result_digest")
        != canonical_digest(
            {key: value for key, value in locked_outer.items() if key != "result_digest"}
        )
        or not _is_sha256(checkpoint_sha256)
        or not _is_sha256(outer_file_sha256)
        or not isinstance(details, Mapping)
        or details.get("final_checkpoint_sha256") != checkpoint_sha256
        or details.get("outer_result_digest") != outer_digest
        or details.get("split_membership_digest")
        != locked_outer.get("split_membership_digest")
        or checkpoint.get("split_membership_digest")
        not in {None, locked_outer.get("split_membership_digest")}
    ):
        raise RuntimeError("FINALIZED_SOURCE_COMPATIBILITY_INVALID")
    history = state.get("source_history", [])
    if not isinstance(history, list):
        raise RuntimeError("FINALIZED_SOURCE_COMPATIBILITY_CHAIN_INVALID")
    transition_count = _advance_source_chain(
        model_dir=directory,
        history=history,
        artifact_digest=str(artifact_digest),
        target_digest=str(target_digest),
        fold=int(expected_fold),
        checkpoint_sha256=checkpoint_sha256,
        outer_result_digest=str(outer_digest),
    )
    committed = checkpoint.get("locked_outer_test_evaluation")
    historical_authority: HistoricalCompatibilityAuthority | None = None
    if isinstance(committed, Mapping) and committed.get("result_digest") != outer_digest:
        backbone = checkpoint.get("backbone")
        if not isinstance(backbone, str):
            raise RuntimeError("FINALIZED_SOURCE_COMPATIBILITY_INVALID")
        historical_authority = load_historical_compatibility_entry(
            HISTORICAL_COMPATIBILITY_RECEIPT,
            backbone=backbone,
            fold=int(expected_fold),
        )
        evidence = historical_authority.evidence
        if (
            evidence.get("checkpoint_sha256") != checkpoint_sha256
            or evidence.get("outer_file_sha256") != outer_file_sha256
            or evidence.get("run_contract_digest") != artifact_digest
            or evidence.get("embedded_result_digest") != committed.get("result_digest")
            or evidence.get("external_result_digest") != outer_digest
        ):
            raise RuntimeError("FINALIZED_SOURCE_COMPATIBILITY_HISTORICAL_MISMATCH")
    return FinalizedSourceCompatibility(
        artifact_contract_digest=str(artifact_digest),
        target_contract_digest=str(target_digest),
        transition_count=transition_count,
        historical_authority=historical_authority,
    )

