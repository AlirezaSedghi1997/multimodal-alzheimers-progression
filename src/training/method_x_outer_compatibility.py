"""Pinned authority for the five historical Method X outer-result mismatches.

The receipt is a forensic exception, not a generic mismatch waiver.  It binds
the exact pre-repair freeze, every source-migration receipt, and exactly five
model/fold rows.  No writer or outer-test loader is called here.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from src.utils.reproducibility import canonical_digest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
HISTORICAL_COMPATIBILITY_RECEIPT = (
    PROJECT_ROOT
    / "governance"
    / "method_x_migration_receipts"
    / "method_x_historical_outer_result_compatibility_receipt.v1.json"
)
TRUSTED_HISTORICAL_COMPATIBILITY_RECEIPT_SHA256 = (
    "82241ac3a12b598ccd17a8a0912c1d8484b4b3de0644f8b6f4735eeac3687a57"
)
TRUSTED_PRE_REPAIR_FREEZE_SHA256 = (
    "55a8d55d64462e363db9987500e73353c32716ac958a47717511a2c658a73cbf"
)
_EXPECTED_ROWS = frozenset(
    {
        ("resnet18_3d", 0),
        ("resnet18_3d", 1),
        ("resnet34_3d", 0),
        ("resnet34_3d", 1),
        ("resnet34_3d", 2),
    }
)
_REPARSE_POINT_ATTRIBUTE = 0x0400
_AUTHORITY_TOKEN = object()


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


def _assert_regular_project_file(path: Path) -> None:
    try:
        relative = path.absolute().relative_to(PROJECT_ROOT.absolute())
    except ValueError as exc:
        raise RuntimeError("HISTORICAL_OUTER_COMPATIBILITY_PATH_INVALID") from exc
    current = PROJECT_ROOT
    for part in relative.parts:
        current = current / part
        try:
            metadata = os.lstat(current)
        except OSError as exc:
            raise RuntimeError("HISTORICAL_OUTER_COMPATIBILITY_EVIDENCE_MISSING") from exc
        if stat.S_ISLNK(metadata.st_mode) or bool(
            getattr(metadata, "st_file_attributes", 0) & _REPARSE_POINT_ATTRIBUTE
        ):
            raise RuntimeError("HISTORICAL_OUTER_COMPATIBILITY_REPARSE_REJECTED")
    if not stat.S_ISREG(os.lstat(path).st_mode):
        raise RuntimeError("HISTORICAL_OUTER_COMPATIBILITY_NOT_REGULAR")


@dataclass(frozen=True, slots=True, init=False)
class HistoricalCompatibilityAuthority:
    """Sealed evidence produced only after verifying the pinned H receipt."""

    evidence: Mapping[str, Any]
    receipt_sha256: str

    def __init__(
        self,
        evidence: Mapping[str, Any],
        receipt_sha256: str,
        *,
        _token: object,
    ) -> None:
        if _token is not _AUTHORITY_TOKEN:
            raise RuntimeError("HISTORICAL_OUTER_COMPATIBILITY_NOT_AUTHORIZED")
        object.__setattr__(self, "evidence", dict(evidence))
        object.__setattr__(self, "receipt_sha256", str(receipt_sha256))


def _validate_source_receipt_link(*, link: Mapping[str, Any], fold: int) -> None:
    required = {
        "receipt_path",
        "receipt_sha256",
        "old_run_contract_digest",
        "new_run_contract_digest",
        "pre_checkpoint_sha256",
        "post_checkpoint_sha256",
        "pre_outer_result_digest",
        "post_outer_result_digest",
    }
    if set(link) != required or not all(
        _is_sha256(link.get(key))
        for key in required
        if key != "receipt_path"
    ):
        raise RuntimeError("HISTORICAL_OUTER_COMPATIBILITY_CHAIN_INVALID")
    relative = link.get("receipt_path")
    if not isinstance(relative, str) or Path(relative).is_absolute() or ".." in Path(relative).parts:
        raise RuntimeError("HISTORICAL_OUTER_COMPATIBILITY_CHAIN_INVALID")
    receipt_path = PROJECT_ROOT / relative
    _assert_regular_project_file(receipt_path)
    if _sha256_file(receipt_path) != link["receipt_sha256"]:
        raise RuntimeError("HISTORICAL_OUTER_COMPATIBILITY_CHAIN_TAMPERED")
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("HISTORICAL_OUTER_COMPATIBILITY_CHAIN_INVALID") from exc
    unsigned = dict(receipt) if isinstance(receipt, dict) else {}
    recorded_integrity = unsigned.pop("integrity_sha256", None)
    folds = receipt.get("folds") if isinstance(receipt, dict) else None
    matches = (
        [row for row in folds if isinstance(row, dict) and row.get("fold") == fold]
        if isinstance(folds, list)
        else []
    )
    if (
        not isinstance(receipt, dict)
        or receipt.get("artifact") != "method_x_source_only_contract_migration_receipt"
        or receipt.get("append_only") is not True
        or recorded_integrity != canonical_digest(unsigned)
        or receipt.get("old_run_contract_digest") != link["old_run_contract_digest"]
        or receipt.get("new_run_contract_digest") != link["new_run_contract_digest"]
        or len(matches) != 1
    ):
        raise RuntimeError("HISTORICAL_OUTER_COMPATIBILITY_CHAIN_INVALID")
    row = matches[0]
    for field in (
        "pre_checkpoint_sha256",
        "post_checkpoint_sha256",
        "pre_outer_result_digest",
        "post_outer_result_digest",
    ):
        if row.get(field) != link[field]:
            raise RuntimeError("HISTORICAL_OUTER_COMPATIBILITY_CHAIN_INVALID")


def _validate_entry(row: Mapping[str, Any]) -> tuple[str, int]:
    if set(row) != {"compatibility_evidence", "migration_chain"}:
        raise RuntimeError("HISTORICAL_OUTER_COMPATIBILITY_RECEIPT_INVALID")
    evidence = row.get("compatibility_evidence")
    chain = row.get("migration_chain")
    evidence_keys = {
        "schema_version",
        "artifact",
        "scope",
        "backbone",
        "fold",
        "checkpoint_sha256",
        "outer_file_sha256",
        "run_contract_digest",
        "split_membership_digest",
        "evaluation_count",
        "embedded_result_digest",
        "external_result_digest",
        "prediction_identity_digest",
        "outer_loader_invocation_count",
    }
    if (
        not isinstance(evidence, dict)
        or set(evidence) != evidence_keys
        or evidence.get("schema_version") != 1
        or evidence.get("artifact")
        != "method_x_historical_outer_result_compatibility_entry"
        or evidence.get("scope") != "EXACT_FIVE_FROZEN_ARTIFACTS_ONLY"
        or type(evidence.get("fold")) is not int
        or type(evidence.get("evaluation_count")) is not int
        or evidence.get("evaluation_count") != 1
        or type(evidence.get("outer_loader_invocation_count")) is not int
        or evidence.get("outer_loader_invocation_count") != 0
        or not all(
            _is_sha256(evidence.get(field))
            for field in (
                "checkpoint_sha256",
                "outer_file_sha256",
                "run_contract_digest",
                "split_membership_digest",
                "embedded_result_digest",
                "external_result_digest",
                "prediction_identity_digest",
            )
        )
        or not isinstance(chain, list)
        or not chain
    ):
        raise RuntimeError("HISTORICAL_OUTER_COMPATIBILITY_RECEIPT_INVALID")
    fold = int(evidence["fold"])
    previous_outer = evidence["embedded_result_digest"]
    previous_checkpoint: str | None = None
    previous_contract: str | None = None
    for item in chain:
        if not isinstance(item, dict):
            raise RuntimeError("HISTORICAL_OUTER_COMPATIBILITY_CHAIN_INVALID")
        _validate_source_receipt_link(link=item, fold=fold)
        if item["pre_outer_result_digest"] != previous_outer:
            raise RuntimeError("HISTORICAL_OUTER_COMPATIBILITY_CHAIN_INVALID")
        if previous_checkpoint is not None and item["pre_checkpoint_sha256"] != previous_checkpoint:
            raise RuntimeError("HISTORICAL_OUTER_COMPATIBILITY_CHAIN_INVALID")
        if previous_contract is not None and item["old_run_contract_digest"] != previous_contract:
            raise RuntimeError("HISTORICAL_OUTER_COMPATIBILITY_CHAIN_INVALID")
        previous_outer = item["post_outer_result_digest"]
        previous_checkpoint = item["post_checkpoint_sha256"]
        previous_contract = item["new_run_contract_digest"]
    if (
        previous_outer != evidence["external_result_digest"]
        or previous_checkpoint != evidence["checkpoint_sha256"]
        or previous_contract != evidence["run_contract_digest"]
        or evidence["embedded_result_digest"] == evidence["external_result_digest"]
    ):
        raise RuntimeError("HISTORICAL_OUTER_COMPATIBILITY_CHAIN_INVALID")
    return str(evidence["backbone"]), fold


def load_historical_compatibility_entry(
    path: str | Path,
    *,
    backbone: str,
    fold: int,
) -> HistoricalCompatibilityAuthority:
    """Load one row only after validating the exact pinned five-row receipt."""

    receipt_path = Path(path)
    if receipt_path.absolute() != HISTORICAL_COMPATIBILITY_RECEIPT.absolute():
        raise RuntimeError("HISTORICAL_OUTER_COMPATIBILITY_PATH_INVALID")
    _assert_regular_project_file(receipt_path)
    receipt_sha256 = _sha256_file(receipt_path)
    if receipt_sha256 != TRUSTED_HISTORICAL_COMPATIBILITY_RECEIPT_SHA256:
        raise RuntimeError("HISTORICAL_OUTER_COMPATIBILITY_RECEIPT_TAMPERED")
    try:
        payload = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("HISTORICAL_OUTER_COMPATIBILITY_RECEIPT_INVALID") from exc
    unsigned = dict(payload) if isinstance(payload, dict) else {}
    recorded_integrity = unsigned.pop("integrity_sha256", None)
    top_keys = {
        "schema_version",
        "artifact",
        "append_only",
        "scope",
        "pre_repair_freeze_path",
        "pre_repair_freeze_sha256",
        "entry_count",
        "entries",
        "integrity_sha256",
    }
    if (
        not isinstance(payload, dict)
        or set(payload) != top_keys
        or payload.get("schema_version") != 1
        or payload.get("artifact")
        != "method_x_historical_outer_result_compatibility_receipt"
        or payload.get("append_only") is not True
        or payload.get("scope") != "EXACT_FIVE_FROZEN_ARTIFACTS_ONLY"
        or payload.get("entry_count") != 5
        or recorded_integrity != canonical_digest(unsigned)
        or payload.get("pre_repair_freeze_sha256")
        != TRUSTED_PRE_REPAIR_FREEZE_SHA256
    ):
        raise RuntimeError("HISTORICAL_OUTER_COMPATIBILITY_RECEIPT_INVALID")
    freeze_relative = payload.get("pre_repair_freeze_path")
    if (
        not isinstance(freeze_relative, str)
        or Path(freeze_relative).is_absolute()
        or ".." in Path(freeze_relative).parts
    ):
        raise RuntimeError("HISTORICAL_OUTER_COMPATIBILITY_RECEIPT_INVALID")
    freeze_path = PROJECT_ROOT / freeze_relative
    _assert_regular_project_file(freeze_path)
    if _sha256_file(freeze_path) != TRUSTED_PRE_REPAIR_FREEZE_SHA256:
        raise RuntimeError("HISTORICAL_OUTER_COMPATIBILITY_FREEZE_TAMPERED")
    entries = payload.get("entries")
    if not isinstance(entries, list) or len(entries) != 5:
        raise RuntimeError("HISTORICAL_OUTER_COMPATIBILITY_RECEIPT_INVALID")
    indexed: dict[tuple[str, int], Mapping[str, Any]] = {}
    for row in entries:
        if not isinstance(row, dict):
            raise RuntimeError("HISTORICAL_OUTER_COMPATIBILITY_RECEIPT_INVALID")
        key = _validate_entry(row)
        if key in indexed:
            raise RuntimeError("HISTORICAL_OUTER_COMPATIBILITY_RECEIPT_INVALID")
        indexed[key] = row["compatibility_evidence"]
    if frozenset(indexed) != _EXPECTED_ROWS:
        raise RuntimeError("HISTORICAL_OUTER_COMPATIBILITY_RECEIPT_INVALID")
    selected = indexed.get((str(backbone), int(fold)))
    if selected is None:
        raise RuntimeError("HISTORICAL_OUTER_COMPATIBILITY_NOT_AUTHORIZED")
    return HistoricalCompatibilityAuthority(
        selected,
        receipt_sha256,
        _token=_AUTHORITY_TOKEN,
    )

