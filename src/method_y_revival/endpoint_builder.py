from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from .atomic_io import atomic_write_json, read_json
from .contracts import sha256_mapping


class EndpointBuildError(ValueError):
    """Raised when a future endpoint cannot be frozen without privacy or membership defects."""


PRIVATE_ENDPOINT_FIELDS = frozenset(
    {"raw_path", "source_subject_id", "BID", "SUBSTUDY", "sidecar_path"}
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_locked_endpoint(
    rows: Iterable[dict[str, Any]],
    *,
    output_path: str | Path,
    contract_path: str | Path,
    source_hashes: dict[str, str],
    endpoint_definition: dict[str, Any],
) -> dict[str, Any]:
    materialized = list(rows)
    if not materialized:
        raise EndpointBuildError("EMPTY_ENDPOINT")
    offenders = sorted(
        {field for row in materialized for field in PRIVATE_ENDPOINT_FIELDS if field in row}
    )
    if offenders:
        raise EndpointBuildError(f"PRIVATE_ENDPOINT_FIELD_FORBIDDEN:{offenders}")
    patient_hashes = [str(row.get("patient_hash", "")) for row in materialized]
    if any(not value for value in patient_hashes):
        raise EndpointBuildError("PATIENT_HASH_REQUIRED")
    if len(set(patient_hashes)) != len(patient_hashes):
        raise EndpointBuildError("DUPLICATE_ENDPOINT_SUBJECT")
    for row in materialized:
        if int(row.get("event", -1)) not in {0, 1}:
            raise EndpointBuildError("BINARY_EVENT_REQUIRED")
        if float(row.get("time_days", 0.0)) <= 0.0:
            raise EndpointBuildError("POSITIVE_OBSERVED_TIME_REQUIRED")

    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(materialized).sort_values("patient_hash")
    storage_format = "parquet"
    parquet_status = "AVAILABLE"
    stored_target = target
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp.parquet", dir=str(target.parent)
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        try:
            frame.to_parquet(temporary, index=False)
        except ImportError:
            storage_format = "csv_fallback"
            parquet_status = "BLOCKED_PARQUET_ENGINE_UNAVAILABLE"
            stored_target = target.with_suffix(".csv")
            temporary.unlink(missing_ok=True)
            descriptor, temporary_name = tempfile.mkstemp(
                prefix=f".{stored_target.name}.", suffix=".tmp.csv", dir=str(target.parent)
            )
            os.close(descriptor)
            temporary = Path(temporary_name)
            frame.to_csv(temporary, index=False, encoding="utf-8", lineterminator="\n")
        os.replace(temporary, stored_target)
    finally:
        temporary.unlink(missing_ok=True)
    endpoint_sha256 = sha256_file(stored_target)
    contract = {
        "schema_version": "1.0",
        "locked": True,
        "row_count": len(materialized),
        "event_count": sum(int(row["event"]) for row in materialized),
        "endpoint_sha256": endpoint_sha256,
        "endpoint_path": stored_target.name,
        "storage_format": storage_format,
        "parquet_status": parquet_status,
        "source_hashes": dict(sorted(source_hashes.items())),
        "endpoint_definition": endpoint_definition,
    }
    contract["contract_sha256"] = sha256_mapping(contract)
    atomic_write_json(contract_path, contract)
    return contract


def build_tau_endpoint_from_sources(
    *, config: dict[str, Any], output_root: str | Path
) -> dict[str, Any]:
    from scripts.build_method_y_tau_progression import (
        build_label_rows,
        load_first_tau_scans,
        read_csv_rows,
    )

    raw_root = Path(config["paths"]["raw_root"])
    sources = {
        "tau_suvr_csv": raw_root / config["clinical_sources"]["tau_suvr"],
        "cdr_csv": raw_root / config["clinical_sources"]["cdr"],
        "pacc_csv": raw_root / config["clinical_sources"]["pacc"],
        "subjinfo_csv": raw_root / config["clinical_sources"]["subject_info"],
    }
    missing = [name for name, path in sources.items() if not path.is_file()]
    if missing:
        raise EndpointBuildError(f"MISSING_ENDPOINT_SOURCES:{missing}")
    source_hashes = {name: sha256_file(path) for name, path in sources.items()}
    endpoint_definition = {
        "primary_name": "sustained_cdr_global_progression",
        "event_time_rule": config["endpoint"]["event_time_rule"],
        "baseline_cdr_required": config["endpoint"]["baseline_cdr_required"],
        "pet_to_baseline_window_days": config["endpoint"][
            "pet_to_baseline_window_days"
        ],
        "followup_window_days": config["endpoint"]["followup_window_days"],
        "forbidden_label_sources": config["endpoint"]["forbidden_label_sources"],
    }
    output_directory = Path(output_root) / "endpoints"
    contract_path = output_directory / "endpoint_contract.json"
    summary_path = output_directory / "tau_endpoint_summary.json"
    if contract_path.exists():
        existing = read_json(contract_path)
        if existing.get("source_hashes") != source_hashes:
            raise EndpointBuildError("LOCKED_ENDPOINT_SOURCE_HASH_MISMATCH")
        if existing.get("endpoint_definition") != endpoint_definition:
            raise EndpointBuildError("LOCKED_ENDPOINT_DEFINITION_MISMATCH")
        stored = output_directory / str(existing.get("endpoint_path", ""))
        if not stored.is_file() or sha256_file(stored) != existing.get("endpoint_sha256"):
            raise EndpointBuildError("LOCKED_ENDPOINT_ARTIFACT_HASH_MISMATCH")
        summary = read_json(summary_path) if summary_path.exists() else {"contract": existing}
        summary["status"] = "ENDPOINT_LOCKED_REUSED"
        return summary

    compatibility_config = {
        "inputs": {
            "tau_primary_suvr_column": "suvr_persi",
            "tau_sensitivity_suvr_column": "suvr_crus",
            "tau_unavailable_suvr_column": "suvr_cer",
            "tau_scan_analyzed_required": "Yes",
            "tau_ligand_required": "Flortaucipir",
        },
        "endpoint": {
            "baseline_tau_to_cdr_window_days": config["endpoint"][
                "pet_to_baseline_window_days"
            ],
            "followup_window_days": config["endpoint"]["followup_window_days"],
            "baseline_cdr_required": config["endpoint"]["baseline_cdr_required"],
            "event_time_rule": config["endpoint"]["event_time_rule"],
        },
    }
    first_tau, tau_audit = load_first_tau_scans(
        sources["tau_suvr_csv"], compatibility_config
    )
    rows, endpoint_audit, region_features = build_label_rows(
        first_tau,
        read_csv_rows(sources["subjinfo_csv"]),
        read_csv_rows(sources["cdr_csv"]),
        read_csv_rows(sources["pacc_csv"]),
        compatibility_config,
    )
    output = output_directory / "tau_progression_endpoint.parquet"
    contract = write_locked_endpoint(
        rows,
        output_path=output,
        contract_path=contract_path,
        source_hashes=source_hashes,
        endpoint_definition=endpoint_definition,
    )
    summary = {
        "status": "ENDPOINT_LOCKED",
        "contract": contract,
        "tau_source_audit": tau_audit,
        "endpoint_audit": endpoint_audit,
        "regional_feature_count": len(region_features),
        "raw_identifiers_exported": False,
        "raw_paths_exported": False,
    }
    atomic_write_json(summary_path, summary)
    return summary
