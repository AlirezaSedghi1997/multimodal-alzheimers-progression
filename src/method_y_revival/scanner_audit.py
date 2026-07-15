from __future__ import annotations

from collections import Counter, defaultdict
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from .atomic_io import atomic_write_json, read_json
from .endpoint_builder import load_first_amyloid_scans, sha256_file


def canonical_manufacturer(value: object) -> str:
    text = "" if value is None else str(value).strip()
    if not text or text.upper() in {"UNKNOWN", "MISSING", "NA", "N/A"}:
        return "UNKNOWN"
    folded = text.casefold()
    if "siemens" in folded:
        return "SIEMENS"
    if folded == "ge" or "general electric" in folded or folded.startswith("ge:"):
        return "GE"
    if "philips" in folded:
        return "PHILIPS"
    if "toshiba" in folded or "canon" in folded:
        return "TOSHIBA_CANON"
    return "OTHER"


def _same_relative_day(left: object, right: object) -> bool:
    try:
        return abs(float(left) - float(right)) <= 1e-6
    except (TypeError, ValueError):
        return False


def link_amyloid_scanner_rows(
    first_scans: dict[tuple[str, str], dict[str, Any]],
    private_manifest_rows: Iterable[dict[str, Any]],
    pet_input_rows: Iterable[dict[str, Any]],
    eligible_patient_hashes: set[str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    from scripts.build_method_y_tau_progression import patient_hash_for_key

    private_by_patient: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in private_manifest_rows:
        private_by_patient[str(row.get("patient_hash", ""))].append(dict(row))
    scanner_by_sample: dict[str, str] = {}
    for row in pet_input_rows:
        if str(row.get("tracer", "")).upper() != "FBP":
            continue
        scanner_by_sample[str(row.get("subject_id_hash", ""))] = canonical_manufacturer(
            row.get("scanner")
        )

    rows: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    for key, scan in sorted(first_scans.items(), key=lambda item: patient_hash_for_key(item[0])):
        patient_hash = patient_hash_for_key(key)
        if patient_hash not in eligible_patient_hashes:
            continue
        visit = str(scan.get("viscode", "")).strip()
        candidates = [
            row
            for row in private_by_patient.get(patient_hash, [])
            if _same_relative_day(row.get("pet_day"), scan.get("scan_date"))
            and (
                not visit
                or not str(row.get("visit_code", "")).strip()
                or str(row.get("visit_code", "")).strip() == visit
            )
        ]
        if not candidates:
            manufacturer = "UNKNOWN"
            status = "MISSING_EXACT_SCAN_LINK"
            counts["missing_exact_scan_link"] += 1
        else:
            manufacturers = {
                scanner_by_sample.get(str(row.get("sample_hash", "")), "UNKNOWN")
                for row in candidates
            }
            known = {value for value in manufacturers if value != "UNKNOWN"}
            if len(known) == 1:
                manufacturer = next(iter(known))
                status = "EXACT"
                counts["exact"] += 1
            elif len(known) > 1:
                manufacturer = "UNKNOWN"
                status = "AMBIGUOUS_EXACT_SCAN_METADATA"
                counts["ambiguous"] += 1
            else:
                manufacturer = "UNKNOWN"
                status = "UNKNOWN_EXACT_METADATA"
                counts["unknown"] += 1
        rows.append(
            {
                "branch": "amyloid",
                "patient_hash": patient_hash,
                "manufacturer": manufacturer,
                "linkage_status": status,
                "scanner_sensitivity_eligible": status == "EXACT",
            }
        )
    return rows, {
        "branch": "amyloid",
        "endpoint_membership": len(eligible_patient_hashes),
        "linked_rows": len(rows),
        "manufacturer_counts": dict(Counter(row["manufacturer"] for row in rows)),
        **dict(counts),
    }


def link_tau_scanner_rows(
    eligible_patient_hashes: set[str],
    manifest_rows: Iterable[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_patient: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for source in manifest_rows:
        patient_hash = str(source.get("patient_hash", ""))
        if patient_hash in eligible_patient_hashes:
            by_patient[patient_hash].append(dict(source))
    rows: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    for patient_hash in sorted(eligible_patient_hashes):
        candidates = by_patient.get(patient_hash, [])
        if len(candidates) != 1:
            manufacturer = "UNKNOWN"
            status = "MISSING_OR_AMBIGUOUS_MANIFEST_LINK"
            counts["missing_or_ambiguous"] += 1
        else:
            source = candidates[0]
            manufacturer = canonical_manufacturer(source.get("manufacturer"))
            if str(source.get("qc_status", "")).upper() != "PASS":
                status = "QC_EXCLUDED"
                counts["qc_excluded"] += 1
            elif manufacturer == "UNKNOWN":
                status = "UNKNOWN_EXACT_METADATA"
                counts["unknown"] += 1
            else:
                status = "EXACT"
                counts["eligible"] += 1
        rows.append(
            {
                "branch": "tau",
                "patient_hash": patient_hash,
                "manufacturer": manufacturer,
                "linkage_status": status,
                "scanner_sensitivity_eligible": status == "EXACT",
            }
        )
    return rows, {
        "branch": "tau",
        "endpoint_membership": len(eligible_patient_hashes),
        "linked_rows": len(rows),
        "manufacturer_counts": dict(Counter(row["manufacturer"] for row in rows)),
        **dict(counts),
    }


def _read_endpoint_hashes(output_root: Path, branch: str) -> set[str]:
    contract_path = output_root / "endpoints" / f"{branch}_endpoint_contract.json"
    if not contract_path.is_file():
        raise ValueError(f"LOCKED_{branch.upper()}_ENDPOINT_REQUIRED")
    contract = read_json(contract_path)
    endpoint_path = output_root / "endpoints" / str(contract.get("endpoint_path", ""))
    if not endpoint_path.is_file() or sha256_file(endpoint_path) != contract.get("endpoint_sha256"):
        raise ValueError(f"{branch.upper()}_ENDPOINT_HASH_MISMATCH")
    if contract.get("storage_format") == "parquet":
        frame = pd.read_parquet(endpoint_path)
    elif contract.get("storage_format") == "csv_fallback":
        frame = pd.read_csv(endpoint_path)
    else:
        raise ValueError("UNSUPPORTED_ENDPOINT_STORAGE_FORMAT")
    return set(frame["patient_hash"].astype(str))


def _atomic_write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp.csv", dir=str(path.parent)
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        pd.DataFrame(rows).to_csv(temporary, index=False, encoding="utf-8", lineterminator="\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def build_scanner_audit_from_sources(
    *, config: dict[str, Any], output_root: str | Path, project_root: str | Path
) -> dict[str, Any]:
    root = Path(output_root)
    project = Path(project_root)
    raw_root = Path(config["paths"]["raw_root"])
    amyloid_source = raw_root / config["clinical_sources"]["amyloid_suvr"]
    private_manifest = project / config["paths"].get(
        "amyloid_private_manifest",
        "data/preprocessed_pet/_private/manifests/fbp_private_manifest.csv",
    )
    pet_input = project / config["paths"]["amyloid_input_manifest"]
    tau_manifest = project / config["paths"]["tau_image_manifest"]
    required = [amyloid_source, private_manifest, pet_input, tau_manifest]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise ValueError(f"MISSING_SCANNER_AUDIT_SOURCES:{missing}")

    amyloid_hashes = _read_endpoint_hashes(root, "amyloid")
    tau_hashes = _read_endpoint_hashes(root, "tau")
    first_amyloid, source_audit = load_first_amyloid_scans(
        pd.read_csv(amyloid_source, low_memory=False).to_dict(orient="records")
    )
    amyloid_rows, amyloid_audit = link_amyloid_scanner_rows(
        first_amyloid,
        pd.read_csv(private_manifest, low_memory=False).to_dict(orient="records"),
        pd.read_csv(pet_input, low_memory=False).to_dict(orient="records"),
        amyloid_hashes,
    )
    tau_rows, tau_audit = link_tau_scanner_rows(
        tau_hashes,
        pd.read_csv(tau_manifest, low_memory=False).to_dict(orient="records"),
    )
    combined = [*amyloid_rows, *tau_rows]
    if any(set(row) != {
        "branch",
        "patient_hash",
        "manufacturer",
        "linkage_status",
        "scanner_sensitivity_eligible",
    } for row in combined):
        raise ValueError("SCANNER_LINKAGE_PRIVATE_SCHEMA_VIOLATION")
    _atomic_write_csv(root / "qc" / "scanner_linkage_private.csv", combined)
    audit = {
        "status": "COMPLETED",
        "amyloid": amyloid_audit,
        "tau": tau_audit,
        "amyloid_source_audit": source_audit,
        "unknown_scanner_imputed": False,
        "raw_identifiers_exported": False,
        "source_hashes": {
            "amyloid_source": sha256_file(amyloid_source),
            "amyloid_private_manifest": sha256_file(private_manifest),
            "amyloid_input_manifest": sha256_file(pet_input),
            "tau_image_manifest": sha256_file(tau_manifest),
        },
    }
    atomic_write_json(root / "qc" / "scanner_metadata_audit.json", audit)
    return audit
