"""Fail-closed audit of tau scan identity and source-table bridges.

The FTP filenames provide a BID and visit code, but the de-identified raw
sidecars in the local A4 tree do not retain DICOM UID/accession fields.  This
module inventories that fact and distinguishes an exact BID/visit bridge to
the A4 linear-SUVR table from the separate Stanford FreeSurfer table used by
the CenTauR comparison.  It never upgrades either bridge to reference or
quantitative equivalence.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Mapping

import pandas as pd
import nibabel as nib

from .atomic_io import atomic_write_json


FTP_PATTERN = re.compile(r"A4_PET_FTP_(B\d+)_(\d+)", re.IGNORECASE)
IDENTITY_ALIASES = {
    "seriesinstanceuid": "SeriesInstanceUID",
    "studyinstanceuid": "StudyInstanceUID",
    "accessionnumber": "AccessionNumber",
    "accession": "AccessionNumber",
}
SOURCE_IDENTITY_FIELDS = {
    "seriesinstanceuid",
    "studyinstanceuid",
    "accessionnumber",
    "accession",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_visit(value: Any) -> str:
    text = str(value).strip()
    if text.isdigit():
        return str(int(text))
    return text


def parse_ftp_identity(path: str | Path) -> tuple[str, str] | None:
    match = FTP_PATTERN.search(Path(path).name)
    if not match:
        return None
    return match.group(1), canonical_visit(match.group(2))


def _flatten_keys(value: Any) -> list[str]:
    if isinstance(value, Mapping):
        keys: list[str] = []
        for key, child in value.items():
            keys.append(str(key))
            keys.extend(_flatten_keys(child))
        return keys
    if isinstance(value, list):
        keys: list[str] = []
        for child in value:
            keys.extend(_flatten_keys(child))
        return keys
    return []


def identity_fields_from_sidecar(payload: Mapping[str, Any]) -> dict[str, str]:
    """Return non-empty exact identity fields from a sidecar payload."""

    found: dict[str, str] = {}

    def walk(value: Any) -> None:
        if isinstance(value, Mapping):
            for key, child in value.items():
                normalized = re.sub(r"[^a-z0-9]", "", str(key).lower())
                canonical = IDENTITY_ALIASES.get(normalized)
                if canonical and not isinstance(child, (Mapping, list)):
                    text = str(child).strip()
                    if text:
                        found.setdefault(canonical, text)
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)

    walk(payload)
    return found


def _source_identity_columns(columns: list[str]) -> list[str]:
    result: list[str] = []
    for column in columns:
        normalized = re.sub(r"[^a-z0-9]", "", str(column).lower())
        if normalized in SOURCE_IDENTITY_FIELDS:
            result.append(str(column))
    return sorted(set(result))


def _source_key_counts(path: Path, *, require_flortaucipir: bool) -> tuple[pd.Series, dict[str, Any]]:
    frame = pd.read_csv(path, low_memory=False)
    required = {"BID", "VISCODE"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"TAU_SOURCE_IDENTITY_COLUMNS_MISSING:{path.name}:{','.join(missing)}")
    if require_flortaucipir:
        if "ligand" not in frame.columns or "scan_analyzed" not in frame.columns:
            raise ValueError(f"TAU_SOURCE_TRACER_COLUMNS_MISSING:{path.name}")
        frame = frame[
            frame["ligand"].astype(str).str.strip().str.lower().eq("flortaucipir")
            & frame["scan_analyzed"].astype(str).str.strip().str.lower().eq("yes")
        ]
    frame = frame.copy()
    frame["_BID"] = frame["BID"].astype(str).str.strip()
    frame["_VISCODE"] = frame["VISCODE"].map(canonical_visit)
    counts = frame.groupby(["_BID", "_VISCODE"]).size()
    meta = {
        "path": str(path),
        "sha256": sha256_file(path),
        "row_count": int(len(frame)),
        "columns": [str(column) for column in frame.columns if not str(column).startswith("_")],
        "identity_columns": _source_identity_columns([str(column) for column in frame.columns]),
        "identity_columns_present": bool(_source_identity_columns([str(column) for column in frame.columns])),
        "key_count": int(len(counts)),
    }
    return counts, meta


def _read_sidecar(path: Path) -> tuple[str, dict[str, str], str]:
    if not path.is_file():
        return "MISSING", {}, "SIDECAR_MISSING"
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        return "PARSE_ERROR", {}, "SIDECAR_PARSE_ERROR"
    fields = identity_fields_from_sidecar(payload)
    if not fields:
        return "NO_EXACT_ID_FIELDS", {}, "EXACT_DICOM_IDENTITY_UNAVAILABLE"
    return "EXACT_ID_FIELDS_PRESENT", fields, "EXACT_DICOM_IDENTITY_PRESENT"


def _read_nifti_header(path: Path) -> tuple[str, dict[str, str]]:
    """Inspect NIfTI header/extensions without loading voxel data."""

    if not path.is_file():
        return "MISSING", {}
    try:
        image = nib.load(str(path), mmap=False)
        fields: dict[str, str] = {}
        for extension in image.header.extensions:
            try:
                content = extension.get_content()
            except Exception:
                continue
            if isinstance(content, bytes):
                content = content.decode("utf-8", errors="ignore")
            if isinstance(content, str):
                for label in IDENTITY_ALIASES.values():
                    match = re.search(rf"{re.escape(label)}\s*[:=]\s*[\"']?([^\"',}}\s]+)", content, re.I)
                    if match:
                        fields.setdefault(label, match.group(1))
        return ("EXACT_ID_FIELDS_PRESENT" if fields else "NO_EXACT_ID_FIELDS"), fields
    except Exception:
        return "PARSE_ERROR", {}


def audit_tau_scan_provenance(
    linkage_path: Path | str,
    *,
    source_suvr_path: Path | str,
    source_stanford_path: Path | str,
    output_json: Path | str | None = None,
    output_rows: Path | str | None = None,
) -> dict[str, Any]:
    """Audit all linked FTP scans without reading or rewriting transform images."""

    linkage_file = Path(linkage_path)
    suvr_file = Path(source_suvr_path)
    stanford_file = Path(source_stanford_path)
    linkage = pd.read_parquet(linkage_file)
    suvr_counts, suvr_meta = _source_key_counts(suvr_file, require_flortaucipir=True)
    stanford_counts, stanford_meta = _source_key_counts(stanford_file, require_flortaucipir=False)

    rows: list[dict[str, Any]] = []
    for record in linkage.to_dict(orient="records"):
        ftp_path = Path(str(record.get("ftp_path", "")))
        parsed = parse_ftp_identity(ftp_path)
        if parsed is None:
            bid, visit = "", ""
        else:
            bid, visit = parsed
        sidecar_path = (
            ftp_path.with_suffix("").with_suffix(".json")
            if ftp_path.name.endswith(".nii.gz")
            else ftp_path.with_suffix(".json")
        )
        sidecar_status, sidecar_fields, identity_evidence = _read_sidecar(sidecar_path)
        nifti_status, nifti_fields = _read_nifti_header(ftp_path)
        key = (bid, visit)
        suvr_rows = int(suvr_counts.get(key, 0)) if bid else 0
        stanford_same_visit_rows = int(stanford_counts.get(key, 0)) if bid else 0
        # The official validator's 4/004 -> Stanford VISCODE 2 mapping is a
        # numerical visit convention, not an exact acquisition identity.
        stanford_bridge_rows = int(stanford_counts.get((bid, "2"), 0)) if bid and visit in {"4", "004"} else 0
        if suvr_rows:
            direct_bridge_status = "EXACT_BID_VISCODE_A4_LINEAR_SUVR_SOURCE"
        else:
            direct_bridge_status = "NO_A4_LINEAR_SUVR_SOURCE_KEY"
        if stanford_same_visit_rows:
            stanford_bridge_status = "EXACT_BID_VISCODE_STANFORD_SOURCE"
        elif stanford_bridge_rows:
            stanford_bridge_status = "NUMERICAL_VISIT_BRIDGE_TO_STANFORD_VISCODE_2"
        else:
            stanford_bridge_status = "NO_STANFORD_SOURCE_KEY"
        rows.append(
            {
                "scan_key": str(record.get("scan_key", "")),
                "subject_key": str(record.get("subject_key", "")),
                "ftp_path": str(ftp_path),
                "bid": bid,
                "ftp_visit_code": visit,
                "sidecar_path": str(sidecar_path),
                "sidecar_status": sidecar_status,
                "identity_evidence": identity_evidence,
                "sidecar_identity_fields": sorted(sidecar_fields),
                "nifti_header_status": nifti_status,
                "nifti_header_identity_fields": sorted(nifti_fields),
                "direct_bridge_status": direct_bridge_status,
                "a4_suvr_source_row_count": suvr_rows,
                "stanford_bridge_status": stanford_bridge_status,
                "stanford_same_visit_row_count": stanford_same_visit_rows,
                "stanford_visit2_bridge_row_count": stanford_bridge_rows,
            }
        )
    frame = pd.DataFrame(rows)
    if output_rows is not None:
        output_rows_path = Path(output_rows)
        output_rows_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = output_rows_path.with_suffix(output_rows_path.suffix + ".tmp")
        frame.to_parquet(temporary, index=False)
        temporary.replace(output_rows_path)

    sidecar_counts = frame["sidecar_status"].value_counts(dropna=False).to_dict()
    direct_counts = frame["direct_bridge_status"].value_counts(dropna=False).to_dict()
    stanford_counts_summary = frame["stanford_bridge_status"].value_counts(dropna=False).to_dict()
    blockers = [
        "EXACT_DICOM_SCAN_IDENTITY_UNAVAILABLE",
        "STANFORD_SOURCE_SCAN_LEVEL_BRIDGE_NOT_EXACT_UID_VERIFIED",
        "REFERENCE_PIPELINE_EQUIVALENCE_NOT_PROVEN",
        "QUANTITATIVE_CALIBRATION_PATH_NOT_FOUND",
    ]
    report: dict[str, Any] = {
        "schema_version": "1.0",
        "status": "COMPLETED_WITH_SCIENCE_GATE_BLOCKED",
        "science_gate_status": "BLOCKED",
        "claim_status": "NO_CLAIM",
        "claim_bearing_allowed": False,
        "scope": "FTP_SIDEcar_AND_SOURCE_TABLE_IDENTITY_AUDIT_NO_TRANSFORM_RERUN",
        "linkage_path": str(linkage_file),
        "linkage_sha256": sha256_file(linkage_file),
        "scan_count": int(len(frame)),
        "sidecar_status_counts": {str(key): int(value) for key, value in sidecar_counts.items()},
        "exact_dicom_identity_scan_count": int((frame["sidecar_status"] == "EXACT_ID_FIELDS_PRESENT").sum()),
        "exact_dicom_identity_fields": sorted({field for fields in frame["sidecar_identity_fields"] for field in fields}),
        "nifti_header_status_counts": {str(key): int(value) for key, value in frame["nifti_header_status"].value_counts(dropna=False).to_dict().items()},
        "nifti_header_identity_scan_count": int((frame["nifti_header_status"] == "EXACT_ID_FIELDS_PRESENT").sum()),
        "nifti_header_identity_fields": sorted({field for fields in frame["nifti_header_identity_fields"] for field in fields}),
        "a4_linear_suvr_direct_bridge_scan_count": int((frame["a4_suvr_source_row_count"] > 0).sum()),
        "a4_linear_suvr_no_source_key_scan_count": int((frame["a4_suvr_source_row_count"] == 0).sum()),
        "a4_linear_suvr_bridge_status_counts": {str(key): int(value) for key, value in direct_counts.items()},
        "stanford_bridge_status_counts": {str(key): int(value) for key, value in stanford_counts_summary.items()},
        "source_tables": {
            "a4_linear_suvr": suvr_meta,
            "stanford_freesurfer_non_pvc": stanford_meta,
        },
        "reference_contracts": {
            "local_centaur": "voi_CerebGry_tau_2mm",
            "stanford": "bi_Cerebellum.Cortex / gray_matter_cerebellar_cortex",
            "a4_linear_suvr": "blcere_all / whole_cerebellum",
            "equivalence_status": "NOT_PROVEN",
            "calibration_path_status": "NOT_FOUND",
        },
        "blockers": blockers,
        "rows_path": str(output_rows) if output_rows is not None else None,
    }
    if output_rows is not None:
        report["rows_sha256"] = sha256_file(Path(output_rows))
    if output_json is not None:
        atomic_write_json(Path(output_json), report)
    return report
