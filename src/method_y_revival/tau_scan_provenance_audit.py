"""Fail-closed, aggregate-only audit of Tau scan provenance.

The filename BID/visit convention is a subject/visit association.  It is not
an acquisition identity, so this audit never upgrades it to an exact scan
match.  Row-level evidence can be written only to an explicitly private root;
the committed JSON and Markdown outputs contain aggregate, path-free facts.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Mapping

import nibabel as nib
import pandas as pd

from .atomic_io import atomic_write_bytes, atomic_write_json, atomic_write_parquet


FTP_PATTERN = re.compile(r"A4_PET_FTP_(B\d+)_(\d+)", re.IGNORECASE)
IDENTITY_ALIASES = {
    "seriesinstanceuid": "SeriesInstanceUID",
    "studyinstanceuid": "StudyInstanceUID",
    "accessionnumber": "AccessionNumber",
    "accession": "AccessionNumber",
}
IDENTITY_TEXT_PATTERNS = {
    r"series[_.\-\s]*instance[_.\-\s]*uid": "SeriesInstanceUID",
    r"study[_.\-\s]*instance[_.\-\s]*uid": "StudyInstanceUID",
    r"accession[_.\-\s]*number": "AccessionNumber",
    r"accession": "AccessionNumber",
}
SOURCE_IDENTITY_FIELDS = set(IDENTITY_ALIASES)
PLACEHOLDER_STRINGS = {"", "null", "none", "false", "n/a", "na", "unknown"}
HEADER_TEXT_FIELDS = ("descrip", "aux_file", "db_name", "intent_name")
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_within(path: Path, root: Path) -> bool:
    """Return whether ``path`` is ``root`` or a descendant after resolution."""

    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _has_git_ancestor(path: Path) -> bool:
    """Return whether ``path`` is inside a Git repository or worktree.

    A worktree can use either a ``.git`` directory or a ``.git`` file that
    points back to the common Git directory. This check intentionally does not
    trust caller-supplied roots: private row evidence must never be materialized
    in any checkout, including a nested or temporary worktree.
    """

    resolved = path.resolve()
    for candidate in (resolved, *resolved.parents):
        git_marker = candidate / ".git"
        if git_marker.is_dir() or git_marker.is_file():
            return True
    return False


def _assert_private_rows_outside_repositories(row_path: Path, private_root: Path) -> None:
    """Fail closed before reading inputs or writing private row evidence."""

    if (
        _is_within(row_path, REPOSITORY_ROOT)
        or _is_within(private_root, REPOSITORY_ROOT)
        or _has_git_ancestor(row_path)
        or _has_git_ancestor(private_root)
    ):
        raise ValueError("PRIVATE_ROWS_OUTPUT_INSIDE_REPOSITORY")


def _load_quantitative_metrics(
    quantitative_validation_path: Path | str,
    crosswalk_audit_path: Path | str,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, str]]]:
    """Load claim-facing metrics only from canonical aggregate artifacts.

    This intentionally projects a small, aggregate-only subset.  In
    particular, it does not propagate any source paths or row-level keys that
    may be present in the canonical artifacts.
    """

    quantitative_file = Path(quantitative_validation_path)
    crosswalk_file = Path(crosswalk_audit_path)
    if (
        quantitative_file.name != "scan_level_tau_quantitative_validation.json"
        or crosswalk_file.name != "tau_source_region_crosswalk_audit_v2.json"
    ):
        raise ValueError("TAU_QUANTITATIVE_ARTIFACT_NAME_INVALID")
    try:
        quantitative = json.loads(quantitative_file.read_text(encoding="utf-8-sig"))
        crosswalk = json.loads(crosswalk_file.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("TAU_QUANTITATIVE_ARTIFACT_UNREADABLE") from exc
    if not isinstance(quantitative, Mapping) or not isinstance(crosswalk, Mapping):
        raise ValueError("TAU_QUANTITATIVE_ARTIFACT_SCHEMA_INVALID")
    if (
        quantitative.get("science_gate_status") != "BLOCKED"
        or quantitative.get("claim_status") != "NO_CLAIM"
        or crosswalk.get("science_gate_status") != "BLOCKED"
        or crosswalk.get("claim_status") != "NO_CLAIM"
        or crosswalk.get("reference_equivalence_proven") is not False
    ):
        raise ValueError("TAU_QUANTITATIVE_ARTIFACT_GATE_NOT_BLOCKED")
    raw_metrics = quantitative.get("metrics")
    if not isinstance(raw_metrics, list) or not raw_metrics:
        raise ValueError("TAU_QUANTITATIVE_ARTIFACT_METRICS_MISSING")
    metrics: list[dict[str, Any]] = []
    for item in raw_metrics:
        if not isinstance(item, Mapping):
            raise ValueError("TAU_QUANTITATIVE_ARTIFACT_METRIC_SCHEMA_INVALID")
        required = (
            "mask_name",
            "n_pairs",
            "pearson_r",
            "spearman_rho",
            "bias_image_minus_source",
            "mae",
            "bland_altman_mean_bias",
            "bland_altman_sd",
            "bland_altman_lower_95",
            "bland_altman_upper_95",
            "status",
        )
        if any(key not in item for key in required):
            raise ValueError("TAU_QUANTITATIVE_ARTIFACT_METRIC_SCHEMA_INVALID")
        if item["status"] != "COMPUTED":
            raise ValueError("TAU_QUANTITATIVE_ARTIFACT_METRIC_NOT_COMPUTED")
        metrics.append({key: item[key] for key in required})
    sources = {
        "scan_level_tau_quantitative_validation": {
            "artifact_name": quantitative_file.name,
            "sha256": sha256_file(quantitative_file),
        },
        "tau_source_region_crosswalk_audit": {
            "artifact_name": crosswalk_file.name,
            "sha256": sha256_file(crosswalk_file),
        },
    }
    return metrics, sources


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


def _normalized_key(value: Any) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def _valid_identity_value(value: Any) -> str | None:
    # JSON booleans/null and numeric sentinels are not DICOM identity values.
    if not isinstance(value, str):
        return None
    text = value.strip().rstrip("\x00").strip()
    if text.lower() in PLACEHOLDER_STRINGS:
        return None
    return text


def identity_fields_from_sidecar(payload: Mapping[str, Any]) -> dict[str, str]:
    """Return normalized, non-placeholder identity fields from nested JSON."""

    found: dict[str, str] = {}

    def walk(value: Any) -> None:
        if isinstance(value, Mapping):
            for key, child in value.items():
                canonical = IDENTITY_ALIASES.get(_normalized_key(key))
                if canonical:
                    text = _valid_identity_value(child)
                    if text is not None:
                        found.setdefault(canonical, text)
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)

    walk(payload)
    return found


def _identity_fields_from_text(value: str) -> dict[str, str]:
    """Extract identity aliases from header text or extension payload."""

    found: dict[str, str] = {}
    for alias, canonical in IDENTITY_TEXT_PATTERNS.items():
        # Permit separators in aliases (e.g. Study_Instance_UID) and either
        # JSON-ish or DICOM-ish key/value separators.
        match = re.search(
            rf"(?:^|[\"'{{,;\s]){alias}[\"']?\s*[:=]\s*[\"']?([^\"'\s,;}}]+)",
            value,
            re.IGNORECASE,
        )
        if match:
            text = _valid_identity_value(match.group(1))
            if text is not None:
                found.setdefault(canonical, text)
    return found


def _source_identity_columns(columns: list[str]) -> list[str]:
    result: list[str] = []
    for column in columns:
        if _normalized_key(column) in SOURCE_IDENTITY_FIELDS:
            result.append(str(column))
    return sorted(set(result))


def _source_key_counts(path: Path, *, require_flortaucipir: bool) -> tuple[pd.Series, dict[str, Any]]:
    frame = pd.read_csv(path, low_memory=False)
    required = {"BID", "VISCODE"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"TAU_SOURCE_IDENTITY_COLUMNS_MISSING:{path.name}:{','.join(missing)}")
    raw_row_count = len(frame)
    if require_flortaucipir:
        if "ligand" not in frame.columns or "scan_analyzed" not in frame.columns:
            raise ValueError(f"TAU_SOURCE_TRACER_COLUMNS_MISSING:{path.name}")
        valid = (
            frame["ligand"].astype(str).str.strip().str.lower().eq("flortaucipir")
            & frame["scan_analyzed"].astype(str).str.strip().str.lower().isin({"yes", "true", "1"})
        )
        frame = frame.loc[valid].copy()
        if frame.empty:
            raise ValueError(f"TAU_SOURCE_TRACER_NO_VALID_FLORTAUCIPIR_ROWS:{path.name}")
    else:
        frame = frame.copy()
    frame["_BID"] = frame["BID"].astype(str).str.strip()
    frame["_VISCODE"] = frame["VISCODE"].map(canonical_visit)
    counts = frame.groupby(["_BID", "_VISCODE"]).size()
    source_columns = [str(column) for column in frame.columns if not str(column).startswith("_")]
    meta = {
        "source_name": path.name,
        "sha256": sha256_file(path),
        "row_count": int(raw_row_count),
        "valid_tracer_row_count": int(len(frame)),
        "columns": source_columns,
        "identity_columns": _source_identity_columns(source_columns),
        "identity_columns_present": bool(_source_identity_columns(source_columns)),
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
    """Inspect all text identity-bearing header fields and extensions."""

    if not path.is_file():
        return "MISSING", {}
    try:
        image = nib.load(str(path), mmap=False)
        fields: dict[str, str] = {}
        header = image.header
        for name in HEADER_TEXT_FIELDS:
            try:
                value = header[name]
            except Exception:
                continue
            if hasattr(value, "tobytes"):
                value = value.tobytes()
            if isinstance(value, bytes):
                value = value.decode("utf-8", errors="ignore")
            if isinstance(value, str):
                fields.update({k: v for k, v in _identity_fields_from_text(value).items() if k not in fields})
        for extension in header.extensions:
            try:
                content = extension.get_content()
            except Exception:
                content = getattr(extension, "content", None)
            if content is None:
                continue
            if isinstance(content, bytes):
                content = content.decode("utf-8", errors="ignore")
            if isinstance(content, str):
                parsed: dict[str, str] = {}
                try:
                    decoded = json.loads(content)
                    if isinstance(decoded, Mapping):
                        parsed = identity_fields_from_sidecar(decoded)
                except json.JSONDecodeError:
                    parsed = _identity_fields_from_text(content)
                for key, value in parsed.items():
                    fields.setdefault(key, value)
        return ("EXACT_ID_FIELDS_PRESENT" if fields else "NO_EXACT_ID_FIELDS"), fields
    except Exception:
        return "PARSE_ERROR", {}


def _render_report(report: Mapping[str, Any]) -> str:
    sidecar = report["sidecar_status_counts"]
    header = report["nifti_header_status_counts"]
    stanford = report["stanford_bridge_status_counts"]
    a4 = report["a4_linear_suvr_bridge_status_counts"]
    metric_lines = [
        "- Canonical aggregate quantitative metrics (not claim-bearing):"
    ]
    for metric in report["quantitative_metrics"]:
        metric_lines.append(
            "  - "
            f"{metric['mask_name']}: pairs `{metric['n_pairs']}`, Pearson `{metric['pearson_r']:.4f}`, "
            f"Spearman `{metric['spearman_rho']:.4f}`, bias `{metric['bias_image_minus_source']:.5f}`, "
            f"MAE `{metric['mae']:.4f}`, Bland–Altman mean bias "
            f"`{metric['bland_altman_mean_bias']:.5f}`, SD `{metric['bland_altman_sd']:.5f}`, "
            f"95% limits [`{metric['bland_altman_lower_95']:.5f}`, "
            f"`{metric['bland_altman_upper_95']:.5f}`]."
        )
    source_hashes = ", ".join(
        f"`{item['artifact_name']}` SHA-256 `{item['sha256']}`"
        for item in report["quantitative_metric_sources"].values()
    )
    return "\n".join(
        [
            "# Tau quantitative validation v2",
            "",
            "- Official masks: CenTauR Mesial, Meta, TP and Frontal; reference: `voi_CerebGry_tau_2mm`.",
            "- Comparisons computed: Pearson, Spearman, bias, MAE and Bland–Altman limits.",
            *metric_lines,
            f"- Metric sources: {source_hashes}.",
            "- Stanford uses the non-PVC gray-matter cerebellum contract (`bi_Cerebellum.Cortex`), while the official CenTauR mask uses an inferior-cerebellar reference; equivalence is not proven.",
            "<!-- scan-provenance audit: begin -->",
            "## Scan-provenance audit (refresh-generated)",
            "",
            f"- Scan count: `{report['scan_count']}`.",
            f"- Locked input manifest: `{report['input_manifest']['artifact_name']}`; rows "
            f"`{report['input_manifest']['row_count']}`; SHA-256 "
            f"`{report['input_manifest']['sha256']}`.",
            f"- Sidecar identity status counts: `{json.dumps(sidecar, sort_keys=True)}`.",
            f"- NIfTI header/extension identity status counts: `{json.dumps(header, sort_keys=True)}`.",
            f"- Aggregate exact identity scans (sidecar or header): `{report['exact_scan_identity_scan_count']}`; fields: `{', '.join(report['exact_scan_identity_fields']) or 'none'}`.",
            f"- A4 linear-SUVR BID+VISCODE subject/visit associations: `{report['a4_linear_suvr_subject_visit_association_scan_count']}`; statuses: `{json.dumps(a4, sort_keys=True)}`.",
            f"- Stanford BID+VISCODE subject/visit associations (including numeric visit convention): `{json.dumps(stanford, sort_keys=True)}`.",
            "- BID+VISCODE associations are not exact acquisition identity; no UID/accession bridge or reference-pipeline equivalence is inferred.",
            f"- Science gate: **{report['science_gate_status']}**; claim status: **{report['claim_status']}**; claim-bearing work remains disallowed.",
            "<!-- scan-provenance audit: end -->",
            "",
            "The quantitative science gate remains blocked until exact scan identity, reference-pipeline equivalence, and a prespecified calibration path are proven.",
            "",
            "Canonical aggregate metric artifacts are identified above by fixed name and SHA-256; no row-level artifact path is published.",
            "",
            "Rows without exact source-visit linkage remain exploratory even when image metrics are finite; no transform rerun, training, HPO, outer evaluation, or image-CNN work is authorized by this audit.",
            "",
        ]
    )


def audit_tau_scan_provenance(
    linkage_path: Path | str,
    *,
    source_suvr_path: Path | str,
    source_stanford_path: Path | str,
    output_json: Path | str | None = None,
    output_report: Path | str | None = None,
    output_rows: Path | str | None = None,
    private_rows_root: Path | str | None = None,
    quantitative_validation_path: Path | str | None = None,
    crosswalk_audit_path: Path | str | None = None,
) -> dict[str, Any]:
    """Audit linked scans and optionally emit explicitly private row evidence."""

    if output_rows is not None:
        if private_rows_root is None:
            raise ValueError("PRIVATE_ROWS_ROOT_REQUIRED")
        private_root = Path(private_rows_root).resolve()
        row_path = Path(output_rows).resolve()
        if private_root not in row_path.parents and row_path != private_root:
            raise ValueError("PRIVATE_ROWS_OUTPUT_OUTSIDE_PRIVATE_ROOT")
        _assert_private_rows_outside_repositories(row_path, private_root)
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
        bid, visit = parsed if parsed is not None else ("", "")
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
        stanford_visit2_rows = int(stanford_counts.get((bid, "2"), 0)) if bid and visit in {"4", "004"} else 0
        direct_bridge_status = (
            "SUBJECT_VISIT_ASSOCIATION_A4_LINEAR_SUVR_SOURCE"
            if suvr_rows
            else "NO_A4_LINEAR_SUVR_SOURCE_KEY"
        )
        if stanford_same_visit_rows:
            stanford_bridge_status = "SUBJECT_VISIT_ASSOCIATION_STANFORD_SOURCE"
        elif stanford_visit2_rows:
            stanford_bridge_status = "NUMERICAL_VISIT_ASSOCIATION_TO_STANFORD_VISCODE_2"
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
                "stanford_visit2_bridge_row_count": stanford_visit2_rows,
            }
        )
    frame = pd.DataFrame(rows)
    if output_rows is not None:
        atomic_write_parquet(Path(output_rows), frame)

    sidecar_counts = frame["sidecar_status"].value_counts(dropna=False).to_dict()
    header_counts = frame["nifti_header_status"].value_counts(dropna=False).to_dict()
    direct_counts = frame["direct_bridge_status"].value_counts(dropna=False).to_dict()
    stanford_counts_summary = frame["stanford_bridge_status"].value_counts(dropna=False).to_dict()
    sidecar_identity = frame["sidecar_status"].eq("EXACT_ID_FIELDS_PRESENT")
    header_identity = frame["nifti_header_status"].eq("EXACT_ID_FIELDS_PRESENT")
    exact_identity = sidecar_identity | header_identity
    blockers = [
        "STANFORD_SOURCE_SCAN_LEVEL_BRIDGE_NOT_EXACT_UID_VERIFIED",
        "REFERENCE_PIPELINE_EQUIVALENCE_NOT_PROVEN",
        "QUANTITATIVE_CALIBRATION_PATH_NOT_FOUND",
    ]
    if not exact_identity.any():
        blockers.insert(0, "EXACT_DICOM_SCAN_IDENTITY_UNAVAILABLE")
    report: dict[str, Any] = {
        "schema_version": "2.0",
        "status": "COMPLETED_WITH_SCIENCE_GATE_BLOCKED",
        "science_gate_status": "BLOCKED",
        "claim_status": "NO_CLAIM",
        "claim_bearing_allowed": False,
        "scope": "FTP_SIDECAR_AND_NIFTI_HEADER_IDENTITY_AUDIT_NO_TRANSFORM_RERUN",
        "scan_count": int(len(frame)),
        "input_manifest": {
            "artifact_name": linkage_file.name,
            "row_count": int(len(frame)),
            "sha256": sha256_file(linkage_file),
        },
        "sidecar_status_counts": {str(k): int(v) for k, v in sidecar_counts.items()},
        "sidecar_identity_scan_count": int(sidecar_identity.sum()),
        "nifti_header_status_counts": {str(k): int(v) for k, v in header_counts.items()},
        "nifti_header_identity_scan_count": int(header_identity.sum()),
        "exact_scan_identity_scan_count": int(exact_identity.sum()),
        "exact_scan_identity_fields": sorted(
            {field for fields in frame.loc[exact_identity, "sidecar_identity_fields"] for field in fields}
            | {field for fields in frame.loc[exact_identity, "nifti_header_identity_fields"] for field in fields}
        ),
        "association_semantics": {
            "a4_linear_suvr": "SUBJECT_VISIT_ASSOCIATION_ONLY_NOT_ACQUISITION_IDENTITY",
            "stanford": "SUBJECT_VISIT_ASSOCIATION_ONLY_NOT_ACQUISITION_IDENTITY",
        },
        "a4_linear_suvr_subject_visit_association_scan_count": int(
            (frame["a4_suvr_source_row_count"] > 0).sum()
        ),
        "a4_linear_suvr_no_source_key_scan_count": int((frame["a4_suvr_source_row_count"] == 0).sum()),
        "a4_linear_suvr_bridge_status_counts": {str(k): int(v) for k, v in direct_counts.items()},
        "stanford_bridge_status_counts": {str(k): int(v) for k, v in stanford_counts_summary.items()},
        "source_tables": {"a4_linear_suvr": suvr_meta, "stanford_freesurfer_non_pvc": stanford_meta},
        "reference_contracts": {
            "local_centaur": "voi_CerebGry_tau_2mm",
            "stanford": "bi_Cerebellum.Cortex / gray_matter_cerebellar_cortex",
            "a4_linear_suvr": "blcere_all / whole_cerebellum",
            "equivalence_status": "NOT_PROVEN",
            "calibration_path_status": "NOT_FOUND",
        },
        "blockers": blockers,
    }
    # Refresh the human-facing report first and publish the machine-readable
    # gate last.  Each replacement is atomic, and the gate is the commit point,
    # so it cannot advertise a refresh whose official report was not written.
    if output_report is not None:
        if quantitative_validation_path is None or crosswalk_audit_path is None:
            raise ValueError("TAU_QUANTITATIVE_ARTIFACTS_REQUIRED_FOR_REPORT")
        metrics, metric_sources = _load_quantitative_metrics(
            quantitative_validation_path, crosswalk_audit_path
        )
        report["quantitative_metrics"] = metrics
        report["quantitative_metric_sources"] = metric_sources
        atomic_write_bytes(Path(output_report), _render_report(report).encode("utf-8"))
    if output_json is not None:
        atomic_write_json(Path(output_json), report)
    return report
