#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

try:
    from ._stage_common import project_path, write_json
except ImportError:  # pragma: no cover
    from _stage_common import project_path, write_json


REQUIRED_COLUMNS = [
    "subject_id_hash",
    "pet_path",
    "tracer",
    "suvr_global",
    "cohort",
    "paired_mri_subject_id_hash",
    "scanner",
    "frame_timing_min_start",
    "frame_timing_min_end",
]
DEFAULT_FRAME_TIMING = {"FBP": (50.0, 70.0), "FBB": (90.0, 110.0), "PIB": (50.0, 70.0)}


def _load_mri_hashes(path: Path) -> set[str]:
    import h5py

    with h5py.File(path, "r") as h5:
        if "sample_hash" not in h5 or "status" not in h5:
            raise RuntimeError("BLOCKED_MRI_CACHE_SCHEMA_INVALID")
        hashes = [v.decode("utf-8", "replace") if isinstance(v, bytes) else str(v) for v in h5["sample_hash"][:]]
        statuses = [v.decode("utf-8", "replace") if isinstance(v, bytes) else str(v) for v in h5["status"][:]]
    return {sample_hash for sample_hash, status in zip(hashes, statuses, strict=True) if str(status).upper() == "OK"}


def _json_get(payload: dict, dotted: str, default=None):
    value = payload
    for part in dotted.split("."):
        if not isinstance(value, dict) or part not in value:
            return default
        value = value[part]
    return value


def _candidate_pet_path(root: Path, sample_hash: str, tracer: str) -> Path | None:
    search_roots = [
        root / tracer.lower() / "normalized",
        root / tracer.lower() / "aligned",
        root / tracer.upper() / "normalized",
        root / tracer.upper() / "aligned",
    ]
    for directory in search_roots:
        if not directory.exists():
            continue
        for suffix in (".nii.gz", ".nii"):
            candidate = directory / f"{sample_hash}{suffix}"
            if candidate.exists():
                return candidate
    return None


def _load_private_manifest_metadata(path: Path | None) -> dict[str, dict[str, str]]:
    if path is None or not path.exists():
        return {}
    out: dict[str, dict[str, str]] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sample_hash = (row.get("sample_hash") or "").strip()
            if not sample_hash:
                continue
            out.setdefault(
                sample_hash,
                {
                    "cohort": (row.get("substudy") or "").strip().upper(),
                    "scanner": (row.get("scanner") or "").strip(),
                    "frame": (row.get("frame") or "").strip(),
                    "patient_hash": (row.get("patient_hash") or "").strip(),
                    "raw_path_private": (row.get("raw_path_private") or row.get("raw_path") or "").strip(),
                },
            )
    return out


def _sidecar_json_for_raw_path(raw_path: str) -> Path | None:
    if not raw_path:
        return None
    p = Path(raw_path)
    if p.name.endswith(".nii.gz"):
        candidate = p.with_suffix("").with_suffix(".json")
    else:
        candidate = p.with_suffix(".json")
    return candidate if candidate.exists() else None


def _scanner_from_sidecar_payload(payload) -> str:
    if isinstance(payload, list):
        values = [_scanner_from_sidecar_payload(item) for item in payload]
        return next((value for value in values if value), "")
    if not isinstance(payload, dict):
        return ""
    manufacturer = str(payload.get("Manufacturer") or "").strip()
    model = str(payload.get("ManufacturersModelName") or payload.get("ManufacturerModelName") or "").strip()
    if manufacturer and model:
        return f"{manufacturer}:{model}"
    return manufacturer


def _scanner_from_sidecar_with_reason(raw_path: str) -> tuple[str, str]:
    sidecar = _sidecar_json_for_raw_path(raw_path)
    if sidecar is None:
        return "", "sidecar_missing"
    try:
        payload = json.loads(sidecar.read_text(encoding="utf-8"))
    except Exception:
        return "", "sidecar_parse_error"
    scanner = _scanner_from_sidecar_payload(payload)
    return scanner, ("sidecar_has_scanner" if scanner else "sidecar_without_manufacturer")


def _scanner_from_sidecar(raw_path: str) -> str:
    scanner, _reason = _scanner_from_sidecar_with_reason(raw_path)
    return scanner


def _bid_from_path_or_meta(raw_path: str, bid: str = "") -> str:
    if bid:
        return str(bid).strip()
    match = re.search(r"B\d+", Path(raw_path).name)
    return match.group(0) if match else ""


def _same_bid_scanner_candidates(raw_path: str, bid: str = "") -> list[str]:
    """Return scanner strings from same-BID sidecars, for reporting only.

    These are deliberately not used to fill scanner because a different visit/frame
    can use different acquisition metadata. The count is still useful for human
    metadata remediation.
    """
    raw = Path(raw_path)
    resolved_bid = _bid_from_path_or_meta(raw_path, bid)
    if not resolved_bid or not raw.parent.exists():
        return []
    exact = _sidecar_json_for_raw_path(raw_path)
    scanners: list[str] = []
    for sidecar in sorted(raw.parent.glob(f"*{resolved_bid}*.json")):
        if exact is not None and sidecar.resolve() == exact.resolve():
            continue
        try:
            payload = json.loads(sidecar.read_text(encoding="utf-8"))
        except Exception:
            continue
        scanner = _scanner_from_sidecar_payload(payload)
        if scanner and scanner not in scanners:
            scanners.append(scanner)
    return scanners


def _frame_from_manifest(value: str, tracer: str) -> tuple[float, float]:
    if value:
        parts = [p for p in value.replace("-", ",").replace(";", ",").split(",") if p.strip()]
        if len(parts) >= 2:
            try:
                return float(parts[0]), float(parts[1])
            except ValueError:
                pass
    return DEFAULT_FRAME_TIMING[tracer]


def build_pet_input_rows(
    *,
    qc_root: Path,
    preprocessed_root: Path,
    mri_cache: Path,
    private_manifest: Path | None = None,
    include_excluded: bool = False,
) -> tuple[list[dict[str, str]], dict]:
    mri_hashes = _load_mri_hashes(mri_cache)
    private_meta = _load_private_manifest_metadata(private_manifest)
    rows: list[dict[str, str]] = []
    counts = {
        "qc_files": 0,
        "status_pass": 0,
        "excluded_from_training": 0,
        "mri_unpaired": 0,
        "missing_pet_path": 0,
        "missing_suvr": 0,
        "rows_written": 0,
        "cohort_defaulted_other": 0,
        "scanner_defaulted_unknown": 0,
        "cohort_from_private_manifest": 0,
        "scanner_from_qc_payload": 0,
        "scanner_from_private_manifest": 0,
        "scanner_from_sidecar": 0,
        "scanner_source_unavailable": 0,
        "scanner_unknown_no_private_manifest": 0,
        "scanner_unknown_no_raw_path": 0,
        "scanner_unknown_sidecar_missing": 0,
        "scanner_unknown_sidecar_without_manufacturer": 0,
        "scanner_unknown_sidecar_parse_error": 0,
        "scanner_unknown_same_bid_sidecar_with_scanner_not_used": 0,
        "scanner_unknown_no_same_bid_sidecar_scanner": 0,
    }
    cohorts: dict[str, int] = {}
    tracers: dict[str, int] = {}
    same_bid_candidate_scanners: dict[str, int] = {}
    for qc_path in sorted(qc_root.rglob("*.json")):
        counts["qc_files"] += 1
        payload = json.loads(qc_path.read_text(encoding="utf-8"))
        if str(payload.get("status", "")).upper() != "PASS":
            continue
        counts["status_pass"] += 1
        tier = str(payload.get("tier", "")).lower()
        if tier == "excluded_from_training" and not include_excluded:
            counts["excluded_from_training"] += 1
            continue
        sample_hash = str(payload.get("sample_hash") or qc_path.stem)
        if sample_hash not in mri_hashes:
            counts["mri_unpaired"] += 1
            continue
        try:
            tracer_from_path = qc_path.relative_to(preprocessed_root).parts[0]
        except Exception:
            tracer_from_path = qc_path.parents[1].name
        tracer = str(payload.get("tracer") or tracer_from_path).upper()
        if tracer not in {"FBP", "FBB", "PIB"}:
            raise RuntimeError(f"BLOCKED_LOCAL_CALIBRATION_REQUIRED:{tracer}")
        pet_path = _candidate_pet_path(preprocessed_root, sample_hash, tracer)
        if pet_path is None:
            counts["missing_pet_path"] += 1
            continue
        suvr = _json_get(payload, "qc_after.image_composite_suvr")
        if suvr is None:
            suvr = _json_get(payload, "image_composite_suvr")
        if suvr is None:
            counts["missing_suvr"] += 1
            continue
        meta = private_meta.get(sample_hash, {})
        cohort = str(payload.get("cohort") or _json_get(payload, "catalog.cohort") or meta.get("cohort") or "OTHER").upper()
        if cohort not in {"A4", "LEARN", "OTHER"}:
            cohort = "OTHER"
        if cohort in {"A4", "LEARN"} and meta.get("cohort"):
            counts["cohort_from_private_manifest"] += 1
        if cohort == "OTHER" and not payload.get("cohort"):
            counts["cohort_defaulted_other"] += 1
        scanner = str(payload.get("scanner") or _json_get(payload, "catalog.scanner") or "").strip()
        if scanner:
            counts["scanner_from_qc_payload"] += 1
        elif meta.get("scanner"):
            scanner = str(meta.get("scanner")).strip()
            counts["scanner_from_private_manifest"] += 1
        else:
            raw_path_private = str(meta.get("raw_path_private") or "")
            if not meta:
                counts["scanner_unknown_no_private_manifest"] += 1
                scanner = "UNKNOWN"
            elif not raw_path_private:
                counts["scanner_unknown_no_raw_path"] += 1
                scanner = "UNKNOWN"
            else:
                sidecar_scanner, sidecar_reason = _scanner_from_sidecar_with_reason(raw_path_private)
                if sidecar_scanner:
                    scanner = sidecar_scanner
                    counts["scanner_from_sidecar"] += 1
                else:
                    scanner = "UNKNOWN"
                    if sidecar_reason == "sidecar_missing":
                        counts["scanner_unknown_sidecar_missing"] += 1
                    elif sidecar_reason == "sidecar_parse_error":
                        counts["scanner_unknown_sidecar_parse_error"] += 1
                    elif sidecar_reason == "sidecar_without_manufacturer":
                        counts["scanner_unknown_sidecar_without_manufacturer"] += 1
                        candidates = _same_bid_scanner_candidates(raw_path_private, meta.get("bid") or "")
                        if candidates:
                            counts["scanner_unknown_same_bid_sidecar_with_scanner_not_used"] += 1
                            for candidate in candidates:
                                same_bid_candidate_scanners[candidate] = same_bid_candidate_scanners.get(candidate, 0) + 1
                        else:
                            counts["scanner_unknown_no_same_bid_sidecar_scanner"] += 1
        if scanner == "UNKNOWN":
            counts["scanner_defaulted_unknown"] += 1
            counts["scanner_source_unavailable"] += 1
        frame = _frame_from_manifest(str(meta.get("frame") or ""), tracer)
        row = {
            "subject_id_hash": sample_hash,
            "pet_path": str(pet_path.resolve()),
            "tracer": tracer,
            "suvr_global": f"{float(suvr):.8g}",
            "cohort": cohort,
            "paired_mri_subject_id_hash": sample_hash,
            "scanner": scanner,
            "frame_timing_min_start": f"{frame[0]:.1f}",
            "frame_timing_min_end": f"{frame[1]:.1f}",
        }
        rows.append(row)
        tracers[tracer] = tracers.get(tracer, 0) + 1
        cohorts[cohort] = cohorts.get(cohort, 0) + 1
    counts["rows_written"] = len(rows)
    report = {
        "status": "READY" if rows else "BLOCKED_NO_PAIRED_QC_PASS_PET",
        "counts": counts,
        "tracers": tracers,
        "cohorts": cohorts,
        "private_manifest": str(private_manifest) if private_manifest else None,
        "scanner_source_status": "AVAILABLE" if (counts["scanner_from_private_manifest"] or counts["scanner_from_sidecar"]) else "UNAVAILABLE_IN_PRIVATE_MANIFEST_AND_QC",
        "scanner_unknown_subjects_count": counts["scanner_defaulted_unknown"],
        "scanner_unknown_reasons": {
            "no_private_manifest_row": counts["scanner_unknown_no_private_manifest"],
            "private_manifest_has_no_raw_path": counts["scanner_unknown_no_raw_path"],
            "raw_sidecar_missing": counts["scanner_unknown_sidecar_missing"],
            "raw_sidecar_parse_error": counts["scanner_unknown_sidecar_parse_error"],
            "raw_sidecar_without_manufacturer": counts["scanner_unknown_sidecar_without_manufacturer"],
            "same_bid_sidecar_with_scanner_not_used": counts["scanner_unknown_same_bid_sidecar_with_scanner_not_used"],
            "no_same_bid_sidecar_scanner": counts["scanner_unknown_no_same_bid_sidecar_scanner"],
        },
        "scanner_same_bid_candidate_scanners_not_used": dict(sorted(same_bid_candidate_scanners.items())),
        "scanner_same_bid_candidate_policy": (
            "Same-BID sidecar scanners are reported for remediation only and are not assigned, "
            "because they are not the exact raw sidecar for the selected PET image."
        ),
    }
    return rows, report


def write_pet_input_list(rows: list[dict[str, str]], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=REQUIRED_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in REQUIRED_COLUMNS})


def main() -> int:
    parser = argparse.ArgumentParser(description="Build Method Y pet_input_list.csv from QC-pass MRI-paired PETs")
    parser.add_argument("--qc_root", default="data/preprocessed_pet")
    parser.add_argument("--preprocessed_root", default="data/preprocessed_pet")
    parser.add_argument("--mri_cache", default="data/cached/archive/t1_volume_cache_128_real_20260712T203222Z_26fff22aa7de.h5")
    parser.add_argument("--private_manifest", default="data/preprocessed_pet/_private/manifests/fbp_private_manifest.csv")
    parser.add_argument("--out", default="data/method_y/pet_input_list.csv")
    parser.add_argument("--report", default="results/method_y_execution/pet_input_list_build_report.json")
    parser.add_argument("--include_excluded", action="store_true")
    args = parser.parse_args()
    rows, report = build_pet_input_rows(
        qc_root=project_path(args.qc_root),
        preprocessed_root=project_path(args.preprocessed_root),
        mri_cache=project_path(args.mri_cache),
        private_manifest=project_path(args.private_manifest) if args.private_manifest else None,
        include_excluded=bool(args.include_excluded),
    )
    if rows:
        write_pet_input_list(rows, project_path(args.out))
        report["out"] = args.out
    write_json(args.report, report)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
