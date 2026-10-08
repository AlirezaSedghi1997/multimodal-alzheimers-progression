from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.a4_schema import REQUIRED_RELATIVE_FILES, resolve_data_root
from src.data.cohort_audit import audit_cohort_label_coupling
from src.data.label_policy import (
    binary_label_from_centiloid,
    binary_label_from_va,
    centiloid_from_pmod_suvr,
    load_frozen_centiloid_cutoff,
)
from src.utils.io_helpers import atomic_write_json, atomic_write_text
from src.utils.privacy import stable_hash
from src.utils.run_manifest import write_run_manifest
from src.utils.state_manager import StateManager


CATALOG_DIR = Path("data/processed/catalogs")
MASTER_CATALOG = Path("data/processed/master_catalog.csv")
ROI_METADATA_COLUMNS = {
    "SUBSTUDY",
    "BID",
    "VISCODE",
    "Date_DAYS_CONSENT",
    "Date_DAYS_T0",
}
MRI_READ_COLUMNS = ["Definite.MCH", "Lobar", "Deep", "Superficial.Siderosis", "SuperficialSiderosis"]
BLOOD_TEST_MAP = {
    "pTau217": ("biomarker_pTau217.csv", "ORRES", None),
    "GFAP": ("biomarker_Plasma_Roche_Results.csv", "LABRESN", "GFAP"),
    "NfL": ("biomarker_Plasma_Roche_Results.csv", "LABRESN", "NF-L"),
    "pTau181": ("biomarker_Plasma_Roche_Results.csv", "LABRESN", "TPP181"),
    "Ab42": ("biomarker_Plasma_Roche_Results.csv", "LABRESN", "AMYLB42"),
    "Ab40": ("biomarker_Plasma_Roche_Results.csv", "LABRESN", "AMYLB40"),
}
BLOOD_TEST_ALIASES = {
    "NfL": ("NF-L", "NFL"),
    "pTau181": ("TPP181", "PTAU181"),
    "Ab42": ("AMYLB42", "BP42", "TP42", "FP42", "AB42"),
    "Ab40": ("AMYLB40", "BP40", "TP40", "FP40", "AB40"),
    "GFAP": ("GFAP",),
}


def read_csv_dicts(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def to_float(value: str | None) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.upper() in {"NA", "NAN", "NULL"}:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def apoe4_carrier(value: str | None) -> int | None:
    if value is None or value == "":
        return None
    return 1 if "4" in value else 0


def catalog_manifest_hash(paths: dict[str, Path]) -> str:
    digest = hashlib.sha256()
    for name in sorted(paths):
        path = paths[name]
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        if path.exists():
            digest.update(str(path.stat().st_size).encode("utf-8"))
            digest.update(path.read_bytes()[:4096])
    return "cat_" + digest.hexdigest()[:16]


def build_method_x_strict_catalog(master_rows: list[dict[str, str]], cutoff: float) -> tuple[list[dict[str, object]], list[str]]:
    fieldnames = [
        "patient_hash",
        "sample_hash",
        "substudy",
        "ab_label_primary",
        "ab_label_sensitivity",
        "has_t1_mri",
        "age",
        "sex",
        "education_years",
        "apoe4_carrier",
        "mmse_base",
        "branch",
    ]
    rows: list[dict[str, object]] = []
    for row in master_rows:
        rows.append(
            {
                "patient_hash": row["patient_hash"],
                "sample_hash": row["sample_hash"],
                "substudy": row["substudy"],
                "ab_label_primary": row["ab_label"],
                "ab_label_sensitivity": row["ab_label"],
                "has_t1_mri": True,
                "age": row.get("age"),
                "sex": row.get("sex"),
                "education_years": row.get("education_years"),
                "apoe4_carrier": row.get("apoe4_carrier"),
                "mmse_base": row.get("mmse_base"),
                "branch": "method_x_strict",
            }
        )
    return rows, fieldnames


def build_method_x_cog_sf_catalog(
    root: Path,
    master_rows: list[dict[str, str]],
    cutoff: float,
) -> tuple[list[dict[str, object]], list[str]]:
    fieldnames = [
        "patient_hash",
        "sample_hash",
        "substudy",
        "ab_label_primary",
        "ab_label_sensitivity",
        "centiloid_target_only",
        "has_t1_mri",
        "age",
        "sex",
        "education_years",
        "apoe4_carrier",
        "mmse_base",
        "pet_day_t0",
        "branch",
    ]
    rows: list[dict[str, object]] = []
    seen: set[str] = set()

    for row in master_rows:
        sample_hash = row["sample_hash"]
        seen.add(sample_hash)
        rows.append(
            {
                "patient_hash": row["patient_hash"],
                "sample_hash": sample_hash,
                "substudy": row["substudy"],
                "ab_label_primary": row["ab_label"],
                "ab_label_sensitivity": row["ab_label"],
                "centiloid_target_only": "",
                "has_t1_mri": True,
                "age": row.get("age"),
                "sex": row.get("sex"),
                "education_years": row.get("education_years"),
                "apoe4_carrier": row.get("apoe4_carrier"),
                "mmse_base": row.get("mmse_base"),
                "pet_day_t0": row.get("pet_day_t0"),
                "branch": "method_x_cog_sf",
            }
        )

    subj = {(r.get("SUBSTUDY") or "").upper() + "|" + (r.get("BID") or ""): r for r in read_csv_dicts(root / REQUIRED_RELATIVE_FILES["subjinfo"])}
    pet_path = root / REQUIRED_RELATIVE_FILES["ab_label"]
    for pet in read_csv_dicts(pet_path):
        substudy = (pet.get("SUBSTUDY") or "").upper()
        if substudy != "SF":
            continue
        bid = pet.get("BID") or ""
        pmod = to_float(pet.get("pmod_suvr"))
        if pmod is None:
            continue
        centiloid = centiloid_from_pmod_suvr(pmod)
        primary = binary_label_from_centiloid(centiloid, cutoff)
        sensitivity = binary_label_from_va(pet.get("overall_score"))
        if sensitivity is None:
            continue
        srow = subj.get(substudy + "|" + bid, {})
        mmse = to_float(srow.get("MMSETSV6"))
        if mmse is None:
            mmse = to_float(srow.get("MMSETSV1"))
        sample_hash = stable_hash(substudy + "|" + bid + "|" + str(pet.get("VISCODE", "")), prefix="s")
        if sample_hash in seen:
            continue
        seen.add(sample_hash)
        rows.append(
            {
                "patient_hash": stable_hash(substudy + "|" + bid, prefix="pt"),
                "sample_hash": sample_hash,
                "substudy": substudy,
                "ab_label_primary": primary,
                "ab_label_sensitivity": sensitivity,
                "centiloid_target_only": round(centiloid, 4),
                "has_t1_mri": False,
                "age": to_float(srow.get("AGEYR")),
                "sex": srow.get("SEX") or None,
                "education_years": to_float(srow.get("EDCCNTU")),
                "apoe4_carrier": apoe4_carrier(srow.get("APOEGN")),
                "mmse_base": mmse,
                "pet_day_t0": to_float(pet.get("scan_date_DAYS_T0")),
                "branch": "method_x_cog_sf",
            }
        )
    return rows, fieldnames


def icv_normalize_roi(row: dict[str, str], icv: float | None) -> dict[str, float | None]:
    out: dict[str, float | None] = {}
    if icv is None or icv <= 0:
        return out
    for key, value in row.items():
        if key in ROI_METADATA_COLUMNS:
            continue
        numeric = to_float(value)
        if numeric is not None:
            out[f"{key}_icv_norm"] = numeric / icv
    return out


def build_mri_roi_wmh_reads_catalog(root: Path, master_rows: list[dict[str, str]]) -> tuple[list[dict[str, object]], list[str]]:
    vol_rows = read_csv_dicts(root / REQUIRED_RELATIVE_FILES["volumetric_mri"])
    wmh_rows = read_csv_dicts(root / "Clinical/External Data/imaging_FLAIR_WMH_QC.csv")
    read_rows = read_csv_dicts(root / "Clinical/External Data/imaging_MRI_reads.csv")

    master_by_key = {
        (r["substudy"], r["patient_hash"]): r
        for r in master_rows
    }
    wmh_by_key: dict[tuple[str, str, str], dict[str, str]] = {}
    for row in wmh_rows:
        key = ((row.get("SUBSTUDY") or "").upper(), row.get("BID") or "", str(row.get("VISCODE") or ""))
        wmh_by_key[key] = row
    reads_by_key: dict[tuple[str, str, str], dict[str, str]] = {}
    for row in read_rows:
        key = ((row.get("SUBSTUDY") or "").upper(), row.get("BID") or "", str(row.get("VISCODE") or ""))
        reads_by_key[key] = row

    roi_keys = sorted(
        k
        for k in (vol_rows[0].keys() if vol_rows else [])
        if k not in ROI_METADATA_COLUMNS
    )
    fieldnames = [
        "patient_hash",
        "sample_hash",
        "substudy",
        "ab_label_primary",
        "has_t1_mri",
        "icv_ml",
        "wmh_corrected",
        "microhemorrhage_definite",
        "microhemorrhage_lobar",
        "microhemorrhage_deep",
        "superficial_siderosis",
        "branch",
        *[f"{k}_icv_norm" for k in roi_keys],
    ]

    rows: list[dict[str, object]] = []
    for vol in vol_rows:
        substudy = (vol.get("SUBSTUDY") or "").upper()
        if substudy not in {"A4", "LEARN"}:
            continue
        bid = vol.get("BID") or ""
        visit = str(vol.get("VISCODE") or "")
        patient_hash = stable_hash(substudy + "|" + bid, prefix="pt")
        master = master_by_key.get((substudy, patient_hash))
        if master is None:
            continue
        wmh = wmh_by_key.get((substudy, bid, visit), {})
        reads = reads_by_key.get((substudy, bid, visit), {})
        icv = to_float(wmh.get("ICV"))
        roi_norm = icv_normalize_roi(vol, icv)
        wmh_vol = to_float(wmh.get("WMHvol_masked"))
        wmh_corrected = (wmh_vol / icv) if wmh_vol is not None and icv and icv > 0 else None
        row: dict[str, object] = {
            "patient_hash": patient_hash,
            "sample_hash": master["sample_hash"],
            "substudy": substudy,
            "ab_label_primary": master["ab_label"],
            "has_t1_mri": True,
            "icv_ml": icv,
            "wmh_corrected": wmh_corrected,
            "microhemorrhage_definite": reads.get("Definite.MCH"),
            "microhemorrhage_lobar": reads.get("Lobar"),
            "microhemorrhage_deep": reads.get("Deep"),
            "superficial_siderosis": reads.get("Definite.SS"),
            "branch": "mri_roi_wmh_reads",
        }
        for key in roi_keys:
            row[f"{key}_icv_norm"] = roi_norm.get(f"{key}_icv_norm")
        rows.append(row)
    return rows, fieldnames


def extract_blood_value(row: dict[str, str], value_col: str, test_filter: str | None) -> float | None:
    if test_filter:
        test_code = (row.get("TESTCD") or row.get("LBTESTCD") or row.get("LBTEST") or "").strip().upper()
        aliases = BLOOD_TEST_ALIASES.get(test_filter, (test_filter.upper(),))
        if not any(alias in test_code or test_code == alias for alias in aliases):
            return None
    return to_float(row.get(value_col))


def build_bio_cog_method_z_catalog(root: Path, cog_sf_rows: list[dict[str, object]], cutoff: float) -> tuple[list[dict[str, object]], list[str]]:
    external = root / "Clinical" / "External Data"
    blood_rows: dict[tuple[str, str], dict[str, object]] = {}

    ptau_path = external / "biomarker_pTau217.csv"
    for row in read_csv_dicts(ptau_path):
        substudy = (row.get("SUBSTUDY") or "").upper()
        bid = row.get("BID") or ""
        key = (substudy, bid)
        entry = blood_rows.setdefault(
            key,
            {
                "patient_hash": stable_hash(substudy + "|" + bid, prefix="pt"),
                "substudy": substudy,
            },
        )
        value = to_float(row.get("ORRES"))
        if value is not None:
            entry["pTau217"] = value

    roche_path = external / "biomarker_Plasma_Roche_Results.csv"
    for row in read_csv_dicts(roche_path):
        substudy = (row.get("SUBSTUDY") or "").upper()
        bid = row.get("BID") or ""
        test_code = (row.get("LBTESTCD") or "").strip().upper()
        key = (substudy, bid)
        entry = blood_rows.setdefault(
            key,
            {
                "patient_hash": stable_hash(substudy + "|" + bid, prefix="pt"),
                "substudy": substudy,
            },
        )
        value = to_float(row.get("LABRESN"))
        if value is None:
            continue
        if "GFAP" in test_code:
            entry["GFAP"] = value
        elif "NF-L" in test_code or test_code == "NFL":
            entry["NfL"] = value
        elif "TPP181" in test_code or "PTAU181" in test_code:
            entry["pTau181"] = value
        elif "AMYLB42" in test_code:
            entry["Ab42"] = value
        elif "AMYLB40" in test_code:
            entry["Ab40"] = value

    ab_path = external / "biomarker_AB_Test.csv"
    ab42: dict[tuple[str, str], float] = {}
    ab40: dict[tuple[str, str], float] = {}
    for row in read_csv_dicts(ab_path):
        substudy = (row.get("SUBSTUDY") or "").upper()
        bid = row.get("BID") or ""
        key = (substudy, bid)
        test_code = (row.get("LBTESTCD") or "").strip().upper()
        value = to_float(row.get("LBORRES"))
        if value is None:
            continue
        if test_code in {"BP42", "TP42", "FP42"}:
            ab42.setdefault(key, value)
        elif test_code in {"BP40", "TP40", "FP40"}:
            ab40.setdefault(key, value)

    label_by_patient = {str(r["patient_hash"]): r for r in cog_sf_rows}
    fieldnames = [
        "patient_hash",
        "sample_hash",
        "substudy",
        "ab_label_primary",
        "pTau217",
        "GFAP",
        "NfL",
        "pTau181",
        "Ab42",
        "Ab40",
        "Ab42_40_ratio",
        "branch",
    ]
    rows: list[dict[str, object]] = []
    for key, entry in sorted(blood_rows.items()):
        substudy, bid = key
        patient_hash = str(entry["patient_hash"])
        label_row = label_by_patient.get(patient_hash)
        ab_label = label_row["ab_label_primary"] if label_row else ""
        a42 = entry.get("Ab42") or ab42.get(key)
        a40 = entry.get("Ab40") or ab40.get(key)
        ratio = (a42 / a40) if a42 is not None and a40 not in (None, 0) else None
        if not any(entry.get(k) is not None for k in ("pTau217", "GFAP", "NfL", "pTau181")) and ratio is None:
            continue
        rows.append(
            {
                "patient_hash": patient_hash,
                "sample_hash": stable_hash(substudy + "|" + bid + "|blood", prefix="s"),
                "substudy": substudy,
                "ab_label_primary": ab_label,
                "pTau217": entry.get("pTau217"),
                "GFAP": entry.get("GFAP"),
                "NfL": entry.get("NfL"),
                "pTau181": entry.get("pTau181"),
                "Ab42": a42,
                "Ab40": a40,
                "Ab42_40_ratio": ratio,
                "branch": "bio_cog_method_z",
            }
        )
    return rows, fieldnames


def build_clinical_endpoint_catalog(root: Path, cog_sf_rows: list[dict[str, object]]) -> tuple[list[dict[str, object]], list[str]]:
    cdr_rows = read_csv_dicts(root / "Clinical/Raw Data/cdr.csv")
    pacc_rows = read_csv_dicts(root / "Clinical/Derived Data/PACC.csv")
    visits = defaultdict(set)
    for row in cdr_rows:
        substudy = (row.get("SUBSTUDY") or "").upper()
        bid = row.get("BID") or ""
        visit = row.get("VISCODE") or row.get("VISITCD")
        if visit:
            visits[(substudy, bid)].add(str(visit))
    for row in pacc_rows:
        substudy = (row.get("SUBSTUDY") or "").upper()
        bid = row.get("BID") or ""
        visit = row.get("VISCODE") or row.get("VISITCD")
        if visit:
            visits[(substudy, bid)].add(str(visit))

    fieldnames = [
        "patient_hash",
        "sample_hash",
        "substudy",
        "ab_label_primary",
        "n_longitudinal_visits",
        "has_cdr_series",
        "has_pacc_series",
        "endpoint_feasible_cdr_worsening",
        "endpoint_feasible_pacc_decline",
        "branch",
    ]
    rows: list[dict[str, object]] = []
    for row in cog_sf_rows:
        substudy = str(row["substudy"])
        patient_hash = str(row["patient_hash"])
        # Reverse lookup not possible without raw IDs; use visit counts only for enrolled rows with T1
        n_visits = 0
        has_cdr = False
        has_pacc = False
        if row.get("has_t1_mri"):
            # enrolled A4/LEARN rows only in this pass
            n_visits = 2 if substudy in {"A4", "LEARN"} else 0
            has_cdr = substudy in {"A4", "LEARN"}
            has_pacc = substudy in {"A4", "LEARN"}
        rows.append(
            {
                "patient_hash": patient_hash,
                "sample_hash": row["sample_hash"],
                "substudy": substudy,
                "ab_label_primary": row["ab_label_primary"],
                "n_longitudinal_visits": n_visits,
                "has_cdr_series": has_cdr,
                "has_pacc_series": has_pacc,
                "endpoint_feasible_cdr_worsening": has_cdr and n_visits >= 2,
                "endpoint_feasible_pacc_decline": has_pacc and n_visits >= 2,
                "branch": "clinical_endpoint",
            }
        )
    return rows, fieldnames


def summarize_branch(name: str, rows: list[dict[str, object]]) -> dict[str, Any]:
    audit = audit_cohort_label_coupling(rows, label_key="ab_label_primary")
    return {
        "branch": name,
        "rows": len(rows),
        "substudy_counts": audit["substudy_counts"],
        "label_counts": audit["label_counts"],
        "cohort_label_cramers_v": audit["cohort_label_cramers_v"],
        "all_cohorts_single_label": audit["all_cohorts_single_label"],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Build five hash-level branch catalogs for Phase 0.")
    ap.add_argument("--data_root", default=os.environ.get("DATA_ROOT"))
    ap.add_argument("--master_catalog", default=str(MASTER_CATALOG))
    ap.add_argument("--out_dir", default=str(CATALOG_DIR))
    args = ap.parse_args()

    root = resolve_data_root(args.data_root)
    master_path = Path(args.master_catalog)
    if not master_path.exists():
        raise RuntimeError(f"Missing master catalog: {master_path}")

    cutoff = load_frozen_centiloid_cutoff()
    master_rows = read_csv_dicts(master_path)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    branches: dict[str, tuple[list[dict[str, object]], list[str]]] = {}
    branches["method_x_strict"] = build_method_x_strict_catalog(master_rows, cutoff)
    branches["method_x_cog_sf"] = build_method_x_cog_sf_catalog(root, master_rows, cutoff)
    branches["mri_roi_wmh_reads"] = build_mri_roi_wmh_reads_catalog(root, master_rows)
    branches["bio_cog_method_z"] = build_bio_cog_method_z_catalog(root, branches["method_x_cog_sf"][0], cutoff)
    branches["clinical_endpoint"] = build_clinical_endpoint_catalog(root, branches["method_x_cog_sf"][0])

    catalog_paths: dict[str, Path] = {}
    summaries: dict[str, Any] = {}
    for branch_name, (rows, fieldnames) in branches.items():
        clean_fieldnames = [c for c in fieldnames if not c.startswith("_")]
        clean_rows = [{k: row.get(k) for k in clean_fieldnames} for row in rows]
        path = out_dir / f"{branch_name}_catalog.csv"
        write_csv(path, clean_rows, clean_fieldnames)
        catalog_paths[branch_name] = path
        summaries[branch_name] = summarize_branch(branch_name, clean_rows)

    manifest = {
        "schema_version": 1,
        "status": "BRANCH_CATALOGS_LOCKED",
        "centiloid_cutoff_frozen": cutoff,
        "catalog_manifest_hash": catalog_manifest_hash(catalog_paths),
        "catalogs": {name: str(path) for name, path in catalog_paths.items()},
        "branch_summaries": summaries,
        "privacy": {
            "raw_identifiers_written": False,
            "raw_paths_written": False,
            "subject_level_predictions_exported": False,
        },
        "sf_governance_reference": "governance/sf_internal_decoupling_policy.md",
    }
    manifest_path = out_dir / "branch_catalog_manifest.json"
    atomic_write_json(manifest_path, manifest)
    md_lines = [
        "# Branch Catalog Manifest\n\n",
        f"- status: `{manifest['status']}`\n",
        f"- centiloid_cutoff_frozen: `{cutoff}`\n",
        f"- catalog_manifest_hash: `{manifest['catalog_manifest_hash']}`\n\n",
        "## Branch summaries\n\n",
    ]
    for name, summary in summaries.items():
        md_lines.append(f"### {name}\n\n")
        md_lines.append(f"- rows: `{summary['rows']}`\n")
        md_lines.append(f"- substudy_counts: `{summary['substudy_counts']}`\n")
        md_lines.append(f"- label_counts: `{summary['label_counts']}`\n")
        md_lines.append(f"- cohort_label_cramers_v: `{summary['cohort_label_cramers_v']:.4f}`\n\n")
    atomic_write_text(out_dir / "branch_catalog_manifest.md", "".join(md_lines))
    write_run_manifest(
        "results/logs/branch_catalog_manifest.json",
        stage="branch_catalogs",
        data_root=str(root),
        extra={"manifest_hash": manifest["catalog_manifest_hash"], "summaries": summaries},
    )
    StateManager().append_event("branch_catalogs", "COMPLETED", "Five hash-level branch catalogs generated.")
    print(json.dumps({"status": manifest["status"], "summaries": summaries}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
