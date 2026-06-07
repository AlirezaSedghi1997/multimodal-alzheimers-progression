from __future__ import annotations

import argparse
import csv
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.a4_schema import (
    AB_LABEL_COLUMN,
    PET_DAY_COLUMN,
    REQUIRED_RELATIVE_FILES,
    T1_DAY_COLUMN,
    T1_NAME_PATTERN,
    resolve_data_root,
)
from src.utils.io_helpers import atomic_write_json, atomic_write_text
from src.utils.privacy import stable_hash
from src.utils.run_manifest import write_run_manifest
from src.utils.state_manager import StateManager


DEFAULT_MRI_QC_EXCLUSIONS = Path("configs/data/mri_qc_exclusions.csv")
def read_csv_dicts(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def to_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def normalize_ab(value: str | None) -> int | None:
    if value is None:
        return None
    v = value.strip().lower()
    if v in {"1", "pos", "positive", "ab_pos", "a+", "elevated"}:
        return 1
    if v in {"0", "neg", "negative", "ab_neg", "a-", "not elevated", "nonelevated"}:
        return 0
    if "pos" in v or "elev" in v:
        return 1
    if "neg" in v or "non" in v:
        return 0
    return None


def apoe4_carrier(value: str | None) -> int | None:
    if value is None or value == "":
        return None
    return 1 if "4" in value else 0


def norm_visit(value: str | None) -> str | None:
    if value is None or value == "":
        return None
    try:
        return str(int(float(value)))
    except ValueError:
        text = value.strip()
        return text if text and text.upper() != "NA" else None


def index_t1(root: Path) -> tuple[dict[tuple[str, str, str], dict[str, object]], dict[tuple[str, str], list[dict[str, object]]]]:
    pattern = re.compile(T1_NAME_PATTERN, re.IGNORECASE)
    exact: dict[tuple[str, str, str], dict[str, object]] = {}
    by_subject: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for p in (root / "T1").rglob("*"):
        if not p.is_file() or not (p.name.endswith(".nii") or p.name.endswith(".nii.gz")):
            continue
        m = pattern.match(p.name)
        if not m:
            continue
        substudy = m.group("substudy").upper()
        bid = m.group("bid")
        scan = int(m.group("scan"))
        key = (substudy, bid, str(scan))
        meta = {
            "t1_file_hash": stable_hash(p.name, prefix="t1"),
            "scan_number": scan,
        }
        exact[key] = meta
        by_subject[(substudy, bid)].append(meta)
    for items in by_subject.values():
        items.sort(key=lambda x: int(x["scan_number"]))
    return exact, by_subject


def read_visit_days(path: Path) -> dict[tuple[str, str, str], float]:
    out: dict[tuple[str, str, str], float] = {}
    if not path.exists():
        return out
    for row in read_csv_dicts(path):
        substudy = (row.get("SUBSTUDY") or "").upper()
        bid = row.get("BID") or ""
        visit = norm_visit(row.get("VISITCD"))
        if not substudy or not bid or visit is None:
            continue
        day = to_float(row.get("SVUSEDTC_DAYS_T0"))
        if day is None:
            day = to_float(row.get("SVSTDTC_DAYS_T0"))
        if day is not None:
            out[(substudy, bid, visit)] = day
    return out


def choose_nearest_t1(pet_day: float | None, candidates: list[dict[str, object]]) -> dict[str, object] | None:
    if pet_day is None or not candidates:
        return None
    valid = [c for c in candidates if c.get("t1_day_t0") is not None]
    if not valid:
        return None
    return min(valid, key=lambda c: (abs(float(c["t1_day_t0"]) - pet_day), float(c["t1_day_t0"])))


def read_mri_qc_exclusions(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    exclusions: dict[str, dict[str, str]] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sample_hash = (row.get("sample_hash") or "").strip()
            active = (row.get("active") or "true").strip().lower()
            if not sample_hash or active not in {"1", "true", "yes", "y"}:
                continue
            exclusions[sample_hash] = {k: (v or "") for k, v in row.items()}
    return exclusions


def apply_mri_qc_exclusions(
    rows: list[dict[str, object]],
    exclusions: dict[str, dict[str, str]],
) -> tuple[list[dict[str, object]], list[dict[str, str]]]:
    if not exclusions:
        return rows, []
    kept: list[dict[str, object]] = []
    excluded_rows: list[dict[str, str]] = []
    for row in rows:
        sample_hash = str(row.get("sample_hash", ""))
        exclusion = exclusions.get(sample_hash)
        if not exclusion:
            kept.append(row)
            continue
        excluded_rows.append(
            {
                "sample_hash": sample_hash,
                "patient_hash": str(row.get("patient_hash", "")),
                "t1_file_hash": str(row.get("t1_file_hash", "")),
                "reason": exclusion.get("reason", "MRI_QC_EXCLUSION"),
                "decision": exclusion.get("decision", "exclude_from_method_x_final_mri_cache_splits"),
            }
        )
    return kept, excluded_rows


def write_mri_qc_exclusion_report(excluded_rows: list[dict[str, str]], exclusions_path: Path) -> None:
    reason_counts = Counter(row["reason"] for row in excluded_rows)
    report = {
        "stage": "mri_qc_exclusions",
        "status": "APPLIED" if excluded_rows else "NO_ACTIVE_EXCLUSIONS_APPLIED",
        "active_exclusion_count": len(excluded_rows),
        "reason_counts": dict(reason_counts),
        "exclusions_path": str(exclusions_path),
        "excluded_rows": excluded_rows,
        "raw_identifiers_written": False,
        "raw_paths_written": False,
        "pet_as_input": False,
    }
    atomic_write_json("results/tables/mri_qc_exclusion_report.json", report)
    lines = ["# MRI QC Exclusion Report\n\n", f"- status: `{report['status']}`\n", f"- active_exclusion_count: `{len(excluded_rows)}`\n"]
    for reason, count in reason_counts.items():
        lines.append(f"- {reason}: `{count}`\n")
    lines.extend(["- raw_identifiers_written: `False`\n", "- raw_paths_written: `False`\n", "- pet_as_input: `False`\n"])
    atomic_write_text("results/tables/mri_qc_exclusion_report.md", "".join(lines))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default=os.environ.get("DATA_ROOT"))
    ap.add_argument("--mri_qc_exclusions", default=str(DEFAULT_MRI_QC_EXCLUSIONS))
    args = ap.parse_args()
    root = resolve_data_root(args.data_root)
    ab_path = root / REQUIRED_RELATIVE_FILES["ab_label"]
    vol_path = root / REQUIRED_RELATIVE_FILES["volumetric_mri"]
    subj_path = root / REQUIRED_RELATIVE_FILES["subjinfo"]
    visits_path = root / REQUIRED_RELATIVE_FILES["visits"]
    if not ab_path.exists() or not vol_path.exists() or not subj_path.exists() or not visits_path.exists():
        raise RuntimeError("Required catalog inputs are missing. Run inventory first.")

    t1_files, t1_by_subject_file = index_t1(root)
    visit_days = read_visit_days(visits_path)
    vol_rows = read_csv_dicts(vol_path)
    t1_by_subject: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    seen_candidates: set[tuple[str, str, str, str]] = set()
    for (substudy, bid), candidates in t1_by_subject_file.items():
        for meta in candidates:
            scan_norm = str(int(meta["scan_number"]))
            day = visit_days.get((substudy, bid, scan_norm))
            if day is None:
                continue
            key = (substudy, bid, scan_norm, str(meta["t1_file_hash"]))
            seen_candidates.add(key)
            t1_by_subject[(substudy, bid)].append({**meta, "t1_day_t0": day, "match_strategy": "sv_visit_timing"})
    for row in vol_rows:
        substudy = (row.get("SUBSTUDY") or "").upper()
        bid = row.get("BID") or ""
        scan = row.get("VISCODE") or row.get("scan_number") or ""
        scan_norm = norm_visit(scan)
        t1_meta = t1_files.get((substudy, bid, scan_norm)) if scan_norm is not None else None
        match_strategy = "exact_scan"
        if not t1_meta:
            subject_candidates = t1_by_subject_file.get((substudy, bid), [])
            if not subject_candidates:
                continue
            t1_meta = subject_candidates[0]
            match_strategy = "subject_fallback_first_t1"
        day = to_float(row.get(T1_DAY_COLUMN))
        if day is None:
            continue
        scan_key = str(int(t1_meta["scan_number"]))
        key = (substudy, bid, scan_key, str(t1_meta["t1_file_hash"]))
        if key in seen_candidates:
            continue
        seen_candidates.add(key)
        t1_by_subject[(substudy, bid)].append({**t1_meta, "t1_day_t0": day, "match_strategy": match_strategy})

    subj_rows = read_csv_dicts(subj_path)
    subj = {(r.get("SUBSTUDY") or "").upper() + "|" + (r.get("BID") or ""): r for r in subj_rows}
    rows: list[dict[str, object]] = []
    excluded = defaultdict(int)
    for pet in read_csv_dicts(ab_path):
        substudy = (pet.get("SUBSTUDY") or "").upper()
        bid = pet.get("BID") or ""
        label = normalize_ab(pet.get(AB_LABEL_COLUMN))
        pet_day = to_float(pet.get(PET_DAY_COLUMN))
        if label is None:
            excluded["missing_or_unknown_ab_label"] += 1
            continue
        nearest = choose_nearest_t1(pet_day, t1_by_subject.get((substudy, bid), []))
        if nearest is None:
            excluded["no_t1_match"] += 1
            continue
        srow = subj.get(substudy + "|" + bid, {})
        mmse = to_float(srow.get("MMSETSV6"))
        if mmse is None:
            mmse = to_float(srow.get("MMSETSV1"))
        t1_day = nearest.get("t1_day_t0")
        rows.append(
            {
                "patient_hash": stable_hash(substudy + "|" + bid, prefix="pt"),
                "sample_hash": stable_hash(substudy + "|" + bid + "|" + str(pet.get("VISCODE", "")), prefix="s"),
                "substudy": substudy,
                "ab_label": label,
                "pet_day_t0": pet_day,
                "t1_day_t0": t1_day,
                "abs_pet_t1_gap_days": abs(float(pet_day) - float(t1_day)) if pet_day is not None and t1_day is not None else None,
                "t1_file_hash": nearest["t1_file_hash"],
                "t1_scan_number": nearest["scan_number"],
                "t1_match_strategy": nearest.get("match_strategy", "unknown"),
                "age": to_float(srow.get("AGEYR")),
                "sex": srow.get("SEX") or None,
                "education_years": to_float(srow.get("EDCCNTU")),
                "apoe4_carrier": apoe4_carrier(srow.get("APOEGN")),
                "mmse_base": mmse,
                "has_tabular": bool(srow),
            }
        )

    exclusions_path = Path(args.mri_qc_exclusions)
    mri_qc_exclusions = read_mri_qc_exclusions(exclusions_path)
    rows, mri_qc_excluded_rows = apply_mri_qc_exclusions(rows, mri_qc_exclusions)
    if mri_qc_excluded_rows:
        excluded["mri_qc_manual_exclusion"] += len(mri_qc_excluded_rows)
    write_mri_qc_exclusion_report(mri_qc_excluded_rows, exclusions_path)

    out_csv = Path("data/processed/master_catalog.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else [
        "patient_hash",
        "sample_hash",
        "substudy",
        "ab_label",
        "pet_day_t0",
        "t1_day_t0",
        "abs_pet_t1_gap_days",
        "t1_file_hash",
        "t1_scan_number",
        "t1_match_strategy",
        "age",
        "sex",
        "education_years",
        "apoe4_carrier",
        "mmse_base",
        "has_tabular",
    ]
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    label_counts: dict[str, int] = defaultdict(int)
    substudy_counts: dict[str, int] = defaultdict(int)
    gap_within = {60: 0, 90: 0, 120: 0}
    match_strategy_counts: dict[str, int] = defaultdict(int)
    for r in rows:
        label_counts[str(r["ab_label"])] += 1
        substudy_counts[str(r["substudy"])] += 1
        match_strategy_counts[str(r["t1_match_strategy"])] += 1
        gap = r["abs_pet_t1_gap_days"]
        if gap is not None:
            for limit in gap_within:
                if float(gap) <= limit:
                    gap_within[limit] += 1
    summary = {
        "rows": len(rows),
        "excluded": dict(excluded),
        "label_counts": dict(label_counts),
        "substudy_counts": dict(substudy_counts),
        "gap_within_days": gap_within,
        "t1_match_strategy_counts": dict(match_strategy_counts),
        "raw_identifiers_written": False,
        "pet_as_input": False,
        "mri_qc_exclusion_count": len(mri_qc_excluded_rows),
        "mri_qc_exclusion_reasons": dict(Counter(row["reason"] for row in mri_qc_excluded_rows)),
        "mri_qc_exclusions_path": str(exclusions_path),
    }
    atomic_write_json("results/tables/master_catalog_summary.json", summary)
    md = [
        "# Master Catalog Summary\n\n",
        f"- rows: `{len(rows)}`\n",
        f"- excluded: `{dict(excluded)}`\n",
        f"- label_counts: `{dict(label_counts)}`\n",
        f"- substudy_counts: `{dict(substudy_counts)}`\n",
        f"- gap_within_days: `{gap_within}`\n",
        f"- t1_match_strategy_counts: `{dict(match_strategy_counts)}`\n",
        f"- mri_qc_exclusion_count: `{len(mri_qc_excluded_rows)}`\n",
        "- raw identifiers written: `False`\n",
        "- PET as model input: `False`\n",
    ]
    atomic_write_text("results/tables/master_catalog_summary.md", "".join(md))
    pkt = Path("results/approval_packets/GATE_H1_DATA")
    pkt.mkdir(parents=True, exist_ok=True)
    atomic_write_text(pkt / "README.md", "# GATE_H1_DATA\n\nAggregate data inventory and master catalog summary are ready for review.\n")
    write_run_manifest("results/logs/catalog_manifest.json", stage="catalog", data_root=str(root), extra={"summary": summary})
    StateManager().append_event("catalog", "COMPLETED", "Master catalog generated with hashed identifiers.")
    print(f"rows={len(rows)} excluded={sum(excluded.values())}")
    return 0 if rows else 2


if __name__ == "__main__":
    raise SystemExit(main())
