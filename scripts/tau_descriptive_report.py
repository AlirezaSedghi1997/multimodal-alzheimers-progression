from __future__ import annotations

import argparse
import csv
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.a4_schema import REQUIRED_RELATIVE_FILES, T1_NAME_PATTERN, resolve_data_root
from src.utils.io_helpers import atomic_write_json, atomic_write_text


def to_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def norm_visit(value: str | None) -> str | None:
    if value is None or value == "":
        return None
    try:
        return str(int(float(value)))
    except ValueError:
        text = value.strip()
        return text if text and text.upper() != "NA" else None


def read_csv_dicts(path: Path):
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        yield from csv.DictReader(f)


def visit_days(path: Path) -> dict[tuple[str, str, str], float]:
    out = {}
    for row in read_csv_dicts(path):
        key = ((row.get("SUBSTUDY") or "").upper(), row.get("BID") or "", norm_visit(row.get("VISITCD")))
        if not key[0] or not key[1] or key[2] is None:
            continue
        day = to_float(row.get("SVUSEDTC_DAYS_T0")) or to_float(row.get("SVSTDTC_DAYS_T0"))
        if day is not None:
            out[key] = day
    return out


def t1_days_by_subject(root: Path, visits: dict[tuple[str, str, str], float]) -> dict[tuple[str, str], list[float]]:
    pattern = re.compile(T1_NAME_PATTERN, re.IGNORECASE)
    out: dict[tuple[str, str], list[float]] = defaultdict(list)
    for p in (root / "T1").rglob("*"):
        if not p.is_file() or not (p.name.endswith(".nii") or p.name.endswith(".nii.gz")):
            continue
        m = pattern.match(p.name)
        if not m:
            continue
        substudy = m.group("substudy").upper()
        bid = m.group("bid")
        scan = str(int(m.group("scan")))
        day = visits.get((substudy, bid, scan))
        if day is not None:
            out[(substudy, bid)].append(day)
    for vals in out.values():
        vals.sort()
    return out


def nearest_gap(day: float | None, candidates: list[float]) -> float | None:
    if day is None or not candidates:
        return None
    return min(abs(day - c) for c in candidates)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default=os.environ.get("DATA_ROOT"))
    args = ap.parse_args()
    root = resolve_data_root(args.data_root)
    tau_path = root / REQUIRED_RELATIVE_FILES["tau"]
    visits = visit_days(root / REQUIRED_RELATIVE_FILES["visits"])
    t1_by_subject = t1_days_by_subject(root, visits)

    scans: dict[tuple[str, str, str, str, str, str], dict[str, object]] = {}
    for row in read_csv_dicts(tau_path):
        if (row.get("scan_analyzed") or "").strip().lower() not in {"yes", "y", "1", "true"}:
            continue
        substudy = (row.get("SUBSTUDY") or "").upper()
        bid = row.get("BID") or ""
        key = (
            substudy,
            bid,
            row.get("VISCODE") or "",
            row.get("scan_number") or "",
            row.get("scan_date_DAYS_T0") or "",
            row.get("ligand") or "",
        )
        rec = scans.setdefault(
            key,
            {
                "substudy": substudy,
                "bid": bid,
                "tau_day": to_float(row.get("scan_date_DAYS_T0")),
                "n_rois": 0,
                "suvr_cer_sum": 0.0,
                "suvr_cer_n": 0,
            },
        )
        rec["n_rois"] = int(rec["n_rois"]) + 1
        val = to_float(row.get("suvr_cer"))
        if val is not None:
            rec["suvr_cer_sum"] = float(rec["suvr_cer_sum"]) + val
            rec["suvr_cer_n"] = int(rec["suvr_cer_n"]) + 1

    total_scans = len(scans)
    aligned = []
    for rec in scans.values():
        gap = nearest_gap(rec["tau_day"], t1_by_subject.get((str(rec["substudy"]), str(rec["bid"])), []))
        if gap is None:
            continue
        aligned.append({"substudy": rec["substudy"], "gap": gap, "n_rois": rec["n_rois"]})

    windows = {60: 0, 90: 0, 120: 0}
    substudy_120: dict[str, int] = defaultdict(int)
    for rec in aligned:
        for w in windows:
            if rec["gap"] <= w:
                windows[w] += 1
        if rec["gap"] <= 120:
            substudy_120[str(rec["substudy"])] += 1

    n120 = windows[120]
    if n120 >= 200:
        mode = "FULL"
    elif n120 >= 150:
        mode = "STRONG_BOOTSTRAP"
    elif n120 >= 50:
        mode = "DESCRIPTIVE_ONLY"
    else:
        mode = "SKIP_WITH_JUSTIFICATION"

    report = {
        "tau_scan_level_records": total_scans,
        "aligned_to_t1_any_gap": len(aligned),
        "aligned_within_days": windows,
        "substudy_within_120": dict(substudy_120),
        "feasibility_mode_120d": mode,
    }
    atomic_write_json("results/tables/tau_feasibility_report.json", report)
    md = [
        "# Tau Feasibility Report\n\n",
        f"- scan_level_records: `{total_scans}`\n",
        f"- aligned_to_t1_any_gap: `{len(aligned)}`\n",
        f"- aligned_within_60_days: `{windows[60]}`\n",
        f"- aligned_within_90_days: `{windows[90]}`\n",
        f"- aligned_within_120_days: `{windows[120]}`\n",
        f"- substudy_within_120: `{dict(substudy_120)}`\n",
        f"- feasibility_mode_120d: `{mode}`\n",
    ]
    atomic_write_text("results/tables/tau_feasibility_report.md", "".join(md))
    print(f"tau_scans={total_scans} within_120={n120} mode={mode}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

