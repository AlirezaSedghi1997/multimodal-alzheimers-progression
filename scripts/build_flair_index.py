#!/usr/bin/env python3
"""Index FLAIR volumes linkable to master catalog (T1+FLAIR sensitivity track)."""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.a4_schema import resolve_data_root
from src.utils.io_helpers import atomic_write_json, atomic_write_text
from src.utils.privacy import stable_hash

FLAIR_PATTERN = re.compile(
    r"^(?P<substudy>A4|LEARN)_MR_FLAIR_(?P<bid>B[0-9]+)_(?P<scan>[0-9]{3})\.nii(\.gz)?$",
    re.IGNORECASE,
)


def read_catalog(path: Path) -> dict[tuple[str, str], dict[str, str]]:
    out: dict[tuple[str, str], dict[str, str]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            substudy = (row.get("substudy") or "").upper()
            patient_hash = row.get("patient_hash") or ""
            if substudy and patient_hash:
                out[(substudy, patient_hash)] = row
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="")
    ap.add_argument("--catalog", default="data/processed/master_catalog.csv")
    ap.add_argument("--out_json", default="results/tables/flair_volume_index.json")
    ap.add_argument("--out_md", default="results/tables/flair_volume_index.md")
    args = ap.parse_args()

    root = resolve_data_root(args.data_root)
    catalog = read_catalog(PROJECT_ROOT / args.catalog)
    flair_dir = root / "Flair"
    if not flair_dir.exists():
        flair_dir = root / "FLAIR"
    rows = []
    if flair_dir.exists():
        for path in flair_dir.rglob("*.nii*"):
            match = FLAIR_PATTERN.match(path.name)
            if not match:
                continue
            substudy = match.group("substudy").upper()
            bid = match.group("bid").upper()
            patient_hash = stable_hash(substudy + "|" + bid, prefix="pt")
            master = catalog.get((substudy, patient_hash))
            rows.append(
                {
                    "substudy": substudy,
                    "patient_hash": patient_hash,
                    "sample_hash": master.get("sample_hash") if master else None,
                    "has_t1_in_catalog": master is not None,
                    "flair_file_hash": stable_hash(path.as_posix(), prefix="fh"),
                    "scan_code": match.group("scan"),
                }
            )

    linked = sum(1 for r in rows if r.get("has_t1_in_catalog"))
    report = {
        "stage": "flair_volume_index",
        "flair_dir_exists": flair_dir.exists(),
        "flair_files_indexed": len(rows),
        "linked_to_master_catalog": linked,
        "link_rate": round(linked / len(rows), 4) if rows else None,
        "rows_sample": rows[:100],
        "note": "Aggregate index only; raw paths not exported. Use for T1+FLAIR sensitivity planning.",
    }
    out_json = PROJECT_ROOT / args.out_json
    out_md = PROJECT_ROOT / args.out_md
    out_json.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(out_json, report)
    atomic_write_text(
        out_md,
        "# FLAIR Volume Index\n\n"
        f"- FLAIR files indexed: **{len(rows)}**\n"
        f"- Linked to master catalog: **{linked}**\n"
        f"- Link rate: **{report['link_rate']}**\n",
    )
    print(json.dumps({"status": "OK", "flair_files_indexed": len(rows), "linked": linked}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
