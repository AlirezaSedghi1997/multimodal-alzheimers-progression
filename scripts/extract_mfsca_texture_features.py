#!/usr/bin/env python3
"""Extract MFSCA texture surrogate features from MRI header stats."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features.mfsca_texture import MFSCA_FEATURE_NAMES, extract_mfsca_from_mri_row
from src.utils.io_helpers import atomic_write_json


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mri_features", default="data/processed/mri_features.csv")
    ap.add_argument("--out", default="results/tables/mfsca_texture_features.json")
    args = ap.parse_args()

    mri_path = PROJECT_ROOT / args.mri_features
    if not mri_path.exists():
        print(json.dumps({"status": "BLOCKED", "blocker": "MRI_FEATURES_MISSING"}, indent=2))
        return 1

    rows = []
    for row in read_rows(mri_path):
        if row.get("mri_feature_status") != "OK":
            continue
        feat = extract_mfsca_from_mri_row(row)
        rows.append({"sample_hash": row["sample_hash"], **feat})

    out = {
        "literature_anchor": "LIT018 arXiv:2606.10222",
        "features": MFSCA_FEATURE_NAMES,
        "n": len(rows),
        "note": "Surrogate on header stats; full 3D Hilbert MFSCA requires slice pipeline (R3).",
        "rows": rows,
    }
    out_path = PROJECT_ROOT / args.out
    atomic_write_json(out_path, out)
    print(json.dumps({"status": "COMPLETED", "n": len(rows), "out": str(out_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
