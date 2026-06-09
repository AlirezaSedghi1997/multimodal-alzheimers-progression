#!/usr/bin/env python3
"""Export Method Y primary_feasible_90 paired sample_hash list (design artifact; no PET cache required)."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.io_helpers import atomic_write_json, atomic_write_text
from src.utils.pet_manifest import read_catalog, tier_matches_requested
from src.utils.pet_spatial_qc import load_yaml


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--catalog", default="data/processed/master_catalog.csv")
    ap.add_argument("--policy", default="configs/model/method_y_spatial_preprocess_policy.yaml")
    ap.add_argument("--tier", default="primary_feasible_90")
    ap.add_argument("--out_csv", default="data/processed/catalogs/method_y_paired_primary_feasible_90.csv")
    ap.add_argument("--out_json", default="results/tables/method_y_paired_hash_list.json")
    args = ap.parse_args()

    policy = load_yaml(args.policy)
    catalog_path = PROJECT_ROOT / args.catalog if not Path(args.catalog).is_absolute() else Path(args.catalog)
    rows = read_catalog(catalog_path)

    selected = []
    for row in rows:
        if not tier_matches_requested(
            tier="", requested_tier=args.tier, gap=row.abs_pet_t1_gap_days, policy=policy
        ):
            continue
        if row.abs_pet_t1_gap_days is None:
            continue
        selected.append(
            {
                "sample_hash": row.sample_hash,
                "patient_hash": row.patient_hash,
                "substudy": row.substudy,
                "ab_label": row.ab_label,
                "abs_pet_t1_gap_days": row.abs_pet_t1_gap_days,
                "tracer": row.tracer,
            }
        )

    pos = sum(1 for r in selected if int(float(r["ab_label"])) == 1)
    neg = len(selected) - pos
    out_csv = PROJECT_ROOT / args.out_csv
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(selected[0].keys()) if selected else ["sample_hash"])
        writer.writeheader()
        writer.writerows(selected)

    summary = {
        "stage": "method_y_paired_hash_list",
        "tier": args.tier,
        "n": len(selected),
        "pos_like": pos,
        "neg_like": neg,
        "out_csv": str(out_csv.as_posix()),
        "interpretation": "Paired-subset hash list for Method Y Y1–Y6; MRI-only baselines must use this same list.",
    }
    atomic_write_json(PROJECT_ROOT / args.out_json, summary)
    md = (
        f"# Method Y Paired Hash List ({args.tier})\n\n"
        f"- N: **{len(selected)}** (pos={pos}, neg={neg})\n"
        f"- CSV: `{out_csv.as_posix()}`\n"
    )
    atomic_write_text(PROJECT_ROOT / "results/tables/method_y_paired_hash_list.md", md)
    print(json.dumps({"status": "OK", **summary}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
