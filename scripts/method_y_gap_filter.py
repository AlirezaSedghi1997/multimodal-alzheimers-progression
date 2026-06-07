#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.pet_manifest import read_catalog, tier_for_gap
from src.utils.pet_spatial_qc import atomic_write_json, atomic_write_text, load_yaml


def chi2_2d(table: dict[tuple[str, str], int]) -> dict:
    rows = sorted({k[0] for k in table})
    cols = sorted({k[1] for k in table})
    if len(rows) < 2 or len(cols) < 2:
        return {"status": "SINGLE_ROW_OR_COLUMN", "chi2": None}
    total = sum(table.values())
    row_tot = {r: sum(table.get((r, c), 0) for c in cols) for r in rows}
    col_tot = {c: sum(table.get((r, c), 0) for r in rows) for c in cols}
    chi2 = 0.0
    for r in rows:
        for c in cols:
            exp = row_tot[r] * col_tot[c] / total if total else 0
            obs = table.get((r, c), 0)
            if exp > 0:
                chi2 += (obs - exp) ** 2 / exp
    # no scipy dependency; provide statistic and a conservative coupling flag
    return {"status": "OK", "chi2": chi2, "rows": rows, "cols": cols, "table": {f"{r}|{c}": table.get((r,c),0) for r in rows for c in cols}}


def main() -> int:
    ap = argparse.ArgumentParser(description="Aggregate PET-T1 temporal window and confound feasibility report.")
    ap.add_argument("--catalog", default="data/processed/master_catalog.csv")
    ap.add_argument("--policy", default="configs/model/method_y_spatial_preprocess_policy.yaml")
    args = ap.parse_args()
    policy = load_yaml(args.policy)
    rows = read_catalog(PROJECT_ROOT / args.catalog if not Path(args.catalog).is_absolute() else args.catalog)

    tier_counts = Counter()
    label_by_tier = defaultdict(Counter)
    cohort_by_label = Counter()
    tracer_by_label = Counter()
    gap_by_label = Counter()
    missing_gap = 0
    for r in rows:
        tier = tier_for_gap(r.abs_pet_t1_gap_days, policy)
        tier_counts[tier] += 1
        label_by_tier[tier][str(r.ab_label)] += 1
        cohort_by_label[(r.substudy or "UNK", str(r.ab_label))] += 1
        tracer_by_label[(r.tracer or "UNK", str(r.ab_label))] += 1
        gap_by_label[(tier, str(r.ab_label))] += 1
        if r.abs_pet_t1_gap_days is None:
            missing_gap += 1
    feasibility = {}
    for tier, counts in label_by_tier.items():
        pos = counts.get("1", 0) + counts.get("AB_POS", 0) + counts.get("positive", 0)
        neg = counts.get("0", 0) + counts.get("AB_NEG", 0) + counts.get("negative", 0)
        n = sum(counts.values())
        feasibility[tier] = {
            "n": n, "pos_like": pos, "neg_like": neg,
            "trainable_binary": pos > 0 and neg > 0 and n >= 100,
            "warn_low_n": n < 300,
        }
    cohort_audit = chi2_2d(cohort_by_label)
    tracer_audit = chi2_2d(tracer_by_label)
    gap_label_audit = chi2_2d(gap_by_label)

    report = {
        "stage": "method_y_gap_filter",
        "catalog_rows": len(rows),
        "tier_counts": dict(tier_counts),
        "label_by_tier": {k: dict(v) for k, v in label_by_tier.items()},
        "missing_gap_rows": missing_gap,
        "feasibility_by_tier": feasibility,
        "cohort_label_audit": cohort_audit,
        "tracer_label_audit": tracer_audit,
        "gap_label_audit": gap_label_audit,
        "primary_training_window_days": policy.get("temporal_pairing", {}).get("primary_training_window_days", 90),
        "gold_standard_window_days": policy.get("temporal_pairing", {}).get("ideal_gold_standard_window_days", 30),
        "gap_is_model_feature": False,
        "raw_paths_exported": False,
        "verdict": "REPORT_READY_NOT_TRAINING_PERMISSION",
    }
    atomic_write_json(PROJECT_ROOT / "results" / "tables" / "method_y_gap_filter_report.json", report)
    md = ["# Method Y PET Gap/Confound Filter Report\n\n", f"- Catalog rows: {len(rows)}\n", f"- Tier counts: `{dict(tier_counts)}`\n", f"- Feasibility: `{feasibility}`\n", "- This report does not grant training readiness.\n"]
    atomic_write_text(PROJECT_ROOT / "results" / "tables" / "method_y_gap_filter_report.md", "".join(md))
    print(report["verdict"])
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
