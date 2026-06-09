#!/usr/bin/env python3
"""Build Method Y incremental-value table scaffold (ΔAUC vs PET-only; metrics filled when Y1–Y4 complete)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.io_helpers import atomic_write_json, atomic_write_text


def _load_metrics(path: Path) -> dict | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    agg = payload.get("aggregate") or payload
    if not isinstance(agg, dict):
        return None
    return {
        "roc_auc": agg.get("roc_auc"),
        "pr_auc": agg.get("pr_auc"),
        "brier": agg.get("brier"),
        "n": agg.get("n"),
        "status": agg.get("status") or payload.get("status"),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_json", default="results/tables/method_y_incremental_value_table.json")
    ap.add_argument("--out_md", default="results/tables/method_y_incremental_value_table.md")
    args = ap.parse_args()

    sources = {
        "Y1_pet_only": PROJECT_ROOT / "results/final_results/method_y/pet_training/method_y_pet_metrics.json",
        "Y2_mri_paired": PROJECT_ROOT / "results/final_results/method_y/mri_paired_training/method_y_mri_paired_metrics.json",
        "Y3_late_fusion": PROJECT_ROOT / "results/final_results/method_y/late_fusion_training/method_y_late_fusion_metrics.json",
        "Y4_gated_fusion": PROJECT_ROOT / "results/final_results/method_y/gated_fusion_training/method_y_gated_fusion_metrics.json",
    }
    metrics = {key: _load_metrics(path) for key, path in sources.items()}
    pet_auc = (metrics.get("Y1_pet_only") or {}).get("roc_auc")
    rows = []
    for comp_id, label in [
        ("Y2_mri_paired", "MRI-only on paired subset"),
        ("Y3_late_fusion", "Late fusion MRI+PET"),
        ("Y4_gated_fusion", "Gated fusion MRI+PET"),
    ]:
        m = metrics.get(comp_id)
        delta = None
        if m and m.get("roc_auc") is not None and pet_auc is not None:
            delta = float(m["roc_auc"]) - float(pet_auc)
        rows.append(
            {
                "comparison_id": comp_id,
                "label": label,
                "metrics_status": "AVAILABLE" if m else "PENDING_TRAINING",
                "roc_auc": (m or {}).get("roc_auc"),
                "pr_auc": (m or {}).get("pr_auc"),
                "delta_auc_vs_pet_only": delta,
                "headline_allowed": False,
                "caveat": "Paired subset N=266 with 232/34 imbalance; descriptive until bootstrap CI and DCA complete.",
            }
        )

    report = {
        "stage": "method_y_incremental_value_table",
        "status": "PARTIAL" if any(r["metrics_status"] == "PENDING_TRAINING" for r in rows) else "COMPLETED",
        "paired_tier": "primary_feasible_90",
        "pet_only_reference": metrics.get("Y1_pet_only"),
        "rows": rows,
        "reporting_rules": {
            "primary_metric": "MRI incremental value over PET-only (ΔAUC + DCA net benefit)",
            "forbidden_headline": "PET-only AUC as title/abstract primary finding",
            "sample_scope": "paired_subset_only_not_1700_cohort",
        },
    }
    out_json = PROJECT_ROOT / args.out_json
    out_md = PROJECT_ROOT / args.out_md
    atomic_write_json(out_json, report)
    lines = [
        "# Method Y Incremental Value Table",
        "",
        f"- Status: `{report['status']}`",
        f"- PET-only reference AUC: `{pet_auc}`",
        "",
        "| comparison | ΔAUC vs PET | status |",
        "|---|---:|---|",
    ]
    for row in rows:
        lines.append(f"| {row['label']} | {row['delta_auc_vs_pet_only']} | {row['metrics_status']} |")
    atomic_write_text(out_md, "\n".join(lines) + "\n")
    print(json.dumps({"status": report["status"], "rows": len(rows)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
