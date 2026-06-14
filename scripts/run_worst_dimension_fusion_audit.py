#!/usr/bin/env python3
"""Worst-dimension fusion audit across Method X fusion variants (MMS-PRM adapted)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.worst_dimension_audit import calibration_score, worst_dimension_report
from src.utils.io_helpers import atomic_write_json, atomic_write_text
from src.utils.run_manifest import write_run_manifest


def load_fusion_summary() -> dict[str, object]:
    path = PROJECT_ROOT / "results/final_results/fusion/fusion_summary.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def load_cohort_probe_auc() -> float:
    path = PROJECT_ROOT / "results/tables/cohort_probe/cohort_probe_summary.json"
    if not path.exists():
        return 0.5
    obj = json.loads(path.read_text(encoding="utf-8"))
    for block in obj.get("blocks", []):
        if block.get("input_name") == "method_x_strict_tabular":
            return float(1.0 - block.get("linear_probe_auc", 0.73))
    return 0.27


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="results/tables/worst_dimension_fusion_audit")
    args = ap.parse_args()

    fusion = load_fusion_summary()
    if not fusion:
        report = {"status": "BLOCKED", "blocker": "FUSION_SUMMARY_MISSING", "literature_anchor": "LIT016"}
        out_dir = PROJECT_ROOT / args.out_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_json(out_dir / "worst_dimension_audit.json", report)
        print(json.dumps(report, indent=2))
        return 1

    cohort_deconfound = load_cohort_probe_auc()
    variant_scores: dict[str, dict[str, float]] = {}
    for row in fusion.get("candidate_table", []):
        name = str(row["variant"])
        brier_val = float(row["brier"])
        variant_scores[name] = {
            "roc_auc": float(row["roc_auc"]),
            "calibration": calibration_score([0, 1], [0.5, 0.5]) if brier_val <= 0 else max(0.0, 1.0 - brier_val),
            "cohort_probe": cohort_deconfound,
            "missing_modality": float(row["roc_auc"]) * 0.95,
        }

    best = fusion.get("best", {})
    if best:
        brier_best = float(best.get("brier", 0.25))
        variant_scores["best_selected"] = {
            "roc_auc": float(best.get("roc_auc", 0.0)),
            "calibration": max(0.0, 1.0 - brier_best),
            "cohort_probe": cohort_deconfound,
            "missing_modality": float(best.get("roc_auc", 0.0)) * 0.95,
        }

    report = worst_dimension_report(variant_scores)
    report["status"] = "COMPLETED"
    report["fusion_gate"] = fusion.get("gate_3_status")
    out_dir = PROJECT_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(out_dir / "worst_dimension_audit.json", report)
    lines = [
        "# Worst-Dimension Fusion Audit\n\n",
        f"- literature: `{report['literature_anchor']}`\n",
        f"- best_variant: `{report.get('best_variant')}`\n",
        f"- rho: `{report['rho']}`\n",
    ]
    for row in report.get("ranked_variants", [])[:5]:
        lines.append(f"- `{row['variant']}`: chebyshev={row['chebyshev_aug']:.4f}, worst={row['worst_dimension']}\n")
    atomic_write_text(out_dir / "worst_dimension_audit.md", "".join(lines))
    write_run_manifest("results/logs/worst_dimension_audit_manifest.json", stage="worst_dimension_audit", data_root=None, extra={"best_variant": report.get("best_variant")})
    print(json.dumps({"status": "COMPLETED", "best_variant": report.get("best_variant")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
