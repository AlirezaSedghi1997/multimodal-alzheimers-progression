#!/usr/bin/env python3
"""Build Method Y incremental-value table from a single metric source index."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.io_helpers import atomic_write_json, atomic_write_text


METHOD_Y_METRIC_SOURCES = {
    "Y1_pet_only": {
        "label": "PET-only baseline",
        "path": "results/final_results/method_y/pet_training/method_y_pet_metrics.json",
        "comparison": "reference",
        "requires_common_hash_intersection": True,
    },
    "Y2_mri_paired": {
        "label": "MRI-only on PET-cache hash intersection",
        "path": "results/final_results/method_y/mri_paired_training/method_y_mri_paired_metrics.json",
        "comparison": "mri_incremental_reference",
        "requires_common_hash_intersection": True,
    },
    "Y3_late_fusion": {
        "label": "Late fusion MRI+PET",
        "path": "results/final_results/method_y/late_fusion_training/method_y_late_fusion_metrics.json",
        "comparison": "fusion",
        "requires_common_hash_intersection": True,
    },
    "Y4_gated_fusion": {
        "label": "Gated fusion MRI+PET",
        "path": "results/final_results/method_y/gated_fusion_training/method_y_gated_fusion_metrics.json",
        "comparison": "fusion",
        "requires_common_hash_intersection": True,
    },
    "Y5_mri_pet_cognitive": {
        "label": "MRI+PET+cognitive summaries",
        "path": "results/final_results/method_y/mri_pet_cognitive_training/method_y_mri_pet_cognitive_metrics.json",
        "comparison": "controlled_extension",
        "requires_common_hash_intersection": True,
    },
}


def _load_metrics(path: Path) -> dict | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    agg = payload.get("aggregate") or payload
    if not isinstance(agg, dict):
        return None
    threshold = agg.get("threshold_selection") or {}
    threshold_folds = threshold.get("folds") or []
    threshold_means: dict[str, float | None] = {}
    for metric in ["accuracy", "sensitivity", "specificity", "f1"]:
        values = [
            float(row.get("test_metrics", {}).get(metric))
            for row in threshold_folds
            if isinstance(row, dict) and row.get("test_metrics", {}).get(metric) is not None
        ]
        threshold_means[metric] = sum(values) / len(values) if values else None
    return {
        "roc_auc": agg.get("roc_auc"),
        "pr_auc": agg.get("pr_auc"),
        "brier": agg.get("brier"),
        "n": agg.get("n"),
        "scenario": agg.get("scenario") or agg.get("name"),
        "status": agg.get("status") or payload.get("status"),
        "threshold_metrics": threshold_means,
    }


def build_metric_index(root: Path = PROJECT_ROOT) -> dict[str, dict | None]:
    return {key: _load_metrics(root / spec["path"]) for key, spec in METHOD_Y_METRIC_SOURCES.items()}


def _common_effective_n(metrics: dict[str, dict | None]) -> int | None:
    completed_ns = [
        int(metric["n"])
        for key, metric in metrics.items()
        if metric and METHOD_Y_METRIC_SOURCES[key]["requires_common_hash_intersection"] and metric.get("n") is not None
    ]
    return min(completed_ns) if completed_ns else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_json", default="results/tables/method_y_incremental_value_table.json")
    ap.add_argument("--out_md", default="results/tables/method_y_incremental_value_table.md")
    args = ap.parse_args()

    metrics = build_metric_index(PROJECT_ROOT)
    pet = metrics.get("Y1_pet_only") or {}
    pet_auc = pet.get("roc_auc")
    pet_n = pet.get("n")
    matched_effective_n = _common_effective_n(metrics)
    rows = []
    for comp_id in ["Y2_mri_paired", "Y3_late_fusion", "Y4_gated_fusion", "Y5_mri_pet_cognitive"]:
        spec = METHOD_Y_METRIC_SOURCES[comp_id]
        m = metrics.get(comp_id)
        row_n = (m or {}).get("n")
        delta_valid = bool(
            m
            and pet_auc is not None
            and row_n == matched_effective_n
            and pet_n == matched_effective_n
        )
        delta = None
        if delta_valid and m and m.get("roc_auc") is not None:
            delta = float(m["roc_auc"]) - float(pet_auc)
        rows.append(
            {
                "comparison_id": comp_id,
                "label": spec["label"],
                "metrics_status": "AVAILABLE" if m else "PENDING_TRAINING",
                "roc_auc": (m or {}).get("roc_auc"),
                "pr_auc": (m or {}).get("pr_auc"),
                "brier": (m or {}).get("brier"),
                "threshold_metrics": (m or {}).get("threshold_metrics"),
                "effective_n": row_n,
                "required_common_hash_intersection_n": matched_effective_n,
                "delta_auc_valid": delta_valid,
                "delta_auc_vs_pet_only": delta,
                "headline_allowed": False,
                "caveat": (
                    "Method Y paired PET-cache hash intersection only; PET-containing rows have PET-label "
                    "tautology caveat and small-N/class-imbalance limitations."
                ),
            }
        )

    comparability_ok = all(row["metrics_status"] != "AVAILABLE" or row["delta_auc_valid"] for row in rows)
    report = {
        "stage": "method_y_incremental_value_table",
        "status": "PARTIAL" if any(r["metrics_status"] == "PENDING_TRAINING" for r in rows) else "COMPLETED",
        "paired_tier": "primary_feasible_90",
        "pet_only_reference": pet or None,
        "matched_effective_n": matched_effective_n,
        "source_of_truth": {
            "metric_index": METHOD_Y_METRIC_SOURCES,
            "hash_intersection_policy": (
                "Compute deltas only when PET reference and comparison aggregate.n match the common "
                "effective PET-cache hash intersection."
            ),
        },
        "comparability_ok": comparability_ok,
        "comparability_issue": None
        if comparability_ok
        else "One or more available rows were not generated on the common PET-cache hash intersection; delta_auc_valid=false for those rows.",
        "rows": rows,
        "reporting_rules": {
            "primary_metric": "MRI incremental value over PET-only (delta AUC + DCA net benefit)",
            "forbidden_headline": "PET-only AUC as title/abstract primary finding",
            "sample_scope": "matched_pet_cache_intersection_not_full_1700_cohort",
            "threshold_metrics": "Accuracy, sensitivity, specificity, and F1 use fold-internal validation-selected thresholds only.",
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
        f"- matched effective N: `{matched_effective_n}`",
        f"- comparability ok: `{comparability_ok}`",
        "",
        "| comparison | n | AUC | delta AUC vs PET | accuracy | sensitivity | specificity | F1 | status | delta valid |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|---:|",
    ]
    for row in rows:
        tm = row.get("threshold_metrics") or {}
        lines.append(
            f"| {row['label']} | {row.get('effective_n')} | {row.get('roc_auc')} | {row['delta_auc_vs_pet_only']} | "
            f"{tm.get('accuracy')} | {tm.get('sensitivity')} | {tm.get('specificity')} | {tm.get('f1')} | "
            f"{row['metrics_status']} | {row['delta_auc_valid']} |"
        )
    atomic_write_text(out_md, "\n".join(lines) + "\n")
    print(json.dumps({"status": report["status"], "rows": len(rows), "comparability_ok": comparability_ok}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
