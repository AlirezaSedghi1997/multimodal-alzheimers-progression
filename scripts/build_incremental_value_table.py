from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.io_helpers import atomic_write_json, atomic_write_text
from src.utils.run_manifest import write_run_manifest
from src.utils.state_manager import StateManager


INCREMENTAL_ORDER = [
    "fixed_tabular",
    "cognitive_only",
    "fixed_tabular_plus_cognitive",
    "roi_icv_norm_only",
    "wmh_only",
    "mri_reads_only",
    "roi_wmh_reads",
    "fixed_tabular_plus_roi_wmh_reads",
]

ROI_COHORT_ARTIFACT_STEPS = {
    "roi_icv_norm_only",
    "roi_wmh_reads",
    "fixed_tabular_plus_roi_wmh_reads",
}


def load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def cohort_probe_nonlinear_auc(cohort_probe: dict | None, input_name: str = "mri_roi_wmh_reads") -> float | None:
    if not cohort_probe:
        return None
    for block in cohort_probe.get("blocks", []):
        if block.get("input_name") == input_name:
            return block.get("nonlinear_probe_auc")
    return None


def validity_flag(step: str, roi_probe: float | None) -> str:
    if step in ROI_COHORT_ARTIFACT_STEPS and roi_probe is not None and roi_probe >= 0.9:
        return "DESCRIPTIVE_ONLY_COHORT_ARTIFACT"
    if step == "wmh_only":
        return "SINGLE_OFFICIAL_WMH_FEATURE"
    if step == "mri_reads_only":
        return "WEAK_STANDALONE_SIGNAL"
    if step == "method_x_strict_3d_cnn":
        return "VALIDITY_SAFE_LOW_COHORT_PROBE"
    if step in {"fixed_tabular", "cognitive_only", "fixed_tabular_plus_cognitive"}:
        return "HEADLINE_DESCRIPTIVE_WITH_CAVEAT"
    return "DESCRIPTIVE"


def main() -> int:
    ap = argparse.ArgumentParser(description="Build Phase 1 incremental-value table from branch baselines.")
    ap.add_argument("--baselines", default="results/final_results/phase1_baselines/phase1_baselines_summary.json")
    ap.add_argument("--three_d", default="results/final_results/three_d/three_d_metrics.json")
    ap.add_argument("--cohort_probe", default="results/tables/cohort_probe/cohort_probe_summary.json")
    ap.add_argument("--out_json", default="results/tables/phase1_incremental_value_table.json")
    ap.add_argument("--out_md", default="results/tables/phase1_incremental_value_table.md")
    args = ap.parse_args()

    baselines = load_json(Path(args.baselines)) or {}
    three_d = load_json(Path(args.three_d)) or {}
    cohort_probe = load_json(Path(args.cohort_probe)) or {}
    roi_probe = cohort_probe_nonlinear_auc(cohort_probe)

    rows = []
    results = baselines.get("results") or {}
    for name in INCREMENTAL_ORDER:
        agg = results.get(name)
        if not agg:
            continue
        rows.append(
            {
                "step": name,
                "description": agg.get("description"),
                "roc_auc": agg.get("roc_auc"),
                "pr_auc": agg.get("pr_auc"),
                "brier": agg.get("brier"),
                "n_features": agg.get("n_features"),
                "validity_flag": validity_flag(name, roi_probe),
            }
        )
    if three_d.get("aggregate"):
        agg = three_d["aggregate"]
        rows.append(
            {
                "step": "method_x_strict_3d_cnn",
                "description": "Method X-Strict 3D T1 CNN + tabular cross-attention (existing run)",
                "roc_auc": agg.get("roc_auc"),
                "pr_auc": agg.get("pr_auc"),
                "brier": agg.get("brier"),
                "n_features": "3D_MRI_embedding_plus_tabular",
                "final_claim_allowed": agg.get("final_claim_allowed"),
                "validity_flag": validity_flag("method_x_strict_3d_cnn", roi_probe),
            }
        )

    best_descriptive = max(rows, key=lambda r: float(r.get("roc_auc") or 0)) if rows else None
    validity_safe_candidates = [
        r for r in rows if r.get("validity_flag") in {"HEADLINE_DESCRIPTIVE_WITH_CAVEAT", "VALIDITY_SAFE_LOW_COHORT_PROBE"}
    ]
    validity_safe = max(validity_safe_candidates, key=lambda r: float(r.get("roc_auc") or 0)) if validity_safe_candidates else None

    report = {
        "schema_version": 2,
        "status": "COMPLETED" if rows else "MISSING_BASELINES",
        "incremental_order": INCREMENTAL_ORDER,
        "rows": rows,
        "caveat": "A4/LEARN enrolled catalog has strong cohort-label coupling; pooled binary AUC is not biological proof.",
        "cohort_probe_nonlinear_auc_roi_wmh_reads": roi_probe,
        "best_descriptive_roc_auc_step": best_descriptive.get("step") if best_descriptive else None,
        "validity_safe_headline_step": validity_safe.get("step") if validity_safe else "fixed_tabular",
        "ranking_policy": (
            "Do not present highest pooled ROC-AUC (often tabular+ROI) as best biological model when "
            "ROI/WMH nonlinear cohort-probe ≈0.99. Prefer fixed tabular or 3D (cohort-probe ≈0.50) for validity narrative."
        ),
    }
    atomic_write_json(args.out_json, report)
    lines = [
        "# Phase 1 Incremental Value Table\n\n",
        f"- status: `{report['status']}`\n",
        f"- ROI nonlinear cohort-probe AUC: `{roi_probe}`\n",
        f"- best **descriptive** pooled AUC: `{report['best_descriptive_roc_auc_step']}`\n",
        f"- validity-safe headline step: `{report['validity_safe_headline_step']}`\n\n",
        "| step | roc_auc | pr_auc | validity_flag |\n",
        "|---|---:|---:|---|\n",
    ]
    for row in rows:
        lines.append(
            f"| {row['step']} | {row.get('roc_auc', 'NA')} | {row.get('pr_auc', 'NA')} | {row.get('validity_flag')} |\n"
        )
    lines.append("\n## Caveat\n\n")
    lines.append(f"{report['ranking_policy']}\n")
    atomic_write_text(args.out_md, "".join(lines))
    write_run_manifest("results/logs/phase1_incremental_value_manifest.json", stage="phase1_incremental_value", data_root=None, extra={"rows": len(rows)})
    StateManager().append_event("phase1_incremental_value", report["status"], "Phase 1 incremental-value table generated with validity flags.")
    print(json.dumps({"status": report["status"], "rows": len(rows), "roi_probe": roi_probe}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
