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


def load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    ap = argparse.ArgumentParser(description="Build Phase 1 incremental-value table from branch baselines.")
    ap.add_argument("--baselines", default="results/final_results/phase1_baselines/phase1_baselines_summary.json")
    ap.add_argument("--three_d", default="results/final_results/three_d/three_d_metrics.json")
    ap.add_argument("--out_json", default="results/tables/phase1_incremental_value_table.json")
    ap.add_argument("--out_md", default="results/tables/phase1_incremental_value_table.md")
    args = ap.parse_args()

    baselines = load_json(Path(args.baselines)) or {}
    three_d = load_json(Path(args.three_d)) or {}
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
            }
        )

    report = {
        "schema_version": 1,
        "status": "COMPLETED" if rows else "MISSING_BASELINES",
        "incremental_order": INCREMENTAL_ORDER,
        "rows": rows,
        "caveat": "A4/LEARN enrolled catalog has strong cohort-label coupling; pooled binary AUC is not biological proof.",
    }
    atomic_write_json(args.out_json, report)
    lines = [
        "# Phase 1 Incremental Value Table\n\n",
        f"- status: `{report['status']}`\n\n",
        "| step | roc_auc | pr_auc | brier | n_features |\n",
        "|---|---:|---:|---:|---|\n",
    ]
    for row in rows:
        lines.append(
            f"| {row['step']} | {row.get('roc_auc', 'NA')} | {row.get('pr_auc', 'NA')} | {row.get('brier', 'NA')} | {row.get('n_features', 'NA')} |\n"
        )
    lines.append("\n## Caveat\n\nA4/LEARN enrolled catalog has strong cohort-label coupling; interpret increments as descriptive only.\n")
    atomic_write_text(args.out_md, "".join(lines))
    write_run_manifest("results/logs/phase1_incremental_value_manifest.json", stage="phase1_incremental_value", data_root=None, extra={"rows": len(rows)})
    StateManager().append_event("phase1_incremental_value", report["status"], "Phase 1 incremental-value table generated.")
    print(json.dumps({"status": report["status"], "rows": len(rows)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
