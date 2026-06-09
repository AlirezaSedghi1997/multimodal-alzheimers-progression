#!/usr/bin/env python3
"""Document Method Y three-way DCA workflow (MRI vs PET vs fusion) — CPU design artifact."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.io_helpers import atomic_write_json, atomic_write_text


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_json", default="results/tables/method_y_dca_workflow_summary.json")
    ap.add_argument("--out_md", default="results/tables/method_y_dca_workflow_summary.md")
    args = ap.parse_args()

    tabular_dca = PROJECT_ROOT / "results/tables/decision_curve_analysis.json"
    tabular = json.loads(tabular_dca.read_text(encoding="utf-8")) if tabular_dca.exists() else {}

    report = {
        "stage": "method_y_dca_workflow_summary",
        "status": "PARTIAL",
        "method_x_tabular_dca_present": bool(tabular),
        "method_y_three_way_plan": {
            "branches": ["Y2_mri_paired", "Y1_pet_only", "Y3_late_fusion"],
            "metric": "net_benefit across threshold probabilities",
            "sample_scope": "primary_feasible_90 paired subset only",
            "blocked_until": ["Y1_metrics", "Y2_metrics", "Y3_metrics"],
        },
        "execution_script": "scripts/run_decision_curve_analysis.py (extend for Method Y paired predictions)",
        "scientific_framing": "Exploratory clinical utility on paired subset; not Method X headline.",
    }
    atomic_write_json(PROJECT_ROOT / args.out_json, report)
    atomic_write_text(
        PROJECT_ROOT / args.out_md,
        "# Method Y DCA Workflow Summary\n\n- Status: `PARTIAL`\n- Requires Y1–Y3 probability exports on paired subset.\n",
    )
    print(json.dumps({"status": report["status"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
