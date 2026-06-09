#!/usr/bin/env python3
"""Phase 4 required figure manifest — tracks which figures exist vs pending."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.io_helpers import atomic_write_json, atomic_write_text
from src.utils.state_manager import StateManager


def status_for(path: str) -> str:
    return "DONE" if Path(path).exists() else "MISSING"


def main() -> int:
    figures = [
        {"id": "fig_cohort_coupling", "title": "Cohort-label coupling audit", "artifact": "results/governance/cohort_label_coupling_audit.json", "phase": 4},
        {"id": "fig_sf_label_dist", "title": "SF label distribution", "artifact": "results/final_results/method_x_cog_sf/sf_stress_tests_summary.json", "phase": 4},
        {"id": "fig_centiloid_within_cohort", "title": "Continuous centiloid within-cohort", "artifact": "results/final_results/centiloid_regression/centiloid_regression_metrics.json", "phase": 2},
        {"id": "fig_calibration", "title": "Calibration plots", "artifact": "results/final_results/calibration/calibration_report.json", "phase": 2},
        {"id": "fig_dca", "title": "Decision curve analysis", "artifact": "results/tables/decision_curve_analysis.json", "phase": 2},
        {"id": "fig_ablation_incremental", "title": "Ablation / incremental value", "artifact": "results/tables/phase1_incremental_value_table.json", "phase": 1},
        {"id": "fig_cohort_probe", "title": "Cohort-probe validity", "artifact": "results/tables/cohort_probe/cohort_probe_summary.json", "phase": 2},
        {"id": "fig_probe_gap", "title": "Linear vs nonlinear cohort-probe gap", "artifact": "results/tables/cohort_probe/cohort_probe_summary.json", "phase": 4},
        {"id": "fig_temporal_gap", "title": "Temporal-gap headline sensitivity", "artifact": "results/tables/headline_temporal_gap_evaluation.json", "phase": 2},
        {"id": "fig_branch_comparison", "title": "Branch comparison table", "artifact": "results/final_results/final_analysis/final_analysis_summary.json", "phase": 4},
    ]
    for fig in figures:
        fig["status"] = status_for(fig["artifact"])

    done = sum(1 for f in figures if f["status"] == "DONE")
    report = {
        "stage": "phase4_figure_manifest",
        "status": "PARTIAL" if done < len(figures) else "COMPLETE",
        "done": done,
        "total": len(figures),
        "figures": figures,
    }
    atomic_write_json("results/governance/phase4_figure_manifest.json", report)
    lines = ["# Phase 4 Figure Manifest\n\n", f"- done: `{done}/{len(figures)}`\n\n", "| id | title | status | artifact |\n", "|---|---|---|---|\n"]
    for fig in figures:
        lines.append(f"| {fig['id']} | {fig['title']} | {fig['status']} | `{fig['artifact']}` |\n")
    atomic_write_text("results/governance/phase4_figure_manifest.md", "".join(lines))
    StateManager().append_event("phase4_figure_manifest", report["status"], f"Phase 4 figure manifest {done}/{len(figures)} done.")
    print(json.dumps({"done": done, "total": len(figures)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
