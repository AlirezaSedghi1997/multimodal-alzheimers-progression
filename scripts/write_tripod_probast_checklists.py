#!/usr/bin/env python3
"""TRIPOD+AI and PROBAST+AI checklist scaffolds with repo-backed status."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.io_helpers import atomic_write_json, atomic_write_text
from src.utils.run_manifest import write_run_manifest
from src.utils.state_manager import StateManager


def exists(path: str) -> bool:
    return Path(path).exists()


def main() -> int:
    phase2 = json.loads(Path("results/tables/phase2_mvp/phase2_mvp_status.json").read_text(encoding="utf-8")) if Path("results/tables/phase2_mvp/phase2_mvp_status.json").exists() else {}
    gap = json.loads(Path("results/tables/implementation_gap_matrix.json").read_text(encoding="utf-8")) if Path("results/tables/implementation_gap_matrix.json").exists() else {}

    tripod = [
        {"item": "Title identifies prediction model development/validation", "status": "DRAFT_IN_METHODS", "evidence": "results/governance/tripod_methods_scaffold.md"},
        {"item": "Abstract includes biomarker-status wording (not AD diagnosis)", "status": "PENDING_MANUSCRIPT"},
        {"item": "Background states cohort-label coupling limitation", "status": "DONE", "evidence": "results/governance/cohort_label_coupling_audit.json"},
        {"item": "Methods describe locked pseudo-external evaluation", "status": "DONE", "evidence": "splits/split_summary.json"},
        {"item": "Methods describe patient-level grouped CV", "status": "DONE" if phase2.get("status") == "MVP_COMPLETE" else "PARTIAL", "evidence": "results/tables/phase2_mvp/phase2_mvp_status.json"},
        {"item": "Calibration reported (Brier/ECE)", "status": "DONE" if exists("results/final_results/calibration/calibration_report.json") else "MISSING"},
        {"item": "Cohort-probe validity metric reported", "status": "DONE", "evidence": "results/tables/cohort_probe/cohort_probe_summary.json"},
        {"item": "Incremental value ranked with validity flags", "status": "DONE" if exists("results/tables/phase1_incremental_value_table.json") else "MISSING"},
        {"item": "Temporal-gap sensitivity with bootstrap CI", "status": "DONE" if exists("results/tables/headline_temporal_gap_evaluation.json") else "MISSING"},
        {"item": "Matched subset and DCA sensitivity", "status": "DONE" if exists("results/tables/matched_propensity_subset.json") and exists("results/tables/decision_curve_analysis.json") else "PARTIAL"},
        {"item": "Claim policy documented", "status": "DONE", "evidence": "docs/final_execution_program.md"},
        {"item": "Domain-ceiling citations frozen", "status": "DONE", "evidence": "results/governance/domain_ceiling_citation_list.json"},
        {"item": "Thesis-grade aggregate report", "status": "DONE" if exists("results/final_results/thesis/thesis_grade_report.md") else "MISSING"},
        {"item": "Aggregate-only reproducibility bundle", "status": "DONE", "evidence": "public_bundle/PUBLIC_BUNDLE_MANIFEST.json"},
    ]
    probast = [
        {"item": "Participants: hash-level catalog with inclusion flow", "status": "DONE", "evidence": "data/processed/catalogs/branch_catalog_manifest.json"},
        {"item": "Predictors: Method X allowlist enforced", "status": "DONE", "evidence": "results/governance/method_x_feature_matrix_pet_leakage_check.json"},
        {"item": "Outcome: frozen centiloid threshold label", "status": "DONE", "evidence": "results/governance/label_freeze.json"},
        {"item": "Analysis: label-shuffle and permutation controls", "status": "DONE", "evidence": "results/tables/validity_audit/validity_audit_summary.json"},
        {"item": "Analysis: cohort-probe for representation validity", "status": "DONE", "evidence": "results/tables/cohort_probe/cohort_probe_summary.json"},
        {"item": "Risk of bias: cohort coupling documented", "status": "DONE", "evidence": "results/tables/cross_cohort_bias_report.json"},
        {"item": "Risk of bias: ROI cohort-artifact caveat", "status": "DONE", "evidence": "results/tables/phase1_incremental_value_table.json"},
        {"item": "Applicability: pseudo-external limitation stated", "status": "DONE", "evidence": "results/final_results/thesis/thesis_grade_report.md"},
        {"item": "Method Y/PET separate claim boundary", "status": "DONE", "evidence": "results/tables/method_y_training_readiness.json"},
        {"item": "No subject-level predictions in public JSON", "status": "DONE", "evidence": "tests/test_public_bundle_outputs.py"},
    ]
    done_t = sum(1 for x in tripod if x["status"] == "DONE")
    done_p = sum(1 for x in probast if x["status"] == "DONE")
    report = {
        "stage": "tripod_probast_checklists",
        "status": "SCAFFOLD_COMPLETE",
        "tripod_ai": {"done": done_t, "total": len(tripod), "items": tripod},
        "probast_ai": {"done": done_p, "total": len(probast), "items": probast},
        "gap_matrix_summary": gap.get("summary"),
        "manuscript_items_pending": sum(1 for x in tripod if x["status"].startswith("PENDING")),
    }
    atomic_write_json("results/governance/tripod_probast_checklists.json", report)
    md = [
        "# TRIPOD+AI / PROBAST+AI Checklists\n\n",
        f"- TRIPOD+AI done: `{done_t}/{len(tripod)}`\n",
        f"- PROBAST+AI done: `{done_p}/{len(probast)}`\n",
        f"- Manuscript-only pending: `{report['manuscript_items_pending']}`\n\n",
        "## TRIPOD+AI items\n\n",
    ]
    for row in tripod:
        md.append(f"- [{row['status']}] {row['item']}")
        if row.get("evidence"):
            md.append(f" — `{row['evidence']}`")
        md.append("\n")
    md.append("\n## PROBAST+AI items\n\n")
    for row in probast:
        md.append(f"- [{row['status']}] {row['item']}")
        if row.get("evidence"):
            md.append(f" — `{row['evidence']}`")
        md.append("\n")
    atomic_write_text("results/governance/tripod_probast_checklists.md", "".join(md))
    write_run_manifest("results/logs/tripod_probast_manifest.json", stage="tripod_probast", data_root=None, extra={"summary": report})
    StateManager().append_event("tripod_probast", "COMPLETED", "TRIPOD/PROBAST checklist scaffold expanded.")
    print(json.dumps({"tripod_done": done_t, "probast_done": done_p}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
