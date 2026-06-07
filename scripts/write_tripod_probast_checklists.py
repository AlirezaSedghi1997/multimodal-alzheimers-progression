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
        {"item": "Title identifies prediction model development/validation", "status": "PENDING_MANUSCRIPT"},
        {"item": "Abstract includes biomarker-status wording (not AD diagnosis)", "status": "PENDING_MANUSCRIPT"},
        {"item": "Background states cohort-label coupling limitation", "status": "DONE" if exists("results/governance/cohort_label_coupling_audit.json") else "MISSING"},
        {"item": "Methods describe locked pseudo-external evaluation", "status": "DONE" if exists("splits/split_summary.json") else "MISSING"},
        {"item": "Methods describe patient-level grouped CV", "status": "DONE" if phase2.get("status") == "MVP_COMPLETE" else "PARTIAL"},
        {"item": "Calibration reported (Brier/ECE)", "status": "DONE" if exists("results/final_results/calibration/calibration_report.json") else "MISSING"},
        {"item": "Cohort-probe validity metric reported", "status": "DONE" if exists("results/tables/cohort_probe/cohort_probe_summary.json") else "MISSING"},
        {"item": "Claim policy documented", "status": "DONE" if exists("docs/final_execution_program.md") else "MISSING"},
        {"item": "Aggregate-only reproducibility bundle", "status": "DONE" if exists("public_bundle/PUBLIC_BUNDLE_MANIFEST.json") else "MISSING"},
    ]
    probast = [
        {"item": "Participants: hash-level catalog with inclusion flow", "status": "DONE" if exists("data/processed/catalogs/branch_catalog_manifest.json") else "MISSING"},
        {"item": "Predictors: Method X allowlist enforced", "status": "DONE" if exists("results/governance/method_x_feature_matrix_pet_leakage_check.json") else "MISSING"},
        {"item": "Outcome: frozen centiloid threshold label", "status": "DONE" if exists("results/governance/label_freeze.json") else "MISSING"},
        {"item": "Analysis: label-shuffle and permutation controls", "status": "DONE" if exists("results/tables/validity_audit/validity_audit_summary.json") else "MISSING"},
        {"item": "Risk of bias: cohort coupling documented", "status": "DONE" if exists("results/tables/cross_cohort_bias_report.json") else "MISSING"},
        {"item": "Applicability: pseudo-external limitation stated", "status": "DONE" if exists("results/final_results/final_analysis/final_analysis_report.md") else "PARTIAL"},
    ]
    done_t = sum(1 for x in tripod if x["status"] == "DONE")
    done_p = sum(1 for x in probast if x["status"] == "DONE")
    report = {
        "stage": "tripod_probast_checklists",
        "status": "SCaffold_COMPLETE",
        "tripod_ai": {"done": done_t, "total": len(tripod), "items": tripod},
        "probast_ai": {"done": done_p, "total": len(probast), "items": probast},
        "gap_matrix_summary": gap.get("summary"),
    }
    atomic_write_json("results/governance/tripod_probast_checklists.json", report)
    md = [
        "# TRIPOD+AI / PROBAST+AI Checklists\n\n",
        f"- TRIPOD+AI done: `{done_t}/{len(tripod)}`\n",
        f"- PROBAST+AI done: `{done_p}/{len(probast)}`\n",
        "- Manuscript-specific items remain PENDING until Phase 4 draft.\n",
    ]
    atomic_write_text("results/governance/tripod_probast_checklists.md", "".join(md))
    write_run_manifest("results/logs/tripod_probast_manifest.json", stage="tripod_probast", data_root=None, extra={"summary": report})
    StateManager().append_event("tripod_probast", "COMPLETED", "TRIPOD/PROBAST checklist scaffold written.")
    print(json.dumps({"tripod_done": done_t, "probast_done": done_p}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
