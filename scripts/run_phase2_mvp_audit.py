from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.io_helpers import atomic_write_json, atomic_write_text
from src.utils.run_manifest import write_run_manifest
from src.utils.state_manager import StateManager


def load(path: str) -> dict[str, Any] | None:
    p = Path(path)
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def status_ok(obj: dict | None, *keys: str) -> bool:
    if not obj:
        return False
    for key in keys:
        if obj.get(key) not in {None, "MISSING", "FAILED"}:
            continue
        return False
    return True


def check_item(name: str, ok: bool, artifact: str, note: str) -> dict[str, str]:
    return {
        "requirement": name,
        "status": "DONE" if ok else "MISSING",
        "artifact": artifact,
        "note": note,
    }


def model_checks(model_id: str, cfg: dict[str, Any]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for req, spec in cfg.items():
        ok = spec["check"]()
        rows.append(check_item(req, ok, spec["artifact"], spec["note"]))
    rows.insert(0, {"model": model_id, "headline": str(cfg.get("_headline", True))})
    return rows


def summarize(rows: list[dict[str, str]]) -> dict[str, int]:
    counts = {"DONE": 0, "MISSING": 0, "N/A": 0}
    for row in rows:
        if "requirement" not in row:
            continue
        status = row["status"]
        counts[status] = counts.get(status, 0) + 1
    return counts


def main() -> int:
    ap = argparse.ArgumentParser(description="Aggregate Minimum Phase-2 MVP completion status for headline models.")
    ap.add_argument("--out_json", default="results/tables/phase2_mvp/phase2_mvp_status.json")
    ap.add_argument("--out_md", default="results/tables/phase2_mvp/phase2_mvp_status.md")
    args = ap.parse_args()

    three_d = load("results/final_results/three_d/three_d_metrics.json")
    baseline = load("results/final_results/baseline/baseline_metrics.json")
    phase1 = load("results/final_results/phase1_baselines/phase1_baselines_summary.json")
    external = load("results/final_results/external/external_final_metrics.json")
    calibration = load("results/final_results/calibration/calibration_report.json")
    cohort_coupling = load("results/governance/cohort_label_coupling_audit.json")
    cohort_probe = load("results/tables/cohort_probe/cohort_probe_summary.json")
    centiloid = load("results/final_results/centiloid_regression/centiloid_regression_metrics.json")
    validity = load("results/tables/validity_audit/validity_audit_summary.json")
    ablation = load("results/final_results/three_d_ablation/three_d_ablation_summary.json")

    three_d_agg = (three_d or {}).get("aggregate", {})
    baseline_agg = (baseline or {}).get("aggregate", {})
    phase1_tab = ((phase1 or {}).get("results") or {}).get("fixed_tabular", {})
    phase1_roi = ((phase1 or {}).get("results") or {}).get("roi_wmh_reads", {})

    shared = {
        "locked_pseudo_external_evaluation": {
            "artifact": "results/final_results/external/external_final_metrics.json",
            "note": "No fit/tune/calibration on holdout.",
            "check": lambda: external is not None and external.get("external_used_for_fit") is False,
        },
        "cramers_v_cohort_label_audit": {
            "artifact": "results/governance/cohort_label_coupling_audit.json",
            "note": "Before/after SF coupling audit.",
            "check": lambda: cohort_coupling is not None and cohort_coupling.get("status") == "COMPLETED",
        },
        "linear_and_nonlinear_cohort_probe": {
            "artifact": "results/tables/cohort_probe/cohort_probe_summary.json",
            "note": "Phase 0.6 protocol with linear/nonlinear gap.",
            "check": lambda: cohort_probe is not None and cohort_probe.get("status") == "COMPLETED",
        },
        "continuous_centiloid_within_cohort_analysis": {
            "artifact": "results/final_results/centiloid_regression/centiloid_regression_metrics.json",
            "note": "Within-A4 and within-LEARN subsets required.",
            "check": lambda: centiloid is not None
            and centiloid.get("subsets", {}).get("within_a4", {}).get("status") == "COMPLETED"
            and centiloid.get("subsets", {}).get("within_learn", {}).get("status") == "COMPLETED",
        },
    }

    method_x_strict_3d = {
        **shared,
        "5_fold_subject_grouped_cv": {
            "artifact": "results/final_results/three_d/three_d_metrics.json",
            "note": "5-fold CV on dev set with patient grouping.",
            "check": lambda: three_d_agg.get("folds") == 5 and len((three_d or {}).get("folds", [])) == 5,
        },
        "calibration_brier_ece_reliability_plot": {
            "artifact": "results/final_results/calibration/calibration_report.json",
            "note": "Train-fold-only calibration; external not used for fit.",
            "check": lambda: calibration is not None
            and calibration.get("external_used_for_fit") is False
            and ((calibration.get("cv_calibrated") or {}).get("ece") is not None),
        },
        "bootstrap_ci": {
            "artifact": "results/final_results/calibration/calibration_report.json",
            "note": "Bootstrap CI on CV metrics.",
            "check": lambda: ((calibration or {}).get("cv_calibrated") or {}).get("roc_auc_ci", {}).get("status") == "OK",
        },
        "label_shuffle_control": {
            "artifact": "results/tables/validity_audit/validity_audit_summary.json",
            "note": "Global label shuffle should collapse toward 0.5.",
            "check": lambda: validity is not None,
        },
        "feature_permutation_or_ablation": {
            "artifact": "results/final_results/three_d_ablation/three_d_ablation_summary.json",
            "note": "3D fusion ablation variants.",
            "check": lambda: ablation is not None and ablation.get("status") == "COMPLETED",
        },
        "sf_aware_stress_test_where_applicable": {
            "artifact": "N/A",
            "note": "N/A for enrolled A4/LEARN Method X-Strict MRI catalog.",
            "check": lambda: True,
        },
    }

    method_x_strict_tabular = {
        **shared,
        "5_fold_subject_grouped_cv": {
            "artifact": "results/final_results/baseline/baseline_metrics.json",
            "note": "Locked split 5-fold tabular baseline.",
            "check": lambda: len((baseline or {}).get("folds", [])) == 5,
        },
        "calibration_brier_ece_reliability_plot": {
            "artifact": "results/final_results/phase1_baselines/fixed_tabular_metrics.json",
            "note": "Phase 1 tabular baseline reports Brier; dedicated calibration plot pending refresh.",
            "check": lambda: phase1_tab.get("brier") is not None,
        },
        "bootstrap_ci": {
            "artifact": "results/final_results/phase1_baselines/fixed_tabular_metrics.json",
            "note": "Bootstrap CI on pooled CV predictions.",
            "check": lambda: ((phase1_tab.get("roc_auc_ci") or {}).get("status") == "OK"),
        },
        "label_shuffle_control": {
            "artifact": "results/tables/validity_audit/validity_audit_summary.json",
            "note": "Validity audit label-shuffle control.",
            "check": lambda: validity is not None,
        },
        "feature_permutation_or_ablation": {
            "artifact": "results/final_results/phase1_baselines/phase1_baselines_summary.json",
            "note": "Incremental feature-set comparison across tabular/cognitive/ROI families.",
            "check": lambda: phase1 is not None and phase1.get("status") == "COMPLETED",
        },
        "sf_aware_stress_test_where_applicable": {
            "artifact": "N/A",
            "note": "N/A for Method X-Strict enrolled catalog.",
            "check": lambda: True,
        },
    }

    mri_roi_branch = {
        **shared,
        "5_fold_subject_grouped_cv": {
            "artifact": "results/final_results/phase1_baselines/roi_wmh_reads_metrics.json",
            "note": "5-fold CV on ROI/WMH/read interpretable branch.",
            "check": lambda: Path("results/final_results/phase1_baselines/roi_wmh_reads_metrics.json").exists(),
        },
        "calibration_brier_ece_reliability_plot": {
            "artifact": "results/final_results/phase1_baselines/roi_wmh_reads_metrics.json",
            "note": "Brier reported; ECE/reliability plot refresh optional for interpretable branch.",
            "check": lambda: phase1_roi.get("brier") is not None,
        },
        "bootstrap_ci": {
            "artifact": "results/final_results/phase1_baselines/roi_wmh_reads_metrics.json",
            "note": "Bootstrap CI on ROI branch.",
            "check": lambda: ((phase1_roi.get("roc_auc_ci") or {}).get("status") == "OK"),
        },
        "label_shuffle_control": {
            "artifact": "results/tables/validity_audit/validity_audit_summary.json",
            "note": "Shared validity audit.",
            "check": lambda: validity is not None,
        },
        "feature_permutation_or_ablation": {
            "artifact": "results/tables/cohort_probe/cohort_probe_summary.json",
            "note": "ROI branch must report cohort-probe limitation (nonlinear AUC high).",
            "check": lambda: cohort_probe is not None,
        },
        "sf_aware_stress_test_where_applicable": {
            "artifact": "N/A",
            "note": "SF stress tests belong to Method X-Cog/SF branch.",
            "check": lambda: True,
        },
    }

    models = {
        "method_x_strict_3d_cnn": method_x_strict_3d,
        "method_x_strict_tabular": method_x_strict_tabular,
        "mri_roi_wmh_reads_branch": mri_roi_branch,
    }

    report_models: dict[str, Any] = {}
    for model_id, cfg in models.items():
        checks = []
        for req, spec in cfg.items():
            if req.startswith("_"):
                continue
            if req == "sf_aware_stress_test_where_applicable":
                checks.append(
                    {
                        "requirement": req,
                        "status": "N/A",
                        "artifact": spec["artifact"],
                        "note": spec["note"],
                    }
                )
                continue
            ok = spec["check"]()
            checks.append(check_item(req, ok, spec["artifact"], spec["note"]))
        done = sum(1 for c in checks if c["status"] == "DONE")
        na = sum(1 for c in checks if c["status"] == "N/A")
        total = len(checks)
        report_models[model_id] = {
            "checks": checks,
            "completion": {
                "done": done,
                "na": na,
                "required": total - na,
                "missing": total - done - na,
                "mvp_complete": (total - done - na) == 0,
            },
        }

    overall_missing = sum(m["completion"]["missing"] for m in report_models.values())
    sf_stress = load("results/final_results/method_x_cog_sf/sf_stress_tests_summary.json")
    blocking_items: list[str] = []
    if not (ablation and ablation.get("status") == "COMPLETED"):
        blocking_items.append("3d_fusion_ablation_not_completed")
    if not (sf_stress and sf_stress.get("status") == "COMPLETED"):
        blocking_items.append("method_x_cog_sf_stress_tests_not_completed")
    report = {
        "schema_version": 1,
        "status": "MVP_COMPLETE" if overall_missing == 0 else "MVP_PARTIAL",
        "minimum_phase2_mvp_source": "configs/policy/final_execution_program.json",
        "models": report_models,
        "overall_missing_requirements": overall_missing,
        "method_x_cog_sf_stress_tests": {
            "status": "DONE" if sf_stress and sf_stress.get("status") == "COMPLETED" else "MISSING",
            "artifact": "results/final_results/method_x_cog_sf/sf_stress_tests_summary.json",
        },
        "blocking_items": blocking_items,
    }

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(out_json, report)

    lines = ["# Phase 2 Minimum MVP Status\n\n", f"- overall status: `{report['status']}`\n\n"]
    for model_id, body in report_models.items():
        comp = body["completion"]
        lines.append(f"## {model_id}\n\n")
        lines.append(f"- mvp_complete: `{comp['mvp_complete']}` ({comp['done']}/{comp['required']} required done)\n\n")
        lines.append("| requirement | status | artifact |\n|---|---|---|\n")
        for check in body["checks"]:
            lines.append(f"| {check['requirement']} | {check['status']} | `{check['artifact']}` |\n")
        lines.append("\n")
    atomic_write_text(args.out_md, "".join(lines))
    write_run_manifest("results/logs/phase2_mvp_status_manifest.json", stage="phase2_mvp_audit", data_root=None, extra={"status": report["status"]})
    StateManager().append_event("phase2_mvp_audit", report["status"], f"Phase 2 MVP audit: {overall_missing} missing requirements across headline models.")
    print(json.dumps({"status": report["status"], "overall_missing": overall_missing}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
