from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.io_helpers import atomic_write_json, atomic_write_text
from src.utils.state_manager import StateManager


SITE_LIKE_TERMS = ("site", "scanner", "manufacturer", "field", "center", "protocol", "vendor", "magnet", "model")


def load_json(path: str) -> dict[str, Any] | None:
    p = Path(path)
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def exists(path: str) -> bool:
    return Path(path).exists()


def master_columns() -> list[str]:
    p = Path("data/processed/master_catalog.csv")
    if not p.exists():
        return []
    header = p.open("r", encoding="utf-8").readline().strip()
    return [c.strip() for c in header.split(",")]


def has_site_scanner_metadata() -> bool:
    cols = [c.lower() for c in master_columns()]
    return any(any(term in col for term in SITE_LIKE_TERMS) for col in cols)


def add(rows: list[dict[str, str]], requirement: str, status: str, artifact: str, note: str) -> None:
    rows.append({"requirement": requirement, "status": status, "artifact": artifact, "note": note})


def build_rows() -> list[dict[str, str]]:
    final = load_json("results/final_results/final_analysis/final_analysis_summary.json") or {}
    three_d = load_json("results/final_results/three_d/three_d_metrics.json") or {}
    three_d_ablation = load_json("results/final_results/three_d_ablation/three_d_ablation_summary.json") or {}
    tau_reg = load_json("results/final_results/tau/tau_regression_metrics.json") or {}
    full_preprocess = load_json("results/tables/full_mri_preprocess_toolchain_report.json") or {}
    xai = load_json("results/final_results/xai/xai_summary.json") or {}
    rows: list[dict[str, str]] = []
    add(rows, "PET is never a model input", "DONE" if final.get("pet_as_model_input") is False else "MISSING", "final_analysis_summary.json", "PET remains label/target only.")
    add(rows, "Allowed tabular schema is fixed", "DONE" if set(master_columns()).issuperset({"age", "sex", "education_years", "apoe4_carrier", "mmse_base"}) else "MISSING", "data/processed/master_catalog.csv", "Only aggregate column-level check; no raw rows exported.")
    add(rows, "Raw IDs/PII excluded from artifacts", "DONE" if final.get("raw_identifiers_used") is False else "MISSING", "public_bundle/PUBLIC_BUNDLE_MANIFEST.json", "Public bundle manifest reports no raw identifiers.")
    add(rows, "Patient-level split locking", "DONE" if exists("splits/split_summary.json") else "MISSING", "splits/split_summary.json", "External holdout remains locked pseudo-external.")
    add(rows, "Baseline tabular and MRI-feature models", "DONE" if exists("results/final_results/baseline/baseline_metrics.json") and exists("results/final_results/mri_feature_baseline/mri_feature_baseline_metrics.json") else "MISSING", "results/final_results/", "Fast-track baselines generated.")
    add(
        rows,
        "Fusion/ablation variants",
        "DONE" if three_d_ablation.get("status") == "COMPLETED" else "PARTIAL",
        "results/final_results/three_d_ablation/three_d_ablation_summary.json",
        "Feature/surrogate fusion exists; DONE requires real 3D concat/cross-attention/alignment ablation.",
    )
    three_d_status = "PARTIAL"
    if three_d.get("aggregate", {}).get("mode") == "FULL_3D_TRAINING":
        agg = three_d.get("aggregate", {})
        three_d_status = "DONE" if agg.get("folds") == 5 and agg.get("cache_resolution") == 96 else "PARTIAL"
    ablation_best = three_d_ablation.get("best_by_cv_roc_auc") or {}
    if ablation_best.get("folds") == 5 and ablation_best.get("cache_resolution") == 96:
        three_d_status = "DONE"
    add(rows, "Real 3D MRI training", three_d_status, "results/final_results/three_d/three_d_metrics.json", "CUDA 64/3-fold exists; final 96/5-fold remains pending.")
    add(
        rows,
        "Deep MRI preprocessing",
        "DONE" if full_preprocess.get("status") == "COMPLETED" else "MISSING",
        "results/tables/full_mri_preprocess_toolchain_report.json",
        "Specialized toolchain-gated stage is required before final 96 training.",
    )
    power = Path("results/tables/power_analysis.json")
    add(rows, "Power analysis with actual N", "DONE" if power.exists() else "MISSING", "results/tables/power_analysis.json", "Must use catalog/split N, class balance, CIs, and caveats.")
    add(rows, "Site/scanner drift sensitivity", "MISSING" if has_site_scanner_metadata() else "SKIP_WITH_JUSTIFICATION", "results/tables/site_scanner_drift_sensitivity_report.json", "No site/scanner metadata columns are present in the locked master catalog.")
    add(rows, "Subgroup performance audit", "MISSING" if not exists("results/final_results/subgroup_performance_audit.json") else "DONE", "results/final_results/subgroup_performance_audit.json", "Required for sex, age bins, APOE4, education, and MMSE with CIs.")
    add(
        rows,
        "XAI / interpretability",
        "DONE" if xai.get("status") == "COMPLETED" else "MISSING",
        "results/final_results/xai/xai_summary.json",
        "Grad-CAM MRI and SHAP tabular must be aggregate-only in public outputs.",
    )
    tau_status = "DONE" if tau_reg.get("mode") == "FULL" else "PARTIAL" if tau_reg else "MISSING"
    tau_note = (
        "Tau regression executed with grouped CV; current result is exploratory and negative versus the grouped mean baseline."
        if tau_status == "DONE"
        else "Tau feasibility is FULL but current regression artifact is skipped; rerun is required when DATA_ROOT resolves."
    )
    add(rows, "Tau exploratory branch", tau_status, "results/final_results/tau/tau_regression_metrics.json", tau_note)
    add(rows, "Public reproducibility bundle", "DONE" if (load_json("public_bundle/PUBLIC_BUNDLE_MANIFEST.json") or {}).get("violations") == [] else "MISSING", "public_bundle/PUBLIC_BUNDLE_MANIFEST.json", "Aggregate text only; no raw data/HDF5/checkpoints/images.")
    add(rows, "Thesis-grade results report", "MISSING", "results/final_results/thesis/", "Write after validation refresh, not from current fast-track-only package.")
    return rows


def main() -> int:
    rows = build_rows()
    summary = {status: sum(1 for row in rows if row["status"] == status) for status in ["DONE", "PARTIAL", "MISSING", "SKIP_WITH_JUSTIFICATION"]}
    out = {"schema_version": 1, "summary": summary, "rows": rows}
    atomic_write_json("results/tables/implementation_gap_matrix.json", out)
    lines = ["# Implementation Gap Matrix\n\n", "| requirement | status | artifact | note |\n", "|---|---|---|---|\n"]
    for row in rows:
        lines.append(f"| {row['requirement']} | {row['status']} | `{row['artifact']}` | {row['note']} |\n")
    atomic_write_text("results/tables/implementation_gap_matrix.md", "".join(lines))
    StateManager().append_event("gap_matrix", "COMPLETED", "Formal implementation gap matrix generated.")
    print(json.dumps({"gap_matrix": "COMPLETED", "summary": summary}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
