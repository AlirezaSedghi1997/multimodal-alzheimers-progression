from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src.evaluation.cohort_probe import (
    cohort_label_from_substudy,
    run_linear_cohort_probe,
    run_nonlinear_cohort_probe,
    summarize_cohort_probe,
)
from src.utils.io_helpers import atomic_write_json, atomic_write_text
from src.utils.run_manifest import write_run_manifest
from src.utils.state_manager import StateManager


def read_catalog_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def load_roi_features(catalog_path: Path, allowed_hashes: set[str] | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    rows = read_catalog_rows(catalog_path)
    if allowed_hashes is not None:
        rows = [row for row in rows if row.get("sample_hash") in allowed_hashes]
    feature_names = [
        c
        for c in rows[0].keys()
        if c.endswith("_icv_norm") or c in {"wmh_corrected", "microhemorrhage_definite", "microhemorrhage_lobar", "microhemorrhage_deep", "superficial_siderosis"}
    ]
    x = []
    substudy = []
    groups = []
    for row in rows:
        vals = []
        ok = True
        for name in feature_names:
            text = (row.get(name) or "").strip()
            if text == "":
                vals.append(np.nan)
            else:
                try:
                    vals.append(float(text))
                except ValueError:
                    vals.append(np.nan)
        x.append(vals)
        substudy.append(row["substudy"])
        groups.append(row["patient_hash"])
    return np.asarray(x, dtype=float), np.asarray(substudy), np.asarray(groups), feature_names


def load_tabular_features(catalog_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = read_catalog_rows(catalog_path)
    x = []
    substudy = []
    groups = []
    for row in rows:
        sex = row.get("sex", "")
        sex_val = 1.0 if str(sex).strip() in {"1", "M", "Male", "male"} else 0.0
        x.append(
            [
                float(row.get("age") or np.nan),
                sex_val,
                float(row.get("education_years") or np.nan),
                float(row.get("apoe4_carrier") or np.nan),
                float(row.get("mmse_base") or np.nan),
            ]
        )
        substudy.append(row["substudy"])
        groups.append(row["patient_hash"])
    return np.asarray(x, dtype=float), np.asarray(substudy), np.asarray(groups)


def load_3d_embeddings(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    obj = np.load(path, allow_pickle=True)
    return obj["embeddings"], obj["substudy"], obj["patient_hash"]


def probe_block(name: str, x: np.ndarray, substudy: np.ndarray, groups: np.ndarray, n_splits: int, seed: int) -> dict[str, object]:
    y_cohort = cohort_label_from_substudy(substudy)
    linear = run_linear_cohort_probe(x, y_cohort, groups, n_splits=n_splits, seed=seed)
    nonlinear = run_nonlinear_cohort_probe(x, y_cohort, groups, n_splits=n_splits, seed=seed + 1)
    summary = summarize_cohort_probe(linear, nonlinear)
    return {
        "input_name": name,
        "n": int(len(substudy)),
        "n_features": int(x.shape[1]),
        "target": "substudy_cohort_A4_vs_LEARN",
        "linear_probe": linear,
        "nonlinear_probe": nonlinear,
        **summary,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Run Phase 0.6 linear + nonlinear cohort-probe protocol.")
    ap.add_argument("--embeddings", default="data/processed/method_x_3d_mri_embeddings_oof.npz")
    ap.add_argument("--strict_catalog", default="data/processed/catalogs/method_x_strict_catalog.csv")
    ap.add_argument("--roi_catalog", default="data/processed/catalogs/mri_roi_wmh_reads_catalog.csv")
    ap.add_argument("--out_json", default="results/tables/cohort_probe/cohort_probe_summary.json")
    ap.add_argument("--out_md", default="results/tables/cohort_probe/cohort_probe_summary.md")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

    blocks = []
    if Path(args.embeddings).exists():
        x, substudy, groups = load_3d_embeddings(Path(args.embeddings))
        blocks.append(probe_block("method_x_3d_mri_latent_oof", x, substudy, groups, args.folds, args.seed))

    x_tab, sub_tab, grp_tab = load_tabular_features(Path(args.strict_catalog))
    blocks.append(probe_block("method_x_strict_tabular", x_tab, sub_tab, grp_tab, args.folds, args.seed))

    allowed_hashes = {row["sample_hash"] for row in read_catalog_rows(Path(args.strict_catalog))}
    x_roi, sub_roi, grp_roi, roi_names = load_roi_features(Path(args.roi_catalog), allowed_hashes=allowed_hashes)
    blocks.append(probe_block("mri_roi_wmh_reads", x_roi, sub_roi, grp_roi, args.folds, args.seed))

    report = {
        "schema_version": 1,
        "status": "COMPLETED",
        "protocol_reference": "results/governance/cohort_probe_protocol.json",
        "headline_validity_metric": True,
        "blocks": blocks,
        "roi_feature_names_count": len(roi_names),
        "interpretation_notes": [
            "High linear probe AUC indicates cohort is linearly encoded.",
            "Low linear but high nonlinear probe AUC indicates covert/structured cohort signal.",
            "Never claim deconfounding succeeded based on the linear probe alone.",
        ],
        "raw_identifiers_written": False,
        "subject_level_predictions_exported": False,
        "public_output_granularity": "aggregate_probe_level",
    }
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(out_json, report)

    lines = [
        "# Cohort-Probe Summary (Phase 0.6 Protocol)\n\n",
        "| input | linear_auc | nonlinear_auc | gap | interpretation |\n",
        "|---|---:|---:|---:|---|\n",
    ]
    for block in blocks:
        lines.append(
            f"| {block['input_name']} | {block['linear_probe_auc']:.4f} | {block['nonlinear_probe_auc']:.4f} | {block['linear_vs_nonlinear_gap']:.4f} | {block['interpretation']} |\n"
        )
    atomic_write_text(args.out_md, "".join(lines))
    write_run_manifest("results/logs/cohort_probe_manifest.json", stage="cohort_probe", data_root=None, extra={"blocks": len(blocks)})
    StateManager().append_event("cohort_probe", "COMPLETED", f"Cohort-probe completed for {len(blocks)} input blocks.")
    print(json.dumps({"status": "COMPLETED", "blocks": len(blocks)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
