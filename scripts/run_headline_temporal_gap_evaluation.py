#!/usr/bin/env python3
"""Evaluate headline tabular and ROI models across PET-T1 temporal-gap catalog variants.

CPU-only; does not touch PET SPM preprocessing or GPU 3D training.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_branch_baselines import (
    DEFAULT_CONFIG,
    evaluate_feature_set,
    load_yaml,
    prepare_rows,
    read_csv_rows,
)
from scripts.run_temporal_window_sensitivity import derive_gap_col, find_col, LABEL_CANDIDATES
from src.utils.io_helpers import atomic_write_json, atomic_write_text
from src.utils.run_manifest import write_run_manifest
from src.utils.state_manager import StateManager


HEADLINE_SETS = ["fixed_tabular", "roi_icv_norm_only"]
THREE_D_REFERENCE = PROJECT_ROOT / "results/final_results/three_d/three_d_metrics.json"


def filter_rows_by_catalog(
    rows_by_sample: dict[str, dict[str, str]],
    allowed_hashes: set[str],
) -> dict[str, dict[str, str]]:
    return {h: r for h, r in rows_by_sample.items() if h in allowed_hashes}


def cohort_descriptives(rows: list[dict[str, str]]) -> dict[str, Any]:
    label_col = find_col(pd.DataFrame(rows), LABEL_CANDIDATES) or "ab_label_primary"
    labels = [int(float(r.get(label_col, r.get("ab_label", 0)))) for r in rows if r.get("sample_hash")]
    pos = sum(1 for y in labels if y == 1)
    neg = sum(1 for y in labels if y == 0)
    subs = {}
    if rows and "substudy" in rows[0]:
        subs = {str(k): int(v) for k, v in pd.Series([r.get("substudy") for r in rows]).value_counts().items()}
    return {"n": len(labels), "pos": pos, "neg": neg, "substudy_counts": subs}


def load_three_d_reference() -> dict[str, Any]:
    if not THREE_D_REFERENCE.exists():
        return {"status": "MISSING", "note": "Full-cohort 3D metrics unavailable."}
    data = json.loads(THREE_D_REFERENCE.read_text(encoding="utf-8"))
    agg = data.get("aggregate", {})
    return {
        "status": "REFERENCE_FULL_COHORT_ONLY",
        "note": "3D CNN not retrained per gap window (GPU/PET isolation). Report overlap N and full-cohort reference metrics.",
        "full_cohort_n": agg.get("n"),
        "full_cohort_roc_auc": agg.get("roc_auc"),
        "full_cohort_pr_auc": agg.get("pr_auc"),
        "final_claim_allowed": agg.get("final_claim_allowed"),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--splits", default="splits")
    ap.add_argument("--catalog", default="data/processed/master_catalog.csv")
    ap.add_argument("--windows", default="0,60,90,120")
    ap.add_argument("--out_json", default="results/tables/headline_temporal_gap_evaluation.json")
    ap.add_argument("--out_md", default="results/tables/headline_temporal_gap_evaluation.md")
    args = ap.parse_args()

    config = load_yaml(Path(args.config))
    split_dir = Path(args.splits)
    df = pd.read_csv(PROJECT_ROOT / args.catalog)
    gap, gap_source = derive_gap_col(df)
    windows = [int(x.strip()) for x in args.windows.split(",") if x.strip()]

    prepared: dict[str, dict[str, dict[str, str]]] = {}
    for set_name in HEADLINE_SETS:
        spec = config["feature_sets"][set_name]
        prepared[set_name] = prepare_rows(config, set_name, spec)

    report: dict[str, Any] = {
        "stage": "headline_temporal_gap_evaluation",
        "status": "COMPLETED",
        "gap_source": gap_source,
        "headline_feature_sets": HEADLINE_SETS,
        "three_d_cnn": load_three_d_reference(),
        "windows": [],
        "uses_pet_as_input": False,
        "claim_boundary": "Temporal-gap sensitivity on Method X headline CPU models; 3D is reference-only per window overlap.",
    }

    hash_col = "sample_hash"
    all_hashes = set(df[hash_col].astype(str))

    for w in windows:
        if w == 0:
            allowed = all_hashes
            policy = "NO_GAP_FILTER"
        else:
            mask = gap.notna() & (gap <= float(w))
            allowed = set(df.loc[mask, hash_col].astype(str))
            policy = f"ABS_GAP_LE_{w}_DAYS"

        window_entry: dict[str, Any] = {
            "window_days": w,
            "policy": policy,
            "n_catalog": int(len(allowed)),
            "models": {},
        }
        master_rows = [r for r in read_csv_rows(PROJECT_ROOT / args.catalog) if r.get(hash_col) in allowed]
        window_entry["cohort_descriptives"] = cohort_descriptives(master_rows)

        overlap_3d = len(allowed & set(prepared["fixed_tabular"].keys()))
        window_entry["three_d_cnn"] = {
            **load_three_d_reference(),
            "eligible_strict_overlap_n": overlap_3d,
        }

        for set_name in HEADLINE_SETS:
            filtered = filter_rows_by_catalog(prepared[set_name], allowed)
            if len(filtered) < 50:
                window_entry["models"][set_name] = {"status": "SKIPPED_LOW_N", "n": len(filtered)}
                continue
            try:
                result = evaluate_feature_set(
                    set_name,
                    config["feature_sets"][set_name],
                    config,
                    filtered,
                    split_dir,
                    config.get("headline_model", "logistic_regression"),
                )
                window_entry["models"][set_name] = {"status": "OK", **result["aggregate"]}
            except Exception as exc:
                window_entry["models"][set_name] = {
                    "status": "FAIL",
                    "error_type": type(exc).__name__,
                    "error_message": str(exc)[:300],
                }
        report["windows"].append(window_entry)

    out_json = PROJECT_ROOT / args.out_json
    out_md = PROJECT_ROOT / args.out_md
    atomic_write_json(out_json, report)

    lines = [
        "# Headline Temporal-Gap Evaluation\n\n",
        "CPU baselines on gap-filtered catalogs. 3D CNN: full-cohort reference only (not retrained per window).\n\n",
        f"- gap source: `{gap_source}`\n\n",
        "| window | n | tabular AUC [95% CI] | ROI AUC [95% CI] | 3D overlap n |\n",
        "|---:|---:|---|---|---:|\n",
    ]
    for row in report["windows"]:
        tab = row["models"].get("fixed_tabular", {})
        roi = row["models"].get("roi_icv_norm_only", {})

        def fmt_model(model: dict) -> str:
            if model.get("status") != "OK":
                return model.get("status", "NA")
            ci = model.get("roc_auc_ci") or {}
            auc = model.get("roc_auc")
            if ci.get("status") == "OK":
                return f"{auc:.4f} [{ci.get('ci_low', ci.get('ci_lower')):.3f}, {ci.get('ci_high', ci.get('ci_upper')):.3f}]"
            return f"{auc:.4f}" if isinstance(auc, float) else "NA"

        lines.append(
            f"| {row['window_days']} | {row['n_catalog']} | {fmt_model(tab)} | {fmt_model(roi)} | "
            f"{row['three_d_cnn'].get('eligible_strict_overlap_n', 'NA')} |\n"
        )
    lines.append(
        "\n## Interpretation\n\n"
        "Window≤60d (n≈154) may show higher point AUC than full cohort, but bootstrap CIs often overlap baseline. "
        "Do not interpret subset AUC without CI and cohort descriptives.\n"
    )
    atomic_write_text(out_md, "".join(lines))
    write_run_manifest(
        "results/logs/headline_temporal_gap_evaluation_manifest.json",
        stage="headline_temporal_gap_evaluation",
        data_root=None,
        extra={"windows": windows},
    )
    StateManager().append_event(
        "headline_temporal_gap_evaluation",
        "COMPLETED",
        f"Headline temporal-gap evaluation for {len(windows)} windows.",
    )
    print(json.dumps({"status": "COMPLETED", "windows": len(windows)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
