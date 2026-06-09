#!/usr/bin/env python3
"""Decision-curve analysis for interpretable branch models (aggregate only)."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_matched_propensity_analysis import enrich_row, feature_vector, read_rows
from src.utils.io_helpers import atomic_write_json, atomic_write_text
from src.utils.run_manifest import write_run_manifest
from src.utils.state_manager import StateManager


def net_benefit(y_true: list[int], y_prob: list[float], threshold: float) -> float:
    n = len(y_true)
    if n == 0:
        return 0.0
    pred_pos = [p >= threshold for p in y_prob]
    tp = sum(1 for yt, pp in zip(y_true, pred_pos) if yt == 1 and pp)
    fp = sum(1 for yt, pp in zip(y_true, pred_pos) if yt == 0 and pp)
    return (tp / n) - (fp / n) * (threshold / (1 - threshold + 1e-12))


def oof_tabular_predictions(catalog: Path, seed: int = 42) -> tuple[list[int], list[float]]:
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    rows = [enrich_row(r) for r in read_rows(catalog)]
    x = [feature_vector(r) for r in rows]
    y = [int(float(r["ab_label"])) for r in rows]
    probs = [0.0] * len(rows)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    for train_idx, test_idx in skf.split(x, y):
        model = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000, random_state=seed)),
            ]
        )
        model.fit([x[i] for i in train_idx], [y[i] for i in train_idx])
        pred = model.predict_proba([x[i] for i in test_idx])[:, 1]
        for j, p in zip(test_idx, pred):
            probs[j] = float(p)
    return y, probs


def curve_from_predictions(name: str, label: str, y: list[int], prob: list[float], thresholds: list[float]) -> dict:
    prevalence = sum(y) / len(y)
    treat_all = [prevalence - (1 - prevalence) * (t / (1 - t + 1e-12)) for t in thresholds]
    return {
        "status": "OK",
        "label": label,
        "n": len(y),
        "thresholds": thresholds,
        "net_benefit_model": [net_benefit(y, prob, t) for t in thresholds],
        "net_benefit_treat_all": treat_all,
        "net_benefit_treat_none": [0.0] * len(thresholds),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--catalog", default="data/processed/master_catalog.csv")
    ap.add_argument("--thresholds", default="0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    thresholds = [float(x) for x in args.thresholds.split(",") if x.strip()]

    y, prob = oof_tabular_predictions(Path(args.catalog), seed=args.seed)
    curves = {
        "method_x_strict_tabular": curve_from_predictions(
            "method_x_strict_tabular",
            "Method X-Strict fixed tabular (5-fold OOF)",
            y,
            prob,
            thresholds,
        )
    }
    zpath = Path("results/final_results/bio_cog_method_z/bio_cog_method_z_summary.json")
    if zpath.exists():
        z = json.loads(zpath.read_text(encoding="utf-8"))
        enrolled = z.get("arms", {}).get("enrolled_a4_learn", {}).get("variants", {}).get("core_blood_only", {})
        if enrolled.get("roc_auc") is not None:
            curves["method_z_blood_core"] = {
                "status": "METRICS_ONLY",
                "label": "Method Z core blood (CV metrics only; DCA curve requires stored OOF)",
                "roc_auc": enrolled.get("roc_auc"),
                "n": z.get("arms", {}).get("enrolled_a4_learn", {}).get("rows"),
            }

    report = {
        "stage": "decision_curve_analysis",
        "status": "COMPLETED",
        "curves": curves,
        "claim_boundary": "Exploratory clinical utility framing only; not Alzheimer diagnosis claim.",
    }
    atomic_write_json("results/tables/decision_curve_analysis.json", report)
    tab = curves.get("method_x_strict_tabular") or {}
    md_lines = [
        "# Decision Curve Analysis\n\n",
        f"- status: `{report['status']}`\n",
        f"- claim boundary: {report['claim_boundary']}\n\n",
        "## Method X-Strict fixed tabular\n\n",
        f"- n: `{tab.get('n')}`\n",
        f"- thresholds: `{tab.get('thresholds')}`\n\n",
        "| threshold | net benefit (model) | treat all | treat none |\n",
        "|---:|---:|---:|---:|\n",
    ]
    for t, nb_m, nb_a, nb_n in zip(
        tab.get("thresholds") or [],
        tab.get("net_benefit_model") or [],
        tab.get("net_benefit_treat_all") or [],
        tab.get("net_benefit_treat_none") or [],
    ):
        md_lines.append(f"| {t:.2f} | {nb_m:.4f} | {nb_a:.4f} | {nb_n:.4f} |\n")
    md_lines.append("\n## Method Z\n\n")
    z = curves.get("method_z_blood_core") or {}
    md_lines.append(f"- status: `{z.get('status')}` — `{z.get('label')}`\n")
    if z.get("roc_auc") is not None:
        md_lines.append(f"- enrolled ROC-AUC (metrics only): `{z.get('roc_auc')}`\n")
    atomic_write_text("results/tables/decision_curve_analysis.md", "".join(md_lines))
    write_run_manifest("results/logs/decision_curve_manifest.json", stage="decision_curve_analysis", data_root=None, extra={"summary": report})
    StateManager().append_event("decision_curve_analysis", "COMPLETED", "Decision curve analysis completed.")
    print(json.dumps({"status": report["status"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
