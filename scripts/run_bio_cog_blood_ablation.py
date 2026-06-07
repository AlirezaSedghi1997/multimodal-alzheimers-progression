from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.train_bio_cog_method_z import (
    BLOOD_FULL,
    average_precision,
    brier,
    fit_predict,
    load_labeled_rows,
    patient_grouped_folds,
    roc_auc,
    subset,
    usable_rows,
)
from src.utils.io_helpers import atomic_write_json, atomic_write_text
from src.utils.state_manager import StateManager


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def evaluate(rows: list[dict[str, str]], features: list[str], seed: int) -> dict[str, Any]:
    all_y: list[int] = []
    all_p: list[float] = []
    for train_rows, test_rows in patient_grouped_folds(rows, 5, seed):
        y_true, y_prob = fit_predict(train_rows, test_rows, features)
        if not y_true:
            continue
        all_y.extend(y_true)
        all_p.extend(y_prob)
    if not all_y:
        return {"status": "SKIPPED", "features": features}
    return {
        "status": "OK",
        "features": features,
        "feature_count": len(features),
        "n": len(all_y),
        "roc_auc": roc_auc(all_y, all_p),
        "pr_auc": average_precision(all_y, all_p),
        "brier": brier(all_y, all_p),
    }


def label_shuffle_control(rows: list[dict[str, str]], features: list[str], seed: int) -> dict[str, Any]:
    shuffled = [dict(row) for row in rows]
    rng = random.Random(seed)
    labels = [row["ab_label"] for row in shuffled]
    rng.shuffle(labels)
    for row, label in zip(shuffled, labels):
        row["ab_label"] = label
    return evaluate(shuffled, features, seed)


def main() -> int:
    ap = argparse.ArgumentParser(description="Bio-Cog/Method Z blood feature ablation (CPU-only).")
    ap.add_argument("--catalog", default="data/processed/catalogs/bio_cog_method_z_catalog.csv")
    ap.add_argument("--out_json", default="results/final_results/bio_cog_method_z/bio_cog_blood_ablation.json")
    ap.add_argument("--out_md", default="results/final_results/bio_cog_method_z/bio_cog_blood_ablation.md")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rows = usable_rows(subset(load_labeled_rows(Path(args.catalog)), {"A4", "LEARN"}), BLOOD_FULL)
    full = evaluate(rows, BLOOD_FULL, args.seed)
    drop_one = []
    for feature in BLOOD_FULL:
        reduced = [f for f in BLOOD_FULL if f != feature]
        drop_one.append({"dropped_feature": feature, **evaluate(rows, reduced, args.seed)})
    single_feature = [{"feature": feature, **evaluate(rows, [feature], args.seed)} for feature in BLOOD_FULL]
    shuffle = label_shuffle_control(rows, BLOOD_FULL, args.seed + 1)

    summary = {
        "schema_version": 1,
        "branch": "bio_cog_method_z",
        "status": "COMPLETED",
        "arm": "enrolled_a4_learn",
        "full_panel": full,
        "drop_one_ablation": drop_one,
        "single_feature_models": single_feature,
        "label_shuffle_control": shuffle,
        "cohort_coupling_caveat": "Enrolled A4/LEARN retains strong cohort-label coupling; do not over-interpret AUC.",
        "supports_method_x_strict_claim": False,
    }
    out_json = Path(args.out_json)
    out_md = Path(args.out_md)
    atomic_write_json(out_json, summary)
    lines = [
        "# Bio-Cog / Method Z Blood Ablation\n\n",
        f"- full panel roc_auc: `{full.get('roc_auc')}`\n",
        f"- label shuffle roc_auc: `{shuffle.get('roc_auc')}`\n\n",
        "## Drop-one\n\n",
    ]
    for row in drop_one:
        lines.append(f"- drop `{row.get('dropped_feature')}` → auc `{row.get('roc_auc')}`\n")
    atomic_write_text(out_md, "".join(lines))
    StateManager().append_event("bio_cog_blood_ablation", "COMPLETED", "Method Z blood ablation completed.")
    print(json.dumps({"status": "COMPLETED", "full_roc_auc": full.get("roc_auc")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
