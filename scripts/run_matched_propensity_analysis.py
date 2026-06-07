#!/usr/bin/env python3
"""Matched / propensity-weighted subset analysis for Method X tabular headline model."""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.train_baseline import average_precision, brier, roc_auc
from src.evaluation.bootstrap_ci import guarded_bootstrap_ci
from src.utils.io_helpers import atomic_write_json, atomic_write_text
from src.utils.run_manifest import write_run_manifest
from src.utils.state_manager import StateManager

MATCH_FEATURES = ["age", "sex_binary", "education_years", "apoe4_carrier", "mmse_base"]


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def safe_float(value: str | None) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.upper() in {"NA", "NAN", "NULL"}:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def enrich_row(row: dict[str, str]) -> dict[str, str]:
    out = dict(row)
    sex = row.get("sex", "")
    out["sex_binary"] = "1" if str(sex).strip() in {"1", "M", "Male", "male"} else "0"
    return out


def feature_vector(row: dict[str, str]) -> list[float]:
    return [safe_float(row.get(k)) or 0.0 for k in MATCH_FEATURES]


def match_distance(a: dict[str, str], b: dict[str, str]) -> float:
    va, vb = feature_vector(a), feature_vector(b)
    return sum((x - y) ** 2 for x, y in zip(va, vb)) ** 0.5


def greedy_match(pos_rows: list[dict[str, str]], neg_rows: list[dict[str, str]], rng: random.Random) -> tuple[list[tuple[dict[str, str], dict[str, str]]], list[dict[str, str]]]:
    neg_pool = list(neg_rows)
    rng.shuffle(neg_pool)
    pairs: list[tuple[dict[str, str], dict[str, str]]] = []
    used_neg: set[str] = set()
    for pos in pos_rows:
        best = None
        best_dist = None
        for neg in neg_pool:
            if neg["sample_hash"] in used_neg:
                continue
            dist = match_distance(pos, neg)
            if best is None or dist < best_dist:
                best, best_dist = neg, dist
        if best is None:
            continue
        used_neg.add(best["sample_hash"])
        pairs.append((pos, best))
    unmatched_neg = [r for r in neg_rows if r["sample_hash"] not in used_neg]
    return pairs, unmatched_neg


def fit_predict(rows: list[dict[str, str]], seed: int = 42) -> tuple[list[int], list[float]]:
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    x = [feature_vector(r) for r in rows]
    y = [int(float(r["ab_label"])) for r in rows]
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    probs = [0.0] * len(rows)
    for train_idx, test_idx in skf.split(x, y):
        x_train = [x[i] for i in train_idx]
        y_train = [y[i] for i in train_idx]
        model = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000, random_state=seed)),
            ]
        )
        model.fit(x_train, y_train)
        pred = model.predict_proba([x[i] for i in test_idx])[:, 1]
        for j, p in zip(test_idx, pred):
            probs[j] = float(p)
    return y, probs


def summarize(name: str, rows: list[dict[str, str]], seed: int) -> dict:
    if len(set(r["ab_label"] for r in rows)) < 2 or len(rows) < 20:
        return {"status": "SKIPPED_LOW_N", "name": name, "n": len(rows)}
    y, probs = fit_predict(rows, seed=seed)
    return {
        "status": "OK",
        "name": name,
        "n": len(rows),
        "pos": sum(y),
        "neg": len(y) - sum(y),
        "roc_auc": roc_auc(y, probs),
        "pr_auc": average_precision(y, probs),
        "brier": brier(y, probs),
        "roc_auc_ci": guarded_bootstrap_ci(y, probs, roc_auc, n_bootstrap=500, seed=seed),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--catalog", default="data/processed/master_catalog.csv")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rows = [enrich_row(r) for r in read_rows(Path(args.catalog))]
    by_label: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_label[row["ab_label"]].append(row)
    pos = by_label.get("1", [])
    neg = by_label.get("0", [])
    rng = random.Random(args.seed)
    pairs, _ = greedy_match(pos, neg, rng)
    matched = [p for pair in pairs for p in pair]

    report = {
        "stage": "matched_propensity_subset",
        "status": "COMPLETED",
        "method": "greedy_1to1_covariate_nearest_neighbor",
        "match_features": MATCH_FEATURES,
        "full_cohort": summarize("full_cohort", rows, args.seed),
        "matched_subset": summarize("matched_1to1_subset", matched, args.seed),
        "n_pairs": len(pairs),
        "claim_boundary": "Covariate-matched subset is a sensitivity analysis only; pseudo-external holdout rules unchanged.",
    }
    out_json = Path("results/tables/matched_propensity_subset.json")
    atomic_write_json(out_json, report)
    md = [
        "# Matched / Propensity-Weighted Subset Analysis\n\n",
        f"- status: `{report['status']}`\n",
        f"- n_pairs: `{report['n_pairs']}`\n",
        f"- full cohort ROC-AUC: `{report['full_cohort'].get('roc_auc')}`\n",
        f"- matched subset ROC-AUC: `{report['matched_subset'].get('roc_auc')}`\n",
    ]
    atomic_write_text("results/tables/matched_propensity_subset.md", "".join(md))
    write_run_manifest("results/logs/matched_propensity_manifest.json", stage="matched_propensity", data_root=None, extra={"summary": report})
    StateManager().append_event("matched_propensity", "COMPLETED", "Matched covariate subset analysis completed.")
    print(json.dumps({"n_pairs": report["n_pairs"], "matched_roc_auc": report["matched_subset"].get("roc_auc")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
