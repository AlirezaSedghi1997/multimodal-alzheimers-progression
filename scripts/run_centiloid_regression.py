from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src.data.a4_schema import REQUIRED_RELATIVE_FILES, resolve_data_root
from src.data.label_policy import centiloid_from_pmod_suvr
from src.utils.io_helpers import atomic_write_json, atomic_write_text
from src.utils.privacy import stable_hash
from src.utils.run_manifest import write_run_manifest
from src.utils.state_manager import StateManager


FEATURES = ["age", "education_years", "apoe4_carrier", "mmse_base"]


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def build_centiloid_lookup(root: Path) -> dict[str, float]:
    pet_path = root / REQUIRED_RELATIVE_FILES["ab_label"]
    lookup: dict[str, float] = {}
    for row in read_rows(pet_path):
        substudy = (row.get("SUBSTUDY") or "").upper()
        bid = row.get("BID") or ""
        pmod = row.get("pmod_suvr")
        if not pmod:
            continue
        try:
            cent = centiloid_from_pmod_suvr(float(pmod))
        except ValueError:
            continue
        sample_hash = stable_hash(substudy + "|" + bid + "|" + str(row.get("VISCODE", "")), prefix="s")
        lookup[sample_hash] = cent
    return lookup


def feature_matrix(rows: list[dict[str, str]], centiloid_lookup: dict[str, float]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x = []
    y = []
    substudy = []
    groups = []
    for row in rows:
        sample_hash = row["sample_hash"]
        if sample_hash not in centiloid_lookup:
            continue
        sex = row.get("sex", "")
        sex_val = 1.0 if str(sex).strip() in {"1", "M", "Male", "male"} else 0.0
        x.append(
            [
                float(row["age"]) if row.get("age") else np.nan,
                sex_val,
                float(row["education_years"]) if row.get("education_years") else np.nan,
                float(row["apoe4_carrier"]) if row.get("apoe4_carrier") else np.nan,
                float(row["mmse_base"]) if row.get("mmse_base") else np.nan,
            ]
        )
        y.append(centiloid_lookup[sample_hash])
        substudy.append(row["substudy"])
        groups.append(row["patient_hash"])
    return (
        np.asarray(x, dtype=float),
        np.asarray(y, dtype=float),
        np.asarray(substudy),
        np.asarray(groups),
    )


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | None]:
    from sklearn.metrics import mean_absolute_error, r2_score
    from scipy.stats import spearmanr

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if len(y_true) < 3:
        return {"n": float(len(y_true)), "spearman_r": None, "r2": None, "mae": None}
    spearman = spearmanr(y_true, y_pred).statistic
    return {
        "n": float(len(y_true)),
        "spearman_r": float(spearman) if spearman is not None else None,
        "r2": float(r2_score(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
    }


def cv_regression(
    x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    model_name: str,
    n_splits: int,
    seed: int,
) -> dict[str, object]:
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import ElasticNet, HuberRegressor
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    if model_name == "huber":
        reg = HuberRegressor(max_iter=1000)
    elif model_name == "elastic_net":
        reg = ElasticNet(alpha=0.05, l1_ratio=0.5, max_iter=5000, random_state=seed)
    elif model_name == "gbt":
        reg = HistGradientBoostingRegressor(max_depth=3, max_iter=300, learning_rate=0.05, random_state=seed)
    else:
        raise ValueError(model_name)

    splitter = GroupKFold(n_splits=n_splits)
    fold_rows = []
    y_all: list[float] = []
    p_all: list[float] = []
    for fold, (tr, te) in enumerate(splitter.split(x, y, groups)):
        model = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("reg", reg),
            ]
        )
        model.fit(x[tr], y[tr])
        pred = model.predict(x[te])
        metrics = regression_metrics(y[te], pred)
        metrics["fold"] = fold
        fold_rows.append(metrics)
        y_all.extend(y[te].tolist())
        p_all.extend(pred.tolist())
    aggregate = regression_metrics(np.asarray(y_all), np.asarray(p_all))
    aggregate["model"] = model_name
    aggregate["folds"] = fold_rows
    return aggregate


def analyze_subset(name: str, x: np.ndarray, y: np.ndarray, groups: np.ndarray, n_splits: int, seed: int) -> dict[str, object]:
    if len(y) < 20:
        return {"subset": name, "status": "INSUFFICIENT_N", "n": int(len(y))}
    models = {}
    for model_name in ("huber", "elastic_net", "gbt"):
        try:
            models[model_name] = cv_regression(x, y, groups, model_name, n_splits=min(n_splits, 5), seed=seed)
        except Exception as exc:
            models[model_name] = {"model": model_name, "status": "FAILED", "error": str(exc)}
    return {"subset": name, "status": "COMPLETED", "n": int(len(y)), "models": models}


def main() -> int:
    ap = argparse.ArgumentParser(description="Continuous centiloid regression with within-cohort analyses.")
    ap.add_argument("--data_root", default=os.environ.get("DATA_ROOT"))
    ap.add_argument("--catalog", default="data/processed/master_catalog.csv")
    ap.add_argument("--out_dir", default="results/final_results/centiloid_regression")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

    root = resolve_data_root(args.data_root)
    rows = read_rows(Path(args.catalog))
    centiloid_lookup = build_centiloid_lookup(root)
    x, y, substudy, groups = feature_matrix(rows, centiloid_lookup)

    subsets = {
        "pooled_enrolled": np.ones(len(y), dtype=bool),
        "within_a4": substudy == "A4",
        "within_learn": substudy == "LEARN",
    }
    results = {}
    for name, mask in subsets.items():
        results[name] = analyze_subset(name, x[mask], y[mask], groups[mask], args.folds, args.seed)

    report = {
        "schema_version": 1,
        "status": "COMPLETED",
        "target": "continuous_centiloid",
        "predictors": FEATURES + ["sex_binary"],
        "subsets": results,
        "caveats": {
            "pooled_enrolled": "Strong cohort-label coupling; pooled result is descriptive only.",
            "within_a4": "Within-cohort biological signal assessment.",
            "within_learn": "Within-cohort biological signal assessment.",
        },
        "centiloid_as_predictor": False,
        "raw_identifiers_written": False,
    }
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(out_dir / "centiloid_regression_metrics.json", report)
    lines = ["# Continuous Centiloid Regression\n\n"]
    for name, subset in results.items():
        lines.append(f"## {name}\n\n")
        if subset.get("status") != "COMPLETED":
            lines.append(f"- status: `{subset.get('status')}`\n\n")
            continue
        lines.append(f"- n: `{subset['n']}`\n\n")
        lines.append("| model | spearman_r | r2 | mae |\n|---|---:|---:|---:|\n")
        for model_name, metrics in subset.get("models", {}).items():
            if metrics.get("status") == "FAILED":
                continue
            lines.append(
                f"| {model_name} | {metrics.get('spearman_r')} | {metrics.get('r2')} | {metrics.get('mae')} |\n"
            )
        lines.append("\n")
    atomic_write_text(out_dir / "centiloid_regression_metrics.md", "".join(lines))
    write_run_manifest("results/logs/centiloid_regression_manifest.json", stage="centiloid_regression", data_root=str(root), extra={"subsets": list(results)})
    StateManager().append_event("centiloid_regression", "COMPLETED", "Continuous centiloid regression completed for pooled/A4/LEARN subsets.")
    print(json.dumps({"status": "COMPLETED", "n": int(len(y))}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
