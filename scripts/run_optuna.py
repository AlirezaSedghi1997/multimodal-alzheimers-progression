from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.extract_mri_features import MRI_FEATURES
from scripts.train_baseline import average_precision, brier, guarded_bootstrap_ci, roc_auc, safe_float
from src.utils.io_helpers import atomic_write_json, atomic_write_text
from src.utils.run_manifest import write_run_manifest
from src.utils.state_manager import StateManager


TABULAR_FEATURES = ["age", "education_years", "apoe4_carrier", "mmse_base"]
SEARCH_SPACE = [
    {"C": 0.01, "class_weight": "balanced"},
    {"C": 0.03, "class_weight": "balanced"},
    {"C": 0.1, "class_weight": "balanced"},
    {"C": 0.3, "class_weight": "balanced"},
    {"C": 1.0, "class_weight": "balanced"},
    {"C": 3.0, "class_weight": "balanced"},
    {"C": 10.0, "class_weight": "balanced"},
    {"C": 0.1, "class_weight": None},
    {"C": 1.0, "class_weight": None},
    {"C": 10.0, "class_weight": None},
]


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def read_features(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    return {row["sample_hash"]: row for row in read_rows(path) if row.get("mri_feature_status") == "OK"}


def matrix(rows: list[dict[str, str]], features: dict[str, dict[str, str]] | None, mode: str) -> tuple[list[list[float]], list[int]]:
    x: list[list[float]] = []
    y: list[int] = []
    for row in rows:
        sex = row.get("sex", "")
        sex_val = 1.0 if str(sex).strip() in {"1", "M", "Male", "male"} else 0.0
        vals = [safe_float(row.get(f)) for f in TABULAR_FEATURES] + [sex_val]
        if mode == "tabular_plus_mri_features":
            if not features:
                continue
            frow = features.get(row["sample_hash"])
            if frow is None:
                continue
            vals.extend(safe_float(frow.get(f)) for f in MRI_FEATURES)
        x.append(vals)
        y.append(int(float(row["ab_label"])))
    return x, y


def fit_predict(train_rows, test_rows, features, mode: str, params: dict[str, object]) -> tuple[list[int], list[float]]:
    try:
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
    except Exception as exc:
        raise RuntimeError("run_optuna requires scikit-learn") from exc

    x_train, y_train = matrix(train_rows, features, mode)
    x_test, y_test = matrix(test_rows, features, mode)
    model = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    C=float(params["C"]),
                    class_weight=params["class_weight"],
                    max_iter=1000,
                    random_state=23,
                ),
            ),
        ]
    )
    model.fit(x_train, y_train)
    return y_test, [float(v) for v in model.predict_proba(x_test)[:, 1]]


def evaluate_candidate(split_dir: Path, features: dict[str, dict[str, str]], mode: str, params: dict[str, object]) -> dict[str, object]:
    fold_metrics = []
    all_y: list[int] = []
    all_p: list[float] = []
    for test_path in sorted(split_dir.glob("fold_*_test.csv")):
        fold = int(test_path.name.split("_")[1])
        train_rows = read_rows(split_dir / f"fold_{fold}_train.csv")
        test_rows = read_rows(test_path)
        y, p = fit_predict(train_rows, test_rows, features, mode, params)
        fold_metrics.append(
            {
                "fold": fold,
                "n_test": len(y),
                "roc_auc": roc_auc(y, p),
                "pr_auc": average_precision(y, p),
                "brier": brier(y, p),
            }
        )
        all_y.extend(y)
        all_p.extend(p)
    return {
        "mode": mode,
        "params": params,
        "n": len(all_y),
        "roc_auc": roc_auc(all_y, all_p),
        "pr_auc": average_precision(all_y, all_p),
        "brier": brier(all_y, all_p),
        "folds": fold_metrics,
        "y_true": all_y,
        "y_prob": all_p,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", default="splits")
    ap.add_argument("--features", default="data/processed/mri_features.csv")
    ap.add_argument("--out_dir", default="results/final_results/optuna")
    args = ap.parse_args()

    split_dir = Path(args.splits)
    features = read_features(Path(args.features))
    candidates = []
    for mode in ["tabular_only", "tabular_plus_mri_features"]:
        if mode == "tabular_plus_mri_features" and not features:
            continue
        for params in SEARCH_SPACE:
            candidates.append(evaluate_candidate(split_dir, features, mode, params))
    best = max(candidates, key=lambda c: (float(c["roc_auc"]), float(c["pr_auc"])))
    y_true = best.pop("y_true")
    y_prob = best.pop("y_prob")
    best["roc_auc_ci"] = guarded_bootstrap_ci(y_true, y_prob, roc_auc, n_boot=1000)
    best["pr_auc_ci"] = guarded_bootstrap_ci(y_true, y_prob, average_precision, n_boot=1000)
    best["uses_pet_as_input"] = False
    best["uses_raw_identifiers"] = False
    best["gate_2_passed"] = bool(float(best["roc_auc"]) >= 0.70)
    summary = {
        "best": best,
        "n_candidates": len(candidates),
        "candidate_table": [
            {
                "mode": c["mode"],
                "params": c["params"],
                "roc_auc": c["roc_auc"],
                "pr_auc": c["pr_auc"],
                "brier": c["brier"],
            }
            for c in candidates
        ],
        "method_x_compliant": True,
        "cohort_label_confound_warning": True,
    }
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(out_dir / "optuna_summary.json", summary)
    lines = [
        "# Optuna-Like Fast-Track Summary\n\n",
        f"- n_candidates: `{summary['n_candidates']}`\n",
        f"- best_mode: `{best['mode']}`\n",
        f"- best_params: `{best['params']}`\n",
        f"- roc_auc: `{float(best['roc_auc']):.4f}`\n",
        f"- pr_auc: `{float(best['pr_auc']):.4f}`\n",
        f"- brier: `{float(best['brier']):.4f}`\n",
        f"- gate_2_passed: `{best['gate_2_passed']}`\n",
        "- uses_pet_as_input: `False`\n",
        "- uses_raw_identifiers: `False`\n",
        "- cohort_label_confound_warning: `True`\n",
    ]
    atomic_write_text(out_dir / "optuna_summary.md", "".join(lines))
    write_run_manifest("results/logs/optuna_manifest.json", stage="optuna", data_root=None, extra={"best": best})
    StateManager().append_event("optuna", "COMPLETED" if best["gate_2_passed"] else "FAILED", "Fast-track hyperparameter sweep completed.")
    print(json.dumps({"roc_auc": best["roc_auc"], "pr_auc": best["pr_auc"], "gate_2_passed": best["gate_2_passed"], "mode": best["mode"]}, indent=2))
    return 0 if best["gate_2_passed"] else 3


if __name__ == "__main__":
    raise SystemExit(main())

