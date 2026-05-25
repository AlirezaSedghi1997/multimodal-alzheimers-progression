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


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def feature_index(path: Path) -> dict[str, dict[str, str]]:
    return {row["sample_hash"]: row for row in read_rows(path) if row.get("mri_feature_status") == "OK"}


def matrix(rows: list[dict[str, str]], features: dict[str, dict[str, str]]) -> tuple[list[list[float]], list[int]]:
    x = []
    y = []
    for row in rows:
        frow = features.get(row["sample_hash"])
        if frow is None:
            continue
        sex = row.get("sex", "")
        sex_val = 1.0 if str(sex).strip() in {"1", "M", "Male", "male"} else 0.0
        vals = [safe_float(row.get(f)) for f in TABULAR_FEATURES] + [sex_val]
        vals.extend(safe_float(frow.get(f)) for f in MRI_FEATURES)
        x.append(vals)
        y.append(int(float(row["ab_label"])))
    return x, y


def train_predict(train_rows: list[dict[str, str]], test_rows: list[dict[str, str]], features: dict[str, dict[str, str]]) -> tuple[list[int], list[float]]:
    try:
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
    except Exception as exc:
        raise RuntimeError("train_mri_feature_baseline requires scikit-learn") from exc

    x_train, y_train = matrix(train_rows, features)
    x_test, y_test = matrix(test_rows, features)
    model = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=17)),
        ]
    )
    model.fit(x_train, y_train)
    return y_test, [float(v) for v in model.predict_proba(x_test)[:, 1]]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", default="splits")
    ap.add_argument("--features", default="data/processed/mri_features.csv")
    ap.add_argument("--out_dir", default="results/final_results/mri_feature_baseline")
    args = ap.parse_args()

    split_dir = Path(args.splits)
    features = feature_index(Path(args.features))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fold_metrics = []
    all_y: list[int] = []
    all_p: list[float] = []
    for test_path in sorted(split_dir.glob("fold_*_test.csv")):
        fold = test_path.name.split("_")[1]
        train_rows = read_rows(split_dir / f"fold_{fold}_train.csv")
        test_rows = read_rows(test_path)
        y, p = train_predict(train_rows, test_rows, features)
        metrics = {
            "fold": int(fold),
            "n_test": len(y),
            "roc_auc": roc_auc(y, p),
            "pr_auc": average_precision(y, p),
            "brier": brier(y, p),
        }
        fold_metrics.append(metrics)
        all_y.extend(y)
        all_p.extend(p)
    aggregate = {
        "n": len(all_y),
        "roc_auc": roc_auc(all_y, all_p),
        "pr_auc": average_precision(all_y, all_p),
        "brier": brier(all_y, all_p),
        "roc_auc_ci": guarded_bootstrap_ci(all_y, all_p, roc_auc, n_boot=1000),
        "pr_auc_ci": guarded_bootstrap_ci(all_y, all_p, average_precision, n_boot=1000),
        "uses_pet_as_input": False,
        "uses_raw_identifiers": False,
        "model": "logistic_regression_tabular_plus_mri_intensity_features",
        "gate_1_passed": False,
    }
    aggregate["gate_1_passed"] = bool(aggregate["roc_auc"] >= 0.65)
    out = {"folds": fold_metrics, "aggregate": aggregate}
    atomic_write_json(out_dir / "mri_feature_baseline_metrics.json", out)
    lines = [
        "# MRI Feature Baseline Metrics\n\n",
        f"- model: `{aggregate['model']}`\n",
        f"- n: `{aggregate['n']}`\n",
        f"- roc_auc: `{aggregate['roc_auc']:.4f}`\n",
        f"- pr_auc: `{aggregate['pr_auc']:.4f}`\n",
        f"- brier: `{aggregate['brier']:.4f}`\n",
        f"- gate_1_passed: `{aggregate['gate_1_passed']}`\n",
        "- uses_pet_as_input: `False`\n",
        "- uses_raw_identifiers: `False`\n",
    ]
    atomic_write_text(out_dir / "mri_feature_baseline_metrics.md", "".join(lines))
    write_run_manifest("results/logs/mri_feature_baseline_manifest.json", stage="mri_feature_baseline", data_root=None, extra={"aggregate": aggregate})
    StateManager().append_event("mri_feature_baseline", "COMPLETED" if aggregate["gate_1_passed"] else "FAILED", "MRI feature baseline evaluated.")
    print(json.dumps({"roc_auc": aggregate["roc_auc"], "pr_auc": aggregate["pr_auc"], "gate_1_passed": aggregate["gate_1_passed"]}, indent=2))
    return 0 if aggregate["gate_1_passed"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
