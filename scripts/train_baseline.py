from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.bootstrap_ci import guarded_bootstrap_ci
from src.utils.io_helpers import atomic_write_json, atomic_write_text
from src.utils.run_manifest import write_run_manifest
from src.utils.state_manager import StateManager


FEATURES = [
    "age",
    "education_years",
    "apoe4_carrier",
    "mmse_base",
]


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def safe_float(value: str | None) -> float:
    if value is None or value == "":
        return 0.0
    try:
        return float(value)
    except ValueError:
        return 0.0


def matrix(rows: list[dict[str, str]]) -> tuple[list[list[float]], list[int]]:
    x: list[list[float]] = []
    y: list[int] = []
    for row in rows:
        sex = row.get("sex", "")
        sex_val = 1.0 if str(sex).strip() in {"1", "M", "Male", "male"} else 0.0
        x.append([safe_float(row.get(f)) for f in FEATURES] + [sex_val])
        y.append(int(float(row["ab_label"])))
    return x, y


def roc_auc(y_true: list[int], y_score: list[float]) -> float:
    pos = [s for y, s in zip(y_true, y_score) if y == 1]
    neg = [s for y, s in zip(y_true, y_score) if y == 0]
    if not pos or not neg:
        raise ValueError("ROC-AUC requires both classes")
    wins = 0.0
    total = len(pos) * len(neg)
    for p in pos:
        for n in neg:
            if p > n:
                wins += 1.0
            elif p == n:
                wins += 0.5
    return wins / total


def average_precision(y_true: list[int], y_score: list[float]) -> float:
    pairs = sorted(zip(y_score, y_true), reverse=True)
    total_pos = sum(y_true)
    if total_pos == 0:
        raise ValueError("PR-AUC requires positives")
    hits = 0
    precisions = []
    for i, (_, y) in enumerate(pairs, 1):
        if y == 1:
            hits += 1
            precisions.append(hits / i)
    return sum(precisions) / total_pos


def brier(y_true: list[int], y_score: list[float]) -> float:
    return sum((float(y) - float(s)) ** 2 for y, s in zip(y_true, y_score)) / len(y_true)


def train_predict(train_rows: list[dict[str, str]], test_rows: list[dict[str, str]]) -> list[float]:
    try:
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
    except Exception as exc:
        raise RuntimeError("train_baseline requires scikit-learn") from exc

    x_train, y_train = matrix(train_rows)
    x_test, _ = matrix(test_rows)
    model = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=13)),
        ]
    )
    model.fit(x_train, y_train)
    return [float(v) for v in model.predict_proba(x_test)[:, 1]]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", default="splits")
    ap.add_argument("--out_dir", default="results/final_results/baseline")
    args = ap.parse_args()
    split_dir = Path(args.splits)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fold_metrics = []
    all_y: list[int] = []
    all_p: list[float] = []
    for test_path in sorted(split_dir.glob("fold_*_test.csv")):
        fold = test_path.name.split("_")[1]
        train_path = split_dir / f"fold_{fold}_train.csv"
        train_rows = read_rows(train_path)
        test_rows = read_rows(test_path)
        probs = train_predict(train_rows, test_rows)
        _, y = matrix(test_rows)
        metrics = {
            "fold": int(fold),
            "n_test": len(y),
            "roc_auc": roc_auc(y, probs),
            "pr_auc": average_precision(y, probs),
            "brier": brier(y, probs),
        }
        fold_metrics.append(metrics)
        all_y.extend(y)
        all_p.extend(probs)

    aggregate = {
        "n": len(all_y),
        "roc_auc": roc_auc(all_y, all_p),
        "pr_auc": average_precision(all_y, all_p),
        "brier": brier(all_y, all_p),
        "roc_auc_ci": guarded_bootstrap_ci(all_y, all_p, roc_auc, n_boot=1000),
        "pr_auc_ci": guarded_bootstrap_ci(all_y, all_p, average_precision, n_boot=1000),
        "features": FEATURES + ["sex_binary"],
        "uses_pet_as_input": False,
        "uses_raw_identifiers": False,
        "model": "logistic_regression_tabular_only_method_x_baseline",
        "gate_1_passed": False,
    }
    aggregate["gate_1_passed"] = bool(aggregate["roc_auc"] >= 0.65)
    out = {"folds": fold_metrics, "aggregate": aggregate}
    atomic_write_json(out_dir / "baseline_metrics.json", out)
    md = [
        "# Baseline Metrics\n\n",
        f"- model: `{aggregate['model']}`\n",
        f"- n: `{aggregate['n']}`\n",
        f"- roc_auc: `{aggregate['roc_auc']:.4f}`\n",
        f"- pr_auc: `{aggregate['pr_auc']:.4f}`\n",
        f"- brier: `{aggregate['brier']:.4f}`\n",
        f"- gate_1_passed: `{aggregate['gate_1_passed']}`\n",
        "- uses_pet_as_input: `False`\n",
        "- uses_raw_identifiers: `False`\n",
        "\n## Fold Metrics\n\n",
        "| fold | n_test | roc_auc | pr_auc | brier |\n",
        "|---:|---:|---:|---:|---:|\n",
    ]
    for m in fold_metrics:
        md.append(f"| {m['fold']} | {m['n_test']} | {m['roc_auc']:.4f} | {m['pr_auc']:.4f} | {m['brier']:.4f} |\n")
    atomic_write_text(out_dir / "baseline_metrics.md", "".join(md))
    write_run_manifest("results/logs/baseline_manifest.json", stage="baseline", data_root=None, extra={"aggregate": aggregate})
    StateManager().append_event("baseline", "COMPLETED" if aggregate["gate_1_passed"] else "FAILED", "Baseline tabular-only model evaluated.")
    print(json.dumps({"roc_auc": aggregate["roc_auc"], "pr_auc": aggregate["pr_auc"], "gate_1_passed": aggregate["gate_1_passed"]}, indent=2))
    return 0 if aggregate["gate_1_passed"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
