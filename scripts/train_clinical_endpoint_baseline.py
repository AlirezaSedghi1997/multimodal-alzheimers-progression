from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.train_bio_cog_method_z import (
    average_precision,
    brier,
    patient_grouped_folds,
    roc_auc,
)
from src.evaluation.bootstrap_ci import guarded_bootstrap_ci
from src.utils.io_helpers import atomic_write_json, atomic_write_text
from src.utils.run_manifest import write_run_manifest
from src.utils.state_manager import StateManager

TABULAR = ["age", "education_years", "apoe4_carrier", "mmse_base"]
ENDPOINTS = ("cdr_worsening_event", "pacc_decline_event")


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def matrix_rows(rows: list[dict[str, str]], features: list[str]) -> tuple[list[list[float]], list[int], list[dict[str, str]]]:
    kept: list[dict[str, str]] = []
    x: list[list[float]] = []
    y: list[int] = []
    for row in rows:
        label = row.get("label")
        if label is None or str(label).strip() == "":
            continue
        sex = row.get("sex", "")
        sex_val = 1.0 if str(sex).strip() in {"1", "M", "Male", "male"} else 0.0
        x.append([float(row.get(f) or 0.0) for f in features] + [sex_val])
        y.append(int(label))
        kept.append(row)
    return x, y, kept


def fit_predict_tabular(train_rows: list[dict[str, str]], test_rows: list[dict[str, str]], features: list[str]) -> tuple[list[int], list[float]]:
    try:
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
    except Exception as exc:
        raise RuntimeError("clinical endpoint baseline requires scikit-learn") from exc

    train = [{**row, "label": row["endpoint_label"]} for row in train_rows]
    test = [{**row, "label": row["endpoint_label"]} for row in test_rows]
    x_train, y_train, _ = matrix_rows(train, features)
    x_test, y_test, _ = matrix_rows(test, features)
    if not x_train or not x_test:
        return [], []
    model = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=13)),
        ]
    )
    model.fit(x_train, y_train)
    probs = [float(v) for v in model.predict_proba(x_test)[:, 1]]
    return y_test, probs


def evaluate_endpoint(rows: list[dict[str, str]], endpoint: str, *, k: int, seed: int) -> dict[str, Any]:
    labeled = [
        {**row, "endpoint_label": row[endpoint]}
        for row in rows
        if str(row.get(endpoint)).strip() in {"0", "1"}
    ]
    if not labeled:
        return {"status": "SKIPPED_NO_LABELS", "endpoint": endpoint}
    pos = sum(1 for row in labeled if row["endpoint_label"] == "1")
    neg = len(labeled) - pos
    if pos == 0 or neg == 0:
        return {"status": "SKIPPED_SINGLE_CLASS", "endpoint": endpoint, "n": len(labeled), "pos": pos, "neg": neg}
    all_y: list[int] = []
    all_p: list[float] = []
    for train_rows, test_rows in patient_grouped_folds(labeled, k, seed):
        y_true, y_prob = fit_predict_tabular(train_rows, test_rows, TABULAR)
        if not y_true:
            continue
        all_y.extend(y_true)
        all_p.extend(y_prob)
    if not all_y:
        return {"status": "SKIPPED_NO_PREDICTIONS", "endpoint": endpoint}
    return {
        "status": "OK",
        "endpoint": endpoint,
        "n": len(all_y),
        "pos": pos,
        "neg": neg,
        "roc_auc": roc_auc(all_y, all_p),
        "pr_auc": average_precision(all_y, all_p),
        "brier": brier(all_y, all_p),
        "roc_auc_ci": guarded_bootstrap_ci(all_y, all_p, roc_auc, n_boot=1000),
        "substudy_counts": dict(Counter(row["substudy"] for row in labeled)),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Clinical endpoint baseline modeling (enrolled only, CPU).")
    ap.add_argument("--labels", default="results/tables/clinical_endpoint_label_table.csv")
    ap.add_argument("--out_dir", default="results/final_results/clinical_endpoint")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    label_path = Path(args.labels)
    if not label_path.exists():
        raise RuntimeError(f"Missing labels: {label_path}. Run build_clinical_endpoint_labels.py first.")
    rows = [row for row in read_rows(label_path) if row.get("substudy") in {"A4", "LEARN"}]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {endpoint: evaluate_endpoint(rows, endpoint, k=args.folds, seed=args.seed) for endpoint in ENDPOINTS}
    summary: dict[str, Any] = {
        "schema_version": 1,
        "branch": "clinical_endpoint",
        "status": "COMPLETED",
        "phase": 3,
        "claim_boundary": (
            "Exploratory progression-endpoint modeling only; not Alzheimer diagnosis or Method X amyloid claim."
        ),
        "supports_method_x_strict_claim": False,
        "pet_as_input": False,
        "enrolled_rows": len(rows),
        "endpoints": results,
        "predictors": TABULAR + ["sex_binary"],
        "label_artifact": str(label_path),
    }
    atomic_write_json(out_dir / "clinical_endpoint_baseline_summary.json", summary)
    lines = ["# Clinical Endpoint Baseline (Enrolled A4/LEARN)\n\n", f"- status: `{summary['status']}`\n\n"]
    for endpoint, result in results.items():
        lines.append(
            f"## {endpoint}\n\n- status: `{result.get('status')}` "
            f"- roc_auc: `{result.get('roc_auc')}` - n: `{result.get('n')}`\n\n"
        )
    atomic_write_text(out_dir / "clinical_endpoint_baseline_summary.md", "".join(lines))
    write_run_manifest(
        "results/logs/clinical_endpoint_baseline_manifest.json",
        stage="clinical_endpoint_baseline",
        data_root=None,
        extra={"status": summary["status"]},
    )
    StateManager().append_event("clinical_endpoint_baseline", summary["status"], "Clinical endpoint baseline CV completed.")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
