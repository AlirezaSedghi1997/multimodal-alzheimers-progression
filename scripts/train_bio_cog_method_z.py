from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.bootstrap_ci import guarded_bootstrap_ci
from src.utils.io_helpers import atomic_write_json, atomic_write_text
from src.utils.run_manifest import write_run_manifest
from src.utils.state_manager import StateManager

BLOOD_CORE = ["pTau217", "GFAP"]
BLOOD_FULL = ["pTau217", "GFAP", "NfL", "pTau181", "Ab42_40_ratio"]
VARIANTS = {
    "core_blood_only": BLOOD_CORE,
    "full_blood_panel": BLOOD_FULL,
}
DECOUPLING_DISCLAIMER = (
    "Internal decoupling / stress-test only. SF is not external validation and must never "
    "be described as independent or quasi-external validation."
)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def present(value: str | None) -> bool:
    if value is None:
        return False
    text = str(value).strip()
    return text not in {"", "NA", "NAN", "NULL"}


def load_labeled_rows(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for row in read_rows(path):
        label = row.get("ab_label_primary") or row.get("ab_label")
        if not present(label):
            continue
        normalized = dict(row)
        normalized["ab_label"] = str(int(float(label)))
        rows.append(normalized)
    return rows


def subset(rows: list[dict[str, str]], substudies: set[str] | None) -> list[dict[str, str]]:
    if substudies is None:
        return rows
    return [row for row in rows if row.get("substudy") in substudies]


def usable_rows(rows: list[dict[str, str]], features: list[str]) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for row in rows:
        if any(present(row.get(feature)) for feature in features):
            out.append(row)
    return out


def patient_grouped_folds(rows: list[dict[str, str]], k: int, seed: int) -> list[tuple[list[dict[str, str]], list[dict[str, str]]]]:
    rng = random.Random(seed)
    by_patient: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        by_patient.setdefault(row["patient_hash"], []).append(row)
    patients = list(by_patient)
    rng.shuffle(patients)
    fold_patients: list[list[str]] = [[] for _ in range(k)]
    for i, patient in enumerate(patients):
        fold_patients[i % k].append(patient)
    folds: list[tuple[list[dict[str, str]], list[dict[str, str]]]] = []
    for test_patients in fold_patients:
        test_ids = set(test_patients)
        test_rows = [row for patient in test_patients for row in by_patient[patient]]
        train_rows = [row for patient in patients if patient not in test_ids for row in by_patient[patient]]
        folds.append((train_rows, test_rows))
    return folds


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


def fit_predict(train_rows: list[dict[str, str]], test_rows: list[dict[str, str]], features: list[str]) -> tuple[list[int], list[float]]:
    try:
        import numpy as np
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
    except Exception as exc:
        raise RuntimeError("train_bio_cog_method_z requires scikit-learn") from exc

    def matrix(rows: list[dict[str, str]]) -> tuple[list[list[float]], list[int]]:
        x: list[list[float]] = []
        y: list[int] = []
        for row in rows:
            if not any(present(row.get(feature)) for feature in features):
                continue
            x.append([float(row[feature]) if present(row.get(feature)) else float("nan") for feature in features])
            y.append(int(row["ab_label"]))
        return x, y

    x_train, y_train = matrix(train_rows)
    x_test, y_test = matrix(test_rows)
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


def evaluate_cv(rows: list[dict[str, str]], features: list[str], *, k: int, seed: int) -> dict[str, Any]:
    all_y: list[int] = []
    all_p: list[float] = []
    fold_rows: list[dict[str, Any]] = []
    for fold_idx, (train_rows, test_rows) in enumerate(patient_grouped_folds(rows, k, seed)):
        y_true, y_prob = fit_predict(train_rows, test_rows, features)
        if not y_true:
            continue
        fold_rows.append(
            {
                "fold": fold_idx,
                "n_train": len(train_rows),
                "n_test": len(y_true),
                "roc_auc": roc_auc(y_true, y_prob),
                "pr_auc": average_precision(y_true, y_prob),
                "brier": brier(y_true, y_prob),
            }
        )
        all_y.extend(y_true)
        all_p.extend(y_prob)
    if not all_y:
        return {"status": "SKIPPED_NO_ROWS", "n": 0, "folds": fold_rows}
    return {
        "status": "OK",
        "n": len(all_y),
        "feature_count": len(features),
        "features": features,
        "folds": fold_rows,
        "roc_auc": roc_auc(all_y, all_p),
        "pr_auc": average_precision(all_y, all_p),
        "brier": brier(all_y, all_p),
        "roc_auc_ci": guarded_bootstrap_ci(all_y, all_p, roc_auc, n_boot=1000),
    }


def cramers_v(rows: list[dict[str, str]]) -> float | None:
    if not rows:
        return None
    table: dict[str, Counter[str]] = {}
    for row in rows:
        table.setdefault(row.get("substudy", ""), Counter())[row.get("ab_label", "")] += 1
    substudies = sorted(table)
    labels = sorted({label for counts in table.values() for label in counts})
    if len(substudies) < 2 or len(labels) < 2:
        return None
    n = sum(sum(counts.values()) for counts in table.values())
    observed = [[table[s].get(l, 0) for l in labels] for s in substudies]
    row_totals = [sum(r) for r in observed]
    col_totals = [sum(observed[i][j] for i in range(len(substudies))) for j in range(len(labels))]
    if min(row_totals) == 0 or min(col_totals) == 0:
        return None
    expected = [[row_totals[i] * col_totals[j] / n for j in range(len(labels))] for i in range(len(substudies))]
    chi2 = sum(
        ((observed[i][j] - expected[i][j]) ** 2) / expected[i][j]
        for i in range(len(substudies))
        for j in range(len(labels))
        if expected[i][j] > 0
    )
    k = min(len(substudies) - 1, len(labels) - 1)
    if k <= 0 or n <= 1:
        return None
    return math.sqrt(chi2 / (n * k))


def main() -> int:
    ap = argparse.ArgumentParser(description="Phase 3 Bio-Cog/Method Z blood branch training (CPU-only).")
    ap.add_argument("--catalog", default="data/processed/catalogs/bio_cog_method_z_catalog.csv")
    ap.add_argument("--out_dir", default="results/final_results/bio_cog_method_z")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    catalog_path = Path(args.catalog)
    if not catalog_path.exists():
        raise RuntimeError(f"Missing catalog: {catalog_path}")
    all_rows = load_labeled_rows(catalog_path)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    arms: dict[str, Any] = {}
    for arm_name, substudies in {
        "enrolled_a4_learn": {"A4", "LEARN"},
        "sf_inclusive": {"A4", "LEARN", "SF"},
    }.items():
        base = subset(all_rows, substudies)
        arms[arm_name] = {
            "rows": len(base),
            "cohort_label_cramers_v": cramers_v(base),
            "variants": {},
        }
        for variant_name, features in VARIANTS.items():
            rows = usable_rows(base, features)
            arms[arm_name]["variants"][variant_name] = evaluate_cv(rows, features, k=args.folds, seed=args.seed)

    sf_rows = usable_rows(subset(all_rows, {"SF"}), BLOOD_FULL)
    enrolled_rows = usable_rows(subset(all_rows, {"A4", "LEARN"}), BLOOD_FULL)
    train_sf_test_enrolled = {"status": "SKIPPED_NO_ROWS"}
    if sf_rows and enrolled_rows:
        y_true, y_prob = fit_predict(sf_rows, enrolled_rows, BLOOD_FULL)
        if y_true:
            train_sf_test_enrolled = {
                "status": "OK",
                "n_train": len(sf_rows),
                "n_test": len(y_true),
                "variant": "full_blood_panel",
                "roc_auc": roc_auc(y_true, y_prob),
                "pr_auc": average_precision(y_true, y_prob),
                "brier": brier(y_true, y_prob),
                "decoupling_disclaimer": DECOUPLING_DISCLAIMER,
            }

    best = None
    for arm in arms.values():
        for variant in arm["variants"].values():
            variant.pop("y_true", None)
            variant.pop("y_prob", None)
            if variant.get("status") != "OK":
                continue
            if best is None or float(variant["roc_auc"]) > float(best["roc_auc"]):
                best = {k: v for k, v in variant.items() if k not in {"y_true", "y_prob"}}

    summary: dict[str, Any] = {
        "schema_version": 1,
        "branch": "bio_cog_method_z",
        "status": "COMPLETED",
        "phase": 3,
        "claim_boundary": (
            "Minimally invasive biomarker-assisted branch; never supports the non-invasive Method X claim."
        ),
        "supports_method_x_strict_claim": False,
        "pet_as_method_x_input": False,
        "uses_pet_as_input": False,
        "uses_raw_identifiers": False,
        "catalog_artifact": str(catalog_path),
        "arms": arms,
        "train_sf_test_enrolled_full_blood": train_sf_test_enrolled,
        "best_cv_result": best,
        "decoupling_disclaimer": DECOUPLING_DISCLAIMER,
    }

    atomic_write_json(out_dir / "bio_cog_method_z_summary.json", summary)
    lines = [
        "# Bio-Cog / Method Z — Phase 3 Blood Branch\n\n",
        f"- status: `{summary['status']}`\n",
        f"- claim boundary: {summary['claim_boundary']}\n\n",
    ]
    for arm_name, arm in arms.items():
        lines.append(f"## {arm_name}\n\n")
        lines.append(f"- rows: `{arm['rows']}`\n")
        lines.append(f"- cohort-label Cramér's V: `{arm.get('cohort_label_cramers_v')}`\n\n")
        for variant_name, variant in arm["variants"].items():
            lines.append(
                f"- **{variant_name}**: status=`{variant.get('status')}` "
                f"roc_auc=`{variant.get('roc_auc')}` n=`{variant.get('n')}`\n"
            )
        lines.append("\n")
    lines.append(f"## train-SF → test-enrolled\n\n`{train_sf_test_enrolled}`\n")
    atomic_write_text(out_dir / "bio_cog_method_z_summary.md", "".join(lines))

    write_run_manifest(
        "results/logs/bio_cog_method_z_training_manifest.json",
        stage="bio_cog_method_z_training",
        data_root=None,
        extra={"status": summary["status"], "best_roc_auc": (best or {}).get("roc_auc")},
    )
    StateManager().append_event(
        "bio_cog_method_z_training",
        summary["status"],
        "Phase 3 Bio-Cog/Method Z blood branch CV training completed.",
    )
    print(json.dumps({"status": summary["status"], "best_roc_auc": (best or {}).get("roc_auc")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
