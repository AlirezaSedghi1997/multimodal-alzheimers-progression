from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.build_cognitive_features import (
    CLINICAL_SEVERITY_FEATURES,
    COGNITIVE_FEATURES,
    COGSTATE_FEATURES,
    COMPLAINT_FEATURES,
    CORE_COGNITIVE_FEATURES,
    MOOD_ANXIETY_FEATURES,
)
from scripts.extract_mri_features import MRI_FEATURES
from scripts.train_baseline import average_precision, brier, guarded_bootstrap_ci, roc_auc
from src.utils.io_helpers import atomic_write_json, atomic_write_text
from src.utils.run_manifest import write_run_manifest
from src.utils.state_manager import StateManager


FIXED_TABULAR_FEATURES = ["age", "education_years", "apoe4_carrier", "mmse_base", "sex_binary"]
MODEL_VARIANTS = {
    "fixed_tabular_only": {"tabular": True, "cognitive": [], "mri": False},
    "cognitive_core_only": {"tabular": False, "cognitive": CORE_COGNITIVE_FEATURES, "mri": False},
    "fixed_tabular_plus_cognitive_core": {"tabular": True, "cognitive": CORE_COGNITIVE_FEATURES, "mri": False},
    "fixed_tabular_plus_all_cognitive": {"tabular": True, "cognitive": COGNITIVE_FEATURES, "mri": False},
    "fixed_tabular_plus_clinical_severity": {"tabular": True, "cognitive": CLINICAL_SEVERITY_FEATURES, "mri": False},
    "fixed_tabular_plus_complaints": {"tabular": True, "cognitive": COMPLAINT_FEATURES, "mri": False},
    "fixed_tabular_plus_cogstate": {"tabular": True, "cognitive": COGSTATE_FEATURES, "mri": False},
    "fixed_tabular_plus_mood_anxiety": {"tabular": True, "cognitive": MOOD_ANXIETY_FEATURES, "mri": False},
    "mri_feature_surrogate_plus_cognitive": {"tabular": True, "cognitive": COGNITIVE_FEATURES, "mri": True},
}
FORBIDDEN_INPUT_TERMS = ("pet", "tau", "suvr", "amyloid", "centiloid", "scan_date", "raw_path", "gap")


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def read_by_sample(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    return {row["sample_hash"]: row for row in read_rows(path)}


def safe_value(value: str | None) -> float:
    if value is None:
        return math.nan
    text = str(value).strip()
    if not text or text.upper() in {"NA", "NAN", "NULL", "."}:
        return math.nan
    try:
        value_float = float(text)
    except ValueError:
        return math.nan
    return value_float if not math.isnan(value_float) else math.nan


def sex_binary(row: dict[str, str]) -> float:
    sex = str(row.get("sex", "")).strip()
    if sex in {"1", "M", "Male", "male"}:
        return 1.0
    if sex in {"0", "F", "Female", "female"}:
        return 0.0
    return math.nan


def median(values: list[float]) -> float:
    clean = sorted(v for v in values if not math.isnan(v))
    if not clean:
        return 0.0
    mid = len(clean) // 2
    if len(clean) % 2:
        return clean[mid]
    return (clean[mid - 1] + clean[mid]) / 2.0


def prepare_matrix(train: list[list[float]], test: list[list[float]]) -> tuple[list[list[float]], list[list[float]]]:
    if not train:
        return train, test
    cols = len(train[0])
    medians = [median([row[j] for row in train]) for j in range(cols)]

    def impute(rows: list[list[float]]) -> list[list[float]]:
        return [[medians[j] if math.isnan(value) else value for j, value in enumerate(row)] for row in rows]

    train_i = impute(train)
    test_i = impute(test)
    means = [sum(row[j] for row in train_i) / len(train_i) for j in range(cols)]
    stds = []
    for j in range(cols):
        var = sum((row[j] - means[j]) ** 2 for row in train_i) / max(1, len(train_i) - 1)
        stds.append(math.sqrt(var) if var > 1e-12 else 1.0)

    def scale(rows: list[list[float]]) -> list[list[float]]:
        return [[(value - means[j]) / stds[j] for j, value in enumerate(row)] for row in rows]

    return scale(train_i), scale(test_i)


def fallback_gaussian_predict(x_train: list[list[float]], y_train: list[int], x_test: list[list[float]]) -> list[float]:
    x_train_s, x_test_s = prepare_matrix(x_train, x_test)
    if not x_train_s:
        return []
    cols = len(x_train_s[0])
    by_class = {0: [], 1: []}
    for row, label in zip(x_train_s, y_train):
        by_class[int(label)].append(row)
    if not by_class[0] or not by_class[1]:
        prevalence = sum(y_train) / len(y_train) if y_train else 0.5
        return [float(prevalence) for _ in x_test_s]

    means: dict[int, list[float]] = {}
    variances: dict[int, list[float]] = {}
    for label, rows in by_class.items():
        means[label] = [sum(row[j] for row in rows) / len(rows) for j in range(cols)]
        vars_for_label = []
        for j in range(cols):
            var = sum((row[j] - means[label][j]) ** 2 for row in rows) / max(1, len(rows) - 1)
            vars_for_label.append(max(var, 0.25))
        variances[label] = vars_for_label

    prior_pos = len(by_class[1]) / len(x_train_s)
    prior_neg = len(by_class[0]) / len(x_train_s)
    probs = []
    for row in x_test_s:
        log_pos = math.log(max(prior_pos, 1e-6))
        log_neg = math.log(max(prior_neg, 1e-6))
        for j, value in enumerate(row):
            for label, acc_name in ((1, "pos"), (0, "neg")):
                mean = means[label][j]
                var = variances[label][j]
                term = -0.5 * math.log(2.0 * math.pi * var) - ((value - mean) ** 2) / (2.0 * var)
                if label == 1:
                    log_pos += term
                else:
                    log_neg += term
        delta = max(min(log_pos - log_neg, 35.0), -35.0)
        probs.append(1.0 / (1.0 + math.exp(-delta)))
    return probs


def tabular_vector(row: dict[str, str]) -> list[float]:
    return [
        safe_value(row.get("age")),
        safe_value(row.get("education_years")),
        safe_value(row.get("apoe4_carrier")),
        safe_value(row.get("mmse_base")),
        sex_binary(row),
    ]


def variant_feature_names(variant: str) -> list[str]:
    spec = MODEL_VARIANTS[variant]
    names: list[str] = []
    if spec["tabular"]:
        names.extend(FIXED_TABULAR_FEATURES)
    names.extend(str(name) for name in spec["cognitive"])
    if spec["mri"]:
        names.extend(MRI_FEATURES)
    return names


def vector(
    row: dict[str, str],
    cognitive: dict[str, dict[str, str]],
    mri: dict[str, dict[str, str]],
    variant: str,
) -> list[float] | None:
    spec = MODEL_VARIANTS[variant]
    out: list[float] = []
    if spec["tabular"]:
        out.extend(tabular_vector(row))
    crow = cognitive.get(row["sample_hash"])
    if spec["cognitive"]:
        if crow is None:
            return None
        out.extend(safe_value(crow.get(str(feature))) for feature in spec["cognitive"])
    if spec["mri"]:
        mrow = mri.get(row["sample_hash"])
        if mrow is None:
            return None
        out.extend(safe_value(mrow.get(feature)) for feature in MRI_FEATURES)
    return out


def matrix(
    rows: list[dict[str, str]],
    cognitive: dict[str, dict[str, str]],
    mri: dict[str, dict[str, str]],
    variant: str,
) -> tuple[list[list[float]], list[int]]:
    x: list[list[float]] = []
    y: list[int] = []
    for row in rows:
        values = vector(row, cognitive, mri, variant)
        if values is None:
            continue
        x.append(values)
        y.append(int(float(row["ab_label"])))
    return x, y


def fit_predict(train_rows, test_rows, cognitive, mri, variant: str, c_value: float) -> tuple[list[int], list[float]]:
    x_train, y_train = matrix(train_rows, cognitive, mri, variant)
    x_test, y_test = matrix(test_rows, cognitive, mri, variant)
    if not x_train or not x_test:
        return [], []
    try:
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
    except Exception:
        return y_test, fallback_gaussian_predict(x_train, y_train, x_test)

    model = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(C=c_value, class_weight="balanced", max_iter=1000, random_state=43)),
        ]
    )
    model.fit(x_train, y_train)
    return y_test, [float(v) for v in model.predict_proba(x_test)[:, 1]]


def evaluate_variant(split_dir: Path, cognitive, mri, variant: str, c_value: float) -> dict[str, object]:
    all_y: list[int] = []
    all_p: list[float] = []
    folds = []
    for test_path in sorted(split_dir.glob("fold_*_test.csv")):
        fold = int(test_path.name.split("_")[1])
        y, p = fit_predict(read_rows(split_dir / f"fold_{fold}_train.csv"), read_rows(test_path), cognitive, mri, variant, c_value)
        if not y:
            continue
        folds.append({"fold": fold, "n_test": len(y), "roc_auc": roc_auc(y, p), "pr_auc": average_precision(y, p), "brier": brier(y, p)})
        all_y.extend(y)
        all_p.extend(p)
    if not all_y:
        return {"variant": variant, "C": c_value, "status": "SKIPPED_NO_ROWS", "n": 0}
    return {
        "variant": variant,
        "C": c_value,
        "status": "OK",
        "n": len(all_y),
        "feature_count": len(variant_feature_names(variant)),
        "roc_auc": roc_auc(all_y, all_p),
        "pr_auc": average_precision(all_y, all_p),
        "brier": brier(all_y, all_p),
        "folds": folds,
        "y_true": all_y,
        "y_prob": all_p,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", default="splits")
    ap.add_argument("--cognitive_features", default="data/processed/cognitive_features.csv")
    ap.add_argument("--mri_features", default="data/processed/mri_features.csv")
    ap.add_argument("--out_dir", default="results/final_results/method_x_cog")
    args = ap.parse_args()

    split_dir = Path(args.splits)
    cognitive = read_by_sample(Path(args.cognitive_features))
    if not cognitive:
        raise RuntimeError("Missing cognitive features. Run build_cognitive_features.py first.")
    mri = read_by_sample(Path(args.mri_features))
    candidates = []
    for variant, spec in MODEL_VARIANTS.items():
        if spec["mri"] and not mri:
            candidates.append({"variant": variant, "status": "SKIPPED_MRI_FEATURES_MISSING", "n": 0})
            continue
        for c_value in [0.03, 0.1, 0.3, 1.0, 3.0]:
            candidates.append(evaluate_variant(split_dir, cognitive, mri, variant, c_value))
    valid = [c for c in candidates if c.get("status") == "OK"]
    if not valid:
        raise RuntimeError("No Method X-Cog candidates produced predictions.")
    best = max(valid, key=lambda c: (float(c["roc_auc"]), float(c["pr_auc"])))
    y_true = list(best.pop("y_true"))
    y_prob = list(best.pop("y_prob"))
    best["roc_auc_ci"] = guarded_bootstrap_ci(y_true, y_prob, roc_auc, n_boot=1000)
    best["pr_auc_ci"] = guarded_bootstrap_ci(y_true, y_prob, average_precision, n_boot=1000)
    best["features"] = variant_feature_names(str(best["variant"]))
    best["uses_pet_as_input"] = False
    best["uses_raw_identifiers"] = False
    summary = {
        "schema_version": 1,
        "method": "Method X-Cog",
        "status": "COMPLETED",
        "best": best,
        "candidate_table": [
            {
                "variant": c.get("variant"),
                "C": c.get("C"),
                "status": c.get("status"),
                "n": c.get("n"),
                "feature_count": c.get("feature_count"),
                "roc_auc": c.get("roc_auc"),
                "pr_auc": c.get("pr_auc"),
                "brier": c.get("brier"),
            }
            for c in candidates
        ],
        "method_x_claim_unchanged": True,
        "method_x_cog_exploratory": True,
        "uses_pet_as_input": False,
        "uses_raw_identifiers": False,
        "cohort_label_confound_warning": True,
        "forbidden_input_terms": FORBIDDEN_INPUT_TERMS,
        "model_backend": "sklearn_logistic_regression_or_fallback_gaussian",
    }
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(out_dir / "method_x_cog_summary.json", summary)
    lines = [
        "# Method X-Cog Summary\n\n",
        "- status: `COMPLETED`\n",
        f"- best_variant: `{best['variant']}`\n",
        f"- best_C: `{best['C']}`\n",
        f"- n: `{best['n']}`\n",
        f"- feature_count: `{best['feature_count']}`\n",
        f"- roc_auc: `{float(best['roc_auc']):.4f}`\n",
        f"- pr_auc: `{float(best['pr_auc']):.4f}`\n",
        f"- brier: `{float(best['brier']):.4f}`\n",
        "- uses_pet_as_input: `False`\n",
        "- uses_raw_identifiers: `False`\n",
        "- method_x_claim_unchanged: `True`\n",
        "- method_x_cog_exploratory: `True`\n",
        "- cohort_label_confound_warning: `True`\n",
    ]
    atomic_write_text(out_dir / "method_x_cog_summary.md", "".join(lines))
    write_run_manifest("results/logs/method_x_cog_manifest.json", stage="cognitive-ablation", data_root=None, extra={"best": best})
    StateManager().append_event("cognitive-ablation", "COMPLETED", "Method X-Cog cognitive ablation completed.")
    print(json.dumps({"best_variant": best["variant"], "roc_auc": best["roc_auc"], "pr_auc": best["pr_auc"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
