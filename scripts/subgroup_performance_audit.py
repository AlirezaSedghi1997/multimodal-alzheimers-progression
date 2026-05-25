from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_external_eval import load_dev_rows
from scripts.run_fusion import fit_predict, matrix, read_features
from scripts.train_baseline import average_precision, brier, guarded_bootstrap_ci, roc_auc, safe_float
from src.utils.io_helpers import atomic_write_json, atomic_write_text
from src.utils.run_manifest import write_run_manifest
from src.utils.state_manager import StateManager


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def eligible_rows(rows: list[dict[str, str]], features: dict[str, dict[str, str]]) -> list[dict[str, str]]:
    return [row for row in rows if row.get("sample_hash") in features]


def sex_group(row: dict[str, str]) -> str:
    value = str(row.get("sex", "")).strip()
    if value == "1":
        return "sex_1"
    if value == "2":
        return "sex_2"
    return "sex_missing_or_other"


def age_group(row: dict[str, str]) -> str:
    age = safe_float(row.get("age"))
    if age < 70:
        return "age_lt_70"
    if age < 75:
        return "age_70_74"
    if age < 80:
        return "age_75_79"
    return "age_ge_80"


def apoe4_group(row: dict[str, str]) -> str:
    return "apoe4_carrier" if safe_float(row.get("apoe4_carrier")) >= 0.5 else "apoe4_non_carrier"


def education_group(row: dict[str, str]) -> str:
    years = safe_float(row.get("education_years"))
    if years <= 12:
        return "education_le_12"
    if years <= 16:
        return "education_13_16"
    return "education_gt_16"


def mmse_group(row: dict[str, str]) -> str:
    mmse = safe_float(row.get("mmse_base"))
    if mmse <= 27:
        return "mmse_le_27"
    if mmse <= 29:
        return "mmse_28_29"
    return "mmse_30"


GROUPERS = {
    "sex": sex_group,
    "age_bins": age_group,
    "apoe4": apoe4_group,
    "education": education_group,
    "mmse": mmse_group,
}


def safe_metric(fn, y: list[int], p: list[float]) -> float | None:
    try:
        return float(fn(y, p))
    except ValueError:
        return None


def summarize_group(y: list[int], p: list[float]) -> dict[str, Any]:
    label_counts = {str(label): y.count(label) for label in sorted(set(y))}
    single_class = len(set(y)) < 2
    out = {
        "n": len(y),
        "label_counts": label_counts,
        "roc_auc": None if single_class else safe_metric(roc_auc, y, p),
        "pr_auc": safe_metric(average_precision, y, p),
        "brier": brier(y, p) if y else None,
        "roc_auc_ci": None if single_class else guarded_bootstrap_ci(y, p, roc_auc, n_boot=1000),
        "pr_auc_ci": guarded_bootstrap_ci(y, p, average_precision, n_boot=1000) if len(set(y)) >= 2 else None,
        "small_n_caveat": len(y) < 30,
        "single_class_caveat": single_class,
    }
    return out


def subgroup_table(rows: list[dict[str, str]], y: list[int], p: list[float]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for group_name, group_fn in GROUPERS.items():
        buckets: dict[str, tuple[list[int], list[float]]] = {}
        for row, label, prob in zip(rows, y, p):
            bucket = group_fn(row)
            labels, probs = buckets.setdefault(bucket, ([], []))
            labels.append(label)
            probs.append(prob)
        out[group_name] = {bucket: summarize_group(labels, probs) for bucket, (labels, probs) in sorted(buckets.items())}
    return out


def cv_predictions(split_dir: Path, features: dict[str, dict[str, str]], variant: str, c_value: float):
    rows_out: list[dict[str, str]] = []
    y_out: list[int] = []
    p_out: list[float] = []
    for test_path in sorted(split_dir.glob("fold_*_test.csv")):
        fold = int(test_path.name.split("_")[1])
        train_rows = read_rows(split_dir / f"fold_{fold}_train.csv")
        test_rows = read_rows(test_path)
        y, p = fit_predict(train_rows, test_rows, features, variant, c_value)
        rows_out.extend(eligible_rows(test_rows, features))
        y_out.extend(y)
        p_out.extend(p)
    return rows_out, y_out, p_out


def external_predictions(split_dir: Path, features: dict[str, dict[str, str]], variant: str, c_value: float):
    try:
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
    except Exception as exc:
        raise RuntimeError("subgroup_performance_audit requires scikit-learn") from exc

    dev_rows = load_dev_rows(split_dir)
    external_rows = read_rows(split_dir / "external_holdout.csv")
    x_train, y_train = matrix(dev_rows, features, variant)
    x_ext, y_ext = matrix(external_rows, features, variant)
    model = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(C=c_value, class_weight="balanced", max_iter=1000, random_state=41)),
        ]
    )
    model.fit(x_train, y_train)
    probs = [float(v) for v in model.predict_proba(x_ext)[:, 1]]
    return eligible_rows(external_rows, features), y_ext, probs


def write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# Subgroup Performance Audit\n\n",
        f"- model_source: `{report['model_source']}`\n",
        f"- hard_fairness_gate: `{report['hard_fairness_gate']}`\n",
        f"- cohort_label_confound_warning: `{report['cohort_label_confound_warning']}`\n",
        "- uses_pet_as_input: `False`\n",
        "- uses_raw_identifiers: `False`\n\n",
    ]
    for split_name, split in report["splits"].items():
        lines.extend([f"## {split_name}\n\n", f"- n: `{split['n']}`\n\n"])
        for group_name, groups in split["subgroups"].items():
            lines.extend([f"### {group_name}\n\n", "| group | n | labels | roc_auc | pr_auc | brier | caveat |\n", "|---|---:|---|---:|---:|---:|---|\n"])
            for bucket, metrics in groups.items():
                caveats = []
                if metrics["small_n_caveat"]:
                    caveats.append("small_n")
                if metrics["single_class_caveat"]:
                    caveats.append("single_class")
                lines.append(
                    f"| {bucket} | {metrics['n']} | `{metrics['label_counts']}` | {metrics['roc_auc']} | {metrics['pr_auc']} | {metrics['brier']} | `{','.join(caveats)}` |\n"
                )
            lines.append("\n")
    atomic_write_text(path, "".join(lines))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", default="splits")
    ap.add_argument("--features", default="data/processed/mri_features.csv")
    ap.add_argument("--fusion_summary", default="results/final_results/fusion/fusion_summary.json")
    ap.add_argument("--out", default="results/final_results/subgroup_performance_audit.json")
    args = ap.parse_args()

    split_dir = Path(args.splits)
    features = read_features(Path(args.features))
    fusion = json.loads(Path(args.fusion_summary).read_text(encoding="utf-8"))
    variant = fusion["best"]["variant"]
    c_value = float(fusion["best"]["C"])
    cv_rows, cv_y, cv_p = cv_predictions(split_dir, features, variant, c_value)
    ext_rows, ext_y, ext_p = external_predictions(split_dir, features, variant, c_value)
    report = {
        "schema_version": 1,
        "model_source": f"{variant}_C_{c_value}",
        "hard_fairness_gate": False,
        "cohort_label_confound_warning": True,
        "external_holdout_type": "locked_pseudo_external_random_patient_holdout",
        "uses_pet_as_input": False,
        "uses_raw_identifiers": False,
        "splits": {
            "cv_oof": {"n": len(cv_y), "subgroups": subgroup_table(cv_rows, cv_y, cv_p)},
            "external_like": {"n": len(ext_y), "subgroups": subgroup_table(ext_rows, ext_y, ext_p)},
        },
        "interpretation": "Aggregate subgroup performance audit only; not a hard fairness certification.",
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(out, report)
    write_markdown(report, out.with_suffix(".md"))
    write_run_manifest("results/logs/subgroup_performance_audit_manifest.json", stage="subgroup_performance_audit", data_root=None, extra={"n_cv": len(cv_y), "n_external": len(ext_y)})
    StateManager().append_event("subgroup_performance_audit", "COMPLETED", "Aggregate subgroup performance audit generated.")
    print(json.dumps({"status": "COMPLETED", "n_cv": len(cv_y), "n_external": len(ext_y)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
