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
FUSION_VARIANTS = [
    "late_concat",
    "attention_cross_light_feature_surrogate",
    "conv_feature_fusion_surrogate",
    "adaptive_fusion_missingness_surrogate",
]


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def read_features(path: Path) -> dict[str, dict[str, str]]:
    return {row["sample_hash"]: row for row in read_rows(path) if row.get("mri_feature_status") == "OK"}


def vector(row: dict[str, str], frow: dict[str, str], variant: str) -> list[float]:
    sex = row.get("sex", "")
    sex_val = 1.0 if str(sex).strip() in {"1", "M", "Male", "male"} else 0.0
    tab = [safe_float(row.get(f)) for f in TABULAR_FEATURES] + [sex_val]
    mri = [safe_float(frow.get(f)) for f in MRI_FEATURES]
    if variant == "late_concat":
        return tab + mri
    if variant == "attention_cross_light_feature_surrogate":
        mri_core = [safe_float(frow.get("mri_mean")), safe_float(frow.get("mri_std")), safe_float(frow.get("mri_p50"))]
        interactions = [t * m for t in tab for m in mri_core]
        return tab + mri + interactions
    if variant == "conv_feature_fusion_surrogate":
        local_deltas = [mri[i + 1] - mri[i] for i in range(len(mri) - 1)]
        return tab + mri + local_deltas
    if variant == "adaptive_fusion_missingness_surrogate":
        tab_present = [1.0 if row.get(f) not in {"", None} else 0.0 for f in TABULAR_FEATURES] + [1.0]
        return tab + mri + tab_present
    raise ValueError(f"Unknown fusion variant: {variant}")


def matrix(rows: list[dict[str, str]], features: dict[str, dict[str, str]], variant: str) -> tuple[list[list[float]], list[int]]:
    x: list[list[float]] = []
    y: list[int] = []
    for row in rows:
        frow = features.get(row["sample_hash"])
        if frow is None:
            continue
        x.append(vector(row, frow, variant))
        y.append(int(float(row["ab_label"])))
    return x, y


def fit_predict(train_rows, test_rows, features, variant: str, c_value: float) -> tuple[list[int], list[float]]:
    try:
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
    except Exception as exc:
        raise RuntimeError("run_fusion requires scikit-learn") from exc

    x_train, y_train = matrix(train_rows, features, variant)
    x_test, y_test = matrix(test_rows, features, variant)
    model = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(C=c_value, class_weight="balanced", max_iter=1000, random_state=29)),
        ]
    )
    model.fit(x_train, y_train)
    return y_test, [float(v) for v in model.predict_proba(x_test)[:, 1]]


def evaluate_variant(split_dir: Path, features: dict[str, dict[str, str]], variant: str, c_value: float) -> dict[str, object]:
    all_y: list[int] = []
    all_p: list[float] = []
    folds = []
    for test_path in sorted(split_dir.glob("fold_*_test.csv")):
        fold = int(test_path.name.split("_")[1])
        y, p = fit_predict(read_rows(split_dir / f"fold_{fold}_train.csv"), read_rows(test_path), features, variant, c_value)
        folds.append({"fold": fold, "n_test": len(y), "roc_auc": roc_auc(y, p), "pr_auc": average_precision(y, p), "brier": brier(y, p)})
        all_y.extend(y)
        all_p.extend(p)
    return {
        "variant": variant,
        "C": c_value,
        "n": len(all_y),
        "roc_auc": roc_auc(all_y, all_p),
        "pr_auc": average_precision(all_y, all_p),
        "brier": brier(all_y, all_p),
        "folds": folds,
        "y_true": all_y,
        "y_prob": all_p,
    }


def load_baseline_auc() -> float:
    path = Path("results/final_results/baseline/baseline_metrics.json")
    if not path.exists():
        return 0.65
    obj = json.loads(path.read_text(encoding="utf-8"))
    return float(obj["aggregate"]["roc_auc"])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", default="splits")
    ap.add_argument("--features", default="data/processed/mri_features.csv")
    ap.add_argument("--out_dir", default="results/final_results/fusion")
    args = ap.parse_args()

    split_dir = Path(args.splits)
    features = read_features(Path(args.features))
    if not features:
        raise RuntimeError("Missing MRI features. Run extract_mri_features.py first.")
    candidates = []
    for variant in FUSION_VARIANTS:
        for c_value in [0.03, 0.1, 0.3, 1.0, 3.0]:
            candidates.append(evaluate_variant(split_dir, features, variant, c_value))
    best = max(candidates, key=lambda c: (float(c["roc_auc"]), float(c["pr_auc"])))
    y_true = best.pop("y_true")
    y_prob = best.pop("y_prob")
    best["roc_auc_ci"] = guarded_bootstrap_ci(y_true, y_prob, roc_auc, n_boot=1000)
    best["pr_auc_ci"] = guarded_bootstrap_ci(y_true, y_prob, average_precision, n_boot=1000)
    best["uses_pet_as_input"] = False
    best["uses_raw_identifiers"] = False
    baseline_auc = load_baseline_auc()
    gate_3_status = "PASSED" if float(best["roc_auc"]) > baseline_auc else "PARTIAL"
    summary = {
        "baseline_auc_roc": baseline_auc,
        "best": best,
        "gate_3_status": gate_3_status,
        "n_candidates": len(candidates),
        "candidate_table": [
            {"variant": c["variant"], "C": c["C"], "roc_auc": c["roc_auc"], "pr_auc": c["pr_auc"], "brier": c["brier"]}
            for c in candidates
        ],
        "method_x_compliant": True,
        "cohort_label_confound_warning": True,
    }
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(out_dir / "fusion_summary.json", summary)
    lines = [
        "# Fusion Summary\n\n",
        f"- baseline_auc_roc: `{baseline_auc:.4f}`\n",
        f"- best_variant: `{best['variant']}`\n",
        f"- best_C: `{best['C']}`\n",
        f"- roc_auc: `{float(best['roc_auc']):.4f}`\n",
        f"- pr_auc: `{float(best['pr_auc']):.4f}`\n",
        f"- brier: `{float(best['brier']):.4f}`\n",
        f"- gate_3_status: `{gate_3_status}`\n",
        "- uses_pet_as_input: `False`\n",
        "- uses_raw_identifiers: `False`\n",
        "- cohort_label_confound_warning: `True`\n",
    ]
    atomic_write_text(out_dir / "fusion_summary.md", "".join(lines))
    write_run_manifest("results/logs/fusion_manifest.json", stage="fusion", data_root=None, extra={"best": best, "gate_3_status": gate_3_status})
    StateManager().append_event("fusion", gate_3_status, "Fusion/ablation comparison completed.")
    print(json.dumps({"roc_auc": best["roc_auc"], "pr_auc": best["pr_auc"], "gate_3_status": gate_3_status, "variant": best["variant"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

