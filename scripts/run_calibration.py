from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_fusion import read_features, vector
from scripts.train_baseline import average_precision, brier, guarded_bootstrap_ci, roc_auc
from src.utils.io_helpers import atomic_write_json, atomic_write_text
from src.utils.run_manifest import write_run_manifest
from src.utils.state_manager import StateManager


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def stratified_cal_split(rows: list[dict[str, str]], fraction: float = 0.2, seed: int = 41) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    rng = random.Random(seed)
    by_label: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_label[row["ab_label"]].append(row)
    train: list[dict[str, str]] = []
    cal: list[dict[str, str]] = []
    for label_rows in by_label.values():
        shuffled = list(label_rows)
        rng.shuffle(shuffled)
        n_cal = max(1, round(len(shuffled) * fraction))
        cal.extend(shuffled[:n_cal])
        train.extend(shuffled[n_cal:])
    rng.shuffle(train)
    rng.shuffle(cal)
    return train, cal


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


def ece(y_true: list[int], y_prob: list[float], n_bins: int = 10) -> float:
    total = len(y_true)
    if total == 0:
        return 0.0
    score = 0.0
    for i in range(n_bins):
        lo = i / n_bins
        hi = (i + 1) / n_bins
        idx = [j for j, p in enumerate(y_prob) if (p >= lo and (p < hi or i == n_bins - 1))]
        if not idx:
            continue
        conf = sum(y_prob[j] for j in idx) / len(idx)
        acc = sum(y_true[j] for j in idx) / len(idx)
        score += (len(idx) / total) * abs(acc - conf)
    return score


def fit_base_model(x_train, y_train, c_value: float):
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    model = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(C=c_value, class_weight="balanced", max_iter=1000, random_state=43)),
        ]
    )
    model.fit(x_train, y_train)
    return model


def fit_calibrator(prob, y):
    from sklearn.linear_model import LogisticRegression

    cal = LogisticRegression(C=1.0, max_iter=1000, random_state=47)
    cal.fit([[float(p)] for p in prob], y)
    return cal


def apply_calibrator(cal, prob: list[float]) -> list[float]:
    return [float(v) for v in cal.predict_proba([[float(p)] for p in prob])[:, 1]]


def summarize(y: list[int], prob: list[float]) -> dict[str, object]:
    return {
        "n": len(y),
        "roc_auc": roc_auc(y, prob) if len(set(y)) > 1 else None,
        "pr_auc": average_precision(y, prob) if len(set(y)) > 1 else None,
        "brier": brier(y, prob),
        "ece": ece(y, prob),
        "roc_auc_ci": guarded_bootstrap_ci(y, prob, roc_auc, n_boot=1000) if len(set(y)) > 1 else None,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", default="splits")
    ap.add_argument("--features", default="data/processed/mri_features.csv")
    ap.add_argument("--fusion_summary", default="results/final_results/fusion/fusion_summary.json")
    ap.add_argument("--out_dir", default="results/final_results/calibration")
    args = ap.parse_args()

    fusion = json.loads(Path(args.fusion_summary).read_text(encoding="utf-8"))
    variant = fusion["best"]["variant"]
    c_value = float(fusion["best"]["C"])
    features = read_features(Path(args.features))
    split_dir = Path(args.splits)

    cv_uncal_y: list[int] = []
    cv_uncal_p: list[float] = []
    cv_cal_y: list[int] = []
    cv_cal_p: list[float] = []
    fit_log = []
    for test_path in sorted(split_dir.glob("fold_*_test.csv")):
        fold = int(test_path.name.split("_")[1])
        train_rows, cal_rows = stratified_cal_split(read_rows(split_dir / f"fold_{fold}_train.csv"), seed=100 + fold)
        test_rows = read_rows(test_path)
        x_train, y_train = matrix(train_rows, features, variant)
        x_cal, y_cal = matrix(cal_rows, features, variant)
        x_test, y_test = matrix(test_rows, features, variant)
        model = fit_base_model(x_train, y_train, c_value)
        cal_prob = [float(v) for v in model.predict_proba(x_cal)[:, 1]]
        test_prob = [float(v) for v in model.predict_proba(x_test)[:, 1]]
        calibrator = fit_calibrator(cal_prob, y_cal)
        test_cal = apply_calibrator(calibrator, test_prob)
        cv_uncal_y.extend(y_test)
        cv_uncal_p.extend(test_prob)
        cv_cal_y.extend(y_test)
        cv_cal_p.extend(test_cal)
        fit_log.append({"fold": fold, "fit_data_source": "fold_train_internal_calibration_split", "external_used_for_fit": False, "n_cal": len(y_cal)})

    # Final pseudo-external calibration: train/cal split is derived only from dev rows.
    dev_rows = []
    dev_rows.extend(read_rows(split_dir / "fold_0_train.csv"))
    dev_rows.extend(read_rows(split_dir / "fold_0_test.csv"))
    dev_unique = {row["sample_hash"]: row for row in dev_rows}
    final_train, final_cal = stratified_cal_split(list(dev_unique.values()), seed=500)
    external_rows = read_rows(split_dir / "external_holdout.csv")
    x_train, y_train = matrix(final_train, features, variant)
    x_cal, y_cal = matrix(final_cal, features, variant)
    x_ext, y_ext = matrix(external_rows, features, variant)
    final_model = fit_base_model(x_train, y_train, c_value)
    ext_prob = [float(v) for v in final_model.predict_proba(x_ext)[:, 1]]
    cal_prob = [float(v) for v in final_model.predict_proba(x_cal)[:, 1]]
    final_calibrator = fit_calibrator(cal_prob, y_cal)
    ext_cal = apply_calibrator(final_calibrator, ext_prob)
    fit_log.append({"fold": "external_like", "fit_data_source": "dev_internal_calibration_split", "external_used_for_fit": False, "n_cal": len(y_cal)})

    report = {
        "variant": variant,
        "C": c_value,
        "cv_uncalibrated": summarize(cv_uncal_y, cv_uncal_p),
        "cv_calibrated": summarize(cv_cal_y, cv_cal_p),
        "external_like_uncalibrated": summarize(y_ext, ext_prob),
        "external_like_calibrated": summarize(y_ext, ext_cal),
        "fit_log": fit_log,
        "external_used_for_fit": False,
        "uses_pet_as_input": False,
        "uses_raw_identifiers": False,
        "cohort_label_confound_warning": True,
    }
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(out_dir / "calibration_report.json", report)
    atomic_write_json("results/final_results/calibration_fit_log.json", {"records": fit_log})
    atomic_write_json("results/final_results/external_calibrated_metrics.json", report["external_like_calibrated"])
    lines = [
        "# Calibration Report\n\n",
        f"- variant: `{variant}`\n",
        f"- external_used_for_fit: `{report['external_used_for_fit']}`\n",
        f"- cv_uncalibrated_brier: `{report['cv_uncalibrated']['brier']:.4f}`\n",
        f"- cv_calibrated_brier: `{report['cv_calibrated']['brier']:.4f}`\n",
        f"- cv_uncalibrated_ece: `{report['cv_uncalibrated']['ece']:.4f}`\n",
        f"- cv_calibrated_ece: `{report['cv_calibrated']['ece']:.4f}`\n",
        f"- external_like_uncalibrated_brier: `{report['external_like_uncalibrated']['brier']:.4f}`\n",
        f"- external_like_calibrated_brier: `{report['external_like_calibrated']['brier']:.4f}`\n",
        "- uses_pet_as_input: `False`\n",
        "- uses_raw_identifiers: `False`\n",
        "- cohort_label_confound_warning: `True`\n",
    ]
    atomic_write_text(out_dir / "calibration_report.md", "".join(lines))
    pkt = Path("results/approval_packets/CALIBRATION")
    pkt.mkdir(parents=True, exist_ok=True)
    atomic_write_text(pkt / "README.md", "# CALIBRATION\n\nCalibration fit used only internal dev/fold calibration splits. External-like data was not used for fitting.\n")
    write_run_manifest("results/logs/calibration_manifest.json", stage="calibration", data_root=None, extra={"external_used_for_fit": False})
    StateManager().append_event("calibration", "COMPLETED", "Calibration completed without external fit.")
    print(json.dumps({"cv_brier": report["cv_calibrated"]["brier"], "cv_ece": report["cv_calibrated"]["ece"], "external_used_for_fit": False}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

