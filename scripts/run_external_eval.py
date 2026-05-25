from __future__ import annotations

import argparse
import csv
import json
import sys
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


def unique_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    out = {}
    for row in rows:
        out[row["sample_hash"]] = row
    return list(out.values())


def load_dev_rows(split_dir: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    first_train = split_dir / "fold_0_train.csv"
    first_test = split_dir / "fold_0_test.csv"
    rows.extend(read_rows(first_train))
    rows.extend(read_rows(first_test))
    return unique_rows(rows)


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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", default="splits")
    ap.add_argument("--features", default="data/processed/mri_features.csv")
    ap.add_argument("--fusion_summary", default="results/final_results/fusion/fusion_summary.json")
    ap.add_argument("--out_dir", default="results/final_results/external")
    args = ap.parse_args()

    try:
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
    except Exception as exc:
        raise RuntimeError("run_external_eval requires scikit-learn") from exc

    split_dir = Path(args.splits)
    fusion = json.loads(Path(args.fusion_summary).read_text(encoding="utf-8"))
    variant = fusion["best"]["variant"]
    c_value = float(fusion["best"]["C"])
    features = read_features(Path(args.features))
    dev_rows = load_dev_rows(split_dir)
    external_rows = read_rows(split_dir / "external_holdout.csv")
    x_train, y_train = matrix(dev_rows, features, variant)
    x_ext, y_ext = matrix(external_rows, features, variant)
    model = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(C=c_value, class_weight="balanced", max_iter=1000, random_state=31)),
        ]
    )
    model.fit(x_train, y_train)
    probs = [float(v) for v in model.predict_proba(x_ext)[:, 1]]
    single_class = len(set(y_ext)) < 2
    metrics = {
        "n_external": len(y_ext),
        "label_counts": {str(label): y_ext.count(label) for label in sorted(set(y_ext))},
        "external_holdout_type": "locked_pseudo_external_random_patient_holdout",
        "is_truly_independent_external": False,
        "single_class_behavior": "CALIBRATION_ONLY_NO_AUC" if single_class else None,
        "roc_auc": None if single_class else roc_auc(y_ext, probs),
        "pr_auc": None if single_class else average_precision(y_ext, probs),
        "brier": brier(y_ext, probs),
        "roc_auc_ci": None if single_class else guarded_bootstrap_ci(y_ext, probs, roc_auc, n_boot=1000),
        "pr_auc_ci": None if single_class else guarded_bootstrap_ci(y_ext, probs, average_precision, n_boot=1000),
        "variant": variant,
        "C": c_value,
        "uses_pet_as_input": False,
        "uses_raw_identifiers": False,
        "cohort_label_confound_warning": True,
        "gate_4_status": "DONE",
    }
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(out_dir / "external_metrics.json", metrics)
    lines = [
        "# External-Like Evaluation\n\n",
        f"- external_holdout_type: `{metrics['external_holdout_type']}`\n",
        f"- is_truly_independent_external: `{metrics['is_truly_independent_external']}`\n",
        f"- n_external: `{metrics['n_external']}`\n",
        f"- label_counts: `{metrics['label_counts']}`\n",
        f"- variant: `{variant}`\n",
        f"- roc_auc: `{metrics['roc_auc']}`\n",
        f"- pr_auc: `{metrics['pr_auc']}`\n",
        f"- brier: `{metrics['brier']:.4f}`\n",
        "- uses_pet_as_input: `False`\n",
        "- uses_raw_identifiers: `False`\n",
        "- cohort_label_confound_warning: `True`\n",
        "- gate_4_status: `DONE`\n\n",
        "## Interpretation\n\n",
        "This is a locked pseudo-external holdout derived from the same local A4/LEARN data layout, not a truly independent external cohort.\n",
    ]
    atomic_write_text(out_dir / "external_metrics.md", "".join(lines))
    pkt = Path("results/approval_packets/GATE_4_EXTERNAL")
    pkt.mkdir(parents=True, exist_ok=True)
    atomic_write_text(pkt / "README.md", "# GATE_4_EXTERNAL\n\nExternal-like evaluation completed once on the locked pseudo-external holdout.\n")
    write_run_manifest("results/logs/external_manifest.json", stage="external", data_root=None, extra={"metrics": metrics})
    StateManager().append_event("external", "DONE", "Locked pseudo-external evaluation completed.")
    print(json.dumps({"roc_auc": metrics["roc_auc"], "pr_auc": metrics["pr_auc"], "brier": metrics["brier"], "gate_4_status": "DONE"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

