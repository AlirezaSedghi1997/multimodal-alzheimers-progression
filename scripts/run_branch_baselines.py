from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.train_baseline import average_precision, brier, roc_auc
from src.evaluation.bootstrap_ci import guarded_bootstrap_ci
from src.utils.io_helpers import atomic_write_json, atomic_write_text
from src.utils.run_manifest import write_run_manifest
from src.utils.state_manager import StateManager


DEFAULT_CONFIG = Path("configs/model/phase1_baseline_feature_sets.yaml")
FORBIDDEN_RE = re.compile(
    r"(pet|suvr|centiloid|tau|pmod|patient_hash|sample_hash|substudy|branch|has_t1_mri|"
    r"icv_ml|t1_|raw_|path|mri_shape|mri_voxel|mri_mean|mri_std|mri_p0|mri_nonzero)",
    re.I,
)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("run_branch_baselines requires PyYAML") from exc
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def safe_float(value: str | None) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.upper() in {"NA", "NAN", "NULL", ""}:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def sex_binary(row: dict[str, str]) -> float:
    sex = row.get("sex", "")
    if sex == "" and "sex_binary" in row:
        return safe_float(row.get("sex_binary")) or 0.0
    return 1.0 if str(sex).strip() in {"1", "M", "Male", "male"} else 0.0


def index_by_sample(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    return {row["sample_hash"]: row for row in rows if row.get("sample_hash")}


def merge_rows(
    base: dict[str, dict[str, str]],
    extra: dict[str, dict[str, str]] | None,
) -> dict[str, dict[str, str]]:
    if not extra:
        return base
    merged: dict[str, dict[str, str]] = {}
    for sample_hash, row in base.items():
        out = dict(row)
        if sample_hash in extra:
            for key, value in extra[sample_hash].items():
                if key not in {"sample_hash", "patient_hash"}:
                    out[key] = value
        merged[sample_hash] = out
    return merged


def resolve_feature_names(
    spec: dict[str, Any],
    config: dict[str, Any],
    sample_row: dict[str, str],
    seen: set[str] | None = None,
) -> list[str]:
    seen = seen or set()
    if "inherit" in spec:
        names: list[str] = []
        for parent in spec["inherit"]:
            if parent in seen:
                continue
            seen.add(parent)
            names.extend(resolve_feature_names(config["feature_sets"][parent], config, sample_row, seen))
        return names
    if spec.get("column_suffix"):
        suffix = str(spec["column_suffix"])
        return sorted(k for k in sample_row if k.endswith(suffix) and not FORBIDDEN_RE.search(k))
    features = list(spec.get("features") or [])
    for name in features:
        if name == "sex_binary":
            continue
        if name not in sample_row and name not in {"sex_binary"}:
            raise RuntimeError(f"Feature '{name}' missing from catalog row for set '{spec.get('description')}'")
    return features


def build_matrix(rows: list[dict[str, str]], feature_names: list[str]) -> tuple[list[list[float]], list[int], list[str]]:
    used: list[str] = []
    for name in feature_names:
        if name == "sex_binary":
            used.append(name)
        elif name in rows[0]:
            used.append(name)
    x: list[list[float]] = []
    y: list[int] = []
    for row in rows:
        label = row.get("ab_label_primary", row.get("ab_label"))
        if label in {None, ""}:
            continue
        vals: list[float] = []
        for name in used:
            if name == "sex_binary":
                vals.append(sex_binary(row))
            else:
                val = safe_float(row.get(name))
                vals.append(val if val is not None else float("nan"))
        x.append(vals)
        y.append(int(float(label)))
    return x, y, used


def train_predict(
    train_rows: list[dict[str, str]],
    test_rows: list[dict[str, str]],
    feature_names: list[str],
    model_name: str,
) -> tuple[list[int], list[float]]:
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    x_train, y_train, used = build_matrix(train_rows, feature_names)
    x_test, y_test, _ = build_matrix(test_rows, feature_names)
    if len(set(y_train)) < 2 or len(set(y_test)) < 2:
        raise ValueError("Both train and test folds must contain both classes")
    if model_name == "logistic_regression":
        clf = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=13)
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    model = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", clf),
        ]
    )
    model.fit(x_train, y_train)
    probs = [float(v) for v in model.predict_proba(x_test)[:, 1]]
    return y_test, probs


def evaluate_feature_set(
    name: str,
    spec: dict[str, Any],
    config: dict[str, Any],
    rows_by_sample: dict[str, dict[str, str]],
    split_dir: Path,
    model_name: str,
) -> dict[str, Any]:
    sample_row = next(iter(rows_by_sample.values()))
    feature_names = resolve_feature_names(spec, config, sample_row)
    if not feature_names:
        raise RuntimeError(f"No features resolved for set {name}")

    fold_metrics = []
    all_y: list[int] = []
    all_p: list[float] = []
    for test_path in sorted(split_dir.glob("fold_*_test.csv")):
        fold = test_path.name.split("_")[1]
        train_path = split_dir / f"fold_{fold}_train.csv"
        train_hashes = {r["sample_hash"] for r in read_csv_rows(train_path)}
        test_hashes = {r["sample_hash"] for r in read_csv_rows(test_path)}
        train_rows = [rows_by_sample[h] for h in train_hashes if h in rows_by_sample]
        test_rows = [rows_by_sample[h] for h in test_hashes if h in rows_by_sample]
        y_true, probs = train_predict(train_rows, test_rows, feature_names, model_name)
        metrics = {
            "fold": int(fold),
            "n_test": len(y_true),
            "roc_auc": roc_auc(y_true, probs),
            "pr_auc": average_precision(y_true, probs),
            "brier": brier(y_true, probs),
        }
        fold_metrics.append(metrics)
        all_y.extend(y_true)
        all_p.extend(probs)

    aggregate = {
        "feature_set": name,
        "branch": spec.get("branch"),
        "description": spec.get("description"),
        "model": model_name,
        "n_features": len(feature_names),
        "features_used": feature_names[:50] + (["..."] if len(feature_names) > 50 else []),
        "n": len(all_y),
        "roc_auc": roc_auc(all_y, all_p),
        "pr_auc": average_precision(all_y, all_p),
        "brier": brier(all_y, all_p),
        "roc_auc_ci": guarded_bootstrap_ci(all_y, all_p, roc_auc, n_boot=1000),
        "pr_auc_ci": guarded_bootstrap_ci(all_y, all_p, average_precision, n_boot=1000),
        "uses_pet_as_input": False,
        "uses_leaky_mri_surrogates": False,
        "uses_raw_identifiers": False,
    }
    return {"folds": fold_metrics, "aggregate": aggregate}


def prepare_rows(config: dict[str, Any], set_name: str, spec: dict[str, Any]) -> dict[str, dict[str, str]]:
    sources = config["catalog_sources"]
    branch = spec["branch"]
    if branch == "method_x_strict":
        base_rows = read_csv_rows(Path(sources["strict"]))
    elif branch == "mri_roi_wmh_reads":
        base_rows = read_csv_rows(Path(sources["roi_wmh_reads"]))
    else:
        raise RuntimeError(f"Unsupported branch: {branch}")

    cognitive = index_by_sample(read_csv_rows(Path(sources["cognitive"]))) if spec.get("join") == "cognitive" else None
    strict = index_by_sample(read_csv_rows(Path(sources["strict"]))) if spec.get("join") == "strict_tabular_from_master" else None

    rows = merge_rows(index_by_sample(base_rows), cognitive)
    if strict:
        tabular_keys = ["age", "sex", "education_years", "apoe4_carrier", "mmse_base"]
        for sample_hash, row in rows.items():
            if sample_hash in strict:
                for key in tabular_keys:
                    row[key] = strict[sample_hash].get(key, row.get(key))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description="Phase 1 interpretable baselines on branch catalogs.")
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--splits", default="splits")
    ap.add_argument("--out_dir", default="results/final_results/phase1_baselines")
    ap.add_argument("--sets", default="", help="Comma-separated feature set names; default runs all.")
    args = ap.parse_args()

    config = load_yaml(Path(args.config))
    split_dir = Path(args.splits)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_name = config.get("headline_model", "logistic_regression")
    selected = [s.strip() for s in args.sets.split(",") if s.strip()] or list(config["feature_sets"])

    results: dict[str, Any] = {}
    for name in selected:
        spec = config["feature_sets"][name]
        rows_by_sample = prepare_rows(config, name, spec)
        results[name] = evaluate_feature_set(name, spec, config, rows_by_sample, split_dir, model_name)

    summary = {
        "schema_version": 1,
        "status": "COMPLETED",
        "phase": 1,
        "headline_model": model_name,
        "feature_sets_evaluated": selected,
        "results": {name: obj["aggregate"] for name, obj in results.items()},
        "contains_pet_input": False,
        "contains_leaky_mri_surrogates": False,
        "contains_raw_identifiers": False,
        "public_output_granularity": "aggregate_feature_set_level",
    }
    atomic_write_json(out_dir / "phase1_baselines_summary.json", summary)
    for name, obj in results.items():
        atomic_write_json(out_dir / f"{name}_metrics.json", obj)

    lines = [
        "# Phase 1 Branch Baselines\n\n",
        f"- headline_model: `{model_name}`\n",
        f"- feature_sets: `{len(selected)}`\n\n",
        "| feature_set | branch | n_features | roc_auc | pr_auc | brier |\n",
        "|---|---|---:|---:|---:|---:|\n",
    ]
    for name, agg in summary["results"].items():
        lines.append(
            f"| {name} | {agg['branch']} | {agg['n_features']} | {agg['roc_auc']:.4f} | {agg['pr_auc']:.4f} | {agg['brier']:.4f} |\n"
        )
    atomic_write_text(out_dir / "phase1_baselines_summary.md", "".join(lines))
    write_run_manifest(
        "results/logs/phase1_baselines_manifest.json",
        stage="phase1_baselines",
        data_root=None,
        extra={"feature_sets": selected},
    )
    StateManager().append_event("phase1_baselines", "COMPLETED", f"Phase 1 interpretable baselines evaluated for {len(selected)} feature sets.")
    print(json.dumps({"status": "COMPLETED", "feature_sets": len(selected)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
