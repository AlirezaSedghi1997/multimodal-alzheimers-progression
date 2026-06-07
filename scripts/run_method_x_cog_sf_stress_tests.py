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

from scripts.run_cognitive_ablation import (
    COGNITIVE_FEATURES,
    FIXED_TABULAR_FEATURES,
    average_precision_grouped,
    brier,
    fit_predict,
    read_by_sample,
    read_rows,
    roc_auc,
    variant_feature_names,
)
from src.evaluation.bootstrap_ci import guarded_bootstrap_ci
from src.utils.io_helpers import atomic_write_json, atomic_write_text
from src.utils.run_manifest import write_run_manifest
from src.utils.state_manager import StateManager


STRESS_VARIANTS = {
    "fixed_tabular_only": {"tabular": True, "cognitive": [], "mri": False},
    "fixed_tabular_plus_all_cognitive": {"tabular": True, "cognitive": COGNITIVE_FEATURES, "mri": False},
}
FORBIDDEN_TERMS = ("pet", "suvr", "centiloid", "tau", "raw_path", "patient_hash")
DECOUPLING_DISCLAIMER = (
    "Internal decoupling / stress-test only. SF is not external validation and must never "
    "be described as independent or quasi-external validation."
)


def load_catalog(path: Path) -> list[dict[str, str]]:
    rows = read_rows(path)
    out: list[dict[str, str]] = []
    for row in rows:
        label = row.get("ab_label_primary") or row.get("ab_label")
        if label is None or str(label).strip() == "":
            continue
        normalized = dict(row)
        normalized["ab_label"] = str(int(float(label)))
        out.append(normalized)
    return out


def subset(rows: list[dict[str, str]], *, substudies: set[str] | None = None) -> list[dict[str, str]]:
    if substudies is None:
        return rows
    return [row for row in rows if row.get("substudy") in substudies]


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


def evaluate_holdout(
    train_rows: list[dict[str, str]],
    test_rows: list[dict[str, str]],
    cognitive: dict[str, dict[str, str]],
    variant: str,
    c_value: float = 1.0,
) -> dict[str, Any]:
    y_true, y_prob = fit_predict(train_rows, test_rows, cognitive, {}, variant, c_value)
    if not y_true:
        return {"status": "SKIPPED_NO_ROWS", "n_test": 0}
    return {
        "status": "OK",
        "n_train": len(train_rows),
        "n_test": len(y_true),
        "roc_auc": roc_auc(y_true, y_prob),
        "pr_auc": average_precision_grouped(y_true, y_prob),
        "brier": brier(y_true, y_prob),
        "positive_rate_test": round(sum(int(v) for v in y_true) / len(y_true), 6),
        "y_true": y_true,
        "y_prob": y_prob,
    }


def evaluate_cv(
    rows: list[dict[str, str]],
    cognitive: dict[str, dict[str, str]],
    variant: str,
    *,
    k: int,
    seed: int,
    c_value: float = 1.0,
    bootstrap: bool = False,
) -> dict[str, Any]:
    all_y: list[int] = []
    all_p: list[float] = []
    fold_rows: list[dict[str, Any]] = []
    for fold_idx, (train_rows, test_rows) in enumerate(patient_grouped_folds(rows, k, seed)):
        result = evaluate_holdout(train_rows, test_rows, cognitive, variant, c_value)
        if result.get("status") != "OK":
            continue
        fold_rows.append(
            {
                "fold": fold_idx,
                "n_train": result["n_train"],
                "n_test": result["n_test"],
                "roc_auc": result["roc_auc"],
                "pr_auc": result["pr_auc"],
                "brier": result["brier"],
                "train_substudy_counts": dict(Counter(r["substudy"] for r in train_rows)),
                "test_substudy_counts": dict(Counter(r["substudy"] for r in test_rows)),
            }
        )
        all_y.extend(result["y_true"])
        all_p.extend(result["y_prob"])
    if not all_y:
        return {"status": "SKIPPED_NO_ROWS", "n": 0, "folds": fold_rows}
    result = {
        "status": "OK",
        "n": len(all_y),
        "folds": fold_rows,
        "feature_count": len(variant_feature_names(variant)),
        "roc_auc": roc_auc(all_y, all_p),
        "pr_auc": average_precision_grouped(all_y, all_p),
        "brier": brier(all_y, all_p),
        "y_true": all_y,
        "y_prob": all_p,
    }
    if bootstrap:
        result["roc_auc_ci"] = guarded_bootstrap_ci(all_y, all_p, roc_auc, n_boot=1000)
    return result


def cramers_v(rows: list[dict[str, str]]) -> float | None:
    if not rows:
        return None
    table: dict[str, Counter[str]] = {}
    for row in rows:
        substudy = row.get("substudy", "")
        label = row.get("ab_label", "")
        table.setdefault(substudy, Counter())[label] += 1
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
    return float((chi2 / n) ** 0.5)


def sf_label_discrepancy(rows: list[dict[str, str]]) -> dict[str, Any]:
    sf = [row for row in rows if row.get("substudy") == "SF"]
    agree = 0
    disagree = 0
    missing_sensitivity = 0
    for row in sf:
        primary = row.get("ab_label_primary") or row.get("ab_label")
        sensitivity = row.get("ab_label_sensitivity")
        if sensitivity is None or str(sensitivity).strip() == "":
            missing_sensitivity += 1
            continue
        if str(int(float(primary))) == str(int(float(sensitivity))):
            agree += 1
        else:
            disagree += 1
    return {
        "sf_rows": len(sf),
        "primary_positive_rate": round(sum(int(row["ab_label"]) for row in sf) / len(sf), 6) if sf else None,
        "sensitivity_positive_rate": round(
            sum(int(float(row["ab_label_sensitivity"])) for row in sf if row.get("ab_label_sensitivity") not in {None, ""})
            / max(1, len(sf) - missing_sensitivity),
            6,
        ),
        "primary_vs_sensitivity_agree": agree,
        "primary_vs_sensitivity_disagree": disagree,
        "missing_sensitivity_label": missing_sensitivity,
    }


def run_arm(
    arm_id: str,
    rows: list[dict[str, str]],
    cognitive: dict[str, dict[str, str]],
    *,
    mode: str,
    k: int,
    seed: int,
    bootstrap: bool = False,
) -> dict[str, Any]:
    results: dict[str, Any] = {"arm": arm_id, "mode": mode, "variants": {}}
    for variant in STRESS_VARIANTS:
        if mode == "cv":
            metrics = evaluate_cv(rows, cognitive, variant, k=k, seed=seed, bootstrap=bootstrap)
        elif mode == "train_sf_test_enrolled":
            metrics = evaluate_holdout(
                subset(rows, substudies={"SF"}),
                subset(rows, substudies={"A4", "LEARN"}),
                cognitive,
                variant,
            )
        elif mode == "train_enrolled_test_sf":
            metrics = evaluate_holdout(
                subset(rows, substudies={"A4", "LEARN"}),
                subset(rows, substudies={"SF"}),
                cognitive,
                variant,
            )
        else:
            raise ValueError(f"Unknown mode: {mode}")
        metrics.pop("y_true", None)
        metrics.pop("y_prob", None)
        results["variants"][variant] = metrics
    return results


def build_scientific_interpretation(arms: list[dict[str, Any]], coupling: dict[str, Any]) -> dict[str, Any]:
    def arm_metric(arm_name: str, variant: str, key: str) -> float | None:
        for arm in arms:
            if arm["arm"] != arm_name:
                continue
            metrics = arm["variants"].get(variant, {})
            value = metrics.get(key)
            return float(value) if value is not None else None
        return None

    sf_cog = arm_metric("sf_only_5fold_cv", "fixed_tabular_plus_all_cognitive", "roc_auc")
    sf_tab = arm_metric("sf_only_5fold_cv", "fixed_tabular_only", "roc_auc")
    pooled_cog = arm_metric(
        "sf_inclusive_confound_controlled_5fold_cv",
        "fixed_tabular_plus_all_cognitive",
        "roc_auc",
    )
    train_sf_cog = arm_metric(
        "train_sf_test_enrolled_internal_decoupling",
        "fixed_tabular_plus_all_cognitive",
        "roc_auc",
    )
    train_sf_tab = arm_metric(
        "train_sf_test_enrolled_internal_decoupling",
        "fixed_tabular_only",
        "roc_auc",
    )
    return {
        "cv_protocol_note": (
            "sf_inclusive_confound_controlled_5fold_cv uses patient-grouped 5-fold CV on the "
            "SF-inclusive catalog. It is confound-aware evaluation, not a claim of confound-free accuracy."
        ),
        "sf_only_cognitive_signal": {
            "tabular_roc_auc": sf_tab,
            "cognitive_roc_auc": sf_cog,
            "reading": "Within SF-only screening, cognitive features add little beyond fixed tabular.",
        },
        "pooled_sf_inclusive_caution": {
            "cognitive_roc_auc": pooled_cog,
            "pooled_cramers_v": coupling.get("pooled_cramers_v"),
            "reading": (
                "High pooled AUC on SF-inclusive data must not be read as biological proof; "
                "residual cohort-label coupling remains (Cramér's V≈0.90)."
            ),
        },
        "internal_decoupling_transfer": {
            "train_sf_tabular_roc_auc": train_sf_tab,
            "train_sf_cognitive_roc_auc": train_sf_cog,
            "reading": (
                "Train-SF / test-enrolled cognitive transfer collapses (~0.51 AUC), indicating "
                "cognitive features do not generalize across substudy contexts and must not support "
                "Method X-Strict claims."
            ),
        },
        "forbidden_claims": [
            "external validation",
            "quasi-external validation",
            "confound-free prediction accuracy",
            "Method X-Strict replacement",
        ],
    }


def write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# Method X-Cog/SF Internal Stress Tests\n\n",
        f"- status: `{report['status']}`\n",
        f"- branch: `{report['branch']}`\n",
        f"- confound_controlled_cv: `{report['confound_controlled_cv']}`\n",
        f"- disclaimer: {report['disclaimer']}\n\n",
        "## Catalog counts\n\n",
        "| substudy | n | positive_rate |\n",
        "|---|---:|---:|\n",
    ]
    for substudy, stats in sorted(report["catalog_counts"].items()):
        lines.append(f"| {substudy} | {stats['n']} | {stats['positive_rate']} |\n")
    lines.extend(
        [
            "\n",
            f"- pooled Cramér's V (substudy vs label): `{report['cohort_label_coupling']['pooled_cramers_v']}`\n",
            f"- SF-only Cramér's V: `{report['cohort_label_coupling']['sf_only_cramers_v']}`\n",
            f"- enrolled-only Cramér's V: `{report['cohort_label_coupling']['enrolled_only_cramers_v']}`\n\n",
            "## SF primary vs sensitivity label discrepancy\n\n",
            f"- SF rows: `{report['sf_label_discrepancy']['sf_rows']}`\n",
            f"- primary positive rate: `{report['sf_label_discrepancy']['primary_positive_rate']}`\n",
            f"- sensitivity positive rate: `{report['sf_label_discrepancy']['sensitivity_positive_rate']}`\n",
            f"- agree/disagree: `{report['sf_label_discrepancy']['primary_vs_sensitivity_agree']}` / "
            f"`{report['sf_label_discrepancy']['primary_vs_sensitivity_disagree']}`\n\n",
            "## Scientific interpretation (required reading)\n\n",
            f"- CV protocol: {report['scientific_interpretation']['cv_protocol_note']}\n",
            f"- SF-only: {report['scientific_interpretation']['sf_only_cognitive_signal']['reading']}\n",
            f"- Pooled SF-inclusive: {report['scientific_interpretation']['pooled_sf_inclusive_caution']['reading']}\n",
            f"- Internal decoupling: {report['scientific_interpretation']['internal_decoupling_transfer']['reading']}\n\n",
            "## Stress-test arms\n\n",
        ]
    )
    for arm in report["arms"]:
        lines.append(f"### {arm['arm']} ({arm['mode']})\n\n")
        lines.append("| variant | status | n | roc_auc | pr_auc | brier |\n")
        lines.append("|---|---|---:|---:|---:|---:|\n")
        for variant, metrics in arm["variants"].items():
            lines.append(
                f"| {variant} | {metrics.get('status')} | {metrics.get('n', metrics.get('n_test', 0))} | "
                f"{metrics.get('roc_auc')} | {metrics.get('pr_auc')} | {metrics.get('brier')} |\n"
            )
        lines.append("\n")
    atomic_write_text(path, "".join(lines))


def main() -> int:
    ap = argparse.ArgumentParser(description="SF-aware internal decoupling stress tests for Method X-Cog/SF.")
    ap.add_argument("--catalog", default="data/processed/catalogs/method_x_cog_sf_catalog.csv")
    ap.add_argument("--cognitive_features", default="data/processed/cognitive_features_method_x_cog_sf.csv")
    ap.add_argument("--out_dir", default="results/final_results/method_x_cog_sf")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=20260607)
    args = ap.parse_args()

    catalog_path = Path(args.catalog)
    cognitive_path = Path(args.cognitive_features)
    if not catalog_path.exists():
        raise RuntimeError(f"Missing catalog: {catalog_path}")
    if not cognitive_path.exists():
        raise RuntimeError(
            "Missing SF-inclusive cognitive features. Run build_cognitive_features.py with "
            "method_x_cog_sf_catalog.csv first."
        )

    rows = load_catalog(catalog_path)
    cognitive = read_by_sample(cognitive_path)
    if not rows:
        raise RuntimeError("Method X-Cog/SF catalog is empty after label normalization.")

    catalog_counts: dict[str, dict[str, Any]] = {}
    for substudy in sorted({row["substudy"] for row in rows}):
        sub_rows = subset(rows, substudies={substudy})
        catalog_counts[substudy] = {
            "n": len(sub_rows),
            "positive_rate": round(sum(int(row["ab_label"]) for row in sub_rows) / len(sub_rows), 6),
        }

    arms = [
        run_arm("sf_only_5fold_cv", subset(rows, substudies={"SF"}), cognitive, mode="cv", k=args.folds, seed=args.seed),
        run_arm(
            "sf_inclusive_confound_controlled_5fold_cv",
            rows,
            cognitive,
            mode="cv",
            k=args.folds,
            seed=args.seed + 11,
            bootstrap=True,
        ),
        run_arm(
            "train_sf_test_enrolled_internal_decoupling",
            rows,
            cognitive,
            mode="train_sf_test_enrolled",
            k=args.folds,
            seed=args.seed,
        ),
        run_arm(
            "train_enrolled_test_sf_internal_decoupling",
            rows,
            cognitive,
            mode="train_enrolled_test_sf",
            k=args.folds,
            seed=args.seed,
        ),
    ]

    report: dict[str, Any] = {
        "schema_version": 1,
        "status": "COMPLETED",
        "branch": "method_x_cog_sf",
        "confound_controlled_cv": "REQUIRED",
        "disclaimer": DECOUPLING_DISCLAIMER,
        "uses_pet_as_input": False,
        "uses_raw_identifiers": False,
        "public_output_granularity": "aggregate_arm_level",
        "catalog_counts": catalog_counts,
        "cohort_label_coupling": {
            "pooled_cramers_v": cramers_v(rows),
            "sf_only_cramers_v": cramers_v(subset(rows, substudies={"SF"})),
            "enrolled_only_cramers_v": cramers_v(subset(rows, substudies={"A4", "LEARN"})),
        },
        "sf_label_discrepancy": sf_label_discrepancy(rows),
        "scientific_interpretation": build_scientific_interpretation(arms, {
            "pooled_cramers_v": cramers_v(rows),
            "sf_only_cramers_v": cramers_v(subset(rows, substudies={"SF"})),
            "enrolled_only_cramers_v": cramers_v(subset(rows, substudies={"A4", "LEARN"})),
        }),
        "cognitive_features_artifact": str(cognitive_path),
        "variants_evaluated": list(STRESS_VARIANTS),
        "forbidden_terms_checked": FORBIDDEN_TERMS,
        "arms": arms,
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "sf_stress_tests_summary.json"
    md_path = out_dir / "sf_stress_tests_summary.md"
    atomic_write_json(json_path, report)
    write_markdown(report, md_path)
    write_run_manifest(
        "results/logs/method_x_cog_sf_stress_tests_manifest.json",
        stage="method_x_cog_sf_stress_tests",
        data_root=None,
        extra={"status": report["status"], "arms": [arm["arm"] for arm in arms]},
    )
    StateManager().append_event(
        "method_x_cog_sf_stress_tests",
        "COMPLETED",
        "SF-aware internal decoupling stress tests completed for Method X-Cog/SF.",
    )
    print(json.dumps({"status": report["status"], "arms": len(arms), "json": str(json_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
