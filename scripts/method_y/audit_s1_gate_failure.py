#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import h5py
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.method_y._stage_common import DEFAULT_METHOD_Y_PET_CACHE, DEFAULT_METHOD_Y_PET_INPUT_LIST


CL_PRIMARY = 24.0
CL_SECONDARY = 30.0
DEFAULT_PET_H5 = DEFAULT_METHOD_Y_PET_CACHE
DEFAULT_PET_CSV = DEFAULT_METHOD_Y_PET_INPUT_LIST
DEFAULT_FOLD_MANIFEST = "results/method_y_execution/fold_manifest.json"
DEFAULT_STAGE_S1_REPORT = "results/method_y_execution/stage_s1_report.json"
DEFAULT_VALIDATION_DIR = "results/method_y_execution/validation_artifacts"
DEFAULT_OUT = "results/method_y_execution/s1_gate_failure_audit.json"


def project_path(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def decode_h5(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    return value


def read_json_lenient(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"), parse_constant=lambda _value: None)


def read_pet_csv(path: Path) -> tuple[dict[str, dict[str, str]], dict[str, Any]]:
    rows: dict[str, dict[str, str]] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            subject = str(row.get("subject_id_hash") or "").strip()
            if not subject:
                continue
            rows[subject] = {
                "tracer": str(row.get("tracer") or "").strip().upper(),
                "cohort": str(row.get("cohort") or "OTHER").strip().upper(),
                "scanner": str(row.get("scanner") or "UNKNOWN").strip(),
                "paired_mri_subject_id_hash": str(row.get("paired_mri_subject_id_hash") or "").strip(),
            }
    return rows, {
        "row_count": len(rows),
        "tracer_counts": dict(Counter(row["tracer"] for row in rows.values())),
        "cohort_counts": dict(Counter(row["cohort"] for row in rows.values())),
        "scanner_counts": dict(Counter(row["scanner"] for row in rows.values())),
        "scanner_unknown_subjects_count": sum(1 for row in rows.values() if row["scanner"].upper() == "UNKNOWN"),
    }


def finite_summary(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "min": None, "mean": None, "median": None, "max": None}
    arr = np.asarray(values, dtype=float)
    return {
        "count": int(arr.size),
        "min": float(np.min(arr)),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "max": float(np.max(arr)),
    }


def load_h5_metadata(h5_path: Path, csv_meta: dict[str, dict[str, str]]) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    subjects: dict[str, dict[str, Any]] = {}
    finite_fractions: list[float] = []
    nonfinite_subjects = 0
    shape_counts: Counter[str] = Counter()
    with h5py.File(h5_path, "r") as h5:
        for subject in sorted(h5.keys()):
            grp = h5[subject]
            centiloid = float(grp["centiloid"][()]) if "centiloid" in grp else float("nan")
            row = csv_meta.get(subject, {})
            scanner_h5 = decode_h5(grp["scanner"][()]) if "scanner" in grp else "UNKNOWN"
            image = np.asarray(grp["image"])
            finite_fraction = float(np.isfinite(image).mean())
            finite_values = image[np.isfinite(image)]
            if finite_values.size:
                image_mean = float(np.mean(finite_values))
                image_std = float(np.std(finite_values))
                image_p95 = float(np.percentile(finite_values, 95))
                image_max = float(np.max(finite_values))
            else:
                image_mean = image_std = image_p95 = image_max = float("nan")
            finite_fractions.append(finite_fraction)
            if finite_fraction < 1.0:
                nonfinite_subjects += 1
            shape_counts[str(tuple(image.shape))] += 1
            subjects[subject] = {
                "centiloid": centiloid,
                "label_cl24": int(centiloid >= CL_PRIMARY),
                "label_cl30": int(centiloid >= CL_SECONDARY),
                "equivocal": int(grp["equivocal"][()]) if "equivocal" in grp else int(18.0 <= centiloid < 35.0),
                "cohort": row.get("cohort") or (decode_h5(grp["cohort"][()]) if "cohort" in grp else "OTHER"),
                "scanner": row.get("scanner") or scanner_h5,
                "scanner_h5": scanner_h5,
                "tracer": row.get("tracer") or (decode_h5(grp["tracer"][()]) if "tracer" in grp else ""),
                "finite_fraction": finite_fraction,
                "suvr_global": float(grp["suvr_global"][()]) if "suvr_global" in grp else float("nan"),
                "image_mean": image_mean,
                "image_std": image_std,
                "image_p95": image_p95,
                "image_max": image_max,
            }
    return subjects, {
        "subjects_checked": len(subjects),
        "image_shape_counts": dict(shape_counts),
        "subjects_with_nonfinite_voxels": nonfinite_subjects,
        "finite_fraction": finite_summary(finite_fractions),
    }


def nested_count(rows: list[dict[str, Any]], left: str, right: str) -> dict[str, dict[str, int]]:
    out: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        out[str(row[left])][str(row[right])] += 1
    return {key: dict(value) for key, value in sorted(out.items())}


def centiloid_by(rows: list[dict[str, Any]], key: str) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        value = row.get("centiloid")
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            groups[str(row[key])].append(float(value))
    return {name: finite_summary(values) for name, values in sorted(groups.items())}


def summarize_subjects(subject_ids: list[str], metadata: dict[str, dict[str, Any]]) -> dict[str, Any]:
    rows = [metadata[subject] for subject in subject_ids if subject in metadata]
    label24 = Counter("CL>=24" if row["label_cl24"] else "CL<24" for row in rows)
    label30 = Counter("CL>=30" if row["label_cl30"] else "CL<30" for row in rows)
    return {
        "n_subjects": len(rows),
        "label_counts_cl24": dict(label24),
        "label_counts_cl30": dict(label30),
        "equivocal_counts": dict(Counter("equivocal" if row["equivocal"] else "non_equivocal" for row in rows)),
        "cohort_counts": dict(Counter(str(row["cohort"]) for row in rows)),
        "scanner_counts": dict(Counter(str(row["scanner"]) for row in rows)),
        "scanner_x_cohort": nested_count(rows, "scanner", "cohort"),
        "label_x_cohort_cl24": nested_count(
            [{"label": "CL>=24" if row["label_cl24"] else "CL<24", "cohort": row["cohort"]} for row in rows],
            "label",
            "cohort",
        ),
        "label_x_scanner_cl24": nested_count(
            [{"label": "CL>=24" if row["label_cl24"] else "CL<24", "scanner": row["scanner"]} for row in rows],
            "label",
            "scanner",
        ),
        "centiloid_summary_by_cohort": centiloid_by(rows, "cohort"),
        "centiloid_summary_by_scanner": centiloid_by(rows, "scanner"),
    }


def safe_auc(labels: list[int], scores: list[float]) -> float | None:
    pos = [score for label, score in zip(labels, scores, strict=True) if int(label) == 1]
    neg = [score for label, score in zip(labels, scores, strict=True) if int(label) == 0]
    if not pos or not neg:
        return None
    wins = 0.0
    for p in pos:
        for n in neg:
            wins += 1.0 if p > n else 0.5 if p == n else 0.0
    return wins / (len(pos) * len(neg))


def safe_spearman(a: list[float], b: list[float]) -> float | None:
    if len(a) < 3 or len(a) != len(b):
        return None
    x = np.asarray(a, dtype=float)
    y = np.asarray(b, dtype=float)
    keep = np.isfinite(x) & np.isfinite(y)
    x = x[keep]
    y = y[keep]
    if x.size < 3:
        return None

    def ranks(values: np.ndarray) -> np.ndarray:
        order = np.argsort(values)
        out = np.empty_like(order, dtype=float)
        sorted_values = values[order]
        start = 0
        while start < len(values):
            end = start + 1
            while end < len(values) and sorted_values[end] == sorted_values[start]:
                end += 1
            out[order[start:end]] = (start + end - 1) / 2.0
            start = end
        return out

    rx = ranks(x)
    ry = ranks(y)
    if float(np.std(rx)) == 0.0 or float(np.std(ry)) == 0.0:
        return None
    corr = float(np.corrcoef(rx, ry)[0, 1])
    return corr if math.isfinite(corr) else None


def summarize_image_signal(metadata: dict[str, dict[str, Any]]) -> dict[str, Any]:
    rows = list(metadata.values())
    features = ["suvr_global", "image_mean", "image_std", "image_p95", "image_max", "finite_fraction"]
    scopes = {
        "all_subjects": rows,
        "non_equivocal_only": [row for row in rows if not bool(row.get("equivocal"))],
    }
    out: dict[str, Any] = {}
    for scope, scope_rows in scopes.items():
        labels = [int(row["label_cl24"]) for row in scope_rows]
        centiloid = [float(row["centiloid"]) for row in scope_rows]
        scope_summary: dict[str, Any] = {
            "n": len(scope_rows),
            "positive_cl24": sum(labels),
            "negative_cl24": len(labels) - sum(labels),
        }
        for feature in features:
            values = [float(row.get(feature, float("nan"))) for row in scope_rows]
            scope_summary[feature] = {
                "auc_cl24": safe_auc(labels, values),
                "spearman_centiloid": safe_spearman(values, centiloid),
                "summary": finite_summary([value for value in values if math.isfinite(value)]),
            }
        out[scope] = scope_summary
    return out


def summarize_validation_artifact(
    path: Path,
    metadata: dict[str, dict[str, Any]],
    expected_val_ids: list[str],
    *,
    artifact_mode: str,
) -> dict[str, Any]:
    if not path.exists():
        return {"path": display_path(path), "artifact_mode": artifact_mode, "status": "MISSING"}
    with np.load(path, allow_pickle=False) as data:
        keys = sorted(data.files)
        subjects = [str(value) for value in data["subject_id_hash"].astype(str).tolist()] if "subject_id_hash" in data else []
        labels24 = [int(value) for value in data["labels_cl24"].tolist()] if "labels_cl24" in data else []
        labels30 = [int(value) for value in data["labels_cl30"].tolist()] if "labels_cl30" in data else []
        probs = [float(value) for value in data["prob_pet"].tolist()] if "prob_pet" in data else []
        if "equivocal" in data:
            equivocal = [int(value) for value in data["equivocal"].tolist()]
        else:
            equivocal = [int(metadata.get(subject, {}).get("equivocal", 0)) for subject in subjects]
    all_mask = [True] * len(subjects)
    non_equivocal_mask = [not bool(value) for value in equivocal]

    def metrics(mask: list[bool]) -> dict[str, Any]:
        y24 = [value for value, keep in zip(labels24, mask, strict=True) if keep]
        y30 = [value for value, keep in zip(labels30, mask, strict=True) if keep]
        p = [value for value, keep in zip(probs, mask, strict=True) if keep]
        return {
            "n": len(y24),
            "auroc_cl24": safe_auc(y24, p),
            "auroc_cl30": safe_auc(y30, p),
            "auroc_cl24_if_probability_complemented": safe_auc(y24, [1.0 - value for value in p]),
        }

    return {
        "path": display_path(path),
        "artifact_mode": artifact_mode,
        "status": "READ",
        "keys": keys,
        "n_rows": len(subjects),
        "subjects_match_fold_val": sorted(subjects) == sorted(expected_val_ids),
        "metrics_by_subject_scope": {
            "all_subjects": metrics(all_mask),
            "non_equivocal_only": metrics(non_equivocal_mask),
        },
    }


def build_root_cause_hypotheses(
    report: dict[str, Any],
    overall: dict[str, Any],
    tracer_status: str,
    image_signal_summary: dict[str, Any],
) -> list[str]:
    hypotheses: list[str] = []
    if tracer_status == "NOT_MEASURABLE_SINGLE_TRACER":
        hypotheses.append(
            "Only one tracer is present, so tracer-probe AUC cannot measure tracer invariance; S1 promotion needs multi-tracer evidence or an explicit single-tracer promotion policy."
        )
    fold_metrics = report.get("fold_metrics") or []
    if fold_metrics and min((int(m.get("n_val") or 0) for m in fold_metrics), default=0) <= 4:
        hypotheses.append(
            "The S1 dry-run uses very small validation folds, so AUROC and cohort-probe values are smoke diagnostics rather than stable scientific estimates."
        )
    equivocal = overall.get("equivocal_counts", {})
    total = int(overall.get("n_subjects") or 0)
    equivocal_n = int(equivocal.get("equivocal") or 0)
    if total and equivocal_n / total > 0.25:
        hypotheses.append(
            "A large equivocal Centiloid band can depress or destabilize all-subject AUROC; non-equivocal-only gate metrics must be reported separately."
        )
    if int((overall.get("scanner_counts") or {}).get("UNKNOWN") or 0) > 0:
        hypotheses.append(
            "Residual UNKNOWN scanner metadata can weaken scanner-balanced sampling and scanner/cohort leakage interpretation."
        )
    max_cohort_probe = (report.get("metrics") or {}).get("max_cohort_probe_auc")
    if isinstance(max_cohort_probe, (int, float)) and max_cohort_probe >= 0.65:
        hypotheses.append(
            "High cohort-probe AUC indicates the PET latent or dry-run fold composition is cohort-predictive; inspect scanner x cohort and Centiloid distribution shifts before S2."
        )
    non_eq_signal = image_signal_summary.get("non_equivocal_only", {})
    suvr_auc = (non_eq_signal.get("suvr_global") or {}).get("auc_cl24")
    image_auc = max(
        [
            value
            for value in [
                (non_eq_signal.get("image_mean") or {}).get("auc_cl24"),
                (non_eq_signal.get("image_std") or {}).get("auc_cl24"),
                (non_eq_signal.get("image_p95") or {}).get("auc_cl24"),
                (non_eq_signal.get("image_max") or {}).get("auc_cl24"),
            ]
            if isinstance(value, (int, float))
        ]
        or [0.0]
    )
    if isinstance(suvr_auc, (int, float)) and suvr_auc >= 0.95 and image_auc < 0.70:
        hypotheses.append(
            "suvr_global is strongly tied to the Centiloid label, but simple voxel-intensity summaries from the per-scan-z-scored HDF5 image are weak; do not feed SUVR/Centiloid scalars as model inputs, but treat this as evidence that the current image cache/training setup may not carry enough quantitative PET signal for S1."
        )
    if str(report.get("mode")) == "train" and not bool(report.get("acceptance_gate_passed")):
        hypotheses.append(
            "The latest train-mode S1 report failed its acceptance gate, so S2-S5 must remain blocked until S1 is rerun with a valid protocol and passes."
        )
    hypotheses.append(
        "Dry-run metrics are smoke/gate diagnostics only and must not be treated as final Method Y science."
    )
    return hypotheses


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit Method Y S1 gate failure without promoting S1.")
    parser.add_argument("--pet_h5", default=DEFAULT_PET_H5)
    parser.add_argument("--pet_csv", default=DEFAULT_PET_CSV)
    parser.add_argument("--fold_manifest", default=DEFAULT_FOLD_MANIFEST)
    parser.add_argument("--stage_s1_report", default=DEFAULT_STAGE_S1_REPORT)
    parser.add_argument("--validation_dir", default=DEFAULT_VALIDATION_DIR)
    parser.add_argument("--out", default=DEFAULT_OUT)
    args = parser.parse_args()

    pet_csv_meta, pet_csv_summary = read_pet_csv(project_path(args.pet_csv))
    h5_metadata, h5_finite_summary = load_h5_metadata(project_path(args.pet_h5), pet_csv_meta)
    fold_manifest = read_json_lenient(project_path(args.fold_manifest))
    stage_s1_report = read_json_lenient(project_path(args.stage_s1_report))
    all_subjects = sorted(h5_metadata)
    overall = summarize_subjects(all_subjects, h5_metadata)
    image_signal_summary = summarize_image_signal(h5_metadata)
    tracers = sorted({str(row.get("tracer") or "") for row in h5_metadata.values() if row.get("tracer")})
    tracer_status = "NOT_MEASURABLE_SINGLE_TRACER" if len(tracers) <= 1 else "MEASURABLE"

    validation_dir = project_path(args.validation_dir)
    report_mode = str(stage_s1_report.get("mode") or "dry_run")
    artifact_suffix = "" if report_mode == "train" else "_dryrun"
    artifact_mode = "train" if report_mode == "train" else "dry_run"
    folds = []
    for fold in fold_manifest.get("folds", []):
        fold_idx = int(fold.get("fold", len(folds)))
        train_ids = [str(value) for value in fold.get("train_subject_id_hashes", [])]
        val_ids = [str(value) for value in fold.get("val_subject_id_hashes", [])]
        artifact_path = validation_dir / f"stage_s1_fold{fold_idx}{artifact_suffix}.npz"
        artifact_mode_used = artifact_mode
        if not artifact_path.exists() and artifact_mode == "train":
            fallback = validation_dir / f"stage_s1_fold{fold_idx}_dryrun.npz"
            if fallback.exists():
                artifact_path = fallback
                artifact_mode_used = "dry_run_fallback_after_missing_train_artifact"
        folds.append(
            {
                "fold": fold_idx,
                "train": summarize_subjects(train_ids, h5_metadata),
                "val": summarize_subjects(val_ids, h5_metadata),
                "validation_artifact": summarize_validation_artifact(
                    artifact_path,
                    h5_metadata,
                    val_ids,
                    artifact_mode=artifact_mode_used,
                ),
            }
        )

    payload = {
        "status": "DONE",
        "purpose": "S1 gate-failure audit; dry-run metrics are smoke/gate diagnostics only.",
        "files_read": [
            args.pet_h5,
            args.pet_csv,
            args.fold_manifest,
            args.stage_s1_report,
            args.validation_dir,
        ],
        "pet_input_list_summary": pet_csv_summary,
        "hdf5_finite_fraction_summary": h5_finite_summary,
        "hdf5_image_signal_summary": image_signal_summary,
        "overall": overall,
        "fold_manifest": {
            "manifest_status": fold_manifest.get("manifest_status"),
            "fold_strategy": fold_manifest.get("fold_strategy"),
            "n_subjects": fold_manifest.get("n_subjects"),
            "n_folds": fold_manifest.get("n_folds"),
            "mode_when_written": fold_manifest.get("mode_when_written"),
        },
        "folds": folds,
        "tracers_observed": tracers,
        "tracer_probe_status": tracer_status,
        "stage_s1_report_summary": {
            "mode": stage_s1_report.get("mode"),
            "status": stage_s1_report.get("status"),
            "acceptance_gate_passed": stage_s1_report.get("acceptance_gate_passed"),
            "blocker": stage_s1_report.get("blocker"),
            "next_allowed_stage": stage_s1_report.get("next_allowed_stage"),
        },
        "stage_s1_metrics": stage_s1_report.get("metrics", {}),
        "root_cause_hypotheses": build_root_cause_hypotheses(
            stage_s1_report,
            overall,
            tracer_status,
            image_signal_summary,
        ),
    }
    out = project_path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(json_safe(payload), indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"status": payload["status"], "out": args.out, "tracer_probe_status": tracer_status}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
