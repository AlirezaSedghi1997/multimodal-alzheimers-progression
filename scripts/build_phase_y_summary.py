#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import sys
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.utils.pet_spatial_qc import atomic_write_text


PROJECT_ROOT = Path(__file__).resolve().parents[1]

FILES = [
    "results/tables/method_y_gap_filter_report.json",
    "results/tables/method_y_pet_manifest_summary.json",
    "results/tables/method_y_pet_preprocess_summary.json",
    "results/tables/method_y_pet_cache_summary.json",
    "results/tables/method_y_training_readiness.json",
    "results/tables/method_y_incremental_value_table.json",
    "results/tables/synthetic_mri_pet_supplement.json",
    "results/tables/proposal_alignment_report.json",
    "results/final_results/method_y/mri_pet_cognitive_training/method_y_mri_pet_cognitive_metrics.json",
]


def read(path):
    p = PROJECT_ROOT / path
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def fmt(value, ndigits: int = 4) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:.{ndigits}f}"
    return str(value)


def status_line(lines: list[str], rel: str, obj: dict | None) -> None:
    label = rel.replace("results/", "")
    if obj is None:
        lines.append(f"- {label}: `MISSING`\n")
        return
    status = obj.get("status") or obj.get("verdict") or obj.get("stage")
    lines.append(f"- {label}: `{status}`\n")


def append_incremental_table(lines: list[str], report: dict | None) -> None:
    lines.append("## Incremental Method Y Metrics\n\n")
    if report is None:
        lines.append("- Method Y incremental value table is missing.\n\n")
        return

    pet = report.get("pet_only_reference") or {}
    pet_threshold = pet.get("threshold_metrics") or {}
    lines.extend(
        [
            f"- PET-only reference AUC: `{fmt(pet.get('roc_auc'))}`\n",
            f"- matched effective N: `{report.get('matched_effective_n', 'NA')}`\n",
            f"- paired tier: `{report.get('paired_tier', 'NA')}`\n",
            "- caveat: PET-containing rows are Method Y only and have PET-label tautology plus small-N/class-imbalance limitations.\n",
            "- threshold metrics: fold-internal validation-selected thresholds only; test labels are never used for threshold selection.\n\n",
            "| row | n | AUC | delta AUC vs PET | accuracy | sensitivity | specificity | F1 | headline allowed |\n",
            "|---|---:|---:|---:|---:|---:|---:|---:|---|\n",
            (
                f"| Y1 PET-only reference | {pet.get('n', 'NA')} | {fmt(pet.get('roc_auc'))} | 0.0000 | "
                f"{fmt(pet_threshold.get('accuracy'))} | {fmt(pet_threshold.get('sensitivity'))} | "
                f"{fmt(pet_threshold.get('specificity'))} | {fmt(pet_threshold.get('f1'))} | False |\n"
            ),
        ]
    )
    for row in report.get("rows") or []:
        threshold = row.get("threshold_metrics") or {}
        lines.append(
            f"| {row.get('label', row.get('comparison_id'))} | {row.get('effective_n', 'NA')} | "
            f"{fmt(row.get('roc_auc'))} | {fmt(row.get('delta_auc_vs_pet_only'))} | "
            f"{fmt(threshold.get('accuracy'))} | {fmt(threshold.get('sensitivity'))} | "
            f"{fmt(threshold.get('specificity'))} | {fmt(threshold.get('f1'))} | "
            f"{row.get('headline_allowed', False)} |\n"
        )
    lines.append("\n")


def main() -> int:
    lines = ["# Method Y Phase 2 Summary\n\n"]
    lines.extend(
        [
            "Method Y is a PET-assisted exploratory branch. It is separate from Method X-Strict, ",
            "which remains PET-free. The target is amyloid biomarker status and preclinical risk, ",
            "not clinical Alzheimer diagnosis.\n\n",
            "## Source Status\n\n",
        ]
    )
    for rel in FILES:
        status_line(lines, rel, read(rel))
    lines.append("\n")

    readiness = read("results/tables/method_y_training_readiness.json") or {}
    progress = readiness.get("pet_preprocess_progress") or {}
    lines.extend(
        [
            "## Execution Boundary\n\n",
            f"- training readiness: `{readiness.get('status', 'MISSING')}`\n",
            f"- PET preprocessing complete: `{progress.get('preprocessing_complete', 'NA')}`\n",
            f"- PET preprocessing progress: `{progress.get('resume_done', 'NA')}/{progress.get('linked_work_items', 'NA')}`\n",
            "- Method X compliant: `False` for PET-containing Method Y rows.\n",
            "- clinical Alzheimer diagnosis allowed: `False`.\n",
            "- public outputs: aggregate reports only; no raw paths, raw IDs, images, HDF5/cache, checkpoints, or subject-level predictions.\n\n",
        ]
    )

    append_incremental_table(lines, read("results/tables/method_y_incremental_value_table.json"))

    synthetic = read("results/tables/synthetic_mri_pet_supplement.json") or {}
    augmentation = read("results/tables/augmentation_policy.json") or {}
    lines.extend(
        [
            "## Supplementary Policies\n\n",
            f"- synthetic MRI-to-PET status: `{synthetic.get('status', 'MISSING')}`\n",
            f"- synthetic generator fit scope: `{synthetic.get('fit_generator_scope', 'NA')}`\n",
            f"- synthetic added to: `{synthetic.get('add_synthetic_to', 'NA')}`\n",
            f"- synthetic counted as real subjects: `{synthetic.get('synthetic_counted_as_real_subjects', 'NA')}`\n",
            f"- augmentation real train N: `{augmentation.get('real_train_n', 'NA')}`\n",
            f"- augmentation training exposures: `{augmentation.get('augmented_training_exposures', 'NA')}`\n",
            f"- augmented or synthetic counted as real subjects: `{augmentation.get('synthetic_or_augmented_counted_as_real_subjects', 'NA')}`\n",
        ]
    )
    out = PROJECT_ROOT / "results" / "final_results" / "method_y" / "phase_y_summary.md"
    atomic_write_text(out, "".join(lines))
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
