from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.build_mri_volume_cache import build_t1_hash_index
from scripts.check_spm_toolchain import check_spm_toolchain, write_spm_toolchain_report
from scripts.spm_preprocess_backend import (
    gzip_file,
    is_windows_matlab,
    prepare_input_nii,
    run_matlab_batch,
    write_spm_batch,
)
from src.data.a4_schema import resolve_data_root
from src.utils.io_helpers import atomic_write_json, atomic_write_text
from src.utils.resumable import ResumableManifest, config_digest
from src.utils.run_manifest import write_run_manifest
from src.utils.state_manager import StateManager


PREPROCESSING_VERSION = "spm_v2_mni_brain_robust_intensity"
QUALITY_STEPS = [
    "spm_unified_segmentation_bias_correction",
    "spm_tissue_probability_brain_mask",
    "spm_forward_deformation_mni_normalization",
    "brain_masked_robust_percentile_clipping_0_5_99_5",
    "brain_masked_zscore_intensity_normalization",
    "hashed_numeric_qc_brain_volume_tissue_ratios_cnr_intensity",
]
QC_WARNING_THRESHOLDS = {
    "min_brain_volume_ml": 800.0,
    "max_brain_volume_ml": 1800.0,
    "min_gm_fraction": 0.25,
    "max_gm_fraction": 0.70,
    "min_gm_wm_cnr": 0.25,
    "max_abs_normalized_brain_mean": 0.15,
    "min_normalized_brain_std": 0.75,
    "max_normalized_brain_std": 1.25,
}


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def make_spm_scratch_dir(out_dir: Path, sample_hash: str) -> Path:
    scratch_root = out_dir.resolve() / "_scratch_spm"
    scratch_root.mkdir(parents=True, exist_ok=True)
    for attempt in range(100):
        suffix = utc_stamp()
        scratch = scratch_root / f"{sample_hash}_{suffix}_{attempt:02d}"
        try:
            scratch.mkdir(parents=True, exist_ok=False)
            return scratch
        except FileExistsError:
            continue
    raise RuntimeError(f"Could not allocate unique SPM scratch directory for {sample_hash}")


def build_resume_manifest(path: str, *, stage: str, config: dict[str, Any]) -> ResumableManifest:
    resume = ResumableManifest(path, stage=stage, config=config)
    try:
        resume.load()
    except ValueError as exc:
        if "config hash mismatch" not in str(exc):
            raise
        manifest_path = Path(path)
        archived = manifest_path.with_name(f"{manifest_path.stem}.mismatch.{utc_stamp()}{manifest_path.suffix}")
        manifest_path.replace(archived)
    return ResumableManifest(path, stage=stage, config=config)


def write_full_mri_backend_report(report: dict[str, object]) -> None:
    atomic_write_json("results/tables/full_mri_preprocess_toolchain_report.json", report)
    lines = [
        "# Full MRI Preprocessing Toolchain Report\n\n",
        f"- status: `{report['status']}`\n",
        "- backend: `spm`\n",
        f"- preprocessing_version: `{report.get('preprocessing_version')}`\n",
        f"- quality_steps: `{report.get('quality_steps')}`\n",
        f"- missing: `{report.get('missing', [])}`\n",
        f"- summary: `{report.get('summary')}`\n",
        "- contains_pet_input: `False`\n",
        "- contains_raw_identifiers: `False`\n",
        "- output_granularity: `hashed_sample_level`\n",
    ]
    atomic_write_text("results/tables/full_mri_preprocess_toolchain_report.md", "".join(lines))


def summarize_spm_numeric_qc(
    *,
    sample_hash: str,
    bias_corrected: Any,
    c1: Any,
    c2: Any,
    c3: Any,
    normalized_brain: Any,
    voxel_volume_ml: float,
) -> dict[str, object]:
    import numpy as np

    gm = np.nan_to_num(c1.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    wm = np.nan_to_num(c2.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    csf = np.nan_to_num(c3.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    corrected = np.nan_to_num(bias_corrected.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    norm = np.nan_to_num(normalized_brain.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)

    tissue_sum = gm + wm + csf
    brain_mask = tissue_sum > 0.20
    norm_mask = norm != 0
    tissue_total = float(tissue_sum[brain_mask].sum()) if np.any(brain_mask) else 0.0

    def fraction(prob):
        return float(prob[brain_mask].sum() / tissue_total) if tissue_total > 0 else 0.0

    gm_mask = gm > 0.50
    wm_mask = wm > 0.50
    brain_values = corrected[brain_mask]
    gm_mean = float(corrected[gm_mask].mean()) if np.any(gm_mask) else 0.0
    wm_mean = float(corrected[wm_mask].mean()) if np.any(wm_mask) else 0.0
    brain_std = float(brain_values.std()) if brain_values.size else 1.0
    if brain_std < 1e-6:
        brain_std = 1.0
    gm_wm_cnr = float(abs(wm_mean - gm_mean) / brain_std)

    norm_values = norm[norm_mask]
    normalized_brain_mean = float(norm_values.mean()) if norm_values.size else 0.0
    normalized_brain_std = float(norm_values.std()) if norm_values.size else 0.0
    brain_volume_ml = float(np.count_nonzero(brain_mask) * voxel_volume_ml)

    warnings = []
    if not (QC_WARNING_THRESHOLDS["min_brain_volume_ml"] <= brain_volume_ml <= QC_WARNING_THRESHOLDS["max_brain_volume_ml"]):
        warnings.append("brain_volume_outside_expected_range")
    gm_fraction = fraction(gm)
    if not (QC_WARNING_THRESHOLDS["min_gm_fraction"] <= gm_fraction <= QC_WARNING_THRESHOLDS["max_gm_fraction"]):
        warnings.append("gray_matter_fraction_outside_expected_range")
    if gm_wm_cnr < QC_WARNING_THRESHOLDS["min_gm_wm_cnr"]:
        warnings.append("low_gm_wm_contrast_to_noise")
    if abs(normalized_brain_mean) > QC_WARNING_THRESHOLDS["max_abs_normalized_brain_mean"]:
        warnings.append("normalized_brain_mean_not_centered")
    if not (
        QC_WARNING_THRESHOLDS["min_normalized_brain_std"]
        <= normalized_brain_std
        <= QC_WARNING_THRESHOLDS["max_normalized_brain_std"]
    ):
        warnings.append("normalized_brain_std_outside_expected_range")

    return {
        "sample_hash": sample_hash,
        "qc_status": "PASS" if not warnings else "WARNING",
        "warnings": warnings,
        "brain_volume_ml": round(brain_volume_ml, 3),
        "brain_mask_voxels": int(np.count_nonzero(brain_mask)),
        "tissue_fraction_gm": round(gm_fraction, 6),
        "tissue_fraction_wm": round(fraction(wm), 6),
        "tissue_fraction_csf": round(fraction(csf), 6),
        "gm_wm_cnr": round(gm_wm_cnr, 6),
        "normalized_brain_mean": round(normalized_brain_mean, 6),
        "normalized_brain_std": round(normalized_brain_std, 6),
        "normalized_nonzero_fraction": round(float(np.count_nonzero(norm_mask) / norm.size), 6),
        "contains_pet_input": False,
        "contains_raw_identifiers": False,
    }


def write_spm_numeric_qc(
    *,
    sample_hash: str,
    scratch: Path,
    final_path: Path,
    out_dir: Path,
) -> dict[str, object]:
    try:
        import nibabel as nib
    except Exception as exc:
        return {
            "sample_hash": sample_hash,
            "qc_status": "UNAVAILABLE",
            "warnings": [f"nibabel_unavailable:{type(exc).__name__}"],
            "contains_pet_input": False,
            "contains_raw_identifiers": False,
        }

    paths = {
        "bias_corrected": scratch / "minput.nii",
        "c1": scratch / "c1input.nii",
        "c2": scratch / "c2input.nii",
        "c3": scratch / "c3input.nii",
    }
    missing = [name for name, path in paths.items() if not path.exists()]
    if missing:
        return {
            "sample_hash": sample_hash,
            "qc_status": "UNAVAILABLE",
            "warnings": [f"missing_spm_qc_inputs:{','.join(missing)}"],
            "contains_pet_input": False,
            "contains_raw_identifiers": False,
        }

    bias_img = nib.load(str(paths["bias_corrected"]))
    zooms = bias_img.header.get_zooms()[:3]
    qc = summarize_spm_numeric_qc(
        sample_hash=sample_hash,
        bias_corrected=bias_img.get_fdata(dtype="float32"),
        c1=nib.load(str(paths["c1"])).get_fdata(dtype="float32"),
        c2=nib.load(str(paths["c2"])).get_fdata(dtype="float32"),
        c3=nib.load(str(paths["c3"])).get_fdata(dtype="float32"),
        normalized_brain=nib.load(str(final_path)).get_fdata(dtype="float32"),
        voxel_volume_ml=float(zooms[0] * zooms[1] * zooms[2] / 1000.0),
    )
    qc_dir = out_dir / "qc"
    qc_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(qc_dir / f"{sample_hash}.json", qc)
    return qc


def preprocess_one_spm(
    *,
    sample_hash: str,
    source_t1: Path,
    spm_dir: Path,
    matlab_cmd: str,
    out_dir: Path,
) -> tuple[bool, str]:
    source_t1 = source_t1.resolve()
    out_dir = out_dir.resolve()
    scratch = make_spm_scratch_dir(out_dir, sample_hash)
    input_nii = scratch / "input.nii"
    spm_output = scratch / "spm_mni_brain.nii"
    batch_path = scratch / "run_spm_preprocess.m"
    log_path = scratch / "matlab_spm.log"
    final_path = out_dir / f"{sample_hash}.nii.gz"
    qc_path = out_dir / "qc" / f"{sample_hash}.json"
    qc_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_final = final_path.with_suffix(final_path.suffix + ".tmp")

    prepare_input_nii(source_t1, input_nii)
    write_spm_batch(
        batch_path=batch_path,
        spm_dir=spm_dir,
        input_nii=input_nii,
        output_nii=spm_output,
        windows_matlab=is_windows_matlab(matlab_cmd),
        sample_hash=sample_hash,
        qc_json=qc_path,
    )
    code = run_matlab_batch(matlab_cmd, batch_path, log_path=log_path)
    if code != 0:
        return False, f"matlab_returncode_{code}_log_{log_path}"
    if not spm_output.exists():
        return False, "spm_output_missing"
    if not qc_path.exists():
        return False, "spm_qc_json_missing"
    try:
        qc = json.loads(qc_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False, "spm_qc_json_unreadable"
    if float(qc.get("normalized_nonzero_fraction") or 0.0) <= 0.0 or float(qc.get("normalized_brain_std") or 0.0) <= 0.0:
        return False, "spm_qc_empty_normalized_output"
    gzip_file(spm_output, tmp_final)
    os.replace(tmp_final, final_path)
    return True, str(final_path)


def count_done_items(rows: list[dict[str, str]], resume: ResumableManifest) -> int:
    return sum(1 for row in rows if resume.item_done(row["sample_hash"]))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default=os.environ.get("DATA_ROOT"))
    ap.add_argument("--catalog", default="data/processed/master_catalog.csv")
    ap.add_argument("--matlab", default=os.environ.get("MATLAB_BIN"))
    ap.add_argument("--spm_dir", default=os.environ.get("SPM_DIR"))
    ap.add_argument("--out_dir", default="data/preprocessed_mri_96")
    ap.add_argument("--manifest", default="results/logs/spm_mri_preprocess_v2_resume.json")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument(
        "--batch_size",
        type=int,
        default=0,
        help="Process at most N pending catalog rows this run. Use for resumable full preprocessing batches.",
    )
    ap.add_argument(
        "--auto_batches",
        action="store_true",
        help="Keep processing pending rows in batches until all rows are completed or a batch fails.",
    )
    ap.add_argument("--execute", action="store_true")
    args = ap.parse_args()
    if args.limit > 0 and args.batch_size > 0:
        print(json.dumps({"status": "BLOCKED", "reason": "use either --limit for smoke or --batch_size for full batches, not both"}, indent=2))
        return 2
    if args.limit > 0 and args.auto_batches:
        print(json.dumps({"status": "BLOCKED", "reason": "use --limit for smoke or --auto_batches for full preprocessing, not both"}, indent=2))
        return 2
    if args.auto_batches and args.batch_size <= 0:
        args.batch_size = 10

    toolchain = check_spm_toolchain(args.matlab, args.spm_dir)
    write_spm_toolchain_report(toolchain)
    report: dict[str, object] = {
        "status": toolchain["status"],
        "backend": "spm",
        "toolchain": toolchain,
        "missing": toolchain["missing"],
        "execute_requested": bool(args.execute),
        "batch_size": args.batch_size,
        "auto_batches": bool(args.auto_batches),
        "contains_pet_input": False,
        "contains_raw_identifiers": False,
        "output_resolution": 96,
        "preprocessing_version": PREPROCESSING_VERSION,
        "quality_steps": QUALITY_STEPS,
    }
    if toolchain["status"] != "READY":
        write_full_mri_backend_report(report)
        StateManager().append_event("preprocess_full_mri", "BLOCKED", "SPM full MRI preprocessing blocked by missing MATLAB/SPM toolchain.")
        print(json.dumps({"status": "BLOCKED", "missing": toolchain["missing"], "backend": "spm"}, indent=2))
        return 2
    if not args.execute:
        report["status"] = "READY_FOR_EXECUTION"
        write_full_mri_backend_report(report)
        print(json.dumps({"status": "READY_FOR_EXECUTION", "backend": "spm", "message": "rerun with --execute"}, indent=2))
        return 0

    data_root = resolve_data_root(args.data_root)
    rows = read_rows(Path(args.catalog))
    if args.limit > 0:
        rows = rows[: args.limit]
    t1_index = build_t1_hash_index(data_root)
    out_dir = Path(args.out_dir)
    config = {
        "backend": "spm",
        "catalog": args.catalog,
        "out_dir": args.out_dir,
        "spm_dir": str(toolchain["spm_dir"]),
        "resolution": 96,
        "preprocessing_version": PREPROCESSING_VERSION,
        "quality_steps": QUALITY_STEPS,
    }
    resume = build_resume_manifest(args.manifest, stage="preprocess_full_mri_spm", config=config)
    ok = failed = missing = deferred = attempted = batches_completed = 0
    initial_done = count_done_items(rows, resume)
    while True:
        pending = [row for row in rows if not resume.item_done(row["sample_hash"])]
        if not pending:
            break
        batch_rows = pending[: args.batch_size] if args.batch_size > 0 else pending
        deferred = max(0, len(pending) - len(batch_rows))
        if not batch_rows:
            break
        batches_completed += 1
        batch_failed = batch_missing = 0
        for row in batch_rows:
            sample_hash = row["sample_hash"]
            attempted += 1
            source = t1_index.get(row["t1_file_hash"])
            if source is None:
                missing += 1
                batch_missing += 1
                resume.set_item(sample_hash, "FAILED", message="matching_t1_not_visible")
                continue
            resume.set_item(sample_hash, "RUNNING")
            success, detail = preprocess_one_spm(
                sample_hash=sample_hash,
                source_t1=source,
                spm_dir=Path(str(toolchain["spm_dir"])),
                matlab_cmd=str(toolchain["matlab"]),
                out_dir=out_dir,
            )
            if success:
                ok += 1
                resume.set_item(sample_hash, "DONE", artifact=detail)
            else:
                failed += 1
                batch_failed += 1
                resume.set_item(sample_hash, "FAILED", message=detail)
        if batch_failed or batch_missing or not args.auto_batches:
            break

    final_done = count_done_items(rows, resume)
    summary = {
        "ok": ok,
        "failed": failed,
        "missing": missing,
        "skipped_done": initial_done,
        "done_total": final_done,
        "deferred_pending": deferred,
        "attempted_this_run": attempted,
        "batches_completed": batches_completed,
        "total": len(rows),
        "limit": args.limit,
        "batch_size": args.batch_size,
        "is_smoke_test": args.limit > 0,
        "is_batch_run": args.batch_size > 0,
        "auto_batches": bool(args.auto_batches),
    }
    if args.limit > 0 and failed == 0 and missing == 0:
        status = "SMOKE_TEST_COMPLETED"
    elif args.batch_size > 0 and failed == 0 and missing == 0 and deferred > 0:
        status = "BATCH_COMPLETED"
    elif failed == 0 and missing == 0:
        status = "COMPLETED"
    else:
        status = "PARTIAL"
    report.update({"status": status, "summary": summary, "config_hash": config_digest(config)})
    write_full_mri_backend_report(report)
    write_run_manifest("results/logs/spm_mri_preprocess_manifest.json", stage="preprocess_full_mri_spm", data_root=str(data_root), extra=summary)
    StateManager().append_event("preprocess_full_mri", status, "SPM full MRI preprocessing stage finished or paused with resumable manifest.")
    print(json.dumps(report, indent=2))
    return 0 if status in {"COMPLETED", "SMOKE_TEST_COMPLETED", "BATCH_COMPLETED"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
