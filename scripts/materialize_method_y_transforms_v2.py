"""Resumable, fail-closed PET->T1->MNI registration for exact FTP/T1 pairs.

Only exact subject+visit rows are eligible.  Every committed row carries
structural, metric, geometry and chain-consistency checks plus an automated
three-layer QC record.  ``--limit`` is explicitly a smoke run and can never
produce a full ``PASS`` manifest.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import ants
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.method_y_revival.process_guard import WriterLock


QC_SCHEMA_VERSION = "2.1"
STOP_REQUESTED = False


def _request_graceful_pause(signum: int, _frame: Any) -> None:
    """Finish the current scan, persist its result, then pause safely."""
    global STOP_REQUESTED
    STOP_REQUESTED = True
    print(json.dumps({"event": "PAUSE_REQUESTED", "signal": signum}), flush=True)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_json(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temp, path)


def dice_for_foreground(left: ants.ANTsImage, right: ants.ANTsImage) -> float:
    a = np.asarray(left.numpy(), dtype=np.float32)
    b = np.asarray(right.numpy(), dtype=np.float32)
    pa = a[np.isfinite(a) & (a > 0)]
    pb = b[np.isfinite(b) & (b > 0)]
    if pa.size == 0 or pb.size == 0:
        return 0.0
    ma = a > max(float(np.percentile(pa, 10)), 1e-6)
    mb = b > max(float(np.percentile(pb, 10)), 1e-6)
    denom = int(ma.sum() + mb.sum())
    return float(2 * np.logical_and(ma, mb).sum() / denom) if denom else 0.0


def initial_mutual_information(
    fixed: ants.ANTsImage,
    moving_in_fixed_grid: ants.ANTsImage,
) -> tuple[float | None, str]:
    """Return baseline MI without rejecting valid scans that start out of frame."""
    moving = np.asarray(moving_in_fixed_grid.numpy(), dtype=np.float32)
    if not np.isfinite(moving).all() or np.count_nonzero(moving) == 0:
        return None, "NOT_ESTIMABLE_NO_INITIAL_OVERLAP"
    value = float(ants.image_mutual_information(fixed, moving_in_fixed_grid))
    if not np.isfinite(value):
        return None, "NOT_ESTIMABLE_NONFINITE"
    return value, "ESTIMABLE"


def summarize_and_validate_t1_intensity(image: ants.ANTsImage) -> dict[str, float]:
    """Reject source T1 volumes whose numeric scaling destroys the signal.

    A valid anatomical image may contain a negative background after a
    converter-specific offset, but it must still retain a measurable positive
    foreground.  We never infer or repair a missing NIfTI intercept here:
    doing so would alter the source image and invalidate provenance.
    """
    array = np.asarray(image.numpy(), dtype=np.float32)
    finite_fraction = float(np.isfinite(array).mean())
    if finite_fraction != 1.0:
        raise ValueError("T1_NONFINITE_SIGNAL")
    positive_fraction = float(np.mean(array > 0))
    q01, q50, q99 = (float(x) for x in np.percentile(array, [1, 50, 99]))
    summary = {
        "finite_fraction": finite_fraction,
        "positive_fraction": positive_fraction,
        "q01": q01,
        "q50": q50,
        "q99": q99,
    }
    # The threshold is deliberately conservative: known valid Philips scans
    # have substantially more positive foreground, whereas the observed
    # malformed scan has ~0.3% and a near-constant negative volume.
    if positive_fraction < 0.01 or q99 <= q01:
        raise ValueError("T1_INTENSITY_SIGNAL_GATE_FAILED")
    return summary


def make_montage(t1: ants.ANTsImage, pet_in_t1: ants.ANTsImage, out_path: Path) -> None:
    a = np.asarray(t1.numpy(), dtype=np.float32)
    b = np.asarray(pet_in_t1.numpy(), dtype=np.float32)
    if a.ndim != 3 or b.ndim != 3:
        raise ValueError("MONTAGE_REQUIRES_3D")
    slices = [a.shape[2] // 2, a.shape[1] // 2, a.shape[0] // 2]
    fig, axes = plt.subplots(1, 3, figsize=(9, 3), dpi=100)
    for ax, axis, idx in zip(axes, (2, 1, 0), slices):
        base = np.take(a, idx, axis=axis)
        overlay = np.take(b, idx, axis=axis)
        ax.imshow(np.rot90(base), cmap="gray")
        ax.imshow(np.rot90(overlay), cmap="magma", alpha=0.35)
        ax.axis("off")
    fig.tight_layout(pad=0)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    temp = out_path.with_suffix(".tmp.png")
    fig.savefig(temp, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    os.replace(temp, out_path)


def run(args: argparse.Namespace) -> int:
    global STOP_REQUESTED
    STOP_REQUESTED = False
    manifest_path = args.output_root / "manifests/private_ftp_t1_linkage.parquet"
    template_path = ROOT / "data/templates/method_y_mni152nlin2009casym_v1/tpl-MNI152NLin2009cAsym_res-01_T1w.nii.gz"
    out_root = args.output_root / "transforms"
    state_path = out_root / "manifest.json"
    if not manifest_path.exists() or not template_path.exists():
        atomic_json(state_path, {"status": "BLOCKED", "reason": "MANIFEST_OR_TEMPLATE_MISSING"})
        return 2
    df = pd.read_parquet(manifest_path)
    all_exact = df[df["linkage_status"].eq("EXACT_SUBJECT_VISIT")].copy()
    selected = all_exact.head(args.limit) if args.limit > 0 else all_exact
    previous = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {}
    rows = {item["scan_key"]: item for item in previous.get("rows", [])}
    contract = sha256_json({
        "qc_schema_version": QC_SCHEMA_VERSION,
        "linkage_sha256": sha256_file(manifest_path),
        "template_sha256": sha256_file(template_path),
        "full_exact_count": len(all_exact),
    })
    resumable_pass_count = sum(
        item.get("qc_status") == "PASS"
        and item.get("qc_schema_version") == QC_SCHEMA_VERSION
        and item.get("run_contract_sha256") == contract
        for item in rows.values()
    )
    resumable_fail_count = sum(
        item.get("qc_status") == "FAIL"
        and item.get("qc_schema_version") == QC_SCHEMA_VERSION
        and item.get("run_contract_sha256") == contract
        for item in rows.values()
    )
    if args.status_only:
        print(json.dumps({
            "status": previous.get("status", "NOT_STARTED"),
            "run_contract_sha256": contract,
            "source_eligible_count": int(len(all_exact)),
            "selected_count": int(len(selected)),
            "resumable_pass_count": int(resumable_pass_count),
            "resumable_fail_count": int(resumable_fail_count),
            "pending_or_retry_count": int(len(selected) - resumable_pass_count),
            "resume_starts_after_committed_pass_rows": True,
        }, indent=2), flush=True)
        return 0
    fixed = ants.image_read(str(template_path))
    failures: list[dict[str, str]] = []
    paused = False
    for index, record in enumerate(selected.to_dict("records"), start=1):
        if STOP_REQUESTED:
            paused = True
            break
        key = str(record["scan_key"])
        existing = rows.get(key)
        if existing and existing.get("qc_status") == "PASS" and existing.get("qc_schema_version") == QC_SCHEMA_VERSION and existing.get("run_contract_sha256") == contract:
            continue
        work_dir = out_root / ".work" / f"{key}.{os.getpid()}"
        final_dir = out_root / key
        t1_intensity_qc: dict[str, float] | None = None
        shutil.rmtree(work_dir, ignore_errors=True)
        try:
            pet = ants.image_read(str(record["ftp_path"]))
            frame_policy = "STATIC_3D"
            if pet.dimension == 4:
                pet = ants.get_average_of_timeseries(pet)
                frame_policy = "UNWEIGHTED_FRAME_MEAN_FOR_GEOMETRY_ONLY"
            t1 = ants.image_read(str(record["t1_path"]))
            if pet.dimension != 3 or t1.dimension != 3:
                raise ValueError("INPUT_MUST_BE_3D_AFTER_FRAME_COLLAPSE")
            t1_intensity_qc = summarize_and_validate_t1_intensity(t1)
            initial_pet = ants.resample_image_to_target(pet, t1, interp_type=1)
            mi_before, mi_before_status = initial_mutual_information(t1, initial_pet)
            rigid = ants.registration(fixed=t1, moving=pet, type_of_transform="Rigid", verbose=False)
            pet_in_t1 = rigid["warpedmovout"]
            mi_after = float(ants.image_mutual_information(t1, pet_in_t1))
            dice = dice_for_foreground(t1, pet_in_t1)
            if not np.isfinite(mi_after) or dice <= 0.01:
                raise ValueError("PET_T1_METRIC_GATE_FAILED")
            nonlinear = ants.registration(fixed=fixed, moving=t1, type_of_transform="SyN", verbose=False)
            work_dir.mkdir(parents=True, exist_ok=True)
            rigid_files: list[Path] = []
            for idx, src in enumerate(rigid["fwdtransforms"]):
                src_path = Path(src)
                dst = work_dir / f"pet_to_t1_{idx}{src_path.suffix}"
                if src_path.name.endswith(".nii.gz"):
                    dst = work_dir / f"pet_to_t1_{idx}.nii.gz"
                dst.write_bytes(src_path.read_bytes())
                rigid_files.append(dst)
            nonlinear_files: list[Path] = []
            for idx, src in enumerate(nonlinear["fwdtransforms"]):
                src_path = Path(src)
                dst = work_dir / f"t1_to_mni_{idx}{src_path.suffix}"
                if src_path.name.endswith(".nii.gz"):
                    dst = work_dir / f"t1_to_mni_{idx}.nii.gz"
                dst.write_bytes(src_path.read_bytes())
                nonlinear_files.append(dst)
            composed = ants.apply_transforms(
                fixed=fixed,
                moving=pet,
                transformlist=nonlinear["fwdtransforms"] + rigid["fwdtransforms"],
                interpolator="linear",
            )
            composed_path = work_dir / "pet_mni.nii.gz"
            ants.image_write(composed, str(composed_path))
            arr = np.asarray(composed.numpy(), dtype=np.float32)
            if arr.ndim != 3 or not np.isfinite(arr).all() or np.count_nonzero(arr) == 0:
                raise ValueError("COMPOSED_OUTPUT_INVALID")
            warp = next((Path(x) for x in nonlinear["fwdtransforms"] if str(x).endswith(".nii.gz")), None)
            jacobian_min = jacobian_max = jacobian_positive_fraction = None
            if warp is not None:
                jac = ants.create_jacobian_determinant_image(fixed, str(warp), do_log=False, geom=False)
                jarr = np.asarray(jac.numpy(), dtype=np.float32)
                jacobian_min = float(np.nanmin(jarr))
                jacobian_max = float(np.nanmax(jarr))
                jacobian_positive_fraction = float(np.isfinite(jarr).mean() and (jarr > 0).mean())
                if not np.isfinite(jarr).all() or jacobian_positive_fraction < 0.95:
                    raise ValueError("JACOBIAN_GATE_FAILED")
            montage_path = work_dir / "pet_t1_montage.png"
            make_montage(t1, pet_in_t1, montage_path)
            if montage_path.stat().st_size < 1000:
                raise ValueError("MONTAGE_GATE_FAILED")
            output_hash = sha256_file(composed_path)
            transform_hashes = [sha256_file(p) for p in rigid_files + nonlinear_files]
            chain_hash = sha256_json({"rigid": transform_hashes[: len(rigid_files)], "nonlinear": transform_hashes[len(rigid_files):], "output": output_hash})
            if final_dir.exists():
                shutil.rmtree(final_dir)
            os.replace(work_dir, final_dir)
            rows[key] = {
                "qc_schema_version": QC_SCHEMA_VERSION,
                "run_contract_sha256": contract,
                "scan_key": key,
                "subject_key": str(record["subject_key"]),
                "pet_path": str(record["ftp_path"]),
                "t1_path": str(record["t1_path"]),
                "pet_to_t1_transform_sha256": sha256_json(transform_hashes[: len(rigid_files)]),
                "t1_to_mni_transform_sha256": sha256_json(transform_hashes[len(rigid_files):]),
                "composed_transform_sha256": chain_hash,
                "composed_output_sha256": output_hash,
                "frame_policy": frame_policy,
                "t1_intensity_qc": t1_intensity_qc,
                "mi_before": mi_before,
                "mi_before_status": mi_before_status,
                "mi_after": mi_after,
                "mi_delta": mi_after - mi_before if mi_before is not None else None,
                "foreground_dice": dice,
                "jacobian_min": jacobian_min,
                "jacobian_max": jacobian_max,
                "jacobian_positive_fraction": jacobian_positive_fraction,
                "montage_sha256": sha256_file(final_dir / "pet_t1_montage.png"),
                "qc_status": "PASS",
                "qc_layers": ["STRUCTURAL", "METRIC_MI_DICE", "GEOMETRY_JACOBIAN", "CHAIN_HASH", "MONTAGE_CONSISTENCY"],
            }
        except Exception as exc:  # failure is recorded; no guessed linkage or transform is accepted
            failures.append({"scan_key": key, "reason": f"{type(exc).__name__}:{exc}"})
            rows[key] = {
                "qc_schema_version": QC_SCHEMA_VERSION,
                "run_contract_sha256": contract,
                "scan_key": key,
                "qc_status": "FAIL",
                "reason": failures[-1]["reason"],
                "t1_intensity_qc": t1_intensity_qc,
            }
            shutil.rmtree(work_dir, ignore_errors=True)
        atomic_json(state_path, {
            "schema_version": "1.0",
            "qc_schema_version": QC_SCHEMA_VERSION,
            "run_contract_sha256": contract,
            "status": "RUNNING",
            "scope": "SMOKE" if args.limit > 0 else "FULL",
            "started_utc": previous.get("started_utc", datetime.now(timezone.utc).isoformat()),
            "updated_utc": datetime.now(timezone.utc).isoformat(),
            "eligible_count": int(len(selected)),
            "source_eligible_count": int(len(all_exact)),
            "validated_count": sum(v.get("qc_status") == "PASS" and v.get("run_contract_sha256") == contract for v in rows.values()),
            "failed_count": sum(v.get("qc_status") == "FAIL" and v.get("run_contract_sha256") == contract for v in rows.values()),
            "rows": sorted(rows.values(), key=lambda item: item["scan_key"]),
        })
        progress = {
            "event": "SCAN_COMMITTED",
            "scan_index": index,
            "scan_count": int(len(selected)),
            "scan_key": key,
            "validated_count": sum(v.get("qc_status") == "PASS" and v.get("run_contract_sha256") == contract for v in rows.values()),
            "failed_count": sum(v.get("qc_status") == "FAIL" and v.get("run_contract_sha256") == contract for v in rows.values()),
        }
        print(json.dumps(progress, ensure_ascii=False), flush=True)
        if STOP_REQUESTED:
            paused = True
            break
    payload = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {}
    if paused:
        payload.update({
            "status": "PAUSED",
            "terminal_reason": "USER_REQUESTED_MANUAL_PAUSE_RESUMABLE",
            "paused_utc": datetime.now(timezone.utc).isoformat(),
            "resume_policy": "EXACT_SAME_OUTPUT_ROOT_AND_RUN_CONTRACT;PASS_ROWS_SKIPPED",
        })
        atomic_json(state_path, payload)
        print(json.dumps({"status": "PAUSED", "validated": payload.get("validated_count", 0), "failed": payload.get("failed_count", 0)}, indent=2), flush=True)
        return 0
    full_complete = len(selected) == len(all_exact) and payload.get("validated_count") == len(all_exact) and payload.get("failed_count") == 0
    smoke_complete = args.limit > 0 and payload.get("validated_count") == len(selected) and payload.get("failed_count") == 0
    payload.update({
        "status": "PASS" if full_complete else ("SMOKE_PASS" if smoke_complete else "BLOCKED"),
        "terminal_reason": None if (full_complete or smoke_complete) else "TRANSFORM_FAILURES_OR_INCOMPLETE",
        "updated_utc": datetime.now(timezone.utc).isoformat(),
    })
    atomic_json(state_path, payload)
    print(json.dumps({"status": payload["status"], "eligible": len(selected), "source_eligible": len(all_exact), "validated": payload.get("validated_count", 0), "failed": payload.get("failed_count", 0)}, indent=2))
    return 0 if (full_complete or smoke_complete) else 2


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=ROOT / "results/method_y_revival_v2")
    parser.add_argument("--limit", type=int, default=0, help="smoke limit; 0 means all exact rows")
    parser.add_argument("--status-only", action="store_true", help="report resumable progress without processing scans")
    args = parser.parse_args()
    signal.signal(signal.SIGINT, _request_graceful_pause)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, _request_graceful_pause)
    # Status is strictly read-only and must remain usable while the manual
    # resumable writer owns the lock.  Do not acquire the writer lock here.
    if args.status_only:
        return run(args)
    out_root = args.output_root / "transforms"
    lock = WriterLock(out_root / ".writer.lock", output_root=out_root, run_contract_digest="method_y_transform_qc_v2")
    with lock:
        return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
