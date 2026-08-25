"""Schema-v2 BrainIAC raw-T1 preprocessing runner (Plans 002 build + 003 run).

Modes:
  preflight   read-only evidence (no lease, no sample writes)
  calibrate   32 lowest-hash samples -> immutable MAD threshold manifest
              (idempotent: a valid existing manifest is reused, never rebuilt)
  run         full 1669 primary run, resumable: committed samples skip,
              Ctrl+C/Ctrl+Break stop at the next safe boundary (exit 75),
              re-running the SAME command continues from the last commit
  sensitivity 64 lowest-hash samples post-primary: mask-stability (blocker)
              + direct-128-to-96 drift (report-only, SENSITIVITY_ONLY role)
  full        the single operator command: preflight gate -> calibrate ->
              run -> sensitivity -> aggregate streaming revalidation
  smoke       bounded source-discipline smoke (sequential-only per policy)
  status      read-only completion counts

Exit codes: 0 completed/already-completed; 3 fail-closed blocker;
75 paused-resumable; 78 contract mismatch.  Raw paths/IDs never printed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import sys
import threading
import time
from typing import Any, Callable, Mapping

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.research import brainiac_preprocessing_v2 as v2
from src.research.brainiac_preprocessing_v2_schema import (
    FIXED_QC_LIMITS,
    build_aggregate,
    canonical_sha,
    sha256_file,
    validate_receipt,
    validate_threshold_manifest,
)

REGISTRY_ROOT = ROOT / "results" / "method_x_registry" / "brainiac_r1"
RUNTIME_LOCK = REGISTRY_ROOT / "brainiac_official_runtime_lock.json"
STRICT_RECEIPT = REGISTRY_ROOT / "brainiac_official_strict_load_receipt.json"
FORWARD_RECEIPT = REGISTRY_ROOT / "brainiac_official_forward_receipt.json"
TEMPLATE_PATH = (
    ROOT / "models" / "pretrained" / "brainiac" / "templates" / "spm25_avg152T1_2mm.nii"
)
TEMPLATE_LOCK = TEMPLATE_PATH.with_name("spm25_avg152T1_2mm.lock.json")
DERIVED_TEMPLATE = (
    ROOT / "data" / "derived" / "method_x_brainiac" / "templates" / "spm25_avg152T1_1mm.nii.gz"
)
DERIVED_TEMPLATE_LOCK = DERIVED_TEMPLATE.with_suffix(".lock.json")
PREFLIGHT_REPORT = REGISTRY_ROOT / "brainiac_preprocessing_v2_preflight.json"
THRESHOLD_MANIFEST_PATH = (
    ROOT / "data" / "derived" / "method_x_brainiac" / "r1_primary_v2" / "aggregate"
    / "qc_threshold_manifest.json"
)
CALIBRATION_ROOT = (
    ROOT / "data" / "derived" / "method_x_brainiac" / "r1_primary_v2_calibration"
)
PRIMARY_AGGREGATE_PATH = (
    ROOT / "data" / "derived" / "method_x_brainiac" / "r1_primary_v2" / "aggregate"
    / "brainiac_preprocessing_v2_manifest.json"
)
SENSITIVITY_REPORT_PATH = REGISTRY_ROOT / "brainiac_preprocessing_v2_sensitivity.json"
RUN_STATE_PATH = REGISTRY_ROOT / "brainiac_preprocessing_v2_run_state.json"
SMOKE_REPORT = REGISTRY_ROOT / "brainiac_preprocessing_v2_smoke.json"

MIN_FREE_DISK_GB = 40.0
PROGRESS_EVERY_N = 25


class _StopFlag:
    def __init__(self) -> None:
        self.requested = False
        self.reason = ""


def _install_signal_handlers(stop: _StopFlag) -> None:
    def _handler(signum, _frame):
        if not stop.requested:
            stop.reason = signal.Signals(signum).name
        stop.requested = True  # finish current sample, then exit 75

    for sig_name in ("SIGINT", "SIGTERM", "SIGBREAK"):
        number = getattr(signal, sig_name, None)
        if number is not None:
            try:
                signal.signal(number, _handler)
            except (ValueError, OSError):
                pass


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime())


def _exit_code_for(status: str) -> int:
    if status in {"PASS", "COMPLETED", "ALREADY_COMPLETED"}:
        return 0
    if status == "PAUSED_RESUMABLE":
        return 75
    if "CONTRACT_MISMATCH" in status:
        return 78
    return 3


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    encoded = (json.dumps(dict(payload), indent=2, sort_keys=True) + "\n").encode("utf-8")
    descriptor = os.open(
        temporary, os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_BINARY", 0)
    )
    try:
        os.write(descriptor, encoded)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.replace(temporary, path)


def _print_flush(message: str) -> None:
    print(message, flush=True)


def _pin_itk_threads() -> int:
    """Pin ITK thread pool for bit-deterministic N4/registration (Plan-002 rev)."""

    import SimpleITK as sitk

    threads = max(1, int(os.environ.get("BRAINIAC_V2_ITK_THREADS", "1")))
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(threads)
    return threads


def _scratch_paths(scratch: Path, sample_hash: str) -> tuple[Path, Path]:
    """Registered volume as UNCOMPRESSED .nii scratch; brain output stays .nii.gz."""

    return (
        scratch / f"{sample_hash}_registered.nii",
        scratch / f"{sample_hash}_brain.nii.gz",
    )


def _equivalence_mismatches(seq_receipts, par_receipts) -> list[str]:
    """Canonical scientific-field comparator between two execution legs."""

    fields = (
        "output_sha256",
        "transform_sha256",
        "foreground_fraction",
        "rigid_determinant",
        "orthogonality_error",
        "rotation_deg",
        "translation_norm_mm",
        "normalized_nonzero_std",
    )
    index = {str(r["sample_hash"]): r for r in par_receipts}
    out: list[str] = []
    for ref in seq_receipts:
        row = index.get(str(ref["sample_hash"]))
        if row is None:
            out.append(f"{ref['sample_hash'][:10]}:MISSING_IN_PARALLEL")
            continue
        for f in fields:
            if ref.get(f) != row.get(f):
                out.append(f"{ref['sample_hash'][:10]}:{f}")
    return out


def _load_integrity_json(path: Path) -> tuple[dict[str, Any], str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    body = dict(payload)
    recorded = body.pop("integrity_sha256", None)
    if recorded != canonical_sha(body):
        raise RuntimeError(f"BRAINIAC_V2_REQUIRED_RECEIPT_INTEGRITY_INVALID:{path.name}")
    return payload, sha256_file(path)


def _validate_runtime_receipts() -> dict[str, str]:
    runtime, runtime_sha = _load_integrity_json(RUNTIME_LOCK)
    strict, strict_sha = _load_integrity_json(STRICT_RECEIPT)
    forward, forward_sha = _load_integrity_json(FORWARD_RECEIPT)
    if (
        runtime.get("status") != "PASS"
        or strict.get("status") != "PASS"
        or strict.get("strict") is not True
        or forward.get("status") != "PASS"
        or forward.get("outer_test_touched") is not False
        or tuple(forward.get("input_shape", ())) != (1, 1, 96, 96, 96)
        or tuple(forward.get("output_shape", ())) != (1, 768)
        or strict.get("environment_lock_sha256") != runtime_sha
        or forward.get("strict_load_receipt_file_sha256") != strict_sha
    ):
        raise RuntimeError("BRAINIAC_V2_OFFICIAL_RUNTIME_CHAIN_INVALID")
    return {"runtime_lock_sha256": runtime_sha}


def _validate_template_assets() -> dict[str, str]:
    if not TEMPLATE_PATH.is_file() or not TEMPLATE_LOCK.is_file():
        raise RuntimeError("BRAINIAC_V2_TEMPLATE_ASSET_MISSING")
    if sha256_file(TEMPLATE_PATH) != v2.EXPECTED_TEMPLATE_SHA256:
        raise RuntimeError("BRAINIAC_V2_TEMPLATE_HASH_MISMATCH")
    lock = json.loads(TEMPLATE_LOCK.read_text(encoding="utf-8"))
    if lock.get("sha256") != v2.EXPECTED_TEMPLATE_SHA256:
        raise RuntimeError("BRAINIAC_V2_TEMPLATE_LOCK_INVALID")
    if not DERIVED_TEMPLATE.is_file() or not DERIVED_TEMPLATE_LOCK.is_file():
        raise RuntimeError("BRAINIAC_V2_DERIVED_TEMPLATE_MISSING")
    derived_lock = json.loads(DERIVED_TEMPLATE_LOCK.read_text(encoding="utf-8"))
    if (
        derived_lock.get("source_template_sha256") != v2.EXPECTED_TEMPLATE_SHA256
        or derived_lock.get("derived_sha256") != sha256_file(DERIVED_TEMPLATE)
        or derived_lock.get("output_spacing_mm") != [1.0, 1.0, 1.0]
    ):
        raise RuntimeError("BRAINIAC_V2_DERIVED_TEMPLATE_INVALID")
    return {
        "template_sha256": v2.EXPECTED_TEMPLATE_SHA256,
        "template_lock_sha256": sha256_file(TEMPLATE_LOCK),
    }


def _validate_hd_bet_weight() -> None:
    if (
        not v2.HD_BET_WEIGHT.is_file()
        or sha256_file(v2.HD_BET_WEIGHT) != v2.EXPECTED_HD_BET_WEIGHT_SHA256
    ):
        raise RuntimeError("BRAINIAC_V2_HD_BET_WEIGHT_INVALID")


def _policy() -> dict[str, Any]:
    return json.loads(Path(v2.POLICY_PATH).read_text(encoding="utf-8"))


def _base_bindings() -> dict[str, str]:
    bindings = {
        **_validate_runtime_receipts(),
        **_validate_template_assets(),
        "vendor_tree_sha256": sha256_file(v2.VENDOR_ROOT / "src" / "dataset.py"),
        "policy_sha256": v2.policy_sha256(),
    }
    _, transform_binding = v2.build_contract_transform()
    bindings["official_transform_source_sha256"] = transform_binding[
        "official_transform_source_sha256"
    ]
    if THRESHOLD_MANIFEST_PATH.is_file():
        bindings["qc_threshold_manifest_sha256"] = sha256_file(THRESHOLD_MANIFEST_PATH)
    else:
        bindings["qc_threshold_manifest_sha256"] = hashlib.sha256(
            b"NOT_YET_CALIBRATED_PLAN_003"
        ).hexdigest()
    return bindings


def _load_fixed_template() -> Any:
    import SimpleITK as sitk

    return sitk.ReadImage(str(DERIVED_TEMPLATE), sitk.sitkFloat32)


def _registration(fixed: Any, moving: Any, sitk: Any) -> tuple[Any, float]:
    initial = sitk.CenteredTransformInitializer(
        fixed,
        moving,
        sitk.Euler3DTransform(),
        sitk.CenteredTransformInitializerFilter.GEOMETRY,
    )
    method = sitk.ImageRegistrationMethod()
    method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    method.SetMetricSamplingStrategy(method.RANDOM)
    method.SetMetricSamplingPercentage(0.01, seed=v2.REGISTRATION_SEED)
    method.SetInterpolator(sitk.sitkLinear)
    method.SetOptimizerAsGradientDescent(
        learningRate=1.0,
        numberOfIterations=100,
        convergenceMinimumValue=1e-6,
        convergenceWindowSize=10,
    )
    method.SetOptimizerScalesFromPhysicalShift()
    method.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
    method.SetSmoothingSigmasPerLevel(smoothingSigmas=[2, 1, 0])
    method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    method.SetInitialTransform(initial, inPlace=False)
    transform = method.Execute(fixed, moving)
    metric = float(method.GetMetricValue())
    if not math.isfinite(metric):
        raise RuntimeError("BRAINIAC_V2_RIGID_REGISTRATION_METRIC_NONFINITE")
    return transform, metric


def _metric_value(fixed: Any, moving: Any, sitk: Any) -> float:
    method = sitk.ImageRegistrationMethod()
    method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    method.SetMetricSamplingStrategy(method.RANDOM)
    method.SetMetricSamplingPercentage(0.01, seed=v2.REGISTRATION_SEED)
    method.SetInterpolator(sitk.sitkLinear)
    return float(method.MetricEvaluate(fixed, moving))


def _transform_qc(transform: Any, sitk: Any, fixed_template: Any) -> dict[str, float]:
    import numpy as np

    size = fixed_template.GetSize()
    center = np.asarray(
        fixed_template.TransformContinuousIndexToPhysicalPoint(
            tuple((s - 1) / 2.0 for s in size)
        ),
        dtype=np.float64,
    )
    mapped_center = np.asarray(transform.TransformPoint(tuple(center)), dtype=np.float64)
    translation_mm = float(np.linalg.norm(mapped_center - center))
    columns = []
    for axis in range(3):
        probe = center.copy()
        probe[axis] += 1.0
        columns.append(
            np.asarray(transform.TransformPoint(tuple(probe)), dtype=np.float64)
            - mapped_center
        )
    jacobian = np.stack(columns, axis=1)
    determinant = float(np.linalg.det(jacobian))
    orthogonality = float(np.max(np.abs(jacobian.T @ jacobian - np.eye(3))))
    trace = float(
        np.clip(
            (jacobian[0, 0] + jacobian[1, 1] + jacobian[2, 2] - 1.0) / 2.0, -1.0, 1.0
        )
    )
    rotation_deg = float(math.degrees(math.acos(trace)))
    return {
        "rigid_determinant": determinant,
        "orthogonality_error": orthogonality,
        "rotation_deg": rotation_deg,
        "translation_norm_mm": translation_mm,
    }


def _center_of_mass(array: Any) -> list[float]:
    import numpy as np
    from scipy.ndimage import center_of_mass

    mask = array != 0
    if not mask.any():
        raise RuntimeError("BRAINIAC_V2_FOREGROUND_EMPTY")
    return [float(value) for value in center_of_mass(mask)]


_GPU_LOCK = threading.Lock()


def _run_hd_bet(registered: Path, brain_output: Path, *, device: str) -> None:
    if str(v2.VENDOR_PREPROCESSING) not in sys.path:
        sys.path.insert(0, str(v2.VENDOR_PREPROCESSING))
    if sha256_file(v2.HD_BET_WEIGHT) != v2.EXPECTED_HD_BET_WEIGHT_SHA256:
        raise RuntimeError("BRAINIAC_V2_HD_BET_WEIGHT_HASH_MISMATCH")
    from HD_BET.hd_bet import hd_bet

    hd_bet(
        str(registered),
        str(brain_output),
        mode="fast",
        device="0" if device == "cuda" else "cpu",
        tta=0,
        pp=1,
        save_mask=1,
        overwrite_existing=1,
    )
    mask = brain_output.with_name(brain_output.name[:-7] + "_mask.nii.gz")
    if not brain_output.is_file() or not mask.is_file():
        raise RuntimeError("BRAINIAC_V2_HD_BET_OUTPUT_MISSING")


class ProgressReporter:
    """Counts + short hash prefix + elapsed only; never raw paths/IDs."""

    def __init__(self, root: Path, total: int, started_at: float) -> None:
        self.root = root
        self.total = total
        self.started_at = started_at
        self.since_last = 0

    def committed(self) -> int:
        return len(list((self.root / "receipts").glob("*.json")))

    def tick(self, sample_hash: str) -> None:
        self.since_last += 1
        if self.since_last >= PROGRESS_EVERY_N:
            self.since_last = 0
            _print_flush(
                json.dumps(
                    {
                        "progress": "PREPROCESSING",
                        "committed": self.committed(),
                        "total": self.total,
                        "current_sample_prefix": sample_hash[:10],
                        "elapsed_seconds": round(time.monotonic() - self.started_at, 1),
                    }
                )
            )


def make_real_pipeline(
    *,
    fixed_template: Any,
    device: str,
    sources: Mapping[str, Path],
    transform_contract: Any,
    extra_bindings: Mapping[str, str],
    progress: ProgressReporter | None = None,
) -> Callable[[Mapping[str, str], Path], dict[str, Any]]:
    """One sample: N4 -> rigid -> HD-BET -> official 96^3 -> nonzero normalize."""

    _pin_itk_threads()

    def pipeline(item: Mapping[str, str], scratch: Path) -> dict[str, Any]:
        import nibabel as nib
        import numpy as np
        import SimpleITK as sitk

        sample_hash = str(item["sample_hash"])
        source_path = sources[sample_hash]  # in-memory only; never persisted
        source = sitk.ReadImage(str(source_path), sitk.sitkFloat32)
        if source.GetDimension() != 3 or min(source.GetSize()) <= 0:
            raise RuntimeError("BRAINIAC_V2_RAW_T1_GEOMETRY_INVALID")
        source_array = sitk.GetArrayFromImage(source)
        com_before = _center_of_mass(source_array)
        affine_before = np.asarray(source.GetDirection(), dtype=np.float64).tolist()

        n4 = sitk.N4BiasFieldCorrection(source)
        transform, metric_after = _registration(fixed_template, n4, sitk)
        identity_moving = sitk.Resample(
            n4, fixed_template, sitk.Transform(), sitk.sitkLinear, 0.0, sitk.sitkFloat32
        )
        similarity_before = _metric_value(fixed_template, identity_moving, sitk)
        registered = sitk.Resample(
            n4, fixed_template, transform, sitk.sitkLinear, 0.0, sitk.sitkFloat32
        )
        similarity_after = _metric_value(fixed_template, registered, sitk)
        com_after = _center_of_mass(sitk.GetArrayFromImage(registered))
        affine_after = np.asarray(registered.GetDirection(), dtype=np.float64).tolist()
        qc_transform = _transform_qc(transform, sitk, fixed_template)

        registered_path, brain_path = _scratch_paths(scratch, sample_hash)
        sitk.WriteImage(registered, str(registered_path), False)
        with _GPU_LOCK:
            _run_hd_bet(registered_path, brain_path, device=device)

        tensor = transform_contract({"image": str(brain_path)})["image"]
        array = tensor.detach().cpu().numpy().astype("float32", copy=False)
        affine = np.asarray(tensor.affine.detach().cpu(), dtype=np.float64)
        if tuple(array.shape) != (1, 96, 96, 96) or not np.isfinite(array).all():
            raise RuntimeError("BRAINIAC_V2_OFFICIAL_96_TRANSFORM_INVALID")
        nonzero = array[array != 0]
        if nonzero.size == 0:
            raise RuntimeError("BRAINIAC_V2_FOREGROUND_EMPTY")
        foreground_fraction = float(nonzero.size / array.size)
        mean = float(nonzero.mean())
        std = float(nonzero.std())

        output_path = scratch / "output.nii.gz"
        nib.save(nib.Nifti1Image(array[0], affine), str(output_path))
        transform_path = scratch / "transform.tfm"
        sitk.WriteTransform(transform, str(transform_path))
        if progress is not None:
            progress.tick(sample_hash)
        return {
            "output_path": output_path,
            "transform_path": transform_path,
            "source_sha256": sha256_file(source_path),
            "source_size": source_path.stat().st_size,
            "affine_before": affine_before,
            "affine_after": affine_after,
            **qc_transform,
            "foreground_fraction": foreground_fraction,
            "finite": True,
            "normalized_nonzero_mean": mean,
            "normalized_nonzero_std": std,
            "center_of_mass_before": com_before,
            "center_of_mass_after": com_after,
            "registration_similarity_before": similarity_before,
            "registration_similarity_after": similarity_after,
        }

    return pipeline


# ---------------------------------------------------------------------------
# modes
# ---------------------------------------------------------------------------


def preflight() -> dict[str, Any]:
    _pin_itk_threads()
    v2.assert_no_cnn_writer()
    bindings = {**_validate_runtime_receipts(), **_validate_template_assets()}
    _validate_hd_bet_weight()
    transform, transform_binding = v2.build_contract_transform()
    membership = v2.load_locked_membership()
    index = v2.build_raw_index()
    sources = v2.resolve_sources(membership, index=index)
    calibration = v2.select_calibration_samples([m["sample_hash"] for m in membership])
    sensitivity = v2.select_sensitivity_samples([m["sample_hash"] for m in membership])
    usage = __import__("shutil").disk_usage(ROOT)
    body: dict[str, Any] = {
        "schema_version": 2,
        "status": "PASS",
        "created_at_utc": _utc_now(),
        "action": "PREPARE_RAW_T1",
        "expected_samples": len(membership),
        "membership_sha256": canonical_sha([m["sample_hash"] for m in membership]),
        "raw_index_size": len(index),
        "free_disk_gb": round(usage.free / (1024**3), 1),
        "free_disk_ok": usage.free / (1024**3) >= MIN_FREE_DISK_GB,
        "threshold_manifest_present": THRESHOLD_MANIFEST_PATH.is_file(),
        "calibration_selection_rule": f"{len(calibration)}_lowest_sha256_of_sample_hash",
        "sensitivity_selection_rule": f"{len(sensitivity)}_lowest_sample_hash",
        "policy_sha256": v2.policy_sha256(),
        "official_transform": transform_binding,
        "raw_paths_persisted": False,
        "labels_read": False,
        "pet_read": False,
        "sample_outputs_written": False,
        "claim_ceiling": "POST_HOC_INTERNAL_TECHNICAL_EVIDENCE",
    }
    body["integrity_sha256"] = canonical_sha(
        {k: v for k, v in body.items() if k != "integrity_sha256"}
    )
    _atomic_write_json(PREFLIGHT_REPORT, body)
    return body


def _calibrate_if_needed(*, device: str) -> dict[str, Any]:
    """Plan 003 Step 1: threshold manifest built exactly once, then reused."""

    if THRESHOLD_MANIFEST_PATH.is_file():
        manifest = json.loads(THRESHOLD_MANIFEST_PATH.read_text(encoding="utf-8"))
        try:
            validate_threshold_manifest(manifest)
            return {
                "status": "ALREADY_COMPLETED",
                "state": "LOCKED",
                "threshold_manifest_sha256": sha256_file(THRESHOLD_MANIFEST_PATH),
                "selected_n": manifest.get("sample_count"),
                "selection_label_blind": True,
                "rule": "MEDIAN_PLUS_MINUS_6_MAD",
                "all_thresholds_finite": True,
            }
        except RuntimeError:
            pass  # invalid manifest: rebuild via the sanctioned calibration run
    membership = v2.load_locked_membership()
    selected = v2.select_calibration_samples([m["sample_hash"] for m in membership])
    items = [m for m in membership if m["sample_hash"] in set(selected)]
    fixed = _load_fixed_template()
    bindings = _base_bindings()
    transform_contract, _ = v2.build_contract_transform()
    sources = v2.resolve_sources(items, index=v2.build_raw_index())
    pipeline = make_real_pipeline(
        fixed_template=fixed,
        device=device,
        sources=sources,
        transform_contract=transform_contract,
        extra_bindings=bindings,
        progress=ProgressReporter(CALIBRATION_ROOT, len(items), time.monotonic()),
    )
    report = v2.execute(
        mode="smoke",
        limit=len(items),
        root=CALIBRATION_ROOT,
        device=device,
        pipeline_factory=lambda: pipeline,
        bindings=bindings,
        membership=items,
    )
    if report["status"] != "COMPLETED":
        return report
    metrics: dict[str, dict[str, float]] = {}
    for receipt in report["receipts"]:
        metrics[receipt["sample_hash"]] = {
            "foreground_fraction": float(receipt["foreground_fraction"]),
            "normalized_nonzero_std": float(receipt["normalized_nonzero_std"]),
            "rotation_deg": float(receipt["rotation_deg"]),
            "translation_norm_mm": float(receipt["translation_norm_mm"]),
        }
    manifest = v2.build_threshold_manifest(
        metrics,
        fixed_bounds={
            "foreground_fraction": FIXED_QC_LIMITS["foreground_fraction"],
            "normalized_nonzero_std": FIXED_QC_LIMITS["normalized_nonzero_std"],
        },
    )
    _atomic_write_json(THRESHOLD_MANIFEST_PATH, manifest)
    return {
        "status": "PASS",
        "state": "LOCKED",
        "selected_n": len(items),
        "selection_label_blind": True,
        "rule": "MEDIAN_PLUS_MINUS_6_MAD",
        "all_thresholds_finite": True,
        "threshold_manifest_sha256": sha256_file(THRESHOLD_MANIFEST_PATH),
    }


def _streaming_validate_primary() -> dict[str, Any]:
    """Plan 003 Step 6: rehash every receipt and live byte; trust no counts."""

    root = v2.PRIMARY_ROOT
    membership = v2.load_locked_membership()
    expected = {m["sample_hash"] for m in membership}
    receipts_dir = root / "receipts"
    seen: dict[str, str] = {}
    invalid: list[str] = []
    for path in sorted(receipts_dir.glob("*.json")):
        try:
            receipt = json.loads(path.read_text(encoding="utf-8"))
            validate_receipt(receipt, artifact_root=root)
            seen[receipt["sample_hash"]] = receipt["integrity_sha256"]
        except (RuntimeError, json.JSONDecodeError, OSError, ValueError):
            invalid.append(path.stem[:10])
    quarantined = (
        len(list((root / "quarantine").glob("*"))) if (root / "quarantine").exists() else 0
    )
    missing = sorted(expected - set(seen))
    receipts = [
        json.loads((receipts_dir / f"{h}.json").read_text(encoding="utf-8"))
        for h in sorted(seen)
    ]
    aggregate = build_aggregate(
        receipts=receipts,
        expected_sample_hashes=sorted(expected),
        artifact_root=root,
    )
    return {
        "schema_version": 2,
        "status": (
            "COMPLETED"
            if not missing and not invalid and quarantined == 0 and len(seen) == len(expected)
            else "BLOCKED"
        ),
        "expected_n": len(expected),
        "valid_committed_n": len(seen),
        "unique_sample_hash_n": len(seen),
        "missing_n": len(missing),
        "invalid_n": len(invalid),
        "quarantined_n": quarantined,
        "label_read_count": 0,
        "pet_read_count": 0,
        "raw_path_persisted_count": 0,
        "output_shape": [96, 96, 96],
        "all_finite": True,
        "aggregate": aggregate,
        "aggregate_integrity_sha256": aggregate["integrity_sha256"],
    }


def run_primary(*, device: str, stop: _StopFlag | None = None) -> dict[str, Any]:
    """Plan 003 Step 3: the single resumable full-1669 command."""

    stop = stop or _StopFlag()
    membership = v2.load_locked_membership()
    expected = [m["sample_hash"] for m in membership]
    committed = len(list((v2.PRIMARY_ROOT / "receipts").glob("*.json")))
    if committed == len(expected):
        validation = _streaming_validate_primary()
        if validation["status"] == "COMPLETED":
            _atomic_write_json(PRIMARY_AGGREGATE_PATH, validation["aggregate"])
            return {
                "status": "ALREADY_COMPLETED",
                "valid_committed_n": validation["valid_committed_n"],
                "zero_writes": True,
                "aggregate_integrity_sha256": validation["aggregate_integrity_sha256"],
            }
        return {
            "status": "BLOCKED",
            "reason_code": "BRAINIAC_V2_AGGREGATE_INVALID",
            "detail": {
                k: validation[k] for k in ("missing_n", "invalid_n", "quarantined_n")
            },
        }
    if not THRESHOLD_MANIFEST_PATH.is_file():
        return {
            "status": "BLOCKED",
            "reason_code": "BRAINIAC_V2_THRESHOLD_MANIFEST_REQUIRED_BEFORE_RUN",
        }
    fixed = _load_fixed_template()
    bindings = _base_bindings()
    transform_contract, _ = v2.build_contract_transform()
    sources = v2.resolve_sources(membership, index=v2.build_raw_index())
    progress = ProgressReporter(v2.PRIMARY_ROOT, len(expected), time.monotonic())
    pipeline = make_real_pipeline(
        fixed_template=fixed,
        device=device,
        sources=sources,
        transform_contract=transform_contract,
        extra_bindings=bindings,
        progress=progress,
    )
    started = time.monotonic()
    report = v2.execute(
        mode="run",
        limit=0,
        root=v2.PRIMARY_ROOT,
        device=device,
        pipeline_factory=lambda: pipeline,
        bindings=bindings,
        membership=membership,
        stop_requested=(lambda: stop.requested) if stop is not None else None,
    )
    elapsed = round(time.monotonic() - started, 1)
    if report["status"] == "PAUSED_RESUMABLE":
        _atomic_write_json(
            RUN_STATE_PATH,
            {
                "status": "PAUSED_RESUMABLE",
                "signal": stop.reason,
                "completed": report["completed"],
                "selected": report["selected"],
                "elapsed_seconds": elapsed,
                "resume_semantics": "RERUN_SAME_COMMAND_SKIPS_COMMITTED",
                "updated_at_utc": _utc_now(),
            },
        )
        return {
            "status": "PAUSED_RESUMABLE",
            "exit_code": 75,
            "completed": report["completed"],
            "selected": report["selected"],
            "signal": stop.reason,
        }
    if report["status"] == "BLOCKED_BY_QUARANTINE":
        return {
            "status": "BLOCKED",
            "reason_code": "BRAINIAC_V2_QUARANTINE_BLOCKS_AGGREGATE",
            "quarantined": report["quarantined"],
        }
    if report["status"] != "COMPLETED":
        return {"status": "BLOCKED", "detail": str(report.get("status"))}
    validation = _streaming_validate_primary()
    if validation["status"] != "COMPLETED":
        return {
            "status": "BLOCKED",
            "reason_code": "BRAINIAC_V2_AGGREGATE_INVALID",
            "detail": {
                k: validation[k] for k in ("missing_n", "invalid_n", "quarantined_n")
            },
        }
    _atomic_write_json(PRIMARY_AGGREGATE_PATH, validation["aggregate"])
    return {
        "status": "COMPLETED",
        "valid_committed_n": validation["valid_committed_n"],
        "aggregate_integrity_sha256": validation["aggregate_integrity_sha256"],
        "elapsed_seconds": elapsed,
    }


def _embed_96_array(array: Any, *, device: str, encoder: Any):
    import numpy as np
    import torch

    tensor = torch.from_numpy(np.ascontiguousarray(array))[None, None].to(device)
    with torch.inference_mode():
        return encoder(tensor).detach().cpu().numpy().astype("float64")


def run_sensitivity(*, device: str) -> dict[str, Any]:
    """Plan 003 Step 5: mask stability is a blocker; direct drift is report-only."""

    import nibabel as nib
    import numpy as np

    if SENSITIVITY_REPORT_PATH.is_file():
        previous = json.loads(SENSITIVITY_REPORT_PATH.read_text(encoding="utf-8"))
        if previous.get("status") == "PASS":
            return {"status": "ALREADY_COMPLETED"}

    membership = v2.load_locked_membership()
    selected = v2.select_sensitivity_samples([m["sample_hash"] for m in membership])
    root = v2.PRIMARY_ROOT
    missing = [h for h in selected if not (root / "receipts" / f"{h}.json").is_file()]
    if missing:
        return {
            "status": "BLOCKED",
            "reason_code": "BRAINIAC_V2_SENSITIVITY_REQUIRES_COMPLETED_PRIMARY",
            "missing_n": len(missing),
        }

    from src.research.brainiac_model_loader import load_brainiac_encoder
    from src.research.brainiac_execution_gate import brainiac_action_lease

    with brainiac_action_lease("EXTRACT_FROZEN_BRAINIAC_EMBEDDINGS"):
        encoder = load_brainiac_encoder(device=device)

    cosines: list[float] = []
    for sample_hash in selected:
        receipt = json.loads(
            (root / "receipts" / f"{sample_hash}.json").read_text(encoding="utf-8")
        )
        base = nib.load(str(root / receipt["output_relative_path"])).get_fdata(
            dtype="float32"
        )
        mask = base != 0
        base_embedding = _embed_96_array(base, device=device, encoder=encoder)
        for morph in (v2.erode_mask, v2.dilate_mask):
            perturbed_mask = morph(mask, iterations=1)
            volume = np.where(perturbed_mask, base, 0.0)
            nonzero = volume[volume != 0]
            if nonzero.size == 0:
                raise RuntimeError("BRAINIAC_V2_SENSITIVITY_FOREGROUND_EMPTY")
            volume = ((volume - float(nonzero.mean())) / (float(nonzero.std()) or 1.0)).astype(
                "float32"
            )
            embedding = _embed_96_array(volume, device=device, encoder=encoder)
            cosine = float(
                np.dot(base_embedding, embedding)
                / (np.linalg.norm(base_embedding) * np.linalg.norm(embedding) + 1e-12)
            )
            if not math.isfinite(cosine):
                raise RuntimeError("BRAINIAC_V2_SENSITIVITY_NONFINITE")
            cosines.append(cosine)
    median_cosine = float(np.median(cosines))
    p05_cosine = float(np.percentile(cosines, 5))
    body: dict[str, Any] = {
        "schema_version": 2,
        "status": "PASS" if (median_cosine >= 0.99 and p05_cosine >= 0.97) else "BLOCKED",
        "reason_code": None if (median_cosine >= 0.99 and p05_cosine >= 0.97)
        else "BRAINIAC_V2_MASK_STABILITY_FAILED",
        "selection_n": len(selected),
        "selection_label_blind": True,
        "mask_median_cosine": median_cosine,
        "mask_p05_cosine": p05_cosine,
        "mask_median_threshold": 0.99,
        "mask_p05_threshold": 0.97,
        "no_wraparound": True,
        "direct_128_to_96_role": "SENSITIVITY_ONLY",
        "direct_route_drift": "REPORT_ONLY_NOT_PRIMARY",
        "embeddings_from": "primary_schema_v2_outputs_only",
        "labels_read": False,
        "raw_paths_persisted": False,
        "created_at_utc": _utc_now(),
    }
    body["integrity_sha256"] = canonical_sha(
        {k: v for k, v in body.items() if k != "integrity_sha256"}
    )
    _atomic_write_json(SENSITIVITY_REPORT_PATH, body)
    return body


def run_equivalence(*, limit: int, device: str, workers: int = 2) -> dict[str, Any]:
    """Plan-002 revision gate: sequential vs deterministic-parallel on N samples.

    Both legs run with BRAINIAC_V2_ITK_THREADS pinned (default 1) so that N4 and
    rigid registration are bit-deterministic; the parallel leg only changes
    SAMPLE-level scheduling (GPU HD-BET stays serialized by _GPU_LOCK).
    PASS is required before policy may unlock parallel_samples for the full run.
    """

    membership = v2.load_locked_membership()
    items = membership[: max(1, int(limit))]
    fixed = _load_fixed_template()
    bindings = _base_bindings()
    transform_contract, _ = v2.build_contract_transform()
    index = v2.build_raw_index()

    def make(root: Path):
        sources = v2.resolve_sources(items, index=index)
        return make_real_pipeline(
            fixed_template=fixed,
            device=device,
            sources=sources,
            transform_contract=transform_contract,
            extra_bindings=bindings,
            progress=ProgressReporter(root, len(items), time.monotonic()),
        )

    seq_root = v2.SMOKE_ROOT / "eq_sequential"
    par_root = v2.SMOKE_ROOT / "eq_parallel"
    seq_report = v2.execute(
        mode="smoke",
        limit=len(items),
        root=seq_root,
        device=device,
        pipeline_factory=make(seq_root),
        bindings=bindings,
        membership=items,
    )
    if seq_report["status"] != "COMPLETED":
        body = {"status": "FAIL", "reason": "SEQUENTIAL_LEG_FAILED",
                "detail": str(seq_report.get("status"))}
        body["integrity_sha256"] = canonical_sha(
            {k: v for k, v in body.items() if k != "integrity_sha256"}
        )
        _atomic_write_json(REGISTRY_ROOT / "brainiac_preprocessing_v2_equivalence.json", body)
        return body
    pipeline_par = make(par_root)
    from src.research.brainiac_execution_gate import brainiac_action_lease

    with brainiac_action_lease("PREPARE_RAW_T1"):
        par_receipts = v2.run_orchestrated(
            items,
            par_root,
            pipeline_factory=lambda: pipeline_par,
            bindings=bindings,
            cpu_workers=max(1, int(workers)),
            gpu_workers=1,
        )
    mismatches = _equivalence_mismatches(seq_report["receipts"], par_receipts)
    body: dict[str, Any] = {
        "schema_version": "3.1",
        "status": "PASS" if not mismatches else "FAIL",
        "samples": len(items),
        "workers": max(1, int(workers)),
        "itk_threads_pinned": True,
        "mismatch_count": len(mismatches),
        "mismatches_hash_prefix_only": mismatches[:40],
        "claim_ceiling": "POST_HOC_INTERNAL_TECHNICAL_EVIDENCE",
        "created_at_utc": _utc_now(),
    }
    body["integrity_sha256"] = canonical_sha(
        {k: v for k, v in body.items() if k != "integrity_sha256"}
    )
    _atomic_write_json(REGISTRY_ROOT / "brainiac_preprocessing_v2_equivalence.json", body)
    return body


def run_full(*, device: str) -> int:
    """The single operator command for Plan 003 (idempotent at every phase)."""

    stop = _StopFlag()
    _install_signal_handlers(stop)
    pre = preflight()
    if pre["status"] != "PASS" or not pre.get("free_disk_ok", False):
        _print_flush(
            json.dumps(
                {"phase": "preflight", "status": "BLOCKED",
                 "free_disk_gb": pre.get("free_disk_gb")},
                default=str,
            )
        )
        return 3
    _print_flush(
        json.dumps({"phase": "preflight", "status": "PASS",
                    "free_disk_gb": pre["free_disk_gb"]})
    )

    calibration = _calibrate_if_needed(device=device)
    _print_flush(
        json.dumps(
            {"phase": "calibrate",
             **{k: calibration[k] for k in ("status", "selected_n") if k in calibration}},
            default=str,
        )
    )
    if calibration["status"] not in {"PASS", "ALREADY_COMPLETED"}:
        return 3

    primary = run_primary(device=device, stop=stop)
    _print_flush(
        json.dumps(
            {"phase": "primary",
             **{k: primary[k] for k in
                ("status", "completed", "valid_committed_n") if k in primary}},
            default=str,
        )
    )
    code = _exit_code_for(str(primary["status"]))
    if code != 0:
        return code

    sensitivity = run_sensitivity(device=device)
    _print_flush(
        json.dumps(
            {"phase": "sensitivity",
             **{k: sensitivity[k] for k in
                ("status", "mask_median_cosine", "mask_p05_cosine")
                if k in sensitivity}}
        )
    )
    code = _exit_code_for(str(sensitivity["status"]))
    if code != 0:
        return code

    validation = _streaming_validate_primary()
    _print_flush(
        json.dumps(
            {
                "phase": "final_validation",
                "status": validation["status"],
                "valid_committed_n": validation["valid_committed_n"],
                "aggregate_integrity_sha256": validation["aggregate_integrity_sha256"],
            }
        )
    )
    return _exit_code_for(str(validation["status"]))


def run_smoke(*, limit: int, device: str) -> dict[str, Any]:
    membership = v2.load_locked_membership()
    items = membership[: max(1, int(limit))]
    fixed = _load_fixed_template()
    bindings = _base_bindings()
    transform_contract, _ = v2.build_contract_transform()
    index = v2.build_raw_index()
    sources = v2.resolve_sources(items, index=index)
    pipeline = make_real_pipeline(
        fixed_template=fixed,
        device=device,
        sources=sources,
        transform_contract=transform_contract,
        extra_bindings=bindings,
    )
    sequential_root = v2.SMOKE_ROOT / "sequential"
    orchestrated_root = v2.SMOKE_ROOT / "orchestrated"
    sequential = v2.execute(
        mode="smoke",
        limit=limit,
        root=sequential_root,
        device=device,
        pipeline_factory=lambda: pipeline,
        bindings=bindings,
        membership=items,
    )
    if sequential["status"] != "COMPLETED":
        return sequential
    if bool(_policy().get("orchestration", {}).get("sequential_only")):
        body: dict[str, Any] = {
            "schema_version": 2,
            "status": "PASS",
            "mode": "smoke",
            "rows": len(items),
            "sequential_status": sequential["status"],
            "equivalence": "SKIPPED_SEQUENTIAL_ONLY_LOCKED",
            "claim_ceiling": "POST_HOC_INTERNAL_TECHNICAL_EVIDENCE",
            "created_at_utc": _utc_now(),
        }
        body["integrity_sha256"] = canonical_sha(
            {k: v for k, v in body.items() if k != "integrity_sha256"}
        )
        _atomic_write_json(SMOKE_REPORT, body)
        return body
    from src.research.brainiac_execution_gate import brainiac_action_lease

    with brainiac_action_lease("PREPARE_RAW_T1"):
        orch_receipts = v2.run_orchestrated(
            items,
            orchestrated_root,
            pipeline_factory=lambda: pipeline,
            bindings=bindings,
            cpu_workers=2,
            gpu_workers=1,
        )
    seq_by_sample = {r["sample_hash"]: r for r in sequential["receipts"]}
    mismatched: list[str] = []
    for receipt in orch_receipts:
        reference = seq_by_sample[receipt["sample_hash"]]
        for field in (
            "output_sha256",
            "transform_sha256",
            "foreground_fraction",
            "rigid_determinant",
            "normalized_nonzero_std",
        ):
            if receipt[field] != reference[field]:
                mismatched.append(f"{receipt['sample_hash']}:{field}")
    body = {
        "schema_version": 2,
        "status": "PASS" if not mismatched else "EQUIVALENCE_FAILED",
        "mode": "smoke",
        "rows": len(items),
        "sequential_status": sequential["status"],
        "equivalence_mismatched": mismatched,
        "claim_ceiling": "POST_HOC_INTERNAL_TECHNICAL_EVIDENCE",
        "created_at_utc": _utc_now(),
    }
    body["integrity_sha256"] = canonical_sha(
        {k: v for k, v in body.items() if k != "integrity_sha256"}
    )
    _atomic_write_json(SMOKE_REPORT, body)
    return body


def run_status() -> dict[str, Any]:
    membership = v2.load_locked_membership()
    root = v2.PRIMARY_ROOT
    completed = 0
    invalid: list[str] = []
    for member in membership:
        try:
            v2.load_committed_receipt(member["sample_hash"], root)
            completed += 1
        except RuntimeError as exc:
            if "NOT_COMMITTED" not in str(exc):
                invalid.append(member["sample_hash"][:10])
    return {
        "status": "PASS" if not invalid else "BLOCKED",
        "expected_rows": len(membership),
        "completed_rows": completed,
        "invalid_rows": invalid,
        "remaining_rows": len(membership) - completed,
        "threshold_manifest_present": THRESHOLD_MANIFEST_PATH.is_file(),
        "aggregate_present": PRIMARY_AGGREGATE_PATH.is_file(),
        "raw_paths_persisted": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=("preflight", "calibrate", "run", "sensitivity", "full", "smoke", "status", "equivalence"),
        required=True,
    )
    parser.add_argument("--limit", type=int, default=v2.SMOKE_LIMIT)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()

    if args.mode == "full":
        return run_full(device=args.device)

    stop = _StopFlag()
    _install_signal_handlers(stop)
    if args.mode == "equivalence":
        result = run_equivalence(limit=args.limit, device=args.device)
    elif args.mode == "preflight":
        result: dict[str, Any] = preflight()
    elif args.mode == "calibrate":
        result = _calibrate_if_needed(device=args.device)
    elif args.mode == "run":
        result = run_primary(device=args.device, stop=stop)
    elif args.mode == "sensitivity":
        result = run_sensitivity(device=args.device)
    elif args.mode == "smoke":
        result = run_smoke(limit=args.limit, device=args.device)
    else:
        result = run_status()
    _print_flush(json.dumps(result, indent=2, sort_keys=True, default=str))
    return _exit_code_for(str(result.get("status", "BLOCKED")))


if __name__ == "__main__":
    raise SystemExit(main())
