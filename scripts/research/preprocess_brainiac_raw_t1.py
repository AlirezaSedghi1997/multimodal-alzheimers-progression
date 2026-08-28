"""Resumable, label-blind raw-T1 preprocessing for BrainIAC.

This command never reads outcome labels or PET data.  It reconstructs raw T1
locations in memory from existing filename hashes, performs N4 correction,
deterministic rigid registration to a hash-locked local MNI reference, the
pinned BrainIAC vendor HD-BET fast-fold0 extraction, and the exact public
validation transform (trilinear resize to 96 cubed followed by nonzero,
channel-wise normalization).

The local SPM avg152 template is protocol-compatible, but the upstream
BrainIAC training template was not distributed.  Every output therefore keeps
the claim ceiling at post-hoc internal technical evidence.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
from typing import Any, Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

RAW_DATA_ROOT = Path(r"H:\A4 Learn")
CACHE_PATH = ROOT / "data" / "cached" / "t1_volume_cache_128.h5"
CATALOG_PATH = ROOT / "data" / "processed" / "master_catalog.csv"
VENDOR_COMMIT = "ba60f45bed832ee0e5683c678caeaeefe2072f0d"
VENDOR_ROOT = ROOT / "vendor" / "BrainIAC" / VENDOR_COMMIT
VENDOR_PREPROCESSING = VENDOR_ROOT / "src" / "preprocessing"
HD_BET_WEIGHT = VENDOR_PREPROCESSING / "hd-bet_params" / "0.model"
TEMPLATE_PATH = (
    ROOT
    / "models"
    / "pretrained"
    / "brainiac"
    / "templates"
    / "spm25_avg152T1_2mm.nii"
)
TEMPLATE_LOCK = TEMPLATE_PATH.with_name("spm25_avg152T1_2mm.lock.json")
ARTIFACT_ROOT = ROOT / "data" / "derived" / "method_x_brainiac"
OUTPUT_ROOT = ARTIFACT_ROOT / "preprocessed_96"
TRANSFORM_ROOT = ARTIFACT_ROOT / "rigid_transforms"
RECEIPT_ROOT = ARTIFACT_ROOT / "preprocessing_receipts"
SCRATCH_ROOT = ARTIFACT_ROOT / "_scratch_preprocessing"
ORPHAN_ROOT = ARTIFACT_ROOT / "_orphaned_preprocessing"
REGISTRY_ROOT = ROOT / "results" / "method_x_registry" / "brainiac_r1"
RUNTIME_LOCK = REGISTRY_ROOT / "brainiac_official_runtime_lock.json"
STRICT_RECEIPT = REGISTRY_ROOT / "brainiac_official_strict_load_receipt.json"
FORWARD_RECEIPT = REGISTRY_ROOT / "brainiac_official_forward_receipt.json"
PREFLIGHT_REPORT = REGISTRY_ROOT / "brainiac_preprocessing_preflight.json"
SMOKE_REPORT = REGISTRY_ROOT / "brainiac_preprocessing_smoke.json"
STATE_PATH = REGISTRY_ROOT / "brainiac_preprocessing_state.json"
AGGREGATE_MANIFEST = ARTIFACT_ROOT / "brainiac_preprocessing_manifest.json"
PROCESS_LOCK = REGISTRY_ROOT / ".brainiac_preprocessing.lock"

EXPECTED_REAL_ROWS = 1669
EXPECTED_TEMPLATE_SHA256 = (
    "213557a43d4387243c51bf208b3a672ae2eb68091f7ffe47c28a9f69d9e3d9fd"
)
EXPECTED_HD_BET_WEIGHT_SHA256 = (
    "6f75233753c4750672815e2b7a86db754995ae44b8f1cd77bccfc37becd2d83c"
)
REGISTRATION_SEED = 20260821
LEGACY_SCHEMA1_PREPROCESSOR_BLOCKER = (
    "BRAINIAC_LEGACY_SCHEMA1_PREPROCESSOR_PERMANENTLY_DISABLED"
)


def _block_legacy_schema1_preprocessing() -> None:
    """Fail before I/O; B2 must provide the receipt-bound schema-v2 entrypoint."""

    raise RuntimeError(LEGACY_SCHEMA1_PREPROCESSOR_BLOCKER)


@dataclass(frozen=True)
class WorkItem:
    sample_hash: str
    t1_file_hash: str
    substudy: str
    source_path: Path


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_sha(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _sha256_file(path: Path) -> str:
    from src.research.brainiac_preprocessing import sha256_file

    return sha256_file(path)


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    encoded = (
        json.dumps(dict(payload), indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")
    with temporary.open("wb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _safe_relative(path: Path, root: Path = ROOT) -> str:
    resolved = path.resolve(strict=True)
    try:
        relative = resolved.relative_to(root.resolve(strict=True))
    except ValueError as exc:
        raise RuntimeError("BRAINIAC_PROJECT_ARTIFACT_OUTSIDE_H_ROOT") from exc
    return relative.as_posix()


def _load_integrity_json(path: Path) -> tuple[dict[str, Any], str]:
    if not path.is_file():
        raise RuntimeError(f"BRAINIAC_REQUIRED_RECEIPT_MISSING:{path.name}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"BRAINIAC_REQUIRED_RECEIPT_INVALID:{path.name}")
    body = {key: value for key, value in payload.items() if key != "integrity_sha256"}
    if payload.get("integrity_sha256") != _canonical_sha(body):
        raise RuntimeError(f"BRAINIAC_REQUIRED_RECEIPT_INTEGRITY_INVALID:{path.name}")
    return payload, _sha256_file(path)


def _decode(value: object) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def _load_real_sample_hashes() -> list[str]:
    import h5py

    if not CACHE_PATH.is_file():
        raise RuntimeError("BRAINIAC_LOCKED_CACHE_MISSING")
    with h5py.File(CACHE_PATH, "r") as handle:
        required = {"sample_hash", "source", "status"}
        if not required.issubset(handle.keys()):
            raise RuntimeError("BRAINIAC_LOCKED_CACHE_SCHEMA_INVALID")
        rows = [
            _decode(sample_hash)
            for sample_hash, source, status in zip(
                handle["sample_hash"][:],
                handle["source"][:],
                handle["status"][:],
                strict=True,
            )
            if _decode(source) == "real" and _decode(status).upper() == "OK"
        ]
        if int(handle.attrs.get("n_real_eligible", -1)) != EXPECTED_REAL_ROWS:
            raise RuntimeError("BRAINIAC_LOCKED_CACHE_REAL_COUNT_ATTRIBUTE_INVALID")
    if len(rows) != EXPECTED_REAL_ROWS or len(rows) != len(set(rows)):
        raise RuntimeError("BRAINIAC_LOCKED_CACHE_REAL_MEMBERSHIP_INVALID")
    return rows


def _build_raw_index() -> dict[str, Path]:
    from scripts.build_mri_volume_cache import iter_t1_files
    from src.data.a4_schema import T1_NAME_PATTERN
    from src.utils.privacy import stable_hash
    import re

    if not RAW_DATA_ROOT.is_dir():
        raise RuntimeError("BRAINIAC_RAW_DATA_ROOT_UNAVAILABLE")
    pattern = re.compile(T1_NAME_PATTERN, re.IGNORECASE)
    index: dict[str, Path] = {}
    for candidate in iter_t1_files(RAW_DATA_ROOT):
        if not pattern.match(candidate.name):
            continue
        file_hash = stable_hash(candidate.name, prefix="t1")
        if file_hash in index and index[file_hash].resolve() != candidate.resolve():
            raise RuntimeError(f"BRAINIAC_RAW_T1_HASH_COLLISION:{file_hash}")
        index[file_hash] = candidate
    if not index:
        raise RuntimeError("BRAINIAC_RAW_T1_INDEX_EMPTY")
    return index


def build_work_items() -> list[WorkItem]:
    samples = _load_real_sample_hashes()
    with CATALOG_PATH.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    by_sample = {str(row.get("sample_hash") or ""): row for row in rows}
    if len(by_sample) != len(rows):
        raise RuntimeError("BRAINIAC_CATALOG_SAMPLE_HASH_DUPLICATE")
    raw_index = _build_raw_index()
    items: list[WorkItem] = []
    for sample_hash in samples:
        row = by_sample.get(sample_hash)
        if row is None:
            raise RuntimeError(f"BRAINIAC_CATALOG_SAMPLE_MISSING:{sample_hash}")
        t1_file_hash = str(row.get("t1_file_hash") or "")
        source_path = raw_index.get(t1_file_hash)
        if source_path is None:
            raise RuntimeError(f"BRAINIAC_RAW_T1_NOT_FOUND:{sample_hash}")
        items.append(
            WorkItem(
                sample_hash=sample_hash,
                t1_file_hash=t1_file_hash,
                substudy=str(row.get("substudy") or "UNKNOWN"),
                source_path=source_path,
            )
        )
    return items


def _validate_template() -> dict[str, Any]:
    if not TEMPLATE_PATH.is_file() or not TEMPLATE_LOCK.is_file():
        raise RuntimeError("BRAINIAC_TEMPLATE_ASSET_MISSING")
    if _sha256_file(TEMPLATE_PATH) != EXPECTED_TEMPLATE_SHA256:
        raise RuntimeError("BRAINIAC_TEMPLATE_HASH_MISMATCH")
    lock = json.loads(TEMPLATE_LOCK.read_text(encoding="utf-8"))
    if (
        lock.get("sha256") != EXPECTED_TEMPLATE_SHA256
        or lock.get("upstream_brainiac_template_identity")
        != "UNKNOWN_NOT_DISTRIBUTED_IN_PINNED_REPOSITORY"
        or lock.get("status") != "LOCKED_WITH_UPSTREAM_IDENTITY_LIMITATION"
    ):
        raise RuntimeError("BRAINIAC_TEMPLATE_CLAIM_BOUNDARY_INVALID")
    return lock


def _validate_runtime_receipts() -> dict[str, Any]:
    runtime, runtime_file_sha = _load_integrity_json(RUNTIME_LOCK)
    strict, strict_file_sha = _load_integrity_json(STRICT_RECEIPT)
    forward, forward_file_sha = _load_integrity_json(FORWARD_RECEIPT)
    if (
        runtime.get("status") != "PASS"
        or strict.get("status") != "PASS"
        or strict.get("strict") is not True
        or int(strict.get("matched_keys", 0)) <= 0
        or forward.get("status") != "PASS"
        or forward.get("outer_test_touched") is not False
        or tuple(forward.get("input_shape", ())) != (1, 1, 96, 96, 96)
        or tuple(forward.get("output_shape", ())) != (1, 768)
    ):
        raise RuntimeError("BRAINIAC_OFFICIAL_RUNTIME_CHAIN_INVALID")
    if strict.get("environment_lock_sha256") != runtime_file_sha:
        raise RuntimeError("BRAINIAC_STRICT_RUNTIME_BINDING_INVALID")
    if forward.get("strict_load_receipt_file_sha256") != strict_file_sha:
        raise RuntimeError("BRAINIAC_FORWARD_STRICT_BINDING_INVALID")
    return {
        "runtime_lock_sha256": runtime_file_sha,
        "strict_receipt_sha256": strict_file_sha,
        "forward_receipt_sha256": forward_file_sha,
        "runtime_integrity_sha256": runtime["integrity_sha256"],
        "strict_integrity_sha256": strict["integrity_sha256"],
        "forward_integrity_sha256": forward["integrity_sha256"],
    }


def _assert_no_cnn_writer() -> None:
    from src.training.method_x_autosupervisor import discover_method_x_processes

    active = discover_method_x_processes(exclude_pids=(os.getpid(),))
    if active:
        raise RuntimeError("BRAINIAC_CNN_WRITER_ACTIVE")
    locks = list((ROOT / "results" / "final_results").glob("*/.battery.lock"))
    if locks:
        raise RuntimeError("BRAINIAC_BATTERY_LOCK_PRESENT")


def _acquire_lock() -> None:
    PROCESS_LOCK.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "pid": os.getpid(),
        "created_at_utc": _utc_now(),
        "role": "BRAINIAC_RAW_T1_PREPROCESSING",
    }
    encoded = (json.dumps(payload, sort_keys=True) + "\n").encode("utf-8")
    try:
        descriptor = os.open(
            PROCESS_LOCK, os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_BINARY", 0)
        )
    except FileExistsError as exc:
        raise RuntimeError("BRAINIAC_PREPROCESSING_LOCK_PRESENT") from exc
    try:
        os.write(descriptor, encoded)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _release_lock() -> None:
    if PROCESS_LOCK.is_file():
        PROCESS_LOCK.unlink()


def _affine_digest(image: Any) -> str:
    import numpy as np

    payload = {
        "size": list(image.GetSize()),
        "spacing": [float(value) for value in image.GetSpacing()],
        "origin": [float(value) for value in image.GetOrigin()],
        "direction": [float(value) for value in image.GetDirection()],
    }
    return _canonical_sha(payload)


def _resample_template_1mm(template: Any, sitk: Any) -> Any:
    old_size = template.GetSize()
    old_spacing = template.GetSpacing()
    new_spacing = (1.0, 1.0, 1.0)
    new_size = [
        int(round(old_size[index] * old_spacing[index] / new_spacing[index]))
        for index in range(3)
    ]
    return sitk.Resample(
        template,
        new_size,
        sitk.Transform(),
        sitk.sitkLinear,
        template.GetOrigin(),
        new_spacing,
        template.GetDirection(),
        0.0,
        sitk.sitkFloat32,
    )


def _registration(fixed: Any, moving: Any, sitk: Any) -> tuple[Any, float, str]:
    initial = sitk.CenteredTransformInitializer(
        fixed,
        moving,
        sitk.Euler3DTransform(),
        sitk.CenteredTransformInitializerFilter.GEOMETRY,
    )
    method = sitk.ImageRegistrationMethod()
    method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    method.SetMetricSamplingStrategy(method.RANDOM)
    method.SetMetricSamplingPercentage(0.01, seed=REGISTRATION_SEED)
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
    if not (metric == metric and abs(metric) < float("inf")):
        raise RuntimeError("BRAINIAC_RIGID_REGISTRATION_METRIC_NONFINITE")
    return transform, metric, str(method.GetOptimizerStopConditionDescription())


def _write_sitk_atomic(image: Any, path: Path, sitk: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + f".{os.getpid()}.tmp.nii.gz")
    sitk.WriteImage(image, str(temporary), True)
    with temporary.open("rb+") as handle:
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _write_transform_atomic(transform: Any, path: Path, sitk: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + f".{os.getpid()}.tmp.tfm")
    sitk.WriteTransform(transform, str(temporary))
    with temporary.open("rb+") as handle:
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _run_hd_bet(registered: Path, brain_output: Path, *, device: str) -> Path:
    if str(VENDOR_PREPROCESSING) not in sys.path:
        sys.path.insert(0, str(VENDOR_PREPROCESSING))
    if _sha256_file(HD_BET_WEIGHT) != EXPECTED_HD_BET_WEIGHT_SHA256:
        raise RuntimeError("BRAINIAC_HD_BET_WEIGHT_HASH_MISMATCH")
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
        raise RuntimeError("BRAINIAC_HD_BET_OUTPUT_MISSING")
    return mask


def _official_96_transform(brain_path: Path) -> tuple[Any, Any]:
    import numpy as np
    from monai.transforms import (
        Compose,
        EnsureChannelFirstd,
        LoadImaged,
        NormalizeIntensityd,
        Resized,
        ToTensord,
    )

    transform = Compose(
        [
            LoadImaged(keys=["image"]),
            EnsureChannelFirstd(keys=["image"]),
            Resized(keys=["image"], spatial_size=(96, 96, 96), mode="trilinear"),
            NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
            ToTensord(keys=["image"]),
        ]
    )
    tensor = transform({"image": str(brain_path)})["image"]
    array = tensor.detach().cpu().numpy().astype(np.float32, copy=False)
    affine = np.asarray(tensor.affine.detach().cpu(), dtype=np.float64)
    if tuple(array.shape) != (1, 96, 96, 96) or not np.isfinite(array).all():
        raise RuntimeError("BRAINIAC_OFFICIAL_96_TRANSFORM_INVALID")
    return array, affine


def _write_nifti_atomic(array: Any, affine: Any, path: Path) -> None:
    import nibabel as nib

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + f".{os.getpid()}.tmp.nii.gz")
    nib.save(nib.Nifti1Image(array[0], affine), str(temporary))
    with temporary.open("rb+") as handle:
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _quarantine_uncommitted(paths: Iterable[Path], sample_hash: str) -> None:
    existing = [path for path in paths if path.exists()]
    if not existing:
        return
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    target = ORPHAN_ROOT / f"{stamp}-{sample_hash}"
    target.mkdir(parents=True, exist_ok=False)
    for path in existing:
        try:
            path.resolve(strict=True).relative_to(ARTIFACT_ROOT.resolve(strict=True))
        except (OSError, ValueError) as exc:
            raise RuntimeError("BRAINIAC_UNCOMMITTED_ARTIFACT_OUTSIDE_ROOT") from exc
        os.replace(path, target / path.name)


def _cleanup_scratch(path: Path) -> None:
    if not path.exists():
        return
    resolved_root = SCRATCH_ROOT.resolve(strict=True)
    resolved = path.resolve(strict=True)
    try:
        resolved.relative_to(resolved_root)
    except ValueError as exc:
        raise RuntimeError("BRAINIAC_SCRATCH_CLEANUP_OUTSIDE_ROOT") from exc
    shutil.rmtree(resolved)


def _receipt_paths(item: WorkItem) -> tuple[Path, Path, Path]:
    return (
        OUTPUT_ROOT / f"{item.sample_hash}.nii.gz",
        TRANSFORM_ROOT / f"{item.sample_hash}.tfm",
        RECEIPT_ROOT / f"{item.sample_hash}.json",
    )


def _load_valid_completed(
    item: WorkItem, *, recover_uncommitted: bool = False
) -> dict[str, Any] | None:
    from src.research.brainiac_preprocessing import validate_completed_item

    output, transform, receipt_path = _receipt_paths(item)
    if receipt_path.is_file():
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        validate_completed_item(receipt, artifact_root=ARTIFACT_ROOT)
        if receipt.get("t1_file_hash") != item.t1_file_hash:
            raise RuntimeError("BRAINIAC_PREPROCESS_SOURCE_BINDING_MISMATCH")
        return receipt
    if recover_uncommitted:
        _quarantine_uncommitted((output, transform), item.sample_hash)
    return None


def preprocess_one(
    item: WorkItem,
    *,
    fixed_template: Any,
    template_sha256: str,
    template_1mm_sha256: str,
    runtime_bindings: Mapping[str, str],
    device: str,
) -> dict[str, Any]:
    completed = _load_valid_completed(item, recover_uncommitted=True)
    if completed is not None:
        return completed

    import numpy as np
    import SimpleITK as sitk

    output_path, transform_path, receipt_path = _receipt_paths(item)
    scratch = SCRATCH_ROOT / item.sample_hash
    if scratch.exists():
        _quarantine_uncommitted((scratch,), item.sample_hash)
    scratch.mkdir(parents=True, exist_ok=False)
    n4_path = scratch / f"{item.sample_hash}_n4.nii.gz"
    registered_path = scratch / f"{item.sample_hash}_0000.nii.gz"
    brain_path = scratch / f"{item.sample_hash}_brain.nii.gz"
    started = time.monotonic()

    source = sitk.ReadImage(str(item.source_path), sitk.sitkFloat32)
    if source.GetDimension() != 3 or min(source.GetSize()) <= 0:
        raise RuntimeError("BRAINIAC_RAW_T1_GEOMETRY_INVALID")
    source_affine_sha = _affine_digest(source)
    source_file_sha = _sha256_file(item.source_path)

    n4_started = time.monotonic()
    n4 = sitk.N4BiasFieldCorrection(source)
    _write_sitk_atomic(n4, n4_path, sitk)
    n4_seconds = time.monotonic() - n4_started

    registration_started = time.monotonic()
    rigid_transform, metric, stop_condition = _registration(fixed_template, n4, sitk)
    registered = sitk.Resample(
        n4,
        fixed_template,
        rigid_transform,
        sitk.sitkLinear,
        0.0,
        sitk.sitkFloat32,
    )
    _write_sitk_atomic(registered, registered_path, sitk)
    registration_seconds = time.monotonic() - registration_started

    hd_bet_started = time.monotonic()
    mask_path = _run_hd_bet(registered_path, brain_path, device=device)
    hd_bet_seconds = time.monotonic() - hd_bet_started

    mask_image = sitk.ReadImage(str(mask_path), sitk.sitkUInt8)
    mask_array = sitk.GetArrayFromImage(mask_image)
    mask_fraction = float(np.count_nonzero(mask_array) / mask_array.size)
    if not 0.01 <= mask_fraction <= 0.80:
        raise RuntimeError(f"BRAINIAC_HD_BET_MASK_QC_INVALID:{mask_fraction}")

    official_started = time.monotonic()
    output_array, output_affine = _official_96_transform(brain_path)
    foreground = output_array[output_array != 0]
    if foreground.size == 0:
        raise RuntimeError("BRAINIAC_OFFICIAL_96_FOREGROUND_EMPTY")
    foreground_fraction = float(foreground.size / output_array.size)
    foreground_mean = float(foreground.mean())
    foreground_std = float(foreground.std())
    if (
        not np.isfinite(output_array).all()
        or not 0.01 <= foreground_fraction <= 0.95
        or abs(foreground_mean) > 1e-3
        or not 0.95 <= foreground_std <= 1.05
    ):
        raise RuntimeError("BRAINIAC_OFFICIAL_96_NUMERIC_QC_INVALID")
    _write_nifti_atomic(output_array, output_affine, output_path)
    _write_transform_atomic(rigid_transform, transform_path, sitk)
    official_seconds = time.monotonic() - official_started

    body: dict[str, Any] = {
        "schema_version": 1,
        "status": "COMPLETED",
        "sample_hash": item.sample_hash,
        "t1_file_hash": item.t1_file_hash,
        "source_role": "raw_t1",
        "source_file_sha256": source_file_sha,
        "source_affine_sha256": source_affine_sha,
        "source_geometry": {
            "size": list(source.GetSize()),
            "spacing_mm": [float(value) for value in source.GetSpacing()],
            "direction_sha256": _canonical_sha(list(source.GetDirection())),
        },
        "label_read": False,
        "pet_read": False,
        "direct_128_to_96_used": False,
        "n4_artifact_sha256": _sha256_file(n4_path),
        "registered_artifact_sha256": _sha256_file(registered_path),
        "registered_affine_sha256": _affine_digest(registered),
        "brain_mask_sha256": _sha256_file(mask_path),
        "brain_mask_fraction": mask_fraction,
        "rigid_transform_path": _safe_relative(transform_path, ARTIFACT_ROOT),
        "rigid_transform_sha256": _sha256_file(transform_path),
        "rigid_transform_bytes": transform_path.stat().st_size,
        "registration_metric": metric,
        "registration_stop_condition": stop_condition,
        "registration_seed": REGISTRATION_SEED,
        "template_sha256": template_sha256,
        "template_1mm_sha256": template_1mm_sha256,
        "upstream_template_identity": "UNKNOWN_NOT_DISTRIBUTED",
        "output_path": _safe_relative(output_path, ARTIFACT_ROOT),
        "output_sha256": _sha256_file(output_path),
        "output_bytes": output_path.stat().st_size,
        "output_shape": [96, 96, 96],
        "output_affine_sha256": _canonical_sha(output_affine.tolist()),
        "finite": True,
        "foreground_fraction": foreground_fraction,
        "foreground_mean": foreground_mean,
        "foreground_std": foreground_std,
        "official_transform": {
            "vendor_file": "src/dataset.py",
            "vendor_file_sha256": _sha256_file(VENDOR_ROOT / "src" / "dataset.py"),
            "steps": [
                "LoadImaged",
                "EnsureChannelFirstd",
                "Resized_96_trilinear",
                "NormalizeIntensityd_nonzero_channel_wise",
                "ToTensord",
            ],
        },
        "hd_bet": {
            "mode": "fast",
            "fold": 0,
            "tta": False,
            "postprocess_largest_component": True,
            "weight_sha256": EXPECTED_HD_BET_WEIGHT_SHA256,
        },
        **dict(runtime_bindings),
        "durations_seconds": {
            "n4": round(n4_seconds, 3),
            "registration": round(registration_seconds, 3),
            "hd_bet": round(hd_bet_seconds, 3),
            "official_96": round(official_seconds, 3),
            "total": round(time.monotonic() - started, 3),
        },
        "completed_at_utc": _utc_now(),
        "claim_ceiling": "POST_HOC_INTERNAL_TECHNICAL_EVIDENCE",
    }
    body["integrity_sha256"] = _canonical_sha(body)
    _atomic_write_json(receipt_path, body)

    from src.research.brainiac_preprocessing import validate_completed_item

    validate_completed_item(body, artifact_root=ARTIFACT_ROOT)
    _cleanup_scratch(scratch)
    return body


def _prepare_fixed_template() -> tuple[Any, str]:
    import SimpleITK as sitk

    derived_path = ARTIFACT_ROOT / "templates" / "spm25_avg152T1_1mm.nii.gz"
    receipt_path = derived_path.with_suffix(".lock.json")
    if derived_path.is_file() and receipt_path.is_file():
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if (
            receipt.get("source_template_sha256") != EXPECTED_TEMPLATE_SHA256
            or receipt.get("derived_sha256") != _sha256_file(derived_path)
            or receipt.get("output_spacing_mm") != [1.0, 1.0, 1.0]
        ):
            raise RuntimeError("BRAINIAC_DERIVED_TEMPLATE_RECEIPT_INVALID")
        return sitk.ReadImage(str(derived_path), sitk.sitkFloat32), str(
            receipt["derived_sha256"]
        )
    if derived_path.exists() or receipt_path.exists():
        raise RuntimeError("BRAINIAC_DERIVED_TEMPLATE_PARTIAL_STATE")
    template = sitk.ReadImage(str(TEMPLATE_PATH), sitk.sitkFloat32)
    derived = _resample_template_1mm(template, sitk)
    _write_sitk_atomic(derived, derived_path, sitk)
    body = {
        "schema_version": 1,
        "status": "LOCKED",
        "source_template_sha256": EXPECTED_TEMPLATE_SHA256,
        "derived_path": _safe_relative(derived_path),
        "derived_sha256": _sha256_file(derived_path),
        "derived_bytes": derived_path.stat().st_size,
        "output_size": list(derived.GetSize()),
        "output_spacing_mm": [float(value) for value in derived.GetSpacing()],
        "output_affine_sha256": _affine_digest(derived),
        "upstream_template_identity": "UNKNOWN_NOT_DISTRIBUTED",
    }
    body["integrity_sha256"] = _canonical_sha(body)
    _atomic_write_json(receipt_path, body)
    return derived, str(body["derived_sha256"])


def preflight() -> tuple[list[WorkItem], Any, str, dict[str, Any]]:
    _block_legacy_schema1_preprocessing()
    _assert_no_cnn_writer()
    template_lock = _validate_template()
    runtime = _validate_runtime_receipts()
    if not HD_BET_WEIGHT.is_file() or _sha256_file(HD_BET_WEIGHT) != EXPECTED_HD_BET_WEIGHT_SHA256:
        raise RuntimeError("BRAINIAC_HD_BET_WEIGHT_INVALID")
    items = build_work_items()
    fixed, fixed_sha = _prepare_fixed_template()
    body = {
        "schema_version": 1,
        "status": "PASS",
        "created_at_utc": _utc_now(),
        "real_rows": len(items),
        "membership_sha256": _canonical_sha([item.sample_hash for item in items]),
        "raw_paths_persisted": False,
        "labels_read": False,
        "pet_read": False,
        "cnn_writer_active": False,
        "template_sha256": EXPECTED_TEMPLATE_SHA256,
        "template_1mm_sha256": fixed_sha,
        "template_claim_boundary": template_lock["claim_boundary"],
        "hd_bet_weight_sha256": EXPECTED_HD_BET_WEIGHT_SHA256,
        "runtime_bindings": runtime,
        "external_validation_status": "OPTIONAL_NOT_AVAILABLE_NOT_EXECUTION_BLOCKER",
        "claim_ceiling": "POST_HOC_INTERNAL_TECHNICAL_EVIDENCE",
    }
    body["integrity_sha256"] = _canonical_sha(body)
    _atomic_write_json(PREFLIGHT_REPORT, body)
    return items, fixed, fixed_sha, runtime


def _smoke_selection(items: Sequence[WorkItem], limit: int) -> list[WorkItem]:
    if limit <= 0:
        raise RuntimeError("BRAINIAC_SMOKE_LIMIT_INVALID")
    selected: list[WorkItem] = []
    seen_substudies: set[str] = set()
    for item in items:
        if item.substudy not in seen_substudies:
            selected.append(item)
            seen_substudies.add(item.substudy)
        if len(selected) == limit:
            return selected
    for item in items:
        if item not in selected:
            selected.append(item)
        if len(selected) == limit:
            break
    return selected


def _write_state(
    *, mode: str, selected: int, completed: int, current_sample_hash: str | None
) -> None:
    body = {
        "schema_version": 1,
        "status": "RUNNING" if completed < selected else "COMPLETED",
        "mode": mode,
        "selected_rows": selected,
        "completed_rows": completed,
        "current_sample_hash": current_sample_hash,
        "updated_at_utc": _utc_now(),
        "resume_semantics": "VALID_RECEIPT_SKIP_INVALID_RECEIPT_FAIL_CLOSED",
        "raw_paths_persisted": False,
    }
    body["integrity_sha256"] = _canonical_sha(body)
    _atomic_write_json(STATE_PATH, body)


def execute(*, mode: str, limit: int, device: str) -> dict[str, Any]:
    _block_legacy_schema1_preprocessing()
    if device == "cuda":
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("BRAINIAC_PREPROCESSING_CUDA_UNAVAILABLE")
    items, fixed, fixed_sha, runtime = preflight()
    selected = _smoke_selection(items, limit) if mode == "smoke" else items
    _acquire_lock()
    receipts: list[dict[str, Any]] = []
    try:
        for index, item in enumerate(selected):
            _assert_no_cnn_writer()
            _write_state(
                mode=mode,
                selected=len(selected),
                completed=len(receipts),
                current_sample_hash=item.sample_hash,
            )
            receipt = preprocess_one(
                item,
                fixed_template=fixed,
                template_sha256=EXPECTED_TEMPLATE_SHA256,
                template_1mm_sha256=fixed_sha,
                runtime_bindings=runtime,
                device=device,
            )
            receipts.append(receipt)
            print(
                json.dumps(
                    {
                        "status": "ITEM_COMPLETED",
                        "index": index + 1,
                        "selected": len(selected),
                        "sample_hash": item.sample_hash,
                        "elapsed_seconds": receipt["durations_seconds"]["total"],
                    }
                ),
                flush=True,
            )
        _write_state(
            mode=mode,
            selected=len(selected),
            completed=len(receipts),
            current_sample_hash=None,
        )
        if mode == "smoke":
            body = {
                "schema_version": 1,
                "status": "PASS",
                "rows": len(receipts),
                "sample_hashes": [record["sample_hash"] for record in receipts],
                "item_receipts_sha256": _canonical_sha(
                    [record["integrity_sha256"] for record in receipts]
                ),
                "raw_paths_persisted": False,
                "labels_read": False,
                "outer_test_touched": False,
                "claim_ceiling": "POST_HOC_INTERNAL_TECHNICAL_EVIDENCE",
            }
            body["integrity_sha256"] = _canonical_sha(body)
            _atomic_write_json(SMOKE_REPORT, body)
            return body

        from src.research.brainiac_preprocessing import build_aggregate_manifest

        body = build_aggregate_manifest(
            receipts=receipts,
            expected_sample_hashes=[item.sample_hash for item in selected],
            artifact_root=ARTIFACT_ROOT,
            template_sha256=EXPECTED_TEMPLATE_SHA256,
            runtime_lock_sha256=runtime["runtime_lock_sha256"],
            strict_receipt_sha256=runtime["strict_receipt_sha256"],
            forward_receipt_sha256=runtime["forward_receipt_sha256"],
        )
        body["item_receipt_directory"] = _safe_relative(RECEIPT_ROOT)
        body["output_directory"] = _safe_relative(OUTPUT_ROOT)
        body["transform_directory"] = _safe_relative(TRANSFORM_ROOT)
        body["completed_at_utc"] = _utc_now()
        body["manifest_sha256"] = _canonical_sha(
            {key: value for key, value in body.items() if key != "manifest_sha256"}
        )
        _atomic_write_json(AGGREGATE_MANIFEST, body)
        return body
    finally:
        _release_lock()


def status() -> dict[str, Any]:
    items = build_work_items()
    receipts: list[dict[str, Any]] = []
    invalid: list[str] = []
    for item in items:
        try:
            receipt = _load_valid_completed(item)
            if receipt is not None:
                receipts.append(receipt)
        except RuntimeError:
            invalid.append(item.sample_hash)
    return {
        "status": "PASS" if not invalid else "BLOCKED",
        "expected_rows": len(items),
        "completed_rows": len(receipts),
        "invalid_rows": invalid,
        "remaining_rows": len(items) - len(receipts),
        "raw_paths_persisted": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("preflight", "smoke", "run", "status"), required=True)
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    if args.mode == "preflight":
        items, _, fixed_sha, runtime = preflight()
        result = {
            "status": "PASS",
            "real_rows": len(items),
            "template_1mm_sha256": fixed_sha,
            "runtime_bindings": runtime,
        }
    elif args.mode == "status":
        result = status()
    else:
        result = execute(mode=args.mode, limit=args.limit, device=args.device)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return 0 if result.get("status") in {"PASS", "COMPLETED"} else 3


if __name__ == "__main__":
    raise SystemExit(main())
