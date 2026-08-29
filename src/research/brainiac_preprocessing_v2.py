"""Schema-v2 raw-T1 preprocessing: transactions, resume, QC and pipeline.

Builds on the proven library mechanics of the hard-disabled schema-v1 script
(SimpleITK N4, deterministic rigid registration, pinned vendor HD-BET fast
fold0, MONAI 96-cube resize) under a new receipt-bound, privacy-preserving,
signal-safe contract.  Legacy entrypoints stay disabled; nothing in this
module reads labels, PET, or any identifier beyond the locked two-field
membership (sample_hash, t1_file_hash).
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time
from typing import Any, Callable, Mapping, Sequence

from src.research.brainiac_readiness import _beneath, _is_reparse
from src.research.brainiac_action_guard import _pid_identity
from src.research.brainiac_execution_gate import brainiac_action_lease
from src.research.brainiac_preprocessing_v2_schema import (
    AGGREGATE_ARTIFACT_TYPE,
    AUDIT_FALSE_FIELDS,
    RECEIPT_ARTIFACT_TYPE,
    RECEIPT_REQUIRED_FIELDS,
    canonical_json,
    canonical_sha,
    seal_receipt,
    sha256_file,
    validate_receipt,
    verify_receipt_artifacts,
    quarantine_decision as schema_quarantine_decision,
)


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
    ROOT / "models" / "pretrained" / "brainiac" / "templates" / "spm25_avg152T1_2mm.nii"
)
TEMPLATE_LOCK = TEMPLATE_PATH.with_name("spm25_avg152T1_2mm.lock.json")
EXPECTED_TEMPLATE_SHA256 = (
    "213557a43d4387243c51bf208b3a672ae2eb68091f7ffe47c28a9f69d9e3d9fd"
)
EXPECTED_HD_BET_WEIGHT_SHA256 = (
    "6f75233753c4750672815e2b7a86db754995ae44b8f1cd77bccfc37becd2d83c"
)
REGISTRATION_SEED = 20260821
EXPECTED_REAL_ROWS = 1669
RECEIPT_GENERATION = 1

PRIMARY_ROOT = ROOT / "data" / "derived" / "method_x_brainiac" / "r1_primary_v2"
SMOKE_ROOT = ROOT / "data" / "derived" / "method_x_brainiac" / "r1_primary_v2_smoke"
POLICY_PATH = (
    ROOT / "governance" / "brainiac_preprocessing_r1_primary_v2_policy_v3.json"
)
LOCAL_SCRATCH_ROOT = Path(
    r"C:\Users\USER\AppData\Local\AMUSE-X\brainiac_preprocessing_staging"
)
LOCAL_SCRATCH_MAX_BYTES = 1_073_741_824
LOCAL_SCRATCH_MIN_FREE_BYTES = 21_474_836_480
LOCAL_SCRATCH_EXPECTED_PEAK_PER_SAMPLE_BYTES = 33_554_432

CALIBRATION_SAMPLE_COUNT = 32
SENSITIVITY_SAMPLE_COUNT = 64
SMOKE_LIMIT = 12


def policy_sha256() -> str:
    if not POLICY_PATH.is_file():
        raise RuntimeError("BRAINIAC_PREPROCESSING_POLICY_MISSING")
    return sha256_file(POLICY_PATH)


# ---------------------------------------------------------------------------
# writer exclusion (mirrors the proven legacy check)
# ---------------------------------------------------------------------------


def assert_no_project_gpu_writer(
    *,
    process_provider: Any | None = None,
    gpu_provider: Any | None = None,
) -> None:
    """Block Method X, Method Y, BrainIAC, or foreign CUDA writers."""

    from src.research.brainiac_readiness import (
        DEFAULT_BATTERY_MANIFEST,
        _load_json_strict,
        _writer_blockers,
        inspect_writer_state,
    )

    battery, _ = _load_json_strict(DEFAULT_BATTERY_MANIFEST)
    writer_state = inspect_writer_state(
        battery,
        process_provider=process_provider,
        gpu_provider=gpu_provider,
    )
    blockers = _writer_blockers(writer_state)
    if blockers:
        raise RuntimeError(
            "BRAINIAC_PROJECT_GPU_WRITER_BLOCKED:" + ",".join(blockers)
        )
    from src.research.brainiac_action_guard import _recorded_lock_status

    for battery_dir in sorted((ROOT / "results" / "final_results").glob("*/")):
        status = _recorded_lock_status(
            battery_dir / ".battery.lock", role="battery_lock"
        )
        if status.get("liveness") == "LIVE":
            raise RuntimeError("BRAINIAC_BATTERY_LOCK_OWNER_ACTIVE")


def assert_no_cnn_writer() -> None:
    """Backward-compatible name for the all-project GPU writer boundary."""

    assert_no_project_gpu_writer()


# ---------------------------------------------------------------------------
# membership (two-field, privacy-bounded)
# ---------------------------------------------------------------------------


def _decode(value: object) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def load_locked_membership(
    *,
    cache_path: Path = CACHE_PATH,
    catalog_path: Path = CATALOG_PATH,
    expected_rows: int | None = None,
) -> list[dict[str, str]]:
    """Return the locked real roster as (sample_hash, t1_file_hash) pairs only."""

    import h5py

    if not Path(cache_path).is_file():
        raise RuntimeError("BRAINIAC_LOCKED_CACHE_MISSING")
    with h5py.File(cache_path, "r") as handle:
        required = {"sample_hash", "source", "status"}
        if not required.issubset(handle.keys()):
            raise RuntimeError("BRAINIAC_LOCKED_CACHE_SCHEMA_INVALID")
        members = [
            _decode(sample_hash)
            for sample_hash, source, status in zip(
                handle["sample_hash"][:],
                handle["source"][:],
                handle["status"][:],
                strict=True,
            )
            if _decode(source) == "real" and _decode(status).upper() == "OK"
        ]
        n_attr = int(handle.attrs.get("n_real_eligible", -1))
    expected = EXPECTED_REAL_ROWS if expected_rows is None else expected_rows
    if n_attr != expected:
        raise RuntimeError("BRAINIAC_LOCKED_CACHE_REAL_COUNT_ATTRIBUTE_INVALID")
    if len(members) != expected or len(members) != len(set(members)):
        raise RuntimeError("BRAINIAC_LOCKED_CACHE_REAL_MEMBERSHIP_INVALID")

    with open(catalog_path, "r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    by_sample: dict[str, dict[str, str]] = {}
    for row in rows:
        sample_hash = str(row.get("sample_hash") or "")
        if sample_hash in by_sample:
            raise RuntimeError("BRAINIAC_CATALOG_SAMPLE_HASH_DUPLICATE")
        by_sample[sample_hash] = row
    membership: list[dict[str, str]] = []
    for sample_hash in members:
        row = by_sample.get(sample_hash)
        if row is None:
            raise RuntimeError(f"BRAINIAC_CATALOG_SAMPLE_MISSING:{sample_hash}")
        t1_file_hash = str(row.get("t1_file_hash") or "")
        if not t1_file_hash:
            raise RuntimeError(f"BRAINIAC_CATALOG_T1_HASH_MISSING:{sample_hash}")
        membership.append(
            {"sample_hash": sample_hash, "t1_file_hash": t1_file_hash}
        )
    return sorted(membership, key=lambda item: item["sample_hash"])


def build_raw_index(*, data_root: Path = RAW_DATA_ROOT) -> dict[str, Path]:
    """Filename-hash index over the read-only raw root; paths stay in memory."""

    from src.data.a4_schema import T1_NAME_PATTERN
    from src.utils.privacy import stable_hash
    import re

    if not Path(data_root).is_dir():
        raise RuntimeError("BRAINIAC_RAW_DATA_ROOT_UNAVAILABLE")
    pattern = re.compile(T1_NAME_PATTERN, re.IGNORECASE)
    index: dict[str, Path] = {}
    t1_root = Path(data_root) / "T1"
    roots = [t1_root] if t1_root.exists() else []
    if not roots:
        roots = [
            path
            for path in Path(data_root).rglob("*")
            if path.is_dir() and path.name.casefold() == "t1"
        ]
    candidates = (
        candidate
        for root in roots
        for candidate in root.rglob("*")
        if candidate.is_file()
        and candidate.name.casefold().endswith((".nii", ".nii.gz"))
    )
    for candidate in candidates:
        if not pattern.match(candidate.name):
            continue
        file_hash = stable_hash(candidate.name, prefix="t1")
        if file_hash in index and index[file_hash].resolve() != candidate.resolve():
            raise RuntimeError(f"BRAINIAC_RAW_T1_HASH_COLLISION:{file_hash}")
        index[file_hash] = candidate
    if not index:
        raise RuntimeError("BRAINIAC_RAW_T1_INDEX_EMPTY")
    return index


def resolve_sources(
    membership: Sequence[Mapping[str, str]], *, index: Mapping[str, Path] | None = None
) -> dict[str, Path]:
    """In-memory sample_hash → raw path map; never persisted, never logged."""

    raw_index = dict(index) if index is not None else build_raw_index()
    resolved: dict[str, Path] = {}
    for member in membership:
        source = raw_index.get(str(member["t1_file_hash"]))
        if source is None:
            raise RuntimeError(f"BRAINIAC_RAW_T1_NOT_FOUND:{member['sample_hash']}")
        resolved[str(member["sample_hash"])] = source
    return resolved


# ---------------------------------------------------------------------------
# label-blind selection and MAD calibration
# ---------------------------------------------------------------------------


def _lowest_by_hash(values: Sequence[str], count: int) -> list[str]:
    if count <= 0 or count > len(values):
        raise RuntimeError("BRAINIAC_SELECTION_COUNT_INVALID")
    return sorted(values, key=lambda value: hashlib.sha256(value.encode()).hexdigest())[
        :count
    ]


def select_calibration_samples(
    members: Sequence[str], *, count: int = CALIBRATION_SAMPLE_COUNT
) -> list[str]:
    return _lowest_by_hash(sorted(set(members)), count)


def select_sensitivity_samples(
    members: Sequence[str], *, count: int = SENSITIVITY_SAMPLE_COUNT
) -> list[str]:
    return _lowest_by_hash(sorted(set(members)), count)


def compute_mad_thresholds(
    values: Sequence[float], *, fixed_bounds: tuple[float, float]
) -> dict[str, Any]:
    """median ± 6*MAD clamped to fixed contract bounds; explicit MAD=0 rule."""

    if not values:
        raise RuntimeError("BRAINIAC_MAD_CALIBRATION_EMPTY")
    ordered = sorted(float(v) for v in values)
    n = len(ordered)
    median = ordered[n // 2] if n % 2 else (ordered[n // 2 - 1] + ordered[n // 2]) / 2
    deviations = sorted(abs(v - median) for v in ordered)
    mad = deviations[n // 2] if n % 2 else (deviations[n // 2 - 1] + deviations[n // 2]) / 2
    if mad == 0.0:
        # Explicit conservative fallback: MAD degenerate → fixed contract
        # bounds only, never a zero-width auto-accept band.
        low, high = fixed_bounds
        return {
            "median": median,
            "mad": 0.0,
            "low": low,
            "high": high,
            "mad_zero_fallback": True,
        }
    # MAD bands are anomaly detection on top of the always-applied fixed
    # geometric QC; they are deliberately not clamped to the fixed bounds.
    return {
        "median": median,
        "mad": mad,
        "low": median - 6 * mad,
        "high": median + 6 * mad,
        "mad_zero_fallback": False,
    }


def build_threshold_manifest(
    metrics_by_sample: Mapping[str, Mapping[str, float]],
    *,
    receipt_integrity_sha256s: Mapping[str, str],
    contract_bindings: Mapping[str, str],
    fixed_bounds: Mapping[str, tuple[float, float]] | None = None,
) -> dict[str, Any]:
    if not metrics_by_sample:
        raise RuntimeError("BRAINIAC_THRESHOLD_MANIFEST_EMPTY")
    sample_hashes = sorted(map(str, metrics_by_sample))
    receipt_hashes = dict(
        sorted((str(key), str(value)) for key, value in receipt_integrity_sha256s.items())
    )
    bound_contract = dict(
        sorted((str(key), str(value)) for key, value in contract_bindings.items())
    )
    if set(sample_hashes) != set(receipt_hashes):
        raise RuntimeError("BRAINIAC_THRESHOLD_RECEIPT_MEMBERSHIP_MISMATCH")
    if any(len(value) != 64 for value in (*receipt_hashes.values(), *bound_contract.values())):
        raise RuntimeError("BRAINIAC_THRESHOLD_BINDING_SHA_INVALID")
    bounds = dict(fixed_bounds or {})
    metric_names = sorted(set().union(*(row.keys() for row in metrics_by_sample.values())))
    thresholds: dict[str, dict[str, float]] = {}
    stats: dict[str, dict[str, float]] = {}
    for metric in metric_names:
        values = [float(row[metric]) for row in metrics_by_sample.values() if metric in row]
        computed = compute_mad_thresholds(
            values, fixed_bounds=bounds.get(metric, (min(values), max(values)))
        )
        stats[metric] = {
            "median": computed["median"],
            "mad": computed["mad"],
            "mad_zero_fallback": computed["mad_zero_fallback"],
        }
        thresholds[metric] = {"low": computed["low"], "high": computed["high"]}
    body: dict[str, Any] = {
        "schema_version": 2,
        "artifact": "BRAINIAC_PREPROCESSING_QC_THRESHOLD_MANIFEST_V2",
        "selection_rule": f"{len(metrics_by_sample)}_LOWEST_SHA256_OF_SAMPLE_HASH",
        "sample_count": len(metrics_by_sample),
        "sample_hashes": sample_hashes,
        "receipt_integrity_sha256s": receipt_hashes,
        "contract_bindings": bound_contract,
        "pipeline_contract_sha256": canonical_sha(bound_contract),
        "metrics": stats,
        "thresholds": thresholds,
        "labels_read": False,
        "pet_read": False,
        "raw_paths_persisted": False,
    }
    body["integrity_sha256"] = canonical_sha(
        {key: value for key, value in body.items() if key != "integrity_sha256"}
    )
    return body


def qc_threshold_anomalies(
    receipt_like: Mapping[str, Any], threshold_manifest: Mapping[str, Any]
) -> list[str]:
    """Return MAD-band violations; missing/non-finite fields fail closed."""

    thresholds = threshold_manifest.get("thresholds")
    if not isinstance(thresholds, Mapping) or not thresholds:
        raise RuntimeError("BRAINIAC_PREPROCESSING_THRESHOLD_MANIFEST_INVALID")
    anomalies: list[str] = []
    for metric, limits in sorted(thresholds.items()):
        if not isinstance(limits, Mapping) or metric not in receipt_like:
            raise RuntimeError(f"BRAINIAC_PREPROCESSING_THRESHOLD_METRIC_MISSING:{metric}")
        value = float(receipt_like[metric])
        low = float(limits.get("low", float("nan")))
        high = float(limits.get("high", float("nan")))
        if not all(math.isfinite(item) for item in (value, low, high)) or low > high:
            raise RuntimeError("BRAINIAC_PREPROCESSING_THRESHOLD_VALUE_INVALID")
        if not low <= value <= high:
            anomalies.append(str(metric))
    return anomalies


def normalize_foreground(volume: Any, foreground_mask: Any) -> Any:
    """Z-normalize foreground only while preserving exact zero background."""

    import numpy as np

    array = np.asarray(volume, dtype=np.float32)
    mask = np.asarray(foreground_mask, dtype=bool)
    if array.shape != mask.shape or not mask.any():
        raise RuntimeError("BRAINIAC_V2_SENSITIVITY_FOREGROUND_INVALID")
    values = array[mask]
    if not np.isfinite(values).all():
        raise RuntimeError("BRAINIAC_V2_SENSITIVITY_NONFINITE")
    std = float(values.std())
    if not math.isfinite(std) or std <= 1e-8:
        raise RuntimeError("BRAINIAC_V2_SENSITIVITY_FOREGROUND_DEGENERATE")
    output = np.zeros_like(array, dtype=np.float32)
    output[mask] = ((values - float(values.mean())) / std).astype(np.float32)
    return output


# ---------------------------------------------------------------------------
# morphology without wrap-around
# ---------------------------------------------------------------------------


_SIX_CONNECTIVITY = None


def _get_6_connectivity():
    global _SIX_CONNECTIVITY
    if _SIX_CONNECTIVITY is None:
        import numpy as np
        s = np.zeros((3, 3, 3), dtype=bool)
        s[1, 1, 1] = True
        s[0, 1, 1] = True
        s[2, 1, 1] = True
        s[1, 0, 1] = True
        s[1, 2, 1] = True
        s[1, 1, 0] = True
        s[1, 1, 2] = True
        _SIX_CONNECTIVITY = s
    return _SIX_CONNECTIVITY


def dilate_mask(mask: Any, *, iterations: int = 1) -> Any:
    """Binary 6-neighbour dilation; outside the volume is background (no wrap)."""

    import numpy as np
    from scipy.ndimage import binary_dilation

    return binary_dilation(
        np.asarray(mask, dtype=bool),
        structure=_get_6_connectivity(),
        iterations=int(iterations),
    )


def erode_mask(mask: Any, *, iterations: int = 1) -> Any:
    """Binary 6-neighbour erosion with a non-wrapping foreground boundary."""

    import numpy as np
    from scipy.ndimage import binary_erosion

    return binary_erosion(
        np.asarray(mask, dtype=bool),
        structure=_get_6_connectivity(),
        iterations=int(iterations),
        # A foreground boundary makes erosion the local inverse of dilation
        # for tissue touching a cropped face, without reading the opposite face.
        border_value=1,
    )


def content_digest(path: Path) -> str:
    """Deterministic digest of voxel + affine content (header-timestamp free)."""

    import nibabel as nib
    import numpy as np

    image = nib.load(str(path))
    array = np.ascontiguousarray(image.get_fdata(dtype="float32"))
    affine = np.asarray(image.affine, dtype="<f8")
    digest = hashlib.sha256()
    digest.update(array.tobytes())
    digest.update(affine.tobytes())
    digest.update(str(array.shape).encode("ascii"))
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# official transform identity (imported, not reimplemented)
# ---------------------------------------------------------------------------


_EXPECTED_VENDOR_COMPOSITION = (
    "LoadImaged",
    "EnsureChannelFirstd",
    "Resized",
    "NormalizeIntensityd",
    "ToTensord",
)


def _import_vendor_dataset() -> Any:
    import importlib.util

    module_path = VENDOR_ROOT / "src" / "dataset.py"
    if not module_path.is_file():
        raise RuntimeError("BRAINIAC_PREPROCESSING_VENDOR_DATASET_MISSING")
    spec = importlib.util.spec_from_file_location(
        "brainiac_vendor_dataset_pinned", module_path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_contract_transform() -> tuple[Any, dict[str, Any]]:
    """Vendor-composed 96³ resize + contract-mandated nonzero normalization.

    The resize path uses the ops of the imported vendor
    ``get_validation_transform`` object itself; the vendor's zero-inclusive
    normalization is replaced by the B2 contract's nonzero channel-wise
    normalization as the explicit terminal stage.  Any composition drift
    fails closed.
    """

    from monai.transforms import Compose, NormalizeIntensityd, ToTensord

    vendor = _import_vendor_dataset()
    factory = getattr(vendor, "get_validation_transform", None)
    if factory is None:
        raise RuntimeError("BRAINIAC_PREPROCESSING_TRANSFORM_IDENTITY_UNKNOWN")
    vendor_transform = factory(image_size=(96, 96, 96))
    ops = list(getattr(vendor_transform, "transforms", ()))
    names = tuple(type(op).__name__ for op in ops)
    if names != _EXPECTED_VENDOR_COMPOSITION:
        raise RuntimeError("BRAINIAC_PREPROCESSING_TRANSFORM_IDENTITY_UNKNOWN")
    contract = Compose(
        [
            *ops[:3],
            NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
            ToTensord(keys=["image"]),
        ]
    )
    binding = {
        "official_transform_source_sha256": sha256_file(VENDOR_ROOT / "src" / "dataset.py"),
        "official_transform_steps": [
            "LoadImaged",
            "EnsureChannelFirstd",
            "Resized_96_trilinear_vendor",
            "NormalizeIntensityd_nonzero_channel_wise_contract",
            "ToTensord",
        ],
    }
    return contract, binding


# ---------------------------------------------------------------------------
# sample-level transaction with journal, quarantine and signal-safe resume
# ---------------------------------------------------------------------------


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    encoded = canonical_json(dict(payload))
    descriptor = os.open(
        temporary, os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_BINARY", 0)
    )
    try:
        os.write(descriptor, encoded)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.replace(temporary, path)


def _fsync_file(path: Path) -> None:
    # FlushFileBuffers on Windows requires write access to the handle.
    descriptor = os.open(path, os.O_RDWR | getattr(os, "O_BINARY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _same_volume(first: Path, second: Path) -> bool:
    """Return whether two paths can use one-volume atomic rename semantics."""

    first = Path(first)
    second = Path(second)
    if os.name == "nt":
        return first.resolve(strict=False).drive.casefold() == second.resolve(
            strict=False
        ).drive.casefold()

    def existing_anchor(path: Path) -> Path:
        candidate = path.resolve(strict=False)
        while not candidate.exists() and candidate != candidate.parent:
            candidate = candidate.parent
        return candidate

    return existing_anchor(first).stat().st_dev == existing_anchor(second).stat().st_dev


def _tree_size_and_validate(root: Path) -> int:
    """Measure a controlled scratch tree without following links/reparse points."""

    root = Path(root)
    if not root.exists():
        return 0
    if not root.is_dir() or _is_reparse(root):
        raise RuntimeError("BRAINIAC_LOCAL_SCRATCH_REPARSE_FORBIDDEN")
    total = 0
    for current, directories, files in os.walk(root, followlinks=False):
        current_path = Path(current)
        for name in list(directories):
            candidate = current_path / name
            if _is_reparse(candidate):
                raise RuntimeError("BRAINIAC_LOCAL_SCRATCH_REPARSE_FORBIDDEN")
        for name in files:
            candidate = current_path / name
            if _is_reparse(candidate) or not candidate.is_file():
                raise RuntimeError("BRAINIAC_LOCAL_SCRATCH_ENTRY_INVALID")
            total += candidate.stat().st_size
    return total


def inspect_local_scratch_storage(
    *, scratch_base: Path | None = None, create: bool = False
) -> dict[str, Any]:
    """Validate the single allow-listed, bounded NVMe scratch authority."""

    configured_lexical = Path(os.path.abspath(str(LOCAL_SCRATCH_ROOT)))
    requested_lexical = Path(
        os.path.abspath(str(Path(scratch_base or LOCAL_SCRATCH_ROOT)))
    )
    if requested_lexical != configured_lexical:
        raise RuntimeError("BRAINIAC_LOCAL_SCRATCH_OUTSIDE_ALLOWLIST")
    anchor = requested_lexical
    while not anchor.exists() and anchor != anchor.parent:
        anchor = anchor.parent
    if not anchor.exists() or _is_reparse(anchor):
        raise RuntimeError("BRAINIAC_LOCAL_SCRATCH_ANCHOR_INVALID")
    current = requested_lexical
    while current != anchor:
        if current.exists() and _is_reparse(current):
            raise RuntimeError("BRAINIAC_LOCAL_SCRATCH_REPARSE_FORBIDDEN")
        current = current.parent
    if create:
        requested_lexical.mkdir(parents=True, exist_ok=True)
    base = requested_lexical.resolve(strict=False)
    if base != configured_lexical.resolve(strict=False):
        raise RuntimeError("BRAINIAC_LOCAL_SCRATCH_REPARSE_FORBIDDEN")
    used = _tree_size_and_validate(base) if base.exists() else 0
    free = shutil.disk_usage(anchor).free
    if used > int(LOCAL_SCRATCH_MAX_BYTES):
        raise RuntimeError("LOCAL_SCRATCH_CAP_EXCEEDED")
    if free < int(LOCAL_SCRATCH_MIN_FREE_BYTES):
        raise RuntimeError("BRAINIAC_LOCAL_SCRATCH_FREE_SPACE_INSUFFICIENT")
    return {
        "status": "PASS",
        "exists": base.exists(),
        "used_bytes": int(used),
        "free_bytes": int(free),
        "maximum_bytes": int(LOCAL_SCRATCH_MAX_BYTES),
        "minimum_free_bytes": int(LOCAL_SCRATCH_MIN_FREE_BYTES),
        "authoritative": False,
    }


def _sample_scratch_path(root: Path, sample_hash: str, base: Path) -> Path:
    run_key = hashlib.sha256(
        str(Path(root).resolve(strict=False)).casefold().encode("utf-8")
    ).hexdigest()[:20]
    return Path(base) / run_key / sample_hash


def resolve_sample_scratch(
    root: Path,
    sample_hash: str,
    *,
    scratch_base: Path | None = None,
) -> Path:
    """Resolve a privacy-bounded local scratch path and enforce its storage cap."""

    base = Path(scratch_base or LOCAL_SCRATCH_ROOT)
    storage = inspect_local_scratch_storage(scratch_base=base, create=True)
    projected = min(
        int(LOCAL_SCRATCH_EXPECTED_PEAK_PER_SAMPLE_BYTES),
        max(0, int(LOCAL_SCRATCH_MAX_BYTES) // 2),
    )
    if int(storage["used_bytes"]) + projected > int(LOCAL_SCRATCH_MAX_BYTES):
        raise RuntimeError("LOCAL_SCRATCH_CAP_EXCEEDED")
    scratch = _sample_scratch_path(root, sample_hash, base.resolve(strict=True))
    base_resolved = base.resolve(strict=True)
    try:
        scratch.resolve(strict=False).relative_to(base_resolved)
    except ValueError as exc:
        raise RuntimeError("BRAINIAC_LOCAL_SCRATCH_OUTSIDE_ALLOWLIST") from exc
    current = scratch
    while current != base_resolved:
        if current.exists() and _is_reparse(current):
            raise RuntimeError("BRAINIAC_LOCAL_SCRATCH_REPARSE_FORBIDDEN")
        current = current.parent
    return scratch


def _commit_staged_file(source: Path, target: Path) -> None:
    """Commit a staged file to H atomically, verifying cross-volume copies."""

    source = Path(source)
    target = Path(target)
    if not source.is_file() or _is_reparse(source):
        raise RuntimeError("BRAINIAC_PREPROCESSING_STAGED_FILE_INVALID")
    if target.exists():
        raise RuntimeError("BRAINIAC_PREPROCESSING_COMMIT_TARGET_EXISTS")
    target.parent.mkdir(parents=True, exist_ok=True)
    source_size = source.stat().st_size
    source_sha = sha256_file(source)
    if _same_volume(source, target):
        os.replace(source, target)
    else:
        temporary = target.with_name(
            target.name + f".tmp-{os.getpid()}-{time.time_ns()}"
        )
        if temporary.exists():
            raise RuntimeError("BRAINIAC_PREPROCESSING_COMMIT_TEMP_EXISTS")
        try:
            with source.open("rb") as reader, temporary.open("xb") as writer:
                shutil.copyfileobj(reader, writer, length=8 * 1024 * 1024)
                writer.flush()
                os.fsync(writer.fileno())
            if (
                temporary.stat().st_size != source_size
                or sha256_file(temporary) != source_sha
            ):
                raise RuntimeError("BRAINIAC_PREPROCESSING_CROSS_VOLUME_COPY_MISMATCH")
            os.replace(temporary, target)
            _fsync_file(target)
            if target.stat().st_size != source_size or sha256_file(target) != source_sha:
                raise RuntimeError("BRAINIAC_PREPROCESSING_CROSS_VOLUME_COMMIT_MISMATCH")
            source.unlink()
        finally:
            if temporary.exists():
                temporary.unlink()
    _fsync_file(target)


def _tree_manifest(root: Path) -> dict[str, tuple[int, str]]:
    manifest: dict[str, tuple[int, str]] = {}
    for path in sorted(Path(root).rglob("*")):
        if _is_reparse(path):
            raise RuntimeError("BRAINIAC_PREPROCESSING_SCRATCH_REPARSE_FORBIDDEN")
        if path.is_file():
            manifest[path.relative_to(root).as_posix()] = (
                path.stat().st_size,
                sha256_file(path),
            )
    return manifest


def _quarantine_move(path: Path, root: Path, sample_hash: str) -> Path | None:
    if not path.exists():
        return None
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    target_dir = Path(root) / "quarantine" / f"{stamp}-{time.time_ns()}-{sample_hash}"
    target_dir.mkdir(parents=True, exist_ok=True)
    _commit_staged_file(path, target_dir / path.name)
    return target_dir


def _archive_stale_scratch(scratch: Path, root: Path, sample_hash: str) -> Path | None:
    if not scratch.exists():
        return None
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    target = Path(root) / "quarantine" / f"{stamp}-{time.time_ns()}-{sample_hash}-scratch"
    target.parent.mkdir(parents=True, exist_ok=True)
    if _same_volume(scratch, target):
        os.replace(scratch, target)
    else:
        temporary = target.with_name(target.name + f".tmp-{os.getpid()}-{time.time_ns()}")
        if temporary.exists():
            raise RuntimeError("BRAINIAC_PREPROCESSING_SCRATCH_ARCHIVE_TEMP_EXISTS")
        try:
            shutil.copytree(scratch, temporary, copy_function=shutil.copy2)
            if _tree_manifest(temporary) != _tree_manifest(scratch):
                raise RuntimeError("BRAINIAC_PREPROCESSING_SCRATCH_ARCHIVE_MISMATCH")
            os.replace(temporary, target)
            if _tree_manifest(target) != _tree_manifest(scratch):
                raise RuntimeError("BRAINIAC_PREPROCESSING_SCRATCH_ARCHIVE_MISMATCH")
            shutil.rmtree(scratch)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
    _atomic_write_json(
        target / "reason.json",
        {
            "sample_hash": sample_hash,
            "reason_code": "STALE_SCRATCH_ARCHIVED",
            "raw_path_persisted": False,
        },
    )
    return target


def load_committed_receipt(
    sample_hash: str,
    root: Path,
    *,
    expected_bindings: Mapping[str, str] | None = None,
    recover_missing_journal: bool = False,
) -> dict[str, Any]:
    receipt_path = Path(root) / "receipts" / f"{sample_hash}.json"
    if not receipt_path.is_file():
        raise RuntimeError("BRAINIAC_PREPROCESSING_NOT_COMMITTED")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if not isinstance(receipt, dict):
        raise RuntimeError("BRAINIAC_PREPROCESSING_RECEIPT_INVALID")
    if int(receipt.get("generation", 0)) < RECEIPT_GENERATION:
        raise RuntimeError("BRAINIAC_PREPROCESSING_GENERATION_ROLLBACK")
    try:
        validate_receipt(
            receipt,
            artifact_root=Path(root),
            expected_bindings=expected_bindings,
        )
    except RuntimeError as exc:
        if "BINDING_MISMATCH" in str(exc):
            raise RuntimeError("BRAINIAC_PREPROCESSING_COMMITTED_MISMATCH") from exc
        raise
    committed = Path(root) / "journals" / sample_hash / "committed.json"
    if not committed.is_file():
        if not recover_missing_journal:
            raise RuntimeError("BRAINIAC_PREPROCESSING_JOURNAL_MISSING_COMMIT")
        _atomic_write_json(
            committed,
            {
                "sample_hash": sample_hash,
                "receipt_sha256": receipt["integrity_sha256"],
                "output_sha256": receipt["output_sha256"],
                "transform_sha256": receipt["transform_sha256"],
                "status": "JOURNAL_COMMITTED",
                "recovered_from_receipt": True,
            },
        )
    journal = json.loads(committed.read_text(encoding="utf-8"))
    if (
        journal.get("receipt_sha256") != receipt.get("integrity_sha256")
        or journal.get("output_sha256") != receipt.get("output_sha256")
        or journal.get("transform_sha256") != receipt.get("transform_sha256")
    ):
        raise RuntimeError("BRAINIAC_PREPROCESSING_JOURNAL_MISMATCH")
    verify_receipt_artifacts(receipt, artifact_root=Path(root))
    return receipt


def _reconcile_stale_sample_lock(
    lock_path: Path, *, root: Path, sample_hash: str
) -> None:
    """Archive a sample lock only with liveness proof (card Step 5).

    Dead owner PID -> provably stale, append-only archive.  Live PID (or
    PID-reuse doubt) -> fail closed; never remove a lock we cannot disprove.
    """

    payload = json.loads(lock_path.read_text(encoding="utf-8"))
    try:
        pid = int(payload.get("pid", -1))
    except (TypeError, ValueError) as exc:
        raise RuntimeError("BRAINIAC_PREPROCESSING_SAMPLE_LOCK_INVALID") from exc
    if pid <= 0:
        raise RuntimeError("BRAINIAC_PREPROCESSING_SAMPLE_LOCK_INVALID")
    live, creation = _pid_identity(pid)
    if live is None:
        raise RuntimeError("BRAINIAC_PREPROCESSING_SAMPLE_LOCK_LIVENESS_UNKNOWN")
    if live is False:
        stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        target = (
            Path(root)
            / "quarantine"
            / f"{stamp}-{time.time_ns()}-{sample_hash}-dead-lock"
        )
        target.mkdir(parents=True, exist_ok=False)
        archived = target / lock_path.name
        os.replace(lock_path, archived)
        _atomic_write_json(
            target / "reason.json",
            {
                "sample_hash": sample_hash,
                "reason_code": "PROVEN_DEAD_SAMPLE_LOCK_ARCHIVED",
                "dead_pid": pid,
                "archived_lock_sha256": sha256_file(archived),
                "raw_path_persisted": False,
            },
        )
        if lock_path.exists() or not archived.is_file():
            raise RuntimeError("BRAINIAC_PREPROCESSING_SAMPLE_LOCK_ARCHIVE_FAILED")
        return
    recorded = payload.get("process_create_time")
    if recorded is not None and str(recorded) != str(creation):
        raise RuntimeError("BRAINIAC_PREPROCESSING_SAMPLE_LOCK_PID_REUSE")
    raise RuntimeError("BRAINIAC_PREPROCESSING_SAMPLE_LOCK_OWNER_ACTIVE")


def run_sample_transaction(
    item: Mapping[str, str],
    root: Path,
    *,
    pipeline: Callable[[Mapping[str, str], Path], Mapping[str, Any]],
    bindings: Mapping[str, str],
    generation: int = RECEIPT_GENERATION,
    quarantine_check: bool = True,
    threshold_manifest: Mapping[str, Any] | None = None,
    fault_hook: Callable[[str], None] | None = None,
    scratch_base: Path | None = None,
) -> dict[str, Any]:
    """One sample as an OS-durable transaction; committed valid samples skip."""

    assert_no_cnn_writer()
    sample_hash = str(item["sample_hash"])
    root = Path(root)
    for directory in ("outputs", "transforms", "receipts", "quarantine", "journals"):
        (root / directory).mkdir(parents=True, exist_ok=True)
    receipts_dir = root / "receipts"
    receipt_path = receipts_dir / f"{sample_hash}.json"
    if receipt_path.is_file():
        receipt = load_committed_receipt(
            sample_hash,
            root,
            expected_bindings=bindings,
            recover_missing_journal=True,
        )
        scratch = (
            resolve_sample_scratch(root, sample_hash, scratch_base=scratch_base)
            if scratch_base is not None
            else root / "scratch" / sample_hash
        )
        if scratch.exists():
            _archive_stale_scratch(scratch, root, sample_hash)
        return receipt

    output_target = root / "outputs" / f"{sample_hash}.nii.gz"
    transform_target = root / "transforms" / f"{sample_hash}.tfm"
    for stale in (output_target, transform_target):
        if stale.exists():
            # payload without committed receipt/journal = uncommitted → quarantine
            _quarantine_move(stale, root, sample_hash)

    journal_dir = root / "journals" / sample_hash
    journal_dir.mkdir(parents=True, exist_ok=True)
    lock_path = journal_dir / ".sample.lock"
    nonce = hashlib.sha256(
        f"{os.getpid()}:{time.time_ns()}:{sample_hash}".encode()
    ).hexdigest()
    owner_live, owner_creation = _pid_identity(os.getpid())
    if owner_live is not True or not owner_creation:
        raise RuntimeError("BRAINIAC_PREPROCESSING_OWN_PROCESS_IDENTITY_UNKNOWN")
    lock_payload = {
        "schema_version": 1,
        "pid": os.getpid(),
        "process_create_time": owner_creation,
        "nonce": nonce,
        "ownership_token": nonce[::-1],
        "sample_hash": sample_hash,
    }
    if lock_path.exists():
        _reconcile_stale_sample_lock(
            lock_path, root=root, sample_hash=sample_hash
        )
    try:
        descriptor = os.open(
            lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_BINARY", 0)
        )
    except FileExistsError as exc:
        raise RuntimeError("BRAINIAC_PREPROCESSING_SAMPLE_LOCK_PRESENT") from exc
    try:
        os.write(descriptor, canonical_json(lock_payload))
        os.fsync(descriptor)
    finally:
        os.close(descriptor)

    try:
        scratch = (
            resolve_sample_scratch(root, sample_hash, scratch_base=scratch_base)
            if scratch_base is not None
            else root / "scratch" / sample_hash
        )
        if scratch.exists():
            _archive_stale_scratch(scratch, root, sample_hash)
        scratch.mkdir(parents=True, exist_ok=True)
    except BaseException:
        if lock_path.exists():
            live = json.loads(lock_path.read_text(encoding="utf-8"))
            if live.get("nonce") != nonce:
                raise RuntimeError(
                    "BRAINIAC_PREPROCESSING_SAMPLE_LOCK_OWNERSHIP_LOST"
                )
            lock_path.unlink()
        raise
    try:
        _atomic_write_json(
            journal_dir / "scratch.json",
            {
                "sample_hash": sample_hash,
                "status": "SCRATCH",
                "event_at_utc": time.strftime(
                    "%Y-%m-%dT%H:%M:%S+00:00", time.gmtime()
                ),
            },
        )
        if fault_hook is not None:
            fault_hook("SCRATCH")
        result = dict(pipeline(item, scratch))
        if scratch_base is not None:
            inspect_local_scratch_storage(scratch_base=scratch_base, create=False)
        runtime_stage_seconds = result.get("runtime_stage_seconds")
        if runtime_stage_seconds is not None:
            if not isinstance(runtime_stage_seconds, Mapping):
                raise RuntimeError("BRAINIAC_PREPROCESSING_RUNTIME_TELEMETRY_INVALID")
            allowed_runtime_fields = (
                "n4",
                "registration_serial",
                "hd_bet_serial",
                "total",
            )
            runtime_payload: dict[str, Any] = {
                "schema_version": 1,
                "sample_hash": sample_hash,
                "status": "MEASURED_NONSCIENTIFIC_RUNTIME_TELEMETRY",
                "raw_path_persisted": False,
            }
            for field in allowed_runtime_fields:
                value = float(runtime_stage_seconds[field])
                if not math.isfinite(value) or value < 0:
                    raise RuntimeError("BRAINIAC_PREPROCESSING_RUNTIME_TELEMETRY_INVALID")
                runtime_payload[field + "_seconds"] = value
            _atomic_write_json(journal_dir / "runtime.json", runtime_payload)
        output_path = Path(result["output_path"])
        transform_path = Path(result["transform_path"])
        scratch_resolved = scratch.resolve(strict=True)
        for staged_path in (output_path, transform_path):
            try:
                staged_path.resolve(strict=True).relative_to(scratch_resolved)
            except (OSError, ValueError) as exc:
                raise RuntimeError(
                    "BRAINIAC_PREPROCESSING_STAGED_FILE_OUTSIDE_SCRATCH"
                ) from exc
        _fsync_file(output_path)
        _fsync_file(transform_path)
        if result.get("finite") is not True:
            raise RuntimeError("BRAINIAC_PREPROCESSING_QC_NONFINITE")
        if quarantine_check and schema_quarantine_decision(
            com_before=result["center_of_mass_before"],
            com_after=result["center_of_mass_after"],
            similarity_before=float(result["registration_similarity_before"]),
            similarity_after=float(result["registration_similarity_after"]),
        ) == "QUARANTINE_BOTH_WORSENED":
            for payload in (output_path, transform_path):
                _quarantine_move(payload, root, sample_hash)
            _atomic_write_json(
                journal_dir / "quarantined.json",
                {"sample_hash": sample_hash, "status": "QUARANTINED"},
            )
            raise RuntimeError("BRAINIAC_PREPROCESSING_QUARANTINE_BOTH_WORSENED")

        body: dict[str, Any] = {
            "schema_version": 2,
            "artifact": RECEIPT_ARTIFACT_TYPE,
            "sample_hash": sample_hash,
            "generation": int(generation),
            "status": "COMPLETED",
            "source_sha256": str(result["source_sha256"]),
            "source_size": int(result["source_size"]),
            "output_relative_path": f"outputs/{sample_hash}.nii.gz",
            "output_sha256": sha256_file(output_path),
            "output_size": output_path.stat().st_size,
            "transform_relative_path": f"transforms/{sample_hash}.tfm",
            "transform_sha256": sha256_file(transform_path),
            "transform_size": transform_path.stat().st_size,
            "vendor_commit": VENDOR_COMMIT,
            "hd_bet_weight_sha256": EXPECTED_HD_BET_WEIGHT_SHA256,
            "template_sha256": EXPECTED_TEMPLATE_SHA256,
            "affine_before": list(result["affine_before"]),
            "affine_after": list(result["affine_after"]),
            "spacing_before": list(result["spacing_before"]),
            "spacing_after": list(result["spacing_after"]),
            "origin_before": list(result["origin_before"]),
            "origin_after": list(result["origin_after"]),
            "direction_before": list(result["direction_before"]),
            "direction_after": list(result["direction_after"]),
            "com_coordinate_frame": str(result["com_coordinate_frame"]),
            "rigid_determinant": float(result["rigid_determinant"]),
            "orthogonality_error": float(result["orthogonality_error"]),
            "rotation_deg": float(result["rotation_deg"]),
            "translation_norm_mm": float(result["translation_norm_mm"]),
            "foreground_fraction": float(result["foreground_fraction"]),
            "finite": True,
            "normalized_nonzero_mean": float(result["normalized_nonzero_mean"]),
            "normalized_nonzero_std": float(result["normalized_nonzero_std"]),
            "center_of_mass_before": list(result["center_of_mass_before"]),
            "center_of_mass_after": list(result["center_of_mass_after"]),
            "registration_similarity_before": float(
                result["registration_similarity_before"]
            ),
            "registration_similarity_after": float(
                result["registration_similarity_after"]
            ),
            "labels_read": False,
            "pet_read": False,
            "raw_identifier_persisted": False,
            "raw_path_persisted": False,
            "direct_128_to_96_used": False,
        }
        body.update(dict(bindings))
        receipt = seal_receipt(body)
        if threshold_manifest is not None:
            anomalies = qc_threshold_anomalies(receipt, threshold_manifest)
            if anomalies:
                for payload in (output_path, transform_path):
                    _quarantine_move(payload, root, sample_hash)
                _atomic_write_json(
                    journal_dir / "quarantined.json",
                    {
                        "sample_hash": sample_hash,
                        "status": "QUARANTINED",
                        "reason_code": "MAD_THRESHOLD_ANOMALY",
                        "anomaly_fields": anomalies,
                    },
                )
                raise RuntimeError("BRAINIAC_PREPROCESSING_QUARANTINE_MAD_ANOMALY")

        _atomic_write_json(
            journal_dir / "prepared.json",
            {
                "sample_hash": sample_hash,
                "output_sha256": receipt["output_sha256"],
                "transform_sha256": receipt["transform_sha256"],
                "status": "PREPARED",
                "event_at_utc": time.strftime(
                    "%Y-%m-%dT%H:%M:%S+00:00", time.gmtime()
                ),
            },
        )
        if fault_hook is not None:
            fault_hook("PREPARED")
        for source, target in (
            (output_path, output_target),
            (transform_path, transform_target),
        ):
            if target.exists():
                raise RuntimeError("BRAINIAC_PREPROCESSING_COMMIT_TARGET_EXISTS")
            _commit_staged_file(source, target)
        _atomic_write_json(
            journal_dir / "payload_committed.json",
            {
                "sample_hash": sample_hash,
                "output_sha256": receipt["output_sha256"],
                "transform_sha256": receipt["transform_sha256"],
                "status": "PAYLOAD_COMMITTED",
                "event_at_utc": time.strftime(
                    "%Y-%m-%dT%H:%M:%S+00:00", time.gmtime()
                ),
            },
        )
        if fault_hook is not None:
            fault_hook("PAYLOAD_COMMITTED")
        _atomic_write_json(receipt_path, receipt)
        verify_receipt_artifacts(receipt, artifact_root=root)
        _atomic_write_json(
            journal_dir / "receipt_committed.json",
            {
                "sample_hash": sample_hash,
                "receipt_sha256": receipt["integrity_sha256"],
                "status": "RECEIPT_COMMITTED",
                "event_at_utc": time.strftime(
                    "%Y-%m-%dT%H:%M:%S+00:00", time.gmtime()
                ),
            },
        )
        if fault_hook is not None:
            fault_hook("RECEIPT_COMMITTED")
        _atomic_write_json(
            journal_dir / "committed.json",
            {
                "sample_hash": sample_hash,
                "receipt_sha256": receipt["integrity_sha256"],
                "output_sha256": receipt["output_sha256"],
                "transform_sha256": receipt["transform_sha256"],
                "status": "JOURNAL_COMMITTED",
                "event_at_utc": time.strftime(
                    "%Y-%m-%dT%H:%M:%S+00:00", time.gmtime()
                ),
            },
        )
        if fault_hook is not None:
            fault_hook("JOURNAL_COMMITTED")
        shutil.rmtree(scratch)
        if scratch.exists():
            raise RuntimeError("BRAINIAC_PREPROCESSING_COMMITTED_SCRATCH_CLEANUP_FAILED")
        return receipt
    finally:
        if lock_path.exists():
            live = json.loads(lock_path.read_text(encoding="utf-8"))
            if live.get("nonce") != nonce:
                raise RuntimeError("BRAINIAC_PREPROCESSING_SAMPLE_LOCK_OWNERSHIP_LOST")
            lock_path.unlink()
            if lock_path.exists():
                raise RuntimeError("BRAINIAC_PREPROCESSING_SAMPLE_LOCK_RELEASE_FAILED")


# ---------------------------------------------------------------------------
# batches, signal-safe resume, orchestration equivalence
# ---------------------------------------------------------------------------


def run_batch(
    items: Sequence[Mapping[str, str]],
    root: Path,
    *,
    pipeline_factory: Callable[[], Callable],
    bindings: Mapping[str, str],
    stop_requested: Callable[[], bool] | None = None,
    threshold_manifest: Mapping[str, Any] | None = None,
    scratch_base: Path | None = None,
) -> dict[str, Any]:
    receipts: list[dict[str, Any]] = []
    quarantined: list[str] = []
    paused = False
    for item in items:
        try:
            receipt = run_sample_transaction(
                item,
                root,
                pipeline=pipeline_factory(),
                bindings=bindings,
                threshold_manifest=threshold_manifest,
                scratch_base=scratch_base,
            )
        except RuntimeError as exc:
            if "BRAINIAC_PREPROCESSING_QUARANTINE_" in str(exc):
                quarantined.append(str(item["sample_hash"]))
                continue
            raise
        receipts.append(receipt)
        if stop_requested is not None and stop_requested():
            paused = True
            break
    status = (
        "PAUSED_RESUMABLE"
        if paused
        else ("BLOCKED_BY_QUARANTINE" if quarantined else "COMPLETED")
    )
    return {
        "status": status,
        "exit_code": 75 if paused else (3 if quarantined else 0),
        "completed": len(receipts),
        "selected": len(items),
        "quarantined": quarantined,
        "receipts": receipts,
    }


def run_orchestrated(
    items: Sequence[Mapping[str, str]],
    root: Path,
    *,
    pipeline_factory: Callable[[], Callable],
    bindings: Mapping[str, str],
    cpu_workers: int = 2,
    gpu_workers: int = 1,
    threshold_manifest: Mapping[str, Any] | None = None,
    stop_requested: Callable[[], bool] | None = None,
    scratch_base: Path | None = None,
) -> list[dict[str, Any]]:
    """Bounded parallel execution whose receipts must equal sequential ones."""

    return run_orchestrated_batch(
        items,
        root,
        pipeline_factory=pipeline_factory,
        bindings=bindings,
        cpu_workers=cpu_workers,
        gpu_workers=gpu_workers,
        threshold_manifest=threshold_manifest,
        stop_requested=stop_requested,
        scratch_base=scratch_base,
    )["receipts"]


def run_orchestrated_batch(
    items: Sequence[Mapping[str, str]],
    root: Path,
    *,
    pipeline_factory: Callable[[], Callable],
    bindings: Mapping[str, str],
    cpu_workers: int = 2,
    gpu_workers: int = 1,
    threshold_manifest: Mapping[str, Any] | None = None,
    stop_requested: Callable[[], bool] | None = None,
    scratch_base: Path | None = None,
) -> dict[str, Any]:
    """At most two in-flight samples; stop submits no new work and drains safely."""

    from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait

    if cpu_workers < 1 or gpu_workers != 1:
        raise RuntimeError("BRAINIAC_ORCHESTRATION_WORKER_COUNT_INVALID")
    max_in_flight = min(2, int(cpu_workers))
    indexed = list(enumerate(items))
    results: dict[int, dict[str, Any]] = {}
    quarantined: list[str] = []
    next_index = 0

    def process(item: Mapping[str, str]) -> dict[str, Any]:
        return run_sample_transaction(
            item,
            root,
            pipeline=pipeline_factory(),
            bindings=bindings,
            threshold_manifest=threshold_manifest,
            scratch_base=scratch_base,
        )

    with ThreadPoolExecutor(max_workers=max_in_flight) as pool:
        pending: dict[Future, int] = {}

        def submit_available() -> None:
            nonlocal next_index
            while (
                next_index < len(indexed)
                and len(pending) < max_in_flight
                and not (stop_requested is not None and stop_requested())
            ):
                position, item = indexed[next_index]
                next_index += 1
                pending[pool.submit(process, item)] = position

        submit_available()
        while pending:
            done, _ = wait(tuple(pending), return_when=FIRST_COMPLETED)
            for future in done:
                position = pending.pop(future)
                try:
                    results[position] = future.result()
                except RuntimeError as exc:
                    if "BRAINIAC_PREPROCESSING_QUARANTINE_" in str(exc):
                        quarantined.append(str(indexed[position][1]["sample_hash"]))
                    else:
                        raise
            submit_available()

    paused = next_index < len(indexed) and bool(stop_requested and stop_requested())
    status = (
        "PAUSED_RESUMABLE"
        if paused
        else ("BLOCKED_BY_QUARANTINE" if quarantined else "COMPLETED")
    )
    ordered = [results[index] for index in sorted(results)]
    return {
        "status": status,
        "exit_code": 75 if paused else (3 if quarantined else 0),
        "completed": len(ordered),
        "selected": len(indexed),
        "quarantined": quarantined,
        "receipts": ordered,
        "max_in_flight": max_in_flight,
    }


def execute(
    *,
    mode: str,
    limit: int,
    root: Path,
    device: str,
    pipeline_factory: Callable[[], Callable],
    bindings: Mapping[str, str],
    membership: Sequence[Mapping[str, str]],
    stop_requested: Callable[[], bool] | None = None,
    threshold_manifest: Mapping[str, Any] | None = None,
    parallel: bool = False,
    scratch_base: Path | None = None,
) -> dict[str, Any]:
    """Run one bounded batch strictly inside the B1 PREPARE_RAW_T1 lease."""

    if mode not in ("smoke", "run"):
        raise RuntimeError("BRAINIAC_PREPROCESSING_MODE_INVALID")
    items = list(membership) if mode == "run" else list(membership)[: max(0, int(limit))]
    if not items:
        raise RuntimeError("BRAINIAC_PREPROCESSING_BATCH_EMPTY")
    if scratch_base is None:
        derived_root = ROOT / "data" / "derived" / "method_x_brainiac"
        try:
            Path(root).resolve(strict=False).relative_to(derived_root.resolve(strict=False))
            scratch_base = LOCAL_SCRATCH_ROOT
        except ValueError:
            scratch_base = None
    with brainiac_action_lease("PREPARE_RAW_T1"):
        if parallel:
            return run_orchestrated_batch(
                items,
                root,
                pipeline_factory=pipeline_factory,
                bindings=bindings,
                cpu_workers=2,
                gpu_workers=1,
                threshold_manifest=threshold_manifest,
                stop_requested=stop_requested,
                scratch_base=scratch_base,
            )
        return run_batch(
            items,
            root,
            pipeline_factory=pipeline_factory,
            bindings=bindings,
            stop_requested=stop_requested,
            threshold_manifest=threshold_manifest,
            scratch_base=scratch_base,
        )
