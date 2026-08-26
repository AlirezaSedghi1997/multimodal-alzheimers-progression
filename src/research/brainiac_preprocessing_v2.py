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
POLICY_PATH = ROOT / "governance" / "brainiac_preprocessing_r1_primary_v2_policy.json"

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


def assert_no_cnn_writer() -> None:
    """Block on live CNN writers; a provably stale battery lock is retained."""

    from src.training.method_x_autosupervisor import discover_method_x_processes

    active = discover_method_x_processes(exclude_pids=(os.getpid(),))
    if active:
        raise RuntimeError("BRAINIAC_CNN_WRITER_ACTIVE")
    from src.research.brainiac_action_guard import _recorded_lock_status

    for battery_dir in sorted((ROOT / "results" / "final_results").glob("*/")):
        status = _recorded_lock_status(
            battery_dir / ".battery.lock", role="battery_lock"
        )
        if status.get("liveness") == "LIVE":
            raise RuntimeError("BRAINIAC_BATTERY_LOCK_OWNER_ACTIVE")


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

    from scripts.build_mri_volume_cache import iter_t1_files
    from src.data.a4_schema import T1_NAME_PATTERN
    from src.utils.privacy import stable_hash
    import re

    if not Path(data_root).is_dir():
        raise RuntimeError("BRAINIAC_RAW_DATA_ROOT_UNAVAILABLE")
    pattern = re.compile(T1_NAME_PATTERN, re.IGNORECASE)
    index: dict[str, Path] = {}
    for candidate in iter_t1_files(Path(data_root)):
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
    fixed_bounds: Mapping[str, tuple[float, float]] | None = None,
) -> dict[str, Any]:
    if not metrics_by_sample:
        raise RuntimeError("BRAINIAC_THRESHOLD_MANIFEST_EMPTY")
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
    """Binary erosion; outside counts as foreground so tissue at the face survives."""

    import numpy as np
    from scipy.ndimage import binary_erosion

    return binary_erosion(
        np.asarray(mask, dtype=bool),
        structure=_get_6_connectivity(),
        iterations=int(iterations),
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


def _quarantine_move(path: Path, root: Path, sample_hash: str) -> Path | None:
    if not path.exists():
        return None
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    target_dir = Path(root) / "quarantine" / f"{stamp}-{sample_hash}"
    target_dir.mkdir(parents=True, exist_ok=True)
    os.replace(path, target_dir / path.name)
    return target_dir


def load_committed_receipt(sample_hash: str, root: Path) -> dict[str, Any]:
    receipt_path = Path(root) / "receipts" / f"{sample_hash}.json"
    if not receipt_path.is_file():
        raise RuntimeError("BRAINIAC_PREPROCESSING_NOT_COMMITTED")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if not isinstance(receipt, dict):
        raise RuntimeError("BRAINIAC_PREPROCESSING_RECEIPT_INVALID")
    if int(receipt.get("generation", 0)) < RECEIPT_GENERATION:
        raise RuntimeError("BRAINIAC_PREPROCESSING_GENERATION_ROLLBACK")
    try:
        validate_receipt(receipt, artifact_root=Path(root))
    except RuntimeError as exc:
        if "BINDING_MISMATCH" in str(exc):
            raise RuntimeError("BRAINIAC_PREPROCESSING_COMMITTED_MISMATCH") from exc
        raise
    committed = Path(root) / "journals" / sample_hash / "committed.json"
    if not committed.is_file():
        raise RuntimeError("BRAINIAC_PREPROCESSING_JOURNAL_MISSING_COMMIT")
    journal = json.loads(committed.read_text(encoding="utf-8"))
    if (
        journal.get("receipt_sha256") != receipt.get("integrity_sha256")
        or journal.get("output_sha256") != receipt.get("output_sha256")
        or journal.get("transform_sha256") != receipt.get("transform_sha256")
    ):
        raise RuntimeError("BRAINIAC_PREPROCESSING_JOURNAL_MISMATCH")
    verify_receipt_artifacts(receipt, artifact_root=Path(root))
    return receipt


def _reconcile_stale_sample_lock(lock_path: Path) -> None:
    """Remove a sample lock only with liveness proof (card Step 5).

    Dead owner PID -> provably stale, remove.  Live PID (or PID-reuse doubt)
    -> fail closed; never delete a lock we cannot disprove.
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
        lock_path.unlink()
        if lock_path.exists():
            raise RuntimeError("BRAINIAC_PREPROCESSING_SAMPLE_LOCK_RELEASE_FAILED")
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
        return load_committed_receipt(sample_hash, root)

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
        _reconcile_stale_sample_lock(lock_path)
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

    scratch = root / "scratch" / sample_hash
    if scratch.exists():
        shutil.rmtree(scratch)
    scratch.mkdir(parents=True, exist_ok=True)
    try:
        result = dict(pipeline(item, scratch))
        output_path = Path(result["output_path"])
        transform_path = Path(result["transform_path"])
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
            "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime()),
        }
        body.update(dict(bindings))
        receipt = seal_receipt(body)

        _atomic_write_json(
            journal_dir / "prepared.json",
            {
                "sample_hash": sample_hash,
                "output_sha256": receipt["output_sha256"],
                "transform_sha256": receipt["transform_sha256"],
                "status": "PREPARED",
            },
        )
        for source, target in (
            (output_path, output_target),
            (transform_path, transform_target),
        ):
            if target.exists():
                raise RuntimeError("BRAINIAC_PREPROCESSING_COMMIT_TARGET_EXISTS")
            os.replace(source, target)
            _fsync_file(target)
        _atomic_write_json(receipt_path, receipt)
        verify_receipt_artifacts(receipt, artifact_root=root)
        _atomic_write_json(
            journal_dir / "committed.json",
            {
                "sample_hash": sample_hash,
                "receipt_sha256": receipt["integrity_sha256"],
                "output_sha256": receipt["output_sha256"],
                "transform_sha256": receipt["transform_sha256"],
                "status": "COMMITTED",
            },
        )
        shutil.rmtree(scratch, ignore_errors=True)
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
) -> dict[str, Any]:
    receipts: list[dict[str, Any]] = []
    quarantined: list[str] = []
    paused = False
    for item in items:
        try:
            receipt = run_sample_transaction(
                item, root, pipeline=pipeline_factory(), bindings=bindings
            )
        except RuntimeError as exc:
            if "QUARANTINE_BOTH_WORSENED" in str(exc):
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
) -> list[dict[str, Any]]:
    """Bounded parallel execution whose receipts must equal sequential ones."""

    from concurrent.futures import ThreadPoolExecutor

    if cpu_workers < 1 or gpu_workers < 1:
        raise RuntimeError("BRAINIAC_ORCHESTRATION_WORKER_COUNT_INVALID")
    # CPU stage prepares scratch payloads; the single GPU lane serializes the
    # device-bound stage.  Callable pipelines execute as one stage on the GPU
    # lane while scratch preparation stays on the CPU lanes.
    results: list[dict[str, Any]] = []

    def process(item: Mapping[str, str]) -> dict[str, Any]:
        return run_sample_transaction(
            item, root, pipeline=pipeline_factory(), bindings=bindings
        )

    with ThreadPoolExecutor(max_workers=cpu_workers + gpu_workers) as pool:
        for receipt in pool.map(process, list(items)):
            results.append(receipt)
    return results


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
) -> dict[str, Any]:
    """Run one bounded batch strictly inside the B1 PREPARE_RAW_T1 lease."""

    if mode not in ("smoke", "run"):
        raise RuntimeError("BRAINIAC_PREPROCESSING_MODE_INVALID")
    items = list(membership) if mode == "run" else list(membership)[: max(0, int(limit))]
    if not items:
        raise RuntimeError("BRAINIAC_PREPROCESSING_BATCH_EMPTY")
    with brainiac_action_lease("PREPARE_RAW_T1"):
        return run_batch(
            items,
            root,
            pipeline_factory=pipeline_factory,
            bindings=bindings,
            stop_requested=stop_requested,
        )
