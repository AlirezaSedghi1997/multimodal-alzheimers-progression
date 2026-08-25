from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import h5py
import nibabel as nib
import numpy as np
import pandas as pd

from .atomic_io import atomic_write_json
from .contracts import sha256_mapping
from .image_survival_v3 import robust_pattern_normalize
from .pet_geometry import resample_to_reference
from .process_guard import WriterLock


class ImageCacheV3Error(RuntimeError):
    """Raised when a resumable pattern cache cannot preserve its contract."""


_REQUIRED_COLUMNS = {
    "patient_hash",
    "image_key",
    "image_storage_kind",
    "image_path_private",
    "h5_group",
    "source_image_sha256",
    "qc_status",
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _membership_digest(frame: pd.DataFrame) -> str:
    columns = sorted(_REQUIRED_COLUMNS)
    payload = (
        frame.loc[:, columns]
        .sort_values("patient_hash", kind="stable")
        .to_dict(orient="records")
    )
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_source(
    row: Mapping[str, Any],
    *,
    reference: nib.spatialimages.SpatialImage,
    source_handles: dict[Path, h5py.File],
) -> np.ndarray:
    kind = str(row["image_storage_kind"])
    source_path = Path(str(row["image_path_private"]))
    if kind == "HDF5_GROUP":
        if not source_path.is_file():
            raise ImageCacheV3Error("SOURCE_HDF5_MISSING")
        handle = source_handles.get(source_path)
        if handle is None:
            handle = h5py.File(source_path, "r")
            source_handles[source_path] = handle
        group_name = str(row["h5_group"])
        if group_name not in handle or "image" not in handle[group_name]:
            raise ImageCacheV3Error("SOURCE_HDF5_GROUP_MISSING")
        # Plan 010 provenance: verify group attr source_sha256 matches membership
        expected_source_sha = str(row.get("source_image_sha256", ""))
        if len(expected_source_sha) == 64:
            actual_group_sha = str(handle[group_name].attrs.get("source_sha256", ""))
            if actual_group_sha and actual_group_sha != expected_source_sha:
                raise ImageCacheV3Error("HDF5_GROUP_SOURCE_SHA_MISMATCH")
            # If group has no attr, fall back to file-level check; do not auto-repair
        value = np.asarray(handle[group_name]["image"], dtype=np.float32)
        if value.ndim != 4 or value.shape[0] != 1:
            raise ImageCacheV3Error("SOURCE_HDF5_IMAGE_SHAPE_INVALID")
        volume = value[0]
        if volume.shape != reference.shape:
            raise ImageCacheV3Error("SOURCE_HDF5_REFERENCE_SHAPE_MISMATCH")
        return volume
    if kind == "NIFTI_TRANSFORM_OUTPUT":
        if not source_path.is_file():
            raise ImageCacheV3Error("SOURCE_NIFTI_MISSING")
        expected = str(row.get("source_image_sha256", ""))
        if len(expected) != 64 or _sha256_file(source_path) != expected:
            raise ImageCacheV3Error("SOURCE_NIFTI_SHA256_MISMATCH")
        image = nib.load(source_path)
        if len(image.shape) != 3:
            raise ImageCacheV3Error("SOURCE_NIFTI_MUST_BE_3D")
        resampled = resample_to_reference(image, reference, order=1)
        return np.asarray(resampled.dataobj, dtype=np.float32)
    raise ImageCacheV3Error(f"UNSUPPORTED_IMAGE_STORAGE_KIND:{kind}")


def build_pattern_cache(
    *,
    membership: pd.DataFrame,
    reference_template: str | Path,
    output_cache: str | Path,
    state_path: str | Path,
    normalization: Mapping[str, Any],
    branch: str,
    progress_every: int = 5,
    readiness_run_contract_sha256: str | None = None,
    source_artifact_sha256: str | None = None,
) -> dict[str, Any]:
    missing = sorted(_REQUIRED_COLUMNS.difference(membership.columns))
    if missing:
        raise ImageCacheV3Error(f"CACHE_MEMBERSHIP_COLUMNS_MISSING:{missing}")
    if membership.empty or membership["patient_hash"].astype(str).duplicated().any():
        raise ImageCacheV3Error("CACHE_MEMBERSHIP_EMPTY_OR_DUPLICATE")
    if (membership["qc_status"].astype(str) != "PASS").any():
        raise ImageCacheV3Error("CACHE_MEMBERSHIP_CONTAINS_NONPASS_ROW")
    branch_name = str(branch).strip().lower()
    if branch_name not in {"amyloid", "tau"}:
        raise ImageCacheV3Error("CACHE_BRANCH_INVALID")
    reference_path = Path(reference_template)
    if not reference_path.is_file():
        raise ImageCacheV3Error("REFERENCE_TEMPLATE_MISSING")
    reference_sha = _sha256_file(reference_path)
    reference = nib.load(reference_path)
    reference_data = np.asarray(reference.dataobj, dtype=np.float32)
    brain_mask = np.isfinite(reference_data) & (reference_data > 0)
    if int(brain_mask.sum()) < 8:
        raise ImageCacheV3Error("REFERENCE_BRAIN_MASK_EMPTY")
    lower = float(normalization["lower_percentile"])
    upper = float(normalization["upper_percentile"])
    clip_values = tuple(float(value) for value in normalization["output_clip"])
    if len(clip_values) != 2 or not (0 <= lower < upper <= 100):
        raise ImageCacheV3Error("NORMALIZATION_CONTRACT_INVALID")
    # Provenance-bound contract (Plan 010): bind source artifact and readiness contract
    # if provided; otherwise fall back to membership-only for backward compat.
    contract_payload = {
        "schema_version": "3.0",
        "branch": branch_name,
        "membership_sha256": _membership_digest(membership),
        "reference_template_sha256": reference_sha,
        "reference_shape": list(reference.shape),
        "normalization": {
            "method": "ROBUST_WITHIN_BRAIN_MEDIAN_IQR",
            "lower_percentile": lower,
            "upper_percentile": upper,
            "output_clip": list(clip_values),
            "quantitative_scale_claim_allowed": False,
        },
        "dtype": "float16",
        "compression": {"name": "gzip", "level": 4, "shuffle": True},
    }
    if readiness_run_contract_sha256 is not None:
        contract_payload["readiness_run_contract_sha256"] = str(readiness_run_contract_sha256)
    if source_artifact_sha256 is not None:
        contract_payload["source_artifact_sha256"] = str(source_artifact_sha256)
    contract_sha = sha256_mapping(contract_payload)
    output_path = Path(output_cache)
    state = Path(state_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    source_handles: dict[Path, h5py.File] = {}
    newly_completed = 0
    resumed = 0
    failures: list[dict[str, str]] = []
    lock = WriterLock(
        output_path.with_suffix(output_path.suffix + ".writer.lock"),
        output_root=output_path.parent,
        run_contract_digest=contract_sha,
    )
    try:
        with lock, h5py.File(output_path, "a") as target:
            existing = str(target.attrs.get("run_contract_sha256", ""))
            if existing and existing != contract_sha:
                raise ImageCacheV3Error("PATTERN_CACHE_CONTRACT_MISMATCH")
            target.attrs["run_contract_sha256"] = contract_sha
            target.attrs["reference_template_sha256"] = reference_sha
            target.attrs["branch"] = branch_name
            target.attrs["input_interpretation"] = (
                "PATTERN_NORMALIZED_NOT_QUANTITATIVE_STANDARD"
            )
            target.attrs["quantitative_scale_claim_allowed"] = False
            for stale in [key for key in target.keys() if key.startswith("__tmp__")]:
                del target[stale]
            rows = membership.sort_values("patient_hash", kind="stable").to_dict(
                orient="records"
            )
            for position, row in enumerate(rows, start=1):
                patient = str(row["patient_hash"])
                if (
                    patient in target
                    and str(target[patient].attrs.get("status", "")) == "COMPLETE"
                    and str(target[patient].attrs.get("source_image_sha256", ""))
                    == str(row["source_image_sha256"])
                ):
                    resumed += 1
                    continue
                try:
                    volume = _load_source(
                        row, reference=reference, source_handles=source_handles
                    )
                except ImageCacheV3Error as exc:
                    # Fail closed for provenance mismatches, not just per-row failure
                    if "HDF5_GROUP_SOURCE_SHA_MISMATCH" in str(exc):
                        raise
                    failures.append(
                        {
                            "patient_hash": patient,
                            "reason": str(exc),
                        }
                    )
                    continue
                try:
                    if volume.shape != reference.shape or not np.isfinite(volume).all():
                        raise ImageCacheV3Error("CACHE_SOURCE_GEOMETRY_OR_FINITE_FAILURE")
                    normalized = robust_pattern_normalize(
                        volume,
                        brain_mask,
                        lower_percentile=lower,
                        upper_percentile=upper,
                        clip=clip_values,
                    )
                    temporary = f"__tmp__{patient}"
                    if temporary in target:
                        del target[temporary]
                    group = target.create_group(temporary)
                    group.create_dataset(
                        "image",
                        data=normalized[None].astype(np.float16),
                        dtype=np.float16,
                        compression="gzip",
                        compression_opts=4,
                        shuffle=True,
                        chunks=(
                            1,
                            *[min(32, int(size)) for size in normalized.shape],
                        ),
                    )
                    group.attrs["status"] = "COMPLETE"
                    group.attrs["image_key"] = str(row["image_key"])
                    group.attrs["source_image_sha256"] = str(
                        row["source_image_sha256"]
                    )
                    group.attrs["finite_fraction"] = 1.0
                    if patient in target:
                        del target[patient]
                    target.move(temporary, patient)
                    target.flush()
                    newly_completed += 1
                except Exception as exc:
                    failures.append(
                        {
                            "patient_hash": patient,
                            "reason": str(exc)
                            if isinstance(exc, ImageCacheV3Error)
                            else type(exc).__name__,
                        }
                    )
                if position % max(1, int(progress_every)) == 0 or position == len(rows):
                    atomic_write_json(
                        state,
                        {
                            "schema_version": "3.0",
                            "status": "RUNNING" if position < len(rows) else "ITERATION_COMPLETE",
                            "branch": branch_name,
                            "run_contract_sha256": contract_sha,
                            "processed_position": position,
                            "total": len(rows),
                            "completed_in_cache": sum(
                                1
                                for key in target.keys()
                                if not key.startswith("__tmp__")
                                and str(target[key].attrs.get("status", "")) == "COMPLETE"
                            ),
                            "failure_count": len(failures),
                        },
                    )
                    print(
                        json.dumps(
                            {
                                "event": "PATTERN_CACHE_PROGRESS",
                                "branch": branch_name,
                                "position": position,
                                "total": len(rows),
                                "newly_completed": newly_completed,
                                "resumed": resumed,
                                "failures": len(failures),
                            }
                        ),
                        flush=True,
                    )
            completed = sum(
                1
                for key in target.keys()
                if not key.startswith("__tmp__")
                and str(target[key].attrs.get("status", "")) == "COMPLETE"
            )
    finally:
        for handle in source_handles.values():
            handle.close()
    status = "COMPLETED" if completed == len(membership) and not failures else "BLOCKED"
    report = {
        "schema_version": "3.0",
        "status": status,
        "branch": branch_name,
        "run_contract_sha256": contract_sha,
        "membership_sha256": contract_payload["membership_sha256"],
        "reference_template_sha256": reference_sha,
        "requested_count": int(len(membership)),
        "completed_count": int(completed),
        "newly_completed_count": int(newly_completed),
        "resumed_existing_count": int(resumed),
        "failure_count": len(failures),
        "failures": failures,
        "cache_sha256": _sha256_file(output_path) if status == "COMPLETED" else None,
        "quantitative_scale_claim_allowed": False,
        "resume_policy": "PER_SUBJECT_ATOMIC_GROUP_COMMIT",
    }
    atomic_write_json(state, report)
    return report

