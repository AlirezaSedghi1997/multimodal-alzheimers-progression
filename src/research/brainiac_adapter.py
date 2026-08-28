"""Label-blind 128->96 adapter, geometry gate, strict load, and frozen probes.

Geometry is the reason this module is defensive. The locked project cache is
``spm_v3_mni_brain_128_isotropic``: 128 cubed at 1.5 mm after *non-linear* SPM
forward-deformation MNI normalization, giving a 192 mm field of view. BrainIAC's
published contract is 96 cubed after *rigid* MNI registration at 1 mm spacing.

Those do not reconcile by resizing.  The vendored implementation establishes a
96-cubed array shape, but without the authoritative template affine it does not
establish a 96 mm physical field of view:

* a direct 128->96 resize keeps the 192 mm field of view and lands at 2.0 mm
  effective spacing, not 1.0 mm;
* the upstream physical extent remains NOT_MEASURABLE until its template and
  output affine are verified, and is recorded as unresolved rather than guessed;
* non-linear normalization removes part of the atrophy signal that a rigid
  pipeline preserves, so the two are not interchangeable even at equal geometry.

The gate therefore reports a documented domain shift instead of silently
resampling. A failure stops the BrainIAC branch only; the main battery is
untouched.

Nothing here reads a label, and no raw identifier or path reaches an artifact.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from src.data.combined_mri_cache import METHOD_X_PHENOTYPE_FEATURES
from src.research.brainiac_provenance import UNRESOLVED, UPSTREAM_PINNED_COMMIT

ADAPTER_SCHEMA_VERSION = 2

# Locked project geometry (scripts/preprocess_mri_spm.py).
PROJECT_RESOLUTION = 128
PROJECT_VOXEL_MM = 1.5
PROJECT_REGISTRATION = "spm_forward_deformation_mni_nonlinear"

# BrainIAC published contract.
BRAINIAC_RESOLUTION = 96
BRAINIAC_DECLARED_VOXEL_MM = 1.0
BRAINIAC_REGISTRATION = "rigid_mni"
BRAINIAC_EMBEDDING_DIM = 768

GEOMETRY_SHIFT_STATUS = "GEOMETRY_DOMAIN_SHIFT_DOCUMENTED"
GEOMETRY_UNKNOWN_STATUS = "UNKNOWN_GEOMETRY_PENDING_VENDOR_TEMPLATE_AFFINE"

#: Never inputs to prediction or gating; audit side-channel only.
AUDIT_ONLY_FIELDS = frozenset(
    {"cohort", "cohort_id", "site", "scanner_id", "manufacturer", "field_strength"}
)

PROBE_C_GRID = (0.01, 0.1, 1.0, 10.0)

_CANONICAL_PREPROCESSING_STEPS = (
    "n4_bias_correction",
    "rigid_registration",
    "brain_extraction",
    "official_96_transform",
)

# These uncertainties limit interpretation but do not make a label-free,
# protocol-compatible internal execution unsafe.  They must never disappear
# from the claim record and they categorically prohibit canonical-reproduction,
# external-generalisation, or confirmatory novelty language.
_CLAIM_ONLY_R1A_BLOCKERS = frozenset(
    {
        "UNKNOWN_PRETRAINING_OVERLAP",
        "UPSTREAM_TEMPLATE_IDENTITY_UNKNOWN",
        "EXTERNAL_VALIDATION_UNAVAILABLE",
    }
)


def _is_sha256(value: Any) -> bool:
    text = str(value or "")
    return len(text) == 64 and all(ch in "0123456789abcdef" for ch in text.lower())


def _canonical_sha(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def classify_r1a_blockers(blockers: Sequence[str]) -> dict[str, Any]:
    """Separate execution-safety blockers from permanent claim limitations."""

    unique = list(dict.fromkeys(str(item) for item in blockers if str(item)))
    claim_blocks = [item for item in unique if item in _CLAIM_ONLY_R1A_BLOCKERS]
    execution_blocks = [item for item in unique if item not in _CLAIM_ONLY_R1A_BLOCKERS]
    internal_ready = not execution_blocks
    canonical_ready = internal_ready and not claim_blocks
    return {
        "status": (
            "PASS"
            if canonical_ready
            else "PASS_WITH_CLAIM_BLOCKS"
            if internal_ready
            else "BLOCKED"
        ),
        "internal_execution_ready": internal_ready,
        "canonical_reproduction_ready": canonical_ready,
        "claim_ceiling": (
            "CANONICAL_REPRODUCTION_TECHNICAL_EVIDENCE"
            if canonical_ready
            else "POST_HOC_INTERNAL_TECHNICAL_EVIDENCE"
        ),
        "execution_blockers": execution_blocks,
        "claim_blockers": claim_blocks,
    }


def validate_locked_split_registry(
    registry: Mapping[str, Any] | None, *, expected_sha256: str | None
) -> dict[str, Any]:
    """Validate an externally pinned subject-role registry."""

    if not isinstance(registry, Mapping) or not _is_sha256(expected_sha256):
        raise RuntimeError("SPLIT_REGISTRY_REQUIRED")
    record = dict(registry)
    body = {key: value for key, value in record.items() if key != "integrity_sha256"}
    if (
        record.get("schema_version") != 1
        or record.get("locked") is not True
        or record.get("integrity_sha256") != _canonical_sha(body)
        or record.get("integrity_sha256") != expected_sha256
    ):
        raise RuntimeError("SPLIT_REGISTRY_DIGEST_MISMATCH")
    if not _is_sha256(record.get("split_hash")) or not _is_sha256(
        record.get("group_hash")
    ):
        raise RuntimeError("SPLIT_REGISTRY_PROVENANCE_INVALID")
    subjects = record.get("subjects")
    if not isinstance(subjects, list) or not subjects:
        raise RuntimeError("SPLIT_REGISTRY_SUBJECTS_INVALID")
    allowed_roles = {
        "inner_train",
        "inner_validation",
        "oof_inner_training",
        "outer_test",
    }
    by_subject: dict[str, dict[str, Any]] = {}
    for row in subjects:
        if not isinstance(row, Mapping):
            raise RuntimeError("SPLIT_REGISTRY_SUBJECTS_INVALID")
        item = dict(row)
        subject_hash = str(item.get("subject_hash") or "")
        if (
            not _is_sha256(subject_hash)
            or subject_hash in by_subject
            or item.get("role") not in allowed_roles
            or not isinstance(item.get("real"), bool)
        ):
            raise RuntimeError("SPLIT_REGISTRY_SUBJECTS_INVALID")
        parent = item.get("parent_hash")
        if parent is not None and not _is_sha256(parent):
            raise RuntimeError("SPLIT_REGISTRY_PARENT_INVALID")
        by_subject[subject_hash] = item
    return {
        "status": "PASS",
        "integrity_sha256": str(expected_sha256),
        "split_hash": str(record["split_hash"]),
        "group_hash": str(record["group_hash"]),
        "subjects": by_subject,
    }


# --- Geometry gate ------------------------------------------------------------


@dataclass(frozen=True)
class GeometryReport:
    status: str
    source_fov_mm: float
    target_fov_mm: float | None
    effective_voxel_mm: float
    declared_target_voxel_mm: float
    voxel_ratio: float
    registration_match: bool
    findings: tuple[str, ...]

    @property
    def passed(self) -> bool:
        """True only when there are no unresolved geometric concerns."""
        return self.status == "PASS" and not self.findings


def evaluate_geometry(
    *,
    source_resolution: int = PROJECT_RESOLUTION,
    source_voxel_mm: float = PROJECT_VOXEL_MM,
    target_resolution: int = BRAINIAC_RESOLUTION,
    declared_target_voxel_mm: float = BRAINIAC_DECLARED_VOXEL_MM,
    source_registration: str = PROJECT_REGISTRATION,
    target_registration: str = BRAINIAC_REGISTRATION,
    target_geometry_verified: bool = False,
) -> GeometryReport:
    """Compare field of view, effective spacing, and registration family."""

    try:
        source_res = float(source_resolution)
        source_voxel = float(source_voxel_mm)
        target_res = float(target_resolution)
        target_voxel = float(declared_target_voxel_mm)
    except (TypeError, ValueError, OverflowError):
        source_res = source_voxel = target_res = target_voxel = float("nan")

    invalid_findings: list[str] = []
    for value, name in (
        (source_res, "SOURCE_RESOLUTION_INVALID"),
        (source_voxel, "SOURCE_VOXEL_MM_INVALID"),
        (target_res, "TARGET_RESOLUTION_INVALID"),
        (target_voxel, "TARGET_VOXEL_MM_INVALID"),
    ):
        if not np.isfinite(value) or value <= 0.0:
            invalid_findings.append(name)
    if not str(source_registration or "").strip():
        invalid_findings.append("SOURCE_REGISTRATION_UNKNOWN")
    if not str(target_registration or "").strip():
        invalid_findings.append("TARGET_REGISTRATION_UNKNOWN")

    source_fov = source_res * source_voxel
    # The pinned reference code proves an array resize to 96 cubed, not the
    # affine or physical field of view of its output. Multiplying array shape by
    # a prose spacing claim would turn an unresolved geometry into a false fact.
    target_fov = target_res * target_voxel if target_geometry_verified else None
    # A direct resize rescales voxels but preserves the physical extent.
    effective_voxel = source_fov / target_res if target_res > 0.0 else float("nan")

    if invalid_findings:
        return GeometryReport(
            status="UNKNOWN_GEOMETRY",
            source_fov_mm=source_fov,
            target_fov_mm=None,
            effective_voxel_mm=effective_voxel,
            declared_target_voxel_mm=target_voxel,
            voxel_ratio=(
                effective_voxel / target_voxel if target_voxel > 0.0 else float("nan")
            ),
            registration_match=False,
            findings=tuple(invalid_findings),
        )

    findings: list[str] = []
    if abs(effective_voxel - declared_target_voxel_mm) > 1e-6:
        findings.append("EFFECTIVE_SPACING_DIFFERS_FROM_DECLARED")
    if target_fov is None:
        findings.append("TARGET_PHYSICAL_FOV_NOT_MEASURABLE")
    elif abs(source_fov - target_fov) > 1e-6:
        findings.append("FIELD_OF_VIEW_MISMATCH")
    registration_match = str(source_registration) == str(target_registration)
    if not registration_match:
        findings.append("REGISTRATION_FAMILY_MISMATCH")

    if target_geometry_verified is not True:
        findings.append("VENDOR_TEMPLATE_AFFINE_UNVERIFIED")
        status = GEOMETRY_UNKNOWN_STATUS
    else:
        status = GEOMETRY_SHIFT_STATUS if findings else "PASS"
    return GeometryReport(
        status=status,
        source_fov_mm=source_fov,
        target_fov_mm=target_fov,
        effective_voxel_mm=effective_voxel,
        declared_target_voxel_mm=target_voxel,
        voxel_ratio=effective_voxel / target_voxel,
        registration_match=registration_match,
        findings=tuple(findings),
    )


def geometry_manifest() -> dict[str, Any]:
    """Serializable record of the documented shift, for the R1 gate artifact."""

    report = evaluate_geometry()
    return {
        "schema_version": ADAPTER_SCHEMA_VERSION,
        "status": report.status,
        "source": {
            "resolution": PROJECT_RESOLUTION,
            "voxel_mm": PROJECT_VOXEL_MM,
            "fov_mm": report.source_fov_mm,
            "registration": PROJECT_REGISTRATION,
        },
        "target": {
            "resolution": BRAINIAC_RESOLUTION,
            "array_shape": [BRAINIAC_RESOLUTION] * 3,
            "declared_voxel_mm": BRAINIAC_DECLARED_VOXEL_MM,
            "declared_fov_mm": report.target_fov_mm,
            "physical_fov_status": "NOT_MEASURABLE",
            "registration": BRAINIAC_REGISTRATION,
            "fov_interpretation": UNRESOLVED,
        },
        "direct_resize": {
            "effective_voxel_mm": report.effective_voxel_mm,
            "voxel_ratio_vs_declared": report.voxel_ratio,
            "preserves_source_fov": True,
        },
        "findings": list(report.findings),
        "target_geometry_verified": False,
        "vendor_observed": {
            "snapshot_commit": UPSTREAM_PINNED_COMMIT,
            "rigid_euler3d": True,
            "n4_bias_correction": True,
            "hd_bet_brain_extraction": True,
            "resize_to_96_after_registration": True,
            "normalize_intensity_nonzero_channel_wise": True,
            "template_nifti_in_snapshot": False,
        },
        "consequence": (
            "Canonical BrainIAC geometry cannot be compared until the pinned "
            "vendor template and output affine are inspected. The direct "
            "128-to-96 route remains sensitivity-only."
        ),
        "branch_scope": "BRAINIAC_ONLY_MAIN_BATTERY_UNAFFECTED",
    }


# --- Label-blind adapter ------------------------------------------------------


def _resize_trilinear(volume: np.ndarray, target: int) -> np.ndarray:
    """Single-stage trilinear resize to a cubic target."""

    import torch

    tensor = torch.as_tensor(np.asarray(volume, dtype=np.float32))[None, None]
    resized = torch.nn.functional.interpolate(
        tensor, size=(target, target, target), mode="trilinear", align_corners=False
    )
    return resized[0, 0].numpy()


def nonzero_normalize(volume: np.ndarray) -> np.ndarray:
    """Official nonzero normalization: statistics from foreground voxels only."""

    array = np.asarray(volume, dtype=np.float32)
    mask = array != 0
    if not mask.any():
        return np.zeros_like(array)
    values = array[mask]
    mean = float(values.mean())
    std = float(values.std())
    if std <= 0.0:
        return np.zeros_like(array)
    out = np.zeros_like(array)
    out[mask] = (array[mask] - mean) / std
    return out


def adapt_volume(
    volume: np.ndarray,
    *,
    target_resolution: int = BRAINIAC_RESOLUTION,
    purpose: str | None = None,
) -> np.ndarray:
    """Direct index-space resize, permitted only as a sensitivity arm."""

    if purpose != "sensitivity_only":
        raise RuntimeError("DIRECT_128_TO_96_SENSITIVITY_ONLY")
    array = np.asarray(volume, dtype=np.float32)
    if array.ndim != 3 or len(set(array.shape)) != 1:
        raise RuntimeError(f"ADAPTER_EXPECTS_CUBIC_VOLUME:{array.shape}")
    return nonzero_normalize(_resize_trilinear(array, int(target_resolution)))


def adapter_provenance(volume: np.ndarray, *, sample_hash: str) -> dict[str, Any]:
    """Hash-indexed provenance. No raw identifier or filesystem path is stored."""

    digest = hashlib.sha256(np.ascontiguousarray(volume, dtype=np.float32).tobytes())
    return {
        "schema_version": ADAPTER_SCHEMA_VERSION,
        "sample_hash": str(sample_hash),
        "output_sha256": digest.hexdigest(),
        "target_resolution": BRAINIAC_RESOLUTION,
        "normalization": "nonzero_channelwise_zscore",
        "resize": "single_stage_trilinear",
        "route_role": "SENSITIVITY_ONLY_NOT_CANONICAL_PREPROCESSING",
        "label_read": False,
        "raw_identifier_stored": False,
        "raw_path_stored": False,
        "geometry_status": GEOMETRY_UNKNOWN_STATUS,
    }


def sensitivity_pair(volume: np.ndarray) -> dict[str, Any]:
    """Direct resize versus a 1 mm-resample route, compared label-blind."""

    direct = adapt_volume(volume, purpose="sensitivity_only")
    # Intermediate 1 mm-equivalent grid, then down to the target.
    intermediate = int(round(PROJECT_RESOLUTION * PROJECT_VOXEL_MM))
    staged = nonzero_normalize(
        _resize_trilinear(_resize_trilinear(volume, intermediate), BRAINIAC_RESOLUTION)
    )
    difference = float(np.abs(direct - staged).mean())
    denominator = float(np.abs(direct).mean()) or 1.0
    return {
        "direct_resize_sha256": hashlib.sha256(direct.tobytes()).hexdigest(),
        "staged_resample_sha256": hashlib.sha256(staged.tobytes()).hexdigest(),
        "mean_absolute_difference": difference,
        "relative_difference": difference / denominator,
        "intermediate_resolution": intermediate,
        "label_read": False,
        "route_role": "SENSITIVITY_ONLY_NOT_PRIMARY",
    }


# --- Canonical raw-T1 preprocessing readiness --------------------------------


def validate_canonical_preprocessing(manifest: Mapping[str, Any] | None) -> dict[str, Any]:
    """Validate evidence for the canonical affine-aware raw-T1 route.

    This function validates provenance; it does not pretend to implement N4,
    registration, skull stripping, or the official BrainIAC transform.  Those
    operations must be run by a pinned local vendor environment and represented
    by hash-locked artifacts.  A direct project-cache 128->96 resize is allowed
    only in the sensitivity arm and therefore blocks a primary-route manifest.
    """

    blockers: list[str] = []
    record = dict(manifest or {})
    if not record:
        return {
            "schema_version": ADAPTER_SCHEMA_VERSION,
            "status": "UNKNOWN",
            "ready": False,
            "affine_aware": False,
            "primary_route": "raw_t1_n4_rigid_brain_extract_official_96",
            "blockers": ["CANONICAL_PREPROCESSING_MANIFEST_ABSENT"],
        }
    if record.get("source_role") != "raw_t1":
        blockers.append("CANONICAL_SOURCE_MUST_BE_RAW_T1")
    if record.get("label_read") is not False:
        blockers.append("CANONICAL_PREPROCESSING_MUST_BE_LABEL_FREE")
    if not _is_sha256(record.get("source_affine_sha256")):
        blockers.append("SOURCE_AFFINE_PROVENANCE_MISSING")
    if record.get("direct_128_to_96_used") is not False:
        blockers.append("DIRECT_128_TO_96_PRIMARY_ROUTE_FORBIDDEN")

    hash_fields = {
        "n4_bias_correction": "artifact_sha256",
        "rigid_registration": "transform_sha256",
        "brain_extraction": "mask_sha256",
        "official_96_transform": "output_affine_sha256",
    }
    for step in _CANONICAL_PREPROCESSING_STEPS:
        evidence = record.get(step)
        if not isinstance(evidence, Mapping):
            blockers.append(f"{step.upper()}_EVIDENCE_MISSING")
            continue
        if evidence.get("status") != "PASS":
            blockers.append(f"{step.upper()}_NOT_PASS")
        if not _is_sha256(evidence.get(hash_fields[step])):
            blockers.append(f"{step.upper()}_HASH_MISSING")
    registration = record.get("rigid_registration")
    if isinstance(registration, Mapping) and not _is_sha256(
        registration.get("template_sha256")
    ):
        blockers.append("OFFICIAL_TEMPLATE_HASH_MISSING")
    official = record.get("official_96_transform")
    if isinstance(official, Mapping) and tuple(official.get("output_shape", ())) != (
        96,
        96,
        96,
    ):
        blockers.append("OFFICIAL_OUTPUT_SHAPE_INVALID")

    status = "PASS" if not blockers else "BLOCKED"
    return {
        "schema_version": ADAPTER_SCHEMA_VERSION,
        "status": status,
        "ready": not blockers,
        "affine_aware": not blockers,
        "primary_route": "raw_t1_n4_rigid_brain_extract_official_96",
        "direct_128_to_96_role": "SENSITIVITY_ONLY",
        "blockers": blockers,
    }


def _path_has_reparse_between(path: Path, root: Path) -> bool:
    current = path
    while True:
        try:
            stat_result = os.lstat(current)
        except OSError:
            return True
        attributes = int(getattr(stat_result, "st_file_attributes", 0))
        if current.is_symlink() or attributes & 0x400:
            return True
        if current == root:
            return False
        if current.parent == current:
            return True
        current = current.parent


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


_SOURCE_TREE_IGNORED_DIRECTORIES = frozenset(
    {".git", ".hg", ".svn", "__pycache__", ".pytest_cache"}
)
_SOURCE_TREE_IGNORED_SUFFIXES = frozenset({".pyc", ".pyo"})


def stable_source_tree_sha256(root: Path) -> str:
    """Hash material vendor files while excluding mutable runtime/VCS noise.

    Python imports may create ``__pycache__`` files and Git may mutate its
    private metadata without changing the checked-out source.  Neither is part
    of the scientific implementation identity.  Reparse points are rejected so
    that a receipt cannot silently hash content outside the pinned tree.
    """

    try:
        resolved_root = root.resolve(strict=True)
    except OSError as exc:
        raise RuntimeError("BRAINIAC_VENDOR_SOURCE_TREE_MISSING") from exc
    if not resolved_root.is_dir() or _path_has_reparse_between(
        resolved_root, resolved_root
    ):
        raise RuntimeError("BRAINIAC_VENDOR_SOURCE_TREE_UNSAFE")

    digest = hashlib.sha256()
    files: list[Path] = []
    for directory, directory_names, file_names in os.walk(
        resolved_root, topdown=True, followlinks=False
    ):
        directory_path = Path(directory)
        kept_directories: list[str] = []
        for name in sorted(directory_names, key=str.casefold):
            if name in _SOURCE_TREE_IGNORED_DIRECTORIES:
                continue
            candidate = directory_path / name
            stat_result = os.lstat(candidate)
            attributes = int(getattr(stat_result, "st_file_attributes", 0))
            if candidate.is_symlink() or attributes & 0x400:
                raise RuntimeError("BRAINIAC_VENDOR_SOURCE_TREE_REPARSE_REJECTED")
            kept_directories.append(name)
        directory_names[:] = kept_directories
        for name in sorted(file_names, key=str.casefold):
            candidate = directory_path / name
            if candidate.suffix.casefold() in _SOURCE_TREE_IGNORED_SUFFIXES:
                continue
            stat_result = os.lstat(candidate)
            attributes = int(getattr(stat_result, "st_file_attributes", 0))
            if candidate.is_symlink() or attributes & 0x400:
                raise RuntimeError("BRAINIAC_VENDOR_SOURCE_TREE_REPARSE_REJECTED")
            if candidate.is_file():
                files.append(candidate)
    files.sort(key=lambda path: path.relative_to(resolved_root).as_posix().casefold())
    if not files:
        return ""
    for path in files:
        relative = path.relative_to(resolved_root).as_posix().encode("utf-8")
        digest.update(relative)
        digest.update(b"\0")
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def _source_tree_sha256(root: Path) -> str:
    """Compatibility alias for existing receipt validation code."""

    return stable_source_tree_sha256(root)


def _validate_vendor_h_locality(vendor: Mapping[str, Any]) -> bool:
    paths = vendor.get("h_local_paths")
    if not isinstance(paths, Mapping):
        return False
    project_root = Path("H:/New project")
    expected_vendor = project_root / "vendor" / "BrainIAC" / UPSTREAM_PINNED_COMMIT
    try:
        resolved_project = project_root.resolve(strict=True)
        resolved_vendor = Path(str(paths.get("vendor_root") or "")).resolve(strict=True)
        expected_resolved = expected_vendor.resolve(strict=True)
    except OSError:
        return False
    if resolved_vendor != expected_resolved or _path_has_reparse_between(
        resolved_vendor, resolved_project
    ):
        return False
    source_root = Path(str(paths.get("source_root") or ""))
    file_bindings = {
        "license_path": "license_sha256",
        "checkpoint_path": "weight_sha256",
        "template_path": "template_sha256",
        "environment_lock_path": "environment_lock_sha256",
        "interpreter_path": "interpreter_sha256",
    }
    try:
        resolved_source = source_root.resolve(strict=True)
        resolved_source.relative_to(resolved_vendor)
    except (OSError, ValueError):
        return False
    if not resolved_source.is_dir() or _path_has_reparse_between(
        resolved_source, resolved_project
    ):
        return False
    for path_field, digest_field in file_bindings.items():
        candidate = Path(str(paths.get(path_field) or ""))
        try:
            resolved = candidate.resolve(strict=True)
            resolved.relative_to(resolved_project)
        except (OSError, ValueError):
            return False
        if (
            not resolved.is_file()
            or _path_has_reparse_between(resolved, resolved_project)
            or _sha256_file(resolved) != vendor.get(digest_field)
        ):
            return False
    return _source_tree_sha256(resolved_source) == vendor.get("source_tree_sha256")


def assess_r1a_readiness(
    *,
    vendor_manifest: Mapping[str, Any] | None,
    preprocessing_manifest: Mapping[str, Any] | None,
    strict_load_report: Mapping[str, Any] | None,
    forward_report: Mapping[str, Any] | None,
    overlap_status: str,
    overlap_evidence: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Fail-closed R1a readiness assessment without fabricating vendor assets."""

    vendor = dict(vendor_manifest or {})
    blockers: list[str] = []
    unknown = False
    if vendor.get("vendored") is not True:
        blockers.append("OFFICIAL_VENDOR_SOURCE_ABSENT")
        unknown = True
    if not _is_sha256(vendor.get("license_sha256")):
        blockers.append("OFFICIAL_LICENSE_LOCK_ABSENT")
        unknown = True
    if not _is_sha256(vendor.get("weight_sha256")):
        blockers.append("OFFICIAL_CHECKPOINT_ABSENT")
        unknown = True
    if not _is_sha256(vendor.get("template_sha256")):
        blockers.append("OFFICIAL_TEMPLATE_ABSENT")
        unknown = True
    if vendor.get("commit") != UPSTREAM_PINNED_COMMIT:
        blockers.append("PINNED_VENDOR_COMMIT_MISSING")
        unknown = True
    if not _is_sha256(vendor.get("source_tree_sha256")):
        blockers.append("VENDOR_SOURCE_TREE_HASH_MISSING")
        unknown = True
    if not _is_sha256(vendor.get("environment_lock_sha256")) or not _is_sha256(
        vendor.get("interpreter_sha256")
    ):
        blockers.append("VENDOR_ENVIRONMENT_LOCK_MISSING")
        unknown = True
    module_fields = (
        vendor.get("encoder_module_path"),
        vendor.get("encoder_class_name"),
        vendor.get("loader_symbol"),
    )
    module_body = {
        "commit": vendor.get("commit"),
        "source_tree_sha256": vendor.get("source_tree_sha256"),
        "encoder_module_path": module_fields[0],
        "encoder_class_name": module_fields[1],
        "loader_symbol": module_fields[2],
    }
    if (
        any(not str(value or "").strip() or value == UNRESOLVED for value in module_fields)
        or vendor.get("module_identity_sha256") != _canonical_sha(module_body)
    ):
        blockers.append("VENDOR_MODULE_IDENTITY_MISSING")
        unknown = True
    if not _validate_vendor_h_locality(vendor):
        blockers.append("VENDOR_H_LOCAL_PATH_EVIDENCE_MISSING")
        unknown = True

    preprocessing = validate_canonical_preprocessing(preprocessing_manifest)
    if preprocessing["status"] != "PASS":
        blockers.extend(str(item) for item in preprocessing["blockers"])
        unknown = unknown or preprocessing["status"] == "UNKNOWN"
    registration = dict((preprocessing_manifest or {}).get("rigid_registration") or {})
    if registration.get("template_sha256") != vendor.get("template_sha256"):
        blockers.append("PREPROCESSING_TEMPLATE_VENDOR_BINDING_MISMATCH")

    strict = dict(strict_load_report or {})
    strict_body = {key: value for key, value in strict.items() if key != "integrity_sha256"}
    strict_bound = (
        strict.get("receipt_type") == "ACTUAL_STRICT_LOAD"
        and strict.get("status") == "PASS"
        and strict.get("strict") is True
        and isinstance(strict.get("matched_keys"), int)
        and not isinstance(strict.get("matched_keys"), bool)
        and int(strict.get("matched_keys", 0)) > 0
        and strict.get("missing_keys") == 0
        and strict.get("unexpected_keys") == 0
        and strict.get("shape_key_count") == strict.get("matched_keys")
        and strict.get("checkpoint_sha256") == vendor.get("weight_sha256")
        and strict.get("source_tree_sha256") == vendor.get("source_tree_sha256")
        and strict.get("environment_lock_sha256")
        == vendor.get("environment_lock_sha256")
        and strict.get("module_identity_sha256")
        == vendor.get("module_identity_sha256")
        and _is_sha256(strict.get("model_keyset_sha256"))
        and _is_sha256(strict.get("shape_map_sha256"))
        and strict.get("integrity_sha256") == _canonical_sha(strict_body)
    )
    if not strict_load_report or strict_load_report.get("strict") is not True:
        blockers.append("STRICT_LOAD_NOT_PROVEN")
        unknown = True
    elif strict_load_report.get("matched_keys", 0) <= 0:
        blockers.append("STRICT_LOAD_EMPTY_STATE_DICT")
    if not strict_bound:
        blockers.append("STRICT_LOAD_RECEIPT_UNBOUND")
        unknown = True

    forward = dict(forward_report or {})
    forward_body = {
        key: value for key, value in forward.items() if key != "integrity_sha256"
    }
    forward_bound = (
        forward.get("receipt_type") == "ACTUAL_FORWARD"
        and forward.get("status") == "PASS"
        and forward.get("finite") is True
        and forward.get("checkpoint_sha256") == vendor.get("weight_sha256")
        and forward.get("source_tree_sha256") == vendor.get("source_tree_sha256")
        and forward.get("environment_lock_sha256")
        == vendor.get("environment_lock_sha256")
        and forward.get("module_identity_sha256")
        == vendor.get("module_identity_sha256")
        and forward.get("strict_load_receipt_sha256") == strict.get("integrity_sha256")
        and _is_sha256(forward.get("output_sha256"))
        and forward.get("integrity_sha256") == _canonical_sha(forward_body)
    )
    if not forward_report or forward_report.get("status") != "PASS":
        blockers.append("OFFICIAL_FORWARD_NOT_PROVEN")
        unknown = True
    elif tuple(forward_report.get("input_shape", ())) != (1, 1, 96, 96, 96) or tuple(
        forward_report.get("output_shape", ())
    ) != (1, BRAINIAC_EMBEDDING_DIM):
        blockers.append("OFFICIAL_FORWARD_SHAPE_INVALID")
    if not forward_bound:
        blockers.append("FORWARD_RECEIPT_UNBOUND")
        unknown = True
    if overlap_status != "NO_PRETRAINING_OVERLAP_PROVEN":
        blockers.append(str(overlap_status or "UNKNOWN_PRETRAINING_OVERLAP"))
        unknown = True
    overlap = dict(overlap_evidence or {})
    overlap_body = {
        key: value for key, value in overlap.items() if key != "integrity_sha256"
    }
    overlap_bound = (
        overlap.get("status") == "PASS"
        and overlap.get("overlap_status") == "NO_PRETRAINING_OVERLAP_PROVEN"
        and overlap.get("intersection_count") == 0
        and _is_sha256(overlap.get("project_subject_set_sha256"))
        and _is_sha256(overlap.get("pretraining_subject_set_sha256"))
        and overlap.get("integrity_sha256") == _canonical_sha(overlap_body)
    )
    if overlap_status == "NO_PRETRAINING_OVERLAP_PROVEN" and not overlap_bound:
        blockers.append("OVERLAP_EVIDENCE_RECEIPT_MISSING")
        unknown = True

    status = "PASS" if not blockers else ("UNKNOWN" if unknown else "BLOCKED")
    return {
        "schema_version": ADAPTER_SCHEMA_VERSION,
        "status": status,
        "ready": status == "PASS",
        "blockers": list(dict.fromkeys(blockers)),
        "preprocessing": preprocessing,
        "branch_scope": "BRAINIAC_ONLY_MAIN_BATTERY_UNAFFECTED",
    }


def _shift_no_wrap(mask: np.ndarray, offset: int, axis: int) -> np.ndarray:
    """Shift a boolean mask along one axis without wrapping around the volume.

    ``np.roll`` wraps, so a voxel at the inferior edge reappears at the superior
    edge. In a mask-edge probe that fabricates anatomy at the opposite boundary
    and corrupts the very signal being measured. Vacated positions are filled
    with ``False`` (background) instead.
    """

    shifted = np.zeros_like(mask, dtype=bool)
    source = [slice(None)] * mask.ndim
    target = [slice(None)] * mask.ndim
    if offset > 0:
        source[axis] = slice(None, -offset)
        target[axis] = slice(offset, None)
    elif offset < 0:
        source[axis] = slice(-offset, None)
        target[axis] = slice(None, offset)
    else:
        return mask.copy()
    shifted[tuple(target)] = mask[tuple(source)]
    return shifted


def mask_edge_perturbation(
    volume: np.ndarray, *, mode: str, voxels: int = 1
) -> np.ndarray:
    """Dilate, erode, or zero the foreground to probe skull-strip shortcuts."""

    array = np.asarray(volume, dtype=np.float32)
    mask = array != 0
    if mode == "erode":
        shifted = mask.copy()
        for axis in range(3):
            for offset in (voxels, -voxels):
                shifted &= _shift_no_wrap(mask, offset, axis)
        return np.where(shifted, array, 0.0).astype(np.float32)
    if mode == "dilate":
        grown = mask.copy()
        for axis in range(3):
            for offset in (voxels, -voxels):
                grown |= _shift_no_wrap(mask, offset, axis)
        filled = array.copy()
        filled[grown & ~mask] = float(array[mask].mean()) if mask.any() else 0.0
        return filled.astype(np.float32)
    if mode == "foreground_only":
        return np.where(mask, array, 0.0).astype(np.float32)
    raise RuntimeError(f"UNKNOWN_MASK_PERTURBATION:{mode}")


# --- Strict checkpoint load ---------------------------------------------------


def validate_strict_load(
    checkpoint_keys: Iterable[str],
    model_keys: Iterable[str],
    *,
    shapes: Mapping[str, Sequence[int]] | None = None,
    expected_shapes: Mapping[str, Sequence[int]] | None = None,
) -> dict[str, Any]:
    """Require 100% key coverage and exact shape agreement."""

    checkpoint = set(checkpoint_keys)
    model = set(model_keys)
    # Two empty sets trivially satisfy "no missing, no unexpected" and would
    # report 100% coverage for a checkpoint that loaded nothing at all.
    if not checkpoint or not model:
        raise RuntimeError("STRICT_LOAD_EMPTY_STATE_DICT")
    missing = sorted(model - checkpoint)
    unexpected = sorted(checkpoint - model)
    if missing:
        raise RuntimeError(f"STRICT_LOAD_MISSING_KEYS:{len(missing)}")
    if unexpected:
        raise RuntimeError(f"STRICT_LOAD_UNEXPECTED_KEYS:{len(unexpected)}")
    if (shapes is None) != (expected_shapes is None):
        raise RuntimeError("STRICT_LOAD_SHAPE_EVIDENCE_INCOMPLETE")
    if shapes is not None and expected_shapes is not None:
        if not shapes or not expected_shapes:
            raise RuntimeError("STRICT_LOAD_SHAPE_EVIDENCE_EMPTY")
        if set(shapes) != model or set(expected_shapes) != model:
            raise RuntimeError("STRICT_LOAD_SHAPE_KEYSET_MISMATCH")
        for key, expected in expected_shapes.items():
            actual = shapes.get(key)
            if actual is None or tuple(actual) != tuple(expected):
                raise RuntimeError(f"STRICT_LOAD_SHAPE_MISMATCH:{key}")
    return {
        "coverage": 1.0,
        "matched_keys": len(model),
        "missing_keys": 0,
        "unexpected_keys": 0,
        "strict": True,
    }


MONAI15_UNUSED_CROSS_ATTN_SUBSTRING = "cross_attn"
VENDOR_EQUIVALENT_STRICT_STATUS = "PASS_VENDOR_EQUIVALENT_STRICT"
NAIVE_BATTERY_VIT_STRICT_FAIL_REASON = (
    "MONAI_1_5_2_TRANSFORMERBLOCK_INSTANTIATES_UNUSED_CROSS_ATTN"
)


def partition_vendor_equivalent_vit_keys(
    model_keys: Iterable[str],
) -> dict[str, Any]:
    """Split a battery-venv ViT keyset from the official BrainIAC encoder keyset.

    The vendored pin is ``monai==1.3.2``. The locked battery venv is MONAI 1.5.2.
    ``TransformerBlock`` in 1.5.2 always constructs unused ``cross_attn`` /
    ``norm_cross_attn`` parameters even when ``with_cross_attention=False``.
    Official ``BrainIAC.ckpt`` has none of those keys. Naive ``strict=True``
    against the battery ViT is therefore a documented FAIL, not a corrupt
    checkpoint. Vendor-equivalent load drops the unused keys, then requires
    100% coverage of the remainder.
    """

    model = [str(key) for key in model_keys]
    unused = sorted(key for key in model if MONAI15_UNUSED_CROSS_ATTN_SUBSTRING in key)
    vendor_keys = [key for key in model if MONAI15_UNUSED_CROSS_ATTN_SUBSTRING not in key]
    return {
        "unused_cross_attn_keys": unused,
        "vendor_equivalent_model_keys": vendor_keys,
        "naive_strict_against_battery_vit": "FAIL" if unused else "PASS",
        "reason": NAIVE_BATTERY_VIT_STRICT_FAIL_REASON if unused else "KEYS_ALREADY_VENDOR_EQUIVALENT",
    }


def _filter_shape_map(
    mapping: Mapping[str, Sequence[int]] | None,
    allowed: set[str],
) -> dict[str, tuple[int, ...]] | None:
    if mapping is None:
        return None
    return {str(key): tuple(value) for key, value in mapping.items() if str(key) in allowed}


def validate_vendor_equivalent_strict_load(
    checkpoint_keys: Iterable[str],
    model_keys: Iterable[str],
    *,
    shapes: Mapping[str, Sequence[int]] | None = None,
    expected_shapes: Mapping[str, Sequence[int]] | None = None,
) -> dict[str, Any]:
    """Strict-load the official encoder against a possibly newer MONAI ViT.

    Never rounds a naive battery-ViT mismatch up to ``PASS``. The returned
    ``status`` is ``PASS_VENDOR_EQUIVALENT_STRICT`` only after unused
    ``cross_attn`` keys are excluded and ``validate_strict_load`` succeeds on
    the remainder.
    """

    checkpoint = [str(key) for key in checkpoint_keys]
    leaked_cross_attn = sorted(
        key for key in checkpoint if MONAI15_UNUSED_CROSS_ATTN_SUBSTRING in key
    )
    if leaked_cross_attn:
        raise RuntimeError(
            f"CHECKPOINT_CONTAINS_UNUSED_CROSS_ATTN:{len(leaked_cross_attn)}"
        )
    partition = partition_vendor_equivalent_vit_keys(model_keys)
    allowed = set(partition["vendor_equivalent_model_keys"])
    result = validate_strict_load(
        checkpoint,
        partition["vendor_equivalent_model_keys"],
        shapes=_filter_shape_map(shapes, allowed),
        expected_shapes=_filter_shape_map(expected_shapes, allowed),
    )
    result["status"] = VENDOR_EQUIVALENT_STRICT_STATUS
    result["naive_strict_status"] = partition["naive_strict_against_battery_vit"]
    result["naive_strict_reason"] = partition["reason"]
    result["battery_monai_unused_cross_attn_keys"] = len(
        partition["unused_cross_attn_keys"]
    )
    return result


# --- Audit / predictive separation --------------------------------------------


def assert_predictive_feature_order(names: Sequence[str]) -> None:
    """The five allowed features, in canonical order, and nothing else."""

    if tuple(names) != METHOD_X_PHENOTYPE_FEATURES:
        raise RuntimeError(f"FEATURE_ORDER_VIOLATION:{tuple(names)}")


def assert_no_audit_fields_in_predictors(names: Iterable[str]) -> None:
    """Cohort/scanner metadata must never reach a predictive tensor."""

    leaked = sorted(AUDIT_ONLY_FIELDS.intersection({str(n).lower() for n in names}))
    if leaked:
        raise RuntimeError(f"AUDIT_FIELD_IN_PREDICTORS:{','.join(leaked)}")


def audit_metadata_status(metadata: Mapping[str, Any] | None) -> dict[str, Any]:
    """Missing metadata is NOT_MEASURABLE, never a silent pass."""

    if not metadata:
        return {"status": "NOT_MEASURABLE", "available_fields": []}
    available = sorted(AUDIT_ONLY_FIELDS.intersection(metadata))
    return {
        "status": "MEASURABLE" if available else "NOT_MEASURABLE",
        "available_fields": available,
        "audit_only": True,
    }


# --- Frozen probes ------------------------------------------------------------

def fit_frozen_probe(
    embeddings: np.ndarray,
    labels: np.ndarray,
    *,
    inner_train: Sequence[int],
    inner_validation: Sequence[int],
    tabular: np.ndarray | None = None,
    c_grid: Sequence[float] = PROBE_C_GRID,
    subject_hashes: Sequence[str],
    split_roles: Sequence[str],
    real_flags: Sequence[bool],
    parent_hashes: Sequence[str | None],
    feature_names: Sequence[str],
    split_provenance: Mapping[str, Any],
    locked_split_registry: Mapping[str, Any] | None = None,
    expected_split_registry_sha256: str | None = None,
    probe_kind: str = "primary_real_only",
) -> dict[str, Any]:
    """L2 probe with fit-only transforms and hash-locked split provenance.

    ``primary_real_only`` refuses every derivative.  The optional
    ``augmented_fit_sensitivity`` permits derivatives in inner train only when
    each has a real inner-train parent; validation remains real-only.  The outer
    test set is never accepted as a selection role.
    """

    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    features = np.asarray(embeddings, dtype=np.float64)
    if features.ndim != 2:
        raise RuntimeError("PROBE_EMBEDDINGS_MUST_BE_2D")
    n = int(features.shape[0])
    y_raw = np.asarray(labels)
    if y_raw.ndim != 1 or y_raw.size != n:
        raise RuntimeError("PROBE_LABELS_MISALIGNED")
    if not np.all(np.isin(y_raw, (0, 1))):
        raise RuntimeError("PROBE_LABEL_INVALID")
    y = y_raw.astype(int)

    sequences = (subject_hashes, split_roles, real_flags, parent_hashes)
    if any(len(values) != n for values in sequences):
        raise RuntimeError("PROBE_LINEAGE_MISALIGNED")
    if any(not _is_sha256(value) for value in subject_hashes):
        raise RuntimeError("PROBE_SUBJECT_HASH_INVALID")
    if len(set(str(value) for value in subject_hashes)) != n:
        raise RuntimeError("PROBE_SUBJECT_OVERLAP")
    if any(str(role) == "outer_test" for role in split_roles):
        raise RuntimeError("PROBE_OUTER_ROWS_PRESENT")
    if any(
        str(role) not in {"inner_train", "inner_validation"}
        for role in split_roles
    ):
        raise RuntimeError("PROBE_SPLIT_ROLE_INVALID")
    if locked_split_registry is None or expected_split_registry_sha256 is None:
        raise RuntimeError("PROBE_SPLIT_REGISTRY_REQUIRED")
    try:
        registry = validate_locked_split_registry(
            locked_split_registry,
            expected_sha256=expected_split_registry_sha256,
        )
    except RuntimeError as exc:
        if "DIGEST_MISMATCH" in str(exc):
            raise RuntimeError("PROBE_SPLIT_REGISTRY_DIGEST_MISMATCH") from exc
        raise RuntimeError(f"PROBE_{exc}") from exc
    provenance = dict(split_provenance or {})
    if (
        provenance.get("split_registry_sha256")
        != registry["integrity_sha256"]
        or provenance.get("split_hash") != registry["split_hash"]
        or provenance.get("group_hash") != registry["group_hash"]
    ):
        raise RuntimeError("PROBE_SPLIT_REGISTRY_PROVENANCE_MISMATCH")
    registry_subjects = registry["subjects"]
    for index, subject_hash in enumerate(subject_hashes):
        item = registry_subjects.get(str(subject_hash))
        if item is None:
            raise RuntimeError("PROBE_SUBJECT_NOT_IN_LOCKED_SPLIT_REGISTRY")
        if (
            item.get("role") != str(split_roles[index])
            or item.get("real") is not bool(real_flags[index])
            or item.get("parent_hash") != parent_hashes[index]
        ):
            raise RuntimeError("PROBE_SPLIT_REGISTRY_ROW_MISMATCH")
    train = tuple(int(i) for i in inner_train)
    validation = tuple(int(i) for i in inner_validation)
    if not train or not validation:
        raise RuntimeError("PROBE_SPLIT_EMPTY")
    if len(set(train)) != len(train) or len(set(validation)) != len(validation):
        raise RuntimeError("PROBE_DUPLICATE_INDEX")
    if set(train).intersection(validation):
        raise RuntimeError("PROBE_INDEX_OVERLAP")
    if min((*train, *validation)) < 0 or max((*train, *validation)) >= n:
        raise RuntimeError("PROBE_INDEX_OUT_OF_RANGE")
    if any(split_roles[i] != "inner_train" for i in train) or any(
        split_roles[i] != "inner_validation" for i in validation
    ):
        raise RuntimeError("PROBE_SPLIT_ROLE_MISMATCH")
    train_subjects = {str(subject_hashes[i]) for i in train}
    validation_subjects = {str(subject_hashes[i]) for i in validation}
    outer_subjects = {
        str(subject_hashes[index])
        for index, role in enumerate(split_roles)
        if role == "outer_test"
    }
    if (
        train_subjects.intersection(validation_subjects)
        or train_subjects.intersection(outer_subjects)
        or validation_subjects.intersection(outer_subjects)
    ):
        raise RuntimeError("PROBE_SUBJECT_OVERLAP")
    if len(validation_subjects) != len(validation):
        raise RuntimeError("PROBE_SUBJECT_DUPLICATE_WITHIN_VALIDATION")
    if not all(bool(real_flags[i]) for i in validation):
        raise RuntimeError("PROBE_VALIDATION_MUST_BE_REAL_ONLY")
    if len(set(y[list(train)].tolist())) != 2:
        raise RuntimeError("PROBE_TRAIN_REQUIRES_BOTH_CLASSES")
    if len(set(y[list(validation)].tolist())) != 2:
        raise RuntimeError("PROBE_VALIDATION_REQUIRES_BOTH_CLASSES")

    if probe_kind == "primary_real_only":
        if not all(bool(real_flags[i]) for i in train):
            raise RuntimeError("PROBE_PRIMARY_FIT_MUST_BE_REAL_ONLY")
        if len(train_subjects) != len(train):
            raise RuntimeError("PROBE_SUBJECT_DUPLICATE_WITHIN_TRAIN")
    elif probe_kind == "augmented_fit_sensitivity":
        real_train = {
            str(subject_hashes[i]) for i in train if bool(real_flags[i])
        }
        for index in train:
            if bool(real_flags[index]):
                continue
            parent = parent_hashes[index]
            if not _is_sha256(parent) or str(parent) not in real_train:
                raise RuntimeError("PROBE_DERIVATIVE_PARENT_LINEAGE_INVALID")
            if str(parent) in validation_subjects:
                raise RuntimeError("PROBE_DERIVATIVE_PARENT_OVERLAP")
            parent_index = next(
                i
                for i in train
                if bool(real_flags[i]) and str(subject_hashes[i]) == str(parent)
            )
            if int(y[index]) != int(y[parent_index]):
                raise RuntimeError("PROBE_DERIVATIVE_PARENT_LABEL_MISMATCH")
    else:
        raise RuntimeError(f"UNKNOWN_PROBE_KIND:{probe_kind}")

    if provenance.get("selection_role") != "inner_validation":
        raise RuntimeError("PROBE_SELECTION_ROLE_INVALID")
    if provenance.get("outer_test_touched") is not False:
        raise RuntimeError("PROBE_OUTER_TEST_EXCLUSION_UNPROVEN")
    for field in ("split_hash", "group_hash", "embedding_artifact_sha256"):
        if not _is_sha256(provenance.get(field)):
            raise RuntimeError(f"PROBE_PROVENANCE_HASH_INVALID:{field}")

    if tabular is not None:
        assert_predictive_feature_order(feature_names)
        assert_no_audit_fields_in_predictors(feature_names)
        tabular_array = np.asarray(tabular, dtype=np.float64)
        if tabular_array.shape != (n, len(METHOD_X_PHENOTYPE_FEATURES)):
            raise RuntimeError("PROBE_TABULAR_SHAPE_INVALID")
        features = np.concatenate([features, tabular_array], axis=1)
    elif tuple(feature_names):
        raise RuntimeError("PROBE_FEATURE_NAMES_WITHOUT_TABULAR")
    if not np.all(np.isfinite(features) | np.isnan(features)):
        raise RuntimeError("PROBE_FEATURE_NOT_FINITE")

    best: dict[str, Any] | None = None
    for c_value in c_grid:
        # sklearn >=1.8 expresses the L2 penalty as l1_ratio=0; passing
        # penalty="l2" is deprecated. L2 is a scientific requirement of the
        # probe contract, so it is stated explicitly rather than left to a
        # library default that could change.
        if not np.isfinite(float(c_value)) or float(c_value) <= 0.0:
            raise RuntimeError("PROBE_C_GRID_INVALID")
        model = Pipeline(
            steps=(
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(C=float(c_value), l1_ratio=0.0, max_iter=2000),
                ),
            )
        )
        model.fit(features[list(train)], y[list(train)])
        scores = model.predict_proba(features[list(validation)])[:, 1]
        auc = float(roc_auc_score(y[list(validation)], scores))
        if best is None or auc > best["inner_validation_auc"]:
            best = {"C": float(c_value), "inner_validation_auc": auc, "model": model}
    assert best is not None
    return {
        "C": best["C"],
        "inner_validation_auc": best["inner_validation_auc"],
        "model": best["model"],
        "c_grid": list(c_grid),
        "selected_on": "inner_validation_only",
        "outer_test_touched": False,
        "fit_real_only": all(bool(real_flags[i]) for i in train),
        "validation_real_only": True,
        "probe_kind": probe_kind,
        "preprocessing_fit_role": "head_fit",
        "feature_order": list(feature_names),
        "split_hash": provenance["split_hash"],
        "group_hash": provenance["group_hash"],
        "embedding_artifact_sha256": provenance["embedding_artifact_sha256"],
        "split_registry_sha256": registry["integrity_sha256"],
    }
