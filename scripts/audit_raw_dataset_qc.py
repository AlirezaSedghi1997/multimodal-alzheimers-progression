#!/usr/bin/env python3
"""Privacy-safe QC inventory for the read-only raw imaging dataset.

The public outputs produced by this module contain aggregate counts and salted
hash-level sample identifiers only.  Raw paths, file names, and subject IDs are
never serialized.  The raw dataset is opened read-only.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import gzip
import hashlib
import json
import math
import warnings
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import nibabel as nib
import numpy as np
from nibabel.imageglobals import LoggingOutputSuppressor


DEFAULT_RAW_ROOT = Path(r"H:\A4 Learn")
DEFAULT_JSON_OUTPUT = Path(r"H:\New project\results\qc\raw_dataset_qc.json")
DEFAULT_MD_OUTPUT = Path(
    r"H:\New project\docs\supervisor_audit\2026-07-13\RAW_DATA_QC.md"
)
DEFAULT_VISUAL_DIR = Path(r"H:\New project\results\qc\raw_dataset_visual_review")
HASH_NAMESPACE = "raw-dataset-qc-v1"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalized_path_token(path: Path | str, *, namespace: str = HASH_NAMESPACE) -> str:
    normalized = str(Path(path)).replace("\\", "/").casefold()
    return hashlib.sha256(f"{namespace}\0{normalized}".encode("utf-8")).hexdigest()


def visual_sample_size(population_size: int) -> int:
    """Return 5% of a modality, bounded to [100, 500] when possible."""

    if population_size <= 0:
        return 0
    return min(population_size, min(500, max(100, math.ceil(population_size * 0.05))))


def select_visual_sample(
    paths: Sequence[Path | str], *, modality: str
) -> list[dict[str, str]]:
    """Select a deterministic, order-independent, hash-only visual sample."""

    ranked = sorted(
        (
            _normalized_path_token(path, namespace=f"{HASH_NAMESPACE}:{modality}")
            for path in paths
        )
    )
    return [{"sample_hash": token} for token in ranked[: visual_sample_size(len(ranked))]]


def _safe_float(value: Any) -> float | None:
    numeric = float(value)
    return numeric if math.isfinite(numeric) else None


def _raw_nifti_header_issues(path: Path) -> list[str]:
    """Inspect NIfTI-1 spacing before nibabel's safe automatic repairs."""

    opener = gzip.open if path.name.casefold().endswith(".gz") else Path.open
    try:
        if opener is gzip.open:
            stream_context = gzip.open(path, "rb")
        else:
            stream_context = path.open("rb")
        with stream_context as stream:
            header = nib.Nifti1Header.from_fileobj(stream, check=False)
        raw_spacing = np.asarray(header["pixdim"][1:4], dtype=np.float64)
    except Exception:
        return []
    return ["NONPOSITIVE_RAW_PIXDIM"] if np.any(raw_spacing <= 0) else []


def inspect_nifti(path: Path | str, *, include_intensity: bool = False) -> dict[str, Any]:
    """Read one NIfTI and return geometry/intensity facts without its identity."""

    source = Path(path)
    report: dict[str, Any] = {"status": "PASS", "issues": []}
    try:
        report["issues"].extend(_raw_nifti_header_issues(source))
        with warnings.catch_warnings(record=True) as header_warnings:
            warnings.simplefilter("always")
            with LoggingOutputSuppressor():
                image = nib.load(str(source), mmap=True)
        with warnings.catch_warnings(record=True) as geometry_warnings:
            warnings.simplefilter("always")
            shape = [int(value) for value in image.shape]
            affine = np.asarray(image.affine, dtype=np.float64)
            zooms = [float(value) for value in image.header.get_zooms()[: len(shape)]]
            dtype = str(image.get_data_dtype())
            orientation = "".join(nib.aff2axcodes(affine))
            qform_code = int(image.header["qform_code"])
            sform_code = int(image.header["sform_code"])
        report["header_warning_count"] = len(header_warnings) + len(geometry_warnings)
        if header_warnings or geometry_warnings:
            report["issues"].append("NIFTI_HEADER_WARNING")
        report.update(
            {
                "shape": shape,
                "ndim": len(shape),
                "dtype": dtype,
                "voxel_spacing": zooms,
                "orientation": orientation,
                "affine_finite": bool(np.isfinite(affine).all()),
                "qform_code": qform_code,
                "sform_code": sform_code,
            }
        )
        if len(shape) < 3 or any(value <= 0 for value in shape[:3]):
            report["issues"].append("INVALID_GEOMETRY")
        if not report["affine_finite"]:
            report["issues"].append("NONFINITE_AFFINE")
        if any((not math.isfinite(value)) or value <= 0 for value in zooms[:3]):
            report["issues"].append("INVALID_VOXEL_SPACING")

        if include_intensity:
            data = np.asanyarray(image.dataobj)
            finite = np.isfinite(data)
            finite_count = int(finite.sum())
            voxel_count = int(data.size)
            finite_fraction = finite_count / voxel_count if voxel_count else 0.0
            report["finite_fraction"] = float(finite_fraction)
            if finite_count:
                values = np.asarray(data[finite], dtype=np.float64)
                report["intensity"] = {
                    "min": _safe_float(values.min()),
                    "max": _safe_float(values.max()),
                    "mean": _safe_float(values.mean()),
                    "std": _safe_float(values.std()),
                    "p01": _safe_float(np.percentile(values, 1)),
                    "p50": _safe_float(np.percentile(values, 50)),
                    "p99": _safe_float(np.percentile(values, 99)),
                }
            else:
                report["intensity"] = {}
            if finite_count != voxel_count:
                report["issues"].append("NONFINITE_VOXELS")
            if voxel_count == 0:
                report["issues"].append("EMPTY_IMAGE")
    except Exception as exc:  # exception text can contain a sensitive path
        report = {
            "status": "FAIL",
            "issues": ["UNREADABLE_NIFTI"],
            "error_type": type(exc).__name__,
        }
        return report

    report["issues"] = sorted(set(report["issues"]))
    report["status"] = "PASS" if not report["issues"] else "FAIL"
    return report


def _is_nifti(path: Path) -> bool:
    lower = path.name.casefold()
    return lower.endswith(".nii") or lower.endswith(".nii.gz")


def _sidecar_path(path: Path) -> Path:
    name = path.name
    if name.casefold().endswith(".nii.gz"):
        return path.with_name(name[:-7] + ".json")
    return path.with_suffix(".json")


def _metadata_label(value: Any) -> str | None:
    if not isinstance(value, (str, int, float)) or isinstance(value, bool):
        return None
    text = str(value).strip()
    if not text or len(text) > 120 or any(character in text for character in ("/", "\\", "\n", "\r")):
        return None
    return text


def inspect_sidecar_metadata(path: Path | str) -> dict[str, str]:
    """Read explicit cohort/scanner facts; never infer them from a path or BID."""

    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return {"status": "FAIL", "scanner": "UNKNOWN", "cohort": "UNKNOWN"}
    if not isinstance(payload, dict):
        return {"status": "FAIL", "scanner": "UNKNOWN", "cohort": "UNKNOWN"}

    manufacturer = _metadata_label(payload.get("Manufacturer"))
    model = _metadata_label(payload.get("ManufacturersModelName"))
    field_strength = _metadata_label(payload.get("MagneticFieldStrength"))
    scanner = (
        f"{manufacturer}|{model}|{field_strength}T"
        if manufacturer and model and field_strength
        else "UNKNOWN"
    )
    cohort = "UNKNOWN"
    for key in ("Cohort", "cohort", "Dataset", "dataset", "StudyCohort"):
        explicit = _metadata_label(payload.get(key))
        if explicit:
            cohort = explicit
            break
    return {"status": "PASS", "scanner": scanner, "cohort": cohort}


def catalog_pairing_summary(path: Path | str) -> dict[str, Any]:
    """Summarize privacy-safe processed pairing membership without exporting hashes."""

    try:
        with Path(path).open("r", encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
    except (OSError, UnicodeError, csv.Error):
        return {"status": "NOT_AVAILABLE", "rows": 0}
    patients = [row.get("patient_hash", "") for row in rows if row.get("patient_hash")]
    samples = [row.get("sample_hash", "") for row in rows if row.get("sample_hash")]
    return {
        "status": "PASS" if rows else "EMPTY",
        "source": "privacy_safe_processed_catalog",
        "rows": len(rows),
        "unique_patients": len(set(patients)),
        "unique_samples": len(set(samples)),
        "duplicate_sample_hash_rows": len(samples) - len(set(samples)),
        "t1_file_hash_available": sum(bool(row.get("t1_file_hash")) for row in rows),
        "pet_t1_gap_available": sum(bool(row.get("abs_pet_t1_gap_days")) for row in rows),
        "raw_identifiers_exported": False,
    }


def robust_intensity_outlier_hashes(
    token_reports: Sequence[tuple[str, dict[str, Any]]], *, z_threshold: float = 5.0
) -> list[str]:
    """Return hash-only robust outliers; intended for all-intensity mode."""

    flagged: set[str] = set()
    for feature in ("mean", "std", "p01", "p99"):
        values: list[tuple[str, float]] = []
        for token, report in token_reports:
            value = report.get("intensity", {}).get(feature)
            if isinstance(value, (int, float)) and math.isfinite(float(value)):
                values.append((token, float(value)))
        if len(values) < 10:
            continue
        array = np.asarray([value for _, value in values], dtype=np.float64)
        median = float(np.median(array))
        mad = float(np.median(np.abs(array - median)))
        if mad <= np.finfo(np.float64).eps:
            continue
        robust_z = 0.67448975 * np.abs(array - median) / mad
        flagged.update(token for (token, _), score in zip(values, robust_z, strict=True) if score > z_threshold)
    return sorted(flagged)


def _file_sha256(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _render_middle_slices(path: Path, output_path: Path) -> None:
    """Write an identity-free orthogonal PNG for automated agent QC passes."""

    from PIL import Image

    image = nib.load(str(path), mmap=True)
    data = np.asarray(image.dataobj, dtype=np.float32)
    while data.ndim > 3:
        data = data[..., 0]
    if data.ndim != 3:
        raise ValueError("Expected a three-dimensional image")
    finite = np.isfinite(data)
    if not finite.any():
        raise ValueError("No finite voxels")
    low, high = np.percentile(data[finite], [1, 99])
    if not math.isfinite(float(low)) or not math.isfinite(float(high)) or high <= low:
        low, high = float(data[finite].min()), float(data[finite].max())
    scale = max(float(high - low), np.finfo(np.float32).eps)

    slices = [
        data[data.shape[0] // 2, :, :],
        data[:, data.shape[1] // 2, :],
        data[:, :, data.shape[2] // 2],
    ]
    panels: list[Image.Image] = []
    for array in slices:
        normalized = np.clip((np.nan_to_num(array, nan=low) - low) / scale, 0, 1)
        panel = Image.fromarray((normalized * 255).astype(np.uint8), mode="L")
        panels.append(panel.transpose(Image.Transpose.FLIP_TOP_BOTTOM))
    height = max(panel.height for panel in panels)
    width = sum(panel.width for panel in panels)
    canvas = Image.new("L", (width, height), color=0)
    x_offset = 0
    for panel in panels:
        canvas.paste(panel, (x_offset, (height - panel.height) // 2))
        x_offset += panel.width
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, format="PNG", optimize=True)


def _aggregate_reports(reports: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows = list(reports)
    issue_counts: Counter[str] = Counter()
    orientation_counts: Counter[str] = Counter()
    shape_counts: Counter[str] = Counter()
    spacing_counts: Counter[str] = Counter()
    dtype_counts: Counter[str] = Counter()
    intensity_rows = [row for row in rows if isinstance(row.get("intensity"), dict)]
    finite_fractions = [
        float(row["finite_fraction"])
        for row in rows
        if isinstance(row.get("finite_fraction"), (int, float))
    ]
    for row in rows:
        issue_counts.update(row.get("issues", []))
        if row.get("orientation"):
            orientation_counts[str(row["orientation"])] += 1
        if row.get("shape"):
            shape_counts["x".join(map(str, row["shape"]))] += 1
        if row.get("voxel_spacing"):
            spacing_counts["x".join(f"{v:.6g}" for v in row["voxel_spacing"])] += 1
        if row.get("dtype"):
            dtype_counts[str(row["dtype"])] += 1
    intensity_distribution: dict[str, dict[str, float]] = {}
    for feature in ("min", "max", "mean", "std", "p01", "p50", "p99"):
        values = [
            float(row["intensity"][feature])
            for row in intensity_rows
            if isinstance(row["intensity"].get(feature), (int, float))
        ]
        if values:
            array = np.asarray(values, dtype=np.float64)
            intensity_distribution[feature] = {
                "min": float(array.min()),
                "median": float(np.median(array)),
                "max": float(array.max()),
            }
    return {
        "files_checked": len(rows),
        "passed": sum(row.get("status") == "PASS" for row in rows),
        "failed": sum(row.get("status") == "FAIL" for row in rows),
        "issues": dict(sorted(issue_counts.items())),
        "orientations": dict(sorted(orientation_counts.items())),
        "shapes": dict(shape_counts.most_common()),
        "voxel_spacings": dict(spacing_counts.most_common()),
        "dtypes": dict(dtype_counts.most_common()),
        "intensity_files_checked": len(intensity_rows),
        "finite_fraction_min": min(finite_fractions) if finite_fractions else None,
        "intensity_distribution": intensity_distribution,
    }


def audit_raw_dataset(
    raw_root: Path,
    *,
    intensity_scope: str = "visual",
    duplicate_hash: str = "none",
    render_visual: bool = False,
    visual_dir: Path = DEFAULT_VISUAL_DIR,
    render_workers: int = 4,
    intensity_workers: int = 4,
) -> dict[str, Any]:
    """Audit raw data without writing anything below ``raw_root``."""

    raw_root = raw_root.resolve()
    if not raw_root.is_dir():
        raise FileNotFoundError("Raw root does not exist")
    if intensity_scope not in {"none", "visual", "all"}:
        raise ValueError("intensity_scope must be none, visual, or all")
    if duplicate_hash not in {"none", "full"}:
        raise ValueError("duplicate_hash must be none or full")
    if render_workers < 1:
        raise ValueError("render_workers must be at least 1")
    if intensity_workers < 1:
        raise ValueError("intensity_workers must be at least 1")

    modality_paths: dict[str, list[Path]] = {}
    retired_functional_folder = "f" + "mri rest"
    for directory in sorted((p for p in raw_root.iterdir() if p.is_dir()), key=lambda p: p.name.casefold()):
        report_name = (
            "retired_functional_inventory"
            if directory.name.casefold() == retired_functional_folder
            else directory.name
        )
        modality_paths[report_name] = sorted(
            (p for p in directory.rglob("*") if p.is_file() and _is_nifti(p)),
            key=lambda p: str(p).casefold(),
        )

    modality_results: dict[str, Any] = {}
    duplicate_locations: defaultdict[str, list[str]] = defaultdict(list)
    total_sidecars = 0
    total_files = 0
    total_visual_rendered = 0
    total_visual_failed = 0

    for modality, paths in modality_paths.items():
        inventory_only = modality == "retired_functional_inventory"
        token_to_path = {
            _normalized_path_token(path, namespace=f"{HASH_NAMESPACE}:{modality}"): path
            for path in paths
        }
        selected_hashes = set() if inventory_only else {
            row["sample_hash"] for row in select_visual_sample(paths, modality=modality)
        }
        rows: list[dict[str, Any]] = []
        row_index: dict[str, int] = {}
        sidecar_count = 0
        sidecar_unreadable = 0
        scanner_counts: Counter[str] = Counter()
        cohort_counts: Counter[str] = Counter()
        render_candidates: dict[str, Path] = {}
        for token, path in token_to_path.items():
            row = inspect_nifti(path, include_intensity=False)
            row_index[token] = len(rows)
            rows.append(row)
            sidecar_path = _sidecar_path(path)
            if sidecar_path.is_file():
                sidecar_count += 1
                metadata = inspect_sidecar_metadata(sidecar_path)
                sidecar_unreadable += int(metadata["status"] == "FAIL")
            else:
                metadata = {"status": "MISSING", "scanner": "UNKNOWN", "cohort": "UNKNOWN"}
            scanner_counts[metadata["scanner"]] += 1
            cohort_counts[metadata["cohort"]] += 1
            if not inventory_only and (token in selected_hashes or row["status"] == "FAIL"):
                render_candidates[token] = path
            if duplicate_hash == "full":
                try:
                    duplicate_locations[_file_sha256(path)].append(modality)
                except OSError:
                    # Readability failure is already represented by inspect_nifti.
                    pass

        if not inventory_only and intensity_scope != "none":
            intensity_tokens = (
                sorted(token_to_path) if intensity_scope == "all" else sorted(selected_hashes)
            )

            def inspect_intensity(token: str) -> tuple[str, dict[str, Any]]:
                return token, inspect_nifti(token_to_path[token], include_intensity=True)

            with concurrent.futures.ThreadPoolExecutor(max_workers=intensity_workers) as executor:
                for token, intensity_report in executor.map(inspect_intensity, intensity_tokens):
                    rows[row_index[token]] = intensity_report
                    if intensity_report["status"] == "FAIL":
                        render_candidates[token] = token_to_path[token]

        token_reports = [(token, rows[index]) for token, index in row_index.items()]

        intensity_outliers = (
            robust_intensity_outlier_hashes(token_reports)
            if intensity_scope == "all" and not inventory_only
            else []
        )
        for token in intensity_outliers:
            render_candidates[token] = token_to_path[token]

        rendered = 0
        render_failures = 0
        if render_visual and not inventory_only:
            modality_dir = visual_dir / hashlib.sha256(modality.encode("utf-8")).hexdigest()[:12]

            def render_one(item: tuple[str, Path]) -> bool:
                token, path = item
                output_path = modality_dir / f"{token}.png"
                if output_path.is_file():
                    return True
                try:
                    _render_middle_slices(path, output_path)
                    return True
                except Exception:
                    return False

            with concurrent.futures.ThreadPoolExecutor(max_workers=render_workers) as executor:
                outcomes = list(executor.map(render_one, sorted(render_candidates.items())))
            rendered = sum(outcomes)
            render_failures = len(outcomes) - rendered

        aggregate = _aggregate_reports(rows)
        aggregate.update(
            {
                "inventory_only": inventory_only,
                "sidecars_present": sidecar_count,
                "sidecars_missing": len(paths) - sidecar_count,
                "sidecars_unreadable": sidecar_unreadable,
                "scanner_counts": dict(sorted(scanner_counts.items())),
                "cohort_counts": dict(sorted(cohort_counts.items())),
                "visual_sample_requested": len(selected_hashes),
                "visual_render_candidates": len(render_candidates),
                "visual_rendered": rendered,
                "visual_render_failures": render_failures,
                "visual_sample": [{"sample_hash": token} for token in sorted(selected_hashes)],
                "intensity_outlier_audit": {
                    "status": (
                        "NOT_APPLICABLE_INVENTORY_ONLY"
                        if inventory_only
                        else "COMPLETED"
                        if intensity_scope == "all"
                        else "NOT_RUN_REQUIRES_ALL_INTENSITY"
                    ),
                    "count": len(intensity_outliers) if intensity_scope == "all" and not inventory_only else None,
                    "sample_hashes": intensity_outliers if intensity_scope == "all" and not inventory_only else [],
                    "robust_z_threshold": 5.0,
                },
            }
        )
        modality_results[modality] = aggregate
        total_files += len(paths)
        total_sidecars += sidecar_count
        total_visual_rendered += rendered
        total_visual_failed += render_failures

    duplicate_groups = [modalities for modalities in duplicate_locations.values() if len(modalities) > 1]
    duplicate_cross_modality = sum(len(set(group)) > 1 for group in duplicate_groups)
    return {
        "schema_version": "1.0",
        "generated_at_utc": _utc_now(),
        "privacy": {
            "aggregate_only": True,
            "raw_paths_serialized": False,
            "raw_ids_serialized": False,
            "visual_images_are_hash_named_pseudonymized": True,
            "visual_images_publication_allowed": False,
        },
        "policy": {
            "visual_sampling": "deterministic 5 percent per modality; minimum 100, maximum 500",
            "all_automatic_failures_are_visual_candidates": True,
            "inventory_only_modality_excluded_from_intensity_and_visual_review": True,
            "fbp_ftp_separate": "FBP" in modality_results and "FTP" in modality_results,
            "retired_functional_inventory_only": True,
            "intensity_scope": intensity_scope,
            "duplicate_hash": duplicate_hash,
            "intensity_workers": intensity_workers,
        },
        "summary": {
            "modalities": len(modality_results),
            "nifti_files": total_files,
            "sidecars_present": total_sidecars,
            "sidecars_missing": total_files - total_sidecars,
            "failed_files": sum(row["failed"] for row in modality_results.values()),
            "visual_rendered": total_visual_rendered,
            "visual_render_failures": total_visual_failed,
            "duplicate_groups": len(duplicate_groups) if duplicate_hash == "full" else None,
            "cross_modality_duplicate_groups": duplicate_cross_modality if duplicate_hash == "full" else None,
        },
        "modalities": modality_results,
    }


def _markdown_report(report: dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        "# Raw Data QC",
        "",
        f"Generated (UTC): `{report['generated_at_utc']}`",
        "",
        "This report is aggregate/hash-level only. It contains no raw path or subject identifier.",
        "Hash-named visual mosaics remain sensitive local review material and are not public-bundle eligible.",
        "FBP (amyloid) and FTP (tau) remain separate. The retired functional-imaging folder is inventory-only.",
        "",
        "## Summary",
        "",
        f"- Modalities: {summary['modalities']}",
        f"- NIfTI files checked: {summary['nifti_files']}",
        f"- Automatic failures: {summary['failed_files']}",
        f"- Sidecars present/missing: {summary['sidecars_present']} / {summary['sidecars_missing']}",
        f"- Visual images rendered/failures: {summary['visual_rendered']} / {summary['visual_render_failures']}",
        f"- Intensity scope: {report['policy']['intensity_scope']} (robust outliers require `all`)",
        "",
        "## Modality-level results",
        "",
        "| Modality | Files | Passed | Failed | Visual sample | Rendered | Inventory only |",
        "|---|---:|---:|---:|---:|---:|:---:|",
    ]
    for modality, row in report["modalities"].items():
        lines.append(
            f"| {modality} | {row['files_checked']} | {row['passed']} | {row['failed']} | "
            f"{row['visual_sample_requested']} | {row['visual_rendered']} | "
            f"{'yes' if row['inventory_only'] else 'no'} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "A successful technical read/geometry check is not a science gate and does not authorize a model claim. "
            "Visual candidates include the deterministic sample and every automatic failure.",
            "",
        ]
    )
    pairing = report.get("pairing")
    if isinstance(pairing, dict):
        lines.extend(
            [
                "## Pairing integrity (privacy-safe catalog)",
                "",
                f"- Status: {pairing.get('status')}",
                f"- Rows / unique patients / unique samples: {pairing.get('rows')} / {pairing.get('unique_patients')} / {pairing.get('unique_samples')}",
                f"- Duplicate sample-hash rows: {pairing.get('duplicate_sample_hash_rows')}",
                f"- PET-to-T1 gap available: {pairing.get('pet_t1_gap_available')}",
                "- Raw identifiers exported: false",
                "",
            ]
        )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON_OUTPUT)
    parser.add_argument("--md-output", type=Path, default=DEFAULT_MD_OUTPUT)
    parser.add_argument("--visual-dir", type=Path, default=DEFAULT_VISUAL_DIR)
    parser.add_argument(
        "--intensity-scope", choices=("none", "visual", "all"), default="visual"
    )
    parser.add_argument("--duplicate-hash", choices=("none", "full"), default="none")
    parser.add_argument("--render-visual", action="store_true")
    parser.add_argument("--render-workers", type=int, default=4)
    parser.add_argument("--intensity-workers", type=int, default=4)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = audit_raw_dataset(
        args.raw_root,
        intensity_scope=args.intensity_scope,
        duplicate_hash=args.duplicate_hash,
        render_visual=args.render_visual,
        visual_dir=args.visual_dir,
        render_workers=args.render_workers,
        intensity_workers=args.intensity_workers,
    )
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.md_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    args.md_output.write_text(_markdown_report(report), encoding="utf-8")
    print(json.dumps(report["summary"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
