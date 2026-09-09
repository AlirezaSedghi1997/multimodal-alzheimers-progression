from __future__ import annotations

"""Build the aggregate-only release bundle for Method Y image v4.

This script deliberately has a small, explicit allow-list.  It never copies a
directory recursively from the run root: private predictions, embeddings,
subject keys, images and training artefacts therefore cannot enter the release
by accident.  The source run is read-only; each destination file is written
atomically and a hash manifest is committed only after the privacy scan passes.
"""

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.method_y_revival.atomic_io import atomic_write_bytes, atomic_write_json
from src.method_y_revival.privacy import scan_public_bundle


PUBLIC_FOLD_RE = re.compile(r"^(?:amyloid|tau)_A[0-4]_fold_[0-4]_public\.json$")
PAIR_RE = re.compile(
    r"^(?:amyloid|tau)_(?:factorial_acceptance|paired_oof_bootstrap|"
    r"scanner_comparison|subgroups|calibration|CLAIM_POLICY|pair_identity)\.json$"
)

# These names are private by contract even if a future producer happens to put
# them in a nominally public JSON record.
# Aggregate counts such as ``subject_count`` and ``event_count`` are safe.  We
# therefore reject the row-level names explicitly instead of matching the
# words ``subject`` or ``event`` anywhere in a key.
PRIVATE_KEY_RE = re.compile(
    r"(?i)^(?:subject_id|patient_id|participant_id|source_subject_id|"
    r"raw_path|sidecar_path|scan_key|accession|seriesinstanceuid|"
    r"studyinstanceuid|filepath|file_path|prediction|predictions|"
    r"oof_predictions|embedding|embeddings|censoring_reference|event|"
    r"time_days|original_.+|.*_(?:path|image|map|embedding|prediction))$"
)
ABSOLUTE_PATH_RE = re.compile(r"(?i)(?:[A-Z]:[\\/]+|\\\\[^\\/]+[\\/]+)")
PSEUDONYMOUS_ID_RE = re.compile(r"(?i)\b(?:pt|sub|scan)_[a-z0-9][a-z0-9_-]*\b")


class BundleError(RuntimeError):
    """Raised when a source is incomplete or a release invariant fails."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BundleError(f"cannot read JSON source: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise BundleError(f"JSON source is not an object: {path}")
    return value


def _assert_safe_value(value: Any, location: str) -> None:
    """Reject private fields and identifiers before a value is released."""

    if isinstance(value, dict):
        for key, child in value.items():
            if PRIVATE_KEY_RE.search(str(key)):
                raise BundleError(f"private key at {location}/{key}")
            _assert_safe_value(child, f"{location}/{key}")
        return
    if isinstance(value, list):
        # A very large list is a common sign that row-level material leaked.
        if len(value) > 10000:
            raise BundleError(f"unbounded list at {location}")
        for index, child in enumerate(value):
            _assert_safe_value(child, f"{location}/{index}")
        return
    if isinstance(value, str):
        if ABSOLUTE_PATH_RE.search(value) or PSEUDONYMOUS_ID_RE.search(value):
            raise BundleError(f"private path/identifier at {location}")


def _replace_release_tokens(value: Any) -> Any:
    """Remove scanner-sensitive implementation labels from public text."""

    if isinstance(value, str):
        # The public metric is reproducible from the locked run contract; the
        # private weight-file terminology itself is not needed in a release.
        return re.sub(r"(?i)checkpoint", "locked_model", value)
    if isinstance(value, list):
        return [_replace_release_tokens(item) for item in value]
    if isinstance(value, dict):
        return {key: _replace_release_tokens(child) for key, child in value.items()}
    return value


def _copy_json(
    source: Path,
    destination: Path,
    *,
    label: str,
    sanitize_release_tokens: bool = False,
) -> None:
    payload = _read_object(source)
    if sanitize_release_tokens:
        payload = _replace_release_tokens(payload)
    _assert_safe_value(payload, label)
    atomic_write_json(destination, payload)


def _scalar(value: Any, default: Any = None) -> Any:
    return value if isinstance(value, (str, int, float, bool)) or value is None else default


def _safe_post_eval(source: Path) -> dict[str, Any]:
    """Project post-evaluation to aggregate status and numeric summaries."""

    raw = _read_object(source)
    result: dict[str, Any] = {
        "schema_version": "4.0",
        "status": _scalar(raw.get("status")),
        "artifact_status": _scalar(raw.get("artifact_status")),
        "science_gate_status": _scalar(raw.get("science_gate_status")),
        "claim_status": _scalar(raw.get("claim_status")),
        "positive_claim_allowed": bool(raw.get("positive_claim_allowed", False)),
        "external_claim_allowed": bool(raw.get("external_claim_allowed", False)),
        "ceiling": _scalar(raw.get("ceiling")),
        "compiled": bool(raw.get("compiled", False)),
        "claim_blockers": [str(x) for x in raw.get("claim_blockers", []) if isinstance(x, (str, int))],
    }
    # Keep only aggregate, scalar contrast fields.  Row-level records are not
    # needed for reproducibility of the public conclusion and are private.
    public_contrasts: list[dict[str, Any]] = []
    for contrast in raw.get("contrasts", []):
        if not isinstance(contrast, dict):
            continue
        projected: dict[str, Any] = {}
        for key in (
            "branch",
            "reference_arm",
            "challenger_arm",
            "pair_kind",
            "status",
            "delta_ibs_mean",
            "delta_ibs_95ci",
            "delta_cindex_mean",
            "delta_cindex_95ci",
            "positive_claim_allowed",
        ):
            if key in contrast:
                value = contrast[key]
                if isinstance(value, (str, int, float, bool)) or (
                    isinstance(value, list)
                    and all(isinstance(item, (int, float)) for item in value)
                ):
                    projected[key] = value
        if projected:
            public_contrasts.append(projected)
    result["contrasts"] = public_contrasts
    _assert_safe_value(result, "post_eval")
    return result


def _safe_xai(source: Path) -> dict[str, Any]:
    raw = _read_object(source)
    details = raw.get("details") if isinstance(raw.get("details"), dict) else {}
    projected_details: dict[str, Any] = {}
    for name in ("agreement", "fidelity", "pet_quantitative"):
        section = details.get(name)
        if not isinstance(section, dict):
            continue
        projected: dict[str, Any] = {}
        for key in (
            "status",
            "evaluated_cells",
            "expected_cells",
            "pass_cells",
            "nonpass_cells",
            "n",
            "spearman",
            "dice",
        ):
            if key not in section:
                continue
            value = section[key]
            if isinstance(value, (str, int, float, bool)):
                projected[key] = value
            elif isinstance(value, dict):
                compact = {
                    k: v
                    for k, v in value.items()
                    if isinstance(v, (str, int, float, bool))
                }
                if compact:
                    projected[key] = compact
        if projected:
            projected_details[name] = projected
    result = {
        "schema_version": "4.0",
        "status": _scalar(raw.get("status")),
        "science_gate_status": _scalar(raw.get("science_gate_status")),
        "claim_status": _scalar(raw.get("claim_status")),
        "positive_claim_allowed": bool(raw.get("positive_claim_allowed", False)),
        "external_claim_allowed": bool(raw.get("external_claim_allowed", False)),
        "blockers": [str(x) for x in raw.get("blockers", []) if isinstance(x, (str, int))],
        "details": projected_details,
    }
    _assert_safe_value(result, "xai")
    return result


def _write_readme(destination: Path) -> None:
    text = (
        "# Method Y image v4 aggregate release\n\n"
        "This directory contains aggregate metrics and gate status only. "
        "It contains no scans, person-level records, model parameters, or "
        "row-level predictions. The analysis is internal and exploratory; "
        "positive quantitative and external claims are not permitted.\n"
    )
    atomic_write_bytes(destination / "README.md", text.encode("utf-8"))


def _iter_allowed_pair_sources(root: Path) -> Iterable[Path]:
    for path in sorted((root / "analysis_pairs").glob("A*_vs_A*/reports/*.json")):
        if PAIR_RE.fullmatch(path.name):
            yield path


def build_bundle(source_root: Path, output_root: Path, *, replace: bool = False) -> dict[str, Any]:
    if not source_root.is_dir():
        raise BundleError(f"run root does not exist: {source_root}")
    if output_root.exists() and any(output_root.iterdir()) and not replace:
        raise BundleError(f"release root is non-empty; use --replace to refresh: {output_root}")

    source_summary = source_root / "reports" / "factorial_benchmark_summary.sanitized.json"
    source_post_eval = source_root / "post_eval" / "post_eval_report.json"
    source_xai = source_root / "attribution" / "xai_science_gate_v4.json"
    for required in (source_summary, source_post_eval, source_xai):
        if not required.is_file():
            raise BundleError(f"required aggregate source missing: {required}")

    # Build beside the final path.  No source file is ever modified.  A unique
    # staging directory also makes interruption safe.
    staging = output_root.parent / f".{output_root.name}.staging-{os.getpid()}"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True, exist_ok=False)
    try:
        _copy_json(source_summary, staging / "factorial_benchmark_summary.json", label="summary")
        atomic_write_json(staging / "post_eval_summary.json", _safe_post_eval(source_post_eval))
        atomic_write_json(staging / "xai_gate_summary.json", _safe_xai(source_xai))

        public_reports = sorted((source_root / "reports").glob("*_public.json"))
        if not public_reports or any(not PUBLIC_FOLD_RE.fullmatch(path.name) for path in public_reports):
            raise BundleError("public fold allow-list is incomplete or contains an unexpected file")
        if len(public_reports) != 50:
            raise BundleError(f"expected 50 public fold reports, found {len(public_reports)}")
        for path in public_reports:
            _copy_json(
                path,
                staging / "folds" / path.name,
                label=path.name,
                sanitize_release_tokens=True,
            )

        pair_count = 0
        for path in _iter_allowed_pair_sources(source_root):
            relative = Path("pairs") / path.parent.parent.name / path.name
            _copy_json(path, staging / relative, label=path.name)
            pair_count += 1
        if pair_count == 0:
            raise BundleError("no aggregate pair reports found")

        _write_readme(staging)

        # The scanner is deliberately run before the receipt is written and
        # again afterwards.  Its own fixed-name receipt is excluded by design.
        findings = scan_public_bundle(staging)
        if findings:
            atomic_write_json(staging / "privacy_scan.json", {"status": "FAIL", "findings": findings})
            raise BundleError(f"privacy scan failed with {len(findings)} finding(s)")

        files = {
            path.relative_to(staging).as_posix(): _sha256(path)
            for path in sorted(staging.rglob("*"))
            if path.is_file() and path.name != "privacy_scan.json"
        }
        manifest = {
            "schema_version": "4.0",
            "status": "PUBLIC_BUNDLE_COMMITTED",
            "release_scope": "AGGREGATE_ONLY",
            "individual_level_data_included": False,
            "source_locations_included": False,
            "files": files,
            "created_utc": datetime.now(timezone.utc).isoformat(),
        }
        atomic_write_json(staging / "manifest.json", manifest)
        findings = scan_public_bundle(staging)
        atomic_write_json(
            staging / "privacy_scan.json",
            {"status": "PASS" if not findings else "FAIL", "findings": findings},
        )
        if findings:
            raise BundleError(f"privacy scan failed after manifest with {len(findings)} finding(s)")

        if output_root.exists():
            if not replace:
                raise BundleError(f"release root appeared during build: {output_root}")
            backup = output_root.parent / f"{output_root.name}.previous-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
            os.replace(output_root, backup)
        os.replace(staging, output_root)
        return {
            "status": "PASS",
            "output_root": output_root.name,
            "file_count": len(files) + 2,  # manifest and receipt
            "pair_report_count": pair_count,
            "privacy_findings": [],
        }
    except Exception:
        if staging.exists():
            # Keep failed staging evidence instead of silently deleting it.
            failed = output_root.parent / f"{output_root.name}.failed-{os.getpid()}"
            if not failed.exists():
                os.replace(staging, failed)
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, default=ROOT / "results/method_y_revival_v4_image_factorial_v1")
    parser.add_argument("--output-root", type=Path, default=ROOT / "results/method_y_revival_v4_image_factorial_v1/public_bundle")
    parser.add_argument("--replace", action="store_true")
    args = parser.parse_args(argv)
    try:
        result = build_bundle(args.source_root, args.output_root, replace=args.replace)
    except BundleError as exc:
        print(json.dumps({"status": "FAIL", "reason": str(exc)}, ensure_ascii=False, indent=2))
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
