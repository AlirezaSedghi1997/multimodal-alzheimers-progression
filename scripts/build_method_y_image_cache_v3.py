from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.method_y_revival.atomic_io import atomic_write_json, read_json  # noqa: E402
from src.method_y_revival.image_cache_v3 import (  # noqa: E402
    ImageCacheV3Error,
    build_pattern_cache,
)


class ImageCacheLauncherError(RuntimeError):
    pass


def _path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(config_path: Path, branches: tuple[str, ...]) -> dict:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    output_root = _path(config["output_root"])
    readiness = read_json(output_root / "readiness/FINAL_IMAGE_TRAINING_READINESS.json")
    if readiness.get("training_execution_allowed") is not True:
        raise ImageCacheLauncherError("IMAGE_TRAINING_READINESS_NOT_PASS")
    # Provenance binding (Plan 010): readiness run contract and source artifact
    run_contract_path = output_root / "run_contract.json"
    if run_contract_path.is_file():
        try:
            rc_data = json.loads(run_contract_path.read_text(encoding="utf-8"))
            run_contract_sha = str(rc_data.get("digest", ""))
            if len(run_contract_sha) != 64:
                run_contract_sha = _sha256(run_contract_path)
        except (OSError, json.JSONDecodeError):
            run_contract_sha = _sha256(run_contract_path)
        expected_run_contract = readiness.get("run_contract_sha256")
        if expected_run_contract and run_contract_sha != expected_run_contract:
            raise ImageCacheLauncherError("READINESS_RUN_CONTRACT_MISMATCH")
    else:
        run_contract_sha = readiness.get("run_contract_sha256", "UNKNOWN")
    normalization = config["shared"]["pattern_normalization"]
    reference = _path(config["shared"]["reference_template"])
    cache_root = output_root / "cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    reports = {}
    for branch in branches:
        membership_path = output_root / "private" / f"{branch}_image_membership.parquet"
        if not membership_path.is_file():
            raise ImageCacheLauncherError(f"{branch.upper()}_PRIVATE_MEMBERSHIP_MISSING")
        expected = readiness["private_membership_sha256"][branch]
        if _sha256(membership_path) != expected:
            raise ImageCacheLauncherError(f"{branch.upper()}_PRIVATE_MEMBERSHIP_HASH_MISMATCH")
        membership = pd.read_parquet(membership_path)
        output_cache = _path(config["branches"][branch]["derived_cache"])
        # Source artifact hash (FBP/FTP HDF5) for provenance
        source_artifact = _path(config["branches"][branch].get("source_cache", ""))
        source_sha = _sha256(source_artifact) if source_artifact.is_file() else "UNKNOWN_SOURCE"
        # Verify against readiness if available
        expected_source = readiness.get("source_artifact_sha256", {}).get(branch) if isinstance(readiness.get("source_artifact_sha256"), dict) else None
        if expected_source and source_sha != expected_source:
            raise ImageCacheLauncherError("SOURCE_ARTIFACT_CHANGED_SINCE_READINESS")
        report = build_pattern_cache(
            membership=membership,
            reference_template=reference,
            output_cache=output_cache,
            state_path=cache_root / f"{branch}_pattern_cache_state.json",
            normalization=normalization,
            branch=branch,
            readiness_run_contract_sha256=run_contract_sha,
            source_artifact_sha256=source_sha,
        )
        atomic_write_json(cache_root / f"{branch}_pattern_cache_report.json", report)
        reports[branch] = report
        if report["status"] != "COMPLETED":
            raise ImageCacheLauncherError(f"{branch.upper()}_PATTERN_CACHE_INCOMPLETE")
    summary = {
        "schema_version": "3.0",
        "status": "COMPLETED",
        "branches": {
            branch: {
                "completed_count": report["completed_count"],
                "cache_sha256": report["cache_sha256"],
                "resumed_existing_count": report["resumed_existing_count"],
                "quantitative_scale_claim_allowed": False,
            }
            for branch, report in reports.items()
        },
    }
    atomic_write_json(cache_root / "PATTERN_CACHE_SUMMARY.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build or resume pattern-normalized Method Y image caches."
    )
    parser.add_argument(
        "--config", default=str(ROOT / "configs/method_y_image_training_v3.yaml")
    )
    parser.add_argument("--branch", choices=("amyloid", "tau", "both"), default="both")
    args = parser.parse_args()
    branches = ("amyloid", "tau") if args.branch == "both" else (args.branch,)
    try:
        summary = run(Path(args.config), branches)
    except (OSError, KeyError, ValueError, ImageCacheV3Error, ImageCacheLauncherError) as exc:
        print(json.dumps({"status": "BLOCKED", "reason": str(exc)}, indent=2), flush=True)
        return 2
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

