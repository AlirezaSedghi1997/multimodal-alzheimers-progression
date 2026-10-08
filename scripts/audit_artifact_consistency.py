#!/usr/bin/env python3
"""Reconcile artifact, live SPM, HPO, science-gate, and claim statuses."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.spm_completion_guard import (
    CATALOG_PATH,
    EXPECTED_CONFIG_HASH,
    MANIFEST_PATH,
    PREPROCESSED_DIR,
    spm_completion_status,
)
from src.utils.artifact_provenance import (
    audit_cache_contract,
    audit_claim_consistency,
    build_provenance,
    status_contract,
)


DEFAULT_ARTIFACT = Path("results/final_results/three_d/three_d_metrics.json")
DEFAULT_HPARAM = Path("results/tables/hyperparameter_selection_manifest.json")
DEFAULT_CACHE = Path("data/cached/t1_volume_cache_128.h5")
DEFAULT_SPLIT_SUMMARY = Path("splits/split_summary.json")
DEFAULT_JSON_OUTPUT = Path("results/tables/artifact_claim_consistency_audit.json")
DEFAULT_MD_OUTPUT = Path("results/tables/artifact_claim_consistency_audit.md")


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Cannot read required JSON artifact: {path.name}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"Required JSON artifact is not an object: {path.name}")
    return value


def build_consistency_report(
    *,
    artifact: dict[str, Any],
    live_spm: dict[str, Any],
    hparam_manifest: dict[str, Any],
    provenance: dict[str, Any],
    cache_contract: dict[str, Any] | None = None,
    artifact_cache_contract: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a privacy-safe report from already-loaded evidence."""

    result = audit_claim_consistency(
        artifact=artifact,
        live_spm=live_spm,
        hparam_manifest=hparam_manifest,
    )
    cache_blockers: list[str] = []
    for contract in (cache_contract, artifact_cache_contract):
        if contract is not None and contract.get("status") != "PASS":
            cache_blockers.extend(contract.get("blockers", []))
    if cache_blockers:
        result["blockers"] = list(dict.fromkeys([*result["blockers"], *cache_blockers]))
        result["status"] = "FAIL"
        result["final_claim_allowed"] = False
        result["status_contract"] = status_contract(
            artifact_status=result["status_contract"]["artifact_status"],
            science_gate_status="BLOCKED",
            claim_status="NOT_ALLOWED",
        )
    result["schema_version"] = "1.0"
    result["generated_at_utc"] = datetime.now(timezone.utc).isoformat()
    result["live_spm"] = {
        "ready": bool(live_spm.get("ready")),
        "status": live_spm.get("status", "UNKNOWN"),
        "expected_total": live_spm.get("expected_total"),
        "done_count": live_spm.get("done_count"),
        "terminal_count": live_spm.get("manifest_terminal_count"),
        "output_contract_valid_count": live_spm.get(
            "preprocessed_output_contract_valid_count"
        ),
        "qc_valid_count": live_spm.get("preprocessed_qc_valid_count"),
        "checks": live_spm.get("checks", {}),
    }
    result["hparam_selection"] = {
        "status": hparam_manifest.get("status", "UNKNOWN"),
        "test_used_for_selection": hparam_manifest.get("selection_contract", {}).get(
            "test_used_for_selection"
        ),
    }
    result["provenance"] = provenance
    result["cache_contract"] = cache_contract or {"status": "NOT_EVALUATED"}
    result["artifact_cache_contract"] = artifact_cache_contract or {"status": "NOT_EVALUATED"}
    result["privacy"] = {
        "raw_paths_exported": False,
        "raw_identifiers_exported": False,
        "subject_level_outputs_exported": False,
    }
    return result


def _markdown(report: dict[str, Any]) -> str:
    contract = report["status_contract"]
    blockers = report["blockers"] or ["none"]
    lines = [
        "# Artifact and claim consistency audit",
        "",
        f"Generated (UTC): `{report['generated_at_utc']}`",
        "",
        f"- Overall consistency: **{report['status']}**",
        f"- Artifact status: `{contract['artifact_status']}`",
        f"- Science gate: `{contract['science_gate_status']}`",
        f"- Claim status: `{contract['claim_status']}`",
        f"- Live SPM: `{report['live_spm']['status']}`",
        f"- HPO selection: `{report['hparam_selection']['status']}`",
        "",
        "## Blockers",
        "",
    ]
    lines.extend(f"- `{blocker}`" for blocker in blockers)
    lines.extend(
        [
            "",
            "`COMPLETED` records execution state only; it does not imply a passed science gate or an allowed claim.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--hparam-manifest", type=Path, default=DEFAULT_HPARAM)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--split-summary", type=Path, default=DEFAULT_SPLIT_SUMMARY)
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON_OUTPUT)
    parser.add_argument("--md-output", type=Path, default=DEFAULT_MD_OUTPUT)
    return parser.parse_args()


def _under_root(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def _artifact_path_under_root(root: Path, value: Any) -> Path | None:
    if not isinstance(value, str) or not value.strip():
        return None
    candidate = _under_root(root, Path(value)).resolve()
    try:
        candidate.relative_to(root)
    except ValueError:
        return None
    return candidate


def _load_expected_folds(path: Path) -> int:
    payload = _load_json(path)
    try:
        folds = int(payload["folds"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError("Split summary does not define a valid positive fold count") from exc
    if folds < 2:
        raise RuntimeError("Split summary does not define a valid positive fold count")
    return folds


def _read_cache_contract(
    path: Path | None, *, missing_blocker: str, expected_folds: int
) -> dict[str, Any]:
    if path is None or not path.is_file():
        return {
            "status": "FAIL",
            "blockers": [missing_blocker],
            "expected_folds": expected_folds,
        }
    import h5py

    with h5py.File(path, "r") as cache:
        attributes = {key: str(value) for key, value in cache.attrs.items()}
    report = audit_cache_contract(attributes, expected_folds=expected_folds)
    report["bytes"] = path.stat().st_size
    return report


def main() -> int:
    args = parse_args()
    root = args.repo_root.resolve()
    artifact_path = _under_root(root, args.artifact)
    hparam_path = _under_root(root, args.hparam_manifest)
    cache_path = _under_root(root, args.cache)
    split_summary_path = _under_root(root, args.split_summary)
    expected_folds = _load_expected_folds(split_summary_path)
    artifact_payload = _load_json(artifact_path)
    aggregate = artifact_payload.get("aggregate") if isinstance(artifact_payload.get("aggregate"), dict) else artifact_payload
    artifact_cache_path = _artifact_path_under_root(root, aggregate.get("cache"))
    live_spm = spm_completion_status(
        manifest_path=root / MANIFEST_PATH,
        preprocessed_dir=root / PREPROCESSED_DIR,
        expected_config_hash=EXPECTED_CONFIG_HASH,
        catalog_path=root / CATALOG_PATH,
    )
    provenance = build_provenance(
        repo_root=root,
        inputs=[artifact_path, hparam_path, root / MANIFEST_PATH],
        parent_artifacts=[
            artifact_path.relative_to(root).as_posix(),
            hparam_path.relative_to(root).as_posix(),
        ],
    )
    cache_report = _read_cache_contract(
        cache_path,
        missing_blocker="CANDIDATE_CACHE_MISSING",
        expected_folds=expected_folds,
    )
    artifact_cache_report = _read_cache_contract(
        artifact_cache_path,
        missing_blocker=(
            "ARTIFACT_CACHE_PATH_INVALID_OR_MISSING"
            if aggregate.get("cache") and artifact_cache_path is None
            else "ARTIFACT_CACHE_MISSING"
        ),
        expected_folds=expected_folds,
    )
    report = build_consistency_report(
        artifact=artifact_payload,
        live_spm=live_spm,
        hparam_manifest=_load_json(hparam_path),
        provenance=provenance,
        cache_contract=cache_report,
        artifact_cache_contract=artifact_cache_report,
    )
    json_output = _under_root(root, args.json_output)
    md_output = _under_root(root, args.md_output)
    json_output.parent.mkdir(parents=True, exist_ok=True)
    md_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    md_output.write_text(_markdown(report), encoding="utf-8")
    print(json.dumps({"status": report["status"], "blockers": report["blockers"]}))
    return 0 if report["status"] == "PASS" else 4


if __name__ == "__main__":
    raise SystemExit(main())
