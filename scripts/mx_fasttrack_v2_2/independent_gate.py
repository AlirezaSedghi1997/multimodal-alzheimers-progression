"""Eligibility preflight, never an external-validation or disease-region result."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _digest(value: object) -> bool:
    return (isinstance(value, str) and len(value) == 64
            and all(character in "0123456789abcdef" for character in value.lower()))


def assess_candidate(records: Sequence[dict], internal_patient_hashes: set[str],
                     *, endpoint_protocol_sha256: str | None,
                     linkage_attestation_sha256: str | None,
                     locked_model_manifest_sha256: str | None) -> dict:
    """Check structural prerequisites without loading outcomes into model selection.

    A passing preflight is only permission to design/execute a locked external
    evaluation in a separately preregistered run. It never upgrades claims.
    """
    blockers: list[str] = []
    if not _digest(endpoint_protocol_sha256):
        blockers.append("ENDPOINT_PROTOCOL_NOT_HASH_LOCKED")
    if not _digest(linkage_attestation_sha256):
        blockers.append("CROSS_DATASET_LINKAGE_NOT_ATTESTED")
    if not _digest(locked_model_manifest_sha256):
        blockers.append("MODEL_SELECTION_NOT_FROZEN")
    if not records:
        blockers.append("EXTERNAL_COHORT_NOT_PROVIDED")
    ids: set[str] = set()
    by_cohort: dict[str, set[int]] = defaultdict(set)
    sites: set[str] = set()
    scanners: set[str] = set()
    regional_reference_count = 0
    for row in records:
        if not isinstance(row, dict):
            blockers.append("INVALID_CANDIDATE_RECORD")
            continue
        patient = row.get("patient_hash")
        label = row.get("label")
        cohort, site, scanner = (row.get(key) for key in ("cohort", "site", "scanner"))
        if not isinstance(patient, str) or not patient:
            blockers.append("PATIENT_LINKAGE_ID_MISSING")
        elif patient in ids:
            blockers.append("DUPLICATE_EXTERNAL_PATIENT")
        elif patient in internal_patient_hashes:
            blockers.append("INTERNAL_EXTERNAL_PATIENT_OVERLAP")
        else:
            ids.add(patient)
        if type(label) is not int or label not in (0, 1):
            blockers.append("BINARY_ENDPOINT_LABEL_INVALID")
        if not _digest(row.get("mri_sha256")):
            blockers.append("MRI_INPUT_HASH_MISSING")
        if not all(isinstance(item, str) and item.strip() for item in (cohort, site, scanner)):
            blockers.append("COHORT_SITE_SCANNER_PROVENANCE_MISSING")
        elif type(label) is int and label in (0, 1):
            by_cohort[cohort].add(label)
            sites.add(site)
            scanners.add(scanner)
        if (row.get("reference_blinded") is True
            and _digest(row.get("regional_reference_sha256"))):
            regional_reference_count += 1
    if by_cohort and any(labels != {0, 1} for labels in by_cohort.values()):
        blockers.append("WITHIN_COHORT_LABEL_OVERLAP_ABSENT")
    blockers = sorted(set(blockers))
    return {"schema_version": 1,
            "status": "BLOCKED_PRECHECK" if blockers else "STRUCTURAL_PRECHECK_PASSED_EVALUATION_NOT_DONE",
            "blockers": blockers, "n_candidates": len(records),
            "n_cohorts_with_both_labels": sum(labels == {0, 1} for labels in by_cohort.values()),
            "n_sites": len(sites), "n_scanners": len(scanners),
            "n_blinded_regional_references": regional_reference_count,
            "declared_reference_coverage_complete": bool(records) and regional_reference_count == len(records),
            "external_validation_completed": False,
            "disease_region_confirmation_completed": False,
            "diagnostic_or_causal_claim_allowed": False,
            "declared_hashes_verified_against_files": False,
            "requires_new_preregistered_evaluation_and_blinded_regional_scoring": True}


def _internal_patient_hashes(root: Path) -> set[str]:
    paths = [root / "docs/fasttrack_private/cache_full_index.csv",
             *(root / "splits" / f"fold_{fold}_{kind}.csv"
               for fold in range(5) for kind in ("train", "test")),
             root / "splits/external_holdout.csv"]
    found: set[str] = set()
    for path in paths:
        if not path.is_file():
            raise RuntimeError(f"V22_INTERNAL_LINKAGE_SOURCE_MISSING:{path.name}")
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            for row in csv.DictReader(stream):
                for key in ("patient_hash", "source_patient_hash"):
                    value = row.get(key)
                    if value:
                        found.add(value)
    return found


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    from scripts.mx_fasttrack_v2.run import read, sha
    from scripts.mx_fasttrack_v2_1.supervise_v21 import immutable_json

    root = args.project_root.resolve()
    manifest_path = args.candidate_manifest.resolve()
    manifest = read(manifest_path)
    records = manifest.get("records")
    if not isinstance(records, list):
        raise RuntimeError("V22_CANDIDATE_RECORDS_MISSING")
    outcome = assess_candidate(
        records, _internal_patient_hashes(root),
        endpoint_protocol_sha256=manifest.get("endpoint_protocol_sha256"),
        linkage_attestation_sha256=manifest.get("linkage_attestation_sha256"),
        locked_model_manifest_sha256=manifest.get("locked_model_manifest_sha256"))
    outcome["candidate_manifest_sha256"] = sha(manifest_path)
    outcome["linkage_namespace_verified_by_this_preflight"] = False
    outcome["representativeness_or_sample_size_adequacy_verified"] = False
    if args.output is not None:
        target = args.output.resolve()
        if not target.is_relative_to(root) or target == root:
            raise RuntimeError("V22_PRECHECK_OUTPUT_OUTSIDE_PROJECT")
        immutable_json(target, outcome)
    print(json.dumps(outcome, ensure_ascii=False, sort_keys=True))
    return 0 if not outcome["blockers"] else 3


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError, KeyError, TypeError) as exc:
        print(json.dumps({"status": "BLOCKED_PRECHECK", "reason": str(exc)}, ensure_ascii=False),
              file=sys.stderr)
        raise SystemExit(3)
