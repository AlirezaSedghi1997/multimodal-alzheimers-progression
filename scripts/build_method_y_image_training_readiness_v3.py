from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_method_y_tau_progression import (  # noqa: E402
    load_first_tau_scans,
    patient_hash_for_key,
)
from src.method_y_revival.atomic_io import (  # noqa: E402
    atomic_write_json,
    atomic_write_parquet,
    read_json,
)
from src.method_y_revival.contracts import RunContract, RunContractMismatch  # noqa: E402
from src.method_y_revival.image_training_manifest_v3 import (  # noqa: E402
    audit_membership,
    validate_model_feature_columns,
)
from src.method_y_revival.image_training_policy import (  # noqa: E402
    assess_image_training_policy,
)
from src.method_y_revival.process_guard import WriterLock  # noqa: E402
from src.method_y_revival.software_contract_v3 import (  # noqa: E402
    bind_software_runtime_contract,
)


class ImageReadinessBuildError(RuntimeError):
    """Raised when the live project cannot support a locked image membership."""


def _path(value: str | Path) -> Path:
    candidate = Path(value)
    return candidate if candidate.is_absolute() else ROOT / candidate


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_file(path: Path, label: str) -> Path:
    if not path.is_file():
        raise ImageReadinessBuildError(f"{label}_REQUIRED:{path}")
    return path


def _future_endpoint_contract(path: Path) -> tuple[bool, bool]:
    payload = read_json(path)
    contract = payload.get("contract", payload)
    definition = contract.get("endpoint_definition", {})
    locked = bool(contract.get("locked"))
    future = bool(
        definition.get("primary_name") == "sustained_cdr_global_progression"
        and definition.get("event_time_rule")
        == "first_increase_if_confirmed_by_next_visit"
        and definition.get("same_pet_derived_target_used", False) is not True
    )
    return locked, future


def _scanner_table(path: Path, branch: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"branch", "patient_hash", "manufacturer"}
    if not required.issubset(frame.columns):
        raise ImageReadinessBuildError("SCANNER_LINKAGE_COLUMNS_MISSING")
    selected = frame.loc[
        frame["branch"].astype(str).str.lower() == branch,
        ["patient_hash", "manufacturer"],
    ].copy()
    selected["manufacturer"] = (
        selected["manufacturer"].fillna("UNKNOWN").astype(str).str.upper()
    )
    if selected["patient_hash"].duplicated().any():
        raise ImageReadinessBuildError(f"{branch.upper()}_SCANNER_LINKAGE_DUPLICATE")
    return selected


def _model_columns(branch_config: dict[str, Any]) -> tuple[str, ...]:
    return validate_model_feature_columns(branch_config.get("model_features", []))


def _attach_endpoint_fields(
    base: dict[str, Any], endpoint_row: pd.Series, model_columns: tuple[str, ...]
) -> dict[str, Any]:
    row = dict(base)
    row["event"] = int(endpoint_row["event"])
    row["time_days"] = float(endpoint_row["time_days"])
    row["cohort"] = str(endpoint_row["cohort"])
    for column in model_columns:
        row[column] = endpoint_row.get(column, np.nan)
    return row


def _build_amyloid_membership(
    branch_config: dict[str, Any], scanner_path: Path
) -> pd.DataFrame:
    endpoint = pd.read_parquet(_require_file(_path(branch_config["endpoint"]), "AMYLOID_ENDPOINT"))
    endpoint = endpoint.set_index("patient_hash", drop=False)
    mapping = pd.read_csv(
        _require_file(_path(branch_config["private_sample_mapping"]), "FBP_PRIVATE_MAPPING"),
        usecols=["sample_hash", "patient_hash"],
    ).drop_duplicates()
    if mapping["sample_hash"].duplicated().any():
        raise ImageReadinessBuildError("FBP_SAMPLE_TO_PATIENT_MAPPING_AMBIGUOUS")
    patient_by_sample = mapping.set_index("sample_hash")["patient_hash"].astype(str)
    scanner = _scanner_table(scanner_path, "amyloid").set_index("patient_hash")
    model_columns = _model_columns(branch_config)
    cache_path = _require_file(_path(branch_config["source_cache"]), "FBP_IMAGE_CACHE")
    development = str(branch_config["development_cohort"])
    transport = str(branch_config["internal_transport_cohort"])
    rows: list[dict[str, Any]] = []
    with h5py.File(cache_path, "r") as cache:
        if str(cache.attrs.get("tracer", "")).upper() != "FBP":
            raise ImageReadinessBuildError("FBP_CACHE_TRACER_MISMATCH")
        for image_key in sorted(cache.keys()):
            if image_key not in patient_by_sample.index:
                continue
            patient_hash = str(patient_by_sample.loc[image_key])
            if patient_hash not in endpoint.index:
                continue
            endpoint_row = endpoint.loc[patient_hash]
            cohort = str(endpoint_row["cohort"])
            if cohort not in {development, transport}:
                continue
            group = cache[image_key]
            image = group.get("image")
            geometry_valid = bool(
                image is not None
                and tuple(image.shape) == (1, 128, 128, 128)
                and str(group.attrs.get("status", "")) == "COMPLETE"
            )
            finite = float(group.attrs.get("finite_fraction", 0.0)) == 1.0
            manufacturer = str(group.attrs.get("scanner", "UNKNOWN")).upper()
            if patient_hash in scanner.index:
                linked = str(scanner.loc[patient_hash, "manufacturer"]).upper()
                if manufacturer in {"", "UNKNOWN", "NAN"}:
                    manufacturer = linked
            row = _attach_endpoint_fields(
                {
                    "patient_hash": patient_hash,
                    "image_key": image_key,
                    "image_storage_kind": "HDF5_GROUP",
                    "image_path_private": str(cache_path),
                    "h5_group": image_key,
                    "source_image_sha256": str(
                        group.attrs.get("source_sha256", "")
                    ),
                    "tracer": "FBP",
                    "manufacturer": manufacturer or "UNKNOWN",
                    "qc_status": "PASS" if geometry_valid and finite else "EXCLUDED",
                    "image_geometry_valid": geometry_valid,
                    "image_finite": finite,
                    "role": "development" if cohort == development else "internal_transport",
                    "input_interpretation": "PATTERN_NORMALIZED_FBP_NOT_OFFICIAL_CENTILOID",
                },
                endpoint_row,
                model_columns,
            )
            rows.append(row)
    frame = pd.DataFrame(rows).sort_values(["role", "patient_hash"], kind="stable")
    if frame["patient_hash"].duplicated().any():
        raise ImageReadinessBuildError("AMYLOID_MEMBERSHIP_NOT_ONE_IMAGE_PER_SUBJECT")
    return frame.reset_index(drop=True)


def _private_transform_lookup(path: Path) -> dict[tuple[str, int], dict[str, str]]:
    frame = pd.read_parquet(path)
    lookup: dict[tuple[str, int], dict[str, str]] = {}
    pattern = re.compile(r"_PET_FTP_(B\d+)_(\d+)", re.IGNORECASE)
    for row in frame.itertuples(index=False):
        match = pattern.search(str(row.ftp_path))
        if match is None:
            continue
        key = (match.group(1).upper(), int(match.group(2)))
        if key in lookup:
            raise ImageReadinessBuildError("AMBIGUOUS_SELECTED_FTP_TRANSFORM")
        lookup[key] = {
            "scan_key": str(row.scan_key),
            "subject_key": str(row.subject_key),
        }
    return lookup


def _build_tau_membership(
    branch_config: dict[str, Any], scanner_path: Path
) -> pd.DataFrame:
    endpoint = pd.read_parquet(_require_file(_path(branch_config["endpoint"]), "TAU_ENDPOINT"))
    endpoint = endpoint.set_index("patient_hash", drop=False)
    scanner = _scanner_table(scanner_path, "tau").set_index("patient_hash")
    model_columns = _model_columns(branch_config)

    source_config_path = _require_file(
        _path(branch_config["legacy_source_config"]), "TAU_SOURCE_CONFIG"
    )
    source_config = yaml.safe_load(source_config_path.read_text(encoding="utf-8"))
    data_root = Path(source_config["data_sources"]["data_root"])
    tau_table = data_root / source_config["data_sources"]["tau_suvr_csv"]
    first_tau, _ = load_first_tau_scans(tau_table, source_config)

    transform_lookup = _private_transform_lookup(
        _require_file(
            _path(branch_config["transform_eligible_manifest"]),
            "FTP_TRANSFORM_ELIGIBILITY",
        )
    )
    transform_manifest = read_json(
        _require_file(_path(branch_config["transform_manifest"]), "FTP_TRANSFORM_MANIFEST")
    )
    transform_rows = {
        str(row.get("scan_key")): row
        for row in transform_manifest.get("rows", [])
        if isinstance(row, dict)
    }
    tracer = pd.read_parquet(
        _require_file(_path(branch_config["tracer_consistency"]), "FTP_TRACER_AUDIT")
    ).set_index("scan_key", drop=False)
    quantitative = pd.read_parquet(
        _path("results/method_y_revival_v2/provenance/scan_level_tau_quantitative_validation.parquet")
    ).set_index("scan_key", drop=False)
    transform_root = _path(branch_config["transform_root"])

    rows: list[dict[str, Any]] = []
    for key, source in sorted(first_tau.items(), key=lambda item: patient_hash_for_key(item[0])):
        patient_hash = patient_hash_for_key(key)
        if patient_hash not in endpoint.index:
            continue
        selected = transform_lookup.get((str(key[1]).upper(), int(float(source["viscode"]))))
        if selected is None:
            continue
        scan_key = selected["scan_key"]
        if scan_key not in tracer.index or scan_key not in transform_rows:
            continue
        tracer_row = tracer.loc[scan_key]
        if str(tracer_row["decision"]) != "ELIGIBLE_FOR_NEXT_GATE":
            continue
        transform_row = transform_rows[scan_key]
        output_path = transform_root / scan_key / "pet_mni.nii.gz"
        quant_row = quantitative.loc[scan_key] if scan_key in quantitative.index else None
        finite = bool(
            quant_row is not None and float(quant_row.get("finite_fraction", 0.0)) == 1.0
        )
        geometry_valid = bool(
            output_path.is_file()
            and str(transform_row.get("qc_status", "")) == "PASS"
            and transform_row.get("composed_output_sha256")
        )
        endpoint_row = endpoint.loc[patient_hash]
        manufacturer = (
            str(scanner.loc[patient_hash, "manufacturer"]).upper()
            if patient_hash in scanner.index
            else "UNKNOWN"
        )
        rows.append(
            _attach_endpoint_fields(
                {
                    "patient_hash": patient_hash,
                    "image_key": scan_key,
                    "image_storage_kind": "NIFTI_TRANSFORM_OUTPUT",
                    "image_path_private": str(output_path),
                    "h5_group": "",
                    "source_image_sha256": str(
                        transform_row.get("composed_output_sha256", "")
                    ),
                    "tracer": "FTP",
                    "manufacturer": manufacturer,
                    "qc_status": "PASS" if geometry_valid and finite else "EXCLUDED",
                    "image_geometry_valid": geometry_valid,
                    "image_finite": finite,
                    "role": "development",
                    "input_interpretation": "PATTERN_NORMALIZED_FTP_NOT_CENTAUR_QUANTITATIVE",
                },
                endpoint_row,
                model_columns,
            )
        )
    frame = pd.DataFrame(rows).sort_values("patient_hash", kind="stable")
    if frame["patient_hash"].duplicated().any():
        raise ImageReadinessBuildError("TAU_MEMBERSHIP_NOT_ONE_IMAGE_PER_SUBJECT")
    return frame.reset_index(drop=True)


def _quantitative_equivalence_passed(path: Path) -> bool:
    payload = read_json(path)
    return bool(
        payload.get("image_training_allowed") is True
        and payload.get("science_gate_status") == "PASS"
    )


def _scanner_support_passed(report: dict[str, Any]) -> bool:
    supported = [
        record
        for manufacturer, record in report.get("manufacturer_support", {}).items()
        if manufacturer != "UNKNOWN"
        and int(record.get("subjects", 0)) >= 30
        and int(record.get("events", 0)) >= 10
    ]
    return len(supported) >= 2


def _write_or_match_contract(path: Path, contract: RunContract) -> None:
    if path.exists():
        existing = RunContract.from_mapping(read_json(path)["payload"])
        existing.require_match(contract)
        return
    atomic_write_json(path, {"digest": contract.digest, "payload": contract.payload})


def build(config_path: Path, *, branches: tuple[str, ...]) -> dict[str, Any]:
    config_path = _require_file(config_path, "IMAGE_TRAINING_CONFIG")
    config = bind_software_runtime_contract(
        yaml.safe_load(config_path.read_text(encoding="utf-8")), ROOT
    )
    if config.get("schema_version") not in {"3.0", "3.2"}:
        raise ImageReadinessBuildError("IMAGE_TRAINING_CONFIG_SCHEMA_MISMATCH")
    if _path(config["project_root"]).resolve() != ROOT.resolve() or ROOT.drive.upper() != "H:":
        raise ImageReadinessBuildError("PROJECT_ROOT_MUST_BE_CANONICAL_H_CHECKOUT")
    output_root = _path(config["output_root"])
    if output_root.drive.upper() != "H:":
        raise ImageReadinessBuildError("OUTPUT_ROOT_MUST_BE_ON_H_DRIVE")
    if (
        output_root.resolve() == (_path("results/method_y_revival_v3_image")).resolve()
        or output_root.name == "method_y_revival_v3_image"
    ):
        raise ImageReadinessBuildError(
            "SUPERSEDED_OUTPUT_ROOT_CANNOT_BE_USED:results/method_y_revival_v3_image"
        )

    shared = config["shared"]
    scanner_path = _require_file(_path(shared["scanner_linkage"]), "SCANNER_LINKAGE")
    inputs: dict[str, str] = {"config": _sha256_file(config_path), "scanner": _sha256_file(scanner_path)}
    for branch in branches:
        branch_config = config["branches"][branch]
        for key in ("endpoint", "endpoint_contract", "splits", "quantitative_validation"):
            path = _require_file(_path(branch_config[key]), f"{branch.upper()}_{key.upper()}")
            inputs[f"{branch}:{key}"] = _sha256_file(path)
        if branch == "amyloid":
            for key in ("source_cache", "source_cache_manifest", "private_sample_mapping"):
                path = _require_file(_path(branch_config[key]), f"AMYLOID_{key.upper()}")
                inputs[f"amyloid:{key}"] = _sha256_file(path)
        else:
            for key in (
                "legacy_source_config",
                "transform_eligible_manifest",
                "transform_manifest",
                "tracer_consistency",
            ):
                path = _require_file(_path(branch_config[key]), f"TAU_{key.upper()}")
                inputs[f"tau:{key}"] = _sha256_file(path)
    contract = RunContract.from_mapping(
        {
            "schema_version": "3.0",
            "stage": "IMAGE_MEMBERSHIP_LOCK",
            "branches": list(branches),
            "seed": int(config["seed"]),
            "inputs": inputs,
            "model_features": {
                branch: list(_model_columns(config["branches"][branch]))
                for branch in branches
            },
            "model_config": config.get("model", {}),
            "runtime_policy": config.get("runtime", {}),
            "software_runtime_contract": config.get("software_runtime_contract", {}),
        }
    )

    output_root.mkdir(parents=True, exist_ok=True)
    lock_path = output_root / ".readiness.writer.lock"
    with WriterLock(lock_path, output_root=output_root, run_contract_digest=contract.digest):
        _write_or_match_contract(output_root / "run_contract.json", contract)
        private_root = output_root / "private"
        readiness_root = output_root / "readiness"
        private_root.mkdir(parents=True, exist_ok=True)
        readiness_root.mkdir(parents=True, exist_ok=True)
        branch_reports: dict[str, Any] = {}
        membership_hashes: dict[str, str] = {}
        for branch in branches:
            branch_config = config["branches"][branch]
            complete = (
                _build_amyloid_membership(branch_config, scanner_path)
                if branch == "amyloid"
                else _build_tau_membership(branch_config, scanner_path)
            )
            development = complete.loc[complete["role"] == "development"].copy()
            splits = read_json(_path(branch_config["splits"]))
            audit = audit_membership(
                development,
                splits=splits,
                branch=branch,
                minimum_outer_events=int(shared["minimum_outer_test_events"]),
                minimum_inner_events=int(shared["minimum_inner_validation_events"]),
            )
            endpoint_locked, endpoint_is_future = _future_endpoint_contract(
                _path(branch_config["endpoint_contract"])
            )
            quantitative_passed = _quantitative_equivalence_passed(
                _path(branch_config["quantitative_validation"])
            )
            scanner_support_available = _scanner_support_passed(audit)
            scanner_robustness_proven = False
            policy = assess_image_training_policy(
                integrity={
                    **audit,
                    "endpoint_locked": endpoint_locked,
                    "endpoint_is_future": endpoint_is_future,
                    "split_locked": True,
                    "outer_fold_count": 5,
                },
                quantitative_equivalence_proven=quantitative_passed,
                scanner_robustness_proven=scanner_robustness_proven,
                external_validation_available=False,
                incremental_value_demonstrated=False,
            )
            private_path = private_root / f"{branch}_image_membership.parquet"
            atomic_write_parquet(private_path, complete)
            private_sha = _sha256_file(private_path)
            membership_hashes[branch] = private_sha
            transport = complete.loc[complete["role"] == "internal_transport"]
            report = {
                "schema_version": "3.0",
                "status": (
                    "TRAINING_EXECUTION_READY"
                    if policy["training_execution_allowed"]
                    else "TRAINING_EXECUTION_BLOCKED"
                ),
                "branch": branch,
                "membership_audit": audit,
                "policy": policy,
                "private_membership_sha256": private_sha,
                "private_membership_path_published": False,
                "internal_transport_subject_count": int(len(transport)),
                "internal_transport_event_count": int(transport["event"].sum()) if len(transport) else 0,
                "quantitative_equivalence_proven": quantitative_passed,
                "scanner_support_available_for_prespecified_analysis": scanner_support_available,
                "scanner_support_gate_passed": scanner_support_available,
                "scanner_robustness_proven": scanner_robustness_proven,
                "external_validation_status": "EXTERNAL_PENDING_NONBLOCKING_FOR_INTERNAL_TRAINING",
                "raw_identifiers_exported": False,
                "raw_paths_exported": False,
            }
            atomic_write_json(readiness_root / f"{branch}_image_readiness.json", report)
            branch_reports[branch] = report

        # Source artifact provenance for cache binding (Plan 010)
        source_artifact_sha256: dict[str, str] = {}
        for branch in branches:
            branch_config = config["branches"][branch]
            if branch == "amyloid":
                src_path = _path(branch_config["source_cache"])
                source_artifact_sha256[branch] = _sha256_file(src_path) if src_path.is_file() else "UNKNOWN_SOURCE"
            else:
                # Tau source is transform manifest (hash of manifest file) as proxy for transform outputs
                src_path = _path(branch_config["transform_manifest"])
                source_artifact_sha256[branch] = _sha256_file(src_path) if src_path.is_file() else "UNKNOWN_SOURCE"
        overall_ready = all(
            report["policy"]["training_execution_allowed"]
            for report in branch_reports.values()
        )
        summary = {
            "schema_version": "3.0",
            "status": "READY_FOR_FINAL_INTERNAL_IMAGE_TRAINING" if overall_ready else "BLOCKED",
            "training_execution_allowed": overall_ready,
            "positive_claim_allowed": False,
            "external_claim_allowed": False,
            "branches": {
                branch: {
                    "status": report["status"],
                    "eligible_subject_count": report["membership_audit"]["eligible_subject_count"],
                    "eligible_event_count": report["membership_audit"]["eligible_event_count"],
                    "training_execution_allowed": report["policy"]["training_execution_allowed"],
                    "positive_claim_allowed": report["policy"]["positive_claim_allowed"],
                    "claim_limitations": report["policy"]["claim_limitations"],
                }
                for branch, report in branch_reports.items()
            },
            "run_contract_sha256": contract.digest,
            "private_membership_sha256": membership_hashes,
            "source_artifact_sha256": source_artifact_sha256,
            "claim_boundary": "Internal image training and locked internal outer-CV are allowed; quantitative-standard, external-generalization, diagnostic and causal claims remain prohibited until their independent gates pass.",
        }
        atomic_write_json(readiness_root / "FINAL_IMAGE_TRAINING_READINESS.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build locked Method Y v3 image-training memberships and dual execution/claim gates."
    )
    parser.add_argument(
        "--config",
        default=str(ROOT / "configs/method_y_image_training_v3.yaml"),
    )
    parser.add_argument(
        "--branch", choices=("amyloid", "tau", "both"), default="both"
    )
    args = parser.parse_args()
    branches = ("amyloid", "tau") if args.branch == "both" else (args.branch,)
    try:
        summary = build(Path(args.config), branches=branches)
    except (ImageReadinessBuildError, RunContractMismatch, OSError, ValueError) as exc:
        print(json.dumps({"status": "BLOCKED", "reason": str(exc)}, indent=2), flush=True)
        return 2
    print(json.dumps(summary, indent=2), flush=True)
    return 0 if summary["training_execution_allowed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
