"""Build Method Y v2 public source-regional feature contracts; never trains."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
from pathlib import Path

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.method_y_revival.atomic_io import atomic_write_json
from src.method_y_revival.branch_readiness import build_branch_readiness, write_branch_readiness
from src.method_y_revival.source_regional_features import (
    SourceRegionalFeatureError,
    build_feature_contract,
    build_paired_feature_contract,
    build_subject_level_centiloid_contract,
    write_public_feature_contract,
    TAU_PERSI_REGION_ALLOWLIST_V1,
    TAU_PERSI_EXCLUDED_VOCABULARY_V1,
    TAU_PERSI_PRIMARY_ALLOWLIST_V1,
)
from src.method_y_revival.contracts import sha256_mapping
from src.utils.privacy import stable_hash


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _read_csv_snapshot(path: Path, *, usecols: list[str] | None = None) -> tuple[pd.DataFrame, str]:
    """Read each source table once, binding parsing and SHA to the same bytes."""
    raw = path.read_bytes()
    return pd.read_csv(io.BytesIO(raw), usecols=usecols, low_memory=False), hashlib.sha256(raw).hexdigest()


def _read_locked_endpoint(root: Path, stem: str, *, required_source_hashes: dict[str, str]) -> tuple[dict, pd.DataFrame]:
    """Bind membership/value reads to the exact locked endpoint byte contract."""
    contract_path = root / "endpoints" / f"{stem}_endpoint_contract.json"
    contract_raw = contract_path.read_bytes()
    payload = json.loads(contract_raw.decode("utf-8-sig"))
    stored = payload.get("contract_sha256")
    recomputed = sha256_mapping({key: value for key, value in payload.items() if key != "contract_sha256"})
    if payload.get("locked") is not True or stored != recomputed:
        raise RuntimeError(f"LOCKED_{stem.upper()}_ENDPOINT_CONTRACT_INVALID")
    source_hashes = payload.get("source_hashes", {})
    if any(source_hashes.get(key) != digest for key, digest in required_source_hashes.items()):
        raise RuntimeError(f"LOCKED_{stem.upper()}_ENDPOINT_SOURCE_HASH_MISMATCH")
    artifact = root / "endpoints" / str(payload.get("endpoint_path", ""))
    artifact_raw = artifact.read_bytes() if artifact.name == payload.get("endpoint_path") and artifact.is_file() else b""
    if not artifact_raw or hashlib.sha256(artifact_raw).hexdigest() != payload.get("endpoint_sha256"):
        raise RuntimeError(f"LOCKED_{stem.upper()}_ENDPOINT_ARTIFACT_HASH_MISMATCH")
    frame = pd.read_parquet(io.BytesIO(artifact_raw))
    if len(frame) != payload.get("row_count") or "patient_hash" not in frame.columns:
        raise RuntimeError(f"LOCKED_{stem.upper()}_ENDPOINT_ROW_COUNT_OR_SCHEMA_MISMATCH")
    if frame["patient_hash"].isna().any():
        raise RuntimeError(f"LOCKED_{stem.upper()}_ENDPOINT_MEMBERSHIP_INVALID")
    membership = frame["patient_hash"].astype(str)
    if membership.eq("").any() or membership.eq("nan").any() or membership.duplicated().any() or membership.nunique() != len(frame):
        raise RuntimeError(f"LOCKED_{stem.upper()}_ENDPOINT_MEMBERSHIP_INVALID")
    if "event_count" in payload:
        if "event" not in frame.columns:
            raise RuntimeError(f"LOCKED_{stem.upper()}_ENDPOINT_EVENT_COUNT_MISMATCH")
        events = pd.to_numeric(frame["event"], errors="coerce")
        if events.isna().any() or not np.isfinite(events.to_numpy()).all() or not np.equal(events.to_numpy(), np.floor(events.to_numpy())).all() or not set(events.astype(int)).issubset({0, 1}) or int(events.sum()) != payload["event_count"]:
            raise RuntimeError(f"LOCKED_{stem.upper()}_ENDPOINT_EVENT_COUNT_MISMATCH")
    return payload, frame


def _sanitized_tau_column(region: str) -> str:
    import re
    return "tau_persi__" + re.sub(r"[^a-zA-Z0-9]+", "_", region.strip().lower()).strip("_")


def _validated_negative_result(path: Path) -> dict[str, str]:
    raw = path.read_bytes()
    payload = json.loads(raw.decode("utf-8-sig"))
    if payload.get("method") != "paired_subject_bootstrap_over_locked_oof_predictions" or payload.get("status") != "COMPLETED" or payload.get("cache_status") != "COMPUTED_AND_LOCKED":
        raise RuntimeError("PAIRED_V1_NEGATIVE_ARTIFACT_SEMANTICS_INVALID")
    if (payload.get("subject_count"), payload.get("requested_iterations"), payload.get("successful_iterations"), payload.get("failed_attempts")) != (369, 2000, 2000, 0):
        raise RuntimeError("PAIRED_V1_NEGATIVE_ARTIFACT_COMPLETENESS_INVALID")
    if not isinstance(payload.get("input_evidence_sha256"), str) or len(payload["input_evidence_sha256"]) != 64:
        raise RuntimeError("PAIRED_V1_NEGATIVE_INPUT_EVIDENCE_INVALID")
    ibs, cindex = payload.get("delta_ibs_95ci"), payload.get("delta_cindex_95ci")
    if not (isinstance(ibs, list) and len(ibs) == 2 and isinstance(cindex, list) and len(cindex) == 2):
        raise RuntimeError("PAIRED_V1_NEGATIVE_INTERVALS_INVALID")
    try:
        ibs = [float(value) for value in ibs]
        cindex = [float(value) for value in cindex]
    except (TypeError, ValueError):
        raise RuntimeError("PAIRED_V1_NEGATIVE_INTERVALS_INVALID")
    if not all(np.isfinite(value) for value in [*ibs, *cindex]) or ibs[0] > ibs[1] or cindex[0] > cindex[1]:
        raise RuntimeError("PAIRED_V1_NEGATIVE_INTERVALS_INVALID")
    positive_gate = float(ibs[1]) < 0.0 and float(cindex[0]) >= 0.0
    if positive_gate:
        raise RuntimeError("PAIRED_V1_ARTIFACT_IS_NOT_NEGATIVE")
    conclusion = "TAU_INCREMENTAL_VALUE_NOT_DEMONSTRATED"
    return {"artifact_basename": path.name, "sha256": hashlib.sha256(raw).hexdigest(), "input_evidence_sha256": payload["input_evidence_sha256"], "conclusion": conclusion, "conclusion_sha256": hashlib.sha256(conclusion.encode()).hexdigest()}


def _blocked(branch: str, *, source_basename: str, source_sha256: str, endpoint_sha256: str, blocker: str) -> dict:
    payload = {
        "schema_version": "1.0", "branch": branch, "status": "BLOCKED", "locked": False,
        "blockers": [blocker], "source": {"source_basename": source_basename, "source_sha256": source_sha256},
        "endpoint_contract_sha256": endpoint_sha256,
        "forbidden_dependencies": ["same_pet_derived_target", "fuzzy_roi_mapping", "image_voxels", "image_cache", "image_scale", "future_outcomes"],
    }
    payload["contract_sha256"] = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return payload


def replace_current_feature_evidence(evidence: dict, current: dict[str, dict], output: Path) -> dict:
    """Remove stale feature PASS entries before accepting this run's contracts."""
    refreshed = dict(evidence)
    for branch, payload in current.items():
        artifact_id = f"{branch}_feature_contract"
        refreshed.pop(artifact_id, None)
        if payload.get("locked") is True:
            refreshed[artifact_id] = {"artifact_id": artifact_id, "status": "PASS", "sha256": _sha(output / f"{branch}_feature_contract.json")}
    return refreshed


def _main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--aggregate-root", type=Path, required=True)
    args = parser.parse_args()
    output = args.project_root / "results" / "method_y_revival_v2" / "feature_contracts"
    state_path = output / "source_regional_validation_state.json"
    atomic_write_json(state_path, {"status": "VALIDATING", "development_allowed": False, "reason": "CURRENT_ARTIFACTS_NOT_COMMITTED"})
    raw = args.raw_root / "Clinical" / "External Data"
    aggregate = args.aggregate_root / "results" / "method_y_revival_v2"
    amyloid_source = args.raw_root / "Clinical" / "Derived Data" / "SUBJINFO.csv"
    tau_source = raw / "imaging_SUVR_tau.csv"
    # The documented continuous Centiloid source is SUBJINFO.AMYLCENT.  It is
    # linked privately to the already locked endpoint only via the exact stable
    # subject hash, then all row-level information is discarded.
    amyloid_rows, amyloid_sha = _read_csv_snapshot(amyloid_source, usecols=["SUBSTUDY", "BID", "AMYLCENT"])
    amyloid_endpoint_payload, amyloid_endpoint_frame = _read_locked_endpoint(aggregate, "amyloid", required_source_hashes={"subjinfo_csv": amyloid_sha})
    amyloid_endpoint = amyloid_endpoint_payload["contract_sha256"]
    endpoint_membership = set(amyloid_endpoint_frame["patient_hash"].astype(str))
    amyloid_rows["subject_key"] = amyloid_rows.apply(lambda row: stable_hash(f"{str(row['SUBSTUDY']).upper()}|{str(row['BID']).strip()}", prefix="pt"), axis=1)
    amyloid_rows = amyloid_rows.loc[amyloid_rows["subject_key"].isin(endpoint_membership)].copy()
    if set(amyloid_rows["subject_key"]) != endpoint_membership:
        raise RuntimeError("AMYLOID_ENDPOINT_SOURCE_MEMBERSHIP_MISMATCH")
    if amyloid_rows["subject_key"].duplicated().any():
        raise RuntimeError("AMYLOID_ENDPOINT_SOURCE_DUPLICATE")
    amyloid_values = amyloid_rows.set_index("subject_key")["AMYLCENT"].astype(float)
    endpoint_values = amyloid_endpoint_frame.set_index("patient_hash")["baseline_amyloid_centiloid"].astype(float).reindex(amyloid_values.index)
    if endpoint_values.isna().any() or not np.allclose(amyloid_values.to_numpy(), endpoint_values.to_numpy(), rtol=0.0, atol=1e-10, equal_nan=False):
        raise RuntimeError("AMYLOID_ENDPOINT_SOURCE_VALUE_MISMATCH")
    try:
        amyloid, amyloid_matrix = build_subject_level_centiloid_contract(
            rows=amyloid_rows, source_provenance={"source_basename": amyloid_source.name, "source_sha256": amyloid_sha}, endpoint_contract_sha256=amyloid_endpoint,
        )
    except SourceRegionalFeatureError as error:
        amyloid = _blocked("amyloid_source_regional", source_basename=amyloid_source.name, source_sha256=amyloid_sha, endpoint_sha256=amyloid_endpoint, blocker=str(error))
        amyloid_matrix = pd.DataFrame(columns=["subject_key"])
    write_public_feature_contract(amyloid, output, "amyloid_source_regional")

    usecols = ["SUBSTUDY", "BID", "VISCODE", "scan_number", "scan_date_DAYS_T0", "scan_analyzed", "ligand", "brain_region", "suvr_persi"]
    tau_rows, tau_sha = _read_csv_snapshot(tau_source, usecols=usecols)
    expected_vocabulary = set(TAU_PERSI_REGION_ALLOWLIST_V1).union(TAU_PERSI_EXCLUDED_VOCABULARY_V1)
    observed_vocabulary = set(tau_rows["brain_region"].dropna().astype(str))
    if observed_vocabulary != expected_vocabulary:
        raise RuntimeError("TAU_SOURCE_VOCABULARY_REGISTRY_MISMATCH")
    tau_endpoint_payload, tau_endpoint_frame = _read_locked_endpoint(aggregate, "tau", required_source_hashes={"tau_suvr_csv": tau_sha, "subjinfo_csv": amyloid_sha})
    paired_fbp_sha = _sha(raw / "imaging_SUVR_amyloid.csv")
    paired_endpoint_payload, paired_endpoint_frame = _read_locked_endpoint(aggregate, "paired_at", required_source_hashes={"amyloid_suvr_csv": paired_fbp_sha, "tau_suvr_csv": tau_sha, "subjinfo_csv": amyloid_sha})
    tau_endpoint = tau_endpoint_payload["contract_sha256"]
    paired_endpoint = paired_endpoint_payload["contract_sha256"]
    if not tau_rows["ligand"].eq("Flortaucipir").all():
        raise RuntimeError("TAU_FTP_LIGAND_IDENTITY_MISMATCH")
    tau_rows = tau_rows.loc[tau_rows["scan_analyzed"].eq("Yes")].copy()
    tau_rows["subject_key"] = tau_rows.apply(lambda row: stable_hash(f"{str(row['SUBSTUDY']).upper()}|{str(row['BID']).strip()}", prefix="pt"), axis=1)
    tau_rows = tau_rows.rename(columns={"VISCODE": "visit_code", "scan_date_DAYS_T0": "scan_date_days_t0"})
    tau_membership = set(tau_endpoint_frame["patient_hash"].astype(str))
    tau_rows = tau_rows.loc[tau_rows["subject_key"].isin(tau_membership)].copy()
    allowed = list(TAU_PERSI_PRIMARY_ALLOWLIST_V1)
    try:
        tau, tau_matrix = build_feature_contract(
            branch="tau_source_regional", rows=tau_rows,
            source_provenance={"source_basename": tau_source.name, "source_sha256": tau_sha,
                               "baseline_visit_code": 4, "reference_region": "PERSI_white_matter",
                               "quantitative_contract_id": "linear_normalization_persi_white_matter_v1",
                               "reference_dictionary_basename": "imaging_SUVR_methods.pdf",
                               "reference_dictionary_sha256": _sha(args.raw_root / "Clinical" / "Documents" / "Methods" / "imaging_SUVR_methods.pdf"),
                               "excluded_region_policy": {"registry_id": "TAU_PERSI_REGION_VOCABULARY_V1", "source_vocabulary_sha256": hashlib.sha256("\n".join(sorted(expected_vocabulary)).encode()).hexdigest(), "primary_allowlist_sha256": hashlib.sha256("\n".join(allowed).encode()).hexdigest(), "explicit_exclusions": ["DELTA_Z 50PCT N141", "reference_regions", "custom_masks", "VOI_regions", "Caudate_L", "Caudate_R", "Pallidum_L", "Pallidum_R", "Putamen_L", "Putamen_R", "Thalamus_L", "Thalamus_R"]},
                               "allowed_region_names": allowed}, endpoint_contract_sha256=tau_endpoint,
        )
    except SourceRegionalFeatureError as error:
        tau = _blocked("tau_source_regional", source_basename=tau_source.name, source_sha256=tau_sha, endpoint_sha256=tau_endpoint, blocker=str(error))
        tau_matrix = pd.DataFrame(columns=["subject_key"])
    if tau.get("locked") is True and int(tau.get("selected_subject_count", -1)) != len(tau_membership):
        raise RuntimeError("TAU_LOCKED_ENDPOINT_MEMBERSHIP_MISMATCH")
    if tau.get("locked") is True:
        if set(tau_matrix["subject_key"]) != tau_membership:
            raise RuntimeError("TAU_ENDPOINT_SOURCE_MEMBERSHIP_MISMATCH")
        endpoint_by_subject = tau_endpoint_frame.set_index("patient_hash")
        for region in allowed:
            endpoint_column = _sanitized_tau_column(region)
            if endpoint_column not in endpoint_by_subject.columns:
                raise RuntimeError("TAU_ENDPOINT_SOURCE_FEATURE_SCHEMA_MISMATCH")
            observed = pd.to_numeric(tau_matrix.set_index("subject_key")[region], errors="coerce")
            expected = pd.to_numeric(endpoint_by_subject[endpoint_column], errors="coerce").reindex(observed.index)
            if observed.isna().any() or expected.isna().any() or not np.allclose(observed.to_numpy(), expected.to_numpy(), rtol=0.0, atol=1e-10):
                raise RuntimeError("TAU_ENDPOINT_SOURCE_VALUE_MISMATCH")
    write_public_feature_contract(tau, output, "tau_source_regional")

    negative = aggregate.parent / "method_y_revival_v1" / "reports" / "paired_at_tau_beyond_amyloid_bootstrap.json"
    negative_reference = _validated_negative_result(negative)
    if "paired_fbp_ftp_gap_days" not in paired_endpoint_frame.columns:
        raise RuntimeError("LOCKED_PAIRED_ENDPOINT_GAP_COLUMN_REQUIRED")
    paired_gaps = pd.to_numeric(paired_endpoint_frame["paired_fbp_ftp_gap_days"], errors="coerce")
    if paired_gaps.isna().any() or bool((paired_gaps > 183).any()):
        raise RuntimeError("LOCKED_PAIRED_ENDPOINT_DATE_WINDOW_VIOLATION")
    paired_membership = set(paired_endpoint_frame["patient_hash"].astype(str))
    pair = set(amyloid_matrix.get("subject_key", [])).intersection(set(tau_matrix.get("subject_key", []))).intersection(paired_membership)
    if pair != paired_membership:
        raise RuntimeError("PAIRED_ENDPOINT_SOURCE_MEMBERSHIP_MISMATCH")
    paired_by_subject = paired_endpoint_frame.set_index("patient_hash")
    paired_amyloid = pd.to_numeric(paired_by_subject["baseline_amyloid_centiloid"], errors="coerce").reindex(sorted(pair))
    source_amyloid = pd.to_numeric(amyloid_matrix.set_index("subject_key")["AMYLCENT"], errors="coerce").reindex(sorted(pair))
    if paired_amyloid.isna().any() or source_amyloid.isna().any() or not np.allclose(paired_amyloid.to_numpy(), source_amyloid.to_numpy(), rtol=0.0, atol=1e-10):
        raise RuntimeError("PAIRED_AMYLOID_SOURCE_VALUE_MISMATCH")
    tau_by_subject = tau_matrix.set_index("subject_key")
    for region in TAU_PERSI_PRIMARY_ALLOWLIST_V1:
        column = _sanitized_tau_column(region)
        if column not in paired_by_subject.columns:
            raise RuntimeError("PAIRED_TAU_SOURCE_FEATURE_SCHEMA_MISMATCH")
        paired_values = pd.to_numeric(paired_by_subject[column], errors="coerce").reindex(sorted(pair))
        source_values = pd.to_numeric(tau_by_subject[region], errors="coerce").reindex(sorted(pair))
        if paired_values.isna().any() or source_values.isna().any() or not np.allclose(paired_values.to_numpy(), source_values.to_numpy(), rtol=0.0, atol=1e-10):
            raise RuntimeError("PAIRED_TAU_SOURCE_VALUE_MISMATCH")
    try:
        paired = build_paired_feature_contract(
            amyloid_contract=amyloid, tau_contract=tau, paired_endpoint_contract_sha256=paired_endpoint,
            negative_result_reference=negative_reference,
            paired_subject_count=len(pair), paired_date_window_days=183,
            paired_membership_digest=hashlib.sha256("\n".join(sorted(pair)).encode("utf-8")).hexdigest(),
            paired_membership_rule="exact_private_hash_join_of_locked_baselines_and_locked_paired_endpoint_with_verified_gap_le_183_days",
        )
    except (SourceRegionalFeatureError, OSError) as error:
        paired = _blocked("paired_source_regional", source_basename="locked_source_regional_contracts", source_sha256=hashlib.sha256((amyloid["contract_sha256"] + tau["contract_sha256"]).encode()).hexdigest(), endpoint_sha256=paired_endpoint, blocker=str(error))
        paired["negative_result_reference"] = negative_reference
        paired["negative_result_preserved"] = True
    write_public_feature_contract(paired, output, "paired_source_regional")

    # Refresh from the prior aggregate evidence snapshot, then bind only LOCKED
    # public contracts from this run.  No image cache or external outcome opens.
    previous = _read_json(args.project_root / "results" / "method_y_revival_v2" / "readiness" / "branch_readiness_v2.json")
    evidence = {item["artifact_id"]: item for branch in previous["branches"].values() for item in branch.get("evidence", [])}
    evidence = replace_current_feature_evidence(evidence, {"amyloid_source_regional": amyloid, "tau_source_regional": tau, "paired_source_regional": paired}, output)
    readiness = build_branch_readiness(evidence, source_role="read_only_project_aggregate_contracts")
    write_branch_readiness(readiness, args.project_root / "results" / "method_y_revival_v2" / "readiness")
    atomic_write_json(output / "source_regional_contract_summary.json", {"amyloid_status": amyloid["status"], "tau_status": tau["status"], "paired_status": paired["status"], "training_started": False})
    artifact_paths = [output / f"{branch}_feature_contract.json" for branch in ("amyloid_source_regional", "tau_source_regional", "paired_source_regional")]
    artifact_paths += [args.project_root / "results" / "method_y_revival_v2" / "readiness" / "branch_readiness_v2.json"]
    atomic_write_json(state_path, {"status": "COMMITTED", "development_allowed": True, "artifact_sha256": {path.name: _sha(path) for path in artifact_paths}})
    return 0


def main() -> int:
    """Fail closed: a rebuild error replaces all current feature PASS evidence."""
    try:
        return _main()
    except Exception as error:
        # Avoid retaining a previous PASS after a changed/tampered source or endpoint.
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument("--project-root", type=Path, required=True)
        args, _ = parser.parse_known_args()
        output = args.project_root / "results" / "method_y_revival_v2" / "feature_contracts"
        blocked = {
            branch: _blocked(branch, source_basename="validation_failed", source_sha256="0" * 64, endpoint_sha256="0" * 64, blocker=f"REBUILD_VALIDATION_FAILED:{type(error).__name__}")
            for branch in ("amyloid_source_regional", "tau_source_regional", "paired_source_regional")
        }
        for branch, payload in blocked.items():
            write_public_feature_contract(payload, output, branch)
        atomic_write_json(output / "source_regional_contract_summary.json", {"amyloid_status": "BLOCKED", "tau_status": "BLOCKED", "paired_status": "BLOCKED", "training_started": False})
        atomic_write_json(output / "source_regional_validation_state.json", {"status": "BLOCKED", "development_allowed": False, "reason": f"REBUILD_VALIDATION_FAILED:{type(error).__name__}"})
        readiness_path = args.project_root / "results" / "method_y_revival_v2" / "readiness" / "branch_readiness_v2.json"
        if readiness_path.is_file():
            previous = _read_json(readiness_path)
            evidence = {item["artifact_id"]: item for record in previous["branches"].values() for item in record.get("evidence", [])}
            readiness = build_branch_readiness(replace_current_feature_evidence(evidence, blocked, output), source_role="read_only_project_aggregate_contracts")
            write_branch_readiness(readiness, readiness_path.parent)
        print(f"METHOD_Y_SOURCE_REGIONAL_BLOCKED:{type(error).__name__}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
