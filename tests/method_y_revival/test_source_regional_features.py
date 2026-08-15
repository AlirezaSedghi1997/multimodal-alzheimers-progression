from __future__ import annotations

import hashlib
import json

import pandas as pd
import pytest


def _provenance(name: str = "amyloid_source.csv") -> dict[str, str]:
    return {"source_basename": name, "source_sha256": "a" * 64, "baseline_visit_code": 2}


def test_amyloid_contract_requires_documented_continuous_centiloid_and_excludes_targets():
    from src.method_y_revival.source_regional_features import (
        SourceRegionalFeatureError,
        build_feature_contract,
    )

    rows = pd.DataFrame(
        {
            "subject_key": ["s1", "s2"],
            "visit_code": [2, 2],
            "scan_date_days_t0": [1, 2],
            "scan_analyzed": ["Yes", "Yes"],
            "AMYLCENT": [12.0, 24.0],
            "amyloid_positive": [0, 1],
        }
    )
    contract, _ = build_feature_contract(
        branch="amyloid_source_regional",
        rows=rows,
        source_provenance=_provenance(),
        endpoint_contract_sha256="b" * 64,
    )

    assert contract["status"] == "LOCKED"
    assert contract["feature_columns"] == ["AMYLCENT"]
    assert contract["feature_semantics"] == "SOURCE_CONTINUOUS_CENTILOID"
    assert "amyloid_positive" not in json.dumps(contract)

    with pytest.raises(SourceRegionalFeatureError, match="AMYLOID_CONTINUOUS_CENTILOID_REQUIRED"):
        build_feature_contract(
            branch="amyloid_source_regional",
            rows=rows.rename(columns={"AMYLCENT": "centiloid_derived_from_image"}),
            source_provenance=_provenance(),
            endpoint_contract_sha256="b" * 64,
        )


def test_tau_contract_uses_exact_source_suvr_names_never_centaur():
    from src.method_y_revival.source_regional_features import build_feature_contract

    rows = pd.DataFrame(
        {
            "subject_key": ["s1", "s2"],
            "visit_code": [2, 2],
            "scan_date_days_t0": [1, 2],
            "scan_analyzed": ["Yes", "Yes"],
            "bi_entorhinal": [1.1, 1.3],
            "bi_precuneus": [1.0, 1.2],
        }
    )
    contract, _ = build_feature_contract(
        branch="tau_source_regional",
        rows=rows,
        source_provenance={
            **_provenance("tau_stanford.csv"),
            "reference_region": "bi_Cerebellum.Cortex",
            "quantitative_contract_id": "stanford_freesurfer_non_pvc_gray_cerebellum_v1",
            "reference_dictionary_basename": "tau_methods.pdf",
            "reference_dictionary_sha256": "c" * 64,
        },
        endpoint_contract_sha256="b" * 64,
    )

    assert contract["status"] == "LOCKED"
    assert contract["feature_semantics"] == "SOURCE_REGIONAL_SUVR"
    assert contract["feature_columns"] == ["bi_entorhinal", "bi_precuneus"]
    assert "CenTauR" not in json.dumps(contract)


def test_tau_long_source_requires_exact_prespecified_regions_without_fuzzy_mapping():
    from src.method_y_revival.source_regional_features import build_feature_contract

    rows = pd.DataFrame(
        {
            "subject_key": ["s1", "s1", "s2", "s2"],
            "visit_code": [4, 4, 4, 4],
            "scan_date_days_t0": [1, 1, 2, 2],
            "scan_analyzed": ["Yes"] * 4,
            "scan_number": [1] * 4,
            "brain_region": ["Temporal_Mid_L", "Precuneus_R", "Temporal_Mid_L", "Precuneus_R"],
            "suvr_persi": [1.1, 1.2, 1.3, 1.4],
        }
    )
    contract, _ = build_feature_contract(
        branch="tau_source_regional",
        rows=rows,
        source_provenance={
            **_provenance("tau_long.csv"),
            "baseline_visit_code": 4,
            "reference_region": "whole_cerebellum_blcere_all",
            "quantitative_contract_id": "linear_normalization_whole_cerebellum_v1",
            "reference_dictionary_basename": "tau_methods.pdf",
            "reference_dictionary_sha256": "c" * 64,
            "allowed_region_names": ["Temporal_Mid_L", "Precuneus_R"],
        },
        endpoint_contract_sha256="b" * 64,
    )
    assert contract["feature_columns"] == ["Precuneus_R", "Temporal_Mid_L"]


def test_baseline_duplicate_tie_fails_closed():
    from src.method_y_revival.source_regional_features import (
        SourceRegionalFeatureError,
        build_feature_contract,
    )

    rows = pd.DataFrame(
        {
            "subject_key": ["s1", "s1"],
            "visit_code": [2, 2],
            "scan_date_days_t0": [1, 1],
            "scan_analyzed": ["Yes", "Yes"],
            "AMYLCENT": [12.0, 13.0],
        }
    )
    with pytest.raises(SourceRegionalFeatureError, match="AMBIGUOUS_BASELINE_ROW"):
        build_feature_contract(
            branch="amyloid_source_regional",
            rows=rows,
            source_provenance=_provenance(),
            endpoint_contract_sha256="b" * 64,
        )


def test_paired_contract_binds_negative_result_without_exposing_private_membership():
    from src.method_y_revival.source_regional_features import build_paired_feature_contract

    amyloid = {"feature_digest": "a" * 64, "baseline_row_digest": "b" * 64, "locked": True}
    tau = {"feature_digest": "c" * 64, "baseline_row_digest": "d" * 64, "locked": True}
    contract = build_paired_feature_contract(
        amyloid_contract=amyloid,
        tau_contract=tau,
        paired_endpoint_contract_sha256="e" * 64,
        negative_result_reference={"artifact_basename": "PAIRED_AT_RESULTS_V1.md", "sha256": "f" * 64, "input_evidence_sha256": "e" * 64, "conclusion": "TAU_INCREMENTAL_VALUE_NOT_DEMONSTRATED", "conclusion_sha256": hashlib.sha256(b"TAU_INCREMENTAL_VALUE_NOT_DEMONSTRATED").hexdigest()},
        paired_subject_count=369,
        paired_date_window_days=183,
        paired_membership_digest="0" * 64,
    )
    assert contract["status"] == "LOCKED"
    assert contract["negative_result_preserved"] is True
    assert "subject_key" not in json.dumps(contract)

    other = build_paired_feature_contract(
        amyloid_contract=amyloid, tau_contract=tau, paired_endpoint_contract_sha256="e" * 64,
        negative_result_reference={"artifact_basename": "PAIRED_AT_RESULTS_V1.md", "sha256": "f" * 64, "input_evidence_sha256": "e" * 64, "conclusion": "TAU_INCREMENTAL_VALUE_NOT_DEMONSTRATED", "conclusion_sha256": hashlib.sha256(b"TAU_INCREMENTAL_VALUE_NOT_DEMONSTRATED").hexdigest()},
        paired_subject_count=369, paired_date_window_days=183, paired_membership_digest="1" * 64,
    )
    assert other["contract_sha256"] != contract["contract_sha256"]


def test_public_contract_writer_is_atomic_and_rejects_private_payload(tmp_path):
    from src.method_y_revival.source_regional_features import (
        SourceRegionalFeatureError,
        write_public_feature_contract,
    )

    payload = {"status": "BLOCKED", "contract_sha256": hashlib.sha256(b"x").hexdigest(), "blockers": ["MISSING"]}
    write_public_feature_contract(payload, tmp_path, "amyloid_source_regional")
    assert (tmp_path / "amyloid_source_regional_feature_contract.json").is_file()
    assert (tmp_path / "amyloid_source_regional_feature_contract.md").is_file()
    with pytest.raises(SourceRegionalFeatureError, match="PRIVATE_FIELD"):
        write_public_feature_contract({**payload, "subject_key": "private"}, tmp_path, "bad")
    with pytest.raises(SourceRegionalFeatureError, match="PRIVATE_FIELD"):
        write_public_feature_contract({**payload, "nested": [{"patient_hash": "pt_private"}]}, tmp_path, "bad")
    with pytest.raises(SourceRegionalFeatureError, match="PRIVATE_FIELD"):
        write_public_feature_contract({**payload, "nested": ["B123456", {"value": "LRN_12345"}]}, tmp_path, "bad")


def test_subject_level_centiloid_binds_endpoint_without_inventing_scan_date():
    from src.method_y_revival.source_regional_features import build_subject_level_centiloid_contract
    from src.utils.privacy import stable_hash

    patient_hash = stable_hash("A4|B100", prefix="pt")
    contract, selected = build_subject_level_centiloid_contract(
        rows=pd.DataFrame({"subject_key": [patient_hash], "AMYLCENT": [42.0]}),
        source_provenance={"source_basename": "SUBJINFO.csv", "source_sha256": "a" * 64},
        endpoint_contract_sha256="b" * 64,
    )
    assert selected.subject_key.tolist() == [patient_hash]
    assert contract["baseline_selection"]["time_zero_rule"] == "bound_only_to_locked_endpoint_contract"
    assert "scan_date" not in json.dumps(contract)


def test_blocked_rerun_removes_stale_feature_pass_evidence(tmp_path):
    import importlib.util
    from pathlib import Path

    script = Path(__file__).resolve().parents[2] / "scripts" / "build_method_y_source_regional_contracts_v2.py"
    spec = importlib.util.spec_from_file_location("source_builder", script)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    artifact = tmp_path / "amyloid_source_regional_feature_contract.json"
    artifact.write_text("{}", encoding="utf-8")
    stale = {"amyloid_source_regional_feature_contract": {"artifact_id": "amyloid_source_regional_feature_contract", "status": "PASS", "sha256": "a" * 64}}
    refreshed = module.replace_current_feature_evidence(stale, {"amyloid_source_regional": {"locked": False}}, tmp_path)
    assert "amyloid_source_regional_feature_contract" not in refreshed


def test_negative_result_validator_rejects_a_positive_incremental_gate(tmp_path):
    import importlib.util
    from pathlib import Path

    script = Path(__file__).resolve().parents[2] / "scripts" / "build_method_y_source_regional_contracts_v2.py"
    spec = importlib.util.spec_from_file_location("source_builder_negative", script)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    payload = {"method": "paired_subject_bootstrap_over_locked_oof_predictions", "status": "COMPLETED", "cache_status": "COMPUTED_AND_LOCKED", "subject_count": 369, "requested_iterations": 2000, "successful_iterations": 2000, "failed_attempts": 0, "input_evidence_sha256": "a" * 64, "delta_ibs_95ci": [-0.2, -0.1], "delta_cindex_95ci": [0.1, 0.2]}
    path = tmp_path / "negative.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="NOT_NEGATIVE"):
        module._validated_negative_result(path)
    payload["delta_ibs_95ci"] = [float("nan"), 0.1]
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="INTERVALS_INVALID"):
        module._validated_negative_result(path)


def test_fold_local_contract_requires_inner_train_then_outer_refit():
    from src.method_y_revival.source_regional_features import build_subject_level_centiloid_contract

    contract, _ = build_subject_level_centiloid_contract(
        rows=pd.DataFrame({"subject_key": ["s1"], "AMYLCENT": [1.0]}),
        source_provenance={"source_basename": "SUBJINFO.csv", "source_sha256": "a" * 64},
        endpoint_contract_sha256="b" * 64,
    )
    policy = contract["fold_local_processing"]
    assert policy["inner_hpo_imputation"] == "fit_on_each_inner_train_only"
    assert policy["post_winner_refit"] == "fit_on_entire_outer_train_only"


def test_locked_endpoint_rejects_nonbinary_events_with_preserved_sum(tmp_path):
    import importlib.util
    from pathlib import Path
    from src.method_y_revival.contracts import sha256_mapping

    script = Path(__file__).resolve().parents[2] / "scripts" / "build_method_y_source_regional_contracts_v2.py"
    spec = importlib.util.spec_from_file_location("source_builder_events", script)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    endpoints = tmp_path / "endpoints"
    endpoints.mkdir()
    artifact = endpoints / "amyloid.parquet"
    pd.DataFrame({"patient_hash": ["p1", "p2"], "event": [-1, 2]}).to_parquet(artifact, index=False)
    payload = {"locked": True, "row_count": 2, "event_count": 1, "endpoint_path": artifact.name, "endpoint_sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(), "source_hashes": {"subjinfo_csv": "a" * 64}}
    payload["contract_sha256"] = sha256_mapping(payload)
    (endpoints / "amyloid_endpoint_contract.json").write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="EVENT_COUNT_MISMATCH"):
        module._read_locked_endpoint(tmp_path, "amyloid", required_source_hashes={"subjinfo_csv": "a" * 64})
