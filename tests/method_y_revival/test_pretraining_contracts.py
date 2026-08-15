from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest


def _endpoint_rows(prefix: str, n: int = 60, *, cohort: str = "A4"):
    rows = []
    for index in range(n):
        event = int(index % 4 == 0)
        rows.append(
            {
                "patient_hash": f"{prefix}-{index:03d}",
                "cohort": cohort,
                "treatment_arm": "Placebo" if index % 2 else "Solanezumab",
                "treatment_active": 0.0 if index % 2 else 1.0,
                "baseline_gap_days_abs": 10.0,
                "index_source": "max_fbp_scan_and_baseline_cdr_dates",
                "event": event,
                "time_days": 300.0 + index,
                "censoring_rule": "event_time" if event else "right_censored",
                "followup_cdr_visits": 2 if event else 1,
                "baseline_cdglobal": 0.0,
                "uses_raw_identifier": False,
                "uses_raw_path": False,
            }
        )
    return rows


def _amyloid_endpoint_rows(prefix: str, n: int = 60):
    split = max(1, n // 4)
    return [
        *_endpoint_rows(f"{prefix}-a4", n=n - split, cohort="A4"),
        *_endpoint_rows(f"{prefix}-learn", n=split, cohort="LEARN"),
    ]


def test_terminal_unconfirmed_cdr_increase_is_right_censored():
    from scripts.build_method_y_tau_progression import detect_sustained_event

    event, event_date = detect_sustained_event(
        [{"_date": 100.0, "_cdglobal": 0.5}],
        event_time_rule="first_increase_if_confirmed_by_next_visit",
    )

    assert event is False
    assert event_date is None


def test_predictor_temporality_forbids_future_change_and_unknown_cognition():
    from src.method_y_revival.pretraining_contracts import predictor_temporality_contract

    contract = predictor_temporality_contract(
        [
            "baseline_pacc_change",
            "baseline_pacc_raw",
            "baseline_cdrsb",
            "followup_diagnosis",
            "sex_binary",
        ],
        timing_evidence={"sex_binary": "TIME_INVARIANT"},
    )

    assert contract["status"] == "PASS"
    assert contract["baseline_pacc_change_allowed"] is False
    assert (
        contract["forbidden_predictors"]["baseline_pacc_change"]
        == "OUTCOME_DERIVED_OR_CIRCULAR"
    )
    assert (
        contract["timing_blocked_predictors"]["baseline_pacc_raw"]
        == "EXACT_NONFUTURE_TIMING_NOT_PROVEN"
    )
    assert contract["forbidden_predictors"]["baseline_cdrsb"]
    assert contract["forbidden_predictors"]["followup_diagnosis"]
    assert contract["allowed_predictors"] == ["sex_binary"]


def test_baseline_pacc_raw_requires_exact_nonfuture_timing():
    from src.method_y_revival.pretraining_contracts import predictor_temporality_contract

    blocked = predictor_temporality_contract(["baseline_pacc_raw"])
    allowed = predictor_temporality_contract(
        ["baseline_pacc_raw"],
        timing_evidence={"baseline_pacc_raw": "EXACT_NONFUTURE"},
    )

    assert blocked["allowed_predictors"] == []
    assert allowed["allowed_predictors"] == ["baseline_pacc_raw"]


def test_locked_endpoint_reads_and_hashes_the_same_bytes(tmp_path):
    from src.method_y_revival.endpoint_builder import write_locked_endpoint
    from src.method_y_revival.pretraining_contracts import (
        PretrainingContractError,
        validate_locked_endpoint,
    )

    root = tmp_path / "source"
    endpoint = root / "endpoints" / "amyloid_progression_endpoint.parquet"
    contract = root / "endpoints" / "amyloid_endpoint_contract.json"
    write_locked_endpoint(
        _amyloid_endpoint_rows("a"),
        output_path=endpoint,
        contract_path=contract,
        source_hashes={"source": "a" * 64},
        endpoint_definition={
            "event_time_rule": "first_increase_if_confirmed_by_next_visit",
            "followup_window_days": 1640.0,
            "baseline_cdr_required": 0.0,
        },
    )

    report, frame = validate_locked_endpoint(root, "amyloid")
    assert report["status"] == "PASS"
    assert report["row_count"] == len(frame) == 60
    assert report["future_only"] is True

    endpoint.write_bytes(endpoint.read_bytes() + b"tamper")
    with pytest.raises(PretrainingContractError, match="ENDPOINT_ARTIFACT_HASH_MISMATCH"):
        validate_locked_endpoint(root, "amyloid")


def test_event_without_confirming_visit_fails_endpoint_gate(tmp_path):
    from src.method_y_revival.endpoint_builder import write_locked_endpoint
    from src.method_y_revival.pretraining_contracts import (
        PretrainingContractError,
        validate_locked_endpoint,
    )

    rows = _amyloid_endpoint_rows("a")
    event_row = next(row for row in rows if row["event"] == 1)
    event_row["followup_cdr_visits"] = 1
    root = tmp_path / "source"
    write_locked_endpoint(
        rows,
        output_path=root / "endpoints" / "amyloid_progression_endpoint.parquet",
        contract_path=root / "endpoints" / "amyloid_endpoint_contract.json",
        source_hashes={"source": "a" * 64},
        endpoint_definition={
            "event_time_rule": "first_increase_if_confirmed_by_next_visit",
            "followup_window_days": 1640.0,
            "baseline_cdr_required": 0.0,
        },
    )

    with pytest.raises(PretrainingContractError, match="EVENT_WITHOUT_CONFIRMING_VISIT"):
        validate_locked_endpoint(root, "amyloid")


@pytest.mark.parametrize("invalid_event", [0.5, float("nan"), float("inf"), "1", None])
def test_endpoint_writer_rejects_nonfinite_fractional_or_nonnumeric_events(
    tmp_path, invalid_event
):
    from src.method_y_revival.endpoint_builder import EndpointBuildError, write_locked_endpoint

    rows = _endpoint_rows("bad")
    rows[0]["event"] = invalid_event
    with pytest.raises(EndpointBuildError, match="BINARY_EVENT_REQUIRED"):
        write_locked_endpoint(
            rows,
            output_path=tmp_path / "endpoints" / "amyloid.parquet",
            contract_path=tmp_path / "endpoints" / "amyloid_endpoint_contract.json",
            source_hashes={"source": "a" * 64},
            endpoint_definition={"event_time_rule": "first_increase_if_confirmed_by_next_visit"},
        )


def _rewrite_endpoint_and_contract(root, mutate):
    import hashlib

    from src.method_y_revival.contracts import sha256_mapping

    contract_path = root / "endpoints" / "amyloid_endpoint_contract.json"
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    endpoint_path = root / "endpoints" / contract["endpoint_path"]
    frame = pd.read_parquet(endpoint_path)
    mutate(frame)
    frame.to_parquet(endpoint_path, index=False)
    contract["endpoint_sha256"] = hashlib.sha256(endpoint_path.read_bytes()).hexdigest()
    event_total = float(pd.to_numeric(frame["event"], errors="raise").sum())
    contract["event_count"] = int(event_total) if event_total.is_integer() else event_total
    contract["contract_sha256"] = sha256_mapping(
        {key: value for key, value in contract.items() if key != "contract_sha256"}
    )
    contract_path.write_text(json.dumps(contract), encoding="utf-8")


def test_endpoint_validator_rejects_forged_contract_self_hash(tmp_path):
    from src.method_y_revival.endpoint_builder import write_locked_endpoint
    from src.method_y_revival.pretraining_contracts import (
        PretrainingContractError,
        validate_locked_endpoint,
    )

    root = tmp_path / "source"
    write_locked_endpoint(
        _amyloid_endpoint_rows("a"),
        output_path=root / "endpoints" / "amyloid.parquet",
        contract_path=root / "endpoints" / "amyloid_endpoint_contract.json",
        source_hashes={"source": "a" * 64},
        endpoint_definition={
            "event_time_rule": "first_increase_if_confirmed_by_next_visit",
            "followup_window_days": 1640.0,
            "baseline_cdr_required": 0.0,
        },
    )
    contract_path = root / "endpoints" / "amyloid_endpoint_contract.json"
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    contract["contract_sha256"] = "0" * 64
    contract_path.write_text(json.dumps(contract), encoding="utf-8")

    with pytest.raises(PretrainingContractError, match="ENDPOINT_CONTRACT_SELF_HASH_INVALID"):
        validate_locked_endpoint(root, "amyloid")


def test_endpoint_validator_rejects_fractional_events_with_preserved_sum(tmp_path):
    from src.method_y_revival.endpoint_builder import write_locked_endpoint
    from src.method_y_revival.pretraining_contracts import (
        PretrainingContractError,
        validate_locked_endpoint,
    )

    root = tmp_path / "source"
    write_locked_endpoint(
        _amyloid_endpoint_rows("a"),
        output_path=root / "endpoints" / "amyloid.parquet",
        contract_path=root / "endpoints" / "amyloid_endpoint_contract.json",
        source_hashes={"source": "a" * 64},
        endpoint_definition={
            "event_time_rule": "first_increase_if_confirmed_by_next_visit",
            "followup_window_days": 1640.0,
            "baseline_cdr_required": 0.0,
        },
    )

    def mutate(frame):
        frame["event"] = frame["event"].astype(float)
        zero = frame.index[frame["event"] == 0][0]
        one = frame.index[frame["event"] == 1][0]
        frame.loc[[zero, one], "event"] = [0.5, 0.5]

    _rewrite_endpoint_and_contract(root, mutate)
    with pytest.raises(PretrainingContractError, match="BINARY_EVENT_SUPPORT_INVALID"):
        validate_locked_endpoint(root, "amyloid")


@pytest.mark.parametrize("placeholder", [None, "unknown", "MISSING", "not available"])
def test_endpoint_validator_rejects_null_or_placeholder_membership(
    tmp_path, placeholder
):
    from src.method_y_revival.endpoint_builder import write_locked_endpoint
    from src.method_y_revival.pretraining_contracts import (
        PretrainingContractError,
        validate_locked_endpoint,
    )

    root = tmp_path / "source"
    write_locked_endpoint(
        _amyloid_endpoint_rows("a"),
        output_path=root / "endpoints" / "amyloid.parquet",
        contract_path=root / "endpoints" / "amyloid_endpoint_contract.json",
        source_hashes={"source": "a" * 64},
        endpoint_definition={
            "event_time_rule": "first_increase_if_confirmed_by_next_visit",
            "followup_window_days": 1640.0,
            "baseline_cdr_required": 0.0,
        },
    )
    _rewrite_endpoint_and_contract(
        root,
        lambda frame: frame.__setitem__(
            "patient_hash", [placeholder, *frame["patient_hash"].tolist()[1:]]
        ),
    )
    with pytest.raises(PretrainingContractError, match="ENDPOINT_MEMBERSHIP_INVALID"):
        validate_locked_endpoint(root, "amyloid")


def test_split_validation_requires_event_classes_not_merely_combined_strata():
    from src.method_y_revival.pretraining_contracts import (
        PretrainingContractError,
        validate_nested_split,
    )
    from src.method_y_revival.splits import build_nested_subject_splits

    frame = pd.DataFrame(_endpoint_rows("s", n=90))
    rows = [
        {
            "subject_key": row.patient_hash,
            "event": row.event,
            "treatment_arm": row.treatment_arm,
        }
        for row in frame.itertuples(index=False)
    ]
    split = build_nested_subject_splits(rows, outer_folds=5, inner_folds=3, seed=17)
    report = validate_nested_split(split, frame)
    assert report["complete_outer_test_partition"] is True
    assert report["all_inner_and_outer_partitions_have_both_event_classes"] is True

    broken = json.loads(json.dumps(split))
    test_members = set(broken["outer_folds"][0]["test_subjects"])
    frame.loc[frame["patient_hash"].isin(test_members), "event"] = 0
    with pytest.raises(PretrainingContractError, match="OUTER_FOLD_EVENT_CLASS_MISSING"):
        validate_nested_split(broken, frame)


def test_direct_split_builder_rejects_fractional_events():
    from src.method_y_revival.splits import SplitContractError, build_nested_subject_splits

    rows = [
        {"subject_key": f"p-{index}", "event": 0.5, "treatment_arm": "A"}
        for index in range(30)
    ]
    with pytest.raises(SplitContractError, match="BINARY_EVENT_REQUIRED"):
        build_nested_subject_splits(rows, outer_folds=5, inner_folds=3, seed=1)


def test_private_splits_separate_a4_development_from_learn_transport(tmp_path):
    from src.method_y_revival.pretraining_contracts import build_private_split_artifacts

    amyloid = pd.DataFrame(
        [
            *_endpoint_rows("a4", n=90, cohort="A4"),
            *_endpoint_rows("learn", n=30, cohort="LEARN"),
        ]
    )
    tau = pd.DataFrame(_endpoint_rows("tau", n=90, cohort="A4"))
    tau["index_source"] = "max_ftp_scan_and_baseline_cdr_dates"
    paired = tau.copy()
    paired["patient_hash"] = paired["patient_hash"].str.replace("tau", "pair")
    paired["index_source"] = "max_fbp_ftp_and_baseline_cdr_dates"
    paired["paired_fbp_ftp_gap_days"] = 30.0
    reports = {
        branch: {"contract_sha256": letter * 64}
        for branch, letter in (("amyloid", "a"), ("tau", "b"), ("paired_at", "c"))
    }

    public = build_private_split_artifacts(
        {"amyloid": amyloid, "tau": tau, "paired_at": paired},
        reports,
        private_output_root=tmp_path / "_private",
        seed=19,
    )

    primary = public["branches"]["amyloid_source_regional"]
    transport = public["branches"]["amyloid_learn_transport"]
    assert primary["analysis_role"] == "PRIMARY_DEVELOPMENT_A4"
    assert primary["subject_count"] == 90
    assert transport["analysis_role"] == "LOCKED_INTERNAL_TRANSPORT_ONLY_NO_HPO"
    assert transport["subject_count"] == 30
    assert transport["development_or_hpo_allowed"] is False
    assert primary["membership_digest"] != transport["membership_digest"]


def test_scanner_contract_never_pools_ge_and_philips_or_imputes_unknown(tmp_path):
    from src.method_y_revival.pretraining_contracts import scanner_contract_from_aggregate

    audit = {
        "status": "COMPLETED",
        "unknown_scanner_imputed": False,
        "amyloid": {"manufacturer_counts": {"GE": 40, "PHILIPS": 35, "UNKNOWN": 2}},
        "tau": {
            "manufacturer_counts": {
                "GE": 44,
                "PHILIPS": 38,
                "SIEMENS": 181,
                "UNKNOWN": 5,
            }
        },
        "amyloid_sensitivity_feasibility": {
            "status": "READY_FOR_LEAVE_MANUFACTURER_OUT",
            "unknown_imputed": False,
            "manufacturer_support": {
                "GE": {"subjects": 40, "events": 12, "gate_eligible": True},
                "PHILIPS": {"subjects": 35, "events": 11, "gate_eligible": True},
            },
            "eligible_manufacturers": ["GE", "PHILIPS"],
            "minimum_subjects_per_manufacturer": 30,
            "minimum_events_per_manufacturer": 10,
            "minimum_eligible_manufacturers": 2,
        },
        "tau_sensitivity_feasibility": {
            "status": "BLOCKED_INSUFFICIENT_MANUFACTURER_EVENT_SUPPORT",
            "unknown_imputed": False,
            "manufacturer_support": {
                "GE": {"subjects": 44, "events": 9, "gate_eligible": False},
                "PHILIPS": {"subjects": 38, "events": 9, "gate_eligible": False},
                "SIEMENS": {"subjects": 181, "events": 50, "gate_eligible": True},
            },
            "eligible_manufacturers": ["SIEMENS"],
            "minimum_subjects_per_manufacturer": 30,
            "minimum_events_per_manufacturer": 10,
            "minimum_eligible_manufacturers": 2,
        },
    }
    path = tmp_path / "scanner.json"
    path.write_text(json.dumps(audit), encoding="utf-8")

    contract = scanner_contract_from_aggregate(path)
    assert contract["unknown_imputed"] is False
    assert contract["ge_philips_pooled"] is False
    assert contract["branches"]["amyloid"]["unknown_metadata_count"] == 2
    assert contract["branches"]["tau"]["unknown_metadata_retained_as_unknown"] is True
    assert "GE" in contract["branches"]["tau"]["manufacturer_support"]
    assert "PHILIPS" in contract["branches"]["tau"]["manufacturer_support"]
    assert (
        contract["branches"]["tau"]["scanner_sensitivity_status"]
        == "NOT_RUN_REQUIRES_LOCKED_OOF_PREDICTIONS"
    )
    assert contract["branches"]["paired_at"]["claim_allowed"] is False


@pytest.mark.parametrize(
    "alias",
    [
        "GE+PHILIPS",
        "GE-PHILIPS",
        "GE/PHILIPS",
        "GE / Philips",
        "non-Siemens",
        "other-combined",
        "",
    ],
)
def test_scanner_contract_rejects_every_pooling_or_unknown_alias(tmp_path, alias):
    from src.method_y_revival.pretraining_contracts import (
        PretrainingContractError,
        scanner_contract_from_aggregate,
    )

    audit = {
        "status": "COMPLETED",
        "unknown_scanner_imputed": False,
        "amyloid": {"manufacturer_counts": {alias: 40, "UNKNOWN": 2}},
        "tau": {"manufacturer_counts": {"SIEMENS": 40, "UNKNOWN": 2}},
        "amyloid_sensitivity_feasibility": {
            "status": "READY_FOR_LEAVE_MANUFACTURER_OUT",
            "unknown_imputed": False,
            "manufacturer_support": {
                alias: {"subjects": 40, "events": 12, "gate_eligible": True},
                "SIEMENS": {"subjects": 40, "events": 12, "gate_eligible": True},
            },
            "eligible_manufacturers": [alias, "SIEMENS"],
            "minimum_subjects_per_manufacturer": 30,
            "minimum_events_per_manufacturer": 10,
            "minimum_eligible_manufacturers": 2,
        },
        "tau_sensitivity_feasibility": {
            "status": "BLOCKED_INSUFFICIENT_MANUFACTURER_EVENT_SUPPORT",
            "unknown_imputed": False,
            "manufacturer_support": {
                "SIEMENS": {"subjects": 40, "events": 12, "gate_eligible": True}
            },
            "eligible_manufacturers": ["SIEMENS"],
            "minimum_subjects_per_manufacturer": 30,
            "minimum_events_per_manufacturer": 10,
            "minimum_eligible_manufacturers": 2,
        },
    }
    path = tmp_path / "scanner.json"
    path.write_text(json.dumps(audit), encoding="utf-8")
    with pytest.raises(PretrainingContractError, match="SCANNER_MANUFACTURER"):
        scanner_contract_from_aggregate(path)


@pytest.mark.parametrize(
    "failure",
    [
        "flag",
        "eligible_list",
        "status",
        "thresholds",
        "events_exceed_subjects",
        "support_exceeds_metadata",
    ],
)
def test_scanner_contract_recomputes_and_rejects_inconsistent_feasibility(
    tmp_path, failure
):
    from src.method_y_revival.pretraining_contracts import (
        PretrainingContractError,
        scanner_contract_from_aggregate,
    )

    support = {
        "GE": {
            "subjects": 2,
            "events": 3 if failure == "events_exceed_subjects" else 0,
            "gate_eligible": failure == "flag",
        },
        "SIEMENS": {"subjects": 40, "events": 12, "gate_eligible": True},
    }
    audit = {
        "status": "COMPLETED",
        "unknown_scanner_imputed": False,
        "amyloid": {
            "manufacturer_counts": {
                "GE": 1 if failure == "support_exceeds_metadata" else 2,
                "SIEMENS": 40,
                "UNKNOWN": 1,
            }
        },
        "tau": {"manufacturer_counts": {"SIEMENS": 40, "UNKNOWN": 2}},
        "amyloid_sensitivity_feasibility": {
            "status": (
                "READY_FOR_LEAVE_MANUFACTURER_OUT"
                if failure == "status"
                else "BLOCKED_INSUFFICIENT_MANUFACTURER_EVENT_SUPPORT"
            ),
            "unknown_imputed": False,
            "manufacturer_support": support,
            "eligible_manufacturers": (
                ["GE", "SIEMENS"] if failure == "eligible_list" else ["SIEMENS"]
            ),
            "minimum_subjects_per_manufacturer": 1 if failure == "thresholds" else 30,
            "minimum_events_per_manufacturer": 1 if failure == "thresholds" else 10,
            "minimum_eligible_manufacturers": 2,
        },
        "tau_sensitivity_feasibility": {
            "status": "BLOCKED_INSUFFICIENT_MANUFACTURER_EVENT_SUPPORT",
            "unknown_imputed": False,
            "manufacturer_support": {
                "SIEMENS": {"subjects": 40, "events": 12, "gate_eligible": True}
            },
            "eligible_manufacturers": ["SIEMENS"],
            "minimum_subjects_per_manufacturer": 30,
            "minimum_events_per_manufacturer": 10,
            "minimum_eligible_manufacturers": 2,
        },
    }
    path = tmp_path / "scanner.json"
    path.write_text(json.dumps(audit), encoding="utf-8")
    with pytest.raises(PretrainingContractError, match="SCANNER_FEASIBILITY_INCONSISTENT"):
        scanner_contract_from_aggregate(path)


def test_external_intake_is_embargoed_and_does_not_block_internal_development():
    from src.method_y_revival.pretraining_contracts import (
        assert_public_contract_privacy,
        external_intake_contracts,
    )

    contract = external_intake_contracts()
    assert contract["status"] == "EXTERNAL_PENDING"
    assert contract["external_outcomes_accessed"] is False
    assert contract["internal_development_blocked_by_external_access"] is False
    assert contract["outcome_embargo"]["enforced"] is True
    for cohort in contract["cohorts"].values():
        assert cohort["external_data_available"] is False
        assert cohort["outcome_access_allowed"] is False
        assert cohort["mapping_locked"] is False
        assert cohort["external_evaluation_allowed"] is False
    assert_public_contract_privacy(contract)


def _write_synthetic_task4_source(root):
    from src.method_y_revival.endpoint_builder import write_locked_endpoint

    endpoint_root = root / "endpoints"
    frames = {
        "amyloid": [
            *_endpoint_rows("a4", n=90, cohort="A4"),
            *_endpoint_rows("learn", n=30, cohort="LEARN"),
        ],
        "tau": _endpoint_rows("tau", n=90, cohort="A4"),
        "paired_at": _endpoint_rows("pair", n=90, cohort="A4"),
    }
    for row in frames["tau"]:
        row["cohort"] = (
            "A4_treatment"
            if row["treatment_active"] == 1.0
            else "A4_placebo"
        )
        row["index_source"] = "max_ftp_scan_and_baseline_cdr_dates"
    for row in frames["paired_at"]:
        row["cohort"] = (
            "A4_treatment"
            if row["treatment_active"] == 1.0
            else "A4_placebo"
        )
        row["index_source"] = "max_fbp_ftp_and_baseline_cdr_dates"
        row["paired_fbp_ftp_gap_days"] = 30.0
    for branch, rows in frames.items():
        definition = {
            "event_time_rule": "first_increase_if_confirmed_by_next_visit",
            "followup_window_days": 1640.0,
            "baseline_cdr_required": 0.0,
        }
        write_locked_endpoint(
            rows,
            output_path=endpoint_root / f"{branch}_endpoint.parquet",
            contract_path=endpoint_root / f"{branch}_endpoint_contract.json",
            source_hashes={"clinical": "a" * 64},
            endpoint_definition=definition,
        )

    pacc = pd.DataFrame(
        [
            {
                "patient_hash": f"a4-{index // 2:03d}",
                "time_days_from_pet": 100.0 + index,
                "time_years": (100.0 + index) / 365.25,
                "pacc_raw": float(index),
            }
            for index in range(10)
        ]
    )
    pacc_path = endpoint_root / "longitudinal_pacc_endpoint.csv"
    pacc.to_csv(pacc_path, index=False, lineterminator="\n")
    import hashlib

    pacc_sha = hashlib.sha256(pacc_path.read_bytes()).hexdigest()
    amyloid_contract = json.loads(
        (endpoint_root / "amyloid_endpoint_contract.json").read_text(encoding="utf-8")
    )
    pacc_contract = {
        "locked": True,
        "status": "LONGITUDINAL_PACC_ENDPOINT_LOCKED",
        "endpoint_path": pacc_path.name,
        "endpoint_sha256": pacc_sha,
        "input_contract_sha256": "d" * 64,
        "parent_endpoint_sha256": amyloid_contract["endpoint_sha256"],
        "raw_identifiers_present": False,
        "source_sha256": {"pacc": "e" * 64},
        "row_count": 10,
        "subject_count": 5,
    }
    (endpoint_root / "longitudinal_pacc_endpoint_contract.json").write_text(
        json.dumps(pacc_contract), encoding="utf-8"
    )
    from src.method_y_revival.contracts import sha256_mapping

    (endpoint_root / "longitudinal_pacc_endpoint_audit.json").write_text(
        json.dumps(
            {
                "status": "LONGITUDINAL_PACC_ENDPOINT_LOCKED",
                "contract_sha256": sha256_mapping(pacc_contract),
                "endpoint_audit": {
                    "status": "LONGITUDINAL_PACC_ENDPOINT_BUILT",
                    "exact_measurement_dates_required": True,
                    "eligible_measurement_count": 10,
                    "eligible_subject_count": 5,
                    "raw_identifiers_in_output": False,
                },
                "exact_date_audit": {
                    "date_rule": "MEDIAN_EXACT_QSDTC_DAYS_T0_ACROSS_PACC_COMPONENTS",
                    "component_codes": ["DIGIT", "FCSRT96", "LMIIA", "MMSE"],
                },
            }
        ),
        encoding="utf-8",
    )
    scanner = {
        "status": "COMPLETED",
        "unknown_scanner_imputed": False,
        "amyloid": {"manufacturer_counts": {"GE": 40, "PHILIPS": 35, "UNKNOWN": 2}},
        "tau": {
            "manufacturer_counts": {
                "GE": 44,
                "PHILIPS": 38,
                "SIEMENS": 181,
                "UNKNOWN": 5,
            }
        },
        "amyloid_sensitivity_feasibility": {
            "status": "READY_FOR_LEAVE_MANUFACTURER_OUT",
            "unknown_imputed": False,
            "manufacturer_support": {
                "GE": {"subjects": 40, "events": 12, "gate_eligible": True},
                "PHILIPS": {"subjects": 35, "events": 11, "gate_eligible": True},
            },
            "eligible_manufacturers": ["GE", "PHILIPS"],
            "minimum_subjects_per_manufacturer": 30,
            "minimum_events_per_manufacturer": 10,
            "minimum_eligible_manufacturers": 2,
        },
        "tau_sensitivity_feasibility": {
            "status": "BLOCKED_INSUFFICIENT_MANUFACTURER_EVENT_SUPPORT",
            "unknown_imputed": False,
            "manufacturer_support": {
                "GE": {"subjects": 44, "events": 9, "gate_eligible": False},
                "PHILIPS": {"subjects": 38, "events": 9, "gate_eligible": False},
                "SIEMENS": {"subjects": 181, "events": 50, "gate_eligible": True},
            },
            "eligible_manufacturers": ["SIEMENS"],
            "minimum_subjects_per_manufacturer": 30,
            "minimum_events_per_manufacturer": 10,
            "minimum_eligible_manufacturers": 2,
        },
    }
    (root / "qc").mkdir(parents=True)
    (root / "qc" / "scanner_metadata_audit.json").write_text(
        json.dumps(scanner), encoding="utf-8"
    )
    return {
        "amyloid": pd.DataFrame(frames["amyloid"]),
        "tau": pd.DataFrame(frames["tau"]),
        "paired_at": pd.DataFrame(frames["paired_at"]),
        "longitudinal_pacc": pacc,
    }


def _initialize_task4_implementation_repo(project_root):
    import hashlib
    import subprocess

    from src.method_y_revival.review_receipt import (
        TASK4_IMPLEMENTATION_RELATIVE_PATHS,
    )

    project_root.mkdir(parents=True, exist_ok=True)
    for relative in TASK4_IMPLEMENTATION_RELATIVE_PATHS:
        path = project_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_text(f"synthetic:{relative}\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q", str(project_root)], check=True)
    subprocess.run(
        ["git", "-C", str(project_root), "config", "user.email", "test@example.invalid"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(project_root), "config", "user.name", "Test Agent"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(project_root), "add", "--", *TASK4_IMPLEMENTATION_RELATIVE_PATHS],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(project_root), "commit", "-q", "-m", "synthetic task4"],
        check=True,
    )
    commit = subprocess.run(
        ["git", "-C", str(project_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    hashes = {
        relative: hashlib.sha256((project_root / relative).read_bytes()).hexdigest()
        for relative in TASK4_IMPLEMENTATION_RELATIVE_PATHS
    }
    return commit, hashes


def _build_synthetic_task4_contracts(
    *, project_root, source, output, private, rebuilt, seed
):
    from unittest.mock import patch

    from scripts import build_method_y_pretraining_contracts_v2 as builder

    commit, implementation_hashes = _initialize_task4_implementation_repo(project_root)

    with (
        patch.object(
            builder, "_clinical_source_hashes", return_value={"clinical": "a" * 64}
        ),
        patch.object(
            builder, "_pacc_source_hashes", return_value={"pacc": "e" * 64}
        ),
        patch.object(builder, "_build_reconstructed_endpoints", return_value=rebuilt),
        patch.object(builder, "_implementation_commit", return_value=commit),
        patch.object(
            builder, "_implementation_file_hashes", return_value=implementation_hashes
        ),
    ):
        return builder.build_all(
            source_root=source,
            output_root=output,
            private_output_root=private,
            config={},
            seed=seed,
            expected_pacc_source_hashes={"pacc": "e" * 64},
            expected_pacc_input_contract_sha256="d" * 64,
        )


def _write_synthetic_task4_review_evidence(
    output, *, implementation_commit_override=None
):
    from src.method_y_revival.contracts import sha256_mapping
    from src.method_y_revival.review_receipt import (
        EVIDENCE_SCHEMA_VERSION,
        synthetic_layer_result,
    )

    state = json.loads(
        (output / "pretraining_validation_state.json").read_text(encoding="utf-8")
    )
    evidence_root = output / "review_evidence"
    evidence_root.mkdir(exist_ok=True)
    relative_paths = []
    for layer_id in (
        "INVARIANT_TEST_SUITE",
        "DETERMINISTIC_REBUILD",
        "ADVERSARIAL_AGENT_REVIEW",
    ):
        relative_path = "review_evidence/" + layer_id.casefold() + "_v2.json"
        payload = {
            "schema_version": EVIDENCE_SCHEMA_VERSION,
            "layer_id": layer_id,
            "status": "PASS",
            "implementation_commit": (
                implementation_commit_override or state["implementation_commit"]
            ),
            "run_contract_sha256": state["run_contract_sha256"],
            "reviewed_artifact_sha256": state["artifact_sha256"],
            **synthetic_layer_result(layer_id, state["artifact_sha256"]),
        }
        payload["evidence_contract_sha256"] = sha256_mapping(payload)
        (output / relative_path).write_text(
            json.dumps(payload, sort_keys=True), encoding="utf-8"
        )
        relative_paths.append(relative_path)

    return relative_paths


def _finalize_synthetic_task4_review(output, *, project_root):
    from src.method_y_revival.review_receipt import finalize_task4_agent_review

    return finalize_task4_agent_review(
        output,
        project_root=project_root,
        evidence_relative_paths=_write_synthetic_task4_review_evidence(output),
    )


def test_task4_builder_commits_reports_before_gate_and_keeps_memberships_private(
    tmp_path,
):
    project = tmp_path / "implementation_repo"
    source = tmp_path / "source"
    output = tmp_path / "public"
    private = output / "_private"
    rebuilt = _write_synthetic_task4_source(source)

    result = _build_synthetic_task4_contracts(
        project_root=project,
        source=source,
        output=output,
        private=private,
        rebuilt=rebuilt,
        seed=23,
    )

    assert result["status"] == "AWAITING_AGENT_MULTI_PASS_QC"
    state = json.loads(
        (output / "pretraining_validation_state.json").read_text(encoding="utf-8")
    )
    assert state["status"] == "AWAITING_AGENT_MULTI_PASS_QC"
    assert state["agent_multi_pass_qc_status"] == "AGENT_MULTI_PASS_QC_PENDING"
    assert state["training_started"] is False
    assert state["external_outcomes_accessed"] is False
    committed = _finalize_synthetic_task4_review(output, project_root=project)
    assert committed["status"] == "COMMITTED"
    assert committed["agent_multi_pass_qc_status"] == "AGENT_MULTI_PASS_QC_PASS"
    assert (private / "amyloid_source_regional_nested_splits_v2_private.json").is_file()
    public = json.loads(
        (output / "split_verification_v2.json").read_text(encoding="utf-8")
    )
    public_text = json.dumps(public)
    assert "a4-" not in public_text
    assert "learn-" not in public_text
    assert public["branches"]["amyloid_source_regional"]["subject_count"] == 90
    assert public["branches"]["amyloid_learn_transport"]["subject_count"] == 30
    temporality = json.loads(
        (output / "predictor_temporality_contract_v2.json").read_text(encoding="utf-8")
    )
    assert not any(
        name.startswith("tau_") for name in temporality["forbidden_predictors"]
    )
    from src.method_y_revival.contracts import sha256_mapping

    assert temporality["contract_sha256"] == sha256_mapping(
        {key: value for key, value in temporality.items() if key != "contract_sha256"}
    )


def test_task4_review_receipt_requires_three_real_bound_evidence_files(tmp_path):
    from src.method_y_revival.review_receipt import (
        AgentReviewReceiptError,
        finalize_task4_agent_review,
    )

    project = tmp_path / "project"
    result_root = project / "results" / "method_y_revival_v2"
    rebuilt = _write_synthetic_task4_source(result_root)
    root = result_root / "pretraining_contracts"
    _build_synthetic_task4_contracts(
        project_root=project,
        source=result_root,
        output=root,
        private=root / "_private",
        rebuilt=rebuilt,
        seed=23,
    )
    with pytest.raises(AgentReviewReceiptError, match="REVIEW_EVIDENCE_.*MISSING"):
        finalize_task4_agent_review(
            root,
            project_root=project,
            evidence_relative_paths=[
                "review_evidence/invariant_test_suite_v2.json",
                "review_evidence/deterministic_rebuild_v2.json",
                "review_evidence/adversarial_agent_review_v2.json",
            ],
        )

    state = _finalize_synthetic_task4_review(root, project_root=project)
    receipt = json.loads(
        (root / "agent_multi_pass_review_receipt_v2.json").read_text(encoding="utf-8")
    )
    from src.method_y_revival.contracts import sha256_mapping

    assert receipt["receipt_sha256"] == sha256_mapping(
        {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    )
    assert state["status"] == "COMMITTED"
    assert state["artifact_sha256"]["agent_multi_pass_review_receipt_v2.json"]


def test_task4_review_receipt_rejects_tampered_or_unbound_evidence(tmp_path):
    from src.method_y_revival.review_receipt import (
        AgentReviewReceiptError,
        finalize_task4_agent_review,
    )

    project = tmp_path / "project"
    result_root = project / "results" / "method_y_revival_v2"
    rebuilt = _write_synthetic_task4_source(result_root)
    root = result_root / "pretraining_contracts"
    _build_synthetic_task4_contracts(
        project_root=project,
        source=result_root,
        output=root,
        private=root / "_private",
        rebuilt=rebuilt,
        seed=23,
    )
    _write_synthetic_task4_review_evidence(
        root, implementation_commit_override="c" * 40
    )
    with pytest.raises(AgentReviewReceiptError, match="REVIEW_EVIDENCE_BINDING_INVALID"):
        finalize_task4_agent_review(
            root,
            project_root=project,
            evidence_relative_paths=[
                "review_evidence/invariant_test_suite_v2.json",
                "review_evidence/deterministic_rebuild_v2.json",
                "review_evidence/adversarial_agent_review_v2.json",
            ],
        )


def test_release_builder_exposes_no_precomputed_rebuild_bypass():
    import inspect

    from scripts.build_method_y_pretraining_contracts_v2 import build_all

    parameters = inspect.signature(build_all).parameters
    assert "precomputed_rebuilt_frames" not in parameters
    assert "independent_rebuild" not in parameters


def test_task4_rejects_self_consistent_forged_run_contract(tmp_path):
    from src.method_y_revival.contracts import sha256_mapping
    from src.method_y_revival.review_receipt import (
        AgentReviewReceiptError,
        finalize_task4_agent_review,
    )

    project = tmp_path / "project"
    result_root = project / "results" / "method_y_revival_v2"
    rebuilt = _write_synthetic_task4_source(result_root)
    root = result_root / "pretraining_contracts"
    _build_synthetic_task4_contracts(
        project_root=project,
        source=result_root,
        output=root,
        private=root / "_private",
        rebuilt=rebuilt,
        seed=23,
    )
    state_path = root / "pretraining_validation_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["implementation_commit"] = "f" * 40
    state["run_contract"]["implementation_commit"] = "f" * 40
    state["run_contract"]["seed"] = 999
    state["run_contract"]["split_seed"] = 999
    state["run_contract"]["implementation_file_sha256"] = {
        name: "f" * 64
        for name in state["run_contract"]["implementation_file_sha256"]
    }
    state["run_contract_sha256"] = sha256_mapping(state["run_contract"])
    state_path.write_text(json.dumps(state), encoding="utf-8")
    paths = _write_synthetic_task4_review_evidence(root)
    with pytest.raises(
        AgentReviewReceiptError,
        match="TASK4_(IMPLEMENTATION|RUN_CONTRACT)",
    ):
        finalize_task4_agent_review(
            root, project_root=project, evidence_relative_paths=paths
        )


def test_task4_rejects_privacy_fields_in_review_evidence(tmp_path):
    from src.method_y_revival.contracts import sha256_mapping
    from src.method_y_revival.review_receipt import (
        AgentReviewReceiptError,
        finalize_task4_agent_review,
    )

    project = tmp_path / "project"
    result_root = project / "results" / "method_y_revival_v2"
    rebuilt = _write_synthetic_task4_source(result_root)
    root = result_root / "pretraining_contracts"
    _build_synthetic_task4_contracts(
        project_root=project,
        source=result_root,
        output=root,
        private=root / "_private",
        rebuilt=rebuilt,
        seed=23,
    )
    paths = _write_synthetic_task4_review_evidence(root)
    path = root / "review_evidence" / "invariant_test_suite_v2.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["raw_path"] = "H:\\A4 Learn\\Clinical\\BID123.csv"
    payload["source_subject_id"] = "BID123"
    payload["evidence_contract_sha256"] = sha256_mapping(
        {
            key: value
            for key, value in payload.items()
            if key != "evidence_contract_sha256"
        }
    )
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(
        AgentReviewReceiptError,
        match="REVIEW_EVIDENCE_(PRIVACY|BINDING)_INVALID",
    ):
        finalize_task4_agent_review(
            root, project_root=project, evidence_relative_paths=paths
        )


def test_task4_review_rejects_schema_20_pass_stubs_without_proof_fields(tmp_path):
    from src.method_y_revival.contracts import sha256_mapping
    from src.method_y_revival.review_receipt import (
        AgentReviewReceiptError,
        validate_task4_review_evidence,
    )

    project = tmp_path / "project"
    result_root = project / "results" / "method_y_revival_v2"
    rebuilt = _write_synthetic_task4_source(result_root)
    root = result_root / "pretraining_contracts"
    _build_synthetic_task4_contracts(
        project_root=project,
        source=result_root,
        output=root,
        private=root / "_private",
        rebuilt=rebuilt,
        seed=23,
    )
    state = json.loads((root / "pretraining_validation_state.json").read_text(encoding="utf-8"))
    _write_synthetic_task4_review_evidence(root)
    path = root / "review_evidence" / "invariant_test_suite_v2.json"
    stub = {
        "schema_version": "2.0",
        "layer_id": "INVARIANT_TEST_SUITE",
        "status": "PASS",
        "implementation_commit": state["implementation_commit"],
        "run_contract_sha256": state["run_contract_sha256"],
        "reviewed_artifact_sha256": state["artifact_sha256"],
        "zero_failures": True,
        "test_count": 150,
    }
    stub["evidence_contract_sha256"] = sha256_mapping(stub)
    path.write_text(json.dumps(stub, sort_keys=True), encoding="utf-8")
    with pytest.raises(AgentReviewReceiptError, match="REVIEW_EVIDENCE_BINDING_INVALID"):
        validate_task4_review_evidence(
            root,
            evidence_relative_paths=[
                "review_evidence/invariant_test_suite_v2.json",
                "review_evidence/deterministic_rebuild_v2.json",
                "review_evidence/adversarial_agent_review_v2.json",
            ],
            implementation_commit=state["implementation_commit"],
            run_contract_sha256=state["run_contract_sha256"],
            reviewed_artifact_sha256=state["artifact_sha256"],
        )


@pytest.mark.parametrize(
    "mutation,expected",
    [
        ({"locked": False}, "PACC_CONTRACT_NOT_LOCKED"),
        ({"raw_identifiers_present": True}, "PACC_PRIVATE_FLAG_INVALID"),
        ({"source_sha256": {}}, "PACC_SOURCE_HASHES_INVALID"),
    ],
)
def test_pacc_contract_fails_closed_on_lock_privacy_and_source_hashes(
    tmp_path, mutation, expected
):
    from src.method_y_revival.pretraining_contracts import (
        PretrainingContractError,
        verify_pacc_secondary_endpoint,
    )

    root = tmp_path / "source"
    rebuilt = _write_synthetic_task4_source(root)
    path = root / "endpoints" / "longitudinal_pacc_endpoint_contract.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload.update(mutation)
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(PretrainingContractError, match=expected):
        verify_pacc_secondary_endpoint(
            root,
            expected_source_hashes={"pacc": "e" * 64},
            expected_input_contract_sha256="d" * 64,
            rebuilt_frame=rebuilt["longitudinal_pacc"],
        )


@pytest.mark.parametrize("mode", ["duplicate", "blank_id", "bad_years"])
def test_pacc_rows_require_unique_identity_time_and_consistent_years(tmp_path, mode):
    import hashlib

    from src.method_y_revival.pretraining_contracts import (
        PretrainingContractError,
        verify_pacc_secondary_endpoint,
    )

    root = tmp_path / "source"
    rebuilt = _write_synthetic_task4_source(root)
    endpoint_root = root / "endpoints"
    contract_path = endpoint_root / "longitudinal_pacc_endpoint_contract.json"
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    artifact = endpoint_root / contract["endpoint_path"]
    frame = pd.read_csv(artifact)
    if mode == "duplicate":
        frame.loc[1, ["patient_hash", "time_days_from_pet"]] = frame.loc[
            0, ["patient_hash", "time_days_from_pet"]
        ]
    elif mode == "blank_id":
        frame.loc[0, "patient_hash"] = ""
    else:
        frame.loc[0, "time_years"] += 1.0
    frame.to_csv(artifact, index=False, lineterminator="\n")
    contract["endpoint_sha256"] = hashlib.sha256(artifact.read_bytes()).hexdigest()
    contract_path.write_text(json.dumps(contract), encoding="utf-8")
    with pytest.raises(PretrainingContractError):
        verify_pacc_secondary_endpoint(
            root,
            expected_source_hashes={"pacc": "e" * 64},
            expected_input_contract_sha256="d" * 64,
            rebuilt_frame=rebuilt["longitudinal_pacc"],
        )


@pytest.mark.parametrize("mode", ["forged_input_digest", "tampered_measurement"])
def test_pacc_rejects_self_consistent_contract_tamper_without_source_rebuild_match(
    tmp_path, mode
):
    import hashlib

    from src.method_y_revival.contracts import sha256_mapping
    from src.method_y_revival.pretraining_contracts import (
        PretrainingContractError,
        verify_pacc_secondary_endpoint,
    )

    root = tmp_path / "source"
    rebuilt = _write_synthetic_task4_source(root)
    endpoint_root = root / "endpoints"
    contract_path = endpoint_root / "longitudinal_pacc_endpoint_contract.json"
    audit_path = endpoint_root / "longitudinal_pacc_endpoint_audit.json"
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    if mode == "forged_input_digest":
        contract["input_contract_sha256"] = "0" * 64
    else:
        artifact = endpoint_root / contract["endpoint_path"]
        frame = pd.read_csv(artifact)
        frame.loc[0, "pacc_raw"] = 999999.0
        frame.to_csv(artifact, index=False, lineterminator="\n")
        contract["endpoint_sha256"] = hashlib.sha256(artifact.read_bytes()).hexdigest()
    contract_path.write_text(json.dumps(contract), encoding="utf-8")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    audit["contract_sha256"] = sha256_mapping(contract)
    audit_path.write_text(json.dumps(audit), encoding="utf-8")

    with pytest.raises(PretrainingContractError, match="PACC_.*INVALID|PACC_REBUILD_MISMATCH"):
        verify_pacc_secondary_endpoint(
            root,
            expected_source_hashes={"pacc": "e" * 64},
            expected_input_contract_sha256="d" * 64,
            rebuilt_frame=rebuilt["longitudinal_pacc"],
        )


def test_pacc_requires_independent_rebuild(tmp_path):
    from src.method_y_revival.pretraining_contracts import (
        PretrainingContractError,
        verify_pacc_secondary_endpoint,
    )

    root = tmp_path / "source"
    _write_synthetic_task4_source(root)
    with pytest.raises(PretrainingContractError, match="PACC_INDEPENDENT_REBUILD_REQUIRED"):
        verify_pacc_secondary_endpoint(
            root,
            expected_source_hashes={"pacc": "e" * 64},
            expected_input_contract_sha256="d" * 64,
        )


def test_branch_readiness_requires_the_exact_private_split_artifact(tmp_path):
    import hashlib

    from src.method_y_revival.branch_readiness import collect_project_evidence

    project = tmp_path / "project"
    result_root = project / "results" / "method_y_revival_v2"
    rebuilt = _write_synthetic_task4_source(result_root)
    contract_root = result_root / "pretraining_contracts"
    _build_synthetic_task4_contracts(
        project_root=project,
        source=result_root,
        output=contract_root,
        private=contract_root / "_private",
        rebuilt=rebuilt,
        seed=29,
    )
    _finalize_synthetic_task4_review(contract_root, project_root=project)
    endpoint_path = contract_root / "endpoint_verification_v2.json"
    endpoint = json.loads(endpoint_path.read_text(encoding="utf-8"))
    assert all(
        record["independent_rebuild"]["performed"] is True
        and record["independent_rebuild"]["row_digest_match"] is True
        for record in endpoint["primary_endpoints"].values()
    )

    feature_root = result_root / "feature_contracts"
    feature_root.mkdir()
    feature_path = feature_root / "amyloid_source_regional_feature_contract.json"
    feature_path.write_text(
        json.dumps({"locked": True, "feature_digest": "f" * 64}),
        encoding="utf-8",
    )
    feature_sha = hashlib.sha256(feature_path.read_bytes()).hexdigest()
    (feature_root / "source_regional_validation_state.json").write_text(
        json.dumps(
            {
                "status": "COMMITTED",
                "development_allowed": True,
                "artifact_sha256": {feature_path.name: feature_sha},
            }
        ),
        encoding="utf-8",
    )
    evidence = collect_project_evidence(
        project, branches=("amyloid_source_regional",)
    )
    assert set(evidence) >= {
        "amyloid_source_regional_feature_contract",
        "amyloid_source_regional_endpoint_contract",
        "amyloid_source_regional_split_contract",
        "method_y_predictor_temporality_contract",
    }

    split = json.loads(
        (contract_root / "split_verification_v2.json").read_text(encoding="utf-8")
    )
    private_name = split["branches"]["amyloid_source_regional"][
        "private_artifact_basename"
    ]
    (contract_root / "_private" / private_name).write_text("{}", encoding="utf-8")
    after_tamper = collect_project_evidence(
        project, branches=("amyloid_source_regional",)
    )
    assert "amyloid_source_regional_split_contract" not in after_tamper


@pytest.mark.parametrize(
    "failure",
    [
        "missing_state",
        "blocked_state",
        "wrong_stage",
        "training_started",
        "outer_started",
        "external_accessed",
        "missing_artifact",
        "wrong_artifact_hash",
        "missing_review_receipt",
        "wrong_review_receipt_hash",
        "tampered_review_evidence",
        "run_contract_tamper",
        "temporality_tamper",
    ],
)
def test_task4_state_is_mandatory_all_or_none_for_source_readiness(tmp_path, failure):
    import hashlib

    from src.method_y_revival.branch_readiness import (
        build_branch_readiness,
        collect_project_evidence,
    )

    project = tmp_path / "project"
    result_root = project / "results" / "method_y_revival_v2"
    rebuilt = _write_synthetic_task4_source(result_root)
    contract_root = result_root / "pretraining_contracts"
    _build_synthetic_task4_contracts(
        project_root=project,
        source=result_root,
        output=contract_root,
        private=contract_root / "_private",
        rebuilt=rebuilt,
        seed=31,
    )
    _finalize_synthetic_task4_review(contract_root, project_root=project)
    endpoint_path = contract_root / "endpoint_verification_v2.json"
    endpoint = json.loads(endpoint_path.read_text(encoding="utf-8"))
    assert all(
        record["independent_rebuild"]["performed"] is True
        and record["independent_rebuild"]["row_digest_match"] is True
        for record in endpoint["primary_endpoints"].values()
    )
    feature_root = result_root / "feature_contracts"
    feature_root.mkdir()
    feature_path = feature_root / "amyloid_source_regional_feature_contract.json"
    feature_path.write_text(
        json.dumps({"locked": True, "feature_digest": "f" * 64}),
        encoding="utf-8",
    )
    feature_sha = hashlib.sha256(feature_path.read_bytes()).hexdigest()
    (feature_root / "source_regional_validation_state.json").write_text(
        json.dumps(
            {
                "status": "COMMITTED",
                "development_allowed": True,
                "artifact_sha256": {feature_path.name: feature_sha},
            }
        ),
        encoding="utf-8",
    )
    state_path = contract_root / "pretraining_validation_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))

    if failure == "missing_state":
        state_path.unlink()
    elif failure == "blocked_state":
        state["status"] = "BLOCKED"
        state_path.write_text(json.dumps(state), encoding="utf-8")
    elif failure == "wrong_stage":
        state["stage"] = "WRONG"
        state_path.write_text(json.dumps(state), encoding="utf-8")
    elif failure == "training_started":
        state["training_started"] = True
        state_path.write_text(json.dumps(state), encoding="utf-8")
    elif failure == "outer_started":
        state["outer_evaluation_started"] = True
        state_path.write_text(json.dumps(state), encoding="utf-8")
    elif failure == "external_accessed":
        state["external_outcomes_accessed"] = True
        state_path.write_text(json.dumps(state), encoding="utf-8")
    elif failure == "missing_artifact":
        (contract_root / "external_intake_contract_v2.json").unlink()
    elif failure == "wrong_artifact_hash":
        state["artifact_sha256"]["external_intake_contract_v2.json"] = "0" * 64
        state_path.write_text(json.dumps(state), encoding="utf-8")
    elif failure == "missing_review_receipt":
        (contract_root / "agent_multi_pass_review_receipt_v2.json").unlink()
    elif failure == "wrong_review_receipt_hash":
        state["artifact_sha256"]["agent_multi_pass_review_receipt_v2.json"] = "0" * 64
        state_path.write_text(json.dumps(state), encoding="utf-8")
    elif failure == "tampered_review_evidence":
        evidence_path = (
            contract_root / "review_evidence" / "invariant_test_suite_v2.json"
        )
        evidence_path.write_text(
            evidence_path.read_text(encoding="utf-8") + " ", encoding="utf-8"
        )
    elif failure == "run_contract_tamper":
        state["run_contract"]["split_seed"] = 999
        state_path.write_text(json.dumps(state), encoding="utf-8")
    else:
        temporality = contract_root / "predictor_temporality_contract_v2.json"
        temporality.write_text(temporality.read_text(encoding="utf-8") + " ", encoding="utf-8")

    evidence = collect_project_evidence(
        project, branches=("amyloid_source_regional",)
    )
    readiness = build_branch_readiness(evidence)["branches"][
        "amyloid_source_regional"
    ]
    assert readiness["development_allowed"] is False
    assert "amyloid_source_regional_endpoint_contract" not in evidence
    assert "amyloid_source_regional_split_contract" not in evidence
    assert "method_y_predictor_temporality_contract" not in evidence


@pytest.mark.parametrize(
    "case",
    ["output_outside_project", "private_outside_output", "source_inside_output"],
)
def test_task4_cli_roots_are_contained_and_source_is_separate_read_only_role(
    tmp_path, case
):
    from scripts.build_method_y_pretraining_contracts_v2 import validate_cli_roots
    from src.method_y_revival.pretraining_contracts import PretrainingContractError

    project = tmp_path / "project"
    source = tmp_path / "source"
    output = project / "results" / "method_y_revival_v2" / "pretraining_contracts"
    private = output / "_private"
    project.mkdir()
    source.mkdir()
    output.mkdir(parents=True)
    private.mkdir()
    if case == "output_outside_project":
        output = tmp_path / "outside"
    elif case == "private_outside_output":
        private = project / "private"
    else:
        source = output / "source"
        source.mkdir()
    with pytest.raises(PretrainingContractError):
        validate_cli_roots(
            project_root=project,
            source_root=source,
            output_root=output,
            private_output_root=private,
        )
