from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from scripts.run_method_y_image_factorial_v4 import (
    FactorialLauncherV4Error,
    _validate_development_split_contract,
)
from scripts.verify_method_y_image_factorial_v4 import _privacy_violations
from src.method_y_revival.atomic_io import read_json
from src.method_y_revival.contracts import sha256_mapping
from src.method_y_revival.image_factorial_orchestrator_v4 import (
    build_factorial_cell_contract,
    resolved_arm_model_config,
)
from src.method_y_revival.image_factorial_reporting_v4 import (
    _scaled_oof_scanner_probe,
    factorial_pair_acceptance_gate,
)


ROOT = Path(__file__).resolve().parents[2]


def _config() -> dict:
    return yaml.safe_load(
        (ROOT / "configs" / "method_y_image_factorial_v4.yaml").read_text(
            encoding="utf-8"
        )
    )


def _contract(config: dict) -> str:
    return build_factorial_cell_contract(
        config=config,
        branch="tau",
        arm_id="A3",
        outer_fold=2,
        candidate_id="candidate",
        membership_sha256="1" * 64,
        cache_sha256="2" * 64,
        mask_sha256="3" * 64,
        split_sha256="4" * 64,
        augmentation_qc_sha256="5" * 64,
        stage="INNER_HPO",
        inner_fold=1,
        partition_contract={"train_subjects_sha256": "6" * 64},
        training_parameters={"learning_rate": 0.0001},
    )


def test_primary_factorial_pairs_have_identical_nonmanipulated_model_config() -> None:
    config = _config()
    assert resolved_arm_model_config(config, "A0") == resolved_arm_model_config(config, "A1")
    assert resolved_arm_model_config(config, "A2") == resolved_arm_model_config(config, "A3")
    assert config["model"]["hpo_candidates"]
    assert config["execution"]["arms"] == ["A0", "A1", "A2", "A3", "A4"]


def test_run_contract_binds_augmentation_software_and_runtime_policy() -> None:
    config = _config()
    config["training_source_sha256"] = "a" * 64
    config["software_runtime_contract"] = {"contract_sha256": "b" * 64}
    baseline = _contract(config)
    changed_source = copy.deepcopy(config)
    changed_source["training_source_sha256"] = "c" * 64
    changed_math = copy.deepcopy(config)
    changed_math["runtime"]["allow_tf32"] = True
    changed_augmentation = copy.deepcopy(config)
    changed_augmentation["augmentation"]["max_rotation_degrees"] = 0.5
    assert baseline != _contract(changed_source)
    assert baseline != _contract(changed_math)
    assert baseline != _contract(changed_augmentation)


def test_run_contract_binds_exact_inner_partition() -> None:
    config = _config()
    config["training_source_sha256"] = "a" * 64
    first = _contract(config)
    changed = build_factorial_cell_contract(
        config=config,
        branch="tau",
        arm_id="A3",
        outer_fold=2,
        candidate_id="candidate",
        membership_sha256="1" * 64,
        cache_sha256="2" * 64,
        mask_sha256="3" * 64,
        split_sha256="4" * 64,
        augmentation_qc_sha256="5" * 64,
        stage="INNER_HPO",
        inner_fold=2,
        partition_contract={"train_subjects_sha256": "7" * 64},
        training_parameters={"learning_rate": 0.0001},
    )
    assert first != changed


def test_locked_qc_contract_selected_primary_and_rejected_upper_envelope() -> None:
    path = (
        ROOT
        / "results"
        / "method_y_revival_v4_image_factorial_v1"
        / "provenance"
        / "pet_augmentation_qc_contract.json"
    )
    payload = read_json(path)
    claimed = payload["contract_sha256"]
    body = {key: value for key, value in payload.items() if key != "contract_sha256"}
    assert claimed == sha256_mapping(body)
    assert payload["selected_envelope"]["candidate_id"] == "primary"
    statuses = {row["candidate_id"]: row["status"] for row in payload["candidate_results"]}
    assert statuses == {"identity": "PASS", "primary": "PASS", "upper": "FAIL"}
    serialized = str(payload).lower()
    assert "patient_hash" not in serialized
    assert "time_days" in payload["explicitly_unused"]


def test_public_privacy_guard_rejects_row_level_survival_content() -> None:
    assert _privacy_violations({"aggregate": {"event": [0, 1], "time_days": [1, 2]}})
    assert not _privacy_violations(
        {
            "aggregate": {
                "event_count": 10,
                "integrated_brier_score": 0.15,
                "confidence_interval": [0.1, 0.2],
            }
        }
    )


def test_augmentation_does_not_change_real_subject_or_event_counts() -> None:
    config = _config()
    assert config["claim_policy"]["augmentation_creates_new_subjects"] is False
    assert config["augmentation"]["exposure_scope"] == "TRAIN_ONLY"
    assert config["augmentation"]["validation_and_test_are_unaugmented"] is True


def _small_nested_split() -> tuple[pd.DataFrame, dict]:
    subjects = [f"p{index}" for index in range(10)]
    development = pd.DataFrame(
        {
            "patient_hash": subjects,
            "event": [0, 1] * 5,
            "time_days": [100.0 + index for index in range(10)],
        }
    )
    outer_folds = []
    for fold in range(5):
        test = {subjects[2 * fold], subjects[2 * fold + 1]}
        train = set(subjects).difference(test)
        inner = []
        train_by_event = {
            event: [key for key in sorted(train) if int(key[1:]) % 2 == event]
            for event in (0, 1)
        }
        for inner_fold in range(3):
            validation = {
                train_by_event[0][inner_fold % len(train_by_event[0])],
                train_by_event[1][inner_fold % len(train_by_event[1])],
            }
            inner.append(
                {
                    "fold": inner_fold,
                    "train_subjects": sorted(train.difference(validation)),
                    "validation_subjects": sorted(validation),
                }
            )
        outer_folds.append(
            {
                "fold": fold,
                "train_subjects": sorted(train),
                "test_subjects": sorted(test),
                "inner_folds": inner,
            }
        )
    return development, {"outer_folds": outer_folds}


def test_nested_split_preflight_accepts_complete_subject_level_contract() -> None:
    development, split = _small_nested_split()
    summary = _validate_development_split_contract(development, split, branch="tau")
    assert summary["subjects"] == 10
    assert summary["events"] == 5
    assert summary["outer_test_coverage"] == 10
    assert summary["outer_folds"] == 5
    assert summary["inner_folds"] == 3


@pytest.mark.parametrize(
    ("mutator", "reason"),
    [
        (
            lambda split: split["outer_folds"][0]["train_subjects"].append("p0"),
            "OUTER_PARTITION_LEAKAGE",
        ),
        (
            lambda split: split["outer_folds"][0]["test_subjects"].remove("p0"),
            "OUTER_PARTITION_INCOMPLETE",
        ),
        (
            lambda split: split["outer_folds"][0]["inner_folds"].pop(),
            "EXPECTED_THREE_INNER_FOLDS",
        ),
    ],
)
def test_nested_split_preflight_fails_closed_on_invalid_membership(mutator, reason: str) -> None:
    development, split = _small_nested_split()
    mutator(split)
    with pytest.raises(FactorialLauncherV4Error, match=reason):
        _validate_development_split_contract(development, split, branch="tau")


def test_factorial_pair_gate_is_generic_and_lmo_fail_closed() -> None:
    margins = _config()["factorial_comparison"]
    blocked = factorial_pair_acceptance_gate(
        delta_ibs_ci=(-0.02, -0.005),
        delta_uno_ci=(0.001, 0.03),
        fold_delta_ibs=[-0.01] * 5,
        calibration_not_worse=True,
        selected_epoch_nll_gap_delta=0.0,
        scanner_probe_not_worse=True,
        leave_manufacturer_out_performance_established=False,
        margins=margins,
    )
    assert blocked["exploratory_operational_acceptance"] is False
    assert "LEAVE_MANUFACTURER_OUT_PERFORMANCE_NOT_ESTABLISHED" in blocked["blockers"]
    assert blocked["positive_claim_allowed"] is False
    assert "scalar_tabular" not in str(blocked)

    accepted = factorial_pair_acceptance_gate(
        delta_ibs_ci=(-0.02, -0.005),
        delta_uno_ci=(0.001, 0.03),
        fold_delta_ibs=[-0.01] * 5,
        calibration_not_worse=True,
        selected_epoch_nll_gap_delta=0.0,
        scanner_probe_not_worse=True,
        leave_manufacturer_out_performance_established=True,
        margins=margins,
    )
    assert accepted["exploratory_operational_acceptance"] is True
    assert accepted["positive_claim_allowed"] is False


def test_v4_scanner_probe_scales_embeddings_and_converges() -> None:
    rng = np.random.default_rng(7)
    manufacturers = ["GE"] * 12 + ["SIEMENS"] * 12
    embeddings = np.vstack(
        (
            rng.normal(loc=-0.5, scale=0.3, size=(12, 6)),
            rng.normal(loc=0.5, scale=0.3, size=(12, 6)),
        )
    )
    result = _scaled_oof_scanner_probe(embeddings, manufacturers, seed=20260714)
    assert result["status"] == "COMPLETED"
    assert result["preprocessing"] == "STANDARD_SCALER_FIT_WITHIN_PROBE_FOLD"
    assert 0.0 <= result["macro_auc"] <= 1.0
    assert result["convergence_warning_count"] == 0
