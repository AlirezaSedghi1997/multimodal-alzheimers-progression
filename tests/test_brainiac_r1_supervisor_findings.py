"""Adversarial regressions for the independent BrainIAC R1 review.

These tests exercise only synthetic arrays and tensors.  They never vendor or
load BrainIAC, train a model, or expose the outer test set.
"""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest
import torch

from src.data.combined_mri_cache import METHOD_X_PHENOTYPE_FEATURES
from src.research.brainiac_adapter import (
    assess_r1a_readiness,
    adapter_provenance,
    evaluate_geometry,
    fit_frozen_probe,
    validate_strict_load,
)
from src.research.brainiac_caft import (
    ABLATION_ARMS,
    BrainIACCAFT,
    CaftConfig,
    build_locked_confound_artifact,
    gate_penalty,
    lora_config,
    paired_bootstrap_auc_delta,
    prediction_artifact_digest,
    residual_fusion_logit,
    evaluate_promotion,
    validate_equal_budget_contract,
)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _feature_schema_sha256() -> str:
    encoded = json.dumps(
        list(METHOD_X_PHENOTYPE_FEATURES), separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def test_default_geometry_remains_unknown_until_vendor_template_affine_is_verified() -> None:
    report = evaluate_geometry()
    assert report.status == "UNKNOWN_GEOMETRY_PENDING_VENDOR_TEMPLATE_AFFINE"
    assert report.passed is False
    assert "VENDOR_TEMPLATE_AFFINE_UNVERIFIED" in report.findings

    verified = evaluate_geometry(
        source_resolution=96,
        source_voxel_mm=2.0,
        target_resolution=96,
        declared_target_voxel_mm=2.0,
        source_registration="rigid_mni",
        target_registration="rigid_mni",
        target_geometry_verified=True,
    )
    assert verified.status == "PASS"
    assert verified.passed is True


def test_strict_load_requires_nonempty_full_shape_maps() -> None:
    with pytest.raises(RuntimeError, match="STRICT_LOAD_SHAPE_EVIDENCE_EMPTY"):
        validate_strict_load(
            ["encoder.weight"],
            ["encoder.weight"],
            shapes={},
            expected_shapes={},
        )
    with pytest.raises(RuntimeError, match="STRICT_LOAD_SHAPE_KEYSET_MISMATCH"):
        validate_strict_load(
            ["encoder.weight", "encoder.bias"],
            ["encoder.weight", "encoder.bias"],
            shapes={"encoder.weight": (2, 2)},
            expected_shapes={"encoder.weight": (2, 2)},
        )


@pytest.mark.parametrize(
    "overrides",
    [
        {"correction_l2_weight": float("nan")},
        {"residual_bound": float("nan")},
        {"gate_bias_init": float("-inf")},
    ],
)
def test_caft_configuration_rejects_nonfinite_hyperparameters(overrides) -> None:
    with pytest.raises(
        RuntimeError,
        match="CORRECTION_MUST_BE_REGULARIZED|CAFT_.*INVALID|CAFT_GATE_BIAS_NOT_NEAR_ZERO",
    ):
        CaftConfig(**overrides).validate()


def test_numpy_fusion_helpers_reject_nonfinite_values() -> None:
    with pytest.raises(RuntimeError, match="GATE_NOT_FINITE"):
        residual_fusion_logit(np.array([0.0]), np.array([np.nan]), np.array([1.0]))
    with pytest.raises(RuntimeError, match="CORRECTION_NOT_FINITE"):
        gate_penalty(np.array([1.0]), np.array([np.nan]), weight=0.1)


def test_lora_version_aliases_and_unregistered_r2_are_fail_closed() -> None:
    for alias in ("r1_caft", "brainiac-r1c", "future", "R2_LORA_ALIAS"):
        with pytest.raises(RuntimeError, match="LORA_PROTOCOL_NOT_REGISTERED"):
            lora_config(
                frozen_stage_promoted=True,
                evidence_scope="inner_oof",
                run_version=alias,
            )
    with pytest.raises(RuntimeError, match="LORA_PROTOCOL_MANIFEST_REQUIRED"):
        lora_config(
            frozen_stage_promoted=True,
            evidence_scope="inner_oof",
            run_version="r2_lora",
        )


def test_probe_rejects_a_unique_unselected_outer_row() -> None:
    rng = np.random.default_rng(37)
    embeddings = rng.normal(size=(13, 4))
    labels = np.asarray([0, 1] * 6 + [0])
    tabular = rng.normal(size=(13, 5))
    roles = ["inner_train"] * 8 + ["inner_validation"] * 4 + ["outer_test"]
    hashes = [_sha(f"probe-{index}") for index in range(13)]
    with pytest.raises(RuntimeError, match="PROBE_OUTER_ROWS_PRESENT"):
        fit_frozen_probe(
            embeddings,
            labels,
            inner_train=tuple(range(8)),
            inner_validation=tuple(range(8, 12)),
            tabular=tabular,
            subject_hashes=hashes,
            split_roles=roles,
            real_flags=[True] * 13,
            parent_hashes=[None] * 13,
            feature_names=METHOD_X_PHENOTYPE_FEATURES,
            split_provenance={
                "selection_role": "inner_validation",
                "outer_test_touched": False,
                "split_hash": _sha("split"),
                "group_hash": _sha("group"),
                "embedding_artifact_sha256": _sha("embedding"),
            },
        )


def test_caft_forward_requires_canonical_feature_and_preprocessing_binding() -> None:
    model = BrainIACCAFT(CaftConfig())
    tensors = (
        torch.zeros(2, 1),
        torch.zeros(2, 128),
        torch.zeros(2, 768),
        torch.zeros(2, 5),
    )
    with pytest.raises(RuntimeError, match="CAFT_TABULAR_CONTRACT_REQUIRED"):
        model(*tensors)

    schema_sha = _feature_schema_sha256()
    receipt = {
        "locked": True,
        "fit_role": "head_fit",
        "imputation": "median",
        "scaling": "standard",
        "feature_schema_sha256": schema_sha,
        "artifact_sha256": _sha("tabular-preprocessing"),
        "finite_output": True,
        "outer_test_touched": False,
    }
    output = model(
        *tensors,
        feature_names=METHOD_X_PHENOTYPE_FEATURES,
        feature_schema_sha256=schema_sha,
        tabular_preprocessing_receipt=receipt,
    )
    assert torch.isfinite(output["final_logit"]).all()

    with pytest.raises(RuntimeError, match="FEATURE_ORDER_VIOLATION"):
        model(
            *tensors,
            feature_names=tuple(reversed(METHOD_X_PHENOTYPE_FEATURES)),
            feature_schema_sha256=schema_sha,
            tabular_preprocessing_receipt=receipt,
        )


def test_caft_forward_rejects_nonfinite_learned_parameters() -> None:
    model = BrainIACCAFT(CaftConfig())
    with torch.no_grad():
        model.residual_head.bias.fill_(float("nan"))
    schema_sha = _feature_schema_sha256()
    with pytest.raises(RuntimeError, match="CAFT_OUTPUT_NOT_FINITE"):
        model(
            torch.zeros(2, 1),
            torch.zeros(2, 128),
            torch.zeros(2, 768),
            torch.zeros(2, 5),
            feature_names=METHOD_X_PHENOTYPE_FEATURES,
            feature_schema_sha256=schema_sha,
            tabular_preprocessing_receipt={
                "locked": True,
                "fit_role": "head_fit",
                "imputation": "median",
                "scaling": "standard",
                "feature_schema_sha256": schema_sha,
                "artifact_sha256": _sha("tabular-preprocessing"),
                "finite_output": True,
                "outer_test_touched": False,
            },
        )


def test_sensitivity_adapter_provenance_keeps_vendor_geometry_unknown() -> None:
    report = adapter_provenance(
        np.zeros((4, 4, 4), dtype=np.float32), sample_hash=_sha("sample")
    )
    assert report["geometry_status"] == "UNKNOWN_GEOMETRY_PENDING_VENDOR_TEMPLATE_AFFINE"


def _bound_budget_contract() -> dict[str, dict[str, object]]:
    common: dict[str, object] = {
        "split_hash": _sha("split"),
        "fit_subject_hash": _sha("fit-subjects"),
        "max_steps": 200,
        "batch_size": 8,
        "optimizer": {
            "name": "adamw",
            "learning_rate": 1e-4,
            "weight_decay": 1e-4,
        },
        "scheduler": {"name": "cosine", "warmup_steps": 10},
        "early_stopping": {"monitor": "inner_validation_auc", "patience": 10},
        "selection_role": "inner_validation",
        "outer_test_touched": False,
        "encoders_frozen": True,
    }
    return {
        arm: {
            **common,
            "arm_id": arm,
            "model_class": f"Synthetic{index}",
            "model_artifact_sha256": _sha(f"model-{arm}"),
            "prediction_artifact_sha256": _sha(f"predictions-{arm}"),
            "step_history_sha256": _sha(f"steps-{arm}"),
            "trainable_parameter_policy": f"locked-policy-{arm}",
        }
        for index, arm in enumerate(ABLATION_ARMS)
    }


def test_equal_budget_contract_rejects_empty_recipe_and_unbound_arms() -> None:
    invalid = _bound_budget_contract()
    for row in invalid.values():
        row["optimizer"] = None
    with pytest.raises(RuntimeError, match="BASELINE_OPTIMIZER_INVALID"):
        validate_equal_budget_contract(invalid)

    invalid = _bound_budget_contract()
    invalid["brainiac_only"].pop("prediction_artifact_sha256")
    with pytest.raises(RuntimeError, match="BASELINE_ARM_BINDING_MISSING"):
        validate_equal_budget_contract(invalid)

    assert validate_equal_budget_contract(_bound_budget_contract())["status"] == "PASS"


def _promotion_prediction_artifact() -> dict[str, object]:
    artifact: dict[str, object] = {
        "locked": True,
        "split_role": "oof_inner_training",
        "outer_test_touched": False,
        "subject_hashes": [_sha(f"promotion-{index}") for index in range(8)],
        "labels": [0, 0, 0, 0, 1, 1, 1, 1],
        "baseline_probabilities": [0.1, 0.2, 0.8, 0.4, 0.3, 0.6, 0.7, 0.9],
        "candidate_probabilities": [0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9],
    }
    artifact["sha256"] = prediction_artifact_digest(artifact)
    return artifact


def _legacy_confound(decodability: float) -> dict[str, object]:
    return {
        "status": "MEASURABLE",
        "decodability": decodability,
        "ci_lower": max(0.5, decodability - 0.05),
        "ci_upper": min(1.0, decodability + 0.05),
        "null_ci_lower": 0.50,
        "null_ci_upper": 0.62,
        "permutation_iterations": 1000,
    }


def test_promotion_recomputes_bootstrap_and_ignores_a_forged_positive_summary() -> None:
    artifact, registry = _prediction_with_split_registry()
    actual = paired_bootstrap_auc_delta(
        artifact["baseline_probabilities"],
        artifact["candidate_probabilities"],
        artifact["labels"],
        iterations=1000,
        seed=13,
    )
    assert actual["iterations"] == 1000
    assert actual["lower_ci_positive"] is False
    hashes = list(artifact["subject_hashes"])
    cohort_labels = [0, 1, 0, 1, 0, 1, 0, 1]
    result = evaluate_promotion(
        locked_predictions=artifact,
        locked_split_registry=registry,
        expected_split_registry_sha256=registry["integrity_sha256"],
        baseline_confound=build_locked_confound_artifact(
            scores=[0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.4, 0.6],
            labels=cohort_labels,
            subject_hashes=hashes,
            prediction_artifact_sha256=artifact["sha256"],
            split_registry_sha256=registry["integrity_sha256"],
            iterations=1000,
            seed=13,
        ),
        candidate_confound=build_locked_confound_artifact(
            scores=[0.45, 0.48, 0.62, 0.41, 0.55, 0.39, 0.58, 0.44],
            labels=cohort_labels,
            subject_hashes=hashes,
            prediction_artifact_sha256=artifact["sha256"],
            split_registry_sha256=registry["integrity_sha256"],
            iterations=1000,
            seed=13,
        ),
        bootstrap={
            "prediction_artifact_sha256": artifact["sha256"],
            "iterations": 1000,
            "seed": 13,
            "lower_ci_positive": True,
            "ci_lower": 0.01,
            "ci_upper": 0.40,
        },
        coverage={
            "coverage_met": True,
            "empirical_coverage": 0.92,
            "target_coverage": 0.90,
            "method": "split_conformal_marginal",
            "calibration_partitions_validated": True,
            "real_only": True,
            "class_coverage_met": True,
            "cohort_coverage_status": "MEASURABLE_PASS",
        },
        sanity_passed=True,
    )
    assert result["promoted"] is False
    assert "PAIRED_CI_LOWER_NOT_POSITIVE" in result["blockers"]
    assert result["paired_bootstrap"]["ci_lower"] == pytest.approx(actual["ci_lower"])


def _split_registry(subject_hashes: list[str], roles: list[str]) -> dict[str, object]:
    body: dict[str, object] = {
        "schema_version": 1,
        "locked": True,
        "split_hash": _sha("split"),
        "group_hash": _sha("group"),
        "subjects": [
            {
                "subject_hash": subject_hash,
                "role": role,
                "real": True,
                "parent_hash": None,
            }
            for subject_hash, role in zip(subject_hashes, roles, strict=True)
        ],
    }
    return {
        **body,
        "integrity_sha256": hashlib.sha256(
            json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
    }


def test_probe_requires_an_externally_pinned_locked_split_registry() -> None:
    rng = np.random.default_rng(41)
    embeddings = rng.normal(size=(12, 4))
    labels = np.asarray([0, 1] * 6)
    tabular = rng.normal(size=(12, 5))
    roles = ["inner_train"] * 8 + ["inner_validation"] * 4
    hashes = [_sha(f"registry-probe-{index}") for index in range(12)]
    registry = _split_registry(hashes, roles)
    provenance = {
        "selection_role": "inner_validation",
        "outer_test_touched": False,
        "split_hash": registry["split_hash"],
        "group_hash": registry["group_hash"],
        "embedding_artifact_sha256": _sha("embedding"),
        "split_registry_sha256": registry["integrity_sha256"],
    }
    common = {
        "inner_train": tuple(range(8)),
        "inner_validation": tuple(range(8, 12)),
        "tabular": tabular,
        "subject_hashes": hashes,
        "split_roles": roles,
        "real_flags": [True] * 12,
        "parent_hashes": [None] * 12,
        "feature_names": METHOD_X_PHENOTYPE_FEATURES,
        "split_provenance": provenance,
    }
    with pytest.raises(RuntimeError, match="PROBE_SPLIT_REGISTRY_REQUIRED"):
        fit_frozen_probe(embeddings, labels, **common)

    result = fit_frozen_probe(
        embeddings,
        labels,
        **common,
        locked_split_registry=registry,
        expected_split_registry_sha256=registry["integrity_sha256"],
    )
    assert result["split_registry_sha256"] == registry["integrity_sha256"]

    tampered = json.loads(json.dumps(registry))
    tampered["subjects"][0]["role"] = "outer_test"
    tampered_body = {
        key: value for key, value in tampered.items() if key != "integrity_sha256"
    }
    tampered["integrity_sha256"] = hashlib.sha256(
        json.dumps(tampered_body, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    with pytest.raises(RuntimeError, match="PROBE_SPLIT_REGISTRY_DIGEST_MISMATCH"):
        fit_frozen_probe(
            embeddings,
            labels,
            **common,
            locked_split_registry=tampered,
            expected_split_registry_sha256=registry["integrity_sha256"],
        )


def _prediction_with_split_registry() -> tuple[dict[str, object], dict[str, object]]:
    artifact = _promotion_prediction_artifact()
    subject_hashes = list(artifact["subject_hashes"])
    outer_hashes = [_sha("outer-0"), _sha("outer-1")]
    registry = _split_registry(
        [*subject_hashes, *outer_hashes],
        ["oof_inner_training"] * len(subject_hashes) + ["outer_test"] * 2,
    )
    artifact.update(
        {
            "split_hash": registry["split_hash"],
            "group_hash": registry["group_hash"],
            "split_registry_sha256": registry["integrity_sha256"],
            "subject_order_sha256": hashlib.sha256(
                json.dumps(subject_hashes, separators=(",", ":")).encode()
            ).hexdigest(),
        }
    )
    artifact["sha256"] = prediction_artifact_digest(artifact)
    return artifact, registry


def test_promotion_binds_prediction_roles_to_the_external_split_registry() -> None:
    artifact, registry = _prediction_with_split_registry()
    common = {
        "locked_predictions": artifact,
        "locked_split_registry": registry,
        "expected_split_registry_sha256": registry["integrity_sha256"],
        "baseline_confound": _legacy_confound(0.60),
        "candidate_confound": _legacy_confound(0.58),
        "bootstrap": {
            "prediction_artifact_sha256": artifact["sha256"],
            "iterations": 1000,
            "seed": 13,
        },
        "coverage": {
            "coverage_met": True,
            "empirical_coverage": 0.92,
            "target_coverage": 0.90,
            "method": "split_conformal_marginal",
            "calibration_partitions_validated": True,
            "real_only": True,
            "class_coverage_met": True,
            "cohort_coverage_status": "MEASURABLE_PASS",
        },
        "sanity_passed": True,
    }
    result = evaluate_promotion(**common)
    assert not any("SPLIT_REGISTRY" in item for item in result["blockers"])

    relabelled = dict(artifact)
    relabelled["split_hash"] = _sha("attacker-split")
    result = evaluate_promotion(**{**common, "locked_predictions": relabelled})
    assert "PREDICTION_ARTIFACT_DIGEST_MISMATCH" in result["blockers"]

    tampered_registry = json.loads(json.dumps(registry))
    tampered_registry["subjects"][0]["role"] = "outer_test"
    tampered_body = {
        key: value
        for key, value in tampered_registry.items()
        if key != "integrity_sha256"
    }
    tampered_registry["integrity_sha256"] = hashlib.sha256(
        json.dumps(tampered_body, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    result = evaluate_promotion(**{**common, "locked_split_registry": tampered_registry})
    assert "PROMOTION_SPLIT_REGISTRY_DIGEST_MISMATCH" in result["blockers"]


def test_r1a_minimal_self_attested_receipts_cannot_reach_pass() -> None:
    preprocessing = {
        "source_role": "raw_t1",
        "label_read": False,
        "source_affine_sha256": _sha("source-affine"),
        "n4_bias_correction": {"status": "PASS", "artifact_sha256": _sha("n4")},
        "rigid_registration": {
            "status": "PASS",
            "transform_sha256": _sha("rigid"),
            "template_sha256": _sha("template"),
        },
        "brain_extraction": {"status": "PASS", "mask_sha256": _sha("mask")},
        "official_96_transform": {
            "status": "PASS",
            "output_shape": [96, 96, 96],
            "output_affine_sha256": _sha("affine"),
        },
        "direct_128_to_96_used": False,
    }
    status = assess_r1a_readiness(
        vendor_manifest={
            "vendored": True,
            "license_sha256": _sha("license"),
            "weight_sha256": _sha("weights"),
            "template_sha256": _sha("template"),
        },
        preprocessing_manifest=preprocessing,
        strict_load_report={"strict": True, "matched_keys": 1},
        forward_report={
            "status": "PASS",
            "input_shape": [1, 1, 96, 96, 96],
            "output_shape": [1, 768],
        },
        overlap_status="NO_PRETRAINING_OVERLAP_PROVEN",
    )
    assert status["ready"] is False
    assert status["status"] in {"UNKNOWN", "BLOCKED"}
    assert set(status["blockers"]) >= {
        "PINNED_VENDOR_COMMIT_MISSING",
        "VENDOR_SOURCE_TREE_HASH_MISSING",
        "VENDOR_ENVIRONMENT_LOCK_MISSING",
        "VENDOR_H_LOCAL_PATH_EVIDENCE_MISSING",
        "STRICT_LOAD_RECEIPT_UNBOUND",
        "FORWARD_RECEIPT_UNBOUND",
        "OVERLAP_EVIDENCE_RECEIPT_MISSING",
    }


def test_promotion_recomputes_confound_from_locked_subject_level_artifacts() -> None:
    prediction, registry = _prediction_with_split_registry()
    hashes = list(prediction["subject_hashes"])
    cohort_labels = [0, 1, 0, 1, 0, 1, 0, 1]
    baseline_confound = build_locked_confound_artifact(
        scores=[0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.4, 0.6],
        labels=cohort_labels,
        subject_hashes=hashes,
        prediction_artifact_sha256=prediction["sha256"],
        split_registry_sha256=registry["integrity_sha256"],
        iterations=1000,
        seed=13,
    )
    candidate_confound = build_locked_confound_artifact(
        scores=[0.45, 0.55, 0.48, 0.52, 0.49, 0.51, 0.47, 0.53],
        labels=cohort_labels,
        subject_hashes=hashes,
        prediction_artifact_sha256=prediction["sha256"],
        split_registry_sha256=registry["integrity_sha256"],
        iterations=1000,
        seed=13,
    )
    common = {
        "locked_predictions": prediction,
        "locked_split_registry": registry,
        "expected_split_registry_sha256": registry["integrity_sha256"],
        "baseline_confound": baseline_confound,
        "candidate_confound": candidate_confound,
        "bootstrap": {
            "prediction_artifact_sha256": prediction["sha256"],
            "iterations": 1000,
            "seed": 13,
        },
        "coverage": {
            "coverage_met": True,
            "empirical_coverage": 0.92,
            "target_coverage": 0.90,
            "method": "split_conformal_marginal",
            "calibration_partitions_validated": True,
            "real_only": True,
            "class_coverage_met": True,
            "cohort_coverage_status": "MEASURABLE_PASS",
        },
        "sanity_passed": True,
    }
    result = evaluate_promotion(**common)
    assert result["confound_source"] == "RECOMPUTED_FROM_LOCKED_SUBJECT_ARTIFACTS"
    assert result["baseline_confound"]["definition"] == "max(AUC,1-AUC)"

    tampered = json.loads(json.dumps(baseline_confound))
    tampered["scores"][0] = 0.99
    result = evaluate_promotion(**{**common, "baseline_confound": tampered})
    assert "CONFOUND_ARTIFACT_DIGEST_MISMATCH:baseline" in result["blockers"]
