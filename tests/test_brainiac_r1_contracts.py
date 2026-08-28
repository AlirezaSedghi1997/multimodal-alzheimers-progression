"""Acceptance contracts for the fail-closed BrainIAC R1 scaffold.

These tests deliberately use only tiny synthetic tensors and arrays.  They do
not vendor BrainIAC, load a checkpoint, train an encoder, or touch an outer
test set.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.combined_mri_cache import METHOD_X_PHENOTYPE_FEATURES  # noqa: E402
from src.research.brainiac_adapter import (  # noqa: E402
    adapt_volume,
    assess_r1a_readiness,
    evaluate_geometry,
    fit_frozen_probe,
    validate_canonical_preprocessing,
    validate_strict_load,
)
from src.research.brainiac_caft import (  # noqa: E402
    ABLATION_ARMS,
    CAFT_FEATURE_SCHEMA_SHA256,
    BrainIACCAFT,
    CaftConfig,
    conformal_coverage_report,
    confound_permutation_audit,
    evaluate_promotion,
    lora_config,
    prediction_artifact_digest,
    split_conformal_calibrate,
    validate_calibration_partition_contract,
    validate_equal_budget_contract,
    build_locked_confound_artifact,
)


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _tabular_contract() -> dict[str, object]:
    return {
        "feature_names": METHOD_X_PHENOTYPE_FEATURES,
        "feature_schema_sha256": CAFT_FEATURE_SCHEMA_SHA256,
        "tabular_preprocessing_receipt": {
            "locked": True,
            "fit_role": "head_fit",
            "imputation": "median",
            "scaling": "standard",
            "feature_schema_sha256": CAFT_FEATURE_SCHEMA_SHA256,
            "artifact_sha256": _sha("tabular-preprocessing"),
            "finite_output": True,
            "outer_test_touched": False,
        },
    }


def _canonical_manifest(**overrides):
    manifest = {
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
            "output_affine_sha256": _sha("output-affine"),
        },
        "direct_128_to_96_used": False,
    }
    manifest.update(overrides)
    return manifest


def test_canonical_preprocessing_is_affine_aware_and_direct_resize_is_sensitivity_only() -> None:
    ready = validate_canonical_preprocessing(_canonical_manifest())
    assert ready["status"] == "PASS"
    assert ready["affine_aware"] is True
    assert ready["primary_route"] == "raw_t1_n4_rigid_brain_extract_official_96"

    direct = _canonical_manifest(direct_128_to_96_used=True)
    blocked = validate_canonical_preprocessing(direct)
    assert blocked["status"] == "BLOCKED"
    assert "DIRECT_128_TO_96_PRIMARY_ROUTE_FORBIDDEN" in blocked["blockers"]

    with pytest.raises(RuntimeError, match="DIRECT_128_TO_96_SENSITIVITY_ONLY"):
        adapt_volume(np.zeros((8, 8, 8), dtype=np.float32), target_resolution=4)
    sensitivity = adapt_volume(
        np.zeros((8, 8, 8), dtype=np.float32),
        target_resolution=4,
        purpose="sensitivity_only",
    )
    assert sensitivity.shape == (4, 4, 4)


def test_missing_vendor_checkpoint_template_and_overlap_stay_unknown() -> None:
    status = assess_r1a_readiness(
        vendor_manifest={"vendored": False, "license_sha256": None},
        preprocessing_manifest=None,
        strict_load_report=None,
        forward_report=None,
        overlap_status="UNKNOWN_PRETRAINING_OVERLAP",
    )
    assert status["status"] == "UNKNOWN"
    assert status["ready"] is False
    assert set(status["blockers"]) >= {
        "OFFICIAL_VENDOR_SOURCE_ABSENT",
        "OFFICIAL_CHECKPOINT_ABSENT",
        "OFFICIAL_TEMPLATE_ABSENT",
        "UNKNOWN_PRETRAINING_OVERLAP",
    }


def test_invalid_geometry_is_unknown_and_never_passes() -> None:
    report = evaluate_geometry(source_resolution=0)
    assert report.status == "UNKNOWN_GEOMETRY"
    assert "SOURCE_RESOLUTION_INVALID" in report.findings
    assert report.passed is False


def test_strict_load_requires_both_shape_maps_when_shape_validation_is_requested() -> None:
    with pytest.raises(RuntimeError, match="STRICT_LOAD_SHAPE_EVIDENCE_INCOMPLETE"):
        validate_strict_load(["a"], ["a"], shapes={"a": (1,)})


def _probe_fixture():
    rng = np.random.default_rng(14)
    embeddings = rng.normal(size=(12, 4))
    labels = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    tabular = rng.normal(size=(12, 5))
    roles = ["inner_train"] * 8 + ["inner_validation"] * 4
    hashes = [_sha(f"subject-{i}") for i in range(12)]
    return embeddings, labels, tabular, roles, hashes


def _fit_probe(**overrides):
    embeddings, labels, tabular, roles, hashes = _probe_fixture()
    kwargs = {
        "embeddings": embeddings,
        "labels": labels,
        "inner_train": tuple(range(8)),
        "inner_validation": tuple(range(8, 12)),
        "tabular": tabular,
        "subject_hashes": hashes,
        "split_roles": roles,
        "real_flags": [True] * 12,
        "parent_hashes": [None] * 12,
        "feature_names": METHOD_X_PHENOTYPE_FEATURES,
        "split_provenance": {
            "selection_role": "inner_validation",
            "outer_test_touched": False,
            "split_hash": _sha("split"),
            "group_hash": _sha("group"),
            "embedding_artifact_sha256": _sha("embedding"),
        },
    }
    kwargs.update(overrides)
    registry_body = {
        "schema_version": 1,
        "locked": True,
        "split_hash": kwargs["split_provenance"]["split_hash"],
        "group_hash": kwargs["split_provenance"]["group_hash"],
        "subjects": [
            {
                "subject_hash": kwargs["subject_hashes"][index],
                "role": kwargs["split_roles"][index],
                "real": kwargs["real_flags"][index],
                "parent_hash": kwargs["parent_hashes"][index],
            }
            for index in range(len(kwargs["subject_hashes"]))
        ],
    }
    registry = {
        **registry_body,
        "integrity_sha256": hashlib.sha256(
            json.dumps(registry_body, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
    }
    kwargs["split_provenance"] = {
        **kwargs["split_provenance"],
        "split_registry_sha256": registry["integrity_sha256"],
    }
    kwargs["locked_split_registry"] = registry
    kwargs["expected_split_registry_sha256"] = registry["integrity_sha256"]
    return fit_frozen_probe(**kwargs)


def test_primary_probe_fits_imputer_and_scaler_on_real_head_fit() -> None:
    result = _fit_probe()
    assert result["selected_on"] == "inner_validation_only"
    assert result["outer_test_touched"] is False
    assert result["fit_real_only"] is True
    assert result["preprocessing_fit_role"] == "head_fit"
    assert result["feature_order"] == list(METHOD_X_PHENOTYPE_FEATURES)


def test_probe_rejects_subject_overlap_and_outer_selection() -> None:
    _, _, _, roles, hashes = _probe_fixture()
    hashes[8] = hashes[0]
    with pytest.raises(RuntimeError, match="PROBE_SUBJECT_OVERLAP"):
        _fit_probe(subject_hashes=hashes)

    roles[8] = "outer_test"
    with pytest.raises(RuntimeError, match="PROBE_OUTER_ROWS_PRESENT"):
        _fit_probe(split_roles=roles)


def test_probe_rejects_selected_subject_overlap_with_an_outer_row() -> None:
    embeddings, labels, tabular, roles, hashes = _probe_fixture()
    embeddings = np.vstack((embeddings, np.zeros((1, embeddings.shape[1]))))
    labels = np.append(labels, 0)
    tabular = np.vstack((tabular, np.zeros((1, tabular.shape[1]))))
    roles.append("outer_test")
    hashes.append(hashes[0])
    with pytest.raises(RuntimeError, match="PROBE_SUBJECT_OVERLAP"):
        _fit_probe(
            embeddings=embeddings,
            labels=labels,
            tabular=tabular,
            split_roles=roles,
            subject_hashes=hashes,
            real_flags=[True] * 13,
            parent_hashes=[None] * 13,
        )


def test_probe_rejects_nonreal_validation_and_bad_feature_order() -> None:
    real_flags = [True] * 12
    real_flags[9] = False
    with pytest.raises(RuntimeError, match="PROBE_VALIDATION_MUST_BE_REAL_ONLY"):
        _fit_probe(real_flags=real_flags)
    with pytest.raises(RuntimeError, match="FEATURE_ORDER_VIOLATION"):
        _fit_probe(feature_names=tuple(reversed(METHOD_X_PHENOTYPE_FEATURES)))


def test_caft_is_a_real_frozen_output_residual_module() -> None:
    config = CaftConfig(gate_bias_init=-4.0)
    model = BrainIACCAFT(config)
    assert isinstance(model, torch.nn.Module)
    assert model.local_projection.out_features == 128
    assert model.global_projection.out_features == 128
    assert model.tabular_projection.out_features == 32

    locked = torch.tensor([[0.2], [-0.3]], requires_grad=True)
    local = torch.randn(2, 128, requires_grad=True)
    global_embedding = torch.randn(2, 768, requires_grad=True)
    tabular = torch.randn(2, 5)
    output = model(locked, local, global_embedding, tabular, **_tabular_contract())

    assert torch.all(output["gate"] < 0.03)
    assert torch.all(output["bounded_residual"].abs() <= config.residual_bound)
    assert torch.allclose(output["final_logit"], locked.detach())
    assert output["regularization"].item() == pytest.approx(0.0)
    output["final_logit"].sum().backward()
    assert locked.grad is None and local.grad is None and global_embedding.grad is None


def test_caft_regularizes_the_actual_correction_not_gate_alone() -> None:
    model = BrainIACCAFT(CaftConfig(gate_bias_init=-4.0))
    with torch.no_grad():
        model.residual_head.weight.zero_()
        model.residual_head.bias.fill_(0.5)
    output = model(
        torch.zeros(3, 1),
        torch.zeros(3, 128),
        torch.zeros(3, 768),
        torch.zeros(3, 5),
        **_tabular_contract(),
    )
    expected = model.config.correction_l2_weight * output["correction"].square().mean()
    assert torch.allclose(output["regularization"], expected)


def test_caft_requires_a_negative_near_zero_gate_initialization() -> None:
    with pytest.raises(RuntimeError, match="CAFT_GATE_BIAS_NOT_NEAR_ZERO"):
        CaftConfig(gate_bias_init=0.0).validate()


def test_equal_budget_contract_requires_all_arms_and_matched_training_budget() -> None:
    common = {
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
    contract = {
        arm: {
            **common,
            "arm_id": arm,
            "model_class": f"Synthetic{index}",
            "model_artifact_sha256": _sha(f"model-{arm}"),
            "prediction_artifact_sha256": _sha(f"predictions-{arm}"),
            "step_history_sha256": _sha(f"steps-{arm}"),
            "trainable_parameter_policy": f"policy-{arm}",
        }
        for index, arm in enumerate(ABLATION_ARMS)
    }
    result = validate_equal_budget_contract(contract)
    assert result["status"] == "PASS"

    contract["naive_concatenation"]["max_steps"] = 201
    with pytest.raises(RuntimeError, match="BASELINE_BUDGET_MISMATCH"):
        validate_equal_budget_contract(contract)


def _split_registry(subject_hashes: list[str], roles: list[str]) -> dict:
    body = {
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


def _locked_predictions():
    n = 32
    hashes = [_sha(f"promotion-{i}") for i in range(n)]
    registry = _split_registry(
        [*hashes, _sha("outer-0"), _sha("outer-1")],
        ["oof_inner_training"] * n + ["outer_test"] * 2,
    )
    half = n // 2
    mediocre = ([0.2, 0.4, 0.6, 0.8] * (half // 4 + 1))[:half] + (
        [0.3, 0.5, 0.7, 0.9] * ((n - half) // 4 + 1)
    )[: n - half]
    artifact = {
        "locked": True,
        "split_role": "oof_inner_training",
        "outer_test_touched": False,
        "subject_hashes": hashes,
        "labels": [0] * half + [1] * (n - half),
        "baseline_probabilities": mediocre,
        "candidate_probabilities": [0.05] * half + [0.95] * (n - half),
        "split_hash": registry["split_hash"],
        "group_hash": registry["group_hash"],
        "split_registry_sha256": registry["integrity_sha256"],
        "subject_order_sha256": hashlib.sha256(
            json.dumps(hashes, separators=(",", ":")).encode()
        ).hexdigest(),
    }
    artifact["sha256"] = prediction_artifact_digest(artifact)
    return artifact, registry


def _locked_confound(artifact: dict, registry: dict, scores: list[float]) -> dict:
    n = len(artifact["subject_hashes"])
    return build_locked_confound_artifact(
        scores=scores,
        labels=[i % 2 for i in range(n)],
        subject_hashes=list(artifact["subject_hashes"]),
        prediction_artifact_sha256=artifact["sha256"],
        split_registry_sha256=registry["integrity_sha256"],
        iterations=1000,
        seed=13,
    )


def _promotion_kwargs(artifact: dict, registry: dict, **overrides) -> dict:
    n = len(artifact["subject_hashes"])
    kwargs = {
        "locked_predictions": artifact,
        "locked_split_registry": registry,
        "expected_split_registry_sha256": registry["integrity_sha256"],
        "baseline_confound": _locked_confound(
            artifact, registry, [0.1 + 0.8 * (i % 2) for i in range(n)]
        ),
        "candidate_confound": _locked_confound(
            artifact, registry, [0.4 + 0.2 * ((i // 2) % 2) for i in range(n)]
        ),
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
    kwargs.update(overrides)
    return kwargs


def test_promotion_requires_locked_inner_or_oof_predictions_and_never_unlocks_lora() -> None:
    artifact, registry = _locked_predictions()
    result = evaluate_promotion(**_promotion_kwargs(artifact, registry))
    assert result["promoted"] is True
    assert result["lora_unlocked"] is False
    assert result["metrics_source"] == "LOCKED_PREDICTION_ARTIFACT_RECOMPUTED"

    bad, bad_registry = _locked_predictions()
    bad["outer_test_touched"] = True
    bad["sha256"] = prediction_artifact_digest(bad)
    blocked = evaluate_promotion(**_promotion_kwargs(bad, bad_registry))
    assert blocked["promoted"] is False
    assert "OUTER_TEST_SELECTION_FORBIDDEN" in blocked["blockers"]


def test_lora_is_hard_disabled_in_r1_even_after_frozen_promotion() -> None:
    with pytest.raises(RuntimeError, match="LORA_HARD_DISABLED_R1C"):
        lora_config(
            frozen_stage_promoted=True,
            evidence_scope="inner_oof",
            run_version="r1c",
        )


def test_conformal_partition_contract_requires_real_disjoint_subsets() -> None:
    result = validate_calibration_partition_contract(
        probability_calibration_hashes=[_sha("p0"), _sha("p1")],
        conformal_calibration_hashes=[_sha("c0"), _sha("c1")],
        outer_training_hashes=[_sha("t0"), _sha("t1")],
        probability_real_flags=[True, True],
        conformal_real_flags=[True, True],
    )
    assert result["status"] == "PASS"
    with pytest.raises(RuntimeError, match="CALIBRATION_PARTITION_OVERLAP"):
        validate_calibration_partition_contract(
            probability_calibration_hashes=[_sha("same")],
            conformal_calibration_hashes=[_sha("same")],
            outer_training_hashes=[_sha("train")],
            probability_real_flags=[True],
            conformal_real_flags=[True],
        )


def test_mondrian_requires_thirty_real_samples_per_class() -> None:
    probs = [0.2] * 29 + [0.8] * 30
    labels = [0] * 29 + [1] * 30
    with pytest.raises(RuntimeError, match="MONDRIAN_MIN_CLASS_COUNT"):
        split_conformal_calibrate(probs, labels, method="mondrian")


def test_conformal_report_includes_ci_class_cohort_and_risk_coverage() -> None:
    report = conformal_coverage_report(
        [0.1, 0.2, 0.8, 0.9],
        [0, 0, 1, 1],
        threshold=0.25,
        target_coverage=0.9,
        cohorts=["A", "B", "A", "B"],
    )
    assert set(report["coverage_ci_95"]) == {"lower", "upper"}
    assert set(report["class_coverage"]) == {0, 1}
    assert set(report["cohort_coverage"]) == {"A", "B"}
    assert len(report["risk_coverage_curve"]) >= 3


def test_missing_real_disjoint_calibration_evidence_blocks_promotion() -> None:
    artifact, registry = _locked_predictions()
    result = evaluate_promotion(
        **_promotion_kwargs(
            artifact,
            registry,
            coverage={
                "coverage_met": True,
                "empirical_coverage": 0.92,
                "target_coverage": 0.90,
                "method": "split_conformal_marginal",
            },
        )
    )
    assert result["promoted"] is False
    assert "CALIBRATION_PARTITION_EVIDENCE_MISSING" in result["blockers"]


def test_confound_audit_reports_folded_auc_permutation_null_and_ci() -> None:
    labels = np.array([0] * 40 + [1] * 40)
    scores = np.array([0.9] * 40 + [0.1] * 40)
    result = confound_permutation_audit(scores, labels, iterations=1000, seed=5)
    assert result["raw_auc"] == pytest.approx(0.0)
    assert result["decodability"] == pytest.approx(1.0)
    assert result["permutation_iterations"] == 1000
    assert 0.0 <= result["null_ci_lower"] <= result["null_ci_upper"] <= 1.0
    assert 0.0 <= result["ci_lower"] <= result["ci_upper"] <= 1.0
