"""Task 7 — CAFT fusion, conformal abstention, promotion gate, XAI contract."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.research.brainiac_caft import (  # noqa: E402
    ABLATION_ARMS,
    EVIDENCE_LABEL,
    INDETERMINATE,
    LORA_CONFIG,
    MIN_BOOTSTRAP_ITERATIONS,
    CaftConfig,
    brier_score,
    conformal_coverage_report,
    conformal_prediction_set,
    decays_to_baseline,
    evaluate_promotion,
    expected_calibration_error,
    gate_penalty,
    linear_cka,
    lora_config,
    paired_bootstrap_auc_delta,
    prediction_artifact_digest,
    residual_fusion_logit,
    residual_predictive_value,
    split_conformal_calibrate,
    validate_xai_plan,
    build_locked_confound_artifact,
)


def _good_metrics(**overrides) -> dict[str, float]:
    base = {"roc_auc": 0.72, "brier": 0.21, "ece": 0.15, "cohort_probe_auc": 0.75}
    base.update(overrides)
    return base


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


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


def _registry_for(artifact: dict) -> dict:
    hashes = list(artifact["subject_hashes"])
    outer = [_sha("outer-0"), _sha("outer-1")]
    return _split_registry(
        [*hashes, *outer],
        ["oof_inner_training"] * len(hashes) + ["outer_test"] * 2,
    )


def _overlapping_baseline(n: int) -> list[float]:
    half = n // 2
    return ([0.2, 0.4, 0.6, 0.8] * (half // 4 + 1))[:half] + (
        [0.3, 0.5, 0.7, 0.9] * ((n - half) // 4 + 1)
    )[: n - half]


def _strong_candidate(n: int) -> list[float]:
    half = n // 2
    return [0.05] * half + [0.95] * (n - half)


def _locked_predictions(*, candidate=None, baseline=None):
    if candidate is not None:
        n = len(candidate)
    elif baseline is not None:
        n = len(baseline)
    else:
        n = 32
    hashes = [
        hashlib.sha256(f"caft-subject-{i}".encode()).hexdigest() for i in range(n)
    ]
    registry = _registry_for({"subject_hashes": hashes})
    half = n // 2
    mediocre = _overlapping_baseline(n)
    strong = _strong_candidate(n)
    artifact = {
        "locked": True,
        "split_role": "oof_inner_training",
        "outer_test_touched": False,
        "subject_hashes": hashes,
        "labels": [0] * half + [1] * (n - half),
        "baseline_probabilities": baseline or mediocre,
        "candidate_probabilities": candidate or strong,
        "split_hash": registry["split_hash"],
        "group_hash": registry["group_hash"],
        "split_registry_sha256": registry["integrity_sha256"],
        "subject_order_sha256": hashlib.sha256(
            json.dumps(hashes, separators=(",", ":")).encode()
        ).hexdigest(),
    }
    artifact["sha256"] = prediction_artifact_digest(artifact)
    return artifact


def _cohort_labels(n: int) -> list[int]:
    return [i % 2 for i in range(n)]


def _strong_cohort_scores(n: int) -> list[float]:
    return [0.1 + 0.8 * (i % 2) for i in range(n)]


def _weak_cohort_scores(n: int) -> list[float]:
    return [0.4 + 0.2 * ((i // 2) % 2) for i in range(n)]


def _locked_confound(artifact: dict, registry: dict, scores: list[float]) -> dict:
    n = len(artifact["subject_hashes"])
    return build_locked_confound_artifact(
        scores=scores,
        labels=_cohort_labels(n),
        subject_hashes=list(artifact["subject_hashes"]),
        prediction_artifact_sha256=artifact["sha256"],
        split_registry_sha256=registry["integrity_sha256"],
        iterations=1000,
        seed=13,
    )


def _promotable_kwargs(**overrides):
    artifact = overrides.pop("locked_predictions", None) or _locked_predictions()
    registry = overrides.pop("locked_split_registry", None) or _registry_for(artifact)
    expected_sha = (
        overrides.pop("expected_split_registry_sha256", None)
        or registry["integrity_sha256"]
    )
    n = len(artifact["subject_hashes"])
    kwargs = {
        "locked_predictions": artifact,
        "locked_split_registry": registry,
        "expected_split_registry_sha256": expected_sha,
        "baseline_confound": _locked_confound(
            artifact, registry, _strong_cohort_scores(n)
        ),
        "candidate_confound": _locked_confound(
            artifact, registry, _weak_cohort_scores(n)
        ),
        "bootstrap": {
            "prediction_artifact_sha256": artifact["sha256"],
            "iterations": MIN_BOOTSTRAP_ITERATIONS,
            "seed": 13,
            "lower_ci_positive": True,
            "ci_lower": 0.01,
            "ci_upper": 0.40,
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


# --- Fusion -------------------------------------------------------------------


def test_closed_gate_reproduces_the_baseline_exactly() -> None:
    locked = np.array([-1.2, 0.4, 2.0])
    residual = np.array([5.0, -3.0, 9.0])
    assert decays_to_baseline(locked, residual)
    assert np.array_equal(
        residual_fusion_logit(locked, np.zeros(3), residual), locked
    )


def test_open_gate_applies_the_full_residual() -> None:
    locked = np.array([0.0, 1.0])
    residual = np.array([2.0, -0.5])
    fused = residual_fusion_logit(locked, np.ones(2), residual)
    assert np.allclose(fused, [2.0, 0.5])


def test_gate_outside_the_unit_interval_is_rejected() -> None:
    with pytest.raises(RuntimeError, match="GATE_OUT_OF_RANGE"):
        residual_fusion_logit(np.zeros(2), np.array([1.5, 0.0]), np.ones(2))
    with pytest.raises(RuntimeError, match="GATE_OUT_OF_RANGE"):
        residual_fusion_logit(np.zeros(2), np.array([-0.1, 0.0]), np.ones(2))


def test_gate_penalty_rewards_silence() -> None:
    residual = np.full(4, 2.0)
    assert gate_penalty(np.zeros(4), residual, weight=0.01) == 0.0
    assert gate_penalty(np.ones(4), residual, weight=0.01) > gate_penalty(
        np.full(4, 0.5), residual, weight=0.01
    )


def test_caft_config_requires_frozen_encoders() -> None:
    CaftConfig().validate()
    with pytest.raises(RuntimeError, match="CAFT_STAGE_REQUIRES_FROZEN_ENCODERS"):
        CaftConfig(freeze_global_encoder=False).validate()


def test_caft_config_rejects_audit_fields_and_extra_features() -> None:
    with pytest.raises(RuntimeError, match="AUDIT_FIELD_IN_PREDICTORS"):
        CaftConfig(audit_fields_used=("scanner_id",)).validate()
    with pytest.raises(RuntimeError, match="ONLY_FIVE_ALLOWED_FEATURES"):
        CaftConfig(tabular_features=6).validate()


def test_caft_config_requires_a_regularized_correction() -> None:
    with pytest.raises(RuntimeError, match="CORRECTION_MUST_BE_REGULARIZED"):
        CaftConfig(correction_l2_weight=0.0).validate()
    with pytest.raises(RuntimeError, match="GATE_ONLY_REGULARIZATION_FORBIDDEN"):
        CaftConfig(gate_l2_weight=0.01).validate()


# --- Complementarity ----------------------------------------------------------


def test_cka_is_one_for_identical_representations() -> None:
    rng = np.random.default_rng(13)
    x = rng.normal(size=(40, 6))
    assert linear_cka(x, x) == pytest.approx(1.0, abs=1e-9)


def test_cka_is_low_for_unrelated_representations() -> None:
    rng = np.random.default_rng(29)
    assert linear_cka(rng.normal(size=(200, 5)), rng.normal(size=(200, 5))) < 0.35


def test_residual_predictive_value_detects_added_information() -> None:
    labels = [0] * 20 + [1] * 20
    weak = [0.5] * 40
    strong = [0.2] * 20 + [0.8] * 20
    result = residual_predictive_value(weak, strong, labels)
    assert result["fused_auc"] > result["baseline_auc"]
    assert result["adds_information"] is True


def test_residual_predictive_value_reports_no_gain_honestly() -> None:
    labels = [0] * 20 + [1] * 20
    same = [0.2] * 20 + [0.8] * 20
    result = residual_predictive_value(same, same, labels)
    assert result["delta_auc"] == pytest.approx(0.0)
    assert result["adds_information"] is False


# --- Conformal ----------------------------------------------------------------


def test_conformal_threshold_targets_the_requested_coverage() -> None:
    rng = np.random.default_rng(7)
    labels = rng.integers(0, 2, size=200)
    probs = np.where(labels == 1, rng.uniform(0.5, 1.0, 200), rng.uniform(0.0, 0.5, 200))
    fitted = split_conformal_calibrate(probs, labels, alpha=0.1)
    assert fitted["target_coverage"] == pytest.approx(0.9)
    assert 0.0 <= fitted["threshold"] <= 1.0
    assert fitted["calibration_n"] == 200


def test_two_label_set_is_an_explicit_indeterminate_output() -> None:
    result = conformal_prediction_set(0.5, threshold=0.9)
    assert result["labels"] == ["AB_NEGATIVE", "AB_POSITIVE"]
    assert result["decision"] == INDETERMINATE
    assert result["abstained"] is True
    assert result["clinical_triage_claim"] is False


def test_confident_predictions_yield_a_single_label() -> None:
    positive = conformal_prediction_set(0.99, threshold=0.2)
    negative = conformal_prediction_set(0.01, threshold=0.2)
    assert positive["decision"] == "AB_POSITIVE" and positive["abstained"] is False
    assert negative["decision"] == "AB_NEGATIVE" and negative["abstained"] is False


def test_empty_set_is_treated_as_uncertainty_not_confidence() -> None:
    result = conformal_prediction_set(0.5, threshold=0.01)
    assert result["labels"] == []
    assert result["decision"] == INDETERMINATE
    assert result["abstained"] is True


def test_coverage_report_measures_coverage_and_abstention() -> None:
    """Conformal coverage is a marginal guarantee, so allow sampling noise.

    At n=150 the standard error on a 0.9 target is about 2.4 points, so
    demanding an exact >= 0.9 on a single split would be a flaky test rather
    than a real check. The band below is roughly three standard errors.
    """

    rng = np.random.default_rng(47)
    labels = rng.integers(0, 2, size=300)
    probs = np.where(
        labels == 1, rng.uniform(0.55, 1.0, 300), rng.uniform(0.0, 0.45, 300)
    )
    fitted = split_conformal_calibrate(probs[:150], labels[:150], alpha=0.1)
    report = conformal_coverage_report(
        probs[150:],
        labels[150:],
        threshold=fitted["threshold"],
        target_coverage=fitted["target_coverage"],
    )
    assert report["empirical_coverage"] == pytest.approx(0.9, abs=0.08)
    assert 0.0 <= report["abstention_rate"] <= 1.0
    assert report["n"] == 150


def test_coverage_holds_on_average_across_many_splits() -> None:
    """The guarantee is about the average, so average over splits to check it."""

    coverages: list[float] = []
    for seed in range(20):
        rng = np.random.default_rng(seed)
        labels = rng.integers(0, 2, size=600)
        probs = np.where(
            labels == 1, rng.uniform(0.55, 1.0, 600), rng.uniform(0.0, 0.45, 600)
        )
        fitted = split_conformal_calibrate(probs[:300], labels[:300], alpha=0.1)
        coverages.append(
            conformal_coverage_report(
                probs[300:],
                labels[300:],
                threshold=fitted["threshold"],
                target_coverage=fitted["target_coverage"],
            )["empirical_coverage"]
        )
    assert float(np.mean(coverages)) >= 0.88


def test_an_empty_set_is_reported_as_uncovered_not_silently_correct() -> None:
    """Empty sets are a real split-conformal outcome and must not be hidden."""

    # threshold 0.3 leaves a gap for p in (0.3, 0.7).
    report = conformal_coverage_report([0.5], [1], threshold=0.3, target_coverage=0.9)
    assert report["empirical_coverage"] == 0.0
    assert report["abstention_rate"] == 1.0
    assert report["indeterminate_count"] == 1


def test_conformal_rejects_bad_inputs() -> None:
    with pytest.raises(RuntimeError, match="INVALID_ALPHA"):
        split_conformal_calibrate([0.5], [1], alpha=0.0)
    with pytest.raises(RuntimeError, match="CONFORMAL_CALIBRATION_EMPTY"):
        split_conformal_calibrate([], [])


# --- Calibration and bootstrap ------------------------------------------------


def test_perfect_calibration_has_near_zero_ece() -> None:
    probs = [0.0] * 50 + [1.0] * 50
    labels = [0] * 50 + [1] * 50
    assert expected_calibration_error(probs, labels) == pytest.approx(0.0, abs=1e-9)
    assert brier_score(probs, labels) == pytest.approx(0.0)


def test_paired_bootstrap_requires_a_thousand_iterations() -> None:
    with pytest.raises(RuntimeError, match="INSUFFICIENT_BOOTSTRAP_ITERATIONS"):
        paired_bootstrap_auc_delta([0.4], [0.6], [1], iterations=999)


def test_paired_bootstrap_detects_a_real_improvement() -> None:
    rng = np.random.default_rng(13)
    labels = np.array([0] * 60 + [1] * 60)
    baseline = rng.uniform(0.3, 0.7, 120)
    candidate = np.where(labels == 1, rng.uniform(0.6, 1.0, 120), rng.uniform(0.0, 0.4, 120))
    result = paired_bootstrap_auc_delta(baseline, candidate, labels, iterations=1000)
    assert result["delta_auc_point"] > 0.0
    assert result["lower_ci_positive"] is True
    assert result["paired"] is True


def test_paired_bootstrap_reports_no_gain_when_there_is_none() -> None:
    rng = np.random.default_rng(29)
    labels = np.array([0] * 60 + [1] * 60)
    scores = rng.uniform(0.0, 1.0, 120)
    result = paired_bootstrap_auc_delta(scores, scores, labels, iterations=1000)
    assert result["delta_auc_point"] == pytest.approx(0.0)
    assert result["lower_ci_positive"] is False


# --- Promotion gate -----------------------------------------------------------


def test_promotion_requires_every_criterion() -> None:
    result = evaluate_promotion(**_promotable_kwargs())
    assert result["promoted"] is True
    assert result["decision"] == "PROMOTE"
    assert result["lora_unlocked"] is False
    assert result["evidence_label"] == EVIDENCE_LABEL


def test_insufficient_auc_gain_blocks_promotion() -> None:
    result = evaluate_promotion(
        **_promotable_kwargs(
            locked_predictions=_locked_predictions(candidate=_overlapping_baseline(32))
        )
    )
    assert result["promoted"] is False
    assert "AUC_GAIN_BELOW_THRESHOLD" in result["blockers"]
    assert result["decision"] == "NO_IMPROVEMENT_DEMONSTRATED"


@pytest.mark.parametrize(
    "overrides,expected",
    [
        (
            {
                "locked_predictions": _locked_predictions(
                    candidate=[0.1, 0.2, 0.8, 0.4, 0.3, 0.6, 0.7, 0.9]
                )
            },
            "PAIRED_CI_LOWER_NOT_POSITIVE",
        ),
        (
            {
                "locked_predictions": _locked_predictions(
                    candidate=[0.85] * 16 + [0.90] * 16
                )
            },
            "BRIER_DEGRADED",
        ),
        (
            {
                "locked_predictions": _locked_predictions(
                    candidate=[0.49] * 16 + [0.51] * 16
                )
            },
            "ECE_DEGRADED",
        ),
        ({"coverage": {"coverage_met": False, "empirical_coverage": 0.80, "target_coverage": 0.90, "method": "split_conformal_marginal", "calibration_partitions_validated": True, "real_only": True, "class_coverage_met": True, "cohort_coverage_status": "MEASURABLE_PASS"}}, "CONFORMAL_COVERAGE_NOT_MET"),
        ({"sanity_passed": False}, "SANITY_FAILED"),
    ],
)
def test_each_criterion_can_block_promotion_alone(overrides, expected) -> None:
    result = evaluate_promotion(**_promotable_kwargs(**overrides))
    assert result["promoted"] is False
    assert expected in result["blockers"]


def test_a_worse_cohort_probe_blocks_promotion() -> None:
    artifact = _locked_predictions()
    registry = _registry_for(artifact)
    n = len(artifact["subject_hashes"])
    result = evaluate_promotion(
        **_promotable_kwargs(
            locked_predictions=artifact,
            locked_split_registry=registry,
            baseline_confound=_locked_confound(
                artifact, registry, _weak_cohort_scores(n)
            ),
            candidate_confound=_locked_confound(
                artifact, registry, _strong_cohort_scores(n)
            ),
        )
    )
    assert result["promoted"] is False
    assert "COHORT_PROBE_WORSENED" in result["blockers"]


def test_a_gain_cannot_buy_off_a_calibration_regression() -> None:
    """A large AUC win must not launder a Brier degradation."""

    result = evaluate_promotion(
        **_promotable_kwargs(
            locked_predictions=_locked_predictions(
                candidate=[0.85] * 16 + [0.90] * 16
            )
        )
    )
    assert result["promoted"] is False
    assert "BRIER_DEGRADED" in result["blockers"]


# --- LoRA gating --------------------------------------------------------------


def test_lora_is_hard_disabled_in_r1c() -> None:
    with pytest.raises(RuntimeError, match="LORA_HARD_DISABLED_R1C"):
        lora_config(
            frozen_stage_promoted=False,
            evidence_scope="inner_oof",
            run_version="r1c",
        )


def test_lora_config_matches_the_prespecified_recipe() -> None:
    protocol_body = {
        "protocol_id": "method_x_brainiac_r2_lora",
        "run_version": "r2_lora",
        "status": "REGISTERED_LOCKED",
        "selection_evidence_scope": "inner_oof_only",
        "outer_test_touched": False,
    }
    protocol = {
        **protocol_body,
        "integrity_sha256": hashlib.sha256(
            json.dumps(protocol_body, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
    }
    receipt_body = {
        "status": "PASS",
        "promoted": True,
        "evidence_scope": "inner_oof",
        "outer_test_touched": False,
        "protocol_sha256": protocol["integrity_sha256"],
        "prediction_artifact_sha256": hashlib.sha256(b"predictions").hexdigest(),
    }
    receipt = {
        **receipt_body,
        "integrity_sha256": hashlib.sha256(
            json.dumps(receipt_body, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
    }
    config = lora_config(
        frozen_stage_promoted=True,
        evidence_scope="inner_oof",
        run_version="r2_lora",
        protocol_manifest=protocol,
        promotion_receipt=receipt,
    )
    assert config["rank"] == 4
    assert config["alpha"] == 8
    assert config["dropout"] == 0.05
    assert config["learning_rate"] == 5e-5
    assert config["layerwise_lr_decay"] == 0.8
    assert config["target_projections"] == ("q", "v")
    assert config["full_fine_tune_permitted"] is False


def test_full_fine_tune_is_forbidden_in_the_first_version() -> None:
    assert LORA_CONFIG["full_fine_tune_permitted"] is False


# --- XAI contract -------------------------------------------------------------


def _xai_plan(**overrides):
    plan = {
        "methods": ["integrated_gradients", "occlusion", "region_additive_sensitivity"],
        "sanity": ["randomization", "occlusion_fidelity"],
        "aggregate_only": True,
        "raw_identifiers_stored": False,
    }
    plan.update(overrides)
    return plan


def test_valid_xai_plan_is_accepted() -> None:
    result = validate_xai_plan(_xai_plan())
    assert "integrated_gradients" in result["methods"]
    assert result["aggregate_only"] is True
    assert result["permitted_phrase"].startswith("Regions with higher attribution")


def test_gradcam_on_a_vit_is_rejected() -> None:
    with pytest.raises(RuntimeError, match="FORBIDDEN_XAI_METHOD"):
        validate_xai_plan(_xai_plan(methods=["grad_cam"]))


def test_missing_sanity_checks_are_rejected() -> None:
    with pytest.raises(RuntimeError, match="MISSING_XAI_SANITY"):
        validate_xai_plan(_xai_plan(sanity=["randomization"]))


def test_subject_level_xai_is_rejected() -> None:
    with pytest.raises(RuntimeError, match="XAI_MUST_BE_AGGREGATE_ONLY"):
        validate_xai_plan(_xai_plan(aggregate_only=False))
    with pytest.raises(RuntimeError, match="XAI_RAW_IDENTIFIERS_FORBIDDEN"):
        validate_xai_plan(_xai_plan(raw_identifiers_stored=True))


# --- Ablation arms ------------------------------------------------------------


def test_all_four_ablation_arms_are_declared() -> None:
    assert set(ABLATION_ARMS) == {
        "locked_resnet34_plus_tabular",
        "brainiac_only",
        "naive_concatenation",
        "caft_residual_gated",
    }
