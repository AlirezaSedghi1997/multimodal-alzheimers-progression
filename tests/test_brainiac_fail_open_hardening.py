"""Fail-open defects found by the Codex supervisory audit (2026-08-10).

Each test here corresponds to a way the BrainIAC/CAFT contracts could report
success on evidence that does not support it. They are grouped separately from
the feature tests because the property under test is *refusal*, not behaviour.

None of these paths is reachable today — R1 is hard-disabled and no BrainIAC
weights are vendored. They are locked down now so the gate cannot silently open
later.
"""

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

from src.research.brainiac_adapter import (  # noqa: E402
    evaluate_geometry,
    mask_edge_perturbation,
    validate_strict_load,
)
from src.research.brainiac_caft import (  # noqa: E402
    conformal_coverage_report,
    confound_decodability,
    evaluate_promotion,
    prediction_artifact_digest,
    split_conformal_calibrate,
    build_locked_confound_artifact,
)


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


def _artifact(*, baseline=None, candidate=None):
    if candidate is not None:
        n = len(candidate)
    elif baseline is not None:
        n = len(baseline)
    else:
        n = 32
    hashes = [
        hashlib.sha256(f"hardening-subject-{i}".encode()).hexdigest() for i in range(n)
    ]
    registry = _registry_for({"subject_hashes": hashes})
    half = n // 2
    mediocre = ([0.2, 0.4, 0.6, 0.8] * (half // 4 + 1))[:half] + (
        [0.3, 0.5, 0.7, 0.9] * ((n - half) // 4 + 1)
    )[: n - half]
    strong = [0.05] * half + [0.95] * (n - half)
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


def _promotable(**overrides):
    artifact = overrides.pop("locked_predictions", None) or _artifact()
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
            "iterations": 1000,
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


# --- Promotion gate: NaN and missing evidence ---------------------------------


@pytest.mark.parametrize("field", ["baseline_probabilities", "candidate_probabilities"])
def test_nan_metric_cannot_reach_promote(field: str) -> None:
    """`NaN < MIN_AUC_GAIN` is False, so every threshold would silently pass.

    Left unguarded, a degenerate run reaches PROMOTE and sets
    ``lora_unlocked=True``.
    """

    artifact = _artifact()
    artifact[field][0] = float("nan")
    artifact["sha256"] = prediction_artifact_digest(artifact)
    result = evaluate_promotion(**_promotable(locked_predictions=artifact))
    assert result["promoted"] is False
    assert "PROMOTION_METRIC_NOT_FINITE:PREDICTIONS" in result["blockers"]


@pytest.mark.parametrize("value", [float("inf"), float("-inf")])
def test_infinite_metric_cannot_reach_promote(value: float) -> None:
    artifact = _artifact()
    artifact["candidate_probabilities"][0] = value
    artifact["sha256"] = prediction_artifact_digest(artifact)
    result = evaluate_promotion(**_promotable(locked_predictions=artifact))
    assert "PROMOTION_METRIC_NOT_FINITE:PREDICTIONS" in result["blockers"]


def test_missing_metric_cannot_reach_promote() -> None:
    artifact = _artifact()
    artifact["candidate_probabilities"].pop()
    artifact["sha256"] = prediction_artifact_digest(artifact)
    result = evaluate_promotion(**_promotable(locked_predictions=artifact))
    assert "PREDICTION_ARTIFACT_MISALIGNED" in result["blockers"]


def test_out_of_range_metric_cannot_reach_promote() -> None:
    artifact = _artifact()
    artifact["candidate_probabilities"][0] = 1.4
    artifact["sha256"] = prediction_artifact_digest(artifact)
    result = evaluate_promotion(**_promotable(locked_predictions=artifact))
    assert "PROMOTION_METRIC_OUT_OF_RANGE:PREDICTIONS" in result["blockers"]


def test_absent_bootstrap_evidence_blocks_rather_than_defaulting_to_false() -> None:
    """Missing evidence must be named, not collapsed into a generic failure."""

    result = evaluate_promotion(**_promotable(bootstrap={"iterations": 1000}))
    assert result["promoted"] is False
    assert "PAIRED_BOOTSTRAP_EVIDENCE_MISSING" in result["blockers"]


def test_absent_coverage_evidence_blocks() -> None:
    result = evaluate_promotion(**_promotable(coverage={}))
    assert result["promoted"] is False
    assert "CONFORMAL_COVERAGE_EVIDENCE_MISSING" in result["blockers"]


def test_absent_sanity_evidence_blocks() -> None:
    result = evaluate_promotion(**_promotable(sanity_passed=None))
    assert result["promoted"] is False
    assert "SANITY_EVIDENCE_MISSING" in result["blockers"]


def test_a_clean_promotion_still_succeeds() -> None:
    """The hardening must not block a genuinely qualifying candidate."""

    result = evaluate_promotion(**_promotable())
    assert result["promoted"] is True
    assert result["lora_unlocked"] is False


# --- Confound decodability ----------------------------------------------------


@pytest.mark.parametrize(
    "auc,expected",
    [(0.10, 0.90), (0.90, 0.90), (0.50, 0.50), (0.25, 0.75), (1.0, 1.0)],
)
def test_decodability_folds_inverted_probes(auc: float, expected: float) -> None:
    """AUC 0.10 means 90% decodable after sign inversion, not 'better'."""

    assert confound_decodability(auc) == pytest.approx(expected)


def test_an_inverted_probe_is_not_mistaken_for_an_improvement() -> None:
    """Raw comparison would read 0.10 < 0.75 as the confound getting better."""

    artifact = _artifact()
    registry = _registry_for(artifact)
    n = len(artifact["subject_hashes"])
    result = evaluate_promotion(
        **_promotable(
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


def test_invalid_probe_auc_is_rejected() -> None:
    for bad in (float("nan"), -0.1, 1.2):
        with pytest.raises(RuntimeError, match="INVALID_PROBE_AUC"):
            confound_decodability(bad)


# --- Conformal honesty --------------------------------------------------------


def test_conformal_reports_itself_as_marginal_not_class_conditional() -> None:
    """Scores are pooled across classes, so the guarantee is marginal."""

    fitted = split_conformal_calibrate([0.2, 0.8, 0.3, 0.7], [0, 1, 0, 1])
    assert fitted["method"] == "split_conformal_marginal"
    assert fitted["coverage_guarantee"] == "MARGINAL_NOT_CLASS_CONDITIONAL"
    assert fitted["class_counts"] == {0: 2, 1: 2}


def test_calibration_rejects_invalid_probabilities_and_labels() -> None:
    with pytest.raises(RuntimeError, match="PROBABILITY_INVALID"):
        split_conformal_calibrate([0.2, 1.4], [0, 1])
    with pytest.raises(RuntimeError, match="PROBABILITY_INVALID"):
        split_conformal_calibrate([0.2, float("nan")], [0, 1])
    with pytest.raises(RuntimeError, match="LABEL_INVALID"):
        split_conformal_calibrate([0.2, 0.8], [0, 2])


def test_coverage_report_refuses_mismatched_lengths() -> None:
    """`zip` would truncate and report coverage over a silent subset."""

    with pytest.raises(RuntimeError, match="CONFORMAL_EVALUATION_MISALIGNED"):
        conformal_coverage_report(
            [0.2, 0.8, 0.5], [0, 1], threshold=0.4, target_coverage=0.9
        )


def test_coverage_report_rejects_invalid_inputs() -> None:
    with pytest.raises(RuntimeError, match="PROBABILITY_INVALID"):
        conformal_coverage_report([1.5], [1], threshold=0.4, target_coverage=0.9)
    with pytest.raises(RuntimeError, match="LABEL_INVALID"):
        conformal_coverage_report([0.5], [7], threshold=0.4, target_coverage=0.9)
    with pytest.raises(RuntimeError, match="CONFORMAL_EVALUATION_EMPTY"):
        conformal_coverage_report([], [], threshold=0.4, target_coverage=0.9)


# --- Geometry gate ------------------------------------------------------------


def test_geometry_gate_does_not_pass_with_unresolved_findings() -> None:
    """A documented domain shift is not a pass; it is a recorded problem."""

    report = evaluate_geometry()
    assert report.findings
    assert report.status == "UNKNOWN_GEOMETRY_PENDING_VENDOR_TEMPLATE_AFFINE"
    assert report.passed is False


def test_geometry_gate_passes_only_with_zero_findings() -> None:
    # 96 voxels at 2 mm spans 192 mm, which comfortably holds a whole brain, so
    # no FOV ambiguity is raised and every other axis reconciles.
    report = evaluate_geometry(
        source_resolution=96,
        source_voxel_mm=2.0,
        target_resolution=96,
        declared_target_voxel_mm=2.0,
        source_registration="rigid_mni",
        target_registration="rigid_mni",
        target_geometry_verified=True,
    )
    assert report.findings == ()
    assert report.passed is True


# --- Strict load --------------------------------------------------------------


def test_strict_load_rejects_two_empty_state_dicts() -> None:
    """Empty sets trivially satisfy 'no missing, no unexpected' at 100%."""

    with pytest.raises(RuntimeError, match="STRICT_LOAD_EMPTY_STATE_DICT"):
        validate_strict_load([], [])


def test_strict_load_rejects_an_empty_checkpoint_against_a_real_model() -> None:
    with pytest.raises(RuntimeError, match="STRICT_LOAD_EMPTY_STATE_DICT"):
        validate_strict_load([], ["encoder.weight"])


def test_strict_load_still_accepts_a_genuine_full_match() -> None:
    report = validate_strict_load(["a.weight", "b.bias"], ["a.weight", "b.bias"])
    assert report["coverage"] == 1.0


# --- Mask perturbation wrap-around --------------------------------------------


def test_dilation_does_not_wrap_anatomy_to_the_opposite_face() -> None:
    """`np.roll` would recreate foreground at the far boundary."""

    volume = np.zeros((5, 5, 5), dtype=np.float32)
    volume[0, 2, 2] = 1.0  # a voxel on the inferior face
    grown = mask_edge_perturbation(volume, mode="dilate", voxels=1)
    assert grown[4, 2, 2] == 0.0, "foreground wrapped to the opposite face"
    assert grown[1, 2, 2] != 0.0, "legitimate neighbour was not filled"


def test_erosion_does_not_wrap_across_the_volume_boundary() -> None:
    volume = np.ones((4, 4, 4), dtype=np.float32)
    eroded = mask_edge_perturbation(volume, mode="erode", voxels=1)
    # Every face voxel loses a neighbour, so the shell must be removed.
    assert eroded[0, 0, 0] == 0.0
    assert eroded[1, 1, 1] != 0.0


def test_foreground_only_mode_is_unchanged_by_the_fix() -> None:
    volume = np.zeros((3, 3, 3), dtype=np.float32)
    volume[1, 1, 1] = 2.0
    result = mask_edge_perturbation(volume, mode="foreground_only")
    assert result[1, 1, 1] == pytest.approx(2.0)
    assert result.sum() == pytest.approx(2.0)
