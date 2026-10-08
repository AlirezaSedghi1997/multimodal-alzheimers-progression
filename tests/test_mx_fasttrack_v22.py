"""Fail-closed tests for the post-lock XAI and thesis addendum."""
from __future__ import annotations

import json

import numpy as np
import pytest

from scripts.mx_fasttrack_v2_2.xai_crossfit import (
    choose_size_matched_controls,
    crossfit_plan,
    paired_target_effect,
    randomization_check,
    run_fold_crossfit,
)
from scripts.mx_fasttrack_v2_2.thesis_report import _metric, build_sections, validate_freeze
from scripts.mx_fasttrack_v2_2.independent_gate import assess_candidate
from scripts.mx_fasttrack_v2_2.xai_runner import _validated_receipts


def test_crossfit_never_assesses_on_its_roi_selection_half():
    plan = crossfit_plan(8)
    assert plan == [((0, 1, 2, 3), (4, 5, 6, 7)),
                    ((4, 5, 6, 7), (0, 1, 2, 3))]
    with pytest.raises(ValueError, match="EIGHT"):
        crossfit_plan(7)


def test_control_regions_match_volume_and_are_not_top():
    atlas = np.asarray([1] * 10 + [2] * 9 + [3] * 11 + [4] * 10 + [5] * 40)
    assert choose_size_matched_controls(atlas, 1) == [4, 3, 2]
    with pytest.raises(ValueError, match="MATCHED"):
        choose_size_matched_controls(np.asarray([1] * 10 + [2] * 100), 1)


def test_paired_target_uses_frozen_predicted_class_for_each_sample():
    # Negative-class sample: a fall in p(positive) means a rise in p(target).
    effect = paired_target_effect(0.2, 0.4, [0.25, 0.3, 0.35])
    assert effect["target_class"] == 0
    assert effect["top_target_drop"] == pytest.approx(0.2)
    assert effect["top_minus_control_drop"] == pytest.approx(0.1)
    effect = paired_target_effect(0.8, 0.6, [0.75, 0.7, 0.65])
    assert effect["target_class"] == 1
    assert effect["top_minus_control_drop"] == pytest.approx(0.1)
    with pytest.raises(ValueError, match="PROBABILITY"):
        paired_target_effect(float("nan"), 0.4, [0.3] * 3)


def test_randomization_constant_map_fails_closed():
    assert randomization_check(np.ones((2, 2)), np.arange(4).reshape(2, 2))["status"] == "FAIL"
    assert randomization_check(np.arange(4).reshape(2, 2), np.ones((2, 2)))["status"] == "FAIL"


def test_cpu_toy_crossfit_runs_with_eight_assessments_and_no_sample_exports():
    import torch

    class Toy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weights = torch.nn.Parameter(torch.arange(1, 11, dtype=torch.float32)
                                              .reshape(1, 1, 2, 5, 1))

        def forward(self, image, tabular, tab_presence=None):
            return (image * self.weights).flatten(1).sum(dim=1, keepdim=True) / 100

    atlas = np.tile(np.arange(1, 6, dtype=np.int32), (2, 1))[:, :, None]
    batches = [(torch.ones(1, 1, 2, 5, 1), torch.zeros(1, 5), torch.ones(1))
               for _ in range(8)]
    result = run_fold_crossfit(model=Toy(), batches=batches,
                               atlas_data=atlas, region_names={i: str(i) for i in range(1, 6)},
                               randomization_seeds=[13, 29, 47])
    assert result["sample_count"] == 8
    assert [row["assessment_n"] for row in result["crossfit_directions"]] == [4, 4]
    assert result["subject_level_outputs_written"] is False
    assert len(result["randomization_sanity"]) == 3
    frozen = run_fold_crossfit(model=Toy(), batches=batches,
                               atlas_data=atlas, region_names={i: str(i) for i in range(1, 6)},
                               randomization_seeds=[13, 29, 47],
                               frozen_original_probabilities=[0.1] * 8)
    assert frozen["frozen_target_class_counts"] == {"negative": 8, "positive": 0}


def test_thesis_requires_hash_bound_parent_and_all_five_xai_receipts():
    frozen = {"scope": "POST_HOC_INTERNAL_TECHNICAL_EVIDENCE", "immutable": True,
              "status": {"MX07_fasttrack": "READY_SCOPED_FASTTRACK"},
              "external_validation": False, "cohort_label_perfectly_confounding": True,
              "paper_numbers": {"selected_inner_task": "hybrid_baseline",
                                "hybrid_baseline_overall": {"roc_auc": 0.6},
                                "hybrid_dense_overall": {"roc_auc": 0.61},
                                "hybrid_paired_difference": {"roc_auc": {"dense_minus_baseline": 0.01}},
                                "mx04_scratch_overall": {"roc_auc": 0.55},
                                "mx04_head_overall": {"roc_auc": 0.56}}}
    validate_freeze(frozen)
    with pytest.raises(ValueError, match="FROZEN"):
        validate_freeze({**frozen, "immutable": False})
    xai = {"status": "COMPLETED_EXPLORATORY_CROSSFIT", "n_images": 40,
           "folds": [{"fold": i, "sample_count": 8, "paired_mean_effect": 0.1} for i in range(5)],
           "not_independent_testing": True, "anatomical_or_causal_region_claim_allowed": False}
    sections = build_sections(frozen, xai, external_status="NOT_DONE_NO_INDEPENDENT_COHORT")
    assert "0.61" in sections["results"]
    assert "NOT_DONE_NO_INDEPENDENT_COHORT" in sections["results"]
    assert "بیماری" in sections["discussion"]
    expanded_numbers = {**frozen["paper_numbers"],
                        "inner_mean_best_validation_loss": {"hybrid_baseline": 0.45},
                        "temporal_gap_counts": {str(day): {"n": 20, "positive": 10,
                                                          "negative": 10}
                                                for day in (90, 180, 365)}}
    expanded = build_sections(
        {**frozen, "paper_numbers": expanded_numbers}, xai,
        external_status="NOT_DONE_NO_INDEPENDENT_COHORT",
        pair_by_fold=[{"fold": i, "n": 5,
                       "baseline": {"roc_auc": 0.6, "average_precision": 0.6, "brier": 0.2},
                       "dense": {"roc_auc": 0.61, "average_precision": 0.61, "brier": 0.19}}
                      for i in range(5)])
    assert "| 4 | 5 |" in expanded["results"]
    assert "365 روز" in expanded["results"]
    with pytest.raises(ValueError, match="FIVE_FOLDS"):
        build_sections(frozen, {**xai, "folds": xai["folds"][:4]},
                       external_status="NOT_DONE_NO_INDEPENDENT_COHORT")


def test_external_preflight_never_declares_validation_or_disease_region_confirmed():
    rows = [
        {"patient_hash": f"outside_{i}", "label": i % 2, "cohort": "independent_A",
         "site": "external_site", "scanner": "scanner_1", "mri_sha256": "e" * 64,
         "reference_blinded": True,
         "regional_reference_sha256": "a" * 64}
        for i in range(4)
    ]
    evidence = assess_candidate(rows, set(), endpoint_protocol_sha256="b" * 64,
                                linkage_attestation_sha256="c" * 64,
                                locked_model_manifest_sha256="d" * 64)
    assert evidence["status"] == "STRUCTURAL_PRECHECK_PASSED_EVALUATION_NOT_DONE"
    assert evidence["external_validation_completed"] is False
    assert evidence["disease_region_confirmation_completed"] is False
    blocked = assess_candidate(rows, {"outside_0"}, endpoint_protocol_sha256="b" * 64,
                               linkage_attestation_sha256="c" * 64,
                               locked_model_manifest_sha256="d" * 64)
    assert "INTERNAL_EXTERNAL_PATIENT_OVERLAP" in blocked["blockers"]
    assert assess_candidate([], set(), endpoint_protocol_sha256=None,
                            linkage_attestation_sha256=None,
                            locked_model_manifest_sha256=None)["status"] == "BLOCKED_PRECHECK"


def test_receipt_rejects_falsified_scientific_gate(tmp_path):
    reports = tmp_path / "reports"
    folder = reports / "xai_v22_folds"
    folder.mkdir(parents=True)
    (reports / "xai_v22_context.json").write_text("{}", encoding="utf-8")
    from scripts.mx_fasttrack_v2.run import sha
    context = {"selected_task": "hybrid_baseline", "checkpoint_sha256": ["a" * 64] * 5,
               "fold_membership_digest": ["b" * 64] * 5}
    direction = {"top_region_id": 1, "top_volume": 10,
                 "controls": [{"atlas_id": i, "volume": 10} for i in (2, 3, 4)],
                 "selection_n": 4, "assessment_n": 4,
                 "assessment_mean_top_target_drop": 0.2,
                 "assessment_mean_paired_effect": 0.1}
    row = {"schema_version": 1, "fold": 0,
           "context_sha256": sha(reports / "xai_v22_context.json"),
           "selected_task": "hybrid_baseline", "checkpoint_sha256": "a" * 64,
           "split_membership_digest": "b" * 64, "sample_count": 8,
           "subject_level_outputs_written": False, "aggregate_only": True,
           "model_selection_overlap": True,
           "target_definition": "FROZEN_UNOCCLUDED_PREDICTED_CLASS_BEFORE_ROI_SELECTION",
           "frozen_target_class_counts": {"negative": 4, "positive": 4},
           "crossfit_directions": [direction, direction],
           "randomization_sanity": [{"seed": seed, "status": "PASS",
                                     "absolute_correlation": 0.1} for seed in (13, 29, 47)],
           "paired_mean_effect": 0.1, "paired_mean_top_target_drop": 0.2,
           "paired_positive_count": 8,
           "peak_reserved_gb": 1.0, "gate_status": "PASS_EXPLORATORY"}
    path = folder / "fold_0.json"
    path.write_text(json.dumps(row), encoding="utf-8")
    with pytest.raises(RuntimeError, match="TARGET_RECEIPT"):
        _validated_receipts(tmp_path, context)
    target_folder = tmp_path / "_private_resume/xai_v22_targets"
    target_folder.mkdir(parents=True)
    target_path = target_folder / "fold_0.json"
    target_path.write_text(json.dumps({"schema_version": 1, "fold": 0,
                                       "context_sha256": sha(reports / "xai_v22_context.json"),
                                       "checkpoint_sha256": "a" * 64,
                                       "split_membership_digest": "b" * 64,
                                       "target_rule": "unoccluded_predicted_class",
                                       "samples": [{"ordinal": i, "original_probability": 0.8,
                                                    "target_class": 1} for i in range(8)]}),
                           encoding="utf-8")
    row["target_receipt_sha256"] = sha(target_path)
    row["frozen_target_class_counts"] = {"negative": 0, "positive": 8}
    path.write_text(json.dumps(row), encoding="utf-8")
    assert _validated_receipts(tmp_path, context)[0][0]["gate_status"] == "PASS_EXPLORATORY"
    path.write_text(json.dumps({**row, "gate_status": "BLOCKED_EXPLORATORY_GATE"}), encoding="utf-8")
    with pytest.raises(RuntimeError, match="GATE_INCONSISTENT"):
        _validated_receipts(tmp_path, context)
    negative_drop = {**direction, "assessment_mean_top_target_drop": -0.1}
    path.write_text(json.dumps({**row, "paired_mean_top_target_drop": -0.1,
                                "crossfit_directions": [negative_drop, negative_drop]}),
                    encoding="utf-8")
    with pytest.raises(RuntimeError, match="GATE_INCONSISTENT"):
        _validated_receipts(tmp_path, context)


def test_protocol_cannot_be_registered_after_mx5_artifact(tmp_path, monkeypatch):
    from scripts.mx_fasttrack_v2_2 import supervise_v22
    monkeypatch.setattr(supervise_v22, "ensure_amendment", lambda root, out: {})
    out = tmp_path / "out"
    (out / "reports").mkdir(parents=True)
    (out / "reports/smoke_baseline.json").write_text("{}", encoding="utf-8")
    with pytest.raises(RuntimeError, match="TOO_LATE"):
        supervise_v22.ensure_protocol(tmp_path, out)
    assert not (out / "reports/fasttrack_v22_protocol.json").exists()


def test_one_class_fold_auc_is_reported_as_not_measurable():
    assert _metric({"status": "NOT_MEASURABLE_ONE_CLASS", "roc_auc": None},
                   "roc_auc") == "غیرقابل‌محاسبه"
