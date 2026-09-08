from __future__ import annotations

import json
from pathlib import Path


def _manifest(root: Path) -> Path:
    output = root / "battery"
    output.mkdir(parents=True, exist_ok=True)
    (output / "battery_manifest.json").write_text(
        json.dumps(
            {
                "locked_protocol": {"resolution": 128, "folds": 5, "seed": 13},
                "method_x_pet_free": True,
                "final_claim_allowed": False,
                "locked_inputs": {"cache_sha256": "a", "split_sha256": "b", "config_sha256": "c", "device_policy_sha256": "d"},
                "cache_characterization": {"total_rows": 6213, "eligible_real_rows": 1669, "materialized_augmented_rows": 4544, "folds": 5, "seed": 13, "resolution": 128},
                "spm_gate_snapshot": {"verified": True, "ready": True},
                "science_audit": {"diagnostic_claim_allowed": False},
                "models": [],
            }
        ),
        encoding="utf-8",
    )
    return output


def test_validity_suite_blocks_without_finalized_private_oof(monkeypatch, tmp_path: Path):
    import src.evaluation.method_x_validity as validity

    output = _manifest(tmp_path)
    monkeypatch.setattr(validity, "evaluate_autonomous_supervision", lambda **kwargs: {"decision": "CONTINUE"})
    monkeypatch.setattr(validity, "enforce_autonomous_supervision", lambda **kwargs: None)

    report = validity.run_method_x_validity_suite(root=tmp_path, output_root=output, resamples=10)

    assert report["status"] == "BLOCKED"
    assert "PAIRED_OOF_ARTIFACT_MISSING_OR_INVALID" in report["blockers"]
    assert (output / "reports/BLOCKED_MX03_VALIDITY.json").is_file()


def test_validity_suite_computes_paired_bootstrap_without_exporting_ids(monkeypatch, tmp_path: Path):
    import src.evaluation.method_x_validity as validity

    output = _manifest(tmp_path)
    private = output / "private"
    private.mkdir()
    y = [0, 1] * 10
    # Locked protocol vocabulary: exactly the effective five; rows carry the
    # fold metadata the suite requires. resamples>=1000 and task ids are
    # suite gates, so the fixture must satisfy them for COMPLETED.
    payload = {
        "status": "FINALIZED",
        "subject_level_vectors_exported": False,
        "membership_digest": "a" * 64,
        "y_true": y,
        "models": {
            name: {
                "probabilities": [0.1, 0.9] * 10,
                "fold_count": 5,
                "prediction_count": len(y),
            }
            for name in (
                "resnet10_3d",
                "resnet18_3d",
                "resnet34_3d",
                "densenet121_3d",
                "mednext_3d_s",
            )
        },
    }
    (private / "oof_predictions.json").write_text(json.dumps(payload), encoding="utf-8")
    for name in ("sanity_random_labels", "sanity_random_noise"):
        task_id = "random_labels" if name == "sanity_random_labels" else "random_noise"
        (private / f"{name}.json").write_text(
            json.dumps({"status": "COMPLETED", "task_id": task_id, "roc_auc": 0.5, "near_chance_gate": True}),
            encoding="utf-8",
        )
    monkeypatch.setattr(validity, "evaluate_autonomous_supervision", lambda **kwargs: {"decision": "CONTINUE"})
    monkeypatch.setattr(validity, "enforce_autonomous_supervision", lambda **kwargs: None)

    report = validity.run_method_x_validity_suite(root=tmp_path, output_root=output, resamples=1000)

    assert report["status"] == "COMPLETED"
    pair = report["pairwise_bootstrap"]["densenet121_3d__vs__resnet34_3d"]
    assert pair["roc_auc"]["status"] == "COMPLETED"
    assert pair["roc_auc"]["subject_level_vectors_exported"] is False


def test_validity_suite_blocks_completed_sanity_artifact_that_fails_near_chance(monkeypatch, tmp_path: Path):
    import src.evaluation.method_x_validity as validity

    output = _manifest(tmp_path)
    private = output / "private"
    private.mkdir()
    (private / "oof_predictions.json").write_text(
        json.dumps(
            {
                "status": "FINALIZED",
                "subject_level_vectors_exported": False,
                "membership_digest": "a" * 64,
                "y_true": [0, 1] * 5,
                "models": {
                    # All five effective members with valid metadata: the
                    # suite must pass the allowlist to reach the sanity
                    # gate under test (near-chance), which is what this
                    # test blocks on.
                    "resnet34_3d": {
                        "probabilities": [0.2, 0.8] * 5,
                        "fold_count": 5,
                        "prediction_count": 10,
                    },
                    "resnet10_3d": {
                        "probabilities": [0.3, 0.7] * 5,
                        "fold_count": 5,
                        "prediction_count": 10,
                    },
                    "resnet18_3d": {
                        "probabilities": [0.25, 0.75] * 5,
                        "fold_count": 5,
                        "prediction_count": 10,
                    },
                    "densenet121_3d": {
                        "probabilities": [0.2, 0.8] * 5,
                        "fold_count": 5,
                        "prediction_count": 10,
                    },
                    "mednext_3d_s": {
                        "probabilities": [0.3, 0.7] * 5,
                        "fold_count": 5,
                        "prediction_count": 10,
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    (private / "sanity_random_labels.json").write_text(
        json.dumps({"status": "COMPLETED", "task_id": "random_labels", "roc_auc": 0.9, "near_chance_gate": False}),
        encoding="utf-8",
    )
    (private / "sanity_random_noise.json").write_text(
        json.dumps({"status": "COMPLETED", "task_id": "random_noise", "roc_auc": 0.5, "near_chance_gate": True}),
        encoding="utf-8",
    )
    monkeypatch.setattr(validity, "evaluate_autonomous_supervision", lambda **kwargs: {"decision": "CONTINUE"})
    monkeypatch.setattr(validity, "enforce_autonomous_supervision", lambda **kwargs: None)

    report = validity.run_method_x_validity_suite(root=tmp_path, output_root=output, resamples=10)

    assert report["status"] == "BLOCKED"
    assert "SANITY_GATE_FAILED:random_labels" in report["blockers"]


def _paired_summary(delta: float, low: float, high: float, *, resamples: int = 1000) -> dict:
    return {
        "status": "COMPLETED",
        "delta_definition": "metric_b_minus_metric_a",
        "delta_mean": delta,
        "ci95": [low, high],
        "resamples_requested": resamples,
        "resamples_completed": resamples,
        "subject_level_vectors_exported": False,
    }


def test_promotion_rule_orients_candidate_minus_reference_and_can_promote():
    from src.evaluation.method_x_validity_tasks import build_promotion_decision

    pairwise = {
        # Alphabetical order makes ResNet34 the right-hand model.  The stored
        # delta is therefore reference-candidate and must be inverted.
        # strict_protocol=False scopes this unit to the orientation math;
        # allowlist enforcement is covered by the suite-level tests above.
        "resnet10_3d__vs__resnet34_3d": {
            "roc_auc": _paired_summary(-0.03, -0.05, -0.01),
            "pr_auc": _paired_summary(-0.02, -0.04, 0.00),
            "brier": _paired_summary(0.005, -0.002, 0.012),
        }
    }
    report = build_promotion_decision(
        pairwise=pairwise,
        model_names=["resnet34_3d", "resnet10_3d"],
        calibration_ece={"resnet34_3d": 0.10, "resnet10_3d": 0.11},
        cohort_probe_auc={"resnet34_3d": 0.70, "resnet10_3d": 0.30},
        sanity_passed=True,
        strict_protocol=False,
    )

    assert report["status"] == "COMPLETED"
    assert report["reporting_backbone"] == "resnet10_3d"
    assert report["decisions"][0]["paired_delta_candidate_minus_reference"]["roc_auc"]["delta"] == 0.03


def test_promotion_rule_rejects_underpowered_bootstrap_and_worse_absolute_cohort_probe():
    from src.evaluation.method_x_validity_tasks import build_promotion_decision

    pairwise = {
        "densenet121_3d__vs__resnet34_3d": {
            "roc_auc": _paired_summary(0.03, 0.01, 0.05, resamples=999),
            "pr_auc": _paired_summary(0.02, 0.00, 0.04, resamples=999),
            "brier": _paired_summary(0.0, -0.01, 0.01, resamples=999),
        }
    }
    report = build_promotion_decision(
        pairwise=pairwise,
        model_names=["resnet34_3d", "densenet121_3d"],
        calibration_ece={"resnet34_3d": 0.10, "densenet121_3d": 0.10},
        # AUC 0.40 is farther from chance than 0.55 even though it is lower.
        cohort_probe_auc={"resnet34_3d": 0.55, "densenet121_3d": 0.40},
        sanity_passed=True,
        strict_protocol=False,
    )

    decision = report["decisions"][0]
    assert decision["status"] == "NOT_PROMOTED"
    assert decision["criteria"]["paired_bootstrap_min_1000_completed"] is False
    assert decision["criteria"]["cohort_probe_not_worse"] is False
