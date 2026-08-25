from __future__ import annotations

import json


def test_contract_diff_classifies_scientific_and_source_changes():
    from src.training.method_x_resume import build_run_contract, contract_diff

    base = build_run_contract(
        cache_sha256="a" * 64,
        split_digest="b" * 64,
        folds=5,
        epochs=10,
        seed=13,
        parameters={"backbone": "resnet18_3d", "lr": 1e-4, "source_contract_digest": "c" * 64},
        loader={"batch_size": 1},
    )
    changed = build_run_contract(
        cache_sha256="a" * 64,
        split_digest="b" * 64,
        folds=5,
        epochs=10,
        seed=13,
        parameters={"backbone": "resnet18_3d", "lr": 2e-4, "source_contract_digest": "d" * 64},
        loader={"batch_size": 1},
    )
    diff = contract_diff({"run_contract": base}, changed)
    assert "parameters.lr" in diff["scientific_changes"]
    assert "parameters.source_contract_digest" in diff["source_changes"]
    assert diff["source_only_migration_allowed"] is False


def test_method_x_training_count_audit_has_fold_local_augmented_rows():
    from scripts.audit_method_x_training_counts import audit

    report = audit(
        cache=__import__("pathlib").Path("data/cached/t1_volume_cache_128.h5"),
        splits=__import__("pathlib").Path("splits"),
        folds=5,
        seed=13,
        val_fraction=0.15,
    )
    assert report["status"] == "PASS"
    assert report["observed_source_counts"]["total"] == 6213
    assert report["totals"]["real_fit_exposures"] == 4544
    assert report["totals"]["augmented_fit_exposures"] == 4544
    assert report["totals"]["fit_exposures"] == 9088


def test_amuse_x_hybrid3d_forward_and_stage_manifest():
    torch = __import__("pytest").importorskip("torch")
    from src.models.backbones.backbone_factory import build_3d_backbone
    from src.training.method_x_stage_maps import stage_map_manifest

    encoder = build_3d_backbone("amuse_x_hybrid3d", out_dim=16)
    output = encoder(torch.zeros(1, 1, 32, 32, 32))
    assert tuple(output.shape) == (1, 16)
    manifest = stage_map_manifest(type("Wrapper", (), {"encoder": encoder, "backbone_name": "amuse_x_hybrid3d"})())
    assert "residual" in manifest["stage_names"]


def test_claim_validator_blocks_partial_final_aggregate():
    from src.utils.final_claim_policy import validate_claim

    report = validate_claim(
        {
            "method": "X",
            "phase": "final",
            "resolution": 128,
            "folds": 5,
            "completed_folds": 2,
            "training_status": "IN_PROGRESS",
            "spm_complete": True,
            "uses_pet_as_input": False,
            "uses_raw_identifiers": False,
            "uses_raw_paths": False,
            "framing_language": "MRI-only amyloid biomarker / preclinical risk estimate",
            "claim_scope": "amyloid biomarker / preclinical risk only",
        },
        strict=False,
    )
    assert report["ok"] is False
    assert "TRAINING_NOT_COMPLETED" in report["blocking_reasons"]
    assert "COMPLETED_FOLD_COUNT_INVALID" in report["blocking_reasons"]


def test_legacy_xai_uses_canonical_method_x_feature_order():
    from scripts.run_xai_summary import TABULAR_CACHE_FEATURES
    from src.data.combined_mri_cache import METHOD_X_PHENOTYPE_FEATURES

    assert TABULAR_CACHE_FEATURES == list(METHOD_X_PHENOTYPE_FEATURES)


def test_paired_bootstrap_metric_delta_is_aggregate_only():
    from src.evaluation.matched_benchmark import paired_bootstrap_metric_delta

    result = paired_bootstrap_metric_delta(
        [0, 1, 0, 1, 0, 1],
        [0.1, 0.8, 0.2, 0.7, 0.3, 0.6],
        [0.2, 0.9, 0.1, 0.8, 0.4, 0.7],
        lambda y, p: sum((float(label) - value) ** 2 for label, value in zip(y, p)) / len(y),
        resamples=100,
        seed=13,
    )
    assert result["status"] == "COMPLETED"
    assert result["subject_level_vectors_exported"] is False
    assert len(result["ci95"]) == 2
