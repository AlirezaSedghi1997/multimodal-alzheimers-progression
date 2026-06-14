import json
from pathlib import Path


def test_train_only_monai_augmentation_policy_keeps_real_n_separate():
    from src.data.volume_augmentation import build_augmentation_policy

    policy = build_augmentation_policy(enabled=True, real_train_n=1500, exposures_per_real=4)
    assert policy["library"] == "MONAI"
    assert policy["train_transform_enabled"] is True
    assert policy["validation_transform_enabled"] is False
    assert policy["test_transform_enabled"] is False
    assert policy["real_train_n"] == 1500
    assert policy["augmented_training_exposures"] == 6000
    assert policy["synthetic_or_augmented_counted_as_real_subjects"] is False


def test_train_only_monai_transform_disabled_outside_train():
    from src.data.volume_augmentation import build_monai_volume_transform

    assert build_monai_volume_transform(split="train", enabled=True) is not None
    assert build_monai_volume_transform(split="validation", enabled=True) is None
    assert build_monai_volume_transform(split="test", enabled=True) is None


def test_threshold_metrics_are_selected_without_test_labels():
    from src.evaluation.threshold_metrics import evaluate_with_train_selected_threshold

    result = evaluate_with_train_selected_threshold(
        validation_y=[0, 0, 1, 1],
        validation_prob=[0.1, 0.4, 0.6, 0.9],
        test_y=[0, 1, 1, 0],
        test_prob=[0.2, 0.7, 0.8, 0.3],
    )
    assert result["threshold_source"] == "validation"
    assert result["selected_threshold"] in result["candidate_thresholds"]
    for key in ["accuracy", "sensitivity", "specificity", "f1"]:
        assert key in result["test_metrics"]
        assert 0.0 <= result["test_metrics"][key] <= 1.0


def test_method_y_mri_pet_cognitive_contract_exists():
    from scripts.train_mri_pet_cognitive_method_y import SCENARIO, build_cognitive_feature_matrix

    assert SCENARIO["name"] == "mri_pet_cognitive"
    assert SCENARIO["method"] == "Y"
    assert SCENARIO["uses_pet_as_input"] is True
    assert SCENARIO["method_x_compliant"] is False
    assert SCENARIO["clinical_alzheimer_diagnosis_allowed"] is False
    assert callable(build_cognitive_feature_matrix)


def test_synthetic_mri_pet_policy_is_fold_local_and_supplementary():
    from scripts.build_synthetic_mri_pet_supplement import build_synthetic_policy

    policy = build_synthetic_policy(folds=5, synthetic_per_real=1)
    assert policy["branch"] == "supplementary_mri_to_pet_synthetic_augmentation"
    assert policy["method_x_input_allowed"] is False
    assert policy["fit_generator_scope"] == "train_fold_only"
    assert policy["add_synthetic_to"] == "same_train_fold_only"
    assert policy["validation_real_only"] is True
    assert policy["test_real_only"] is True
    assert policy["holdout_real_only"] is True
    assert policy["synthetic_counted_as_real_subjects"] is False


def test_method_y_incremental_table_uses_single_metric_index():
    from scripts.build_method_y_incremental_value_table import METHOD_Y_METRIC_SOURCES

    assert "Y5_mri_pet_cognitive" in METHOD_Y_METRIC_SOURCES
    source_paths = {spec["path"] for spec in METHOD_Y_METRIC_SOURCES.values()}
    assert len(source_paths) == len(METHOD_Y_METRIC_SOURCES)


def test_completed_method_y_rows_have_threshold_metrics_if_generated():
    path = Path("results/tables/method_y_incremental_value_table.json")
    if not path.exists():
        return
    report = json.loads(path.read_text(encoding="utf-8"))
    for row in report["rows"]:
        if row["metrics_status"] != "AVAILABLE":
            continue
        metrics = row.get("threshold_metrics") or {}
        missing = [key for key in ["accuracy", "sensitivity", "specificity", "f1"] if metrics.get(key) is None]
        assert not missing, f"{row['comparison_id']} missing threshold metrics: {missing}"


def test_method_x_aggregate_forbids_pet_and_synthetic_pet_inputs():
    from types import SimpleNamespace

    from scripts.train_3d_mri import build_aggregate

    args = SimpleNamespace(
        phase="dev",
        resolution=64,
        fusion="concat",
        backbone="tiny",
        input_mode="mri_plus_tabular",
        latent_alignment_lambda=0.0,
        synib_lambda=0.0,
        modality_dropout=0.0,
        batch_size=1,
        val_fraction=0.15,
        early_stopping_patience=1,
        cache="data/cached/t1_volume_cache_64.h5",
        device="cpu",
        num_workers=0,
        monai_augment=False,
        augmentation_exposures_per_real=4,
    )
    agg = build_aggregate(
        mode="CPU_DEBUG_ONLY",
        args=args,
        device=SimpleNamespace(type="cpu"),
        epochs=1,
        folds=1,
        use_synib=False,
        use_amp=False,
        cache_path=Path(args.cache),
        cache_resolution=64,
        fold_metrics=[],
        all_y=[],
        all_p=[],
        checkpoints=[],
    )
    assert agg["uses_pet_as_input"] is False
    assert agg["uses_synthetic_pet_as_input"] is False
    assert agg["final_contract"]["pet_input_allowed"] is False
    assert agg["final_contract"]["synthetic_pet_input_allowed"] is False


def test_proposal_alignment_report_if_generated_maps_claims():
    path = Path("results/tables/proposal_alignment_report.json")
    if not path.exists():
        return
    report = json.loads(path.read_text(encoding="utf-8"))
    statuses = {row["status"] for row in report["claims"]}
    assert {"DONE", "BETTER_THAN_PROPOSAL", "ADDED_CONTROLLED_EXTENSION", "SCIENTIFIC_ETHICAL_LIMIT"}.issubset(statuses)
    assert report["claim_boundary"]["clinical_alzheimer_diagnosis_requires_independent_label"] is True
    assert report["privacy_contract"]["public_subject_level_predictions"] is False
