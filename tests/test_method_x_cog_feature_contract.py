from scripts.build_cognitive_features import COGNITIVE_FEATURES, FORBIDDEN_INPUT_TERMS
from scripts.run_cognitive_ablation import FIXED_TABULAR_FEATURES, MODEL_VARIANTS, variant_feature_names


def test_method_x_cog_features_exclude_pet_derived_terms():
    bad = [
        feature
        for feature in COGNITIVE_FEATURES
        if any(term in feature.lower() for term in FORBIDDEN_INPUT_TERMS)
    ]
    assert not bad


def test_method_x_cog_variants_keep_method_x_fixed_tabular_contract():
    assert FIXED_TABULAR_FEATURES == ["age", "education_years", "apoe4_carrier", "mmse_base", "sex_binary"]
    assert "fixed_tabular_only" in MODEL_VARIANTS
    assert "fixed_tabular_plus_cognitive_core" in MODEL_VARIANTS
    assert "mri_feature_surrogate_plus_cognitive" in MODEL_VARIANTS


def test_method_x_cog_model_inputs_exclude_forbidden_terms():
    for variant in MODEL_VARIANTS:
        bad = [
            feature
            for feature in variant_feature_names(variant)
            if any(term in feature.lower() for term in FORBIDDEN_INPUT_TERMS)
        ]
        assert not bad, (variant, bad)
