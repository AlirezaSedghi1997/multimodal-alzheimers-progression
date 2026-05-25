from scripts.train_baseline import FEATURES as TABULAR_BASELINE_FEATURES
from scripts.train_mri_feature_baseline import TABULAR_FEATURES as MRI_BASELINE_TABULAR_FEATURES


def test_baselines_do_not_use_pet_timing_gap():
    forbidden_terms = ("pet", "tau", "suvr", "gap")
    all_features = [*TABULAR_BASELINE_FEATURES, *MRI_BASELINE_TABULAR_FEATURES]
    bad = [f for f in all_features if any(term in f.lower() for term in forbidden_terms)]
    assert not bad

