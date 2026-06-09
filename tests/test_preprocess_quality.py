from src.utils.preprocess_quality import classify_mri_numeric_qc, MRI_FAIL_RULES


def test_mri_classify_fail_low_brain_volume():
    qc = classify_mri_numeric_qc(
        {
            "sample_hash": "s_test",
            "brain_volume_ml": 500.0,
            "gm_wm_cnr": 0.5,
            "normalized_nonzero_fraction": 0.2,
            "normalized_brain_std": 1.0,
            "tissue_fraction_gm": 0.4,
            "normalized_brain_mean": 0.0,
        }
    )
    assert qc["qc_status"] == "FAIL"
    assert "brain_volume_fail_range" in qc["failures"]


def test_mri_classify_hold_warning_only():
    qc = classify_mri_numeric_qc(
        {
            "sample_hash": "s_test",
            "brain_volume_ml": 790.0,
            "gm_wm_cnr": 0.5,
            "normalized_nonzero_fraction": 0.2,
            "normalized_brain_std": 1.0,
            "tissue_fraction_gm": 0.4,
            "normalized_brain_mean": 0.0,
        }
    )
    assert qc["qc_status"] == "HOLD"
    assert "brain_volume_outside_expected_range" in qc["warnings"]


def test_mri_fail_rules_documented():
    assert MRI_FAIL_RULES["min_brain_volume_ml"] == 750.0
