from src.utils.cache_qc_filters import mri_cache_allowed, pet_cache_allowed


def test_mri_cache_blocks_fail(tmp_path):
    qc_dir = tmp_path / "qc"
    qc_dir.mkdir()
    (qc_dir / "s_fail.json").write_text(
        '{"sample_hash":"s_fail","qc_status":"FAIL","brain_volume_ml":500,"gm_wm_cnr":0.5,'
        '"normalized_nonzero_fraction":0.2,"normalized_brain_std":1.0,"tissue_fraction_gm":0.4,'
        '"normalized_brain_mean":0.0}',
        encoding="utf-8",
    )
    allowed, reason = mri_cache_allowed(sample_hash="s_fail", qc_dir=qc_dir, require_qc_pass=True)
    assert not allowed
    assert reason == "MRI_QC_FAIL"


def test_pet_cache_requires_visual_signoff():
    qc = {"status": "PASS", "gate": {"status": "PASS"}, "visual_qc_required_for_training": True, "visual_qc_signoff": False}
    allowed, reason = pet_cache_allowed(qc, require_visual_signoff=True)
    assert not allowed
    assert reason == "VISUAL_QC_UNSIGNED"
