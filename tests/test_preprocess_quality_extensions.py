from pathlib import Path

import pytest

from src.utils.preprocess_quality import (
    attach_pet_official_qc_fields,
    derived_centiloid_from_official_suvr,
    classify_mri_numeric_qc,
    pearson_correlation,
)
from src.utils.visual_qc import apply_visual_qc_signoff, build_visual_review_manifest


def test_derived_centiloid_from_fbp_suvr():
    # FBP: CL = 188.22 * SUVR - 189.16
    derived = derived_centiloid_from_official_suvr(1.20, tracer="fbp")
    assert derived is not None
    assert abs(derived - (188.22 * 1.20 - 189.16)) < 1e-3


def test_attach_pet_official_qc_fields():
    qc = {}
    out = attach_pet_official_qc_fields(
        qc,
        substudy="A4",
        bid="B1234",
        tracer="fbp",
        centiloid_lookup={("A4", "B1234"): 45.0},
        suvr_lookup={("A4", "B1234"): 1.20},
    )
    assert out["official_centiloid"] == 45.0
    assert out["official_suvr"] == 1.20
    assert "derived_centiloid" in out


def test_mri_icv_mismatch_hold():
    qc = classify_mri_numeric_qc(
        {
            "sample_hash": "s_test",
            "brain_volume_ml": 1200.0,
            "icv_ml_official": 500.0,
            "gm_wm_cnr": 0.5,
            "normalized_nonzero_fraction": 0.2,
            "normalized_brain_std": 1.0,
            "tissue_fraction_gm": 0.4,
            "normalized_brain_mean": 0.0,
        }
    )
    assert qc["qc_status"] == "HOLD"
    assert "icv_brain_volume_mismatch_review" in qc["warnings"]


def test_pearson_correlation_basic():
    assert pearson_correlation([1, 2, 3, 4], [2, 4, 6, 8]) == pytest.approx(1.0)


def test_visual_qc_manifest_and_signoff(tmp_path):
    qc_dir = tmp_path / "qc"
    qc_dir.mkdir()
    payload = {
        "sample_hash": "s_visual",
        "status": "PASS",
        "tracer": "fbp",
        "gate": {"status": "PASS", "failures": [], "warnings": []},
        "qc_after": {"ncc_pet_t1": 0.95},
        "visual_qc_required_for_training": True,
    }
    qc_path = qc_dir / "s_visual.json"
    qc_path.write_text(__import__("json").dumps(payload), encoding="utf-8")

    manifest = build_visual_review_manifest(qc_dir, project_root=tmp_path, pass_sample_fraction=1.0)
    assert manifest["total_review_items"] >= 1
    assert manifest["pending_signoff_count"] >= 1

    signed = apply_visual_qc_signoff(qc_path, decision="pass")
    assert signed["visual_qc_signoff"] is True

    manifest2 = build_visual_review_manifest(qc_dir, project_root=tmp_path, pass_sample_fraction=1.0)
    assert manifest2["pending_signoff_count"] == 0
