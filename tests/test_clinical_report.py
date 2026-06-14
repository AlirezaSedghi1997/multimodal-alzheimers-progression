import pytest


def test_generate_clinical_report_basic():
    from scripts.generate_clinical_report import generate_clinical_report

    report = generate_clinical_report(0.8, 0.1)
    assert report["prediction"]["amyloid_status"] == "Positive"
    assert report["prediction"]["probability"] == 0.8
    assert report["risk_assessment"] in ["HIGH", "MODERATE", "LOW", "VERY_LOW", "BORDERLINE", "UNCERTAIN"]
    assert len(report["recommendation"]) > 0
    assert len(report["caveats"]) > 0


def test_generate_clinical_report_negative():
    from scripts.generate_clinical_report import generate_clinical_report

    report = generate_clinical_report(0.2, 0.1)
    assert report["prediction"]["amyloid_status"] == "Negative"
    assert report["prediction"]["probability"] == 0.2


def test_generate_clinical_report_high_uncertainty():
    from scripts.generate_clinical_report import generate_clinical_report

    report = generate_clinical_report(0.5, 0.4)
    assert report["uncertainty"]["level"] == "HIGH"
    assert report["risk_assessment"] == "UNCERTAIN"


def test_format_report_text():
    from scripts.generate_clinical_report import format_report_text, generate_clinical_report

    report = generate_clinical_report(0.7, 0.1)
    text = format_report_text(report)
    assert "AMYLOID BIOMARKER PREDICTION REPORT" in text
    assert "Positive" in text
    assert "CAVEATS" in text


def test_caveats_by_modality():
    from scripts.generate_clinical_report import generate_clinical_report

    mri_report = generate_clinical_report(0.7, 0.1, modality="mri")
    assert any("PET-free" in c for c in mri_report["caveats"])
    
    pet_report = generate_clinical_report(0.7, 0.1, modality="pet")
    assert any("PET-assisted" in c for c in pet_report["caveats"])
