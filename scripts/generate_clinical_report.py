#!/usr/bin/env python3
"""Generate human-readable clinical reports from model predictions."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def generate_clinical_report(
    prediction: float,
    uncertainty: float,
    features: dict | None = None,
    *,
    modality: str = "mri",
) -> dict:
    risk_level = _classify_risk(prediction, uncertainty)
    recommendation = _get_recommendation(risk_level, prediction, uncertainty)
    key_findings = _extract_key_findings(features) if features else []
    
    report = {
        "prediction": {
            "amyloid_status": "Positive" if prediction > 0.5 else "Negative",
            "probability": float(prediction),
            "confidence": float(max(prediction, 1 - prediction)),
        },
        "uncertainty": {
            "level": _classify_uncertainty(uncertainty),
            "score": float(uncertainty),
            "interpretation": _interpret_uncertainty(uncertainty),
        },
        "risk_assessment": risk_level,
        "recommendation": recommendation,
        "key_findings": key_findings,
        "caveats": _get_caveats(modality),
        "technical_details": {
            "model": "Method X-Strict" if modality == "mri" else "Method Y",
            "modality": modality,
            "version": "1.0",
        },
    }
    return report


def _classify_risk(prediction: float, uncertainty: float) -> str:
    if uncertainty > 0.3:
        return "UNCERTAIN"
    if prediction > 0.8:
        return "HIGH"
    if prediction > 0.6:
        return "MODERATE"
    if prediction > 0.4:
        return "BORDERLINE"
    if prediction > 0.2:
        return "LOW"
    return "VERY_LOW"


def _classify_uncertainty(uncertainty: float) -> str:
    if uncertainty > 0.3:
        return "HIGH"
    if uncertainty > 0.15:
        return "MODERATE"
    return "LOW"


def _interpret_uncertainty(uncertainty: float) -> str:
    if uncertainty > 0.3:
        return "High uncertainty suggests the model is not confident in its prediction. Additional testing recommended."
    if uncertainty > 0.15:
        return "Moderate uncertainty. Results should be interpreted with caution."
    return "Low uncertainty. Model is confident in its prediction."


def _get_recommendation(risk_level: str, prediction: float, uncertainty: float) -> str:
    recommendations = {
        "HIGH": "Strong positive prediction. Recommend clinical evaluation and consideration of amyloid PET or CSF biomarker confirmation.",
        "MODERATE": "Moderate risk. Recommend follow-up testing within 6-12 months.",
        "BORDERLINE": "Borderline result. Consider additional cognitive testing and follow-up.",
        "LOW": "Low risk. Routine follow-up per clinical guidelines.",
        "VERY_LOW": "Very low risk. Continue routine screening.",
        "UNCERTAIN": "High uncertainty in prediction. Recommend additional imaging or biomarker testing for clarification.",
    }
    return recommendations.get(risk_level, "Consult with clinical team for interpretation.")


def _extract_key_findings(features: dict) -> list[str]:
    findings = []
    if not features:
        return findings
    
    important_features = [
        ("age", "Age"),
        ("sex", "Sex"),
        ("mmse_base", "MMSE Score"),
        ("apoe4_carrier", "APOE4 Carrier"),
        ("education_years", "Education"),
    ]
    
    for key, name in important_features:
        if key in features:
            value = features[key]
            findings.append(f"{name}: {value}")
    
    return findings[:5]


def _get_caveats(modality: str) -> list[str]:
    base_caveats = [
        "This is a research prediction, not a clinical diagnosis.",
        "Results should be interpreted by qualified healthcare professionals.",
        "The model predicts amyloid biomarker status, not Alzheimer's disease diagnosis.",
    ]
    
    if modality == "mri":
        base_caveats.append("Method X-Strict uses T1 MRI and clinical variables only (PET-free).")
    elif modality == "pet":
        base_caveats.append("Method Y is PET-assisted exploratory with PET-label tautology caveat.")
    
    return base_caveats


def format_report_text(report: dict) -> str:
    lines = [
        "=" * 60,
        "AMYLOID BIOMARKER PREDICTION REPORT",
        "=" * 60,
        "",
        "PREDICTION:",
        f"  Status: {report['prediction']['amyloid_status']}",
        f"  Probability: {report['prediction']['probability']:.1%}",
        f"  Confidence: {report['prediction']['confidence']:.1%}",
        "",
        "UNCERTAINTY:",
        f"  Level: {report['uncertainty']['level']}",
        f"  Score: {report['uncertainty']['score']:.3f}",
        f"  {report['uncertainty']['interpretation']}",
        "",
        "RISK ASSESSMENT:",
        f"  {report['risk_assessment']}",
        "",
        "RECOMMENDATION:",
        f"  {report['recommendation']}",
        "",
    ]
    
    if report["key_findings"]:
        lines.append("KEY FINDINGS:")
        for finding in report["key_findings"]:
            lines.append(f"  - {finding}")
        lines.append("")
    
    lines.append("CAVEATS:")
    for caveat in report["caveats"]:
        lines.append(f"  - {caveat}")
    lines.append("")
    
    lines.append("=" * 60)
    lines.append(f"Model: {report['technical_details']['model']}")
    lines.append(f"Modality: {report['technical_details']['modality']}")
    lines.append(f"Version: {report['technical_details']['version']}")
    lines.append("=" * 60)
    
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Generate clinical report")
    parser.add_argument("--prediction", type=float, required=True, help="Model prediction probability")
    parser.add_argument("--uncertainty", type=float, default=0.1, help="Uncertainty score")
    parser.add_argument("--modality", choices=["mri", "pet"], default="mri", help="Input modality")
    parser.add_argument("--output", type=str, help="Output JSON file path")
    args = parser.parse_args()
    
    report = generate_clinical_report(args.prediction, args.uncertainty, modality=args.modality)
    
    if args.output:
        output_path = Path(args.output)
        output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Report saved to {output_path}")
    else:
        print(format_report_text(report))
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
