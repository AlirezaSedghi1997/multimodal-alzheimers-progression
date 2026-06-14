"""Tests for literature-backed innovation modules (SynIB, MFSCA, worst-dimension)."""
from __future__ import annotations

import json
from pathlib import Path


def test_synib_loss_decomposes():
    import torch

    from src.models.losses.synib_synergy_loss import SynIBConfig, synib_multimodal_loss

    logits = torch.tensor([0.5, -0.5, 1.0])
    targets = torch.tensor([1.0, 0.0, 1.0])
    total, parts = synib_multimodal_loss(logits, logits * 0.1, logits * 0.1, targets, config=SynIBConfig())
    assert total.item() > 0
    assert "syn_penalty" in parts


def test_mfsca_features_from_row():
    from src.features.mfsca_texture import extract_mfsca_from_mri_row

    row = {"mri_mean": "70", "mri_std": "137", "mri_p10": "1", "mri_p50": "5", "mri_p90": "280"}
    feat = extract_mfsca_from_mri_row(row)
    assert set(feat) == {"mfsca_hurst_proxy", "mfsca_spectrum_width_proxy", "mfsca_asymmetry_proxy"}
    assert 0.0 <= feat["mfsca_hurst_proxy"] <= 1.0


def test_worst_dimension_ranks_by_chebyshev():
    from src.evaluation.worst_dimension_audit import worst_dimension_report

    scores = {
        "good": {"roc_auc": 0.75, "calibration": 0.8, "cohort_probe": 0.7, "missing_modality": 0.72},
        "bad": {"roc_auc": 0.76, "calibration": 0.3, "cohort_probe": 0.2, "missing_modality": 0.5},
    }
    report = worst_dimension_report(scores)
    assert report["best_variant"] == "good"


def test_literature_registry_exists():
    p = Path("docs/literature_backbone_evidence_2023_2026.yaml")
    assert p.exists()
    text = p.read_text(encoding="utf-8")
    assert "LIT019" in text
    assert "LIT016" in text
    assert "LIT018" in text


def test_mfsca_artifact_contract_if_generated():
    p = Path("results/tables/mfsca_texture_features.json")
    if not p.exists():
        return
    obj = json.loads(p.read_text(encoding="utf-8"))
    assert obj["literature_anchor"] == "LIT018 arXiv:2606.10222"
    assert obj["n"] > 0


def test_synib_fusion_artifact_contract_if_generated():
    p = Path("results/final_results/synib_fusion/synib_fusion_metrics.json")
    if not p.exists():
        return
    obj = json.loads(p.read_text(encoding="utf-8"))
    assert obj["status"] == "COMPLETED"
    assert "LIT019" in obj["aggregate"]["literature_anchor"]
