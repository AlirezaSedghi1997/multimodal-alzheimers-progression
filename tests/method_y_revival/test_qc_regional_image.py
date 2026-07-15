from __future__ import annotations

import numpy as np
import pytest


def test_automated_qc_fails_nonfinite_and_reference_geometry_mismatch():
    import nibabel as nib

    from src.method_y_revival.pet_qc import automated_image_qc

    data = np.ones((8, 8, 8), dtype=np.float32)
    data[0, 0, 0] = np.nan
    image = nib.Nifti1Image(data, np.diag([2.0, 2.0, 2.0, 1.0]))
    reference = nib.Nifti1Image(np.zeros((8, 8, 8), dtype=np.float32), np.eye(4))

    report = automated_image_qc(image, reference)

    assert report["finite_fraction"] < 1.0
    assert report["affine_matches_template"] is False
    assert report["qc_status"] == "FAIL"
    assert set(report["failure_reasons"]) == {"NONFINITE_VOXELS", "AFFINE_MISMATCH"}


def test_visual_qc_selection_includes_every_failure_and_is_deterministic():
    from src.method_y_revival.pet_qc import select_visual_qc

    rows = [
        {
            "scan_key": f"scan-{index:03d}",
            "tracer": "FTP",
            "cohort": "A4",
            "manufacturer": "M1" if index % 2 else "M2",
            "qc_status": "FAIL" if index in {3, 8} else "PASS",
        }
        for index in range(120)
    ]

    first = select_visual_qc(rows, seed=7, fraction=0.05, minimum=100, maximum=500)
    repeated = select_visual_qc(list(reversed(rows)), seed=7, fraction=0.05, minimum=100, maximum=500)

    assert first == repeated
    assert len(first) == 100
    assert {"scan-003", "scan-008"}.issubset(set(first))


def test_regional_feature_policy_excludes_outcomes_and_builds_matched_sets():
    from src.method_y_revival.regional_features import build_matched_feature_sets, select_tau_regions

    columns = [
        "age",
        "baseline_pacc_raw",
        "event",
        "time_days",
        "tau_persi__entorhinal_lh_voi",
        "tau_persi__inferiortemporal_lh_voi",
    ]
    tau = select_tau_regions(columns)
    feature_sets = build_matched_feature_sets(
        base_features=["age", "baseline_pacc_raw"], pet_features=tau
    )

    assert tau == ["tau_persi__entorhinal_lh_voi", "tau_persi__inferiortemporal_lh_voi"]
    assert feature_sets["clinical_cognition"] == ["age", "baseline_pacc_raw"]
    assert set(feature_sets["clinical_cognition_pet"]) - set(feature_sets["clinical_cognition"]) == set(tau)


def test_cox_image_loss_is_censoring_aware_and_differentiable():
    import torch

    from src.method_y_revival.image_survival import negative_cox_partial_log_likelihood

    risk = torch.tensor([0.2, 0.4, -0.1], requires_grad=True)
    event = torch.tensor([1.0, 0.0, 1.0])
    time = torch.tensor([3.0, 2.0, 1.0])

    loss = negative_cox_partial_log_likelihood(risk, event, time)
    loss.backward()

    assert torch.isfinite(loss)
    assert risk.grad is not None
    assert torch.isfinite(risk.grad).all()


def test_cox_image_loss_rejects_a_batch_without_events():
    import torch

    from src.method_y_revival.image_survival import ImageSurvivalError, negative_cox_partial_log_likelihood

    with pytest.raises(ImageSurvivalError, match="COX_BATCH_REQUIRES_EVENT"):
        negative_cox_partial_log_likelihood(
            torch.tensor([0.1, 0.2]), torch.tensor([0.0, 0.0]), torch.tensor([1.0, 2.0])
        )
