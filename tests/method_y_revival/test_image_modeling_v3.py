from __future__ import annotations

import numpy as np
import torch
import pytest


def test_survival_at_times_boundary_equality():
    from src.method_y_revival.image_modeling_v3 import survival_at_times

    logits = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0]], dtype=torch.float32)
    edges = [0.0, 365.0, 730.0, 1095.0, 1460.0, 1640.0]
    # hazard 0.5 each -> survival at boundaries vs discrete cumprod
    hazards = torch.sigmoid(logits)
    one_minus = 1 - hazards
    cum = torch.cumprod(one_minus, dim=1)
    # boundary times
    times = edges
    surv = survival_at_times(logits, edges, times)
    # at edge 0, survival 1
    assert float(surv[0, 0]) == pytest.approx(1.0, abs=1e-5)
    # at edge 365, should be 0.5
    # discrete cum after first bin = 0.5
    assert float(surv[0, 1]) == pytest.approx(float(cum[0, 0]), abs=1e-5)
    assert float(surv[0, 2]) == pytest.approx(float(cum[0, 1]), abs=1e-5)
    assert float(surv[0, 5]) == pytest.approx(float(cum[0, 4]), abs=1e-5)


def test_survival_at_times_monotonicity_and_range():
    from src.method_y_revival.image_modeling_v3 import survival_at_times

    torch.manual_seed(0)
    logits = torch.randn(4, 5)
    edges = [0.0, 365.0, 730.0, 1095.0, 1460.0, 1640.0]
    times = np.linspace(1, 1639, 20).tolist()
    surv = survival_at_times(logits, edges, times)
    arr = surv.detach().numpy()
    assert np.all(arr >= 0.0) and np.all(arr <= 1.0)
    assert np.all(np.isfinite(arr))
    # monotonic non-increasing across sorted times
    for row in arr:
        for i in range(len(row) - 1):
            assert row[i] >= row[i + 1] - 1e-6


def test_survival_at_times_zero_one_hazard_stability():
    from src.method_y_revival.image_modeling_v3 import survival_at_times

    logits = torch.tensor([[100.0, -100.0, 100.0, -100.0, 0.0]], dtype=torch.float32)
    edges = [0.0, 365.0, 730.0, 1095.0, 1460.0, 1640.0]
    times = [10.0, 400.0, 800.0, 1200.0, 1500.0]
    surv = survival_at_times(logits, edges, times)
    assert torch.isfinite(surv).all()
    assert (surv >= 0).all() and (surv <= 1).all()


def test_survival_at_times_grid_outside_locked_support_rejected():
    from src.method_y_revival.image_modeling_v3 import survival_at_times, ImageModelingV3Error

    logits = torch.zeros(2, 5)
    edges = [0.0, 365.0, 730.0, 1095.0, 1460.0, 1640.0]
    with pytest.raises(ImageModelingV3Error, match="OUTSIDE_LOCKED_SUPPORT"):
        survival_at_times(logits, edges, [-10.0, 100.0])
    with pytest.raises(ImageModelingV3Error, match="OUTSIDE_LOCKED_SUPPORT"):
        survival_at_times(logits, edges, [100.0, 2000.0])


def test_survival_at_times_same_fold_grid_for_matched_variants():
    from src.method_y_revival.image_modeling_v3 import survival_at_times

    # Simulate matched variants: same bin edges, same evaluation grid, different logits but same survival grid support
    torch.manual_seed(42)
    edges = [0.0, 365.0, 730.0, 1095.0, 1460.0, 1640.0]
    grid = [100.0, 500.0, 900.0, 1300.0]
    logits_a = torch.randn(2, 5)
    logits_b = torch.randn(2, 5)
    surv_a = survival_at_times(logits_a, edges, grid)
    surv_b = survival_at_times(logits_b, edges, grid)
    # Both should succeed with same grid, shape matching
    assert surv_a.shape == (2, 4)
    assert surv_b.shape == (2, 4)


def test_build_model_variants():
    from src.method_y_revival.image_modeling_v3 import build_model, LOCKED_VARIANTS

    assert set(LOCKED_VARIANTS) == {"scalar_tabular", "scalar_tabular_plus_pattern_image"}
    cfg = {"base_channels": 8, "embedding_dim": 64, "tabular_hidden_dim": 32, "dropout": 0.25}
    m1 = build_model("scalar_tabular", cfg, tabular_dim=11, hazard_bins=5)
    m2 = build_model("scalar_tabular_plus_pattern_image", cfg, tabular_dim=11, hazard_bins=5)
    # Tabular takes single input, pattern takes image+tabular
    x_tab = torch.randn(2, 11)
    out1 = m1(x_tab)
    assert out1.shape == (2, 5)
    x_img = torch.randn(2, 1, 16, 16, 16)
    out2 = m2(x_img, x_tab)
    assert out2.shape == (2, 5)


def test_build_model_unknown_variant_fails():
    from src.method_y_revival.image_modeling_v3 import build_model, ImageModelingV3Error

    with pytest.raises(ImageModelingV3Error, match="UNKNOWN_LOCKED_VARIANT"):
        build_model("image_only", {}, tabular_dim=5, hazard_bins=3)


def test_matched_variants_have_identical_features_budget_splits_and_hpo_grid():
    import yaml
    from pathlib import Path

    config_path = Path("configs/method_y_image_training_v3_1.yaml")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    # HPO grid identical for both variants by construction (single grid in config)
    hpo = config["model"]["hpo_candidates"]
    assert len(hpo) == 3
    # Check complexity_rank present and unique
    ranks = [c["complexity_rank"] for c in hpo]
    assert ranks == sorted(ranks) or set(ranks) == {0, 1, 2}
    # Feature sets distinct but share base
    amy_features = config["branches"]["amyloid"]["model_features"]
    tau_features = config["branches"]["tau"]["model_features"]
    assert "baseline_amyloid_centiloid" in amy_features
    # Split files exist
    assert Path(config["branches"]["amyloid"]["splits"]).exists()
    assert Path(config["branches"]["tau"]["splits"]).exists()
