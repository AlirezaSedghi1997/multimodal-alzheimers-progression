"""ResNet GroupNorm challengers must train under the locked deterministic contract.

Regression test for the A2-start BLOCKED:
`max_pool3d_with_indices_backward_cuda does not have a deterministic
implementation` with `torch.use_deterministic_algorithms(True)`.
Every factorial arm (including ResNet10/18-GN) must complete forward+backward
on CUDA under determinism; otherwise exact-resume is a lie and production
blocks at the first ResNet epoch.
"""
from __future__ import annotations

import os

# Mirror the locked production runtime (configs/...v4.yaml runtime +
# run_method_y_image_factorial_v4.ps1): determinism requires a CUBLAS
# workspace config selected BEFORE any CUDA/cuBLAS call in this process.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import pytest
import torch

from src.method_y_revival.image_factorial_v4 import build_factorial_image_model

CUDA = torch.cuda.is_available()

MODEL_CONFIG = {
    "embedding_dim": 16,
    "tabular_hidden_dim": 8,
    "dropout": 0.0,
    "base_channels": 8,
    "resnet_width_multiplier": 0.25,
}


@pytest.mark.parametrize("arm_id", ["A2", "A4"])
@pytest.mark.skipif(not CUDA, reason="requires CUDA for the deterministic-maxpool check")
def test_resnet_arm_backward_is_deterministic_on_cuda(arm_id):
    torch.use_deterministic_algorithms(True)
    try:
        model = build_factorial_image_model(
            arm_id, MODEL_CONFIG, tabular_dim=4, hazard_bins=3
        ).to("cuda")
        model.train()
        image = torch.randn(1, 1, 32, 32, 32, device="cuda")
        tabular = torch.randn(1, 4, device="cuda")
        logits = model(image, tabular)
        loss = logits.pow(2).mean()
        loss.backward()
    finally:
        torch.use_deterministic_algorithms(False)
    assert torch.isfinite(loss).item()
