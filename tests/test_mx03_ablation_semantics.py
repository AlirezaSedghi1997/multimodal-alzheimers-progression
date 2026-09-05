"""Regression tests for MX-03 executable-arm semantics (CPU-only, no training)."""
from __future__ import annotations


def test_feature_drop_contracts_emit_parsed_flag():
    from src.evaluation.method_x_validity_tasks import (
        ALLOWED_FEATURES,
        build_task_contracts,
    )

    manifest = {
        "locked_protocol": {
            "resolution": 128,
            "folds": 5,
            "epochs": 100,
            "seed": 13,
            "common": {
                "fusion": "cross_attention",
                "task_loss": "bce",
                "learning_rate": 0.0001,
                "scheduler": "cosine",
                "dropout": 0.2,
                "batch_size": 1,
                "grad_accum": 8,
            },
        },
        "locked_inputs": {"cache_sha256": "c" * 64},
        "spm_gate_snapshot": {"snapshot_sha256": "e" * 64},
    }
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        contracts = build_task_contracts(
            root=root, output_root=root / "battery", manifest=manifest, python="python.exe"
        )
    by_id = {row["task_id"]: row for row in contracts}
    for name in ALLOWED_FEATURES:
        task_id = f"feature_drop_{name}"
        command = by_id[task_id]["command"]
        assert "--drop_tabular_feature" in command
        assert command[command.index("--drop_tabular_feature") + 1] == name


def test_resnet34_backbone_builds_without_fallback():
    import torch

    from src.models.backbones.backbone_factory import build_3d_backbone

    encoder = build_3d_backbone("resnet34_3d", in_channels=1, out_dim=128)
    encoder.eval()
    with torch.no_grad():
        out = encoder(torch.zeros(1, 1, 32, 32, 32))
    assert tuple(out.shape) == (1, 128)


def test_canonical_resolver_drops_worker_only_knobs_at_zero_workers():
    from src.utils.dataloader_config import resolve_dataloader_kwargs

    assert resolve_dataloader_kwargs(num_workers=0, pin_memory=True) == {
        "num_workers": 0,
        "pin_memory": True,
    }
    full = resolve_dataloader_kwargs(
        num_workers=2, pin_memory=True, persistent_workers=False, prefetch_factor=2
    )
    assert full == {
        "num_workers": 2,
        "pin_memory": True,
        "persistent_workers": False,
        "prefetch_factor": 2,
    }
