from __future__ import annotations

from pathlib import Path

import torch

from scripts.migrate_method_y_image_factorial_v4_contract import (
    _semantic_fingerprint,
    _update_torch_file,
)


def test_semantic_fingerprint_ignores_only_contract_digest() -> None:
    left = {
        "run_contract_digest": "a" * 64,
        "epoch_completed": 3,
        "model_state": {"weight": torch.tensor([1.0, 2.0])},
    }
    right = dict(left, run_contract_digest="b" * 64)
    assert _semantic_fingerprint(left) == _semantic_fingerprint(right)


def test_torch_contract_migration_is_atomic_and_value_preserving(tmp_path: Path) -> None:
    path = tmp_path / "epoch_checkpoint.pt"
    old = "a" * 64
    new = "b" * 64
    payload = {
        "run_contract_digest": old,
        "epoch_completed": 4,
        "best_model_state": {"weight": torch.tensor([1.0, 2.0])},
        "model_state": {"weight": torch.tensor([3.0, 4.0])},
    }
    torch.save(payload, path)
    row = {
        "path": str(path),
        "semantic_fingerprint": _semantic_fingerprint(payload),
        "new_sha256": None,
    }

    _update_torch_file(row, old, new)

    migrated = torch.load(path, map_location="cpu", weights_only=False)
    assert migrated["run_contract_digest"] == new
    assert migrated["epoch_completed"] == 4
    assert torch.equal(migrated["best_model_state"]["weight"], torch.tensor([1.0, 2.0]))
    assert row["new_sha256"]

