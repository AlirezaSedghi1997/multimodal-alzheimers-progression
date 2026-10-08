"""Inner-only, per-fold selection never consumes outer metrics."""
from __future__ import annotations

import json

import pytest

from scripts.mx_science_extension_v1 import selection


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_fold_local_selection_and_tie_break(tmp_path, monkeypatch):
    reports = tmp_path / selection.PARENT / "reports"
    _write(reports / "fasttrack_preregistration.json", {"cache_sha256": "c" * 64})
    _write(reports / "fasttrack_v241_verifier_repair_before_outer.json", {"status": "LOCKED"})
    _write(reports / "fasttrack_v242_checkpoint_projection_before_outer.json", {"status": "LOCKED"})
    _write(reports / "fasttrack_v244_posthoc_xai_atlas_eligibility.json", {"status": "LOCKED"})
    battery = (tmp_path / "results/final_results/"
               "method_x_model_battery_v3_depth_and_diversity_20260729/battery_manifest.json")
    _write(battery, {"spm_gate_snapshot": {"snapshot_sha256": "s" * 64}})
    monkeypatch.setattr(selection, "require_parent_mx7_terminal", lambda root: {"status": "VERIFIED"})

    def fake_checked(directory, *, seed, variant, cache_sha256, spm_sha256):
        assert cache_sha256 == "c" * 64 and spm_sha256 == "s" * 64
        base = 0.4 if variant == "baseline" else 0.3
        losses = [base] * 5
        if variant == "plus_dense_skip":
            losses[2] = 0.4  # Exact tie chooses baseline.
            losses[4] = 0.5  # Baseline wins this fold.
        return {"metric_sha256": f"metric-{seed}-{variant}",
                "contract_sha256": f"contract-{seed}-{variant}",
                "contract_digest": f"digest-{seed}-{variant}",
                "folds": {fold: {"best_val_loss": loss,
                                 "checkpoint_sha256": f"ckpt-{seed}-{variant}-{fold}"}
                          for fold, loss in enumerate(losses)}}

    monkeypatch.setattr(selection, "checked_inner_arm", fake_checked)
    result = selection.build_selection(tmp_path)
    assert len(result["choices"]) == 15
    assert [row["selected_variant"] for row in result["choices"][:5]] == [
        "plus_dense_skip", "plus_dense_skip", "baseline", "plus_dense_skip", "baseline"]
    assert result["outer_used_for_selection"] is False
    assert "outer" not in str(result["choices"]).lower()


def test_parent_mx7_terminal_gate_blocks_before_any_arm_read(tmp_path, monkeypatch):
    def blocked(_root):
        raise RuntimeError("EXTENSION_PARENT_MX7_NOT_TERMINAL")

    monkeypatch.setattr(selection, "require_parent_mx7_terminal", blocked)
    with pytest.raises(RuntimeError, match="PARENT_MX7_NOT_TERMINAL"):
        selection.build_selection(tmp_path)
