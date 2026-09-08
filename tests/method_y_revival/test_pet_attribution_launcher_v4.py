"""Regression tests for attribution launcher repairs (C-fix 2026-09-08).

Covers the BLOCKER/CRITICAL findings of three independent supervisor audits
that the module battery (test_pet_attribution_v4.py) never exercised, because
the launcher had zero tests. All tests are CPU-only, synthetic, no checkpoints,
no real data. RED before the fix (each asserts absence of a crashed pattern or
presence of repaired behavior), GREEN after.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, r"H:\New project")

LAUNCHER = Path(r"H:\New project\scripts\run_method_y_pet_attribution_v4.py")
LAUNCHER_SRC = LAUNCHER.read_text(encoding="utf-8")


def test_no_undefined_pilot_names():
    """F3/F4: pilot block must not reference undefined names."""
    assert "len(pilot)" not in LAUNCHER_SRC.replace("len(pilot_keys)", "").replace(
        "len(pilot_rows)", "")
    assert "_layercam_full_for_agreement" not in LAUNCHER_SRC
    assert "LEG-02-FULL" not in LAUNCHER_SRC


def test_no_stale_batch_indexing():
    """F2/F8: pilot must reload per-cell batches, never batch[subj]."""
    assert "batch[subj]" not in LAUNCHER_SRC
    assert "img, tab = batch[subj]" not in LAUNCHER_SRC
    assert "cell_batch = _load_cell_batch(cell" in LAUNCHER_SRC
    assert "lookup = {s: (i, t) for s, i, t in cell_batch}" in LAUNCHER_SRC


def test_score_times_is_list_and_maps_keyed_by_cell():
    """F1/F5/F9: score_times list; maps keyed by (cell, subject)."""
    assert "score_times: list = []" in LAUNCHER_SRC
    assert "score_maps[(key, subj)]" in LAUNCHER_SRC
    assert "layer_maps[(key, subj)]" in LAUNCHER_SRC


def test_config_final_gate_present():
    """B-F5: production must refuse DRAFT params."""
    assert "ATTRIBUTION_CONFIG_NOT_FINAL" in LAUNCHER_SRC
    assert 'POST_HOC_INTERNAL_EXPLORATORY_FINAL' in LAUNCHER_SRC


def test_per_branch_dims_and_bins_inference():
    """C-F1/F2: tau dim=12 honored; bins inferred from payload."""
    assert "_arch_dims_for_branch" in LAUNCHER_SRC
    assert "_infer_bins_from_payload" in LAUNCHER_SRC
    assert "tabular_dim=11, bins=5" not in LAUNCHER_SRC
    assert '"dropout": 0.0' not in LAUNCHER_SRC


def test_build_production_legs_schema_valid():
    """A-F11/B-F2: every emitted leg id/status passes evaluate_legs_table."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("attr_launcher", str(LAUNCHER))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    from src.method_y_revival.pet_attribution_v4 import evaluate_legs_table

    legs, extra = mod.build_production_legs(
        agree_status="RED_FLAG_INVESTIGATE",
        share_records=[{"cell": "c", "image_share": 0.2, "stamp": "PARTIAL_EXPLANATION"}],
        share_floor=0.5, pilot_seconds_per_volume=1.0,
        projected_hours=99.0, deadline_hours=24.0, pilot_timed_out=False)
    assert {l["id"] for l in legs} == {"LEG-01", "LEG-02", "LEG-03", "LEG-04",
                                       "LEG-05", "LEG-06", "LEG-07", "LEG-08"}
    assert [l["id"] for l in legs].count("LEG-02") == 1
    by_id = {l["id"]: l for l in legs}
    assert by_id["LEG-04"]["status"] == "NOT_MEASURABLE"
    assert by_id["LEG-06"]["status"] == "NOT_MEASURABLE"
    assert by_id["LEG-02"]["status"] == "NOT_MEASURABLE"
    assert evaluate_legs_table(legs)["status"] == "PASS"

    legs_ok, _ = mod.build_production_legs(
        agree_status="PASS",
        share_records=[{"cell": "c", "image_share": 0.8, "stamp": "FULL_EXPLANATION"}],
        share_floor=0.5, pilot_seconds_per_volume=1.0,
        projected_hours=2.0, deadline_hours=24.0, pilot_timed_out=False)
    by_ok = {l["id"]: l for l in legs_ok}
    assert by_ok["LEG-04"]["status"] == "PASS"
    assert by_ok["LEG-06"]["status"] == "PASS"
    assert by_ok["LEG-02"]["status"] == "PASS"
    assert evaluate_legs_table(legs_ok)["status"] == "PASS"

    legs_to, _ = mod.build_production_legs(
        agree_status="PASS", share_records=[],
        share_floor=0.5, pilot_seconds_per_volume=1.0,
        projected_hours=0.0, deadline_hours=24.0, pilot_timed_out=True)
    assert {l["id"]: l for l in legs_to}["LEG-02"]["status"] == "TIMEOUT"
    assert evaluate_legs_table(legs_to)["status"] == "PASS"


def test_verifier_gate_tampered_artifact_blocked(tmp_path):
    """C-F12: present-but-invalid artifact must BLOCK, never fall through."""
    import json

    from src.method_y_revival.pet_attribution_v4 import verifier_gate

    (tmp_path / "final_verification_pass.json").write_text(
        json.dumps({"status": "PASS", "run_contract_sha256": "x" * 64,
                    "integrity_sha256": "0" * 64}), encoding="utf-8")
    with pytest.raises(RuntimeError, match="BLOCKED_VERIFIER_ARTIFACT_TAMPERED"):
        verifier_gate(run_root=tmp_path)


def test_privacy_scan_labeled_digest_passes_bare_hash_blocked():
    """C-F10: labeled provenance digests must not self-BLOCK; bare hashes must."""
    from src.method_y_revival.pet_attribution_v4 import scan_subject_identifiers

    ok = scan_subject_identifiers(
        "run_contract_sha256: 969ebd0f61d9ffc3bc262c5476c75b23575d64d1f5703d23fe04d1095d07743e")
    assert ok["status"] == "PASS"
    bad = scan_subject_identifiers(
        "subject 969ebd0f61d9ffc3bc262c5476c75b23575d64d1f5703d23fe04d1095d07743e leaked")
    assert bad["status"] == "BLOCKED"
    assert scan_subject_identifiers("see H:\\New project\\results\\x")["status"] == "BLOCKED"


def test_arch_dims_branch_shapes():
    """C-F1: amyloid tabular_dim=11, tau=12 from locked factorial config."""
    import importlib.util
    import yaml

    spec = importlib.util.spec_from_file_location("attr_launcher2", str(LAUNCHER))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    factor_cfg = yaml.safe_load(open(
        r"H:\New project\configs\method_y_image_factorial_v4.yaml", encoding="utf-8"))
    assert mod._arch_dims_for_branch(factor_cfg, "amyloid")[0] == 11
    assert mod._arch_dims_for_branch(factor_cfg, "tau")[0] == 12
    import torch
    payload = {"model_state": {"head.1.weight": torch.zeros(5, 96),
                               "tabular.0.weight": torch.zeros(32, 11)}}
    assert mod._infer_bins_from_payload(payload) == 5
    assert mod._infer_bins_from_payload({"model_state": {}}) == 5


def test_blocked_log_never_dumps_firewall_subjects():
    """Privacy: BLOCKED cell log must not dump the cell dict (subject lists)."""
    assert '{"status": "BLOCKED", "cell": cell' not in LAUNCHER_SRC
    assert '"cell": key' in LAUNCHER_SRC
    assert '"n_subjects"' in LAUNCHER_SRC


def test_verify_final_provenance_frozen_chain(tmp_path):
    """Frozen-chain binding: bytes locked by FINALIZED outer state PASS."""
    import hashlib as _hl
    import json as _js

    import torch

    from src.method_y_revival.pet_attribution_v4 import verify_final_provenance

    v4 = tmp_path / "v4"
    state_dir = v4 / "private" / "outer_evaluation" / "amyloid" / "A0" / "fold_0"
    state_dir.mkdir(parents=True)
    ckpt = tmp_path / "locked_checkpoint.pt"
    torch.save({"model_state": {"w": torch.zeros(2, 2)},
                "run_contract_digest": "a" * 64}, ckpt)
    file_sha = _hl.sha256(ckpt.read_bytes()).hexdigest()
    body = {"schema_version": "test", "state": "FINALIZED", "fold": 0,
            "checkpoint_sha256": file_sha, "run_contract_sha256": "a" * 64,
            "result_record_sha256": "b" * 64}
    body["state_sha256"] = _hl.sha256(
        _js.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    (state_dir / "state.json").write_text(_js.dumps(body), encoding="utf-8")
    out = verify_final_provenance(ckpt, v4_root=v4, branch="amyloid", arm="A0", outer=0)
    assert out["status"] == "PASS" and out["generation_digest"] == "a" * 64
    with ckpt.open("ab") as fh:
        fh.write(b"\x00")
    with pytest.raises(RuntimeError, match="BLOCKED"):
        verify_final_provenance(ckpt, v4_root=v4, branch="amyloid", arm="A0", outer=0)
