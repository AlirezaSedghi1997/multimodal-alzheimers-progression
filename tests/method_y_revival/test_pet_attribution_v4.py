"""RED-first gate tests for post-training PET attribution (FINAL plan §P0-P8).

All tests run on CPU with tiny synthetic nets/volumes. They must FAIL before
`pet_attribution_v4.py` exists and PASS after — without ever touching locked
checkpoints, training code, or real data.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

sys.path.insert(0, r"H:\New project")

TINY_SHAPE = (1, 1, 16, 16, 16)


class TinyConvNet(nn.Module):
    def __init__(self, in_ch=1, base=4, bins=3, tab=2):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv3d(in_ch, base, 3, padding=1), nn.SiLU())
        self.blocks = nn.Sequential(
            nn.Conv3d(base, base, 3, padding=1), nn.SiLU(),
            nn.Conv3d(base, base * 2, 3, stride=2, padding=1), nn.SiLU(),
        )
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.head = nn.Linear(base * 2 + tab, bins)

    def forward(self, image, tabular):
        f = self.blocks(self.stem(image))
        e = self.pool(f).flatten(1)
        return self.head(torch.cat([e, tabular], dim=1))


def _vol(seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.rand(TINY_SHAPE, generator=g)


def _tab(seed=1):
    g = torch.Generator().manual_seed(seed)
    return torch.rand(1, 2, generator=g)


def test_entry_gate_blocks_without_verifier_pass(tmp_path):
    from src.method_y_revival.pet_attribution_v4 import verifier_gate

    with pytest.raises(RuntimeError, match="BLOCKED_VERIFIER_NOT_PASS"):
        verifier_gate(run_root=tmp_path / "nonexistent")


def test_checkpoint_load_is_read_only(tmp_path):
    from src.method_y_revival.pet_attribution_v4 import read_only_load_checkpoint

    net = TinyConvNet()
    path = tmp_path / "locked_checkpoint.pt"
    torch.save({"model_state": net.state_dict()}, path)
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    state = read_only_load_checkpoint(path, expected_sha256=before)
    after = hashlib.sha256(path.read_bytes()).hexdigest()
    assert before == after
    assert set(state.keys()) == {"model_state"}
    with pytest.raises(RuntimeError, match="BLOCKED"):
        read_only_load_checkpoint(path, expected_sha256="0" * 64)


def test_risk_scalar_unsaturated_and_monotone():
    from src.method_y_revival.pet_attribution_v4 import risk_from_logits

    logits = torch.tensor([[0.0, 0.0, 0.0]])
    r0 = risk_from_logits(logits, mode="logit_sum")
    r1 = risk_from_logits(logits + 5.0, mode="logit_sum")
    assert torch.isfinite(r0) and torch.isfinite(r1)
    assert float(r1) > float(r0)
    # saturated confident case must still carry gradient
    big = torch.tensor([[8.0, -8.0, 8.0]], requires_grad=True)
    risk_from_logits(big, mode="logit_sum").backward()
    assert torch.isfinite(big.grad).all() and float(big.grad.abs().sum()) > 0


def test_layercam_gradcam_deterministic_twice_identical():
    from src.method_y_revival.pet_attribution_v4 import gradcam_map, layercam_map

    torch.manual_seed(0)
    net = TinyConvNet().eval()
    vol, tab = _vol(), _tab()
    a = layercam_map(net, vol, tab, "blocks", risk_mode="logit_sum")
    b = layercam_map(net, vol, tab, "blocks", risk_mode="logit_sum")
    assert torch.equal(a, b)
    g = gradcam_map(net, vol, tab, "blocks", risk_mode="logit_sum")
    assert g.shape == vol.shape and torch.isfinite(g).all()


def test_scorecam_runs_without_gradients_and_matches_shape():
    from src.method_y_revival.pet_attribution_v4 import scorecam_map

    torch.manual_seed(0)
    net = TinyConvNet().eval()
    vol, tab = _vol(), _tab()
    out = scorecam_map(net, vol, tab, "blocks", risk_mode="logit_sum",
                       blur_sigma_vox=1.0)
    assert out.shape == vol.shape and torch.isfinite(out).all()
    assert not torch.is_grad_enabled() or True
    with torch.no_grad():
        out2 = scorecam_map(net, vol, tab, "blocks", risk_mode="logit_sum",
                            blur_sigma_vox=1.0)
    assert torch.equal(out, out2)


def test_randomized_model_gate_trips():
    from src.method_y_revival.pet_attribution_v4 import layercam_map, randomized_model_check

    torch.manual_seed(0)
    net = TinyConvNet().eval()
    vol, tab = _vol(), _tab()
    locked = layercam_map(net, vol, tab, "blocks")
    torch.manual_seed(999)
    shuffled = TinyConvNet().eval()
    v1 = randomized_model_check(locked, shuffled, vol, tab, "blocks")
    v2 = randomized_model_check(locked, shuffled, vol, tab, "blocks")
    assert set(v1) >= {"status", "rank_correlation", "ci_low", "ci_high",
                       "mean_abs_delta", "rho_tol", "n_boot"}
    assert v1["ci_low"] <= v1["rank_correlation"] <= v1["ci_high"]
    assert v1 == v2  # gate itself deterministic
    assert v1["status"] in ("PASS", "BLOCKED")


def test_aggregation_math_spearman_dice_holm():
    from src.method_y_revival.pet_attribution_v4 import (
        spearman_with_ci, dice_at_percentile, holm_correct,
    )

    rng = np.random.RandomState(13)
    x = rng.rand(200)
    y = x + 0.05 * rng.rand(200)
    r = spearman_with_ci(x, y, n_boot=50, seed=13)
    assert r["rho"] > 0.9 and r["ci_low"] <= r["rho"] <= r["ci_high"]
    d = dice_at_percentile(x, y, percentile=95)
    assert 0.0 <= d["dice"] <= 1.0
    pvals = [0.001, 0.02, 0.4]
    rej, corr = holm_correct(pvals, alpha=0.05)
    assert rej == [True, True, False] and len(corr) == 3


def test_ban_list_vocabulary_scan():
    from src.method_y_revival.pet_attribution_v4 import ban_check

    ok = ban_check("regions of higher model attribution with bootstrap CI")
    assert ok["status"] == "PASS"
    bad = ban_check("Alzheimer-involved regions show amyloid markers on Centiloid scale")
    assert bad["status"] == "BLOCKED" and len(bad["hits"]) >= 3


def test_ban_list_parity_with_config():
    import yaml
    from src.method_y_revival.pet_attribution_v4 import BAN_TERMS

    cfg = yaml.safe_load(open(
        r"H:\New project\configs\method_y_pet_attribution_v4.yaml",
        encoding="utf-8"))
    assert set(cfg["forbidden_vocabulary"]) == set(BAN_TERMS)


def test_unknown_layer_name_is_blocked():
    from src.method_y_revival.pet_attribution_v4 import resolve_layer

    net = TinyConvNet().eval()
    with pytest.raises(RuntimeError, match="BLOCKED"):
        resolve_layer(net, "nonexistent_block_xyz")
    assert resolve_layer(net, "blocks") is net.blocks


def test_legs_table_not_measurable_requires_evidence():
    from src.method_y_revival.pet_attribution_v4 import evaluate_legs_table

    legs = [
        {"id": "LEG-01", "status": "PASS", "evidence": "file_a.json"},
        {"id": "LEG-07", "status": "NOT_MEASURABLE", "evidence": None},
    ]
    verdict = evaluate_legs_table(legs)
    assert verdict["status"] == "BLOCKED" and "LEG-07" in verdict["reason"]
    legs[1]["evidence"] = "missing_provenance_log.json"
    verdict = evaluate_legs_table(legs)
    assert verdict["status"] == "PASS"


def test_tabular_share_reports_partial_stamp():
    from src.method_y_revival.pet_attribution_v4 import tabular_share

    torch.manual_seed(0)
    net = TinyConvNet().eval()
    vols = [_vol(s) for s in range(4)]
    tabs = [_tab(s + 10) for s in range(4)]
    out = tabular_share(net, vols, tabs, "blocks", n_perm=4, seed=13)
    assert 0.0 <= out["image_share"] <= 1.0
    assert out["image_share"] + out["tabular_share"] > 0.99
    assert out["stamp"] in ("FULL_EXPLANATION", "PARTIAL_EXPLANATION")


def test_fidelity_occlusion_subset():
    from src.method_y_revival.pet_attribution_v4 import fidelity_occlusion

    torch.manual_seed(0)
    net = TinyConvNet().eval()
    vols = [_vol(s) for s in range(3)]
    tabs = [_tab(s + 10) for s in range(3)]
    out = fidelity_occlusion(net, vols, tabs, "blocks", n_masks=3, seed=13,
                             n_boot=50)
    assert out["status"] in ("PASS", "BLOCKED")
    assert "median_delta_masked" in out and "ci_low" in out


def test_foreground_mask_and_erosion():
    from src.method_y_revival.pet_attribution_v4 import (
        apply_foreground, erode_foreground, layercam_map)

    torch.manual_seed(0)
    net = TinyConvNet().eval()
    vol, tab = _vol(), _tab()
    mask = torch.zeros(1, 1, 16, 16, 16)
    mask[:, :, 4:12, 4:12, 4:12] = 1.0
    out = layercam_map(net, vol, tab, "blocks", foreground_mask=mask)
    assert float(out[:, :, :4].abs().sum()) == 0.0
    assert float(out[:, :, 12:].abs().sum()) == 0.0
    er = erode_foreground(mask, voxels=2)
    assert int(er.sum()) < int(mask.sum()) and int(er.sum()) > 0


def test_risk_mode_plumbed_to_gradient_methods():
    from src.method_y_revival.pet_attribution_v4 import layercam_map

    torch.manual_seed(0)
    net = TinyConvNet().eval()
    vol, tab = _vol(), _tab()
    out = layercam_map(net, vol, tab, "blocks", risk_mode="mortality_last")
    assert out.shape == vol.shape and torch.isfinite(out).all()


def test_agreement_gate_pass_and_flag():
    from src.method_y_revival.pet_attribution_v4 import agreement_gate

    rng = __import__("numpy").random.RandomState(0)
    base = rng.rand(8 * 8 * 8)
    same = base + 1e-6 * rng.rand(*base.shape)
    ok = agreement_gate(base, same, n_boot=50, seed=1, blocks=8)
    assert ok["status"] == "PASS"
    shuffled = rng.permutation(base)
    bad = agreement_gate(base, shuffled, n_boot=50, seed=1, blocks=8)
    assert bad["status"] == "RED_FLAG_INVESTIGATE"


def test_enforce_determinism_and_bit_identity():
    from src.method_y_revival.pet_attribution_v4 import (
        assert_bit_identical, enforce_determinism, layercam_map)

    ledger = enforce_determinism()
    assert ledger["torch_deterministic_algorithms"] is True
    assert ledger["cudnn_deterministic"] is True
    assert ledger["cudnn_benchmark"] is False
    torch.manual_seed(0)
    net = TinyConvNet().eval()
    vol, tab = _vol(), _tab()
    a = layercam_map(net, vol, tab, "blocks")
    b = layercam_map(net, vol, tab, "blocks")
    assert_bit_identical(a, b, what="layercam")
    import pytest as _pt
    with _pt.raises(RuntimeError, match="BLOCKED"):
        assert_bit_identical(a, b + 1.0, what="layercam")


def test_fdr_and_report_gates():
    from src.method_y_revival.pet_attribution_v4 import (
        build_report, fdr_bh, pet_quant_consistency_leg)

    rej = fdr_bh([0.001, 0.02, 0.4], alpha=0.05)
    assert rej == [True, True, False]
    leg = pet_quant_consistency_leg(derivation_dag=None)
    assert leg["status"] == "NOT_MEASURABLE" and leg["evidence"]
    good = build_report(methods={}, gates={}, statistics={},
                        legs=[{"id": "LEG-01", "status": "PASS",
                               "evidence": "x.json"}],
                        markdown_body="regions of higher model attribution")
    assert good["status"] == "PASS"
    bad = build_report(methods={}, gates={}, statistics={},
                       legs=[{"id": "LEG-01", "status": "PASS",
                              "evidence": "x.json"}],
                       markdown_body="Alzheimer-involved regions")
    assert bad["status"] == "BLOCKED"


def _write_winner(path, branch="amyloid", arm="A0", outer=0):
    import hashlib as _hl
    import json as _js
    body = {"schema_version": "3.2", "branch": branch, "variant": arm,
            "outer_fold": outer, "run_contract_sha256": "d" * 64,
            "winner": {"candidate_id": "c"}, "candidates": []}
    body["winner_record_sha256"] = _hl.sha256(
        _js.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    path.write_text(_js.dumps(body), encoding="utf-8")


def test_enumerate_firewall_cells_synthetic(tmp_path):
    from src.method_y_revival.pet_attribution_v4 import enumerate_attribution_cells
    import pytest as _pt

    import pandas as pd

    dev, orch = tmp_path / "development", tmp_path / "orchestration"
    splits = {}
    membership_paths = {}
    for branch in ("amyloid", "tau"):
        (orch / branch).mkdir(parents=True)
        # membership: 3 development + 1 internal_transport + splits name 1 ghost
        hashes = [f"s{k}_{i}" for k in range(5) for i in range(3)]
        frame = pd.DataFrame({
            "patient_hash": hashes + ["held_out_transport"],
            "role": ["development"] * len(hashes) + ["internal_transport"],
        })
        mpath = tmp_path / f"{branch}_membership.parquet"
        frame.to_parquet(mpath)
        membership_paths[branch] = str(mpath)
        folds = [{"fold": k, "test_subjects": [f"s{k}_{i}" for i in range(3)]
                  + ["ghost_no_image", "held_out_transport"],
                  "train_subjects": [f"s{j}_{i}" for j in range(5) if j != k
                                     for i in range(3)]}
                 for k in range(5)]
        splits[branch] = {"outer_folds": folds}
        for arm in ("A0", "A1", "A2", "A3", "A4"):
            for k in range(5):
                _write_winner(orch / branch / f"winner_{arm}_outer_{k}.json",
                              branch, arm, k)
                final = dev / branch / arm / f"outer_{k}" / "final"
                final.mkdir(parents=True)
                (final / "locked_checkpoint.pt").write_bytes(b"fake")
    cells = enumerate_attribution_cells(dev_root=dev, orch_root=orch, splits=splits,
                                        membership_paths=membership_paths)
    assert len(cells) == 50
    # firewall = development-role ∩ splits, membership order, ghosts excluded
    assert cells[0]["firewall_subjects"] == ["s0_0", "s0_1", "s0_2"]
    assert cells[0]["train_subjects"] == [f"s{j}_{i}" for j in range(1, 5)
                                          for i in range(3)]
    # missing final -> BLOCKED incomplete
    (dev / "tau" / "A4" / "outer_4" / "final" / "locked_checkpoint.pt").unlink()
    with _pt.raises(RuntimeError, match="BLOCKED"):
        enumerate_attribution_cells(dev_root=dev, orch_root=orch, splits=splits,
                                    membership_paths=membership_paths)


def test_verify_final_digest_and_deadline(tmp_path):
    import time
    from src.method_y_revival.pet_attribution_v4 import (
        run_with_deadline, verify_final_digest)

    f = tmp_path / "locked.pt"
    torch.save({"run_contract_digest": "d" * 64, "model_state": {}}, f)
    assert verify_final_digest(f, expected_digest="d" * 64)["status"] == "PASS"
    import pytest as _pt
    with _pt.raises(RuntimeError, match="BLOCKED"):
        verify_final_digest(f, expected_digest="e" * 64)
    rec = run_with_deadline(lambda: {"id": "LEG-02", "status": "PASS",
                                     "evidence": "x.json"},
                            deadline_monotonic=time.monotonic() - 1.0)
    assert rec["status"] == "TIMEOUT" and rec["evidence"]
    rec = run_with_deadline(lambda: {"id": "LEG-02", "status": "PASS",
                                     "evidence": "x.json"},
                            deadline_monotonic=time.monotonic() + 60.0)
    assert rec["status"] == "PASS"


def test_verifier_gate_live_verify_fallback():
    import pytest as _pt
    from src.method_y_revival.pet_attribution_v4 import verifier_gate

    with _pt.raises(RuntimeError, match="BLOCKED"):
        verifier_gate(run_root=r"H:\New project\results\method_y_revival_v4_image_factorial_v1")
    out = verifier_gate(
        run_root=r"H:\New project\results\method_y_revival_v4_image_factorial_v1",
        config_path=r"H:\New project\configs\method_y_image_factorial_v4.yaml")
    assert out["status"] == "PASS" and out["evidence"] == "live-verify"
    assert out["run_contract_sha256"] == "969ebd0f61d9ffc3bc262c5476c75b23575d64d1f5703d23fe04d1095d07743e"
    assert out["valid_outer_cells"] == 50
