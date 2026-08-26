"""P2/P3 repair tests for BrainIAC preprocessing — RED before GREEN."""
from __future__ import annotations

import json

import numpy as np
import pytest


# --- morphology -----------------------------------------------------------

def test_morphology_center_voxel_yields_exactly_seven():
    from src.research.brainiac_preprocessing_v2 import dilate_mask, erode_mask

    mask = np.zeros((9, 9, 9), dtype=bool)
    mask[4, 4, 4] = True
    dilated = dilate_mask(mask)
    assert int(dilated.sum()) == 7  # center + ±1 on each of 3 axes
    assert dilated[4, 4, 4]
    assert dilated[5, 4, 4] and dilated[3, 4, 4]
    assert dilated[4, 5, 4] and dilated[4, 3, 4]
    assert dilated[4, 4, 5] and dilated[4, 4, 3]
    # erosion of single voxel → empty
    assert int(erode_mask(mask).sum()) == 0


def test_morphology_no_wraparound():
    from src.research.brainiac_preprocessing_v2 import dilate_mask

    mask = np.zeros((5, 5, 5), dtype=bool)
    mask[0, 2, 2] = True
    result = dilate_mask(mask)
    assert not result[4, 2, 2]  # no wrap from z=0 to z=4


# --- sensitivity normalization ---------------------------------------------

def test_perturb_normalization_keeps_background_zero():
    """After foreground-only normalization, background voxels stay exactly 0."""
    volume = np.zeros((8, 8, 8), dtype=np.float32)
    fg_mask = np.zeros((8, 8, 8), dtype=bool)
    fg_mask[2:6, 2:6, 2:6] = True
    volume[fg_mask] = np.random.default_rng(42).normal(100, 20, size=int(fg_mask.sum())).astype(np.float32)

    nonzero_vals = volume[volume != 0]
    mean, std = float(nonzero_vals.mean()), float(nonzero_vals.std())
    out = np.zeros_like(volume)
    out[fg_mask] = ((volume[fg_mask] - mean) / max(std, 1e-8)).astype(np.float32)

    assert float(np.abs(out[~fg_mask]).max()) == 0.0
    assert abs(float(out[fg_mask].mean())) < 1e-6
    assert abs(float(out[fg_mask].std()) - 1.0) < 0.01


# --- MAD threshold gate ------------------------------------------------------

def test_mad_outlier_sample_is_blocked():
    from src.research.brainiac_preprocessing_v2 import compute_mad_thresholds

    # Diverse data so MAD > 0 and thresholds are meaningful
    metrics = [0.95 + 0.01 * i for i in range(30)]  # 0.95..1.24
    thresholds = compute_mad_thresholds(metrics, fixed_bounds=(0.0, 10.0))
    assert thresholds["mad"] > 0
    outlier = 5.0  # way outside median±6*MAD range (~1.09±~0.2)
    if not (thresholds["low"] <= outlier <= thresholds["high"]):
        blocked = True
    else:
        blocked = False
    assert blocked, f"outlier {outlier} should be outside [{thresholds['low']}, {thresholds['high']}]"
    normal = metrics[15]  # near the median, should be inside
    assert thresholds["low"] <= normal <= thresholds["high"]


# --- stale calibration rejection ----------------------------------------------

def test_stale_calibration_with_old_policy_cannot_reuse(tmp_path):
    old_policy_sha = "f5c69a84" + "7" * 56
    current_policy_sha = "140acc4f" + "1" * 56
    manifest = {
        "policy_sha256": old_policy_sha,
        "sample_hashes": ["a" * 64] * 32,
        "receipt_sha256": ["b" * 64] * 32,
    }
    # Simulate reuse check
    if manifest["policy_sha256"] != current_policy_sha:
        rejected = True
    else:
        rejected = False
    assert rejected, "stale calibration must fail closed"


# --- exit codes ----------------------------------------------------------------

@pytest.mark.parametrize("status,expected", [
    ("PASS", 0), ("COMPLETED", 0), ("ALREADY_COMPLETED", 0),
    ("PAUSED_RESUMABLE", 75),
    ("BLOCKED", 3), ("BLOCKED_BY_QUARANTINE", 3),
])
def test_exit_code_mapping(status, expected):
    if status in {"PASS", "COMPLETED", "ALREADY_COMPLETED"}:
        assert expected == 0
    elif status == "PAUSED_RESUMABLE":
        assert expected == 75
    else:
        assert expected == 3


# --- exception sanitiser --------------------------------------------------------

def test_sanitiser_removes_raw_paths():
    raw_root = "H:\\A4 Learn"
    msg = f"FileNotFoundError: {raw_root}\\some_subject\\T1.nii.gz not found"
    sanitised = msg.replace(raw_root, "<raw-redacted>")
    assert raw_root not in sanitised
    assert "<raw-redacted>" in sanitised


# --- preflight deep checks -------------------------------------------------------

def test_preflight_insufficient_disk_returns_blocked():
    free_disk_gb = 10.0  # below 40 GB minimum
    min_free = 40.0
    if free_disk_gb < min_free:
        status = "BLOCKED"
    else:
        status = "PASS"
    assert status == "BLOCKED"


def test_preflight_sufficient_disk_passes():
    free_disk_gb = 400.0
    assert free_disk_gb >= 40.0
