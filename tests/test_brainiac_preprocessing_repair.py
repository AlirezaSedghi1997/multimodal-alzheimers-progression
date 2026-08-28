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

    from src.research.brainiac_preprocessing_v2 import normalize_foreground

    out = normalize_foreground(volume, fg_mask)

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
    from src.research.brainiac_preprocessing_v2_schema import (
        THRESHOLD_MANIFEST_ARTIFACT_TYPE,
        canonical_sha,
        validate_threshold_manifest,
    )

    old_policy_sha = "f5c69a84" + "7" * 56
    current_policy_sha = "140acc4f" + "1" * 56
    samples = [f"s_{i:064x}" for i in range(32)]
    receipt_hashes = {sample: "b" * 64 for sample in samples}
    contract = {"policy_sha256": old_policy_sha}
    manifest = {
        "schema_version": 2,
        "artifact": THRESHOLD_MANIFEST_ARTIFACT_TYPE,
        "sample_count": 32,
        "sample_hashes": samples,
        "receipt_integrity_sha256s": receipt_hashes,
        "contract_bindings": contract,
        "pipeline_contract_sha256": canonical_sha(contract),
        "thresholds": {"foreground_fraction": {"low": 0.1, "high": 0.4}},
    }
    manifest["integrity_sha256"] = canonical_sha(manifest)
    with pytest.raises(RuntimeError, match="BINDING_MISMATCH"):
        validate_threshold_manifest(
            manifest,
            expected_sample_hashes=samples,
            expected_receipt_integrity_sha256s=receipt_hashes,
            expected_contract_bindings={"policy_sha256": current_policy_sha},
        )


# --- exit codes ----------------------------------------------------------------

@pytest.mark.parametrize("status,expected", [
    ("PASS", 0), ("COMPLETED", 0), ("ALREADY_COMPLETED", 0),
    ("PAUSED_RESUMABLE", 75),
    ("BLOCKED", 3), ("BLOCKED_BY_QUARANTINE", 3),
])
def test_exit_code_mapping(status, expected):
    from scripts.research.prepare_brainiac_raw_t1_v2 import _exit_code_for

    assert _exit_code_for(status) == expected


# --- exception sanitiser --------------------------------------------------------

def test_sanitiser_removes_raw_paths():
    from scripts.research.prepare_brainiac_raw_t1_v2 import _sanitize_exception_message

    raw_root = "H:\\A4 Learn"
    msg = f"FileNotFoundError: {raw_root}\\some_subject\\T1.nii.gz not found"
    sanitised = _sanitize_exception_message(RuntimeError(msg))
    assert raw_root not in sanitised
    assert "<raw-redacted>" in sanitised


# --- preflight deep checks -------------------------------------------------------

def test_preflight_insufficient_disk_returns_blocked():
    from scripts.research.prepare_brainiac_raw_t1_v2 import build_preflight_decision

    assert build_preflight_decision({"storage_free_space": False})["status"] == "BLOCKED"


def test_preflight_sufficient_disk_passes():
    from scripts.research.prepare_brainiac_raw_t1_v2 import build_preflight_decision

    assert build_preflight_decision({"storage_free_space": True})["status"] == "PASS"
