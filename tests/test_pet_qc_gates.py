import numpy as np
from src.utils.pet_spatial_qc import compute_volume_qc, assess_qc_gates, normalize_pet_array, scrub_sensitive_text


def test_qc_gate_passes_good_volume():
    arr = np.random.default_rng(42).normal(size=(8, 8, 8)).astype("float32")
    qc = compute_volume_qc(arr, np.eye(4))
    policy = {"qc_gates": {"finite_fraction_min": 0.95, "foreground_fraction_min": 0.01, "std_min": 1e-6}}
    gate = assess_qc_gates(qc, policy)
    assert gate["status"] == "PASS"


def test_qc_gate_fails_nan_volume():
    arr = np.full((8,8,8), np.nan, dtype="float32")
    qc = compute_volume_qc(arr, np.eye(4))
    policy = {"qc_gates": {"finite_fraction_min": 0.95, "foreground_fraction_min": 0.01, "std_min": 1e-6}}
    gate = assess_qc_gates(qc, policy)
    assert gate["status"] == "FAIL"


def test_normalization_uses_foreground():
    arr = np.zeros((8,8,8), dtype="float32")
    arr[2:6,2:6,2:6] = np.random.default_rng(0).normal(loc=10, scale=2, size=(4,4,4))
    out, info = normalize_pet_array(arr)
    assert out.shape == arr.shape
    assert info["foreground_voxels"] > 0


def test_scrub_sensitive_text_removes_paths_and_ids():
    msg = "Cannot read /mnt/h/A4 Learn/B12345/pet.nii.gz PREVBID"
    clean = scrub_sensitive_text(msg)
    assert "B12345" not in clean
    assert "/mnt/h" not in clean
