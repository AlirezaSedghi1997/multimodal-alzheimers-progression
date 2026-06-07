from pathlib import Path
import sys
import json
import yaml
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.pet_spatial_qc import validate_single_source_forbidden_terms, compute_volume_qc, assess_qc_gates, centiloid_from_suvr, scrub_sensitive_text


def load_policy():
    return yaml.safe_load((ROOT / "configs/model/method_y_spatial_preprocess_policy.yaml").read_text(encoding="utf-8"))


def load_suvr_policy():
    return yaml.safe_load((ROOT / "configs/model/method_y_suvr_reference_policy.yaml").read_text(encoding="utf-8"))


def test_forbidden_terms_single_source_contains_pet_scalar_terms():
    terms = validate_single_source_forbidden_terms(load_policy())
    for t in ["pet", "suvr", "centiloid", "fbp", "ftp", "gap"]:
        assert t in terms


def test_qc_gates_fail_nan_and_pass_good_volume():
    p = load_policy()
    bad = np.zeros((8, 8, 8), dtype=np.float32)
    bad[0,0,0] = np.nan
    qc_bad = compute_volume_qc(bad, np.eye(4))
    assert assess_qc_gates(qc_bad, p)["status"] == "FAIL"
    good = np.random.default_rng(1).normal(1, 0.1, size=(8,8,8)).astype("float32")
    qc_good = compute_volume_qc(good, np.eye(4))
    assert assess_qc_gates(qc_good, p)["status"] == "PASS"


def test_centiloid_equations_available_for_fbp_fbb():
    p = load_suvr_policy()
    assert round(centiloid_from_suvr(1.11, "fbp", p), 2) == round(188.22*1.11-189.16, 2)
    assert round(centiloid_from_suvr(1.08, "fbb", p), 2) == round(157.15*1.08-151.87, 2)


def test_scrub_sensitive_text_removes_ids_and_paths():
    text = "Cannot read /mnt/h/A4 Learn/B12345/file.nii.gz PREVBID"
    clean = scrub_sensitive_text(text)
    assert "B12345" not in clean
    assert "PREVBID" not in clean
    assert "/mnt/h" not in clean
