from pathlib import Path

import numpy as np
import pytest


def test_gpu_preflight_accepts_combined_supplementary_cache(tmp_path: Path):
    pytest.importorskip("h5py")
    import h5py

    from scripts.gpu_preflight_check import inspect_cache

    cache = tmp_path / "combined.h5"
    with h5py.File(cache, "w") as h5:
        h5.create_dataset("image", data=np.zeros((3, 1, 4, 4, 4), dtype=np.float16))
        h5.create_dataset("label", data=np.array([0, 1, 1], dtype=np.int8))
        h5.create_dataset("tabular", data=np.zeros((3, 2), dtype=np.float32))
        h5.create_dataset("sample_hash", data=[b"real_a", b"real_b", b"syn_global"])
        h5.create_dataset("status", data=[b"OK", b"OK", b"OK"])
        h5.create_dataset("source", data=[b"real", b"real", b"simulated_augment"])
        h5.create_dataset("source_fold", data=np.array([-1, -1, -1], dtype=np.int16))
        h5.create_dataset("source_train_only", data=np.array([False, False, False], dtype=bool))

    info = inspect_cache(cache, {"real_a", "real_b"})
    assert info["ok"] is True
    assert info["mode"] == "supplementary_combined"
    assert info["overlap"] == 2
    assert info["real_n"] == 2
    assert info["catalog_n"] == 2


def test_gpu_preflight_accepts_qc_gated_real_subset_in_combined_cache(tmp_path: Path):
    pytest.importorskip("h5py")
    import h5py

    from scripts.gpu_preflight_check import inspect_cache

    cache = tmp_path / "qc_gated.h5"
    with h5py.File(cache, "w") as h5:
        h5.create_dataset("image", data=np.zeros((2, 1, 4, 4, 4), dtype=np.float16))
        h5.create_dataset("label", data=np.array([0, 1], dtype=np.int8))
        h5.create_dataset("tabular", data=np.zeros((2, 2), dtype=np.float32))
        h5.create_dataset("sample_hash", data=[b"real_a", b"aug_a"])
        h5.create_dataset("status", data=[b"OK", b"OK"])
        h5.create_dataset("source", data=[b"real", b"fold_local_augmentation"])
        h5.create_dataset("source_fold", data=np.array([-1, 0], dtype=np.int16))
        h5.create_dataset("source_train_only", data=np.array([False, True], dtype=bool))

    info = inspect_cache(cache, {"real_a", "qc_excluded"})
    assert info["ok"] is True
    assert info["real_n"] == 1
    assert info["overlap"] == 1
    assert info["catalog_n"] == 2


def test_gpu_preflight_rejects_cache_without_identity_or_supplementary_metadata(tmp_path: Path):
    pytest.importorskip("h5py")
    import h5py

    from scripts.gpu_preflight_check import inspect_cache

    cache = tmp_path / "invalid.h5"
    with h5py.File(cache, "w") as h5:
        h5.create_dataset("image", data=np.zeros((1, 1, 4, 4, 4), dtype=np.float16))
        h5.create_dataset("label", data=np.array([0], dtype=np.int8))
        h5.create_dataset("tabular", data=np.zeros((1, 2), dtype=np.float32))
        h5.create_dataset("sample_hash", data=[b"real_a"])
        h5.create_dataset("status", data=[b"OK"])

    info = inspect_cache(cache, {"real_a"})
    assert info["ok"] is False
    assert info["missing"] == ["patient_hash_or_supplementary_metadata"]
