from __future__ import annotations

from pathlib import Path

import pytest


def test_cache_matches_resume_requires_matching_shape(tmp_path):
    pytest.importorskip("h5py")
    import h5py
    import numpy as np

    from scripts.build_mri_volume_cache import cache_matches_resume

    path = tmp_path / "cache.h5"
    with h5py.File(path, "w") as h5:
        h5.attrs["resolution"] = 4
        h5.attrs["source_mode"] = "preprocessed_mri"
        h5.create_dataset("image", data=np.zeros((2, 1, 4, 4, 4), dtype="float16"))
        h5.create_dataset("label", data=np.zeros((2,), dtype=np.int8))
        h5.create_dataset("sample_hash", data=np.array(["a", "b"], dtype=object))
        h5.create_dataset("patient_hash", data=np.array(["p1", "p2"], dtype=object))
        h5.create_dataset("tabular", data=np.zeros((2, 5), dtype=np.float32))
        h5.create_dataset("status", data=np.array(["OK", "OK"], dtype=object))

    assert cache_matches_resume(path, rows=2, resolution=4, source_mode="preprocessed_mri")
    assert not cache_matches_resume(path, rows=3, resolution=4, source_mode="preprocessed_mri")


def test_pet_pending_skips_terminal_failures_without_retry():
    from scripts.preprocess_pet_spatial import _pending_work_items
    from src.utils.pet_resume import PetResumeManifest

    class Item:
        catalog = type("C", (), {"sample_hash": "sample_a", "patient_hash": "patient_a"})()
        candidate = type("Cand", (), {"file_hash": "file_a"})()

    class NewItem:
        catalog = type("C", (), {"sample_hash": "sample_c", "patient_hash": "patient_c"})()
        candidate = type("Cand", (), {"file_hash": "file_c"})()

    resume = PetResumeManifest(Path("unused.json"), stage="method_y_pet_preprocess", config={"spatial_mode": "spm_pet_t1_mni"})
    resume.data["items"] = {
        "sample_a_file_a_fbp": {"status": "FAIL"},
        "sample_b_file_b_fbp": {"status": "PASS"},
    }
    work = [
        Item(),
        type(
            "ItemB",
            (),
            {
                "catalog": type("C", (), {"sample_hash": "sample_b", "patient_hash": "patient_b"})(),
                "candidate": type("Cand", (), {"file_hash": "file_b"})(),
            },
        )(),
        NewItem(),
    ]
    pending = _pending_work_items(work, resume, "fbp", overwrite=False, retry_failed=False)
    assert [item.catalog.sample_hash for item in pending] == ["sample_c"]

    retry_pending = _pending_work_items(work, resume, "fbp", overwrite=False, retry_failed=True)
    assert {item.catalog.sample_hash for item in retry_pending} == {"sample_a", "sample_c"}
