from __future__ import annotations

import h5py
import nibabel as nib
import numpy as np
import pandas as pd
import pytest
import yaml
from pathlib import Path


def _reference(tmp_path):
    path = tmp_path / "reference.nii.gz"
    nib.save(nib.Nifti1Image(np.ones((8, 8, 8), dtype=np.float32), np.eye(4)), path)
    return path


def _source_h5(tmp_path):
    path = tmp_path / "source.h5"
    with h5py.File(path, "w") as handle:
        grp = handle.create_group("sample_1")
        grp.create_dataset("image", data=np.arange(512, dtype=np.float32).reshape(1, 8, 8, 8))
        grp.attrs["status"] = "COMPLETE"
        grp.attrs["source_sha256"] = "a" * 64
    return path


def test_launcher_rejects_source_drift(tmp_path):
    from scripts.build_method_y_image_cache_v3 import run
    import json

    # Minimal config
    reference = _reference(tmp_path)
    source = _source_h5(tmp_path)
    # Create dummy readiness
    output_root = tmp_path / "v1"
    (output_root / "private").mkdir(parents=True)
    (output_root / "readiness").mkdir(parents=True)
    # Create membership parquet
    membership = pd.DataFrame(
        [
            {
                "patient_hash": "pt_1",
                "image_key": "sample_1",
                "image_storage_kind": "HDF5_GROUP",
                "image_path_private": str(source),
                "h5_group": "sample_1",
                "source_image_sha256": "a" * 64,
                "qc_status": "PASS",
            }
        ]
    )
    membership_path = output_root / "private" / "amyloid_image_membership.parquet"
    membership.to_parquet(membership_path)
    import hashlib

    def sha256(p):
        d = hashlib.sha256()
        with p.open("rb") as fh:
            for chunk in iter(lambda: fh.read(8192), b""):
                d.update(chunk)
        return d.hexdigest()

    import hashlib as _hashlib

    def _file_sha(p):
        d = _hashlib.sha256()
        with p.open("rb") as fh:
            for chunk in iter(lambda: fh.read(8192), b""):
                d.update(chunk)
        return d.hexdigest()

    run_contract_path = output_root / "run_contract.json"
    run_contract_path.write_text(json.dumps({"run_contract": {}}), encoding="utf-8")
    run_contract_sha = _file_sha(run_contract_path)
    readiness = {
        "training_execution_allowed": True,
        "private_membership_sha256": {"amyloid": sha256(membership_path), "tau": sha256(membership_path)},
        "run_contract_sha256": run_contract_sha,
        "source_artifact_sha256": {"amyloid": sha256(source)},
    }
    (output_root / "readiness" / "FINAL_IMAGE_TRAINING_READINESS.json").write_text(
        json.dumps(readiness), encoding="utf-8"
    )
    config = {
        "output_root": str(output_root),
        "shared": {
            "pattern_normalization": {
                "lower_percentile": 1.0,
                "upper_percentile": 99.0,
                "output_clip": [-5.0, 5.0],
            },
            "reference_template": str(reference),
        },
        "branches": {
            "amyloid": {"derived_cache": str(tmp_path / "amyloid.h5"), "source_cache": str(source)},
            "tau": {"derived_cache": str(tmp_path / "tau.h5"), "source_cache": str(source)},
        },
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    # Should pass with correct source
    summary = run(config_path, ("amyloid",))
    assert summary["status"] == "COMPLETED"
    # Now modify source to cause drift
    with h5py.File(source, "a") as handle:
        handle["sample_1"].attrs["source_sha256"] = "b" * 64
    with pytest.raises(Exception, match="SOURCE_ARTIFACT_CHANGED_SINCE_READINESS|HDF5_GROUP_SOURCE_SHA_MISMATCH"):
        run(config_path, ("amyloid",))
