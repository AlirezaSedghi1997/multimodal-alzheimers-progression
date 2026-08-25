from __future__ import annotations

import h5py
import nibabel as nib
import numpy as np
import pandas as pd
import pytest


def _reference(tmp_path):
    path = tmp_path / "reference.nii.gz"
    nib.save(
        nib.Nifti1Image(np.ones((8, 8, 8), dtype=np.float32), np.eye(4)),
        path,
    )
    return path


def _source_cache(tmp_path):
    path = tmp_path / "source.h5"
    with h5py.File(path, "w") as handle:
        group = handle.create_group("sample_1")
        group.create_dataset(
            "image", data=np.arange(512, dtype=np.float32).reshape(1, 8, 8, 8)
        )
        group.attrs["status"] = "COMPLETE"
        group.attrs["source_sha256"] = "a" * 64
    return path


def _membership(source):
    return pd.DataFrame(
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


def test_pattern_cache_commits_per_subject_and_resumes(tmp_path):
    from src.method_y_revival.image_cache_v3 import build_pattern_cache

    source = _source_cache(tmp_path)
    membership = _membership(source)
    output = tmp_path / "pattern.h5"
    state = tmp_path / "state.json"

    first = build_pattern_cache(
        membership=membership,
        reference_template=_reference(tmp_path),
        output_cache=output,
        state_path=state,
        normalization={
            "lower_percentile": 1.0,
            "upper_percentile": 99.0,
            "output_clip": [-5.0, 5.0],
        },
        branch="amyloid",
    )
    second = build_pattern_cache(
        membership=membership,
        reference_template=tmp_path / "reference.nii.gz",
        output_cache=output,
        state_path=state,
        normalization={
            "lower_percentile": 1.0,
            "upper_percentile": 99.0,
            "output_clip": [-5.0, 5.0],
        },
        branch="amyloid",
    )

    assert first["status"] == "COMPLETED"
    assert first["newly_completed_count"] == 1
    assert second["status"] == "COMPLETED"
    assert second["resumed_existing_count"] == 1
    with h5py.File(output, "r") as handle:
        assert handle["pt_1/image"].shape == (1, 8, 8, 8)
        assert np.isfinite(handle["pt_1/image"][:]).all()


def test_pattern_cache_contract_mismatch_fails_closed(tmp_path):
    from src.method_y_revival.image_cache_v3 import (
        ImageCacheV3Error,
        build_pattern_cache,
    )

    source = _source_cache(tmp_path)
    membership = _membership(source)
    reference = _reference(tmp_path)
    output = tmp_path / "pattern.h5"
    state = tmp_path / "state.json"
    build_pattern_cache(
        membership=membership,
        reference_template=reference,
        output_cache=output,
        state_path=state,
        normalization={
            "lower_percentile": 1.0,
            "upper_percentile": 99.0,
            "output_clip": [-5.0, 5.0],
        },
        branch="amyloid",
    )

    with pytest.raises(ImageCacheV3Error, match="PATTERN_CACHE_CONTRACT_MISMATCH"):
        build_pattern_cache(
            membership=membership,
            reference_template=reference,
            output_cache=output,
            state_path=state,
            normalization={
                "lower_percentile": 2.0,
                "upper_percentile": 99.0,
                "output_clip": [-5.0, 5.0],
            },
            branch="amyloid",
        )


def test_hdf5_group_source_sha_must_match_membership(tmp_path):
    from src.method_y_revival.image_cache_v3 import ImageCacheV3Error, build_pattern_cache

    source = tmp_path / "source.h5"
    with h5py.File(source, "w") as handle:
        group = handle.create_group("sample_1")
        group.create_dataset("image", data=np.arange(512, dtype=np.float32).reshape(1, 8, 8, 8))
        group.attrs["status"] = "COMPLETE"
        group.attrs["source_sha256"] = "b" * 64
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
    with pytest.raises(ImageCacheV3Error, match="HDF5_GROUP_SOURCE_SHA_MISMATCH"):
        build_pattern_cache(
            membership=membership,
            reference_template=_reference(tmp_path),
            output_cache=tmp_path / "pattern.h5",
            state_path=tmp_path / "state.json",
            normalization={"lower_percentile": 1.0, "upper_percentile": 99.0, "output_clip": [-5.0, 5.0]},
            branch="amyloid",
        )


def test_source_cache_file_hash_change_breaks_cache_contract(tmp_path):
    from src.method_y_revival.image_cache_v3 import build_pattern_cache

    source = _source_cache(tmp_path)
    membership = _membership(source)
    reference = _reference(tmp_path)
    output = tmp_path / "pattern.h5"
    state = tmp_path / "state.json"
    first = build_pattern_cache(
        membership=membership,
        reference_template=reference,
        output_cache=output,
        state_path=state,
        normalization={"lower_percentile": 1.0, "upper_percentile": 99.0, "output_clip": [-5.0, 5.0]},
        branch="amyloid",
        source_artifact_sha256="a" * 64,
    )
    assert first["status"] == "COMPLETED"
    # Changing source artifact hash must create a new contract, not silently reuse
    second = build_pattern_cache(
        membership=membership,
        reference_template=reference,
        output_cache=tmp_path / "pattern2.h5",
        state_path=tmp_path / "state2.json",
        normalization={"lower_percentile": 1.0, "upper_percentile": 99.0, "output_clip": [-5.0, 5.0]},
        branch="amyloid",
        source_artifact_sha256="b" * 64,
    )
    assert second["run_contract_sha256"] != first["run_contract_sha256"]


def test_readiness_contract_hash_is_part_of_cache_contract(tmp_path):
    from src.method_y_revival.image_cache_v3 import build_pattern_cache

    source = _source_cache(tmp_path)
    membership = _membership(source)
    reference = _reference(tmp_path)
    first = build_pattern_cache(
        membership=membership,
        reference_template=reference,
        output_cache=tmp_path / "pattern.h5",
        state_path=tmp_path / "state.json",
        normalization={"lower_percentile": 1.0, "upper_percentile": 99.0, "output_clip": [-5.0, 5.0]},
        branch="amyloid",
        readiness_run_contract_sha256="a" * 64,
    )
    second = build_pattern_cache(
        membership=membership,
        reference_template=reference,
        output_cache=tmp_path / "pattern2.h5",
        state_path=tmp_path / "state2.json",
        normalization={"lower_percentile": 1.0, "upper_percentile": 99.0, "output_clip": [-5.0, 5.0]},
        branch="amyloid",
        readiness_run_contract_sha256="b" * 64,
    )
    assert second["run_contract_sha256"] != first["run_contract_sha256"]


def test_completed_cache_rejects_membership_or_source_drift(tmp_path):
    from src.method_y_revival.image_cache_v3 import ImageCacheV3Error, build_pattern_cache

    source = _source_cache(tmp_path)
    membership = _membership(source)
    reference = _reference(tmp_path)
    output = tmp_path / "pattern.h5"
    state = tmp_path / "state.json"
    build_pattern_cache(
        membership=membership,
        reference_template=reference,
        output_cache=output,
        state_path=state,
        normalization={"lower_percentile": 1.0, "upper_percentile": 99.0, "output_clip": [-5.0, 5.0]},
        branch="amyloid",
    )
    # Drift: different patient hash must not be silently accepted as resume
    drifted = membership.copy()
    drifted.loc[0, "patient_hash"] = "pt_2"
    with pytest.raises(ImageCacheV3Error, match="PATTERN_CACHE_CONTRACT_MISMATCH"):
        build_pattern_cache(
            membership=drifted,
            reference_template=reference,
            output_cache=output,
            state_path=state,
            normalization={"lower_percentile": 1.0, "upper_percentile": 99.0, "output_clip": [-5.0, 5.0]},
            branch="amyloid",
        )


def test_cache_group_must_be_complete_and_finite(tmp_path):
    from src.method_y_revival.image_cache_v3 import build_pattern_cache
    import h5py

    source = _source_cache(tmp_path)
    membership = _membership(source)
    reference = _reference(tmp_path)
    output = tmp_path / "pattern.h5"
    state = tmp_path / "state.json"
    build_pattern_cache(
        membership=membership,
        reference_template=reference,
        output_cache=output,
        state_path=state,
        normalization={"lower_percentile": 1.0, "upper_percentile": 99.0, "output_clip": [-5.0, 5.0]},
        branch="amyloid",
    )
    with h5py.File(output, "r") as handle:
        assert handle["pt_1"].attrs["status"] == "COMPLETE"
        assert handle["pt_1"].attrs["finite_fraction"] == 1.0
        assert handle["pt_1/image"].dtype == np.float16
        assert np.isfinite(handle["pt_1/image"][:]).all()


def test_float16_roundtrip_error_is_below_locked_tolerance(tmp_path):
    from src.method_y_revival.image_survival_v3 import robust_pattern_normalize
    import nibabel as nib

    reference = nib.load(_reference(tmp_path))
    ref_data = np.asarray(reference.dataobj, dtype=np.float32)
    brain_mask = np.isfinite(ref_data) & (ref_data > 0)
    rng = np.random.default_rng(0)
    volume = rng.normal(size=(8, 8, 8)).astype(np.float32)
    normalized = robust_pattern_normalize(
        volume, brain_mask, lower_percentile=1.0, upper_percentile=99.0, clip=[-5.0, 5.0]
    )
    roundtrip = normalized.astype(np.float16).astype(np.float32)
    max_abs_error = float(np.max(np.abs(normalized - roundtrip)))
    assert max_abs_error <= 0.005
    assert np.isfinite(roundtrip).all()



def test_cache_group_tracer_mismatch_fails_closed(tmp_path):
    from src.method_y_revival.image_cache_v3 import ImageCacheV3Error, build_pattern_cache
    import h5py

    source = _source_cache(tmp_path)
    membership = _membership(source)
    reference = _reference(tmp_path)
    output = tmp_path / "pattern.h5"
    state = tmp_path / "state.json"
    first = build_pattern_cache(
        membership=membership,
        reference_template=reference,
        output_cache=output,
        state_path=state,
        normalization={"lower_percentile": 1.0, "upper_percentile": 99.0, "output_clip": [-5.0, 5.0]},
        branch="amyloid",
        expected_tracer="FBP",
    )
    assert first["status"] == "COMPLETED"
    with h5py.File(output, "r") as handle:
        assert handle["pt_1"].attrs["tracer"] == "FBP"
    # Same cache rebuilt under a different tracer must fail closed.
    with pytest.raises(ImageCacheV3Error, match="CACHE_GROUP_TRACER_MISMATCH"):
        build_pattern_cache(
            membership=membership,
            reference_template=reference,
            output_cache=output,
            state_path=state,
            normalization={"lower_percentile": 1.0, "upper_percentile": 99.0, "output_clip": [-5.0, 5.0]},
            branch="amyloid",
            expected_tracer="FTP",
        )
