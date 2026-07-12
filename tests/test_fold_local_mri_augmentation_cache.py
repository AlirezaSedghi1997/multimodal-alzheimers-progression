from pathlib import Path

import numpy as np
import pytest


def _write_split(path: Path, rows: list[tuple[str, int]]) -> None:
    path.write_text(
        "sample_hash,ab_label\n" + "".join(f"{sample_hash},{label}\n" for sample_hash, label in rows),
        encoding="utf-8",
    )


def test_materialized_cache_is_qc_gated_fold_local_and_lineage_complete(tmp_path: Path):
    pytest.importorskip("h5py")
    import h5py

    from scripts.build_fold_local_mri_augmented_cache import build_materialized_cache

    real_cache = tmp_path / "real.h5"
    image = np.zeros((4, 1, 128, 128, 128), dtype=np.float16)
    image[:, :, 20:108, 20:108, 20:108] = 1.0
    with h5py.File(real_cache, "w") as h5:
        h5.create_dataset("image", data=image, compression="gzip")
        h5.create_dataset("label", data=np.array([0, 1, 0, 1], dtype=np.int8))
        h5.create_dataset("tabular", data=np.zeros((4, 5), dtype=np.float32))
        h5.create_dataset("patient_hash", data=[b"p_a", b"p_b", b"p_c", b"p_d"])
        h5.create_dataset("sample_hash", data=[b"real_a", b"real_b", b"real_c", b"real_d"])
        h5.create_dataset("status", data=[b"OK", b"OK", b"OK", b"OK"])
        h5.attrs["resolution"] = 128
        h5.attrs["contains_pet_input"] = False

    splits = tmp_path / "splits"
    splits.mkdir()
    _write_split(splits / "fold_0_train.csv", [("real_a", 0), ("real_b", 1)])
    _write_split(splits / "fold_0_test.csv", [("real_c", 0), ("real_d", 1)])
    _write_split(splits / "fold_1_train.csv", [("real_c", 0), ("real_d", 1)])
    _write_split(splits / "fold_1_test.csv", [("real_a", 0), ("real_b", 1)])

    combined_cache = tmp_path / "combined.h5"
    summary = build_materialized_cache(
        real_cache=real_cache,
        split_dir=splits,
        out_path=combined_cache,
        qc_dir=None,
        requested_folds=2,
        training_seed=13,
        inner_val_fraction=0.5,
        augments_per_fit_real=1,
    )

    assert summary["status"] == "VALIDATED_STAGED"
    assert summary["eligible_real_rows"] == 4
    assert summary["materialized_augmented_rows"] == 2
    assert summary["total_rows"] == 6
    with h5py.File(combined_cache, "r") as h5:
        assert list(h5["image"].shape) == [6, 1, 128, 128, 128]
        assert all(bool(value) for value in h5["source_train_only"][4:])
        assert all(bool(value) for value in h5["is_synthetic"][4:])
        assert {int(value) for value in h5["source_fold"][4:]} == {0, 1}
        parents = {value.decode() for value in h5["source_sample_hash"][4:]}
        assert parents <= {"real_a", "real_b", "real_c", "real_d"}
        assert np.all(np.isfinite(h5["image"][:]))
        assert not bool(np.all(h5["image"][4:] == 0.0))
        assert float(np.mean(h5["image"][4:] == 0.0)) > 0.50


def test_full_field_zscore_augmentation_avoids_geometric_and_additive_noise():
    from scripts.build_fold_local_mri_augmented_cache import augment_image

    grid = np.linspace(-1.0, 1.0, num=128**3, dtype=np.float32).reshape(1, 128, 128, 128)
    output, parameters = augment_image(grid, np.random.default_rng(17))

    assert parameters["mask_mode"] == "full_field_no_explicit_background"
    assert parameters["affine_applied"] is False
    assert parameters["noise_std"] == 0.0
    assert parameters["offset_zscore"] == 0.0
    assert float(np.mean(output == 0.0)) < 0.01
    assert np.all(np.isfinite(output))
