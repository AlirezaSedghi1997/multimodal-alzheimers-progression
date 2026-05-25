from pathlib import Path

import pytest


def test_hdf5_cache_contains_no_pet_like_keys():
    h5_files = list(Path("data/cached").glob("*.h5")) + list(Path("data/cached").glob("*.hdf5"))
    if not h5_files:
        pytest.skip("No HDF5 cache generated yet.")
    import h5py

    forbidden = ("pet", "suvr", "amyloid", "tau")
    for fp in h5_files:
        with h5py.File(fp, "r") as h5:
            keys = []
            h5.visit(keys.append)
        bad = [k for k in keys if any(term in k.lower() for term in forbidden)]
        assert not bad, f"{fp} contains forbidden PET-like keys: {bad}"

