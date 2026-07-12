from pathlib import Path

import numpy as np
import pytest


def test_supplementary_samples_merge_into_train_only(tmp_path):
    pytest.importorskip("h5py")
    import h5py

    from src.data.combined_mri_cache import (
        is_supplementary_sample,
        merge_train_with_supplementary,
        supplementary_train_indices,
    )

    cache = tmp_path / "combined.h5"
    with h5py.File(cache, "w") as h5:
        h5.create_dataset("sample_hash", data=[b"real_a", b"syn_ddpm_1", b"syn_sim_2"])
        h5.create_dataset("status", data=[b"OK", b"OK", b"OK"])
        h5.create_dataset("source", data=[b"real", b"synthetic_ddpm", b"simulated_augment"])

    with h5py.File(cache, "r") as h5:
        hashes = [v.decode() for v in h5["sample_hash"][:]]
        statuses = [v.decode() for v in h5["status"][:]]
        sources = [v.decode() for v in h5["source"][:]]

    assert is_supplementary_sample("syn_ddpm_1")
    assert not is_supplementary_sample("real_a")
    supp = supplementary_train_indices(hashes, statuses, sources)
    assert supp == [1, 2]
    merged = merge_train_with_supplementary([0], supp)
    assert merged == [0, 1, 2]


def test_resolve_train_indices_adds_supplementary(tmp_path):
    pytest.importorskip("h5py")
    import csv
    import h5py

    from scripts.train_3d_mri import cache_index, ok_indices, resolve_train_indices

    cache = tmp_path / "cache.h5"
    with h5py.File(cache, "w") as h5:
        h5.create_dataset("image", data=np.zeros((3, 1, 4, 4, 4), dtype=np.float16))
        h5.create_dataset("label", data=np.array([0, 1, 1], dtype=np.int8))
        h5.create_dataset("tabular", data=np.zeros((3, 5), dtype=np.float32))
        h5.create_dataset("sample_hash", data=[b"real_a", b"syn_ddpm_0", b"syn_sim_0"])
        h5.create_dataset("status", data=[b"OK", b"OK", b"OK"])
        h5.create_dataset("source", data=[b"real", b"synthetic_ddpm", b"simulated_augment"])
        h5.create_dataset("source_sample_hash", data=[b"", b"real_a", b"real_a"])
        h5.create_dataset("source_fold", data=np.array([-1, 0, 0], dtype=np.int16))
        h5.create_dataset("source_train_only", data=np.array([False, True, True], dtype=bool))

    split_dir = tmp_path / "splits"
    split_dir.mkdir()
    with (split_dir / "fold_0_train.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample_hash", "ab_label"])
        writer.writeheader()
        writer.writerow({"sample_hash": "real_a", "ab_label": "0"})

    sample_to_idx = cache_index(cache)
    ok = ok_indices(cache)
    train_idx, train_y = resolve_train_indices(
        split_dir / "fold_0_train.csv",
        sample_to_idx,
        ok,
        cache_path=cache,
        include_supplementary=True,
    )
    assert train_idx == [0, 1, 2]
    assert train_y == [0, 1, 1]

    test_idx, test_y = resolve_train_indices(
        split_dir / "fold_0_train.csv",
        sample_to_idx,
        ok,
        cache_path=cache,
        include_supplementary=False,
    )
    assert test_idx == [0]
    assert test_y == [0]


def test_resolve_train_indices_excludes_wrong_fold_supplementary(tmp_path):
    pytest.importorskip("h5py")
    import csv
    import h5py

    from scripts.train_3d_mri import cache_index, ok_indices, resolve_train_indices

    cache = tmp_path / "cache.h5"
    with h5py.File(cache, "w") as h5:
        h5.create_dataset("image", data=np.zeros((4, 1, 4, 4, 4), dtype=np.float16))
        h5.create_dataset("label", data=np.array([0, 1, 1, 1], dtype=np.int8))
        h5.create_dataset("tabular", data=np.zeros((4, 5), dtype=np.float32))
        h5.create_dataset("sample_hash", data=[b"real_a", b"syn_ddpm_f0", b"syn_ddpm_f1", b"syn_ddpm_global"])
        h5.create_dataset("status", data=[b"OK", b"OK", b"OK", b"OK"])
        h5.create_dataset("source", data=[b"real", b"synthetic_ddpm", b"synthetic_ddpm", b"synthetic_ddpm"])
        h5.create_dataset("source_sample_hash", data=[b"", b"real_a", b"real_a", b"real_a"])
        h5.create_dataset("source_fold", data=np.array([-1, 0, 1, -1], dtype=np.int16))
        h5.create_dataset("source_train_only", data=np.array([False, True, True, False], dtype=bool))

    split_dir = tmp_path / "splits"
    split_dir.mkdir()
    with (split_dir / "fold_0_train.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample_hash", "ab_label"])
        writer.writeheader()
        writer.writerow({"sample_hash": "real_a", "ab_label": "0"})

    train_idx, train_y = resolve_train_indices(
        split_dir / "fold_0_train.csv",
        cache_index(cache),
        ok_indices(cache),
        cache_path=cache,
        include_supplementary=True,
    )
    assert train_idx == [0, 1]
    assert train_y == [0, 1]


def test_global_supplementary_requires_explicit_flag(tmp_path):
    pytest.importorskip("h5py")
    import csv
    import h5py

    from scripts.train_3d_mri import cache_index, ok_indices, resolve_train_indices

    cache = tmp_path / "cache.h5"
    with h5py.File(cache, "w") as h5:
        h5.create_dataset("image", data=np.zeros((2, 1, 4, 4, 4), dtype=np.float16))
        h5.create_dataset("label", data=np.array([0, 1], dtype=np.int8))
        h5.create_dataset("tabular", data=np.zeros((2, 5), dtype=np.float32))
        h5.create_dataset("sample_hash", data=[b"real_a", b"syn_global"])
        h5.create_dataset("status", data=[b"OK", b"OK"])
        h5.create_dataset("source", data=[b"real", b"simulated_augment"])
        h5.create_dataset("source_sample_hash", data=[b"", b"real_a"])
        h5.create_dataset("source_fold", data=np.array([-1, -1], dtype=np.int16))
        h5.create_dataset("source_train_only", data=np.array([False, False], dtype=bool))

    split_dir = tmp_path / "splits"
    split_dir.mkdir()
    with (split_dir / "fold_0_train.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample_hash", "ab_label"])
        writer.writeheader()
        writer.writerow({"sample_hash": "real_a", "ab_label": "0"})

    common = {
        "split_path": split_dir / "fold_0_train.csv",
        "sample_to_idx": cache_index(cache),
        "ok": ok_indices(cache),
        "cache_path": cache,
        "include_supplementary": True,
    }
    train_idx, train_y = resolve_train_indices(**common)
    assert train_idx == [0]
    assert train_y == [0]

    train_idx, train_y = resolve_train_indices(**common, include_global_supplementary=True)
    assert train_idx == [0, 1]
    assert train_y == [0, 1]


def test_cache_has_supplementary_decodes_hdf5_bytes(tmp_path):
    pytest.importorskip("h5py")
    import h5py

    from src.data.combined_mri_cache import cache_has_supplementary, count_by_source

    cache = tmp_path / "cache.h5"
    with h5py.File(cache, "w") as h5:
        h5.create_dataset("sample_hash", data=[b"real_a", b"syn_global"])
        h5.create_dataset("status", data=[b"OK", b"OK"])
        h5.create_dataset("source", data=[b"real", b"simulated_augment"])

    assert cache_has_supplementary(cache) is True
    assert count_by_source(cache) == {"real": 1, "synthetic_ddpm": 0, "simulated_augment": 1}


def test_hyperparameter_search_global_supplementary_is_opt_in(tmp_path):
    from scripts.run_3d_hyperparameter_search import (
        DEFAULT_CANDIDATES,
        PROJECT_ROOT,
        build_parser,
        candidate_command,
    )

    candidate_names = {row["name"] for row in DEFAULT_CANDIDATES}
    assert "resnet18_healnet_auc_ema_lr5e5_drop15" in candidate_names
    assert "mednext_s_healnet_focal_q2phenotype_lr5e5_drop20" in candidate_names
    assert any(row["task_loss"] == "bce_auc" for row in DEFAULT_CANDIDATES)
    assert any(row["moe_load_balance_lambda"] > 0 for row in DEFAULT_CANDIDATES if row["fusion"] == "moe")

    parser = build_parser()
    args = parser.parse_args([])
    assert args.include_global_supplementary_train is False

    args.combined_cache = "data/cached/t1_combined_128.h5"
    args.folds = 4
    args.epochs = 40
    args.batch_size = 1
    args.grad_accum = 4
    args.pretrained_checkpoint = ""
    args.max_runtime_minutes = 0
    args.limit = 0
    trial_dir = PROJECT_ROOT / "results" / "final_results" / "_unit_trial"
    cmd = candidate_command(args, DEFAULT_CANDIDATES[0], trial_dir)
    assert "--include_global_supplementary_train" not in cmd
    assert "--restore_best_at_end" in cmd
    assert "--resolution" in cmd and cmd[cmd.index("--resolution") + 1] == "128"
    assert "--grad_accum" in cmd and cmd[cmd.index("--grad_accum") + 1] == "4"
    assert "--moe_load_balance_lambda" in cmd
    assert "96" not in cmd

    args.include_global_supplementary_train = True
    cmd = candidate_command(args, DEFAULT_CANDIDATES[0], trial_dir)
    assert "--include_global_supplementary_train" in cmd


def test_train_validation_split_keeps_supplementary_out_of_validation(tmp_path):
    pytest.importorskip("h5py")
    import h5py

    from scripts.train_3d_mri import ok_indices, split_fit_val_indices

    cache = tmp_path / "cache.h5"
    with h5py.File(cache, "w") as h5:
        h5.create_dataset("image", data=np.zeros((6, 1, 4, 4, 4), dtype=np.float16))
        h5.create_dataset("label", data=np.array([0, 1, 0, 1, 0, 1], dtype=np.int8))
        h5.create_dataset("tabular", data=np.zeros((6, 5), dtype=np.float32))
        h5.create_dataset("sample_hash", data=[b"real_a", b"real_b", b"real_c", b"real_d", b"syn_f0_a", b"syn_f0_b"])
        h5.create_dataset("status", data=[b"OK"] * 6)
        h5.create_dataset("source", data=[b"real", b"real", b"real", b"real", b"simulated_augment", b"simulated_augment"])
        h5.create_dataset("source_sample_hash", data=[b"", b"", b"", b"", b"real_a", b"real_b"])
        h5.create_dataset("source_fold", data=np.array([-1, -1, -1, -1, 0, 0], dtype=np.int16))
        h5.create_dataset("source_train_only", data=np.array([False, False, False, False, True, True], dtype=bool))

    fit_idx, fit_y, val_idx, val_y = split_fit_val_indices(
        [0, 1, 2, 3, 4, 5],
        [0, 1, 0, 1, 0, 1],
        cache_path=cache,
        ok=ok_indices(cache),
        val_fraction=0.5,
        seed=7,
        fold=0,
    )

    assert set(val_idx).isdisjoint({4, 5})
    sample_hashes = ["real_a", "real_b", "real_c", "real_d", "syn_f0_a", "syn_f0_b"]
    parent_hashes = ["", "", "", "", "real_a", "real_b"]
    validation_hashes = {sample_hashes[idx] for idx in val_idx}
    for idx in set(fit_idx).intersection({4, 5}):
        assert parent_hashes[idx] not in validation_hashes
    assert len(fit_y) == len(fit_idx)
    assert len(val_y) == len(val_idx)
