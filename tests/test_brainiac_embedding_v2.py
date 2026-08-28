"""Adversarial tests for the BrainIAC v2 embedding extractor (Plan 004 / B3)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from src.research import brainiac_embedding_v2 as ev


@pytest.fixture(autouse=True)
def _isolated_guard(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ev, "assert_no_cnn_writer", lambda: None)


def _fake_forward(batch: list[tuple[str, object]]) -> list[object]:
    """Deterministic synthetic projector: hash-derived 768 vector."""

    import numpy as np

    out = []
    for sample_hash, _payload in batch:
        seed = int(hashlib.sha256(sample_hash.encode()).hexdigest()[:8], 16)
        rng = np.random.default_rng(seed)
        out.append(rng.standard_normal(768).astype("float32"))
    return out


# ---------------------------------------------------------------------------
# deny boundary
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "forbidden",
    [
        "data/cached/t1_volume_cache_128.h5",
        "data/processed/master_catalog.csv",
        "splits/fold_0_test.csv",
        "any/path/ab_label.parquet",
        "pet/something.h5",
        "derived/augmentation_v1/x.nii",
    ],
)
def test_forbidden_sources_fail_before_any_write(forbidden: str) -> None:
    with pytest.raises(RuntimeError, match="FORBIDDEN_SOURCE"):
        ev.assert_no_forbidden_source(forbidden)


def test_allowed_sources_pass() -> None:
    ev.assert_no_forbidden_source(
        "data/derived/method_x_brainiac/r1_primary_v2/outputs/s_x.nii.gz"
    )
    ev.assert_no_forbidden_source("vendor/BrainIAC/ba60f45/src/model.py")


def test_loader_opens_never_touch_forbidden_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    opened: list[str] = []

    real_open = open

    def tracking_open(file, *args, **kwargs):
        opened.append(str(file))
        return real_open(file, *args, **kwargs)

    import builtins

    monkeypatch.setattr(builtins, "open", tracking_open)
    ev.assert_no_forbidden_source("ok/path.nii")
    joined = "\n".join(opened)
    for banned in ("t1_volume_cache_128", "master_catalog", "ab_label", "pet"):
        assert banned not in joined


# ---------------------------------------------------------------------------
# shards
# ---------------------------------------------------------------------------


def _rows(n: int, start: int = 0) -> list[tuple[str, object]]:
    import numpy as np

    return [
        (f"s_{start + i:04d}", np.zeros(768, dtype="float32") + i)
        for i in range(n)
    ]


def test_shard_write_validate_and_duplicate_rejection(tmp_path: Path) -> None:
    shard = tmp_path / "shard_0000.h5"
    ev.write_shard(shard, _rows(4))
    validated = ev.validate_shard(shard)
    assert validated["rows"] == 4
    duplicate = tmp_path / "shard_dup.h5"
    rows = _rows(2) + _rows(1, start=1)
    # duplicates are rejected at write time, before any output byte lands
    with pytest.raises(RuntimeError, match="SHARD_INVALID"):
        ev.write_shard(duplicate, rows)
    assert not duplicate.exists()


def test_shard_rejects_nonfinite_and_wrong_dim(tmp_path: Path) -> None:
    import numpy as np

    shard = tmp_path / "shard_bad.h5"
    rows = [("s_a", np.full(768, np.nan, dtype="float32"))]
    ev.write_shard(shard, rows)
    with pytest.raises(RuntimeError, match="SHARD_INVALID"):
        ev.validate_shard(shard)
    shard2 = tmp_path / "shard_dim.h5"
    ev.write_shard(shard2, [("s_b", np.zeros(128, dtype="float32"))])
    with pytest.raises(RuntimeError, match="SHARD_INVALID"):
        ev.validate_shard(shard2)


def test_merge_requires_all_shards_valid_and_atomic(tmp_path: Path) -> None:
    good_a, good_b = tmp_path / "a.h5", tmp_path / "b.h5"
    ev.write_shard(good_a, _rows(3))
    ev.write_shard(good_b, _rows(2, start=3))
    bad = tmp_path / "bad.h5"
    ev.write_shard(bad, [("s_x", __import__("numpy").full(768, float("inf"), dtype="float32"))])
    final = tmp_path / "final.h5"
    with pytest.raises(RuntimeError, match="SHARD_INVALID"):
        ev.merge_shards([good_a, bad, good_b], final, expected_rows=5)
    assert not final.exists()  # atomic: no partial merge artifact
    ev.merge_shards([good_a, good_b], final, expected_rows=5)
    report = ev.validate_final_embeddings(final, expected_rows=5)
    assert report["rows"] == 5


def test_final_embeddings_forbid_extra_datasets(tmp_path: Path) -> None:
    import h5py
    import numpy as np

    final = tmp_path / "final.h5"
    ev.write_shard(final, _rows(2), datasets_only=False)
    with h5py.File(final, "a") as handle:
        handle.create_dataset("ab_label", data=[0, 1])
    with pytest.raises(RuntimeError, match="FORBIDDEN_DATASET"):
        ev.validate_final_embeddings(final, expected_rows=2)


# ---------------------------------------------------------------------------
# batch selection contract
# ---------------------------------------------------------------------------


def test_batch_selection_prefers_largest_equivalent() -> None:
    probes = [(f"s_{i:04d}", object()) for i in range(16)]
    chosen = ev.select_batch_size(
        candidates=(4, 2, 1), probe_payloads=probes, forward_fn=_fake_forward
    )
    assert chosen == 4


def test_batch_selection_rejects_divergent_candidate() -> None:
    import numpy as np

    probes = [(f"s_{i:04d}", object()) for i in range(16)]

    def batched_forward(batch):
        # batch>1 introduces drift beyond 1e-6
        scale = 1.0 if len(batch) == 1 else 1.00001
        return [vec * scale for vec in _fake_forward(batch)]

    chosen = ev.select_batch_size(
        candidates=(4, 2, 1), probe_payloads=probes, forward_fn=batched_forward
    )
    assert chosen == 1


def test_batch_selection_rejects_nonfinite_candidate() -> None:
    import numpy as np

    probes = [(f"s_{i:04d}", object()) for i in range(16)]

    def poisoned(batch):
        out = _fake_forward(batch)
        if len(batch) > 1:
            out[-1] = np.full(768, np.nan, dtype="float32")
        return out

    chosen = ev.select_batch_size(
        candidates=(4, 2, 1), probe_payloads=probes, forward_fn=poisoned
    )
    assert chosen == 1


# ---------------------------------------------------------------------------
# contract live-byte validation
# ---------------------------------------------------------------------------


def test_embedding_contract_is_live_byte_bound(tmp_path: Path) -> None:
    contract = ev.build_embedding_contract(
        source_aggregate_sha="a" * 64, model_checkpoint_sha="b" * 64,
        runtime_lock_sha="c" * 64,
    )
    path = tmp_path / "contract.json"
    path.write_text(json.dumps(contract, sort_keys=True), encoding="utf-8")
    ev.validate_embedding_contract(path)
    tampered = dict(contract)
    tampered["model_checkpoint_sha"] = "0" * 64
    path.write_text(json.dumps(tampered, sort_keys=True), encoding="utf-8")
    with pytest.raises(RuntimeError, match="CONTRACT_INVALID"):
        ev.validate_embedding_contract(path)
