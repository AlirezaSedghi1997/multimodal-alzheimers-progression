"""BrainIAC frozen embedding extraction, schema v2 (Plan 004 / B3 build).

Reads ONLY schema-v2 preprocessing outputs (96-cube NIfTI + validated
receipts) and the pinned official runtime.  The legacy cache128 extractor
stays permanently disabled; any reference to the cache, catalog labels, PET,
augmentation or outer split data fails closed before a single output byte is
written.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Callable, Mapping, Sequence

from src.research.brainiac_preprocessing_v2 import (
    PRIMARY_ROOT as V2_ROOT,
    assert_no_cnn_writer,
)
from src.research.brainiac_preprocessing_v2_schema import validate_receipt

ROOT = Path(__file__).resolve().parents[2]
FEATURES_ROOT = ROOT / "data" / "derived" / "method_x_brainiac" / "r1_features_v1"
SHARD_DIR = FEATURES_ROOT / "brainiac_shards"
FINAL_H5_PATH = FEATURES_ROOT / "brainiac_embeddings_1669x768.h5"
MANIFEST_PATH = FEATURES_ROOT / "brainiac_embeddings_manifest.json"
CONTRACT_PATH = (
    ROOT / "results" / "method_x_registry" / "brainiac_r1"
    / "brainiac_embedding_contract_v1.json"
)

EMBEDDING_DIM = 768
EXPECTED_ROWS = 1669
BATCH_PROBE_SAMPLE_COUNT = 16
BATCH_COSINE_MIN = 0.999999
BATCH_ABS_DRIFT_MAX = 1e-6
ALLOWED_FINAL_DATASETS = ("sample_hash", "embedding")

_FORBIDDEN_TOKENS: tuple[str, ...] = (
    "t1_volume_cache_128",
    "master_catalog",
    "ab_label",
    "pet",
    "is_synthetic",
    "augmentation",
    "fold_",
    "external_holdout",
)


def assert_no_forbidden_source(path_or_name: str) -> None:
    text = str(path_or_name).casefold().replace("\\", "/")
    for token in _FORBIDDEN_TOKENS:
        if token.casefold() in text:
            raise RuntimeError(
                f"BRAINIAC_EMBEDDING_FORBIDDEN_SOURCE:{token}"
            )


def verify_v2_sample(sample_hash: str, *, v2_root: Path = V2_ROOT) -> dict[str, Any]:
    """Bind a sample to its validated schema-v2 receipt and NIfTI bytes."""

    receipt_path = Path(v2_root) / "receipts" / f"{sample_hash}.json"
    if not receipt_path.is_file():
        raise RuntimeError("BRAINIAC_EMBEDDING_V2_SOURCE_MISSING")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    validate_receipt(receipt, artifact_root=Path(v2_root))
    nifti = Path(v2_root) / receipt["output_relative_path"]
    if not nifti.is_file():
        raise RuntimeError("BRAINIAC_EMBEDDING_V2_OUTPUT_MISSING")
    return {
        "sample_hash": sample_hash,
        "receipt": receipt,
        "nifti_path": nifti,
        "output_sha256": receipt["output_sha256"],
    }


def select_batch_size(
    *,
    candidates: Sequence[int],
    probe_payloads: Sequence[tuple[str, Any]],
    forward_fn: Callable[[list[tuple[str, Any]]], Sequence[Any]],
) -> int:
    """Largest candidate whose batched forward matches batch=1 exactly."""

    import numpy as np

    if len(probe_payloads) != BATCH_PROBE_SAMPLE_COUNT:
        raise RuntimeError("BRAINIAC_EMBEDDING_PROBE_COUNT_INVALID")
    ordered = list(candidates)
    if not ordered or sorted(ordered) != list(dict.fromkeys(sorted(ordered))):
        raise RuntimeError("BRAINIAC_EMBEDDING_CANDIDATES_INVALID")

    baseline: dict[str, Any] = {}
    for start in range(0, len(probe_payloads), 1):
        batch = probe_payloads[start : start + 1]
        for sample_hash, vector in zip(
            [h for h, _ in batch], forward_fn(batch), strict=True
        ):
            baseline[sample_hash] = np.asarray(vector, dtype="float32")

    chosen = 1
    for candidate in sorted(set(ordered), reverse=True):
        if candidate <= 1:
            continue
        ok = True
        for start in range(0, len(probe_payloads), candidate):
            batch = probe_payloads[start : start + candidate]
            outputs = forward_fn(batch)
            if len(outputs) != len(batch):
                ok = False
                break
            for (sample_hash, _payload), vector in zip(batch, outputs, strict=True):
                array = np.asarray(vector, dtype="float32")
                reference = baseline.get(sample_hash)
                if (
                    reference is None
                    or array.shape != (EMBEDDING_DIM,)
                    or not np.isfinite(array).all()
                ):
                    ok = False
                    break
                drift = float(np.max(np.abs(array - reference)))
                cosine = float(
                    np.dot(array, reference)
                    / (np.linalg.norm(array) * np.linalg.norm(reference))
                )
                if not (
                    math.isfinite(drift)
                    and drift <= BATCH_ABS_DRIFT_MAX
                    and cosine >= BATCH_COSINE_MIN
                ):
                    ok = False
                    break
            if not ok:
                break
        if ok:
            chosen = candidate
            break
    return chosen


def _atomic_replace(temp: Path, target: Path) -> None:
    os.replace(temp, target)


def write_shard(
    path: Path,
    rows: Sequence[tuple[str, Any]],
    *,
    datasets_only: bool = True,
) -> None:
    """Atomic shard write: sample_hash + embedding[768] only."""

    import h5py
    import numpy as np

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    hashes = [str(h) for h, _ in rows]
    if len(set(hashes)) != len(hashes):
        raise RuntimeError("BRAINIAC_EMBEDDING_SHARD_INVALID:duplicate")
    matrix = np.stack(
        [np.asarray(v, dtype="float32") for _, v in rows]
    ) if rows else np.zeros((0, EMBEDDING_DIM), dtype="float32")
    descriptor, temp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temp = Path(temp_name)
    try:
        with h5py.File(temp, "w") as handle:
            dataset = handle.create_dataset(
                "sample_hash", data=[h.encode("utf-8") for h in hashes]
            )
            dataset.attrs["role"] = "identifier"
            handle.create_dataset("embedding", data=matrix)
            handle.attrs["artifact"] = "BRAINIAC_EMBEDDING_SHARD_V1"
            handle.attrs["embedding_dim"] = EMBEDDING_DIM
        _atomic_replace(temp, path)
    finally:
        if temp.exists():
            temp.unlink()


def validate_shard(path: Path) -> dict[str, Any]:
    import h5py
    import numpy as np

    with h5py.File(Path(path), "r") as handle:
        names = set(handle.keys())
        if not {"sample_hash", "embedding"} <= names:
            raise RuntimeError("BRAINIAC_EMBEDDING_SHARD_INVALID:schema")
        if any(name not in {"sample_hash", "embedding"} for name in names):
            raise RuntimeError("BRAINIAC_EMBEDDING_FORBIDDEN_DATASET")
        hashes = [h.decode("utf-8") if isinstance(h, bytes) else str(h)
                  for h in handle["sample_hash"][:]]
        matrix = handle["embedding"][:]
    if matrix.ndim != 2 or matrix.shape[1] != EMBEDDING_DIM:
        raise RuntimeError("BRAINIAC_EMBEDDING_SHARD_INVALID:dim")
    if len(hashes) != matrix.shape[0] or len(set(hashes)) != len(hashes):
        raise RuntimeError("BRAINIAC_EMBEDDING_SHARD_INVALID:duplicate")
    if not np.isfinite(matrix).all():
        raise RuntimeError("BRAINIAC_EMBEDDING_SHARD_INVALID:nonfinite")
    return {"rows": len(hashes), "path": str(Path(path))}


def merge_shards(
    shard_paths: Sequence[Path], final_path: Path, *, expected_rows: int
) -> dict[str, Any]:
    """Merge only after every shard independently validates; atomic output."""

    import h5py
    import numpy as np

    validated = [validate_shard(Path(p)) for p in shard_paths]
    total = sum(row["rows"] for row in validated)
    if total != expected_rows:
        raise RuntimeError("BRAINIAC_EMBEDDING_MERGE_COUNT_INVALID")
    final_path = Path(final_path)
    final_path.parent.mkdir(parents=True, exist_ok=True)
    hashes: list[str] = []
    rows: list[Any] = []
    with h5py.File(validated[0]["path"], "r") as first:
        pass
    for shard in shard_paths:
        with h5py.File(Path(shard), "r") as handle:
            shard_hashes = [h.decode("utf-8") if isinstance(h, bytes) else str(h)
                            for h in handle["sample_hash"][:]]
            hashes.extend(shard_hashes)
            rows.append(handle["embedding"][:])
    if len(set(hashes)) != len(hashes):
        raise RuntimeError("BRAINIAC_EMBEDDING_MERGE_OVERLAP")
    matrix = np.concatenate(rows, axis=0)
    write_shard(final_path, list(zip(hashes, matrix)))
    return {"rows": len(hashes), "path": str(final_path)}


def validate_final_embeddings(
    path: Path = FINAL_H5_PATH, *, expected_rows: int = EXPECTED_ROWS
) -> dict[str, Any]:
    import h5py
    import numpy as np

    path = Path(path)
    if not path.is_file():
        raise RuntimeError("BRAINIAC_EMBEDDING_FINAL_MISSING")
    with h5py.File(path, "r") as handle:
        names = sorted(handle.keys())
        if names != sorted(ALLOWED_FINAL_DATASETS):
            raise RuntimeError("BRAINIAC_EMBEDDING_FORBIDDEN_DATASET")
        hashes = [h.decode("utf-8") if isinstance(h, bytes) else str(h)
                  for h in handle["sample_hash"][:]]
        matrix = handle["embedding"][:]
    if matrix.shape != (expected_rows, EMBEDDING_DIM):
        raise RuntimeError("BRAINIAC_EMBEDDING_FINAL_SHAPE_INVALID")
    if len(set(hashes)) != expected_rows:
        raise RuntimeError("BRAINIAC_EMBEDDING_FINAL_DUPLICATE")
    if not np.isfinite(matrix).all():
        raise RuntimeError("BRAINIAC_EMBEDDING_FINAL_NONFINITE")
    return {
        "rows": expected_rows,
        "embedding_dim": EMBEDDING_DIM,
        "sample_hash_domain_sha256": hashlib.sha256(
            "\n".join(sorted(hashes)).encode("utf-8")
        ).hexdigest(),
    }


def _canonical_sha(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
            "utf-8"
        )
    ).hexdigest()


def build_embedding_contract(
    *,
    source_aggregate_sha: str,
    model_checkpoint_sha: str,
    runtime_lock_sha: str,
) -> dict[str, Any]:
    body = {
        "schema_version": 1,
        "artifact": "BRAINIAC_EMBEDDING_CONTRACT_V1",
        "claim_ceiling": "POST_HOC_INTERNAL_TECHNICAL_EVIDENCE",
        "source_aggregate_sha256": source_aggregate_sha,
        "model_checkpoint_sha256": model_checkpoint_sha,
        "runtime_lock_sha256": runtime_lock_sha,
        "embedding_dim": EMBEDDING_DIM,
        "expected_rows": EXPECTED_ROWS,
        "batch_rule": (
            "largest of {4,2,1} with cosine>=0.999999 and |drift|<=1e-6 vs "
            "batch=1 on the 16 lowest-hash samples; NaN/Inf/OOM rejects the "
            "candidate without altering math"
        ),
        "forbidden_sources": list(_FORBIDDEN_TOKENS),
        "final_datasets": list(ALLOWED_FINAL_DATASETS),
        "labels_read": False,
        "pet_read": False,
        "outer_test_touched": False,
    }
    body["integrity_sha256"] = _canonical_sha(
        {k: v for k, v in body.items() if k != "integrity_sha256"}
    )
    return body


def validate_embedding_contract(path: Path = CONTRACT_PATH) -> dict[str, Any]:
    contract = json.loads(Path(path).read_text(encoding="utf-8"))
    if contract.get("artifact") != "BRAINIAC_EMBEDDING_CONTRACT_V1":
        raise RuntimeError("BRAINIAC_EMBEDDING_CONTRACT_INVALID")
    expected = _canonical_sha(
        {k: v for k, v in contract.items() if k != "integrity_sha256"}
    )
    if contract.get("integrity_sha256") != expected:
        raise RuntimeError("BRAINIAC_EMBEDDING_CONTRACT_INVALID")
    for flag in ("labels_read", "pet_read", "outer_test_touched"):
        if contract.get(flag) is not False:
            raise RuntimeError("BRAINIAC_EMBEDDING_CONTRACT_INVALID")
    return dict(contract)
