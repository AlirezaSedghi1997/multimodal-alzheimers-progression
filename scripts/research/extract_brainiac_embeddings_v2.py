"""BrainIAC frozen embedding extraction, schema v2 (Plan 004 build / 005 run).

--mode preflight : read-only source/contract evidence (no lease, no writes)
--mode smoke     : source-discipline shard over real schema-v2 NIfTIs using
                   the pinned official model under .venvs/brainiac (GPU when
                   free; CPU fallback is explicit).  Writes only into a
                   separate smoke shard root.
--mode run       : full 1669 extraction with lease (Plan 005 territory).

Forbidden sources (cache128, catalog labels, PET, augmentation, outer
splits) fail closed before any output byte.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.research import brainiac_embedding_v2 as ev
from src.research import brainiac_preprocessing_v2 as v2

SMOKE_SHARD_ROOT = ev.FEATURES_ROOT / "brainiac_shards_smoke"
RUNTIME_LOCK = (
    ROOT / "results" / "method_x_registry" / "brainiac_r1"
    / "brainiac_official_runtime_lock.json"
)
MODEL_CHECKPOINT_SHA_PLACEHOLDER = "0" * 64  # bound live at run time


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime())


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    encoded = (json.dumps(dict(payload), indent=2, sort_keys=True) + "\n").encode("utf-8")
    descriptor = os.open(
        temporary, os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_BINARY", 0)
    )
    try:
        os.write(descriptor, encoded)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.replace(temporary, path)


def _load_official_forward(device: str):
    """Strict pinned BrainIAC forward from the vendor runtime (1x1x96^3->768)."""

    from src.research.brainiac_model_loader import load_brainiac_encoder
    from src.research.brainiac_execution_gate import brainiac_action_lease

    with brainiac_action_lease("EXTRACT_FROZEN_BRAINIAC_EMBEDDINGS"):
        encoder = load_brainiac_encoder(device=device)
    import torch

    def forward(batch):
        import nibabel as nib
        import numpy as np

        tensors = []
        for sample_hash, source in batch:
            ev.assert_no_forbidden_source(str(source["nifti_path"]))
            array = nib.load(str(source["nifti_path"])).get_fdata(dtype="float32")
            if array.shape != (96, 96, 96) or not np.isfinite(array).all():
                raise RuntimeError("BRAINIAC_EMBEDDING_INPUT_INVALID")
            tensors.append(torch.from_numpy(array)[None, None])
        stacked = torch.cat(tensors, dim=0)
        with torch.inference_mode():
            output = encoder(stacked.to(device))
        return [row.detach().cpu().numpy() for row in output]

    return forward


def preflight() -> dict[str, Any]:
    membership = v2.load_locked_membership()
    lowest = v2.select_calibration_samples(
        [m["sample_hash"] for m in membership], count=ev.BATCH_PROBE_SAMPLE_COUNT
    )
    v2_available = (v2.PRIMARY_ROOT / "receipts").is_dir()
    committed = 0
    if v2_available:
        committed = len(list((v2.PRIMARY_ROOT / "receipts").glob("*.json")))
    body = {
        "schema_version": 1,
        "status": "PASS" if committed == v2.EXPECTED_REAL_ROWS else "PENDING_B2",
        "membership_rows": len(membership),
        "v2_committed_receipts": committed,
        "v2_full_run_required_before_extraction": committed != v2.EXPECTED_REAL_ROWS,
        "batch_probe_samples": ev.BATCH_PROBE_SAMPLE_COUNT,
        "forbidden_sources": list(ev._FORBIDDEN_TOKENS),
        "labels_read": False,
        "pet_read": False,
        "created_at_utc": _utc_now(),
    }
    return body


def smoke(*, limit: int, device: str) -> dict[str, Any]:
    import numpy as np

    membership = v2.load_locked_membership()
    candidates = [
        m["sample_hash"] for m in membership
        if (v2.PRIMARY_ROOT / "receipts" / f"{m['sample_hash']}.json").is_file()
    ]
    if not candidates:
        raise RuntimeError("BRAINIAC_EMBEDDING_NO_V2_SAMPLES_AVAILABLE")
    selected = candidates[: max(1, int(limit))]
    sources = [ev.verify_v2_sample(h) for h in selected]
    forward = _load_official_forward(device=device)
    rows = []
    for sample_hash, source in zip(selected, sources, strict=True):
        vector = forward([(sample_hash, source)])[0]
        rows.append((sample_hash, np.asarray(vector, dtype="float32")))
    shard = SMOKE_SHARD_ROOT / "smoke_0000.h5"
    ev.write_shard(shard, rows)
    validated = ev.validate_shard(shard)
    return {
        "status": "PASS",
        "rows": validated["rows"],
        "shard": str(shard),
        "sources": "schema_v2_nifti_only",
        "labels_read": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("preflight", "smoke", "run"), required=True)
    parser.add_argument("--limit", type=int, default=2)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    if args.mode == "preflight":
        result: dict[str, Any] = preflight()
    elif args.mode == "smoke":
        result = smoke(limit=args.limit, device=args.device)
    else:
        raise RuntimeError("BRAINIAC_EMBEDDING_FULL_RUN_BELONGS_TO_PLAN_005")
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return 0 if result.get("status") == "PASS" else 3


if __name__ == "__main__":
    raise SystemExit(main())
