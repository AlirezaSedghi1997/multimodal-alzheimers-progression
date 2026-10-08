"""Read-only parent receipts and output-bound, resumable analysis units."""
from __future__ import annotations

import copy
import importlib.metadata
import io
import json
import sys
from contextlib import redirect_stdout
from pathlib import Path

import pandas as pd

from scripts.xyz_fusion_v1.store import Store, atomic_json, digest, file_sha


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def load_parent(root):
    root = Path(root)
    parent = root / "results/xyz_mri_baseline_fast_v1"
    original = read_json(parent / "private/contract.json")
    if read_json(parent / "orchestration_state.json").get("stage") != "PRIMARY_FINALIZED":
        raise ValueError("PRIMARY_NOT_FINALIZED")
    current = copy.deepcopy(original)
    current["source_sha256"] = {k: file_sha(root / k) for k in original["source_sha256"]}
    current["runtime"] = {k: importlib.metadata.version(k) for k in original["runtime"]}
    import torch
    current["python"] = sys.version
    current["torch_cuda"] = torch.version.cuda
    current["cuda_device"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    # Store verifies the narrowly evidenced fsync/AMP amendments; no new override.
    store = Store(parent, current)
    with redirect_stdout(io.StringIO()):
        cache = store.job("cache")
        splits = store.job("splits")
        cells = [store.job(f"outer_{i}") for i in range(5)]
        report = store.job("reports")
    if read_json(parent / "public/results.json") != report:
        raise ValueError("PRIMARY_PUBLIC_REPORT_DRIFT")
    frame = pd.DataFrame(original["manifest"]).set_index("mri_sample_hash")
    frame = frame.loc[cache["selected_sample_keys"]].reset_index()
    frame["cache_row"] = range(len(frame))
    if frame.patient_hash.duplicated().any() or len(splits) != 5:
        raise ValueError("PRIMARY_MEMBERSHIP_INVALID")
    return dict(root=root, parent=parent, contract=current, store=store, cache=cache,
                config=original["config"], frame=frame, splits=splits, cells=cells, report=report)


def verify_primary(ctx):
    from scripts.xyz_fusion_v1.pipeline import verify_bound_outputs
    from src.method_y_revival.privacy import scan_public_bundle
    with redirect_stdout(io.StringIO()):
        proof = verify_bound_outputs(ctx["parent"], ctx["store"], ctx["config"])
    if scan_public_bundle(ctx["parent"] / "public"):
        raise ValueError("PRIMARY_PRIVACY_FAILED")
    if ctx["report"]["outer_folds_completed"] != 5:
        raise ValueError("PRIMARY_FOLD_COUNT_INVALID")
    return dict(status="PRIMARY_VERIFIED", **proof, mri_fits=50, cox_fits=45,
                subjects=len(ctx["frame"]), events=int(ctx["frame"].event.sum()),
                execution_contract_sha256=ctx["store"].execution_binding)


def bound_job(store, name, compute=None):
    """Recovery is permitted only when both receipt and artifact bytes verify."""
    payload = store.job(name, compute)
    for relative, sha in payload.get("outputs", {}).items():
        path = (store.root / relative).resolve()
        if not path.is_relative_to(store.root.resolve()) or file_sha(path) != sha:
            raise ValueError("POSTPRIMARY_OUTPUT_HASH_MISMATCH:" + name)
    return payload


def write_array(path, array):
    import os
    import numpy as np
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as stream:
        np.save(stream, array, allow_pickle=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)


def output_hashes(root, paths):
    return {Path(p).relative_to(root).as_posix(): file_sha(p) for p in paths}
