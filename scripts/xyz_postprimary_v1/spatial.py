"""Signed MRI-risk attribution and its exact linear contribution to frozen Cox fusion."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import torch

from scripts.xyz_fusion_v1.mri_training import MRISurvival
from scripts.xyz_fusion_v1.store import file_sha
from .context import read_json


def select_cases(ctx, count_per_fold, seed):
    from scripts.xyz_fusion_v1.store import digest
    rows = ctx["frame"].set_index("patient_hash")
    cases = []
    for split in ctx["splits"]:
        # No outcome, risk, correct prediction or attractive map influences selection.
        keys = sorted(split["test"], key=lambda k: digest({"seed": seed, "key": k}))
        for index, key in enumerate(keys[:count_per_fold]):
            row = rows.loc[key]
            cases.append({"case_id": f"fold{split['fold']}_case{index}", "fold": split["fold"],
                          "patient_hash": key, "sample_key": row.mri_sample_hash,
                          "cache_row": int(row.cache_row), "raw_path": row.raw_path})
    return cases


def score_from_logits(logits, policy, grid):
    # CPU cumprod retains autograd and avoids nondeterministic CUDA backward.
    boundary = torch.cat([torch.ones((len(logits), 1), dtype=logits.dtype, device="cpu"),
                          torch.cumprod(torch.sigmoid(-logits.cpu()), dim=1)], dim=1)
    edges = np.asarray(policy["hazard_edges_days"], float)
    values = []
    for t in grid:
        hi = int(np.searchsorted(edges, t, side="right")); hi = min(hi, len(edges)-1)
        lo = hi-1; weight = float((t-edges[lo])/(edges[hi]-edges[lo]))
        values.append(boundary[:, lo]*(1-weight) + boundary[:, hi]*weight)
    return (1-torch.stack(values, dim=1)).mean(dim=1)


def load_case(ctx, case, device):
    folder = ctx["parent"] / "private/models" / f"outer{case['fold']}_final_mri"
    receipt = read_json(folder / "locked.json")
    if file_sha(folder / "locked.pt") != receipt["sha256"]:
        raise ValueError("XAI_MODEL_HASH_DRIFT")
    if case["patient_hash"] in receipt["train_subjects"] or receipt["selection_subjects"]:
        raise ValueError("XAI_SUBJECT_SEEN_BY_MODEL")
    state = torch.load(folder / "locked.pt", map_location="cpu", weights_only=True)
    if state["binding"] != receipt["binding"] or state["ancestry_subjects"] != receipt["train_subjects"]:
        raise ValueError("XAI_MODEL_CONTRACT_MISMATCH")
    policy = ctx["config"]["training"]; grid = ctx["config"]["fusion"]["ibs_grid_days"]
    model = MRISurvival(policy).to(device)
    model.load_state_dict(state["model"], strict=True)
    model.eval()
    for p in model.parameters(): p.requires_grad_(False)
    scans = ctx["parent"] / "private/mri_scans" / case["sample_key"]
    scan = read_json(scans / "receipt.json")["payload"]
    for name in ("tensor.npy", "registered_t1.nii.gz", "registration_0GenericAffine.mat",
                 "registration_1InverseWarp.nii.gz", "registration_1Warp.nii.gz"):
        if file_sha(scans / name) != scan["outputs"][name]:
            raise ValueError("XAI_INPUT_OR_TRANSFORM_DRIFT")
    if file_sha(case["raw_path"]) != scan["raw_sha256"]:
        raise ValueError("XAI_NATIVE_RAW_IMAGE_DRIFT")
    array = np.load(scans / "tensor.npy", allow_pickle=False)
    image = torch.tensor(array, dtype=torch.float32, device=device)[None, None]
    expected = None
    for path in sorted((ctx["parent"] / "private/jobs").glob(f"outer_predictions_{case['fold']}_*.json")):
        p = ctx["store"].job(path.stem)
        if case["patient_hash"] in p["subjects"]:
            expected = p["mri_risk"][p["subjects"].index(case["patient_hash"])]; break
    if expected is None: raise ValueError("XAI_COMMITTED_PREDICTION_MISSING")
    def forward(x): return score_from_logits(model(x).float(), policy, grid).squeeze()
    with torch.no_grad():
        with torch.autocast(device.type, enabled=policy["amp"] and device.type == "cuda"):
            replay = float(score_from_logits(model(image).float(), policy, grid)[0])
        fp32 = float(forward(image))
    if abs(replay-expected) > 2e-6: raise ValueError("XAI_LOCKED_PREDICTION_NOT_REPRODUCED")
    if abs(fp32-expected) > 1e-3: raise ValueError("XAI_FP32_TARGET_DEVIATES_FROM_AMP_PREDICTION")
    return model, image, forward, scan, dict(amp_replay_error=abs(replay-expected),
        fp32_attribution_target_error=abs(fp32-expected), model_sha256=receipt["sha256"],
        target="MEAN_MRI_PROGRESSION_RISK_OVER_LOCKED_IBS_GRID_NOT_DIAGNOSIS")


def fusion_coefficient(ctx, fold):
    import pickle
    path = ctx["parent"] / "private/models" / f"outer{fold}_clinical_mri_amyloid.pkl"
    receipt = read_json(path.with_suffix(".json"))
    if file_sha(path) != receipt["sha256"]: raise ValueError("FUSION_MODEL_HASH_DRIFT")
    with path.open("rb") as stream: model = pickle.load(stream)  # verified trusted local model
    index = receipt["features"].index("mri_risk")
    coefficient = float(model[-1].coef_[index] / model[-2].scale_[index])
    if not np.isfinite(coefficient): raise ValueError("INVALID_FUSION_MRI_COEFFICIENT")
    return coefficient
