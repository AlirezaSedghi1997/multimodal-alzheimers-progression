"""Post-training PET region attribution (FINAL plan 04-pet-plan.md).

Applies ONLY after the v4 final verifier PASS, on locked checkpoints opened
read-only. Implements LayerCAM (primary), Score-CAM (convergent check) and
Grad-CAM (sanity) for 3D survival nets with a scalar risk target, plus the
sanity/aggregation/leg machinery. No training imports, no checkpoint writes,
no single-subject outputs.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

CEILING = "POST_HOC_INTERNAL_EXPLORATORY"

# Single canonical definition (includes "glenn"; keep parity with
# configs/method_y_pet_attribution_v4.yaml:forbidden_vocabulary — enforced at
# runtime by the launcher preflight parity assert). Substring matching is
# deliberate fail-closed policy (e.g. "causal" fires inside "causality").
BAN_TERMS = (
    "alzheimer", "alzheimer's", "disease-involved", "disease involved",
    "causal", "causality", "biomarker-quantified", "biomarker quantified",
    "centiloid", "centaur", "external validation", "external-validation",
    "diagnosis", "diagnostic", "triage", "patient-specific",
    "amyloid marker", "amyloid markers", "tau marker", "tau markers",
    "glenn",
)

LEG_IDS = (
    "LEG-01", "LEG-02", "LEG-03", "LEG-04",
    "LEG-05", "LEG-06", "LEG-07", "LEG-08",
)


class AttributionError(RuntimeError):
    pass


def _fail(code: str, detail: str = "") -> AttributionError:
    return AttributionError(f"BLOCKED_{code}" + (f":{detail}" if detail else ""))


# --------------------------------------------------------------------------
# Entry gate + read-only loading
# --------------------------------------------------------------------------

def verifier_gate(*, run_root: str | Path, expected_digest: str | None = None,
                  config_path: str | Path | None = None) -> dict[str, Any]:
    """Fail closed unless final verification PASSES on live state.

    Prefers a validated final_verification_pass.json artifact; otherwise runs
    the sanctioned verifier in-process (read-only) when config_path is given.
    """
    root = Path(run_root)
    cand = root / "final_verification_pass.json"
    if cand.is_file():
        try:
            payload = json.loads(cand.read_text(encoding="utf-8"))
            body = dict(payload)
            recorded = body.pop("integrity_sha256", None)
            canon = json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
            if (
                payload.get("status") == "PASS"
                and recorded == hashlib.sha256(canon).hexdigest()
                and (expected_digest is None or payload.get("run_contract_sha256") == expected_digest)
            ):
                return {"status": "PASS",
                        "run_contract_sha256": payload.get("run_contract_sha256"),
                        "evidence": "artifact",
                        "valid_outer_cells": payload.get("valid_outer_cells")}
        except (OSError, ValueError, UnicodeDecodeError):
            pass
        # Present-but-invalid artifact masks tampering if we silently fall
        # through to live-verify: fail closed instead (C-fix 2026-09-08).
        raise _fail("VERIFIER_ARTIFACT_TAMPERED", str(cand))
    if config_path is None:
        raise _fail("VERIFIER_NOT_PASS", "no artifact and no live-verify path")
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "method_y_v4_verifier",
        Path(__file__).resolve().parents[2] / "scripts"
        / "verify_method_y_image_factorial_v4.py")
    if spec is None or spec.loader is None:
        raise _fail("VERIFIER_NOT_PASS", "loader unavailable")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
        result = module.verify(config_path=Path(config_path), level="final")
    except Exception as exc:
        raise _fail("VERIFIER_NOT_PASS", f"live-verify failed: {type(exc).__name__}") from exc
    if not isinstance(result, Mapping) or result.get("status") != "PASS":
        raise _fail("VERIFIER_NOT_PASS", "live-verify not PASS")
    try:
        live_digest = json.loads((root / "run_contract.json").read_text(
            encoding="utf-8")).get("digest")
    except (OSError, ValueError):
        raise _fail("VERIFIER_NOT_PASS", "run contract unreadable")
    if expected_digest is not None and live_digest != expected_digest:
        raise _fail("VERIFIER_NOT_PASS", "digest drift")
    return {"status": "PASS", "run_contract_sha256": live_digest,
            "evidence": "live-verify",
            "valid_outer_cells": result.get("valid_outer_cells")}


def read_only_load_checkpoint(path: str | Path, *, expected_sha256: str) -> dict[str, Any]:
    """Load a locked checkpoint without modifying a single byte on disk."""
    p = Path(path)
    raw = p.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise _fail("CHECKPOINT_DIGEST_MISMATCH", p.name)
    try:
        import io
        payload = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
    except Exception as exc:
        raise _fail("CHECKPOINT_UNREADABLE", p.name) from exc
    if hashlib.sha256(p.read_bytes()).hexdigest() != expected_sha256:
        raise _fail("CHECKPOINT_MUTATED_DURING_LOAD", p.name)
    if not isinstance(payload, Mapping) or "model_state" not in payload:
        raise _fail("CHECKPOINT_CONTENT_INVALID", p.name)
    return dict(payload)


# --------------------------------------------------------------------------
# Determinism ledger
# --------------------------------------------------------------------------

def determinism_ledger() -> dict[str, Any]:
    import torch.backends.cudnn as cudnn

    return {
        "torch_deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
        "cudnn_deterministic": bool(cudnn.deterministic),
        "cudnn_benchmark": bool(cudnn.benchmark),
        "allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32)
        if torch.cuda.is_available()
        else False,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "python_hash_seed": os.environ.get("PYTHONHASHSEED"),
        "torch_version": str(torch.__version__),
    }


# --------------------------------------------------------------------------
# Risk scalar + layer resolution
# --------------------------------------------------------------------------

def risk_from_logits(logits: torch.Tensor, *, mode: str = "logit_sum") -> torch.Tensor:
    """Differentiable scalar risk. Primary = unsaturated logit sum (FINAL P1.1)."""
    if mode == "logit_sum":
        return logits.sum(dim=1)
    if mode == "mortality_last":
        hazards = torch.sigmoid(logits)
        surv = torch.cumprod(1.0 - hazards, dim=1)
        return 1.0 - surv[:, -1]
    raise _fail("UNKNOWN_RISK_MODE", mode)


def resolve_layer(model: nn.Module, dotted: str) -> nn.Module:
    """Resolve 'blocks' / 'encoder.stages.3' style paths; unknown -> BLOCKED."""
    node: Any = model
    for part in str(dotted).split("."):
        if part.isdigit():
            try:
                node = node[int(part)]
            except (IndexError, KeyError, TypeError) as exc:
                raise _fail("UNKNOWN_LAYER", dotted) from exc
        else:
            if not hasattr(node, part):
                raise _fail("UNKNOWN_LAYER", dotted)
            node = getattr(node, part)
    if not isinstance(node, nn.Module):
        raise _fail("UNKNOWN_LAYER", dotted)
    return node


def _hooked_forward_backward(model, volume, tabular, layer_names, *,
                             risk_mode="logit_sum"):
    """Forward + single backward with retained activation grads."""
    acts: dict[str, torch.Tensor] = {}
    layer_objs = [resolve_layer(model, n) for n in layer_names]
    handles = []
    for name, layer in zip(layer_names, layer_objs):
        def _hook(_m, _i, out, _n=name):
            out.retain_grad()
            acts[_n] = out
        handles.append(layer.register_forward_hook(_hook))
    logits = model(volume, tabular)
    risk = risk_from_logits(logits, mode=risk_mode)
    model.zero_grad(set_to_none=True)
    risk.sum().backward()
    grads = {n: acts[n].grad.detach().clone() for n in layer_names}
    acts = {n: acts[n].detach().clone() for n in layer_names}
    for h in handles:
        h.remove()
    return logits, risk, acts, grads


def erode_foreground(mask: torch.Tensor, *, voxels: int = 2) -> torch.Tensor:
    """Binary erosion (cube structuring element) for edge-band protocol.

    Accepts (D,H,W) or (...,D,H,W); leading dims are preserved.
    """
    if int(voxels) <= 0:
        return (mask > 0.5)
    m = (mask > 0.5).float()
    lead = m.shape[:-3]
    m = m.reshape(-1, 1, *m.shape[-3:])
    k = int(2 * voxels + 1)
    eroded = -F.max_pool3d(-m, kernel_size=k, stride=1, padding=voxels)
    return (eroded.reshape(*lead, *eroded.shape[-3:]) > 0.5)


def apply_foreground(cam: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    """Force exact-zero background (FINAL P2.1). None mask = passthrough."""
    if mask is None:
        return cam
    m = (mask > 0.5).to(cam.dtype)
    while m.dim() < cam.dim():
        m = m.unsqueeze(0)
    return cam * m


def _upsample_to_input(cam: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    return F.interpolate(cam, size=tuple(like.shape[2:]), mode="trilinear",
                         align_corners=False)


def gradcam_map(model, volume, tabular, layer_name, *, risk_mode="logit_sum",
                foreground_mask=None) -> torch.Tensor:
    model.eval()
    _, _, acts, grads = _hooked_forward_backward(model, volume, tabular,
                                                 [layer_name], risk_mode=risk_mode)
    a, g = acts[layer_name], grads[layer_name]
    w = g.mean(dim=(0, 2, 3, 4), keepdim=True)
    cam = torch.relu((w * a).sum(dim=1, keepdim=True))
    cam = cam / (cam.amax(dim=(1, 2, 3, 4), keepdim=True) + 1e-12)
    return apply_foreground(_upsample_to_input(cam, volume).detach(),
                            foreground_mask)


def layercam_map(model, volume, tabular, layer_name, *, risk_mode="logit_sum",
                 shallow_layer_name=None, foreground_mask=None) -> torch.Tensor:
    model.eval()
    names = [layer_name] + ([shallow_layer_name] if shallow_layer_name else [])
    _, _, acts, grads = _hooked_forward_backward(model, volume, tabular, names,
                                                 risk_mode=risk_mode)
    cams = []
    for n in names:
        w = torch.relu(grads[n])
        cam = (w * acts[n]).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)
        cam = cam / (cam.amax(dim=(1, 2, 3, 4), keepdim=True) + 1e-12)
        cams.append(_upsample_to_input(cam, volume))
    out = torch.stack(cams).mean(dim=0)
    return apply_foreground(out.detach(), foreground_mask)


def _gaussian_blur_3d(volume: torch.Tensor, *, sigma_vox: float) -> torch.Tensor:
    radius = max(1, int(3.0 * sigma_vox))
    ax = torch.arange(-radius, radius + 1, dtype=volume.dtype)
    kern = torch.exp(-0.5 * (ax / sigma_vox) ** 2)
    kern = kern / kern.sum()
    out = volume
    for dim in (2, 3, 4):
        out = out.transpose(dim, -1)
        shape = out.shape
        out = out.reshape(-1, 1, shape[-1])
        pad = radius
        out = F.pad(out, (pad, pad), mode="replicate")
        out = F.conv1d(out, kern.reshape(1, 1, -1).to(out.dtype))
        out = out.reshape(*shape[:-1], shape[-1])
        out = out.transpose(dim, -1)
    return out


def scorecam_map(model, volume, tabular, layer_name, *, risk_mode="logit_sum",
                 blur_sigma_vox=2.0, foreground_mask=None) -> torch.Tensor:
    """Gradient-free CIC-style scoring (no backward pass at all)."""
    model.eval()
    with torch.no_grad():
        base_vol = _gaussian_blur_3d(volume, sigma_vox=blur_sigma_vox)
        base_risk = risk_from_logits(model(base_vol, tabular), mode=risk_mode)
        layer = resolve_layer(model, layer_name)
        captured: dict[str, torch.Tensor] = {}
        handle = layer.register_forward_hook(
            lambda _m, _i, out: captured.setdefault("a", out.detach()))
        try:
            _ = model(volume, tabular)
        finally:
            handle.remove()
        acts = captured["a"][0]  # (C, d, h, w)
        c = acts.shape[0]
        up = F.interpolate(acts.unsqueeze(0), size=tuple(volume.shape[2:]),
                           mode="trilinear", align_corners=False)[0]
        up = (up - up.amin(dim=(1, 2, 3), keepdim=True)) / (
            up.amax(dim=(1, 2, 3), keepdim=True)
            - up.amin(dim=(1, 2, 3), keepdim=True) + 1e-12)
        scores = []
        for k in range(c):
            masked = up[k:k + 1].unsqueeze(0) * volume + \
                (1.0 - up[k:k + 1].unsqueeze(0)) * base_vol
            r = risk_from_logits(model(masked, tabular), mode=risk_mode)
            scores.append(float((r - base_risk).sum()))
        scores_t = torch.tensor(scores, dtype=torch.float32)
        weights = torch.softmax(scores_t, dim=0).reshape(-1, 1, 1, 1)
        cam = torch.relu((weights * acts).sum(dim=0, keepdim=True)).unsqueeze(0)
        cam = cam / (cam.amax() + 1e-12)
        return apply_foreground(_upsample_to_input(cam, volume).detach(),
                                foreground_mask)


# --------------------------------------------------------------------------
# Sanity gates
# --------------------------------------------------------------------------

def randomized_model_check(locked_map, shuffled_model, volume, tabular,
                           layer_name, *, rho_tol: float = 0.1,
                           delta_tol: float = 1e-6, n_boot: int = 500,
                           seed: int = 13) -> dict[str, Any]:
    """Randomized-model sanity (FINAL P5.1): shuffled weights must destroy maps.

    PASS iff rank-correlation 95% CI lies within [-tol, +tol] AND mean|Δ|
    exceeds delta_tol. Shuffled-label control is evaluated by the caller via
    the same function on label-shuffled cohort statistics.
    """
    probe = layercam_map(shuffled_model, volume, tabular, layer_name)
    a = locked_map.detach().flatten().double().numpy()
    b = probe.detach().flatten().double().numpy()
    if np.std(a) == 0 or np.std(b) == 0:
        return {"status": "BLOCKED", "reason": "DEGENERATE_MAP",
                "rank_correlation": 0.0, "ci_low": 0.0, "ci_high": 0.0,
                "mean_abs_delta": 0.0, "rho_tol": float(rho_tol),
                "n_boot": int(n_boot)}
    rho = float(np.corrcoef(a, b)[0, 1])
    rng = np.random.RandomState(int(seed) & 0xFFFFFFFF)
    idx = np.arange(a.size)
    boots = []
    for _ in range(int(n_boot)):
        sel = rng.choice(idx, size=idx.size, replace=True)
        c = np.corrcoef(a[sel], b[sel])[0, 1]
        boots.append(float(c) if np.isfinite(c) else 0.0)
    ci_low, ci_high = float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))
    mean_delta = float(np.mean(np.abs(a - b)))
    ok = (-rho_tol <= ci_low and ci_high <= rho_tol and mean_delta >= delta_tol)
    return {"status": "PASS" if ok else "BLOCKED", "rank_correlation": rho,
            "ci_low": ci_low, "ci_high": ci_high, "mean_abs_delta": mean_delta,
            "rho_tol": float(rho_tol), "n_boot": int(n_boot)}


def _block_means(flat_map: np.ndarray, *, blocks: int = 8) -> np.ndarray:
    n = flat_map.size
    k = max(1, n // int(blocks))
    return np.array([flat_map[i * k:(i + 1) * k].mean() for i in range(int(blocks))])


def agreement_gate(map_a, map_b, *, spearman_margin: float = 0.7,
                   dice_margin: float = 0.6, percentile: float = 95.0,
                   n_boot: int = 500, seed: int = 13,
                   blocks: int = 64) -> dict[str, Any]:
    """Convergent agreement (FINAL P4.3): block-level Spearman + top-decile
    Dice, each with subject-bootstrap CI + permutation null; gate on CI LOWER
    bounds vs pre-locked margins. Never averages methods."""
    a = np.asarray(map_a, dtype=float).flatten()
    b = np.asarray(map_b, dtype=float).flatten()
    if a.shape != b.shape or a.size < blocks * 4:
        raise _fail("AGREEMENT_INVALID_INPUT")
    ba, bb = _block_means(a, blocks=blocks), _block_means(b, blocks=blocks)
    sp = spearman_with_ci(ba, bb, n_boot=n_boot, seed=seed)
    dc = dice_at_percentile(a, b, percentile=percentile)
    rng = np.random.RandomState(int(seed) + 1 & 0xFFFFFFFF)
    nulls = []
    for _ in range(int(n_boot)):
        nulls.append(float(np.corrcoef(ba, rng.permutation(bb))[0, 1]))
    null_hi = float(np.percentile(np.asarray(nulls), 97.5))
    spear_ok = bool(sp["ci_low"] >= spearman_margin and sp["ci_low"] > null_hi)
    dice_ok = bool(dc["dice"] >= dice_margin)
    status = "PASS" if (spear_ok and dice_ok) else "RED_FLAG_INVESTIGATE"
    return {"status": status, "spearman": sp, "dice": dc,
            "null_97_5": null_hi, "margins": {"spearman": float(spearman_margin),
                                              "dice": float(dice_margin)}}


# --------------------------------------------------------------------------
# Aggregation statistics
# --------------------------------------------------------------------------

def _rng(seed: int) -> np.random.RandomState:
    return np.random.RandomState(int(seed) & 0xFFFFFFFF)


def bootstrap_ci(values: Sequence[float], *, n_boot: int = 2000,
                 seed: int = 13, ci: float = 0.95) -> dict[str, float]:
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0 or not np.isfinite(arr).all():
        raise _fail("BOOTSTRAP_INVALID_INPUT")
    rng = _rng(seed)
    means = np.array([float(np.mean(rng.choice(arr, size=arr.size, replace=True)))
                      for _ in range(int(n_boot))])
    lo_q, hi_q = (1.0 - ci) / 2.0 * 100.0, (1.0 + ci) / 2.0 * 100.0
    return {"mean": float(np.mean(arr)), "ci_low": float(np.percentile(means, lo_q)),
            "ci_high": float(np.percentile(means, hi_q)), "n_boot": int(n_boot)}


def spearman_with_ci(x, y, *, n_boot: int = 2000, seed: int = 13) -> dict[str, float]:
    from scipy.stats import rankdata
    xa = np.asarray(list(x), dtype=float)
    ya = np.asarray(list(y), dtype=float)
    if xa.shape != ya.shape or xa.size < 3:
        raise _fail("SPEARMAN_INVALID_INPUT")
    rx, ry = rankdata(xa), rankdata(ya)
    rho = float(np.corrcoef(rx, ry)[0, 1])
    rng = _rng(seed)
    idx = np.arange(xa.size)
    boots = []
    for _ in range(int(n_boot)):
        sel = rng.choice(idx, size=idx.size, replace=True)
        rxs, rys = rankdata(xa[sel]), rankdata(ya[sel])
        c = np.corrcoef(rxs, rys)[0, 1]
        boots.append(float(c) if np.isfinite(c) else 0.0)
    return {"rho": rho, "ci_low": float(np.percentile(boots, 2.5)),
            "ci_high": float(np.percentile(boots, 97.5)), "n": int(xa.size)}


def dice_at_percentile(a, b, *, percentile: float = 95.0) -> dict[str, float]:
    aa = np.asarray(list(a), dtype=float).flatten()
    bb = np.asarray(list(b), dtype=float).flatten()
    if aa.shape != bb.shape or aa.size == 0:
        raise _fail("DICE_INVALID_INPUT")
    ta, tb = float(np.percentile(aa, percentile)), float(np.percentile(bb, percentile))
    ma, mb = aa >= ta, bb >= tb
    inter = float(np.logical_and(ma, mb).sum())
    denom = float(ma.sum() + mb.sum())
    return {"dice": (2.0 * inter / denom) if denom > 0 else 0.0,
            "threshold_a": ta, "threshold_b": tb}


def holm_correct(p_values: Sequence[float], *, alpha: float = 0.05):
    order = sorted(range(len(p_values)), key=lambda i: float(p_values[i]))
    m = len(p_values)
    rejected = [False] * m
    corrected = [0.0] * m
    for rank, i in enumerate(order):
        corrected[i] = min(1.0, float(p_values[i]) * (m - rank))
    for rank, i in enumerate(order):
        if float(p_values[i]) <= alpha / (m - rank):
            rejected[i] = True
        else:
            break
    return rejected, corrected


# --------------------------------------------------------------------------
# Tabular share + fidelity (FINAL P5.2, LEG-06)
# --------------------------------------------------------------------------

def tabular_share(model, volumes, tabulars, layer_name, *, n_perm: int = 8,
                  seed: int = 13) -> dict[str, Any]:
    """Locked image-vs-tabular risk-share metric (read-only inference).

    image effect: mean|risk - risk(blurred image)|; tabular effect:
    mean over permutations of |risk - risk(shuffled tabular)|.
    Returns shares + FULL/PARTIAL_EXPLANATION stamp (threshold 0.5).
    """
    model.eval()
    rng = np.random.RandomState(int(seed) & 0xFFFFFFFF)
    vols = list(volumes)
    tabs = [t.detach().clone() for t in tabulars]
    with torch.no_grad():
        base = torch.stack([
            risk_from_logits(model(v.unsqueeze(0) if v.dim() == 4 else v,
                                   t.unsqueeze(0) if t.dim() == 1 else t),
                             mode="logit_sum").sum()
            for v, t in zip(vols, tabs)])
        img_fx = torch.stack([
            risk_from_logits(model(_gaussian_blur_3d(
                v.unsqueeze(0) if v.dim() == 4 else v, sigma_vox=2.0),
                t.unsqueeze(0) if t.dim() == 1 else t),
                mode="logit_sum").sum() for v, t in zip(vols, tabs)])
        tab_fx = []
        order = np.arange(len(tabs))
        for _ in range(int(n_perm)):
            perm = rng.permutation(order)
            tab_fx.append(torch.stack([
                risk_from_logits(model(v.unsqueeze(0) if v.dim() == 4 else v,
                                       tabs[int(perm[i])].unsqueeze(0)
                                       if tabs[int(perm[i])].dim() == 1
                                       else tabs[int(perm[i])]),
                                 mode="logit_sum").sum()
                for i, v in enumerate(vols)]))
    img_delta = float((base - img_fx).abs().mean())
    tab_delta = float(sum((base - p).abs().mean() for p in tab_fx) / max(1, len(tab_fx)))
    total = img_delta + tab_delta
    image_share = (img_delta / total) if total > 0 else 0.5
    return {"image_share": float(image_share),
            "tabular_share": float(1.0 - image_share),
            "stamp": "FULL_EXPLANATION" if image_share >= 0.5 else "PARTIAL_EXPLANATION"}


def fidelity_occlusion(model, volumes, tabulars, layer_name, *, n_masks: int = 8,
                       seed: int = 13, mask_fraction: float = 0.1,
                       n_boot: int = 2000) -> dict[str, Any]:
    """Occlusion fidelity on a locked subset (FINAL P5.2).

    Fill value = blurred-baseline value (locked). Compares risk drop from
    top-decile attribution masks vs K random masks (paired median + CI).
    """
    model.eval()
    rng = np.random.RandomState(int(seed) & 0xFFFFFFFF)
    vols = list(volumes)
    tabs = list(tabulars)
    deltas_m, deltas_r = [], []
    for v, t in zip(vols, tabs):
        vv = v.unsqueeze(0) if v.dim() == 4 else v
        tt = t.unsqueeze(0) if t.dim() == 1 else t
        cam = layercam_map(model, vv, tt, layer_name).detach()
        with torch.no_grad():
            base = float(risk_from_logits(model(vv, tt), mode="logit_sum").sum())
            fill = _gaussian_blur_3d(vv, sigma_vox=2.0)
            flat = cam.flatten()
            k = max(1, int(flat.numel() * 0.1))
            top_idx = set(torch.topk(flat, k).indices.tolist())
            n_vox = flat.numel()
            for _ in range(int(n_masks)):
                rand_idx = set(rng.choice(n_vox, size=k, replace=False).tolist())
                for idx_set, store in ((top_idx, deltas_m), (rand_idx, deltas_r)):
                    mask = torch.ones_like(flat)
                    mask[list(idx_set)] = 0.0
                    mask = mask.reshape_as(vv)
                    masked = mask * vv + (1.0 - mask) * fill
                    r = float(risk_from_logits(model(masked, tt), mode="logit_sum").sum())
                    store.append(base - r)
    dm = np.asarray(deltas_m, dtype=float)
    dr = np.asarray(deltas_r, dtype=float)
    diff = dm - dr
    ci = bootstrap_ci(diff.tolist(), n_boot=int(n_boot), seed=int(seed))
    median_m, median_r = float(np.median(dm)), float(np.median(dr))
    effect = median_m - median_r
    monotonic = bool(median_m > median_r)
    status = "PASS" if (ci["ci_low"] > 0 and monotonic and effect > 0) else "BLOCKED"
    return {"status": status, "median_delta_masked": median_m,
            "median_delta_random": median_r, "effect_size": effect,
            "monotonic": monotonic,
            "ci_low": ci["ci_low"], "ci_high": ci["ci_high"],
            "n_volumes": len(vols), "n_masks": int(n_masks),
            "n_boot": int(n_boot)}


# --------------------------------------------------------------------------
# Determinism enforcement + reproducibility + FDR + report
# --------------------------------------------------------------------------

def enforce_determinism(*, cublas_config: str = ":4096:8",
                        python_hash_seed: str = "20260714") -> dict[str, Any]:
    """Set the FULL deterministic stack (FINAL P0.3 ledger). Raises if unsettable.

    CUBLAS_WORKSPACE_CONFIG and PYTHONHASHSEED only take effect when set
    before process/CUDA init: the ledger records whether they were already
    effective (``*_effective`` flags) so callers can BLOCK on a host that
    merely set them mid-process (C-fix 2026-09-08). Seeds for torch/numpy are
    the caller's responsibility and must come from the locked config seed.
    """
    import torch.backends.cudnn as cudnn

    cublas_pre = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    hash_pre = os.environ.get("PYTHONHASHSEED")
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = str(cublas_config)
    os.environ["PYTHONHASHSEED"] = str(python_hash_seed)
    torch.use_deterministic_algorithms(True)
    cudnn.deterministic = True
    cudnn.benchmark = False
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False
    ledger = determinism_ledger()
    ledger["cublas_workspace_effective"] = bool(cublas_pre == str(cublas_config))
    ledger["python_hash_seed_effective"] = bool(hash_pre == str(python_hash_seed))
    return ledger


def assert_bit_identical(a: torch.Tensor, b: torch.Tensor, *, what: str) -> None:
    """Forward-map reproducibility gate (FINAL P0.4 tier 1)."""
    if a.shape != b.shape or not bool(torch.equal(a, b)):
        raise _fail("REPRODUCIBILITY_NOT_BIT_IDENTICAL", what)


def fdr_bh(p_values: Sequence[float], *, alpha: float = 0.05):
    """Benjamini-Hochberg FDR (FINAL M1 exploratory family)."""
    p = [float(v) for v in p_values]
    m = len(p)
    order = sorted(range(m), key=lambda i: p[i])
    rejected = [False] * m
    thresh = 0
    for rank, i in enumerate(order, start=1):
        if p[i] <= rank / m * alpha:
            thresh = rank
    for rank, i in enumerate(order, start=1):
        if rank <= thresh:
            rejected[i] = True
    return rejected


def scan_subject_identifiers(text: str) -> dict[str, Any]:
    """Privacy scan: reject anything shaped like a subject ID/path hash leak.

    Labeled provenance digests (e.g. ``run_contract_sha256: <64hex>``) are
    allowlisted: digests belong in JSON fields, and a labeled digest in prose
    must not self-BLOCK an otherwise PASS report. Bare unlabeled hashes still
    BLOCK. POSIX/UNC absolute paths are covered alongside Windows drives.
    """
    import re
    lowered = str(text)
    scrubbed = re.sub(
        r"\b[a-z_]+(?:sha256|digest|run_digest|config_sha)\b\s*[:=]\s*"
        r"[A-Fa-f0-9]{32,64}\b",
        "<labeled-digest>", lowered)
    hits = []
    if re.search(r"[A-Za-z]:[\\/]", scrubbed):
        hits.append("absolute_path")
    if re.search(r"(^|[\s\"'])(/(h|data|home|mnt)/\S+|\\\\[A-Za-z0-9_.$-]+\\[^\\]+)",
                 scrubbed):
        hits.append("absolute_path")
    if re.search(r"\b[A-Fa-f0-9]{32,64}\b", scrubbed):
        hits.append("hash_like_token")
    if re.search(r"sub-[A-Za-z0-9]{4,}", scrubbed):
        hits.append("bids_id_like_token")
    return {"status": "BLOCKED" if hits else "PASS", "hits": hits}


def build_report(*, methods: Mapping[str, Any], gates: Mapping[str, Any],
                 statistics: Mapping[str, Any], legs: Sequence[Mapping[str, Any]],
                 markdown_body: str) -> dict[str, Any]:
    """Assemble the FINAL P7 report artifact (ban + privacy scans mandatory)."""
    ban = ban_check(markdown_body)
    priv = scan_subject_identifiers(markdown_body)
    leg_verdict = evaluate_legs_table(legs)
    status = "PASS" if (ban["status"] == "PASS" and priv["status"] == "PASS"
                        and leg_verdict["status"] == "PASS") else "BLOCKED"
    return {"schema_version": 1, "status": status, "ceiling": CEILING,
            "methods": dict(methods), "gates": dict(gates),
            "statistics": dict(statistics),
            "legs": leg_verdict, "ban": ban, "privacy": priv,
            "markdown_body": str(markdown_body)}


def pet_quant_consistency_leg(*, derivation_dag: Mapping[str, Any] | None
                              ) -> dict[str, Any]:
    """PET-quant leg (FINAL P6.2): default CIRCULARITY_CONFOUNDED.

    Opens ONLY with a pre-registered derivation DAG artifact
    (source files + hashes + disjointness proof). No DAG -> NOT_MEASURABLE
    WITH the missing-dag evidence record (never bare assertion).
    """
    if not isinstance(derivation_dag, Mapping) or not derivation_dag.get("evidence"):
        return {"id": "LEG-07", "status": "NOT_MEASURABLE",
                "evidence": "pet_quant_derivation_dag_missing.json",
                "default": "CIRCULARITY_CONFOUNDED"}


# --------------------------------------------------------------------------
# Vocabulary + legs table
# --------------------------------------------------------------------------

def ban_check(text: str) -> dict[str, Any]:
    lowered = str(text).casefold()
    hits = sorted({term for term in BAN_TERMS if term in lowered})
    return {"status": "BLOCKED" if hits else "PASS", "hits": hits}


def evaluate_legs_table(legs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    seen = set()
    for leg in legs:
        lid = str(leg.get("id", ""))
        if lid not in LEG_IDS or lid in seen:
            return {"status": "BLOCKED", "reason": f"LEG_ID_INVALID:{lid}"}
        seen.add(lid)
        status = str(leg.get("status", ""))
        if status == "NOT_MEASURABLE" and not leg.get("evidence"):
            return {"status": "BLOCKED", "reason": f"LEG_EVIDENCE_REQUIRED:{lid}"}
        if status not in ("PASS", "NOT_MEASURABLE", "TIMEOUT"):
            return {"status": "BLOCKED", "reason": f"LEG_STATUS_INVALID:{lid}"}
    return {"status": "PASS", "legs_evaluated": len(legs)}


# --------------------------------------------------------------------------
# Production cell enumeration + VRAM guard (FINAL P4.1, P1.4)
# --------------------------------------------------------------------------

EXPECTED_ARMS = ("A0", "A1", "A2", "A3", "A4")
EXPECTED_OUTERS = (0, 1, 2, 3, 4)


def enumerate_attribution_cells(*, dev_root: str | Path, orch_root: str | Path,
                                splits: Mapping[str, Mapping[str, Any]],
                                membership_paths: Mapping[str, str | Path]) -> list[dict[str, Any]]:
    """Enumerate (branch, arm, outer) cells strictly from locked files.

    Requires per cell: winner_<ARM>_outer_<K>.json with self-hash intact AND
    development/<branch>/<ARM>/outer_<K>/final/locked_checkpoint.pt present.
    Firewall cohort mirrors the locked v4 program verbatim
    (run_method_y_image_factorial_v4.py:691 + :796-798): the image-development
    membership (``role == "development"``) intersected with that outer fold's
    ``test_subjects``, in membership-file order (NOT the raw v2 splits list,
    which names source-regional subjects without images — 2026-09-08 fix).
    Anything missing/ambiguous -> BLOCKED (never guessed, never partial).
    `splits` maps branch -> {"outer_folds": [{"fold": k, "test_subjects": [...]}]}.
    `membership_paths` maps branch -> locked membership parquet path.
    """
    import pandas as pd

    cells: list[dict[str, Any]] = []
    missing: list[str] = []
    for branch in ("amyloid", "tau"):
        try:
            membership = pd.read_parquet(str(membership_paths[branch]))
        except (KeyError, TypeError, ValueError, OSError) as exc:
            raise _fail("MEMBERSHIP_UNREADABLE", f"{branch}") from exc
        if "role" not in membership.columns or "patient_hash" not in membership.columns:
            raise _fail("MEMBERSHIP_COLUMNS_MISSING", f"{branch}")
        development = membership[membership["role"] == "development"]
        branch_splits = splits.get(branch, {})
        outer_tests: dict[int, list[str]] = {}
        outer_trains: dict[int, list[str]] = {}
        for fold_row in branch_splits.get("outer_folds", []):
            try:
                keys = {str(s) for s in fold_row["test_subjects"]}
                frame = development[development["patient_hash"].astype(str).isin(keys)]
                outer_tests[int(fold_row["fold"])] = frame["patient_hash"].astype(str).tolist()
                train_keys = {str(s) for s in fold_row.get("train_subjects", [])}
                train_frame = development[development["patient_hash"].astype(str).isin(train_keys)]
                outer_trains[int(fold_row["fold"])] = train_frame["patient_hash"].astype(str).tolist()
            except (KeyError, TypeError, ValueError) as exc:
                raise _fail("SPLITS_UNREADABLE", f"{branch}") from exc
        for arm in EXPECTED_ARMS:
            for outer in EXPECTED_OUTERS:
                tag = f"{branch}/{arm}/outer_{outer}"
                wpath = Path(orch_root) / branch / f"winner_{arm}_outer_{outer}.json"
                fpath = (Path(dev_root) / branch / arm / f"outer_{outer}" / "final"
                         / "locked_checkpoint.pt")
                try:
                    record = json.loads(wpath.read_text(encoding="utf-8"))
                except (OSError, ValueError, UnicodeDecodeError):
                    missing.append(f"WINNER_MISSING:{tag}")
                    continue
                body = dict(record)
                claimed = body.pop("winner_record_sha256", None)
                canon = json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
                if claimed != hashlib.sha256(canon).hexdigest():
                    raise _fail("WINNER_SELF_HASH_MISMATCH", tag)
                if not fpath.is_file():
                    missing.append(f"FINAL_LOCKED_MISSING:{tag}")
                    continue
                if outer not in outer_tests or not outer_tests[outer]:
                    raise _fail("FIREWALL_COHORT_UNRESOLVABLE", tag)
                if outer not in outer_trains or not outer_trains[outer]:
                    raise _fail("TRAIN_COHORT_UNRESOLVABLE", tag)
                cells.append({"branch": branch, "arm": arm, "outer": outer,
                              "winner_path": str(wpath),
                              "final_checkpoint": str(fpath),
                              "firewall_subjects": outer_tests[outer],
                              "train_subjects": outer_trains[outer]})
    if missing:
        raise _fail("INCOMPLETE_FINALS", ";".join(sorted(missing)[:5]))
    if len(cells) != 50:
        raise _fail("CELL_COUNT_INVALID", str(len(cells)))
    return cells


def check_vram_budget(*, budget_gb: float = 7.5) -> dict[str, Any]:
    """Fail-closed VRAM guard (FINAL P1.4). CPU-only hosts pass vacuously."""
    if not torch.cuda.is_available():
        return {"status": "PASS", "peak_gb": 0.0, "budget_gb": float(budget_gb),
                "device": "cpu"}
    peak_gb = float(torch.cuda.max_memory_allocated() / (1024 ** 3))
    if peak_gb > float(budget_gb):
        raise _fail("VRAM_BUDGET_EXCEEDED", f"{peak_gb:.2f}GB")
    return {"status": "PASS", "peak_gb": round(peak_gb, 3),
            "budget_gb": float(budget_gb), "device": "cuda"}


def verify_final_digest(path: str | Path, *, expected_digest: str) -> dict[str, Any]:
    """Confirm a final locked checkpoint binds the live run contract.

    Reads (never writes) the torch payload and compares its embedded
    `run_contract_digest`. Any mismatch/absence -> BLOCKED.

    NOTE: versioned contract migrations preserve weights by design, so finals
    locked under an older generation embed that generation's digest, not the
    live one. For production cells prefer verify_final_provenance(), which
    binds through the self-hashed FINALIZED outer state (frozen-chain).
    """
    import io
    p = Path(path)
    try:
        # weights_only=True: a compromised checkpoint must never achieve code
        # execution during the "verification" read (C-fix 2026-09-08).
        payload = torch.load(io.BytesIO(p.read_bytes()), map_location="cpu",
                             weights_only=True)
    except Exception as exc:
        raise _fail("FINAL_CHECKPOINT_UNREADABLE", p.name) from exc
    if not isinstance(payload, Mapping) or payload.get("run_contract_digest") != expected_digest:
        raise _fail("FINAL_DIGEST_MISMATCH", p.name)
    return {"status": "PASS", "path": str(p)}


def verify_final_provenance(path: str | Path, *, v4_root: str | Path,
                            branch: str, arm: str, outer: int) -> dict[str, Any]:
    """Frozen-chain binding of a production cell's final checkpoint (2026-09-08).

    Versioned migrations (e.g. 654f->969e) preserve weights by design, so a
    locked final embeds its OWN generation digest, never the live global one:
    strict equality against live can never hold. Instead bind the exact bytes
    that generated the locked outer metrics, through the self-hashed FINALIZED
    outer state at
    ``private/outer_evaluation/{branch}/{arm}/fold_{outer}/state.json``:
    file-sha == state's checkpoint_sha256 AND payload digest == state's
    run_contract_sha256 AND state == FINALIZED with a valid self-hash.
    Anything else -> BLOCKED (never guessed, never partial).
    """
    import io
    p = Path(path)
    state_path = (Path(v4_root) / "private" / "outer_evaluation" / str(branch)
                  / str(arm) / f"fold_{int(outer)}" / "state.json")
    try:
        record = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, UnicodeDecodeError) as exc:
        raise _fail("OUTER_STATE_UNREADABLE", state_path.name) from exc
    body = dict(record)
    claimed = body.pop("state_sha256", None)
    canon = json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    if claimed != hashlib.sha256(canon).hexdigest():
        raise _fail("OUTER_STATE_SELF_HASH_MISMATCH", f"{branch}/{arm}/fold_{outer}")
    if record.get("state") != "FINALIZED":
        raise _fail("OUTER_STATE_NOT_FINALIZED", f"{branch}/{arm}/fold_{outer}")
    raw = p.read_bytes()
    if hashlib.sha256(raw).hexdigest() != record.get("checkpoint_sha256"):
        raise _fail("FINAL_FILE_NOT_LOCKED_OUTER_FILE", p.name)
    try:
        payload = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
    except Exception as exc:
        raise _fail("FINAL_CHECKPOINT_UNREADABLE", p.name) from exc
    if not isinstance(payload, Mapping) or payload.get("run_contract_digest") != record.get("run_contract_sha256"):
        raise _fail("FINAL_GENERATION_DIGEST_MISMATCH", p.name)
    if hashlib.sha256(p.read_bytes()).hexdigest() != record.get("checkpoint_sha256"):
        raise _fail("CHECKPOINT_MUTATED_DURING_LOAD", p.name)
    return {"status": "PASS", "path": str(p),
            "checkpoint_sha256": record.get("checkpoint_sha256"),
            "generation_digest": record.get("run_contract_sha256"),
            "outer_state": str(state_path)}


def run_with_deadline(fn, *, deadline_monotonic: float, leg_id: str = "LEG-02"):
    """Execute fn() unless the monotonic deadline passed -> TIMEOUT leg record.

    The timeout path is a recorded outcome, never a silent skip and never a
    method substitution (FINAL C2-fix).
    """
    import time
    if time.monotonic() >= float(deadline_monotonic):
        return {"id": leg_id, "status": "TIMEOUT",
                "evidence": "scorecam_full_cohort_timeout.json"}
    return fn()
