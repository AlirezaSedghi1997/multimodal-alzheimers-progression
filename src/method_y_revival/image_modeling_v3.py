from __future__ import annotations

import math
from typing import Any, Iterable

import numpy as np

from .image_survival_v3 import PatternSurvivalNet, TabularSurvivalNet


class ImageModelingV3Error(ValueError):
    pass


LOCKED_VARIANTS = ("scalar_tabular", "scalar_tabular_plus_pattern_image")


def build_model(
    variant: str,
    model_config: dict[str, Any],
    tabular_dim: int,
    hazard_bins: int,
):
    if variant == "scalar_tabular":
        return TabularSurvivalNet(
            tabular_dim=int(tabular_dim),
            hazard_bins=int(hazard_bins),
            tabular_hidden_dim=int(model_config.get("tabular_hidden_dim", 32)),
            dropout=float(model_config.get("dropout", 0.25)),
        )
    if variant == "scalar_tabular_plus_pattern_image":
        return PatternSurvivalNet(
            tabular_dim=int(tabular_dim),
            hazard_bins=int(hazard_bins),
            base_channels=int(model_config.get("base_channels", 8)),
            embedding_dim=int(model_config.get("embedding_dim", 64)),
            tabular_hidden_dim=int(model_config.get("tabular_hidden_dim", 32)),
            dropout=float(model_config.get("dropout", 0.25)),
        )
    raise ImageModelingV3Error(f"UNKNOWN_LOCKED_VARIANT:{variant}")


def survival_at_times(
    logits: Any,
    bin_edges: Iterable[float],
    evaluation_times: Iterable[float],
) -> Any:
    """Piecewise-constant hazard interpolation over locked bin edges.

    Guarantees:
    - monotone non-increasing in time
    - values in [0, 1]
    - boundary equality: at each bin edge, equals discrete cumulative survival
    - zero/one hazard stability (no NaNs)
    - raises if grid outside locked support
    """
    import torch

    if logits.ndim != 2:
        raise ImageModelingV3Error("HAZARD_LOGITS_MUST_BE_2D")
    edges = torch.as_tensor(list(bin_edges), dtype=torch.float32, device=logits.device)
    times = torch.as_tensor(list(evaluation_times), dtype=torch.float32, device=logits.device)
    if edges.numel() != logits.shape[1] + 1:
        raise ImageModelingV3Error("HAZARD_BIN_EDGE_COUNT_MISMATCH")
    if not torch.all(edges[1:] > edges[:-1]):
        raise ImageModelingV3Error("INVALID_BIN_EDGES_NOT_STRICTLY_INCREASING")
    if torch.any(times < edges[0]) or torch.any(times > edges[-1]):
        raise ImageModelingV3Error("EVALUATION_TIME_OUTSIDE_LOCKED_SUPPORT")
    if not torch.isfinite(logits).all() or not torch.isfinite(edges).all() or not torch.isfinite(times).all():
        raise ImageModelingV3Error("SURVIVAL_INPUT_NONFINITE")

    hazards = torch.sigmoid(logits)  # [N, K] in (0,1)
    # clamp for numerical stability at 0/1
    hazards = torch.clamp(hazards, min=1e-7, max=1 - 1e-7)
    one_minus_h = 1.0 - hazards  # [N, K]
    # cumulative product for discrete boundaries
    cum_survival = torch.cumprod(one_minus_h, dim=1)  # [N, K]
    # survival at edge 0 is 1.0, at edge k (k>=1) is cum_survival[:, k-1]
    # Build survival boundary matrix: [N, K+1]
    N, K = logits.shape
    boundary = torch.ones((N, K + 1), device=logits.device, dtype=torch.float32)
    boundary[:, 1:] = cum_survival

    # For each evaluation time, find interval k where edges[k] <= t < edges[k+1]
    # bucket = index of first edge > t  -> interval = bucket -1  but we need handling for exact edge
    # Use bucketize with right=False: returns index of first upper edge > t
    # For t == edges[-1], bucket = K+1? Actually bucket returns len(edges) if t >= last edge. But we clamp above ensures t <= edges[-1]; for t==edges[-1], bucket == K+1 (since >?). Need to clamp.
    # Alternative: use searchsorted with right=False
    # torch.bucketize with right=False: bucket = count of edges <= t ? Let's test logic.
    # Simpler: compute interval index via bucketize on upper edges
    upper = edges[1:]  # size K
    # interval idx = number of upper edges <= t  (with right=False? need exact)
    # For t at edge boundary, we want fraction 0 in next interval, survival == boundary at that edge.
    # So we want interval = torch.bucketize(t, edges[1:-1]?) Hmm.
    # Safer: interval = torch.searchsorted(edges, times, right=False) -1? But searchsorted returns insertion point.
    # Use bucketize: interval = torch.bucketize(times, edges[1:], right=False) is similar to number of upper edges <= t?
    # Let's implement manual: for each t, edges[k] <= t < edges[k+1]
    # bucket = torch.bucketize(times, edges, right=False) gives first index i where edges[i] > t? Actually torch.bucketize with right=False returns first i where edges[i] > t? Wait docs: bucketize(input, boundaries, right=False) -> boundaries[b-1] <= input < boundaries[b]
    # So bucket index b satisfies boundaries[b-1] <= input < boundaries[b]. For input == boundaries[-1], b = len(boundaries) (out of range). For input < boundaries[0], b=0.
    # We have edges size K+1. For valid t in [edges[0], edges[-1]], bucket in [1, K+1].
    # interval k = bucket -1  where t in [edges[k], edges[k+1]) except for t == edges[-1] where bucket=K+1 => k=K (out of hazard range). But t==edges[-1] survival = cum_survival[:,-1] = boundary[:,K].
    # So for bucket == K+1 (t == last edge), we set output = boundary[:,K]
    # For bucket in 1..K, interval k = bucket-1, fraction = (t - edges[k])/(edges[k+1]-edges[k]), survival = boundary[:,k] * (1 - h_k)^fraction ??? But need mapping for hazard indexing.
    # Actually hazard k corresponds to interval [edges[k], edges[k+1]) for k=0..K-1. So interval k = bucket-1 for bucket <=K, and for bucket==K+1, we are at last boundary.
    result = torch.empty((N, times.numel()), device=logits.device, dtype=torch.float32)
    bucket = torch.bucketize(times, edges, right=False)  # [T]
    for t_idx in range(times.numel()):
        b = int(bucket[t_idx].item())
        t = float(times[t_idx].item())
        if b == 0:
            # t < edges[0] should have been rejected, but clamp
            result[:, t_idx] = 1.0
        elif b == len(edges):
            # t == last edge (or slightly beyond due to float)
            result[:, t_idx] = boundary[:, -1]
        else:
            k = b - 1  # interval index 0..K-1, t in [edges[k], edges[k+1])
            if k >= K:
                result[:, t_idx] = boundary[:, -1]
            else:
                # fraction within interval
                edge_lo = float(edges[k].item())
                edge_hi = float(edges[k + 1].item())
                denom = edge_hi - edge_lo
                f = (t - edge_lo) / denom if denom > 0 else 0.0
                f = max(0.0, min(1.0, f))
                if f == 0.0:
                    result[:, t_idx] = boundary[:, k]
                elif f == 1.0:
                    result[:, t_idx] = boundary[:, k + 1]
                else:
                    # boundary[:,k] * (1 - h_k)^f  but boundary already = product up to k-1
                    # Check: boundary[:,k] = cum up to k-1, (1 - h_k)^f interpolates within interval
                    # cum up to k-1 * (1 - h_k)^f  should equal? For f=0 -> boundary[k], f=1 -> boundary[k+1]
                    # Since boundary[k+1] = boundary[k] * (1 - h_k)
                    # So indeed survival = boundary[k] * (1 - h_k)^f
                    h_k = one_minus_h[:, k]  # 1 - h_k already? Actually one_minus_h = 1 - hazard
                    # hazards are in one_minus_h, so (1 - h_k) = one_minus_h[:,k]
                    result[:, t_idx] = boundary[:, k] * torch.pow(one_minus_h[:, k], f)
        # ensure [0,1]
    result = torch.clamp(result, min=0.0, max=1.0)
    # monotonicity: ensure non-increasing across sorted times
    # But evaluation_times may not be sorted; we enforce per-time independently, monotonic across sorted order is guaranteed by construction if hazard constant
    # Verify no NaN/Inf
    if not torch.isfinite(result).all():
        raise ImageModelingV3Error("SURVIVAL_AT_TIMES_NONFINITE")
    # boundary equality check (optional, caller can test)
    return result


def build_run_contract(
    *,
    branch: str,
    variant: str,
    outer_fold: int,
    inner_fold_or_final: str,
    candidate_id: str,
    model_config: dict[str, Any],
    optimizer_config: dict[str, Any],
    hazard_bin_edges: list[float],
    feature_columns: list[str],
    tabular_transformer: dict[str, Any] | None,
    membership_file_sha256: str,
    cache_file_sha256: str,
    split_file_sha256: str,
    readiness_run_contract_sha256: str,
    seed: int,
    device_math_policy: str,
) -> dict[str, Any]:
    if variant not in LOCKED_VARIANTS:
        raise ImageModelingV3Error(f"UNKNOWN_LOCKED_VARIANT:{variant}")
    if len(membership_file_sha256) != 64 or len(cache_file_sha256) != 64 or len(split_file_sha256) != 64 or len(readiness_run_contract_sha256) != 64:
        raise ImageModelingV3Error("RUN_CONTRACT_SHA_REQUIRED")
    return {
        "schema_version": "3.0",
        "branch": branch,
        "variant": variant,
        "outer_fold": int(outer_fold),
        "inner_fold_or_final": str(inner_fold_or_final),
        "candidate_id": str(candidate_id),
        "model_config": dict(model_config),
        "optimizer_config": dict(optimizer_config),
        "hazard_bin_edges": list(hazard_bin_edges),
        "feature_columns": list(feature_columns),
        "tabular_transformer": tabular_transformer,
        "membership_file_sha256": membership_file_sha256,
        "cache_file_sha256": cache_file_sha256,
        "split_file_sha256": split_file_sha256,
        "readiness_run_contract_sha256": readiness_run_contract_sha256,
        "seed": int(seed),
        "device_math_policy": str(device_math_policy),
    }
