from __future__ import annotations

from typing import Any

import numpy as np


class ImageSurvivalV3Error(ValueError):
    """Raised when a censoring-aware image survival contract is violated."""


def robust_pattern_normalize(
    volume: np.ndarray,
    brain_mask: np.ndarray,
    *,
    lower_percentile: float = 1.0,
    upper_percentile: float = 99.0,
    clip: tuple[float, float] = (-5.0, 5.0),
) -> np.ndarray:
    """Remove arbitrary scan scale while preserving within-brain spatial pattern.

    This transform intentionally forfeits quantitative SUVR/Centiloid/CenTauR
    interpretation.  It is appropriate only for the explicitly labelled
    pattern-normalized exploratory image branch.
    """

    data = np.asarray(volume, dtype=np.float32)
    mask = np.asarray(brain_mask, dtype=bool)
    if data.shape != mask.shape:
        raise ImageSurvivalV3Error("BRAIN_MASK_SHAPE_MISMATCH")
    if not np.isfinite(data).all():
        raise ImageSurvivalV3Error("NONFINITE_IMAGE_CANNOT_BE_NORMALIZED")
    values = data[mask]
    if values.size < 8:
        raise ImageSurvivalV3Error("INSUFFICIENT_BRAIN_MASK_VOXELS")
    lo, hi = np.percentile(values, [float(lower_percentile), float(upper_percentile)])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        raise ImageSurvivalV3Error("DEGENERATE_PATTERN_INTENSITY_RANGE")
    clipped = np.clip(data, lo, hi)
    in_brain = clipped[mask]
    median = float(np.median(in_brain))
    q25, q75 = np.percentile(in_brain, [25.0, 75.0])
    robust_scale = float((q75 - q25) / 1.3489795003921634)
    if not np.isfinite(robust_scale) or robust_scale <= 1e-6:
        robust_scale = float(np.std(in_brain))
    if not np.isfinite(robust_scale) or robust_scale <= 1e-6:
        raise ImageSurvivalV3Error("DEGENERATE_PATTERN_ROBUST_SCALE")
    normalized = np.zeros_like(data, dtype=np.float32)
    normalized[mask] = (clipped[mask] - median) / robust_scale
    normalized[mask] = np.clip(normalized[mask], float(clip[0]), float(clip[1]))
    if not np.isfinite(normalized).all():
        raise ImageSurvivalV3Error("NORMALIZED_IMAGE_NONFINITE")
    return normalized


def survival_from_logits(logits: Any):
    import torch

    if logits.ndim != 2:
        raise ImageSurvivalV3Error("HAZARD_LOGITS_MUST_BE_2D")
    hazards = torch.sigmoid(logits)
    return torch.cumprod(1.0 - hazards, dim=1)


def discrete_time_survival_nll(
    logits: Any,
    event: Any,
    observed_time: Any,
    bin_edges: Any,
    sample_weight: Any | None = None,
):
    """Exact minibatchable discrete-time hazard likelihood with censoring."""

    import torch
    import torch.nn.functional as functional

    if logits.ndim != 2:
        raise ImageSurvivalV3Error("HAZARD_LOGITS_MUST_BE_2D")
    events = event.reshape(-1).to(device=logits.device, dtype=logits.dtype)
    times = observed_time.reshape(-1).to(device=logits.device, dtype=logits.dtype)
    edges = bin_edges.reshape(-1).to(device=logits.device, dtype=logits.dtype)
    if logits.shape[0] != events.numel() or events.numel() != times.numel():
        raise ImageSurvivalV3Error("SURVIVAL_BATCH_LENGTH_MISMATCH")
    if edges.numel() != logits.shape[1] + 1:
        raise ImageSurvivalV3Error("HAZARD_BIN_EDGE_COUNT_MISMATCH")
    if not torch.isfinite(logits).all() or not torch.isfinite(times).all():
        raise ImageSurvivalV3Error("SURVIVAL_BATCH_NONFINITE")
    if not torch.all((events == 0) | (events == 1)):
        raise ImageSurvivalV3Error("EVENT_MUST_BE_BINARY")
    if not torch.all(edges[1:] > edges[:-1]) or not torch.all(times > edges[0]):
        raise ImageSurvivalV3Error("INVALID_SURVIVAL_TIME_OR_BIN_EDGES")
    if torch.any((events == 1) & (times > edges[-1])):
        raise ImageSurvivalV3Error("EVENT_AFTER_LAST_HAZARD_BIN")

    upper_edges = edges[1:]
    interval = torch.bucketize(times, upper_edges, right=False)
    interval = torch.clamp(interval, max=logits.shape[1] - 1)
    grid = torch.arange(logits.shape[1], device=logits.device).reshape(1, -1)
    # An event contributes survival through earlier intervals and the event
    # hazard in its interval.  A censor contributes only fully observed prior
    # intervals; censoring beyond the last edge contributes every interval.
    event_rows = events.reshape(-1, 1) == 1
    at_risk = torch.where(
        event_rows,
        grid <= interval.reshape(-1, 1),
        grid < torch.bucketize(times, upper_edges, right=False).reshape(-1, 1),
    )
    event_cell = event_rows & (grid == interval.reshape(-1, 1))
    log_likelihood = torch.where(
        event_cell,
        functional.logsigmoid(logits),
        functional.logsigmoid(-logits),
    )
    per_subject = -(log_likelihood * at_risk.to(logits.dtype)).sum(dim=1)
    if sample_weight is None:
        return per_subject.mean()
    weight = sample_weight.reshape(-1).to(device=logits.device, dtype=logits.dtype)
    if weight.numel() != per_subject.numel() or not torch.isfinite(weight).all():
        raise ImageSurvivalV3Error("INVALID_SAMPLE_WEIGHT")
    if torch.any(weight <= 0):
        raise ImageSurvivalV3Error("SAMPLE_WEIGHT_MUST_BE_POSITIVE")
    return (per_subject * weight).sum() / weight.sum()


def _group_count(channels: int) -> int:
    for groups in (8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


def _modules():
    import torch
    from torch import nn

    class ResidualBlock3D(nn.Module):
        def __init__(self, in_channels: int, out_channels: int, stride: int) -> None:
            super().__init__()
            self.main = nn.Sequential(
                nn.Conv3d(
                    in_channels,
                    out_channels,
                    kernel_size=3,
                    stride=stride,
                    padding=1,
                    bias=False,
                ),
                nn.GroupNorm(_group_count(out_channels), out_channels),
                nn.SiLU(inplace=True),
                nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.GroupNorm(_group_count(out_channels), out_channels),
            )
            self.skip = (
                nn.Identity()
                if in_channels == out_channels and stride == 1
                else nn.Conv3d(
                    in_channels, out_channels, kernel_size=1, stride=stride, bias=False
                )
            )
            self.activation = nn.SiLU(inplace=True)

        def forward(self, value):
            return self.activation(self.main(value) + self.skip(value))

    class _PatternSurvivalNet(nn.Module):
        def __init__(
            self,
            *,
            tabular_dim: int,
            hazard_bins: int,
            base_channels: int = 8,
            embedding_dim: int = 64,
            tabular_hidden_dim: int = 32,
            dropout: float = 0.25,
        ) -> None:
            super().__init__()
            channels = int(base_channels)
            self.stem = nn.Sequential(
                nn.Conv3d(1, channels, kernel_size=5, stride=2, padding=2, bias=False),
                nn.GroupNorm(_group_count(channels), channels),
                nn.SiLU(inplace=True),
            )
            self.encoder = nn.Sequential(
                ResidualBlock3D(channels, channels * 2, 2),
                ResidualBlock3D(channels * 2, channels * 4, 2),
                ResidualBlock3D(channels * 4, channels * 8, 2),
                ResidualBlock3D(channels * 8, channels * 8, 2),
            )
            self.pool = nn.AdaptiveAvgPool3d(1)
            self.image_projection = nn.Sequential(
                nn.Flatten(),
                nn.Linear(channels * 8, int(embedding_dim)),
                nn.SiLU(inplace=True),
            )
            self.tabular = nn.Sequential(
                nn.Linear(int(tabular_dim), int(tabular_hidden_dim)),
                nn.SiLU(inplace=True),
                nn.Dropout(float(dropout)),
                nn.Linear(int(tabular_hidden_dim), int(tabular_hidden_dim)),
                nn.SiLU(inplace=True),
            )
            self.head = nn.Sequential(
                nn.Dropout(float(dropout)),
                nn.Linear(int(embedding_dim) + int(tabular_hidden_dim), int(hazard_bins)),
            )

        def forward(self, image, tabular, *, return_embedding: bool = False):
            encoded = self.encoder(self.stem(image))
            embedding = self.image_projection(self.pool(encoded))
            tabular_embedding = self.tabular(tabular)
            logits = self.head(torch.cat([embedding, tabular_embedding], dim=1))
            return (logits, embedding) if return_embedding else logits

    return _PatternSurvivalNet


PatternSurvivalNet = _modules()


def _tabular_module():
    from torch import nn

    class _TabularSurvivalNet(nn.Module):
        def __init__(
            self,
            *,
            tabular_dim: int,
            hazard_bins: int,
            tabular_hidden_dim: int = 32,
            dropout: float = 0.25,
        ) -> None:
            super().__init__()
            hidden = int(tabular_hidden_dim)
            self.network = nn.Sequential(
                nn.Linear(int(tabular_dim), hidden),
                nn.SiLU(inplace=True),
                nn.Dropout(float(dropout)),
                nn.Linear(hidden, hidden),
                nn.SiLU(inplace=True),
                nn.Dropout(float(dropout)),
                nn.Linear(hidden, int(hazard_bins)),
            )

        def forward(self, tabular, *, return_embedding: bool = False):
            value = tabular
            for module in list(self.network.children())[:-1]:
                value = module(value)
            logits = self.network[-1](value)
            return (logits, value) if return_embedding else logits

    return _TabularSurvivalNet


TabularSurvivalNet = _tabular_module()
