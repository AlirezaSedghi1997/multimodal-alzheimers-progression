from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from typing import Any

from .image_survival_v3 import PatternSurvivalNet, _group_count


class ImageFactorialV4Error(ValueError):
    pass


@dataclass(frozen=True)
class FactorialArm:
    arm_id: str
    encoder_family: str
    augmentation_profile: str
    analysis_role: str
    allowed_width_multipliers: tuple[float, ...]


FACTORIAL_ARMS: dict[str, FactorialArm] = {
    "A0": FactorialArm(
        "A0", "GROUPNORM_RESIDUAL_TINY_3D", "NONE", "LOCKED_V3_BASELINE", (1.0,)
    ),
    "A1": FactorialArm(
        "A1", "GROUPNORM_RESIDUAL_TINY_3D", "PET_SAFE_V1", "AUGMENTATION_EFFECT", (1.0,)
    ),
    "A2": FactorialArm(
        "A2", "RESNET10_3D_GROUPNORM", "NONE", "PRIMARY_ARCHITECTURE_EFFECT", (0.25,)
    ),
    "A3": FactorialArm(
        "A3", "RESNET10_3D_GROUPNORM", "PET_SAFE_V1", "PRIMARY_FACTORIAL_COMBINATION", (0.25,)
    ),
    "A4": FactorialArm(
        "A4",
        "RESNET18_3D_GROUPNORM",
        "PET_SAFE_V1",
        "SECONDARY_ARCHITECTURE_CHALLENGER",
        (0.25, 0.5),
    ),
}


def matched_arm_differences(left: str, right: str) -> tuple[str, ...]:
    try:
        a, b = FACTORIAL_ARMS[left], FACTORIAL_ARMS[right]
    except KeyError as exc:
        raise ImageFactorialV4Error(f"UNKNOWN_FACTORIAL_ARM:{exc.args[0]}") from exc
    fields = ("encoder_family", "augmentation_profile")
    return tuple(field for field in fields if getattr(a, field) != getattr(b, field))


def select_inner_only_factorial_arm(
    evidence: dict[str, list[float]], *, complexity_ranks: dict[str, int]
) -> dict[str, Any]:
    """Lock the family preference without consulting any outer-test result."""

    expected = set(FACTORIAL_ARMS)
    if set(evidence) != expected or set(complexity_ranks) != expected:
        raise ImageFactorialV4Error("ALL_FIVE_FACTORIAL_ARMS_REQUIRED")
    summaries: dict[str, dict[str, Any]] = {}
    for arm_id in FACTORIAL_ARMS:
        values = [float(value) for value in evidence[arm_id]]
        if len(values) < 2 or not all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in values):
            raise ImageFactorialV4Error(f"INNER_ARM_EVIDENCE_INVALID:{arm_id}")
        mean = float(statistics.fmean(values))
        standard_error = float(statistics.stdev(values) / math.sqrt(len(values)))
        summaries[arm_id] = {
            "independent_outer_partition_summary_count": len(values),
            "mean_ibs": mean,
            "standard_error": standard_error,
            "complexity_rank": int(complexity_ranks[arm_id]),
            "outer_partition_mean_inner_ibs": values,
        }
    best_arm = min(summaries, key=lambda arm: summaries[arm]["mean_ibs"])
    threshold = float(
        summaries[best_arm]["mean_ibs"] + summaries[best_arm]["standard_error"]
    )
    eligible = [
        arm_id
        for arm_id in FACTORIAL_ARMS
        if float(summaries[arm_id]["mean_ibs"]) <= threshold
    ]
    selected = min(
        eligible,
        key=lambda arm: (
            int(summaries[arm]["complexity_rank"]),
            float(summaries[arm]["mean_ibs"]),
            arm,
        ),
    )
    return {
        "selection_metric": "IPCW_INTEGRATED_BRIER_SCORE",
        "selection_rule": "ONE_STANDARD_ERROR_THEN_LOWEST_COMPLEXITY",
        "outer_test_consulted": False,
        "best_mean_arm": best_arm,
        "one_standard_error_threshold": threshold,
        "eligible_within_one_standard_error": eligible,
        "selected_arm": selected,
        "arm_summaries": summaries,
    }


def _resnet_modules():
    import torch
    from torch import nn

    class BasicBlock3DGroupNorm(nn.Module):
        expansion = 1

        def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
            super().__init__()
            self.conv1 = nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False,
            )
            self.norm1 = nn.GroupNorm(_group_count(out_channels), out_channels)
            self.activation = nn.SiLU(inplace=True)
            self.conv2 = nn.Conv3d(
                out_channels, out_channels, kernel_size=3, padding=1, bias=False
            )
            self.norm2 = nn.GroupNorm(_group_count(out_channels), out_channels)
            if stride != 1 or in_channels != out_channels:
                self.skip = nn.Sequential(
                    nn.Conv3d(
                        in_channels,
                        out_channels,
                        kernel_size=1,
                        stride=stride,
                        bias=False,
                    ),
                    nn.GroupNorm(_group_count(out_channels), out_channels),
                )
            else:
                self.skip = nn.Identity()

        def forward(self, value):
            residual = self.skip(value)
            value = self.activation(self.norm1(self.conv1(value)))
            value = self.norm2(self.conv2(value))
            return self.activation(value + residual)

    class ResNetEncoder3DGroupNorm(nn.Module):
        def __init__(self, layers: tuple[int, int, int, int], base_channels: int) -> None:
            super().__init__()
            self.in_channels = int(base_channels)
            self.stem = nn.Sequential(
                nn.Conv3d(
                    1,
                    self.in_channels,
                    kernel_size=7,
                    stride=2,
                    padding=3,
                    bias=False,
                ),
                nn.GroupNorm(_group_count(self.in_channels), self.in_channels),
                nn.SiLU(inplace=True),
                nn.MaxPool3d(kernel_size=3, stride=2, padding=1),
            )
            widths = tuple(int(base_channels * (2**index)) for index in range(4))
            self.stages = nn.Sequential(
                self._make_stage(widths[0], layers[0], stride=1),
                self._make_stage(widths[1], layers[1], stride=2),
                self._make_stage(widths[2], layers[2], stride=2),
                self._make_stage(widths[3], layers[3], stride=2),
            )
            self.pool = nn.AdaptiveAvgPool3d(1)
            self.output_channels = widths[-1]

        def _make_stage(self, out_channels: int, block_count: int, stride: int):
            blocks = [BasicBlock3DGroupNorm(self.in_channels, out_channels, stride)]
            self.in_channels = out_channels
            blocks.extend(
                BasicBlock3DGroupNorm(self.in_channels, out_channels, 1)
                for _ in range(1, int(block_count))
            )
            return nn.Sequential(*blocks)

        def forward(self, image):
            return self.pool(self.stages(self.stem(image))).flatten(1)

    class ResNetPatternSurvivalNet(nn.Module):
        def __init__(
            self,
            *,
            layers: tuple[int, int, int, int],
            width_multiplier: float,
            tabular_dim: int,
            hazard_bins: int,
            embedding_dim: int,
            tabular_hidden_dim: int,
            dropout: float,
        ) -> None:
            super().__init__()
            base_channels = max(8, int(round(64.0 * float(width_multiplier))))
            self.encoder = ResNetEncoder3DGroupNorm(layers, base_channels)
            self.image_projection = nn.Sequential(
                nn.Linear(self.encoder.output_channels, int(embedding_dim)),
                nn.SiLU(inplace=True),
            )
            # This branch and the survival head deliberately mirror v3 Tiny.
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
            embedding = self.image_projection(self.encoder(image))
            tabular_embedding = self.tabular(tabular)
            logits = self.head(torch.cat((embedding, tabular_embedding), dim=1))
            return (logits, embedding) if return_embedding else logits

    return ResNetPatternSurvivalNet


ResNetPatternSurvivalNet = _resnet_modules()


def build_factorial_image_model(
    arm_id: str,
    model_config: dict[str, Any],
    *,
    tabular_dim: int,
    hazard_bins: int,
):
    try:
        arm = FACTORIAL_ARMS[arm_id]
    except KeyError as exc:
        raise ImageFactorialV4Error(f"UNKNOWN_FACTORIAL_ARM:{arm_id}") from exc
    common = {
        "tabular_dim": int(tabular_dim),
        "hazard_bins": int(hazard_bins),
        "embedding_dim": int(model_config.get("embedding_dim", 64)),
        "tabular_hidden_dim": int(model_config.get("tabular_hidden_dim", 32)),
        "dropout": float(model_config.get("dropout", 0.25)),
    }
    if arm.encoder_family == "GROUPNORM_RESIDUAL_TINY_3D":
        return PatternSurvivalNet(
            **common,
            base_channels=int(model_config.get("base_channels", 8)),
        )
    width = float(model_config.get("resnet_width_multiplier", 0.25))
    if width not in arm.allowed_width_multipliers:
        raise ImageFactorialV4Error(
            f"WIDTH_NOT_ALLOWED_FOR_{arm_id}:{width}"
        )
    layers = (
        (1, 1, 1, 1)
        if arm.encoder_family == "RESNET10_3D_GROUPNORM"
        else (2, 2, 2, 2)
    )
    return ResNetPatternSurvivalNet(
        layers=layers,
        width_multiplier=width,
        **common,
    )
