"""Pure Method X 5+2 battery-terminal predicate.

This module is intentionally standard-library only.  The formal XAI preflight
and the read-only readiness audit both call this function, so the auditor can
exercise the production predicate without importing torch or hashing the
multi-gigabyte MRI cache merely to reach an inline gate.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any


EFFECTIVE_FIVE = (
    "resnet10_3d",
    "resnet18_3d",
    "resnet34_3d",
    "densenet121_3d",
    "mednext_3d_s",
)
EXCLUDED_BACKBONES = ("resnet50_3d", "efficientnet3d_b0")
TERMINAL_TRAINING_STATUSES = frozenset(
    {
        "COMPLETED",
        "COMPLETED_VALIDATED_REPAIRED",
        "COMPLETED_VALIDATED_RECONCILED",
    }
)


class MethodXBatteryGateError(RuntimeError):
    """A fail-closed Method X battery-gate result."""


def validate_xai_battery_terminal(
    *,
    battery: Mapping[str, Any],
    backbone: str,
    selection: Mapping[str, Any] | None = None,
    model_root: Path | None = None,
) -> None:
    rows = battery.get("models") if isinstance(battery.get("models"), list) else []
    selected_rows = [
        row for row in rows if isinstance(row, dict) and str(row.get("backbone")) == backbone
    ]
    if len(rows) != 7:
        raise MethodXBatteryGateError("XAI_BATTERY_NOT_FINALIZED")
    by_name: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise MethodXBatteryGateError("XAI_BATTERY_NOT_FINALIZED")
        name = str(row.get("backbone") or "")
        if not name or name in by_name:
            raise MethodXBatteryGateError("XAI_BATTERY_NOT_FINALIZED")
        by_name[name] = row
    if set(by_name) != set(EFFECTIVE_FIVE) | set(EXCLUDED_BACKBONES):
        raise MethodXBatteryGateError("XAI_BATTERY_NOT_FINALIZED")
    for name in EFFECTIVE_FIVE:
        row = by_name[name]
        training = row.get("training")
        if str(row.get("status", "")).upper() != "COMPLETED":
            raise MethodXBatteryGateError("XAI_BATTERY_NOT_FINALIZED")
        if not isinstance(training, Mapping):
            raise MethodXBatteryGateError("XAI_BATTERY_NOT_FINALIZED")
        if str(training.get("status", "")).upper() not in TERMINAL_TRAINING_STATUSES:
            raise MethodXBatteryGateError("XAI_BATTERY_NOT_FINALIZED")
    for name in EXCLUDED_BACKBONES:
        row = by_name[name]
        training = row.get("training")
        if str(row.get("status", "")) != "PROTOCOL_EXCLUDED_NOT_STARTED":
            raise MethodXBatteryGateError("XAI_BATTERY_NOT_FINALIZED")
        if not isinstance(training, Mapping):
            raise MethodXBatteryGateError("XAI_BATTERY_NOT_FINALIZED")
        if str(training.get("status", "")) != "PROTOCOL_EXCLUDED_NOT_STARTED":
            raise MethodXBatteryGateError("XAI_BATTERY_NOT_FINALIZED")
    if str(battery.get("status", "")).upper() != "COMPLETED":
        raise MethodXBatteryGateError("XAI_BATTERY_NOT_FINALIZED")
    if backbone in EFFECTIVE_FIVE and len(selected_rows) != 1:
        raise MethodXBatteryGateError("XAI_SELECTED_BACKBONE_ROW_INVALID")
    if backbone == "amuse_x_hybrid3d" and (
        model_root is None
        or not isinstance(selection, Mapping)
        or str(selection.get("model_kind")) != "MX05_HYBRID"
    ):
        raise MethodXBatteryGateError("XAI_HYBRID_SELECTION_BINDING_INVALID")
