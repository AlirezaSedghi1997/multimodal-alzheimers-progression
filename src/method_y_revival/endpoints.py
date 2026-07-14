from __future__ import annotations

from dataclasses import asdict, dataclass
from math import isfinite
from typing import Any, Iterable


class EndpointContractError(ValueError):
    """Raised when an endpoint is temporally invalid or derived from the input PET."""


@dataclass(frozen=True)
class ProgressionEndpoint:
    event: int
    time_days: float
    event_day: float | None
    confirmation_day: float | None
    censor_day: float | None
    terminal_reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def validate_target_contract(
    *, branch: str, target_name: str, target_source: str, claim_bearing: bool
) -> None:
    if branch not in {"amyloid", "tau", "paired_at"}:
        raise EndpointContractError(f"UNSUPPORTED_BRANCH:{branch}")
    same_pet_targets = {"amyloid_pos", "centiloid", "suvr", "tau_suvr"}
    if (
        claim_bearing
        and target_source == "same_input_pet"
        and target_name.strip().casefold() in same_pet_targets
    ):
        raise EndpointContractError("SAME_PET_DERIVED_TARGET_FORBIDDEN")


def build_sustained_progression(
    *,
    pet_index_day: float,
    baseline_cdr: float,
    visits: Iterable[dict[str, Any]],
    max_followup_days: float,
) -> ProgressionEndpoint:
    if float(baseline_cdr) != 0.0:
        raise EndpointContractError("PRIMARY_ENDPOINT_REQUIRES_BASELINE_CDR_ZERO")
    if max_followup_days <= 0:
        raise EndpointContractError("POSITIVE_FOLLOWUP_WINDOW_REQUIRED")

    valid: list[tuple[float, float]] = []
    upper_day = float(pet_index_day) + float(max_followup_days)
    for visit in visits:
        try:
            day = float(visit["day"])
            cdr = float(visit["cdr_global"])
        except (KeyError, TypeError, ValueError):
            continue
        if not isfinite(day) or not isfinite(cdr):
            continue
        if float(pet_index_day) < day <= upper_day:
            valid.append((day, cdr))
    valid.sort(key=lambda item: item[0])
    if not valid:
        raise EndpointContractError("NO_VALID_POST_PET_FOLLOWUP")

    for current, following in zip(valid, valid[1:]):
        current_day, current_cdr = current
        confirmation_day, confirmation_cdr = following
        if current_cdr > baseline_cdr and confirmation_cdr >= current_cdr:
            return ProgressionEndpoint(
                event=1,
                time_days=current_day - float(pet_index_day),
                event_day=current_day,
                confirmation_day=confirmation_day,
                censor_day=None,
                terminal_reason="SUSTAINED_CDR_PROGRESSION",
            )

    last_day, last_cdr = valid[-1]
    reason = (
        "RIGHT_CENSORED_UNCONFIRMED_INCREASE"
        if last_cdr > baseline_cdr
        else "RIGHT_CENSORED_NO_PROGRESSION"
    )
    return ProgressionEndpoint(
        event=0,
        time_days=last_day - float(pet_index_day),
        event_day=None,
        confirmation_day=None,
        censor_day=last_day,
        terminal_reason=reason,
    )
