from __future__ import annotations


class ExternalValidationError(ValueError):
    """Raised if external outcomes influence model fitting or selection."""


def external_claim_status(
    *, internal_gate_passed: bool, external_gate_passed: bool, refit_used: bool
) -> str:
    if refit_used:
        raise ExternalValidationError("EXTERNAL_REFIT_FORBIDDEN")
    if internal_gate_passed and external_gate_passed:
        return "EXTERNALLY_VALIDATED"
    if internal_gate_passed:
        return "INTERNAL_ONLY"
    return "EXPLORATORY"


def validate_external_policy(
    *, refit_used: bool, feature_reselection_used: bool, recalibration_used: bool
) -> None:
    if refit_used:
        raise ExternalValidationError("EXTERNAL_REFIT_FORBIDDEN")
    if feature_reselection_used:
        raise ExternalValidationError("EXTERNAL_FEATURE_RESELECTION_FORBIDDEN")
    if recalibration_used:
        raise ExternalValidationError("PRIMARY_EXTERNAL_RECALIBRATION_FORBIDDEN")
