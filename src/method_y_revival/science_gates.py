from __future__ import annotations

from typing import Any


def incremental_value_gate(
    *,
    delta_ibs_ci: tuple[float, float],
    delta_cindex_ci: tuple[float, float],
    outer_folds_completed: int,
    expected_outer_folds: int,
    scanner_sensitivity_passed: bool,
) -> dict[str, Any]:
    blockers: list[str] = []
    if int(outer_folds_completed) != int(expected_outer_folds):
        blockers.append("INCOMPLETE_OUTER_CV")
    if float(delta_ibs_ci[1]) >= 0.0:
        blockers.append("IBS_IMPROVEMENT_CI_INCLUDES_ZERO")
    if float(delta_cindex_ci[0]) < 0.0:
        blockers.append("CINDEX_NOT_NONINFERIOR")
    if not scanner_sensitivity_passed:
        blockers.append("SCANNER_SHORTCUT_NOT_CONTROLLED")
    passed = not blockers
    return {
        "science_gate_status": "PASS" if passed else "FAIL",
        "claim_status": "INTERNAL_ONLY" if passed else "EXPLORATORY",
        "blockers": blockers,
        "delta_ibs_ci": list(delta_ibs_ci),
        "delta_cindex_ci": list(delta_cindex_ci),
    }


def assert_claim_consistency(record: dict[str, Any]) -> None:
    claim = record.get("claim_status")
    gate = record.get("science_gate_status")
    if claim in {"INTERNAL_ONLY", "EXTERNALLY_VALIDATED"} and gate != "PASS":
        raise ValueError("POSITIVE_CLAIM_WITHOUT_PASSED_SCIENCE_GATE")
