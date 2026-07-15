from __future__ import annotations

from pathlib import Path
from typing import Any

from .atomic_io import atomic_write_json, read_json
from .contracts import RunContract


class OrchestrationError(RuntimeError):
    """Raised when a required stage is skipped or an immutable root is reused incorrectly."""


_ALLOWED_TRANSITIONS = {
    "INIT": {"PREFLIGHT_DONE"},
    "PREFLIGHT_DONE": {"MANIFEST_LOCKED"},
    "MANIFEST_LOCKED": {"PREPROCESSING_DONE"},
    "PREPROCESSING_DONE": {"QC_PASSED"},
    "QC_PASSED": {"ENDPOINTS_LOCKED"},
    "ENDPOINTS_LOCKED": {"SPLITS_LOCKED"},
    "SPLITS_LOCKED": {"LOCAL_HPO_DONE"},
    "LOCAL_HPO_DONE": {"MODEL_CONFIGS_LOCKED"},
    "MODEL_CONFIGS_LOCKED": {"OUTER_EVALUATION_DONE"},
    "OUTER_EVALUATION_DONE": {"EXTERNAL_PENDING", "EXTERNAL_DONE"},
    "EXTERNAL_PENDING": {"EXTERNAL_DONE", "REPORTS_DONE"},
    "EXTERNAL_DONE": {"REPORTS_DONE"},
    "REPORTS_DONE": {"FINALIZED"},
    "FINALIZED": set(),
}


def next_state(current: str, requested: str) -> str:
    if requested not in _ALLOWED_TRANSITIONS.get(current, set()):
        raise OrchestrationError(f"INVALID_STAGE_TRANSITION:{current}->{requested}")
    return requested


def _legacy_records() -> list[dict[str, Any]]:
    return [
        {
            "artifact_id": "legacy_amyloid_pet_to_pet_label",
            "artifact_status": "COMPLETED",
            "science_gate_status": "FAIL",
            "claim_status": "TECHNICAL_ONLY",
            "terminal_reason": "SAME_MODALITY_DERIVED_TARGET_AND_COHORT_CONFOUNDING",
            "source": "results/method_y_execution",
        },
        {
            "artifact_id": "current_tau_progression",
            "artifact_status": "COMPLETED",
            "science_gate_status": "FAIL",
            "claim_status": "EXPLORATORY",
            "terminal_reason": [
                "TAU_INCREMENTAL_VALUE_NOT_DEMONSTRATED",
                "SCANNER_SHORTCUT_DETECTED",
                "EXTERNAL_VALIDATION_MISSING",
                "INCOMPLETE_SURVIVAL_FOLD_EVALUATION",
            ],
            "source": "results/method_y_tau_progression",
        },
    ]


def initialize_run(
    *, config: dict[str, Any], output_root: str | Path, repo_snapshot: dict[str, Any]
) -> dict[str, Any]:
    root = Path(output_root)
    for relative in (
        "provenance",
        "manifests",
        "qc",
        "endpoints",
        "splits",
        "development/amyloid",
        "development/tau",
        "development/paired_at",
        "locked",
        "outer_evaluation",
        "external_validation/adni",
        "external_validation/oasis3",
        "reports",
        "public_bundle",
    ):
        (root / relative).mkdir(parents=True, exist_ok=True)

    repository_contract = {
        key: repo_snapshot.get(key)
        for key in ("git_commit", "git_branch", "dirty", "dirty_tree_digest")
        if key in repo_snapshot
    }
    contract = RunContract.from_mapping(
        {"config": config, "repository": repository_contract}
    )
    contract_path = root / "run_contract.json"
    if contract_path.exists():
        existing_payload = read_json(contract_path)
        existing = RunContract.from_mapping(existing_payload["payload"])
        existing.require_match(contract)
    else:
        atomic_write_json(contract_path, {"payload": contract.payload, "sha256": contract.digest})

    state_path = root / "orchestration_state.json"
    if not state_path.exists():
        atomic_write_json(
            state_path,
            {"stage": "PREFLIGHT_DONE", "run_contract_sha256": contract.digest},
        )
    atomic_write_json(root / "provenance" / "repository_snapshot.json", repo_snapshot)
    legacy = _legacy_records()
    atomic_write_json(root / "provenance" / "legacy_artifact_registry.json", legacy)
    atomic_write_json(root / "artifact_registry.json", legacy)
    external_status = root / "external_validation" / "status.json"
    if not external_status.exists():
        atomic_write_json(
            external_status,
            {
                "status": "EXTERNAL_PENDING",
                "science_gate_status": "BLOCKED",
                "claim_status": "NO_CLAIM",
                "blockers": ["ADNI_DUA_DATA_NOT_AVAILABLE", "OASIS3_DATA_NOT_AVAILABLE"],
            },
        )
    return {"output_root": str(root), "run_contract_sha256": contract.digest}
