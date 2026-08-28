from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "research" / "brainiac_caft_20260808" / "evidence_registry_v3.json"

EFFECTIVE = (
    "resnet10_3d",
    "resnet18_3d",
    "resnet34_3d",
    "densenet121_3d",
    "mednext_3d_s",
)
EXCLUDED = ("resnet50_3d", "efficientnet3d_b0")
ACTIONS = (
    "PREPARE_RAW_T1",
    "EXTRACT_FROZEN_BRAINIAC_EMBEDDINGS",
    "EXTRACT_RESNET34_INNER_FEATURES",
    "CAFT_TRAINING",
    "OUTER_FEATURE_SNAPSHOT",
    "PROMOTION",
    "EXTERNAL_EVALUATION",
)


def _battery(*, terminal: bool = False) -> dict:
    rows = [
        {"backbone": name, "status": "COMPLETED"}
        for name in EFFECTIVE
    ]
    if not terminal:
        rows[-1]["status"] = "TRAINING_RUNNING"
    rows.extend(
        {"backbone": name, "status": "PROTOCOL_EXCLUDED_NOT_STARTED"}
        for name in EXCLUDED
    )
    return {"schema_version": 1, "models": rows}


def _evidence(*, b2: bool = False, b3: bool = False, b4: bool = False) -> dict:
    return {
        "r0_complete": True,
        "official_runtime": True,
        "strict_load": True,
        "official_forward": True,
        "preprocessing_v2_complete": b2,
        "brainiac_embeddings_complete": b3,
        "split_registry_locked": b3,
        "resnet34_representation_contract_ready": b3,
        "resnet34_inner_features_complete": b3,
        "brainiac_extractor_contract_ready": b3,
        "caft_trainer_contract_ready": b4,
        "outer_head_calibration_locked": False,
        "promotion_evidence_ready": False,
        "external_dua": False,
        "external_lineage": False,
        "external_no_refit": False,
        "external_exactly_once": False,
        "pretraining_overlap_status": "UNKNOWN_PRETRAINING_OVERLAP",
    }


def _writer_clear() -> dict:
    return {
        "inspection_status": "PASS",
        "active_writers": [],
        "recorded_owners": [],
    }


def test_method_y_writer_is_detected_by_common_writer_scan() -> None:
    """Dropping Method-Y markers must make the shared writer scan fail this test."""

    from src.research.brainiac_readiness import inspect_writer_state

    report = inspect_writer_state(
        _battery(terminal=True),
        process_provider=lambda: [
            {
                "ProcessId": 43210,
                "Name": "python.exe",
                "CommandLine": (
                    r"H:\New project\.venv\Scripts\python.exe "
                    r"scripts\run_method_y_image_training_v3.py --resume"
                ),
                "CreationDate": "20260828070000.000000+210",
            }
        ],
    )

    assert report["inspection_status"] == "PASS"
    assert report["active_writers"] == [
        {
            "pid": 43210,
            "name": "python.exe",
            "creation_time": "20260828070000.000000+210",
        }
    ]


@pytest.mark.parametrize(
    "entrypoint",
    [
        r"scripts\run_method_y_training_v2.py",
        r"scripts\train_fold_method_y_image.py",
        r"scripts\research\prepare_brainiac_raw_t1_v2.py",
        r"scripts\research\extract_brainiac_embeddings_v2.py",
    ],
)
def test_all_project_gpu_writer_entrypoint_families_are_detected(
    entrypoint: str,
) -> None:
    """Removing any required writer family must expose an undetected GPU writer."""

    from src.research.brainiac_readiness import inspect_writer_state

    report = inspect_writer_state(
        _battery(terminal=True),
        process_provider=lambda: [
            {
                "ProcessId": 43211,
                "Name": "PYTHON.EXE",
                "CommandLine": rf"python.exe {entrypoint} --resume",
                "CreationDate": "20260828070100.000000+210",
            }
        ],
    )

    assert report["inspection_status"] == "PASS"
    assert [row["pid"] for row in report["active_writers"]] == [43211]


def test_writer_scan_exposes_an_injectable_gpu_boundary() -> None:
    """Removing GPU-provider injection would make CUDA contention untestable."""

    import inspect

    from src.research.brainiac_readiness import inspect_writer_state

    assert "gpu_provider" in inspect.signature(inspect_writer_state).parameters


def test_foreign_cuda_python_process_blocks_all_action_decisions() -> None:
    """Ignoring a non-project Python CUDA process would permit GPU contention."""

    from src.research.brainiac_readiness import (
        derive_action_decisions,
        inspect_writer_state,
    )

    report = inspect_writer_state(
        _battery(terminal=True),
        process_provider=lambda: [
            {
                "ProcessId": 43212,
                "Name": "python.exe",
                "CommandLine": "python.exe -c \"import torch; torch.zeros(1, device='cuda')\"",
                "CreationDate": "20260828070200.000000+210",
            }
        ],
        gpu_provider=lambda: [
            {
                "pid": 43212,
                "process_name": "python.exe",
                "used_gpu_memory_mb": 512,
            }
        ],
    )
    decisions = derive_action_decisions(
        battery=_battery(terminal=True),
        evidence=_evidence(b2=True, b3=True, b4=True),
        writer_state=report,
    )

    assert report["inspection_status"] == "PASS"
    assert [row["pid"] for row in report.get("foreign_gpu_processes", [])] == [
        43212
    ]
    assert all(not row["allowed"] for row in decisions.values())
    assert all(
        "FOREIGN_CUDA_COMPUTE_PROCESS" in row["blockers"]
        for row in decisions.values()
    )


def test_action_exclusion_exposes_test_only_writer_providers() -> None:
    """Removing provider injection would make the filesystem guard non-hermetic."""

    import inspect

    from src.research.brainiac_action_guard import inspect_action_exclusion

    parameters = inspect.signature(inspect_action_exclusion).parameters
    assert "process_provider" in parameters
    assert "gpu_provider" in parameters


def test_action_exclusion_blocks_method_y_writer_with_no_lock_present(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Lock absence must not launder a live Method-Y writer into PASS."""

    from src.research import brainiac_action_guard as guard

    manifest = tmp_path / "battery_manifest.json"
    manifest.write_text(json.dumps(_battery(terminal=True)), encoding="utf-8")
    monkeypatch.setattr(guard, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(guard, "DEFAULT_BATTERY_MANIFEST", manifest)
    monkeypatch.setattr(guard, "BATTERY_ROOT", tmp_path)
    method_y = {
        "ProcessId": 43214,
        "Name": "python.exe",
        "CommandLine": r"python.exe scripts\run_method_y_training_v2.py --resume",
        "CreationDate": "20260828070400.000000+210",
    }

    report = guard.inspect_action_exclusion(
        process_provider=lambda: [method_y],
        gpu_provider=lambda: [],
    )

    assert report["status"] == "BLOCKED"
    assert "ACTIVE_BRAINIAC_OR_CNN_WRITER" in report["blockers"]
    assert [row["pid"] for row in report["writer_state"]["active_writers"]] == [
        43214
    ]


def test_action_guard_rechecks_gpu_processes_after_atomic_acquire(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A CUDA process appearing in the acquisition race must revoke the lease."""

    import os

    from src.research import brainiac_action_guard as guard
    from src.research import brainiac_readiness as readiness

    manifest = tmp_path / "battery_manifest.json"
    manifest.write_text(json.dumps(_battery(terminal=True)), encoding="utf-8")
    monkeypatch.setattr(guard, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(guard, "DEFAULT_BATTERY_MANIFEST", manifest)
    monkeypatch.setattr(guard, "BATTERY_ROOT", tmp_path)
    monkeypatch.setattr(guard, "_pid_identity", lambda pid: (True, "self-created"))
    process_rows = [
        {
            "ProcessId": os.getpid(),
            "Name": "python.exe",
            "CommandLine": "python.exe -m pytest",
            "CreationDate": "self-created",
        },
        {
            "ProcessId": 43215,
            "Name": "python.exe",
            "CommandLine": "python.exe -c \"import torch\"",
            "CreationDate": "foreign-created",
        },
    ]
    gpu_snapshots = iter(
        [
            [],
            [
                {
                    "pid": 43215,
                    "process_name": "python.exe",
                    "used_gpu_memory_mb": 128,
                }
            ],
        ]
    )
    process_provider = lambda: process_rows
    gpu_provider = lambda: next(gpu_snapshots)
    monkeypatch.setattr(readiness, "_default_process_provider", process_provider)
    monkeypatch.setattr(readiness, "_default_gpu_process_provider", gpu_provider)

    with pytest.raises(
        RuntimeError,
        match="BRAINIAC_ACTION_EXCLUSION_POST_ACQUIRE_BLOCKED:FOREIGN_CUDA_COMPUTE_PROCESS",
    ):
        guard.acquire_action_guard("PREPARE_RAW_T1")
    assert not (tmp_path / guard.SHARED_GUARD_NAME).exists()


def test_live_r0_is_complete_but_citation_corroboration_is_partial() -> None:
    from src.research.brainiac_readiness import audit_r0_registry

    report = audit_r0_registry(REGISTRY, project_root=ROOT)
    assert report["total_in_window"] == 29
    assert report["peer_reviewed"] == 13
    assert report["full_text_reviewed"] == 28
    assert report["r0_status"] == "R0_COMPLETE"
    assert report["citation_corroboration"] == "CITATION_CORROBORATION_PARTIAL"
    assert report["fully_verified"] is False
    assert report["novelty_status"] == "NOVELTY_CANDIDATE"


def test_r0_rejects_duplicate_canonical_identifier_and_review_hash_drift(
    tmp_path: Path,
) -> None:
    from src.research.brainiac_readiness import audit_r0_registry

    payload = json.loads(REGISTRY.read_text(encoding="utf-8"))
    payload["sources"][1]["doi"] = payload["sources"][0]["doi"]
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="R0_CANONICAL_IDENTIFIER_DUPLICATE"):
        audit_r0_registry(duplicate, project_root=ROOT)

    payload = json.loads(REGISTRY.read_text(encoding="utf-8"))
    payload["sources"][0]["review_evidence"]["source_sha256"] = "0" * 64
    drift = tmp_path / "drift.json"
    drift.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="R0_REVIEW_EVIDENCE_HASH_MISMATCH"):
        audit_r0_registry(drift, project_root=ROOT)


def test_exactly_seven_actions_have_explicit_decisions() -> None:
    from src.research.brainiac_readiness import derive_action_decisions

    report = derive_action_decisions(
        battery=_battery(terminal=False),
        evidence=_evidence(),
        writer_state=_writer_clear(),
    )
    assert tuple(report) == ACTIONS
    assert all(set(report[action]) >= {"allowed", "status", "blockers"} for action in ACTIONS)


def test_nonterminal_battery_does_not_block_ready_preparation_actions() -> None:
    from src.research.brainiac_readiness import derive_action_decisions

    report = derive_action_decisions(
        battery=_battery(terminal=False),
        evidence=_evidence(b2=True, b3=True),
        writer_state=_writer_clear(),
    )
    assert report["PREPARE_RAW_T1"]["allowed"] is True
    assert report["EXTRACT_FROZEN_BRAINIAC_EMBEDDINGS"]["allowed"] is True
    assert report["EXTRACT_RESNET34_INNER_FEATURES"]["allowed"] is True
    for action in ("CAFT_TRAINING", "OUTER_FEATURE_SNAPSHOT", "PROMOTION"):
        assert report[action]["allowed"] is False
        assert "FIVE_CNN_BATTERY_NOT_TERMINAL" in report[action]["blockers"]


def test_exact_battery_identity_cannot_be_laundered_by_status_or_exclusions() -> None:
    from src.research.brainiac_readiness import derive_action_decisions

    reset = _battery(terminal=True)
    reset["models"] = reset["models"][:5]
    reset["models"].append({"backbone": "efficientnet3d_b0", "status": "COMPLETED"})
    reset["models"].append({"backbone": "resnet50_3d", "status": "COMPLETED"})
    reset["models"][4]["backbone"] = "invented_model"
    report = derive_action_decisions(
        battery=reset,
        evidence=_evidence(b2=True, b3=True, b4=True),
        writer_state=_writer_clear(),
    )
    assert all(not row["allowed"] for row in report.values())
    assert all("FIVE_CNN_BATTERY_IDENTITY_INVALID" in row["blockers"] for row in report.values())


@pytest.mark.parametrize(
    "writer_state,expected",
    [
        ({"inspection_status": "UNKNOWN", "active_writers": [], "recorded_owners": []}, "WRITER_LIVENESS_UNKNOWN"),
        ({"inspection_status": "PASS", "active_writers": [{"pid": 7}], "recorded_owners": []}, "ACTIVE_BRAINIAC_OR_CNN_WRITER"),
        ({"inspection_status": "PASS", "active_writers": [], "recorded_owners": [{"pid": 8, "liveness": "LIVE", "creation_time_matches": True}]}, "UNLISTED_LIVE_RECORDED_OWNER"),
        ({"inspection_status": "PASS", "active_writers": [], "recorded_owners": [{"pid": 8, "liveness": "LIVE", "creation_time_matches": False}]}, "RECORDED_OWNER_PID_REUSE_OR_CREATION_MISMATCH"),
    ],
)
def test_writer_uncertainty_or_ownership_anomaly_blocks_before_action(
    writer_state: dict, expected: str
) -> None:
    from src.research.brainiac_readiness import derive_action_decisions

    report = derive_action_decisions(
        battery=_battery(terminal=False),
        evidence=_evidence(b2=True, b3=True),
        writer_state=writer_state,
    )
    assert all(not row["allowed"] for row in report.values())
    assert all(expected in row["blockers"] for row in report.values())


def test_unknown_overlap_never_unlocks_lora_or_external_or_confirmatory_claims() -> None:
    from src.research.brainiac_readiness import derive_readiness_payload

    payload = derive_readiness_payload(
        battery=_battery(terminal=True),
        evidence=_evidence(b2=True, b3=True, b4=True),
        writer_state=_writer_clear(),
        input_bindings={"x": {"path": "x", "bytes": 1, "sha256": "a" * 64}},
        battery_terminal_report={
            "identity_valid": True,
            "terminal": True,
            "blockers": [],
        },
    )
    assert payload["lora_unlocked"] is False
    assert payload["claim_limits"]["confirmatory_pretraining_independence"] is False
    assert payload["claim_limits"]["external_generalization"] is False
    assert payload["claim_limits"]["claim_ceiling"] == "POST_HOC_INTERNAL_TECHNICAL_EVIDENCE"
    assert payload["external_evaluation_status"] == "EXTERNAL_UNAVAILABLE_NONBLOCKING"
    assert payload["actions"]["CAFT_TRAINING"]["allowed"] is True
    assert payload["actions"]["EXTERNAL_EVALUATION"]["allowed"] is False


def test_schema4_validation_rejects_self_hash_tamper_input_drift_and_nonfinite(
    tmp_path: Path,
) -> None:
    from src.research.brainiac_readiness import (
        bind_file,
        validate_materialized_readiness,
        write_readiness_artifact,
    )

    source = tmp_path / "source.json"
    source.write_text('{"finite": 1}\n', encoding="utf-8")
    payload = derive = {
        "schema_version": 4,
        "status": "BLOCKED",
        "ready": False,
        "preparation_ready": False,
        "training_allowed": False,
        "actions": {action: {"allowed": False, "status": "BLOCKED", "blockers": ["MISSING"]} for action in ACTIONS},
        "authorized_actions": [],
        "input_bindings": {"source": bind_file(source, project_root=tmp_path)},
        "outer_test_touched": False,
        "lora_unlocked": False,
        "training_started": False,
        "human_approval_required": False,
        "manual_override_allowed": False,
        "unknown_is_pass": False,
    }
    target = tmp_path / "readiness.json"
    write_readiness_artifact(target, payload)
    validate_materialized_readiness(target, project_root=tmp_path)

    tampered = json.loads(target.read_text(encoding="utf-8"))
    tampered["ready"] = True
    target.write_text(json.dumps(tampered, sort_keys=True), encoding="utf-8")
    with pytest.raises(RuntimeError, match="READINESS_INTEGRITY_SHA256_MISMATCH"):
        validate_materialized_readiness(target, project_root=tmp_path)

    write_readiness_artifact(target, payload)
    source.write_text('{"finite": 2}\n', encoding="utf-8")
    with pytest.raises(RuntimeError, match="READINESS_INPUT_BINDING_MISMATCH"):
        validate_materialized_readiness(target, project_root=tmp_path)

    source.write_text('{"finite": NaN}\n', encoding="utf-8")
    payload["input_bindings"]["source"] = bind_file(source, project_root=tmp_path)
    write_readiness_artifact(target, payload)
    with pytest.raises(RuntimeError, match="NONFINITE_OR_INVALID_JSON"):
        validate_materialized_readiness(target, project_root=tmp_path)


@pytest.mark.parametrize(
    "field",
    ["human_approval_required", "manual_override_allowed", "unknown_is_pass"],
)
@pytest.mark.parametrize("invalid", [None, True])
def test_schema4_requires_explicit_fail_closed_supervision_fields(
    tmp_path: Path, field: str, invalid: object
) -> None:
    from src.research.brainiac_readiness import (
        bind_file,
        validate_materialized_readiness,
        write_readiness_artifact,
    )

    source = tmp_path / "source.json"
    source.write_text('{"finite": 1}\n', encoding="utf-8")
    payload = {
        "schema_version": 4,
        "status": "BLOCKED",
        "ready": False,
        "preparation_ready": False,
        "training_allowed": False,
        "actions": {
            action: {"allowed": False, "status": "BLOCKED", "blockers": ["MISSING"]}
            for action in ACTIONS
        },
        "authorized_actions": [],
        "input_bindings": {"source": bind_file(source, project_root=tmp_path)},
        "outer_test_touched": False,
        "lora_unlocked": False,
        "training_started": False,
        "human_approval_required": False,
        "manual_override_allowed": False,
        "unknown_is_pass": False,
    }
    if invalid is None:
        payload.pop(field)
    else:
        payload[field] = invalid
    target = tmp_path / "readiness.json"
    write_readiness_artifact(target, payload)
    with pytest.raises(RuntimeError, match="READINESS_SUPERVISION_INVALID"):
        validate_materialized_readiness(target, project_root=tmp_path)


def test_schema3_predecessor_is_not_rewritten_by_schema4_materialization() -> None:
    predecessor = ROOT / "research" / "brainiac_caft_20260808" / "r1_readiness_status_20260822.json"
    before = predecessor.read_bytes()
    assert hashlib.sha256(before).hexdigest() == "09b360b1381c903d38d05af2a6f30859f511d2db2b2af05c1c2928fb01f83f12"
    assert json.loads(before)["schema_version"] == 3


def test_policy_scope_amendment_and_live_schema4_are_independently_pinned(
    tmp_path: Path,
) -> None:
    from src.research.brainiac_readiness import (
        DEFAULT_POLICY,
        DEFAULT_SCOPE_AMENDMENT,
        PINNED_POLICY_SHA256,
        PINNED_SCOPE_AMENDMENT_SHA256,
        build_live_readiness,
        validate_policy_and_scope,
        validate_materialized_readiness,
        write_readiness_artifact,
    )

    validated = validate_policy_and_scope()
    assert validated["policy"]["sha256"] == PINNED_POLICY_SHA256
    assert validated["scope_amendment"]["sha256"] == PINNED_SCOPE_AMENDMENT_SHA256

    predecessor = ROOT / "research" / "brainiac_caft_20260808" / "r1_readiness_status_20260822.json"
    before = predecessor.read_bytes()
    payload = build_live_readiness(process_provider=lambda: [])
    assert payload["schema_version"] == 4
    assert set(payload["actions"]) == set(ACTIONS)
    assert payload["training_allowed"] is False
    assert payload["battery_terminal_report"]["terminal"] is True
    assert "FIVE_CNN_BATTERY_NOT_TERMINAL" not in payload["actions"]["CAFT_TRAINING"]["blockers"]
    assert "BRAINIAC_PREPROCESSING_V2_INCOMPLETE" in payload["actions"]["CAFT_TRAINING"]["blockers"]
    target = tmp_path / "schema4.json"
    receipt = write_readiness_artifact(target, payload)
    validate_materialized_readiness(
        target,
        project_root=ROOT,
        expected_sha256=receipt["sha256"],
    )
    assert predecessor.read_bytes() == before

    changed_policy = tmp_path / "policy.json"
    policy = json.loads(DEFAULT_POLICY.read_text(encoding="utf-8"))
    policy["manual_override_allowed"] = True
    changed_policy.write_text(json.dumps(policy), encoding="utf-8")
    with pytest.raises(RuntimeError, match="BRAINIAC_POLICY_SHA256_MISMATCH"):
        validate_policy_and_scope(
            policy_path=changed_policy,
            scope_amendment_path=DEFAULT_SCOPE_AMENDMENT,
            expected_policy_sha256=PINNED_POLICY_SHA256,
            expected_scope_amendment_sha256=PINNED_SCOPE_AMENDMENT_SHA256,
            project_root=ROOT,
        )


def test_reparse_and_malformed_authorities_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import src.research.brainiac_readiness as readiness

    authority = tmp_path / "authority.json"
    authority.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(
        readiness,
        "_is_reparse",
        lambda path: Path(path).resolve() == authority.resolve(),
    )
    with pytest.raises(RuntimeError, match="REPARSE_AUTHORITY_FORBIDDEN"):
        readiness.bind_file(authority, project_root=tmp_path)

    malformed = tmp_path / "malformed.json"
    malformed.write_text("{", encoding="utf-8")
    monkeypatch.undo()
    with pytest.raises(RuntimeError, match="NONFINITE_OR_INVALID_JSON"):
        readiness.validate_policy_and_scope(
            policy_path=malformed,
            scope_amendment_path=readiness.DEFAULT_SCOPE_AMENDMENT,
            expected_policy_sha256=hashlib.sha256(malformed.read_bytes()).hexdigest(),
            expected_scope_amendment_sha256=readiness.PINNED_SCOPE_AMENDMENT_SHA256,
            project_root=ROOT,
        )


def test_execution_gate_recomputes_action_and_writer_state_before_io(
    tmp_path: Path,
) -> None:
    from src.research.brainiac_execution_gate import assess_brainiac_execution_gate
    from src.research.brainiac_readiness import derive_action_decisions

    del tmp_path
    allowed = assess_brainiac_execution_gate("PREPARE_RAW_T1")
    assert allowed["allowed"] is True

    active = derive_action_decisions(
        battery=_battery(terminal=False),
        evidence=_evidence(),
        writer_state={
            "inspection_status": "PASS",
            "active_writers": [{"pid": 76543}],
            "recorded_owners": [],
        },
    )
    assert active["PREPARE_RAW_T1"]["allowed"] is False
    assert "ACTIVE_BRAINIAC_OR_CNN_WRITER" in active["PREPARE_RAW_T1"]["blockers"]

    training = assess_brainiac_execution_gate("CAFT_TRAINING")
    assert training["allowed"] is False
    assert "FIVE_CNN_BATTERY_NOT_TERMINAL" not in training["blockers"]
    assert "BRAINIAC_PREPROCESSING_V2_INCOMPLETE" in training["blockers"]


def test_execution_gate_rejects_readiness_tamper_even_with_ready_true(
    tmp_path: Path,
) -> None:
    from src.research.brainiac_readiness_ledger import (
        append_readiness_transition,
        validate_readiness_ledger,
    )
    ledger = tmp_path / "ledger"
    append_readiness_transition(
        ledger,
        {"schema_version": 4, "ready": False, "authorized_actions": []},
        project_root=tmp_path,
    )
    target = next(ledger.glob("*.json"))
    payload = json.loads(target.read_text(encoding="utf-8"))
    payload["readiness_payload"]["ready"] = True
    target.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    with pytest.raises(RuntimeError, match="LEDGER"):
        validate_readiness_ledger(ledger, project_root=tmp_path)


def test_schema4_publication_is_append_only_but_idempotent(tmp_path: Path) -> None:
    from src.research.brainiac_readiness import (
        build_live_readiness,
        write_readiness_artifact,
    )

    target = tmp_path / "schema4.json"
    payload = build_live_readiness(process_provider=lambda: [])
    first = write_readiness_artifact(target, payload, append_only=True)
    second = write_readiness_artifact(target, payload, append_only=True)
    assert first == second
    before = target.read_bytes()
    changed = dict(payload)
    changed["status"] = "TAMPERED"
    with pytest.raises(RuntimeError, match="APPEND_ONLY_ARTIFACT_DRIFT"):
        write_readiness_artifact(target, changed, append_only=True)
    assert target.read_bytes() == before


@pytest.mark.parametrize("forbidden", ["subject_id", "raw_path", "ab_label", "pet_input"])
def test_readiness_refuses_private_or_predictive_payload_fields(
    tmp_path: Path, forbidden: str
) -> None:
    from src.research.brainiac_readiness import (
        build_live_readiness,
        write_readiness_artifact,
    )

    payload = build_live_readiness(process_provider=lambda: [])
    payload[forbidden] = "forbidden"
    with pytest.raises(RuntimeError, match="READINESS_PRIVACY_FIELD_FORBIDDEN"):
        write_readiness_artifact(tmp_path / "readiness.json", payload)
