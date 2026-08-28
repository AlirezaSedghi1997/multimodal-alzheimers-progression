from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_dynamic_ledger_is_anchored_to_the_pinned_b1_genesis(tmp_path: Path) -> None:
    from src.research.brainiac_readiness import _canonical_json_bytes, _sha256_bytes
    from src.research.brainiac_readiness_ledger import (
        append_readiness_transition,
        validate_readiness_ledger,
    )

    ledger = tmp_path / "ledger"
    event = append_readiness_transition(
        ledger, {"schema_version": 4, "stage": "B1"}, project_root=tmp_path
    )
    original_sha = event["event_sha256"]
    path = next(ledger.glob("*.json"))
    forged = json.loads(path.read_text(encoding="utf-8"))
    forged["readiness_payload"]["stage"] = "FORGED"
    forged["readiness_payload_sha256"] = _sha256_bytes(
        _canonical_json_bytes(forged["readiness_payload"])
    )
    body = dict(forged)
    body.pop("event_sha256")
    forged_sha = _sha256_bytes(_canonical_json_bytes(body))
    forged["event_sha256"] = forged_sha
    path.unlink()
    (ledger / f"00000001-{forged_sha}.json").write_bytes(
        _canonical_json_bytes(forged)
    )
    with pytest.raises(RuntimeError, match="LEDGER_GENESIS_MISMATCH"):
        validate_readiness_ledger(
            ledger,
            project_root=tmp_path,
            expected_genesis_sha256=original_sha,
        )


def test_ledger_blocks_concurrent_or_reparse_append_authority(
    tmp_path: Path,
) -> None:
    from src.research.brainiac_readiness_ledger import (
        append_readiness_transition,
        validate_readiness_ledger,
    )

    ledger = tmp_path / "ledger"
    append_readiness_transition(
        ledger, {"schema_version": 4, "stage": "B1"}, project_root=tmp_path
    )
    (ledger / ".append.lock").write_text("999", encoding="ascii")
    with pytest.raises(RuntimeError, match="APPEND_IN_PROGRESS"):
        validate_readiness_ledger(ledger, project_root=tmp_path)
    (ledger / ".append.lock").unlink()

    target = next(ledger.glob("*.json"))
    original = target.with_suffix(".original")
    target.rename(original)
    try:
        target.symlink_to(original)
    except OSError as exc:
        original.rename(target)
        pytest.skip(f"file symlink unavailable: {exc}")
    with pytest.raises(RuntimeError, match="LEDGER_ENTRY_INVALID|REPARSE"):
        validate_readiness_ledger(ledger, project_root=tmp_path)


def _patch_guard_fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> object:
    import src.research.brainiac_action_guard as guard

    manifest = tmp_path / "battery_manifest.json"
    _write_json(manifest, {"models": []})
    monkeypatch.setattr(guard, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(guard, "BATTERY_ROOT", tmp_path)
    monkeypatch.setattr(guard, "DEFAULT_BATTERY_MANIFEST", manifest)
    monkeypatch.setattr(guard, "_default_process_provider", lambda: [])
    monkeypatch.setattr(guard, "_default_gpu_process_provider", lambda: [])
    monkeypatch.setattr(
        guard,
        "_pid_identity",
        lambda pid: (True, "owner-create") if pid == os.getpid() else (False, None),
    )
    return guard


def test_action_guard_is_owned_rechecked_and_cleanly_released(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    guard = _patch_guard_fixture(tmp_path, monkeypatch)
    lease = guard.acquire_action_guard("PREPARE_RAW_T1")
    payload = json.loads(lease.guard_path.read_text(encoding="utf-8"))
    assert payload["role"] == "method_x_reconciliation"
    assert payload["purpose"] == "brainiac_action_exclusion"
    assert payload["nonce"] == lease.nonce
    token = guard.activate_lease(lease)
    try:
        assert guard.require_active_lease("PREPARE_RAW_T1") == lease
        with pytest.raises(RuntimeError, match="LEASE_REQUIRED"):
            guard.require_active_lease("CAFT_TRAINING")
    finally:
        guard.deactivate_lease(token)
        guard.release_action_guard(lease)
    assert not lease.guard_path.exists()


@pytest.mark.parametrize("tamper", ["missing", "rewritten"])
def test_action_guard_release_failure_is_never_silent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, tamper: str
) -> None:
    guard = _patch_guard_fixture(tmp_path, monkeypatch)
    lease = guard.acquire_action_guard("PREPARE_RAW_T1")
    if tamper == "missing":
        lease.guard_path.unlink()
    else:
        lease.guard_path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="BRAINIAC_ACTION_GUARD_RELEASE"):
        guard.release_action_guard(lease)


def test_live_battery_owner_blocks_but_proven_stale_lock_is_retained(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    guard = _patch_guard_fixture(tmp_path, monkeypatch)
    lock = tmp_path / ".battery.lock"
    _write_json(lock, {"pid": 777})
    monkeypatch.setattr(
        guard,
        "_pid_identity",
        lambda pid: (True, "live-777") if pid == 777 else (True, "owner-create"),
    )
    with pytest.raises(RuntimeError, match="ACTIVE_BATTERY_LOCK_OWNER"):
        guard.acquire_action_guard("PREPARE_RAW_T1")
    before = lock.read_bytes()

    monkeypatch.setattr(
        guard,
        "_pid_identity",
        lambda pid: (False, None) if pid == 777 else (True, "owner-create"),
    )
    lease = guard.acquire_action_guard("PREPARE_RAW_T1")
    try:
        assert lock.read_bytes() == before
    finally:
        guard.release_action_guard(lease)


def _amendment_payload(config_sha: str) -> dict:
    from src.research.brainiac_battery_terminal import _canonical_payload_sha256
    from src.research.brainiac_readiness import EFFECTIVE_FIVE, EXCLUDED_BACKBONES

    body = {
        "schema_version": 1,
        "amendment_id": "method_x_battery_v3_five_model_amendment.v1",
        "append_only": True,
        "effective_order": list(EFFECTIVE_FIVE),
        "excluded_backbones": list(EXCLUDED_BACKBONES),
        "locked_config_sha256": config_sha,
        "not_started_proof": {
            name: "SMOKE_PASS_ONLY_NO_TRAINING_CHECKPOINT_OR_METRICS_ARTIFACTS"
            for name in EXCLUDED_BACKBONES
        },
        "pre_amendment_manifest_sha256": "a" * 64,
    }
    return {**body, "scientific_payload_sha256": _canonical_payload_sha256(body)}


def _completed_battery_fixture(root: Path) -> Path:
    from src.research.brainiac_readiness import EFFECTIVE_FIVE, EXCLUDED_BACKBONES
    from src.training.method_x_resume import build_locked_outer_result
    from src.utils.reproducibility import canonical_digest

    config_sha = "c" * 64
    rows = []
    for backbone in EFFECTIVE_FIVE:
        model_dir = root / "models" / backbone
        model_dir.mkdir(parents=True)
        contract_body = {
            "schema_version": 1,
            "folds": 5,
            "seed": 13,
            "parameters": {
                "backbone": backbone,
                "folds": 5,
                "seed": 13,
                "resolution": 128,
                "phase": "final",
                "require_gpu": True,
                "source_contract_digest": "a" * 64,
            },
        }
        contract_digest = canonical_digest(contract_body)
        _write_json(
            model_dir / "run_contract.json",
            {
                "run_contract": {**contract_body, "digest": contract_digest},
                "run_contract_digest": contract_digest,
            },
        )
        fold_metrics = []
        for fold in range(5):
            checkpoint = model_dir / f"method_x_3d_fold_{fold}.pt"
            split_sha = hashlib.sha256(f"{backbone}:{fold}:split".encode()).hexdigest()
            outer = build_locked_outer_result(
                fold=fold,
                run_contract_digest=contract_digest,
                split_membership_digest=split_sha,
                y_true=[0, 1],
                y_prob=[0.2, 0.8],
                test_logits=[-1.0, 1.0],
            )
            import torch

            torch.save(
                {
                    "fold": fold,
                    "backbone": backbone,
                    "run_contract": {**contract_body, "digest": contract_digest},
                    "run_contract_digest": contract_digest,
                    "split_membership_digest": split_sha,
                    "locked_outer_test_evaluation": {
                        "evaluation_count": 1,
                        "metrics": {"roc_auc": 1.0},
                        "y_true": list(outer["y_true"]),
                        "y_prob": list(outer["y_prob"]),
                        "test_logits": list(outer["test_logits"]),
                        "calibrated_prob": list(outer["y_prob"]),
                        "split_membership_digest": split_sha,
                        "result_digest": outer["result_digest"],
                        "raw_identifiers_stored": False,
                    },
                },
                checkpoint,
            )
            checkpoint_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            outer_path = model_dir / f"method_x_3d_fold_{fold}_outer_eval.json"
            _write_json(outer_path, outer)
            _write_json(
                model_dir / f"method_x_3d_fold_{fold}_state.json",
                {
                    "schema_version": 1,
                    "stage": "FINALIZED",
                    "fold": fold,
                    "contract_digest": contract_digest,
                    "details": {
                        "final_checkpoint": checkpoint.relative_to(root).as_posix(),
                        "final_checkpoint_sha256": checkpoint_sha,
                        "locked_result": outer_path.relative_to(root).as_posix(),
                        "outer_result_digest": outer["result_digest"],
                        "split_membership_digest": split_sha,
                    },
                },
            )
            fold_metrics.append(
                {"fold": fold, "outer_test_evaluation_count": 1}
            )
        _write_json(
            model_dir / "three_d_metrics.json",
            {
                "aggregate": {
                    "status": "COMPLETED",
                    "backbone": backbone,
                    "folds": 5,
                },
                "folds": fold_metrics,
            },
        )
        rows.append(
            {
                "backbone": backbone,
                "status": "COMPLETED",
                "output_dir": f"models/{backbone}",
            }
        )
    for backbone in EXCLUDED_BACKBONES:
        model_dir = root / "models" / backbone
        model_dir.mkdir(parents=True)
        _write_json(model_dir / "smoke.json", {"backbone": backbone, "status": "PASS"})
        rows.append(
            {
                "backbone": backbone,
                "status": "PROTOCOL_EXCLUDED_NOT_STARTED",
                "output_dir": f"models/{backbone}",
                "training": {"status": "PROTOCOL_EXCLUDED_NOT_STARTED"},
            }
        )
    _write_json(
        root / "amendments" / "method_x_battery_v3_five_model_amendment.v1.json",
        _amendment_payload(config_sha),
    )
    manifest = root / "battery_manifest.json"
    _write_json(manifest, {"locked_inputs": {"config_sha256": config_sha}, "models": rows})
    return manifest


def _migrate_fixture_contract(
    root: Path, *, backbone: str = "resnet18_3d", scientific_drift: bool = False
) -> Path:
    from src.utils.reproducibility import canonical_digest

    model_dir = root / "models" / backbone
    wrapper = json.loads((model_dir / "run_contract.json").read_text(encoding="utf-8"))
    old_contract = dict(wrapper["run_contract"])
    old_digest = str(wrapper["run_contract_digest"])
    new_body = {key: value for key, value in old_contract.items() if key != "digest"}
    new_body["parameters"] = dict(new_body["parameters"])
    if scientific_drift:
        new_body["parameters"]["dropout"] = 0.9
    else:
        new_body["parameters"]["source_contract_digest"] = "b" * 64
    new_digest = canonical_digest(new_body)
    folds = []
    for fold in range(5):
        checkpoint = model_dir / f"method_x_3d_fold_{fold}.pt"
        outer_path = model_dir / f"method_x_3d_fold_{fold}_outer_eval.json"
        outer = json.loads(outer_path.read_text(encoding="utf-8"))
        checkpoint_sha = _sha(checkpoint)
        folds.append(
            {
                "fold": fold,
                "pre_checkpoint_sha256": checkpoint_sha,
                "post_checkpoint_sha256": checkpoint_sha,
                "pre_outer_result_digest": outer["result_digest"],
                "post_outer_result_digest": outer["result_digest"],
            }
        )
    receipt = {
        "schema_version": 1,
        "artifact": "method_x_source_only_contract_migration_receipt",
        "append_only": True,
        "old_run_contract_digest": old_digest,
        "new_run_contract_digest": new_digest,
        "folds": folds,
    }
    receipt["integrity_sha256"] = canonical_digest(receipt)
    relative = (
        Path("source_only_contract_migrations")
        / f"{old_digest[:16]}_to_{new_digest[:16]}.completed.json"
    )
    receipt_path = model_dir / relative
    _write_json(receipt_path, receipt)
    receipt_sha = _sha(receipt_path)
    for fold in range(5):
        state_path = model_dir / f"method_x_3d_fold_{fold}_state.json"
        state = json.loads(state_path.read_text(encoding="utf-8"))
        state["contract_digest"] = new_digest
        state["source_history"] = [
            {
                "from_run_contract_digest": old_digest,
                "to_run_contract_digest": new_digest,
                "receipt_path": relative.as_posix(),
                "receipt_sha256": receipt_sha,
            }
        ]
        _write_json(state_path, state)
    _write_json(
        model_dir / "run_contract.json",
        {
            "run_contract": {**new_body, "digest": new_digest},
            "run_contract_digest": new_digest,
        },
    )
    return receipt_path


def test_terminal_verifier_requires_all_25_linked_exactly_once_folds(
    tmp_path: Path,
) -> None:
    from src.research.brainiac_battery_terminal import verify_battery_terminal

    manifest = _completed_battery_fixture(tmp_path)
    report = verify_battery_terminal(
        manifest_path=manifest,
        battery_root=tmp_path,
        project_root=tmp_path,
        hash_checkpoints=True,
    )
    assert report["terminal"] is True
    assert report["outer_evaluation_count"] == 25

    checkpoint = tmp_path / "models" / "resnet34_3d" / "method_x_3d_fold_2.pt"
    checkpoint.write_bytes(b"tampered")
    blocked = verify_battery_terminal(
        manifest_path=manifest,
        battery_root=tmp_path,
        project_root=tmp_path,
        hash_checkpoints=True,
    )
    assert blocked["terminal"] is False
    assert any("CHECKPOINT_SHA256_MISMATCH" in item for item in blocked["blockers"])


def test_terminal_verifier_cannot_authorize_without_checkpoint_rehash(
    tmp_path: Path,
) -> None:
    from src.research.brainiac_battery_terminal import verify_battery_terminal

    manifest = _completed_battery_fixture(tmp_path)
    report = verify_battery_terminal(
        manifest_path=manifest,
        battery_root=tmp_path,
        project_root=tmp_path,
        hash_checkpoints=False,
    )
    assert report["terminal"] is False
    assert report["status"] == "BLOCKED"
    assert report["blockers"] == ["CHECKPOINT_HASH_RECOMPUTATION_REQUIRED"]


def test_terminal_verifier_accepts_only_receipt_bound_source_only_chain(
    tmp_path: Path,
) -> None:
    from src.research.brainiac_battery_terminal import verify_battery_terminal

    manifest = _completed_battery_fixture(tmp_path)
    _migrate_fixture_contract(tmp_path)
    report = verify_battery_terminal(
        manifest_path=manifest,
        battery_root=tmp_path,
        project_root=tmp_path,
        hash_checkpoints=True,
    )
    assert report["terminal"] is True
    migrated = next(row for row in report["models"] if row["backbone"] == "resnet18_3d")
    assert {fold["source_transition_count"] for fold in migrated["folds"]} == {1}


@pytest.mark.parametrize("tamper", ["missing", "bytes"])
def test_terminal_verifier_rejects_missing_or_tampered_source_receipt(
    tmp_path: Path, tamper: str
) -> None:
    from src.research.brainiac_battery_terminal import verify_battery_terminal

    manifest = _completed_battery_fixture(tmp_path)
    receipt = _migrate_fixture_contract(tmp_path)
    if tamper == "missing":
        receipt.unlink()
    else:
        receipt.write_bytes(receipt.read_bytes() + b" ")
    report = verify_battery_terminal(
        manifest_path=manifest,
        battery_root=tmp_path,
        project_root=tmp_path,
        hash_checkpoints=True,
    )
    assert report["terminal"] is False
    assert any("FINALIZED_SOURCE_COMPATIBILITY" in code for code in report["blockers"])


def test_terminal_verifier_rejects_scientific_drift_disguised_as_source_migration(
    tmp_path: Path,
) -> None:
    from src.research.brainiac_battery_terminal import verify_battery_terminal

    manifest = _completed_battery_fixture(tmp_path)
    _migrate_fixture_contract(tmp_path, scientific_drift=True)
    report = verify_battery_terminal(
        manifest_path=manifest,
        battery_root=tmp_path,
        project_root=tmp_path,
        hash_checkpoints=True,
    )
    assert report["terminal"] is False
    assert any("FINALIZED_SOURCE_COMPATIBILITY" in code for code in report["blockers"])


def test_terminal_verifier_rejects_checkpoint_embedded_contract_tamper(
    tmp_path: Path,
) -> None:
    import torch

    from src.research.brainiac_battery_terminal import verify_battery_terminal
    from src.utils.reproducibility import canonical_digest

    manifest = _completed_battery_fixture(tmp_path)
    model_dir = tmp_path / "models" / "resnet34_3d"
    checkpoint = model_dir / "method_x_3d_fold_2.pt"
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    tampered_body = {
        key: value for key, value in payload["run_contract"].items() if key != "digest"
    }
    tampered_body["parameters"] = dict(tampered_body["parameters"])
    tampered_body["parameters"]["resolution"] = 192
    tampered_digest = canonical_digest(tampered_body)
    payload["run_contract"] = {**tampered_body, "digest": tampered_digest}
    payload["run_contract_digest"] = tampered_digest
    torch.save(payload, checkpoint)
    state_path = model_dir / "method_x_3d_fold_2_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["details"]["final_checkpoint_sha256"] = _sha(checkpoint)
    _write_json(state_path, state)

    report = verify_battery_terminal(
        manifest_path=manifest,
        battery_root=tmp_path,
        project_root=tmp_path,
        hash_checkpoints=True,
    )
    assert report["terminal"] is False
    assert any("FINALIZED_SOURCE_COMPATIBILITY" in code for code in report["blockers"])
