"""Receipt-backed import of completed MX-04 and terminal MX-05 baseline folds.

The parent tree and its C-side checkpoint staging are read-only.  A crash during
copying is recoverable: each destination file is either an exact copied byte
stream or the next attempt refuses it.  No outer-test artifact is imported.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.mx_fasttrack_v2.run import (
    CHECKPOINT_STAGING_ROOT, TASKS, build_fold_manifest, read, sha,
    task_command, update_state, write_once,
)
from src.training.method_x_resume import (
    _migration_validate_checkpoint, _migration_validate_completed_receipt,
    source_history_binds_digest,
)

PARENT = Path("results/final_results/method_x_fasttrack_v2_20260926_mx04_import")
PROTOCOL = "method_x_fasttrack_v2_20260927_mx05_gate_import"
RECEIPT = "mx05_resume_import.v1.json"
MX04 = ("mx04_scratch", "mx04_head_last")
ALLOWED_PARENT_DRIFT = {
    "scripts/train_3d_mri.py", "src/training/method_x_resume.py",
    "scripts/mx_fasttrack_v2/run.py",
    "scripts/mx_fasttrack_v2/integrity.py",
}
SKIP_BASELINE = {"BLOCKED_RUN.json", "three_d_metrics.json", "selection_evidence.json"}


def _copy_exact(source: Path, target: Path) -> str:
    source_hash = sha(source)
    if target.is_file():
        if sha(target) != source_hash:
            raise RuntimeError(f"V24_IMPORT_DESTINATION_DRIFT:{target}")
        return source_hash
    if target.exists():
        raise RuntimeError(f"V24_IMPORT_DESTINATION_NOT_FILE:{target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.importing")
    try:
        shutil.copy2(source, temporary)
        if sha(temporary) != source_hash:
            raise RuntimeError(f"V24_IMPORT_COPY_HASH_MISMATCH:{target}")
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return source_hash


def _copy_task(parent: Path, out: Path, task: str) -> dict:
    source = parent / "inner_tasks" / task
    target = out / "inner_tasks" / task
    if not source.is_dir():
        raise RuntimeError(f"V24_IMPORT_PARENT_TASK_MISSING:{task}")
    copied: dict[str, str] = {}
    for path in sorted(source.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(source)
        if task == "hybrid_baseline" and path.name in SKIP_BASELINE:
            continue
        if "outer_eval" in path.name.casefold():
            raise RuntimeError(f"V24_IMPORT_OUTER_ARTIFACT_FORBIDDEN:{task}")
        copied[relative.as_posix()] = _copy_exact(path, target / relative)
    return copied


def _check_parent(root: Path, parent: Path, out: Path) -> tuple[dict, dict]:
    if not parent.is_dir() or parent == out or out.is_relative_to(parent):
        raise RuntimeError("V24_IMPORT_PARENT_OR_TARGET_INVALID")
    prereg_path = parent / "reports/fasttrack_preregistration.json"
    old = read(prereg_path)
    new = read(out / "reports/fasttrack_preregistration.json")
    if (old.get("protocol_id") != "method_x_fasttrack_v2_20260926_mx04_import"
        or new.get("protocol_id") != PROTOCOL
        or old.get("status") != "LOCKED_BEFORE_GPU"
        or new.get("status") != "LOCKED_BEFORE_GPU"):
        raise RuntimeError("V24_IMPORT_PREREGISTRATION_INVALID")
    drift = {
        relative for relative, expected in old["training_source_sha256"].items()
        if sha(root / relative) != expected
    }
    if drift != ALLOWED_PARENT_DRIFT:
        raise RuntimeError(f"V24_IMPORT_SOURCE_DELTA_UNEXPECTED:{sorted(drift)}")
    import_manifest = parent / "reports/mx04_import_manifest.v1.json"
    historical = read(import_manifest)
    if (historical.get("manifest_id") !=
            "method_x_fasttrack_v2_20260926_mx04_import_manifest.v1"
        or historical.get("status") != "MX04_IMPORTED_UNDER_PROOF"):
        raise RuntimeError("V24_IMPORT_MX04_PARENT_MANIFEST_INVALID")
    if any((parent / "reports").glob("evaluation_outer_*")):
        raise RuntimeError("V24_IMPORT_PARENT_OUTER_ALREADY_ACCESSED")
    old_state = read(parent / "reports/fasttrack_task_state.json")
    for task in MX04:
        row = (old_state.get("tasks") or {}).get(task) or {}
        metrics = parent / "inner_tasks" / task / "three_d_metrics.json"
        if row.get("status") != "COMPLETED" or row.get("metrics_sha256") != sha(metrics):
            raise RuntimeError(f"V24_IMPORT_MX04_NOT_TERMINAL:{task}")
        aggregate = read(metrics).get("aggregate") or {}
        if (aggregate.get("status") != "COMPLETED"
            or aggregate.get("hpo_selection_only") is not True
            or aggregate.get("outer_test_access_count") != 0
            or aggregate.get("completed_folds") != 5):
            raise RuntimeError(f"V24_IMPORT_MX04_SCOPE_INVALID:{task}")
    return old, old_state


def _check_baseline(parent: Path) -> tuple[str, dict[str, str]]:
    import torch
    source = parent / "inner_tasks/hybrid_baseline"
    contract = read(source / "run_contract.json")
    old_digest = contract.get("run_contract_digest")
    if old_digest != (contract.get("run_contract") or {}).get("digest"):
        raise RuntimeError("V24_IMPORT_BASELINE_CONTRACT_INVALID")
    hashes: dict[str, str] = {}
    for fold in range(5):
        path = source / f"method_x_3d_fold_{fold}.pt"
        _migration_validate_checkpoint(
            path, expected_contract=contract["run_contract"],
            old_digest=str(old_digest), fold=fold,
        )
        state = read(source / f"method_x_3d_fold_{fold}_state.json")
        details = state.get("details") or {}
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        selection = checkpoint.get("selection_only_evaluation") or {}
        if (state.get("stage") != "FINALIZED"
            or state.get("contract_digest") != old_digest
            or details.get("selection_only") is not True
            or details.get("hpo_selection_only") is not True
            or details.get("outer_test_access_count") != 0
            or details.get("checkpoint_sha256") != sha(path)
            or checkpoint.get("run_contract_digest") != old_digest
            or checkpoint.get("fold_terminal") is not True
            or checkpoint.get("hpo_selection_only") is not True
            or checkpoint.get("outer_test_access_count") != 0
            or selection.get("fold") != fold
            or selection.get("run_contract_digest") != old_digest
            or selection.get("split_membership_digest") != details.get("split_membership_digest")
            or selection.get("outer_test_access_count") != 0
            or selection.get("outer_test_used_for_selection") is not False
            or (source / f"method_x_3d_fold_{fold}_outer_eval.json").exists()):
            raise RuntimeError(f"V24_IMPORT_BASELINE_FOLD_INVALID:{fold}")
        hashes[str(fold)] = sha(path)
    return str(old_digest), hashes


def _new_command_hashes(root: Path, parent: Path, out: Path, old_state: dict) -> dict:
    battery = root / "results/final_results/method_x_model_battery_v3_depth_and_diversity_20260729"
    battery_manifest = read(battery / "battery_manifest.json")
    result = {}
    for task in TASKS:
        if task[0] not in (*MX04, "hybrid_baseline"):
            continue
        fold_manifest = (
            build_fold_manifest(root, battery, out)
            if task[0] == "mx04_head_last" else None
        )
        command, _ = task_command(task, root, battery, out, battery_manifest, fold_manifest)
        digest = hashlib.sha256(json.dumps(command, ensure_ascii=False).encode()).hexdigest()
        remapped = [str(part).replace(str(out), str(parent)) for part in command]
        old_digest = hashlib.sha256(json.dumps(remapped, ensure_ascii=False).encode()).hexdigest()
        prior = (old_state.get("tasks") or {}).get(task[0]) or {}
        if old_digest != prior.get("command_sha256"):
            raise RuntimeError(f"V24_IMPORT_COMMAND_NOT_EQUIVALENT:{task[0]}")
        result[task[0]] = digest
    return result


def _mx04_prune_plan(root: Path, parent: Path, out: Path, task: str) -> None:
    old_plan = read(parent / "reports" / f"staging_prune_{task}_plan.json")
    old_receipt = read(parent / "reports" / f"staging_prune_{task}.json")
    if (old_receipt.get("status") != "COMPLETED"
        or old_receipt.get("plan_sha256") != sha(parent / "reports" / f"staging_prune_{task}_plan.json")
        or old_receipt.get("c_payloads_absent_after_verified_h_commit") is not True):
        raise RuntimeError(f"V24_IMPORT_PARENT_PRUNE_INVALID:{task}")
    for row in old_plan.get("c_files") or []:
        if (CHECKPOINT_STAGING_ROOT / row["relative_path"]).exists():
            raise RuntimeError(f"V24_IMPORT_PARENT_C_STAGING_REAPPEARED:{task}")
    h_folds = []
    for fold in range(5):
        directory = out / "inner_tasks" / task
        final = directory / f"method_x_3d_fold_{fold}.pt"
        mirror = directory / "checkpoint_mirror" / f"fold_{fold}" / "resume_current.pt"
        mirror_state = mirror.with_name("checkpoint_mirror_state.json")
        binding = read(mirror_state)
        if (binding.get("payload_sha256") != sha(mirror)
            or binding.get("run_id") != old_plan.get("run_id")
            or binding.get("fold") != fold):
            raise RuntimeError(f"V24_IMPORT_MX04_MIRROR_INVALID:{task}:{fold}")
        h_folds.append({
            "fold": fold,
            "final_checkpoint": final.relative_to(root).as_posix(),
            "final_checkpoint_sha256": sha(final),
            "mirror": mirror.relative_to(root).as_posix(),
            "mirror_sha256": sha(mirror),
            "mirror_state": mirror_state.relative_to(root).as_posix(),
            "mirror_state_sha256": sha(mirror_state),
        })
    if [row["final_checkpoint_sha256"] for row in h_folds] != [
        row["final_checkpoint_sha256"] for row in old_plan["h_folds"]
    ]:
        raise RuntimeError(f"V24_IMPORT_MX04_CHECKPOINT_DRIFT:{task}")
    plan = {**old_plan, "h_folds": h_folds,
            "c_payload_removal_location": "parent_tree_before_v24_import",
            "parent_plan_sha256": sha(parent / "reports" / f"staging_prune_{task}_plan.json")}
    write_once(out / "reports" / f"staging_prune_{task}_plan.json", plan)


def verify_import(root: Path, out: Path) -> dict:
    path = out / "reports" / RECEIPT
    receipt = read(path)
    parent = root / PARENT
    if (receipt.get("status") != "IMPORTED_UNDER_HASH_PROOF"
        or receipt.get("protocol_id") != PROTOCOL
        or receipt.get("parent_preregistration_sha256") != sha(parent / "reports/fasttrack_preregistration.json")
        or receipt.get("new_preregistration_sha256") != sha(out / "reports/fasttrack_preregistration.json")
        or receipt.get("parent_mx04_import_sha256") != sha(parent / "reports/mx04_import_manifest.v1.json")):
        raise RuntimeError("V24_IMPORT_RECEIPT_INVALID")
    for task in MX04:
        for relative, expected in receipt["copied_files"][task].items():
            if sha(out / "inner_tasks" / task / relative) != expected:
                raise RuntimeError(f"V24_IMPORT_MX04_COPY_CHANGED:{task}:{relative}")
    for fold, expected in receipt["baseline_checkpoint_sha256"].items():
        path = out / "inner_tasks/hybrid_baseline" / f"method_x_3d_fold_{fold}.pt"
        if sha(path) != expected:
            raise RuntimeError(f"V24_IMPORT_BASELINE_CHECKPOINT_CHANGED:{fold}")
    return receipt


def _seed_mx04_task_state(out: Path, old_state: dict, commands: dict,
                          receipt_path: Path) -> None:
    state_path = out / "reports/fasttrack_task_state.json"
    if state_path.exists():
        state = read(state_path)
    else:
        state = {"schema_version": 1, "tasks": {},
                 "identity_sha256": sha(out / "reports/fasttrack_preregistration.json")}
    if state.get("identity_sha256") != sha(out / "reports/fasttrack_preregistration.json"):
        raise RuntimeError("V24_IMPORT_TASK_STATE_IDENTITY_DRIFT")
    changed = False
    for task in MX04:
        metrics = out / "inner_tasks" / task / "three_d_metrics.json"
        row = dict((old_state.get("tasks") or {})[task])
        row.update({"command_sha256": commands[task], "metrics_sha256": sha(metrics),
                    "imported": True, "import_receipt_sha256": sha(receipt_path)})
        if task in state["tasks"] and state["tasks"][task] != row:
            raise RuntimeError(f"V24_IMPORT_TASK_STATE_DRIFT:{task}")
        if task not in state["tasks"]:
            state["tasks"][task] = row
            changed = True
    if changed or not state_path.exists():
        update_state(state_path, state)


def import_previous(root: Path, out: Path) -> dict:
    from scripts.mx_fasttrack_v2.integrity import check_inputs
    root, out = root.resolve(), out.resolve()
    if root != ROOT.resolve() or not out.is_relative_to(root) or out == root:
        raise RuntimeError("V24_IMPORT_ROOT_INVALID")
    check_inputs(root, out)
    if (out / "reports" / RECEIPT).exists():
        receipt = verify_import(root, out)
        parent = root / PARENT
        old_state = read(parent / "reports/fasttrack_task_state.json")
        _seed_mx04_task_state(
            out, old_state, receipt["command_sha256"], out / "reports" / RECEIPT
        )
        return receipt
    parent = root / PARENT
    old_prereg, old_state = _check_parent(root, parent, out)
    old_digest, baseline_hashes = _check_baseline(parent)
    commands = _new_command_hashes(root, parent, out, old_state)
    if any((out / "reports").glob("evaluation_outer_*")):
        raise RuntimeError("V24_IMPORT_TARGET_OUTER_ALREADY_ACCESSED")
    copies = {task: _copy_task(parent, out, task) for task in (*MX04, "hybrid_baseline")}
    for task in MX04:
        _mx04_prune_plan(root, parent, out, task)
    receipt = {
        "schema_version": 1,
        "status": "IMPORTED_UNDER_HASH_PROOF",
        "protocol_id": PROTOCOL,
        "parent_protocol_id": old_prereg["protocol_id"],
        "parent_preregistration_sha256": sha(parent / "reports/fasttrack_preregistration.json"),
        "new_preregistration_sha256": sha(out / "reports/fasttrack_preregistration.json"),
        "parent_mx04_import_sha256": sha(parent / "reports/mx04_import_manifest.v1.json"),
        "parent_source_delta": sorted(ALLOWED_PARENT_DRIFT),
        "copied_files": copies,
        "baseline_checkpoint_sha256": baseline_hashes,
        "baseline_historical_run_contract_digest": old_digest,
        "command_sha256": commands,
        "mx04_retrained": False,
        "mx05_baseline_retrained": False,
        "outer_test_access_count": 0,
        "parent_c_staging_preserved": True,
        "new_gpu_training_occurs_only_after_new_preregistration": True,
        "claim_cap": "POST_HOC_INTERNAL_TECHNICAL_EVIDENCE",
        "historical_source_disclosure": (
            "Imported MX04 and hybrid_baseline checkpoints were produced under the parent "
            "preregistration, not under the new source bytes. Their checkpoint bytes stay "
            "immutable; source-only linkage is checked before reuse."
        ),
    }
    receipt_path = out / "reports" / RECEIPT
    write_once(receipt_path, receipt)
    _seed_mx04_task_state(out, old_state, commands, receipt_path)
    return verify_import(root, out)


def imported_baseline_prune(root: Path, out: Path, model_root: Path, digest: str,
                            expected_folds: int, *, verify_only: bool = False) -> dict:
    """Attest an imported final without deleting its *parent* C-stage payloads."""
    if expected_folds != 5:
        raise RuntimeError("V24_IMPORTED_PRUNE_FOLD_COUNT_INVALID")
    receipt = verify_import(root, out)
    old_digest = receipt["baseline_historical_run_contract_digest"]
    migration_dir = model_root / "source_only_contract_migrations"
    if not migration_dir.is_dir():
        raise RuntimeError("V24_IMPORTED_BASELINE_MIGRATION_MISSING")
    h_folds = []
    for fold in range(5):
        final = model_root / f"method_x_3d_fold_{fold}.pt"
        state_path = model_root / f"method_x_3d_fold_{fold}_state.json"
        state = read(state_path)
        details = state.get("details") or {}
        payload = _migration_validate_checkpoint(
            final, expected_contract=read(model_root / "run_contract.json")["run_contract"],
            old_digest=old_digest, fold=fold,
            allowed_digests={old_digest, digest},
        )
        if (state.get("stage") != "FINALIZED"
            or state.get("contract_digest") != digest
            or not source_history_binds_digest(
                history=state.get("source_history"), from_digest=old_digest,
                to_digest=digest)
            or details.get("checkpoint_sha256") != sha(final)
            or details.get("outer_test_access_count") != 0
            or payload.get("hpo_selection_only") is not True):
            raise RuntimeError(f"V24_IMPORTED_BASELINE_LINK_INVALID:{fold}")
        audit = details.get("source_only_migration_receipt") or {}
        receipt_relative = Path(str(audit.get("path") or ""))
        receipt_path = (model_root / receipt_relative).resolve()
        if (receipt_relative.is_absolute() or ".." in receipt_relative.parts
            or not receipt_path.is_relative_to(migration_dir.resolve())
            or not receipt_path.is_file()
            or audit.get("sha256") != sha(receipt_path)):
            raise RuntimeError(f"V24_IMPORTED_BASELINE_MIGRATION_RECEIPT_INVALID:{fold}")
        migration = _migration_validate_completed_receipt(
            receipt_path, old_digest=old_digest, new_digest=digest
        )
        rows = [row for row in migration.get("folds") or []
                if isinstance(row, dict) and row.get("fold") == fold]
        if (len(rows) != 1
            or rows[0].get("pre_checkpoint_sha256") != receipt["baseline_checkpoint_sha256"][str(fold)]
            or rows[0].get("post_checkpoint_sha256") != sha(final)
            or not any(
                isinstance(link, dict)
                and link.get("from_run_contract_digest") == old_digest
                and link.get("to_run_contract_digest") == digest
                and link.get("receipt_sha256") == sha(receipt_path)
                for link in state.get("source_history") or []
            )):
            raise RuntimeError(f"V24_IMPORTED_BASELINE_MIGRATION_LINK_INVALID:{fold}")
        mirror = model_root / "checkpoint_mirror" / f"fold_{fold}" / "resume_current.pt"
        mirror_state = mirror.with_name("checkpoint_mirror_state.json")
        binding = read(mirror_state)
        if (binding.get("payload_sha256") != sha(mirror)
            or binding.get("run_id") != old_digest[:16]
            or binding.get("fold") != fold):
            raise RuntimeError(f"V24_IMPORTED_BASELINE_MIRROR_INVALID:{fold}")
        h_folds.append({
            "fold": fold, "final_checkpoint": final.relative_to(root).as_posix(),
            "final_checkpoint_sha256": sha(final),
            "mirror": mirror.relative_to(root).as_posix(), "mirror_sha256": sha(mirror),
            "mirror_state": mirror_state.relative_to(root).as_posix(),
            "mirror_state_sha256": sha(mirror_state),
        })
    # The new digest had no C staging at import or during checkpoint reuse.  The
    # old digest's C payloads belong to the parent run and must not be pruned.
    new_stage = CHECKPOINT_STAGING_ROOT / digest[:16] / "amuse_x_hybrid3d"
    if new_stage.exists():
        raise RuntimeError("V24_IMPORTED_BASELINE_NEW_C_STAGE_UNEXPECTED")
    plan = {
        "schema_version": 1, "task_id": "hybrid_baseline",
        "run_contract_digest": digest, "run_id": digest[:16],
        "backbone": "amuse_x_hybrid3d", "expected_folds": 5,
        "metrics_sha256": sha(model_root / "three_d_metrics.json"),
        "h_folds": h_folds, "c_files": [],
        "scope": "IMPORTED_FINAL_SOURCE_ONLY_PARENT_C_STAGING_PRESERVED",
        "import_receipt_sha256": sha(out / "reports" / RECEIPT),
        "parent_run_contract_digest": old_digest,
        "preexisting_staging_outside_task_preserved": True,
    }
    plan_path = out / "reports/staging_prune_hybrid_baseline_plan.json"
    if verify_only:
        if read(plan_path) != plan:
            raise RuntimeError("V24_IMPORTED_BASELINE_PRUNE_PLAN_DRIFT")
    else:
        write_once(plan_path, plan)
    result = {"schema_version": 1, "status": "COMPLETED",
              "task_id": "hybrid_baseline", "plan_sha256": sha(plan_path),
              "removed_file_count": 0, "removed_bytes": 0,
              "parent_c_staging_preserved": True,
              "c_payloads_absent_after_verified_h_commit": True,
              "preexisting_staging_outside_task_preserved": True}
    result_path = out / "reports/staging_prune_hybrid_baseline.json"
    if verify_only:
        if read(result_path) != result:
            raise RuntimeError("V24_IMPORTED_BASELINE_PRUNE_RECEIPT_DRIFT")
    else:
        write_once(result_path, result)
    return result
