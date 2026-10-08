"""Read-only fast-track evidence chain. A missing or changed byte fails closed."""
from __future__ import annotations

from pathlib import Path

from scripts.mx_fasttrack_v2.run import (
    CHECKPOINT_STAGING_ROOT, FASTTRACK_CONFIGS, FASTTRACK_PROTOCOL_IDS, SOURCE_PATHS, TASKS,
    choose_without_contaminated_finetune_validation, cross_check_metrics_membership,
    config_for_tree, fold_membership_digests, inner, read, sha)


def check_inputs(root: Path, out: Path, *, hash_cache: bool = True) -> tuple[dict, dict]:
    reports = out / "reports"
    audit_path = reports / "data_semantics_audit.json"
    prereg_path = reports / "fasttrack_preregistration.json"
    audit, prereg = read(audit_path), read(prereg_path)
    preflight_path = reports / "machine_preflight.json"
    preflight = read(preflight_path)
    if (audit.get("status") != "READY" or audit.get("verified_full_hdf5_real_n") != 1669
        or audit.get("cohort_label_perfectly_confounding") is not True
        or prereg.get("data_audit_sha256") != sha(audit_path)
        or prereg.get("cache_sha256") != audit.get("cache_sha256")
        or prereg.get("status") != "LOCKED_BEFORE_GPU"
        or prereg.get("outer_test_access_count") != 0):
        raise RuntimeError("FASTTRACK_INPUT_AUDIT_OR_PREREG_CHANGED")
    if (preflight.get("status") != "READY" or preflight.get("blockers") != []
        or prereg.get("machine_preflight_sha256") != sha(preflight_path)
        or preflight.get("live_method_x_python_writers") != []
        or (preflight.get("source_package") or {}).get("all_match") is not True):
        raise RuntimeError("FASTTRACK_MACHINE_PREFLIGHT_CHANGED_OR_INVALID")
    # Resolve the config that belongs to *this* tree's preregistered protocol id, so
    # a sibling protocol version can never be validated against the wrong file.
    protocol_id = prereg.get("protocol_id")
    config_relative = FASTTRACK_CONFIGS.get(str(protocol_id))
    if not config_relative:
        raise RuntimeError(f"FASTTRACK_PROTOCOL_ID_UNKNOWN:{protocol_id}")
    config = root / config_relative
    if prereg.get("fasttrack_config_sha256") != sha(config):
        raise RuntimeError("FASTTRACK_CONFIG_CHANGED")
    source = prereg.get("training_source_sha256")
    if not isinstance(source, dict) or set(source) != set(SOURCE_PATHS):
        raise RuntimeError("FASTTRACK_SOURCE_BINDING_MISSING")
    for relative, expected in source.items():
        if not isinstance(relative, str) or Path(relative).is_absolute() or ".." in Path(relative).parts:
            raise RuntimeError("FASTTRACK_SOURCE_PATH_UNSAFE")
        if sha(root / relative) != expected:
            raise RuntimeError(f"FASTTRACK_SOURCE_CHANGED:{relative}")
    splits = audit.get("split_sha256") or {}
    expected = {*(f"fold_{fold}_{kind}" for fold in range(5) for kind in ("test", "train")), "external_holdout"}
    if set(splits) != expected:
        raise RuntimeError("FASTTRACK_SPLIT_BINDING_INCOMPLETE")
    for split, expected_digest in splits.items():
        if sha(root / "splits" / f"{split}.csv") != expected_digest:
            raise RuntimeError(f"FASTTRACK_SPLIT_CHANGED:{split}")
    if sha(root / "docs/fasttrack_private/cache_full_index.csv") != audit.get("index_sha256"):
        raise RuntimeError("FASTTRACK_INDEX_CHANGED")
    if hash_cache and sha(root / "data/cached/t1_volume_cache_128.h5") != prereg["cache_sha256"]:
        raise RuntimeError("FASTTRACK_CACHE_CHANGED")
    return prereg, audit


def check_lock(root: Path, out: Path, *, hash_cache: bool = True) -> dict:
    prereg, _ = check_inputs(root, out, hash_cache=hash_cache)
    reports = out / "reports"
    lock = read(reports / "fasttrack_inner_lock.json")
    if (lock.get("status") != "LOCKED_INNER_ONLY_EXPLORATORY"
        or lock.get("preregistration_sha256") != sha(reports / "fasttrack_preregistration.json")
        or lock.get("outer_test_access_count") != 0
        or lock.get("outer_test_used_for_selection") is not False
        or lock.get("fine_tuning_inner_validation_valid_for_selection") is not False
        or lock.get("external_validation") is not False
        or lock.get("hybrid_pair") != ["hybrid_baseline", "hybrid_dense"]):
        raise RuntimeError("FASTTRACK_LOCK_SCOPE_OR_PROVENANCE_INVALID")
    ids = {task[0] for task in TASKS}
    if set(lock.get("tasks") or {}) != ids or lock.get("final_task") not in {
        "mx04_scratch", "hybrid_baseline", "hybrid_dense"
    }:
        raise RuntimeError("FASTTRACK_LOCK_TASKS_INVALID")
    if set(lock.get("exploratory_outer_tasks", [])) != {"mx04_scratch", "mx04_head_last"}:
        raise RuntimeError("FASTTRACK_MX04_COMPARISON_NOT_PREDECLARED")
    for tid in sorted(ids):
        task = lock["tasks"][tid]
        directory = out / "inner_tasks" / tid
        metric = directory / "three_d_metrics.json"
        contract = directory / "run_contract.json"
        if (task.get("metrics_sha256") != sha(metric)
            or task.get("run_contract_sha256") != sha(contract)
            or task.get("run_contract_digest") != read(contract).get("run_contract_digest")
            or task.get("run_contract_digest") != read(contract).get("run_contract", {}).get("digest")):
            raise RuntimeError(f"FASTTRACK_METRIC_OR_CONTRACT_CHANGED:{tid}")
        metrics = read(metric)
        inner(metric)
        # Same helper the lock writer uses, so the recorded and the verified value
        # cannot diverge.  This previously re-derived the digest from the metrics
        # fold rows, which never carried it.
        membership = fold_membership_digests(directory, task_id=tid)
        cross_check_metrics_membership(directory, task_id=tid, membership=membership)
        if (metrics["aggregate"].get("run_contract_digest") != task["run_contract_digest"]
                or membership != task.get("fold_membership_digest")):
            raise RuntimeError(f"FASTTRACK_FOLD_CONTRACT_CHANGED:{tid}")
        if len(task.get("checkpoint_sha256") or []) != 5:
            raise RuntimeError(f"FASTTRACK_CHECKPOINT_COUNT_INVALID:{tid}")
        for fold in range(5):
            if sha(directory / f"method_x_3d_fold_{fold}.pt") != task["checkpoint_sha256"][fold]:
                raise RuntimeError(f"FASTTRACK_CHECKPOINT_CHANGED:{tid}:{fold}")
        prune_plan = reports / f"staging_prune_{tid}_plan.json"
        prune_receipt = reports / f"staging_prune_{tid}.json"
        if (task.get("staging_prune_plan_sha256") != sha(prune_plan)
            or task.get("staging_prune_receipt_sha256") != sha(prune_receipt)):
            raise RuntimeError(f"FASTTRACK_STAGING_PRUNE_EVIDENCE_CHANGED:{tid}")
        plan, receipt = read(prune_plan), read(prune_receipt)
        if (plan.get("task_id") != tid or plan.get("run_contract_digest") != task["run_contract_digest"]
            or receipt.get("status") != "COMPLETED" or receipt.get("task_id") != tid
            or receipt.get("plan_sha256") != sha(prune_plan)
            or receipt.get("c_payloads_absent_after_verified_h_commit") is not True
            or receipt.get("preexisting_staging_outside_task_preserved") is not True):
            raise RuntimeError(f"FASTTRACK_STAGING_PRUNE_EVIDENCE_INVALID:{tid}")
        for row in plan.get("c_files") or []:
            relative = Path(str(row.get("relative_path") or ""))
            candidate = (CHECKPOINT_STAGING_ROOT / relative).resolve()
            if (relative.is_absolute() or ".." in relative.parts
                or not candidate.is_relative_to(CHECKPOINT_STAGING_ROOT.resolve())
                or candidate.exists()):
                raise RuntimeError(f"FASTTRACK_COMPLETED_STAGING_PAYLOAD_REAPPEARED:{tid}")
        if plan.get("scope") == "IMPORTED_FINAL_SOURCE_ONLY_PARENT_C_STAGING_PRESERVED":
            if tid != "hybrid_baseline" or plan.get("c_files") != []:
                raise RuntimeError("FASTTRACK_IMPORTED_STAGING_SCOPE_INVALID")
            from scripts.mx_fasttrack_v2_4.import_resume import imported_baseline_prune

            imported_baseline_prune(
                root, out, directory, task["run_contract_digest"], 5,
                verify_only=True,
            )
    config = config_for_tree(root, out)
    margin, count = config["practical_improvement_margin_loss"], config["minimum_folds_improved"]
    comparison, baseline, hybrid, selected, advantage = (
        choose_without_contaminated_finetune_validation(
            lock["tasks"], margin=margin, minimum_folds=count))
    if (lock.get("comparisons") != comparison or lock.get("mx04_candidate") != baseline
        or lock.get("hybrid_candidate") != hybrid or lock.get("final_task") != selected
        or lock.get("hybrid_over_mx04_practical_rule") != advantage
        or lock.get("practical_margin_loss") != margin
        or lock.get("minimum_folds_improved") != count):
        raise RuntimeError("FASTTRACK_SELECTION_LOCK_DECISION_CHANGED")
    if prereg["protocol_id"] not in FASTTRACK_PROTOCOL_IDS:
        raise RuntimeError("FASTTRACK_PROTOCOL_ID_INVALID")
    return lock
