"""MX07_fasttrack: verify a complete scoped evidence chain and freeze once.

Partial output is a DRAFT and never unlocks the original MX07 completion gate.
No test labels or subject vectors are read here, only locked aggregate artifacts.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.mx_fasttrack_v2.run import resolve_config
from scripts.mx_fasttrack_v2.integrity import check_inputs, check_lock
from scripts.mx_fasttrack_v2.run import TASKS, read, sha, write_once


def valid_binary_metrics(row: dict, n: int) -> bool:
    if not isinstance(row, dict) or row.get("n") != n:
        return False
    p, m = row.get("positive"), row.get("negative")
    if type(p) is not int or type(m) is not int or p + m != n or min(p, m) <= 0:
        return False
    for field in ("roc_auc", "average_precision", "brier"):
        value = row.get(field)
        if type(value) not in (float, int) or not math.isfinite(value) or not 0 <= value <= 1:
            return False
    return True


def text_once(path: Path, value: str) -> None:
    if path.exists():
        if path.read_text(encoding="utf-8") != value:
            raise RuntimeError(f"IMMUTABLE_DOCUMENT_CHANGED_CREATE_NEW_VERSION:{path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".new")
    with temp.open("x", encoding="utf-8") as stream:
        stream.write(value)
    temp.replace(path)


def evaluated(out: Path, name: str, lock_sha: str, task: str | None,
              *, paired: bool = False, pseudo: bool = False) -> dict | None:
    reports = out / "reports"
    result_path = reports / f"{name}.json"
    state_path = reports / f"{name}_state.json"
    if not result_path.exists() and not state_path.exists():
        return None
    if not result_path.exists() or not state_path.exists():
        # An unfinished resumable evaluation is not a completed result.
        if state_path.exists() and read(state_path).get("status") in {"STARTED", "RUNNING"}:
            return None
        raise RuntimeError(f"MX07_MISMATCHED_EVALUATION_FILES:{name}")
    state, result = read(state_path), read(result_path)
    if (state.get("status") != "COMPLETED" or state.get("result_sha256") != sha(result_path)
        or state.get("lock_sha256") != lock_sha or result.get("lock_sha256") != lock_sha):
        raise RuntimeError(f"MX07_EVALUATION_STATE_OR_HASH_INVALID:{name}")
    if paired:
        if (result.get("status") != "COMPLETED_EXPLORATORY_PAIRED"
            or result.get("folds_evaluated") != list(range(5))
            or result.get("n") != 1336 or len(result.get("by_fold") or []) != 5
            or sum(int(row["n"]) for row in result["by_fold"]) != 1336
            or set(result.get("arms") or {}) != {"hybrid_baseline", "hybrid_dense"}
            or any(not valid_binary_metrics(result["arms"][arm], 1336) for arm in result["arms"])
            or set(result.get("paired_dense_minus_baseline") or {}) !=
                {"roc_auc", "average_precision", "brier"}
            or result.get("test_used_for_model_selection") is not False
            or result.get("external_validation") is not False
            or result.get("subject_level_predictions_written") is not True
            or result.get("subject_level_predictions_exported") is not False
            or result.get("transient_identifier_free_resume_vectors") !=
                "REMOVED_AFTER_AGGREGATE_COMMIT"
            or any(not isinstance(v, dict) or type(v.get("dense_minus_baseline")) not in (int, float)
                   or not math.isfinite(v["dense_minus_baseline"])
                   or not isinstance(v.get("patient_bootstrap_ci95"), list)
                   or len(v["patient_bootstrap_ci95"]) != 2
                   or v.get("valid_draws", 0) < 900
                   for v in result["paired_dense_minus_baseline"].values())):
            raise RuntimeError("MX07_HYBRID_PAIR_CONTENT_INVALID")
    else:
        expected_scope = "random_pseudo_holdout" if pseudo else "internal_fold_outer"
        expected_folds = [0] if pseudo else list(range(5))
        expected_n = 333 if pseudo else 1336
        if (result.get("status") != "COMPLETED" or result.get("scope") != expected_scope
            or result.get("model_task") != task or state.get("task") != task
            or result.get("folds_evaluated") != expected_folds
            or not valid_binary_metrics((result.get("subsets") or {}).get("overall"), expected_n)
            or result.get("external_validation") is not False
            or result.get("subject_level_vectors_exported") is not False):
            raise RuntimeError(f"MX07_EVALUATION_CONTENT_INVALID:{name}")
        if pseudo:
            from scripts.mx_fasttrack_v2.evaluate_locked import verify_holdout_history
            history = reports / "holdout_history_audit.json"
            if sha(history) != result.get("holdout_history_audit_sha256"):
                raise RuntimeError("MX07_PSEUDO_HISTORY_HASH_INVALID")
            verify_holdout_history(ROOT, history)
    return result


def xai_status(root: Path, out: Path, lock_sha: str, selected: str,
               checkpoints: list[str]) -> tuple[str, dict | None]:
    path = out / "reports/xai_limited_summary.json"
    state = out / "reports/xai_limited_state.json"
    if not path.exists() and not state.exists():
        return "NOT_DONE_OPTIONAL", None
    if state.exists() and not path.exists() and read(state).get("status") in {"STARTED", "RUNNING"}:
        return "NOT_DONE_STARTED", None
    if not path.exists() or not state.exists():
        raise RuntimeError("MX07_XAI_RESULT_STATE_MISMATCH")
    result, recorded = read(path), read(state)
    if (recorded.get("status") != "COMPLETED" or recorded.get("result_sha256") != sha(path)
        or recorded.get("lock_sha256") != lock_sha
        or result.get("lock_sha256") != lock_sha or result.get("selected_task") != selected
        or result.get("n_images") != 40 or result.get("scope") != "inner_validation_probe_8_per_fold"
        or result.get("attribution") != "NORMALIZED_ABSOLUTE_INPUT_GRADIENT_NOT_GRADCAM"
        or result.get("anatomical_or_causal_region_claim_allowed") is not False
        or result.get("subject_level_outputs_written") is not False
        or [row.get("checkpoint_sha256") for row in result.get("folds", [])] != checkpoints
        or [row.get("fold") for row in result.get("folds", [])] != list(range(5))
        or any(row.get("sample_count") != 8 for row in result["folds"])):
        raise RuntimeError("MX07_XAI_SCOPE_OR_BINDING_INVALID")
    receipts = recorded.get("fold_receipt_sha256") or {}
    if set(receipts) != {str(fold) for fold in range(5)}:
        raise RuntimeError("MX07_XAI_FOLD_RECEIPT_BINDING_MISSING")
    for fold in range(5):
        receipt = out / "reports/xai_limited_folds" / f"fold_{fold}.json"
        if not receipt.is_file() or sha(receipt) != receipts[str(fold)]:
            raise RuntimeError(f"MX07_XAI_FOLD_RECEIPT_CHANGED:{fold}")
    from scripts.mx_fasttrack_v2.xai_limited import atlas_binding
    _, current_atlas_provenance = atlas_binding(
        root, read(out / "reports/fasttrack_preregistration.json"))
    if current_atlas_provenance != result.get("provenance"):
        raise RuntimeError("MX07_XAI_ATLAS_OR_SPM_PROVENANCE_CHANGED")
    if result.get("status") == "COMPLETED_EXPLORATORY_LIMITED":
        if result.get("gates", {}).get("status") != "COMPLETED":
            raise RuntimeError("MX07_XAI_GATES_INVALID")
        return "COMPLETED_EXPLORATORY_LIMITED", result
    if result.get("status") == "BLOCKED_EXPLORATORY_GATE":
        return "BLOCKED_EXPLORATORY_GATE", result
    raise RuntimeError("MX07_XAI_UNKNOWN_RESULT_STATE")


def rows_for_paper(lock: dict, pair: dict, scratch: dict, head: dict, audit: dict) -> dict:
    delta = pair["paired_dense_minus_baseline"]
    return {
        "comparison": "exploratory internal historically inspected five fold tests",
        "selected_inner_task": lock["final_task"],
        "inner_mean_best_validation_loss": {tid: lock["tasks"][tid]["mean_best_val_loss"]
                                            for tid, _, _ in TASKS},
        "mx04_scratch_overall": scratch["subsets"]["overall"],
        "mx04_head_overall": head["subsets"]["overall"],
        "hybrid_baseline_overall": pair["arms"]["hybrid_baseline"],
        "hybrid_dense_overall": pair["arms"]["hybrid_dense"],
        "hybrid_paired_difference": delta,
        "temporal_gap_counts": audit["time_window_sensitivity_counts"],
    }


def report_documents(lock_sha: str, ledger_sha: str, scoped: dict) -> dict[str, str]:
    rows = scoped["paper_numbers"]
    evidence = (f"Internal lock SHA256 `{lock_sha}`; negative-results ledger SHA256 `{ledger_sha}`.\n"
                "Each metric below must be traced to the frozen manifest and is descriptive.\n")
    methods = ("# Method X: scoped fast-track Methods\n\n" + evidence + "\n"
               "Five patient-disjoint internal folds at seed 13; four prespecified tasks: "
               "ResNet34 scratch, ResNet34 supervised fold-local pretraining with head/last-stage tuning, "
               "AMUSE-X baseline hybrid and the same hybrid with dense skip. One common 20-epoch-per-fold "
               "cap and fold-local normalization. Inner loss selects among scratch and hybrids; pretrained "
               "ResNet34 inner validation is exposed to prior supervised labels and excluded from selection. "
               "One frozen evaluation on historical fold tests, paired patient-bootstrap deltas for the "
               "two hybrid arms; MX4 scores are descriptive without paired difference CI. "
               "Optional 8 inner validation examples per fold use normalized absolute input gradients and "
               "fold-level region bootstrap. Pseudo holdout requires verified untouched provenance. "
               "PET is a label source for Method X, never a model input.\n")
    numbers = json.dumps(rows, ensure_ascii=False, indent=2, sort_keys=True)
    results = ("# Method X: scoped fast-track Results\n\n" + evidence + "\n"
               "Only these verified aggregate values may populate the paper; report both arms, regardless "
               "of which scored higher. Pseudo holdout and XAI have separately scoped statuses.\n\n"
               "```json\n" + numbers + "\n```\n")
    discussion = ("# Method X: scoped fast-track Discussion\n\n" + evidence + "\n"
                  "A4 subjects are positive and LEARN subjects negative; cohort and label are perfectly "
                  "confounded, so within-cohort binary AUC cannot be estimated. Historical fold tests "
                  "were inspected before this fast track; no external validation or independent amyloid "
                  "MRI signal is established. Single seed and limited MX4/XAI do not complete the "
                  "original MX4–MX7 program. Saliency uses the same inner examples to choose and probe "
                  "the top atlas ROI, uses first atlas IDs as control ROIs, and yields no causal or "
                  "biomarker anatomy inference. Method Y (PET input) requires a separate evidence chain, "
                  "endpoint, split audit, and method-specific interpretation; these files certify no "
                  "Method Y results.\n")
    return {"methods": methods, "results": results, "discussion": discussion}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--project-root", type=Path, default=ROOT)
    ap.add_argument("--output-root", type=Path,
                    default=Path("results/final_results/method_x_fasttrack_v2_20260923"))
    ap.add_argument("--allow-partial", action="store_true")
    args = ap.parse_args()
    root = args.project_root.resolve()
    out = (root / args.output_root).resolve()
    if root != ROOT.resolve() or not out.is_relative_to(root):
        raise RuntimeError("MX07_PROJECT_OR_OUTPUT_ROOT_INVALID")
    prereg, audit = check_inputs(root, out)
    battery = root / "results/final_results/method_x_model_battery_v3_depth_and_diversity_20260729/battery_manifest.json"
    if sha(battery) != prereg["battery_manifest_sha256"]:
        raise RuntimeError("MX07_BATTERY_MANIFEST_CHANGED")
    lock_path = out / "reports/fasttrack_inner_lock.json"
    lock = check_lock(root, out) if lock_path.exists() else None
    if lock is None and not args.allow_partial:
        raise RuntimeError("MX07_INNER_LOCK_MISSING")
    lock_sha = sha(lock_path) if lock is not None else None
    selected = lock["final_task"] if lock is not None else None
    scratch = (evaluated(out, "evaluation_outer_mx04_scratch", lock_sha, "mx04_scratch")
               if lock is not None else None)
    head = (evaluated(out, "evaluation_outer_mx04_head_last", lock_sha, "mx04_head_last")
            if lock is not None else None)
    pair = (evaluated(out, "evaluation_outer_hybrid_pair", lock_sha, None, paired=True)
            if lock is not None else None)
    pseudo = (evaluated(out, "evaluation_pseudo", lock_sha, selected, pseudo=True)
              if lock is not None else None)
    if lock is not None:
        xai, xai_result = xai_status(root, out, lock_sha, selected,
                                     lock["tasks"][selected]["checkpoint_sha256"])
    else:
        xai, xai_result = "NOT_DONE_OPTIONAL", None
    selected_outer = (scratch if selected == "mx04_scratch" else
                      pair["arms"].get(selected) if pair and selected in pair["arms"] else None)
    ready = bool(lock is not None and scratch is not None and head is not None
                 and pair is not None and selected_outer is not None)
    if not ready and not args.allow_partial:
        raise RuntimeError("MX07_MANDATORY_EVALUATIONS_MISSING_USE_ALLOW_PARTIAL_FOR_DRAFT")
    stage = {
        "MX00_MX03_original": "COMPLETED_BY_PRIOR_WORK_NOT_REVERIFIED_BY_FASTTRACK",
        "MX04_original_full_grid": "NOT_DONE_BY_FASTTRACK",
        "MX04_focused_two_arm": "COMPLETED_EXPLORATORY" if scratch and head else "NOT_DONE",
        "MX05_original_full_battery_replication": "NOT_DONE_BY_FASTTRACK",
        "MX05_two_hybrid_arms": "COMPLETED_EXPLORATORY" if pair else "NOT_DONE",
        "MX06_original_formal": "NOT_DONE_BY_FASTTRACK",
        "MX06_limited_optional": xai,
        "MX07_original": "NOT_DONE_BY_FASTTRACK",
        "MX07_fasttrack": "READY_SCOPED_FASTTRACK" if ready else "PARTIAL_DRAFT",
        "selected_internal_outer": "COMPLETED_EXPLORATORY" if selected_outer is not None else "NOT_DONE",
        "pseudo_random_holdout": "COMPLETED_INTERNAL_IF_HISTORY_VERIFIED" if pseudo else "UNKNOWN_OR_NOT_DONE",
        "true_external_validation": "NOT_DONE",
        "method_y_pet_branch": "UNKNOWN_SEPARATE_EVIDENCE_REQUIRED",
    }
    assets = {"data_audit": out / "reports/data_semantics_audit.json",
              "machine_preflight": out / "reports/machine_preflight.json",
              "preregistration": out / "reports/fasttrack_preregistration.json",
              "battery_manifest": battery,
              "config": resolve_config(root, str(prereg.get("protocol_id")))}
    if lock is not None:
        assets["inner_lock"] = lock_path
        for tid, _, _ in TASKS:
            base = out / "inner_tasks" / tid
            assets[f"{tid}_metrics"] = base / "three_d_metrics.json"
            assets[f"{tid}_contract"] = base / "run_contract.json"
            assets[f"{tid}_staging_prune_plan"] = (
                out / "reports" / f"staging_prune_{tid}_plan.json")
            assets[f"{tid}_staging_prune_receipt"] = (
                out / "reports" / f"staging_prune_{tid}.json")
            for fold in range(5):
                assets[f"{tid}_fold_{fold}_checkpoint"] = base / f"method_x_3d_fold_{fold}.pt"
        for key, name in (("mx04_scratch", "evaluation_outer_mx04_scratch"),
                          ("mx04_head", "evaluation_outer_mx04_head_last"),
                          ("hybrid_pair", "evaluation_outer_hybrid_pair"),
                          ("pseudo", "evaluation_pseudo"),
                          ("xai_limited", "xai_limited_summary")):
            if (out / "reports" / f"{name}.json").exists():
                assets[key] = out / "reports" / f"{name}.json"
                assets[key + "_state"] = out / "reports" / f"{name}_state.json"
        if pseudo is not None:
            history = out / "reports/holdout_history_audit.json"
            assets["pseudo_history"] = history
            for number, evidence in enumerate(read(history)["evidence_files"]):
                assets[f"pseudo_history_evidence_{number}"] = root / evidence["relative_path"]
        if xai_result is not None:
            for fold in range(5):
                assets[f"xai_limited_fold_{fold}"] = (
                    out / "reports/xai_limited_folds" / f"fold_{fold}.json")
    relative = {key: {"relative_path": path.relative_to(root).as_posix(), "sha256": sha(path)}
                for key, path in assets.items()}
    if any(value["sha256"] is None for value in relative.values()):
        raise RuntimeError("MX07_REFERENCED_ASSET_MISSING")
    scoped = {
        "schema_version": 2, "scope": "POST_HOC_INTERNAL_TECHNICAL_EVIDENCE",
        "status": stage, "immutable": ready, "selected_inner_task": selected,
        "cohort_label_perfectly_confounding": True,
        "within_cohort_binary_auc": "NOT_MEASURABLE_ONE_CLASS_PER_COHORT",
        "external_validation": False, "amyloid_specific_or_clinical_claim_allowed": False,
        "method_x_pet_input": False, "method_y_results_attested": False,
        "selected_outer": selected_outer, "pseudo_aggregate": pseudo["subsets"] if pseudo else None,
        "feature_drop_label_correction": audit["feature_drop_label_correction"],
        "paper_numbers": rows_for_paper(lock, pair, scratch, head, audit) if ready else None,
        "asset_inventory": relative,
        "negative_results_policy": "report_all_four_inner_arms_and_both_hybrid_outer_arms_even_if_unfavorable",
        "unmeasured": ["independent_external_testing", "within_cohort_binary_auc",
                       "seed_29_47_training_replication", "original_full_MX04_MX05_MX06_MX07",
                       "Method_Y_separate_results_in_this_inventory"],
        "subject_level_vectors_exported": False,
    }
    destination = out / "reports/fasttrack_thesis_traceability.json"
    if not ready:
        # A draft is a separate, mutable-in-new-version artifact. It must
        # never occupy the immutable ready manifest path.
        write_once(out / "reports/fasttrack_thesis_draft.json", scoped)
        print(json.dumps({"status": "PARTIAL_DRAFT", "selected": selected,
                          "missing": [name for name, value in
                                      (("lock", lock), ("mx04_scratch", scratch),
                                       ("mx04_head", head), ("hybrid_pair", pair),
                                       ("selected_outer", selected_outer)) if value is None]}))
        return 0
    ledger = {
        "schema_version": 2, "status": "COMPLETE_EXACT_FOUR_TASK_INVENTORY",
        "source_lock_sha256": lock_sha,
        "tasks": {tid: {"mean_inner_validation_loss": lock["tasks"][tid]["mean_best_val_loss"],
                        "metrics_sha256": lock["tasks"][tid]["metrics_sha256"]}
                  for tid, _, _ in TASKS},
        "comparison_limitations": {
            "mx04_head_inner_selection": "INVALID_SUPERVISED_PRETRAINING_LABEL_EXPOSURE",
            "hybrid_paired_roc_auc_delta_dense_minus_baseline":
                pair["paired_dense_minus_baseline"]["roc_auc"],
            "cohort_label_overlap": "ABSENT_ONE_CLASS_PER_COHORT",
            "mx04_paired_difference_ci": "NOT_ESTIMATED",
        },
        "all_arms_reported_even_if_negative": True,
    }
    ledger_path = out / "reports/fasttrack_negative_results_ledger.json"
    write_once(ledger_path, ledger)
    scoped["negative_results_ledger_sha256"] = sha(ledger_path)
    docs = report_documents(lock_sha, sha(ledger_path), scoped)
    for key, document in docs.items():
        path = out / "reports" / f"paper_method_x_{key}.md"
        text_once(path, document)
        scoped["asset_inventory"][f"paper_{key}"] = {
            "relative_path": path.relative_to(root).as_posix(), "sha256": sha(path)}
    scoped["asset_inventory"]["negative_results_ledger"] = {
        "relative_path": ledger_path.relative_to(root).as_posix(), "sha256": sha(ledger_path)}
    write_once(destination, scoped)
    print(json.dumps({"status": stage["MX07_fasttrack"], "selected": selected,
                      "manifest_sha256": sha(destination), "original_MX07": stage["MX07_original"]}))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, OSError, ValueError, KeyError, TypeError) as exc:
        print(json.dumps({"status": "BLOCKED", "reason": str(exc)}), file=sys.stderr)
        raise SystemExit(3)
