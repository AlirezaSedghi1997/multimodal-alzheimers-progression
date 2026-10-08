"""One logical aggregate-only evaluation of a frozen inner-selected checkpoint.

`outer` evaluates either the selected model or the two predeclared MX4 arms on
their matching fold TEST splits. MX4's paired comparison is exploratory:
supervised source weights had seen the future inner-validation labels.
`pseudo` evaluates only the predeclared selected fold-0 checkpoint, provided
that a separate provenance audit attests it was untouched.
For interruption recovery, identifier-free fold vectors are written only under
the ignored private-resume directory and removed after the aggregate result is
committed. They are never exported or used for tuning.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic(path: Path, content: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".new")
    temp.write_text(json.dumps(content, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    os.replace(temp, path)


def completed_evaluation_reusable(
    state: dict,
    result: dict,
    *,
    result_sha256: str,
    lock_sha256: str,
    scope: str,
    task: str,
) -> bool:
    """Require the completed result to remain bound to the current lock."""

    return bool(
        state.get("status") == "COMPLETED"
        and state.get("result_sha256") == result_sha256
        and state.get("lock_sha256") == lock_sha256
        and result.get("lock_sha256") == lock_sha256
        and state.get("scope") == scope
        and state.get("task") == task
        and result.get("model_task") == task
    )


def begin_resumable_evaluation(
    state_path: Path,
    result_path: Path,
    *,
    lock_sha256: str,
    scope: str,
    task: str,
    expected_folds: list[int],
) -> dict:
    """Start or resume the same prespecified logical evaluation."""

    if state_path.is_file():
        state = read(state_path)
        if (state.get("status") not in {"STARTED", "RUNNING"}
            or state.get("lock_sha256") != lock_sha256
            or state.get("scope") != scope
            or state.get("task") != task
            or state.get("expected_folds") not in (None, expected_folds)):
            raise RuntimeError("EVALUATION_RESUME_STATE_OR_PROTOCOL_DRIFT")
        attempts = int(state.get("execution_attempts", 1)) + 1
        completed = state.get("completed_folds") or {}
        first_started = state.get("first_started_at_utc")
    else:
        attempts, completed, first_started = 1, {}, None
    if not isinstance(completed, dict):
        raise RuntimeError("EVALUATION_COMPLETED_FOLD_STATE_INVALID")
    from datetime import datetime, timezone

    now = datetime.now(timezone.utc).isoformat()
    state = {
        "schema_version": 2,
        "status": "RUNNING",
        "scope": scope,
        "task": task,
        "lock_sha256": lock_sha256,
        "expected_folds": expected_folds,
        "completed_folds": completed,
        "logical_evaluation_count": 1,
        "execution_attempts": attempts,
        "first_started_at_utc": first_started or now,
        "last_resumed_at_utc": now,
        "result_present_during_resume": result_path.is_file(),
    }
    atomic(state_path, state)
    return state


def load_or_adopt_receipt(
    receipt_path: Path,
    state: dict,
    state_path: Path,
    *,
    fold: int,
    lock_sha256: str,
    scope: str,
    task: str,
    checkpoint_sha256: str,
    split_sha256: str,
) -> dict | None:
    """Validate a committed fold receipt and bind it to state after a torn write."""

    if not receipt_path.is_file():
        if str(fold) in state.get("completed_folds", {}):
            raise RuntimeError(f"EVALUATION_RECEIPT_MISSING_AFTER_COMMIT:{fold}")
        return None
    receipt = read(receipt_path)
    digest = hash_file(receipt_path)
    expected = state.get("completed_folds", {}).get(str(fold))
    if expected not in (None, digest):
        raise RuntimeError(f"EVALUATION_RECEIPT_HASH_CHANGED:{fold}")
    if (receipt.get("schema_version") != 1 or receipt.get("fold") != fold
        or receipt.get("lock_sha256") != lock_sha256
        or receipt.get("scope") != scope or receipt.get("task") != task
        or receipt.get("checkpoint_sha256") != checkpoint_sha256
        or receipt.get("split_sha256") != split_sha256
        or receipt.get("contains_identifiers") is not False):
        raise RuntimeError(f"EVALUATION_RECEIPT_BINDING_INVALID:{fold}")
    state.setdefault("completed_folds", {})[str(fold)] = digest
    atomic(state_path, state)
    return receipt


def commit_receipt(receipt_path: Path, state: dict, state_path: Path,
                   *, fold: int, receipt: dict) -> None:
    atomic(receipt_path, receipt)
    state.setdefault("completed_folds", {})[str(fold)] = hash_file(receipt_path)
    atomic(state_path, state)


def remove_transient_receipts(receipt_root: Path, expected_folds: list[int]) -> None:
    """Remove only private fold vectors created by this evaluation."""

    for fold in expected_folds:
        path = receipt_root / f"fold_{fold}.json"
        if path.is_file():
            path.unlink()
    if receipt_root.is_dir() and not any(receipt_root.iterdir()):
        receipt_root.rmdir()


def csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def verify_holdout_history(root: Path, path: Path) -> dict:
    history = read(path)
    entries = history.get("evidence_files")
    if (history.get("status") != "VERIFIED_UNTOUCHED"
        or history.get("reviewed_execution_logs") is not True
        or history.get("model_or_threshold_selected_with_holdout") is not False
        or history.get("prior_inference_count") != 0
        or history.get("scope") != "METHOD_X_FASTTRACK_V2_PSEUDO_HOLDOUT"
        or not isinstance(history.get("reviewer_note"), str) or not history["reviewer_note"].strip()
        or not isinstance(entries, list) or not entries):
        raise RuntimeError("PSEUDO_HOLDOUT_HISTORY_UNVERIFIED")
    for evidence in entries:
        relative = Path(str(evidence.get("relative_path", ""))) if isinstance(evidence, dict) else Path("..")
        expected = evidence.get("sha256") if isinstance(evidence, dict) else None
        if (relative.is_absolute() or ".." in relative.parts or not relative.parts
            or not isinstance(expected, str) or len(expected) != 64
            or hash_file(root / relative) != expected):
            raise RuntimeError("PSEUDO_HOLDOUT_HISTORY_EVIDENCE_MISSING_OR_CHANGED")
    return history


def scores(y: list[int], p: list[float], *, seed: int = 13) -> dict:
    import numpy as np
    from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

    yy, pp = np.asarray(y), np.asarray(p)
    positive = int(yy.sum())
    negative = len(y) - positive
    result = {"n": len(y), "positive": positive, "negative": negative,
              "brier": float(brier_score_loss(yy, pp)) if len(y) else None,
              "mean_predicted_probability": float(pp.mean()) if len(y) else None}
    if min(positive, negative) == 0:
        result.update({"roc_auc": None, "average_precision": None,
                       "roc_auc_ci95": None, "status": "NOT_MEASURABLE_ONE_CLASS"})
        return result
    result["roc_auc"] = float(roc_auc_score(yy, pp))
    result["average_precision"] = float(average_precision_score(yy, pp))
    result["status"] = "ESTIMATED" if min(positive, negative) >= 10 else "SPARSE_CLASS_EXPLORATORY"
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(1000):
        sample = rng.integers(0, len(yy), size=len(yy))
        if len(np.unique(yy[sample])) == 2:
            draws.append(float(roc_auc_score(yy[sample], pp[sample])))
    result["roc_auc_ci95"] = (
        [float(x) for x in np.quantile(draws, [0.025, 0.975])]
        if len(draws) >= 900 else None
    )
    result["bootstrap_valid_draws"] = len(draws)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--output-root", type=Path, default=Path("results/final_results/method_x_fasttrack_v2_20260923"))
    parser.add_argument("--scope", choices=("outer", "pseudo"), required=True)
    parser.add_argument("--task", choices=("selected", "mx04_scratch", "mx04_head_last"),
                        default="selected")
    parser.add_argument("--holdout-history-audit", type=Path)
    args = parser.parse_args()
    root = args.project_root.resolve()
    if root != ROOT.resolve():
        raise RuntimeError("PROJECT_ROOT_MISMATCH")
    out = (root / args.output_root).resolve()
    locked_path = out / "reports/fasttrack_inner_lock.json"
    from scripts.mx_fasttrack_v2.integrity import check_lock
    lock = check_lock(root, out)
    if args.scope == "pseudo" and args.task != "selected":
        raise RuntimeError("PSEUDO_HOLDOUT_SELECTED_MODEL_ONLY")
    task = lock["final_task"] if args.task == "selected" else args.task
    if args.task != "selected" and task not in lock.get("exploratory_outer_tasks", []):
        raise RuntimeError("EXPLORATORY_TASK_NOT_PREDECLARED_IN_LOCK")
    folder = out / "inner_tasks" / task
    selected = lock["tasks"][task]
    if hash_file(folder / "three_d_metrics.json") != selected["metrics_sha256"]:
        raise RuntimeError("SELECTED_METRICS_CHANGED")
    if args.scope == "pseudo":
        if args.holdout_history_audit is None:
            raise RuntimeError("PSEUDO_HOLDOUT_HISTORY_UNVERIFIED")
        if args.holdout_history_audit.resolve() != (out / "reports/holdout_history_audit.json").resolve():
            raise RuntimeError("PSEUDO_HOLDOUT_AUDIT_MUST_BE_VERSIONED_WITH_RESULT")
        verify_holdout_history(root, args.holdout_history_audit)
    if args.scope == "outer" and task.startswith("hybrid_"):
        paired_path = out / "reports/evaluation_outer_hybrid_pair.json"
        paired_state = read(out / "reports/evaluation_outer_hybrid_pair_state.json")
        paired = read(paired_path)
        if (paired.get("status") != "COMPLETED_EXPLORATORY_PAIRED"
            or paired.get("lock_sha256") != hash_file(locked_path)
            or paired.get("folds_evaluated") != list(range(5))
            or paired_state.get("status") != "COMPLETED"
            or paired_state.get("result_sha256") != hash_file(paired_path)
            or (paired.get("arms") or {}).get(task, {}).get("n") != 1336):
            raise RuntimeError("HYBRID_PAIRED_OUTER_EVALUATION_INVALID")
        print(json.dumps({"status": "REUSED_PREDECLARED_HYBRID_PAIR", "task": task,
                          "path": str(paired_path)}))
        return 0
    basename = f"evaluation_outer_{task}" if args.scope == "outer" else "evaluation_pseudo"
    result_path = out / "reports" / f"{basename}.json"
    state_path = out / "reports" / f"{basename}_state.json"
    expected_folds = list(range(5)) if args.scope == "outer" else [0]
    receipt_root = out / "_private_resume" / basename
    current_lock_sha = hash_file(locked_path)
    if state_path.exists():
        state = read(state_path)
        existing = read(result_path) if result_path.is_file() else {}
        if (result_path.is_file() and completed_evaluation_reusable(
            state, existing, result_sha256=hash_file(result_path),
            lock_sha256=current_lock_sha, scope=args.scope, task=task,
        )):
            remove_transient_receipts(receipt_root, expected_folds)
            if state.get("transient_receipts_cleanup_pending") is not False:
                state["transient_receipts_cleanup_pending"] = False
                atomic(state_path, state)
            print(json.dumps({"status": "REUSED_EXISTING_EVALUATION", "path": str(result_path)}))
            return 0
    state = begin_resumable_evaluation(
        state_path, result_path, lock_sha256=current_lock_sha,
        scope=args.scope, task=task, expected_folds=expected_folds,
    )
    import torch
    from torch.utils.data import DataLoader
    from src.evaluation.method_x_xai_forward import build_model_from_checkpoint
    from scripts.train_3d_mri import H5MethodXDataset, cache_index, ok_indices, predict

    cache = root / "data/cached/t1_volume_cache_128.h5"
    sample_to_index = cache_index(cache)
    ok = ok_indices(cache)
    y: list[int] = []
    p: list[float] = []
    gap: list[float] = []
    direction: list[str] = []
    groups: list[str] = []
    folds = expected_folds
    for fold in folds:
        checkpoint_file = folder / f"method_x_3d_fold_{fold}.pt"
        if hash_file(checkpoint_file) != selected["checkpoint_sha256"][fold]:
            raise RuntimeError(f"CHECKPOINT_CHANGED:{fold}")
        ckpt = torch.load(checkpoint_file, map_location="cpu", weights_only=True)
        if ckpt.get("fold") != fold or ckpt.get("hpo_selection_only") is not True:
            raise RuntimeError("CHECKPOINT_NOT_INNER_ONLY")
        if (ckpt.get("run_contract_digest") != selected["run_contract_digest"]
            or ckpt.get("split_membership_digest") != selected["fold_membership_digest"][fold]):
            raise RuntimeError("CHECKPOINT_RUN_OR_SPLIT_BINDING_CHANGED")
        if ckpt.get("outer_test_access_count") != 0 or ckpt.get("outer_test_evaluation_count") != 0:
            raise RuntimeError("CHECKPOINT_TEST_PREVIOUSLY_ACCESSED")
        if ckpt.get("input_mode") != "mri_plus_tabular" or ckpt.get("drop_tabular_feature") != "none":
            raise RuntimeError("CHECKPOINT_INPUT_CONTRACT_FAILED")
        variant = ckpt.get("hybrid_variant", "baseline")
        expected_variant = task.removeprefix("hybrid_")
        if task == "hybrid_dense":
            expected_variant = "plus_dense_skip"
        if task.startswith("hybrid_") and variant != expected_variant:
            raise RuntimeError("HYBRID_CHECKPOINT_VARIANT_MISMATCH")
        path = root / "splits" / (f"fold_{fold}_test.csv" if args.scope == "outer" else "external_holdout.csv")
        receipt_path = receipt_root / f"fold_{fold}.json"
        receipt = load_or_adopt_receipt(
            receipt_path, state, state_path, fold=fold,
            lock_sha256=current_lock_sha, scope=args.scope, task=task,
            checkpoint_sha256=selected["checkpoint_sha256"][fold],
            split_sha256=hash_file(path),
        )
        if receipt is None:
            from scripts.mx_fasttrack_v2.run import gpu_idle

            gpu_idle()
            eligible = [row for row in csv_rows(path) if row["sample_hash"] in sample_to_index
                        and sample_to_index[row["sample_hash"]] in ok]
            if not eligible or len({row["patient_hash"] for row in eligible}) != len(eligible):
                raise RuntimeError("EVALUATION_COHORT_EMPTY_OR_DUPLICATED")
            indices = [sample_to_index[row["sample_hash"]] for row in eligible]
            labels = [int(row["ab_label"]) for row in eligible]
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model, state_source = build_model_from_checkpoint(ckpt, device=device)
            if state_source != "model_state_dict":
                raise RuntimeError("OUTER_CHECKPOINT_STATE_SOURCE_UNEXPECTED")
            loader = DataLoader(H5MethodXDataset(cache, indices, labels,
                input_mode="mri_plus_tabular", drop_tabular_feature="none"),
                batch_size=1, shuffle=False, num_workers=0)
            observed, probs = predict(model, loader, device, ckpt["tab_mean"], ckpt["tab_std"])
            if observed != labels or not all(math.isfinite(v) and 0 <= v <= 1 for v in probs):
                raise RuntimeError("PREDICTIONS_OR_LABELS_INVALID")
            fold_gap = [float(row["abs_pet_t1_gap_days"]) for row in eligible]
            fold_direction = [
                "PET_AFTER_MRI" if float(row["pet_day_t0"]) > float(row["t1_day_t0"])
                else "PET_BEFORE_MRI" if float(row["pet_day_t0"]) < float(row["t1_day_t0"])
                else "SAME_DAY" for row in eligible
            ]
            fold_groups = [row["substudy"] for row in eligible]
            receipt = {
                "schema_version": 1, "fold": fold, "lock_sha256": current_lock_sha,
                "scope": args.scope, "task": task,
                "checkpoint_sha256": selected["checkpoint_sha256"][fold],
                "split_sha256": hash_file(path), "contains_identifiers": False,
                "labels": labels, "probabilities": probs, "gap_days": fold_gap,
                "directions": fold_direction, "groups": fold_groups,
            }
            commit_receipt(receipt_path, state, state_path, fold=fold, receipt=receipt)
            del model, loader
            if device.type == "cuda":
                torch.cuda.empty_cache()
        labels = [int(value) for value in receipt.get("labels", [])]
        probs = [float(value) for value in receipt.get("probabilities", [])]
        fold_gap = [float(value) for value in receipt.get("gap_days", [])]
        fold_direction = [str(value) for value in receipt.get("directions", [])]
        fold_groups = [str(value) for value in receipt.get("groups", [])]
        lengths = {len(labels), len(probs), len(fold_gap), len(fold_direction), len(fold_groups)}
        if (lengths != {len(labels)} or not labels or not set(labels) <= {0, 1}
            or not all(math.isfinite(value) and 0 <= value <= 1 for value in probs)
            or not all(math.isfinite(value) and value >= 0 for value in fold_gap)
            or not set(fold_direction) <= {"PET_AFTER_MRI", "PET_BEFORE_MRI", "SAME_DAY"}):
            raise RuntimeError(f"EVALUATION_RECEIPT_VECTOR_INVALID:{fold}")
        y.extend(labels); p.extend(probs); gap.extend(fold_gap)
        direction.extend(fold_direction); groups.extend(fold_groups)
        del ckpt
    expected_n = 1336 if args.scope == "outer" else 333
    if len(y) != expected_n:
        raise RuntimeError(f"EVALUATION_QC_COUNT_CHANGED_EXPECTED_{expected_n}_FOUND_{len(y)}")
    subsets = {"overall": scores(y, p),
               "gap_le_90_days": scores([a for a, g in zip(y, gap) if g <= 90],
                                        [b for b, g in zip(p, gap) if g <= 90]),
               "gap_le_180_days": scores([a for a, g in zip(y, gap) if g <= 180],
                                          [b for b, g in zip(p, gap) if g <= 180]),
               "gap_le_365_days": scores([a for a, g in zip(y, gap) if g <= 365],
                                         [b for b, g in zip(p, gap) if g <= 365])}
    for group in sorted(set(groups)):
        yy = [a for a, g in zip(y, groups) if g == group]
        subsets[f"cohort_{group}"] = {"n": len(yy), "positive": sum(yy),
                                       "negative": len(yy) - sum(yy),
                                       "auc": None, "status": "NOT_MEASURABLE_ONE_CLASS"}
    result = {"schema_version": 1, "status": "COMPLETED",
              "scope": "internal_fold_outer" if args.scope == "outer" else "random_pseudo_holdout",
              "model_task": task, "lock_sha256": hash_file(locked_path),
              "comparison_role": ("exploratory_paired_mx04_supervised_pretraining_inner_validation_exposure"
                                  if args.scope == "outer" and task.startswith("mx04_")
                                  else "selected_model"),
              "fine_tuning_inner_validation_valid_for_selection": False,
              "folds_evaluated": list(folds), "canonical_fold_for_pseudo": 0 if args.scope == "pseudo" else None,
              "subsets": subsets, "pet_mri_direction_counts": dict(Counter(direction)),
              "external_validation": False, "causal_amyloid_claim_allowed": False,
              "subject_level_vectors_exported": False,
              "transient_identifier_free_resume_vectors": "REMOVED_AFTER_AGGREGATE_COMMIT",
              "site_scanner_sensitivity": "UNKNOWN_NO_SITE_SCANNER_FIELD_IN_PROVIDED_SPLITS"}
    if args.scope == "pseudo":
        result["holdout_history_audit_sha256"] = hash_file(args.holdout_history_audit)
    atomic(result_path, result)
    state.update({"status": "COMPLETED", "scope": args.scope, "task": task,
                  "lock_sha256": current_lock_sha,
                  "result_sha256": hash_file(result_path),
                  "logical_evaluation_count": 1,
                  "transient_receipts_cleanup_pending": True})
    atomic(state_path, state)
    remove_transient_receipts(receipt_root, expected_folds)
    state["transient_receipts_cleanup_pending"] = False
    atomic(state_path, state)
    print(json.dumps({"status": "COMPLETED", "scope": args.scope,
                      "n": len(y), "roc_auc": subsets["overall"]["roc_auc"],
                      "subject_level_vectors_exported": False}))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError, KeyError, TypeError) as exc:
        print(json.dumps({"status": "BLOCKED", "reason": str(exc)}, ensure_ascii=False), file=sys.stderr)
        raise SystemExit(3)
