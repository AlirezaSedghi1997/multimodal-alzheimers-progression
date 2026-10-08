"""One logical, prespecified paired outer analysis of the two hybrid arms.

No identifiers are written. Identifier-free fold vectors exist only as private
resume receipts and are removed after the aggregate commit. This remains an
exploratory comparison on historically inspected internal tests.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.mx_fasttrack_v2.evaluate_locked import (
    atomic, begin_resumable_evaluation, commit_receipt, csv_rows, hash_file,
    load_or_adopt_receipt, read, remove_transient_receipts, scores,
)
from scripts.mx_fasttrack_v2.integrity import check_lock


def paired_differences(y: list[int], base: list[float], dense: list[float], seed: int = 13) -> dict:
    """Resample the SAME patient rows for both models (1000 draws)."""
    import numpy as np
    from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

    if len(y) != len(base) or len(y) != len(dense) or len(set(y)) != 2:
        raise RuntimeError("PAIRED_INPUTS_MISMATCH_OR_ONE_CLASS")
    yy = np.asarray(y); bb = np.asarray(base); dd = np.asarray(dense)
    metrics = {"roc_auc": roc_auc_score, "average_precision": average_precision_score,
               "brier": brier_score_loss}
    effect = {name: float(fn(yy, dd) - fn(yy, bb)) for name, fn in metrics.items()}
    rng = np.random.default_rng(seed)
    draws = {name: [] for name in metrics}
    for _ in range(1000):
        indices = rng.integers(0, len(y), size=len(y))
        if len(set(yy[indices])) != 2:
            continue
        for name, fn in metrics.items():
            draws[name].append(float(fn(yy[indices], dd[indices]) - fn(yy[indices], bb[indices])))
    if min(len(v) for v in draws.values()) < 900:
        raise RuntimeError("PAIRED_BOOTSTRAP_INSUFFICIENT_TWO_CLASS_DRAWS")
    return {name: {"dense_minus_baseline": effect[name],
                   "patient_bootstrap_ci95": [float(v) for v in np.quantile(draws[name], [.025, .975])],
                   "valid_draws": len(draws[name])} for name in metrics}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--project-root", type=Path, default=ROOT)
    ap.add_argument("--output-root", type=Path,
                    default=Path("results/final_results/method_x_fasttrack_v2_20260923"))
    args = ap.parse_args()
    root = args.project_root.resolve()
    if root != ROOT.resolve():
        raise RuntimeError("PROJECT_ROOT_MISMATCH")
    out = (root / args.output_root).resolve()
    lock = check_lock(root, out)
    if lock.get("hybrid_pair") != ["hybrid_baseline", "hybrid_dense"]:
        raise RuntimeError("PAIRED_ANALYSIS_NOT_PREDECLARED")
    lock_path = out / "reports/fasttrack_inner_lock.json"
    result = out / "reports/evaluation_outer_hybrid_pair.json"
    state_path = out / "reports/evaluation_outer_hybrid_pair_state.json"
    current_lock_sha = hash_file(lock_path)
    expected_folds = list(range(5))
    receipt_root = out / "_private_resume/evaluation_outer_hybrid_pair"
    if state_path.exists():
        state = read(state_path)
        if (state.get("status") == "COMPLETED" and result.is_file()
            and state.get("result_sha256") == hash_file(result)
            and state.get("lock_sha256") == current_lock_sha):
            remove_transient_receipts(receipt_root, expected_folds)
            if state.get("transient_receipts_cleanup_pending") is not False:
                state["transient_receipts_cleanup_pending"] = False
                atomic(state_path, state)
            print(json.dumps({"status": "REUSED_EXISTING_PAIRED_EVALUATION"}))
            return 0
    state = begin_resumable_evaluation(
        state_path, result, lock_sha256=current_lock_sha,
        scope="outer_hybrid_pair", task="hybrid_pair", expected_folds=expected_folds,
    )
    import torch
    from torch.utils.data import DataLoader
    from src.evaluation.method_x_xai_forward import build_model_from_checkpoint
    from scripts.train_3d_mri import H5MethodXDataset, cache_index, ok_indices, predict

    cache = root / "data/cached/t1_volume_cache_128.h5"
    sample_to_index, ok = cache_index(cache), ok_indices(cache)
    all_y: list[int] = []
    predictions: dict[str, list[float]] = {"hybrid_baseline": [], "hybrid_dense": []}
    per_fold = []
    seen: set[str] = set()
    for fold in expected_folds:
        split_path = root / "splits" / f"fold_{fold}_test.csv"
        checkpoint_hashes = [lock["tasks"][task]["checkpoint_sha256"][fold]
                             for task in predictions]
        checkpoint_binding = hashlib.sha256(
            json.dumps(checkpoint_hashes, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        receipt_path = receipt_root / f"fold_{fold}.json"
        receipt = load_or_adopt_receipt(
            receipt_path, state, state_path, fold=fold,
            lock_sha256=current_lock_sha, scope="outer_hybrid_pair", task="hybrid_pair",
            checkpoint_sha256=checkpoint_binding, split_sha256=hash_file(split_path),
        )
        if receipt is None:
            from scripts.mx_fasttrack_v2.run import gpu_idle

            gpu_idle()
            eligible = [r for r in csv_rows(split_path)
                        if r["sample_hash"] in sample_to_index and sample_to_index[r["sample_hash"]] in ok]
            indices = [sample_to_index[r["sample_hash"]] for r in eligible]
            labels = [int(r["ab_label"]) for r in eligible]
            if (not indices or len({r["patient_hash"] for r in eligible}) != len(eligible)
                or any(r["patient_hash"] in seen for r in eligible)):
                raise RuntimeError("PAIRED_FOLD_EMPTY_OR_PATIENT_OVERLAP")
            seen.update(r["patient_hash"] for r in eligible)
            fold_predictions = {}
        else:
            labels = [int(value) for value in receipt.get("labels", [])]
            fold_predictions = {
                "hybrid_baseline": [float(value) for value in receipt.get("baseline_probabilities", [])],
                "hybrid_dense": [float(value) for value in receipt.get("dense_probabilities", [])],
            }
        for task in predictions if receipt is None else ():
            info = lock["tasks"][task]
            ckpt_path = out / "inner_tasks" / task / f"method_x_3d_fold_{fold}.pt"
            if hash_file(ckpt_path) != info["checkpoint_sha256"][fold]:
                raise RuntimeError("PAIRED_CHECKPOINT_CHANGED")
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
            variant = "baseline" if task == "hybrid_baseline" else "plus_dense_skip"
            if (ckpt.get("fold") != fold or ckpt.get("hpo_selection_only") is not True
                or ckpt.get("backbone") != "amuse_x_hybrid3d"
                or ckpt.get("hybrid_variant") != variant
                or ckpt.get("run_contract_digest") != info["run_contract_digest"]
                or ckpt.get("split_membership_digest") != info["fold_membership_digest"][fold]
                or ckpt.get("outer_test_access_count") != 0
                or ckpt.get("outer_test_evaluation_count") != 0
                or ckpt.get("drop_tabular_feature") != "none"
                or ckpt.get("input_mode") != "mri_plus_tabular"):
                raise RuntimeError("PAIRED_CHECKPOINT_IDENTITY_INVALID")
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model, state_source = build_model_from_checkpoint(ckpt, device=device)
            if state_source != "model_state_dict":
                raise RuntimeError("PAIRED_CHECKPOINT_STATE_SOURCE_UNEXPECTED")
            dataset = H5MethodXDataset(cache, indices, labels)
            loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
            observed, probability = predict(model, loader, device, ckpt["tab_mean"], ckpt["tab_std"])
            if observed != labels or not all(math.isfinite(p) and 0 <= p <= 1 for p in probability):
                raise RuntimeError("PAIRED_PREDICTION_INVALID")
            fold_predictions[task] = probability
            del dataset, loader, model, ckpt
            if device.type == "cuda":
                torch.cuda.empty_cache()
        if receipt is None:
            receipt = {
                "schema_version": 1, "fold": fold, "lock_sha256": current_lock_sha,
                "scope": "outer_hybrid_pair", "task": "hybrid_pair",
                "checkpoint_sha256": checkpoint_binding,
                "split_sha256": hash_file(split_path), "contains_identifiers": False,
                "unique_patient_count": len(eligible),
                "labels": labels,
                "baseline_probabilities": fold_predictions["hybrid_baseline"],
                "dense_probabilities": fold_predictions["hybrid_dense"],
            }
            commit_receipt(receipt_path, state, state_path, fold=fold, receipt=receipt)
        if (not labels or not set(labels) <= {0, 1}
            or int(receipt.get("unique_patient_count", -1)) != len(labels)
            or len(labels) != len(fold_predictions["hybrid_baseline"])
            or len(labels) != len(fold_predictions["hybrid_dense"])
            or not all(math.isfinite(value) and 0 <= value <= 1
                       for values in fold_predictions.values() for value in values)):
            raise RuntimeError(f"PAIRED_RESUME_RECEIPT_VECTOR_INVALID:{fold}")
        for task in predictions:
            predictions[task].extend(fold_predictions[task])
        per_fold.append({"fold": fold, "n": len(labels),
                         "baseline": scores(labels, fold_predictions["hybrid_baseline"]),
                         "dense": scores(labels, fold_predictions["hybrid_dense"])})
        all_y.extend(labels)
    if (len(all_y) != 1336 or sum(row["n"] for row in per_fold) != 1336
        or [row["fold"] for row in per_fold] != list(range(5))):
        raise RuntimeError("PAIRED_QC_COUNT_CHANGED")
    payload = {"schema_version": 2, "status": "COMPLETED_EXPLORATORY_PAIRED",
               "scope": "historically_inspected_internal_outer_folds",
               "lock_sha256": hash_file(lock_path), "folds_evaluated": list(range(5)),
               "n": len(all_y), "by_fold": per_fold,
               "arms": {task: scores(all_y, probs) for task, probs in predictions.items()},
               "paired_dense_minus_baseline": paired_differences(
                   all_y, predictions["hybrid_baseline"], predictions["hybrid_dense"]),
               "external_validation": False,
               "paired_effect_causal_or_confirmatory": False,
               "test_used_for_model_selection": False,
               "subject_level_predictions_written": True,
               "subject_level_predictions_exported": False,
               "transient_identifier_free_resume_vectors": "REMOVED_AFTER_AGGREGATE_COMMIT"}
    atomic(result, payload)
    state.update({"status": "COMPLETED", "lock_sha256": current_lock_sha,
                  "scope": "outer_hybrid_pair", "task": "hybrid_pair",
                  "result_sha256": hash_file(result), "logical_evaluation_count": 1,
                  "transient_receipts_cleanup_pending": True})
    atomic(state_path, state)
    remove_transient_receipts(receipt_root, expected_folds)
    state["transient_receipts_cleanup_pending"] = False
    atomic(state_path, state)
    print(json.dumps({"status": payload["status"], "n": len(all_y)}))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError, KeyError, TypeError) as exc:
        print(json.dumps({"status": "BLOCKED", "reason": str(exc)}), file=sys.stderr)
        raise SystemExit(3)
