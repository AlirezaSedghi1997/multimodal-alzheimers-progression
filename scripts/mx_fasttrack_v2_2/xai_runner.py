"""Resumable five-fold v2.2 internal XAI, bound to a post-lock protocol."""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.mx_fasttrack_v2.integrity import check_lock
from scripts.mx_fasttrack_v2.run import read, sha
from scripts.mx_fasttrack_v2.xai_limited import atlas_binding, inner_validation
from scripts.mx_fasttrack_v2_1.supervise_v21 import immutable_json, require_internal_coordinator_parent
from scripts.mx_fasttrack_v2_2.xai_crossfit import aggregate_five_folds, run_fold_crossfit

CONTEXT = "xai_v22_context.json"
STATE = "xai_v22_state.json"
SUMMARY = "xai_v22_summary.json"
RECEIPTS = "xai_v22_folds"
PRIVATE_TARGETS = "xai_v22_targets"


def _atomic_state(path: Path, value: dict) -> None:
    """Unique temporary names permit retry after an interrupted write."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.{uuid.uuid4().hex}.new")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def expected_context(root: Path, out: Path, protocol: dict) -> dict:
    lock = check_lock(root, out)
    prereg = read(out / "reports/fasttrack_preregistration.json")
    _, provenance = atlas_binding(root, prereg)
    selected = lock["final_task"]
    return {"schema_version": 1, "status": "LOCKED_BEFORE_XAI_V22",
            "protocol_sha256": sha(out / "reports/fasttrack_v22_protocol.json"),
            "inner_lock_sha256": sha(out / "reports/fasttrack_inner_lock.json"),
            "selected_task": selected,
            "checkpoint_sha256": lock["tasks"][selected]["checkpoint_sha256"],
            "fold_membership_digest": lock["tasks"][selected]["fold_membership_digest"],
            "cache_sha256": prereg["cache_sha256"],
            "atlas_spm_provenance": provenance,
            "xai_config_sha256": protocol["source_sha256"]["configs/experiments/method_x_xai_v22.json"]}


def ensure_context(root: Path, out: Path, protocol: dict) -> dict:
    expected = expected_context(root, out, protocol)
    path = out / "reports" / CONTEXT
    if path.exists() and read(path) != expected:
        raise RuntimeError("V22_XAI_CONTEXT_OR_SOURCE_DRIFT")
    immutable_json(path, expected)
    return expected


def _validated_receipts(out: Path, context: dict) -> tuple[list[dict], dict[str, str]]:
    receipt_root = out / "reports" / RECEIPTS
    if receipt_root.exists() and not receipt_root.is_dir():
        raise RuntimeError("V22_RECEIPT_ROOT_NOT_DIRECTORY")
    if receipt_root.is_dir() and any(path.name not in {f"fold_{i}.json" for i in range(5)}
                                     for path in receipt_root.iterdir()):
        raise RuntimeError("V22_UNEXPECTED_FOLD_RECEIPT")
    rows, hashes = [], {}
    for fold in range(5):
        path = receipt_root / f"fold_{fold}.json"
        if not path.exists():
            continue
        row = read(path)
        target_path = out / "_private_resume" / PRIVATE_TARGETS / f"fold_{fold}.json"
        if (not target_path.is_file()
            or row.get("target_receipt_sha256") != sha(target_path)):
            raise RuntimeError(f"V22_TARGET_RECEIPT_MISSING_OR_CHANGED:{fold}")
        target_receipt = read(target_path)
        target_samples = target_receipt.get("samples")
        if (target_receipt.get("schema_version") != 1
            or target_receipt.get("fold") != fold
            or target_receipt.get("context_sha256") != sha(out / "reports" / CONTEXT)
            or target_receipt.get("checkpoint_sha256") != context["checkpoint_sha256"][fold]
            or target_receipt.get("split_membership_digest") != context["fold_membership_digest"][fold]
            or target_receipt.get("target_rule") != "unoccluded_predicted_class"
            or not isinstance(target_samples, list) or len(target_samples) != 8
            or any(not isinstance(item, dict) or item.get("ordinal") != ordinal
                   or type(item.get("original_probability")) not in (float, int)
                   or not math.isfinite(item["original_probability"])
                   or not 0 <= item["original_probability"] <= 1
                   or item.get("target_class") != int(item["original_probability"] >= 0.5)
                   for ordinal, item in enumerate(target_samples))):
            raise RuntimeError(f"V22_TARGET_RECEIPT_INVALID:{fold}")
        expected_target_counts = {
            "negative": sum(item["target_class"] == 0 for item in target_samples),
            "positive": sum(item["target_class"] == 1 for item in target_samples),
        }
        if (row.get("schema_version") != 1 or row.get("fold") != fold
            or row.get("context_sha256") != sha(out / "reports" / CONTEXT)
            or row.get("selected_task") != context["selected_task"]
            or row.get("checkpoint_sha256") != context["checkpoint_sha256"][fold]
            or row.get("split_membership_digest") != context["fold_membership_digest"][fold]
            or row.get("sample_count") != 8
            or row.get("subject_level_outputs_written") is not False
            or row.get("aggregate_only") is not True
            or row.get("model_selection_overlap") is not True
            or row.get("target_definition") != "FROZEN_UNOCCLUDED_PREDICTED_CLASS_BEFORE_ROI_SELECTION"
            or not isinstance(row.get("frozen_target_class_counts"), dict)
            or set(row["frozen_target_class_counts"]) != {"negative", "positive"}
            or any(type(value) is not int or value < 0
                   for value in row["frozen_target_class_counts"].values())
            or sum(row["frozen_target_class_counts"].values()) != 8
            or row["frozen_target_class_counts"] != expected_target_counts
            or len(row.get("crossfit_directions", [])) != 2
            or [item.get("seed") for item in row.get("randomization_sanity", [])]
               != [13, 29, 47]
            or type(row.get("paired_mean_effect")) not in (float, int)
            or not math.isfinite(row["paired_mean_effect"])
            or type(row.get("paired_mean_top_target_drop")) not in (float, int)
            or not math.isfinite(row["paired_mean_top_target_drop"])
            or type(row.get("peak_reserved_gb")) not in (float, int)
            or not 0 <= row["peak_reserved_gb"] <= 7.5):
            raise RuntimeError(f"V22_FOLD_RECEIPT_INVALID:{fold}")
        directions = row["crossfit_directions"]
        if (any(not isinstance(direction, dict)
                or direction.get("selection_n") != 4
                or direction.get("assessment_n") != 4
                or type(direction.get("assessment_mean_paired_effect")) not in (float, int)
                or not math.isfinite(direction["assessment_mean_paired_effect"])
                or type(direction.get("assessment_mean_top_target_drop")) not in (float, int)
                or not math.isfinite(direction["assessment_mean_top_target_drop"])
                or type(direction.get("top_volume")) is not int
                or direction["top_volume"] <= 0
                or len(direction.get("controls", [])) != 3
                or len({item.get("atlas_id") for item in direction["controls"]}) != 3
                or any(item.get("atlas_id") == direction.get("top_region_id")
                       or type(item.get("volume")) is not int
                       or not 1 / 1.5 <= item["volume"] / direction["top_volume"] <= 1.5
                       for item in direction["controls"])
                for direction in directions)
            or not math.isclose(row["paired_mean_effect"],
                                sum(item["assessment_mean_paired_effect"] for item in directions) / 2,
                                rel_tol=1e-8, abs_tol=1e-10)
            or not math.isclose(row["paired_mean_top_target_drop"],
                                sum(item["assessment_mean_top_target_drop"] for item in directions) / 2,
                                rel_tol=1e-8, abs_tol=1e-10)
            or type(row.get("paired_positive_count")) is not int
            or not 0 <= row["paired_positive_count"] <= 8):
            raise RuntimeError(f"V22_CROSSFIT_OR_PAIRED_EFFECT_INVALID:{fold}")
        for sanity in row["randomization_sanity"]:
            correlation = sanity.get("absolute_correlation")
            expected_sanity = ("PASS" if type(correlation) in (float, int)
                               and math.isfinite(correlation) and 0 <= correlation < 0.9
                               else "FAIL")
            if sanity.get("status") != expected_sanity:
                raise RuntimeError(f"V22_RANDOMIZATION_GATE_INCONSISTENT:{fold}")
        expected_gate = ("PASS_EXPLORATORY" if row["paired_mean_effect"] > 0
                         and row["paired_mean_top_target_drop"] > 0
                         and all(item["status"] == "PASS" for item in row["randomization_sanity"])
                         else "BLOCKED_EXPLORATORY_GATE")
        if row.get("gate_status") != expected_gate:
            raise RuntimeError(f"V22_FOLD_GATE_INCONSISTENT:{fold}")
        rows.append(row)
        hashes[str(fold)] = sha(path)
    return rows, hashes


def verify_xai_summary(root: Path, out: Path, protocol: dict) -> dict:
    context = ensure_context(root, out, protocol)
    reports = out / "reports"
    summary_path, state_path = reports / SUMMARY, reports / STATE
    if not summary_path.is_file() or not state_path.is_file():
        raise RuntimeError("V22_XAI_SUMMARY_OR_STATE_NOT_TERMINAL")
    summary, state = read(summary_path), read(state_path)
    rows, hashes = _validated_receipts(out, context)
    if [row["fold"] for row in rows] != list(range(5)):
        raise RuntimeError("V22_XAI_FIVE_FOLD_RECEIPTS_MISSING")
    aggregate = aggregate_five_folds(rows)
    if (state.get("status") != "COMPLETED" or state.get("result_sha256") != sha(summary_path)
        or state.get("context_sha256") != sha(reports / CONTEXT)
        or state.get("completed_folds") != hashes
        or summary.get("status") != aggregate["gate_status"]
        or summary.get("aggregate") != aggregate
        or summary.get("context_sha256") != sha(reports / CONTEXT)
        or summary.get("fold_receipt_sha256") != hashes
        or summary.get("folds") != rows
        or summary.get("n_images") != 40
        or summary.get("not_independent_testing") is not True
        or summary.get("anatomical_or_causal_region_claim_allowed") is not False
        or summary.get("subject_level_outputs_written") is not False
        or summary.get("fold_mean_paired_effect") != aggregate["fold_mean_effect"]):
        raise RuntimeError("V22_XAI_AGGREGATE_OR_RECEIPT_DRIFT")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--output-root", type=Path,
                        default=Path("results/final_results/method_x_fasttrack_v2_20260923"))
    parser.add_argument("--mode", choices=("audit", "run"), required=True)
    args = parser.parse_args()
    root, out = args.project_root.resolve(), (args.project_root / args.output_root).resolve()
    if root != ROOT.resolve() or not out.is_relative_to(root) or out == root:
        raise RuntimeError("V22_PROJECT_OR_OUTPUT_ROOT_INVALID")
    from scripts.mx_fasttrack_v2_2.supervise_v22 import ensure_protocol

    protocol = ensure_protocol(root, out)
    config = read(root / "configs/experiments/method_x_xai_v22.json")
    if (config.get("samples_per_fold") != 8 or config.get("randomization_seeds") != [13, 29, 47]
        or config.get("bootstrap_resamples") != 1000
        or config.get("claim_ceiling") != "INTERNAL_EXPLORATORY_MODEL_ATTRIBUTION_NOT_DISEASE_REGION"
        or config.get("cuda_required") is not True):
        raise RuntimeError("V22_XAI_CONFIG_PROTOCOL_DRIFT")
    if args.mode == "run":
        require_internal_coordinator_parent(root)
    context = ensure_context(root, out, protocol)
    if args.mode == "audit":
        print(json.dumps({"status": "V22_XAI_ATLAS_SPM_CONTEXT_PASS",
                          "context_sha256": sha(out / "reports" / CONTEXT)}))
        return 0
    import nibabel as nib
    import numpy as np
    import torch
    from scripts.train_3d_mri import H5MethodXDataset
    from scripts.mx_fasttrack_v2.run import gpu_idle
    from src.evaluation.method_x_xai_forward import build_model_from_checkpoint

    reports = out / "reports"
    summary_path, state_path = reports / SUMMARY, reports / STATE
    if summary_path.exists() and state_path.exists() and read(state_path).get("status") == "COMPLETED":
        result = verify_xai_summary(root, out, protocol)
        print(json.dumps({"status": "REUSED_VERIFIED_V22_XAI", "gate_status": result["status"]}))
        return 0 if result["status"] == "COMPLETED_EXPLORATORY_CROSSFIT" else 3
    prior = read(state_path) if state_path.exists() else None
    if prior and (prior.get("status") not in ("RUNNING", "STARTED")
                  or prior.get("context_sha256") != sha(reports / CONTEXT)):
        raise RuntimeError("V22_XAI_RESUME_STATE_DRIFT")
    rows, hashes = _validated_receipts(out, context)
    if prior and any(prior.get("completed_folds", {}).get(key) not in (None, value)
                     for key, value in hashes.items()):
        raise RuntimeError("V22_XAI_COMMITTED_RECEIPT_CHANGED")
    if prior and set(prior.get("completed_folds", {})) - set(hashes):
        raise RuntimeError("V22_XAI_COMMITTED_RECEIPT_MISSING")
    if len(rows) < 5:
        if not torch.cuda.is_available():
            raise RuntimeError("V22_XAI_CUDA_REQUIRED_FOR_UNFINISHED_FOLDS")
        # The process retains its own CUDA context after a completed fold.
        # Check for other writers once, before opening the first model.
        gpu_idle()
    now = datetime.now(timezone.utc).isoformat()
    state = {"schema_version": 1, "status": "RUNNING",
             "context_sha256": sha(reports / CONTEXT),
             "completed_folds": hashes,
             "execution_attempts": int(prior.get("execution_attempts", 0) if prior else 0) + 1,
             "first_started_at_utc": prior.get("first_started_at_utc", now) if prior else now,
             "last_resumed_at_utc": now}
    _atomic_state(state_path, state)
    prereg = read(reports / "fasttrack_preregistration.json")
    resolved, _ = atlas_binding(root, prereg)
    atlas = np.asarray(nib.load(resolved["paths"]["atlas_path"]).dataobj).astype(np.int32)
    cache = root / "data/cached/t1_volume_cache_128.h5"
    lock = check_lock(root, out)
    selected = context["selected_task"]
    for fold in range(5):
        if str(fold) in hashes:
            continue
        cp = out / "inner_tasks" / selected / f"method_x_3d_fold_{fold}.pt"
        if sha(cp) != context["checkpoint_sha256"][fold]:
            raise RuntimeError(f"V22_CHECKPOINT_CHANGED:{fold}")
        checkpoint = torch.load(cp, map_location="cpu", weights_only=True)
        info = lock["tasks"][selected]
        if (checkpoint.get("fold") != fold
            or checkpoint.get("hpo_selection_only") is not True
            or checkpoint.get("run_contract_digest") != info["run_contract_digest"]
            or checkpoint.get("split_membership_digest") != context["fold_membership_digest"][fold]
            or checkpoint.get("outer_test_access_count") != 0
            or checkpoint.get("outer_test_evaluation_count") != 0
            or checkpoint.get("input_mode") != "mri_plus_tabular"
            or checkpoint.get("drop_tabular_feature") != "none"):
            raise RuntimeError(f"V22_CHECKPOINT_SCOPE_INVALID:{fold}")
        indices, labels = inner_validation(root, checkpoint, fold, cache)
        dataset = H5MethodXDataset(cache, indices, labels, input_mode="mri_plus_tabular")
        mean = torch.as_tensor(checkpoint["tab_mean"], dtype=torch.float32)
        std = torch.as_tensor(checkpoint["tab_std"], dtype=torch.float32).clamp_min(1.0)
        if mean.numel() != 5 or std.numel() != 5:
            raise RuntimeError("V22_TABULAR_NORMALIZATION_INVALID")

        class Batches:
            def __len__(self):
                return len(dataset)

            def __getitem__(self, item):
                if item < 0 or item >= len(dataset):
                    raise IndexError(item)
                image, tab, _, presence = dataset[item]
                return image.unsqueeze(0), ((tab - mean) / std).unsqueeze(0), presence.reshape(1)

        model, source = build_model_from_checkpoint(checkpoint, device=torch.device("cuda"))
        if source != "model_state_dict":
            raise RuntimeError("V22_MODEL_STATE_SOURCE_UNEXPECTED")
        torch.cuda.reset_peak_memory_stats()
        target_path = out / "_private_resume" / PRIVATE_TARGETS / f"fold_{fold}.json"
        probabilities = []
        with torch.no_grad():
            for ordinal in range(8):
                image, tabular, presence = Batches()[ordinal]
                probability = torch.sigmoid(model(
                    image.to("cuda"), tabular.to("cuda"),
                    tab_presence=presence.to("cuda")).reshape(-1))[0].item()
                probabilities.append(float(probability))
        target_receipt = {
            "schema_version": 1, "fold": fold,
            "context_sha256": sha(reports / CONTEXT),
            "checkpoint_sha256": context["checkpoint_sha256"][fold],
            "split_membership_digest": context["fold_membership_digest"][fold],
            "target_rule": "unoccluded_predicted_class",
            "samples": [{"ordinal": ordinal, "original_probability": value,
                         "target_class": int(value >= 0.5)}
                        for ordinal, value in enumerate(probabilities)],
            "public_subject_level_outputs_written": False,
        }
        if target_path.exists():
            recorded = read(target_path)
            if (set(recorded) != set(target_receipt)
                or any(recorded.get(key) != value for key, value in target_receipt.items()
                       if key != "samples")
                or len(recorded.get("samples", [])) != 8
                or any(old.get("ordinal") != new["ordinal"]
                       or old.get("target_class") != new["target_class"]
                       or not math.isclose(old.get("original_probability", float("nan")),
                                           new["original_probability"],
                                           rel_tol=0, abs_tol=1e-6)
                       for old, new in zip(recorded["samples"], target_receipt["samples"], strict=True))):
                raise RuntimeError(f"V22_PRECOMMITTED_TARGET_DRIFT:{fold}")
            target_receipt = recorded
        else:
            immutable_json(target_path, target_receipt)
        result = run_fold_crossfit(model=model, batches=Batches(), atlas_data=atlas,
                                   region_names=resolved["names"],
                                   randomization_seeds=config["randomization_seeds"],
                                   frozen_original_probabilities=[
                                       item["original_probability"]
                                       for item in target_receipt["samples"]])
        reserved = torch.cuda.max_memory_reserved() / 1024**3
        if reserved >= config["hard_stop_reserved_gb"]:
            raise RuntimeError(f"V22_XAI_RESERVED_VRAM_LIMIT_EXCEEDED:{fold}")
        result.update({"schema_version": 1, "fold": fold,
                       "context_sha256": sha(reports / CONTEXT),
                       "selected_task": selected,
                       "target_receipt_sha256": sha(target_path),
                       "checkpoint_sha256": context["checkpoint_sha256"][fold],
                       "split_membership_digest": context["fold_membership_digest"][fold],
                       "state_source": source, "peak_reserved_gb": round(reserved, 3)})
        path = reports / RECEIPTS / f"fold_{fold}.json"
        immutable_json(path, result)
        state["completed_folds"][str(fold)] = sha(path)
        _atomic_state(state_path, state)
        del model, checkpoint, dataset
        torch.cuda.empty_cache()
    rows, hashes = _validated_receipts(out, context)
    if [row["fold"] for row in rows] != list(range(5)):
        raise RuntimeError("V22_XAI_FIVE_FOLDS_NOT_COMPLETE")
    aggregate = aggregate_five_folds(rows)
    summary = {"schema_version": 1, "status": aggregate["gate_status"],
               "scope": "inner_validation_crossfit_8_per_fold",
               "attribution": "NORMALIZED_ABSOLUTE_INPUT_GRADIENT_NOT_GRADCAM",
               "context_sha256": sha(reports / CONTEXT),
               "selected_task": selected, "n_images": 40,
               "folds": rows, "fold_receipt_sha256": hashes,
               "aggregate": aggregate,
               "fold_mean_paired_effect": aggregate["fold_mean_effect"],
               "not_independent_testing": True,
               "anatomical_or_causal_region_claim_allowed": False,
               "subject_level_outputs_written": False}
    immutable_json(summary_path, summary)
    state.update({"status": "COMPLETED", "result_sha256": sha(summary_path),
                  "completed_folds": hashes})
    _atomic_state(state_path, state)
    verify_xai_summary(root, out, protocol)
    print(json.dumps({"status": summary["status"], "n_images": 40,
                      "fold_mean_paired_effect": aggregate["fold_mean_effect"]}))
    return 0 if summary["status"] == "COMPLETED_EXPLORATORY_CROSSFIT" else 3


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError, KeyError, TypeError) as exc:
        print(json.dumps({"status": "BLOCKED", "reason": str(exc)}, ensure_ascii=False), file=sys.stderr)
        raise SystemExit(4)
