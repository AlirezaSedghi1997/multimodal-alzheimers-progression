"""Optional MX06 fast-track: five fold-local inner-validation saliency probes.

Eight prespecified real validation images per fold. The attribution is normalized
absolute input gradient; neither its atlas labels nor its fold-bootstrap interval
identify disease anatomy. The original MX06 gate remains NOT_DONE.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.mx_fasttrack_v2.evaluate_locked import atomic, hash_file, read
from scripts.mx_fasttrack_v2.integrity import check_lock


def atlas_binding(root: Path, prereg: dict) -> tuple[dict, dict]:
    """Verify full atlas geometry and a preprocessing snapshot bound to battery."""
    from scripts.run_method_x_anatomical_xai import resolve_locked_atlas
    from src.evaluation.xai_validation import load_registered_region_names, validate_registered_atlas
    from src.training.method_x_spm_snapshot import verify_spm_gate_snapshot

    atlas = resolve_locked_atlas()
    validate_registered_atlas(registry_path=atlas["registry_path"],
                              atlas_path=atlas["atlas_path"], reference_path=atlas["reference_path"])
    names = load_registered_region_names(registry_path=atlas["registry_path"],
                                         label_table_path=atlas["labels_path"])
    battery_path = root / "results/final_results/method_x_model_battery_v3_depth_and_diversity_20260729/battery_manifest.json"
    if hash_file(battery_path) != prereg["battery_manifest_sha256"]:
        raise RuntimeError("XAI_BATTERY_BINDING_CHANGED")
    binding = read(battery_path).get("spm_gate_snapshot") or {}
    relative = Path(str(binding.get("path") or ""))
    digest = binding.get("snapshot_sha256")
    if not relative.parts or relative.is_absolute() or ".." in relative.parts or not isinstance(digest, str) or len(digest) != 64:
        raise RuntimeError("XAI_SPM_BATTERY_BINDING_MISSING")
    spm = verify_spm_gate_snapshot(root / relative, root=root, verify_sources=True)
    if spm.get("verified") is not True or spm.get("ready") is not True or spm.get("snapshot_sha256") != digest:
        raise RuntimeError("XAI_SPM_SOURCE_OR_SNAPSHOT_NOT_VERIFIED")
    # Public output carries relative paths and SHA only.
    provenance = {key: atlas[key] for key in
                  ("registry_sha256", "labels_sha256", "atlas_sha256", "reference_sha256", "asset_id", "license_id")}
    provenance["spm_snapshot_sha256"] = digest
    return {"paths": atlas, "names": names}, provenance


def inner_validation(root: Path, ckpt: dict, fold: int, cache: Path) -> tuple[list[int], list[int]]:
    """Recreate the exact trainer split, then bind it to the saved fold digest."""
    from scripts.train_3d_mri import cache_index, ok_indices, resolve_train_indices, split_fit_val_indices
    from src.data.combined_mri_cache import cache_has_supplementary
    from src.utils.reproducibility import split_membership_digest

    parameters = ((ckpt.get("run_contract") or {}).get("parameters") or {})
    if not isinstance(parameters, dict) or parameters.get("seed") not in (None, 13):
        raise RuntimeError("XAI_TRAINING_SPLIT_PARAMETERS_MISSING")
    val_fraction = parameters.get("val_fraction")
    if not isinstance(val_fraction, (int, float)) or not 0 < float(val_fraction) < 1:
        raise RuntimeError("XAI_VALIDATION_FRACTION_MISSING")
    ok = ok_indices(cache)
    train_idx, train_y = resolve_train_indices(
        root / "splits" / f"fold_{fold}_train.csv", cache_index(cache), ok,
        cache_path=cache,
        include_supplementary=(not bool(parameters.get("no_supplementary_train", False))
                               and cache_has_supplementary(cache)),
        include_global_supplementary=bool(parameters.get("include_global_supplementary_train", False)),
        limit=0,
    )
    fit_idx, _, val_idx, val_y = split_fit_val_indices(
        train_idx, train_y, cache_path=cache, ok=ok,
        val_fraction=float(val_fraction), seed=13, fold=fold)
    digest = split_membership_digest(fit=fit_idx, validation=val_idx, outer_test=())
    if digest != ckpt.get("split_membership_digest") or len(val_idx) < 8:
        raise RuntimeError(f"XAI_INNER_VALIDATION_DIGEST_OR_COVERAGE_FAILED:{fold}")
    return list(val_idx[:8]), list(val_y[:8])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--project-root", type=Path, default=ROOT)
    ap.add_argument("--output-root", type=Path,
                    default=Path("results/final_results/method_x_fasttrack_v2_20260923"))
    ap.add_argument("--mode", choices=("audit", "run"), required=True)
    args = ap.parse_args()
    root = args.project_root.resolve()
    if root != ROOT.resolve():
        raise RuntimeError("PROJECT_ROOT_MISMATCH")
    out = (root / args.output_root).resolve()
    lock = check_lock(root, out)
    lock_path = out / "reports/fasttrack_inner_lock.json"
    prereg = read(out / "reports/fasttrack_preregistration.json")
    config = config_for_tree(root, out)["mx06_limited"]
    if (config["sample_source"] != "locked_inner_validation_not_independent"
        or config["samples_per_fold"] != 8 or config["randomization_seeds"] != [13, 29, 47]):
        raise RuntimeError("XAI_PREDECLARED_DESIGN_DRIFT")
    resolved, provenance = atlas_binding(root, prereg)
    if args.mode == "audit":
        print(json.dumps({"status": "ATLAS_SPM_PROVENANCE_PASS", "provenance": provenance,
                          "sample_source": config["sample_source"]}))
        return 0
    import nibabel as nib
    import numpy as np
    import torch
    from scripts.train_3d_mri import H5MethodXDataset
    from src.evaluation.method_x_xai_forward import (
        aggregate_fold_regions, build_model_from_checkpoint, run_fold_attribution)
    from src.evaluation.xai_validation import evaluate_fold_xai_gates

    if not torch.cuda.is_available() or not config["cuda_required"]:
        raise RuntimeError("XAI_CUDA_REQUIRED")
    output = out / "reports/xai_limited_summary.json"
    state_path = out / "reports/xai_limited_state.json"
    receipt_root = out / "reports/xai_limited_folds"
    current_lock_sha = hash_file(lock_path)
    if state_path.exists():
        state = read(state_path)
        if (state.get("status") == "COMPLETED" and output.is_file()
            and state.get("lock_sha256") == current_lock_sha
            and state.get("result_sha256") == hash_file(output)):
            print(json.dumps({"status": "REUSED_EXISTING_LIMITED_XAI",
                              "gate_status": state.get("gate_status")}))
            return 0 if state.get("gate_status") == "COMPLETED_EXPLORATORY_LIMITED" else 3
        if (state.get("status") not in {"STARTED", "RUNNING"}
            or state.get("lock_sha256") != current_lock_sha
            or state.get("source") != "inner_validation"
            or state.get("expected_n") != 40):
            raise RuntimeError("XAI_RESUME_STATE_OR_PROTOCOL_DRIFT")
        completed_folds = state.get("completed_folds") or {}
        attempts = int(state.get("execution_attempts", 1)) + 1
        first_started = state.get("first_started_at_utc")
    else:
        completed_folds, attempts, first_started = {}, 1, None
    if not isinstance(completed_folds, dict):
        raise RuntimeError("XAI_COMPLETED_FOLD_STATE_INVALID")
    atlas_data = np.asarray(nib.load(resolved["paths"]["atlas_path"]).dataobj).astype(np.int32)
    if {int(i) for i in np.unique(atlas_data) if int(i) > 0} != set(resolved["names"]):
        raise RuntimeError("XAI_ATLAS_LABELS_INCOMPLETE")
    now = datetime.now(timezone.utc).isoformat()
    state = {"schema_version": 2, "status": "RUNNING", "lock_sha256": current_lock_sha,
             "source": "inner_validation", "expected_n": 40,
             "expected_folds": list(range(5)), "completed_folds": completed_folds,
             "execution_attempts": attempts, "first_started_at_utc": first_started or now,
             "last_resumed_at_utc": now}
    atomic(state_path, state)
    cache = root / "data/cached/t1_volume_cache_128.h5"
    fold_results, gate_rows = [], []
    device = torch.device("cuda")
    for fold in range(5):
        task = lock["final_task"]
        info = lock["tasks"][task]
        cp = out / "inner_tasks" / task / f"method_x_3d_fold_{fold}.pt"
        receipt_path = receipt_root / f"fold_{fold}.json"
        if receipt_path.is_file():
            result = read(receipt_path)
            digest = hash_file(receipt_path)
            expected = state["completed_folds"].get(str(fold))
            if expected not in (None, digest):
                raise RuntimeError(f"XAI_FOLD_RECEIPT_HASH_CHANGED:{fold}")
            if (result.get("schema_version") != 1 or result.get("fold") != fold
                or result.get("lock_sha256") != current_lock_sha
                or result.get("selected_task") != task
                or result.get("checkpoint_sha256") != info["checkpoint_sha256"][fold]
                or result.get("subject_level_outputs_written") is not False
                or result.get("sample_count") != 8):
                raise RuntimeError(f"XAI_FOLD_RECEIPT_BINDING_INVALID:{fold}")
            state["completed_folds"][str(fold)] = digest
            atomic(state_path, state)
            fold_results.append(result)
            gate_rows.append({"fold": fold, **result["gate"]})
            continue
        if str(fold) in state["completed_folds"]:
            raise RuntimeError(f"XAI_FOLD_RECEIPT_MISSING_AFTER_COMMIT:{fold}")
        ckpt = torch.load(cp, map_location="cpu", weights_only=True)
        if (hash_file(cp) != info["checkpoint_sha256"][fold]
            or ckpt.get("fold") != fold or ckpt.get("hpo_selection_only") is not True
            or ckpt.get("run_contract_digest") != info["run_contract_digest"]
            or ckpt.get("split_membership_digest") != info["fold_membership_digest"][fold]
            or ckpt.get("outer_test_access_count") != 0
            or ckpt.get("outer_test_evaluation_count") != 0
            or ckpt.get("input_mode") != "mri_plus_tabular"
            or ckpt.get("drop_tabular_feature") != "none"):
            raise RuntimeError(f"XAI_CHECKPOINT_SCOPE_INVALID:{fold}")
        indices, labels = inner_validation(root, ckpt, fold, cache)
        dataset = H5MethodXDataset(cache, indices, labels, input_mode="mri_plus_tabular")
        mean = torch.as_tensor(ckpt["tab_mean"], dtype=torch.float32)
        std = torch.as_tensor(ckpt["tab_std"], dtype=torch.float32).clamp_min(1.0)
        if mean.numel() != 5 or std.numel() != 5:
            raise RuntimeError("XAI_TABULAR_NORMALIZATION_INVALID")

        class Batches:
            def __len__(self) -> int:
                return len(dataset)

            def __getitem__(self, item: int):
                if item < 0 or item >= len(dataset):
                    raise IndexError(item)
                image, tab, _, presence = dataset[item]
                return image.unsqueeze(0), ((tab - mean) / std).unsqueeze(0), presence.reshape(1)

        from scripts.mx_fasttrack_v2.run import config_for_tree,  gpu_idle

        gpu_idle()
        model, state_source = build_model_from_checkpoint(ckpt, device=device)
        if state_source != "model_state_dict":
            raise RuntimeError("XAI_MODEL_STATE_SOURCE_UNEXPECTED")
        torch.cuda.reset_peak_memory_stats()
        result = run_fold_attribution(
            model=model, batches=Batches(), atlas_data=atlas_data,
            region_names=resolved["names"],
            randomization_seeds=config["randomization_seeds"],
            bootstrap_resamples=config["bootstrap_resamples"])
        reserved = torch.cuda.max_memory_reserved() / 1024**3
        if reserved >= config["hard_stop_reserved_gb"]:
            raise RuntimeError(f"XAI_RESERVED_VRAM_LIMIT_EXCEEDED:{fold}")
        result.update({"schema_version": 1, "fold": fold,
                       "lock_sha256": current_lock_sha, "selected_task": task,
                       "checkpoint_sha256": info["checkpoint_sha256"][fold],
                       "state_source": state_source, "peak_reserved_gb": round(reserved, 3),
                       "subject_level_outputs_written": False})
        if result["sample_count"] != 8:
            raise RuntimeError("XAI_SAMPLE_COUNT_INVALID")
        atomic(receipt_path, result)
        state["completed_folds"][str(fold)] = hash_file(receipt_path)
        atomic(state_path, state)
        fold_results.append(result)
        gate_rows.append({"fold": fold, **result["gate"]})
        del model, ckpt, dataset
        torch.cuda.empty_cache()
    gates = evaluate_fold_xai_gates(gate_rows, expected_folds=5, required_randomization_seeds=3)
    status = "COMPLETED_EXPLORATORY_LIMITED" if gates.get("status") == "COMPLETED" else "BLOCKED_EXPLORATORY_GATE"
    payload = {"schema_version": 2, "status": status, "scope": "inner_validation_probe_8_per_fold",
               "attribution": "NORMALIZED_ABSOLUTE_INPUT_GRADIENT_NOT_GRADCAM",
               "lock_sha256": hash_file(lock_path), "selected_task": lock["final_task"],
               "n_images": 40, "folds": fold_results, "gates": gates,
               "region_summary": aggregate_fold_regions(
                   fold_results, bootstrap_resamples=config["bootstrap_resamples"]),
               "region_summary_ci_resampling_unit": "fold_mean_five_folds_not_patient",
               "provenance": provenance, "randomization_seeds": [13, 29, 47],
               "same_data_used_to_choose_top_roi_and_probe_occlusion": True,
               "random_roi_controls_selected_from_first_atlas_ids": True,
               "not_independent_testing": True, "anatomical_or_causal_region_claim_allowed": False,
               "subject_level_outputs_written": False}
    atomic(output, payload)
    state.update({"status": "COMPLETED", "lock_sha256": current_lock_sha,
                  "result_sha256": hash_file(output), "gate_status": status,
                  "fold_receipt_sha256": dict(state["completed_folds"])})
    atomic(state_path, state)
    print(json.dumps({"status": status, "n": 40, "gates": gates["status"]}))
    return 0 if status == "COMPLETED_EXPLORATORY_LIMITED" else 3


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError, KeyError, TypeError) as exc:
        print(json.dumps({"status": "BLOCKED", "reason": str(exc)}), file=sys.stderr)
        raise SystemExit(3)
