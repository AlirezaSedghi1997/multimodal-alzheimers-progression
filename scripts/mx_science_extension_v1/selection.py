"""Write-once, fold-local Hybrid3D choices using only inner-validation loss."""
from __future__ import annotations

import math
from pathlib import Path

from scripts.mx_fasttrack_v2.run import read, sha
from scripts.mx_fasttrack_v2_1.supervise_v21 import immutable_json
from scripts.mx_science_extension_v1.audit_and_lock import OUTPUT, PARENT
from scripts.mx_science_extension_v1.terminal import require_parent_mx7_terminal

SEEDS = (13, 29, 47)
VARIANTS = ("baseline", "plus_dense_skip")
NAME = "fold_local_hybrid_selection.v1.json"


def arm_directory(root: Path, seed: int, variant: str) -> Path:
    if seed not in SEEDS or variant not in VARIANTS:
        raise ValueError("EXTENSION_SELECTION_ARM_INVALID")
    if seed == 13:
        name = "hybrid_baseline" if variant == "baseline" else "hybrid_dense"
        return root / PARENT / "inner_tasks" / name
    return root / OUTPUT / "inner_tasks" / f"hybrid_{variant}_seed{seed}"


def checked_inner_arm(directory: Path, *, seed: int, variant: str,
                      cache_sha256: str, spm_sha256: str) -> dict:
    metrics_path = directory / "three_d_metrics.json"
    contract_path = directory / "run_contract.json"
    metrics, contract_file = read(metrics_path), read(contract_path)
    contract = contract_file.get("run_contract") or {}
    parameters = contract.get("parameters") or {}
    aggregate = metrics.get("aggregate") or {}
    if (contract_file.get("run_contract_digest") != contract.get("digest")
        or contract.get("seed") != seed or contract.get("folds") != 5
        or contract.get("epochs") != 100
        or contract.get("cache_sha256") != cache_sha256
        or contract.get("spm_snapshot_sha256") != spm_sha256
        or parameters.get("seed") != seed
        or parameters.get("hybrid_variant") != variant
        or parameters.get("backbone") != "amuse_x_hybrid3d"
        or parameters.get("input_mode") != "mri_plus_tabular"
        or parameters.get("phase") != "final"
        or parameters.get("hpo_selection_only") is not True
        or aggregate.get("status") != "COMPLETED"
        or aggregate.get("hpo_selection_only") is not True
        or aggregate.get("outer_test_access_count") != 0
        or aggregate.get("outer_test_used_for_selection") is not False
        or aggregate.get("folds") != 5
        or aggregate.get("completed_folds") != 5
        or aggregate.get("epochs_per_fold") != 20):
        raise RuntimeError(f"EXTENSION_INNER_ARM_CONTRACT_INVALID:{directory}")
    rows = metrics.get("folds")
    if not isinstance(rows, list) or len(rows) != 5:
        raise RuntimeError("EXTENSION_INNER_FOLD_ROWS_INVALID")
    by_fold = {}
    for row in rows:
        fold, loss = row.get("fold"), row.get("best_val_loss")
        if (type(fold) is not int or fold not in range(5) or fold in by_fold
            or type(loss) not in (int, float) or not math.isfinite(loss)
            or loss < 0 or row.get("outer_test_evaluation_count") != 0
            or row.get("hpo_selection_only") is not True):
            raise RuntimeError("EXTENSION_INNER_FOLD_EVIDENCE_INVALID")
        checkpoint = directory / f"method_x_3d_fold_{fold}.pt"
        if not checkpoint.is_file():
            raise RuntimeError("EXTENSION_INNER_CHECKPOINT_MISSING")
        by_fold[fold] = {"best_val_loss": float(loss),
                         "checkpoint_sha256": sha(checkpoint)}
    if set(by_fold) != set(range(5)):
        raise RuntimeError("EXTENSION_INNER_FOLD_COVERAGE_INVALID")
    return {"metric_sha256": sha(metrics_path), "contract_sha256": sha(contract_path),
            "contract_digest": contract["digest"], "folds": by_fold}


def build_selection(root: Path) -> dict:
    require_parent_mx7_terminal(root)
    reports = root / PARENT / "reports"
    prereg = read(reports / "fasttrack_preregistration.json")
    spm = read(root / "results/final_results/method_x_model_battery_v3_depth_and_diversity_20260729/battery_manifest.json")
    cache_sha = str(prereg.get("cache_sha256") or "")
    spm_sha = str((spm.get("spm_gate_snapshot") or {}).get("snapshot_sha256") or "")
    if len(cache_sha) != 64 or len(spm_sha) != 64:
        raise RuntimeError("EXTENSION_PARENT_CACHE_OR_SPM_BINDING_MISSING")
    arms = {(seed, variant): checked_inner_arm(
        arm_directory(root, seed, variant), seed=seed, variant=variant,
        cache_sha256=cache_sha, spm_sha256=spm_sha)
        for seed in SEEDS for variant in VARIANTS}
    choices = []
    for seed in SEEDS:
        for fold in range(5):
            baseline = arms[(seed, "baseline")]["folds"][fold]
            dense = arms[(seed, "plus_dense_skip")]["folds"][fold]
            chosen = "plus_dense_skip" if dense["best_val_loss"] < baseline["best_val_loss"] else "baseline"
            choices.append({"seed": seed, "fold": fold, "selected_variant": chosen,
                            "baseline_inner_bce": baseline["best_val_loss"],
                            "dense_inner_bce": dense["best_val_loss"],
                            "selected_checkpoint_sha256": (dense if chosen == "plus_dense_skip"
                                                           else baseline)["checkpoint_sha256"]})
    return {
        "schema_version": 1, "status": "LOCKED_INNER_ONLY_BEFORE_EXTENSION_OUTER",
        "selection_rule": "within_fold_and_seed_min_best_inner_bce_tie_baseline",
        "outer_used_for_selection": False, "external_validation": False,
        "parent_preregistration_sha256": sha(reports / "fasttrack_preregistration.json"),
        "parent_v241_amendment_sha256": sha(reports / "fasttrack_v241_verifier_repair_before_outer.json"),
        "parent_v242_amendment_sha256": sha(reports / "fasttrack_v242_checkpoint_projection_before_outer.json"),
        "parent_v244_amendment_sha256": sha(reports / "fasttrack_v244_posthoc_xai_atlas_eligibility.json"),
        "arms": {f"{seed}|{variant}": {"metric_sha256": row["metric_sha256"],
                                        "contract_sha256": row["contract_sha256"],
                                        "contract_digest": row["contract_digest"],
                                        "checkpoint_sha256": [row["folds"][fold]["checkpoint_sha256"]
                                                              for fold in range(5)]}
                 for (seed, variant), row in sorted(arms.items())},
        "choices": choices,
    }


def lock_selection(root: Path) -> dict:
    selection = build_selection(root)
    path = root / OUTPUT / "reports" / NAME
    immutable_json(path, selection)
    return selection
