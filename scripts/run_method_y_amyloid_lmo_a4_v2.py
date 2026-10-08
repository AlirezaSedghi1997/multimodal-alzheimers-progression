"""Manual, resumable A0-versus-A4 amyloid manufacturer-holdout study.

This is a post-hoc internal policy comparison. A4 differs from A0 in both
encoder and augmentation. No external, anatomical, quantitative, or positive
clinical claim can be inferred from this run.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import shutil
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_method_y_amyloid_lmo_v1 as common
from scripts.audit_method_y_amyloid_lmo_release_v1 import PUBLIC_METRIC_KEYS, _check_public


class A4LMOError(RuntimeError):
    pass


def _validate_policy(policy: dict[str, Any]) -> None:
    required = {
        "schema_version": "2.0",
        "analysis_role": "POST_HOC_INTERNAL_SCANNER_TRANSPORT_EXPLORATORY",
        "source_factorial_config": "configs/method_y_image_factorial_v4.yaml",
        "output_root": "results/method_y_revival_v4_amyloid_lmo_a4_v2",
        "branch": "amyloid",
        "tracer": "FBP",
        "development_cohort": "A4",
        "manufacturer_order": ["GE", "PHILIPS", "SIEMENS"],
        "minimum_subjects_per_heldout_manufacturer": 30,
        "minimum_events_per_heldout_manufacturer": 10,
        "inner_folds": 3,
        "active_arms": ["A0", "A4"],
        "baseline_arm": "A0",
        "selected_v4_arm": "A4",
        "claim_status": "POST_HOC_INTERNAL_EXPLORATORY",
        "external_validation_status": "EXTERNAL_PENDING",
        "quantitative_equivalence_status": "BLOCKED",
        "positive_claim_allowed": False,
        "requested_horizons_days": [547.875, 1095.75, 1640.0],
    }
    for key, value in required.items():
        if policy.get(key) != value:
            raise A4LMOError(f"POLICY_CONTRACT_INVALID:{key}")
    if int(policy.get("minimum_h_free_gib", 0)) < 20:
        raise A4LMOError("MINIMUM_H_FREE_SPACE_POLICY_INVALID")


def _preflight(policy_path: Path) -> dict[str, Any]:
    import pandas as pd
    import torch
    import yaml

    from src.method_y_revival.atomic_io import read_json
    from src.method_y_revival.contracts import sha256_mapping
    from src.method_y_revival.image_lmo_v1 import build_lmo_plan, public_plan_summary
    from src.method_y_revival.software_contract_v4 import (
        bind_software_runtime_contract_v4, binding_core,
    )

    policy = yaml.safe_load(policy_path.read_text(encoding="utf-8"))
    if not isinstance(policy, dict):
        raise A4LMOError("POLICY_OBJECT_REQUIRED")
    _validate_policy(policy)
    source_config = common._resolve(str(policy["source_factorial_config"]))
    source = yaml.safe_load(source_config.read_text(encoding="utf-8"))
    if not isinstance(source, dict):
        raise A4LMOError("SOURCE_FACTORIAL_CONFIG_INVALID")
    arms = source.get("factorial_arms", {})
    if (
        arms.get("A0", {}).get("encoder_family") != "GROUPNORM_RESIDUAL_TINY_3D"
        or arms.get("A0", {}).get("augmentation_profile") != "NONE"
        or arms.get("A4", {}).get("encoder_family") != "RESNET18_3D_GROUPNORM"
        or arms.get("A4", {}).get("augmentation_profile") != "PET_SAFE_V1"
    ):
        raise A4LMOError("A0_A4_POLICY_DEFINITION_DRIFT")
    root = common._resolve(str(policy["output_root"])).resolve()
    if root.drive.upper() != "H:" or not root.is_relative_to(ROOT.resolve()):
        raise A4LMOError("OUTPUT_MUST_BE_UNDER_H_PROJECT")
    if shutil.disk_usage(root.anchor).free < int(policy["minimum_h_free_gib"]) * 1024**3:
        raise A4LMOError("H_MINIMUM_FREE_SPACE_NOT_MET")
    config = copy.deepcopy(source)
    config["output_root"] = str(root)
    common._set_h_runtime(root, int(config["seed"]), str(config["runtime"].get("cublas_workspace_config", ":4096:8")))
    config = bind_software_runtime_contract_v4(config, ROOT)
    v4_root = common._resolve(str(source["output_root"]))
    paths = {
        "policy": policy_path,
        "source_config": source_config,
        "source_v4_run_contract": v4_root / "run_contract.json",
        "source_v4_orchestration_state": v4_root / "orchestration_state.json",
        "launcher": Path(__file__).resolve(),
        "legacy_helper_source": Path(common.__file__).resolve(),
        "release_privacy_guard": ROOT / "scripts/audit_method_y_amyloid_lmo_release_v1.py",
        "lmo_split_source": ROOT / "src/method_y_revival/image_lmo_v1.py",
        "membership": common._resolve(config["branches"]["amyloid"]["membership"]),
        "cache": common._resolve(config["branches"]["amyloid"]["pattern_cache"]),
        "mask": common._resolve(config["shared"]["foreground_mask"]),
        "augmentation_qc": common._resolve(config["shared"]["augmentation_qc_contract"]),
        "template": common._resolve(config["shared"]["reference_template"]),
    }
    for name, path in paths.items():
        if not path.is_file():
            raise A4LMOError(f"INPUT_MISSING:{name}")
    fingerprints = {name: common._sha(path) for name, path in paths.items()}
    v4_state = read_json(paths["source_v4_orchestration_state"])
    if (
        v4_state.get("stage") != "FINALIZED"
        or v4_state.get("outer_cells_finalized") != 50
        or v4_state.get("inner_only_selected_arms", {}).get("amyloid") != "A4"
    ):
        raise A4LMOError("SOURCE_V4_A4_WINNER_NOT_VERIFIED")
    qc = read_json(paths["augmentation_qc"])
    if qc.get("status") != "PASS" or qc.get("contract_sha256") != config["shared"]["augmentation_qc_expected_contract_sha256"]:
        raise A4LMOError("AUGMENTATION_QC_NOT_LOCKED")
    membership = pd.read_parquet(paths["membership"])
    if set(membership["tracer"].dropna().astype(str)) != {"FBP"}:
        raise A4LMOError("NON_FBP_IN_AMYLOID_MEMBERSHIP")
    development = membership[membership["role"] == "development"].copy()
    if set(development["cohort"].astype(str)) != {"A4"}:
        raise A4LMOError("NON_A4_DEVELOPMENT_SUBJECT")
    plan = build_lmo_plan(
        development,
        manufacturer_order=list(policy["manufacturer_order"]),
        minimum_subjects=int(policy["minimum_subjects_per_heldout_manufacturer"]),
        minimum_events=int(policy["minimum_events_per_heldout_manufacturer"]),
        inner_folds=int(policy["inner_folds"]),
        seed=int(config["seed"]),
    )
    if list(policy["requested_horizons_days"]) != list(config["shared"]["requested_horizons_days"]):
        raise A4LMOError("HORIZON_POLICY_MISMATCH")
    contract_body = {
        "schema_version": "2.0",
        "analysis_role": policy["analysis_role"],
        "policy": policy,
        "input_sha256": fingerprints,
        "plan_sha256": plan["plan_sha256"],
        "metric_support_policy": "GRID_COMPUTED_ONLY_INSIDE_LOCKED_OUTER_EVALUATION_AND_MATCHED_WITHIN_MANUFACTURER",
        "software_runtime_contract": binding_core(config["software_runtime_contract"]),
        "active_arms": ["A0", "A4"],
        "model_features": list(config["branches"]["amyloid"]["model_features"]),
        "no_reuse_of_v4_or_v1_winners_or_checkpoints": True,
    }
    return {
        "policy": policy,
        "config": config,
        "development": development,
        "plan": plan,
        "qc": qc,
        "paths": paths,
        "input_sha256": fingerprints,
        "output_root": root,
        "contract": {**contract_body, "digest": sha256_mapping(contract_body)},
        "public_preflight": {
            "status": "READY" if torch.cuda.is_available() else "GPU_UNAVAILABLE",
            "study": public_plan_summary(plan),
            "active_arms": ["A0", "A4"],
            "outer_cells_required": 6,
            "metric_support_policy": contract_body["metric_support_policy"],
            "h_free_gib": round(shutil.disk_usage(root.anchor).free / 1024**3, 2),
            "claim_status": "POST_HOC_INTERNAL_EXPLORATORY",
            "positive_claim_allowed": False,
        },
    }


def _make_public_summary(cells: list[dict[str, Any]], policy: dict[str, Any]) -> dict[str, Any]:
    if len(cells) != 6:
        raise A4LMOError("SIX_OUTER_CELLS_REQUIRED")
    comparisons = []
    for manufacturer in policy["manufacturer_order"]:
        pair = {cell["arm"]: cell for cell in cells if cell["manufacturer"] == manufacturer}
        if set(pair) != {"A0", "A4"}:
            raise A4LMOError("A0_A4_PAIR_MISSING")
        if pair["A0"]["n_test"] != pair["A4"]["n_test"] or pair["A0"]["test_events"] != pair["A4"]["test_events"]:
            raise A4LMOError("A0_A4_TEST_SUPPORT_MISMATCH")
        if pair["A0"]["evaluation_grid_days"] != pair["A4"]["evaluation_grid_days"]:
            raise A4LMOError("A0_A4_IBS_GRID_MISMATCH")
        comparisons.append({
            "manufacturer": manufacturer,
            "a4_minus_a0_ibs": pair["A4"]["integrated_brier_score"] - pair["A0"]["integrated_brier_score"],
            "a4_minus_a0_uno": pair["A4"]["uno_c_index"] - pair["A0"]["uno_c_index"],
        })
    return {
        "schema_version": "2.0",
        "status": "PASS_TECHNICAL_LOCAL_LMO_ONLY",
        "analysis_role": "POST_HOC_INTERNAL_SCANNER_TRANSPORT_EXPLORATORY",
        "selected_v4_arm": "A4",
        "architecture_only_ablation": False,
        "outer_cells_finalized": 6,
        "cells": cells,
        "comparisons": comparisons,
        "claim_status": "POST_HOC_INTERNAL_EXPLORATORY",
        "positive_claim_allowed": False,
        "external_validation_status": "EXTERNAL_PENDING",
        "source_space_status": "UNPROVEN_FOR_FBP_INPUTS",
        "quantitative_equivalence_status": "BLOCKED",
        "limitations": [
            "A4 and A0 differ in encoder and augmentation; this is a policy comparison, not an architecture-only effect.",
            "Only three local manufacturers; no independent cohort.",
            "Manufacturer-specific IBS grids differ and must not be averaged as identical estimands.",
            "FBP source-space and quantitative provenance remain blocked.",
            "The study was planned after observing the original v4 outputs and is exploratory.",
        ],
    }


def _validate_outer_cell_membership(metrics: dict[str, Any], outer: dict[str, Any]) -> None:
    expected = set(map(str, outer["test_subjects"]))
    for field in ("oof_predictions", "oof_embeddings"):
        rows = metrics.get(field)
        if not isinstance(rows, list) or len(rows) != len(expected):
            raise A4LMOError(f"HELDOUT_MEMBERSHIP_COUNT_INVALID:{field}")
        keys = [str(row.get("patient_hash", "")) for row in rows if isinstance(row, dict)]
        if len(keys) != len(expected) or len(set(keys)) != len(keys) or set(keys) != expected:
            raise A4LMOError(f"HELDOUT_MEMBERSHIP_MISMATCH:{field}")
    if metrics.get("evaluation_source") != "LOCKED_V4_CHECKPOINT_WEIGHTS_UNAUGMENTED_OUTER_TEST":
        raise A4LMOError("HELDOUT_EVALUATION_SOURCE_INVALID")
    if metrics.get("outer_test_augmentation_applied") is not False:
        raise A4LMOError("HELDOUT_TEST_AUGMENTATION_FORBIDDEN")


def _verify(preflight: dict[str, Any], *, require_finalized: bool = True) -> dict[str, Any]:
    from src.method_y_revival.atomic_io import read_json
    from src.method_y_revival.image_factorial_orchestrator_v4 import ImageFactorialOrchestratorV4
    from src.method_y_revival.image_training_orchestrator_v3 import read_verified_private_outer_record
    from src.method_y_revival.outer_evaluation_control import OuterFoldController

    root = preflight["output_root"]
    contract_path = root / "run_contract.json"
    plan_path = root / "private" / "lmo_plan.json"
    if not contract_path.is_file() or read_json(contract_path) != preflight["contract"]:
        raise A4LMOError("RUN_CONTRACT_MISMATCH_OR_MISSING")
    if not plan_path.is_file() or read_json(plan_path) != preflight["plan"]:
        raise A4LMOError("SPLIT_PLAN_MISMATCH_OR_MISSING")
    if not (root / "orchestration" / "amyloid" / "master_contract.json").is_file():
        raise A4LMOError("ORCHESTRATOR_MASTER_CONTRACT_MISSING")
    state_path = root / "orchestration_state.json"
    if require_finalized:
        state = read_json(state_path)
        if (
            state.get("stage") != "FINALIZED"
            or state.get("outer_cells_finalized") != 6
            or state.get("run_contract_sha256") != preflight["contract"]["digest"]
            or state.get("positive_claim_allowed") is not False
        ):
            raise A4LMOError("FINAL_ORCHESTRATION_STATE_INVALID")
    config = preflight["config"]
    orchestrator = ImageFactorialOrchestratorV4(
        config=config, output_root=root, branch="amyloid",
        membership_sha256=preflight["input_sha256"]["membership"],
        cache_sha256=preflight["input_sha256"]["cache"],
        mask_sha256=preflight["input_sha256"]["mask"],
        split_sha256=preflight["plan"]["plan_sha256"],
        augmentation_qc_sha256=preflight["qc"]["contract_sha256"],
        h5_cache_path=preflight["paths"]["cache"],
        mask_path=preflight["paths"]["mask"],
        augmentation_qc=preflight["qc"],
    )
    cells: list[dict[str, Any]] = []
    patient_keys: set[str] = set()
    for outer in preflight["plan"]["outer_folds"]:
        fold = int(outer["fold"])
        patient_keys.update(str(key) for key in outer["test_subjects"])
        for arm in ("A0", "A4"):
            common._verify_winner_checkpoints(orchestrator, arm, fold)
            checkpoint = root / "development" / "amyloid" / arm / f"outer_{fold}" / "final" / "locked_checkpoint.pt"
            cell_root = root / "private" / "outer_evaluation" / "amyloid" / arm
            cell_state_path = cell_root / f"fold_{fold}" / "state.json"
            if not checkpoint.is_file() or not cell_state_path.is_file():
                raise A4LMOError(f"OUTER_CELL_MISSING:{arm}:{fold}")
            cell_state = read_json(cell_state_path)
            controller = OuterFoldController(cell_root, fold=fold, run_contract_sha256=cell_state["run_contract_sha256"])
            locked = controller._load_state()
            checkpoint_sha = common._sha(checkpoint)
            if locked.get("state") != "FINALIZED" or locked.get("checkpoint_sha256") != checkpoint_sha:
                raise A4LMOError(f"OUTER_CELL_NOT_FINALIZED:{arm}:{fold}")
            journal = controller._load_access_journal()
            result = controller._load_result(checkpoint_sha)
            if journal is None or result is None or journal.get("checkpoint_sha256") != checkpoint_sha:
                raise A4LMOError(f"OUTER_ACCESS_OR_RESULT_INVALID:{arm}:{fold}")
            metrics = result["metrics"]
            private_path = cell_root / f"fold_{fold}_private.json"
            private_record = read_verified_private_outer_record(
                private_path, expected_branch="amyloid", expected_variant=arm,
                expected_outer_fold=fold,
                expected_run_contract_sha256=cell_state["run_contract_sha256"],
            )
            if private_record["metrics"] != metrics:
                raise A4LMOError("PRIVATE_OUTER_RESULT_MISMATCH")
            _validate_outer_cell_membership(metrics, outer)
            uno = float(metrics["uno_c_index"])
            ibs = float(metrics["integrated_brier_score"])
            if not math.isfinite(uno) or not math.isfinite(ibs) or not 0 <= ibs <= 1:
                raise A4LMOError(f"OUTER_METRIC_INVALID:{arm}:{fold}")
            if (int(metrics["n_test"]), int(metrics["test_events"])) != (int(outer["test_count"]), int(outer["test_events"])):
                raise A4LMOError(f"OUTER_SUPPORT_MISMATCH:{arm}:{fold}")
            for horizon in preflight["policy"]["requested_horizons_days"]:
                item = metrics.get("horizon_metrics", {}).get(str(float(horizon)))
                if not isinstance(item, dict) or item.get("status") not in {"ESTIMABLE", "NOT_ESTIMABLE"}:
                    raise A4LMOError(f"HORIZON_STATUS_MISSING:{arm}:{fold}:{horizon}")
            public = read_json(root / "reports" / f"amyloid_{arm}_fold_{fold}_public.json")
            aggregate = {key: metrics[key] for key in PUBLIC_METRIC_KEYS if key in metrics}
            aggregate["outer_test_augmentation_applied"] = False
            if public != {
                "schema_version": "4.0", "aggregate": aggregate,
                "run_contract_sha256": cell_state["run_contract_sha256"],
                "claim_status": "POST_HOC_INTERNAL_EXPLORATORY",
            }:
                raise A4LMOError(f"PUBLIC_OUTER_REPORT_MISMATCH:{arm}:{fold}")
            cells.append({
                "manufacturer": outer["heldout_manufacturer"], "arm": arm,
                "n_test": int(metrics["n_test"]), "test_events": int(metrics["test_events"]),
                "uno_c_index": uno, "integrated_brier_score": ibs,
                "horizon_metrics": metrics["horizon_metrics"],
                "evaluation_grid_days": metrics["evaluation_grid_days"],
                "checkpoint_sha256": checkpoint_sha,
            })
    report = _make_public_summary(cells, preflight["policy"])
    for public_path in (root / "reports").rglob("*.json"):
        _check_public(read_json(public_path), patient_keys)
    _check_public(report, patient_keys)
    if require_finalized:
        summary_path = root / "reports" / "amyloid_lmo_a4_public_summary.json"
        if not summary_path.is_file() or read_json(summary_path) != report:
            raise A4LMOError("STORED_PUBLIC_SUMMARY_MISMATCH")
    return report


def _run(preflight: dict[str, Any]) -> dict[str, Any]:
    import torch

    from src.method_y_revival.atomic_io import atomic_write_json, read_json
    from src.method_y_revival.image_factorial_orchestrator_v4 import ImageFactorialOrchestratorV4
    from src.method_y_revival.image_outer_eval_v4 import evaluate_locked_outer_v4
    from src.method_y_revival.process_guard import WriterLock

    if not torch.cuda.is_available():
        raise A4LMOError("CUDA_REQUIRED")
    common._assert_no_competing_gpu_writer()
    root = preflight["output_root"]
    with WriterLock(root / ".amyloid_lmo_a4_v2.writer.lock", output_root=root, run_contract_digest=preflight["contract"]["digest"]):
        common._require_same_json(root / "run_contract.json", preflight["contract"])
        common._require_same_json(root / "private" / "lmo_plan.json", preflight["plan"])
        state_path = root / "orchestration_state.json"
        if state_path.is_file():
            state = read_json(state_path)
            if state.get("run_contract_sha256") != preflight["contract"]["digest"]:
                raise A4LMOError("ORCHESTRATION_STATE_CONTRACT_MISMATCH")
            if state.get("stage") == "FINALIZED":
                return _verify(preflight)
        config = preflight["config"]
        orchestrator = ImageFactorialOrchestratorV4(
            config=config, output_root=root, branch="amyloid",
            membership_sha256=preflight["input_sha256"]["membership"],
            cache_sha256=preflight["input_sha256"]["cache"],
            mask_sha256=preflight["input_sha256"]["mask"],
            split_sha256=preflight["plan"]["plan_sha256"],
            augmentation_qc_sha256=preflight["qc"]["contract_sha256"],
            h5_cache_path=preflight["paths"]["cache"],
            mask_path=preflight["paths"]["mask"], augmentation_qc=preflight["qc"],
        )
        orchestrator.verify_inputs()
        development = preflight["development"]
        features = config["branches"]["amyloid"]["model_features"]
        edges = config["shared"]["hazard_bin_edges_days"]
        winners: dict[tuple[int, str], dict[str, Any]] = {}
        atomic_write_json(state_path, {
            "stage": "INNER_SELECTION_RUNNING",
            "run_contract_sha256": preflight["contract"]["digest"],
            "claim_status": "POST_HOC_INTERNAL_EXPLORATORY",
        })
        for outer in preflight["plan"]["outer_folds"]:
            fold = int(outer["fold"])
            train = development[development["patient_hash"].astype(str).isin(outer["train_subjects"])].copy()
            for arm in ("A0", "A4"):
                if orchestrator.winner_path(arm, fold).is_file():
                    winner = common._verify_winner_checkpoints(orchestrator, arm, fold)
                    event = "INNER_WINNER_RECOVERED"
                else:
                    winner = orchestrator.run_inner_hpo(
                        outer_fold=fold, arm_id=arm, outer_train_frame=train,
                        inner_splits=outer["inner_folds"], tabular_columns=features,
                        hazard_bin_edges=edges, device="cuda",
                    )["winner"]
                    common._verify_winner_checkpoints(orchestrator, arm, fold)
                    event = "INNER_WINNER_LOCKED"
                winners[(fold, arm)] = winner
                print(json.dumps({"event": event, "manufacturer": outer["heldout_manufacturer"], "arm": arm}, sort_keys=True), flush=True)
        atomic_write_json(state_path, {
            "stage": "ALL_INNER_WINNERS_LOCKED", "inner_winners": 6,
            "run_contract_sha256": preflight["contract"]["digest"],
            "claim_status": "POST_HOC_INTERNAL_EXPLORATORY",
        })
        for outer in preflight["plan"]["outer_folds"]:
            fold = int(outer["fold"])
            train = development[development["patient_hash"].astype(str).isin(outer["train_subjects"])].copy()
            heldout_keys = frozenset(outer["test_subjects"])
            for arm in ("A0", "A4"):
                final = orchestrator.run_final_training(
                    outer_fold=fold, arm_id=arm, outer_train_frame=train,
                    tabular_columns=features, hazard_bin_edges=edges,
                    winner=winners[(fold, arm)], device="cuda",
                )

                def evaluate_after_authorization() -> dict[str, Any]:
                    test = development[development["patient_hash"].astype(str).isin(heldout_keys)].copy()
                    return evaluate_locked_outer_v4(
                        locked_checkpoint_path=final["locked_checkpoint_path"],
                        run_contract_digest=final["run_contract_sha256"],
                        train_frame=train, test_frame=test,
                        h5_cache_path=preflight["paths"]["cache"],
                        tabular_columns=features, arm_id=arm,
                        hazard_bin_edges=edges,
                        model_config=final["resolved_model_config"],
                        device="cuda", amp=bool(config["model"]["amp"]),
                        batch_size=int(config["model"]["batch_size"]),
                        requested_horizons_days=preflight["policy"]["requested_horizons_days"],
                    )

                metrics = orchestrator.outer_evaluate(
                    outer_fold=fold, arm_id=arm,
                    run_contract_sha256=final["run_contract_sha256"],
                    evaluator=evaluate_after_authorization,
                )
                print(json.dumps({
                    "event": "HELDOUT_CELL_COMMITTED_OR_RECOVERED",
                    "manufacturer": outer["heldout_manufacturer"], "arm": arm,
                    "fold": fold, "ibs": metrics["integrated_brier_score"],
                    "uno": metrics["uno_c_index"],
                }, sort_keys=True), flush=True)
        report = _verify(preflight, require_finalized=False)
        atomic_write_json(root / "reports" / "amyloid_lmo_a4_public_summary.json", report)
        atomic_write_json(state_path, {
            "stage": "FINALIZED", "outer_cells_finalized": 6,
            "run_contract_sha256": preflight["contract"]["digest"],
            "claim_status": "POST_HOC_INTERNAL_EXPLORATORY",
            "positive_claim_allowed": False,
        })
        return _verify(preflight)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, default=ROOT / "configs/method_y_amyloid_lmo_a4_v2.yaml")
    parser.add_argument("--mode", choices=("preflight", "run", "verify"), default="preflight")
    args = parser.parse_args(argv)
    try:
        preflight = _preflight(args.policy)
        if args.mode == "preflight":
            print(json.dumps(preflight["public_preflight"], indent=2, sort_keys=True), flush=True)
            return 0 if preflight["public_preflight"]["status"] == "READY" else 2
        report = _verify(preflight) if args.mode == "verify" else _run(preflight)
        print(json.dumps(report, indent=2, sort_keys=True), flush=True)
        return 0
    except KeyboardInterrupt:
        print(json.dumps({"status": "INTERRUPTED", "resume": "RERUN_SAME_COMMAND"}), flush=True)
        return 130
    except Exception as exc:
        print(json.dumps({"status": "BLOCKED", "reason": f"{type(exc).__name__}:{exc}"}, indent=2), flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
