"""Resumable, versioned A4 amyloid manufacturer-held-out sensitivity study.

This post-hoc internal analysis reuses frozen image inputs, not v4 winners or
checkpoints.  It does not establish quantitative PET or external validity.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class LMOLauncherError(RuntimeError):
    pass


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _set_h_runtime(output_root: Path, seed: int, cublas: str) -> None:
    temp = output_root / "tmp"
    pycache = output_root / "runtime" / "pycache"
    temp.mkdir(parents=True, exist_ok=True)
    pycache.mkdir(parents=True, exist_ok=True)
    os.environ["TEMP"] = str(temp)
    os.environ["TMP"] = str(temp)
    os.environ["PYTHONPYCACHEPREFIX"] = str(pycache)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = cublas
    os.environ["PYTHONHASHSEED"] = str(seed)


def _preflight(policy_path: Path) -> dict[str, Any]:
    import pandas as pd
    import torch
    import yaml

    from src.method_y_revival.contracts import sha256_mapping
    from src.method_y_revival.image_lmo_v1 import build_lmo_plan, public_plan_summary
    from src.method_y_revival.software_contract_v4 import (
        bind_software_runtime_contract_v4,
        binding_core,
    )

    policy = yaml.safe_load(policy_path.read_text(encoding="utf-8"))
    if not isinstance(policy, dict) or policy.get("schema_version") != "1.0":
        raise LMOLauncherError("LMO_POLICY_INVALID")
    if policy.get("branch") != "amyloid" or policy.get("tracer") != "FBP":
        raise LMOLauncherError("FBP_AMYLOID_ONLY")
    if policy.get("active_arms") != ["A0", "A2"]:
        raise LMOLauncherError("LOCKED_A0_A2_PAIR_REQUIRED")
    if policy.get("positive_claim_allowed") is not False:
        raise LMOLauncherError("POSITIVE_CLAIM_FORBIDDEN")
    source_config_path = _resolve(str(policy["source_factorial_config"]))
    source = yaml.safe_load(source_config_path.read_text(encoding="utf-8"))
    output_root = _resolve(str(policy["output_root"]))
    if output_root.drive.upper() != "H:" or not str(output_root.resolve()).lower().startswith(str(ROOT.resolve()).lower()):
        raise LMOLauncherError("LMO_OUTPUT_MUST_BE_UNDER_H_PROJECT")
    config = copy.deepcopy(source)
    config["output_root"] = str(output_root)
    _set_h_runtime(
        output_root,
        int(config["seed"]),
        str(config["runtime"].get("cublas_workspace_config", ":4096:8")),
    )
    config = bind_software_runtime_contract_v4(config, ROOT)
    membership_path = _resolve(config["branches"]["amyloid"]["membership"])
    cache_path = _resolve(config["branches"]["amyloid"]["pattern_cache"])
    mask_path = _resolve(config["shared"]["foreground_mask"])
    qc_path = _resolve(config["shared"]["augmentation_qc_contract"])
    template_path = _resolve(config["shared"]["reference_template"])
    source_run_contract_path = _resolve(source["output_root"]) / "run_contract.json"
    paths = {
        "policy": policy_path,
        "source_config": source_config_path,
        "source_v4_run_contract": source_run_contract_path,
        "launcher": Path(__file__).resolve(),
        "lmo_split_source": ROOT / "src/method_y_revival/image_lmo_v1.py",
        "membership": membership_path,
        "cache": cache_path,
        "mask": mask_path,
        "augmentation_qc": qc_path,
        "template": template_path,
    }
    for name, path in paths.items():
        if not path.is_file():
            raise LMOLauncherError(f"INPUT_MISSING:{name}:{path}")
    fingerprints = {name: _sha(path) for name, path in paths.items()}
    from src.method_y_revival.atomic_io import read_json

    qc = read_json(qc_path)
    if qc.get("status") != "PASS" or qc.get("contract_sha256") != config["shared"]["augmentation_qc_expected_contract_sha256"]:
        raise LMOLauncherError("AUGMENTATION_QC_NOT_LOCKED")
    membership = pd.read_parquet(membership_path)
    if set(membership["tracer"].dropna().astype(str)) != {"FBP"}:
        raise LMOLauncherError("NON_FBP_IN_AMYLOID_MEMBERSHIP")
    development = membership[membership["role"] == "development"].copy()
    if set(development["cohort"].astype(str)) != {"A4"}:
        raise LMOLauncherError("NON_A4_DEVELOPMENT_SUBJECT")
    plan = build_lmo_plan(
        development,
        manufacturer_order=[str(x) for x in policy["manufacturer_order"]],
        minimum_subjects=int(policy["minimum_subjects_per_heldout_manufacturer"]),
        minimum_events=int(policy["minimum_events_per_heldout_manufacturer"]),
        inner_folds=int(policy["inner_folds"]),
        seed=int(config["seed"]),
    )
    from src.method_y_revival.image_outer_eval_v3 import _evaluation_grid

    metric_support: dict[str, Any] = {}
    for outer in plan["outer_folds"]:
        train = development[development["patient_hash"].astype(str).isin(outer["train_subjects"])]
        test = development[development["patient_hash"].astype(str).isin(outer["test_subjects"])]
        try:
            grid = _evaluation_grid(
                train["time_days"].to_numpy(dtype=float),
                test["time_days"].to_numpy(dtype=float),
                config["shared"]["hazard_bin_edges_days"],
            )
        except Exception as exc:
            raise LMOLauncherError(
                f"PRESPECIFIED_METRIC_SUPPORT_NOT_ESTIMABLE:{outer['heldout_manufacturer']}:{type(exc).__name__}"
            ) from exc
        metric_support[outer["heldout_manufacturer"]] = [float(value) for value in grid]
    if list(policy["requested_horizons_days"]) != list(config["shared"]["requested_horizons_days"]):
        raise LMOLauncherError("HORIZON_POLICY_MISMATCH")
    contract_body = {
        "schema_version": "1.0",
        "analysis_role": policy["analysis_role"],
        "policy": policy,
        "input_sha256": fingerprints,
        "plan_sha256": plan["plan_sha256"],
        "prespecified_ibs_grids_by_manufacturer": metric_support,
        "software_runtime_contract": binding_core(config["software_runtime_contract"]),
        "active_arms": list(policy["active_arms"]),
        "model_features": list(config["branches"]["amyloid"]["model_features"]),
        "no_reuse_of_v4_winners_or_checkpoints": True,
    }
    return {
        "policy": policy,
        "config": config,
        "development": development,
        "plan": plan,
        "qc": qc,
        "paths": paths,
        "input_sha256": fingerprints,
        "output_root": output_root,
        "contract": {**contract_body, "digest": sha256_mapping(contract_body)},
        "public_preflight": {
            "status": "READY" if torch.cuda.is_available() else "GPU_UNAVAILABLE",
            "study": public_plan_summary(plan),
            "active_arms": list(policy["active_arms"]),
            "outer_cells_required": len(plan["outer_folds"]) * len(policy["active_arms"]),
            "prespecified_ibs_grids_by_manufacturer": metric_support,
            "cuda_available": bool(torch.cuda.is_available()),
            "claim_status": policy["claim_status"],
            "positive_claim_allowed": False,
        },
    }


def _require_same_json(path: Path, expected: dict[str, Any]) -> None:
    from src.method_y_revival.atomic_io import atomic_write_json, read_json

    if path.is_file():
        if read_json(path) != expected:
            raise LMOLauncherError(f"RUN_CONTRACT_MISMATCH:{path.name}")
    else:
        atomic_write_json(path, expected)


def _verify_winner_checkpoints(orchestrator: Any, arm: str, fold: int) -> dict[str, Any]:
    from src.method_y_revival.image_training_orchestrator_v3 import read_verified_winner_record

    record = read_verified_winner_record(
        orchestrator.winner_path(arm, fold),
        expected_branch="amyloid",
        expected_variant=arm,
        expected_outer_fold=fold,
        expected_master_contract_sha256=orchestrator.master_contract_sha256,
    )
    if len(record["candidates"]) != 3:
        raise LMOLauncherError("INNER_HPO_CANDIDATE_COUNT_INVALID")
    for candidate in record["candidates"]:
        if candidate.get("status") != "COMPLETED" or len(candidate.get("inner_fold_metric_records", [])) != 3:
            raise LMOLauncherError("INNER_HPO_INCOMPLETE")
        for metric in candidate["inner_fold_metric_records"]:
            value = float(metric["metric_value"])
            if metric.get("metric_name") != "IPCW_INTEGRATED_BRIER_SCORE" or not 0 <= value <= 1:
                raise LMOLauncherError("INNER_HPO_METRIC_INVALID")
            path = (
                orchestrator.output_root / "development" / "amyloid" / arm
                / f"outer_{fold}" / "inner" / str(candidate["candidate_id"])
                / f"fold_{int(metric['inner_fold'])}" / "locked_checkpoint.pt"
            )
            if not path.is_file() or _sha(path) != metric["locked_checkpoint_sha256"]:
                raise LMOLauncherError("INNER_HPO_CHECKPOINT_HASH_MISMATCH")
    return record["winner"]


def _assert_no_competing_gpu_writer() -> None:
    import subprocess

    # The project environment does not require psutil.  Query live Win32
    # process command lines read-only; exclude our own parent launch chain.
    command = (
        "Get-CimInstance Win32_Process | "
        "Where-Object { $_.Name -match 'python|powershell|matlab' } | "
        "Select-Object ProcessId,ParentProcessId,CommandLine | ConvertTo-Json -Compress"
    )
    helper = subprocess.Popen(
        ["powershell", "-NoProfile", "-Command", command],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    stdout, stderr = helper.communicate(timeout=30)
    if helper.returncode != 0:
        raise LMOLauncherError(f"PROCESS_GUARD_QUERY_FAILED:{stderr.strip()[:200]}")
    records = json.loads(stdout) if stdout.strip() else []
    if isinstance(records, dict):
        records = [records]
    by_pid = {int(row["ProcessId"]): row for row in records}
    own_lineage = {os.getpid(), helper.pid}
    current = os.getpid()
    while current in by_pid:
        parent = int(by_pid[current].get("ParentProcessId") or 0)
        if parent <= 0 or parent in own_lineage:
            break
        own_lineage.add(parent)
        current = parent
    for pid, row in by_pid.items():
        if pid in own_lineage:
            continue
        line = str(row.get("CommandLine") or "").lower().replace("/", "\\")
        if "h:\\new project" not in line:
            continue
        if any(marker in line for marker in (
            "run_method_x_", "run_mx04", "run_mx05",
            "run_method_y_image_", "run_method_y_amyloid_lmo",
        )):
            raise LMOLauncherError(f"COMPETING_PROJECT_GPU_WRITER:pid={pid}")


def _verify(preflight: dict[str, Any]) -> dict[str, Any]:
    import math

    from src.method_y_revival.atomic_io import read_json
    from src.method_y_revival.image_factorial_orchestrator_v4 import ImageFactorialOrchestratorV4
    from src.method_y_revival.image_training_orchestrator_v3 import read_verified_private_outer_record
    from src.method_y_revival.outer_evaluation_control import OuterFoldController

    root = preflight["output_root"]
    if not (root / "run_contract.json").is_file() or read_json(root / "run_contract.json") != preflight["contract"]:
        raise LMOLauncherError("RUN_CONTRACT_MISMATCH_OR_MISSING")
    if not (root / "private" / "lmo_plan.json").is_file() or read_json(root / "private" / "lmo_plan.json") != preflight["plan"]:
        raise LMOLauncherError("SPLIT_PLAN_MISMATCH_OR_MISSING")
    if not (root / "orchestration" / "amyloid" / "master_contract.json").is_file():
        raise LMOLauncherError("ORCHESTRATOR_MASTER_CONTRACT_MISSING")
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
    cells: list[dict[str, Any]] = []
    forbidden = {
        "patient_hash", "source_subject_id", "raw_path", "image_path_private",
        "event", "time_days", "original_event", "original_time_days",
        "oof_predictions", "oof_embeddings", "censoring_reference", "embedding",
    }

    def scan_public(value: Any) -> None:
        if isinstance(value, dict):
            if forbidden & set(value):
                raise LMOLauncherError("PUBLIC_ROW_LEVEL_LEAK")
            for child in value.values():
                scan_public(child)
        elif isinstance(value, list):
            for child in value:
                scan_public(child)

    for outer in preflight["plan"]["outer_folds"]:
        fold = int(outer["fold"])
        for arm in preflight["policy"]["active_arms"]:
            _verify_winner_checkpoints(orchestrator, arm, fold)
            final = root / "development" / "amyloid" / arm / f"outer_{fold}" / "final" / "locked_checkpoint.pt"
            controller_root = root / "private" / "outer_evaluation" / "amyloid" / arm
            state_path = controller_root / f"fold_{fold}" / "state.json"
            if not final.is_file() or not state_path.is_file():
                raise LMOLauncherError(f"OUTER_CELL_MISSING:{arm}:{fold}")
            state = read_json(state_path)
            controller = OuterFoldController(controller_root, fold=fold, run_contract_sha256=state["run_contract_sha256"])
            locked = controller._load_state()
            if locked.get("state") != "FINALIZED" or locked.get("checkpoint_sha256") != _sha(final):
                raise LMOLauncherError(f"OUTER_CELL_NOT_FINALIZED:{arm}:{fold}")
            journal = controller._load_access_journal()
            result = controller._load_result(locked["checkpoint_sha256"])
            if journal is None or result is None:
                raise LMOLauncherError(f"OUTER_ACCESS_OR_RESULT_INVALID:{arm}:{fold}")
            metrics = result["metrics"]
            private_path = root / "private" / "outer_evaluation" / "amyloid" / arm / f"fold_{fold}_private.json"
            private_record = read_verified_private_outer_record(
                private_path,
                expected_branch="amyloid", expected_variant=arm,
                expected_outer_fold=fold,
                expected_run_contract_sha256=state["run_contract_sha256"],
            )
            if private_record["metrics"] != metrics:
                raise LMOLauncherError("PRIVATE_OUTER_RESULT_MISMATCH")
            uno = float(metrics["uno_c_index"])
            ibs = float(metrics["integrated_brier_score"])
            if not math.isfinite(uno) or not math.isfinite(ibs) or not 0 <= ibs <= 1:
                raise LMOLauncherError(f"OUTER_METRIC_INVALID:{arm}:{fold}")
            if int(metrics["n_test"]) != int(outer["test_count"]) or int(metrics["test_events"]) != int(outer["test_events"]):
                raise LMOLauncherError(f"OUTER_SUPPORT_MISMATCH:{arm}:{fold}")
            public_path = root / "reports" / f"amyloid_{arm}_fold_{fold}_public.json"
            if not public_path.is_file():
                raise LMOLauncherError(f"PUBLIC_OUTER_REPORT_MISSING:{arm}:{fold}")
            public = read_json(public_path)
            if set(public) != {"schema_version", "aggregate", "run_contract_sha256", "claim_status"}:
                raise LMOLauncherError("PUBLIC_REPORT_SCHEMA_INVALID")
            scan_public(public)
            cells.append({
                "manufacturer": outer["heldout_manufacturer"], "arm": arm,
                "n_test": int(metrics["n_test"]), "test_events": int(metrics["test_events"]),
                "uno_c_index": uno, "integrated_brier_score": ibs,
                "horizon_metrics": metrics.get("horizon_metrics", {}),
                "evaluation_grid_days": metrics.get("evaluation_grid_days", []),
                "checkpoint_sha256": locked["checkpoint_sha256"],
            })
    if len(cells) != 6:
        raise LMOLauncherError("SIX_OUTER_CELLS_REQUIRED")
    for public_path in (root / "reports").rglob("*.json"):
        scan_public(read_json(public_path))
    report = {
        "schema_version": "1.0", "status": "PASS_TECHNICAL_LOCAL_LMO_ONLY",
        "analysis_role": "POST_HOC_INTERNAL_SCANNER_TRANSPORT_EXPLORATORY",
        "outer_cells_finalized": len(cells), "cells": cells,
        "comparisons": [
            {
                "manufacturer": maker,
                "a2_minus_a0_ibs": next(c["integrated_brier_score"] for c in cells if c["manufacturer"] == maker and c["arm"] == "A2") - next(c["integrated_brier_score"] for c in cells if c["manufacturer"] == maker and c["arm"] == "A0"),
                "a2_minus_a0_uno": next(c["uno_c_index"] for c in cells if c["manufacturer"] == maker and c["arm"] == "A2") - next(c["uno_c_index"] for c in cells if c["manufacturer"] == maker and c["arm"] == "A0"),
            }
            for maker in preflight["policy"]["manufacturer_order"]
        ],
        "claim_status": "POST_HOC_INTERNAL_EXPLORATORY",
        "positive_claim_allowed": False,
        "external_validation_status": "EXTERNAL_PENDING",
        "source_space_status": "UNPROVEN_FOR_FBP_INPUTS",
        "quantitative_equivalence_status": "BLOCKED",
        "limitations": [
            "Only three local manufacturers; no independent cohort.",
            "Manufacturer-specific IBS grids differ and must not be averaged as identical estimands.",
            "FBP source-space and quantitative provenance remain blocked.",
            "The study was planned after observing the original v4 outputs and is exploratory.",
        ],
    }
    return report


def _run(preflight: dict[str, Any]) -> dict[str, Any]:
    import pandas as pd
    import torch

    from src.method_y_revival.atomic_io import atomic_write_json
    from src.method_y_revival.image_factorial_orchestrator_v4 import ImageFactorialOrchestratorV4
    from src.method_y_revival.image_outer_eval_v4 import evaluate_locked_outer_v4
    from src.method_y_revival.process_guard import WriterLock

    if not torch.cuda.is_available():
        raise LMOLauncherError("CUDA_REQUIRED")
    _assert_no_competing_gpu_writer()
    root = preflight["output_root"]
    with WriterLock(root / ".amyloid_lmo.writer.lock", output_root=root, run_contract_digest=preflight["contract"]["digest"]):
        _require_same_json(root / "run_contract.json", preflight["contract"])
        _require_same_json(root / "private" / "lmo_plan.json", preflight["plan"])
        state_path = root / "orchestration_state.json"
        if state_path.is_file():
            from src.method_y_revival.atomic_io import read_json

            state = read_json(state_path)
            if state.get("run_contract_sha256") != preflight["contract"]["digest"]:
                raise LMOLauncherError("ORCHESTRATION_STATE_CONTRACT_MISMATCH")
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
        atomic_write_json(root / "orchestration_state.json", {
            "stage": "INNER_SELECTION_RUNNING", "run_contract_sha256": preflight["contract"]["digest"],
            "claim_status": "POST_HOC_INTERNAL_EXPLORATORY",
        })
        # All six inner winners are committed before any held-out image is loaded.
        for outer in preflight["plan"]["outer_folds"]:
            fold = int(outer["fold"])
            train = development[development["patient_hash"].astype(str).isin(outer["train_subjects"])].copy()
            for arm in preflight["policy"]["active_arms"]:
                if orchestrator.winner_path(arm, fold).is_file():
                    winner = _verify_winner_checkpoints(orchestrator, arm, fold)
                    event = "INNER_WINNER_RECOVERED"
                else:
                    winner = orchestrator.run_inner_hpo(
                        outer_fold=fold, arm_id=arm, outer_train_frame=train,
                        inner_splits=outer["inner_folds"], tabular_columns=features,
                        hazard_bin_edges=edges, device="cuda",
                    )["winner"]
                    _verify_winner_checkpoints(orchestrator, arm, fold)
                    event = "INNER_WINNER_LOCKED"
                winners[(fold, arm)] = winner
                print(json.dumps({"event": event, "manufacturer": outer["heldout_manufacturer"], "arm": arm}, sort_keys=True), flush=True)
        atomic_write_json(root / "orchestration_state.json", {
            "stage": "ALL_INNER_WINNERS_LOCKED", "inner_winners": 6,
            "run_contract_sha256": preflight["contract"]["digest"],
            "claim_status": "POST_HOC_INTERNAL_EXPLORATORY",
        })
        for outer in preflight["plan"]["outer_folds"]:
            fold = int(outer["fold"])
            train = development[development["patient_hash"].astype(str).isin(outer["train_subjects"])].copy()
            heldout_keys = frozenset(outer["test_subjects"])
            for arm in preflight["policy"]["active_arms"]:
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
                        hazard_bin_edges=edges, model_config=final["resolved_model_config"],
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
                    "event": "HELDOUT_CELL_COMMITTED_OR_RECOVERED", "manufacturer": outer["heldout_manufacturer"],
                    "arm": arm, "fold": fold, "ibs": metrics["integrated_brier_score"],
                    "uno": metrics["uno_c_index"],
                }, sort_keys=True), flush=True)
        report = _verify(preflight)
        atomic_write_json(root / "reports" / "amyloid_lmo_public_summary.json", report)
        atomic_write_json(root / "orchestration_state.json", {
            "stage": "FINALIZED", "outer_cells_finalized": 6,
            "run_contract_sha256": preflight["contract"]["digest"],
            "claim_status": "POST_HOC_INTERNAL_EXPLORATORY",
            "positive_claim_allowed": False,
        })
        return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", default=str(ROOT / "configs/method_y_amyloid_lmo_v1.yaml"))
    parser.add_argument("--mode", choices=("preflight", "run", "verify"), default="preflight")
    args = parser.parse_args()
    try:
        preflight = _preflight(Path(args.policy))
        if args.mode == "preflight":
            print(json.dumps(preflight["public_preflight"], indent=2, sort_keys=True), flush=True)
            return 0 if preflight["public_preflight"]["status"] == "READY" else 2
        if args.mode == "verify":
            report = _verify(preflight)
        else:
            report = _run(preflight)
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
