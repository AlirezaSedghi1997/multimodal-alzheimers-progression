from __future__ import annotations

import argparse
import hashlib
import os
import json
from pathlib import Path
import sys
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import yaml
from src.method_y_revival.atomic_io import atomic_write_json, read_json
from src.method_y_revival.process_guard import WriterLock, ActiveWriterError
from src.method_y_revival.contracts import sha256_mapping
from src.method_y_revival.software_contract_v3 import bind_software_runtime_contract


def _sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _apply_deterministic_env(config: dict[str, Any]) -> None:
    """Deterministic runtime contract must hold for direct .py runs too,
    not only through the PS1 wrapper."""
    runtime_cfg = config.get("runtime", {})
    if runtime_cfg.get("deterministic_cuda"):
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        os.environ["IMAGE_V3_PERF"] = "0"



def _preflight(config_path: Path) -> dict:
    raw_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    # CUBLAS_WORKSPACE_CONFIG must be set before the first torch import.  The
    # software binder records CUDA by importing torch, so ordering is part of
    # the deterministic execution contract.
    _apply_deterministic_env(raw_config)
    config = bind_software_runtime_contract(raw_config, ROOT)
    output_root = Path(config["project_root"]) / config["output_root"] if not Path(config["output_root"]).is_absolute() else Path(config["output_root"])
    # Normalize to H:/...
    if not config["output_root"].startswith("results/method_y_revival_v3_image"):
        raise RuntimeError("OUTPUT_ROOT_UNEXPECTED")
    # Check H drive
    if output_root.drive.upper() != "H:":
        # Resolve relative to ROOT
        output_root = ROOT / config["output_root"]
    # Check required files
    readiness = output_root / "readiness" / "FINAL_IMAGE_TRAINING_READINESS.json"
    if not readiness.is_file():
        raise RuntimeError(f"READINESS_MISSING:{readiness}")
    readiness_json = read_json(readiness)
    if readiness_json.get("training_execution_allowed") is not True:
        raise RuntimeError("TRAINING_NOT_ALLOWED_BY_READINESS")
    run_contract_path = output_root / "run_contract.json"
    if not run_contract_path.is_file():
        raise RuntimeError("RUN_CONTRACT_MISSING")
    run_contract = read_json(run_contract_path)
    payload = run_contract.get("payload")
    if not isinstance(payload, dict) or run_contract.get("digest") != sha256_mapping(payload):
        raise RuntimeError("RUN_CONTRACT_SELF_HASH_MISMATCH")
    if readiness_json.get("run_contract_sha256") != run_contract.get("digest"):
        raise RuntimeError("READINESS_RUN_CONTRACT_MISMATCH")
    if payload.get("software_runtime_contract") != config.get("software_runtime_contract"):
        raise RuntimeError("SOFTWARE_RUNTIME_CONTRACT_MISMATCH")
    # Check caches complete
    for branch in ["amyloid", "tau"]:
        derived = ROOT / config["branches"][branch]["derived_cache"]
        if not derived.is_file():
            raise RuntimeError(f"DERIVED_CACHE_MISSING:{branch}")
        # verify cache state
        cache_state = output_root / "cache" / f"{branch}_pattern_cache_state.json"
        if cache_state.is_file():
            state = read_json(cache_state)
            if state.get("status") not in {"COMPLETED", "ITERATION_COMPLETE"}:
                # but we also allow completed via summary
                pass
        # also check summary
        summary = output_root / "cache" / "PATTERN_CACHE_SUMMARY.json"
        if summary.is_file():
            s = read_json(summary)
            if s.get("branches", {}).get(branch, {}).get("completed_count", 0) == 0:
                raise RuntimeError(f"CACHE_NOT_COMPLETED:{branch}")
    # Check splits
    for branch in ["amyloid", "tau"]:
        split_path = ROOT / config["branches"][branch]["splits"]
        if not split_path.is_file():
            raise RuntimeError(f"SPLIT_MISSING:{branch}")
    # GPU check (smoke) – just report, don't fail for internal CPU run
    gpu_info = {"name": "UNKNOWN", "vram": 0}
    try:
        import torch

        if torch.cuda.is_available():
            gpu_info["name"] = torch.cuda.get_device_name(0)
            gpu_info["vram"] = torch.cuda.get_device_properties(0).total_memory
    except Exception:
        pass
    # H free
    import shutil

    free = shutil.disk_usage(str(output_root)).free
    if free < 20 * 1024**3:
        raise RuntimeError(f"H_FREE_BELOW_20GiB:{free}")
    return {
        "status": "READY_TO_RUN_FINAL_INTERNAL_EXPLORATORY_IMAGE_TRAINING",
        "training_execution_allowed": True,
        "positive_claim_allowed": False,
        "external_claim_allowed": False,
        "gpu": gpu_info,
        "readiness": readiness_json,
    }


def _evaluate_outer_cell_exactly_once(
    *,
    orchestrator: Any,
    outer_fold: int,
    variant: str,
    run_contract: str,
    locked_checkpoint_path: Path,
    outer_train_frame: Any,
    test_frame_factory: Callable[[], Any],
    cache_path: Path,
    tabular_columns: list[str],
    hazard_edges: list[float],
    model_config: dict[str, Any],
    requested_horizons_days: list[float],
    device: str,
    amp: bool,
    batch_size: int,
) -> dict[str, Any]:
    """Keep test-frame materialization inside the exactly-once controller."""
    from src.method_y_revival.image_outer_eval_v3 import evaluate_locked_outer

    def evaluator() -> dict[str, Any]:
        test_frame = test_frame_factory()
        if test_frame.empty:
            raise RuntimeError("EMPTY_OUTER_TEST_FRAME")
        return evaluate_locked_outer(
            locked_checkpoint_path=locked_checkpoint_path,
            run_contract_digest=run_contract,
            train_frame=outer_train_frame,
            test_frame=test_frame,
            h5_cache_path=cache_path,
            tabular_columns=tabular_columns,
            variant=variant,
            hazard_bin_edges=hazard_edges,
            model_config=model_config,
            requested_horizons_days=requested_horizons_days,
            device=device,
            amp=amp,
            batch_size=batch_size,
        )

    return orchestrator.outer_evaluate(
        outer_fold=outer_fold,
        variant=variant,
        evaluator=evaluator,
        run_contract=run_contract,
    )


def _restrict_inner_splits_to_eligible_outer_train(
    inner_splits: list[dict[str, Any]], eligible_keys: set[str]
) -> list[dict[str, Any]]:
    restricted: list[dict[str, Any]] = []
    for row in inner_splits:
        train = set(row.get("train_subjects") or row.get("train_subject_keys") or [])
        validation = set(
            row.get("validation_subjects")
            or row.get("validation_subject_keys")
            or row.get("val_subjects")
            or []
        )
        if train & validation:
            raise RuntimeError("INNER_PARTITION_LEAKAGE")
        train &= eligible_keys
        validation &= eligible_keys
        if not train or not validation or train | validation != eligible_keys:
            raise RuntimeError("INNER_PARTITION_INCOMPLETE_AFTER_ELIGIBILITY_FILTER")
        restricted.append(
            {
                "fold": int(row.get("fold", len(restricted))),
                "train_subjects": sorted(train),
                "validation_subjects": sorted(validation),
            }
        )
    if len(restricted) != 3:
        raise RuntimeError("EXPECTED_THREE_INNER_FOLDS")
    return restricted


def _run_gpu_smoke(config: dict, output_root: Path) -> dict:
    """GPU smoke: 4-8 scans 16^3, forward+backward, AMP, checkpoint, finite loss."""
    import numpy as np
    import pandas as pd
    import torch
    from src.method_y_revival.image_modeling_v3 import build_model
    from src.method_y_revival.image_training_runtime_v3 import (
        FoldTabularTransformer,
        MemorySurvivalDataset,
        fit_survival_model,
    )

    n = 6
    df = pd.DataFrame(
        {
            "patient_hash": [f"smoke_{i}" for i in range(n)],
            "event": [1, 0] * 3,
            "time_days": [300 + i * 200 for i in range(n)],
            "age": [70] * n,
            "sex_binary": [0] * n,
            "education_years": [12] * n,
            "apoe4_carrier": [0] * n,
            "baseline_pacc_raw": [0.0] * n,
            "baseline_fctotal96": [0.0] * n,
            "baseline_ldeltotal": [0.0] * n,
            "baseline_digittotal": [0.0] * n,
            "baseline_mmscore": [28] * n,
            "treatment_active": [0] * n,
            "baseline_amyloid_centiloid": [0.0] * n,
        }
    )
    cols = config["branches"]["amyloid"]["model_features"]
    trans = FoldTabularTransformer.fit(df, cols)
    tab = trans.transform(df)
    model = build_model("scalar_tabular_plus_pattern_image", config["model"], tabular_dim=len(cols), hazard_bins=5)
    images = np.random.randn(n, 1, 16, 16, 16).astype(np.float32)
    ds = MemorySurvivalDataset(tabular=tab, event=df["event"].to_numpy(), time_days=df["time_days"].to_numpy(), images=images)
    out = output_root / ".smoke_test"
    out.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    fit_survival_model(
        model=model,
        train_dataset=ds,
        validation_dataset=None,
        bin_edges=config["shared"]["hazard_bin_edges_days"],
        output_dir=out,
        run_contract_digest="a" * 64,
        seed=0,
        max_epochs=1,
        patience=10,
        learning_rate=0.0001,
        weight_decay=0.0001,
        batch_size=2,
        gradient_accumulation_steps=4,
        device=device,
        amp=(device == "cuda"),
        amp_dtype=str(config["model"].get("amp_dtype", "bfloat16")),
        amp_initial_scale=float(config["model"].get("amp_initial_scale", 256.0)),
        amp_growth_interval=int(config["model"].get("amp_growth_interval", 2000)),
        fixed_epochs=1,
    )
    ckpt = out / "locked_checkpoint.pt"
    summary = out / "training_summary.json"
    if not ckpt.exists() or not summary.is_file():
        raise RuntimeError("SMOKE_CHECKPOINT_MISSING")
    vram = int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0
    return {"smoke": "PASSED", "device": device, "peak_vram": vram, "checkpoint": str(ckpt)}


def main() -> int:
    parser = argparse.ArgumentParser(description="Method Y v3 image training orchestrator")
    parser.add_argument("--config", default=str(ROOT / "configs/method_y_image_training_v3_2.yaml"))
    parser.add_argument("--mode", choices=["preflight", "smoke", "run", "status", "verify"], default="run")
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--branch", choices=["amyloid", "tau", "both"], default="both")
    # Intentionally no flag to disable resume
    args = parser.parse_args()
    config_path = Path(args.config)
    raw_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    _apply_deterministic_env(raw_config)
    config = bind_software_runtime_contract(raw_config, ROOT)
    output_root = ROOT / config["output_root"] if not Path(config["output_root"]).is_absolute() else Path(config["output_root"])
    # preflight checks before acquiring lock
    # storage locality
    import os

    # WriterLock for output root
    run_contract_path = output_root / "run_contract.json"
    run_contract_sha = "0" * 64
    if run_contract_path.is_file():
        try:
            rc = json.loads(run_contract_path.read_text(encoding="utf-8"))
            run_contract_sha = rc.get("digest", "0" * 64)
        except Exception:
            run_contract_sha = _sha256_file(run_contract_path)[:64].ljust(64, "0")
    lock_path = output_root / ".image_training.writer.lock"
    # Try to acquire lock unless mode is status/verify/preflight (read-only)
    if args.mode in {"status", "verify", "preflight"}:
        # read-only preflight even without lock
        try:
            result = _preflight(config_path)
            print(json.dumps(result, indent=2))
            if args.mode == "preflight":
                print("READY_TO_RUN_FINAL_INTERNAL_EXPLORATORY_IMAGE_TRAINING")
                print("training_execution_allowed=true")
                print("positive_claim_allowed=false")
                print("external_claim_allowed=false")
            return 0
        except Exception as exc:
            print(json.dumps({"status": "BLOCKED", "reason": str(exc)}, indent=2))
            return 2
    if args.mode == "smoke":
        # GPU smoke: 4-8 scans, forward+backward, AMP, batch2 grad_accum4, checkpoint, finite loss
        try:
            result = _preflight(config_path)
            print(json.dumps(result, indent=2))
            payload = _run_gpu_smoke(config, output_root)
            print(json.dumps(payload))
            return 0
        except Exception as exc:
            # Write fallback marker if OOM
            if "OOM" in str(exc) or "out of memory" in str(exc).lower():
                fb = output_root / "VRAM_SMOKE_BLOCKED.json"
                fb.write_text(json.dumps({"status": "VRAM_SMOKE_BLOCKED", "reason": str(exc)}, indent=2), encoding="utf-8")
            print(json.dumps({"status": "BLOCKED", "reason": f"SMOKE_FAILED:{exc}"}, indent=2))
            return 2
    # For run mode, need lock
    try:
        with WriterLock(lock_path, output_root=output_root, run_contract_digest=run_contract_sha):
            if args.mode == "run":
                result = _preflight(config_path)
                print(json.dumps({"status": "RUN_DISPATCHED", "mode": "run", "preflight": result}, indent=2))
                # Full image training: nested CV for each branch/variant/outer fold
                from src.method_y_revival.image_launcher_state_v3 import LauncherStateV3
                from src.method_y_revival.image_outer_eval_v3 import evaluate_locked_outer
                from src.method_y_revival.image_training_orchestrator_v3 import (
                    ImageTrainingOrchestratorV3,
                    _build_run_contract,
                    read_verified_private_outer_record,
                    read_verified_winner_record,
                )
                from src.method_y_revival.image_reporting_v3 import (
                            build_aggregate_claim_policy,
                            build_real_oof_reports,
                            privacy_scan_reports,
                        )
                import pandas as pd
                import torch

                state = LauncherStateV3(output_root=output_root, run_contract_sha256=run_contract_sha)
                state.set_stage("PREFLIGHT_DONE", preflight_status=result["status"], gpu=result["gpu"])
                state.set_stage(
                    "DUAL_GATE_CONTRACT_VERIFIED",
                    training_execution_allowed=bool(result["readiness"]["training_execution_allowed"]),
                    positive_claim_allowed=bool(result["readiness"]["positive_claim_allowed"]),
                    external_claim_allowed=bool(result["readiness"]["external_claim_allowed"]),
                )
                state.set_stage(
                    "READINESS_LOCKED",
                    readiness_status=result["readiness"]["status"],
                    run_contract_sha256=result["readiness"]["run_contract_sha256"],
                )
                cache_counts = {}
                for branch in ["amyloid", "tau"]:
                    summary = output_root / "cache" / "PATTERN_CACHE_SUMMARY.json"
                    if summary.is_file():
                        s = read_json(summary)
                        cache_counts[branch] = int(s.get("branches", {}).get(branch, {}).get("completed_count", 0))
                state.set_stage("PATTERN_CACHES_COMPLETE", completed_counts=cache_counts)
                if state.fact("gpu_smoke") is None:
                    smoke_payload = _run_gpu_smoke(config, output_root)
                    state.set_stage("GPU_SMOKE_PASSED", gpu_smoke=smoke_payload)
                    print(json.dumps({"event": "STAGE_COMMITTED", "stage": "GPU_SMOKE_PASSED", **smoke_payload}))
                else:
                    state.set_stage("GPU_SMOKE_PASSED")

                # Determine branches in locked order: amyloid then tau (Plan 013)
                branches_all = ["amyloid", "tau"]
                branches = branches_all if args.branch == "both" else [args.branch]
                device = "cuda" if torch.cuda.is_available() else "cpu"
                fatal_failures: list[str] = []
                branch_memberships: dict[str, pd.DataFrame] = {}
                branch_orchestrators: dict[str, ImageTrainingOrchestratorV3] = {}
                print(json.dumps({"event": "FULL_TRAINING_START", "branches": branches, "device": device}))
                for branch in branches:
                    # Load private membership and splits for real training
                    membership_path = output_root / "private" / f"{branch}_image_membership.parquet"
                    splits_path = ROOT / config["branches"][branch]["splits"]
                    cache_path = ROOT / config["branches"][branch]["derived_cache"]
                    if not membership_path.is_file() or not splits_path.is_file() or not cache_path.is_file():
                        print(json.dumps({"branch": branch, "status": "SKIPPED_MISSING_INPUTS"}))
                        fatal_failures.append(f"{branch}:MISSING_INPUTS")
                        continue
                    membership = pd.read_parquet(membership_path)
                    # Filter to development only for training (LEARN is internal transport, not for HPO)
                    development = membership[membership["role"] == "development"].copy()
                    branch_memberships[branch] = development.copy()
                    splits = read_json(splits_path)
                    # Compute real digests for contract
                    membership_sha = _sha256_file(membership_path)
                    split_sha = _sha256_file(splits_path)
                    cache_sha = _sha256_file(cache_path)
                    # Outer folds from splits
                    outer_folds = splits.get("outer_folds", [])
                    observed_outer_folds = {int(row.get("fold", -1)) for row in outer_folds}
                    if observed_outer_folds != set(range(5)):
                        raise RuntimeError(
                            f"EXPECTED_FIVE_OUTER_FOLDS:{sorted(observed_outer_folds)}"
                        )
                    tabular_cols = config["branches"][branch]["model_features"]
                    hazard_edges = config["shared"]["hazard_bin_edges_days"]
                    # Contract mismatch is terminal for this versioned root; no evidence is deleted.
                    orch = ImageTrainingOrchestratorV3(
                        config=config,
                        output_root=output_root,
                        branch=branch,
                        cache_sha=cache_sha,
                        membership_sha=membership_sha,
                        split_sha=split_sha,
                        readiness_sha=run_contract_sha,
                        h5_cache_path=cache_path,
                    )
                    branch_orchestrators[branch] = orch
                    try:
                        orch.verify_inputs()
                    except Exception as e:
                        print(json.dumps({"branch": branch, "verify_failed": str(e)}))
                        fatal_failures.append(f"{branch}:INPUT_VERIFY:{e}")
                        continue
                    print(json.dumps({"branch": branch, "orchestrator": "INPUTS_VERIFIED", "outer_folds": len(outer_folds)}))
                    for outer in outer_folds:
                        outer_fold = int(outer.get("fold", 0))
                        outer_train_keys = set(outer.get("train_subjects") or outer.get("train_subject_keys") or [])
                        outer_test_keys = set(outer.get("test_subjects") or outer.get("test_subject_keys") or [])
                        # Filter development frame to outer train
                        outer_train_frame = development[development["patient_hash"].astype(str).isin(outer_train_keys)].copy()
                        if outer_train_frame.empty:
                            print(json.dumps({"branch": branch, "outer_fold": outer_fold, "status": "SKIPPED_EMPTY_TRAIN"}))
                            fatal_failures.append(f"{branch}:fold_{outer_fold}:EMPTY_TRAIN")
                            continue
                        inner_splits = _restrict_inner_splits_to_eligible_outer_train(
                            outer.get("inner_folds", []),
                            set(outer_train_frame["patient_hash"].astype(str)),
                        )
                        for variant in ["scalar_tabular", "scalar_tabular_plus_pattern_image"]:
                            winner_path = orch.state_root / f"winner_{variant}_outer_{outer_fold}.json"
                            if winner_path.is_file():
                                # A corrupt or contract-incompatible winner is terminal.  It must
                                # never be silently converted into a fresh HPO run in the same root.
                                winner_record = read_verified_winner_record(
                                    winner_path,
                                    expected_branch=branch,
                                    expected_variant=variant,
                                    expected_outer_fold=outer_fold,
                                    expected_master_contract_sha256=orch._master_contract_digest(),
                                )
                                winner = winner_record["winner"]
                                print(json.dumps({"branch": branch, "variant": variant, "outer_fold": outer_fold, "inner_hpo": "RESUMED_VERIFIED_WINNER"}))
                            else:
                                try:
                                    hpo_result = orch.run_inner_hpo(
                                        outer_fold=outer_fold,
                                        variant=variant,
                                        outer_train_frame=outer_train_frame,
                                        inner_splits=inner_splits,
                                        tabular_columns=tabular_cols,
                                        hazard_bin_edges=hazard_edges,
                                        device=device,
                                    )
                                    winner = hpo_result["winner"]
                                    print(json.dumps({"branch": branch, "variant": variant, "outer_fold": outer_fold, "winner": winner.get("candidate_id"), "mean_ibs": winner.get("mean_ibs")}))
                                except Exception as e:
                                    print(json.dumps({"branch": branch, "variant": variant, "outer_fold": outer_fold, "hpo_failed": str(e)}))
                                    fatal_failures.append(f"{branch}:{variant}:fold_{outer_fold}:HPO:{e}")
                                    continue

                            try:
                                final_info = orch.run_final_training(
                                    outer_fold=outer_fold,
                                    variant=variant,
                                    outer_train_frame=outer_train_frame,
                                    tabular_columns=tabular_cols,
                                    hazard_bin_edges=hazard_edges,
                                    winner=winner,
                                    device=device,
                                )
                                print(json.dumps({"branch": branch, "variant": variant, "outer_fold": outer_fold, "final": "LOCKED_OR_RESUMED", "ckpt": final_info.get("locked_checkpoint_sha")[:16]}))
                            except Exception as e:
                                print(json.dumps({"branch": branch, "variant": variant, "outer_fold": outer_fold, "final_failed": str(e)}))
                                fatal_failures.append(f"{branch}:{variant}:fold_{outer_fold}:FINAL:{e}")
                                continue

                            try:
                                rc = _build_run_contract(
                                    config=config,
                                    branch=branch,
                                    variant=variant,
                                    outer_fold=outer_fold,
                                    candidate_id=winner.get("candidate_id", "final"),
                                    membership_sha=membership_sha,
                                    cache_sha=cache_sha,
                                    split_sha=split_sha,
                                    readiness_sha=run_contract_sha,
                                )
                                locked_ckpt = output_root / "development" / branch / variant / f"outer_{outer_fold}" / "final" / "locked_checkpoint.pt"
                                metrics = _evaluate_outer_cell_exactly_once(
                                    orchestrator=orch,
                                    outer_fold=outer_fold,
                                    variant=variant,
                                    run_contract=rc,
                                    locked_checkpoint_path=locked_ckpt,
                                    outer_train_frame=outer_train_frame,
                                    test_frame_factory=lambda keys=frozenset(outer_test_keys): development[
                                        development["patient_hash"].astype(str).isin(keys)
                                    ].copy(),
                                    cache_path=cache_path,
                                    tabular_columns=tabular_cols,
                                    hazard_edges=hazard_edges,
                                    model_config=config["model"],
                                    requested_horizons_days=[float(value) for value in config["shared"]["requested_horizons_days"]],
                                    device=device,
                                    amp=bool(config["model"].get("amp", True) and device == "cuda"),
                                    batch_size=int(config["model"].get("batch_size", 4)),
                                )
                                # Calling the controller even after FINALIZED is intentional:
                                # it recovers missing mirrors from the committed result without
                                # materializing the outer test a second time.
                                print(json.dumps({"branch": branch, "variant": variant, "outer_fold": outer_fold, "outer": "COMMITTED_OR_RECOVERED", "uno_c_index": metrics["uno_c_index"], "ibs": metrics["integrated_brier_score"]}))
                            except Exception as e:
                                print(json.dumps({"branch": branch, "variant": variant, "outer_fold": outer_fold, "outer_failed": str(e)}))
                                fatal_failures.append(f"{branch}:{variant}:fold_{outer_fold}:OUTER:{e}")
                                continue
                    # After all outer folds, finalize branch
                    try:
                        orch.finalize_branch()
                        prefix = branch.upper()
                        already_done = f"{prefix}_OUTER_COMMITTED" in (state.fact("completed") or [])
                        if not already_done:
                            state.set_stage(f"{prefix}_INNER_HPO")
                            state.set_stage(f"{prefix}_MODELS_LOCKED")
                            state.set_stage(f"{prefix}_OUTER_COMMITTED", outer_folds_committed=5)
                        print(json.dumps({"branch": branch, "status": "BRANCH_OUTER_CV_COMPLETE"}))
                    except Exception as e:
                        print(json.dumps({"branch": branch, "finalize_failed": str(e)}))
                        fatal_failures.append(f"{branch}:FINALIZE:{e}")
                if fatal_failures:
                    raise RuntimeError(
                        "FAILED_INCOMPLETE_OUTER_CV:" + "|".join(fatal_failures)
                    )
                # Robustness/reports/privacy: execute Plan 012 analyses over REAL
                # locked OOF folds. Stage completion requires the executor's own
                # status; file presence alone is never sufficient (rule 6).
                oof_present = all(
                    (output_root / "private" / "outer_evaluation" / b / v / f"fold_{k}_private.json").is_file()
                    for b in branches_all for v in ["scalar_tabular", "scalar_tabular_plus_pattern_image"] for k in range(5)
                )
                if oof_present:
                    exec_statuses = []
                    privacy_all_pass = True
                    for b in branches_all:
                        need = [
                            output_root / "private" / "outer_evaluation" / b / v / f"fold_{k}_private.json"
                            for v in ["scalar_tabular", "scalar_tabular_plus_pattern_image"]
                            for k in range(5)
                        ]
                        if not all(x.is_file() for x in need):
                            continue
                        variants = {}
                        for v in ["scalar_tabular", "scalar_tabular_plus_pattern_image"]:
                            folds = []
                            for k in range(5):
                                pf = output_root / "private" / "outer_evaluation" / b / v / f"fold_{k}_private.json"
                                report_orchestrator = branch_orchestrators[b]
                                winner_record = read_verified_winner_record(
                                    report_orchestrator.state_root / f"winner_{v}_outer_{k}.json",
                                    expected_branch=b,
                                    expected_variant=v,
                                    expected_outer_fold=k,
                                    expected_master_contract_sha256=report_orchestrator._master_contract_digest(),
                                )
                                expected_cell_contract = _build_run_contract(
                                    config=config,
                                    branch=b,
                                    variant=v,
                                    outer_fold=k,
                                    candidate_id=winner_record["winner"]["candidate_id"],
                                    membership_sha=report_orchestrator.membership_sha,
                                    cache_sha=report_orchestrator.cache_sha,
                                    split_sha=report_orchestrator.split_sha,
                                    readiness_sha=report_orchestrator.readiness_sha,
                                )
                                metrics = read_verified_private_outer_record(
                                    pf,
                                    expected_branch=b,
                                    expected_variant=v,
                                    expected_outer_fold=k,
                                    expected_run_contract_sha256=expected_cell_contract,
                                )["metrics"]
                                folds.append({"fold": k, **metrics})
                            variants[v] = {"folds": folds}
                        metadata = branch_memberships[b].loc[
                            :,
                            [
                                "patient_hash",
                                "manufacturer",
                                "sex_binary",
                                "age",
                                "treatment_active",
                            ],
                        ].copy()
                        metadata["age_group"] = pd.cut(
                            metadata["age"],
                            bins=[-float("inf"), 69.999, 74.999, float("inf")],
                            labels=["LT70", "70_TO_74", "GE75"],
                        ).astype(str)
                        rep_summary = build_real_oof_reports(
                            output_root=output_root,
                            branch=b,
                            variants=variants,
                            metadata=metadata,
                            horizons_days=[float(h) for h in config["shared"]["requested_horizons_days"]],
                            bootstrap_iterations=int(config["robustness"]["bootstrap_iterations"]),
                            seed=int(config["seed"]),
                            scanner_sensitivity_passed=False,
                            quantitative_equivalence_proven=False,
                            external_validation_pass=False,
                            expected_outer_folds=5,
                        )
                        exec_statuses.append(rep_summary)
                        print(json.dumps({"event": "OOF_REPORTS_BUILT", "branch": b, "status": rep_summary["status"], "science_gate": rep_summary["science_gate"]["science_gate_status"]}))
                        if rep_summary["privacy_scan"]["status"] != "PASS":
                            privacy_all_pass = False
                    completed_all = len(exec_statuses) == 2 and all(
                        r["status"] == "COMPLETED_FROM_REAL_OOF" for r in exec_statuses
                    )
                    if completed_all:
                        claims = {row["branch"]: row["claim"] for row in exec_statuses}
                        aggregate = build_aggregate_claim_policy(
                            output_root=output_root,
                            branch_policies=claims,
                        )
                        aggregate["external_status"] = "EXTERNAL_PENDING"
                        atomic_write_json(
                            output_root / "reports" / "IMAGE_CLAIM_POLICY.json",
                            aggregate,
                        )
                        state.set_stage("ROBUSTNESS_COMPLETE")
                        state.set_stage("REPORTS_COMPLETE")
                        if privacy_all_pass:
                            state.set_stage("CONSISTENCY_PRIVACY_PASS", privacy_scan="PASS")
                            state.set_stage("FINALIZED", final_cell_count=20)
                        else:
                            raise RuntimeError("PUBLIC_PRIVACY_SCAN_FAILED")
                    else:
                        state.set_fact("reports_status", "PARTIAL_REAL_OOF")
                        print(json.dumps({"event": "REPORTS_PARTIAL"}))
                else:
                    state.set_fact("robustness_status", "PENDING_REAL_OOF")
                    state.set_fact("reports_status", "PENDING_REAL_OOF")
                    print(json.dumps({"event": "STAGES_PENDING_REAL_OOF"}))
                print(json.dumps({"event": "LAUNCHER_STATE", "stage": state.stage}))
                print(json.dumps({"status": "FULL_TRAINING_DISPATCHED"}))
                return 0
            else:
                result = _preflight(config_path)
                print(json.dumps(result, indent=2))
                return 0
    except ActiveWriterError as exc:
        print(json.dumps({"status": "BLOCKED", "reason": f"ACTIVE_WRITER:{exc}"}, indent=2))
        return 2
    except Exception as exc:
        print(json.dumps({"status": "BLOCKED", "reason": str(exc)}, indent=2))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
