from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any

import h5py
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.method_y_revival.atomic_io import read_json  # noqa: E402
from src.method_y_revival.contracts import sha256_mapping  # noqa: E402
from src.method_y_revival.image_reporting_v3 import privacy_scan_reports  # noqa: E402
from src.method_y_revival.image_training_orchestrator_v3 import _build_run_contract  # noqa: E402
from src.method_y_revival.outer_evaluation_control import OuterFoldController  # noqa: E402
from src.method_y_revival.software_contract_v3 import bind_software_runtime_contract  # noqa: E402


class FinalVerificationV3Error(RuntimeError):
    pass


BRANCHES = ("amyloid", "tau")
VARIANTS = ("scalar_tabular", "scalar_tabular_plus_pattern_image")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _load_context(config_path: Path, output_override: Path | None = None):
    config = bind_software_runtime_contract(
        yaml.safe_load(config_path.read_text(encoding="utf-8")), ROOT
    )
    return config, output_override or _resolve(config["output_root"])


def verify_readiness(config_path: Path, output_root: Path | None = None) -> dict[str, Any]:
    config, root = _load_context(config_path, output_root)
    invalidation = root / "provenance" / "INVALIDATION_20260826.json"
    if invalidation.is_file():
        return {
            "status": "INVALIDATED",
            "level": "readiness",
            "invalidation": read_json(invalidation),
        }
    readiness_path = root / "readiness" / "FINAL_IMAGE_TRAINING_READINESS.json"
    contract_path = root / "run_contract.json"
    if not readiness_path.is_file() or not contract_path.is_file():
        raise FinalVerificationV3Error("READINESS_OR_RUN_CONTRACT_MISSING")
    readiness = read_json(readiness_path)
    contract = read_json(contract_path)
    payload = contract.get("payload")
    if not isinstance(payload, dict) or contract.get("digest") != sha256_mapping(payload):
        raise FinalVerificationV3Error("RUN_CONTRACT_SELF_HASH_MISMATCH")
    if readiness.get("run_contract_sha256") != contract["digest"]:
        raise FinalVerificationV3Error("READINESS_RUN_CONTRACT_MISMATCH")
    if payload.get("software_runtime_contract") != config.get("software_runtime_contract"):
        raise FinalVerificationV3Error("SOFTWARE_RUNTIME_CONTRACT_MISMATCH")
    if readiness.get("training_execution_allowed") is not True:
        raise FinalVerificationV3Error("TRAINING_EXECUTION_NOT_ALLOWED")

    cache_counts: dict[str, int] = {}
    for branch in BRANCHES:
        cache = _resolve(config["branches"][branch]["derived_cache"])
        if not cache.is_file():
            raise FinalVerificationV3Error(f"CACHE_MISSING:{branch}")
        with h5py.File(cache, "r") as handle:
            keys = [key for key in handle.keys() if not key.startswith("__tmp__")]
        if not keys:
            raise FinalVerificationV3Error(f"CACHE_EMPTY:{branch}")
        cache_counts[branch] = len(keys)
        split = read_json(_resolve(config["branches"][branch]["splits"]))
        observed = {int(row.get("fold", -1)) for row in split.get("outer_folds", [])}
        if observed != set(range(5)):
            raise FinalVerificationV3Error(f"SPLIT_NOT_FIVE_FOLD:{branch}")
        if not (root / "private" / f"{branch}_image_membership.parquet").is_file():
            raise FinalVerificationV3Error(f"MEMBERSHIP_MISSING:{branch}")
    return {
        "status": "PASS",
        "level": "readiness",
        "output_root": str(root),
        "run_contract_sha256": contract["digest"],
        "training_execution_allowed": True,
        "positive_claim_allowed": False,
        "external_claim_allowed": False,
        "cache_counts": cache_counts,
    }


def _verify_inner_scores(root: Path, branch: str, master_digest: str) -> int:
    completed = 0
    state_root = root / "orchestration" / branch
    for variant in VARIANTS:
        for fold in range(5):
            path = state_root / f"winner_{variant}_outer_{fold}.json"
            if not path.is_file():
                raise FinalVerificationV3Error(f"WINNER_MISSING:{branch}:{variant}:{fold}")
            payload = read_json(path)
            if payload.get("run_contract_sha256") != master_digest:
                raise FinalVerificationV3Error("WINNER_MASTER_CONTRACT_MISMATCH")
            candidates = payload.get("candidates", [])
            if not candidates:
                raise FinalVerificationV3Error("HPO_CANDIDATES_MISSING")
            for candidate in candidates:
                records = candidate.get("inner_fold_metric_records", [])
                if candidate.get("status") != "COMPLETED" or len(records) != 3:
                    raise FinalVerificationV3Error("INNER_HPO_INCOMPLETE")
                values = []
                for record in records:
                    value = float(record.get("metric_value", float("nan")))
                    if (
                        record.get("metric_name") != "IPCW_INTEGRATED_BRIER_SCORE"
                        or not math.isfinite(value)
                        or not 0.0 <= value <= 1.0
                        or len(str(record.get("locked_checkpoint_sha256", ""))) != 64
                    ):
                        raise FinalVerificationV3Error("INNER_IBS_INVALID_OR_FALLBACK")
                    values.append(value)
                if abs(float(candidate.get("mean_ibs")) - sum(values) / len(values)) > 1e-12:
                    raise FinalVerificationV3Error("INNER_IBS_MEAN_MISMATCH")
            completed += 1
    return completed


def verify_final(config_path: Path, output_root: Path | None = None) -> dict[str, Any]:
    readiness = verify_readiness(config_path, output_root)
    if readiness["status"] != "PASS":
        return {**readiness, "level": "final"}
    config, root = _load_context(config_path, output_root)
    readiness_sha = readiness["run_contract_sha256"]
    final_cells = 0
    matched_members: dict[tuple[str, int], dict[str, set[str]]] = {}

    for branch in BRANCHES:
        membership_path = root / "private" / f"{branch}_image_membership.parquet"
        split_path = _resolve(config["branches"][branch]["splits"])
        cache_path = _resolve(config["branches"][branch]["derived_cache"])
        membership_sha = _sha256_file(membership_path)
        split_sha = _sha256_file(split_path)
        cache_sha = _sha256_file(cache_path)
        master_path = root / "orchestration" / branch / "orchestrator_state.json"
        if not master_path.is_file():
            raise FinalVerificationV3Error(f"ORCHESTRATOR_STATE_MISSING:{branch}")
        master = read_json(master_path)
        master_body = {key: value for key, value in master.items() if key != "state_sha256"}
        if master.get("state_sha256") != sha256_mapping(master_body):
            raise FinalVerificationV3Error("ORCHESTRATOR_STATE_HASH_MISMATCH")
        if master.get("state") != "BRANCH_OUTER_CV_COMPLETE":
            raise FinalVerificationV3Error(f"BRANCH_NOT_COMPLETE:{branch}")
        _verify_inner_scores(root, branch, str(master.get("run_contract_sha256")))

        for variant in VARIANTS:
            outer_root = root / "private" / "outer_evaluation" / branch / variant
            for fold in range(5):
                winner = read_json(
                    root / "orchestration" / branch / f"winner_{variant}_outer_{fold}.json"
                )["winner"]
                expected_contract = _build_run_contract(
                    config=config,
                    branch=branch,
                    variant=variant,
                    outer_fold=fold,
                    candidate_id=winner["candidate_id"],
                    membership_sha=membership_sha,
                    cache_sha=cache_sha,
                    split_sha=split_sha,
                    readiness_sha=readiness_sha,
                )
                controller = OuterFoldController(
                    outer_root, fold=fold, run_contract_sha256=expected_contract
                )
                state = controller._load_state()
                if state.get("state") != "FINALIZED":
                    raise FinalVerificationV3Error(f"OUTER_NOT_FINALIZED:{branch}:{variant}:{fold}")
                result = controller._load_result(str(state.get("checkpoint_sha256")))
                journal = controller._load_access_journal()
                if result is None or journal is None or journal.get("access_count") != 1:
                    raise FinalVerificationV3Error("OUTER_RESULT_OR_ACCESS_JOURNAL_INVALID")
                metrics = result["metrics"]
                ibs = float(metrics.get("integrated_brier_score", float("nan")))
                uno = float(metrics.get("uno_c_index", float("nan")))
                if not math.isfinite(ibs) or not 0.0 <= ibs <= 1.0 or not math.isfinite(uno):
                    raise FinalVerificationV3Error("OUTER_METRIC_INVALID")
                members = {str(row.get("patient_hash")) for row in metrics.get("oof_predictions", [])}
                if len(members) != int(metrics.get("n_test", -1)):
                    raise FinalVerificationV3Error("OUTER_MEMBERSHIP_COUNT_MISMATCH")
                matched_members.setdefault((branch, fold), {})[variant] = members
                final_cells += 1

    for key, variants in matched_members.items():
        if set(variants) != set(VARIANTS) or variants[VARIANTS[0]] != variants[VARIANTS[1]]:
            raise FinalVerificationV3Error(f"MATCHED_MEMBERSHIP_MISMATCH:{key}")
    if final_cells != 20:
        raise FinalVerificationV3Error(f"FINAL_CELL_COUNT_MISMATCH:{final_cells}")

    for branch in BRANCHES:
        for suffix in (
            "calibration.json",
            "subgroups.json",
            "scanner_robustness.json",
            "image_incremental.json",
            "IMAGE_CLAIM_POLICY.json",
        ):
            if not (root / "reports" / f"{branch}_{suffix}").is_file():
                raise FinalVerificationV3Error(f"REPORT_MISSING:{branch}:{suffix}")
    privacy = privacy_scan_reports(root / "reports")
    if privacy["status"] != "PASS":
        raise FinalVerificationV3Error("PUBLIC_PRIVACY_SCAN_FAILED")
    global_claim = read_json(root / "reports" / "IMAGE_CLAIM_POLICY.json")
    if global_claim.get("positive_claim_allowed") is not False or global_claim.get("external_claim_allowed") is not False:
        raise FinalVerificationV3Error("CLAIM_BOUNDARY_VIOLATION")
    launcher = read_json(root / "orchestration" / "launcher_state.json")
    if launcher.get("stage") != "FINALIZED" or launcher.get("facts", {}).get("final_cell_count") != 20:
        raise FinalVerificationV3Error("LAUNCHER_NOT_FINALIZED")
    return {
        "status": "PASS",
        "level": "final",
        "output_root": str(root),
        "valid_outer_cells": final_cells,
        "outer_access_count_per_cell": 1,
        "inner_metric": "IPCW_INTEGRATED_BRIER_SCORE",
        "privacy": "PASS",
        "positive_claim_allowed": False,
        "external_claim_allowed": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify Method Y image v3.2")
    parser.add_argument("--level", choices=("readiness", "final"), default="readiness")
    parser.add_argument("--config", default=str(ROOT / "configs" / "method_y_image_training_v3_2.yaml"))
    parser.add_argument("--output-root")
    args = parser.parse_args()
    try:
        result = (
            verify_final(Path(args.config), Path(args.output_root) if args.output_root else None)
            if args.level == "final"
            else verify_readiness(Path(args.config), Path(args.output_root) if args.output_root else None)
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result.get("status") == "PASS" else 2
    except Exception as exc:
        print(json.dumps({"status": "FAIL", "level": args.level, "reason": str(exc)}, indent=2, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
