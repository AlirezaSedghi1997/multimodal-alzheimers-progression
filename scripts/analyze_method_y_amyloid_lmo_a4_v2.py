"""Conditional, post-hoc paired uncertainty for the locked A4 manufacturer holdout.

No image inference, model fitting, or new outer-test access occurs here. Bootstrap
intervals condition on the six fitted models and do not measure selection or
training variance; different manufacturer IBS grids are never pooled.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_method_y_amyloid_lmo_a4_v2 import _preflight, _verify, _check_public
from src.method_y_revival.atomic_io import atomic_write_json, read_json
from src.method_y_revival.contracts import sha256_mapping
from src.method_y_revival.posttraining_bootstrap_v4_2 import _score_fold, _surv


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _paired_data(root: Path, fold: int) -> dict:
    metrics = {}
    paths = {}
    for arm in ("A0", "A4"):
        path = root / "private/outer_evaluation/amyloid" / arm / f"fold_{fold}/outer_result.json"
        record = read_json(path)
        metrics[arm] = record["metrics"]
        paths[arm] = _sha(path)
    a0, a4 = metrics["A0"], metrics["A4"]
    if a0["evaluation_grid_days"] != a4["evaluation_grid_days"] or a0["censoring_reference"] != a4["censoring_reference"]:
        raise RuntimeError("PAIRED_GRID_OR_TRAIN_REFERENCE_MISMATCH")
    by_arm = {}
    for arm, item in metrics.items():
        rows = {str(row["patient_hash"]): row for row in item["oof_predictions"]}
        if len(rows) != len(item["oof_predictions"]):
            raise RuntimeError("DUPLICATE_OUTER_SUBJECT")
        by_arm[arm] = rows
    if set(by_arm["A0"]) != set(by_arm["A4"]):
        raise RuntimeError("PAIRED_MEMBERSHIP_MISMATCH")
    ids = sorted(by_arm["A0"])
    for key in ids:
        for column in ("event", "time_days", "original_event", "original_time_days"):
            if by_arm["A0"][key][column] != by_arm["A4"][key][column]:
                raise RuntimeError("PAIRED_OUTCOME_MISMATCH")
    rows = {arm: [by_arm[arm][key] for key in ids] for arm in ("A0", "A4")}
    grid = np.asarray(a0["evaluation_grid_days"], dtype=float)
    train = _surv(a0["censoring_reference"])
    point = {}
    for arm in ("A0", "A4"):
        point[arm] = _score_fold(train, rows[arm], grid)
        for actual, stored in zip(point[arm], (
            metrics[arm]["integrated_brier_score"], metrics[arm]["uno_c_index"]
        ), strict=True):
            if not math.isclose(actual, float(stored), rel_tol=1e-8, abs_tol=1e-8):
                raise RuntimeError("LOCKED_METRIC_RECOMPUTATION_MISMATCH")
    return {"rows": rows, "train": train, "grid": grid, "point": point,
            "n": len(ids), "events": sum(int(row["event"]) for row in rows["A0"]),
            "outer_result_sha256": paths}


def _bootstrap_one(data: dict, *, iterations: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    draw_ibs, draw_uno = [], []
    rejected = 0
    for _ in range(iterations * 5):
        if len(draw_ibs) >= iterations:
            break
        take = rng.integers(0, data["n"], size=data["n"])
        try:
            values = {}
            for arm in ("A0", "A4"):
                sample = [data["rows"][arm][int(index)] for index in take]
                values[arm] = _score_fold(data["train"], sample, data["grid"])
            draw_ibs.append(values["A4"][0] - values["A0"][0])
            draw_uno.append(values["A4"][1] - values["A0"][1])
        except (ValueError, ZeroDivisionError):
            rejected += 1
    point = {
        "a4_minus_a0_ibs": data["point"]["A4"][0] - data["point"]["A0"][0],
        "a4_minus_a0_uno": data["point"]["A4"][1] - data["point"]["A0"][1],
    }
    if len(draw_ibs) < iterations or rejected / max(1, rejected + len(draw_ibs)) > 0.05:
        return {**point, "uncertainty_status": "NOT_ESTIMABLE",
                "reason": "BOOTSTRAP_SUPPORT_OR_REJECTION_RATE",
                "valid_resamples": len(draw_ibs), "rejected_attempts": rejected}
    return {**point, "uncertainty_status": "CONDITIONAL_POST_HOC_EXPLORATORY",
            "delta_ibs_95ci": np.quantile(draw_ibs, [0.025, 0.975]).tolist(),
            "delta_uno_95ci": np.quantile(draw_uno, [0.025, 0.975]).tolist(),
            "valid_resamples": len(draw_ibs), "rejected_attempts": rejected}


def analyze(policy: Path, output_root: Path, *, iterations: int, seed: int) -> dict:
    if iterations < 1000:
        raise RuntimeError("BOOTSTRAP_REQUIRES_AT_LEAST_1000_RESAMPLES")
    preflight = _preflight(policy)
    summary = _verify(preflight)
    if summary["status"] != "PASS_TECHNICAL_LOCAL_LMO_ONLY" or summary["outer_cells_finalized"] != 6:
        raise RuntimeError("A4_LMO_NOT_VERIFIED_COMPLETE")
    root = preflight["output_root"]
    by_name = {row["heldout_manufacturer"]: row for row in preflight["plan"]["outer_folds"]}
    results, datasets = {}, {}
    inputs = {"public_summary_sha256": _sha(root / "reports/amyloid_lmo_a4_public_summary.json"),
              "run_contract_sha256": _sha(root / "run_contract.json")}
    patient_keys = set()
    for name in ("GE", "PHILIPS", "SIEMENS"):
        outer = by_name[name]
        patient_keys.update(str(key) for key in outer["test_subjects"])
        fold = int(outer["fold"])
        data = _paired_data(root, fold)
        if data["n"] != int(outer["test_count"]) or data["events"] != int(outer["test_events"]):
            raise RuntimeError("HELDOUT_SUPPORT_MISMATCH")
        for arm, digest in data["outer_result_sha256"].items():
            inputs[f"{name}_{arm}_outer_result_sha256"] = digest
        datasets[name] = (fold, data)
    contract = {"schema_version": "1.0", "input_sha256": inputs,
                "analysis_source_sha256": _sha(Path(__file__)),
                "iterations": iterations, "seed": seed}
    contract_path = output_root / "run_contract.json"
    report_path = output_root / "public_bundle/amyloid_lmo_a4_conditional_uncertainty.json"
    if contract_path.is_file():
        if read_json(contract_path) != contract:
            raise RuntimeError("POSTANALYSIS_CONTRACT_MISMATCH")
        if report_path.is_file():
            existing = read_json(report_path)
            if existing.get("report_sha256") != sha256_mapping({k: v for k, v in existing.items() if k != "report_sha256"}):
                raise RuntimeError("POSTANALYSIS_REPORT_HASH_MISMATCH")
            return existing
    else:
        if report_path.exists():
            raise RuntimeError("POSTANALYSIS_REPORT_WITHOUT_CONTRACT")
        atomic_write_json(contract_path, contract)
    for name, (fold, data) in datasets.items():
        results[name] = {"n": data["n"], "events": data["events"],
                         "evaluation_grid_days": data["grid"].tolist(),
                         **_bootstrap_one(data, iterations=iterations, seed=seed + fold)}
    report = {"schema_version": "1.0", "status": "COMPLETED_CONDITIONAL_EXPLORATORY",
              "science_gate_status": "BLOCKED", "claim_status": "POST_HOC_INTERNAL_EXPLORATORY",
              "positive_claim_allowed": False, "external_validation_status": "EXTERNAL_PENDING",
              "manufacturer_results": results,
              "interpretation_limit": "FIXED_LOCKED_MODELS_ONLY; NO_TRAINING_OR_SELECTION_VARIANCE; THREE_LOCAL_MANUFACTURERS; POLICY_NOT_ARCHITECTURE_COMPARISON; NO_CROSS_GRID_IBS_POOLING",
              "source_public_summary_sha256": inputs["public_summary_sha256"]}
    report["report_sha256"] = sha256_mapping(report)
    _check_public(report, patient_keys)
    atomic_write_json(report_path, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", type=Path, default=ROOT / "configs/method_y_amyloid_lmo_a4_v2.yaml")
    parser.add_argument("--output-root", type=Path, default=ROOT / "results/method_y_amyloid_lmo_a4_postanalysis_v1")
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()
    try:
        report = analyze(args.policy, args.output_root, iterations=args.iterations, seed=args.seed)
        print(json.dumps({"status": report["status"], "claim_status": report["claim_status"],
                          "uncertainty_status": {k: v["uncertainty_status"] for k, v in report["manufacturer_results"].items()},
                          "report_sha256": report["report_sha256"]}, sort_keys=True), flush=True)
        return 0
    except Exception as exc:
        print(json.dumps({"status": "BLOCKED", "reason": f"{type(exc).__name__}:{exc}"}), flush=True)
        return 2


if __name__ == "__main__":
    print(json.dumps({"status": "SUPERSEDED", "reason": "USE_run_method_y_local_validation_v5.ps1"}))
    raise SystemExit(2)
