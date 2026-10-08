"""Grid-based descriptive calibration of finalized A4 manufacturer holdouts.

Uses the locked 1095-day evaluation grid, not the distinct prespecified
1095.75-day dynamic-AUC horizon. No image inference or model refitting.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_method_y_amyloid_lmo_a4_v2 import _check_public, _preflight, _verify
from src.method_y_revival.atomic_io import atomic_write_json, read_json
from src.method_y_revival.contracts import sha256_mapping
from src.method_y_revival.post_eval_v4 import (
    _km, _km_step_values, calibration_at_horizon, ipcw_weights,
)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def analyze() -> dict:
    policy = ROOT / "configs/method_y_amyloid_lmo_a4_v2.yaml"
    preflight = _preflight(policy)
    summary = _verify(preflight)
    if summary["outer_cells_finalized"] != 6:
        raise RuntimeError("A4_LMO_INCOMPLETE")
    source = preflight["output_root"]
    # v1 had mis-indexed CIs; v2 lacked a perfect-separation gate; v3 bound
    # the entire unrelated reporting module rather than the metric functions.
    output = ROOT / "results/method_y_amyloid_lmo_a4_calibration_v4"
    report_path = output / "public_bundle/amyloid_lmo_a4_calibration.json"
    contract_path = output / "run_contract.json"
    input_sha = {"summary": _sha(source / "reports/amyloid_lmo_a4_public_summary.json")}
    results = {}
    patient_keys = set()
    for outer in preflight["plan"]["outer_folds"]:
        name, fold = outer["heldout_manufacturer"], int(outer["fold"])
        patient_keys.update(str(key) for key in outer["test_subjects"])
        results[name] = {}
        for arm in ("A0", "A4"):
            path = source / f"private/outer_evaluation/amyloid/{arm}/fold_{fold}/outer_result.json"
            input_sha[f"{name}_{arm}"] = _sha(path)
            metrics = read_json(path)["metrics"]
            grid = list(map(float, metrics["evaluation_grid_days"]))
            horizon = 1095.0
            if horizon not in grid:
                results[name][arm] = {"status": "NOT_ESTIMABLE", "reason": "LOCKED_GRID_LACKS_1095"}
                continue
            index = grid.index(horizon)
            rows = metrics["oof_predictions"]
            time = np.asarray([float(row["time_days"]) for row in rows])
            event = np.asarray([bool(row["event"]) for row in rows])
            usable = event | (time >= horizon)
            events = int(np.sum(event & (time <= horizon)))
            controls = int(np.sum(usable)) - events
            base = {"n": len(rows), "n_used": int(np.sum(usable)),
                    "events_by_horizon": events, "observed_controls": controls}
            if min(events, controls) < 10:
                results[name][arm] = {**base, "status": "NOT_ESTIMABLE",
                                      "reason": "FEWER_THAN_TEN_OBSERVED_EVENTS_OR_CONTROLS"}
                continue
            train = metrics["censoring_reference"]
            fitted = calibration_at_horizon(
                time, event,
                np.asarray([float(row["survival_probabilities"][index]) for row in rows]),
                horizon=horizon,
                train_time=np.asarray([float(row["time_days"]) for row in train]),
                train_event=np.asarray([bool(row["event"]) for row in train]),
            )
            if fitted["status"] == "PASS":
                if not (fitted["slope_ci"][0] <= fitted["slope"] <= fitted["slope_ci"][1]
                        and fitted["intercept_ci"][0] <= fitted["intercept"] <= fitted["intercept_ci"][1]):
                    raise RuntimeError("CALIBRATION_ESTIMATE_OUTSIDE_OWN_CI")
                # Bin means are aggregate but sparse held-out cells are not
                # released; provide slope/intercept only at manufacturer level.
                fitted.pop("calibration_bins", None)
                fitted["status"] = "ESTIMABLE_TECHNICAL"
            results[name][arm] = {**base, **fitted}
    import numpy, sksurv, statsmodels
    calibration_source = "\n".join(inspect.getsource(function) for function in (
        _km, _km_step_values, ipcw_weights, calibration_at_horizon
    ))
    contract = {"schema_version": "1.0", "source_sha256": input_sha,
                "analysis_source_sha256": _sha(Path(__file__)),
                "calibration_function_bundle_sha256": hashlib.sha256(calibration_source.encode()).hexdigest(),
                "runtime_versions": {"numpy": numpy.__version__, "sksurv": sksurv.__version__,
                                     "statsmodels": statsmodels.__version__},
                "horizon_days": 1095.0}
    if contract_path.is_file():
        if read_json(contract_path) != contract:
            raise RuntimeError("CALIBRATION_CONTRACT_MISMATCH")
        if report_path.is_file():
            previous = read_json(report_path)
            if previous.get("report_sha256") != sha256_mapping({k: v for k, v in previous.items() if k != "report_sha256"}):
                raise RuntimeError("CALIBRATION_REPORT_HASH_MISMATCH")
            return previous
    else:
        if report_path.exists():
            raise RuntimeError("CALIBRATION_REPORT_WITHOUT_CONTRACT")
        atomic_write_json(contract_path, contract)
    report = {"schema_version": "1.0", "status": "COMPLETED_DESCRIPTIVE_EXPLORATORY",
              "claim_status": "POST_HOC_INTERNAL_EXPLORATORY", "science_gate_status": "BLOCKED",
              "horizon_days": 1095.0,
              "horizon_role": "LOCKED_IBS_GRID_SECONDARY_NOT_PRESPECIFIED_1095_75_DYNAMIC_AUC_HORIZON",
              "interpretation": "ESTIMABLE_TECHNICAL_IS_NOT_A_SCIENTIFIC_CALIBRATION_PASS",
              "uncertainty_limit": "IPCW_GLM_HESSIAN_INTERVALS_CONDITIONAL_ON_FITTED_MODELS_AND_NO_CLUSTER_CORRECTION",
              "manufacturer_results": results,
              "external_validation_status": "EXTERNAL_PENDING",
              "source_summary_sha256": input_sha["summary"]}
    report["report_sha256"] = sha256_mapping(report)
    _check_public(report, patient_keys)
    atomic_write_json(report_path, report)
    return report


if __name__ == "__main__":
    print(json.dumps({"status": "SUPERSEDED", "reason": "USE_run_method_y_local_validation_v5.ps1"}))
    raise SystemExit(2)
