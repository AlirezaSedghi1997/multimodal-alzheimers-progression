"""Versioned MX5--MX7 safety amendment for the live fast-track v2 run.

The v2 preregistration and its MX4 writer remain byte-for-byte unchanged.  This
additive coordinator locks its own source before MX5 GPU work, binds every MX6
attempt to one atlas/SPM snapshot, and repairs the MX7 XAI state-file inventory.
It does not alter the four candidate models, selection rule, or outcome data.
"""
from __future__ import annotations

import argparse
import json
import math
import msvcrt
import os
import subprocess
import sys
import uuid
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.mx_fasttrack_v2.capture_preflight import windows_python_processes
from scripts.mx_fasttrack_v2.integrity import check_inputs, check_lock
from scripts.mx_fasttrack_v2.run import read, sha
from scripts.mx_fasttrack_v2.supervise import (
    DEFAULT_OUTPUT, DEFAULT_PACKAGE, Supervisor as V2Supervisor, utc,
)
from scripts.mx_fasttrack_v2.xai_limited import atlas_binding

AMENDMENT_NAME = "fasttrack_v21_amendment.json"
CONTEXT_NAME = "xai_v21_context.json"
FOLD_BINDING_NAME = "xai_v21_fold_binding.json"
WAIVER_NAME = "xai_v21_explicit_omission.json"
TEMPORAL_APPENDIX_NAME = "fasttrack_v21_temporal_appendix.json"
SKIP_REASONS = ("ATLAS_UNAVAILABLE", "VRAM_LIMIT", "TIME_BUDGET", "OPERATOR_CHOICE")
SOURCE_RELATIVE = "scripts/mx_fasttrack_v2_1/supervise_v21.py"
INTERNAL_PARENT_ENV = "MX_FASTTRACK_V21_COORDINATOR_PID"


def immutable_json(path: Path, value: dict) -> None:
    """Crash-safe write-once JSON without a fixed-name torn-write obstacle."""
    if path.exists():
        if read(path) != value:
            raise RuntimeError(f"V21_IMMUTABLE_FILE_CHANGED:{path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.{uuid.uuid4().hex}.new")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.flush()
            os.fsync(stream.fileno())
        if path.exists():
            if read(path) != value:
                raise RuntimeError(f"V21_IMMUTABLE_FILE_CHANGED:{path}")
            return
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def has_nonfinite_number(value: object) -> bool:
    if isinstance(value, float):
        return not math.isfinite(value)
    if isinstance(value, dict):
        return any(has_nonfinite_number(item) for item in value.values())
    if isinstance(value, list):
        return any(has_nonfinite_number(item) for item in value)
    return False


def foreign_method_x_writers(root: Path) -> list[dict]:
    """Do not enter the shared v2 ledger while another coordinator is live."""
    found = []
    for row in windows_python_processes(cwd=root):
        pid = int(row.get("ProcessId", -1))
        if pid == os.getpid():
            continue
        command = str(row.get("CommandLine") or "").lower().replace("/", "\\")
        if ("\\mx_fasttrack_v2\\supervise.py" in command
            or "\\mx_fasttrack_v2\\run.py" in command
            or "train_3d_mri.py" in command
            or "\\mx_fasttrack_v2\\evaluate_locked.py" in command
            or "\\mx_fasttrack_v2\\evaluate_hybrid_pair.py" in command
            or "\\mx_fasttrack_v2\\xai_limited.py" in command
            or "\\mx_fasttrack_v2\\freeze_report.py" in command):
            found.append({"pid": pid, "command": command})
    return found


def require_internal_coordinator_parent(root: Path) -> None:
    """Accept the venv launcher hop without admitting an unrelated caller."""
    claimed = os.environ.get(INTERNAL_PARENT_ENV)
    if claimed is None or not claimed.isdecimal():
        raise RuntimeError("V21_INTERNAL_TARGET_REQUIRES_ACTIVE_COORDINATOR")
    coordinator_pid = int(claimed)
    launcher_pid = os.getppid()
    if launcher_pid == coordinator_pid:
        return
    for row in windows_python_processes(cwd=root):
        if (int(row.get("ProcessId", -1)) == launcher_pid
            and int(row.get("ParentProcessId", -1)) == coordinator_pid
            and "mx_fasttrack_v2_1\\supervise_v21.py" in
                str(row.get("CommandLine") or "").lower().replace("/", "\\")):
            return
    raise RuntimeError("V21_INTERNAL_TARGET_REQUIRES_ACTIVE_COORDINATOR")


@contextmanager
def coordinator_lock(reports: Path):
    """One v2.1 coordinator at a time; Windows releases the byte lock on exit."""
    reports.mkdir(parents=True, exist_ok=True)
    path = reports / "fasttrack_v21_coordinator.lock"
    with path.open("a+b") as stream:
        if stream.seek(0, os.SEEK_END) == 0:
            stream.write(b"\0")
            stream.flush()
        stream.seek(0)
        try:
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError as exc:
            raise RuntimeError("FASTTRACK_V21_COORDINATOR_ALREADY_ACTIVE") from exc
        try:
            yield
        finally:
            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)


def preserve_orphaned_write_once_temps(reports: Path, names: tuple[str, ...]) -> list[Path]:
    """Unblock a retry after a crash without discarding an uncommitted byte.

    The locked v2 writers create a fixed ``.new`` name with exclusive mode.
    If the process dies before the rename, that name blocks every retry.  A
    v2.1 coordinator may preserve only these known uncommitted temps, while
    the final target is still absent, before asking v2 to recompute the file.
    """
    preserved = []
    for name in names:
        target = reports / name
        temporary = target.with_suffix(target.suffix + ".new")
        if target.exists() or not temporary.is_file():
            continue
        archive = reports / "orphaned_atomic_writes"
        archive.mkdir(parents=True, exist_ok=True)
        destination = archive / f"{name}.{uuid.uuid4().hex}.uncommitted"
        temporary.rename(destination)
        preserved.append(destination)
    return preserved


def require_unambiguous_outer_resume(out: Path, step_name: str) -> None:
    """Never silently re-infer an outer fold after an ambiguous interruption.

    The v2 evaluator records RUNNING before it reads a fold's test rows.  If
    there is no committed receipt for a fold, its state alone cannot prove
    whether labels/inference were already accessed.  Complete receipt sets
    can be reaggregated without another inference; partial sets require an
    explicit access audit and a separately versioned recovery decision.
    """
    basename = {
        "mx5_outer_hybrid_pair": "evaluation_outer_hybrid_pair",
        "mx4_outer_mx04_scratch": "evaluation_outer_mx04_scratch",
        "mx4_outer_mx04_head_last": "evaluation_outer_mx04_head_last",
    }.get(step_name)
    if basename is None:
        return
    state_path = out / "reports" / f"{basename}_state.json"
    result_path = out / "reports" / f"{basename}.json"
    if not state_path.exists():
        if result_path.exists():
            raise RuntimeError(f"V21_OUTER_RESULT_WITHOUT_STATE:{basename}")
        return
    state = read(state_path)
    if state.get("status") == "COMPLETED":
        return
    if state.get("status") not in {"STARTED", "RUNNING"}:
        raise RuntimeError(f"V21_OUTER_STATE_NOT_RESUMABLE:{basename}")
    receipts = out / "_private_resume" / basename
    missing = [fold for fold in range(5)
               if not (receipts / f"fold_{fold}.json").is_file()]
    if missing:
        raise RuntimeError(
            f"V21_OUTER_ACCESS_AMBIGUOUS_REVIEW_LOGS_BEFORE_REINFERENCE:"
            f"{basename}:missing_receipts={missing}")


def verify_hybrid_pair_effect(out: Path) -> None:
    """The signed paired effect must agree with the two reported arm metrics."""
    reports = out / "reports"
    result_path = reports / "evaluation_outer_hybrid_pair.json"
    state_path = reports / "evaluation_outer_hybrid_pair_state.json"
    if not result_path.is_file() or not state_path.is_file():
        raise RuntimeError("V21_HYBRID_PAIR_RESULT_OR_STATE_MISSING")
    result, state = read(result_path), read(state_path)
    if (state.get("status") != "COMPLETED"
        or state.get("result_sha256") != sha(result_path)
        or state.get("lock_sha256") != result.get("lock_sha256")
        or result.get("status") != "COMPLETED_EXPLORATORY_PAIRED"
        or result.get("n") != 1336):
        raise RuntimeError("V21_HYBRID_PAIR_RESULT_OR_STATE_INVALID")
    arms, effects = result.get("arms"), result.get("paired_dense_minus_baseline")
    if not isinstance(arms, dict) or not isinstance(effects, dict):
        raise RuntimeError("V21_HYBRID_PAIR_METRICS_MISSING")
    for metric in ("roc_auc", "average_precision", "brier"):
        baseline = (arms.get("hybrid_baseline") or {}).get(metric)
        dense = (arms.get("hybrid_dense") or {}).get(metric)
        row = effects.get(metric)
        if (type(baseline) not in (float, int)
            or type(dense) not in (float, int)
            or not isinstance(row, dict)
            or type(row.get("dense_minus_baseline")) not in (float, int)
            or not all(math.isfinite(float(v)) and 0 <= float(v) <= 1
                       for v in (baseline, dense))
            or not math.isfinite(float(row["dense_minus_baseline"]))):
            raise RuntimeError(f"V21_HYBRID_PAIR_METRIC_INVALID:{metric}")
        if not math.isclose(float(row["dense_minus_baseline"]),
                            float(dense) - float(baseline),
                            rel_tol=1e-9, abs_tol=1e-10):
            raise RuntimeError(f"V21_HYBRID_PAIR_SIGNED_EFFECT_MISMATCH:{metric}")
        interval = row.get("patient_bootstrap_ci95")
        draws = row.get("valid_draws")
        if (not isinstance(interval, list) or len(interval) != 2
            or any(type(v) not in (float, int) or not math.isfinite(v) for v in interval)
            or interval[0] > interval[1]
            or type(draws) is not int or not 900 <= draws <= 1000):
            raise RuntimeError(f"V21_HYBRID_PAIR_INTERVAL_INVALID:{metric}")


def temporal_appendix_payload(out: Path) -> dict:
    """Expose existing temporal evidence without inventing hybrid subgroup scores."""
    reports = out / "reports"
    audit_path = reports / "data_semantics_audit.json"
    audit = read(audit_path)
    direction = audit.get("pet_vs_mri_direction")
    windows = audit.get("time_window_sensitivity_counts")
    if (audit.get("status") != "READY"
        or audit.get("real_retained_n") != 1669
        or not isinstance(direction, dict)
        or set(direction) != {"PET_AFTER_MRI", "PET_BEFORE_MRI", "SAME_DAY"}
        or any(type(value) is not int or value < 0 for value in direction.values())
        or sum(direction.values()) != 1669
        or not isinstance(windows, dict)
        or set(windows) != {"90", "180", "365"}):
        raise RuntimeError("V21_TEMPORAL_AUDIT_COUNTS_INVALID")
    for threshold in ("90", "180", "365"):
        row = windows[threshold]
        if (not isinstance(row, dict) or type(row.get("n")) is not int
            or type(row.get("positive")) is not int
            or type(row.get("negative")) is not int
            or row["n"] != row["positive"] + row["negative"]
            or row["n"] < 0 or row["n"] > 1669):
            raise RuntimeError(f"V21_TEMPORAL_WINDOW_COUNT_INVALID:{threshold}")
    if not windows["90"]["n"] <= windows["180"]["n"] <= windows["365"]["n"]:
        raise RuntimeError("V21_TEMPORAL_WINDOW_COUNTS_NOT_NESTED")
    mx04 = {}
    source_hashes = {"data_audit": sha(audit_path)}
    for task in ("mx04_scratch", "mx04_head_last"):
        path = reports / f"evaluation_outer_{task}.json"
        result = read(path)
        subsets = result.get("subsets")
        keys = [f"gap_le_{threshold}_days" for threshold in (90, 180, 365)]
        if (result.get("status") != "COMPLETED"
            or result.get("scope") != "internal_fold_outer"
            or result.get("model_task") != task
            or not isinstance(subsets, dict)
            or any(not isinstance(subsets.get(key), dict) for key in keys)):
            raise RuntimeError(f"V21_MX04_TEMPORAL_SUBSETS_MISSING:{task}")
        for threshold, key in zip(("90", "180", "365"), keys, strict=True):
            row = subsets[key]
            n, positive, negative = (row.get(field) for field in
                                     ("n", "positive", "negative"))
            if (any(type(value) is not int for value in (n, positive, negative))
                or min(n, positive, negative) < 0
                or positive + negative != n
                or n > windows[threshold]["n"]
                or row.get("status") not in {
                    "ESTIMATED", "SPARSE_CLASS_EXPLORATORY",
                    "NOT_MEASURABLE_ONE_CLASS"}):
                raise RuntimeError(f"V21_MX04_TEMPORAL_SUBSET_INVALID:{task}:{threshold}")
            for metric in ("roc_auc", "average_precision", "brier"):
                value = row.get(metric)
                if value is not None and (
                    type(value) not in (float, int)
                    or not math.isfinite(value) or not 0 <= value <= 1
                ):
                    raise RuntimeError(f"V21_MX04_TEMPORAL_METRIC_INVALID:{task}:{threshold}:{metric}")
            if ((min(positive, negative) == 0) != (row.get("roc_auc") is None)):
                raise RuntimeError(f"V21_MX04_TEMPORAL_AUC_CLASS_MISMATCH:{task}:{threshold}")
        if [subsets[key]["n"] for key in keys] != sorted(subsets[key]["n"] for key in keys):
            raise RuntimeError(f"V21_MX04_TEMPORAL_SUBSET_COUNTS_NOT_NESTED:{task}")
        mx04[task] = {key: subsets[key] for key in keys}
        source_hashes[task] = sha(path)
    return {
        "schema_version": 1,
        "status": "DESCRIPTIVE_TEMPORAL_APPENDIX_NOT_EXTERNAL_VALIDATION",
        "source_sha256": source_hashes,
        "qc_population": "all_1669_retained_real_samples_including_333_pseudo_holdout",
        "pet_mri_gap_median_days_all_qc": audit.get("pet_mri_gap_median_days"),
        "pet_vs_mri_direction_counts_all_qc": direction,
        "time_window_counts_all_qc": windows,
        "mx04_internal_outer_time_window_metrics": mx04,
        "hybrid_time_window_performance": "NOT_ESTIMATED_BY_PREDECLARED_PAIRED_EVALUATION",
        "no_new_test_inference_for_appendix": True,
        "site_scanner_sensitivity": "UNKNOWN_NO_SITE_SCANNER_FIELD_IN_PROVIDED_SPLITS",
        "external_validation": False,
    }


def amendment_payload(root: Path, out: Path) -> dict:
    prereg_path = out / "reports/fasttrack_preregistration.json"
    prereg = read(prereg_path)
    if (prereg.get("status") != "LOCKED_BEFORE_GPU"
        or prereg.get("outer_test_access_count") != 0):
        raise RuntimeError("V21_PARENT_PREREGISTRATION_INVALID")
    return {
        "schema_version": 1,
        "protocol_id": "METHOD_X_FASTTRACK_V2_POSTLOCK_AMENDMENT_1",
        "parent_preregistration_sha256": sha(prereg_path),
        "parent_protocol_id": prereg.get("protocol_id"),
        "source_sha256": {SOURCE_RELATIVE: sha(root / SOURCE_RELATIVE)},
        "selection_and_training_rules_unchanged": True,
        "outer_test_access_count_at_amendment": 0,
        "scope": "MX5_MX6_MX7_COORDINATION_AND_EVIDENCE_BINDING_ONLY",
        "repairs": ["XAI_ATLAS_SPM_CONTEXT_ACROSS_RESUME",
                    "XAI_RECEIPT_AND_AGGREGATE_VALIDATION",
                    "MX7_XAI_STATE_INVENTORY_PATH",
                    "MX7_EXPLICIT_OPTIONAL_XAI_OMISSION_GATE",
                    "PRESERVE_ORPHANED_WRITE_ONCE_TEMPS_ON_RETRY",
                    "BLOCK_AMBIGUOUS_OUTER_TEST_REINFERENCE",
                    "VERIFY_SIGNED_HYBRID_PAIRED_EFFECT_BEFORE_FREEZE",
                    "BIND_DESCRIPTIVE_TEMPORAL_APPENDIX_WITH_NO_HYBRID_SUBGROUP_CLAIM",
                    "EXPLICITLY_FORBID_OCCLUSION_FAITHFULNESS_CLAIM"],
    }


def ensure_amendment(root: Path, out: Path) -> dict:
    path = out / "reports" / AMENDMENT_NAME
    expected = amendment_payload(root, out)
    if not path.exists():
        # A post-hoc amendment after MX5 GPU work or outer access would not be
        # a preregistered repair.  It must instead start a new protocol.
        supervisor_state_path = out / "reports/fasttrack_supervisor_state.json"
        supervisor_steps = (read(supervisor_state_path).get("steps") or {}
                            if supervisor_state_path.exists() else {})
        if any(str(step).startswith(("mx5_", "mx6_", "mx7_", "mx4_outer_"))
               for step in supervisor_steps):
            raise RuntimeError("V21_AMENDMENT_TOO_LATE_CREATE_NEW_PROTOCOL")
        task_state_path = out / "reports/fasttrack_task_state.json"
        task_state = (read(task_state_path).get("tasks") or {}
                      if task_state_path.exists() else {})
        if any(task in task_state for task in ("hybrid_baseline", "hybrid_dense")):
            raise RuntimeError("V21_AMENDMENT_TOO_LATE_CREATE_NEW_PROTOCOL")
        if any((out / relative).exists() for relative in (
            "reports/smoke_baseline.json", "reports/smoke_plus_dense_skip.json",
            "inner_tasks/profile_baseline", "inner_tasks/profile_plus_dense_skip",
            "inner_tasks/hybrid_baseline", "inner_tasks/hybrid_dense",
            "reports/fasttrack_inner_lock.json",
            "reports/evaluation_outer_hybrid_pair_state.json",
            "reports/evaluation_outer_mx04_scratch_state.json",
            "reports/evaluation_outer_mx04_head_last_state.json",
        )):
            raise RuntimeError("V21_AMENDMENT_TOO_LATE_CREATE_NEW_PROTOCOL")
        check_inputs(root, out)
        immutable_json(path, expected)
    elif read(path) != expected:
        raise RuntimeError("V21_AMENDMENT_OR_SOURCE_DRIFT")
    return expected


def expected_xai_context(root: Path, out: Path) -> dict:
    ensure_amendment(root, out)
    lock = check_lock(root, out)
    prereg = read(out / "reports/fasttrack_preregistration.json")
    _, provenance = atlas_binding(root, prereg)
    selected = lock["final_task"]
    return {
        "schema_version": 1,
        "status": "LOCKED_BEFORE_MX6_ATTRIBUTION",
        "amendment_sha256": sha(out / "reports" / AMENDMENT_NAME),
        "inner_lock_sha256": sha(out / "reports/fasttrack_inner_lock.json"),
        "selected_task": selected,
        "checkpoint_sha256": lock["tasks"][selected]["checkpoint_sha256"],
        "atlas_spm_provenance": provenance,
        "xai_config_sha256": sha(root / "configs/experiments/method_x_fasttrack_v2.json"),
    }


def ensure_xai_context(root: Path, out: Path) -> dict:
    path = out / "reports" / CONTEXT_NAME
    expected = expected_xai_context(root, out)
    if not path.exists():
        if any((out / relative).exists() for relative in (
            "reports/xai_limited_state.json", "reports/xai_limited_summary.json",
            "reports/xai_limited_folds",
        )):
            raise RuntimeError("V21_UNBOUND_PRIOR_XAI_RECEIPTS_CREATE_NEW_PROTOCOL")
        immutable_json(path, expected)
    elif read(path) != expected:
        raise RuntimeError("V21_XAI_ATLAS_SPM_OR_LOCK_DRIFT")
    return expected


def validate_xai_receipts(out: Path, context: dict) -> dict | None:
    """A completed XAI result must be tied to the frozen v2.1 context."""
    reports = out / "reports"
    summary_path = reports / "xai_limited_summary.json"
    state_path = reports / "xai_limited_state.json"
    if not summary_path.exists():
        if state_path.exists() and read(state_path).get("status") == "COMPLETED":
            raise RuntimeError("V21_XAI_COMPLETED_WITHOUT_SUMMARY")
        return None
    if not state_path.exists():
        raise RuntimeError("V21_XAI_SUMMARY_WITHOUT_STATE")
    summary, state = read(summary_path), read(state_path)
    if not isinstance(summary, dict) or not isinstance(state, dict):
        raise RuntimeError("V21_XAI_SUMMARY_OR_STATE_INVALID")
    if has_nonfinite_number(summary) or has_nonfinite_number(state):
        raise RuntimeError("V21_XAI_NONFINITE_RESULT_OR_STATE")
    if (state.get("status") != "COMPLETED"
        or state.get("schema_version") != 2
        or state.get("result_sha256") != sha(summary_path)
        or state.get("lock_sha256") != context["inner_lock_sha256"]
        or state.get("source") != "inner_validation"
        or state.get("expected_n") != 40
        or state.get("expected_folds") != list(range(5))
        or state.get("gate_status") != summary.get("status")
        or summary.get("schema_version") != 2
        or summary.get("status") not in {
            "COMPLETED_EXPLORATORY_LIMITED", "BLOCKED_EXPLORATORY_GATE"}
        or summary.get("lock_sha256") != context["inner_lock_sha256"]
        or summary.get("selected_task") != context["selected_task"]
        or summary.get("provenance") != context["atlas_spm_provenance"]
        or summary.get("scope") != "inner_validation_probe_8_per_fold"
        or summary.get("attribution") != "NORMALIZED_ABSOLUTE_INPUT_GRADIENT_NOT_GRADCAM"
        or summary.get("n_images") != 40
        or summary.get("randomization_seeds") != [13, 29, 47]
        or summary.get("region_summary_ci_resampling_unit") != "fold_mean_five_folds_not_patient"
        or summary.get("not_independent_testing") is not True
        or summary.get("anatomical_or_causal_region_claim_allowed") is not False
        or summary.get("subject_level_outputs_written") is not False):
        raise RuntimeError("V21_XAI_SUMMARY_CONTEXT_MISMATCH")
    receipts = state.get("fold_receipt_sha256")
    if not isinstance(receipts, dict) or set(receipts) != {str(i) for i in range(5)}:
        raise RuntimeError("V21_XAI_FOLD_RECEIPTS_INCOMPLETE")
    if state.get("completed_folds") != receipts:
        raise RuntimeError("V21_XAI_STATE_RECEIPT_MISMATCH")
    folds = summary.get("folds")
    if not isinstance(folds, list) or len(folds) != 5:
        raise RuntimeError("V21_XAI_SUMMARY_FOLDS_INCOMPLETE")
    receipt_root = reports / "xai_limited_folds"
    if (not receipt_root.is_dir()
        or {path.name for path in receipt_root.iterdir()}
           != {f"fold_{fold}.json" for fold in range(5)}):
        raise RuntimeError("V21_XAI_RECEIPT_DIRECTORY_NOT_EXACT_FIVE")
    for fold in range(5):
        receipt_path = reports / "xai_limited_folds" / f"fold_{fold}.json"
        row = folds[fold]
        if (not receipt_path.is_file() or sha(receipt_path) != receipts[str(fold)]
            or not isinstance(row, dict)
            or read(receipt_path) != row
            or row.get("schema_version") != 1
            or row.get("fold") != fold
            or row.get("lock_sha256") != context["inner_lock_sha256"]
            or row.get("selected_task") != context["selected_task"]
            or row.get("checkpoint_sha256") != context["checkpoint_sha256"][fold]
            or row.get("state_source") != "model_state_dict"
            or row.get("sample_count") != 8
            or row.get("bootstrap_resamples") != 1000
            or row.get("randomization_seeds") != [13, 29, 47]
            or type(row.get("peak_reserved_gb")) not in (float, int)
            or not math.isfinite(row["peak_reserved_gb"])
            or not 0 <= row["peak_reserved_gb"] <= 7.5
            or row.get("aggregate_only") is not True
            or row.get("contains_subject_level_outputs") is not False
            or row.get("subject_level_outputs_written") is not False
            or not isinstance(row.get("gate"), dict)
            or not isinstance(row.get("region_attribution"), list)
            or not row["region_attribution"]
            or not isinstance(row["gate"].get("randomization_sanity"), list)
            or any(not isinstance(item, dict)
                   for item in row["gate"]["randomization_sanity"])
            or [item.get("seed") for item in row["gate"]["randomization_sanity"]]
               != [13, 29, 47]):
            raise RuntimeError(f"V21_XAI_FOLD_RECEIPT_CONTEXT_MISMATCH:{fold}")
    from src.evaluation.xai_validation import evaluate_fold_xai_gates

    gates = evaluate_fold_xai_gates(
        [{"fold": fold, **folds[fold]["gate"]} for fold in range(5)],
        expected_folds=5, required_randomization_seeds=3)
    expected_status = ("COMPLETED_EXPLORATORY_LIMITED" if gates["status"] == "COMPLETED"
                       else "BLOCKED_EXPLORATORY_GATE")
    if summary.get("gates") != gates or summary["status"] != expected_status:
        raise RuntimeError("V21_XAI_AGGREGATE_GATE_MISMATCH")
    from src.evaluation.method_x_xai_forward import aggregate_fold_regions

    if summary.get("region_summary") != aggregate_fold_regions(
        folds, bootstrap_resamples=1000):
        raise RuntimeError("V21_XAI_AGGREGATE_REGION_MISMATCH")
    return {
        "schema_version": 1,
        "status": "FIVE_FOLD_RECEIPTS_BOUND_TO_ONE_ATLAS_SPM_CONTEXT",
        "context_sha256": sha(reports / CONTEXT_NAME),
        "summary_sha256": sha(summary_path),
        "folds": [{"fold": fold,
                   "receipt_sha256": receipts[str(fold)],
                   "checkpoint_sha256": context["checkpoint_sha256"][fold]}
                  for fold in range(5)],
    }


def xai_guard(root: Path, out: Path, output_relative: Path) -> int:
    if (out / "reports" / WAIVER_NAME).exists():
        raise RuntimeError("V21_XAI_ALREADY_EXPLICITLY_OMITTED")
    context = ensure_xai_context(root, out)
    command = [sys.executable, str(root / "scripts/mx_fasttrack_v2/xai_limited.py"),
               "--project-root", str(root), "--output-root", str(output_relative),
               "--mode", "run"]
    result = subprocess.run(command, cwd=root, capture_output=True, text=True, check=False)
    if result.stdout:
        print(result.stdout, end="", flush=True)
    if result.stderr:
        print(result.stderr, end="", file=sys.stderr, flush=True)
    if result.returncode == 3:
        try:
            terminal = json.loads(result.stdout.strip().splitlines()[-1])
        except (IndexError, json.JSONDecodeError) as exc:
            raise RuntimeError("V21_XAI_EXIT_3_WITHOUT_SCIENTIFIC_GATE_RECEIPT") from exc
        if not isinstance(terminal, dict) or not (
            terminal.get("status") == "BLOCKED_EXPLORATORY_GATE"
            or (terminal.get("status") == "REUSED_EXISTING_LIMITED_XAI"
                and terminal.get("gate_status") == "BLOCKED_EXPLORATORY_GATE")
        ):
            raise RuntimeError("V21_XAI_EXIT_3_NOT_A_SCIENTIFIC_GATE")
    if ensure_xai_context(root, out) != context:
        raise RuntimeError("V21_XAI_CONTEXT_CHANGED_DURING_ATTEMPT")
    binding = validate_xai_receipts(out, context)
    if binding is not None:
        immutable_json(out / "reports" / FOLD_BINDING_NAME, binding)
    elif result.returncode == 0:
        raise RuntimeError("V21_XAI_EXIT_ZERO_WITHOUT_COMPLETED_SUMMARY")
    return result.returncode


def xai_omission_payload(root: Path, out: Path, reason: str) -> dict:
    """Bind an explicit optional-XAI omission to all already-written evidence."""
    if reason not in SKIP_REASONS:
        raise RuntimeError("V21_XAI_OMISSION_REASON_INVALID")
    ensure_amendment(root, out)
    lock = check_lock(root, out)
    reports = out / "reports"
    if ((reports / "xai_limited_summary.json").exists()
        or (reports / FOLD_BINDING_NAME).exists()):
        raise RuntimeError("V21_COMPLETED_OR_BOUND_XAI_CANNOT_BE_OMITTED")
    context_path = reports / CONTEXT_NAME
    state_path = reports / "xai_limited_state.json"
    receipt_root = reports / "xai_limited_folds"
    receipt_paths = sorted(receipt_root.iterdir()) if receipt_root.is_dir() else []
    if any(not path.is_file() or path.name not in {f"fold_{fold}.json" for fold in range(5)}
           for path in receipt_paths):
        raise RuntimeError("V21_XAI_UNEXPECTED_PARTIAL_RECEIPT")
    if (state_path.exists() or receipt_paths) and not context_path.is_file():
        raise RuntimeError("V21_UNBOUND_PARTIAL_XAI_CANNOT_BE_OMITTED")
    if receipt_paths and not state_path.is_file():
        raise RuntimeError("V21_XAI_OMISSION_RECEIPTS_WITHOUT_STATE")
    if context_path.is_file():
        context = read(context_path)
        expected_context_fields = {
            "schema_version": 1,
            "status": "LOCKED_BEFORE_MX6_ATTRIBUTION",
            "amendment_sha256": sha(reports / AMENDMENT_NAME),
            "inner_lock_sha256": sha(reports / "fasttrack_inner_lock.json"),
            "selected_task": lock["final_task"],
            "checkpoint_sha256": lock["tasks"][lock["final_task"]]["checkpoint_sha256"],
            "xai_config_sha256": sha(root / "configs/experiments/method_x_fasttrack_v2.json"),
        }
        if (not isinstance(context, dict)
            or not isinstance(context.get("atlas_spm_provenance"), dict)
            or not context["atlas_spm_provenance"]
            or {key: context.get(key) for key in expected_context_fields} != expected_context_fields
            or set(context) != set(expected_context_fields) | {"atlas_spm_provenance"}):
            raise RuntimeError("V21_XAI_OMISSION_CONTEXT_INVALID")
    if state_path.is_file():
        state = read(state_path)
        if (state.get("schema_version") != 2
            or state.get("status") not in {"STARTED", "RUNNING"}
            or state.get("lock_sha256") != sha(reports / "fasttrack_inner_lock.json")
            or state.get("source") != "inner_validation"
            or state.get("expected_n") != 40
            or state.get("expected_folds") != list(range(5))
            or not isinstance(state.get("completed_folds"), dict)):
            raise RuntimeError("V21_XAI_OMISSION_STATE_NOT_INCOMPLETE")
        completed = state["completed_folds"]
        if not set(completed) <= {str(fold) for fold in range(5)}:
            raise RuntimeError("V21_XAI_OMISSION_COMPLETED_FOLDS_INVALID")
    else:
        completed = {}
    for receipt_path in receipt_paths:
        fold = int(receipt_path.stem.removeprefix("fold_"))
        row = read(receipt_path)
        if (row.get("schema_version") != 1 or row.get("fold") != fold
            or row.get("lock_sha256") != sha(reports / "fasttrack_inner_lock.json")
            or row.get("selected_task") != lock["final_task"]
            or row.get("checkpoint_sha256") != lock["tasks"][lock["final_task"]]["checkpoint_sha256"][fold]
            or row.get("sample_count") != 8
            or row.get("subject_level_outputs_written") is not False
            or completed.get(str(fold)) not in (None, sha(receipt_path))):
            raise RuntimeError(f"V21_XAI_OMISSION_PARTIAL_RECEIPT_INVALID:{fold}")
    if set(completed) - {str(int(path.stem.removeprefix("fold_"))) for path in receipt_paths}:
        raise RuntimeError("V21_XAI_OMISSION_COMMITTED_RECEIPT_MISSING")
    return {
        "schema_version": 1,
        "status": "OPTIONAL_XAI_EXPLICITLY_OMITTED_NOT_COMPLETED",
        "reason": reason,
        "amendment_sha256": sha(reports / AMENDMENT_NAME),
        "inner_lock_sha256": sha(reports / "fasttrack_inner_lock.json"),
        "selected_task": lock["final_task"],
        "xai_context_sha256": sha(context_path) if context_path.is_file() else None,
        "xai_state_sha256": sha(state_path) if state_path.is_file() else None,
        "partial_fold_receipt_sha256": {path.stem: sha(path) for path in receipt_paths},
        "anatomical_or_causal_region_claim_allowed": False,
    }


def verify_frozen_scope(root: Path, out: Path, *, xai_skipped: bool) -> dict:
    """Check that the parent freeze represented the MX6 choice and claim ceiling."""
    reports = out / "reports"
    frozen = read(reports / "fasttrack_thesis_traceability.json")
    stage = frozen.get("status") or {}
    state_path = reports / "xai_limited_state.json"
    expected_xai = (
        "NOT_DONE_STARTED" if state_path.is_file() else "NOT_DONE_OPTIONAL"
    ) if xai_skipped else read(reports / "xai_limited_summary.json")["status"]
    if (frozen.get("scope") != "POST_HOC_INTERNAL_TECHNICAL_EVIDENCE"
        or frozen.get("immutable") is not True
        or frozen.get("cohort_label_perfectly_confounding") is not True
        or frozen.get("external_validation") is not False
        or frozen.get("amyloid_specific_or_clinical_claim_allowed") is not False
        or frozen.get("method_x_pet_input") is not False
        or frozen.get("subject_level_vectors_exported") is not False
        or stage.get("MX04_focused_two_arm") != "COMPLETED_EXPLORATORY"
        or stage.get("MX05_two_hybrid_arms") != "COMPLETED_EXPLORATORY"
        or stage.get("MX06_limited_optional") != expected_xai
        or stage.get("MX07_fasttrack") != "READY_SCOPED_FASTTRACK"
        or stage.get("MX06_original_formal") != "NOT_DONE_BY_FASTTRACK"
        or stage.get("MX07_original") != "NOT_DONE_BY_FASTTRACK"
        or stage.get("true_external_validation") != "NOT_DONE"):
        raise RuntimeError("V21_PARENT_FREEZE_SCOPE_OR_XAI_STATUS_INVALID")
    inventory = frozen.get("asset_inventory") or {}
    if not isinstance(inventory, dict):
        raise RuntimeError("V21_PARENT_FREEZE_INVENTORY_INVALID")
    if xai_skipped:
        if any(str(key).startswith("xai_limited") for key in inventory):
            raise RuntimeError("V21_OMITTED_XAI_PRESENT_IN_FROZEN_INVENTORY")
    else:
        expected = {
            "xai_limited": reports / "xai_limited_summary.json",
            "xai_limited_state": reports / "xai_limited_summary_state.json",
            **{f"xai_limited_fold_{fold}": reports / "xai_limited_folds" / f"fold_{fold}.json"
               for fold in range(5)},
        }
        for key, path in expected.items():
            if inventory.get(key) != {
                "relative_path": path.relative_to(root).as_posix(), "sha256": sha(path)
            }:
                raise RuntimeError(f"V21_XAI_FROZEN_INVENTORY_MISMATCH:{key}")
    return frozen


def freeze_guard(root: Path, out: Path, output_relative: Path,
                 *, xai_skip_reason: str | None = None) -> int:
    ensure_amendment(root, out)
    reports = out / "reports"
    xai_state = reports / "xai_limited_state.json"
    xai_summary = reports / "xai_limited_summary.json"
    context_path = reports / CONTEXT_NAME
    fold_binding_path = reports / FOLD_BINDING_NAME
    waiver_path = reports / WAIVER_NAME
    alias = reports / "xai_limited_summary_state.json"
    if xai_skip_reason is not None:
        expected_waiver = xai_omission_payload(root, out, xai_skip_reason)
        if not waiver_path.is_file() or read(waiver_path) != expected_waiver:
            raise RuntimeError("V21_XAI_OMISSION_RECEIPT_MISSING_OR_CHANGED")
        if alias.exists():
            raise RuntimeError("V21_XAI_STATE_ALIAS_PRESENT_FOR_OMITTED_XAI")
    elif waiver_path.exists():
        raise RuntimeError("V21_XAI_OMISSION_REQUIRES_EXPLICIT_SKIP_FLAG")
    elif xai_summary.exists() and xai_state.exists() and context_path.exists():
        context = ensure_xai_context(root, out)
        expected_binding = validate_xai_receipts(out, context)
        if (expected_binding is None or not fold_binding_path.is_file()
            or read(fold_binding_path) != expected_binding):
            raise RuntimeError("V21_XAI_FOLD_BINDING_MISSING_OR_CHANGED")
        # v2 freeze_report expects this spelling while its validator reads
        # xai_limited_state.json. The alias is a hash-bound semantic copy.
        immutable_json(alias, read(xai_state))
    else:
        raise RuntimeError("V21_XAI_NOT_TERMINAL_RESUME_OR_EXPLICITLY_OMIT")
    temporal_payload = temporal_appendix_payload(out)
    command = [sys.executable, str(root / "scripts/mx_fasttrack_v2/freeze_report.py"),
               "--project-root", str(root), "--output-root", str(output_relative)]
    result = subprocess.run(command, cwd=root, check=False)
    if result.returncode:
        return result.returncode
    final = reports / "fasttrack_thesis_traceability.json"
    if not final.is_file():
        raise RuntimeError("V21_PARENT_FREEZE_NOT_SCOPED_READY")
    frozen = verify_frozen_scope(root, out, xai_skipped=xai_skip_reason is not None)
    if xai_skip_reason is None:
        expected_binding = validate_xai_receipts(out, ensure_xai_context(root, out))
        if expected_binding is None or read(fold_binding_path) != expected_binding:
            raise RuntimeError("V21_XAI_FOLD_BINDING_DRIFT_DURING_FREEZE")
    else:
        if not waiver_path.is_file() or read(waiver_path) != xai_omission_payload(root, out, xai_skip_reason):
            raise RuntimeError("V21_XAI_OMISSION_DRIFT_DURING_FREEZE")
    if xai_skip_reason is None and read(alias) != read(xai_state):
        raise RuntimeError("V21_XAI_STATE_ALIAS_DRIFT_DURING_FREEZE")
    if temporal_appendix_payload(out) != temporal_payload:
        raise RuntimeError("V21_TEMPORAL_EVIDENCE_DRIFT_DURING_FREEZE")
    temporal_appendix = reports / TEMPORAL_APPENDIX_NAME
    immutable_json(temporal_appendix, temporal_payload)
    attestation = {
        "schema_version": 1,
        "status": "READY_SCOPED_FASTTRACK_V21_AMENDMENT",
        "amendment_sha256": sha(reports / AMENDMENT_NAME),
        "parent_freeze_sha256": sha(final),
        "xai_context_sha256": sha(context_path) if context_path.exists() else None,
        "xai_fold_binding_sha256": sha(fold_binding_path) if fold_binding_path.exists() else None,
        "xai_canonical_state_sha256": sha(xai_state) if xai_state.exists() else None,
        "xai_state_alias_sha256": sha(alias) if alias.exists() else None,
        "xai_explicit_omission_sha256": sha(waiver_path) if waiver_path.exists() else None,
        "temporal_appendix_sha256": sha(temporal_appendix),
        "xai_limited_status": frozen["status"]["MX06_limited_optional"],
        "xai_region_claim_allowed": False,
        "xai_occlusion_faithfulness_claim_allowed": False,
        "xai_occlusion_limitation": (
            "POSITIVE_CLASS_MEAN_OVER_MIXED_LABELS_NOT_A_CLASS_TARGETED_PAIRED_GATE"
        ),
        "original_MX05_MX06_MX07": "NOT_DONE_BY_FASTTRACK",
    }
    immutable_json(reports / "fasttrack_v21_final_attestation.json", attestation)
    return 0


class Supervisor(V2Supervisor):
    def __init__(self, root: Path, output_relative: Path, package_root: Path,
                 *, xai_skip_reason: str | None = None) -> None:
        super().__init__(root, output_relative, package_root)
        self.xai_skip_reason = xai_skip_reason

    def run_step(self, name: str, command: list[str], *, allowed: set[int] | None = None) -> int:
        writers = foreign_method_x_writers(self.root)
        if writers:
            raise RuntimeError(f"V21_FOREIGN_METHOD_X_WRITER_ACTIVE:{[row['pid'] for row in writers]}")
        require_unambiguous_outer_resume(self.out, name)
        if name in {"mx4_outer_mx04_scratch", "mx4_outer_mx04_head_last",
                    "mx5_selected_outer_receipt", "mx7_freeze_scoped_report"}:
            verify_hybrid_pair_effect(self.out)
        if name == "mx5_common_inner_lock":
            preserved = preserve_orphaned_write_once_temps(
                self.reports, ("fasttrack_inner_lock.json",))
        elif name == "mx7_freeze_scoped_report":
            preserved = preserve_orphaned_write_once_temps(self.reports, (
                "fasttrack_negative_results_ledger.json",
                "paper_method_x_methods.md", "paper_method_x_results.md",
                "paper_method_x_discussion.md", "fasttrack_thesis_traceability.json",
            ))
        else:
            preserved = []
        for path in preserved:
            print(json.dumps({"supervisor_step": name,
                              "preserved_uncommitted_temp": str(path)},
                             ensure_ascii=False), flush=True)
        return super().run_step(name, command, allowed=allowed)

    def mx4(self) -> None:
        super().mx4()
        ensure_amendment(self.root, self.out)

    def mx6(self) -> None:
        if (self.reports / WAIVER_NAME).exists():
            raise RuntimeError("V21_XAI_ALREADY_EXPLICITLY_OMITTED")
        self.mx5()
        self.run_step("mx6_atlas_spm_audit", self.py(
            "scripts/mx_fasttrack_v2/xai_limited.py",
            "--project-root", self.root, "--output-root", self.output_relative,
            "--mode", "audit"))
        code = self.run_step("mx6_limited_xai", [
            sys.executable, str(Path(__file__).resolve()), "xai-guard",
            "--project-root", str(self.root), "--output-root", str(self.output_relative),
        ], allowed={0, 3})
        if code == 3:
            state_path = self.reports / "xai_limited_state.json"
            state = read(state_path) if state_path.is_file() else {}
            if (state.get("status") != "COMPLETED"
                or state.get("gate_status") != "BLOCKED_EXPLORATORY_GATE"):
                self.state["steps"]["mx6_limited_xai"]["status"] = "PAUSED_OR_FAILED_RESUMABLE"
                from scripts.mx_fasttrack_v2.supervise import atomic
                atomic(self.state_path, self.state)
                raise RuntimeError("MX6_EXIT_3_IS_NOT_A_TERMINAL_SCIENTIFIC_GATE_RESULT")

    def mx7(self) -> None:
        if self.xai_skip_reason is None:
            if (self.reports / WAIVER_NAME).exists():
                raise RuntimeError("V21_XAI_ALREADY_EXPLICITLY_OMITTED")
            self.mx6()
        else:
            self.mx5()
            verify_hybrid_pair_effect(self.out)
            temporal_appendix_payload(self.out)
            immutable_json(self.reports / WAIVER_NAME,
                           xai_omission_payload(self.root, self.out, self.xai_skip_reason))
        command = [
            sys.executable, str(Path(__file__).resolve()), "freeze-guard",
            "--project-root", str(self.root), "--output-root", str(self.output_relative),
        ]
        if self.xai_skip_reason is not None:
            command.extend(["--xai-skip-reason", self.xai_skip_reason])
        self.run_step("mx7_freeze_scoped_report", command)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("target", choices=("mx4", "mx5", "mx6", "mx7", "all",
                                           "xai-guard", "freeze-guard"))
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--package-root", type=Path, default=DEFAULT_PACKAGE)
    parser.add_argument("--skip-optional-xai", action="store_true")
    parser.add_argument("--xai-skip-reason", choices=SKIP_REASONS)
    args = parser.parse_args()
    root = args.project_root.resolve()
    out = (root / args.output_root).resolve()
    if root != ROOT.resolve() or not out.is_relative_to(root) or out == root:
        raise RuntimeError("V21_PROJECT_OR_OUTPUT_ROOT_INVALID")
    if args.target in {"xai-guard", "freeze-guard"}:
        require_internal_coordinator_parent(root)
        if args.target == "xai-guard" and (args.skip_optional_xai or args.xai_skip_reason):
            raise RuntimeError("V21_XAI_GUARD_SKIP_FLAG_INVALID")
        return (xai_guard(root, out, args.output_root) if args.target == "xai-guard"
                else freeze_guard(root, out, args.output_root,
                                  xai_skip_reason=args.xai_skip_reason))
    if args.skip_optional_xai != (args.xai_skip_reason is not None):
        raise RuntimeError("V21_XAI_SKIP_REQUIRES_EXPLICIT_REASON")
    if args.skip_optional_xai and args.target not in {"mx7", "all"}:
        raise RuntimeError("V21_XAI_SKIP_ONLY_FOR_MX7_OR_ALL")
    writers = foreign_method_x_writers(root)
    if writers:
        raise RuntimeError(f"V21_FOREIGN_METHOD_X_WRITER_ACTIVE:{[row['pid'] for row in writers]}")
    with coordinator_lock(out / "reports"):
        prior_parent = os.environ.get(INTERNAL_PARENT_ENV)
        os.environ[INTERNAL_PARENT_ENV] = str(os.getpid())
        try:
            supervisor = Supervisor(root, args.output_root, args.package_root,
                                    xai_skip_reason=args.xai_skip_reason)
            target = "mx7" if args.target == "all" else args.target
            getattr(supervisor, target)()
            print(json.dumps({"status": f"{target.upper()}_V21_SUPERVISOR_TARGET_COMPLETED",
                              "output_root": str(out)}, ensure_ascii=False))
        finally:
            if prior_parent is None:
                os.environ.pop(INTERNAL_PARENT_ENV, None)
            else:
                os.environ[INTERNAL_PARENT_ENV] = prior_parent
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError, KeyError, TypeError) as exc:
        print(json.dumps({"status": "BLOCKED", "reason": str(exc)}, ensure_ascii=False), file=sys.stderr)
        internal = len(sys.argv) > 1 and sys.argv[1] in {"xai-guard", "freeze-guard"}
        raise SystemExit(4 if internal else 3)
