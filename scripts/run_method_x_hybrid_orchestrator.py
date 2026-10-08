"""Autonomous MX-05 AMUSE-X-Hybrid3D ablation and replication graph."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.smoke_method_x_backbone import smoke, validate_hybrid_component_evidence
from src.method_y_revival.strict_yaml import load_unique_yaml
from src.training.method_x_autosupervisor import enforce_autonomous_supervision, evaluate_autonomous_supervision, run_supervised_process
from src.training.method_x_loop_guard import record_failure, record_success
from src.utils.io_helpers import atomic_write_json


DEFAULT_OUTPUT_ROOT = ROOT / "results/final_results/method_x_model_battery_v3_depth_and_diversity_20260729"
DEFAULT_CONFIG = ROOT / "configs/experiments/method_x_hybrid_v1.yaml"
LOOP_STATE = "reports/method_x_loop_guard.json"
EXPECTED_ABLATIONS = ["baseline", "plus_dense_skip", "plus_large_kernel", "plus_attention", "plus_a3", "plus_a4", "full_model"]
EXPECTED_COMPONENTS = {
    "stem": "shared_conv3d_norm_gelu",
    "residual_stages": True,
    "capped_dense_skip_reuse": True,
    "depthwise_large_kernel_se": True,
    "window_attention": "low_width_optional",
    "multiscale_gated_fusion": True,
    "a3_tabular_projection": "linear_then_optional_mlp",
    "cross_attention_bottleneck": True,
    "a4_channel_gate": "phenotype_only_causal_inspired",
    "calibrated_head": "temperature_scaling_inner_validation_only",
    "a6_claim_validator": True,
}
EXPECTED_ATTENTION = {"enabled": True, "heads": 2, "window_size": 4}
EXPECTED_FALLBACK = {"name": "amuse_x_hybrid_lite", "remove_attention_if_smoke_fails": True}
EXPECTED_CLAIM_BOUNDARY = {
    "pet_input": False,
    "diagnosis": False,
    "causal_region_claim": False,
    "external_generalization": False,
}
EXPECTED_PROMOTION = {
    "all_five_folds_finalized": True,
    "sanity_checks_required": True,
    "bootstrap_resamples": 1000,
    "roc_auc_delta_min": 0.02,
    "brier_degradation_max": 0.01,
    "ece_degradation_max": 0.02,
    "cohort_probe_auc_increase_max": 0.0,
    "memory_hard_stop_gb": 7.5,
}
EXPECTED_EXACT_RESUME = {
    "required": True,
    "checkpoint_staging_root": r"C:\Users\USER\AppData\Local\AMUSE-X\checkpoint_staging",
    "checkpoint_full_cadence": 1,
    "checkpoint_h_mirror": "epoch",
    "resume_replay_policy": "deterministic",
    "storage_policy": "configs/policy/method_x_storage_locality_v1.yaml",
    "no_resume_forbidden": True,
}
EXPECTED_THROUGHPUT_PROFILE = {
    "evidence_artifact": "results/method_x_registry/MX05_HYBRID_THROUGHPUT_PROFILE.json",
    "receipts_artifact": "results/method_x_registry/MX05_HYBRID_THROUGHPUT_PROFILE_RECEIPTS.json",
    "profile_variant": "full_model",
    "profile_limit": 64,
    "folds": 1,
    "epochs": 1,
    "seed": 13,
    "loader_trials": 2,
    "metric_equivalence_tolerance": 0.000001,
    "checkpoint_soft_cap_gb": 7.0,
    "memory_hard_stop_gb": 7.5,
    "persistent_workers": False,
}
EXPECTED_RUNTIME_READINESS = {
    "evidence_artifact": "results/method_x_registry/MX05_RUNTIME_READINESS.json",
    "junit_artifact": "results/method_x_registry/MX05_RUNTIME_READINESS_TESTS.xml",
}
EXPECTED_OUTER_FINALIZATION = {
    "canonical_seed": 13,
    "destination_root": "final_model_v1",
    "logical_evaluation_count": 1,
    "journal_batch_size": 1,
    "outer_used_for_selection": False,
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"INVALID_JSON:{path.name}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"INVALID_JSON_OBJECT:{path.name}")
    return value


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()


HYBRID_PROFILE_SOURCE_PATHS = (
    "scripts/train_3d_mri.py",
    "scripts/run_method_x_hybrid_orchestrator.py",
    "scripts/profile_mx05_hybrid_throughput.py",
    "src/models/backbones/backbone_factory.py",
    "src/models/backbones/lightweight_3d.py",
    "src/models/backbones/amuse_x_hybrid3d.py",
)


def validate_hybrid_throughput_profile(
    profile: Mapping[str, Any],
    *,
    config: Mapping[str, Any],
    root: Path = ROOT,
) -> list[str]:
    """Validate the prespecified, Hybrid-family-specific execution receipt."""

    blockers: list[str] = []
    body = {key: value for key, value in profile.items() if key != "integrity_sha256"}
    if profile.get("integrity_sha256") != _canonical_sha(body):
        blockers.append("HYBRID_THROUGHPUT_PROFILE_INTEGRITY_INVALID")
    if (
        profile.get("schema_version") != 1
        or profile.get("status") != "COMPLETED"
        or profile.get("artifact") != "mx05_hybrid_throughput_profile"
        or profile.get("family") != "amuse_x_hybrid3d"
        or profile.get("profile_variant") != "full_model"
        or profile.get("profile_only_never_selection_evidence") is not True
    ):
        blockers.append("HYBRID_THROUGHPUT_PROFILE_HEADER_INVALID")
    binding = profile.get("source_binding")
    if not isinstance(binding, Mapping):
        blockers.append("HYBRID_THROUGHPUT_PROFILE_BINDING_MISSING")
    else:
        config_path = root / "configs/experiments/method_x_hybrid_v1.yaml"
        if not config_path.is_file() or binding.get("config_sha256") != _sha(config_path):
            blockers.append("HYBRID_THROUGHPUT_PROFILE_CONFIG_DRIFT")
        sources = binding.get("source_sha256")
        if not isinstance(sources, Mapping):
            blockers.append("HYBRID_THROUGHPUT_PROFILE_SOURCE_BINDING_INVALID")
        else:
            for relative in HYBRID_PROFILE_SOURCE_PATHS:
                path = root / relative
                if not path.is_file() or sources.get(relative) != _sha(path):
                    blockers.append(
                        f"HYBRID_THROUGHPUT_PROFILE_SOURCE_DRIFT:{relative}"
                    )
    execution = profile.get("execution_contract")
    if not isinstance(execution, Mapping):
        blockers.append("HYBRID_THROUGHPUT_EXECUTION_CONTRACT_MISSING")
        return sorted(set(blockers))
    if execution.get("checkpoint") not in {"on", "off"}:
        blockers.append("HYBRID_THROUGHPUT_CHECKPOINT_SETTING_INVALID")
    loader = execution.get("loader")
    if not isinstance(loader, Mapping):
        blockers.append("HYBRID_THROUGHPUT_LOADER_CONTRACT_INVALID")
    else:
        if loader.get("num_workers") not in {2, 4}:
            blockers.append("HYBRID_THROUGHPUT_WORKER_COUNT_INVALID")
        if (
            loader.get("persistent_workers") is not False
            or loader.get("pin_memory") is not True
            or loader.get("prefetch_factor") != 2
        ):
            blockers.append("HYBRID_THROUGHPUT_LOADER_CONTRACT_INVALID")
    if execution.get("exact_resume") != config.get("exact_resume"):
        blockers.append("HYBRID_THROUGHPUT_EXACT_RESUME_DRIFT")
    yellow = execution.get("yellow_gates")
    if (
        not isinstance(yellow, Mapping)
        or set(yellow) != {"torch_compile", "channels_last"}
        or any(not str(value).startswith("DROPPED:") for value in yellow.values())
    ):
        blockers.append("HYBRID_THROUGHPUT_YELLOW_GATE_INVALID")
    equivalence = profile.get("metric_equivalence")
    try:
        maximum = float((equivalence or {}).get("max_abs_diff"))
        tolerance = float((equivalence or {}).get("tolerance"))
    except (TypeError, ValueError, OverflowError):
        maximum, tolerance = float("nan"), float("nan")
    expected_tolerance = float(
        (config.get("throughput_profile") or {}).get(
            "metric_equivalence_tolerance", -1
        )
    )
    if (
        not math.isfinite(maximum)
        or not math.isfinite(tolerance)
        or tolerance != expected_tolerance
        or maximum > tolerance
    ):
        blockers.append("HYBRID_THROUGHPUT_METRIC_EQUIVALENCE_FAILED")
    memory = profile.get("selected_memory")
    try:
        allocated = float((memory or {}).get("allocated_peak_gb"))
        reserved = float((memory or {}).get("reserved_peak_gb"))
    except (TypeError, ValueError, OverflowError):
        allocated, reserved = float("nan"), float("nan")
    if (
        not math.isfinite(allocated)
        or not math.isfinite(reserved)
        or allocated < 0
        or reserved < allocated
        or reserved >= float(config.get("training_vram_hard_stop_gb", 0))
    ):
        blockers.append("HYBRID_THROUGHPUT_SELECTED_VRAM_INVALID")
    application = profile.get("hybrid_application")
    if (
        not isinstance(application, Mapping)
        or application.get("family") != "amuse_x_hybrid3d"
        or application.get("profile_variant") != "full_model"
        or application.get("execution_contract") != execution
    ):
        blockers.append("HYBRID_THROUGHPUT_APPLICATION_INVALID")
    return sorted(set(blockers))


def load_hybrid_execution_contract(
    config: Mapping[str, Any], *, root: Path = ROOT
) -> dict[str, Any]:
    block = config.get("throughput_profile")
    if not isinstance(block, Mapping):
        raise RuntimeError("HYBRID_THROUGHPUT_PROFILE_CONFIG_MISSING")
    path = (root / str(block.get("evidence_artifact") or "")).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as exc:
        raise RuntimeError("HYBRID_THROUGHPUT_PROFILE_PATH_INVALID") from exc
    profile = _read(path)
    blockers = validate_hybrid_throughput_profile(profile, config=config, root=root)
    if blockers:
        raise RuntimeError("HYBRID_THROUGHPUT_PROFILE_INVALID:" + ",".join(blockers))
    return dict(profile["execution_contract"])


EFFECTIVE_FIVE = (
    "resnet10_3d",
    "resnet18_3d",
    "resnet34_3d",
    "densenet121_3d",
    "mednext_3d_s",
)
EXCLUDED_BACKBONES = ("resnet50_3d", "efficientnet3d_b0")
TERMINAL_TRAINING_STATUSES = frozenset(
    {
        "COMPLETED",
        "COMPLETED_VALIDATED_REPAIRED",
        "COMPLETED_VALIDATED_RECONCILED",
    }
)


def _baseline_terminal(output_root: Path) -> bool:
    try:
        manifest = _read(output_root / "battery_manifest.json")
    except RuntimeError:
        return False
    rows = manifest.get("models")
    if manifest.get("status") != "COMPLETED":
        return False
    if not isinstance(rows, list) or len(rows) != 7:
        return False
    by_name: dict = {}
    for row in rows:
        if not isinstance(row, Mapping):
            return False
        name = str(row.get("backbone") or "")
        if not name or name in by_name:
            return False
        by_name[name] = row
    if set(by_name) != set(EFFECTIVE_FIVE) | set(EXCLUDED_BACKBONES):
        return False
    for name in EFFECTIVE_FIVE:
        row = by_name[name]
        training = row.get("training")
        if row.get("status") != "COMPLETED":
            return False
        # See the MX-04 orchestrator: a truthy non-mapping ``training`` must be
        # rejected, not dereferenced.
        if not isinstance(training, Mapping):
            return False
        if str(training.get("status") or "") not in TERMINAL_TRAINING_STATUSES:
            return False
    for name in EXCLUDED_BACKBONES:
        row = by_name[name]
        training = row.get("training")
        if row.get("status") != "PROTOCOL_EXCLUDED_NOT_STARTED":
            return False
        if not isinstance(training, Mapping):
            return False
        if training.get("status") != "PROTOCOL_EXCLUDED_NOT_STARTED":
            return False
    return True


def _upstream_terminal(output_root: Path, stage: str, allowed: set[str]) -> bool:
    path = output_root / "reports" / f"{stage}.json"
    if not path.is_file():
        return False
    try:
        return _read(path).get("status") in allowed
    except RuntimeError:
        return False


def _protocol(battery: Mapping[str, Any]) -> dict[str, Any]:
    locked = battery.get("locked_protocol") if isinstance(battery.get("locked_protocol"), Mapping) else {}
    common = locked.get("common") if isinstance(locked.get("common"), Mapping) else {}
    return {
        "folds": int(locked.get("folds", 5)),
        "epochs": int(locked.get("epochs", 100)),
        "resolution": int(locked.get("resolution", 128)),
        "seed": int(locked.get("seed", 13)),
        "cache_sha256": str((battery.get("locked_inputs") or {}).get("cache_sha256", "")),
        "spm_snapshot_sha256": str((battery.get("spm_gate_snapshot") or {}).get("snapshot_sha256", "")),
        "common": dict(common),
    }


# Early-track 033: the YAML plus_* definitions are NON-cumulative isolated
# toggles (e.g. plus_large_kernel has dense_skip False), despite the additive
# names. This matrix pins that reading so plan 021 never re-litigates it.
ISOLATED_ABLATION_FLAG_MATRIX = {
    "baseline": {"dense_skip": False, "large_kernel": False, "attention": False, "a3_alignment": False, "a4_channel_gate": False},
    "plus_dense_skip": {"dense_skip": True, "large_kernel": False, "attention": False, "a3_alignment": False, "a4_channel_gate": False},
    "plus_large_kernel": {"dense_skip": False, "large_kernel": True, "attention": False, "a3_alignment": False, "a4_channel_gate": False},
    "plus_attention": {"dense_skip": False, "large_kernel": False, "attention": True, "a3_alignment": False, "a4_channel_gate": False},
    "plus_a3": {"dense_skip": False, "large_kernel": False, "attention": False, "a3_alignment": True, "a4_channel_gate": False},
    "plus_a4": {"dense_skip": False, "large_kernel": False, "attention": False, "a3_alignment": False, "a4_channel_gate": True},
    "full_model": {"dense_skip": True, "large_kernel": True, "attention": True, "a3_alignment": True, "a4_channel_gate": True},
}


def count_model_parameters_staged(model: Any) -> int:
    """STAGED post-gate parameter counter (no repo-torch import at call time).

    Early-track 033: implemented but NOT wired into the launch or audit
    path -- the factory is allowlisted, so activation waits for plan 021
    post-MX-03. Counts parameters of any object exposing .parameters()
    (fake-testable).
    """
    total = 0
    parameters = model.parameters() if callable(getattr(model, "parameters", None)) else model
    for tensor in parameters:
        count = getattr(tensor, "numel", None)
        total += int(count()) if callable(count) else int(count)
    return total


def validate_hybrid_smoke_report(
    report: Mapping[str, Any],
    *,
    expected_variant: str,
    parameter_budget_max: int,
    vram_budget_gb: float,
) -> list[str]:
    """Fail closed on incomplete/non-finite smoke and parameter evidence."""

    blockers: list[str] = []
    if str(report.get("status")) != "PASS":
        blockers.append(f"HYBRID_SMOKE_STATUS_NOT_PASS:{expected_variant}")
    if str(report.get("hybrid_variant")) != str(expected_variant):
        blockers.append(f"HYBRID_SMOKE_VARIANT_MISMATCH:{expected_variant}")
    try:
        resolution = int(report.get("resolution", -1))
    except (TypeError, ValueError, OverflowError):
        resolution = -1
    if str(report.get("device")) != "cuda" or resolution != 128:
        blockers.append(f"HYBRID_SMOKE_EXECUTION_CONTRACT_INVALID:{expected_variant}")
    if str(report.get("model_scope")) != "FULL_CLASSIFIER_WITH_TABULAR_FUSION":
        blockers.append(f"HYBRID_SMOKE_NOT_FULL_MODEL:{expected_variant}")
    component_errors = validate_hybrid_component_evidence(
        report.get("component_evidence"), expected_variant=expected_variant
    )
    blockers.extend(
        f"HYBRID_SMOKE_COMPONENT_EVIDENCE_INVALID:{expected_variant}:{error}"
        for error in component_errors
    )
    if report.get("canonical_tabular_features") != [
        "age",
        "sex",
        "education_years",
        "apoe4_carrier",
        "mmse_base",
    ]:
        blockers.append(f"HYBRID_SMOKE_TABULAR_FEATURES_INVALID:{expected_variant}")
    if report.get("output_shape") != [1]:
        blockers.append(f"HYBRID_SMOKE_OUTPUT_SHAPE_INVALID:{expected_variant}")
    if report.get("output_finite") is not True:
        blockers.append(f"HYBRID_SMOKE_OUTPUT_NONFINITE:{expected_variant}")
    count = report.get("total_parameter_count")
    if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
        blockers.append(f"HYBRID_PARAMETER_COUNT_MISSING_OR_INVALID:{expected_variant}")
    elif count > int(parameter_budget_max):
        blockers.append(f"HYBRID_PARAMETER_BUDGET_EXCEEDED:{expected_variant}:{count}")
    memory_values: dict[str, float] = {}
    for key in ("max_memory_allocated_gb", "max_memory_reserved_gb"):
        try:
            value = float(report[key])
        except (KeyError, TypeError, ValueError, OverflowError):
            value = float("nan")
        memory_values[key] = value
        if not math.isfinite(value) or value < 0:
            blockers.append(f"HYBRID_SMOKE_VRAM_INVALID:{expected_variant}:{key}")
        elif value >= float(vram_budget_gb):
            blockers.append(f"HYBRID_SMOKE_VRAM_BUDGET_EXCEEDED:{expected_variant}:{key}")
    allocated = memory_values.get("max_memory_allocated_gb", float("nan"))
    reserved = memory_values.get("max_memory_reserved_gb", float("nan"))
    if math.isfinite(allocated) and math.isfinite(reserved) and allocated > reserved:
        blockers.append(f"HYBRID_SMOKE_VRAM_ORDER_INVALID:{expected_variant}")
    return blockers


def _validate_config(config: Mapping[str, Any]) -> list[str]:
    blockers: list[str] = []
    for key, expected in (
        ("schema_version", 1),
        ("protocol_name", "amuse_x_hybrid3d_v1"),
        ("input_mode", "mri_plus_tabular"),
        ("resolution", 128),
        ("folds", 5),
        ("seed", 13),
    ):
        actual = config.get(key)
        if isinstance(expected, int):
            matches = isinstance(actual, int) and not isinstance(actual, bool) and actual == expected
        else:
            matches = actual == expected
        if not matches:
            blockers.append(f"HYBRID_CONFIG_MISMATCH:{key}")
    if config.get("backbone_name") != "amuse_x_hybrid3d":
        blockers.append("HYBRID_BACKBONE_NAME_MISMATCH")
    claim_boundary = config.get("claim_boundary")
    if isinstance(claim_boundary, Mapping):
        if claim_boundary.get("causal_region_claim") is not False:
            blockers.append("HYBRID_CAUSAL_REGION_CLAIM_NOT_FENCED")
        if claim_boundary.get("diagnosis") is not False:
            blockers.append("HYBRID_DIAGNOSIS_CLAIM_NOT_FENCED")
        if claim_boundary.get("external_generalization") is not False:
            blockers.append("HYBRID_EXTERNAL_CLAIM_NOT_FENCED")
    if claim_boundary != EXPECTED_CLAIM_BOUNDARY:
        blockers.append("HYBRID_CLAIM_BOUNDARY_MISMATCH")
    definitions = config.get("variant_definitions")
    if isinstance(definitions, Mapping):
        for variant, expected_flags in ISOLATED_ABLATION_FLAG_MATRIX.items():
            actual = definitions.get(variant)
            if not isinstance(actual, Mapping):
                continue
            actual_flags = {key: bool(actual.get(key)) for key in expected_flags}
            if actual_flags != expected_flags:
                blockers.append(f"HYBRID_ABLATION_MATRIX_MISMATCH:{variant}")
    features = config.get("tabular_features", [])
    if not isinstance(features, list) or list(features) != ["age", "sex", "education_years", "apoe4_carrier", "mmse_base"]:
        blockers.append("HYBRID_TABULAR_FEATURE_ORDER_MISMATCH")
    claim_boundary = config.get("claim_boundary", {})
    if not isinstance(claim_boundary, Mapping) or claim_boundary.get("pet_input") is not False:
        blockers.append("HYBRID_PET_INPUT_BOUNDARY_FAILED")
    ablations = config.get("ablation_order", [])
    if not isinstance(ablations, list) or list(ablations) != EXPECTED_ABLATIONS:
        blockers.append("HYBRID_ABLATION_ORDER_INCOMPLETE")
    if config.get("inner_selection_only") is not True:
        blockers.append("HYBRID_INNER_SELECTION_POLICY_MISSING")
    seeds = config.get("replication_seeds", [])
    try:
        seeds_ok = [int(value) for value in seeds] == [13, 29, 47]
    except (TypeError, ValueError):
        seeds_ok = False
    if not seeds_ok:
        blockers.append("HYBRID_REPLICATION_SEEDS_MISMATCH")
    max_attempts = config.get("max_same_failure_attempts")
    if isinstance(max_attempts, bool) or not isinstance(max_attempts, int) or max_attempts < 1:
        blockers.append("HYBRID_LOOP_GUARD_MAX_ATTEMPTS_INVALID")
    definitions = config.get("variant_definitions")
    if not isinstance(definitions, Mapping) or set(str(key) for key in definitions) != set(EXPECTED_ABLATIONS):
        blockers.append("HYBRID_VARIANT_DEFINITIONS_INCOMPLETE")
    elif dict(definitions) != ISOLATED_ABLATION_FLAG_MATRIX:
        blockers.append("HYBRID_VARIANT_DEFINITIONS_MISMATCH")
    try:
        budget_ok = int(config.get("parameter_budget_max", 0)) == 20_000_000
    except (TypeError, ValueError):
        budget_ok = False
    if not budget_ok:
        blockers.append("HYBRID_PARAMETER_BUDGET_INVALID")
    try:
        smoke_budget = float(config.get("smoke_vram_budget_gb"))
    except (TypeError, ValueError, OverflowError):
        smoke_budget = float("nan")
    if not math.isfinite(smoke_budget) or smoke_budget != 6.5:
        blockers.append("HYBRID_SMOKE_VRAM_BUDGET_INVALID")
    if config.get("components") != EXPECTED_COMPONENTS:
        blockers.append("HYBRID_COMPONENT_CONTRACT_MISMATCH")
    if config.get("attention") != EXPECTED_ATTENTION:
        blockers.append("HYBRID_ATTENTION_CONTRACT_MISMATCH")
    if config.get("fallback") != EXPECTED_FALLBACK:
        blockers.append("HYBRID_FALLBACK_CONTRACT_MISMATCH")
    if config.get("max_same_failure_attempts") != 2 or isinstance(config.get("max_same_failure_attempts"), bool):
        blockers.append("HYBRID_LOOP_GUARD_POLICY_MISMATCH")
    if config.get("training_vram_hard_stop_gb") != 7.5:
        blockers.append("HYBRID_TRAINING_VRAM_HARD_STOP_MISMATCH")
    for key, expected, code in (
        ("promotion", EXPECTED_PROMOTION, "HYBRID_PROMOTION_CONTRACT_MISMATCH"),
        ("exact_resume", EXPECTED_EXACT_RESUME, "HYBRID_EXACT_RESUME_CONTRACT_MISMATCH"),
        (
            "throughput_profile",
            EXPECTED_THROUGHPUT_PROFILE,
            "HYBRID_THROUGHPUT_PROFILE_CONTRACT_MISMATCH",
        ),
        (
            "runtime_readiness",
            EXPECTED_RUNTIME_READINESS,
            "HYBRID_RUNTIME_READINESS_CONTRACT_MISMATCH",
        ),
        (
            "outer_finalization",
            EXPECTED_OUTER_FINALIZATION,
            "HYBRID_OUTER_FINALIZATION_CONTRACT_MISMATCH",
        ),
    ):
        if config.get(key) != expected:
            blockers.append(code)
    return blockers


# Hybrid-specific flags emitted by the orchestrator. The audit below requires
# every one to remain registered by the shared trainer; removing a parser
# binding therefore fails closed before launch.
ORCHESTRATOR_ONLY_FLAGS = (
    "--hybrid_variant",
    "--hybrid_attention_heads",
    "--hybrid_attention_window_size",
    "--parameter_budget_max",
)


def _trainer_flag_subset_blockers(*, root: Path = ROOT) -> list[str]:
    import ast as _ast

    parser_flags: set[str] = set()
    for relative in (
        "scripts/train_3d_mri.py",
        "src/training/checkpoint_staging.py",
    ):
        try:
            tree = _ast.parse((root / relative).read_text(encoding="utf-8"))
        except (OSError, UnicodeError, SyntaxError):
            return ["TRAINER_PARSER_UNREADABLE"]
        for node in _ast.walk(tree):
            if not isinstance(node, _ast.Call):
                continue
            func = node.func
            if not isinstance(func, _ast.Attribute) or func.attr != "add_argument":
                continue
            if not node.args or not isinstance(node.args[0], _ast.Constant):
                continue
            value = node.args[0].value
            if isinstance(value, str) and value.startswith("--"):
                parser_flags.add(value)
    missing = set(ORCHESTRATOR_ONLY_FLAGS) - parser_flags
    if missing:
        return [f"TRAINER_FLAGS_ABSENT:{','.join(sorted(flag.lstrip('-') for flag in missing))}"]
    return []


def build_hybrid_command(
    *,
    root: Path,
    output_dir: Path,
    variant: str,
    seed: int,
    selection_only: bool,
    protocol: Mapping[str, Any],
    config: Mapping[str, Any],
    execution_contract: Mapping[str, Any] | None = None,
) -> list[str]:
    common = protocol["common"]
    exact_resume = config.get("exact_resume")
    if not isinstance(exact_resume, Mapping):
        raise RuntimeError("HYBRID_EXACT_RESUME_CONFIG_MISSING")
    if execution_contract is None:
        execution_contract = {
            "checkpoint": "on",
            "loader": {
                "num_workers": 2,
                "persistent_workers": False,
                "prefetch_factor": 2,
                "pin_memory": True,
            },
            "exact_resume": dict(exact_resume),
            "yellow_gates": {
                "torch_compile": "DROPPED:not profile-authorized",
                "channels_last": "DROPPED:not profile-authorized",
            },
        }
    loader = execution_contract.get("loader")
    resume = execution_contract.get("exact_resume")
    if not isinstance(loader, Mapping) or not isinstance(resume, Mapping):
        raise RuntimeError("HYBRID_EXECUTION_CONTRACT_INVALID")
    if dict(resume) != dict(exact_resume):
        raise RuntimeError("HYBRID_EXECUTION_RESUME_CONTRACT_DRIFT")
    if loader.get("persistent_workers") is not False:
        raise RuntimeError("HYBRID_PERSISTENT_WORKERS_FORBIDDEN")
    checkpoint = str(execution_contract.get("checkpoint") or "")
    if checkpoint not in {"on", "off"}:
        raise RuntimeError("HYBRID_CHECKPOINT_SETTING_INVALID")
    command = [
        sys.executable, "-u", str(root / "scripts/train_3d_mri.py"),
        "--cache", "data/cached/t1_volume_cache_128.h5", "--cache_sha256", str(protocol["cache_sha256"]),
        "--spm_snapshot_sha256", str(protocol["spm_snapshot_sha256"]), "--require_exact_resume_contract",
        "--splits", "splits", "--out_dir", str(output_dir), "--phase", "final",
        "--backbone", "amuse_x_hybrid3d", "--hybrid_variant", variant,
        "--hybrid_attention_heads", str((config.get("attention") or {}).get("heads", 2)),
        "--hybrid_attention_window_size", str((config.get("attention") or {}).get("window_size", 4)),
        "--fusion", str(common.get("fusion", "cross_attention")), "--task_loss", str(common.get("task_loss", "bce")),
        "--lr", str(common.get("learning_rate", 1e-4)), "--scheduler", str(common.get("scheduler", "cosine")),
        "--dropout", str(common.get("dropout", 0.2)), "--resolution", str(protocol["resolution"]),
        "--folds", str(protocol["folds"]), "--epochs", str(protocol["epochs"]), "--seed", str(seed),
        "--batch_size", str(common.get("batch_size", 1)), "--grad_accum", str(common.get("grad_accum", 8)),
        "--device", "cuda", "--require_gpu", "--restore_best_at_end", "--amp",
        "--validate_claim_strict",
        "--parameter_budget_max", str(config.get("parameter_budget_max", 20_000_000)),
        "--num_workers", str(loader.get("num_workers", 2)),
        "--no-persistent_workers",
        "--prefetch_factor", str(loader.get("prefetch_factor", 2)),
        "--checkpoint_staging_root", str(resume.get("checkpoint_staging_root")),
        "--checkpoint_full_cadence", str(resume.get("checkpoint_full_cadence")),
        "--checkpoint_h_mirror", str(resume.get("checkpoint_h_mirror")),
        "--resume_replay_policy", str(resume.get("resume_replay_policy")),
        "--storage_policy", str(resume.get("storage_policy")),
    ]
    if loader.get("pin_memory") is True:
        command.append("--pin_memory")
    if checkpoint == "on":
        command.append("--checkpoint")
    if selection_only:
        command.append("--hpo_selection_only")
    if variant == "plus_a3":
        command.extend(["--latent_alignment_lambda", "0.05", "--alignment_head", "linear"])
    elif variant == "plus_a4":
        command.extend(["--q2_channel_gate", "phenotype"])
    elif variant == "full_model":
        command.extend(["--latent_alignment_lambda", "0.05", "--alignment_head", "linear", "--q2_channel_gate", "phenotype"])
    if "--no_resume" in command:
        raise RuntimeError("HYBRID_NO_RESUME_FORBIDDEN")
    return command


def _inner_metric(path: Path, *, folds: int) -> float:
    payload = _read(path)
    aggregate = payload.get("aggregate") if isinstance(payload.get("aggregate"), Mapping) else {}
    rows = payload.get("folds")
    if aggregate.get("status") != "COMPLETED" or aggregate.get("hpo_selection_only") is not True or not isinstance(rows, list) or len(rows) != folds:
        raise RuntimeError("HYBRID_INNER_METRICS_NOT_COMPLETED")
    if aggregate.get("folds") != int(folds) or aggregate.get("completed_folds") != int(folds):
        raise RuntimeError("HYBRID_INNER_AGGREGATE_FOLD_COUNT_INVALID")
    if any(not isinstance(row, Mapping) for row in rows):
        raise RuntimeError("HYBRID_INNER_FOLD_SCHEMA_INVALID")
    if any(isinstance(row.get("fold"), bool) for row in rows):
        raise RuntimeError("HYBRID_INNER_FOLD_IDENTITIES_INVALID")
    try:
        fold_ids = [int(row["fold"]) for row in rows]
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise RuntimeError("HYBRID_INNER_FOLD_IDENTITIES_INVALID") from exc
    if set(fold_ids) != set(range(int(folds))) or len(set(fold_ids)) != int(folds):
        raise RuntimeError("HYBRID_INNER_FOLD_IDENTITIES_INVALID")
    if any("outer_test_evaluation_count" not in row for row in rows):
        raise RuntimeError("HYBRID_OUTER_TEST_COUNTER_MISSING")
    if any(isinstance(row.get("outer_test_evaluation_count"), bool) for row in rows):
        raise RuntimeError("HYBRID_OUTER_TEST_COUNTER_INVALID")
    try:
        outer_counts = [int(row["outer_test_evaluation_count"]) for row in rows]
    except (TypeError, ValueError, OverflowError) as exc:
        raise RuntimeError("HYBRID_OUTER_TEST_COUNTER_INVALID") from exc
    if any(value != 0 for value in outer_counts):
        raise RuntimeError("HYBRID_OUTER_TEST_USED_DURING_SELECTION")
    try:
        values = [float(row["best_val_loss"]) for row in rows]
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise RuntimeError("HYBRID_INNER_VALIDATION_METRICS_MISSING") from exc
    if len(values) != folds:
        raise RuntimeError("HYBRID_INNER_VALIDATION_METRICS_MISSING")
    if any(not math.isfinite(value) for value in values):
        raise RuntimeError("HYBRID_INNER_VALIDATION_METRICS_NONFINITE")
    return float(statistics.fmean(values))


def _report_ablation_order(config: Mapping[str, Any]) -> list[str]:
    """Best-effort echo for the readiness packet; never raises on malformed config."""
    order = config.get("ablation_order", []) if isinstance(config, Mapping) else []
    return [str(value) for value in order] if isinstance(order, list) else []


def _runtime_readiness_blockers(
    *, config: Mapping[str, Any], output_root: Path, root: Path
) -> list[str]:
    block = config.get("runtime_readiness")
    if not isinstance(block, Mapping):
        return ["MX05_RUNTIME_READINESS_CONFIG_MISSING"]
    artifact = (root / str(block.get("evidence_artifact") or "")).resolve()
    junit = (root / str(block.get("junit_artifact") or "")).resolve()
    try:
        artifact.relative_to(root.resolve())
        junit.relative_to(root.resolve())
        evidence = _read(artifact)
    except (OSError, RuntimeError, ValueError):
        return ["MX05_RUNTIME_READINESS_MISSING_OR_INVALID"]
    body = {
        key: value for key, value in evidence.items() if key != "integrity_sha256"
    }
    provenance = evidence.get("provenance")
    validation = evidence.get("revalidation")
    blockers: list[str] = []
    if (
        evidence.get("schema_version") != 1
        or evidence.get("status") != "COMPLETED"
        or evidence.get("artifact") != "mx05_runtime_readiness"
        or evidence.get("integrity_sha256") != _canonical_sha(body)
    ):
        blockers.append("MX05_RUNTIME_READINESS_INTEGRITY_INVALID")
    if not isinstance(validation, Mapping) or not junit.is_file():
        blockers.append("MX05_RUNTIME_READINESS_JUNIT_INVALID")
    elif (
        validation.get("junit_sha256") != _sha(junit)
        or validation.get("failed") != 0
        or validation.get("errors") != 0
        or validation.get("skipped") != 0
        or not isinstance(validation.get("passed"), int)
        or int(validation["passed"]) <= 0
    ):
        blockers.append("MX05_RUNTIME_READINESS_JUNIT_INVALID")
    if not isinstance(provenance, Mapping):
        blockers.append("MX05_RUNTIME_READINESS_PROVENANCE_INVALID")
        return blockers
    source_manifest = provenance.get("source_manifest")
    if not isinstance(source_manifest, Mapping):
        blockers.append("MX05_RUNTIME_READINESS_SOURCE_MANIFEST_INVALID")
    else:
        for relative, expected in source_manifest.items():
            path = root / str(relative)
            if not path.is_file() or _sha(path) != expected:
                blockers.append(f"MX05_RUNTIME_READINESS_SOURCE_DRIFT:{relative}")
    upstream = provenance.get("upstream_manifest")
    if not isinstance(upstream, Mapping):
        blockers.append("MX05_RUNTIME_READINESS_UPSTREAM_MANIFEST_INVALID")
    else:
        for relative, expected in upstream.items():
            path = output_root / str(relative)
            if not path.is_file() or _sha(path) != expected:
                blockers.append(f"MX05_RUNTIME_READINESS_UPSTREAM_DRIFT:{relative}")
    if provenance.get("output_root") != str(output_root.resolve()):
        blockers.append("MX05_RUNTIME_READINESS_OUTPUT_ROOT_MISMATCH")
    return sorted(set(blockers))


def _validated_smoke_rows(
    *, output_root: Path, config: Mapping[str, Any]
) -> dict[str, dict[str, Any]]:
    from scripts.smoke_method_x_hybrid_variants import VARIANTS, _smoke_binding

    report = _read(output_root / "reports/amuse_x_hybrid_smoke.json")
    rows = report.get("variants")
    if (
        report.get("schema_version") != 2
        or report.get("stage") != "MX05_HYBRID_FULL_CLASSIFIER_SMOKE"
        or report.get("status") != "PASS"
        or report.get("training_started") is not False
        or report.get("smoke_binding") != _smoke_binding()
        or not isinstance(rows, list)
    ):
        raise RuntimeError("MX05_HYBRID_SMOKE_REPORT_INVALID")
    by_variant = {
        str(row.get("hybrid_variant")): dict(row)
        for row in rows
        if isinstance(row, Mapping) and row.get("hybrid_variant")
    }
    if set(by_variant) != set(VARIANTS) or len(rows) != len(VARIANTS):
        raise RuntimeError("MX05_HYBRID_SMOKE_VARIANT_COVERAGE_INVALID")
    blockers: list[str] = []
    for variant in VARIANTS:
        blockers.extend(
            validate_hybrid_smoke_report(
                by_variant[variant],
                expected_variant=variant,
                parameter_budget_max=int(config.get("parameter_budget_max", 0)),
                vram_budget_gb=float(config.get("smoke_vram_budget_gb", 0)),
            )
        )
    if blockers:
        raise RuntimeError("MX05_HYBRID_SMOKE_EVIDENCE_INVALID:" + ",".join(blockers))
    return by_variant


def _task_row_from_evidence(
    *,
    task_id: str,
    variant: str,
    seed: int,
    model_root: Path,
    execution_contract: Mapping[str, Any],
    smoke_row: Mapping[str, Any],
    folds: int,
    training_hard_stop_gb: float,
) -> dict[str, Any]:
    from src.evaluation.method_x_finetune_selection import validate_selection_run

    evidence = validate_selection_run(
        model_root,
        expected_backbone="amuse_x_hybrid3d",
        expected_seed=int(seed),
        folds=int(folds),
    )
    allocated = float(evidence["peak_vram_allocated_gb"])
    reserved = float(evidence["peak_vram_reserved_gb"])
    if (
        not math.isfinite(allocated)
        or not math.isfinite(reserved)
        or allocated < 0
        or reserved < allocated
        or reserved >= float(training_hard_stop_gb)
    ):
        raise RuntimeError(f"HYBRID_TRAINING_VRAM_HARD_STOP:{task_id}")
    metrics_path = model_root / "three_d_metrics.json"
    return {
        "task_id": task_id,
        "variant": variant,
        "seed": int(seed),
        "status": "COMPLETED",
        "model_root": str(model_root.resolve()),
        "score": _inner_metric(metrics_path, folds=int(folds)),
        "roc_auc": float(evidence["roc_auc"]),
        "brier": float(evidence["brier"]),
        "ece": float(evidence["ece"]),
        "cohort_probe_auc": float(evidence["cohort_probe_auc"]),
        "peak_vram_allocated_gb": allocated,
        "peak_vram_reserved_gb": reserved,
        "selection_evidence_sha256": evidence["selection_evidence_sha256"],
        "run_contract_digest": evidence["run_contract_digest"],
        "execution_contract": dict(execution_contract),
        "parameter_count": int(smoke_row["total_parameter_count"]),
        "outer_test_access_count": 0,
        "outer_test_used_for_selection": False,
        "training_started": True,
    }


def _run_or_reuse_hybrid_task(
    *,
    root: Path,
    output_root: Path,
    state_path: Path,
    state: dict[str, Any],
    variant: str,
    seed: int,
    protocol: Mapping[str, Any],
    config: Mapping[str, Any],
    execution_contract: Mapping[str, Any],
    smoke_row: Mapping[str, Any],
) -> tuple[dict[str, Any], bool]:
    task_id = f"{variant}__seed{int(seed)}__inner"
    model_root = output_root / "hybrid_v1" / variant / f"seed_{int(seed)}_inner"
    command = build_hybrid_command(
        root=root,
        output_dir=model_root,
        variant=variant,
        seed=int(seed),
        selection_only=True,
        protocol=protocol,
        config=config,
        execution_contract=execution_contract,
    )
    contract_digest = _canonical_sha(command)
    tasks = state.setdefault("tasks", {})
    existing = tasks.get(task_id, {}) if isinstance(tasks, Mapping) else {}
    if existing.get("contract") and existing.get("contract") != contract_digest:
        raise RuntimeError(f"TASK_CONTRACT_MISMATCH:{task_id}")
    if model_root.is_dir():
        try:
            row = _task_row_from_evidence(
                task_id=task_id,
                variant=variant,
                seed=int(seed),
                model_root=model_root,
                execution_contract=execution_contract,
                smoke_row=smoke_row,
                folds=int(protocol["folds"]),
                training_hard_stop_gb=float(config["training_vram_hard_stop_gb"]),
            )
        except RuntimeError:
            if existing.get("status") == "COMPLETED":
                raise
        else:
            tasks[task_id] = {
                "status": "COMPLETED",
                "contract": contract_digest,
                "metrics_sha256": _sha(model_root / "three_d_metrics.json"),
                "selection_evidence_sha256": row["selection_evidence_sha256"],
                "updated_at_utc": _now(),
            }
            atomic_write_json(state_path, state)
            return row, False
    model_root.mkdir(parents=True, exist_ok=True)
    tasks[task_id] = {
        "status": "RUNNING",
        "contract": contract_digest,
        "updated_at_utc": _now(),
    }
    atomic_write_json(state_path, state)
    execution = run_supervised_process(
        command,
        cwd=root,
        output_root=model_root,
        interval_seconds=15.0,
        max_idle_seconds=3600.0,
        check=lambda pid: evaluate_autonomous_supervision(
            root=root,
            output_root=output_root,
            phase="hybrid_watch",
            config={
                "resolution": 128,
                "folds": 5,
                "seed": int(seed),
                "hardware": {"hard_stop_vram_gb": 7.5},
            },
            expected_pid=pid,
        ),
    )
    code = int(execution.get("returncode", 78))
    if code == 75:
        tasks[task_id].update(
            status="PAUSED_RESUMABLE", execution=execution, updated_at_utc=_now()
        )
        atomic_write_json(state_path, state)
        raise RuntimeError(f"MX05_TASK_PAUSED_RESUMABLE:{task_id}")
    if code != 0:
        guard = record_failure(
            output_root / LOOP_STATE,
            stage="MX-05",
            contract_digest=contract_digest,
            failure_signature=f"TASK_EXIT_{code}:{task_id}",
            max_same_failure_attempts=int(config["max_same_failure_attempts"]),
        )
        tasks[task_id].update(
            status="BLOCKED",
            execution=execution,
            loop_guard=guard,
            updated_at_utc=_now(),
        )
        atomic_write_json(state_path, state)
        raise RuntimeError(f"MX05_TASK_EXECUTION_BLOCKED:{task_id}")
    row = _task_row_from_evidence(
        task_id=task_id,
        variant=variant,
        seed=int(seed),
        model_root=model_root,
        execution_contract=execution_contract,
        smoke_row=smoke_row,
        folds=int(protocol["folds"]),
        training_hard_stop_gb=float(config["training_vram_hard_stop_gb"]),
    )
    tasks[task_id].update(
        status="COMPLETED",
        execution=execution,
        metrics_sha256=_sha(model_root / "three_d_metrics.json"),
        selection_evidence_sha256=row["selection_evidence_sha256"],
        updated_at_utc=_now(),
    )
    atomic_write_json(state_path, state)
    record_success(
        output_root / LOOP_STATE,
        stage="MX-05",
        contract_digest=contract_digest,
    )
    return row, True


def _mx04_handoff_runs(
    *, output_root: Path
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    from src.evaluation.method_x_finetune_selection import validate_selection_run

    path = output_root / "reports/finetune_selection_manifest.json"
    manifest = _read(path)
    handoff = manifest.get("handoff")
    winners = manifest.get("selected_candidates_by_family")
    if (
        manifest.get("schema_version") != 2
        or manifest.get("status") != "COMPLETED"
        or manifest.get("outer_test_used_for_selection") is not False
        or (manifest.get("outer_evaluation") or {}).get("count") != 0
        or not isinstance(handoff, Mapping)
        or not isinstance(winners, Mapping)
    ):
        raise RuntimeError("MX05_MX04_HANDOFF_MANIFEST_INVALID")
    family = str(handoff.get("selected_family") or "")
    candidate = str(handoff.get("selected_candidate") or "")
    winner = winners.get(family)
    if not isinstance(winner, Mapping) or winner.get("candidate") != candidate:
        raise RuntimeError("MX05_MX04_HANDOFF_WINNER_INVALID")
    roots = winner.get("model_roots")
    if not isinstance(roots, Mapping) or set(str(key) for key in roots) != {
        "13",
        "29",
        "47",
    }:
        raise RuntimeError("MX05_MX04_HANDOFF_REPLICATION_ROOTS_INVALID")
    runs = [
        validate_selection_run(
            str(roots[str(seed)]),
            expected_backbone=family,
            expected_seed=seed,
            folds=5,
        )
        for seed in (13, 29, 47)
    ]
    summary = {
        "family": family,
        "candidate": candidate,
        "canonical_model_root": str(roots["13"]),
        "model_roots": {str(seed): str(roots[str(seed)]) for seed in (13, 29, 47)},
        "manifest_sha256": _sha(path),
    }
    sanity = manifest.get("mx03_sanity_evidence")
    if not isinstance(sanity, Mapping) or sanity.get("passed") is not True:
        raise RuntimeError("MX05_MX03_SANITY_EVIDENCE_INVALID")
    return summary, runs, dict(sanity)


def _write_stable_json(path: Path, payload: Mapping[str, Any]) -> None:
    value = dict(payload)
    if path.is_file():
        if _read(path) != value:
            raise RuntimeError(f"MX05_IMMUTABLE_EVIDENCE_MISMATCH:{path.name}")
        return
    atomic_write_json(path, value)


def run_hybrid_orchestrator(
    *,
    root: Path,
    output_root: Path,
    config_path: Path,
    mode: str = "audit",
    persist_audit: bool = True,
) -> dict[str, Any]:
    root, output_root, config_path = (
        root.resolve(),
        output_root.resolve(),
        config_path.resolve(),
    )
    config_load_blockers: list[str] = []
    try:
        loaded_config = load_unique_yaml(config_path)
    except (OSError, UnicodeError, ValueError) as exc:
        loaded_config = {}
        config_load_blockers.append(f"HYBRID_CONFIG_INVALID:{type(exc).__name__}")
    config = loaded_config if isinstance(loaded_config, Mapping) else {}
    if not isinstance(loaded_config, Mapping):
        config_load_blockers.append("HYBRID_CONFIG_NOT_MAPPING")
    try:
        battery = _read(output_root / "battery_manifest.json")
        protocol = _protocol(battery)
    except RuntimeError as exc:
        battery = {}
        protocol = {"resolution": 128, "folds": 5, "seed": 13, "epochs": 100, "common": {}}
        config_load_blockers.append(str(exc))
    supervision = evaluate_autonomous_supervision(
        root=root,
        output_root=output_root,
        phase="hybrid_orchestrator",
        config={
            "resolution": protocol["resolution"],
            "folds": protocol["folds"],
            "seed": protocol["seed"],
            "hardware": {"hard_stop_vram_gb": 7.5},
        },
    )
    if mode != "audit" or persist_audit:
        enforce_autonomous_supervision(
            root=root, output_root=output_root, report=supervision
        )
    blockers = [*config_load_blockers, *_validate_config(config)]
    blockers.extend(_trainer_flag_subset_blockers(root=root))
    if supervision.get("decision") == "BLOCK":
        blockers.append("AUTOSUPERVISOR_BLOCKED")
    if not _baseline_terminal(output_root):
        blockers.append("HYBRID_BASELINE_BATTERY_NOT_FINALIZED")
    if not _upstream_terminal(
        output_root, "mx03_validity_orchestrator", {"COMPLETED"}
    ):
        blockers.append("MX03_VALIDITY_NOT_COMPLETED")
    if not _upstream_terminal(
        output_root, "mx04_finetune_orchestrator", {"COMPLETED"}
    ):
        blockers.append("MX04_FINETUNE_NOT_COMPLETED")
    blockers.extend(
        _runtime_readiness_blockers(
            config=config, output_root=output_root, root=root
        )
    )
    smoke_rows: dict[str, dict[str, Any]] = {}
    execution_contract: dict[str, Any] = {}
    require_dynamic_evidence = mode == "run" or persist_audit
    if require_dynamic_evidence and not blockers:
        try:
            smoke_rows = _validated_smoke_rows(
                output_root=output_root, config=config
            )
        except RuntimeError as exc:
            blockers.append(str(exc))
        try:
            execution_contract = load_hybrid_execution_contract(
                config, root=root
            )
        except RuntimeError as exc:
            blockers.append(str(exc))
    report: dict[str, Any] = {
        "schema_version": 1,
        "stage": "MX-05",
        "mode": mode,
        "status": "BLOCKED" if blockers else "READY",
        "training_started": False,
        "outer_test_used_for_selection": False,
        "ablation_order": _report_ablation_order(config),
        "candidate_tasks": [],
        "replication_tasks": [],
        "blockers": sorted(set(blockers)),
        "supervision_decision": supervision.get("decision"),
        "autonomous_supervision": {
            "mode": "AUTONOMOUS_MULTI_LAYER_FAIL_CLOSED",
            "human_approval_required": False,
            "manual_override_allowed": False,
            "fail_closed": True,
        },
        "next_automatic_action": (
            "RESOLVE_MX05_PREFLIGHT_BLOCKERS" if blockers else "RUN_MX05_ABLATIONS"
        ),
    }
    reports = output_root / "reports"
    if blockers or mode == "audit":
        if mode != "audit" or persist_audit:
            if blockers:
                atomic_write_json(reports / "BLOCKED_MX05_HYBRID.json", report)
            atomic_write_json(reports / "mx05_hybrid_readiness.json", report)
            atomic_write_json(reports / "mx05_hybrid_orchestrator.json", report)
        return report

    state_path = reports / "mx05_task_state.json"
    state = (
        _read(state_path)
        if state_path.is_file()
        else {"schema_version": 1, "tasks": {}}
    )
    screening: list[dict[str, Any]] = []
    replication: list[dict[str, Any]] = []
    try:
        report["status"] = "RUNNING_ABLATIONS"
        for variant in EXPECTED_ABLATIONS:
            row, started = _run_or_reuse_hybrid_task(
                root=root,
                output_root=output_root,
                state_path=state_path,
                state=state,
                variant=variant,
                seed=13,
                protocol=protocol,
                config=config,
                execution_contract=execution_contract,
                smoke_row=smoke_rows[variant],
            )
            screening.append(row)
            report["candidate_tasks"] = screening
            report["training_started"] = bool(report["training_started"] or started)
            atomic_write_json(reports / "mx05_hybrid_orchestrator.json", report)
        if [row["variant"] for row in screening] != EXPECTED_ABLATIONS:
            raise RuntimeError("HYBRID_ABLATION_SET_INCOMPLETE")
        winner = min(
            screening,
            key=lambda row: (
                float(row["score"]), EXPECTED_ABLATIONS.index(str(row["variant"]))
            ),
        )
        selected_variant = str(winner["variant"])
        selection_manifest = {
            "schema_version": 2,
            "status": "LOCKED_INNER_VALIDATION",
            "selection_metric": "mean_best_inner_validation_loss",
            "screening_seed": 13,
            "selected_variant": selected_variant,
            "selected_model_root": winner["model_root"],
            "ablation_semantics": "ISOLATED_COMPONENT_TOGGLES",
            "ablation_order": list(EXPECTED_ABLATIONS),
            "scores": screening,
            "outer_test_used_for_selection": False,
            "outer_test_access_count": 0,
            "replication_seeds": [13, 29, 47],
            "execution_contract": execution_contract,
        }
        _write_stable_json(
            reports / "hybrid_selection_manifest.json", selection_manifest
        )
        architecture_manifest = {
            "schema_version": 2,
            "status": "LOCKED_INNER_VALIDATION",
            "single_model": True,
            "variant": selected_variant,
            "parameter_budget_max": config["parameter_budget_max"],
            "parameter_counts": {
                row["variant"]: row["parameter_count"] for row in screening
            },
            "smoke_vram_budget_gb": config["smoke_vram_budget_gb"],
            "training_vram_hard_stop_gb": config["training_vram_hard_stop_gb"],
            "training_vram_peaks": {
                row["variant"]: {
                    "allocated_gb": row["peak_vram_allocated_gb"],
                    "reserved_gb": row["peak_vram_reserved_gb"],
                }
                for row in screening
            },
            "components": config["components"],
            "variant_definitions": config["variant_definitions"],
            "claim_boundary": config["claim_boundary"],
            "outer_test_used_for_selection": False,
        }
        _write_stable_json(
            reports / "amuse_x_hybrid_architecture_manifest.json",
            architecture_manifest,
        )

        report["status"] = "RUNNING_WINNER_REPLICATION"
        for seed in (13, 29, 47):
            if seed == 13:
                row = winner
                started = False
            else:
                row, started = _run_or_reuse_hybrid_task(
                    root=root,
                    output_root=output_root,
                    state_path=state_path,
                    state=state,
                    variant=selected_variant,
                    seed=seed,
                    protocol=protocol,
                    config=config,
                    execution_contract=execution_contract,
                    smoke_row=smoke_rows[selected_variant],
                )
            replication.append(row)
            report["replication_tasks"] = replication
            report["training_started"] = bool(report["training_started"] or started)
            atomic_write_json(reports / "mx05_hybrid_orchestrator.json", report)
        if [int(row["seed"]) for row in replication] != [13, 29, 47]:
            raise RuntimeError("HYBRID_REPLICATION_SEEDS_INCOMPLETE")

        from scripts.run_method_x_finetune_orchestrator import (
            evaluate_promotion_gate,
        )
        from src.evaluation.method_x_finetune_selection import (
            build_promotion_evidence,
            validate_selection_run,
        )

        candidate_runs = [
            validate_selection_run(
                row["model_root"],
                expected_backbone="amuse_x_hybrid3d",
                expected_seed=int(row["seed"]),
                folds=5,
            )
            for row in replication
        ]
        comparator, comparator_runs, sanity = _mx04_handoff_runs(
            output_root=output_root
        )
        promotion_evidence = build_promotion_evidence(
            candidate_runs=candidate_runs,
            comparator_runs=comparator_runs,
            sanity_checks_passed=bool(sanity["passed"]),
            bootstrap_resamples=int(config["promotion"]["bootstrap_resamples"]),
            bootstrap_seed=13,
        )
        would_promote, promotion_reasons = evaluate_promotion_gate(
            promotion=config["promotion"], evidence=promotion_evidence
        )
        if would_promote:
            decision = "PROMOTE_MX05_HYBRID"
            model_kind = "MX05_HYBRID"
            selected_backbone = "amuse_x_hybrid3d"
            source_model_root = Path(str(winner["model_root"]))
            promotion_gate = {
                "status": "PASS",
                "would_promote": True,
                "blockers": [],
            }
        else:
            if not promotion_reasons:
                raise RuntimeError("HYBRID_NO_PROMOTION_WITHOUT_REASONS")
            decision = "RETAIN_MX04_NO_HYBRID_PROMOTION"
            model_kind = "MX04_FINETUNE"
            selected_backbone = str(comparator["family"])
            source_model_root = Path(str(comparator["canonical_model_root"]))
            promotion_gate = {
                "status": "NO_PROMOTION",
                "would_promote": False,
                "blockers": list(promotion_reasons),
            }
        run_id = "mx05-" + _canonical_sha(
            {
                "config_sha256": _sha(config_path),
                "selected_variant": selected_variant,
                "hybrid_selection_evidence": [
                    row["selection_evidence_sha256"] for row in replication
                ],
                "comparator_selection_evidence": [
                    row["selection_evidence_sha256"] for row in comparator_runs
                ],
                "promotion_evidence": promotion_evidence,
                "decision": decision,
            }
        )[:16]
        replication_report = {
            "schema_version": 1,
            "status": "COMPLETED",
            "run_id": run_id,
            "selected_backbone": selected_backbone,
            "replication_seeds": [13, 29, 47],
            "canonical_checkpoint_seed": 13,
            "hybrid_candidate_variant": selected_variant,
            "hybrid_candidate_runs": replication,
            "comparator": comparator,
            "outer_test_access_count": 0,
            "outer_test_used_for_selection": False,
        }
        bootstrap_report = {
            "schema_version": 1,
            "status": "COMPLETED",
            "run_id": run_id,
            "selected_backbone": selected_backbone,
            "resamples": int(promotion_evidence["bootstrap_resamples"]),
            "bootstrap_unit": promotion_evidence["bootstrap_unit"],
            "evidence": promotion_evidence,
            "outer_test_used_for_selection": False,
        }
        cohort_report = {
            "schema_version": 1,
            "status": "COMPLETED",
            "run_id": run_id,
            "selected_backbone": selected_backbone,
            "scope": "INNER_VALIDATION_AUDIT_ONLY",
            "hybrid_candidate_max_auc": promotion_evidence[
                "cohort_probe_auc_candidate_max"
            ],
            "comparator_max_auc": promotion_evidence[
                "cohort_probe_auc_comparator_max"
            ],
            "increase": promotion_evidence["cohort_probe_auc_increase"],
            "outer_test_used": False,
        }
        promotion_report = {
            "schema_version": 1,
            "status": promotion_gate["status"],
            "run_id": run_id,
            "selected_backbone": selected_backbone,
            "would_promote": promotion_gate["would_promote"],
            "blockers": promotion_gate["blockers"],
            "selection_decision": decision,
            "selected_variant": selected_variant,
            "evidence": promotion_evidence,
            "outer_test_used_for_selection": False,
        }
        evidence_paths = {
            "replication_report": reports / "mx05_replication_report.json",
            "paired_bootstrap_report": reports / "mx05_paired_bootstrap_report.json",
            "cohort_probe_report": reports / "mx05_cohort_probe_report.json",
            "promotion_report": reports / "mx05_promotion_report.json",
        }
        for path, payload in (
            (evidence_paths["replication_report"], replication_report),
            (evidence_paths["paired_bootstrap_report"], bootstrap_report),
            (evidence_paths["cohort_probe_report"], cohort_report),
            (evidence_paths["promotion_report"], promotion_report),
        ):
            _write_stable_json(path, payload)

        from scripts.finalize_mx05_outer import create_finalization_lock

        lock = create_finalization_lock(
            output_root=output_root,
            run_id=run_id,
            selection_decision=decision,
            model_kind=model_kind,
            selected_backbone=selected_backbone,
            source_model_root=source_model_root,
            selected_variant=selected_variant if would_promote else None,
            promotion_gate=promotion_gate,
            evidence_paths=evidence_paths,
        )
        report.update(
            status="RUNNING_LOCKED_OUTER_FINALIZATION",
            selected_variant=selected_variant,
            selection_decision=decision,
            selected_backbone=selected_backbone,
            promotion_gate=promotion_gate,
            finalization_lock_integrity=lock["integrity_sha256"],
            next_automatic_action="RUN_OR_RESUME_LOCKED_OUTER_FINALIZATION",
        )
        atomic_write_json(reports / "mx05_hybrid_orchestrator.json", report)
        final_command = [
            sys.executable,
            "-u",
            str(root / "scripts/finalize_mx05_outer.py"),
            "--output-root",
            str(output_root),
            "--config",
            str(config_path),
        ]
        execution = run_supervised_process(
            final_command,
            cwd=root,
            output_root=output_root,
            interval_seconds=15.0,
            max_idle_seconds=3600.0,
            check=lambda pid: evaluate_autonomous_supervision(
                root=root,
                output_root=output_root,
                phase="mx05_outer_finalization",
                config={
                    "resolution": 128,
                    "folds": 5,
                    "seed": 13,
                    "hardware": {"hard_stop_vram_gb": 7.5},
                },
                expected_pid=pid,
            ),
        )
        code = int(execution.get("returncode", 78))
        if code == 75:
            raise RuntimeError("MX05_OUTER_PAUSED_RESUMABLE")
        if code != 0:
            raise RuntimeError(f"MX05_OUTER_FINALIZATION_EXIT_{code}")
        finalization = _read(reports / "mx05_outer_finalization.json")
        if (
            finalization.get("status") != "COMPLETED"
            or finalization.get("selection_decision") != decision
            or (finalization.get("outer_evaluation") or {}).get("count") != 1
        ):
            raise RuntimeError("MX05_OUTER_FINALIZATION_INVALID")
        negative_entries = [
            {
                "variant": row["variant"],
                "decision": "NOT_SELECTED_INNER_VALIDATION",
                "reason": "higher_mean_best_inner_validation_loss",
            }
            for row in screening
            if row["variant"] != selected_variant
        ]
        if not would_promote:
            negative_entries.append(
                {
                    "variant": selected_variant,
                    "decision": "NOT_PROMOTED",
                    "reasons": list(promotion_reasons),
                }
            )
        _write_stable_json(
            reports / "mx05_negative_ledger.json",
            {
                "schema_version": 1,
                "status": "COMPLETED",
                "entries": negative_entries,
                "pending_or_not_run_count": 0,
            },
        )
        report.update(
            status="COMPLETED",
            blockers=[],
            run_id=run_id,
            selected_model_root=finalization["selected_model_root"],
            outer_evaluation={"status": "COMPLETED", "count": 1},
            outer_test_used_for_selection=False,
            next_automatic_action="RUN_MX06_ANATOMICAL_XAI",
        )
    except RuntimeError as exc:
        message = str(exc)
        if "PAUSED_RESUMABLE" in message:
            report["status"] = "PAUSED_RESUMABLE"
            report["next_automatic_action"] = "RERUN_IDENTICAL_MX05_COMMAND"
        else:
            report["status"] = "BLOCKED"
            report["next_automatic_action"] = "RESOLVE_REPORTED_MX05_BLOCKER"
        report["blockers"] = sorted(set([*report["blockers"], message]))

    report["candidate_tasks"] = screening
    report["replication_tasks"] = replication
    report["blockers"] = sorted(set(str(value) for value in report["blockers"]))
    atomic_write_json(reports / "mx05_hybrid_readiness.json", report)
    atomic_write_json(reports / "mx05_hybrid_orchestrator.json", report)
    if report["blockers"]:
        atomic_write_json(reports / "BLOCKED_MX05_HYBRID.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--mode", choices=("audit", "run"), default="audit")
    parser.add_argument(
        "--no-persist-audit",
        action="store_true",
        help="Evaluate audit gates without writing reports or autosupervisor events.",
    )
    args = parser.parse_args()
    output_root = args.output_root if args.output_root.is_absolute() else ROOT / args.output_root
    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    if args.no_persist_audit and args.mode != "audit":
        parser.error("--no-persist-audit is valid only with --mode audit")
    report = run_hybrid_orchestrator(
        root=ROOT,
        output_root=output_root,
        config_path=config_path,
        mode=args.mode,
        persist_audit=not args.no_persist_audit,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True))
    if report.get("status") == "PAUSED_RESUMABLE":
        return 75
    # Inner-only ablation selection cannot unlock XAI or publication.  The
    # selected variant must first complete its three-seed replication.
    return 0 if report.get("status") in {"READY", "COMPLETED"} else 3


if __name__ == "__main__":
    raise SystemExit(main())
