"""Autonomous MX-04 fine-tuning graph.

The graph is deliberately data-dependent: audit mode writes contracts and
truthful blockers, while run mode is unlocked only after the matched battery
and MX-03 validity are terminal.  Pretrained weights are bound per fold; a
single fold-0 checkpoint can never be broadcast to the other folds.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import signal
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.training.method_x_autosupervisor import (
    enforce_autonomous_supervision,
    evaluate_autonomous_supervision,
    run_supervised_process,
)
from src.training.storage_locality import watch_blockers_are_live_inventory_race
from src.training.method_x_loop_guard import record_failure, record_success
from src.method_y_revival.strict_yaml import load_unique_yaml
from src.utils.io_helpers import atomic_write_json
from src.evaluation.method_x_finetune_selection import (
    build_promotion_evidence,
    build_replication_spec,
    choose_handoff,
    validate_selection_run,
)


DEFAULT_OUTPUT_ROOT = ROOT / "results/final_results/method_x_model_battery_v3_depth_and_diversity_20260729"
DEFAULT_CONFIG = ROOT / "configs/experiments/method_x_finetune_v1.yaml"
LOOP_STATE = "reports/method_x_loop_guard.json"
DEFAULT_CHECKPOINT_STAGING_ROOT = r"C:\Users\USER\AppData\Local\AMUSE-X\checkpoint_staging"
DEFAULT_STORAGE_POLICY = "configs/policy/method_x_storage_locality_v1.yaml"
THROUGHPUT_PROFILE_ARTIFACT = "method_x_mx04_throughput_profile"
# Cadence-1 staging rotates ~173 MB files on C via os.replace. A watch probe
# that collides with that rotation can see a transient Windows lock and report
# C_STAGING_INVENTORY_UNREADABLE. That is not scientific drift; the battery
# scheduler already retries+defers this exact code instead of killing the
# child. MX-04 must do the same (fail-closed elsewhere).
WATCH_INVENTORY_RACE_RETRY_SECONDS = 0.5
_MX04_STOP_REQUESTED = False


def _supervised_watch_report(supervised: Mapping[str, Any] | None) -> Mapping[str, Any] | None:
    if not isinstance(supervised, Mapping):
        return None
    last_report = supervised.get("last_report")
    if isinstance(last_report, Mapping):
        return last_report
    return supervised


def defer_watch_inventory_race(watch_report: Mapping[str, Any]) -> dict[str, Any]:
    """Keep a healthy trainer alive across a transient C inventory lock."""

    report = dict(watch_report)
    report["decision"] = "CONTINUE"
    report["watch_inventory_race_deferred"] = True
    automation = dict(report.get("automation") or {})
    automation["next_action"] = "RUN_NEXT_LOCKED_STAGE"
    report["automation"] = automation
    return report


def resolve_watch_inventory_race(
    *,
    first_report: Mapping[str, Any],
    retry_probe: Callable[[], Mapping[str, Any]],
) -> tuple[dict[str, Any], bool]:
    """Return ``(report, should_enforce)``. Inventory-only watch BLOCK is not a kill."""

    first = dict(first_report)
    if first.get("decision") != "BLOCK":
        return first, False
    if not watch_blockers_are_live_inventory_race(first):
        return first, True
    retry = dict(retry_probe())
    if retry.get("decision") != "BLOCK":
        return retry, False
    if watch_blockers_are_live_inventory_race(retry):
        return defer_watch_inventory_race(retry), False
    return retry, True


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _request_mx04_stop(_signum: int, _frame: Any) -> None:
    """Let the trainer checkpoint at an optimizer boundary before we exit."""

    global _MX04_STOP_REQUESTED
    _MX04_STOP_REQUESTED = True


def _install_mx04_signal_handlers() -> None:
    global _MX04_STOP_REQUESTED
    _MX04_STOP_REQUESTED = False
    for name in ("SIGINT", "SIGTERM"):
        signum = getattr(signal, name, None)
        if signum is not None:
            signal.signal(signum, _request_mx04_stop)


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


def _rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve())).replace("\\", "/")
    except ValueError:
        return str(path.resolve())


def _is_sha256(value: Any) -> bool:
    text = str(value or "").strip().lower()
    return len(text) == 64 and all(char in "0123456789abcdef" for char in text)


def validate_fold_pretrained_manifest(
    manifest: Mapping[str, Any],
    *,
    folds: int = 5,
    root: Path = ROOT,
    expected_backbone: str | None = None,
    expected_output_root: Path | None = None,
) -> dict[str, dict[str, str]]:
    """Validate complete fold-local provenance and the files it binds."""

    entries = manifest.get("folds") if isinstance(manifest, Mapping) else None
    expected = {str(index) for index in range(int(folds))}
    if not isinstance(entries, Mapping) or set(str(key) for key in entries) != expected:
        raise RuntimeError("PRETRAINED_FOLD_MANIFEST_INCOMPLETE")
    normalized: dict[str, dict[str, str]] = {}
    for fold in sorted(expected, key=int):
        value = entries[fold]
        if not isinstance(value, Mapping):
            raise RuntimeError(f"PRETRAINED_FOLD_MANIFEST_INVALID:{fold}")
        checkpoint = str(value.get("checkpoint", "")).strip()
        sidecar = str(value.get("manifest", "")).strip()
        sha256 = str(value.get("sha256", "")).strip().lower()
        if not checkpoint or not sidecar or not _is_sha256(sha256):
            raise RuntimeError(f"PRETRAINED_FOLD_MANIFEST_INVALID:{fold}")
        checkpoint_path = Path(checkpoint)
        sidecar_path = Path(sidecar)
        checkpoint_path = checkpoint_path if checkpoint_path.is_absolute() else root / checkpoint_path
        sidecar_path = sidecar_path if sidecar_path.is_absolute() else root / sidecar_path
        try:
            checkpoint_path = checkpoint_path.resolve()
            sidecar_path = sidecar_path.resolve()
            checkpoint_path.relative_to(root.resolve())
            sidecar_path.relative_to(root.resolve())
            if expected_output_root is not None:
                checkpoint_path.relative_to(expected_output_root.resolve())
                sidecar_path.relative_to(expected_output_root.resolve())
        except (OSError, RuntimeError, ValueError) as exc:
            raise RuntimeError(f"PRETRAINED_FOLD_PATH_OUTSIDE_ALLOWED_ROOT:{fold}") from exc
        if not checkpoint_path.is_file() or not sidecar_path.is_file():
            raise RuntimeError(f"PRETRAINED_FOLD_FILE_MISSING:{fold}")
        if _sha(checkpoint_path) != sha256:
            raise RuntimeError(f"PRETRAINED_FOLD_CHECKPOINT_HASH_MISMATCH:{fold}")
        sidecar_payload = _read(sidecar_path)
        sidecar_sha = str(sidecar_payload.get("checkpoint_sha256") or "").lower()
        if sidecar_sha != sha256 or str(sidecar_payload.get("sha256") or "").lower() != sha256:
            raise RuntimeError(f"PRETRAINED_FOLD_SIDECAR_HASH_MISMATCH:{fold}")
        if sidecar_payload.get("source_type") != "FOLD_LOCAL_METHOD_X_BASELINE":
            raise RuntimeError(f"PRETRAINED_FOLD_SIDECAR_SOURCE_INVALID:{fold}")
        if isinstance(sidecar_payload.get("source_fold"), bool) or sidecar_payload.get("source_fold") != int(fold):
            raise RuntimeError(f"PRETRAINED_FOLD_SIDECAR_FOLD_MISMATCH:{fold}")
        backbone = str(sidecar_payload.get("source_backbone") or "")
        if expected_backbone is not None and backbone != expected_backbone:
            raise RuntimeError(f"PRETRAINED_FOLD_SIDECAR_BACKBONE_MISMATCH:{fold}")
        if sidecar_payload.get("external_download") is not False or sidecar_payload.get("raw_identifiers_stored") is not False:
            raise RuntimeError(f"PRETRAINED_FOLD_SIDECAR_POLICY_INVALID:{fold}")
        if not _is_sha256(sidecar_payload.get("source_run_contract_digest")):
            raise RuntimeError(f"PRETRAINED_FOLD_RUN_CONTRACT_DIGEST_INVALID:{fold}")
        if not _is_sha256(sidecar_payload.get("source_split_membership_digest")):
            raise RuntimeError(f"PRETRAINED_FOLD_SPLIT_DIGEST_INVALID:{fold}")
        normalized[fold] = {"checkpoint": checkpoint, "manifest": sidecar, "sha256": sha256}
    if expected_backbone is not None and str(manifest.get("backbone") or "") != expected_backbone:
        raise RuntimeError("PRETRAINED_FOLD_MANIFEST_BACKBONE_MISMATCH")
    source_contract_digest = str(manifest.get("source_contract_digest") or "").lower()
    if not _is_sha256(source_contract_digest):
        raise RuntimeError("PRETRAINED_FOLD_MANIFEST_CONTRACT_DIGEST_INVALID")
    for fold in sorted(expected, key=int):
        sidecar_path = Path(normalized[fold]["manifest"])
        sidecar_path = sidecar_path if sidecar_path.is_absolute() else root / sidecar_path
        if str(_read(sidecar_path.resolve()).get("source_run_contract_digest") or "").lower() != source_contract_digest:
            raise RuntimeError(f"PRETRAINED_FOLD_MANIFEST_CONTRACT_MISMATCH:{fold}")
    return normalized


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


def _baseline_terminal(manifest: Mapping[str, Any]) -> bool:
    if str(manifest.get("status", "")).upper() != "COMPLETED":
        return False
    rows = manifest.get("models")
    if not isinstance(rows, list) or len(rows) != 7:
        return False
    by_name: dict[str, Mapping[str, Any]] = {}
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
        if str(row.get("status", "")).upper() != "COMPLETED":
            return False
        # ``training`` must be a Mapping, not merely truthy.  ``row.get("training")
        # or {}`` returns a non-empty string unchanged, so ``.get`` on it raises
        # AttributeError and the gate escapes as an unhandled crash instead of a
        # clean False.  A malformed battery is not terminal.
        if not isinstance(training, Mapping):
            return False
        if str(training.get("status", "")).upper() not in TERMINAL_TRAINING_STATUSES:
            return False
    for name in EXCLUDED_BACKBONES:
        row = by_name[name]
        training = row.get("training")
        if str(row.get("status", "")) != "PROTOCOL_EXCLUDED_NOT_STARTED":
            return False
        if not isinstance(training, Mapping):
            return False
        if str(training.get("status", "")) != "PROTOCOL_EXCLUDED_NOT_STARTED":
            return False
    return True


def _mx03_terminal(output_root: Path) -> bool:
    path = output_root / "reports/mx03_validity_orchestrator.json"
    if not path.is_file():
        return False
    try:
        return _read(path).get("status") == "COMPLETED"
    except RuntimeError:
        return False


_LEGACY_AMP_CALLS = ("torch.cuda.amp.GradScaler(", "torch.cuda.amp.autocast(")
_MODERNIZATION_ARTIFACT = "method_x_mx04_runtime_modernization"


def _trainer_uses_legacy_amp(trainer_path: Path) -> bool:
    """True when the trainer still calls the deprecated torch.cuda.amp API."""
    try:
        text = Path(trainer_path).read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        return True  # unreadable trainer fails closed at the call site
    for line in text.splitlines():
        code = line.split("#", 1)[0]
        if any(call in code for call in _LEGACY_AMP_CALLS):
            return True
    return False


def _modernization_integrity(payload: Mapping[str, Any]) -> str:
    body = {key: value for key, value in payload.items() if key != "integrity_sha256"}
    return hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def _validate_runtime_modernization(config: Mapping[str, Any], *, root: Path) -> list[str]:
    """Enforce the MX-04 versioned-boundary runtime work (fail-closed).

    The torch.cuda.amp deprecation and attention non-determinism warnings
    observed during MX-03 were triaged warn-only-by-design with a mid-run
    source freeze.  They must be resolved here — mechanical migration to
    torch.amp plus an explicit determinism evaluation with full
    re-validation — before any MX-04 training launches.
    """
    blockers: list[str] = []
    block = config.get("runtime_modernization")
    if (
        not isinstance(block, Mapping)
        or block.get("amp_api") != "torch.amp"
        or block.get("determinism_evaluation_required") is not True
        or not str(block.get("evidence_artifact") or "").strip()
    ):
        return ["FINETUNE_RUNTIME_MODERNIZATION_BLOCK_MISSING"]
    trainer_path = root / "scripts/train_3d_mri.py"
    if _trainer_uses_legacy_amp(trainer_path):
        blockers.append("FINETUNE_RUNTIME_AMP_MIGRATION_MISSING")
    try:
        evidence_path = (root / str(block["evidence_artifact"])).resolve()
        evidence_path.relative_to(root.resolve())
        evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError):
        return blockers + ["FINETUNE_RUNTIME_MODERNIZATION_EVIDENCE_MISSING_OR_INVALID"]
    if (
        not isinstance(evidence, dict)
        or evidence.get("schema_version") != 1
        or evidence.get("artifact") != _MODERNIZATION_ARTIFACT
        or evidence.get("integrity_sha256") != _modernization_integrity(evidence)
    ):
        return blockers + ["FINETUNE_RUNTIME_MODERNIZATION_EVIDENCE_MISSING_OR_INVALID"]
    amp = evidence.get("amp_migration")
    live_trainer_sha: str | None = None
    try:
        live_trainer_sha = _sha(trainer_path)
    except OSError:
        live_trainer_sha = None
    if (
        not isinstance(amp, Mapping)
        or amp.get("status") != "COMPLETED"
        or not isinstance(amp.get("trainer_sha256"), str)
        or amp.get("trainer_sha256") != live_trainer_sha
    ):
        return blockers + ["FINETUNE_RUNTIME_MODERNIZATION_EVIDENCE_MISSING_OR_INVALID"]
    determinism = evidence.get("determinism_evaluation")
    if (
        not isinstance(determinism, Mapping)
        or determinism.get("status") != "COMPLETED"
        or not str(determinism.get("decision") or "").strip()
        or not str(determinism.get("measurement") or "").strip()
    ):
        return blockers + ["FINETUNE_RUNTIME_MODERNIZATION_EVIDENCE_MISSING_OR_INVALID"]
    revalidation = evidence.get("revalidation")
    if (
        not isinstance(revalidation, Mapping)
        or revalidation.get("status") != "COMPLETED"
        or not isinstance(revalidation.get("passed"), int)
        or isinstance(revalidation.get("passed"), bool)
        or int(revalidation.get("passed")) <= 0
        or int(revalidation.get("failed", 0)) != 0
    ):
        return blockers + ["FINETUNE_RUNTIME_MODERNIZATION_EVIDENCE_MISSING_OR_INVALID"]
    return blockers


PROTOCOL_EXCLUDED_FAMILIES = ("resnet50_3d", "efficientnet3d_b0")
_ALLOWED_A4_FEATURES = ["age", "sex", "education_years", "apoe4_carrier", "mmse_base"]
EXPECTED_FAMILIES = (
    "resnet10_3d",
    "resnet18_3d",
    "resnet34_3d",
    "densenet121_3d",
    "mednext_3d_s",
)

EXPECTED_EXACT_RESUME = {
    "required": True,
    "checkpoint_staging_root": DEFAULT_CHECKPOINT_STAGING_ROOT,
    "checkpoint_full_cadence": 1,
    "checkpoint_h_mirror": "epoch",
    "resume_replay_policy": "deterministic",
    "storage_policy": DEFAULT_STORAGE_POLICY,
    "no_resume_forbidden": True,
}
EXPECTED_CANDIDATES: tuple[dict[str, Any], ...] = (
    {
        "id": "scratch_baseline",
        "pretrained": False,
        "freeze_schedule": "none",
        "layerwise_lr_decay": 1.0,
        "discriminative_lr": False,
        "ema": False,
        "swa": False,
        "task_loss": "bce",
    },
    {
        "id": "head_then_last_stage_llrd",
        "pretrained": True,
        "freeze_schedule": "head_8_then_last_stage",
        "layerwise_lr_decay": 0.80,
        "discriminative_lr": True,
        "finetune_lr": 0.00005,
        "ema": False,
        "swa": False,
        "task_loss": "bce",
    },
    {
        "id": "full_llrd_ema",
        "pretrained": True,
        "freeze_schedule": "full_after_warmup",
        "layerwise_lr_decay": 0.75,
        "discriminative_lr": True,
        "finetune_lr": 0.00005,
        "ema": True,
        "ema_decay": 0.999,
        "swa": False,
        "task_loss": "bce",
    },
    {
        "id": "full_focal_calibrated",
        "pretrained": True,
        "freeze_schedule": "full_after_warmup",
        "layerwise_lr_decay": 0.75,
        "discriminative_lr": True,
        "finetune_lr": 0.00005,
        "ema": False,
        "swa": True,
        "task_loss": "focal",
        "focal_gamma": 2.0,
    },
)
EXPECTED_PROMOTION: dict[str, Any] = {
    "all_five_folds_finalized": True,
    "sanity_checks_required": True,
    "bootstrap_resamples": 1000,
    "roc_auc_delta_min": 0.02,
    "brier_degradation_max": 0.01,
    "ece_degradation_max": 0.02,
    "cohort_probe_auc_increase_max": 0.0,
    "memory_hard_stop_gb": 7.5,
}


def _promotion_config_blockers(promotion: Mapping[str, Any]) -> list[str]:
    if not isinstance(promotion, Mapping):
        return ["FINETUNE_PROMOTION_CONFIG_MISSING"]
    blockers: list[str] = []
    if set(promotion) != set(EXPECTED_PROMOTION):
        blockers.append("FINETUNE_PROMOTION_KEYS_MISMATCH")
    for key, expected in EXPECTED_PROMOTION.items():
        actual = promotion.get(key)
        if isinstance(expected, bool):
            matches = actual is expected
        elif isinstance(expected, int):
            matches = isinstance(actual, int) and not isinstance(actual, bool) and actual == expected
        else:
            try:
                numeric = float(actual)
            except (TypeError, ValueError, OverflowError):
                matches = False
            else:
                if not math.isfinite(numeric):
                    blockers.append(f"FINETUNE_PROMOTION_NONFINITE:{key}")
                matches = math.isfinite(numeric) and numeric == float(expected)
        if not matches:
            blockers.append(f"FINETUNE_PROMOTION_THRESHOLD_MISMATCH:{key}")
    return blockers


def evaluate_promotion_gate(*, promotion: Mapping[str, Any], evidence: Mapping[str, Any]) -> tuple[bool, list[str]]:
    """Pure promotion rule: WOULD_PROMOTE/WOULD_NOT_PROMOTE + reasons.

    Early-track 032: audit-mode validator only. It takes no MX-03 result and
    locks nobody; plan 020 wires it into the locking path post-MX-03.
    The YAML must equal the pre-specified thresholds exactly; a caller cannot
    loosen the gate by supplying a different configuration. Evidence remains a
    plain dict so the rule is unit-testable on synthetic inputs.
    """
    reasons: list[str] = []
    for blocker in _promotion_config_blockers(promotion):
        if blocker.startswith("FINETUNE_PROMOTION_NONFINITE:"):
            reasons.append(blocker.replace("FINETUNE_PROMOTION_NONFINITE:", "PROMOTION_CONFIG_NONFINITE:", 1))
        else:
            reasons.append(blocker.replace("FINETUNE_", "PROMOTION_CONFIG_INVALID:", 1))
    required_resamples = int(EXPECTED_PROMOTION["bootstrap_resamples"])
    roc_min = float(EXPECTED_PROMOTION["roc_auc_delta_min"])
    brier_max = float(EXPECTED_PROMOTION["brier_degradation_max"])
    ece_max = float(EXPECTED_PROMOTION["ece_degradation_max"])
    cohort_max = float(EXPECTED_PROMOTION["cohort_probe_auc_increase_max"])
    vram_stop = float(EXPECTED_PROMOTION["memory_hard_stop_gb"])
    if promotion.get("all_five_folds_finalized") is not True or evidence.get("all_five_folds_finalized") is not True:
        reasons.append("PROMOTION_FOLDS_NOT_FINALIZED")
    if promotion.get("sanity_checks_required") is not True or evidence.get("sanity_checks_passed") is not True:
        reasons.append("PROMOTION_SANITY_CHECKS_NOT_PASSED")
    try:
        resamples = int(evidence.get("bootstrap_resamples", 0))
    except (TypeError, ValueError, OverflowError):
        resamples = 0
    if resamples < required_resamples:
        reasons.append("PROMOTION_BOOTSTRAP_INSUFFICIENT")

    def _finite_evidence(key: str) -> float | None:
        try:
            value = float(evidence[key])
        except (KeyError, TypeError, ValueError, OverflowError):
            value = float("nan")
        if not math.isfinite(value):
            reasons.append(f"PROMOTION_EVIDENCE_NONFINITE:{key}")
            return None
        return value

    delta = _finite_evidence("roc_auc_delta")
    lower = _finite_evidence("roc_auc_delta_lower_ci")
    if delta is not None and lower is not None and not (delta >= roc_min and lower > 0):
        reasons.append("PROMOTION_ROC_DELTA_INSUFFICIENT")
    for key, limit, code in (
        ("brier_degradation", brier_max, "PROMOTION_BRIER_DEGRADED"),
        ("ece_degradation", ece_max, "PROMOTION_ECE_DEGRADED"),
    ):
        value = _finite_evidence(key)
        if value is not None and not value <= limit:
            reasons.append(code)
    cohort = _finite_evidence("cohort_probe_auc_increase")
    if cohort is not None and not cohort <= cohort_max:
        reasons.append("PROMOTION_COHORT_PROBE_WORSE")
    for key in ("peak_vram_allocated_gb", "peak_vram_reserved_gb"):
        peak = _finite_evidence(key)
        if peak is not None and (peak < 0 or not peak < vram_stop):
            reasons.append(f"PROMOTION_VRAM_EXCEEDED:{key}")
            break
    return (not reasons, reasons)


def _finite_number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def _validate_exact_resume_config(config: Mapping[str, Any], *, root: Path) -> list[str]:
    block = config.get("exact_resume")
    if not isinstance(block, Mapping):
        return ["FINETUNE_EXACT_RESUME_CONFIG_MISSING"]
    blockers: list[str] = []
    for key, expected in EXPECTED_EXACT_RESUME.items():
        if block.get(key) != expected:
            blockers.append(f"FINETUNE_EXACT_RESUME_CONFIG_MISMATCH:{key}")
    if "--no_resume" in {str(value) for value in block.values()}:
        blockers.append("FINETUNE_NO_RESUME_FORBIDDEN")
    try:
        policy_path = (root / str(block.get("storage_policy") or "")).resolve()
        policy_path.relative_to(root.resolve())
        policy = load_unique_yaml(policy_path)
        policy_root = Path(str((policy.get("checkpoint_staging") or {}).get("root") or "")).resolve()
        configured_root = Path(str(block.get("checkpoint_staging_root") or "")).resolve()
        if str(policy_root).casefold() != str(configured_root).casefold():
            blockers.append("FINETUNE_EXACT_RESUME_STAGING_POLICY_MISMATCH")
        if int((policy.get("checkpoint_staging") or {}).get("minimum_free_gb", -1)) < 1:
            blockers.append("FINETUNE_EXACT_RESUME_STORAGE_FLOOR_INVALID")
    except (OSError, UnicodeError, ValueError, TypeError):
        blockers.append("FINETUNE_EXACT_RESUME_STORAGE_POLICY_INVALID")
    return blockers


def _throughput_profile_path(config: Mapping[str, Any], *, root: Path) -> Path:
    block = config.get("throughput_profile")
    if not isinstance(block, Mapping) or not str(block.get("evidence_artifact") or "").strip():
        raise RuntimeError("FINETUNE_THROUGHPUT_PROFILE_CONFIG_MISSING")
    path = (root / str(block["evidence_artifact"])).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as exc:
        raise RuntimeError("FINETUNE_THROUGHPUT_PROFILE_PATH_OUTSIDE_PROJECT") from exc
    return path


def validate_throughput_profile(
    profile: Mapping[str, Any],
    *,
    families: tuple[str, ...] = EXPECTED_FAMILIES,
) -> dict[str, Any]:
    """Validate and re-derive the plan-041 execution choices fail-closed."""

    if not isinstance(profile, Mapping) or profile.get("schema_version") != 1:
        raise RuntimeError("FINETUNE_THROUGHPUT_PROFILE_SCHEMA_INVALID")
    required_top = {
        "schema_version",
        "operational_preflight",
        "profile_run",
        "decision",
        "per_family_checkpoint",
        "cross_mode_receipt",
        "loader",
        "yellow_gates",
    }
    if not required_top <= set(profile):
        raise RuntimeError("FINETUNE_THROUGHPUT_PROFILE_KEYS_MISSING")
    operational = profile.get("operational_preflight")
    required_operational = {
        "sleep_hibernate_off",
        "cooling_secured",
        "stable_power",
        "signal_handling_intact_no_no_resume",
    }
    if not isinstance(operational, Mapping) or set(operational) != required_operational:
        raise RuntimeError("FINETUNE_THROUGHPUT_OPERATIONAL_PREFLIGHT_INVALID")
    for key in required_operational:
        value = str(operational.get(key) or "")
        if key == "signal_handling_intact_no_no_resume":
            valid = value == "PASS"
        else:
            valid = value == "PASS" or value.startswith("N/A:")
        if not valid:
            raise RuntimeError(f"FINETUNE_THROUGHPUT_OPERATIONAL_PREFLIGHT_FAILED:{key}")

    run = profile.get("profile_run")
    if not isinstance(run, Mapping) or run.get("mode") != "smoke_only_never_selection_evidence":
        raise RuntimeError("FINETUNE_THROUGHPUT_PROFILE_RUN_INVALID")
    output_dir = str(run.get("output_dir") or "")
    if not output_dir or "finetune_v1" in output_dir.replace("\\", "/").lower():
        raise RuntimeError("FINETUNE_THROUGHPUT_PROFILE_OUTPUT_NOT_DISPOSABLE")
    utilization = _finite_number(run.get("sm_utilization_pct"))
    allocated = _finite_number(run.get("vram_allocated_peak_gb"))
    reserved = _finite_number(run.get("vram_reserved_peak_gb"))
    wallclock = _finite_number(run.get("epoch_wallclock_s"))
    if (
        utilization is None
        or not 0.0 <= utilization <= 100.0
        or allocated is None
        or reserved is None
        or allocated < 0.0
        or reserved < allocated
        or reserved >= 7.5
        or wallclock is None
        or wallclock <= 0.0
    ):
        raise RuntimeError("FINETUNE_THROUGHPUT_PROFILE_MEASUREMENT_INVALID")
    derived_decision = "loader_first" if utilization < 70.0 else "compute_first"
    if profile.get("decision") != derived_decision:
        raise RuntimeError("FINETUNE_THROUGHPUT_DECISION_NOT_REDERIVABLE")

    checkpoint = profile.get("per_family_checkpoint")
    if not isinstance(checkpoint, Mapping) or set(checkpoint) != set(families):
        raise RuntimeError("FINETUNE_THROUGHPUT_FAMILY_CHECKPOINT_MAP_INVALID")
    normalized_checkpoint: dict[str, dict[str, Any]] = {}
    settings: set[str] = set()
    for family in families:
        row = checkpoint.get(family)
        if not isinstance(row, Mapping) or row.get("setting") not in {"on", "off"}:
            raise RuntimeError(f"FINETUNE_THROUGHPUT_FAMILY_CHECKPOINT_INVALID:{family}")
        family_allocated = _finite_number(row.get("allocated_peak_gb"))
        family_reserved = _finite_number(row.get("reserved_peak_gb"))
        if (
            family_allocated is None
            or family_reserved is None
            or family_allocated < 0.0
            or family_reserved < family_allocated
            or family_reserved >= 7.5
        ):
            raise RuntimeError(f"FINETUNE_THROUGHPUT_FAMILY_VRAM_INVALID:{family}")
        derived_under_cap = max(family_allocated, family_reserved) < 7.0
        if row.get("under_7p0_soft_cap") is not derived_under_cap:
            raise RuntimeError(f"FINETUNE_THROUGHPUT_FAMILY_CAP_RECEIPT_INVALID:{family}")
        if row.get("setting") == "off" and not derived_under_cap:
            raise RuntimeError(f"FINETUNE_THROUGHPUT_CHECKPOINT_OFF_UNSAFE:{family}")
        settings.add(str(row["setting"]))
        normalized_checkpoint[family] = {
            "setting": str(row["setting"]),
            "allocated_peak_gb": family_allocated,
            "reserved_peak_gb": family_reserved,
            "under_7p0_soft_cap": derived_under_cap,
        }

    receipt = profile.get("cross_mode_receipt")
    if not isinstance(receipt, Mapping):
        raise RuntimeError("FINETUNE_THROUGHPUT_CROSS_MODE_RECEIPT_INVALID")
    receipt_diff = _finite_number(receipt.get("max_abs_metric_diff"))
    # Any run with recomputation disabled relies on a measured on/off
    # equivalence receipt.  This also covers the uniform-all-off case: the
    # absence of a mixed family map does not make the numerical-equivalence
    # obligation disappear.
    if "off" in settings:
        if receipt.get("verdict") != "PASS" or receipt_diff is None or receipt_diff > 1e-6:
            raise RuntimeError("FINETUNE_THROUGHPUT_CROSS_MODE_EQUIVALENCE_FAILED")
    elif settings == {"on"} and receipt.get("verdict") != "NOT_APPLICABLE(uniform-all-on)":
        raise RuntimeError("FINETUNE_THROUGHPUT_CROSS_MODE_RECEIPT_INVALID")

    loader = profile.get("loader")
    if not isinstance(loader, Mapping):
        raise RuntimeError("FINETUNE_THROUGHPUT_LOADER_INVALID")
    workers = loader.get("num_workers")
    before = _finite_number(loader.get("epoch_before_s"))
    after = _finite_number(loader.get("epoch_after_s"))
    if (
        isinstance(workers, bool)
        or workers not in {2, 4}
        or loader.get("persistent_workers") is not False
        or loader.get("prefetch_factor") != 2
        or loader.get("pin_memory") is not True
        or before is None
        or after is None
        or before <= 0.0
        or after <= 0.0
        or not isinstance(loader.get("fallback_to_2_recorded"), bool)
    ):
        raise RuntimeError("FINETUNE_THROUGHPUT_LOADER_INVALID")
    if workers == 4 and not after < before:
        raise RuntimeError("FINETUNE_THROUGHPUT_WORKERS4_NOT_FASTER")
    if workers == 2 and after < before and loader.get("fallback_to_2_recorded") is True:
        raise RuntimeError("FINETUNE_THROUGHPUT_LOADER_FALLBACK_CONTRADICTORY")

    yellow = profile.get("yellow_gates")
    if not isinstance(yellow, Mapping) or set(yellow) != {"torch_compile", "channels_last"}:
        raise RuntimeError("FINETUNE_THROUGHPUT_YELLOW_GATES_INVALID")
    for name in ("torch_compile", "channels_last"):
        verdict = str(yellow.get(name) or "")
        if not (verdict == "PASSED" or verdict.startswith("DROPPED:")):
            raise RuntimeError(f"FINETUNE_THROUGHPUT_YELLOW_GATE_INVALID:{name}")
        if verdict == "PASSED":
            raise RuntimeError(f"FINETUNE_THROUGHPUT_YELLOW_GATE_UNIMPLEMENTED:{name}")

    normalized_loader = {
        "num_workers": int(workers),
        "persistent_workers": False,
        "prefetch_factor": 2,
        "pin_memory": True,
    }
    return {
        "decision": derived_decision,
        "per_family_checkpoint": normalized_checkpoint,
        "loader": normalized_loader,
        "yellow_gates": dict(yellow),
    }


def _validate_throughput_profile(config: Mapping[str, Any], *, root: Path) -> list[str]:
    try:
        path = _throughput_profile_path(config, root=root)
        profile = _read(path)
        validate_throughput_profile(profile)
        receipts = _read(path.with_name("MX04_THROUGHPUT_PROFILE_RECEIPTS.json"))
        profile_digest = hashlib.sha256(
            json.dumps(profile, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        ).hexdigest()
        profiler_path = root / "scripts/profile_mx04_throughput.py"
        if (
            receipts.get("schema_version") != 1
            or receipts.get("status") != "COMPLETED"
            or receipts.get("profile_sha256") != profile_digest
            or receipts.get("trainer_sha256") != _sha(root / "scripts/train_3d_mri.py")
            or receipts.get("profiler_sha256") != _sha(profiler_path)
            or _finite_number(receipts.get("loader_metric_max_abs_diff")) is None
            or float(receipts["loader_metric_max_abs_diff"]) > 1e-6
        ):
            raise RuntimeError("FINETUNE_THROUGHPUT_PROFILE_RECEIPTS_INVALID")
    except (OSError, RuntimeError, ValueError):
        return ["FINETUNE_THROUGHPUT_PROFILE_MISSING_OR_INVALID"]
    return []


def _exact_resume_storage_preflight(config: Mapping[str, Any], *, root: Path) -> list[str]:
    """Read-only free-space gate; trainer repeats it before every fold."""

    block = config.get("exact_resume")
    if not isinstance(block, Mapping):
        return ["FINETUNE_EXACT_RESUME_CONFIG_MISSING"]
    try:
        policy_path = (root / str(block["storage_policy"])).resolve()
        policy = load_unique_yaml(policy_path)
        minimum = int((policy.get("checkpoint_staging") or {})["minimum_free_gb"])
        staging_root = Path(str(block["checkpoint_staging_root"]))
        probe = staging_root if staging_root.exists() else Path(staging_root.anchor)
        free_gb = shutil.disk_usage(probe).free / (1024 ** 3)
    except (OSError, KeyError, TypeError, ValueError):
        return ["FINETUNE_EXACT_RESUME_STORAGE_PREFLIGHT_UNAVAILABLE"]
    return [] if free_gb >= minimum else ["FINETUNE_EXACT_RESUME_STORAGE_FLOOR_NOT_MET"]


def load_mx04_execution_contracts(
    config: Mapping[str, Any],
    *,
    root: Path = ROOT,
    families: tuple[str, ...] = EXPECTED_FAMILIES,
) -> dict[str, dict[str, Any]]:
    profile = _read(_throughput_profile_path(config, root=root))
    validated = validate_throughput_profile(profile, families=families)
    exact_resume = dict(config["exact_resume"])
    return {
        family: {
            "checkpoint": validated["per_family_checkpoint"][family]["setting"],
            "loader": dict(validated["loader"]),
            "exact_resume": exact_resume,
            "yellow_gates": dict(validated["yellow_gates"]),
        }
        for family in families
    }


def _validate_config(
    config: Mapping[str, Any],
    *,
    root: Path = ROOT,
    require_throughput: bool = True,
) -> list[str]:
    blockers: list[str] = []
    families = [str(value) for value in config.get("candidate_families", []) or []]
    for excluded in PROTOCOL_EXCLUDED_FAMILIES:
        if excluded in families:
            blockers.append(f"PROTOCOL_EXCLUDED_BACKBONE:{excluded}")
    if tuple(families) != EXPECTED_FAMILIES:
        blockers.append("FINETUNE_FAMILY_SET_MISMATCH")
    if config.get("screening_seed") != 13 or isinstance(config.get("screening_seed"), bool):
        blockers.append("FINETUNE_SCREENING_SEED_MISMATCH")
    constraints = config.get("constraints")
    if not isinstance(constraints, Mapping):
        blockers.append("FINETUNE_CONSTRAINTS_BLOCK_MISSING")
    else:
        if str(constraints.get("a2_phenotypic_contrastive", "")) != "disabled_until_physical_batch_or_queue":
            blockers.append("FINETUNE_CONSTRAINT_A2_ENABLED")
        features = constraints.get("a4_conditioning_features")
        if not isinstance(features, list) or list(features) != _ALLOWED_A4_FEATURES:
            blockers.append("FINETUNE_CONSTRAINT_A4_FEATURES_MISMATCH")
        if constraints.get("pet_input") is not False:
            blockers.append("FINETUNE_CONSTRAINT_PET_INPUT")
        if str(constraints.get("external_weights", "")) != "explicit_local_checkpoint_only":
            blockers.append("FINETUNE_CONSTRAINT_WEIGHTS_POLICY_MISMATCH")
    if config.get("selection_scope") != "inner_validation_only":
        blockers.append("FINETUNE_SELECTION_SCOPE_NOT_INNER_VALIDATION_ONLY")
    if config.get("outer_test_access") != "exactly_once_after_lock":
        blockers.append("FINETUNE_OUTER_TEST_POLICY_MISMATCH")
    if config.get("inner_selection_only") is not True:
        blockers.append("FINETUNE_INNER_SELECTION_POLICY_MISSING")
    if config.get("fold_local_pretrained_required") is not True:
        blockers.append("FINETUNE_FOLD_LOCAL_PRETRAINED_POLICY_MISSING")
    seeds = config.get("replication_seeds", [])
    try:
        seeds_ok = [int(value) for value in seeds] == [13, 29, 47]
    except (TypeError, ValueError):
        seeds_ok = False
    if not seeds_ok:
        blockers.append("FINETUNE_REPLICATION_SEEDS_MISMATCH")
    max_attempts = config.get("max_same_failure_attempts")
    if isinstance(max_attempts, bool) or not isinstance(max_attempts, int) or max_attempts < 1:
        blockers.append("FINETUNE_LOOP_GUARD_MAX_ATTEMPTS_INVALID")
    candidates = config.get("candidates")
    if not isinstance(candidates, list):
        blockers.append("FINETUNE_CANDIDATE_GRID_MISSING")
    else:
        ids = [str(row.get("id")) for row in candidates if isinstance(row, Mapping)]
        if len(candidates) != 4 or len(ids) != 4 or len(ids) != len(set(ids)):
            blockers.append("FINETUNE_CANDIDATE_IDS_INVALID")
        if candidates != list(EXPECTED_CANDIDATES):
            blockers.append("FINETUNE_CANDIDATE_GRID_MISMATCH")
        for row in candidates:
            if not isinstance(row, Mapping):
                blockers.append("FINETUNE_CANDIDATE_SCHEMA_INVALID")
                continue
            if bool(row.get("ema")) and bool(row.get("swa")):
                blockers.append(f"EMA_AND_SWA_SIMULTANEOUS:{row.get('id')}")
    blockers.extend(_promotion_config_blockers(config.get("promotion", {})))
    blockers.extend(_validate_runtime_modernization(config, root=root))
    blockers.extend(_validate_exact_resume_config(config, root=root))
    if require_throughput:
        blockers.extend(_validate_throughput_profile(config, root=root))
    return blockers


def build_pretrained_fold_manifest(*, output_root: Path, backbone: str, folds: int = 5) -> tuple[Path, dict[str, Any]]:
    """Create sidecars from the already-finalized baseline fold checkpoints."""

    baseline_dir = output_root / "models" / backbone
    contract_path = baseline_dir / "run_contract.json"
    contract = _read(contract_path).get("run_contract") if contract_path.is_file() else {}
    if not isinstance(contract, dict):
        raise RuntimeError(f"BASELINE_RUN_CONTRACT_MISSING:{backbone}")
    sidecar_dir = output_root / "pretrained_provenance" / backbone
    entries: dict[str, dict[str, str]] = {}
    for fold in range(int(folds)):
        checkpoint = baseline_dir / f"method_x_3d_fold_{fold}.pt"
        state_path = baseline_dir / f"method_x_3d_fold_{fold}_state.json"
        if not checkpoint.is_file() or not state_path.is_file():
            raise RuntimeError(f"BASELINE_CHECKPOINT_MISSING:{backbone}:FOLD_{fold}")
        state = _read(state_path)
        if str(state.get("stage", "")).upper() != "FINALIZED":
            raise RuntimeError(f"BASELINE_FOLD_NOT_FINALIZED:{backbone}:FOLD_{fold}")
        sidecar = sidecar_dir / f"fold_{fold}.json"
        checkpoint_sha = _sha(checkpoint)
        atomic_write_json(
            sidecar,
            {
                "schema_version": 1,
                "checkpoint_sha256": checkpoint_sha,
                "sha256": checkpoint_sha,
                "license": "PROJECT_GENERATED_INTERNAL_RESEARCH_WEIGHTS",
                "source_type": "FOLD_LOCAL_METHOD_X_BASELINE",
                "source_backbone": backbone,
                "source_fold": fold,
                "source_run_contract_digest": contract.get("digest"),
                "source_split_membership_digest": (state.get("details") or {}).get("split_membership_digest"),
                "raw_identifiers_stored": False,
                "external_download": False,
            },
        )
        entries[str(fold)] = {
            "checkpoint": _rel(checkpoint),
            "manifest": _rel(sidecar),
            "sha256": checkpoint_sha,
        }
    payload = {
        "schema_version": 1,
        "backbone": backbone,
        "folds": entries,
        "coverage_minimum": 0.95,
        "source_contract_digest": contract.get("digest"),
        "outer_test_used_for_selection": False,
        "created_at_utc": _now(),
    }
    manifest_path = sidecar_dir / "pretrained_fold_manifest.json"
    atomic_write_json(manifest_path, payload)
    validate_fold_pretrained_manifest(
        payload,
        folds=folds,
        root=ROOT,
        expected_backbone=backbone,
        expected_output_root=output_root,
    )
    return manifest_path, payload


def _protocol_args(battery: Mapping[str, Any]) -> dict[str, Any]:
    protocol = battery.get("locked_protocol") if isinstance(battery.get("locked_protocol"), Mapping) else {}
    common = protocol.get("common") if isinstance(protocol.get("common"), Mapping) else {}
    return {
        "folds": int(protocol.get("folds", 5)),
        "epochs": int(protocol.get("epochs", 100)),
        "resolution": int(protocol.get("resolution", 128)),
        "seed": int(protocol.get("seed", 13)),
        "cache_sha256": str((battery.get("locked_inputs") or {}).get("cache_sha256", "")),
        "spm_snapshot_sha256": str((battery.get("spm_gate_snapshot") or {}).get("snapshot_sha256", "")),
        "common": dict(common),
    }


def build_candidate_command(
    *,
    output_root: Path,
    family: str,
    candidate: Mapping[str, Any],
    seed: int,
    selection_only: bool,
    fold_manifest: Path | None,
    protocol: Mapping[str, Any],
    execution_contract: Mapping[str, Any] | None = None,
) -> list[str]:
    common = protocol["common"]
    contract = dict(execution_contract or {})
    checkpoint_setting = str(contract.get("checkpoint", "on"))
    if checkpoint_setting not in {"on", "off"}:
        raise RuntimeError("FINETUNE_CHECKPOINT_SETTING_INVALID")
    loader = contract.get("loader") if isinstance(contract.get("loader"), Mapping) else {}
    exact_resume = (
        contract.get("exact_resume")
        if isinstance(contract.get("exact_resume"), Mapping)
        else EXPECTED_EXACT_RESUME
    )
    num_workers = int(loader.get("num_workers", common.get("num_workers", 2)))
    persistent_workers = bool(loader.get("persistent_workers", False))
    prefetch_factor = int(loader.get("prefetch_factor", common.get("prefetch_factor", 2)))
    pin_memory = bool(loader.get("pin_memory", True))
    if persistent_workers:
        raise RuntimeError("EXACT_RESUME_PERSISTENT_WORKERS_UNSUPPORTED")
    if int(exact_resume.get("checkpoint_full_cadence", 0)) != 1:
        raise RuntimeError("CHECKPOINT_REPLAY_NOT_AUTHORIZED")
    if exact_resume.get("required") is not True or exact_resume.get("no_resume_forbidden") is not True:
        raise RuntimeError("FINETUNE_EXACT_RESUME_REQUIRED")
    command = [
        sys.executable,
        "-u",
        str(ROOT / "scripts/train_3d_mri.py"),
        "--cache", "data/cached/t1_volume_cache_128.h5",
        "--cache_sha256", str(protocol["cache_sha256"]),
        "--spm_snapshot_sha256", str(protocol["spm_snapshot_sha256"]),
        "--require_exact_resume_contract",
        "--splits", "splits",
        "--out_dir", str(output_root),
        "--phase", "final",
        "--backbone", family,
        "--fusion", str(common.get("fusion", "cross_attention")),
        "--task_loss", str(candidate.get("task_loss", common.get("task_loss", "bce"))),
        "--lr", str(common.get("learning_rate", 1e-4)),
        "--scheduler", str(common.get("scheduler", "cosine")),
        "--dropout", str(common.get("dropout", 0.2)),
        "--resolution", str(protocol["resolution"]),
        "--folds", str(protocol["folds"]),
        "--epochs", str(protocol["epochs"]),
        "--seed", str(seed),
        "--batch_size", str(common.get("batch_size", 1)),
        "--grad_accum", str(common.get("grad_accum", 8)),
        "--device", "cuda",
        "--require_gpu",
        "--restore_best_at_end",
        "--amp",
        "--num_workers", str(num_workers),
        "--no-persistent_workers",
        "--prefetch_factor", str(prefetch_factor),
        "--checkpoint_staging_root", str(exact_resume["checkpoint_staging_root"]),
        "--checkpoint_full_cadence", "1",
        "--checkpoint_h_mirror", str(exact_resume["checkpoint_h_mirror"]),
        "--resume_replay_policy", str(exact_resume["resume_replay_policy"]),
        "--storage_policy", str(exact_resume["storage_policy"]),
    ]
    if pin_memory:
        command.append("--pin_memory")
    else:
        command.append("--no-pin_memory")
    if checkpoint_setting == "on":
        command.append("--checkpoint")
    if selection_only:
        command.append("--hpo_selection_only")
    if bool(candidate.get("pretrained")):
        if fold_manifest is None:
            raise RuntimeError("PRETRAINED_FOLD_MANIFEST_REQUIRED")
        command.extend(["--pretrained_fold_manifest", str(fold_manifest), "--require_pretrained_provenance"])
        command.extend(["--finetune_lr", str(candidate.get("finetune_lr", 5e-5))])
        command.extend(["--layerwise_lr_decay", str(candidate.get("layerwise_lr_decay", 0.75)), "--discriminative_lr"])
        schedule = str(candidate.get("freeze_schedule", "full_after_warmup"))
        if schedule == "head_8_then_last_stage":
            command.extend(["--freeze_encoder", "--unfreeze_schedule", schedule, "--unfreeze_epoch", "8"])
        elif schedule == "full_after_warmup":
            command.extend(["--freeze_encoder", "--unfreeze_schedule", schedule, "--unfreeze_epoch", "8"])
    if bool(candidate.get("ema")):
        command.extend(["--ema_decay", str(candidate.get("ema_decay", 0.999))])
    if bool(candidate.get("swa")):
        command.extend(["--swa", "--swa_start_fraction", "0.8"])
    if str(candidate.get("task_loss")) == "focal":
        command.extend(["--focal_gamma", str(candidate.get("focal_gamma", 2.0))])
    return command


ORCHESTRATOR_ONLY_FLAGS = (
    "--hpo_selection_only",
    "--pretrained_fold_manifest",
    "--require_pretrained_provenance",
    "--unfreeze_schedule",
    "--unfreeze_epoch",
)


def _trainer_flag_subset_blockers(*, root: Path = ROOT) -> list[str]:
    """Fail-closed audit check: orchestrator flags must exist in the trainer.

    Early-track 032: build_candidate_command emits 5 flags the trainer
    parser lacks (verified hermetically: orchestrator-emitted minus
    trainer-parser = exactly the tuple below; --finetune_lr DOES exist at
    train_3d_mri.py:3247). Report TRAINER_FLAGS_ABSENT in audit mode
    instead of crashing at launch with exit 2; plan 020 adds the
    trainer-side flags post-MX-03 (trainer is allowlisted now).
    """
    import ast as _ast

    try:
        tree = _ast.parse((root / "scripts/train_3d_mri.py").read_text(encoding="utf-8"))
    except (OSError, UnicodeError, SyntaxError):
        return ["TRAINER_PARSER_UNREADABLE"]
    parser_flags: set[str] = set()
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


def _inner_metric(path: Path, *, folds: int) -> float:
    payload = _read(path)
    aggregate = payload.get("aggregate") if isinstance(payload.get("aggregate"), Mapping) else {}
    if aggregate.get("status") != "COMPLETED":
        raise RuntimeError("FINETUNE_INNER_METRICS_NOT_COMPLETED")
    if aggregate.get("hpo_selection_only") is not True:
        raise RuntimeError("FINETUNE_INNER_METRICS_NOT_INNER_ONLY")
    if aggregate.get("folds") != int(folds) or aggregate.get("completed_folds") != int(folds):
        raise RuntimeError("FINETUNE_INNER_AGGREGATE_FOLD_COUNT_INVALID")
    rows = payload.get("folds")
    if not isinstance(rows, list) or len(rows) != int(folds):
        raise RuntimeError("FINETUNE_INNER_FOLD_COUNT_INVALID")
    if any(not isinstance(row, Mapping) for row in rows):
        raise RuntimeError("FINETUNE_INNER_FOLD_SCHEMA_INVALID")
    if any(isinstance(row.get("fold"), bool) for row in rows):
        raise RuntimeError("FINETUNE_INNER_FOLD_IDENTITIES_INVALID")
    try:
        fold_ids = [int(row["fold"]) for row in rows]
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise RuntimeError("FINETUNE_INNER_FOLD_IDENTITIES_INVALID") from exc
    if fold_ids != list(range(int(folds))) and set(fold_ids) != set(range(int(folds))):
        raise RuntimeError("FINETUNE_INNER_FOLD_IDENTITIES_INVALID")
    if len(set(fold_ids)) != int(folds):
        raise RuntimeError("FINETUNE_INNER_FOLD_IDENTITIES_INVALID")
    if any("outer_test_evaluation_count" not in row for row in rows):
        raise RuntimeError("FINETUNE_OUTER_TEST_COUNTER_MISSING")
    if any(isinstance(row.get("outer_test_evaluation_count"), bool) for row in rows):
        raise RuntimeError("FINETUNE_OUTER_TEST_COUNTER_INVALID")
    try:
        outer_counts = [int(row["outer_test_evaluation_count"]) for row in rows]
    except (TypeError, ValueError, OverflowError) as exc:
        raise RuntimeError("FINETUNE_OUTER_TEST_COUNTER_INVALID") from exc
    if any(value != 0 for value in outer_counts):
        raise RuntimeError("FINETUNE_OUTER_TEST_USED_DURING_SELECTION")
    try:
        values = [float(row["best_val_loss"]) for row in rows]
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise RuntimeError("FINETUNE_INNER_VALIDATION_METRICS_MISSING") from exc
    if len(values) != int(folds):
        raise RuntimeError("FINETUNE_INNER_VALIDATION_METRICS_MISSING")
    if any(not math.isfinite(value) for value in values):
        raise RuntimeError("FINETUNE_INNER_VALIDATION_METRICS_NONFINITE")
    return float(statistics.fmean(values))


def _assert_inner_vram_below_hard_stop(
    path: Path,
    *,
    hard_stop_gb: float,
) -> dict[str, float]:
    """Validate the measured full-run CUDA peaks before accepting a task.

    Profiling is only a launch-time estimate.  Every completed screening and
    replication run must independently prove that both allocator peaks stayed
    strictly below the prespecified hard stop.
    """

    payload = _read(path)
    aggregate = payload.get("aggregate")
    if not isinstance(aggregate, Mapping):
        raise RuntimeError("FINETUNE_INNER_VRAM_MISSING_OR_INVALID")
    allocated = _finite_number(aggregate.get("vram_peak_allocated_gb"))
    reserved = _finite_number(aggregate.get("vram_peak_reserved_gb"))
    limit = _finite_number(hard_stop_gb)
    if (
        allocated is None
        or reserved is None
        or limit is None
        or limit <= 0.0
        or allocated < 0.0
        or reserved < allocated
    ):
        raise RuntimeError("FINETUNE_INNER_VRAM_MISSING_OR_INVALID")
    if not allocated < limit or not reserved < limit:
        raise RuntimeError("FINETUNE_INNER_VRAM_HARD_STOP")
    return {
        "peak_vram_allocated_gb": allocated,
        "peak_vram_reserved_gb": reserved,
    }


def select_inner_winners_by_family(
    scores: list[dict[str, Any]],
    *,
    families: list[str],
    candidate_ids: list[str],
) -> dict[str, dict[str, Any]]:
    """Select exactly one inner-validation winner for every family."""

    expected = {(family, candidate) for family in families for candidate in candidate_ids}
    observed: dict[tuple[str, str], dict[str, Any]] = {}
    for row in scores:
        key = (str(row.get("family")), str(row.get("candidate")))
        if key in observed:
            raise RuntimeError(f"FINETUNE_DUPLICATE_INNER_SCORE:{key[0]}:{key[1]}")
        try:
            score = float(row["score"])
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise RuntimeError(f"FINETUNE_INNER_SCORE_INVALID:{key[0]}:{key[1]}") from exc
        if not math.isfinite(score):
            raise RuntimeError(f"FINETUNE_INNER_SCORE_NONFINITE:{key[0]}:{key[1]}")
        observed[key] = row
    if set(observed) != expected:
        raise RuntimeError("FINETUNE_INNER_GRID_INCOMPLETE")
    return {
        family: min(
            (observed[(family, candidate)] for candidate in candidate_ids),
            key=lambda row: float(row["score"]),
        )
        for family in families
    }


def _mx03_sanity_evidence(output_root: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for task_id, filename in (
        ("random_labels", "sanity_random_labels.json"),
        ("random_noise", "sanity_random_noise.json"),
    ):
        path = output_root / "private" / filename
        if not path.is_file():
            return {
                "status": "BLOCKED",
                "passed": False,
                "reason": f"MX03_SANITY_ARTIFACT_MISSING:{task_id}",
                "tasks": rows,
            }
        payload = _read(path)
        passed = (
            payload.get("status") == "COMPLETED"
            and payload.get("gate") == "PASS"
            and payload.get("near_chance_gate") is True
            and payload.get("outer_test_used_for_selection") is False
            and payload.get("subject_level_vectors_exported") is False
        )
        rows.append(
            {
                "task_id": task_id,
                "passed": passed,
                "artifact": _rel(path),
                "sha256": _sha(path),
                "roc_auc": payload.get("roc_auc"),
            }
        )
    return {
        "status": "COMPLETED" if all(row["passed"] for row in rows) else "BLOCKED",
        "passed": all(row["passed"] for row in rows),
        "tasks": rows,
    }


def _blocked_next_action(blockers: list[str]) -> str:
    if any("MX03_VALIDITY_NOT_COMPLETED" in blocker for blocker in blockers):
        return "RUN_OR_RESUME_MX03"
    if any("RUNTIME_MODERNIZATION" in blocker or "AMP_MIGRATION" in blocker for blocker in blockers):
        return "FINALIZE_MX04_RUNTIME_MODERNIZATION_EVIDENCE"
    if any("THROUGHPUT_PROFILE" in blocker for blocker in blockers):
        return "RUN_MX04_THROUGHPUT_PROFILE"
    if any("EXACT_RESUME" in blocker for blocker in blockers):
        return "REPAIR_MX04_EXACT_RESUME_PREFLIGHT"
    if any("TRAINER_FLAGS_ABSENT" in blocker for blocker in blockers):
        return "IMPLEMENT_MX04_TRAINER_CONTRACT"
    if any("BASELINE_BATTERY_NOT_FINALIZED" in blocker for blocker in blockers):
        return "FINALIZE_METHOD_X_BASELINE_BATTERY"
    return "RESOLVE_MX04_AUDIT_BLOCKERS"


def _supervise(root: Path, output_root: Path, phase: str, protocol: Mapping[str, Any], pid: int | None = None) -> dict[str, Any]:
    first_report = evaluate_autonomous_supervision(
        root=root,
        output_root=output_root,
        phase=phase,
        config={"resolution": protocol["resolution"], "folds": protocol["folds"], "seed": protocol["seed"], "hardware": {"hard_stop_vram_gb": 7.5}},
        expected_pid=pid,
    )
    # Only the live-trainer watch path defers the transient C inventory race.
    # Preflight/audit/finalize phases keep fail-closed BLOCK on the same code.
    if str(phase) not in {"finetune_watch", "watch"}:
        return first_report

    def _retry_watch() -> dict[str, Any]:
        time.sleep(WATCH_INVENTORY_RACE_RETRY_SECONDS)
        return evaluate_autonomous_supervision(
            root=root,
            output_root=output_root,
            phase=phase,
            config={"resolution": protocol["resolution"], "folds": protocol["folds"], "seed": protocol["seed"], "hardware": {"hard_stop_vram_gb": 7.5}},
            expected_pid=pid,
        )

    watch_report, _should_enforce = resolve_watch_inventory_race(
        first_report=first_report,
        retry_probe=_retry_watch,
    )
    return watch_report


def _run_or_reuse_candidate_task(
    *,
    root: Path,
    output_root: Path,
    state_path: Path,
    state: dict[str, Any],
    report: dict[str, Any],
    protocol: Mapping[str, Any],
    config: Mapping[str, Any],
    family: str,
    candidate: Mapping[str, Any],
    seed: int,
    fold_manifest_path: Path | None,
    execution_contract: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Run one exact selection contract or validate and reuse its terminal result."""

    candidate_id = str(candidate.get("id"))
    task_id = f"{family}__{candidate_id}__seed{int(seed)}__inner"
    candidate_root = (
        output_root
        / "finetune_v1"
        / family
        / candidate_id
        / f"seed_{int(seed)}_inner"
    )
    metrics_path = candidate_root / "three_d_metrics.json"
    tasks = state.setdefault("tasks", {})
    task = tasks.get(task_id, {}) if isinstance(tasks, Mapping) else {}
    command = build_candidate_command(
        output_root=candidate_root,
        family=family,
        candidate=candidate,
        seed=int(seed),
        selection_only=True,
        fold_manifest=fold_manifest_path,
        protocol=protocol,
        execution_contract=execution_contract,
    )
    contract_digest = hashlib.sha256(json.dumps(command, sort_keys=True).encode()).hexdigest()
    if task and task.get("contract") and str(task.get("contract")) != contract_digest:
        report["status"] = "BLOCKED"
        report["blockers"].append(f"TASK_CONTRACT_MISMATCH:{task_id}")
        return None
    if task.get("status") == "COMPLETED" and metrics_path.is_file():
        try:
            score = _inner_metric(metrics_path, folds=int(protocol["folds"]))
            vram = _assert_inner_vram_below_hard_stop(
                metrics_path,
                hard_stop_gb=float(config["promotion"]["memory_hard_stop_gb"]),
            )
        except RuntimeError as exc:
            report["status"] = "BLOCKED"
            report["blockers"].append(f"{exc}:{task_id}")
            return None
        return {
            "task_id": task_id,
            "family": family,
            "candidate": candidate_id,
            "seed": int(seed),
            "score": score,
            "model_root": str(candidate_root.resolve()),
            "status": "COMPLETED",
            "training_started": True,
            "outer_test_used_for_selection": False,
            "contract": contract_digest,
            "execution_contract": dict(execution_contract),
            **vram,
            "reused": True,
        }

    if _MX04_STOP_REQUESTED:
        tasks[task_id] = {
            **(dict(task) if isinstance(task, Mapping) else {}),
            "status": "PAUSED_RESUMABLE",
            "contract": contract_digest,
            "execution_contract": dict(execution_contract),
            "updated_at_utc": _now(),
            "pause_reason": "OPERATOR_STOP_BEFORE_TASK_LAUNCH",
        }
        atomic_write_json(state_path, state)
        report["status"] = "PAUSED_RESUMABLE"
        report["blockers"].append(f"TASK_PAUSED_RESUMABLE:{task_id}")
        report["next_automatic_action"] = "RESUME_MX04_EXACT_CONTRACT"
        return None

    candidate_root.mkdir(parents=True, exist_ok=True)
    tasks[task_id] = {
        **(dict(task) if isinstance(task, Mapping) else {}),
        "status": "RUNNING",
        "contract": contract_digest,
        "resume_attempt": int(task.get("resume_attempt", 0)) + (1 if task else 0),
        "execution_contract": dict(execution_contract),
        "updated_at_utc": _now(),
    }
    atomic_write_json(state_path, state)
    execution = run_supervised_process(
        command,
        cwd=root,
        output_root=candidate_root,
        interval_seconds=15.0,
        max_idle_seconds=3600.0,
        check=lambda pid, _protocol=protocol: _supervise(
            root,
            output_root,
            "finetune_watch",
            _protocol,
            pid,
        ),
    )
    report["training_started"] = True
    code = int(execution.get("returncode", 78))
    if code == 75:
        tasks[task_id].update(
            status="PAUSED_RESUMABLE",
            execution=execution,
            updated_at_utc=_now(),
        )
        atomic_write_json(state_path, state)
        report["status"] = "PAUSED_RESUMABLE"
        report["blockers"].append(f"TASK_PAUSED_RESUMABLE:{task_id}")
        report["next_automatic_action"] = "RESUME_MX04_EXACT_CONTRACT"
        return None
    if code != 0:
        failure = record_failure(
            output_root / LOOP_STATE,
            stage="MX-04",
            contract_digest=contract_digest,
            failure_signature=f"TASK_EXIT_{code}:{task_id}",
            max_same_failure_attempts=int(config["max_same_failure_attempts"]),
        )
        tasks[task_id].update(
            status="BLOCKED",
            execution=execution,
            loop_guard=failure,
            updated_at_utc=_now(),
        )
        atomic_write_json(state_path, state)
        report["status"] = "BLOCKED"
        report["blockers"].append(f"TASK_EXECUTION_BLOCKED:{task_id}")
        return None
    try:
        score = _inner_metric(metrics_path, folds=int(protocol["folds"]))
        vram = _assert_inner_vram_below_hard_stop(
            metrics_path,
            hard_stop_gb=float(config["promotion"]["memory_hard_stop_gb"]),
        )
    except RuntimeError as exc:
        report["status"] = "BLOCKED"
        report["blockers"].append(f"{exc}:{task_id}")
        return None
    tasks[task_id].update(
        status="COMPLETED",
        execution=execution,
        score=score,
        **vram,
        updated_at_utc=_now(),
    )
    atomic_write_json(state_path, state)
    record_success(
        output_root / LOOP_STATE,
        stage="MX-04",
        contract_digest=contract_digest,
    )
    return {
        "task_id": task_id,
        "family": family,
        "candidate": candidate_id,
        "seed": int(seed),
        "status": "COMPLETED",
        "score": score,
        "model_root": str(candidate_root.resolve()),
        "training_started": True,
        "outer_test_used_for_selection": False,
        "contract": contract_digest,
        "execution_contract": dict(execution_contract),
        **vram,
        "reused": False,
    }


def _run_replication_and_finalize(
    *,
    root: Path,
    output_root: Path,
    state_path: Path,
    state: dict[str, Any],
    report: dict[str, Any],
    protocol: Mapping[str, Any],
    config: Mapping[str, Any],
    families: list[str],
    candidates: list[Mapping[str, Any]],
    fold_manifests: Mapping[str, Path],
    family_winners: Mapping[str, Mapping[str, Any]],
    screening_scores: list[dict[str, Any]],
    execution_contracts: Mapping[str, Mapping[str, Any]],
) -> None:
    """Replicate five family winners plus the fixed comparator and hand off."""

    candidate_by_id = {str(candidate["id"]): candidate for candidate in candidates}
    seeds = [int(seed) for seed in config.get("replication_seeds", [])]
    if seeds != [13, 29, 47]:
        report["status"] = "BLOCKED"
        report["blockers"].append("FINETUNE_REPLICATION_SEEDS_MISMATCH")
        return
    comparator_family = "resnet34_3d"
    comparator_candidate = "scratch_baseline"
    if comparator_candidate not in candidate_by_id:
        report["status"] = "BLOCKED"
        report["blockers"].append("FINETUNE_FIXED_COMPARATOR_MISSING")
        return

    try:
        replication_spec = build_replication_spec(
            family_winners,
            family_order=families,
            seeds=seeds,
            comparator_family=comparator_family,
            comparator_candidate=comparator_candidate,
        )
    except RuntimeError as exc:
        report["status"] = "BLOCKED"
        report["blockers"].append(str(exc))
        return

    replication_tasks: list[dict[str, Any]] = []
    validated_runs: dict[tuple[str, str, int], dict[str, Any]] = {}
    for spec in replication_spec:
        family = str(spec["family"])
        candidate_id = str(spec["candidate"])
        seed = int(spec["seed"])
        candidate = candidate_by_id.get(candidate_id)
        if candidate is None:
            report["status"] = "BLOCKED"
            report["blockers"].append(f"FINETUNE_REPLICATION_CANDIDATE_INVALID:{candidate_id}")
            return
        row = _run_or_reuse_candidate_task(
            root=root,
            output_root=output_root,
            state_path=state_path,
            state=state,
            report=report,
            protocol=protocol,
            config=config,
            family=family,
            candidate=candidate,
            seed=seed,
            fold_manifest_path=fold_manifests[family],
            execution_contract=execution_contracts[family],
        )
        if row is None:
            return
        try:
            validated = validate_selection_run(
                row["model_root"],
                expected_backbone=family,
                expected_seed=seed,
                folds=int(protocol["folds"]),
            )
        except RuntimeError as exc:
            report["status"] = "BLOCKED"
            report["blockers"].append(f"{exc}:{row['task_id']}")
            return
        validated_runs[(family, candidate_id, seed)] = validated
        replication_tasks.append(
            {
                **row,
                "role": spec["role"],
                "selection_evidence_sha256": validated["selection_evidence_sha256"],
                "roc_auc": validated["roc_auc"],
                "brier": validated["brier"],
                "ece": validated["ece"],
                "peak_vram_allocated_gb": validated["peak_vram_allocated_gb"],
                "peak_vram_reserved_gb": validated["peak_vram_reserved_gb"],
            }
        )
    report["replication_tasks"] = replication_tasks

    comparator_runs = [
        validated_runs[(comparator_family, comparator_candidate, seed)]
        for seed in seeds
    ]
    sanity = _mx03_sanity_evidence(output_root)
    if sanity.get("passed") is not True:
        report["status"] = "BLOCKED"
        report["blockers"].append(str(sanity.get("reason") or "MX03_SANITY_CHECKS_NOT_PASSED"))
        return

    family_rows: list[dict[str, Any]] = []
    for family in families:
        candidate_id = str(family_winners[family]["candidate"])
        runs = [validated_runs[(family, candidate_id, seed)] for seed in seeds]
        try:
            evidence = build_promotion_evidence(
                candidate_runs=runs,
                comparator_runs=comparator_runs,
                sanity_checks_passed=True,
                bootstrap_resamples=int(config["promotion"]["bootstrap_resamples"]),
                bootstrap_seed=13,
            )
        except RuntimeError as exc:
            report["status"] = "BLOCKED"
            report["blockers"].append(f"{exc}:{family}:{candidate_id}")
            return
        would_promote, reasons = evaluate_promotion_gate(
            promotion=config["promotion"],
            evidence=evidence,
        )
        family_rows.append(
            {
                "family": family,
                "candidate": candidate_id,
                "replication_seeds": seeds,
                "canonical_seed": 13,
                "canonical_model_root": str(
                    output_root
                    / "finetune_v1"
                    / family
                    / candidate_id
                    / "seed_13_inner"
                ),
                "model_roots": {str(run["seed"]): run["model_root"] for run in runs},
                "mean_inner_roc_auc": statistics.fmean(float(run["roc_auc"]) for run in runs),
                "mean_inner_brier": statistics.fmean(float(run["brier"]) for run in runs),
                "mean_inner_ece": statistics.fmean(float(run["ece"]) for run in runs),
                "promotion_evidence": evidence,
                "would_promote": would_promote,
                "promotion_reasons": reasons,
                "outer_test_access_count": 0,
            }
        )

    comparator_root = str(
        output_root
        / "finetune_v1"
        / comparator_family
        / comparator_candidate
        / "seed_13_inner"
    )
    handoff = choose_handoff(
        family_rows,
        family_order=families,
        comparator={
            "family": comparator_family,
            "candidate": comparator_candidate,
            "model_root": comparator_root,
        },
    )
    execution_contract_by_family = {
        family: dict(execution_contracts[family]) for family in families
    }
    comparator_summary = {
        "family": comparator_family,
        "candidate": comparator_candidate,
        "replication_seeds": seeds,
        "canonical_seed": 13,
        "canonical_model_root": comparator_root,
        "model_roots": {str(run["seed"]): run["model_root"] for run in comparator_runs},
        "outer_test_access_count": 0,
        "peak_vram_allocated_gb": max(
            float(run["peak_vram_allocated_gb"]) for run in comparator_runs
        ),
        "peak_vram_reserved_gb": max(
            float(run["peak_vram_reserved_gb"]) for run in comparator_runs
        ),
        "execution_contract": execution_contract_by_family[comparator_family],
    }
    negative_ledger = [
        {
            "family": row["family"],
            "candidate": row["candidate"],
            "decision": "NOT_PROMOTED",
            "reasons": row["promotion_reasons"],
        }
        for row in family_rows
        if row["would_promote"] is not True
    ]
    manifest = {
        "schema_version": 2,
        "status": "COMPLETED",
        "stage": "MX-04",
        "selection_scope": "inner_validation_only",
        "screening_seed": 13,
        "screening_scores": screening_scores,
        "selected_candidates_by_family": {
            row["family"]: {
                **row,
                "execution_contract": execution_contract_by_family[row["family"]],
            }
            for row in family_rows
        },
        "family_count": len(family_rows),
        "replication_seeds": seeds,
        "fixed_comparator": comparator_summary,
        "mx03_sanity_evidence": sanity,
        "handoff": handoff,
        "no_forced_winner": True,
        "outer_test_used_for_selection": False,
        "outer_evaluation": {"status": "DEFERRED_TO_MX05", "count": 0},
        "final_model_selection_manifest_written_by_mx04": False,
        "negative_ledger": negative_ledger,
        "created_at_utc": _now(),
    }
    atomic_write_json(output_root / "reports/finetune_selection_manifest.json", manifest)
    atomic_write_json(
        output_root / "reports/mx04_negative_ledger.json",
        {
            "schema_version": 1,
            "status": "COMPLETED",
            "entries": negative_ledger,
            "created_at_utc": _now(),
        },
    )
    report.update(
        {
            "status": "COMPLETED",
            "selected_candidates_by_family": manifest["selected_candidates_by_family"],
            "fixed_comparator": comparator_summary,
            "mx03_sanity_evidence": sanity,
            "handoff": handoff,
            "outer_evaluation": {"status": "DEFERRED_TO_MX05", "count": 0},
            "outer_test_used_for_selection": False,
            "next_automatic_action": "RUN_MX05_HYBRID_ABLATIONS",
        }
    )


def run_finetune_orchestrator(
    *,
    root: Path,
    output_root: Path,
    config_path: Path,
    mode: str = "audit",
    persist_audit: bool = True,
) -> dict[str, Any]:
    root = root.resolve()
    output_root = output_root.resolve()
    config_load_blockers: list[str] = []
    try:
        loaded_config = load_unique_yaml(config_path)
    except (OSError, UnicodeError, ValueError) as exc:
        loaded_config = {}
        config_load_blockers.append(f"FINETUNE_CONFIG_INVALID:{type(exc).__name__}")
    config = loaded_config if isinstance(loaded_config, Mapping) else {}
    if not isinstance(loaded_config, Mapping):
        config_load_blockers.append("FINETUNE_CONFIG_NOT_MAPPING")
    battery = _read(output_root / "battery_manifest.json")
    protocol = _protocol_args(battery)
    supervision = _supervise(root, output_root, "finetune_orchestrator", protocol)
    if mode != "audit" or persist_audit:
        enforce_autonomous_supervision(root=root, output_root=output_root, report=supervision)
    blockers = [*config_load_blockers, *_validate_config(config, root=root)]
    blockers.extend(_exact_resume_storage_preflight(config, root=root))
    blockers.extend(_trainer_flag_subset_blockers(root=root))
    if supervision.get("decision") == "BLOCK":
        blockers.append("AUTOSUPERVISOR_BLOCKED")
    if not _baseline_terminal(battery):
        blockers.append("BASELINE_BATTERY_NOT_FINALIZED")
    if not _mx03_terminal(output_root):
        blockers.append("MX03_VALIDITY_NOT_COMPLETED")
    report: dict[str, Any] = {
        "schema_version": 1,
        "stage": "MX-04",
        "mode": mode,
        "status": "BLOCKED" if blockers else "READY",
        "training_started": False,
        "selection_scope": config.get("selection_scope"),
        "outer_test_used_for_selection": False,
        "blockers": sorted(set(blockers)),
        "candidate_tasks": [],
        "supervision_decision": supervision.get("decision"),
        "autonomous_supervision": {"mode": "AUTONOMOUS_MULTI_LAYER_FAIL_CLOSED", "human_approval_required": False, "manual_override_allowed": False, "fail_closed": True},
        "next_automatic_action": _blocked_next_action(blockers) if blockers else "RUN_MX04_INNER_SCREENING",
    }
    if blockers or mode == "audit":
        if mode != "audit" or persist_audit:
            if blockers:
                atomic_write_json(output_root / "reports/BLOCKED_MX04_FINETUNE.json", report)
            atomic_write_json(output_root / "reports/mx04_finetune_readiness.json", report)
            atomic_write_json(output_root / "reports/mx04_finetune_orchestrator.json", report)
        return report

    candidates = [row for row in config.get("candidates", []) if isinstance(row, Mapping)]
    families = [str(value) for value in config.get("candidate_families", [])]
    execution_contracts = load_mx04_execution_contracts(
        config,
        root=root,
        families=tuple(families),
    )
    report["execution_contracts_by_family"] = execution_contracts
    state_path = output_root / "reports/mx04_task_state.json"
    state = _read(state_path) if state_path.is_file() else {"schema_version": 1, "tasks": {}}
    scores: list[dict[str, Any]] = []
    fold_manifests: dict[str, Path] = {}
    for family in families:
        fold_manifest_path, _fold_manifest = build_pretrained_fold_manifest(
            output_root=output_root,
            backbone=family,
            folds=int(protocol["folds"]),
        )
        fold_manifests[family] = fold_manifest_path
        for candidate in candidates:
            row = _run_or_reuse_candidate_task(
                root=root,
                output_root=output_root,
                state_path=state_path,
                state=state,
                report=report,
                protocol=protocol,
                config=config,
                family=family,
                candidate=candidate,
                seed=13,
                fold_manifest_path=fold_manifest_path,
                execution_contract=execution_contracts[family],
            )
            if row is None:
                break
            scores.append(row)
            report["candidate_tasks"].append(row)
        if report["status"] in {"BLOCKED", "PAUSED_RESUMABLE"}:
            break

    if report["status"] not in {"BLOCKED", "PAUSED_RESUMABLE"}:
        if not scores:
            report["status"] = "BLOCKED"
            report["blockers"].append("FINETUNE_NO_COMPLETED_INNER_CANDIDATE")
        elif len(scores) != len(families) * len(candidates):
            report["status"] = "BLOCKED"
            report["blockers"].append("FINETUNE_INNER_GRID_INCOMPLETE")
        else:
            # Lock one candidate per family. Promotion remains unevaluated
            # until all five winners and the fixed comparator have exact
            # three-seed, inner-only evidence.
            family_winners = select_inner_winners_by_family(
                scores,
                families=families,
                candidate_ids=[str(row["id"]) for row in candidates],
            )
            report["selected_candidates_by_family"] = family_winners
            report["promotion_gate_at_inner_lock"] = {
                "status": "NOT_EVALUATED_REQUIRES_THREE_SEED_INNER_REPLICATION",
                "would_promote": None,
                "reasons": ["PROMOTION_EVIDENCE_NOT_YET_AVAILABLE"],
            }
            report["status"] = "LOCKED_INNER_VALIDATION"
            report["next_automatic_action"] = "RUN_MX04_REPLICATION_SEEDS_INNER_ONLY"
            atomic_write_json(
                output_root / "reports/finetune_selection_manifest.json",
                {
                    "schema_version": 2,
                    "status": "LOCKED_INNER_VALIDATION",
                    "selected_candidates_by_family": family_winners,
                    "family_count": len(family_winners),
                    "replicate_each_family_winner": True,
                    "selection_scope": "inner_validation_only",
                    "outer_test_used_for_selection": False,
                    "scores": scores,
                    "promotion_gate_at_inner_lock": report.get("promotion_gate_at_inner_lock"),
                    "replication_seeds": config.get("replication_seeds"),
                    "execution_contracts_by_family": execution_contracts,
                    "outer_evaluation": {"status": "DEFERRED_TO_MX05", "count": 0},
                    "created_at_utc": _now(),
                },
            )
            _run_replication_and_finalize(
                root=root,
                output_root=output_root,
                state_path=state_path,
                state=state,
                report=report,
                protocol=protocol,
                config=config,
                families=families,
                candidates=candidates,
                fold_manifests=fold_manifests,
                family_winners=family_winners,
                screening_scores=scores,
                execution_contracts=execution_contracts,
            )
    report["blockers"] = sorted(set(str(value) for value in report["blockers"]))
    atomic_write_json(output_root / "reports/mx04_finetune_readiness.json", report)
    atomic_write_json(output_root / "reports/mx04_finetune_orchestrator.json", report)
    if report["blockers"]:
        atomic_write_json(output_root / "reports/BLOCKED_MX04_FINETUNE.json", report)
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
    _install_mx04_signal_handlers()
    output_root = args.output_root if args.output_root.is_absolute() else ROOT / args.output_root
    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    if args.no_persist_audit and args.mode != "audit":
        parser.error("--no-persist-audit is valid only with --mode audit")
    report = run_finetune_orchestrator(
        root=ROOT,
        output_root=output_root,
        config_path=config_path,
        mode=args.mode,
        persist_audit=not args.no_persist_audit,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True))
    if report.get("status") == "PAUSED_RESUMABLE":
        return 75
    # An inner-validation lock is a scientifically valid intermediate state,
    # not a terminal fine-tuning result.  Fail closed so MX-05/XAI cannot be
    # unlocked until the prespecified replication stage is materialized.
    return 0 if report.get("status") in {"READY", "COMPLETED"} else 3


if __name__ == "__main__":
    raise SystemExit(main())
