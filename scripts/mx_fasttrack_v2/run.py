"""Versioned four-task MX0-3 + focused MX4 + two-arm hybrid runner.

It NEVER invokes either full original orchestrator, changes completed battery
artifacts, reads an outer-test label during selection, or disables resume.
Use `plan` first, then `mx4`, `smoke`, `profile`, `mx5`, `lock`.  The legacy
`run` action remains an alias for both training groups. Run from the actual
project root.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SOURCE_PATHS = (
    "scripts/train_3d_mri.py", "scripts/run_method_x_finetune_orchestrator.py",
    "scripts/run_method_x_hybrid_orchestrator.py",
    "scripts/smoke_method_x_backbone.py", "src/training/method_x_finetune.py",
    "src/training/method_x_resume.py", "src/training/method_x_spm_snapshot.py",
    "src/utils/final_claim_policy.py",
    "src/utils/reproducibility.py",
    "src/models/backbones/pretrained_loader.py",
    "src/models/backbones/backbone_factory.py",
    "src/models/backbones/deterministic_pool.py",
    "src/models/backbones/lightweight_3d.py",
    "src/models/backbones/modern_3d.py",
    "src/models/backbones/amuse_x_hybrid3d.py",
    "src/data/combined_mri_cache.py",
    "scripts/mx_fasttrack_v2/capture_preflight.py",
    "scripts/mx_fasttrack_v2/supervise.py",
    "scripts/mx_fasttrack_v2/run.py",
    "scripts/mx_fasttrack_v2/evaluate_locked.py",
    "scripts/mx_fasttrack_v2/audit_inputs.py",
    "scripts/mx_fasttrack_v2/integrity.py",
    "scripts/mx_fasttrack_v2/evaluate_hybrid_pair.py",
    "scripts/mx_fasttrack_v2/xai_limited.py",
    "scripts/mx_fasttrack_v2/freeze_report.py",
    # The tools that produced the MX-04 import evidence.  They must be bound too,
    # otherwise the evidence chain is produced by unbound code.
    "scripts/mx_fasttrack_v2/import_mx04.py",
    "scripts/mx_fasttrack_v2/reattest_mx04_staging_prune.py",
    "configs/experiments/method_x_fasttrack_v2.json",
    "configs/experiments/method_x_fasttrack_v2_20260926_mx04_import.json",
    "configs/experiments/method_x_fasttrack_v2_20260927_mx05_gate_import.json",
    "scripts/mx_fasttrack_v2_4/__init__.py",
    "scripts/mx_fasttrack_v2_4/import_resume.py",
    "scripts/mx_fasttrack_v2_4/outer_guard_runner.py",
    "scripts/mx_fasttrack_v2_4/supervise_v24.py",
    "docs/method_x_fasttrack_v24_resume_FA.md",
    "docs/method_x_fasttrack_v2_runbook_FA.md",
    "src/evaluation/method_x_xai_forward.py",
    "src/evaluation/xai_validation.py",
    "src/evaluation/xai_regions.py",
    "scripts/run_method_x_anatomical_xai.py",
    "configs/experiments/method_x_finetune_v1.yaml",
    "configs/experiments/method_x_hybrid_v1.yaml",
)
TASKS = (
    ("mx04_scratch", "mx04", "scratch_baseline"),
    ("mx04_head_last", "mx04", "head_then_last_stage_llrd"),
    ("hybrid_baseline", "hybrid", "baseline"),
    ("hybrid_dense", "hybrid", "plus_dense_skip"),
)
# Accepted protocol ids for this scope check.  The 20260923 tree is the historical
# one; 20260926_mx04_import is the same locked protocol re-preregistered after the
# MX-05 adaptive-pool determinism repair, with the completed MX-04 inner runs
# imported under an explicit manifest rather than retrained.  Both share identical
# scientific scope: same seed, folds, epochs, candidates, variants and
# inner-validation-only selection.  Widening this set is a versioning act and must
# never be used to admit a different experimental scope.
FASTTRACK_PROTOCOL_IDS = (
    "method_x_fasttrack_v2_20260923",
    "method_x_fasttrack_v2_20260926_mx04_import",
    "method_x_fasttrack_v2_20260927_mx05_gate_import",
)
# One config file per protocol id.  Sharing a single mutable file across two live
# protocol versions made them bind the same bytes, so bumping protocol_id silently
# invalidated the parent preregistration's fasttrack_config_sha256.
FASTTRACK_CONFIGS = {
    "method_x_fasttrack_v2_20260923": "configs/experiments/method_x_fasttrack_v2.json",
    "method_x_fasttrack_v2_20260926_mx04_import":
        "configs/experiments/method_x_fasttrack_v2_20260926_mx04_import.json",
    "method_x_fasttrack_v2_20260927_mx05_gate_import":
        "configs/experiments/method_x_fasttrack_v2_20260927_mx05_gate_import.json",
}
CHECKPOINT_STAGING_ROOT = Path(r"C:\Users\USER\AppData\Local\AMUSE-X\checkpoint_staging")


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fp:
        for block in iter(lambda: fp.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"EXPECTED_JSON_OBJECT:{path}")
    return value


def write_once(path: Path, value: dict) -> None:
    data = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)
    if path.exists():
        if read(path) != value:
            raise RuntimeError(f"IMMUTABLE_FILE_CHANGED:{path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".new")
    with tmp.open("x", encoding="utf-8") as fp:
        fp.write(data)
    os.replace(tmp, path)


def update_state(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".new")
    tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)


def prune_completed_task_staging(
    *,
    root: Path,
    out: Path,
    task_id: str,
    model_root: Path,
    backbone: str,
    expected_folds: int,
    staging_root: Path = CHECKPOINT_STAGING_ROOT,
) -> dict:
    """Prune only terminal task payloads after their H mirrors are verified.

    Incomplete tasks retain both C generations.  The immutable plan makes a
    crash during pruning resumable without broad directory deletion.
    """

    if (
        task_id == "hybrid_baseline"
        and (out / "reports/mx05_resume_import.v1.json").is_file()
    ):
        from scripts.mx_fasttrack_v2_4.import_resume import imported_baseline_prune

        contract = read(model_root / "run_contract.json")
        digest = str(contract.get("run_contract_digest") or "")
        if (len(digest) != 64 or contract.get("run_contract", {}).get("digest") != digest
            or not (model_root / "three_d_metrics.json").is_file()):
            raise RuntimeError("V24_IMPORTED_BASELINE_NOT_TERMINAL")
        return imported_baseline_prune(root, out, model_root, digest, expected_folds)

    contract_path = model_root / "run_contract.json"
    metrics_path = model_root / "three_d_metrics.json"
    contract = read(contract_path)
    digest = str(contract.get("run_contract_digest") or "")
    if (len(digest) != 64 or contract.get("run_contract", {}).get("digest") != digest
        or not metrics_path.is_file()):
        raise RuntimeError(f"STAGING_PRUNE_TASK_NOT_TERMINAL:{task_id}")
    run_id = digest[:16]
    staging_root = staging_root.resolve()
    task_staging = (staging_root / run_id / backbone).resolve()
    if (not task_staging.is_relative_to(staging_root)
        or task_staging.parent.parent != staging_root
        or task_staging.name != backbone):
        raise RuntimeError(f"STAGING_PRUNE_PATH_UNSAFE:{task_id}")
    reports = out / "reports"
    plan_path = reports / f"staging_prune_{task_id}_plan.json"
    receipt_path = reports / f"staging_prune_{task_id}.json"
    h_folds = []
    for fold in range(expected_folds):
        final = model_root / f"method_x_3d_fold_{fold}.pt"
        mirror = model_root / "checkpoint_mirror" / f"fold_{fold}" / "resume_current.pt"
        mirror_state = mirror.with_name("checkpoint_mirror_state.json")
        if not final.is_file() or not mirror.is_file() or not mirror_state.is_file():
            raise RuntimeError(f"STAGING_PRUNE_H_TERMINAL_EVIDENCE_MISSING:{task_id}:{fold}")
        mirror_binding = read(mirror_state)
        mirror_digest = sha(mirror)
        if (mirror_binding.get("payload_sha256") != mirror_digest
            or mirror_binding.get("run_id") != run_id
            or mirror_binding.get("backbone") != backbone
            or mirror_binding.get("fold") != fold):
            raise RuntimeError(f"STAGING_PRUNE_H_MIRROR_BINDING_INVALID:{task_id}:{fold}")
        h_folds.append({
            "fold": fold,
            "final_checkpoint": final.resolve().relative_to(root).as_posix(),
            "final_checkpoint_sha256": sha(final),
            "mirror": mirror.resolve().relative_to(root).as_posix(),
            "mirror_sha256": mirror_digest,
            "mirror_state": mirror_state.resolve().relative_to(root).as_posix(),
            "mirror_state_sha256": sha(mirror_state),
        })
    if plan_path.is_file():
        plan = read(plan_path)
        if (plan.get("schema_version") != 1 or plan.get("task_id") != task_id
            or plan.get("run_contract_digest") != digest or plan.get("run_id") != run_id
            or plan.get("backbone") != backbone or plan.get("expected_folds") != expected_folds
            or plan.get("metrics_sha256") != sha(metrics_path)
            or plan.get("h_folds") != h_folds):
            raise RuntimeError(f"STAGING_PRUNE_PLAN_DRIFT:{task_id}")
    else:
        c_files = []
        for fold in range(expected_folds):
            fold_dir = task_staging / str(fold)
            if not fold_dir.is_dir():
                raise RuntimeError(f"STAGING_PRUNE_C_FOLD_MISSING:{task_id}:{fold}")
            children = sorted(fold_dir.iterdir(), key=lambda path: path.name.casefold())
            if (any(not child.is_file() for child in children)
                or {child.name for child in children} - {"resume_current.pt", "resume_previous.pt"}
                or not (fold_dir / "resume_current.pt").is_file()):
                raise RuntimeError(f"STAGING_PRUNE_C_LAYOUT_UNEXPECTED:{task_id}:{fold}")
            current = fold_dir / "resume_current.pt"
            if sha(current) != h_folds[fold]["mirror_sha256"]:
                raise RuntimeError(f"STAGING_PRUNE_C_H_MIRROR_MISMATCH:{task_id}:{fold}")
            for child in children:
                c_files.append({
                    "relative_path": child.resolve().relative_to(staging_root).as_posix(),
                    "sha256": sha(child),
                    "size_bytes": child.stat().st_size,
                })
        plan = {
            "schema_version": 1,
            "task_id": task_id,
            "run_contract_digest": digest,
            "run_id": run_id,
            "backbone": backbone,
            "expected_folds": expected_folds,
            "metrics_sha256": sha(metrics_path),
            "h_folds": h_folds,
            "c_files": c_files,
            "scope": "ONLY_THIS_COMPLETED_FASTTRACK_TASK",
            "preexisting_staging_outside_task_preserved": True,
        }
        write_once(plan_path, plan)
    recorded = plan.get("c_files") or []
    if not isinstance(recorded, list) or not recorded:
        raise RuntimeError(f"STAGING_PRUNE_PLAN_C_FILES_INVALID:{task_id}")
    for row in recorded:
        relative = Path(str(row.get("relative_path") or ""))
        candidate = (staging_root / relative).resolve()
        if (relative.is_absolute() or ".." in relative.parts
            or not candidate.is_relative_to(task_staging)
            or candidate.name not in {"resume_current.pt", "resume_previous.pt"}):
            raise RuntimeError(f"STAGING_PRUNE_RECORDED_PATH_UNSAFE:{task_id}")
        if candidate.is_file():
            if sha(candidate) != row.get("sha256") or candidate.stat().st_size != row.get("size_bytes"):
                raise RuntimeError(f"STAGING_PRUNE_PAYLOAD_CHANGED:{task_id}:{candidate.name}")
            candidate.unlink()
    for fold in range(expected_folds):
        fold_dir = task_staging / str(fold)
        if fold_dir.is_dir() and not any(fold_dir.iterdir()):
            fold_dir.rmdir()
    for directory in (task_staging, task_staging.parent):
        if directory.is_dir() and not any(directory.iterdir()):
            directory.rmdir()
    receipt = {
        "schema_version": 1,
        "status": "COMPLETED",
        "task_id": task_id,
        "plan_sha256": sha(plan_path),
        "removed_file_count": len(recorded),
        "removed_bytes": sum(int(row["size_bytes"]) for row in recorded),
        # "removed_*" is the task's historical staging volume as recorded by the plan.
        # When the plan was transferred from an earlier tree those C payloads were
        # deleted *there*, so this receipt must not imply a deletion performed here.
        # The transfer marker lives in the plan, which is immutable, so the receipt
        # stays byte-stable across re-invocations of this idempotent step.
        "plan_declared_c_payload_removal_location": plan.get(
            "c_payload_removal_location", "this_tree_at_plan_creation"),
        "c_payloads_absent_after_verified_h_commit": all(
            not (staging_root / Path(str(row["relative_path"]))).exists() for row in recorded),
        "preexisting_staging_outside_task_preserved": True,
    }
    if receipt["c_payloads_absent_after_verified_h_commit"] is not True:
        raise RuntimeError(f"STAGING_PRUNE_POSTCONDITION_FAILED:{task_id}")
    write_once(receipt_path, receipt)
    return receipt


def resolve_config(root: Path, protocol_id: str) -> Path:
    """Return the config file that belongs to ``protocol_id``.

    Every reader must go through this.  A hardcoded path lets one protocol version
    validate against another version's config, which is how the parent
    preregistration's ``fasttrack_config_sha256`` was invalidated when the protocol
    id was bumped.
    """

    relative = FASTTRACK_CONFIGS.get(str(protocol_id))
    if not relative:
        raise RuntimeError(f"FASTTRACK_PROTOCOL_ID_UNKNOWN:{protocol_id}")
    return root / relative


def config_for_tree(root: Path, out: Path) -> dict:
    """Read the preregistered protocol id, then that protocol's own config."""

    prereg = read(out / "reports/fasttrack_preregistration.json")
    return read(resolve_config(root, str(prereg.get("protocol_id"))))


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value.lower()
    )


def fold_membership_digests(task_dir: Path, *, task_id: str, folds: int = 5) -> list[str]:
    """Return the authoritative per-fold split-membership digests for one task.

    The trainer records the digest in each per-fold state file
    (``details.split_membership_digest``, alongside ``stage=FINALIZED``,
    ``outer_test_access_count`` and ``hpo_selection_only``).  That is the location
    ``build_fold_manifest`` already treats as authoritative for the battery
    checkpoints, and it is the only place the digest exists.

    It used to be read from the ``three_d_metrics.json`` fold rows instead, which
    never carried the field, so every entry resolved to ``None`` and the lock step
    raised ``FOLD_MEMBERSHIP_DIGEST_MISSING`` for MX-04 -- a dead end reachable
    even when MX-05 had never been attempted.  Both the lock writer and
    ``integrity.check_lock`` now call this one function so the recorded value and
    the verified value can no longer diverge.

    Fails closed on a missing/unfinalized fold state or a malformed digest.  When
    a metrics row does carry the digest it must agree, which turns the two sources
    into a cross-check instead of a fallback.
    """

    digests: list[str] = []
    for fold in range(folds):
        state_path = task_dir / f"method_x_3d_fold_{fold}_state.json"
        if not state_path.is_file():
            raise RuntimeError(f"FOLD_STATE_MISSING:{task_id}:{fold}")
        state = read(state_path)
        if state.get("stage") != "FINALIZED":
            raise RuntimeError(f"FOLD_NOT_FINALIZED:{task_id}:{fold}")
        details = state.get("details") or {}
        value = details.get("split_membership_digest")
        if not _is_sha256(value):
            raise RuntimeError(f"FOLD_MEMBERSHIP_DIGEST_MISSING:{task_id}:{fold}")
        digests.append(value)
    return digests


def cross_check_metrics_membership(
    task_dir: Path, *, task_id: str, membership: Sequence[str], folds: int = 5
) -> None:
    """Reject a metrics file whose own digests contradict the fold state files."""

    metric_file = task_dir / "three_d_metrics.json"
    if not metric_file.is_file():
        raise RuntimeError(f"INNER_ONLY_METRICS_MISSING:{task_id}")
    rows = read(metric_file)["folds"]
    for fold in range(folds):
        recorded = next((r.get("split_membership_digest") for r in rows
                         if r.get("fold") == fold), None)
        if recorded is not None and recorded != membership[fold]:
            raise RuntimeError(f"FOLD_MEMBERSHIP_DIGEST_CONTRADICTION:{task_id}:{fold}")


def utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def paths(args: argparse.Namespace):
    root = Path(args.project_root).resolve()
    battery = (root / args.battery_root).resolve()
    out = (root / args.output_root).resolve()
    if root != ROOT.resolve() or out == battery or out == root or out.is_relative_to(battery):
        raise RuntimeError("RUNNER_ROOT_OR_OUTPUT_NOT_ISOLATED")
    if not (root / "scripts/train_3d_mri.py").is_file():
        raise RuntimeError("TRAINER_NOT_FOUND")
    return root, battery, out


def confirm(
    root: Path,
    battery: Path,
    out: Path,
    config_path: Path,
    *,
    persist_preregistration: bool = True,
) -> tuple[dict, dict, dict]:
    config = read(config_path)
    if (config["schema_version"], config["mx04_candidates"], config["hybrid_variants"],
        config["seed"], config["folds"], config["locked_epochs_field"]) != (
        1, ["scratch_baseline", "head_then_last_stage_llrd"],
        ["baseline", "plus_dense_skip"], 13, 5, 100):
        raise RuntimeError("FASTTRACK_CONFIG_DRIFT")
    if (config.get("protocol_id") not in FASTTRACK_PROTOCOL_IDS
        or config.get("effective_epoch_cap_per_fold") != 20
        or config.get("predeclared_outer_comparisons") != [
            "mx04_scratch_vs_head_exploratory", "hybrid_baseline_vs_dense_paired_exploratory"]
        or config.get("hybrid_a3") is not False or config.get("hybrid_a4") is not False
        or config.get("outer_test_used_for_selection") is not False
        or config.get("external_validation") is not False
        or config.get("mx06_limited", {}).get("sample_source") !=
           "locked_inner_validation_not_independent"):
        raise RuntimeError("FASTTRACK_V2_SCOPE_CONFIG_DRIFT")
    battery_manifest = read(battery / "battery_manifest.json")
    locked = battery_manifest.get("locked_protocol") or {}
    if (locked.get("folds"), locked.get("epochs"), locked.get("resolution"),
        locked.get("seed")) != (5, 100, 128, 13):
        raise RuntimeError("BATTERY_PROTOCOL_NOT_FIVE_FOLD_SEED13_EPOCH100_128")
    audit_path = out / "reports/data_semantics_audit.json"
    if not audit_path.is_file():
        raise RuntimeError("FULL_CACHE_DATA_SEMANTICS_AUDIT_MISSING")
    audit = read(audit_path)
    if audit.get("status") != "READY" or audit.get("verified_full_hdf5_real_n") != 1669:
        raise RuntimeError("FULL_CACHE_DATA_SEMANTICS_AUDIT_NOT_READY")
    cache_expected = str((battery_manifest.get("locked_inputs") or {}).get("cache_sha256") or "")
    if len(cache_expected) != 64 or cache_expected != audit.get("cache_sha256"):
        raise RuntimeError("CACHE_SHA_NOT_BOUND_TO_BATTERY")
    if audit.get("physical_order") != ["age", "education_years", "apoe4_carrier", "mmse_base", "sex"]:
        raise RuntimeError("TABULAR_PHYSICAL_ORDER_DRIFT")
    if audit.get("cohort_label_perfectly_confounding") is not True:
        raise RuntimeError("COHORT_LABEL_STATUS_CHANGED_REVIEW_PROTOCOL")
    if (root / "data/cached/t1_volume_cache_128.h5").is_file() is False:
        raise RuntimeError("FULL_CACHE_MISSING")
    preflight_path = out / "reports/machine_preflight.json"
    if not preflight_path.is_file():
        raise RuntimeError("MACHINE_PREFLIGHT_MISSING")
    preflight = read(preflight_path)
    if (preflight.get("status") != "READY" or preflight.get("blockers") != []
        or preflight.get("live_method_x_python_writers") != []
        or (preflight.get("source_package") or {}).get("all_match") is not True
        or (preflight.get("checkpoint_staging") or {}).get("preserved_without_deletion") is not True):
        raise RuntimeError("MACHINE_PREFLIGHT_NOT_READY")
    if sha(root / "data/cached/t1_volume_cache_128.h5") != cache_expected:
        raise RuntimeError("CACHE_CHANGED_SINCE_DATA_AUDIT")
    if (root / "splits/fold_0_train.csv").is_file() is False:
        raise RuntimeError("TRAIN_SPLITS_MISSING")
    split_sha = audit.get("split_sha256") or {}
    if set(split_sha) != {*(f"fold_{i}_{kind}" for i in range(5) for kind in ("test", "train")), "external_holdout"}:
        raise RuntimeError("SPLIT_AUDIT_COVERAGE_INCOMPLETE")
    for split_name, expected_digest in split_sha.items():
        if sha(root / "splits" / (split_name + ".csv")) != expected_digest:
            raise RuntimeError(f"SPLIT_CHANGED_SINCE_DATA_AUDIT:{split_name}")
    if sha(root / "docs/fasttrack_private/cache_full_index.csv") != audit.get("index_sha256"):
        raise RuntimeError("CACHE_INDEX_CHANGED_SINCE_DATA_AUDIT")
    fp = {rel: sha(root / rel) for rel in SOURCE_PATHS}
    identity = {
        "schema_version": 1, "protocol_id": config["protocol_id"],
        "battery_manifest_sha256": sha(battery / "battery_manifest.json"),
        "data_audit_sha256": sha(audit_path), "fasttrack_config_sha256": sha(config_path),
        "machine_preflight_sha256": sha(preflight_path),
        "training_source_sha256": fp, "cache_sha256": cache_expected,
        "selection_scope": "inner_validation_only", "outer_test_access_count": 0,
        "status": "LOCKED_BEFORE_GPU", "claim_cap": "POST_HOC_INTERNAL_TECHNICAL_EVIDENCE",
    }
    if persist_preregistration:
        write_once(out / "reports/fasttrack_preregistration.json", identity)
    return config, battery_manifest, identity


def execution_contract(root: Path, family: str, exact_resume: dict) -> dict:
    profile = read(root / "results/method_x_registry/MX04_THROUGHPUT_PROFILE.json")
    loader = profile.get("loader") or {}
    if (int(loader.get("num_workers", -1)), loader.get("persistent_workers"),
        bool(loader.get("pin_memory"))) != (2, False, True):
        raise RuntimeError("THROUGHPUT_LOADER_DRIFT_REQUIRES_REVIEW")
    checkpoint = profile.get("per_family_checkpoint", {}).get(family, {}).get("setting", "on")
    if checkpoint != "on":
        raise RuntimeError("CHECKPOINT_OFF_REQUIRES_NEW_PROFILE")
    if exact_resume.get("required") is not True or exact_resume.get("no_resume_forbidden") is not True:
        raise RuntimeError("EXACT_RESUME_POLICY_FAILED")
    return {
        "checkpoint": "on", "exact_resume": exact_resume,
        "loader": {"num_workers": 2, "persistent_workers": False,
                   "prefetch_factor": int(loader.get("prefetch_factor", 2)),
                   "pin_memory": True},
        "yellow_gates": {"torch_compile": "DROPPED", "channels_last": "DROPPED"},
    }


def build_fold_manifest(root: Path, battery: Path, out: Path) -> Path:
    """Sidecars are written ONLY to fast-track output; source weights are read-only."""
    source = battery / "models/resnet34_3d"
    contract = read(source / "run_contract.json")["run_contract"]
    contract_digest = contract.get("digest", "")
    if len(contract_digest) != 64:
        raise RuntimeError("REFERENCE_CONTRACT_DIGEST_MISSING")
    sidecars = out / "pretrained_provenance/resnet34_3d"
    entries = {}
    for fold in range(5):
        checkpoint = source / f"method_x_3d_fold_{fold}.pt"
        state = read(source / f"method_x_3d_fold_{fold}_state.json")
        if state.get("stage") != "FINALIZED":
            raise RuntimeError(f"REFERENCE_FOLD_NOT_FINALIZED:{fold}")
        split_digest = (state.get("details") or {}).get("split_membership_digest")
        if not isinstance(split_digest, str) or len(split_digest) != 64:
            raise RuntimeError(f"REFERENCE_SPLIT_DIGEST_INVALID:{fold}")
        checkpoint_hash = sha(checkpoint)
        sidecar = sidecars / f"fold_{fold}.json"
        write_once(sidecar, {
            "schema_version": 1, "checkpoint_sha256": checkpoint_hash,
            "sha256": checkpoint_hash, "license": "PROJECT_GENERATED_INTERNAL_RESEARCH_WEIGHTS",
            "source_type": "FOLD_LOCAL_METHOD_X_BASELINE", "source_backbone": "resnet34_3d",
            "source_fold": fold, "source_run_contract_digest": contract_digest,
            "source_split_membership_digest": split_digest,
            "raw_identifiers_stored": False, "external_download": False,
        })
        entries[str(fold)] = {
            "checkpoint": checkpoint.resolve().relative_to(root).as_posix(),
            "manifest": sidecar.resolve().relative_to(root).as_posix(),
            "sha256": checkpoint_hash,
        }
    manifest = sidecars / "pretrained_fold_manifest.json"
    write_once(manifest, {
        "schema_version": 1, "backbone": "resnet34_3d", "folds": entries,
        "coverage_minimum": 0.95, "source_contract_digest": contract_digest,
        "outer_test_used_for_selection": False,
    })
    from src.training.method_x_finetune import validate_pretrained_fold_manifest
    validate_pretrained_fold_manifest(manifest, folds=5,
        expected_backbone="resnet34_3d", allowed_root=root, minimum_coverage=0.95)
    return manifest


def task_command(task: tuple[str, str, str], root: Path, battery: Path,
                 out: Path, manifest: dict, fold_manifest: Path | None) -> tuple[list[str], Path]:
    task_id, kind, variant = task
    output = out / "inner_tasks" / task_id
    if kind == "mx04":
        import yaml
        from scripts.run_method_x_finetune_orchestrator import build_candidate_command, _protocol_args
        full_config = yaml.safe_load((root / "configs/experiments/method_x_finetune_v1.yaml").read_text(encoding="utf-8"))
        candidate = next((x for x in full_config["candidates"] if x["id"] == variant), None)
        if candidate is None:
            raise RuntimeError("MX04_CANDIDATE_ABSENT")
        cmd = build_candidate_command(
            output_root=output, family="resnet34_3d", candidate=candidate,
            seed=13, selection_only=True,
            fold_manifest=fold_manifest if candidate.get("pretrained") else None,
            protocol=_protocol_args(manifest),
            execution_contract=execution_contract(root, "resnet34_3d", full_config["exact_resume"]),
        )
    else:
        import yaml
        from scripts.run_method_x_hybrid_orchestrator import build_hybrid_command, _protocol
        full_config = yaml.safe_load((root / "configs/experiments/method_x_hybrid_v1.yaml").read_text(encoding="utf-8"))
        cmd = build_hybrid_command(
            root=root, output_dir=output, variant=variant, seed=13, selection_only=True,
            protocol=_protocol(manifest), config=full_config,
            execution_contract=execution_contract(root, "amuse_x_hybrid3d", full_config["exact_resume"]),
        )
    forbidden = {"--no_resume", "--no-resume", "--force_retrain"}
    if forbidden.intersection(cmd) or "--hpo_selection_only" not in cmd or "--phase" not in cmd:
        raise RuntimeError("INNER_ONLY_OR_RESUME_GUARD_FAILED")
    if "--q2_channel_gate" in cmd or "--latent_alignment_lambda" in cmd:
        raise RuntimeError("A3_A4_FORBIDDEN_IN_FASTTRACK")
    return cmd, output


def inner(path: Path) -> tuple[float, list[float]]:
    metrics = read(path)
    aggregate = metrics.get("aggregate") or {}
    rows = metrics.get("folds") or []
    if aggregate.get("status") != "COMPLETED" or aggregate.get("hpo_selection_only") is not True:
        raise RuntimeError("INNER_ONLY_METRICS_INCOMPLETE")
    if aggregate.get("outer_test_used_for_selection") is not False or aggregate.get("outer_test_access_count") != 0:
        raise RuntimeError("INNER_SELECTION_OUTER_ACCESS_INVALID")
    if int(aggregate.get("folds", -1)) != 5 or int(aggregate.get("completed_folds", -1)) != 5:
        raise RuntimeError("INNER_ONLY_FOLDS_INCOMPLETE")
    if int(aggregate.get("epoch_target", -1)) != 20:
        raise RuntimeError("EPOCH_BUDGET_20_PER_FOLD_REQUIRED")
    for field in ("vram_peak_allocated_gb", "vram_peak_reserved_gb"):
        peak = aggregate.get(field)
        if peak is None or not (0 <= float(peak) < 7.5):
            raise RuntimeError(f"TRAINING_VRAM_UNVERIFIED_OR_TOO_HIGH:{field}")
    if len(rows) != 5 or {row.get("fold") for row in rows} != set(range(5)):
        raise RuntimeError("INNER_ONLY_FOLD_IDENTITIES_INVALID")
    if any(row.get("outer_test_evaluation_count") != 0 for row in rows):
        raise RuntimeError("OUTER_TEST_ACCESSED_DURING_SELECTION")
    ordered = sorted(rows, key=lambda row: row["fold"])
    scores = [float(row["best_val_loss"]) for row in ordered]
    if not all(0 <= score < float("inf") for score in scores):
        raise RuntimeError("INNER_ONLY_LOSS_INVALID")
    return sum(scores) / 5, scores


def choose_without_contaminated_finetune_validation(
    rows: dict, *, margin: float, minimum_folds: int
) -> tuple[dict, str, str, str, bool]:
    """Keep fold-local supervised pretraining out of inner-val selection."""
    scratch, head = rows["mx04_scratch"], rows["mx04_head_last"]
    base, dense = rows["hybrid_baseline"], rows["hybrid_dense"]
    wins_dense = sum(a > b for a, b in zip(
        base["fold_best_val_loss"], dense["fold_best_val_loss"], strict=True))
    advantage_dense = base["mean_best_val_loss"] - dense["mean_best_val_loss"]
    comparisons = {
        "mx04_finetuning": {
            "baseline": "mx04_scratch", "candidate": "mx04_head_last",
            "mean_loss_improvement_descriptive_only": (
                scratch["mean_best_val_loss"] - head["mean_best_val_loss"]),
            "valid_for_inner_selection": False,
            "reason": "SUPERVISED_FOLD_LOCAL_PRETRAINING_EXPOSED_INNER_VALIDATION_LABELS",
            "paired_outer_comparison": "PREDECLARED_EXPLORATORY_HISTORICAL_OUTER_TEST",
        },
        "hybrid_dense_skip_effect": {
            "baseline": "hybrid_baseline", "candidate": "hybrid_dense",
            "mean_loss_improvement": advantage_dense,
            "folds_improved": wins_dense,
            "passes_practical_rule": advantage_dense >= margin and wins_dense >= minimum_folds,
        },
    }
    selected_b = "mx04_scratch"
    selected_h = ("hybrid_dense" if comparisons["hybrid_dense_skip_effect"]
                  ["passes_practical_rule"] else "hybrid_baseline")
    hv = rows[selected_h]
    wins_hybrid = sum(a > b for a, b in zip(
        scratch["fold_best_val_loss"], hv["fold_best_val_loss"], strict=True))
    hybrid_over_b = (scratch["mean_best_val_loss"] - hv["mean_best_val_loss"] >= margin
                     and wins_hybrid >= minimum_folds)
    selected = selected_h if hybrid_over_b else selected_b
    return comparisons, selected_b, selected_h, selected, hybrid_over_b


def active_gpu_compute_processes(output: str) -> list[dict[str, str]]:
    """Return GPU processes that can conflict with a fast-track writer.

    On Windows/WDDM, the Codex desktop host can appear in the compute-app
    query with an unavailable memory value even while the GPU is otherwise
    idle.  Ignore only that exact UI signature; fail closed for every other
    row, including malformed or unavailable-memory entries.
    """

    active: list[dict[str, str]] = []
    for row in csv.reader(line for line in output.splitlines() if line.strip()):
        if len(row) != 3:
            active.append({"pid": "UNKNOWN", "process_name": "UNPARSEABLE", "used_gpu_memory": ",".join(row)})
            continue
        pid, process_name, used = (value.strip() for value in row)
        normalized_used = used.strip("[] ").upper()
        if process_name.lower().endswith("chatgpt.exe") and normalized_used == "N/A":
            continue
        active.append({"pid": pid, "process_name": process_name, "used_gpu_memory": used})
    return active


def gpu_idle() -> None:
    query = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,process_name,used_gpu_memory",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=False,
    )
    active = active_gpu_compute_processes(query.stdout) if query.returncode == 0 else []
    if query.returncode != 0 or active:
        detail = json.dumps(active, ensure_ascii=False, sort_keys=True)
        raise RuntimeError(
            "GPU_BUSY_OR_NVIDIA_SMI_UNAVAILABLE_STOP_ORIGINAL_WRITER_SAFELY:"
            f"returncode={query.returncode}:active={detail}"
        )


def validate_hybrid_smoke(report: dict, variant: str, config: dict) -> None:
    if (report.get("status") != "PASS"
        or report.get("total_parameter_count", 10**9) > config["hybrid_parameter_ceiling"]):
        raise RuntimeError(f"HYBRID_SMOKE_NOT_PASS:{variant}")
    evidence = report.get("component_evidence") or {}
    flags = evidence.get("observed_variant_flags") or {}
    if (flags.get("dense_skip") is not (variant == "plus_dense_skip")
        or not evidence.get("all_checks_pass")):
        raise RuntimeError(f"HYBRID_VARIANTS_NOT_DISTINCT:{variant}")


def validate_training_profile(path: Path, *, variant: str, out: Path) -> None:
    report = read(path)
    metrics = out / "inner_tasks" / f"profile_{variant}" / "three_d_metrics.json"
    if (report.get("status") != "PASS" or report.get("variant") != variant
        or report.get("scope") != "one_fold_one_epoch_limit64_not_selection_evidence"
        or report.get("source_preregistration_sha256") !=
           sha(out / "reports/fasttrack_preregistration.json")
        or not metrics.is_file() or report.get("metrics_sha256") != sha(metrics)
        or max(float(report.get("allocated_gb", 99)),
               float(report.get("reserved_gb", 99))) >= 7.5):
        raise RuntimeError(f"HYBRID_PROFILE_EXISTING_NOT_PASS:{variant}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action",
        choices=("plan", "smoke", "profile", "mx4", "mx5", "run", "lock"),
    )
    parser.add_argument("--project-root", default=str(ROOT))
    parser.add_argument("--battery-root", default="results/final_results/method_x_model_battery_v3_depth_and_diversity_20260729")
    parser.add_argument("--output-root", default="results/final_results/method_x_fasttrack_v2_20260926_mx04_import")
    parser.add_argument("--config", default=FASTTRACK_CONFIGS["method_x_fasttrack_v2_20260926_mx04_import"])
    args = parser.parse_args()
    root, battery, out = paths(args)
    config_path = root / args.config
    preregistration_path = out / "reports/fasttrack_preregistration.json"
    if args.action != "plan" and not preregistration_path.is_file():
        raise RuntimeError("FASTTRACK_PLAN_PREREGISTRATION_REQUIRED_BEFORE_GPU_OR_LOCK")
    config, manifest, identity = confirm(
        root, battery, out, config_path,
        persist_preregistration=(args.action != "plan"),
    )
    if args.action == "smoke":
        for variant in config["hybrid_variants"]:
            smoke_path = out / "reports" / f"smoke_{variant}.json"
            if smoke_path.is_file():
                validate_hybrid_smoke(read(smoke_path), variant, config)
                continue
            gpu_idle()
            cmd = [sys.executable, str(root / "scripts/smoke_method_x_backbone.py"),
                   "--backbone", "amuse_x_hybrid3d", "--device", "cuda", "--resolution", "128",
                   "--hard-stop-vram-gb", str(config["hybrid_smoke_vram_ceiling_gb"]),
                   "--hybrid_variant", variant]
            result = subprocess.run(cmd, cwd=root, capture_output=True, text=True, check=False)
            if result.returncode:
                raise RuntimeError(f"HYBRID_SMOKE_FAILED:{variant}:{result.stdout[-600:]}:{result.stderr[-600:]}")
            report = json.loads(result.stdout)
            validate_hybrid_smoke(report, variant, config)
            write_once(smoke_path, report)
        print(json.dumps({"status": "PASS", "variants": config["hybrid_variants"]}))
        return 0
    if args.action == "plan":
        planned = []
        for task in TASKS:
            cmd, location = task_command(task, root, battery, out, manifest,
                out / "pretrained_provenance/resnet34_3d/pretrained_fold_manifest.json")
            planned.append((task, cmd, location))
        # Persist only after every command has passed construction and scope
        # validation. A failed plan must not strand an immutable preregistration.
        write_once(preregistration_path, identity)
        for task, cmd, location in planned:
            print(json.dumps({"task": task[0], "output": str(location),
                              "command": cmd,
                              "inner_only": "--hpo_selection_only" in cmd,
                              "folds": 5, "seed": 13, "epochs_field": 100,
                              "source_contract": identity["training_source_sha256"]}, ensure_ascii=False))
        return 0
    if args.action == "profile":
        for variant in config["hybrid_variants"]:
            smoke_path = out / "reports" / f"smoke_{variant}.json"
            if not smoke_path.is_file():
                raise RuntimeError(f"HYBRID_FORWARD_SMOKE_REQUIRED:{variant}")
            validate_hybrid_smoke(read(smoke_path), variant, config)
            report_path = out / "reports" / f"training_profile_{variant}.json"
            profile_task_id = "profile_" + variant
            profile_output = out / "inner_tasks" / profile_task_id
            if report_path.exists():
                validate_training_profile(report_path, variant=variant, out=out)
                prune_completed_task_staging(
                    root=root, out=out, task_id=profile_task_id,
                    model_root=profile_output, backbone="amuse_x_hybrid3d",
                    expected_folds=1,
                )
                continue
            gpu_idle()
            task = (profile_task_id, "hybrid", variant)
            cmd, output_dir = task_command(task, root, battery, out, manifest, None)
            cmd[cmd.index("--phase") + 1] = "dev"
            cmd[cmd.index("--folds") + 1] = "1"
            cmd[cmd.index("--epochs") + 1] = "1"
            cmd.extend(["--limit", "64", "--throughput_profile_only"])
            code = subprocess.run(cmd, cwd=root, check=False).returncode
            if code != 0:
                raise RuntimeError(f"TRAINING_PROFILE_FAILED:{variant}:EXIT_{code}")
            aggregate = read(output_dir / "three_d_metrics.json").get("aggregate") or {}
            if (aggregate.get("throughput_profile_only") is not True or
                aggregate.get("hpo_selection_only") is not True or
                aggregate.get("outer_test_access_count") != 0 or
                int(aggregate.get("completed_folds", -1)) != 1):
                raise RuntimeError(f"TRAINING_PROFILE_SCOPE_INVALID:{variant}")
            allocated = aggregate.get("vram_peak_allocated_gb")
            reserved = aggregate.get("vram_peak_reserved_gb")
            if allocated is None or reserved is None or max(float(allocated), float(reserved)) >= 7.5:
                raise RuntimeError(f"TRAINING_PROFILE_VRAM_TOO_HIGH:{variant}")
            write_once(report_path, {"status": "PASS", "variant": variant,
                "scope": "one_fold_one_epoch_limit64_not_selection_evidence",
                "allocated_gb": float(allocated), "reserved_gb": float(reserved),
                "metrics_sha256": sha(output_dir / "three_d_metrics.json"),
                "source_preregistration_sha256": sha(out / "reports/fasttrack_preregistration.json")})
            prune_completed_task_staging(
                root=root, out=out, task_id=profile_task_id,
                model_root=output_dir, backbone="amuse_x_hybrid3d",
                expected_folds=1,
            )
        print(json.dumps({"status": "PASS", "training_profiles": config["hybrid_variants"]}))
        return 0
    if args.action in {"mx4", "mx5", "run"}:
        selected_tasks = (
            tuple(task for task in TASKS if task[1] == "mx04")
            if args.action == "mx4"
            else tuple(task for task in TASKS if task[1] == "hybrid")
            if args.action == "mx5"
            else TASKS
        )
        state_path = out / "reports/fasttrack_task_state.json"
        state = read(state_path) if state_path.exists() else {
            "schema_version": 1,
            "tasks": {},
            "identity_sha256": sha(out / "reports/fasttrack_preregistration.json"),
        }
        if state.get("identity_sha256") != sha(out / "reports/fasttrack_preregistration.json"):
            raise RuntimeError("TASK_STATE_PROTOCOL_DRIFT")
        if args.action == "mx5":
            for prerequisite in (task for task in TASKS if task[1] == "mx04"):
                task_id = prerequisite[0]
                prior = state.get("tasks", {}).get(task_id, {})
                metrics = out / "inner_tasks" / task_id / "three_d_metrics.json"
                if prior.get("status") != "COMPLETED" or not metrics.is_file():
                    raise RuntimeError(f"MX04_MUST_COMPLETE_BEFORE_MX05:{task_id}")
                inner(metrics)
                if prior.get("metrics_sha256") != sha(metrics):
                    raise RuntimeError(f"MX04_METRICS_CHANGED_BEFORE_MX05:{task_id}")
        if args.action in {"mx5", "run"}:
            for variant in config["hybrid_variants"]:
                path = out / "reports" / f"smoke_{variant}.json"
                if not path.is_file():
                    raise RuntimeError("ALL_TWO_HYBRID_SMOKES_REQUIRED")
                validate_hybrid_smoke(read(path), variant, config)
                profile = out / "reports" / f"training_profile_{variant}.json"
                if not profile.is_file():
                    raise RuntimeError("ALL_TWO_HYBRID_TRAINING_PROFILES_REQUIRED")
                validate_training_profile(profile, variant=variant, out=out)
        for task in selected_tasks:
            task_id = task[0]
            fold_manifest = build_fold_manifest(root, battery, out) if task_id == "mx04_head_last" else None
            cmd, model_root = task_command(task, root, battery, out, manifest, fold_manifest)
            fingerprint = hashlib.sha256(json.dumps(cmd, ensure_ascii=False).encode()).hexdigest()
            prior = state["tasks"].get(task_id, {})
            if prior and prior.get("command_sha256") != fingerprint:
                raise RuntimeError(f"COMMAND_DRIFT:{task_id}")
            metrics = model_root / "three_d_metrics.json"
            if prior.get("status") == "COMPLETED":
                inner(metrics)
                if prior.get("metrics_sha256") != sha(metrics):
                    raise RuntimeError(f"METRICS_CHANGED_AFTER_COMPLETION:{task_id}")
                prune_completed_task_staging(
                    root=root, out=out, task_id=task_id, model_root=model_root,
                    backbone=("resnet34_3d" if task[1] == "mx04" else "amuse_x_hybrid3d"),
                    expected_folds=5,
                )
                continue
            gpu_idle()
            state["tasks"][task_id] = {"status": "RUNNING", "command_sha256": fingerprint,
                                       "started_at_utc": utc(), "attempts": prior.get("attempts", 0) + 1}
            update_state(state_path, state)
            print(json.dumps({"task": task_id, "state": "RUNNING", "output": str(model_root)}), flush=True)
            code = subprocess.run(cmd, cwd=root, check=False).returncode
            if code == 75:
                state["tasks"][task_id]["status"] = "PAUSED_RESUMABLE"
                update_state(state_path, state)
                raise RuntimeError(f"TASK_PAUSED_EXACT_RESUME:{task_id}")
            if code != 0:
                state["tasks"][task_id]["status"] = "FAILED"
                state["tasks"][task_id]["returncode"] = code
                update_state(state_path, state)
                raise RuntimeError(f"TASK_FAILED:{task_id}:EXIT_{code}")
            inner(metrics)
            state["tasks"][task_id].update({"status": "COMPLETED",
                                            "metrics_sha256": sha(metrics),
                                            "finished_at_utc": utc()})
            update_state(state_path, state)
            prune_completed_task_staging(
                root=root, out=out, task_id=task_id, model_root=model_root,
                backbone=("resnet34_3d" if task[1] == "mx04" else "amuse_x_hybrid3d"),
                expected_folds=5,
            )
        status = {
            "mx4": "MX04_INNER_TASKS_COMPLETED_AWAITING_COMMON_MX04_MX05_LOCK",
            "mx5": "MX05_INNER_TASKS_COMPLETED_READY_TO_LOCK",
            "run": "FOUR_INNER_TASKS_COMPLETED",
        }[args.action]
        print(json.dumps({"status": status, "outer_test_used_for_selection": False,
                          "completed_tasks": [task[0] for task in selected_tasks]}))
        return 0
    if args.action != "lock":
        raise RuntimeError(f"UNHANDLED_FASTTRACK_ACTION:{args.action}")
    state = read(out / "reports/fasttrack_task_state.json")
    rows = {}
    for task in TASKS:
        tid = task[0]
        result = state.get("tasks", {}).get(tid, {})
        metric_file = out / "inner_tasks" / tid / "three_d_metrics.json"
        if result.get("status") != "COMPLETED" or result.get("metrics_sha256") != sha(metric_file):
            raise RuntimeError(f"TASK_NOT_FINALIZED:{tid}")
        prune_completed_task_staging(
            root=root, out=out, task_id=tid, model_root=out / "inner_tasks" / tid,
            backbone=("resnet34_3d" if task[1] == "mx04" else "amuse_x_hybrid3d"),
            expected_folds=5,
        )
        mean, folds = inner(metric_file)
        checkpoints = [out / "inner_tasks" / tid / f"method_x_3d_fold_{i}.pt" for i in range(5)]
        rows[tid] = {"mean_best_val_loss": mean, "fold_best_val_loss": folds,
                     "metrics_sha256": sha(metric_file),
                     "checkpoint_sha256": [sha(p) for p in checkpoints]}
        contract_file = out / "inner_tasks" / tid / "run_contract.json"
        contract = read(contract_file)
        digest = str(contract.get("run_contract_digest") or "")
        if len(digest) != 64 or contract.get("run_contract", {}).get("digest") != digest:
            raise RuntimeError(f"RUN_CONTRACT_DIGEST_MISSING:{tid}")
        metric_rows = read(metric_file)["folds"]
        membership = fold_membership_digests(out / "inner_tasks" / tid, task_id=tid)
        cross_check_metrics_membership(out / "inner_tasks" / tid, task_id=tid,
                                      membership=membership)
        rows[tid].update({"run_contract_sha256": sha(contract_file),
                          "run_contract_digest": digest,
                          "fold_membership_digest": membership,
                          "staging_prune_plan_sha256": sha(
                              out / "reports" / f"staging_prune_{tid}_plan.json"),
                          "staging_prune_receipt_sha256": sha(
                              out / "reports" / f"staging_prune_{tid}.json")})
    margin = config["practical_improvement_margin_loss"]
    min_folds = config["minimum_folds_improved"]
    comparisons, selected_b, selected_h, selected, hybrid_over_b = (
        choose_without_contaminated_finetune_validation(
            rows, margin=margin, minimum_folds=min_folds))
    lock = {"schema_version": 1, "status": "LOCKED_INNER_ONLY_EXPLORATORY",
            "selection_scope": "five_fold_inner_validation_seed_13",
            "outer_test_access_count": 0, "outer_test_used_for_selection": False,
            "original_mx04_mx05_completed": False, "final_task": selected,
            "mx04_candidate": selected_b, "hybrid_candidate": selected_h,
            "exploratory_outer_tasks": ["mx04_scratch", "mx04_head_last"],
            "hybrid_pair": ["hybrid_baseline", "hybrid_dense"],
            "fine_tuning_inner_validation_valid_for_selection": False,
            "hybrid_over_mx04_practical_rule": hybrid_over_b,
            "practical_margin_loss": margin, "minimum_folds_improved": min_folds,
            "comparisons": comparisons, "tasks": rows,
            "preregistration_sha256": sha(out / "reports/fasttrack_preregistration.json"),
            "claim_cap": "POST_HOC_INTERNAL_TECHNICAL_EVIDENCE",
            "external_validation": False}
    write_once(out / "reports/fasttrack_inner_lock.json", lock)
    print(json.dumps({"status": lock["status"], "selected": selected,
                      "outer_test_access_count": 0}))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as exc:
        print(json.dumps({"status": "BLOCKED", "reason": str(exc)}, ensure_ascii=False), file=sys.stderr)
        raise SystemExit(3)
