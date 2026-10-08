#!/usr/bin/env python3
"""Resumable 3D CNN hyperparameter search on real + supplementary MRI caches."""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.merge_real_synthetic import merge_caches
from src.utils.io_helpers import atomic_write_json, atomic_write_text
from src.utils.reproducibility import canonical_digest
from src.utils.run_manifest import write_run_manifest

APPROVED_PHENOTYPE_FEATURES = {"age", "sex", "education_years", "apoe4_carrier", "mmse_base"}
FORBIDDEN_METHOD_X_TOKENS = ("pet", "suvr", "centiloid", "amyloid_pet")
FORBIDDEN_CONDITIONING_TOKENS = ("cohort", "site", "scanner")
FORBIDDEN_CONDITIONING_FIELDS = ("site_embedding", "scanner_adversary", "cohort_adapter")
HPO_CRITIC_COHORT_AUC_CEILING = 0.65


CANDIDATE_DEFAULTS: dict[str, Any] = {
    "attention": "none",
    "alignment_head": "linear",
    "tab_encoder": "linear",
    "q2_channel_gate": "none",
    "fusion_num_heads": 2,
    "n_experts": 4,
    "moe_load_balance_lambda": 0.0,
    "task_loss": "bce",
    "focal_gamma": 2.0,
    "focal_alpha": -1.0,
    "auc_margin": 1.0,
    "auc_loss_weight": 1.0,
    "ema_decay": 0.0,
    "swa": False,
    "swa_start_fraction": 0.8,
    "restore_best_at_end": True,
    "supervised_contrastive_alignment": False,
    "phenotypic_contrastive_alignment": False,
    "contrastive_temperature": 0.07,
    "synib_lambda": 0.0,
    "scheduler": "cosine",
    "warmup_epochs": 5,
    "pos_weight": "auto",
    "aug_prob_scale": 0.2,
    "layerwise_lr_decay": 1.0,
    "discriminative_lr": False,
}


def candidate(**overrides: Any) -> dict[str, Any]:
    row = {**CANDIDATE_DEFAULTS, **overrides}
    if row["task_loss"] != "bce" and float(row["synib_lambda"]) > 0.0:
        raise ValueError("Non-BCE task losses are not combined with SynIB in the default grid")
    return row


DEFAULT_CANDIDATES: list[dict[str, Any]] = [
    candidate(
        name="resnet10_cross_attention_bce_lr1e4_drop10",
        backbone="resnet10_3d",
        fusion="cross_attention",
        lr=1e-4,
        dropout=0.10,
        latent_dim=128,
        tab_hidden_dim=32,
        bottleneck_dim=32,
        latent_alignment_lambda=0.0,
        label_smoothing=0.0,
    ),
    candidate(
        name="resnet18_cross_attention_bce_lr1e4_drop10",
        backbone="resnet18_3d",
        fusion="cross_attention",
        lr=1e-4,
        dropout=0.10,
        latent_dim=128,
        tab_hidden_dim=32,
        bottleneck_dim=32,
        latent_alignment_lambda=0.0,
        label_smoothing=0.0,
    ),
    candidate(
        name="resnet18_cross_attention_linearstack_lr5e5_drop15",
        backbone="resnet18_3d",
        fusion="cross_attention",
        lr=5e-5,
        dropout=0.15,
        latent_dim=128,
        tab_hidden_dim=48,
        bottleneck_dim=32,
        latent_alignment_lambda=0.02,
        label_smoothing=0.03,
        alignment_head="linear_stack",
        tab_encoder="bn_gelu",
        ema_decay=0.995,
    ),
    candidate(
        name="densenet121_cross_attention_focal_lr5e5_drop20",
        backbone="densenet121_3d",
        fusion="cross_attention",
        lr=5e-5,
        dropout=0.20,
        latent_dim=128,
        tab_hidden_dim=48,
        bottleneck_dim=32,
        latent_alignment_lambda=0.0,
        label_smoothing=0.0,
        task_loss="focal",
        focal_gamma=2.0,
        ema_decay=0.995,
    ),
    candidate(
        name="efficientnet3d_b0_moe_bce_lb_lr5e5_drop20",
        backbone="efficientnet3d_b0",
        fusion="moe",
        lr=5e-5,
        dropout=0.20,
        latent_dim=128,
        tab_hidden_dim=48,
        bottleneck_dim=32,
        latent_alignment_lambda=0.0,
        label_smoothing=0.05,
        n_experts=4,
        moe_load_balance_lambda=0.01,
    ),
    candidate(
        name="efficientnet3d_b0_moe_auc_lb_lr5e5_drop20",
        backbone="efficientnet3d_b0",
        fusion="moe",
        lr=5e-5,
        dropout=0.20,
        latent_dim=128,
        tab_hidden_dim=48,
        bottleneck_dim=32,
        latent_alignment_lambda=0.0,
        label_smoothing=0.0,
        n_experts=4,
        moe_load_balance_lambda=0.01,
        task_loss="bce_auc",
        auc_loss_weight=0.2,
    ),
    candidate(
        name="resnet18_healnet_auc_ema_lr5e5_drop15",
        backbone="resnet18_3d",
        fusion="healnet",
        lr=5e-5,
        dropout=0.15,
        latent_dim=128,
        tab_hidden_dim=48,
        bottleneck_dim=32,
        latent_alignment_lambda=0.01,
        label_smoothing=0.0,
        task_loss="bce_auc",
        auc_loss_weight=0.2,
        ema_decay=0.995,
    ),
    candidate(
        name="mednext_s_healnet_focal_q2phenotype_lr5e5_drop20",
        backbone="mednext_3d_s",
        fusion="healnet",
        lr=5e-5,
        dropout=0.20,
        latent_dim=128,
        tab_hidden_dim=48,
        bottleneck_dim=32,
        latent_alignment_lambda=0.0,
        label_smoothing=0.0,
        task_loss="focal",
        q2_channel_gate="phenotype",
        ema_decay=0.995,
    ),
]


DEFAULT_CANDIDATE_CONFIG = "configs/experiments/method_x_3d_candidate_grid.yaml"


def read_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def read_structured_config(path: Path) -> dict[str, Any] | list[Any]:
    if path.suffix.lower() == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    try:
        import yaml
    except Exception as exc:  # pragma: no cover - dependency is pinned in requirements
        raise RuntimeError("PyYAML is required for --candidate_config") from exc
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def load_candidate_config(path: Path | str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load B5/NiftyNet-style declarative candidate definitions."""
    config_path = PROJECT_ROOT / path
    if not config_path.exists():
        raise FileNotFoundError(f"Missing candidate config: {config_path}")
    obj = read_structured_config(config_path)
    if isinstance(obj, list):
        entries = obj
        defaults: dict[str, Any] = {}
        policy: dict[str, Any] = {}
    elif isinstance(obj, dict):
        entries = obj.get("candidates") or []
        defaults = obj.get("candidate_defaults") or {}
        policy = obj.get("policy") or {}
    else:
        raise ValueError("candidate config must be a mapping with candidates or a list")
    if not isinstance(entries, list) or not entries:
        raise ValueError("candidate config must define a non-empty candidates list")
    candidates: list[dict[str, Any]] = []
    for i, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ValueError(f"candidate entry {i} is not a mapping")
        merged = {**defaults, **entry}
        if not merged.get("name"):
            raise ValueError(f"candidate entry {i} is missing name")
        candidates.append(candidate(**merged))
    names = [str(row["name"]) for row in candidates]
    if len(names) != len(set(names)):
        raise ValueError("candidate config contains duplicate names")
    return candidates, {
        "status": "LOADED",
        "path": str(config_path.relative_to(PROJECT_ROOT)),
        "schema_version": obj.get("schema_version") if isinstance(obj, dict) else None,
        "policy": policy,
        "excluded_candidates": (
            obj.get("excluded_candidates") or [] if isinstance(obj, dict) else []
        ),
        "n_candidates": len(candidates),
        "candidate_names": names,
        "b5_niftynet_config_driven": True,
    }


def ensure_combined_cache(args: argparse.Namespace) -> dict[str, Any]:
    real = PROJECT_ROOT / args.real_cache
    out = PROJECT_ROOT / args.combined_cache
    if not real.exists():
        raise FileNotFoundError(f"Missing real MRI cache: {real}")

    extras: list[tuple[Path, str]] = []
    for path_text, source in [
        (args.ddpm_cache, "synthetic_ddpm"),
        (args.simulated_cache, "simulated_augment"),
        (args.legacy_synthetic_cache, "simulated_augment"),
    ]:
        path = PROJECT_ROOT / path_text
        if path.exists():
            extras.append((path, source))

    if not extras:
        return {"status": "REAL_ONLY_NO_SUPPLEMENTARY_FOUND", "combined_cache": str(real)}

    newest_input = max([real.stat().st_mtime, *(p.stat().st_mtime for p, _ in extras)])
    if out.exists() and out.stat().st_mtime >= newest_input and not args.rebuild_combined:
        return {"status": "REUSED", "combined_cache": str(out), "extras": [str(p) for p, _ in extras]}

    stats = merge_caches(real, *extras, out_path=out)
    return {"status": "BUILT", "combined_cache": str(out), "extras": [str(p) for p, _ in extras], "stats": stats}


def trial_metrics(path: Path) -> dict[str, Any] | None:
    obj = read_json(path / "three_d_metrics.json")
    if not obj:
        return None
    aggregate = obj.get("aggregate", {})
    if aggregate.get("status") != "COMPLETED":
        return aggregate
    return aggregate


def validate_completed_selection_trial(trial_dir: Path, *, folds: int) -> dict[str, Any]:
    payload = read_json(trial_dir / "three_d_metrics.json")
    aggregate = payload.get("aggregate") if isinstance(payload.get("aggregate"), dict) else {}
    policy = aggregate.get("outer_test_evaluation_policy") or {}
    if (
        aggregate.get("status") != "COMPLETED"
        or int(aggregate.get("folds", 0)) != int(folds)
        or aggregate.get("hpo_selection_only") is not True
        or policy.get("during_training_epochs") is not False
        or int(policy.get("max_evaluations_per_fold", -1)) != 0
        or policy.get("selection_only_no_outer_test") is not True
    ):
        raise RuntimeError("HPO_TRIAL_SELECTION_ONLY_CONTRACT_INVALID")
    rows = payload.get("folds")
    if (
        not isinstance(rows, list)
        or len(rows) != int(folds)
        or {int(row.get("fold", -1)) for row in rows if isinstance(row, dict)} != set(range(int(folds)))
    ):
        raise RuntimeError("HPO_TRIAL_FOLDS_INCOMPLETE")
    best_losses: list[float] = []
    for fold in range(int(folds)):
        row = next(row for row in rows if int(row.get("fold", -1)) == fold)
        state = read_json(trial_dir / f"method_x_3d_fold_{fold}_state.json")
        try:
            loss = float(row["best_val_loss"])
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError("HPO_TRIAL_VALIDATION_LOSS_INVALID") from exc
        if (
            not math.isfinite(loss)
            or int(row.get("outer_test_evaluation_count", -1)) != 0
            or row.get("hpo_selection_only") is not True
            or state.get("stage") != "FINALIZED"
            or int(state.get("fold", -1)) != fold
            or not (trial_dir / f"method_x_3d_fold_{fold}.pt").is_file()
            or (trial_dir / f"method_x_3d_fold_{fold}_outer_eval.json").exists()
        ):
            raise RuntimeError("HPO_TRIAL_FOLD_LOCK_INVALID")
        best_losses.append(loss)
    declared = aggregate.get("mean_best_val_loss")
    actual = sum(best_losses) / len(best_losses)
    if declared is None or not math.isclose(float(declared), actual, rel_tol=1e-12, abs_tol=1e-12):
        raise RuntimeError("HPO_TRIAL_MEAN_VALIDATION_LOSS_MISMATCH")
    return aggregate


def _flatten_text(value: Any) -> str:
    if isinstance(value, dict):
        return " ".join(f"{_flatten_text(k)} {_flatten_text(v)}" for k, v in value.items())
    if isinstance(value, (list, tuple, set)):
        return " ".join(_flatten_text(item) for item in value)
    return str(value).lower()


def _metric_from_nested(row: dict[str, Any], *paths: tuple[str, ...]) -> float | None:
    for path in paths:
        value: Any = row
        for key in path:
            if not isinstance(value, dict) or key not in value:
                value = None
                break
            value = value[key]
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def critique_candidate(candidate: dict[str, Any], *, grad_accum: int = 8) -> dict[str, Any]:
    """A7 AMUSE-X HPO critic: reject candidates outside the Method X contract before compute is spent."""
    violations: list[str] = []
    warnings: list[str] = []

    resolution = int(candidate.get("resolution", 128))
    if resolution != 128:
        violations.append("A7_REJECT_NON_128_RESOLUTION")

    input_mode = str(candidate.get("input_mode", "mri_plus_tabular"))
    if input_mode not in {"mri_plus_tabular", "mri_only"}:
        violations.append("A7_REJECT_METHOD_X_INPUT_MODE")

    q2_gate = str(candidate.get("q2_channel_gate", "none"))
    if q2_gate not in {"none", "latent_logistic", "phenotype"}:
        violations.append("A7_REJECT_UNKNOWN_Q2_CHANNEL_GATE")
    if q2_gate == "phenotype":
        features = candidate.get("q2_conditioning_features") or APPROVED_PHENOTYPE_FEATURES
        feature_set = {str(item).lower() for item in features}
        if not feature_set <= APPROVED_PHENOTYPE_FEATURES:
            violations.append("A7_REJECT_UNAPPROVED_Q2_CONDITIONING_FEATURES")

    for key, value in candidate.items():
        text = f"{str(key).lower()} {_flatten_text(value)}"
        if any(token in text for token in FORBIDDEN_METHOD_X_TOKENS):
            violations.append(f"A7_REJECT_PET_LIKE_FIELD:{key}")
        if key in {"conditioning_features", "q2_conditioning_features", "hpo_features"} and any(
            token in text for token in FORBIDDEN_CONDITIONING_TOKENS
        ):
            violations.append(f"A7_REJECT_COHORT_SITE_SCANNER_CONDITIONING:{key}")

    for field in FORBIDDEN_CONDITIONING_FIELDS:
        if candidate.get(field) not in {None, False, "", "none"}:
            violations.append(f"A7_REJECT_COHORT_SITE_SCANNER_FIELD:{field}")

    if any(key in candidate for key in ("teacher_checkpoint", "teacher_checkpoint_dir", "distillation_teacher")):
        violations.append("A7_REJECT_C1_DISTILLATION_WITHOUT_GOVERNANCE")

    if str(candidate.get("task_loss", "bce")) != "bce" and float(candidate.get("synib_lambda", 0.0)) > 0.0:
        violations.append("A7_REJECT_NON_BCE_SYNIB_COMBINATION")

    if int(grad_accum) < 8:
        warnings.append("A7_WARN_GRAD_ACCUM_BELOW_16GB_POLICY_DEFAULT")

    return {
        "candidate": candidate.get("name"),
        "status": "ACCEPTED" if not violations else "REJECTED",
        "violations": sorted(set(violations)),
        "warnings": sorted(set(warnings)),
        "policy": {
            "method_x_pet_free": True,
            "official_resolution": 128,
            "cohort_probe_auc_ceiling": HPO_CRITIC_COHORT_AUC_CEILING,
            "cohort_probe_auc_policy": "diagnostic_warning_only",
            "approved_phenotype_features": sorted(APPROVED_PHENOTYPE_FEATURES),
            "c1_distillation_requires_governance": True,
        },
    }


def apply_hpo_critic(
    candidates: list[dict[str, Any]],
    *,
    grad_accum: int = 8,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    decisions = [critique_candidate(row, grad_accum=grad_accum) for row in candidates]
    accepted_names = {row["candidate"] for row in decisions if row["status"] == "ACCEPTED"}
    accepted = [row for row in candidates if row.get("name") in accepted_names]
    return accepted, {
        "enabled": True,
        "critic": "AMUSE_X_A7_HPO_CRITIC",
        "n_input_candidates": len(candidates),
        "n_accepted_candidates": len(accepted),
        "n_rejected_candidates": len(candidates) - len(accepted),
        "decisions": decisions,
    }


def critique_trial_result(row: dict[str, Any]) -> dict[str, Any]:
    warnings: list[str] = []
    cohort_auc = _metric_from_nested(
        row,
        ("cohort_probe_auc",),
        ("cohort_probe", "auc"),
        ("amuse_x_min", "cohort_probe_auc"),
    )
    if cohort_auc is not None and cohort_auc > HPO_CRITIC_COHORT_AUC_CEILING:
        warnings.append("A7_WARN_RESULT_COHORT_PROBE_AUC_ABOVE_DIAGNOSTIC_THRESHOLD")
    evidential_kl = _metric_from_nested(row, ("evidential_kl",), ("amuse_x_min", "evidential_kl"))
    if evidential_kl is not None and evidential_kl <= 0.0:
        warnings.append("A7_WARN_RESULT_EVIDENTIAL_KL_COLLAPSE")
    return {
        "status": "ACCEPTED",
        "violations": [],
        "warnings": warnings,
        "cohort_probe_auc": cohort_auc,
        "cohort_probe_auc_ceiling": HPO_CRITIC_COHORT_AUC_CEILING,
        "evidential_kl": evidential_kl,
        "winner_ranking_eligible": not (
            evidential_kl is not None and evidential_kl <= 0.0
        ),
    }


def validate_reusable_trial_contract(
    trial_dir: Path,
    args: argparse.Namespace,
    candidate: dict[str, Any],
) -> None:
    """Fail closed before reusing a completed trial under a changed contract."""

    if not bool(args.require_exact_resume_contract):
        return
    contract_path = trial_dir / "run_contract.json"
    try:
        payload = json.loads(contract_path.read_text(encoding="utf-8"))
        contract = payload["run_contract"]
        parameters = contract["parameters"]
        loader = contract["loader"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise RuntimeError(
            f"STALE_TRIAL_RUN_CONTRACT:{candidate['name']}:MISSING_OR_INVALID"
        ) from exc
    body = {key: value for key, value in contract.items() if key != "digest"}
    digest = canonical_digest(body)
    # A completed HPO trial is reusable only under the exact training source
    # that produced it.  Candidate equality alone is insufficient: a source
    # change can alter model, loader, loss, or resume semantics while leaving
    # the declarative candidate row unchanged.
    from scripts.train_3d_mri import method_x_training_source_contract

    current_source_digest, current_runtime_versions = method_x_training_source_contract()
    common_ok = (
        payload.get("run_contract_digest") == digest
        and contract.get("digest") == digest
        and contract.get("cache_sha256") == args.cache_sha256
        and int(contract.get("folds", 0)) == int(args.folds)
        and int(contract.get("epochs", 0)) == int(args.epochs)
        and int(contract.get("seed", -1)) == int(args.seed)
        and int(loader.get("batch_size", -1)) == int(args.batch_size)
        and parameters.get("hpo_selection_only") is True
        and parameters.get("source_contract_digest") == current_source_digest
        and parameters.get("runtime_versions") == current_runtime_versions
    )
    candidate_ok = all(
        parameters.get(key) == value
        for key, value in candidate.items()
        if key != "name"
    )
    if not common_ok or not candidate_ok:
        raise RuntimeError(f"STALE_TRIAL_RUN_CONTRACT:{candidate['name']}:MISMATCH")


def candidate_command(args: argparse.Namespace, candidate: dict[str, Any], trial_dir: Path) -> list[str]:
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "train_3d_mri.py"),
        "--cache",
        args.combined_cache,
        "--resolution",
        "128",
        "--phase",
        "final",
        "--folds",
        str(args.folds),
        "--epochs",
        str(args.epochs),
        "--batch_size",
        str(args.batch_size),
        "--backbone",
        candidate["backbone"],
        "--fusion",
        candidate["fusion"],
        "--lr",
        str(candidate["lr"]),
        "--dropout",
        str(candidate["dropout"]),
        "--latent_dim",
        str(candidate["latent_dim"]),
        "--tab_hidden_dim",
        str(candidate["tab_hidden_dim"]),
        "--alignment_head",
        str(candidate["alignment_head"]),
        "--tab_encoder",
        str(candidate["tab_encoder"]),
        "--attention",
        str(candidate["attention"]),
        "--bottleneck_dim",
        str(candidate["bottleneck_dim"]),
        "--n_experts",
        str(candidate["n_experts"]),
        "--fusion_num_heads",
        str(candidate["fusion_num_heads"]),
        "--moe_load_balance_lambda",
        str(candidate["moe_load_balance_lambda"]),
        "--latent_alignment_lambda",
        str(candidate["latent_alignment_lambda"]),
        "--label_smoothing",
        str(candidate["label_smoothing"]),
        "--task_loss",
        str(candidate["task_loss"]),
        "--focal_gamma",
        str(candidate["focal_gamma"]),
        "--focal_alpha",
        str(candidate["focal_alpha"]),
        "--auc_margin",
        str(candidate["auc_margin"]),
        "--auc_loss_weight",
        str(candidate["auc_loss_weight"]),
        "--ema_decay",
        str(candidate["ema_decay"]),
        "--swa_start_fraction",
        str(candidate["swa_start_fraction"]),
        "--q2_channel_gate",
        str(candidate["q2_channel_gate"]),
        "--scheduler",
        str(candidate["scheduler"]),
        "--warmup_epochs",
        str(candidate["warmup_epochs"]),
        "--pos_weight",
        str(candidate["pos_weight"]),
        "--contrastive_temperature",
        str(candidate["contrastive_temperature"]),
        "--synib_lambda",
        str(candidate["synib_lambda"]),
        "--aug_prob_scale",
        str(candidate["aug_prob_scale"]),
        "--layerwise_lr_decay",
        str(candidate["layerwise_lr_decay"]),
        "--amp",
        "--checkpoint",
        "--grad_accum",
        str(args.grad_accum),
        "--monai_augment",
        "--device",
        "cuda",
        "--require_gpu",
        "--out_dir",
        str(trial_dir.relative_to(PROJECT_ROOT)),
    ]
    if candidate.get("restore_best_at_end"):
        cmd.append("--restore_best_at_end")
    if candidate.get("swa"):
        cmd.append("--swa")
    if candidate.get("supervised_contrastive_alignment"):
        cmd.append("--supervised_contrastive_alignment")
    if candidate.get("phenotypic_contrastive_alignment"):
        cmd.append("--phenotypic_contrastive_alignment")
    if candidate.get("discriminative_lr"):
        cmd.append("--discriminative_lr")
    if args.include_global_supplementary_train:
        cmd.append("--include_global_supplementary_train")
    if args.pretrained_checkpoint:
        cmd.extend(["--pretrained_checkpoint", args.pretrained_checkpoint])
    if args.max_runtime_minutes > 0:
        cmd.extend(["--max_runtime_minutes", str(args.max_runtime_minutes)])
    if args.limit > 0:
        cmd.extend(["--limit", str(args.limit)])
    return cmd


def selection_key(row: dict[str, Any]) -> tuple[float, float]:
    mean_best_val_loss = row.get("mean_best_val_loss")
    if mean_best_val_loss is None:
        return 999.0, 999.0
    mean_final_val_loss = row.get("mean_final_val_loss")
    return float(mean_best_val_loss), float(mean_final_val_loss if mean_final_val_loss is not None else mean_best_val_loss)


def write_summary(out_dir: Path, summary: dict[str, Any]) -> None:
    atomic_write_json(out_dir / "three_d_hparam_search_summary.json", summary)
    lines = [
        "# 3D CNN Hyperparameter Search\n\n",
        f"- status: `{summary['status']}`\n",
        f"- combined_cache: `{summary['combined_cache']}`\n",
        f"- epochs_per_trial: `{summary['epochs']}`\n",
        f"- folds: `{summary['folds']}`\n",
        f"- include_global_supplementary_train: `{summary['include_global_supplementary_train']}`\n",
    ]
    best = summary.get("best")
    if best:
        lines.extend(
            [
                f"- best_trial: `{best['name']}`\n",
                f"- best_backbone: `{best['params']['backbone']}`\n",
                f"- best_fusion: `{best['params']['fusion']}`\n",
                f"- best_lr: `{best['params']['lr']}`\n",
                f"- best_dropout: `{best['params']['dropout']}`\n",
                f"- best_roc_auc: `{best.get('roc_auc')}`\n",
                f"- best_pr_auc: `{best.get('pr_auc')}`\n",
                f"- best_brier: `{best.get('brier')}`\n",
            ]
        )
    best_by_backbone = summary.get("best_by_backbone") or {}
    if best_by_backbone:
        lines.append("\n## Best by backbone\n\n")
        for backbone, row in sorted(best_by_backbone.items()):
            lines.append(
                f"- {backbone}: `{row.get('name')}` "
                f"(mean_best_val_loss `{row.get('mean_best_val_loss')}`)\n"
            )
    atomic_write_text(out_dir / "three_d_hparam_search_summary.md", "".join(lines))
    write_run_manifest(
        "results/logs/three_d_hparam_search_manifest.json",
        stage="three_d_hparam_search",
        data_root=None,
        extra=summary,
    )


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Search full 3D CNN settings on real + synthetic MRI data.")
    ap.add_argument("--real_cache", default="data/cached/t1_volume_cache_128.h5")
    ap.add_argument("--combined_cache", default="data/cached/t1_volume_cache_128.h5")
    ap.add_argument("--ddpm_cache", default="data/cached/t1_synthetic_ddpm_128.h5")
    ap.add_argument("--simulated_cache", default="data/cached/t1_simulated_aug_128.h5")
    ap.add_argument("--legacy_synthetic_cache", default="data/cached/t1_synthetic_128.h5")
    ap.add_argument("--out_dir", default="results/final_results/three_d_hparam_search")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max_runtime_minutes", type=float, default=0.0)
    ap.add_argument("--pretrained_checkpoint", default="")
    ap.add_argument("--candidate", action="append", default=[], help="Run only named candidate(s).")
    ap.add_argument(
        "--candidate_config",
        default=DEFAULT_CANDIDATE_CONFIG,
        help="B5/NiftyNet-style YAML/JSON candidate grid. Pass an empty string to use code defaults.",
    )
    ap.add_argument("--include_global_supplementary_train", action="store_true", default=False)
    ap.add_argument("--fold_local_only", action="store_false", dest="include_global_supplementary_train")
    ap.add_argument("--rebuild_combined", action="store_true")
    ap.add_argument("--disable_hpo_critic", action="store_true", help="Disable the default AMUSE-X A7 candidate critic.")
    ap.add_argument(
        "--critic_min_valid_candidates",
        type=int,
        default=3,
        help="Minimum accepted default candidates before the A7 critic blocks the default search.",
    )
    return ap


def main() -> int:
    ap = build_parser()
    args = ap.parse_args()

    if args.epochs < 40:
        raise SystemExit("3D final hyperparameter search requires --epochs >= 40.")

    out_dir = PROJECT_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    combined = ensure_combined_cache(args)

    candidate_config_report = {
        "status": "CODE_DEFAULTS",
        "path": None,
        "b5_niftynet_config_driven": False,
        "n_candidates": len(DEFAULT_CANDIDATES),
    }
    selected = DEFAULT_CANDIDATES
    if str(args.candidate_config).strip():
        selected, candidate_config_report = load_candidate_config(args.candidate_config)
    if args.candidate:
        wanted = set(args.candidate)
        selected = [c for c in selected if c["name"] in wanted]
        missing = wanted - {c["name"] for c in selected}
        if missing:
            raise SystemExit(f"Unknown candidate(s): {sorted(missing)}")

    requested_candidate_count = len(selected)
    critic_report = {
        "enabled": False,
        "critic": "AMUSE_X_A7_HPO_CRITIC",
        "reason": "disabled_by_cli",
    }
    if not args.disable_hpo_critic:
        selected, critic_report = apply_hpo_critic(selected, grad_accum=args.grad_accum)
        if not selected:
            raise SystemExit("A7 HPO critic rejected all selected candidates; no training was started.")
        if not args.candidate and len(selected) < int(args.critic_min_valid_candidates):
            raise SystemExit(
                "A7 HPO critic left fewer than critic_min_valid_candidates; "
                "fall back to the conservative Optuna baseline or revise the candidate grid."
            )

    results: list[dict[str, Any]] = []
    rejected_completed_trials: list[dict[str, Any]] = []
    status = "COMPLETED"
    paused_trial: dict[str, Any] | None = None
    for candidate in selected:
        trial_dir = out_dir / "trials" / candidate["name"]
        trial_dir.mkdir(parents=True, exist_ok=True)
        aggregate = trial_metrics(trial_dir)
        if aggregate and aggregate.get("status") == "COMPLETED":
            validate_completed_selection_trial(trial_dir, folds=args.folds)
            validate_reusable_trial_contract(trial_dir, args, candidate)
            result_critic = critique_trial_result(aggregate)
            if result_critic["status"] == "ACCEPTED" and result_critic.get("winner_ranking_eligible", True):
                results.append({"name": candidate["name"], "params": candidate, "a7_result_critic": result_critic, **aggregate})
            else:
                rejected_completed_trials.append(
                    {"name": candidate["name"], "params": candidate, "a7_result_critic": result_critic, **aggregate}
                )
            continue

        cmd = candidate_command(args, candidate, trial_dir)
        completed = subprocess.run(cmd, check=False)
        aggregate = trial_metrics(trial_dir)
        if aggregate and aggregate.get("status") == "COMPLETED":
            validate_completed_selection_trial(trial_dir, folds=args.folds)
            validate_reusable_trial_contract(trial_dir, args, candidate)
            result_critic = critique_trial_result(aggregate)
            if result_critic["status"] == "ACCEPTED" and result_critic.get("winner_ranking_eligible", True):
                results.append({"name": candidate["name"], "params": candidate, "a7_result_critic": result_critic, **aggregate})
            else:
                rejected_completed_trials.append(
                    {"name": candidate["name"], "params": candidate, "a7_result_critic": result_critic, **aggregate}
                )
            continue

        status = "PAUSED_OR_INCOMPLETE"
        paused_trial = {
            "name": candidate["name"],
            "params": candidate,
            "returncode": completed.returncode,
            "metrics_status": (aggregate or {}).get("status"),
            "trial_dir": str(trial_dir.relative_to(PROJECT_ROOT)),
            "resume_command": cmd,
        }
        break

    best = min(results, key=selection_key) if results else None
    best_by_backbone: dict[str, dict[str, Any]] = {}
    for backbone in sorted({str(row.get("params", {}).get("backbone", "")) for row in results if row.get("params")}):
        if not backbone:
            continue
        rows = [row for row in results if row.get("params", {}).get("backbone") == backbone]
        if rows:
            best_by_backbone[backbone] = min(rows, key=selection_key)
    summary = {
        "status": status,
        "combined_cache": args.combined_cache,
        "combined_cache_build": combined,
        "epochs": args.epochs,
        "folds": args.folds,
        "selection_source": "fold_internal_validation_loss_only",
        "selection_metric": "mean_best_val_loss",
        "external_or_pseudo_external_used_for_selection": False,
        "test_used_for_selection": False,
        "include_global_supplementary_train": args.include_global_supplementary_train,
        "n_candidates_requested": requested_candidate_count,
        "n_candidates_after_critic": len(selected),
        "n_candidates_completed": len(results),
        "candidate_config": candidate_config_report,
        "a7_hpo_critic": critic_report,
        "best": best,
        "best_by_backbone": best_by_backbone,
        "paused_trial": paused_trial,
        "rejected_completed_trials": rejected_completed_trials,
        "completed_trials": [
            {
                "name": row["name"],
                "params": row["params"],
                "roc_auc": row.get("roc_auc"),
                "pr_auc": row.get("pr_auc"),
                "brier": row.get("brier"),
                "mean_best_val_loss": row.get("mean_best_val_loss"),
                "mean_final_val_loss": row.get("mean_final_val_loss"),
                "mean_final_evaluation_loss": row.get("mean_final_evaluation_loss"),
                "a7_result_critic": row.get("a7_result_critic"),
                "out_dir": str((out_dir / "trials" / row["name"]).relative_to(PROJECT_ROOT)),
                "final_claim_allowed": row.get("final_claim_allowed"),
            }
            for row in results
        ],
        "available_candidates": [c["name"] for c in selected],
    }
    write_summary(out_dir, summary)
    print(json.dumps({"status": status, "completed": len(results), "best": (best or {}).get("name"), "paused_trial": paused_trial}, indent=2))
    if status == "COMPLETED":
        return 0
    if paused_trial and paused_trial.get("metrics_status") in {"PAUSED_RESUMABLE", "INTERRUPTED_RESUMABLE"}:
        return 0
    return 75


if __name__ == "__main__":
    raise SystemExit(main())
