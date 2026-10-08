"""Locked, fold-resumable internal outer predictions for the MX-05 add-on.

The cohort is historically inspected, so these results remain exploratory.
Private, write-once per-fold/per-arm receipts make an interrupted inference
retry deterministic without allowing a new model choice or test-set tuning.
"""
from __future__ import annotations

import math
import random
from pathlib import Path

from scripts.mx_fasttrack_v2.audit_inputs import read_csv
from scripts.mx_fasttrack_v2.run import read, sha
from scripts.mx_fasttrack_v2_1.supervise_v21 import immutable_json
from scripts.mx_fasttrack_v2_4_2.checkpoint_projection import baseline_binding
from scripts.mx_science_extension_v1.analysis import ARMS, validate_panel
from scripts.mx_science_extension_v1.audit_and_lock import OUTPUT, PARENT, _digest
from scripts.mx_science_extension_v1.baselines import fit_controls
from scripts.mx_science_extension_v1.execution import (
    LOCK, arm_evidence, ensure_execution_lock,
)
from scripts.mx_science_extension_v1.plan_arms import make_arm_commands
from scripts.mx_science_extension_v1.selection import (
    NAME as SELECTION_NAME, SEEDS, VARIANTS, arm_directory, build_selection,
)

OUTER_LOCK = "extension_outer_lock.v1.json"
PANEL = "patient_panel.v1.json"
EXPECTED_N = 1336


def _validate_probabilities(probability: object, n: int, identity: str) -> None:
    if (not isinstance(probability, list) or len(probability) != n
        or any(type(p) not in (int, float) or not math.isfinite(p)
               or not 0 <= p <= 1 for p in probability)):
        raise RuntimeError(f"EXTENSION_OUTER_MODEL_PROBABILITIES_INVALID:{identity}")


def _private(root: Path) -> Path:
    return root / OUTPUT / "_private_resume" / "outer_v1"


def _arm_name(seed: int, variant: str) -> str:
    return f"hybrid_{variant}_seed{seed}"


def _control_name(kind: str) -> str:
    return f"{kind}_baseline_seed13"


def outer_lock_payload(root: Path) -> dict:
    ensure_execution_lock(root, create=False)
    reports = root / OUTPUT / "reports"
    selection_path = reports / SELECTION_NAME
    if not selection_path.is_file() or read(selection_path) != build_selection(root):
        raise RuntimeError("EXTENSION_OUTER_SELECTION_NOT_LOCKED_OR_DRIFTED")
    commands = make_arm_commands(root)
    controls = {}
    for kind in ("mri_only", "random_label"):
        name = _control_name(kind)
        evidence = arm_evidence(root, name, commands[name])
        receipt = reports / "inner_arm_receipts" / f"{name}.json"
        if read(receipt) != evidence:
            raise RuntimeError(f"EXTENSION_OUTER_CONTROL_NOT_LOCKED:{name}")
        controls[name] = {"receipt_sha256": sha(receipt),
                          "contract_sha256": evidence["contract_sha256"],
                          "checkpoint_sha256": evidence["checkpoint_sha256"]}
    for seed in (29, 47):
        for variant in VARIANTS:
            name = _arm_name(seed, variant)
            receipt = reports / "inner_arm_receipts" / f"{name}.json"
            if read(receipt) != arm_evidence(root, name, commands[name]):
                raise RuntimeError(f"EXTENSION_OUTER_HYBRID_NOT_LOCKED:{name}")
    cache = root / "data/cached/t1_volume_cache_128.h5"
    cache_stat = cache.stat()
    parent_prereg = read(root / PARENT / "reports/fasttrack_preregistration.json")
    return {
        "schema_version": 1, "status": "LOCKED_BEFORE_EXTENSION_OUTER_ACCESS",
        "execution_preregistration_sha256": sha(reports / LOCK),
        "fold_local_selection_sha256": sha(selection_path),
        "control_arms": controls,
        "split_sha256": {str(fold): sha(root / "splits" / f"fold_{fold}_test.csv")
                         for fold in range(5)},
        "cache_sha256": parent_prereg["cache_sha256"],
        "cache_bytes": cache_stat.st_size,
        "cache_mtime_ns": cache_stat.st_mtime_ns,
        "expected_patient_count": EXPECTED_N,
        "model_inference_scope": "fixed_checkpoints_and_fold_local_inner_choices_only",
        "outer_used_for_selection": False,
        "historically_inspected_internal_outer": True,
        "external_validation": False,
    }


def ensure_outer_lock(root: Path, *, create: bool) -> dict:
    path = root / OUTPUT / "reports" / OUTER_LOCK
    if not path.is_file() and not create:
        raise RuntimeError("EXTENSION_OUTER_LOCK_MISSING")
    expected = outer_lock_payload(root)
    if path.is_file():
        if read(path) != expected:
            raise RuntimeError("EXTENSION_OUTER_LOCK_DRIFT")
    else:
        private = _private(root)
        if private.exists() and any(private.iterdir()):
            raise RuntimeError("EXTENSION_OUTER_LOCK_CREATED_AFTER_ACCESS")
        if sha(root / "data/cached/t1_volume_cache_128.h5") != expected["cache_sha256"]:
            raise RuntimeError("EXTENSION_OUTER_CACHE_HASH_CHANGED")
        immutable_json(path, expected)
    return expected


def _fold_context(root: Path, fold: int, sample_to_index: dict[str, int],
                  ok: set[int], lock_sha: str) -> dict:
    split = root / "splits" / f"fold_{fold}_test.csv"
    rows = [row for row in read_csv(split)
            if row["sample_hash"] in sample_to_index
            and sample_to_index[row["sample_hash"]] in ok]
    context = {
        "schema_version": 1, "fold": fold, "outer_lock_sha256": lock_sha,
        "split_sha256": sha(split), "private_patient_identifiers": True,
        "rows": [{"patient_hash": row["patient_hash"],
                  "sample_hash": row["sample_hash"],
                  "label": int(row["ab_label"]),
                  "gap_days": float(row["abs_pet_t1_gap_days"])} for row in rows],
    }
    if (not rows or len({r["patient_hash"] for r in rows}) != len(rows)
        or any(r["label"] not in (0, 1) or not math.isfinite(r["gap_days"])
               or r["gap_days"] < 0 for r in context["rows"])):
        raise RuntimeError(f"EXTENSION_OUTER_FOLD_COHORT_INVALID:{fold}")
    audit = read(root / OUTPUT / "reports/day1_patient_provenance_audit.json")
    if len(rows) != audit["folds"][fold]["outer_qc_n"]:
        raise RuntimeError(f"EXTENSION_OUTER_FOLD_QC_COUNT_DRIFT:{fold}")
    immutable_json(_private(root) / f"fold_{fold}" / "context.json", context)
    return context


def _checkpoint_metadata(root: Path, seed: int, variant: str, fold: int,
                         selection: dict, migration: dict) -> tuple[Path, str, str]:
    info = selection["arms"][f"{seed}|{variant}"]
    checkpoint = arm_directory(root, seed, variant) / f"method_x_3d_fold_{fold}.pt"
    digest = info["checkpoint_sha256"][fold]
    if sha(checkpoint) != digest:
        raise RuntimeError(f"EXTENSION_OUTER_CHECKPOINT_HASH_CHANGED:{seed}:{variant}:{fold}")
    historical = seed == 13 and variant == "baseline"
    expected_contract = (migration["old_digest"] if historical
                         else info["contract_digest"])
    if historical and (checkpoint.resolve() not in migration["paths"]
                       or migration["new_digest"] != info["contract_digest"]):
        raise RuntimeError("EXTENSION_HISTORICAL_MIGRATION_BINDING_INVALID")
    return checkpoint, digest, expected_contract


def _model_probabilities(root: Path, *, fold: int, seed: int, variant: str,
                         mode: str, checkpoint: Path, checkpoint_sha: str,
                         contract_digest: str, context: dict,
                         sample_to_index: dict[str, int], lock_sha: str,
                         receipt_name: str) -> list[float]:
    import torch
    from torch.utils.data import DataLoader
    from scripts.train_3d_mri import H5MethodXDataset, predict
    from src.evaluation.method_x_xai_forward import build_model_from_checkpoint

    path = _private(root) / f"fold_{fold}" / f"{receipt_name}.json"
    envelope = {"schema_version": 1, "fold": fold, "seed": seed,
                "variant": variant, "input_mode": mode,
                "outer_lock_sha256": lock_sha,
                "fold_context_sha256": sha(path.parent / "context.json"),
                "checkpoint_sha256": checkpoint_sha,
                "contract_digest_as_stored": contract_digest,
                "arm": receipt_name}
    if path.is_file():
        receipt = read(path)
        probability = receipt.get("probabilities")
        if ({key: receipt.get(key) for key in envelope} != envelope
            or not isinstance(probability, list)):
            raise RuntimeError(f"EXTENSION_OUTER_MODEL_RECEIPT_DRIFT:{receipt_name}:{fold}")
    else:
        if sha(checkpoint) != checkpoint_sha:
            raise RuntimeError("EXTENSION_OUTER_CHECKPOINT_CHANGED_DURING_INFERENCE")
        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=True)
        state = read(checkpoint.with_name(f"method_x_3d_fold_{fold}_state.json"))
        if (ckpt.get("fold") != fold or ckpt.get("backbone") != "amuse_x_hybrid3d"
            or ckpt.get("hybrid_variant") != variant
            or ckpt.get("input_mode") != mode
            or ckpt.get("drop_tabular_feature") != "none"
            or ckpt.get("run_contract_digest") != contract_digest
            or ckpt.get("split_membership_digest") !=
               (state.get("details") or {}).get("split_membership_digest")
            or ckpt.get("hpo_selection_only") is not True
            or ckpt.get("outer_test_access_count") != 0
            or ckpt.get("outer_test_evaluation_count") != 0):
            raise RuntimeError(f"EXTENSION_OUTER_CHECKPOINT_IDENTITY_INVALID:{receipt_name}:{fold}")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device.type != "cuda":
            raise RuntimeError("EXTENSION_CUDA_REQUIRED_FOR_LOCKED_INFERENCE")
        model, state_source = build_model_from_checkpoint(ckpt, device=device)
        if state_source != "model_state_dict":
            raise RuntimeError("EXTENSION_MODEL_STATE_SOURCE_UNEXPECTED")
        rows = context["rows"]
        indices = [sample_to_index[row["sample_hash"]] for row in rows]
        labels = [row["label"] for row in rows]
        dataset = H5MethodXDataset(root / "data/cached/t1_volume_cache_128.h5",
                                   indices, labels, input_mode=mode,
                                   sanity_random_labels=False)
        loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
        observed, probability = predict(model, loader, device,
                                        ckpt["tab_mean"], ckpt["tab_std"])
        if observed != labels:
            raise RuntimeError("EXTENSION_OUTER_LABEL_ORDER_MISMATCH")
        # Never commit an invalid immutable receipt: it would block every retry.
        _validate_probabilities(probability, len(rows), f"{receipt_name}:{fold}")
        immutable_json(path, {**envelope, "probabilities": probability,
                              "attempt_semantics": "deterministic_fixed_model_reinference_if_interrupted"})
        del model, loader, dataset, ckpt
        torch.cuda.empty_cache()
    _validate_probabilities(probability, len(context["rows"]), f"{receipt_name}:{fold}")
    return [float(p) for p in probability]


def _control_probabilities(root: Path, fold: int, seed: int, context: dict,
                           lock_sha: str, real_samples: set[str]) -> dict:
    path = _private(root) / f"fold_{fold}" / f"tabular_controls_seed{seed}.json"
    train = [row for row in read_csv(root / "splits" / f"fold_{fold}_train.csv")
             if row["sample_hash"] in real_samples]
    random.Random(seed + fold).shuffle(train)
    n_val = min(max(int(len(train) * 0.15), 1), len(train) - 1)
    val, fit = train[:n_val], train[n_val:]
    audit = read(root / OUTPUT / "reports/day1_patient_provenance_audit.json")
    expected = audit["folds"][fold]["inner"][[13, 29, 47].index(seed)]
    membership = _digest({"fit": sorted(r["patient_hash"] for r in fit),
                          "validation": sorted(r["patient_hash"] for r in val)})
    if (expected["membership_sha256"] != membership
        or len(fit) != expected["fit_n"] or len(val) != expected["inner_validation_n"]
        or {r["patient_hash"] for r in train} &
           {r["patient_hash"] for r in context["rows"]}):
        raise RuntimeError(f"EXTENSION_TABULAR_INNER_MEMBERSHIP_INVALID:{fold}:{seed}")
    outer_rows = read_csv(root / "splits" / f"fold_{fold}_test.csv")
    lookup = {r["sample_hash"]: r for r in outer_rows}
    covariates = [lookup[row["sample_hash"]] for row in context["rows"]]
    evidence = {"schema_version": 1, "fold": fold, "seed": seed,
                "outer_lock_sha256": lock_sha,
                "fold_context_sha256": sha(path.parent / "context.json"),
                "inner_membership_sha256": membership,
                "train_split_sha256": sha(root / "splits" / f"fold_{fold}_train.csv")}
    if path.is_file():
        saved = read(path)
        if {key: saved.get(key) for key in evidence} != evidence:
            raise RuntimeError(f"EXTENSION_TABULAR_RECEIPT_DRIFT:{fold}:{seed}")
        result = saved["result"]
    else:
        result = fit_controls(fit, val, covariates)
        for key in ("tabular_outer_probabilities", "prevalence_outer_probabilities"):
            _validate_probabilities(result.get(key), len(covariates), f"{key}:{fold}:{seed}")
        immutable_json(path, {**evidence, "result": result})
    for key in ("tabular_outer_probabilities", "prevalence_outer_probabilities"):
        _validate_probabilities(result.get(key), len(covariates), f"{key}:{fold}:{seed}")
    if (result.get("preprocessing_fit_scope") != "fit_partition_only"
        or result.get("selection_scope") != "fit_to_inner_only_no_outer_label_access"
        or len(result.get("tabular_outer_probabilities") or []) != len(covariates)
        or len(result.get("prevalence_outer_probabilities") or []) != len(covariates)):
        raise RuntimeError(f"EXTENSION_TABULAR_RECEIPT_INVALID:{fold}:{seed}")
    return result


def run_outer(root: Path) -> dict:
    lock = ensure_outer_lock(root, create=False)
    selection = read(root / OUTPUT / "reports" / SELECTION_NAME)
    migration = baseline_binding(root, root / PARENT)
    from scripts.train_3d_mri import cache_index, ok_indices

    cache = root / "data/cached/t1_volume_cache_128.h5"
    sample_to_index, ok = cache_index(cache), ok_indices(cache)
    index = read_csv(root / "docs/fasttrack_private/cache_full_index.csv")
    real_samples = {row["sample_hash"] for row in index
                    if str(row["is_synthetic"]).lower() in {"0", "false"}}
    lock_sha = sha(root / OUTPUT / "reports" / OUTER_LOCK)
    panel = []
    seen = set()
    for fold in range(5):
        context = _fold_context(root, fold, sample_to_index, ok, lock_sha)
        model_values: dict[str, list[float]] = {}
        for seed in SEEDS:
            for variant in VARIANTS:
                cp, digest, contract = _checkpoint_metadata(
                    root, seed, variant, fold, selection, migration)
                name = _arm_name(seed, variant)
                model_values[name] = _model_probabilities(
                    root, fold=fold, seed=seed, variant=variant,
                    mode="mri_plus_tabular", checkpoint=cp,
                    checkpoint_sha=digest, contract_digest=contract,
                    context=context, sample_to_index=sample_to_index,
                    lock_sha=lock_sha, receipt_name=name)
        for kind, mode in (("mri_only", "mri_only"),
                           ("random_label", "mri_plus_tabular")):
            name = _control_name(kind)
            cp = root / OUTPUT / "inner_tasks" / name / f"method_x_3d_fold_{fold}.pt"
            evidence = lock["control_arms"][name]
            digest = evidence["checkpoint_sha256"][fold]
            if sha(cp) != digest:
                raise RuntimeError(f"EXTENSION_CONTROL_CHECKPOINT_DRIFT:{name}:{fold}")
            contract = read(cp.parent / "run_contract.json")["run_contract_digest"]
            model_values[name] = _model_probabilities(
                root, fold=fold, seed=13, variant="baseline", mode=mode,
                checkpoint=cp, checkpoint_sha=digest,
                contract_digest=contract, context=context,
                sample_to_index=sample_to_index, lock_sha=lock_sha,
                receipt_name=name)
        control_values = {seed: _control_probabilities(root, fold, seed, context,
                                                       lock_sha, real_samples)
                          for seed in SEEDS}
        choices = {(row["seed"], row["fold"]): row["selected_variant"]
                   for row in selection["choices"]}
        for index_in_fold, row in enumerate(context["rows"]):
            patient = row["patient_hash"]
            if patient in seen:
                raise RuntimeError("EXTENSION_OUTER_PATIENT_REPEATED_ACROSS_FOLDS")
            seen.add(patient)
            predictions = {arm: {} for arm in ARMS}
            for seed in SEEDS:
                for variant, arm in (("baseline", "hybrid_baseline"),
                                     ("plus_dense_skip", "hybrid_dense")):
                    predictions[arm][str(seed)] = model_values[
                        _arm_name(seed, variant)][index_in_fold]
                selected = choices[(seed, fold)]
                predictions["hybrid_selected"][str(seed)] = model_values[
                    _arm_name(seed, selected)][index_in_fold]
                predictions["tabular_only"][str(seed)] = control_values[seed][
                    "tabular_outer_probabilities"][index_in_fold]
                predictions["prevalence"][str(seed)] = control_values[seed][
                    "prevalence_outer_probabilities"][index_in_fold]
            predictions["mri_only"]["13"] = model_values[
                _control_name("mri_only")][index_in_fold]
            predictions["random_label"]["13"] = model_values[
                _control_name("random_label")][index_in_fold]
            panel.append({"patient_hash": patient, "label": row["label"],
                          "fold": fold, "gap_days": row["gap_days"],
                          "predictions": predictions})
    validate_panel(panel, expected_n=EXPECTED_N)
    payload = {"schema_version": 1, "status": "COMPLETED_PRIVATE_PATIENT_PANEL",
               "outer_lock_sha256": lock_sha, "patient_count": len(panel),
               "private_identifiers_and_predictions": True,
               "external_validation": False, "rows": panel}
    path = _private(root) / PANEL
    immutable_json(path, payload)
    return {"status": payload["status"], "patient_count": len(panel),
            "private_panel_sha256": sha(path)}
