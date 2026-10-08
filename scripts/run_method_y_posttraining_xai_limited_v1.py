"""Bounded, resumable inference from frozen Method Y v4 PET checkpoints.

No training, model selection, or outer-test performance metric is computed.
All subject-level maps/identifiers stay in the private versioned root.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import io
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from src.method_y_revival import pet_attribution_v4 as pa  # noqa: E402
from src.method_y_revival.atomic_io import atomic_write_bytes, atomic_write_json  # noqa: E402
from src.method_y_revival.contracts import sha256_mapping  # noqa: E402
from src.method_y_revival.posttraining_correction_v4_2 import CorrectionV42Error  # noqa: E402
from src.method_y_revival.posttraining_xai_limited_v1 import (  # noqa: E402
    cluster_bootstrap, map_agreement, select_subjects,
)
from scripts.run_method_y_pet_attribution_v4 import (  # noqa: E402
    _arch_dims_for_branch, _build_real_arch, _infer_bins_from_payload,
    _load_cell_batch, _load_config, _locked_model_cfg,
)

OUTPUT = PROJECT / "results" / "method_y_revival_v4_posttraining_xai_limited_v1"
V4_ROOT = PROJECT / "results" / "method_y_revival_v4_image_factorial_v1"
XAI_CONFIG = PROJECT / "configs" / "method_y_pet_attribution_v4.yaml"
FACTOR_CONFIG = PROJECT / "configs" / "method_y_image_factorial_v4.yaml"
FAMILIES = {"A0": "GROUPNORM_RESIDUAL_TINY_3D", "A1": "GROUPNORM_RESIDUAL_TINY_3D",
            "A2": "RESNET10_3D_GROUPNORM", "A3": "RESNET10_3D_GROUPNORM",
            "A4": "RESNET18_3D_GROUPNORM"}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _require(ok: bool, code: str) -> None:
    if not ok:
        raise CorrectionV42Error(code)


def _locked_inputs() -> tuple[dict, dict, dict, list[dict]]:
    import yaml

    xai = _load_config(XAI_CONFIG)
    _require(xai.get("status") == "POST_HOC_INTERNAL_EXPLORATORY_FINAL",
             "LIMITED_XAI_CONFIG_NOT_FINAL")
    gate = pa.verifier_gate(run_root=V4_ROOT, config_path=FACTOR_CONFIG)
    rc = json.loads((V4_ROOT / "run_contract.json").read_text(encoding="utf-8"))
    _require(gate["run_contract_sha256"] == rc["digest"], "LIMITED_XAI_CONTRACT_MISMATCH")
    factor = yaml.safe_load(FACTOR_CONFIG.read_text(encoding="utf-8"))
    splits = {branch: json.loads(Path(rc["input_fingerprints"][f"{branch}_splits"]
                                       ["path"]).read_text(encoding="utf-8"))
              for branch in ("amyloid", "tau")}
    membership = {branch: rc["input_fingerprints"][f"{branch}_membership"]["path"]
                  for branch in ("amyloid", "tau")}
    cells = pa.enumerate_attribution_cells(
        dev_root=V4_ROOT / "development", orch_root=V4_ROOT / "orchestration",
        splits=splits, membership_paths=membership,
    )
    _require(len(cells) == 50, "LIMITED_XAI_CELL_COVERAGE_INVALID")
    return xai, factor, rc, cells


def build_plan(*, seed: int = 20260714, subjects_per_cell: int = 2) -> dict:
    xai, _factor, rc, cells = _locked_inputs()
    selected = []
    for cell in cells:
        chosen = select_subjects(cell, seed=seed, n=subjects_per_cell)
        prov = pa.verify_final_provenance(
            cell["final_checkpoint"], v4_root=V4_ROOT,
            branch=cell["branch"], arm=cell["arm"], outer=cell["outer"],
        )
        selected.append({
            "branch": cell["branch"], "arm": cell["arm"], "outer": cell["outer"],
            "subjects": chosen, "checkpoint_sha256": prov["checkpoint_sha256"],
            "generation_digest": prov["generation_digest"],
        })
    for branch in ("amyloid", "tau"):
        for fold in range(5):
            matched = [row["subjects"] for row in selected
                       if row["branch"] == branch and row["outer"] == fold]
            _require(len(matched) == 5 and all(row == matched[0] for row in matched),
                     "LIMITED_XAI_ARM_SAMPLE_NOT_MATCHED")
    helper = PROJECT / "src" / "method_y_revival" / "posttraining_xai_limited_v1.py"
    attribution = PROJECT / "src" / "method_y_revival" / "pet_attribution_v4.py"
    plan = {
        "schema_version": "1.0", "status": "LOCKED_INFERENCE_ONLY",
        "science_gate_status": "BLOCKED", "claim_status": "POST_HOC_INTERNAL_EXPLORATORY",
        "positive_claim_allowed": False, "external_claim_allowed": False,
        "source_v4_run_contract_sha256": rc["digest"],
        "source_factor_config_sha256": _sha(FACTOR_CONFIG),
        "source_xai_config_sha256": _sha(XAI_CONFIG),
        "runner_sha256": _sha(Path(__file__)),
        "helper_sha256": _sha(helper), "attribution_module_sha256": _sha(attribution),
        "seed": seed, "subjects_per_cell": subjects_per_cell,
        "selection_rule": "SHA256_SEED_BRANCH_FOLD_SUBJECT_OUTCOME_BLIND_MATCHED_ACROSS_ARMS",
        "device": "cpu", "cells": selected,
        "private_subject_level_maps": True,
        "outer_test_metric_recomputation": False,
    }
    plan["plan_sha256"] = sha256_mapping(plan)
    return plan


def _load_or_lock_plan(output: Path) -> dict:
    plan = build_plan()
    target = output / "private" / "inference_plan.json"
    if target.exists():
        _require(json.loads(target.read_text(encoding="utf-8")) == plan,
                 "LIMITED_XAI_PLAN_OR_SOURCE_MISMATCH")
    else:
        atomic_write_json(target, plan)
    return plan


def _stored_map_paths(output: Path, row: dict, subject: str) -> tuple[Path, Path]:
    private = (output / "private" / "maps" / row["branch"] / row["arm"]
               / f"fold_{row['outer']}")
    tag = hashlib.sha256(subject.encode("utf-8")).hexdigest()[:20]
    return private / f"{tag}.npz", private / f"{tag}.json"


def _read_committed(output: Path, plan: dict, row: dict, subject: str) -> dict | None:
    map_path, meta_path = _stored_map_paths(output, row, subject)
    if not meta_path.exists():
        return None
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    _require(meta.get("status") == "COMMITTED"
             and meta.get("plan_sha256") == plan["plan_sha256"]
             and meta.get("subject") == subject
             and meta.get("checkpoint_sha256") == row["checkpoint_sha256"]
             and map_path.is_file() and _sha(map_path) == meta.get("maps_sha256"),
             "LIMITED_XAI_MAP_RECEIPT_MISMATCH")
    return meta


def _compute_maps(net, img: torch.Tensor, tab: torch.Tensor, *, layer: str,
                  shallow: str | None) -> tuple[dict[str, np.ndarray], dict]:
    volume, table = img.unsqueeze(0), tab.unsqueeze(0)
    methods = {
        "layercam_primary": pa.layercam_map(net, volume, table, layer,
                                             risk_mode="logit_sum",
                                             shallow_layer_name=shallow),
        "layercam_sensitivity": pa.layercam_map(net, volume, table, layer,
                                                 risk_mode="mortality_last",
                                                 shallow_layer_name=shallow),
        "gradcam_primary": pa.gradcam_map(net, volume, table, layer,
                                          risk_mode="logit_sum"),
        "scorecam_primary": pa.scorecam_map(net, volume, table, layer,
                                            risk_mode="logit_sum"),
    }
    maps = {name: value.detach().cpu().numpy().astype(np.float32).squeeze()
            for name, value in methods.items()}
    _require(all(value.shape == tuple(img.shape[-3:]) and np.isfinite(value).all()
                 for value in maps.values()), "LIMITED_XAI_MAP_INVALID")
    agree = {
        "risk_target": map_agreement(maps["layercam_primary"], maps["layercam_sensitivity"]),
        "layer_vs_score": map_agreement(maps["layercam_primary"], maps["scorecam_primary"]),
        "layer_vs_grad": map_agreement(maps["layercam_primary"], maps["gradcam_primary"]),
    }
    return maps, agree


def _commit_one(output: Path, plan: dict, row: dict, subject: str,
                maps: dict[str, np.ndarray], agree: dict) -> dict:
    map_path, meta_path = _stored_map_paths(output, row, subject)
    packed = io.BytesIO()
    np.savez_compressed(packed, **maps)
    atomic_write_bytes(map_path, packed.getvalue())
    meta = {
        "status": "COMMITTED", "privacy_tier": "PRIVATE_SUBJECT_MAP",
        "plan_sha256": plan["plan_sha256"], "subject": subject,
        "branch": row["branch"], "arm": row["arm"], "fold": row["outer"],
        "checkpoint_sha256": row["checkpoint_sha256"],
        "maps_sha256": _sha(map_path), "agreement": agree,
    }
    atomic_write_json(meta_path, meta)
    return meta


def _aggregate(output: Path, plan: dict) -> dict:
    private_rows = []
    for row in plan["cells"]:
        for subject in row["subjects"]:
            meta = _read_committed(output, plan, row, subject)
            _require(meta is not None, "LIMITED_XAI_INCOMPLETE_CELL")
            private_rows.append(meta)
    _require(len(private_rows) == 50 * plan["subjects_per_cell"],
             "LIMITED_XAI_SAMPLE_COUNT_INVALID")
    by_branch_arm = {}
    for branch in ("amyloid", "tau"):
        for arm in ("A0", "A1", "A2", "A3", "A4"):
            rows = [r for r in private_rows if r["branch"] == branch and r["arm"] == arm]
            metrics = {}
            for label in ("risk_target", "layer_vs_score", "layer_vs_grad"):
                values = [{"fold": r["fold"], "rho": r["agreement"][label]["rho"]}
                          for r in rows]
                metrics[label] = cluster_bootstrap(
                    values, field="rho", seed=plan["seed"] + len(by_branch_arm),
                )
            by_branch_arm[f"{branch}/{arm}"] = metrics
    report = {
        "schema_version": "1.0", "status": "COMPLETED_LIMITED_INFERENCE",
        "science_gate_status": "BLOCKED", "claim_status": "POST_HOC_INTERNAL_EXPLORATORY",
        "positive_claim_allowed": False, "external_claim_allowed": False,
        "plan_sha256": plan["plan_sha256"], "cells_completed": 50,
        "private_subject_maps_completed": len(private_rows),
        "methods": ["LayerCAM", "GradCAM", "ScoreCAM"],
        "risk_targets": ["logit_sum", "mortality_last"],
        "map_inference_device": "cpu", "outer_test_metric_recomputation": False,
        "limitations": [
            "TWO_OUTCOME_BLIND_SUBJECTS_PER_CELL_ONLY",
            "CONDITIONAL_FIXED_MODELS_AND_POST_HOC_XAI",
            "FOLD_THEN_SUBJECT_BOOTSTRAP_EXCLUDES_MODEL_REFIT_UNCERTAINTY",
            "PET_QUANTITATIVE_AND_SCANNER_AND_EXTERNAL_GATES_STILL_BLOCKED",
        ],
        "aggregate_agreement": by_branch_arm,
    }
    report["report_sha256"] = sha256_mapping(report)
    target = output / "public_bundle" / "limited_xai_aggregate.json"
    if target.exists():
        _require(json.loads(target.read_text(encoding="utf-8")) == report,
                 "LIMITED_XAI_PUBLIC_REPORT_MISMATCH")
    else:
        atomic_write_json(target, report)
    return report


def run(output: Path) -> dict:
    import msvcrt

    output.mkdir(parents=True, exist_ok=True)
    lock_path = output / ".writer.lock"
    with lock_path.open("a+b") as lock:
        lock.seek(0)
        try:
            msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError as exc:
            raise CorrectionV42Error("LIMITED_XAI_ACTIVE_WRITER") from exc
        try:
            plan = _load_or_lock_plan(output)
            xai, factor, rc, cells = _locked_inputs()
            by_key = {(c["branch"], c["arm"], c["outer"]): c for c in cells}
            torch.set_num_threads(4)
            for row in plan["cells"]:
                cell = by_key[(row["branch"], row["arm"], row["outer"])]
                committed = [_read_committed(output, plan, row, subject)
                             for subject in row["subjects"]]
                if all(meta is not None for meta in committed):
                    continue
                prov = pa.verify_final_provenance(
                    cell["final_checkpoint"], v4_root=V4_ROOT,
                    branch=cell["branch"], arm=cell["arm"], outer=cell["outer"],
                )
                _require(prov["checkpoint_sha256"] == row["checkpoint_sha256"],
                         "LIMITED_XAI_CHECKPOINT_DRIFT")
                payload = pa.read_only_load_checkpoint(
                    cell["final_checkpoint"], expected_sha256=row["checkpoint_sha256"],
                )
                branch, arm = row["branch"], row["arm"]
                features = factor["branches"][branch]["model_features"]
                net = _build_real_arch(arm, _locked_model_cfg(),
                                       tabular_dim=len(features),
                                       bins=_infer_bins_from_payload(payload))
                net.load_state_dict(payload["model_state"], strict=True)
                net.eval()
                selected_cell = dict(cell, firewall_subjects=row["subjects"])
                batch = _load_cell_batch(selected_cell, run_contract=rc,
                                         model_cfg=_locked_model_cfg(), model_features=features)
                family = FAMILIES[arm]
                layer = xai["targets"][family][0]
                shallow = xai["targets"]["shallow_fusion"][family]
                for subject, img, tab in batch:
                    if _read_committed(output, plan, row, subject) is not None:
                        continue
                    maps, agree = _compute_maps(net, img, tab, layer=layer, shallow=shallow)
                    _commit_one(output, plan, row, subject, maps, agree)
                    print(json.dumps({"status": "MAP_COMMITTED", "branch": branch,
                                      "arm": arm, "fold": row["outer"]}), flush=True)
                del net, payload, batch
                gc.collect()
            return _aggregate(output, plan)
        finally:
            lock.seek(0)
            msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("plan", "run", "verify"), required=True)
    parser.add_argument("--output-root", type=Path, default=OUTPUT)
    args = parser.parse_args()
    output = args.output_root.resolve()
    _require(output.is_relative_to(PROJECT.resolve() / "results"),
             "LIMITED_XAI_OUTPUT_NOT_UNDER_PROJECT_RESULTS")
    if args.mode == "plan":
        plan = _load_or_lock_plan(output)
        print(json.dumps({"status": "PLAN_LOCKED", "cells": len(plan["cells"]),
                          "maps_expected": len(plan["cells"]) * plan["subjects_per_cell"],
                          "plan_sha256": plan["plan_sha256"]}), flush=True)
    elif args.mode == "run":
        report = run(output)
        print(json.dumps({"status": report["status"],
                          "cells": report["cells_completed"],
                          "subject_maps": report["private_subject_maps_completed"],
                          "science_gate_status": report["science_gate_status"]}), flush=True)
    else:
        plan = _load_or_lock_plan(output)
        report = _aggregate(output, plan)
        print(json.dumps({"status": "PASS_TECHNICAL_MAP_LINEAGE_ONLY",
                          "report_sha256": report["report_sha256"]}), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(json.dumps({"status": "BLOCKED", "reason": f"{type(exc).__name__}:{exc}"}),
              flush=True)
        raise SystemExit(3)
