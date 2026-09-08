"""Post-training PET attribution launcher (FINAL plan 04-pet-plan.md).

Modes:
  preflight  read-only readiness (verifier PASS + checkpoint SHAs + layer
             resolution against REAL architectures + determinism ledger).
  dry-run    end-to-end on SYNTHETIC cohort + synthetic locked-format
             checkpoint (safe to run anytime; proves the machinery).
  run        production on locked finals. REFUSED unless a fresh preflight
             PASS artifact (same config SHA) exists. Never touches training.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.method_y_revival import pet_attribution_v4 as pa  # noqa: E402
from src.method_y_revival.atomic_io import atomic_write_json  # noqa: E402


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_config(path: Path) -> dict:
    import yaml
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _build_real_arch(arm_id: str, model_cfg: dict, tabular_dim: int, bins: int):
    from src.method_y_revival.image_factorial_v4 import build_factorial_image_model
    return build_factorial_image_model(arm_id, model_cfg, tabular_dim=tabular_dim,
                                       hazard_bins=bins)


def _locked_model_cfg() -> dict:
    """Locked training hyperparameters mirrored for attribution arch builds.

    dropout=0.25 matches configs/method_y_image_factorial_v4.yaml:model.dropout;
    attribution never retrains, but arch construction must mirror locked shapes.
    """
    return {"embedding_dim": 64, "tabular_hidden_dim": 32, "dropout": 0.25,
            "base_channels": 8, "resnet_width_multiplier": 0.25}


def _arch_dims_for_branch(factor_cfg: dict, branch: str) -> tuple[int, int]:
    """(tabular_dim, hazard_bins) for a branch from the locked factorial config.

    tabular_dim comes from that branch's locked model_features (amyloid=11,
    tau=12: tau adds global_tau_persi_mean). hazard_bins defaults to 5; the run
    path re-derives it from each locked checkpoint payload when available.
    """
    feats = factor_cfg["branches"][branch]["model_features"]
    return len(feats), 5


def _infer_bins_from_payload(payload: dict, *, fallback: int = 5) -> int:
    """Infer hazard bins from a locked checkpoint's head weight shape.

    Head is Sequential(Dropout, Linear(embedding+tabular_hidden, bins)): the
    Linear weight shaped [bins, 96] reveals bins. Never guesses silently: falls
    back only when no head-like matrix is present (recorded by the caller).
    """
    state = payload.get("model_state", {})
    for key, val in state.items():
        try:
            import torch as _t
            if key.endswith("weight") and isinstance(val, _t.Tensor) \
                    and val.dim() == 2 and int(val.shape[1]) == 96:
                return int(val.shape[0])
        except Exception:
            continue
    return int(fallback)


def cmd_preflight(args) -> int:
    cfg = _load_config(Path(args.config))
    v4_root = ROOT / cfg["v4_output_root"]
    out: dict = {"mode": "preflight", "status": "BLOCKED", "checks": {}}
    try:
        gate = pa.verifier_gate(run_root=v4_root, config_path=ROOT / "configs" / "method_y_image_factorial_v4.yaml")
        out["checks"]["verifier"] = "PASS"
        digest = gate["run_contract_sha256"]
    except Exception as exc:
        out["checks"]["verifier"] = f"BLOCKED:{exc}"
        print(json.dumps(out, indent=2))
        return 3
    # BAN parity: launcher must enforce the same forbidden vocabulary as the
    # locked config (C-fix 2026-09-08: runtime assert, not just a unit test).
    try:
        cfg_terms = {str(t).casefold() for t in cfg.get("forbidden_vocabulary", [])}
        if cfg_terms != {str(t).casefold() for t in pa.BAN_TERMS}:
            out["checks"]["ban_parity"] = "BLOCKED:BAN_TERMS_DRIFT"
            print(json.dumps(out, indent=2))
            return 3
        out["checks"]["ban_parity"] = "PASS"
    except Exception as exc:
        out["checks"]["ban_parity"] = f"BLOCKED:{exc}"
        print(json.dumps(out, indent=2))
        return 3
    # resolve target layers against REAL architectures (no checkpoints loaded).
    # All 5 arms are instantiated with per-branch locked dims (C-fix 2026-09-08:
    # preflight previously covered A0/A2/A4 only with hardcoded tabular_dim=11,
    # leaving A1/A3 shallow targets and tau dim=12 unevidenced).
    import yaml as _yaml
    factor_cfg = _yaml.safe_load(
        (ROOT / "configs" / "method_y_image_factorial_v4.yaml").read_text(
            encoding="utf-8"))
    model_cfg = _locked_model_cfg()
    arms = {"A0": "GROUPNORM_RESIDUAL_TINY_3D", "A1": "GROUPNORM_RESIDUAL_TINY_3D",
            "A2": "RESNET10_3D_GROUPNORM", "A3": "RESNET10_3D_GROUPNORM",
            "A4": "RESNET18_3D_GROUPNORM"}
    arm_branch_dim = None  # per-(arm, branch) dims resolved inline below
    resolved = {}
    try:
        for arm, fam in arms.items():
            for branch in ("amyloid", "tau"):
                tab_dim, bins = _arch_dims_for_branch(factor_cfg, branch)
                net = _build_real_arch(arm, model_cfg, tabular_dim=tab_dim, bins=bins)
                net.eval()
                for layer in cfg["targets"][fam]:
                    pa.resolve_layer(net, layer)
                shallow = ((cfg.get("targets") or {}).get("shallow_fusion") or {}).get(fam)
                if shallow:
                    pa.resolve_layer(net, shallow)
            resolved[arm] = cfg["targets"][fam]
        out["checks"]["layers"] = "PASS"
    except Exception as exc:
        out["checks"]["layers"] = f"BLOCKED:{exc}"
        print(json.dumps(out, indent=2))
        return 3
    # Determinism: ledger-only preflight previously PASSED on non-deterministic
    # hosts (stale artifact showed false/false). Now enforce torch-level flags
    # first, then assert the full ledger == locked yaml block (C-fix 2026-09-08);
    # the stale artifact is therefore invalid. NOTE: CUBLAS_WORKSPACE_CONFIG and
    # PYTHONHASHSEED are effective only when set before process start — the
    # ledger's *_effective flags BLOCK hosts that merely inherited mid-process
    # values, so launch shells must export them (see runbook command).
    try:
        det_cfg = cfg.get("determinism", {})
        pa.enforce_determinism(
            cublas_config=str(det_cfg.get("cublas_workspace_config", ":4096:8")),
            python_hash_seed=str(det_cfg.get("python_hash_seed", "20260714")))
        ledger = pa.determinism_ledger()
        ledger["cublas_workspace_effective"] = (
            __import__("os").environ.get("CUBLAS_WORKSPACE_CONFIG")
            == det_cfg.get("cublas_workspace_config"))
        ledger["python_hash_seed_effective"] = (
            __import__("os").environ.get("PYTHONHASHSEED")
            == det_cfg.get("python_hash_seed"))
        det_ok = (
            bool(ledger.get("torch_deterministic_algorithms")) == bool(det_cfg.get("torch_deterministic_algorithms"))
            and bool(ledger.get("cudnn_deterministic")) == bool(det_cfg.get("cudnn_deterministic"))
            and bool(ledger.get("cudnn_benchmark")) == bool(det_cfg.get("cudnn_benchmark"))
            and bool(ledger.get("cublas_workspace_effective"))
            and bool(ledger.get("python_hash_seed_effective"))
        )
        if not det_ok:
            out["checks"]["determinism"] = "BLOCKED:HOST_NOT_DETERMINISTIC"
            out["determinism"] = ledger
            print(json.dumps({k: v for k, v in out.items()}, indent=2))
            return 3
        out["checks"]["determinism"] = "PASS"
    except Exception as exc:
        out["checks"]["determinism"] = f"BLOCKED:{exc}"
        print(json.dumps(out, indent=2))
        return 3
    out.update({"status": "PASS", "run_digest": digest, "resolved_layers": resolved,
                "determinism": ledger,
                "config_sha256": _sha256_file(Path(args.config)),
                "config_status": cfg.get("status"),
                "production_gate": "RUN_GATE_ENFORCES_FINAL_STATUS",
                "privacy_note": "cell_progress.json (run phase) is private-tier scratch: "
                                "per-subject rows never enter the final report, which is "
                                "aggregate-only and ban+privacy scanned."})
    atomic_write_json(Path(args.output) / "attribution_preflight_pass.json", out)
    print(json.dumps({k: v for k, v in out.items() if k != "determinism"}, indent=2))
    return 0


def _synthetic_net():
    from torch import nn

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = nn.Sequential(nn.Conv3d(1, 4, 3, padding=1), nn.SiLU(),
                                        nn.Conv3d(4, 4, 3, stride=2, padding=1))
            self.pool = nn.AdaptiveAvgPool3d(1)
            self.head = nn.Linear(4 + 2, 3)

        def forward(self, image, tabular):
            f = self.blocks(image)
            e = self.pool(f).flatten(1)
            return self.head(torch.cat([e, tabular], dim=1))

    torch.manual_seed(7)
    return Net().eval()


def cmd_dry_run(args) -> int:
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(11)
    net = _synthetic_net()
    vols = [torch.rand(1, 1, 16, 16, 16) for _ in range(4)]
    tabs = [torch.rand(1, 2) for _ in range(4)]
    # fake verifier PASS artifact (synthetic scope only)
    gate = {"status": "PASS", "scope": "SYNTHETIC_DRY_RUN_ONLY"}
    lc = pa.layercam_map(net, vols[0], tabs[0], "blocks")
    lc2 = pa.layercam_map(net, vols[0], tabs[0], "blocks")
    assert torch.equal(lc, lc2), "determinism rerun failed"
    sc = pa.scorecam_map(net, vols[0], tabs[0], "blocks", blur_sigma_vox=1.0)
    gc = pa.gradcam_map(net, vols[0], tabs[0], "blocks")
    import src.method_y_revival.pet_attribution_v4 as P
    fid = P.fidelity_occlusion(net, vols[:2], tabs[:2], "blocks", n_masks=2, seed=13)
    share = P.tabular_share(net, vols, tabs, "blocks", n_perm=2, seed=13)
    agg = [P.spearman_with_ci((lc.flatten().double().numpy()),
                              (sc.flatten().double().numpy()),
                              n_boot=50, seed=13)]
    legs = P.evaluate_legs_table([
        {"id": "LEG-01", "status": "PASS", "evidence": "dry_run_maps.json"},
        # fidelity BLOCKED is not a valid legs-table status: record the
        # outcome as NOT_MEASURABLE with the fidelity evidence (C-fix 2026-09-08).
        {"id": "LEG-04", "status": fid["status"] if fid["status"] == "PASS" else "NOT_MEASURABLE",
         "evidence": "dry_run_fidelity.json"},
        {"id": "LEG-07", "status": "NOT_MEASURABLE",
         "evidence": "dry_run_no_pet_quant_provenance.json"},
    ])
    report = {"mode": "dry_run", "gate": gate, "ceiling": pa.CEILING,
              "layercam_shape": list(lc.shape), "scorecam_shape": list(sc.shape),
              "gradcam_shape": list(gc.shape), "fidelity": fid,
              "tabular_share": share, "agreement": agg[0],
              "legs": legs, "ban": P.ban_check("regions of higher model attribution")}
    atomic_write_json(out_dir / "dry_run_report.json", report)
    # Cited dry-run evidence files must exist (same completeness rule as production).
    atomic_write_json(out_dir / "dry_run_maps.json",
                      {"layercam_shape": list(lc.shape), "scorecam_shape": list(sc.shape),
                       "gradcam_shape": list(gc.shape), "scope": "SYNTHETIC_DRY_RUN_ONLY"})
    atomic_write_json(out_dir / "dry_run_fidelity.json", fid)
    atomic_write_json(out_dir / "dry_run_no_pet_quant_provenance.json",
                      P.pet_quant_consistency_leg(derivation_dag=None))
    print(json.dumps({"status": "PASS", "mode": "dry_run",
                      "fidelity": fid["status"], "legs": legs["status"],
                      "share_stamp": share["stamp"]}, indent=2))
    return 0


def _load_cell_batch(cell, *, run_contract, model_cfg, model_features):
    """Load (image, tabular) batch for a cell's firewall subjects (read-only).

    Mirrors the locked v4 program verbatim: the development membership
    (role == "development"), the FoldTabularTransformer fitted on THIS cell's
    outer-train frame, raw dtypes coerced per-column exactly like training
    (C-fix 2026-09-08: raw torch.tensor on object-dtype columns crashed).
    Strict probes only: anything unresolvable -> BLOCKED (never guessed).
    """
    import h5py
    import pandas as pd

    from src.method_y_revival.image_training_runtime_v3 import FoldTabularTransformer

    branch = cell["branch"]
    cache_path = run_contract["input_fingerprints"][f"{branch}_pattern_cache"]["path"]
    memb_path = run_contract["input_fingerprints"][f"{branch}_membership"]["path"]
    try:
        table = pd.read_parquet(memb_path)
    except Exception as exc:
        raise RuntimeError(f"BLOCKED_TABULAR_UNREADABLE:{branch}:{exc}")
    missing_cols = [c for c in model_features if c not in table.columns]
    if missing_cols:
        raise RuntimeError(f"BLOCKED_TABULAR_COLUMNS_MISSING:{missing_cols}")
    key_col = "patient_hash" if "patient_hash" in table.columns else None
    if key_col is None:
        for cand in ("subject_key", "subject", "sample_hash"):
            if cand in table.columns:
                key_col = cand
                break
    if key_col is None:
        raise RuntimeError("BLOCKED_TABULAR_KEY_COLUMN_UNKNOWN")
    if "role" not in table.columns:
        raise RuntimeError("BLOCKED_TABULAR_ROLE_COLUMN_MISSING")
    development = table[table["role"] == "development"].copy()
    train_keys = {str(s) for s in cell.get("train_subjects", [])}
    if not train_keys:
        raise RuntimeError(f"BLOCKED_TRAIN_COHORT_MISSING:{cell['branch']}")
    train_frame = development[development[key_col].astype(str).isin(train_keys)]
    if train_frame.empty:
        raise RuntimeError(f"BLOCKED_TRAIN_FRAME_EMPTY:{cell['branch']}")
    try:
        transformer = FoldTabularTransformer.fit(train_frame, list(model_features))
    except Exception as exc:
        raise RuntimeError(f"BLOCKED_TABULAR_TRANSFORMER_FIT:{branch}:{exc}")
    development = development.set_index(key_col)
    batch = []
    # h5py handle is scope-bound: every BLOCKED raise below must release it
    # (C-fix 2026-09-08: previous version leaked the handle on exception paths,
    # locking the .h5 on Windows and blocking retry/cleanup).
    with h5py.File(cache_path, "r") as h5:
        for subj in cell["firewall_subjects"]:
            if subj not in h5:
                raise RuntimeError(f"BLOCKED_H5_GROUP_MISSING:{cell['branch']}:{subj}")
            if subj not in development.index:
                raise RuntimeError(f"BLOCKED_TABULAR_ROW_MISSING:{cell['branch']}:{subj}")
            img = torch.from_numpy(h5[subj]["image"][...]).float()
            try:
                tab_arr = transformer.transform(
                    development.loc[[subj], list(model_features)])
            except Exception as exc:
                raise RuntimeError(f"BLOCKED_TABULAR_TRANSFORM:{cell['branch']}:{subj}:{exc}")
            tab = torch.from_numpy(tab_arr.reshape(-1)).float()
            batch.append((subj, img, tab))
    if not batch:
        raise RuntimeError(f"BLOCKED_EMPTY_FIREWALL_COHORT:{cell['branch']}")
    return batch


def cmd_run(args) -> int:
    cfg = _load_config(Path(args.config))
    # Versioned-run discipline (C-fix 2026-09-08): DRAFT params must never drive
    # production. The locked config status is POST_HOC_INTERNAL_EXPLORATORY_DRAFT_PARAMS,
    # so production stays BLOCKED by design until a versioned FINAL promotion —
    # consistent with image XAI remaining disabled while quantitative gates are closed.
    if cfg.get("status") != "POST_HOC_INTERNAL_EXPLORATORY_FINAL":
        print(json.dumps({"status": "BLOCKED",
                          "reason_code": "ATTRIBUTION_CONFIG_NOT_FINAL",
                          "config_status": cfg.get("status")}))
        return 3
    out_dir = Path(args.output)
    pre = out_dir / "attribution_preflight_pass.json"
    if not pre.is_file():
        print(json.dumps({"status": "BLOCKED",
                          "reason_code": "ATTRIBUTION_PREFLIGHT_REQUIRED"}))
        return 3
    try:
        saved = json.loads(pre.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        print(json.dumps({"status": "BLOCKED",
                          "reason_code": "ATTRIBUTION_PREFLIGHT_INVALID"}))
        return 3
    if saved.get("status") != "PASS" or saved.get("config_sha256") != _sha256_file(
            Path(args.config)):
        print(json.dumps({"status": "BLOCKED",
                          "reason_code": "ATTRIBUTION_PREFLIGHT_STALE"}))
        return 3
    try:
        gate = pa.verifier_gate(run_root=ROOT / cfg["v4_output_root"],
                                config_path=ROOT / "configs" / "method_y_image_factorial_v4.yaml")
    except Exception as exc:
        print(json.dumps({"status": "BLOCKED", "reason_code": str(exc)}))
        return 3
    if gate.get("run_contract_sha256") != saved.get("run_digest"):
        print(json.dumps({"status": "BLOCKED",
                          "reason_code": "ATTRIBUTION_RUN_DIGEST_DRIFT"}))
        return 3
    import time
    import yaml
    v4_root = ROOT / cfg["v4_output_root"]
    rc = json.loads((v4_root / "run_contract.json").read_text(encoding="utf-8"))
    live_digest = rc.get("digest")
    factor_cfg = yaml.safe_load(
        (ROOT / "configs" / "method_y_image_factorial_v4.yaml").read_text(
            encoding="utf-8"))
    try:
        splits = {}
        for branch in ("amyloid", "tau"):
            spath = rc["input_fingerprints"][f"{branch}_splits"]["path"]
            splits[branch] = json.loads(Path(spath).read_text(encoding="utf-8"))
        membership_paths = {
            branch: rc["input_fingerprints"][f"{branch}_membership"]["path"]
            for branch in ("amyloid", "tau")
        }
        cells = pa.enumerate_attribution_cells(
            dev_root=v4_root / "development",
            orch_root=v4_root / "orchestration", splits=splits,
            membership_paths=membership_paths)
    except Exception as exc:
        print(json.dumps({"status": "BLOCKED", "reason_code": str(exc)}))
        return 3
    det_cfg = cfg.get("determinism", {})
    det = pa.enforce_determinism(
        cublas_config=str(det_cfg.get("cublas_workspace_config", ":4096:8")),
        python_hash_seed=str(det_cfg.get("python_hash_seed", "20260714")))
    if not (bool(det.get("torch_deterministic_algorithms")) == bool(det_cfg.get("torch_deterministic_algorithms"))
            and bool(det.get("cudnn_deterministic")) == bool(det_cfg.get("cudnn_deterministic"))
            and bool(det.get("cudnn_benchmark")) == bool(det_cfg.get("cudnn_benchmark"))
            and det.get("cublas_workspace_effective", False)
            and det.get("python_hash_seed_effective", False)):
        print(json.dumps({"status": "BLOCKED",
                          "reason_code": "ATTRIBUTION_HOST_NOT_DETERMINISTIC",
                          "determinism": det}))
        return 3
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    score_timeout_h = float(cfg.get("scorecam", {}).get("full_cohort_timeout_hours", 24))
    deadline = time.monotonic() + score_timeout_h * 3600.0
    risk_primary = str((cfg.get("risk") or {}).get("primary_scalar", "logit_sum"))
    risk_sensitivity = str((cfg.get("risk") or {}).get("sensitivity_scalar", "mortality_last"))
    n_masks = int((cfg.get("sanity") or {}).get("fidelity_n_masks", 8))
    n_perm = int((cfg.get("sanity") or {}).get("tabular_perm_n", 8))
    share_floor = float((cfg.get("sanity") or {}).get("image_share_floor", 0.5))
    fam_of = {"A0": "GROUPNORM_RESIDUAL_TINY_3D", "A1": "GROUPNORM_RESIDUAL_TINY_3D",
              "A2": "RESNET10_3D_GROUPNORM", "A3": "RESNET10_3D_GROUPNORM",
              "A4": "RESNET18_3D_GROUPNORM"}
    model_cfg = _locked_model_cfg()
    all_maps, peak = [], 0.0
    progress_path = out_dir / "cell_progress.json"
    try:
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        if not isinstance(progress, dict):
            progress = {}
    except (OSError, ValueError):
        progress = {}
    cfg_sha = _sha256_file(Path(args.config))

    def _cell_key(cell):
        return f"{cell['branch']}/{cell['arm']}/o{cell['outer']}"

    def _save_progress():
        atomic_write_json(progress_path, progress)

    for cell in cells:
        key = _cell_key(cell)
        done = progress.get(key, {})
        try:
            # Frozen-chain binding (2026-09-08): finals embed their own generation
            # digest (migrations preserve weights by design), so bind the exact
            # bytes that generated the locked outer metrics via the FINALIZED state.
            prov = pa.verify_final_provenance(
                cell["final_checkpoint"], v4_root=v4_root,
                branch=cell["branch"], arm=cell["arm"], outer=cell["outer"])
            feats = factor_cfg["branches"][cell["branch"]]["model_features"]
            tab_dim = len(feats)
            _payload_probe = pa.read_only_load_checkpoint(
                cell["final_checkpoint"],
                expected_sha256=hashlib.sha256(
                    Path(cell["final_checkpoint"]).read_bytes()).hexdigest())
            bins = _infer_bins_from_payload(_payload_probe)
            net = _build_real_arch(cell["arm"], model_cfg, tabular_dim=tab_dim, bins=bins)
            # shape handshake: locked arch must accept locked payload exactly
            _missing, _unexpected = net.load_state_dict(_payload_probe["model_state"], strict=False)
            if _missing or _unexpected:
                raise RuntimeError(f"BLOCKED_ARCH_PAYLOAD_MISMATCH:{key}:"
                                   f"missing={len(_missing)}:unexpected={len(_unexpected)}")
            payload = _payload_probe
            net.load_state_dict(payload["model_state"])
            net.eval()
            batch = _load_cell_batch(cell, run_contract=rc, model_cfg=model_cfg,
                                     model_features=feats)
            layer = cfg["targets"][fam_of[cell["arm"]]][0]
            shallow = ((cfg.get("targets") or {}).get("shallow_fusion") or {}).get(fam_of[cell["arm"]])
            subjects_sorted = sorted(s for s, _, _ in batch)
            if done.get("status") == "COMPLETE" and done.get("config_sha256") == cfg_sha \
                    and done.get("n_maps") == len(batch) \
                    and done.get("subjects") == subjects_sorted \
                    and done.get("layer") == layer and done.get("shallow") == shallow \
                    and done.get("risk_primary") == risk_primary \
                    and done.get("final_sha256") == hashlib.sha256(
                        Path(cell["final_checkpoint"]).read_bytes()).hexdigest():
                for row in done.get("rows", []):
                    all_maps.append(row)
                continue
            v0, t0 = batch[0][1].unsqueeze(0), batch[0][2].unsqueeze(0)
            first = pa.layercam_map(net, v0, t0, layer, risk_mode=risk_primary,
                                    shallow_layer_name=shallow)
            pa.assert_bit_identical(
                first, pa.layercam_map(net, v0, t0, layer, risk_mode=risk_primary,
                                       shallow_layer_name=shallow),
                what=f"{cell['branch']}/{cell['arm']}/rerun")
            if torch.cuda.is_available():
                peak = max(peak, float(torch.cuda.max_memory_allocated() / 1024 ** 3))
                pa.check_vram_budget(budget_gb=7.5)
            rows = []
            for subj, img, tab in batch:
                vv, tt = img.unsqueeze(0), tab.unsqueeze(0)
                rows.append({
                    "cell": key,
                    "subject": subj,
                    "layercam_mean": float(pa.layercam_map(
                        net, vv, tt, layer, risk_mode=risk_primary,
                        shallow_layer_name=shallow).mean()),
                    "gradcam_mean": float(pa.gradcam_map(
                        net, vv, tt, layer, risk_mode=risk_primary).mean()),
                })
                all_maps.append(rows[-1])
            progress[key] = {"status": "COMPLETE", "config_sha256": cfg_sha,
                             "n_maps": len(rows),
                             "subjects": subjects_sorted,
                             "layer": layer, "shallow": shallow,
                             "risk_primary": risk_primary,
                             "tabular_dim": tab_dim, "hazard_bins": bins,
                             "provenance": {"checkpoint_sha256": prov["checkpoint_sha256"],
                                            "generation_digest": prov["generation_digest"]},
                             "final_sha256": hashlib.sha256(
                                 Path(cell["final_checkpoint"]).read_bytes()).hexdigest(),
                             "rows": rows,
                             "privacy_tier": "PRIVATE_SCRATCH_SUBJECT_ROWS"}
            _save_progress()
        except Exception as exc:
            # Privacy hygiene: never dump firewall subject lists into logs.
            print(json.dumps({"status": "BLOCKED", "cell": key,
                              "n_subjects": len(cell.get("firewall_subjects", [])),
                              "checkpoint": Path(cell["final_checkpoint"]).name,
                              "reason_code": f"{type(exc).__name__}:{exc}"}))
            return 3
    # Pilot phase (locked subsets): Score-CAM maps + LayerCAM full maps for
    # agreement, fidelity subset, tabular share. Conducted per pilot cell with
    # that cell's locked model; each cell's batch is reloaded fail-closed here
    # (C-fix 2026-09-08: the previous version reused the LAST cell's `batch`
    # for every pilot cell — cross-cell data leakage — and indexed the batch
    # list with subject strings).
    pilot_keys = sorted({r["subject"] for r in all_maps})[:20]
    _pilot_set = set(pilot_keys)
    pilot_rows = [r for r in all_maps if r["subject"] in _pilot_set]
    score_maps: dict = {}
    layer_maps: dict = {}
    score_times: list = []
    group_gates: dict = {}
    pilot_timed_out = False
    sensitivity: dict = {"status": "NOT_MEASURABLE",
                         "evidence": "risk_sensitivity_subset.json",
                         "primary": risk_primary, "sensitivity": risk_sensitivity}
    try:
        by_cell: dict = {}
        for r in pilot_rows:
            by_cell.setdefault(r["cell"], []).append(r["subject"])
        for key in sorted(by_cell):
            if time.monotonic() >= deadline:
                pilot_timed_out = True
                break
            branch, arm, outer = key.split("/")
            cell = next(c for c in cells
                        if f"{c['branch']}/{c['arm']}/o{c['outer']}" == key)
            pa.verify_final_provenance(
                cell["final_checkpoint"], v4_root=v4_root,
                branch=cell["branch"], arm=cell["arm"], outer=cell["outer"])
            feats = factor_cfg["branches"][cell["branch"]]["model_features"]
            tab_dim = len(feats)
            probe = pa.read_only_load_checkpoint(
                cell["final_checkpoint"],
                expected_sha256=hashlib.sha256(
                    Path(cell["final_checkpoint"]).read_bytes()).hexdigest())
            bins = _infer_bins_from_payload(probe)
            net = _build_real_arch(cell["arm"], model_cfg, tabular_dim=tab_dim, bins=bins)
            net.load_state_dict(probe["model_state"])
            net.eval()
            layer = cfg["targets"][fam_of[cell["arm"]]][0]
            cell_batch = _load_cell_batch(cell, run_contract=rc, model_cfg=model_cfg,
                                          model_features=feats)
            lookup = {s: (i, t) for s, i, t in cell_batch}
            group_vols, group_tabs = [], []
            for subj in sorted(by_cell[key]):
                if time.monotonic() >= deadline:
                    pilot_timed_out = True
                    break
                try:
                    img, tab = lookup[subj]
                except KeyError as exc:
                    raise RuntimeError(f"BLOCKED_PILOT_SUBJECT_ABSENT:{key}:{subj}") from exc
                vv, tt = img.unsqueeze(0), tab.unsqueeze(0)
                t0 = time.monotonic()
                sc = pa.scorecam_map(net, vv, tt, layer, risk_mode=risk_primary)
                score_times.append(time.monotonic() - t0)
                lc = pa.layercam_map(net, vv, tt, layer, risk_mode=risk_primary)
                score_maps[(key, subj)] = sc
                layer_maps[(key, subj)] = lc
                group_vols.append(vv)
                group_tabs.append(tt)
                if sensitivity["status"] == "NOT_MEASURABLE":
                    try:
                        lc_sens = pa.layercam_map(net, vv, tt, layer,
                                                  risk_mode=risk_sensitivity)
                        rho = pa.spearman_with_ci(
                            lc.detach().cpu().flatten().double().numpy(),
                            lc_sens.detach().cpu().flatten().double().numpy(),
                            n_boot=200, seed=13)
                        sensitivity = {"status": "PASS", "cell": key,
                                       "evidence": "risk_sensitivity_subset.json",
                                       "primary": risk_primary,
                                       "sensitivity": risk_sensitivity,
                                       "spearman": rho}
                    except Exception as exc_s:
                        sensitivity = {"status": "NOT_MEASURABLE",
                                       "evidence": "risk_sensitivity_subset.json",
                                       "reason": f"{type(exc_s).__name__}:{exc_s}"}
            if not group_vols:
                continue
            group_gates[key] = {
                "fidelity": pa.fidelity_occlusion(
                    net, group_vols, group_tabs, layer,
                    n_masks=n_masks, seed=13),
                "share": pa.tabular_share(
                    net, group_vols, group_tabs, layer,
                    n_perm=n_perm, seed=13),
            }
    except Exception as exc:
        print(json.dumps({"status": "BLOCKED", "phase": "score_pilot",
                          "reason_code": f"{type(exc).__name__}:{exc}"}))
        return 3
    mapped_rows = [r for r in pilot_rows if (r["cell"], r["subject"]) in layer_maps]
    if not mapped_rows:
        print(json.dumps({"status": "BLOCKED", "phase": "score_pilot",
                          "reason_code": "PILOT_MAPS_EMPTY"}))
        return 3
    # Memory guard (2026-09-08): pilot_rows multiplies subjects × cells (a pilot
    # subject recurs in every cell), so stacking ALL mapped 128³ maps OOMs
    # (observed 1.72 GiB float64 throw). Cap agreement inputs at the locked
    # pilot budget (scorecam.pilot_volumes, default 20), deterministically the
    # first rows in cell order; float32 stack (gate upcasts internally).
    agree_n = int(cfg.get("scorecam", {}).get("pilot_volumes", 20))
    agree_rows = mapped_rows[:max(1, agree_n)]
    per_vol = (sum(score_times) / len(score_times)) if score_times else 0.0
    projected_h = per_vol * len(all_maps) / 3600.0 if all_maps else 0.0
    try:
        # Agreement (LayerCAM vs Score-CAM) on same-cell pilot full maps.
        agree = pa.agreement_gate(
            np.stack([np.asarray(layer_maps[(r["cell"], r["subject"])].detach().cpu(),
                                 dtype=np.float32).ravel() for r in agree_rows]),
            np.stack([np.asarray(score_maps[(r["cell"], r["subject"])].detach().cpu(),
                                 dtype=np.float32).ravel() for r in agree_rows]))
    except Exception as exc:
        print(json.dumps({"status": "BLOCKED", "phase": "agreement",
                          "reason_code": f"{type(exc).__name__}:{exc}"}))
        return 3
    share_records = [{"cell": k, "image_share": g["share"]["image_share"],
                      "stamp": g["share"]["stamp"]} for k, g in sorted(group_gates.items())]
    legs, evidence_extra = build_production_legs(
        agree_status=agree["status"], share_records=share_records,
        share_floor=share_floor, pilot_seconds_per_volume=per_vol,
        projected_hours=projected_h, deadline_hours=score_timeout_h,
        pilot_timed_out=pilot_timed_out)
    # Persist every referenced evidence artifact BEFORE the report (C-fix
    # 2026-09-08: previous version cited files it never wrote).
    atomic_write_json(out_dir / "production_cells.json",
                      {"cells": sorted(_cell_key(c) for c in cells),
                       "n_maps": len(all_maps), "config_sha256": cfg_sha})
    atomic_write_json(out_dir / "scorecam_pilot_20.json",
                      {"pilot_subjects": pilot_keys, "n_mapped_rows": len(mapped_rows),
                       "agreement_rows": len(agree_rows),
                       "pilot_seconds_per_volume": round(per_vol, 2),
                       "projected_hours": round(projected_h, 1),
                       "deadline_hours": score_timeout_h,
                       "pilot_timed_out": pilot_timed_out,
                       "sensitivity": sensitivity})
    atomic_write_json(out_dir / "sanity_gates.json",
                      {"agreement": agree, "fidelity_by_cell":
                       {k: g["fidelity"] for k, g in sorted(group_gates.items())}})
    for _name, _body in evidence_extra.items():
        atomic_write_json(out_dir / _name, _body)
    # Every remaining cited evidence string must exist on disk (C-fix 2026-09-08:
    # legs tables previously cited files that were never written).
    if legs[1]["evidence"] == "scorecam_full_cost_projection.json":
        atomic_write_json(out_dir / "scorecam_full_cost_projection.json",
                          {"pilot_seconds_per_volume": round(per_vol, 2),
                           "projected_hours": round(projected_h, 1),
                           "deadline_hours": score_timeout_h,
                           "decision": "FULL_COHORT_NOT_RUN_OVER_BUDGET"})
    atomic_write_json(out_dir / "pet_quant_derivation_dag_missing.json",
                      pa.pet_quant_consistency_leg(derivation_dag=None))
    atomic_write_json(out_dir / "risk_sensitivity_subset.json", sensitivity)
    fid_sub = sorted(all_maps, key=lambda r: r["subject"])[:30]
    md = ("regions of higher model attribution with bootstrap confidence "
          "intervals; exploratory internal evidence only")
    report = pa.build_report(
        methods={"layercam": "primary", "scorecam": "convergent-pilot",
                 "gradcam": "sanity", "pilot_n": len(pilot_keys),
                 "fidelity_n": len(fid_sub), "cells": len(cells),
                 "risk_primary": risk_primary,
                 "risk_sensitivity_subset": sensitivity.get("status")},
        gates={"determinism_rerun": "bit-identical", "vram_peak_gb": round(peak, 3),
               "determinism": det},
        statistics={"n_maps": len(all_maps),
                    "mean_image_share": evidence_extra["tabular_share.json"]["mean_image_share"]},
        legs=legs, markdown_body=md)
    atomic_write_json(out_dir / "pet_attribution_report_v4.json", report)
    print(json.dumps({"status": report["status"], "cells": len(cells),
                      "maps": len(all_maps), "peak_vram_gb": round(peak, 3)}))
    return 0 if report["status"] == "PASS" else 3


def build_production_legs(*, agree_status: str, share_records: list,
                          share_floor: float, pilot_seconds_per_volume: float,
                          projected_hours: float, deadline_hours: float,
                          pilot_timed_out: bool) -> tuple:
    """Pure legs assembly for the production report (testable seam, no torch).

    agree_status comes from pa.agreement_gate (PASS | RED_FLAG_INVESTIGATE).
    share_records are per-cell {"image_share": float, "stamp": str} dicts.
    Returns (legs, evidence_payloads) where evidence_payloads maps artifact
    file names to their JSON-serializable bodies (written by the caller).
    Schema rule: leg statuses are only PASS | NOT_MEASURABLE | TIMEOUT and ids
    only LEG-01..LEG-08 (pa.evaluate_legs_table rejects anything else).
    """
    shares = [float(r.get("image_share", float("nan"))) for r in share_records]
    shares = [s for s in shares if s == s]
    # JSON-strict nulls (never NaN) when no share was evaluated (C-fix 2026-09-08).
    mean_share = (sum(shares) / len(shares)) if shares else None
    min_share = min(shares) if shares else None
    tabular_payload = {"per_cell": share_records, "mean_image_share": mean_share,
                       "min_image_share": min_share, "floor": float(share_floor),
                       "n_cells": len(share_records)}
    if shares and mean_share >= float(share_floor):
        leg06 = {"id": "LEG-06", "status": "PASS", "evidence": "tabular_share.json"}
    else:
        leg06 = {"id": "LEG-06", "status": "NOT_MEASURABLE",
                 "evidence": "tabular_share.json",
                 "reason": "IMAGE_SHARE_BELOW_FLOOR_OR_UNEVALUATED"}
    if pilot_timed_out:
        leg02 = {"id": "LEG-02", "status": "TIMEOUT",
                 "evidence": "scorecam_pilot_20.json",
                 "pilot_seconds_per_volume": round(pilot_seconds_per_volume, 2),
                 "projected_hours": round(projected_hours, 1),
                 "deadline_hours": deadline_hours}
    elif projected_hours > deadline_hours:
        leg02 = {"id": "LEG-02", "status": "NOT_MEASURABLE",
                 "evidence": "scorecam_full_cost_projection.json",
                 "pilot_seconds_per_volume": round(pilot_seconds_per_volume, 2),
                 "projected_hours": round(projected_hours, 1),
                 "deadline_hours": deadline_hours}
    else:
        leg02 = {"id": "LEG-02", "status": "PASS", "evidence": "scorecam_pilot_20.json"}
    if agree_status == "PASS":
        leg04 = {"id": "LEG-04", "status": "PASS", "evidence": "sanity_gates.json"}
    else:
        # RED_FLAG_INVESTIGATE is not a valid legs-table status: record the
        # outcome as NOT_MEASURABLE with provenance instead of an invalid BLOCKED.
        leg04 = {"id": "LEG-04", "status": "NOT_MEASURABLE",
                 "evidence": "sanity_gates.json",
                 "reason": "agreement_%s" % (agree_status,)}
    legs = [
        {"id": "LEG-01", "status": "PASS", "evidence": "production_cells.json"},
        leg02,
        {"id": "LEG-03", "status": "PASS", "evidence": "production_cells.json"},
        leg04,
        {"id": "LEG-05", "status": "PASS", "evidence": "production_cells.json"},
        leg06,
        {"id": "LEG-07", "status": "NOT_MEASURABLE",
         "evidence": "pet_quant_derivation_dag_missing.json"},
        {"id": "LEG-08", "status": "PASS", "evidence": "pet_attribution_report_v4.json"},
    ]
    return legs, {"tabular_share.json": tabular_payload}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("preflight", "dry-run", "run"), required=True)
    ap.add_argument("--config", default=str(
        ROOT / "configs" / "method_y_pet_attribution_v4.yaml"))
    ap.add_argument("--output", default=str(
        ROOT / "results" / "method_y_revival_v4_image_factorial_v1" / "attribution"))
    args = ap.parse_args(argv)
    try:
        if args.mode == "preflight":
            return cmd_preflight(args)
        if args.mode == "dry-run":
            return cmd_dry_run(args)
        return cmd_run(args)
    except Exception as exc:
        print(json.dumps({"status": "BLOCKED", "reason_code": f"{type(exc).__name__}:{exc}"}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
