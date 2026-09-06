"""Post-training evaluation launcher, v4 image factorial (03-combined-plan + A1).

Modes:
  preflight  read-only readiness (verifier PASS + locked finals + splits +
             horizons + config SHA). No evaluation, no writes except report.
  dry-run    end-to-end on SYNTHETIC cohort (safe anytime; proves machinery).
  run        production on locked finals. REFUSED unless a fresh preflight
             PASS artifact (same config SHA) exists. Never trains/refits.

Seams are explicit; every unresolvable input FAILS closed with a reason.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.method_y_revival import post_eval_v4 as pe  # noqa: E402
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


def cmd_preflight(args) -> int:
    cfg = _load_config(Path(args.config))
    v4_root = ROOT / cfg["v4_output_root"]
    out: dict = {"mode": "preflight", "status": "BLOCKED", "checks": {}}
    try:
        gate = pe.entry_gate(run_root=v4_root)
        out["checks"]["verifier"] = "PASS"
        digest = gate["run_contract_sha256"]
    except Exception as exc:
        out["checks"]["verifier"] = f"BLOCKED:{exc}"
        print(json.dumps(out, indent=2))
        return 3
    try:
        rc = json.loads((v4_root / "run_contract.json").read_text(encoding="utf-8"))
        out["checks"]["run_contract_readable"] = "PASS"
    except Exception as exc:
        out["checks"]["run_contract_readable"] = f"BLOCKED:{exc}"
        print(json.dumps(out, indent=2))
        return 3
    out.update({"status": "PASS", "run_digest": digest,
                "config_sha256": _sha256_file(Path(args.config)),
                "horizons": cfg["horizons_days"],
                "determinism": {"rng": "numpy-RandomState-SeedSequence",
                                "libs": pe.lib_versions()}})
    atomic_write_json(Path(args.output) / "eval_preflight_pass.json", out)
    print(json.dumps({k: v for k, v in out.items()}, indent=2))
    return 0


def _synthetic_oof(n=160, seed=13, shift=0.0):
    rng = np.random.RandomState(seed)
    t = rng.exponential(scale=800.0, size=n)
    c = rng.exponential(scale=1200.0, size=n)
    time = np.minimum(t, c)
    event = t <= c
    base = 1.0 - t / t.max()
    return time, event, np.clip(base + shift, 0.0, 1.0)


def cmd_dry_run(args) -> int:
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = _load_config(Path(args.config))
    B = 300
    seed = 13
    time_a, event_a, risk_a = _synthetic_oof(160, seed, 0.0)
    _, _, risk_b = _synthetic_oof(160, seed, 0.0)
    time_b, event_b = time_a, event_a
    folds = np.random.RandomState(seed).randint(0, 5, size=160)
    d_ibs = risk_a - risk_b
    pb = pe.paired_bootstrap_delta(d_ibs, np.zeros_like(d_ibs), n_boot=B,
                                   seed=seed, fold_id=folds)
    pval = pe.permutation_paired_p(risk_a, risk_b, n_perm=100, seed=seed)
    ni = pe.ni_retention(pb["delta"], (pb["ci_low"], pb["ci_high"]),
                         margin=cfg["ni_margins"]["dIBS"], higher_better=False)
    uno = pe.uno_c(time_a, event_a, time_a, event_a, risk_a, tau=1000.0)
    cal = pe.calibration_at_horizon(time_a, event_a,
                                    np.clip(1.0 - risk_a, 0.05, 0.95),
                                    horizon=800.0, train_time=time_a,
                                    train_event=event_a, n_bins=5)
    rt = pe.risk_table(time_a, event_a, horizons=cfg["horizons_days"])
    legs = pe.run_legs([
        {"id": "LEG-EVAL-00", "status": "PASS"},
        {"id": "LEG-EVAL-02", "status": "PASS"},
        {"id": "LEG-EVAL-05", "status": "PASS"},
        {"id": "LEG-EVAL-12", "status": "NOT_MEASURABLE",
         "evidence": "dry_run_no_external_intake.json"},
    ])
    md = pe.thesis_write(
        [{"pointer": "$.uno", "text": f"Uno-C {uno['c']:.2f} internal exploratory",
          "tokens": ["Uno-C"]}],
        ceiling=cfg["claim_policy"]["ceiling"], section="results",
        not_measurable=["LEG-EVAL-12"])
    final = {"mode": "dry_run", "status": "PASS",
             "ceiling": cfg["claim_policy"]["ceiling"],
             "paired_delta": pb, "perm_p": pval, "ni": ni, "uno": uno,
             "calibration": cal, "risk_table": rt, "legs": legs,
             "thesis": md}
    atomic_write_json(out_dir / "dry_run_report.json", final)
    print(json.dumps({"status": "PASS", "mode": "dry_run",
                      "uno_c": round(uno["c"], 3),
                      "ni_decision": ni["decision"],
                      "legs": legs["status"], "thesis": md["status"]}, indent=2))
    return 0


def cmd_run(args) -> int:
    cfg = _load_config(Path(args.config))
    out_dir = Path(args.output)
    pre = out_dir / "eval_preflight_pass.json"
    if not pre.is_file():
        print(json.dumps({"status": "BLOCKED",
                          "reason_code": "EVAL_PREFLIGHT_REQUIRED"}))
        return 3
    try:
        saved = json.loads(pre.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        print(json.dumps({"status": "BLOCKED",
                          "reason_code": "EVAL_PREFLIGHT_INVALID"}))
        return 3
    if saved.get("status") != "PASS" or saved.get("config_sha256") != _sha256_file(
            Path(args.config)):
        print(json.dumps({"status": "BLOCKED",
                          "reason_code": "EVAL_PREFLIGHT_STALE"}))
        return 3
    try:
        gate = pe.entry_gate(run_root=ROOT / cfg["v4_output_root"])
    except Exception as exc:
        print(json.dumps({"status": "BLOCKED", "reason_code": str(exc)}))
        return 3
    if gate.get("run_contract_sha256") != saved.get("run_digest"):
        print(json.dumps({"status": "BLOCKED",
                          "reason_code": "EVAL_RUN_DIGEST_DRIFT"}))
        return 3
    print(json.dumps({"status": "READY_FOR_PRODUCTION_CELLS",
                      "run_digest": gate.get("run_contract_sha256"),
                      "note": "per-cell evaluation follows 03-combined-plan legs; "
                              "no training/refit; aggregate-only"}))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("preflight", "dry-run", "run"), required=True)
    ap.add_argument("--config", default=str(
        ROOT / "configs" / "method_y_post_eval_v4.yaml"))
    ap.add_argument("--output", default=str(
        ROOT / "results" / "method_y_revival_v4_image_factorial_v1" / "post_eval"))
    args = ap.parse_args(argv)
    try:
        if args.mode == "preflight":
            return cmd_preflight(args)
        if args.mode == "dry-run":
            return cmd_dry_run(args)
        return cmd_run(args)
    except Exception as exc:
        print(json.dumps({"status": "BLOCKED",
                          "reason_code": f"{type(exc).__name__}:{exc}"}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
