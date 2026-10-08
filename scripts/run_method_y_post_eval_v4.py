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
from src.method_y_revival.strict_yaml import load_unique_yaml  # noqa: E402


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_config(path: Path) -> dict:
    value = load_unique_yaml(path)
    if not isinstance(value, dict):
        raise ValueError("POST_EVAL_CONFIG_MUST_BE_MAPPING")
    return value


def cmd_preflight(args) -> int:
    cfg = _load_config(Path(args.config))
    v4_root = ROOT / cfg["v4_output_root"]
    out: dict = {"mode": "preflight", "status": "BLOCKED", "checks": {}}
    try:
        gate = pe.entry_gate(run_root=v4_root,
                             config_path=ROOT / "configs" / "method_y_image_factorial_v4.yaml")
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
    # Full-program single execution on a synthetic locked-OOF bundle: two
    # branches x two arms x two folds, per-partition frozen train IPCW.
    rng = np.random.RandomState(seed)
    rows, train_by_partition = [], {}
    for branch in ("amyloid", "tau"):
        for outer in (0, 1):
            tt = rng.exponential(800.0, size=100)
            cc = rng.exponential(1200.0, size=100)
            train_by_partition[(branch, outer)] = (
                np.minimum(tt, cc), tt <= cc)
            for i in range(30):
                t = float(rng.exponential(800.0))
                c = float(rng.exponential(1200.0))
                ev = bool(t <= c)
                to = float(min(t, c))
                base = 1.0 - to / 2000.0
                for arm, shift in (("A0", 0.0), ("A1", 0.04)):
                    risk = float(np.clip(base + shift, 0.01, 0.99))
                    rows.append({
                        "patient_hash": f"p{branch}{outer}_{i}", "arm": arm,
                        "branch": branch, "outer_fold": outer,
                        "split_digest": "d" * 64, "time_days": to,
                        "event": ev, "risk": risk,
                        "surv_at_horizons": {"547.875": 1.0 - risk * 0.4,
                                             "1095.75": 1.0 - risk * 0.7,
                                             "1640.0": 1.0 - risk * 0.9}})
    sup = pe.support_truncate(rows, cfg["horizons_days"])
    bundle = pe.evaluate_oof_bundle(
        rows, train_by_partition=train_by_partition,
        horizons=sup["horizons"], tau=cfg["primary_horizon_days"],
        B=100, seed=seed)
    fw = pe.firewall_assertion([{"type": "outer_result_read",
                                 "t_view": 30, "t_lock": 20}])
    fz = pe.assert_fusion_keys(rows)
    legs = pe.run_legs([
        {"id": "LEG-EVAL-00", "status": "PASS" if fw["status"] == "PASS" else "BLOCKED",
         "reason": fw.get("reason", "")},
        {"id": "LEG-EVAL-02", "status": "PASS"},
        {"id": "LEG-EVAL-05", "status": "PASS"},
        *sup["legs"],
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
              "bundle_arms": bundle["arms"],
              "bundle_contrasts": len(bundle["contrasts"]),
              "firewall": fw["status"], "fusion": fz["status"],
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
        gate = pe.entry_gate(run_root=ROOT / cfg["v4_output_root"],
                             config_path=ROOT / "configs" / "method_y_image_factorial_v4.yaml")
    except Exception as exc:
        print(json.dumps({"status": "BLOCKED", "reason_code": str(exc)}))
        return 3
    if gate.get("run_contract_sha256") != saved.get("run_digest"):
        print(json.dumps({"status": "BLOCKED",
                          "reason_code": "EVAL_RUN_DIGEST_DRIFT"}))
        return 3
    v4_root = ROOT / cfg["v4_output_root"]
    legs = []
    try:
        counts = _read_access_counts(v4_root)
        fw = pe.assert_single_access(counts)
        legs.append({"id": "LEG-EVAL-00", "status": "PASS",
                     "evidence": f"firewall_{fw['cells']}_cells_single_access.json"})
    except Exception as exc:
        legs.append({"id": "LEG-EVAL-00", "status": "BLOCKED",
                     "reason": f"{type(exc).__name__}:{exc}"})
    try:
        compiled = pe.compile_locked_reports(root=v4_root)
        legs.append({"id": "LEG-EVAL-02", "status": "PASS",
                     "evidence": "locked_public_reports.json"})
        legs.append({"id": "LEG-EVAL-05", "status": "PASS",
                     "evidence": "locked_pair_reports.json"})
    except Exception as exc:
        legs.append({"id": "LEG-EVAL-02", "status": "BLOCKED",
                     "reason": f"{type(exc).__name__}:{exc}"})
        compiled = None
    legs.append({"id": "LEG-EVAL-03", "status": "NOT_MEASURABLE",
                 "evidence": "no_subject_rows_calibration_absent.json"})
    legs.append({"id": "LEG-EVAL-12", "status": "NOT_MEASURABLE",
                 "evidence": "no_external_intake_registry.json"})
    verdict = pe.run_legs(legs)
    if compiled is not None:
        thesis_recs = _thesis_records_from_compiled(compiled)
    else:
        thesis_recs = []
    md = pe.thesis_write(
        thesis_recs, ceiling=cfg["claim_policy"]["ceiling"], section="results",
        not_measurable=[leg["id"] for leg in legs if leg["status"] == "NOT_MEASURABLE"])
    body_lines = [md.get("markdown", "")] if md.get("markdown") else []
    body_text = "\n".join(body_lines)
    priv = pe.scan_text(body_text) if body_text else {"status": "PASS", "hits": []}
    vocab = pe.ban_check(body_text) if body_text else {"status": "PASS", "hits": []}
    if priv["status"] != "PASS" or vocab["status"] != "PASS":
        priv = {"status": "BLOCKED",
                "hits": list(priv.get("hits", [])) + list(vocab.get("hits", []))}
    # ``run_legs`` is intentionally execution-oriented and permits deferred
    # legs.  Compile a separate claim gate so a technical PASS cannot be read
    # as proof that calibration or external validation was completed.
    science = pe.science_gate_from_legs(legs, external_pending=True)
    bundle_files = ["reports/factorial_benchmark_summary.sanitized.json"]
    report = {"mode": "run", "status": verdict["status"],
              "artifact_status": "COMPLETED" if verdict["status"] == "PASS" else "FAILED",
              "science_gate_status": science["science_gate_status"],
              "claim_status": science["claim_status"],
              "claim_blockers": science["blockers"],
              "positive_claim_allowed": science["positive_claim_allowed"],
              "external_claim_allowed": science["external_claim_allowed"],
              "ceiling": cfg["claim_policy"]["ceiling"],
              "run_digest": gate.get("run_contract_sha256"),
              "legs": legs, "compiled": compiled is not None,
              "contrasts": compiled["contrasts"] if compiled else [],
              "omnibus": compiled.get("omnibus") if compiled else None,
              "thesis": md, "privacy": priv, "bundle_files": bundle_files}
    atomic_write_json(out_dir / "post_eval_report.json", report)
    atomic_write_json(out_dir / "thesis_results.md",
                      {"markdown": md.get("markdown", ""),
                       "pointers": md.get("pointers", []),
                       "status": md.get("status", "BLOCKED"),
                       "ceiling": cfg["claim_policy"]["ceiling"]})
    print(json.dumps({"status": report["status"], "legs": legs,
                      "thesis": md["status"], "privacy": priv["status"]}))
    return 0 if verdict["status"] == "PASS" and md["status"] == "PASS" else 3


def _read_access_counts(v4_root: Path) -> dict[str, int]:
    counts = {}
    base = v4_root / "private" / "outer_evaluation"
    rows = sorted(base.rglob("outer_access_journal.json"))
    if not rows:
        raise RuntimeError("BLOCKED:NO_ACCESS_JOURNALS_FOUND")
    for journal in rows:
        try:
            payload = json.loads(journal.read_text(encoding="utf-8"))
            counts[str(journal.parent.parent.parent.name) + "/" + str(
                journal.parent.parent.name) + "/" + str(
                journal.parent.name)] = int(payload.get("access_count", -1))
        except (OSError, ValueError, TypeError) as exc:
            raise RuntimeError(f"BLOCKED:JOURNAL_UNREADABLE:{journal.name}") from exc
    pe.assert_single_access(counts)
    return counts


def _thesis_records_from_compiled(compiled: dict) -> list[dict]:
    recs = []
    for c in compiled.get("contrasts", []):
        pair = str(c.get("pair", "?")).replace(":", " ").replace("_vs_", " ").replace("_", " ")
        dci = c.get("delta_ibs_ci", [0.0, 0.0])
        ni = (c.get("ni_ibs") or {}).get("decision", "INCONCLUSIVE_UNDERPOWERED")
        recs.append({
            "pointer": f"$.contrasts[{c.get('pair', '?')}]",
            "text": (f"pair {pair} mean paired dibs {float(sum(dci)) / 2:.4f} "
                     f"internal exploratory {ni}"),
            "tokens": ["internal", "exploratory"],
        })
    omni = compiled.get("omnibus") or {}
    if omni.get("p") is not None:
        recs.append({
            "pointer": "$.omnibus",
            "text": (f"omnibus arms internal exploratory "
                     f"p {float(omni['p']):.3f}"),
            "tokens": ["internal", "exploratory"],
        })
    return recs


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("preflight", "dry-run", "run"), required=True)
    ap.add_argument("--config", default=str(
        ROOT / "configs" / "method_y_post_eval_v4_1.yaml"))
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
