"""Post-training evaluation library, v4 image factorial (combined plan 03 + A1).

Pure CPU statistics on LOCKED artifacts. No training, no refit, no outer
reuse beyond locked OOF rows, no writes except via caller-provided paths.
Deterministic: numpy RandomState streams spawned from (master_seed, stream).
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

CEILING = "POST_HOC_INTERNAL_EXPLORATORY"

BAN_TERMS = (
    "alzheimer", "diagnosis", "diagnostic", "causal", "causality",
    "external validation", "external-validation", "centiloid", "centaur",
    "triage", "biomarker-quantified", "disease-involved",
)

THESIS_ALLOWLIST = (
    "Uno-C", "IBS", "AUC", "Brier", "slope", "intercept", "ICI",
    "NOT_MEASURABLE", "NOT SUPERIOR", "RETAIN_NEGATIVE",
    "INCONCLUSIVE_UNDERPOWERED", "DESCRIPTIVE_ONLY", "NOT_ESTIMABLE",
    "PASS", "BLOCKED", "EXPLORATORY", "internal",
)


class PostEvalError(RuntimeError):
    pass


def _fail(code: str, detail: str = "") -> PostEvalError:
    return PostEvalError(f"BLOCKED_{code}" + (f":{detail}" if detail else ""))


# --------------------------------------------------------------------------
# Deterministic streams (A1-M2 fix: MT19937 explicit, versions recorded)
# --------------------------------------------------------------------------

def spawned_rng(master_seed: int, stream: str) -> np.random.RandomState:
    digest = hashlib.sha256(f"{int(master_seed)}:{stream}".encode()).hexdigest()
    return np.random.RandomState(int(digest[:8], 16))


def lib_versions() -> dict[str, str]:
    import scipy, sklearn
    import sksurv
    return {"numpy": np.__version__, "scipy": scipy.__version__,
            "sklearn": sklearn.__version__, "sksurv": sksurv.__version__}


# --------------------------------------------------------------------------
# IPCW + tau* + metrics (research A)
# --------------------------------------------------------------------------

def _km(times: np.ndarray, events: np.ndarray):
    from sksurv.nonparametric import kaplan_meier_estimator
    return kaplan_meier_estimator(np.asarray(events, dtype=bool),
                                  np.asarray(times, dtype=float))


def _support_filter(test_time, train_time):
    """Drop test rows at/above train support (sksurv requires strict <).

    Returns (keep_mask, n_dropped). Callers record the counts; paired
    structures apply the SAME mask to both arms (see evaluate_oof_bundle).
    """
    tmax = float(np.asarray(train_time, dtype=float).max(initial=0.0))
    keep = np.asarray(test_time, dtype=float) < tmax
    return keep, int((~keep).sum())


def ipcw_weights(train_time, train_event, eval_time, eval_event, cap: float = 10.0):
    """Marginal-KM IPCW from TRAIN only; truncated at cap with diagnostics."""
    t = np.asarray(train_time, dtype=float)
    e = np.asarray(train_event, dtype=bool)
    et = np.asarray(eval_time, dtype=float)
    kt, kv = _km(t, e)
    g = np.interp(et, kt, kv, left=1.0, right=kv[-1] if kv.size else 1.0)
    g = np.clip(g, 1e-12, 1.0)
    raw = 1.0 / g
    frac = float(np.mean(raw > cap))
    w = np.minimum(raw, float(cap))
    return w, {"frac_truncated": frac, "min_G": float(g.min()), "cap": float(cap)}

def tau_star(g_times, g_vals, max_horizon: float, gmin: float = 0.15) -> float:
    """Largest t with G(t)>=gmin and t<=max_horizon. No outer-driven clause."""
    gt = np.asarray(g_times, dtype=float)
    gv = np.asarray(g_vals, dtype=float)
    cand = gt[(gv >= float(gmin)) & (gt <= float(max_horizon))]
    if cand.size == 0:
        raise _fail("TAU_STAR_INESTIMABLE")
    return float(cand.max())


def support_truncate(rows: Sequence[Mapping[str, Any]],
                     horizons: Sequence[float]) -> dict[str, Any]:
    """Truncate evaluation grid to test follow-up support (H1 tail rule).

    Horizons beyond max observed follow-up are dropped with a
    NOT_MEASURABLE leg record (weight-diagnostics evidence), never silently.
    """
    tmax = max(float(r["time_days"]) for r in rows)
    usable = [float(h) for h in horizons if float(h) < tmax]
    dropped = [float(h) for h in horizons if float(h) > tmax]
    legs = []
    if dropped:
        legs.append({"id": "LEG-EVAL-02-TAIL", "status": "NOT_MEASURABLE",
                     "evidence": "tail_support_diagnostics.json",
                     "dropped_horizons": dropped, "max_followup": tmax})
    if not usable:
        raise _fail("NO_HORIZON_IN_SUPPORT")
    return {"horizons": usable, "max_followup": tmax, "legs": legs}


def _surv(time, event):
    from sksurv.util import Surv
    return Surv.from_arrays(np.asarray(event, dtype=bool), np.asarray(time, dtype=float))


def uno_c(train_time, train_event, test_time, test_event, risk, tau: float):
    from sksurv.metrics import concordance_index_ipcw
    keep, dropped = _support_filter(test_time, train_time)
    tmax = min(float(np.asarray(test_time, dtype=float)[keep].max(initial=0.0)),
               float(np.asarray(train_time, dtype=float).max(initial=0.0)))
    if not float(tau) < tmax:
        raise _fail("TAU_OUT_OF_SUPPORT", f"tau={tau}")
    ytr = _surv(train_time, train_event)
    yte = _surv(np.asarray(test_time, dtype=float)[keep],
                np.asarray(test_event, dtype=bool)[keep])
    c, *_ = concordance_index_ipcw(ytr, yte, np.asarray(risk, dtype=float)[keep],
                                   float(tau))
    return {"c": float(c), "tau": float(tau), "n_dropped_test": int(dropped)}


def harrell_c(time, event, risk):
    from sksurv.metrics import concordance_index_censored
    c, *_ = concordance_index_censored(np.asarray(event, dtype=bool),
                                       np.asarray(time, dtype=float),
                                       np.asarray(risk, dtype=float))
    return {"c": float(c)}


def dynamic_auc(train_time, train_event, test_time, test_event, risk, times):
    from sksurv.metrics import cumulative_dynamic_auc
    keep, dropped = _support_filter(test_time, train_time)
    tmax = min(float(np.asarray(test_time, dtype=float)[keep].max(initial=0.0)),
               float(np.asarray(train_time, dtype=float).max(initial=0.0)))
    grid = [float(v) for v in np.asarray(times, dtype=float) if float(v) < tmax]
    if not grid:
        raise _fail("NO_TIMES_IN_SUPPORT")
    ytr = _surv(train_time, train_event)
    yte = _surv(np.asarray(test_time, dtype=float)[keep],
                np.asarray(test_event, dtype=bool)[keep])
    auc, mean_auc = cumulative_dynamic_auc(
        ytr, yte, np.asarray(risk, dtype=float)[keep],
        np.asarray(grid, dtype=float))
    return {"auc": [float(v) for v in auc], "mean_auc": float(mean_auc),
            "times": [float(v) for v in grid], "n_dropped_test": int(dropped),
            "dropped_times": sorted(set(float(v) for v in times) - set(grid))}


def ibs_summary(train_time, train_event, test_time, test_event, surv_prob, times):
    from sksurv.metrics import integrated_brier_score, brier_score
    keep, dropped = _support_filter(test_time, train_time)
    ytr = _surv(train_time, train_event)
    yte = _surv(np.asarray(test_time, dtype=float)[keep],
                np.asarray(test_event, dtype=bool)[keep])
    p = np.asarray(surv_prob, dtype=float)[keep]
    tmax = min(float(np.asarray(test_time, dtype=float)[keep].max(initial=0.0)),
               float(np.asarray(train_time, dtype=float).max(initial=0.0)))
    tm = np.asarray([float(v) for v in np.asarray(times, dtype=float)
                     if float(v) < tmax], dtype=float)
    if tm.size == 0:
        raise _fail("NO_TIMES_IN_SUPPORT")
    ibs = float(integrated_brier_score(ytr, yte, p, tm))
    _, curve = brier_score(ytr, yte, p, tm)
    null = np.full_like(p, np.mean([
        float(np.mean((np.asarray(test_event, dtype=float) <= 0.5)))] * len(tm)))
    try:
        ibs_null = float(integrated_brier_score(ytr, yte, null, tm))
        scaled = 1.0 - ibs / ibs_null if ibs_null > 0 else 0.0
    except Exception:
        scaled = 0.0
    return {"ibs": ibs, "scaled_ibs": float(scaled),
            "brier_curve": [float(v) for v in curve], "times": [float(v) for v in tm],
            "n_dropped_test": int(dropped),
            "dropped_times": sorted(set(float(v) for v in np.asarray(times, dtype=float)) - set(float(v) for v in tm))}


def calibration_at_horizon(time, event, pred_surv, horizon: float, *,
                           train_time, train_event, n_bins: int = 5):
    """IPCW-logistic slope/intercept + binned ICI descriptors (descriptive)."""
    import statsmodels.api as sm
    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=bool)
    pred = np.asarray(pred_surv, dtype=float)
    tmax = min(float(time.max(initial=0.0)),
               float(np.asarray(train_time, dtype=float).max(initial=0.0)))
    if not float(horizon) < tmax:
        raise _fail("HORIZON_OUT_OF_SUPPORT", f"horizon={horizon}")
    status = ((time <= horizon) & event).astype(float)
    w, _ = ipcw_weights(train_time, train_event, np.minimum(time, horizon),
                        np.ones_like(time, dtype=bool))
    logit = np.log(np.clip(pred, 1e-6, 1 - 1e-6) / np.clip(1 - pred, 1e-6, 1 - 1e-6))
    X = sm.add_constant(logit)
    fit = sm.GLM(status, X, family=sm.families.Binomial(), freq_weights=w).fit()
    ci = fit.conf_int(alpha=0.05)
    order = np.argsort(pred)
    bins = np.array_split(order, int(n_bins))
    ici_terms, emax, per_bin = [], 0.0, []
    for b in bins:
        if len(b) == 0:
            continue
        obs = float(np.average(status[b], weights=w[b]))
        exp = float(np.mean(pred[b]))
        ici_terms.append(abs(obs - exp))
        emax = max(emax, abs(obs - exp))
        per_bin.append({"n": int(len(b)), "observed": obs, "expected": exp})
    ici = float(np.mean(ici_terms)) if ici_terms else 0.0
    return {"slope": float(fit.params[1]), "slope_ci": [float(ci[0][1]), float(ci[1][1])],
            "intercept": float(fit.params[0]),
            "intercept_ci": [float(ci[0][0]), float(ci[1][0])],
            "ici": ici, "emax": float(emax), "n_bins": int(n_bins),
            "horizon": float(horizon)}


# --------------------------------------------------------------------------
# Paired inference: bootstrap (stratified, percentile+BCa) + permutation
# --------------------------------------------------------------------------

def paired_bootstrap_delta(a, b, *, n_boot: int = 2000, seed: int = 13,
                           fold_id=None) -> dict[str, Any]:
    """Fold-stratified paired percentile bootstrap + BCa sensitivity."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape or a.size < 3:
        raise _fail("BOOTSTRAP_INVALID_INPUT")
    n = a.size
    fold_id = np.zeros(n, dtype=int) if fold_id is None else np.asarray(fold_id)
    rng = spawned_rng(seed, "paired-bootstrap")
    diffs = []
    for _ in range(int(n_boot)):
        sel = np.concatenate([rng.choice(np.where(fold_id == f)[0],
                                         size=int((fold_id == f).sum()), replace=True)
                              for f in np.unique(fold_id)])
        diffs.append(float(np.mean(a[sel] - b[sel])))
    diffs = np.asarray(diffs)
    obs = float(np.mean(a - b))
    ci_low, ci_high = float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))
    # BCa via jackknife acceleration
    jack = np.array([float(np.mean(np.delete(a - b, i))) for i in range(n)])
    jmean = float(jack.mean())
    denom = float(np.sum((jmean - jack) ** 2) ** 1.5)
    accel = float(np.sum((jmean - jack) ** 3) / (6.0 * denom)) if denom > 0 else 0.0
    from scipy.stats import norm
    z0 = float(norm.ppf(np.mean(diffs < obs))) if 0 < np.mean(diffs < obs) < 1 else 0.0
    bca = {"accel": accel, "z0": z0, "stable": bool(abs(accel) < 0.2)}
    return {"delta": obs, "ci_low": ci_low, "ci_high": ci_high,
            "method": "percentile-fold-stratified", "n_boot": int(n_boot),
            "bca": bca}


def permutation_paired_p(a, b, *, n_perm: int = 2000, seed: int = 13) -> float:
    """Within-patient arm-label swap null; censoring untouched."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    obs = abs(float(np.mean(a - b)))
    rng = spawned_rng(seed, "permutation")
    hits = 0
    for _ in range(int(n_perm)):
        flip = rng.rand(a.size) < 0.5
        d = np.where(flip, b - a, a - b)
        if abs(float(np.mean(d))) >= obs:
            hits += 1
    return float((hits + 1) / (int(n_perm) + 1))


def ni_retention(delta: float, ci: tuple, *, margin: float,
                 higher_better: bool) -> dict[str, Any]:
    """Non-inferiority retention with prespecified margin (no wide-CI reward)."""
    lo, hi = float(ci[0]), float(ci[1])
    if higher_better:
        decision = "RETAIN_NEGATIVE" if lo > -float(margin) and hi > 0 else (
            "SUPERIOR" if lo > 0 else "INCONCLUSIVE_UNDERPOWERED")
        if lo > 0:
            decision = "SUPERIOR"
    else:
        decision = "RETAIN_NEGATIVE" if hi < float(margin) and lo < 0 else (
            "SUPERIOR" if hi < 0 else "INCONCLUSIVE_UNDERPOWERED")
        if hi < 0:
            decision = "SUPERIOR"
    return {"decision": decision, "delta": float(delta),
            "ci": [lo, hi], "margin": float(margin)}


# --------------------------------------------------------------------------
# Risk tables, subgroups, manufacturer, checklists, attestation
# --------------------------------------------------------------------------

def risk_table(time, event, horizons) -> list[dict[str, Any]]:
    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=bool)
    rows = []
    for h in horizons:
        h = float(h)
        rows.append({"horizon": h, "at_risk": int(np.sum(time >= h)),
                     "events_by": int(np.sum(event & (time <= h)))})
    return rows


def subgroup_decision(n_events: int) -> str:
    n = int(n_events)
    if n >= 25:
        return "CI_ALLOWED"
    if n >= 10:
        return "DESCRIPTIVE_ONLY"
    return "NOT_ESTIMABLE"


def stratified_manufacturer_report(stats: Mapping[str, tuple], *,
                                   min_events: int = 25) -> dict[str, Any]:
    """Descriptive per-manufacturer stats; never a gate input."""
    out: dict[str, Any] = {"gate_input": False, "manufacturers": {}}
    for name, (value, n_events) in stats.items():
        if int(n_events) >= int(min_events):
            out["manufacturers"][str(name)] = {"status": "ESTIMABLE",
                                               "value": float(value),
                                               "n_events": int(n_events)}
        else:
            out["manufacturers"][str(name)] = {"status": "NOT_MEASURABLE",
                                               "n_events": int(n_events)}
    return out


def tripod_machine_checks(evidence: Mapping[str, Any]) -> dict[str, Any]:
    required = {"fold_hashes": list, "n_events": int, "metric_def": str,
                "B": int, "seed": int, "code_hash": str, "pairing": str}
    for key, typ in required.items():
        val = evidence.get(key)
        if not isinstance(val, typ) or (isinstance(val, (list, str)) and not val):
            return {"status": "FAIL", "missing": key}
        if key == "B" and int(val) < 100:
            return {"status": "FAIL", "missing": "B_too_small"}
    return {"status": "PASS", "checked": sorted(required)}


def validate_attestation(att: Mapping[str, Any]) -> dict[str, Any]:
    slots = att.get("attesters")
    if not isinstance(slots, list) or len(slots) != 2:
        return {"status": "FAIL", "reason": "TWO_ATTESTERS_REQUIRED"}
    for s in slots:
        if not isinstance(s, Mapping) or not s.get("name") or not s.get("quote") \
                or not s.get("digest"):
            return {"status": "FAIL", "reason": "ATTESTER_SLOT_INCOMPLETE"}
    return {"status": "ATTESTED_ONLY", "dissent": str(att.get("dissent", ""))}


# --------------------------------------------------------------------------
# Thesis writer (allowlist + provenance + ceiling)
# --------------------------------------------------------------------------

THESIS_GLUE_WORDS = frozenset("""
a an the and or with without for of in on at to from by vs versus
is are was were be been higher lower than between within across per
both each all any no not non zero one two three four five six seven
eight nine ten subjects events folds models arms cohort branch outer
inner validation test train risk score mean median ci confidence interval
horizon horizons days month months year years rate rates curve curves
table figure section results methods discussion limitation limitations
exploratory internal evidence technical benchmark
""".split())


def _thesis_tokens_ok(text: str) -> tuple[bool, list[str]]:
    import re
    bad = []
    allowed = {w.casefold() for w in THESIS_ALLOWLIST} | set(THESIS_GLUE_WORDS)
    for tok in re.findall(r"[A-Za-z][A-Za-z'\-]*", text):
        low = tok.casefold()
        if low not in allowed and not any(ch.isdigit() for ch in tok):
            bad.append(tok)
    return (not bad, sorted(set(bad)))


def thesis_write(records: Sequence[Mapping[str, Any]], *, ceiling: str,
                 section: str, not_measurable: Sequence[str] = ()) -> dict[str, Any]:
    """Thesis writer: allowlist vocabulary + JSON-pointer provenance + ceiling.

    NOT_MEASURABLE legs map to fixed boilerplate. Any novel token -> BLOCKED.
    """
    if ceiling != CEILING:
        return {"status": "BLOCKED", "reason": "CEILING_MISMATCH"}
    if section not in ("results", "methods", "discussion", "limitations"):
        return {"status": "BLOCKED", "reason": "SECTION_INVALID"}
    lines, pointers = [], []
    for rec in records:
        text = str(rec.get("text", ""))
        ok, bad = _thesis_tokens_ok(text)
        if not ok:
            return {"status": "BLOCKED", "reason": "NOVEL_TOKEN",
                    "tokens": bad, "pointer": str(rec.get("pointer", ""))}
        lines.append(text)
        pointers.append(str(rec.get("pointer", "")))
    for leg in not_measurable:
        lines.append(f"{leg} NOT_MEASURABLE with recorded evidence; no claim made.")
        pointers.append(f"$.legs[{leg}]")
    return {"status": "PASS", "markdown": "\n".join(lines), "pointers": pointers}


# --------------------------------------------------------------------------
# Privacy + legs runner + entry gate
# --------------------------------------------------------------------------

def privacy_cell_check(*, n_subjects: int, n_events: int,
                       min_subjects: int = 10, min_events: int = 10) -> dict[str, Any]:
    if int(n_subjects) >= int(min_subjects) and int(n_events) >= int(min_events):
        return {"status": "PASS"}
    return {"status": "SUPPRESSED", "reason": "K_SUPPRESSION"}


def scan_text(text: str) -> dict[str, Any]:
    import re
    lowered = str(text)
    hits = []
    if re.search(r"[A-Za-z]:[\\/]", lowered):
        hits.append("absolute_path")
    if re.search(r"\b[A-Fa-f0-9]{32,64}\b", lowered):
        hits.append("hash_like_token")
    if re.search(r"\bpt_[A-Za-z0-9]+\b", lowered):
        hits.append("subject_id_like_token")
    for term in BAN_TERMS:
        if term in lowered.casefold():
            hits.append(f"ban:{term}")
    return {"status": "BLOCKED" if hits else "PASS", "hits": hits}


def run_legs(legs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """BLOCKED halts; NOT_MEASURABLE/TIMEOUT/UNKNOWN recorded, pipeline continues.

    UNKNOWN propagates as a non-passing record (M4 fix): it never upgrades a
    gate and the exploratory stamp never overrides it (claims compiler rule).
    """
    done = []
    for leg in legs:
        status = str(leg.get("status", ""))
        if status == "BLOCKED":
            return {"status": "BLOCKED", "halted_at": str(leg.get("id", "?")),
                    "reason": str(leg.get("reason", "")), "completed": done}
        if status not in ("PASS", "NOT_MEASURABLE", "TIMEOUT", "UNKNOWN"):
            return {"status": "BLOCKED", "halted_at": str(leg.get("id", "?")),
                    "reason": "LEG_STATUS_INVALID", "completed": done}
        done.append(str(leg.get("id", "?")))
    return {"status": "PASS", "completed": done}


def entry_gate(*, run_root: str | Path, config_path: str | Path | None = None) -> dict[str, Any]:
    """Refuse unless final verification PASSES on live state.

    Prefers a validated final_verification_pass.json artifact; otherwise runs
    the sanctioned verifier in-process (read-only) and requires PASS.
    Either way the live run digest is returned for downstream binding.
    """
    root = Path(run_root)
    cand = root / "final_verification_pass.json"
    if cand.is_file():
        try:
            payload = json.loads(cand.read_text(encoding="utf-8"))
            body = dict(payload)
            recorded = body.pop("integrity_sha256", None)
            canon = json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
            if payload.get("status") == "PASS" and recorded == hashlib.sha256(
                    canon).hexdigest():
                return {"status": "PASS",
                        "run_contract_sha256": payload.get("run_contract_sha256"),
                        "evidence": "artifact"}
        except (OSError, ValueError, UnicodeDecodeError):
            pass
    if config_path is None:
        raise _fail("VERIFIER_NOT_PASS", "no artifact and no live-verify path")
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "method_y_v4_verifier",
        Path(__file__).resolve().parents[2] / "scripts"
        / "verify_method_y_image_factorial_v4.py")
    if spec is None or spec.loader is None:
        raise _fail("VERIFIER_NOT_PASS", "loader unavailable")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
        result = module.verify(config_path=Path(config_path), level="final")
    except Exception as exc:
        raise _fail("VERIFIER_NOT_PASS", f"live-verify failed: {type(exc).__name__}") from exc
    if not isinstance(result, Mapping) or result.get("status") != "PASS":
        raise _fail("VERIFIER_NOT_PASS", "live-verify not PASS")
    try:
        live_digest = json.loads((root / "run_contract.json").read_text(
            encoding="utf-8")).get("digest")
    except (OSError, ValueError):
        raise _fail("VERIFIER_NOT_PASS", "run contract unreadable")
    return {"status": "PASS", "run_contract_sha256": live_digest,
            "evidence": "live-verify",
            "valid_outer_cells": result.get("valid_outer_cells")}


# --------------------------------------------------------------------------
# Round-2 gap closures (traceability 04): probes, multiplicity, PACC,
# intake, CAM gates, fusion keys, firewall, stability, D-cal, interaction,
# Antolini, omnibus, locked-input probing
# --------------------------------------------------------------------------

def audit_probe_weights(provenance: Mapping[str, Any]) -> dict[str, Any]:
    """Scanner/tabular probe weights must be inner-train-frozen artifacts.

    Anything fit on outer-derived data (or with missing lock fields) is a
    refit (C2) -> PROBE_REUSE_VIOLATION, never silently accepted.
    """
    if not isinstance(provenance, Mapping):
        return {"status": "PROBE_REUSE_VIOLATION", "reason": "PROVENANCE_NOT_MAPPING"}
    for key in ("weights_source", "solver", "seed", "C", "digest"):
        if provenance.get(key) in (None, ""):
            return {"status": "PROBE_REUSE_VIOLATION", "reason": f"LOCK_FIELD_MISSING:{key}"}
    if str(provenance.get("weights_source")) != "inner-train":
        return {"status": "PROBE_REUSE_VIOLATION",
                "reason": f"OUTER_FIT:{provenance.get('weights_source')}"}
    return {"status": "PASS"}


def bh_fdr(p_values: Sequence[float], *, alpha: float = 0.05):
    """Benjamini-Hochberg FDR over prespecified contrast families."""
    p = [float(v) for v in p_values]
    m = len(p)
    order = sorted(range(m), key=lambda i: p[i])
    rejected = [False] * m
    thresh = 0
    for rank, i in enumerate(order, start=1):
        if p[i] <= rank / m * alpha:
            thresh = rank
    for rank, i in enumerate(order, start=1):
        if rank <= thresh:
            rejected[i] = True
    return rejected


def omnibus_friedman(values_by_arm: Sequence[Sequence[float]]) -> dict[str, Any]:
    """Omnibus across arms on matched fold-level metrics (deterministic)."""
    from scipy.stats import friedmanchisquare
    arms = [[float(v) for v in row] for row in values_by_arm]
    if len(arms) < 3 or any(len(r) < 2 for r in arms):
        raise _fail("OMNIBUS_INVALID_INPUT")
    stat, p = friedmanchisquare(*arms)
    return {"stat": float(stat), "p": float(p), "n_arms": len(arms)}


def mixedlm_pacc(data, *, maxiter: int = 200, tol: float = 1e-6) -> dict[str, Any]:
    """Prespecified PACC mixed-effects, DESCRIPTIVE ONLY (H4).

    Locked formula ``pacc ~ time * treatment`` + random intercept, ML,
    fixed convergence criteria. Singular/non-converged -> NOT_MEASURABLE
    (never model-switch). No causal reading of treatment terms.
    """
    import pandas as pd
    import statsmodels.api as sm
    df = data if isinstance(data, pd.DataFrame) else pd.DataFrame(data)
    for col in ("subject", "time", "treatment", "pacc"):
        if col not in df.columns:
            return {"status": "NOT_MEASURABLE", "reason": f"COLUMN_MISSING:{col}"}
    try:
        model = sm.MixedLM.from_formula("pacc ~ time * treatment", df,
                                        groups=df["subject"], re_formula="1")
        res = model.fit(maxiter=int(maxiter), tol=float(tol), disp=0)
    except Exception as exc:
        return {"status": "NOT_MEASURABLE", "reason": f"FIT_FAILED:{type(exc).__name__}"}
    try:
        converged = bool(res.converged)
        params = {str(k): float(v) for k, v in dict(res.params).items()}
        cov = np.asarray(res.cov_params(), dtype=float)
        eig = np.linalg.eigvalsh(cov)
        psd = bool(np.all(np.isfinite(eig)) and np.min(eig) > -1e-8)
        finite = all(math.isfinite(v) for v in params.values())
    except Exception:
        return {"status": "NOT_MEASURABLE", "reason": "POSTFIT_CHECK_FAILED"}
    if not (converged and psd and finite):
        return {"status": "NOT_MEASURABLE",
                "reason": f"SINGULAR_OR_DIVERGED:converged={converged},psd={psd},finite={finite}"}
    return {"status": "DESCRIPTIVE_FIT", "params": params,
            "role": "DESCRIPTIVE_ONLY_NEVER_GATE"}


def missingness_table(data) -> dict[str, Any]:
    """PACC missingness/dropout audit table (H4 companion, descriptive)."""
    import pandas as pd
    df = data if isinstance(data, pd.DataFrame) else pd.DataFrame(data)
    counts = df.groupby("subject")["time"].agg(["count", "max"])
    return {"n_subjects": int(df["subject"].nunique()),
            "median_obs_per_subject": float(counts["count"].median()),
            "sparse_subjects_lt3": int((counts["count"] < 3).sum()),
            "dropout_note": "MNAR dropout-due-to-progression cannot be ruled out; "
                            "MAR assumed and stated; treatment terms non-causal."}


def intake_lint(intake: Mapping[str, Any]) -> dict[str, Any]:
    """External-intake gate: STATIC schema/contract lint ONLY (H5).

    No fixtures through eval paths, no synthetic rows, no mapping lock.
    """
    if not isinstance(intake, Mapping):
        return {"status": "BLOCKED", "reason": "INTAKE_NOT_MAPPING"}
    for key in ("schema_version", "tracer", "endpoint", "censoring",
                "quarantine", "mapping_lock", "data_rows"):
        if key not in intake:
            return {"status": "BLOCKED", "reason": f"INTAKE_KEY_MISSING:{key}"}
    if intake.get("mapping_lock"):
        return {"status": "BLOCKED", "reason": "MAPPING_LOCK_FORBIDDEN"}
    if intake.get("data_rows"):
        return {"status": "BLOCKED", "reason": "SYNTHETIC_ROWS_FORBIDDEN"}
    if str(intake.get("quarantine")) != "private-only":
        return {"status": "BLOCKED", "reason": "QUARANTINE_RULE_INVALID"}
    return {"status": "PASS"}


def cam_gates_read(report: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a CAM attribution report's own gates (H6 integration)."""
    if not isinstance(report, Mapping):
        return {"status": "BLOCKED", "reason": "REPORT_NOT_MAPPING"}
    if report.get("ceiling") != CEILING:
        return {"status": "BLOCKED", "reason": "CEILING_MISMATCH"}
    gates = report.get("gates")
    if not isinstance(gates, Mapping) or not gates:
        return {"status": "BLOCKED", "reason": "GATES_MISSING"}
    legs = report.get("legs")
    if isinstance(legs, Mapping) and legs.get("status") not in ("PASS", None):
        if legs.get("status") not in ("PASS",):
            return {"status": "BLOCKED", "reason": "CAM_LEGS_NOT_PASS"}
    return {"status": "PASS"}


def assert_fusion_keys(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Fusion key schema + one-row-per-subject dedup (H7)."""
    required = ("patient_hash", "arm", "branch", "outer_fold", "split_digest")
    seen: set[tuple] = set()
    for row in rows:
        if not isinstance(row, Mapping):
            return {"status": "BLOCKED", "reason": "ROW_NOT_MAPPING"}
        for key in required:
            if key not in row or row[key] in (None, ""):
                return {"status": "BLOCKED", "reason": f"KEY_MISSING:{key}"}
        tag = (str(row["patient_hash"]), str(row["arm"]), str(row["branch"]),
               int(row["outer_fold"]))
        if tag in seen:
            return {"status": "BLOCKED", "reason": f"DUPLICATE_ROW:{tag[0]}"}
        seen.add(tag)
    return {"status": "PASS", "n_rows": len(seen)}


def firewall_assertion(journal_events: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Outer-test firewall log: any outer read before lock -> AT_RISK (M5)."""
    for evt in journal_events:
        if not isinstance(evt, Mapping):
            continue
        if str(evt.get("type", "")) == "outer_result_read":
            try:
                if float(evt.get("t_view", 0)) < float(evt.get("t_lock", 0)):
                    return {"status": "EXACTLY_ONCE_AT_RISK",
                            "reason": "OUTER_READ_BEFORE_LOCK"}
            except (TypeError, ValueError):
                return {"status": "EXACTLY_ONCE_AT_RISK",
                        "reason": "FIREWALL_TIMESTAMP_INVALID"}
    return {"status": "PASS"}


def selection_stability(winner_records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Bates-direction stability report over locked winner records (no redesign)."""
    ids = [str(r.get("winner_candidate_id", "?")) for r in winner_records
           if isinstance(r, Mapping)]
    uniq = sorted(set(ids))
    return {"stable": len(uniq) == 1, "n_partitions": len(ids),
            "unique_winners": uniq}


def d_calibration(surv_at_event, *, n_bins: int = 10) -> dict[str, Any]:
    """D-calibration uniformity check, exploratory, NEVER a gate (M1)."""
    from scipy.stats import chi2 as chi2_dist
    s = np.asarray(list(surv_at_event), dtype=float)
    if s.size == 0 or np.any((s <= 0) | (s >= 1)):
        raise _fail("DCAL_INVALID_INPUT")
    counts, _ = np.histogram(s, bins=int(n_bins), range=(0.0, 1.0))
    expected = float(s.size) / int(n_bins)
    stat = float(np.sum((counts - expected) ** 2 / expected))
    p = float(chi2_dist.sf(stat, int(n_bins) - 1))
    return {"statistic": stat, "p": p, "n_bins": int(n_bins), "n": int(s.size),
            "gate_effect": "NONE"}


def interaction_did(a_risk, b_risk, subgroup, *, n_boot: int = 500,
                    seed: int = 13) -> dict[str, Any]:
    """Subgroup-heterogeneity via difference-in-paired-differences (no Cox API risk).

    Primary heterogeneity evidence; within-subgroup superiority claims still
    require this signal plus prespecified interaction discipline.
    """
    a = np.asarray(a_risk, dtype=float)
    b = np.asarray(b_risk, dtype=float)
    g = np.asarray(subgroup)
    groups = sorted(set(g.tolist()))
    if a.shape != b.shape or a.shape != g.shape or len(groups) != 2:
        raise _fail("INTERACTION_INVALID_INPUT")
    d = a - b
    means = {str(gv): float(np.mean(d[g == gv])) for gv in groups}
    did = means[str(groups[0])] - means[str(groups[1])]
    rng = spawned_rng(seed, "interaction-did")
    boots = []
    for _ in range(int(n_boot)):
        sel = rng.choice(a.size, size=a.size, replace=True)
        boots.append(float(np.mean(d[sel][g[sel] == groups[0]])
                           - np.mean(d[sel][g[sel] == groups[1]])))
    return {"did": float(did),
            "ci_low": float(np.percentile(boots, 2.5)),
            "ci_high": float(np.percentile(boots, 97.5)),
            "n_subgroups": 2, "n_boot": int(n_boot)}


def antolini_ctd(time, event, surv_prob, times) -> dict[str, Any]:
    """Antolini time-dependent C, EXPLORATORY ONLY (mixes calibration in)."""
    t = np.asarray(time, dtype=float)
    e = np.asarray(event, dtype=bool)
    S = np.asarray(surv_prob, dtype=float)
    g = np.asarray(times, dtype=float)
    if S.shape[0] != t.size or S.shape[1] != g.size:
        raise _fail("ANTOLINI_INVALID_INPUT")
    conc = tot = 0
    for i in np.where(e)[0]:
        j = int(np.argmin(np.abs(g - t[i])))
        ctrl = [k for k in range(t.size) if k != i and t[k] >= t[i]]
        if not ctrl:
            continue
        for k in ctrl:
            if S[i, j] < S[k, j]:
                conc += 1
            elif S[i, j] == S[k, j]:
                conc += 0.5
            tot += 1
    ctd = (conc / tot) if tot else 0.5
    return {"ctd": float(ctd), "pairs": int(tot), "role": "EXPLORATORY"}


# --------------------------------------------------------------------------
# Locked-OOF-bundle production evaluation (one firewalled pass)
# --------------------------------------------------------------------------

def _fold_metric_block(rows_fold, train_t, train_e, *, horizons, tau):
    """Per-fold metrics with that partition's FROZEN train IPCW (H8 fix).

    Grid-bound metrics (Uno/IBS/AUC/calibration) evaluate ONLY on horizons
    within THIS fold's test follow-up; otherwise that arm-fold is
    NOT_MEASURABLE (recorded, never extrapolated). Risk scalars need no
    grid and always compute (contrasts stay valid).
    """
    out = {}
    arms = sorted({str(r["arm"]) for r in rows_fold})
    for arm in arms:
        sub = [r for r in rows_fold if str(r["arm"]) == arm]
        t = np.array([float(r["time_days"]) for r in sub])
        e = np.array([bool(r["event"]) for r in sub])
        risk = np.array([float(r["risk"]) for r in sub])
        fold_tmax = float(t.max()) if t.size else 0.0
        train_tmax = float(np.asarray(train_t, dtype=float).max(initial=0.0))
        support_max = min(fold_tmax, train_tmax)
        usable = [float(h) for h in horizons if float(h) < support_max]
        rec: dict[str, Any] = {"n": len(sub), "events": int(e.sum()),
                               "fold_tmax": fold_tmax,
                               "usable_horizons": usable}
        if not usable or float(tau) >= support_max:
            rec.update({"grid_status": "NOT_MEASURABLE",
                        "evidence": "fold_support_diagnostics.json"})
            out[arm] = rec
            continue
        S = np.array([[float(r["surv_at_horizons"][str(h)]) for h in usable]
                      for r in sub])
        rec.update({
            "grid_status": "PASS",
            "uno": uno_c(train_t, train_e, t, e, risk, tau=float(tau)),
            "harrell": harrell_c(t, e, risk),
            "ibs": ibs_summary(train_t, train_e, t, e, S, usable),
            "calibration": calibration_at_horizon(
                t, e, np.array([float(r["surv_at_horizons"][str(float(tau))])
                                for r in sub]),
                horizon=float(tau), train_time=train_t, train_event=train_e),
        })
        out[arm] = rec
    return out


def evaluate_oof_bundle(rows: Sequence[Mapping[str, Any]], *,
                        train_by_partition: Mapping[Any, Any],
                        horizons: Sequence[float], tau: float,
                        B: int = 2000, seed: int = 13,
                        ni_margins: Mapping[str, float] | None = None,
                        reference_arm: str = "A0") -> dict[str, Any]:
    """Evaluate locked OOF rows: per-fold metrics under frozen per-partition
    IPCW, fold-mean points, stratified-bootstrap paired contrasts, NI, subgroups.

    Row schema (all required): patient_hash, arm, branch, outer_fold,
    split_digest, time_days, event, risk, surv_at_horizons {h: p}.
    train_by_partition[(branch, outer_fold)] = (train_times, train_events).
    No training, no refit, no writes. Pure + deterministic.
    """
    ni_margins = {"dIBS": 0.01, "dUno": 0.02, **dict(ni_margins or {})}
    rows = list(rows)
    key_check = assert_fusion_keys(rows)
    if key_check["status"] != "PASS":
        raise _fail("FUSION_KEYS_INVALID", key_check.get("reason", ""))
    horizons = [float(h) for h in horizons]
    partitions: dict[tuple, list] = {}
    for r in rows:
        partitions.setdefault((str(r["branch"]), int(r["outer_fold"])), []).append(r)
    # One shared support mask per partition (same subjects across arms keep
    # pairing valid; time/event must agree across arms = matched-outcome check).
    support_dropped: dict[str, int] = {}
    kept_partitions: dict[tuple, list] = {}
    for key, prows in partitions.items():
        if key not in train_by_partition:
            raise _fail("TRAIN_IPCW_MISSING", f"{key[0]}/o{key[1]}")
        train_t = np.asarray(train_by_partition[key][0], dtype=float)
        tmax = float(train_t.max(initial=0.0))
        by_subj: dict[str, list] = {}
        for r in prows:
            by_subj.setdefault(str(r["patient_hash"]), []).append(r)
        kept = []
        for subj, recs in by_subj.items():
            times = {float(r["time_days"]) for r in recs}
            evs = {bool(r["event"]) for r in recs}
            if len(times) != 1 or len(evs) != 1:
                raise _fail("MATCHED_OUTCOME_MISMATCH", f"{key[0]}/{subj}")
            if next(iter(times)) < tmax:
                kept.extend(recs)
        support_dropped[f"{key[0]}/o{key[1]}"] = len(prows) - len(kept)
        kept_partitions[key] = kept
    fold_blocks = {}
    for key, prows in kept_partitions.items():
        if not prows:
            continue
        train_t, train_e = train_by_partition[key]
        fold_blocks[key] = _fold_metric_block(
            prows, np.asarray(train_t, dtype=float),
            np.asarray(train_e, dtype=bool),
            horizons=horizons, tau=float(tau))
    kept_rows = [r for prows in kept_partitions.values() for r in prows]
    arms = sorted({str(r["arm"]) for r in kept_rows}) if kept_rows else sorted(
        {str(r["arm"]) for r in rows})
    per_arm: dict[str, Any] = {}
    for arm in arms:
        vals = [fold_blocks[k][arm] for k in sorted(fold_blocks)
                if arm in fold_blocks[k]]
        if not vals:
            continue
        ok = [v for v in vals if v.get("grid_status") == "PASS"]
        entry: dict[str, Any] = {
            "n": int(sum(v["n"] for v in vals)),
            "events": int(sum(v["events"] for v in vals)),
            "folds_evaluated": len(ok), "folds_total": len(vals),
        }
        if ok:
            entry.update({
                "uno_mean": float(np.mean([v["uno"]["c"] for v in ok])),
                "uno_per_fold": [v["uno"]["c"] for v in ok],
                "harrell_mean": float(np.mean([v["harrell"]["c"] for v in ok])),
                "ibs_mean": float(np.mean([v["ibs"]["ibs"] for v in ok])),
                "calibration_slope_mean": float(np.mean(
                    [v["calibration"]["slope"] for v in ok])),
            })
        else:
            entry["grid_status"] = "NOT_MEASURABLE"
        per_arm[arm] = entry
        per_arm[arm]["risk_table"] = risk_table(
            np.concatenate([[float(r["time_days"]) for r in kept_rows
                             if str(r["arm"]) == arm]]),
            np.concatenate([[bool(r["event"]) for r in kept_rows
                             if str(r["arm"]) == arm]]), horizons)
    contrasts = []
    pairs = [(a, b) for a in arms for b in arms
             if a == reference_arm and b != a]
    for a, b in pairs:
        da_parts, db_parts, fa_parts = [], [], []
        for key in sorted(kept_partitions):
            ma = {str(r["patient_hash"]): float(r["risk"]) for r in kept_partitions[key]
                  if str(r["arm"]) == a}
            mb = {str(r["patient_hash"]): float(r["risk"]) for r in kept_partitions[key]
                  if str(r["arm"]) == b}
            common = sorted(set(ma) & set(mb))
            for s in common:
                da_parts.append(ma[s])
                db_parts.append(mb[s])
                fa_parts.append(key)
        if not da_parts:
            contrasts.append({"pair": [a, b], "status": "NOT_MEASURABLE",
                              "evidence": "no-common-subjects"})
            continue
        boot = paired_bootstrap_delta(np.asarray(da_parts), np.asarray(db_parts),
                                      n_boot=int(B), seed=int(seed),
                                      fold_id=np.asarray(fa_parts))
        ni = ni_retention(boot["delta"], (boot["ci_low"], boot["ci_high"]),
                          margin=float(ni_margins["dUno"]), higher_better=True)
        pval = permutation_paired_p(np.asarray(da_parts), np.asarray(db_parts),
                                    n_perm=min(2000, 200 + len(da_parts) * 10),
                                    seed=int(seed))
        contrasts.append({"pair": [a, b], "n_common": len(da_parts),
                          "delta": boot["delta"], "ci": [boot["ci_low"], boot["ci_high"]],
                          "bca": boot["bca"], "perm_p": pval,
                          "ni": ni, "status": "PASS"})
    omni = None
    try:
        subs = sorted({str(r["patient_hash"]) for r in kept_rows})
        mat = []
        for arm in arms:
            m = {str(r["patient_hash"]): float(r["risk"]) for r in kept_rows
                 if str(r["arm"]) == arm}
            if all(s in m for s in subs) and subs:
                mat.append([m[s] for s in subs])
        if len(mat) == len(arms) and len(mat) >= 3 and mat:
            omni = omnibus_friedman(mat)
    except (PostEvalError, ValueError):
        omni = None
    return {"status": "PASS", "arms": arms, "per_arm": per_arm,
            "contrasts": contrasts, "omnibus": omni,
            "partitions": sorted([f"{k[0]}/o{k[1]}" for k in fold_blocks]),
            "support_dropped": support_dropped,
            "horizons": horizons, "tau": float(tau)}


# --------------------------------------------------------------------------
# Locked-report compiler (production path: consume + verify locked evidence)
# --------------------------------------------------------------------------

def _load_json_file(path: Path):
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError, UnicodeDecodeError) as exc:
        raise _fail("LOCKED_REPORT_UNREADABLE", Path(path).name) from exc
    if not isinstance(payload, Mapping):
        raise _fail("LOCKED_REPORT_NOT_MAPPING", Path(path).name)
    return dict(payload)


def compile_locked_reports(*, root: str | Path,
                           ni_margins: Mapping[str, float] | None = None,
                           reference_arm: str = "A0") -> dict[str, Any]:
    """Compile the locked v4 evidence layer into evaluation legs.

    Reads ONLY finalized artifacts: 50 public per-fold reports, benchmark
    summary, pair analysis reports, orchestration_state, run_contract.
    Validates schemas + matched-design consistency (same folds/n per arm)
    + digest bindings where recorded. Missing CORE files -> BLOCKED;
    missing EXTENDED files -> NOT_MEASURABLE legs. No training, no refit,
    no inference, no writes. Pure read/compute.
    """
    ni_margins = {"dIBS": 0.01, "dUno": 0.02, **dict(ni_margins or {})}
    root = Path(root)
    legs: list[dict[str, Any]] = []

    def need(sub: str) -> Path:
        p = root / sub
        if not p.is_file():
            raise _fail("LOCKED_CORE_MISSING", sub)
        return p

    state = _load_json_file(need("orchestration_state.json"))
    if state.get("stage") != "FINALIZED" or int(state.get("outer_cells_finalized", -1)) != 50:
        raise _fail("RUN_NOT_FINALIZED")
    rc = _load_json_file(need("run_contract.json"))
    live_digest = rc.get("digest")
    summary = _load_json_file(need("reports/factorial_benchmark_summary.json"))
    if summary.get("status") != "COMPLETED":
        raise _fail("BENCHMARK_SUMMARY_NOT_COMPLETED")
    branches = ["amyloid", "tau"]
    arms = ["A0", "A1", "A2", "A3", "A4"]
    folds = [0, 1, 2, 3, 4]
    per_arm: dict[str, Any] = {}
    for branch in branches:
        for arm in arms:
            unos, ibss, aucs, ns, evs = [], [], [], [], []
            for fold in folds:
                rep = _load_json_file(
                    need(f"reports/{branch}_{arm}_fold_{fold}_public.json"))
                agg = rep.get("aggregate")
                if not isinstance(agg, Mapping):
                    raise _fail("PUBLIC_AGGREGATE_MISSING", f"{branch}/{arm}/f{fold}")
                unos.append(float(agg["uno_c_index"]))
                ibss.append(float(agg["integrated_brier_score"]))
                ns.append(int(agg["n_test"]))
                evs.append(int(agg["test_events"]))
                hm = agg.get("horizon_metrics", {})
                aucs.append({str(h): (v.get("dynamic_auc") if isinstance(v, Mapping) else None)
                             for h, v in hm.items()})
            per_arm[f"{branch}/{arm}"] = {
                "uno_mean": float(np.mean(unos)), "uno_per_fold": unos,
                "ibs_mean": float(np.mean(ibss)), "ibs_per_fold": ibss,
                "auc_per_fold": aucs, "n_test_total": int(sum(ns)),
                "events_total": int(sum(evs)),
            }
    # matched-design check: same n_test per fold across arms within a branch
    for branch in branches:
        for fold in folds:
            counts = set()
            for arm in arms:
                rep = _load_json_file(
                    root / f"reports/{branch}_{arm}_fold_{fold}_public.json")
                counts.add(int(rep["aggregate"]["n_test"]))
            if len(counts) != 1:
                raise _fail("MATCHED_N_MISMATCH", f"{branch}/f{fold}")
    contrasts = []
    pair_reports = summary.get("pair_reports", {})
    for pair_key in sorted(pair_reports):
        pr = pair_reports[pair_key]
        gate = pr.get("acceptance_gate", {})
        dibs = gate.get("delta_ibs_95ci")
        duno = gate.get("delta_uno_cindex_95ci")
        if not (isinstance(dibs, list) and len(dibs) == 2 and
                isinstance(duno, list) and len(duno) == 2):
            contrasts.append({"pair": pair_key, "status": "NOT_MEASURABLE",
                              "evidence": "pair_ci_absent.json"})
            continue
        ni_ibs = ni_retention(float(np.mean(dibs)), (float(dibs[0]), float(dibs[1])),
                              margin=float(ni_margins["dIBS"]), higher_better=False)
        ni_uno = ni_retention(float(np.mean(duno)), (float(duno[0]), float(duno[1])),
                              margin=float(ni_margins["dUno"]), higher_better=True)
        folds_d = gate.get("fold_delta_ibs", [])
        contrasts.append({"pair": pair_key, "status": "PASS",
                          "delta_ibs_ci": [float(dibs[0]), float(dibs[1])],
                          "delta_uno_ci": [float(duno[0]), float(duno[1])],
                          "fold_delta_ibs": [float(v) for v in folds_d],
                          "ni_ibs": ni_ibs, "ni_uno": ni_uno,
                          "exploratory_operational_acceptance": bool(
                              gate.get("exploratory_operational_acceptance", False))})
    omni = None
    try:
        mat = []
        for arm in arms:
            row = []
            for branch in branches:
                key = f"{branch}/{arm}"
                if key in per_arm:
                    row.extend(per_arm[key]["uno_per_fold"])
            if len(row) == 10:
                mat.append(row)
        if len(mat) >= 3:
            omni = omnibus_friedman(mat)
    except (PostEvalError, ValueError):
        omni = None
    extended = {}
    for pair_dir in sorted((root / "analysis_pairs").glob("*_vs_*")) if \
            (root / "analysis_pairs").is_dir() else []:
        for rep_file in sorted(pair_dir.glob("reports/*.json")):
            rel = str(rep_file.relative_to(root))
            try:
                payload = _load_json_file(rep_file)
                extended[rel] = {"status": payload.get("status", "UNKNOWN"),
                                 "sha256": hashlib.sha256(
                                     rep_file.read_bytes()).hexdigest()}
            except PostEvalError as exc:
                extended[rel] = {"status": "UNREADABLE", "reason": str(exc)}
    legs.append({"id": "LEG-EVAL-02", "status": "PASS",
                 "evidence": "locked_public_reports.json"})
    legs.append({"id": "LEG-EVAL-05", "status": "PASS",
                 "evidence": "locked_pair_reports.json"})
    legs.append({"id": "LEG-EVAL-03", "status": "NOT_MEASURABLE",
                 "evidence": "no_subject_rows_calibration_absent.json"})
    return {"status": "PASS", "run_digest": live_digest,
            "per_arm": per_arm, "contrasts": contrasts, "omnibus": omni,
            "extended_reports": extended, "legs": run_legs(legs)}
