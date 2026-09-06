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


def _surv(time, event):
    from sksurv.util import Surv
    return Surv.from_arrays(np.asarray(event, dtype=bool), np.asarray(time, dtype=float))


def uno_c(train_time, train_event, test_time, test_event, risk, tau: float):
    from sksurv.metrics import concordance_index_ipcw
    ytr = _surv(train_time, train_event)
    yte = _surv(test_time, test_event)
    c, *_ = concordance_index_ipcw(ytr, yte, np.asarray(risk, dtype=float), float(tau))
    return {"c": float(c), "tau": float(tau)}


def harrell_c(time, event, risk):
    from sksurv.metrics import concordance_index_censored
    c, *_ = concordance_index_censored(np.asarray(event, dtype=bool),
                                       np.asarray(time, dtype=float),
                                       np.asarray(risk, dtype=float))
    return {"c": float(c)}


def dynamic_auc(train_time, train_event, test_time, test_event, risk, times):
    from sksurv.metrics import cumulative_dynamic_auc
    ytr = _surv(train_time, train_event)
    yte = _surv(test_time, test_event)
    auc, mean_auc = cumulative_dynamic_auc(ytr, yte, np.asarray(risk, dtype=float),
                                           np.asarray(times, dtype=float))
    return {"auc": [float(v) for v in auc], "mean_auc": float(mean_auc),
            "times": [float(v) for v in times]}


def ibs_summary(train_time, train_event, test_time, test_event, surv_prob, times):
    from sksurv.metrics import integrated_brier_score, brier_score
    ytr = _surv(train_time, train_event)
    yte = _surv(test_time, test_event)
    p = np.asarray(surv_prob, dtype=float)
    tm = np.asarray(times, dtype=float)
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
            "brier_curve": [float(v) for v in curve], "times": [float(v) for v in tm]}


def calibration_at_horizon(time, event, pred_surv, horizon: float, *,
                           train_time, train_event, n_bins: int = 5):
    """IPCW-logistic slope/intercept + binned ICI descriptors (descriptive)."""
    import statsmodels.api as sm
    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=bool)
    pred = np.asarray(pred_surv, dtype=float)
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
    """BLOCKED halts; NOT_MEASURABLE/TIMEOUT recorded, pipeline continues."""
    done = []
    for leg in legs:
        status = str(leg.get("status", ""))
        if status == "BLOCKED":
            return {"status": "BLOCKED", "halted_at": str(leg.get("id", "?")),
                    "reason": str(leg.get("reason", "")), "completed": done}
        if status not in ("PASS", "NOT_MEASURABLE", "TIMEOUT"):
            return {"status": "BLOCKED", "halted_at": str(leg.get("id", "?")),
                    "reason": "LEG_STATUS_INVALID", "completed": done}
        done.append(str(leg.get("id", "?")))
    return {"status": "PASS", "completed": done}


def entry_gate(*, run_root: str | Path) -> dict[str, Any]:
    """Refuse unless a final-verification PASS artifact binds live state."""
    root = Path(run_root)
    cand = root / "final_verification_pass.json"
    if not cand.is_file():
        raise _fail("VERIFIER_NOT_PASS", "final_verification_pass.json missing")
    try:
        payload = json.loads(cand.read_text(encoding="utf-8"))
    except (OSError, ValueError, UnicodeDecodeError) as exc:
        raise _fail("VERIFIER_NOT_PASS", "unreadable") from exc
    body = dict(payload)
    recorded = body.pop("integrity_sha256", None)
    canon = json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    if payload.get("status") != "PASS" or recorded != hashlib.sha256(canon).hexdigest():
        raise _fail("VERIFIER_NOT_PASS", "integrity_or_status")
    return {"status": "PASS", "run_contract_sha256": payload.get("run_contract_sha256")}
