> Traceability audit 2026-09-06 — every prior finding mapped to 03-combined-plan.md (+A1) and to code/tests, verified by grep/read (not memory). Verdicts: COVERED (with pointers) or GAP→fix (implemented same session in post_eval_v4.py unless noted).

# 04 — Finding → Plan → Code Traceability

## Supervisor round 1 (adversarial review)
| ID | Verdict | Evidence |
|---|---|---|
| C1 LMO rename | COVERED | plan E6; code `stratified_manufacturer_report` (gate_input False) + test |
| C2 probe audit | GAP→FIXED | new `audit_probe_weights()` (inner-train digest required else PROBE_REUSE_VIOLATION) + test |
| C3 NI margin | COVERED | plan E10 + A1; `ni_retention` + config margins dIBS 0.01/dUno 0.02 + tests all 3 branches |
| C4 calibration descriptive | COVERED | `calibration_at_horizon` + config `descriptive_only: true` + test |
| C5 two attesters | COVERED | `validate_attestation` + test |
| C6 thesis writer | COVERED | `thesis_write` (allowlist/pointers/ceiling/boilerplate) + tests |
| H1 tau*/IPCW | COVERED | `tau_star` (no outer clause) + `ipcw_weights` (cap+diagnostics) + tests |
| H2 stratified paired + BCa + permutation | COVERED | `paired_bootstrap_delta` + `permutation_paired_p` + tests |
| H3 bars/multiplicity | COVERED+P1: `subgroup_decision` (25/10) + k-suppression + NEW `bh_fdr` + hierarchy (omnibus→Holm/BH) |
| H4 MixedLM | GAP→FIXED | new `mixedlm_pacc()` (locked formula/maxiter/tol, singular→NOT_MEASURABLE) + `missingness_table()` + tests |
| H5 intake lint | GAP→FIXED | new `intake_lint()` (static schema only, no fixtures) + test |
| H6 CAM locks | COVERED in CAM build; GAP→FIXED here: new `cam_gates_read()` validating attribution report gates |
| H7 fusion keys | GAP→FIXED | new `assert_fusion_keys()` (schema + one-row-per-subject dedup + digest) + test |
| H8 IPCW lock | COVERED+P1: signature takes train arrays + NEW `selection_stability()` reader over winner records |
| M1 D-cal rule | GAP→FIXED | new `d_calibration()` (statistic+p+bin counts, never-gate rule documented) + test |
| M2 PRNG | COVERED | `spawned_rng` + `lib_versions` + test |
| M3 suppression | COVERED | `privacy_cell_check` + `scan_text` + tests |
| M4 UNKNOWN propagation | GAP→FIXED | `run_legs` now accepts UNKNOWN as recorded-nonhalting (was BLOCKED-halt); exploratory stamp never overrides |
| M5 firewall | GAP→FIXED | new `firewall_assertion()` over journal records + LEG-EVAL-00 wiring |
| M6 min-n E7-O2 | COVERED | `subgroup_decision` reused; application point documented in runner |

## Brief A (metrics)
| Item | Verdict |
|---|---|
| Uno/Harrell/Antolini ranking | COVERED+P1: `uno_c`, `harrell_c` + NEW `antolini_ctd` (exploratory) with tests |
| IBS grid/Brier/scaled | COVERED (`ibs_summary`) |
| C/D AUC + mean | COVERED (`dynamic_auc`); I/D recorded NOT_IMPLEMENTED (unstable at 75 events per brief itself — honest scope cut, in plan) |
| Calibration slope/ICI | COVERED; Emax in output dict |
| Determinism checklist | COVERED: tau/grid/risk(`1-S(tau)`)/tie 0.5/IPCW-marginal-train/seeds/versions/B/cap — tie_rule + risk_score ADDED to config in this fix round |

## Brief B (methodology)
| Item | Verdict |
|---|---|
| Paired bootstrap/BCa/permutation | COVERED (see H2) |
| Bates direction | COVERED+P1: `selection_stability()` + variance-exclusion statement attached by runner |
| LMO descriptive | COVERED (C1) |
| Subgroup interaction-first | GAP→FIXED: NEW `interaction_did()` (difference-in-differences with bootstrap CI; avoids Cox-API risk, deterministic) + test |
| TRIPOD/CLAIM machine | COVERED (`tripod_machine_checks`); judgement = attestation (C5) |
| Negatives template | COVERED (`ni_retention` + full-table built by runner from legs) |
| Omnibus→Holm hierarchy | GAP→FIXED: NEW `omnibus_friedman()` (scipy, deterministic) + `bh_fdr()` + tests |

## R-C brief (similar studies)
| Lesson | Disposition |
|---|---|
| Survival framing support (JAD/Brain 2026) | Cited in plan rationale; no code action (framing already survival) |
| Aggregate voxel attribution precedent (2017 ensemble) | Supports CAM-build reporting style; no action |
| Augmentation-isolation pairs | Already locked in v4 factorial design (A0/A1, A2/A3); referenced, no action |
| PACC-LME off-target / sanity-API gaps / tau-fetch MISS | Recorded as UNCERTAIN in 02-pack; E9 stays descriptive; no invented landmarks |

## Files changed this round (all NEW or own-tracked; 25 bound sources untouched)
- `src/method_y_revival/post_eval_v4.py` (extended; NOT in contract SOURCE_PATHS — verified)
- `tests/method_y_revival/test_post_eval_v4.py` (extended)
- `configs/method_y_post_eval_v4.yaml` (new locked fields)
- `scripts/run_method_y_post_eval_v4.py` (production cell loop over parameterized root + fixture-tested)
