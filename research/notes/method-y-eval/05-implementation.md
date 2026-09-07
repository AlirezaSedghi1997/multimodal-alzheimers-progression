> Implementation record 2026-09-06 — final outputs of the 5-stage program (CAM-adapted evaluation).

# 05 — Final Deliverables (ready to apply post-verifier-PASS)

## Files
| File | Role | Evidence |
|---|---|---|
| `src/method_y_revival/post_eval_v4.py` | Eval library: IPCW/tau*/Uno/Harrell/AUC/IBS/calibration/ICI/bootstrap+BCa/permutation/NI/risk-tables/subgroups/stratified-manufacturer/TRIPOD-checks/attestation/thesis-writer/privacy/legs/firewall/probes-audit/BH/omnibus/MixedLM/intake/CAM-gates/fusion-keys/stability/D-cal/interaction/Antolini/bundle-evaluator | 35 tests |
| `tests/method_y_revival/test_post_eval_v4.py` | 35 RED-first tests, CPU, synthetic | 35 passed |
| `scripts/run_method_y_post_eval_v4.py` | preflight/dry-run/run; fail-closed chain | dry-run PASS; refusal proven |
| `configs/method_y_post_eval_v4.yaml` | All locked numbers (horizons, tau* rule, B=2000/seed 13, NI 0.01/0.02, MixedLM formula+criteria, D-cal bins, k≥10/≥10, subgroup list, BH) | parity-checked |
| `research/notes/method-y-eval/01..04` | Initial plan, knowledge pack, combined plan+A1+A2, traceability audit | 3 supervisor rounds |

## Executed evidence (all CPU, zero training contact)
- 35/35 unit tests GREEN (incl. fixture end-to-end bundle evaluation).
- Full dry-run PASS: synthetic OOF bundle → per-fold metrics → paired contrasts → NI → legs → thesis → report.
- Production refusal proven (no preflight → BLOCKED).
- Supervisor rounds: R1 (6C/8H/6M) → R2 (17 findings, all applied) → R3 final (all PASS, no blockers).

## Run order post-verifier (operator)
1. `--mode preflight` → PASS required.
2. `--mode run` (fail-closed chain; legs LEG-EVAL-00..16; BLOCKED halts, NOT_MEASURABLE/TIMEOUT recorded).
3. Thesis writer → `docs/thesis_traceability/method_y_post_training/` + limitations appendix.
4. Privacy + k-suppression → public bundle → claims compiler (positive/external FALSE).

## Known honest limits (not hidden)
- Tau arm-cells (~3 events) and 1640d tail will be NOT_MEASURABLE/descriptive in most cells.
- Bootstrap CIs are conditional (test noise only, not training/tuning variance) — stated in every report.
- MixedLM descriptive; MAR assumed + dropout audit attached.
- I/D AUC not implemented (documented scope cut).
