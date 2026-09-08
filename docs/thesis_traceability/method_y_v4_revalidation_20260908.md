# Method Y v4 — Re-validation After Attribution Repairs (append-only, 2026-09-08)

> Question answered: are the recorded final results still the valid results after
> the attribution-stack repairs, or stale pre-fix numbers?
> Answer: **they stand, proven by re-execution — not assumed.**
> Ceiling unchanged: `POST_HOC_INTERNAL_EXPLORATORY`. Zero thesis corrections required.
> Supervision: D (protocol, approve-with-7-amendments) → execution → E (evidence, PASS) →
> own cross-check (layer 3). All amendments adopted.

## 1. Why re-validation was required (and what could NOT have changed)

- The attribution repairs (commit `c6ccef5`) added/fixed NEW analysis files only
  (attribution module/launcher/config/tests). They never import, retrain, reload, or
  rewrite survival checkpoints, metrics, splits, or membership — so survival numbers are
  unchanged **by construction**. Construction arguments are not evidence, hence this
  re-execution.
- Caveat carried (D-A3): the live recompile used the UNCOMMITTED R2-dirty post_eval code
  (101/24 + 80/6 + 53/1 lines vs HEAD `c6ccef5`); the claim is therefore
  "reproducible under R2-dirty code", not code-independent. Verifier script and locked
  factorial config were verified pristine (empty diff). Gate-semantics relaxation in the
  dirty tree (live-verify fallback, split scan/ban) is recorded as an intentional R2
  change (D-A5), not inherited silently.

## 2. Protocol (D-approved, all 7 amendments adopted)

- P1 baseline hashes (first-16): summary `5302f90f`, sanitized `701c197f`,
  post_eval_report `bf963e82`, locked preflight `923fa4d2`, orch_state `9ced5ca3`,
  run_contract `01430160`; no `final_verification_pass.json` (live-verify path).
- P2 verifier (pristine): `--level readiness` + `--level final`, CPU, read-only.
- P3 post_eval `preflight` + `run` BOTH to scratch
  `post_eval/_revalidation_20260908/` (A1: never default output — locked
  `eval_preflight_pass.json` untouched). Three-tier compare (A2): byte → canonical →
  science at 1e-12. Joint gate for STAND (A4). Pre/post mutation guard (A6).
  Explicit allowlist commit (A7); post_eval dirty files stay UNCOMMITTED.

## 3. Results (2026-09-08 live re-execution)

| Check | Outcome |
|---|---|
| Verifier readiness | PASS (amyloid 560/110, tau 306/75, coverage full) |
| Verifier final | PASS (50/50, `failures=[]`, claims closed) |
| post_eval run (scratch) | PASS (legs 00/02/05 PASS, 03/12 NOT_MEASURABLE; thesis PASS; privacy PASS) |
| Scratch vs locked report | **BYTE-identical** (`bf963e82…` both, 11975 B) → canonical-identical → **zero science mismatches** (10/10 contrasts: delta CIs, fold vectors, ni deltas+decisions; omnibus `p=0.1007760268867577 stat=7.759999999999991`; legs/thesis/privacy equal; `thesis_results.md` identical) |
| Locked-state mutation | NONE (all 6 hashes identical pre/post; locked preflight `923fa4d2` untouched; results/ git-ignored so mtime+hash guarded) |

## 4. Thesis confirmation (no corrections)

- IBS `RETAIN_NEGATIVE` 9/10 (only tau A3→A4 `INCONCLUSIVE_UNDERPOWERED`,
  `+0.01603990695177199 [0.005897799225448076, 0.026182014678095906]`);
- Uno `RETAIN_NEGATIVE` exactly `amyloid:A0_vs_A1`, `amyloid:A2_vs_A3`, `tau:A2_vs_A3`;
- Omnibus `p=0.101`; ceiling/privacy/legs as recorded.
- The numbers in `method_y_v4_results_20260908.md` are therefore **current post-fix
  results**, re-validated — E verdict PASS, `stand_scope_approved=true`,
  `thesis_corrections_required=[]`.
- Scope (D-A4/E-e2): STAND covers the post_eval recompilation layer + verifier final;
  it does not re-prove splits/membership (unchanged hashes are the evidence there) and
  does not bless attribution production (still DRAFT-gated by design).
