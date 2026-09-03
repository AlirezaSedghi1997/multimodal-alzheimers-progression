# Method Y Image v3/v4 — Master Handover for Next Agents

**Date:** 2026-09-03 | **Branch:** `method-y-post-revival-sanitized-20260812` | **Live repo:** `H:\New project` | **Raw data:** `H:\A4 Learn` (READ-ONLY)

You are an execution agent in a longitudinal MRI/PET survival research project.
This document is the complete state of the Method Y image branch, everything
done, everything remaining, and exactly how to continue. Read it fully before
any action. When in doubt, artifacts (hashes, manifests, state files, tests)
 outrank prose — including this document.

---

## 1. Non-negotiable scientific contract

- Two independent gates: `training_execution_allowed` (internal exploratory
  training may run) vs `positive_claim_allowed`/`external_claim_allowed`
  (remain FALSE until quantitative equivalence, scanner robustness and a
  locked external cohort independently pass). Never widen claims.
- Hard execution blockers (never repair with statistics): wrong tracer,
  same-PET/circular target, non-future/unlocked endpoint, subject leakage,
  invalid transform/geometry, NaN/Inf, invalid numeric scaling, inadequate
  fold event support, contract/hash mismatch. Three T1-invalid scans stay
  excluded forever.
- Image interpretation: FBP/FTP are PATTERN-NORMALIZED, NOT official
  Centiloid/CenTauR. v3.2/v4 are secondary POST_HOC_INTERNAL_EXPLORATORY
  analyses; source-regional v2 remains the primary local result.
- Real N never changes with augmentation: amyloid 560/110, tau 306/75.
- Forbidden: Alzheimer diagnosis, causal claims, official Centiloid/CenTauR
  equivalence, external generalization, LEARN as external validation.
- Rules: process guard before every write; RED test first; never
  reset/clean/checkout/branch-switch/mass-commit/push; never delete or
  overwrite v1/v2/raw; UNKNOWN/NOT_ESTIMABLE is never PASS; mismatch on the
  same output root = STOP, not auto-fix; no human sign-off required.

## 2. Versioned lineage (do not mix)

| Root | Config | Status |
|---|---|---|
| `results/method_y_revival_v3_image` | (old) | SUPERSEDED — never touch |
| `results/method_y_revival_v3_image_v1` | `method_y_image_training_v3_1.yaml` | INVALIDATED 2026-08-26 (4 reasons; fold_0 exposure recorded) |
| `results/method_y_revival_v3_image_v2` | `configs/method_y_image_training_v3_2.yaml` | **COMPLETE + VERIFIED**: final verifier PASS (20/20 cells, exactly-once, privacy, IPCW-IBS) |
| `results/method_y_revival_v4_image_factorial_v1` | `configs/method_y_image_factorial_v4.yaml` | **ACTIVE**. Training STOPPED by operator; resume point = last atomic epoch checkpoint (A1 mid-outer_0). Active run-contract digest `ec968b9b18978ab0508381c394eb2b7144c73cf0d79747599c47a43690abb602` |

v3.2 headline (exploratory, internal only): amyloid base 0.676C/0.097IBS vs
aug 0.695C/0.096IBS (dIBS CI includes zero → non-superiority retained); tau
base 0.616C/0.149IBS vs aug 0.706C/0.125IBS (internal improvement; claims
closed). Both branch claim policies + aggregate `IMAGE_CLAIM_POLICY.json`
exist under `results/method_y_revival_v3_image_v2/reports/`.

## 3. v4 factorial benchmark (Codex design, audited + completed by this session's fixes)

- 5 matched arms: A0 baseline / A1 +augmentation / A2 ResNet10-GroupN /
  A3 +aug+ResNet10 / A4 ResNet18-GroupN(width 0.25).
- PET-safe augmentation: rotation ≤1°, translation ≤0.5 voxel, train-only
  (inner/outer train), validation/test untouched, QC-locked (Pearson p5
  0.99194, Dice p5 0.99199; 2°/1-voxel envelope REJECTED; scale=1.0; PSF and
  count-noise disabled_not_evidenced). Contract:
  `results/method_y_revival_v4_image_factorial_v1/provenance/pet_augmentation_qc_contract.json`
- Selection: inner-only IPCW-IBS; one-SE over the 5 outer-training partitions;
  complexity-rank tie-break; deterministic seed subject×fold×epoch; outer
  test exactly-once after lock; per-epoch atomic checkpoints; 50-cell final
  verifier (2 branches × 5 arms × 5 folds... structured as arms × outer cells).

## 4. Contract-provenance fixes you must understand (the hard-won part)

### 4.1 The two recurrences of RUN_CONTRACT_MISMATCH
1. First: `software_contract_v4.contract_sha256` included the live git HEAD.
   Any commit (even docs-only) changed the digest → resume fail-closed.
2. Second (subtle): git metadata keys stayed inside the contract payload —
   including `git_provenance_status` = verbatim dirty-status text → digest
   still moved with HEAD *and* with working-tree dirt.

### 4.2 Final semantics (commit `7628305`)
- Digest binds ONLY to: bound-source bytes (`training_source.bundle_sha256`),
  package versions, CUDA runtime/hardware, and the locked
  runtime/model/factorial_arms/augmentation policies.
- ALL git metadata is a single `git_provenance` block marked
  `INFORMATIONAL_ONLY_NEVER_RESUME_BINDING`; excluded from digest and from
  `_ensure_run_contract` equality.
- Byte change in any bound source STILL flips the digest (fail-closed,
  adversarially tested) — see
  `tests/method_y_revival/test_software_contract_v4_binding.py`.

### 4.3 Frozen-chain migration (`scripts/migrate_method_y_image_factorial_v4_contract.py`)
- Per-cell/master/winner OLD digests are read FROM the artifacts themselves
  (authoritative labels). Never recompute them with today's formula — older
  generations embedded live git-status text and are not bit-reproducible.
- winner_*.json files migrate via digest-rewrite ONLY after
  `_reverify_winner_selection` proves the recorded winner reproducible from
  the inner-IBS candidates in the file (one-SE + rank). Winner self-hash
  (`winner_record_sha256`) is refreshed on rewrite; idempotent re-entry goes
  through the REPAIR_ONLY_CURRENT path.
- Outer results/access journals are ABSOLUTELY fail-closed: if any exist,
  migration refuses (no post-hoc outer manipulation ever).
- Applied chain: `72c2d83 → d521fa67 → a2013821 → ec968b9b`; current state is
  the last one. Provenance: timestamped JSONs under the root's `provenance/`.
- The same frozen semantics fixed the v3 verifier (`211dd49`): readiness binds
  to the frozen contract inside `run_contract.json`; final recovers per-fold
  digests from self-hashed `state.json`; `audit_post_execution_drift` is
  informational only.

## 5. Exact continuation commands (in this order)

```powershell
# 0. Process guard (never kill by name; judge the command line)
Get-CimInstance Win32_Process | Where-Object { $_.Name -match 'python|powershell|matlab|spm' -and $_.CommandLine -match 'method_y|PET|tau|amyloid|factorial_v4' } | Select-Object ProcessId,ParentProcessId,Name,CommandLine

# 1. Regression gates (all must be green)
& 'H:\New project\.venv\Scripts\python.exe' -m pytest -p no:cacheprovider tests\method_y_revival\test_software_contract_v4_binding.py -q
& 'H:\New project\.venv\Scripts\python.exe' -m pytest -p no:cacheprovider tests\method_y_revival\test_pet_factorial_runtime_v4.py tests\method_y_revival\test_pet_factorial_contract_v4.py tests\method_y_revival\test_pet_factorial_v4.py -q
# (48+6 tests; broader: tests\method_y_revival -q ≈ 510+)

# 2. Preflight (expect READY, digest ec968b9b...)
& 'H:\New project\.venv\Scripts\python.exe' 'H:\New project\scripts\run_method_y_image_factorial_v4.py' --mode preflight

# 3. OPTIONAL idempotence check (expect REPAIR_ONLY_CURRENT, no changes)
& 'H:\New project\.venv\Scripts\python.exe' 'H:\New project\scripts\migrate_method_y_image_factorial_v4_contract.py' --apply

# 4. Resume production (resumable at every epoch; interruption = same command again)
powershell -NoProfile -ExecutionPolicy Bypass -File 'H:\New project\scripts\run_method_y_image_factorial_v4.ps1'
```

Live monitoring:
```powershell
Get-Content "H:\New project\results\method_y_revival_v4_image_factorial_v1\runtime\logs\*.stdout.log" -Tail 20 -Wait
& 'H:\New project\.venv\Scripts\python.exe' 'H:\New project\scripts\run_method_y_image_factorial_v4.py' --mode status
```

Estimated remaining wall-time from telemetry: ~12-15 s/epoch on A0/A1 amyloid
on the RTX 4060 Laptop; full 5-arm × 2-branch programme ≈ 18-30 h total.
Only ONE writer may run at a time; never launch a second copy while one runs.

## 6. After the production run completes (the actual remaining work)

1. **50-cell completion check**: final verifier for v4 must require every
   (arm × outer) cell committed exactly-once with valid IPCW-IBS inner
   records. It intentionally FAILs until then (current status output shows
   which cells are missing). Locate the verifier:
   `scripts/verify_method_y_image_factorial_v4.py` (contract-bound; do not
   casually rewrite — extend, don't weaken).
2. **Factorial reporting**: `src/method_y_revival/image_factorial_reporting_v4.py`
   (`build_factorial_pair_reports`) produces the paired A0-vs-A1..A4 OOF
   comparisons + POST_HOC_INTERNAL_EXPLORATORY claim stamp. Robustness reports
   for v4 mirror v3.2's executor (scanner probe on OOF embeddings, LMO
   eligibility, subgroups with NOT_ESTIMABLE for low support; NEVER pool
   GE+Philips; UNKNOWN stays UNKNOWN).
3. **Acceptance criteria for ResNet/augmentation** (from the locked plan): a
   challenger is accepted only if IBS non-inferior-or-better, Uno-C and
   calibration not worse, scanner probe and LMO not worse, stable across all
   5 folds, and no train-validation gap blow-up. Otherwise record the
   negative result verbatim — negative results are publishable, reruns for
   significance are not.
4. **Privacy/claim gates**: run the recursive privacy scan over public
   reports; `IMAGE_CLAIM_POLICY` per branch + aggregate; positive/external
   claims stay FALSE. v1 fold_0 exposure stays excluded from any aggregation.
5. **Status flips**: only after the final verifier PASS may
   `advisor-plans/README.md` rows and memory index move from
   IN_PROGRESS_INVALIDATED_V1 to DONE — citing verifier output, never prose.

## 7. Known repo-wide caveats

- The worktree is intentionally dirty with unrelated (Brainiac/Method X)
  changes; NEVER stage beyond the Method-Y scope. Verify with
  `git status --short` before any commit.
- Full-repo `pytest tests -q` has ~120 pre-existing out-of-scope failures
  (legacy MX `train_3d_mri` import surface + redesign-stage modules) plus 3
  MX collection errors. Do not "fix" them inside a Method Y session; document
  and move on.
- `results/` is gitignored — provenance JSONs worth preserving must be
  mirrored to a tracked path (see `governance/method_y_revival_v3_image_v1_invalidation_20260826.json`).
- Global Codex memory is outside the repo; this ledger + memory index are the
  in-repo authority (see `docs/PROJECT_MEMORY_INDEX.md` supersession block).

## 8. Key artifact map

- Ledger: `docs/plans/agent_session_ledger_20260815.md` (latest appends at top)
- Memory index: `docs/PROJECT_MEMORY_INDEX.md` (latest sections at top)
- v3.2 verifier: `scripts/verify_method_y_image_training_v3.py` (`--level readiness|final`, `--output-root`)
- v4 launcher: `scripts/run_method_y_image_factorial_v4.py` (modes preflight/run/status) + `.ps1`
- v4 migration: `scripts/migrate_method_y_image_factorial_v4_contract.py` (`--dry-run|--apply`)
- v4 contracts: `src/method_y_revival/software_contract_v4.py`, cell/master
  builders in `src/method_y_revival/image_factorial_orchestrator_v4.py`
- v3 contract/orchestration: `src/method_y_revival/software_contract_v3.py`,
  `image_training_orchestrator_v3.py`, `image_outer_eval_v3.py`,
  `image_launcher_state_v3.py`, `image_reporting_v3.py`
- Tests: `tests/method_y_revival/test_software_contract_v4_binding.py`,
  `test_pet_factorial_*.py`, `test_verifier_frozen_contract_v3.py`,
  `test_method_y_image_v3_supervisor_repairs.py`, `test_audit_round3_v3.py`,
  `test_audit_round4_supervisor_plan.py`
- Reports so far: `advisor-plans/agent-reports/method-y-image-v3-*.md`

## 9. Session commit trail (this branch, newest first)

- `7628305` v4 contract fully HEAD/dirty-independent + winner-aware frozen-chain migration
- `675ae22` ledger: v4 HEAD-independence fix
- `bebad7b` v4 contract HEAD-independence (first pass)
- `5d932d6` docs: v3.2 VERIFY PASS + v4 handover
- `211dd49` verifier frozen-contract semantics (v3.2 final PASS)
- `483aa81`/`c42ef08` supervisor-plan repairs + docs (v2 root, invalidation)
- Earlier: `7a50e84` STAGE_ORDER resume fix, `70372cd` audit round-3, `8a2ea26`/`d680748` audit round-2, `7c30be5` hard-audit 009-013

**First action for the next agent:** section 5, step 0 (process guard), then
step 1 (tests), then resume production. Do not redesign contracts — they are
now correct; extend only with RED tests first.
