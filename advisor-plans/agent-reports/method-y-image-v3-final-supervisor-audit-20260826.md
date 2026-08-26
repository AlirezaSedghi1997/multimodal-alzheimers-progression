# Method Y Image v3.2 — Final Supervisor Audit Before Production

- Date: 2026-08-26
- Live repository: `H:\New project`
- Raw-data boundary: `H:\A4 Learn` remained read-only
- Active production writer during audit: none
- Long production training executed by this audit: no

## Decision

The prior handover commits `c42ef08` and `483aa81` were materially useful but
not sufficient for a scientifically safe production launch. Their seven main
repairs were present, yet the live launcher still had resume/integrity gaps.
Those gaps are now repaired and covered by adversarial tests. The invalidated
v1 result remains evidence of a protocol failure only; no v1 winner,
checkpoint, or outer result is reusable.

## Additional defects found and repaired

1. **Incomplete executable source closure.** The software contract omitted the
   readiness builder, launcher state machine, winner selection, statistical
   reporting/gates, atomic I/O, and writer guard. A one-byte change in those
   files could previously evade the training contract. The full executable
   dependency closure is now hashed.
2. **Readiness bootstrap was built by uncommitted code.** The readiness builder
   now explicitly binds the software/runtime contract and schema 3.2. The old
   bootstrap digest `7fa02033...` must not be used after this audit.
3. **Master-state self-hash was invalid after resumed transitions.** A resumed
   state retained its prior `state_sha256` inside the next digest. It is now
   verified on load, stripped from the mutable body, and recomputed atomically.
4. **Existing winner corruption/mismatch was swallowed.** The launcher could
   treat a bad winner as absent and run HPO again in the same root; it could
   also rerun HPO when a valid winner existed but the final checkpoint did not.
   Winner records are now self-hashed, identity-bound, contract-bound, and
   recomputed from completed inner-only candidate evidence. Any mismatch is
   terminal; HPO resumes only when no winner record exists.
5. **Committed outer result could leave an unrecoverable missing mirror.** The
   launcher skipped a `FINALIZED` controller before recreating the private
   mirror/report. It now always enters the controller recovery path: committed
   metrics are read without invoking the test-frame factory, then missing
   mirrors are regenerated.
6. **Private OOF mirrors were trusted by reports without integrity binding.**
   They now carry branch/variant/fold identity, run-contract digest,
   `metrics_sha256`, and a record self-hash. Reporting and final verification
   use the verified reader and compare the mirror to the controller result.
7. **Checkpoint verification was superficial.** Final verification now checks
   the actual bytes against the controller hash and checks the checkpoint's
   internal run-contract and non-empty model state. Inner checkpoint bytes are
   checked against every IPCW-IBS record.
8. **AMP could advance the LR schedule without updating weights.** Float16
   gradient overflow caused GradScaler to skip every optimizer step while the
   scheduler still advanced. Production AMP is now contract-locked to
   `bfloat16`; the discrete-time survival likelihood is computed in float32;
   zero finite optimizer steps fail the epoch; scheduler advances only after a
   real optimizer step. The CUDA interruption/resume test remains bitwise
   identical.
9. **Deterministic environment ordering was late.** CUBLAS determinism is now
   configured before the software binder's first torch import, including
   direct Python launches outside the PowerShell wrapper.

## Verification evidence

The new tests were first observed failing at all intended boundaries and then
made green. Final results on the corrected source tree:

```text
Focused critical/adversarial battery: 63 passed
Complete tests/method_y_revival:       475 passed
Failures:                              0
Warnings:                              0
```

The complete repository suite was not reclassified here. The earlier handover
reported unrelated Method X/redesign failures in the dirty repository; they do
not authorize weakening the Method Y verifier and are outside this scoped
repair.

## Pre-production artifact migration rule

No production v2 orchestration state or outer result existed when these source
changes were made. Therefore a one-time, provenance-recorded **pre-training
contract rebind** is allowed: preserve the old run contract/readiness hashes,
move only the obsolete bootstrap contract/readiness artifacts aside, rebuild
them from the committed source, and verify that membership/cache/split inputs
are unchanged. This exception is forbidden after any HPO/checkpoint/outer
artifact exists.

After the rebind, the source of truth is always:

```text
results/method_y_revival_v3_image_v2/run_contract.json
results/method_y_revival_v3_image_v2/readiness/FINAL_IMAGE_TRAINING_READINESS.json
```

Never copy a digest from this Markdown report into a run command.

## Status and claim boundary

- Plans 009 and 010 remain `VERIFIED_DONE`.
- Plans 011 and 012 remain `REVISE_REQUIRED` until real 5-fold OOF execution
  and the final verifier complete.
- Plan 013 remains `IN_PROGRESS_INVALIDATED_V1` until all 20 cells finalize.
- v2 internal image training is allowed only as an exploratory internal model.
- Pattern-normalized FBP is not official Centiloid; pattern-normalized FTP is
  not CenTauR.
- Scanner robustness, quantitative equivalence, and external validation remain
  unresolved claim gates.
- LEARN is internal transport, not external validation.
- No Alzheimer diagnosis, causal localization, quantitative equivalence, or
  external-generalization claim is permitted.

## Manual resumable production command

Only after readiness and GPU smoke pass on the committed/rebound contract:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File 'H:\New project\scripts\run_method_y_image_training_v3.ps1'
```

Re-running the exact command resumes the same contract from the last committed
epoch/fold/stage. A changed source, config, cache, membership, split, runtime
math policy, winner record, or checkpoint fails closed instead of restarting
inside the same output root.
