# P1 independent review — Tau provenance commits `cf0fc23` / `a0161d6`

Date: 2026-08-15
Scope: code and public aggregate only. No NIfTI re-scan. No training.

## Commits

- `cf0fc23` Fix Tau audit identity and privacy gates (deleted working-tree parquet; blob remains in leaky history)
- `a0161d6` Harden Tau provenance gate audit

Publishable twin without private history: `methody/pretraining-sanitized-v2`.

## SPEC

`PASS` for the public gate semantics:

- Exact scan identity remains 0/1314.
- BID/VISCODE is subject/visit association, not identity. Live field name is `a4_linear_suvr_subject_visit_association_scan_count`.
- Null/false/placeholder UIDs are not counted as identity.
- Public JSON/Markdown are aggregate and path-free.
- `science_gate_status=BLOCKED`, `claim_status=NO_CLAIM`.
- Private row output is rejected inside the module on any git ancestor, not only the CLI.

## QUALITY

`PASS` for the audit implementation against those commits, with the isolation caveat:

- Live HEAD still reaches blob `2be6257e1d7f946805bc6eff09d42ea47983a152`. That is a P0 publishability FAIL, not a Tau science PASS.
- Do not promote Tau claim from this review.
- Do not treat A4 association counts as exact FTP identity.

## Verdict

Tau quantitative science gate stays `BLOCKED` / `NO_CLAIM`. No new identity evidence was introduced by this repair.
