# Agent Loop Engineering Policy

Status: active project memory.

This project uses loop engineering, not only prompt engineering. Each agent
iteration must be a bounded observe -> decide -> act -> verify -> record cycle.

## Loop State

- Primary state files: `.agent/PROGRESS.yaml`, `SESSION_HANDOFF.md`.
- Metrics source of truth: generated JSON under `results/tables/` and
  `results/final_results/`.
- Method X protected contract: T1 MRI plus age, sex, education, APOE4, and
  MMSE only; PET and synthetic PET are never Method X inputs.
- Raw data remain read-only and are never published.

## Context Budget

- Prefer compact state, manifest, and metric summaries.
- Do not paste raw training logs, raw JSON vectors, raw paths, cache contents,
  HDF5, images, checkpoints, raw IDs, or subject-level predictions into public
  outputs.
- Use RTK when installed; otherwise emulate RTK manually by reading only the
  schema, summary, failure, or final metric lines needed for the decision.

## Action Rule

One iteration should make one bounded change:

1. Observe current state, dirty tree, and the relevant source-of-truth metric.
2. Decide the highest-risk unresolved task.
3. Patch or run one bounded step.
4. Verify with targeted tests, privacy checks, and scientific consistency gates.
5. Record the result in `.agent/PROGRESS.yaml` and `SESSION_HANDOFF.md`.

## Verifiers

- Tests: targeted `pytest` after meaningful edits; full `pytest -q` before
  finishing.
- Privacy: public bundle manifest must report no raw data, raw paths, HDF5,
  images, raw identifiers, or subject-level vectors.
- Scientific gates: Method X PET-free; Method Y PET-assisted exploratory with
  PET-label tautology caveat; synthetic MRI-to-PET train-fold-only; augmentation
  counted as train exposure multiplication, never real-N expansion.
- Claim control: thesis title framing is early prediction of amyloid biomarker
  status and preclinical risk, not clinical Alzheimer diagnosis.

## Stop Conditions

- Stop as complete only after the requested artifacts are rebuilt and tests pass.
- Stop as blocked only when the same blocking condition repeats and no useful
  bounded action remains.
- Stop for budget only after recording what passed, what is incomplete, and the
  next exact command.

## Subagent Policy

Use subagents only when their work is independent and worth the extra context:
literature carding, privacy audit, Method Y review, 3D training review, or
scientific claim audit. Merge only summarized findings into the main loop.

## Literature Memory

Paper cards live in `docs/literature_loop_cards.md`. Each card uses ten fields:
problem, data, modalities, architecture, leakage risk, validation, metric,
claim boundary, code/repo, and project decision. These cards are the RAG memory
surface; do not re-read large paper sets when the card already answers the loop
question.
