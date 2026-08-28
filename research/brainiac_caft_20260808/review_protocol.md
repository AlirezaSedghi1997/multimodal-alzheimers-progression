# BrainIAC-CAFT R0 review protocol

Systematic scoping review with a PRISMA-style log.

- Review window: 2025-01-01 to 2026-08-08 (literature freeze).
- Target: 24-30 primary studies, at least 10 peer-reviewed.
- Preprints are held in a separate evidence tier and never merged into the
  peer-reviewed count.
- Per source, record endpoint, cohort, modality, preprocessing, split, external
  validation, confound audit, uncertainty handling, and code/weight availability.

## Screening depth

A source found by search has been **screened**, not read. The registry tracks
`screening_stage` per source and refuses to report the review complete until the
target number of sources is `FULL_TEXT_REVIEWED`. Counting search hits as
reviewed evidence is the easiest way to inflate a scoping review, so the depth
gate is independent of the count gate.

## Current state (corrected 2026-08-21)

| Measure | Value |
|---|---|
| Primary sources | 29 |
| Peer-reviewed | 13 |
| Preprints | 16 |
| Full-text reviewed with bound read evidence | 6 |
| Full text retrieved, review pending | 19 |
| Title/abstract screened only | 4 |
| Inconclusive citations | 1 |
| Meets count target | True |
| Status | `INCOMPLETE_PENDING_R0_COMPLETION` |

Blockers: FULL_TEXT_REVIEW_BELOW_TARGET

The count and peer-reviewed floors are met. The review is still incomplete:
retrieving a PDF/XML/note is not evidence that it was read. Registry v1 rounded
retrieval up to review for 19 records; append-only `evidence_registry_v2.json`
corrects those rows to `FULL_TEXT_AVAILABLE_PENDING_REVIEW`. Six explicitly
logged read events remain `FULL_TEXT_REVIEWED`. The remaining 18 reviews needed
for the floor are non-blocking for the five-CNN battery but mandatory before a
BrainIAC novelty argument is finalized.

## Claim ceiling

`NOVELTY_CANDIDATE` until the matrix is complete. All BrainIAC results carry
`POST_HOC_INTERNAL_TECHNICAL_EVIDENCE`.

## Closest prior art

`s19` (AJNR 2025) shares the endpoint and includes A4, with external validation
on an independent clinical cohort: T1-only AUC 0.61, T1+FLAIR 0.67 internal /
0.65 external, MCI 0.71. Any added-value claim must be argued against this, not
against ADNI-only numbers. See `novelty_matrix.json`.
