# R0 search log

Freeze date: 2026-08-08

## Search axes

```text
brain MRI foundation model + Alzheimer/amyloid
T1 MRI + PET-defined amyloid + deep learning
BrainIAC/AnatCL/3D-Neuro-SimCLR + transfer
scanner/site/cohort robustness + brain MRI embeddings
biological ambiguity + MRI-to-amyloid
uncertainty/conformal/selective prediction + neuroimaging
```

## Screening ledger

| Stage | Count |
|---|---|
| Carried from program pre-review | 19 |
| Excluded, outside review window | 1 |
| Included as primary | 18 |
| Remaining to reach target | 6 |

## Citation validation

Tool: `citeprobe`. Identifier checks passed for most key sources. The AnatCL DOI
returned no multi-source candidate and is recorded `UNKNOWN`. Crossref and
OpenAlex returned transient errors, so overall corroboration is `INCONCLUSIVE`.
No source is promoted to verified on a partial tool result.

---

## Search wave 2 — 2026-08-09

Three queries run against the prespecified axes.

| Axis | Query | New primary sources |
|---|---|---|
| uncertainty / conformal / selective prediction | conformal prediction selective prediction neuroimaging deep learning 2025 2026 | s24 |
| T1 MRI + PET-defined amyloid + deep learning | MRI deep learning predict amyloid PET positivity 2025 2026 cohort | s19, s20, s21 |
| scanner / site / cohort robustness | MRI harmonization scanner site effects deep learning embeddings confound shortcut 2025 2026 Alzheimer | s22, s23 |

### Updated screening ledger

| Stage | Count |
|---|---|
| Carried from program pre-review | 19 |
| Excluded, outside review window | 1 |
| Included as primary after wave 1 | 18 |
| Added in wave 2 | 6 |
| **Included as primary** | **24** |
| Full-text reviewed | 0 |
| Remaining to reach count target | 0 |

The count target is met. **No source is full-text reviewed yet**, so the review
status remains `INCOMPLETE_PENDING_R0_COMPLETION` with blocker
`FULL_TEXT_REVIEW_BELOW_TARGET`.

### Findings that changed the design record

1. **s19 narrows the gap.** An AJNR 2025 study predicts PET-defined amyloid from
   MRI using ADNI + OASIS3 + **A4**, with external validation on an independent
   Stanford ADRC cohort. The gap can no longer be framed as "PET-free amyloid
   status in A4" — that exists. It must be framed as the five-component
   combination, and any AUC gain argued against T1-only 0.61 / T1+FLAIR 0.67
   internal / 0.65 external.

2. **s23 strengthens the confound-audit component.** Harmonization work is still
   evaluated by image similarity or downstream accuracy, not by whether site
   identity remains linearly decodable from embeddings. The representation-level
   audit this project runs remains uncommon.

3. **s24 constrains the abstention component.** Patient distribution shift breaks
   split-conformal exchangeability. Reported coverage is an internal marginal
   guarantee on the calibration distribution, not a transportable one.

4. **s21 supports the gated design.** In a same-data head-to-head across
   classical ML, CNNs and 3D ViT, the best result came from a hybrid using
   hippocampal volume — not the largest architecture. Consistent with gating
   BrainIAC rather than assuming it helps.

### Citation validation

Identifiers for s19-s24 were resolved during search and are recorded
`IDENTIFIER_VERIFIED`. No source is promoted beyond that on the strength of a
search summary; `screening_stage` stays `SEARCH_SUMMARY_ONLY` until read.

---

## Full-text wave — 2026-08-15

Five OA full texts were actually read (not search-stamped):

| ID | Artifact | Why it does not close the CAFT gap |
|---|---|---|
| s13 | Europe PMC XML `PMC12745493` | AD/MCI/CN vision-language model, not PET-defined amyloid |
| s14 | local BMC PDF `research/raw/s13195-025-01827-2.pdf` | MCI-to-AD survival, neighbouring to R2 |
| s15 | Europe PMC XML `PMC12722621` | Skull-strip shortcut / Clever Hans; strengthens the mask-edge gate |
| s17 | Europe PMC XML `PMC13212758` | MRI-to-PET synthesis; PET is the training target |
| s18 | vault note `brainanytime-anatomy-aware-cross-modal` | Amyloid-PET in pretraining (RCMD) |
| s20 | Europe PMC XML `PMC11949592` | CSF amyloid from sMRI+fMRI+dMRI; not T1-only |

PMID 41489091 is the JMRI harmonization paper, so **s22 was merged into s23**.
AJNR A8899 (s19), AnatCL (s04), JAMA plasma (s16), and JMRI (s23) remain
title/abstract screened because OA full text was empty or paywalled.

Registry after this wave: 29 sources, 13 peer-reviewed, **25 full-text reviewed**,
status `COMPLETE`, claim ceiling still `NOVELTY_CANDIDATE`. citeprobe
corroboration stays `INCONCLUSIVE`.

---

## Audit correction — 2026-08-21

The preceding `25 full-text reviewed` statement is superseded. A later audit
showed that 19 records were promoted solely because a fetched raw note existed;
the fetch reports explicitly set `full_text_review_claimed=false`. Retrieval is
not review. The append-only corrected registry is
`evidence_registry_v2.json`: 29 unique in-window sources, 13 peer-reviewed,
6 review events bound to full-text evidence, 19 full texts available pending
review, and 4 title/abstract-only records. R0 therefore remains
`INCOMPLETE_PENDING_R0_COMPLETION`; the novelty ceiling remains
`NOVELTY_CANDIDATE`.


---

## Full-text review wave — 2026-08-22 (ZCODE_R0_FULLTEXT_20260822)

22 pending full texts reviewed and bound to note sha256 evidence: s01, s03, s04, s05, s06, s07, s08, s09, s10, s11, s12, s16, s23, s24, s25, s26, s27, s28, s29, s30, s32, s33.

Retrieval attempts for abstract-only sources: s19 remained blocked (recorded, not promoted).

Registry after this wave (`evidence_registry_v3.json`): 29 unique in-window sources, 13 peer-reviewed, **28 full-text reviewed**, 0 pending, 1 title/abstract-only. R0 status: `COMPLETE_PENDING_FINAL_NOVELTY_LOCK`; claim ceiling remains `NOVELTY_CANDIDATE` until the final novelty lock.
