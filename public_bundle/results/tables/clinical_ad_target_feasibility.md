# Clinical Alzheimer Target Feasibility

- status: `FEASIBILITY_ONLY_NOT_TRAINED`
- training started: `False`
- PET preprocessing started: `False`
- Method X target remains: `A-beta positive vs A-beta negative biomarker status`
- clinical Alzheimer output allowed now: `False`
- independent clinical label required: `True`
- subject-level outputs exported: `False`

## Availability

| candidate target | availability status | supported now |
|---|---|---:|
| CN/MCI/AD dementia classes | NOT_DETECTED_AS_DIRECT_LABEL | False |
| diagnosis at visit | CDR_SCALE_AVAILABLE_NOT_DIAGNOSIS | False |
| conversion follow-up | CANDIDATE_ENDPOINT_REQUIRES_PROTOCOL_LOCK | False |
| adjudicated diagnosis | NOT_DETECTED | False |

## Aggregate CDR Coverage

- locked catalog patients: `1700`
- locked patients with CDR: `1700`
- locked CDR rows matched in memory: `10858`
- patients with two or more CDR timepoints: `1621`
- first-observed Global CDR counts: `{"0": 1700}`
- latest-observed Global CDR counts: `{"0": 1201, "0.5": 424, "1": 58, "2": 17}`
- conversion candidate counts: `{"first_cdr_missing": 0, "first_cdr_nonzero": 0, "first_cdr_zero_with_later_positive_global_cdr": 720, "first_cdr_zero_without_later_cdr": 94, "first_cdr_zero_without_later_positive_global_cdr": 886}`

## Governance Rule

A clinical `Alzheimer disease yes/no` output remains blocked until an independent clinical diagnosis or conversion label is defined, locked, and validated. Biomarker status must continue to be reported as `A-beta positive vs A-beta negative`, not as a definitive clinical diagnosis.

## Leakage Boundary

PET, SUVR, centiloid, tau, and PET-derived features remain forbidden as Method X or Method X-Cog inputs. This artifact is aggregate feasibility only and does not start PET preprocessing, PET modeling, MRI 3D ablation, fusion, XAI refresh, or final analysis refresh.
