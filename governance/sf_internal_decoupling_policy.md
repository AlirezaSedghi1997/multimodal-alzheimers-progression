# SF Internal Decoupling Policy

Status: **FROZEN_BEFORE_MODELING**  
Date: 2026-06-07  
Applies to: Method X-Cog/SF and SF-aware validation analyses

## Purpose

The screen-fail (SF) group is an **internal decoupling and stress-test instrument** within the A4/LEARN dataset family. It is used to test whether PET-free clinical/cognitive signal survives cohort-label coupling in a single screening context where both Aβ-positive and Aβ-negative subjects coexist.

## Required terminology

Use:

- internal decoupling / stress-test instrument
- SF-aware internal stress-test
- SF-inclusive screening analysis

Never use:

- external validation
- quasi-external validation
- independent validation
- external cohort

## Role relative to Method X-Strict

- **Method X-Strict** remains the protected thesis anchor (T1 MRI + fixed tabular only; A4 + LEARN enrolled MRI sample).
- **Method X-Cog/SF** is scientifically central for confounding analysis but does **not** replace Method X-Strict.
- SF must **never** be described as an MRI external validation cohort; T1 MRI overlap with SF is weak or essentially absent.

## Label policy in SF-inclusive analyses

- Primary binary label: frozen centiloid threshold (see `results/governance/label_freeze.json`).
- Sensitivity label: VA / visual-read consensus (`overall_score` in PET VA table).
- Document explicitly the discrepancy between SF visual-read positives and centiloid-threshold positives.
- Centiloid, SUVR, and PET-derived quantities are **targets/labels only**, never Method X predictors.

## Validation boundaries

SF analyses may support:

- confound-aware evaluation where label ≠ cohort
- SF-only and SF-aware stress tests
- train-SF / test-enrolled and train-enrolled / test-SF internal decoupling analyses

SF analyses must **not** support:

- true external validation claims
- non-invasive Method X claims based on blood biomarkers
- Alzheimer disease diagnosis claims

## Confound-controlled CV

Confound-controlled / confound-reduced CV is **required** for SF-inclusive screening models because label is not equivalent to cohort.

Confound-controlled CV is **N/A-DEGENERATE** for binary A4/LEARN-only catalogs where label ≈ cohort.

## Privacy

All SF catalog outputs remain hash-level or aggregate-only. No raw subject IDs, raw paths, or subject-level prediction exports.
