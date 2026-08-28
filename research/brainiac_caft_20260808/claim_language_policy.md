# Claim language policy

Every BrainIAC-CAFT output is labelled `POST_HOC_INTERNAL_TECHNICAL_EVIDENCE`.
The novelty ceiling is `NOVELTY_CANDIDATE` until the review matrix is complete.

## Permitted

```text
Regions with higher attribution in the model decision
PET-free technical estimate of biomarker status
Internal, single-programme technical evidence
Indeterminate research output
```

## Forbidden

Enforced mechanically by `assert_claim_language` in
`src/research/brainiac_provenance.py`. The rejected set covers diagnosis
language, claims of external validation, causal or disease-region attribution,
clinical triage, and regional quantitative biomarker claims.

A confirmatory claim would require a genuinely untouched cohort, which this
programme does not have.
