# Clinical Endpoint Progression Policy

- status: **`FROZEN_BEFORE_MODELING`**
- date frozen: `2026-06-07`
- branch: `clinical_endpoint`

## Claim boundary

Exploratory clinical/cognitive progression proxy only. Do not call Alzheimer diagnosis or conversion-to-AD without independent validated label.

## CDR worsening (frozen)

Binary progression endpoint: baseline global CDR equals 0.0 and any later visit global CDR is greater than or equal to 0.5 within the locked enrolled A4/LEARN follow-up window.

- threshold: `global_cdr >= 0.5 after baseline global_cdr == 0.0`

## PACC decline (frozen)

Binary decline endpoint: PACC composite decline of at least 1.0 point from participant baseline PACC on any later locked visit within enrolled A4/LEARN follow-up.

- threshold: `delta_pacc <= -1.0 from baseline`

## SF policy

SF may be used only for internal decoupling/stress tests; never external validation.

## Forbidden claims

- Alzheimer disease diagnosis
- conversion-to-AD
- clinical AD classification
