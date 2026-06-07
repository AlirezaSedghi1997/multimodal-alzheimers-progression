# Clinical Endpoint Progression Policy (Template)

- status: **`TEMPLATE_NOT_FROZEN`**
- branch: `clinical_endpoint`

## Claim boundary

Exploratory clinical/cognitive progression proxy only. Do not call Alzheimer diagnosis or conversion-to-AD without independent validated label.

## Endpoints (not frozen)

### CDR worsening

- definition: [INSERT PRESPECIFIED CDR WORSENING DEFINITION BEFORE MODELING]
- worsening threshold: [INSERT THRESHOLD]
- SF limitation: SF rows lack enrolled-style longitudinal coverage; report separately.

### PACC decline

- definition: [INSERT PRESPECIFIED PACC DECLINE DEFINITION BEFORE MODELING]
- decline threshold: [INSERT THRESHOLD]
- SF limitation: SF rows lack enrolled-style longitudinal coverage; report separately.

## Required before freeze

- endpoint_definition
- worsening_or_decline_threshold
- baseline_predictor_allowlist
- non_future_visit_discipline
- sf_reporting_policy
- date_frozen
- approved_by
- change_policy

## Forbidden claims

- Alzheimer disease diagnosis
- conversion-to-AD
- clinical AD classification
