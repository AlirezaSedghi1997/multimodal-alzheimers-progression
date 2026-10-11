# Additional statistical analyses for thesis revision

Status: **completed internal, post-hoc analyses**, 11 October 2026.
These results supplement the original study and do not constitute independent
validation. Detailed methods and execution commands are in the
[statistical supplement guide](../../docs/STATISTICAL_SUPPLEMENT.md).

## Stronger comparators and prediction accuracy

In the 1,059-person cohort, the outer-training Kaplan–Meier null model had IBS
0.042797. The clinical-plus-amyloid baseline fitted on the complete outer training
set had IBS 0.040296 and IPA 5.84%. Adding deep MRI yielded IBS 0.040402.
The paired IBS difference was +0.000106, with the two-comparison-adjusted interval
approximately [−0.000234, +0.000437]. Its Uno difference also included zero.
The comparison retaining regional volumes gave the same substantive conclusion:
**a stronger clinical baseline did not reveal a stable incremental MRI benefit**.

These are original-AMP-parent comparisons; the separately published FP32
sensitivity numbers answer a different precision-specific question.

## All tau contrasts, with multiplicity adjustment

All three analyses used the 358-person tau subset (92 events). Differences are
augmented minus base; intervals adjust three comparisons within each metric.

| Base model | ΔIBS | Adjusted interval | ΔUno | Adjusted interval |
|---|---:|---:|---:|---:|
| Cox ridge | −0.0028 | [−0.0064, +0.0012] | +0.073 | [+0.023, +0.135] |
| Random survival forest | −0.0014 | [−0.0045, +0.0013] | +0.055 | [−0.026, +0.151] |
| MRI fusion head | −0.0030 | [−0.0062, −0.00003] | +0.082 | [+0.033, +0.144] |

Only the MRI-head contrast retained favorable adjusted intervals for both
metrics. Its IBS boundary was very close to zero. Calibration varied substantially
across folds, and only 79–109 people with 18–29 events supported each 12-input tau
head. The result is an exploratory signal that needs a larger independent sample.
Per-fold calibration and valid bootstrap counts are retained in the JSON evidence.

## Cognition sensitivity

Removing PACC and MMSE from risk construction reduced prediction performance
(IBS 0.041731; Uno 0.6456), but the association between higher risk and faster
subsequent PACC decline remained: risk-by-time coefficient **−0.216 PACC units per
year per SD risk**, 95% CI [−0.285, −0.147]. The analysis included 1,049 participants
and 8,557 observations and retained baseline cognition in the longitudinal
adjustment. It refitted 45 heads with frozen MRI experts; it did not retrain CNNs.

## Population, chronology and information support

The flow from 1,169 source A4 participants to 1,059 analyzed participants is
documented with endpoint, linkage and automated anatomical-QC exclusions. Reverse
KM median follow-up was 3.92 years. Demographics and standardized differences are
reported for event/censoring groups and imaging/blood availability subsets.
Every tau measurement occurred at or before the individual origin (median lead
12 days). Subset differences remain a potential selection issue.

Retrospective Riley scenarios give minimum sample sizes ranging from 319 to 1,400
for eight parameters and assumed Cox–Snell R² from 0.20 to 0.05. The shrinkage
criterion dominates the three criteria in these scenarios. These are transparent
assumption checks, not proof of adequate deep-network sample size or prospective
sample planning.

Seven recorded deaths occurred after the observed CDR endpoint/censoring time;
six followed the last CDR in participants classified as censored. Event-free
status in that unobserved interval was not adjudicated. No competing-risk model
was fitted using an invented interval. Clinical safety MRI reads also do not
substitute for visual QC of processed registration.

## Evidence retained

[Eight aggregate evidence files](../../results/aggregate/10_statistical_supplement/)
cover cohort descriptions, stronger baselines/IPA, all tau contrasts, noncognitive
PACC sensitivity, disposition/QC, sample-size scenarios, actual tau training
support and source-summary hashes. Participant-level intermediates and thesis
documents are not included.
