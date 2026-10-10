# FP32 precision sensitivity

Reviewed on 9 October 2026. This analysis recomputed image scores in FP32 with
frozen CNN weights and refitted the small prediction heads within the recorded
training ancestry. It is a separate sensitivity model from the AMP parent.

The execution verifier checked 90 locked components, 180 heads and 10 outer-fold
cells. The focused FP32 and corrected post-fusion test suites passed 63 tests.
Stored-prediction statistics and spatial arrays were reviewed independently of
the execution-completion label.

## Matched comparisons

| Comparison | Participants / events over follow-up | Base IBS | Augmented IBS | Paired IBS difference, 95% CI |
|---|---:|---:|---:|---|
| Clinical + amyloid versus the same predictors + MRI | 1,059 / 239 | 0.04029953 | 0.04031900 | +0.00001946 [−0.00025102, +0.00027647] |
| Clinical + MRI + amyloid versus the same predictors + FBP image | 544 / 112 | 0.04062053 | 0.04041816 | −0.00020237 [−0.00076960, +0.00031639] |

Differences are augmented minus base; lower IBS is better. Both intervals include
zero. Incremental value was not demonstrated in these comparisons. This finding
does not establish equivalence or absence of an effect.

Intervals use 2,000 paired bootstrap resamples within outer folds, conditional on
the fitted models and training censoring estimates. They do not quantify the
uncertainty of repeating the entire selection and training procedure. The two
comparisons use different eligible populations and should not be pooled.

![Paired conditional confidence intervals](../figures/fp32_paired_comparisons.svg)

## Spatial review

The fixed panel contained 10 people and 30 requested person-by-model/modality
units. Twenty-two units were eligible; eight lacked a locked PET image. Evaluating
three baselines per eligible unit produced 66 baselines, of which 36 passed the
component and final-ensemble numerical checks. All 22 zero-image baselines failed.

The randomized-network maps were numerically degenerate, so that control was not
estimable. Label-permutation training was not performed. Numerical convergence
therefore does not establish full attribution validity. Maps show model behavior
in model space with an approximate atlas; they do not identify individual disease
lesions or a causal brain mechanism. Individual maps are excluded from this
repository.

The four-way MRI/FBP/tau/blood model remains unestimated under the recorded support
rules. This sensitivity analysis does not change that decision or provide external
validation.

## Aggregate sources

- [Recomputed fixed-prediction statistics](../results/aggregate/09_fp32/statistical_review.json)
- [Numerical spatial-control review](../results/aggregate/09_fp32/spatial_review.json)
- [Source lineage](result_lineage.json)

The source hashes identify the reviewed local artifacts. Only aggregate material
is included; individual outcomes, predictions, images and training records require
authorized access to the underlying research environment.
