# Calibration Policy

## Core Rule

Calibration metrics may be reported on all evaluation sets, but calibration parameters must be fit only inside the development/fold scope.

## Forbidden Fit Scopes

- External holdout.
- External-like locked holdout.
- Full dataset.
- Test fold.

## Required Reporting

- Brier score.
- Expected calibration error.
- Reliability diagram from aggregate predictions.
- Calibration fit log proving external data was not used for fitting.

## Single-Class External Behavior

If the external-like holdout contains one class only, do not report ROC-AUC or PR-AUC. Mark the result as `CALIBRATION_ONLY_NO_AUC`.

