# پروتکل آزمایش و ablation

1. Lock subject-grouped outer five-fold/inner three-fold membership.
2. Fit imputer, scaler, feature selection and any residual model inside training folds only.
3. Fit clinical-only, clinical+FBP, clinical+FTP and combined models with equal family/budget.
4. Evaluate outer test once after winner lock; recover committed results by hash.
5. For optional discordance: compare raw components, linear fold-local residual, nonlinear residual, and shuffled residual.
6. Report Uno/Harrell C-index, dynamic AUC, Brier/IBS, calibration slope/intercept, event counts, and fold-wise deltas.
7. Apply clustered subject bootstrap and Holm adjustment; do not use DeLong for survival C-index.
8. Stop extension if shuffled residual performs similarly, scanner leave-out collapses, or any fold is incomplete.
