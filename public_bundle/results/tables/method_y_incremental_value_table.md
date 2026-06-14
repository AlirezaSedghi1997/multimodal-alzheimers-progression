# Method Y Incremental Value Table

- Status: `COMPLETED`
- PET-only reference AUC: `0.6332772166105499`
- matched effective N: `184`
- comparability ok: `True`

| comparison | n | AUC | delta AUC vs PET | accuracy | sensitivity | specificity | F1 | status | delta valid |
|---|---:|---:|---:|---:|---:|---:|---:|---|---:|
| MRI-only on PET-cache hash intersection | 184 | 0.5712682379349046 | -0.062008978675645365 | 0.8811646514935989 | 1.0 | 0.0 | 0.9364043057157515 | AVAILABLE | True |
| Late fusion MRI+PET | 184 | 0.5326879910213244 | -0.10058922558922556 | 0.8811646514935989 | 1.0 | 0.0 | 0.9364043057157515 | AVAILABLE | True |
| Gated fusion MRI+PET | 184 | 0.46997755331088664 | -0.16329966329966328 | 0.8864278093883357 | 0.9741935483870968 | 0.14285714285714285 | 0.9366941607882154 | AVAILABLE | True |
| MRI+PET+cognitive summaries | 184 | 0.6332772166105499 | 0.0 | 0.8706383357041252 | 0.9806451612903226 | 0.02857142857142857 | 0.9290018530959075 | AVAILABLE | True |
