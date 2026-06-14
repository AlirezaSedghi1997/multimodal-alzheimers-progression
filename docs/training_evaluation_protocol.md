# Training & Evaluation Protocol (Repository Map)

## Method X 3D MRI (T1)

- **Optimizer:** AdamW, lr=`1e-4`, weight_decay=`1e-4`
- **Loss:** BCE-with-logits (+ optional latent alignment); **not** plain gradient descent
- **Evaluation:** Patient-grouped k-fold CV; metrics on **held-out test fold** each fold
- **Overfitting checks:** OOF AUC, cohort-probe ≈0.5 on latents, label-shuffle control, `final_claim_allowed` gate
- **Gap vs literature:** consider validation early-stopping (Frontiers 2024; arXiv 2411.12061)

## Other branches

See JSON for tabular, Method Y, Method Z, T1+FLAIR, EfficientNet ablation details.

All deep trainers use AdamW: adaptive per-parameter learning rates with momentum-like moment estimates—not classical fixed-lr batch gradient descent. Default lr=1e-4, weight_decay=1e-4; gradients via backprop on mini-batches (batch_size=1 for 3D due to VRAM).
