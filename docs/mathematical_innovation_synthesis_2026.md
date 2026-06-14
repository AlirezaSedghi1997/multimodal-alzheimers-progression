# Mathematical Innovation Synthesis (2026)

Status: `ACTIVE` | Perspective: information-theoretic multimodal fusion at small N

## Core scientific claim (Method X innovation axis)

The thesis contribution is **not** a new ResNet variant. It is a **mathematically grounded fusion framework** that:

1. **Maximizes synergistic information** (SynIB, LIT019) between T1-derived representations and fixed tabular covariates.
2. **Audits fusion non-compensatorily** (MMS-PRM adapted, LIT016) so strong AUC cannot hide cohort-probe or calibration failure.
3. **Extends texture with multifractal geometry** (MFSCA, LIT018) as an interpretable R3 branch when 96³ cache is ready.

## Paper-by-paper extraction

### LIT019 — SynIB (2606.09853) → IP1, R1.2

**Math:** PID decomposition $I(Y;X_1,X_2)=R+U_1+U_2+S$. SynIB penalizes confident $\hat Y$ when modality $i$ is masked:

$$\mathcal L = \mathcal L_{\text{BCE}}(Y,\hat Y_{12}) + \lambda \sum_{i\in\{1,2\}} \mathbb E\big[\text{Conf}(\hat Y \mid \tilde X_i, X_{-i})\big]$$

**Project mapping:** `synib_synergy_loss.py` + modality dropout in Method X/Y fusion training.  
**Interpretation:** At A4/LEARN coupling, synergy penalty reduces tabular-only shortcut learning.

### LIT016 — MMS-PRM (2606.07801) → IP2

**Math:** Augmented Chebyshev scalarization (Eq. 8):

$$g_{\text{aug}}(\mathbf v)=-\max_j(z^*_j-v_j)-\rho\sum_j(z^*_j-v_j)$$

**Project mapping:** Dimensions = {ROC-AUC, 1−Brier, 1−cohort_probe_AUC, missing-modality AUC}.  
**Why:** Tabular branch encodes substudy (probe AUC≈0.73); fusion must not "buy" AUC while worsening deconfounding.

### LIT018 — MFSCA (2606.10222) → IP3, R3

**Math:** Hilbert curve → 1D intensity series → MFDFA → $H=h(2)$, $\Delta\alpha$, asymmetry $A$.  
**Finding:** Multifractal → monofractal transition with dementia progression.  
**Project mapping:** Header-stat surrogate now; full slice MFSCA after 96³ preprocessing.

### LIT006 — PET-guided KD (2604.12574) → IP4

**Use:** Teacher=fusion PET+MRI; student=MRI-only. **Method X inference remains PET-free.**  
**Repo:** `pet-guided-mri-amyloid-detection` — sensitivity analysis only.

### LIT014 — LLM MRI reports (2606.07721)

**Use:** ROI/reads sensitivity (`mri_roi_wmh_reads` cohort probe block).  
**Not:** Primary Method X input — report extraction branch for ablation only.

### LIT015 — GNOVA (2606.07798) → Method X-Cog stretch

**Math:** GRU encoder + Neural ODE decoder + VAE uncertainty on irregular visits.  
**Use:** CDR-SB/MMSE trajectory without MRI/PET — clinical endpoint exploratory branch.

### LIT017 — Rationale SFT hurts (2606.10279) → ANTI-PATTERN

Do **not** add chain-of-thought/rationale LLM to tabular Method X.

### LIT021 — 2606.07722 → EXCLUDED

Likely mislink (chatbot/philosophy). Confirm with user before any citation.

### Similar papers (same quality tier)

| ID | Paper | Relevance |
|---|---|---|
| LIT022 | OMIB (2505.19996) | Dynamic $\lambda$ per modality — extends SynIB |
| LIT023 | MRdIB (2509.20225) | Representation-disentangled IB for Method Y |
| LIT024 | PID tau synthesis (2602.22545) | PID latent factors for T1+FLAIR |
| LIT002 | EfficientNet T1+FLAIR (2411.12061) | +0.05 AUC FLAIR sensitivity |

## GitHub pattern adoption

| Repo | Adopt | Avoid |
|---|---|---|
| pet-guided-mri-amyloid-detection | KD train scaffold | PET at Method X inference |
| DiaMond | HDF5 paired cache | ViT headline at N=184 |
| MMS-PRM | Chebyshev audit | Full MCTS (compute) |
| TransMF_AD | Fusion comparison row | Unablated transformer claim |
| multimodalAD | Small-N caution | ADNI-scale architecture claim |

## Negative results policy

SynIB surrogate may underperform plain logistic fusion on header stats — report honestly.  
Innovation value = objective + audit framework, not guaranteed AUC lift at surrogate stage.
