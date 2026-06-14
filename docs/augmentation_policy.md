# Data Augmentation Policy

## Current Augmentation (MONAI)

### Transforms Applied (Train Only)

| Transform | Parameters | Purpose |
|-----------|------------|---------|
| RandFlip | axis=0, prob=0.5 | Left-right flip |
| RandFlip | axis=1, prob=0.5 | Anterior-posterior flip |
| RandAffine | rotate=0.1, translate=5, scale=0.05, prob=0.3 | Spatial deformation |
| RandAdjustContrast | gamma=(0.8, 1.2), prob=0.3 | Intensity variation |
| RandGaussianSmooth | sigma=(0.5, 1.0), prob=0.2 | Smoothness variation |
| RandGaussianNoise | std=0.02, prob=0.3 | Noise injection |
| RandBiasField | prob=0.2 | MRI bias field simulation |

### Configuration

- **Exposures per real sample**: 4
- **Real training samples**: ~1,087 (per fold)
- **Augmented training exposures**: ~4,348 (1,087 × 4)
- **Total effective samples**: ~4,348 (not 6,000 as proposed)

### Proposal vs Implementation

| Aspect | Proposal | Implementation |
|--------|----------|----------------|
| Target samples | ~6,000 | ~4,348 |
| Method | Data augmentation | MONAI transforms |
| Transforms | Not specified | 7 transforms |
| Validation/test | No augmentation | No augmentation (correct) |

## Justification

1. **4 exposures per real** is standard for medical imaging (Roth et al., 2015)
2. **7 transforms** cover spatial, intensity, and noise variations
3. **Validation/test untouched** ensures fair evaluation
4. **Synthetic data** (GAN/diffusion) planned but not yet implemented

## Future Work

- Implement synthetic MRI generation (flow matching, arXiv:2606.11833)
- Increase exposures to 8 per real sample
- Target 6,000+ effective samples
