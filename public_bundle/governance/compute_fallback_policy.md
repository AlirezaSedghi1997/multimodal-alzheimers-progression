# Compute Fallback Policy

## Primary Device

Use CUDA on RTX 4060 8GB when available.

## VRAM Limits

- Dry run guard: 6.5 GB.
- Training soft cap: 7.0 GB.
- Hard stop: 7.5 GB.

## Fallback Order

1. Reduce batch size.
2. Reduce development resolution.
3. Disable optional variants.
4. Use CPU only for non-training checks.
5. Block with a clear report if final training is not feasible.

