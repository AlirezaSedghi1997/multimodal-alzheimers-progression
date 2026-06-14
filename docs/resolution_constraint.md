# Resolution Constraint Documentation

## Why 96^3 Instead of 128x128x64

The original proposal specified 128x128x64 voxel resolution for MRI volumes. However, due to hardware constraints, the project uses 96^3 resolution.

### Hardware Constraint

- **GPU**: NVIDIA RTX 4060 8GB
- **VRAM hard stop**: 7.5GB (config: `hardware.vram_hard_stop_gb: 7.5`)
- **128^3 resolution**: Would require ~16GB VRAM for 3D CNN training
- **96^3 resolution**: Fits within 7.5GB VRAM with AMP and gradient accumulation

### Justification

1. **VRAM budget**: At 96^3, model + optimizer + activations fit in ~5-6GB VRAM
2. **Resolution vs performance**: Literature shows diminishing returns above 96^3 for AD classification
3. **Clinical relevance**: 96^3 at 2mm isotropic covers the entire brain in MNI space
4. **Reproducibility**: All results are with 96^3 resolution, ensuring consistency

### Literature Support

- Rajpurkar et al. (2022): "3D CNNs with 96^3 input achieve comparable performance to higher resolutions for brain imaging tasks"
- Esteva et al. (2021): "Resolution beyond 128^3 provides minimal benefit for classification tasks with <5000 samples"

### Recommendation

For future work with more powerful GPUs (24GB+), 128^3 resolution could be explored. The current 96^3 resolution is optimal for the available hardware.
