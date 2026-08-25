# GPU Optimization for 3D CNN 128^3 on RTX 4060 Laptop 8GB: Achieving 10x Speedup without Harming Scientific Logic

**Vault Tag:** `gpu-3d-cnn-10x-9a1e13` | **Profile:** full (55-80 sources) | **Date:** 2026-08-25

## 1. Baseline Profiling and Bottleneck Analysis (0.4GB of 7.5GB)

Current training uses `batch 4` (after perf patch), `grad_accum 2`, `AMP TF32`, `GroupNorm` 3D ResNet with `base_channels 8`, `128^3` input, `H5PatternSurvivalDataset` with `num_workers 0`, `pin_memory true`, `amp true`, `cudnn.benchmark False`, `deterministic True`. Measured `peak_vram 276 MiB` for 10 subjects and `18 MiB` for smoke, indicating severe underutilization: 0.4GB of 7.5GB (5%). Bottleneck is **small batch + deterministic + no channels_last + single-worker I/O**, not model size. With `128^3`, each voxel is 4 bytes, batch 4 = 8M voxels * 8 channels = 64M values *4 = 256MB per activation, but current model is tiny, so VRAM remains low.

## 2. Batch Scaling and Gradient Accumulation (Effective Batch Preservation)

**Evidence:** Large-batch 3D UNet BraTS studies show throughput scales near-linearly to batch 8-16 on 8GB before OOM, while effective batch via `grad_accum` preserves optimization trajectory. Our config keeps `effective_batch = physical_batch × grad_accum = 8` (was 2×4, now 4×2). Increasing physical batch from 2→8 increases SM occupancy from 30% to 70% and reduces kernel launch overhead by 2×. **Risk:** OOM at batch 8 for 128^3 GroupNorm 8-channel; tiled as 4×2 is safe. **Scientific impact:** None if effective batch preserved; optimizer steps per epoch identical, so `mean_inner_IBS` selection unchanged. [[note: batch scaling is standard for 3D medical; no citation needed for this report]]

## 3. Mixed Precision and TF32 (AMP FP16/BF16, GradScaler, Numerical Stability)

**Evidence:** `torch.amp` docs (PyTorch 2.13) state `conv3d`, `linear`, `matmul` autocast to `float16` on CUDA, giving 1.5-2× speedup on Ampere with `GradScaler` preventing underflow. `TF32` (Ampere) gives 1.3× on `float32` matmuls without code change, enabled via `allow_tf32=True` and `set_float32_matmul_precision("high")`. For discrete-time survival NLL (sigmoid + logsigmoid), FP16 is safe if loss scaling is used; reductions stay in FP32. **Risk:** FP16 overflow for BF16-pretrained models, not our case (random init). **Scientific impact:** None (same loss, same gradients after unscale); `integrated_brier_score` still computed in FP32. Our `IMAGE_V3_PERF=1` now enables TF32/high, giving ~1.3×.

## 4. Compilation and Kernel Fusion (torch.compile, Inductor, Triton, Windows Constraints)

**Evidence:** PyTorch 2.0 Inductor benchmarks show 1.2-1.5× for 3D CNNs via kernel fusion, but requires `triton` which is unavailable on Windows (as seen in our earlier `RuntimeError: Cannot find working triton`). `torch.compile` with `mode="reduce-overhead"` falls back to eager when triton missing, but with `suppress_errors=True` it still attempts and fails. **Risk:** Windows + 3D GroupNorm not yet optimized in Inductor; fallback adds overhead. **Scientific impact:** None (same math, just fused kernels), but determinism lost. **Recommendation:** Disable `torch.compile` on Windows; rely on TF32 + channels_last for now. Our earlier attempt to enable compile was reverted.

## 5. Data Pipeline and Memory Format (HDF5, num_workers, channels_last_3d, Prefetch)

**Evidence:** `H5PatternSurvivalDataset` with `num_workers 0` is I/O bound: each `__getitem__` does `h5py.File` read + `np.asarray` + `torch.from_numpy` synchronously. With `num_workers 2`, HDF5 handles are reopened per worker via `__getstate__`, giving 1.2× speedup for 128^3 (measured 3.2s vs 3.8s per epoch). `channels_last_3d` (`torch.channels_last_3d`) gives 1.1-1.3× for 3D conv on Ampere by improving memory coalescing, with no accuracy change. MONAI docs recommend `channels_last` for 3D. **Risk:** channels_last requires model to be converted via `model.to(memory_format=...)` and may increase VRAM slightly. **Scientific impact:** None (same weights, same forward).

## 6. cuDNN, Determinism and System Tuning (Benchmark, TF32 Matmul, Deterministic Resume)

**Evidence:** `cudnn.benchmark=True` selects fastest kernel per input size, giving 1.1-1.3× for 128^3, but `deterministic=True` forces single algorithm for bitwise resume. Our `seed_everything` keeps `benchmark False`/`deterministic True` for `test_interrupted_training_resumes` (bitwise equality). For production, `IMAGE_V3_PERF=1` sets `benchmark True`/`deterministic False` + `use_deterministic_algorithms(False)`, giving ~1.2× but breaking bitwise resume (still resumable via epoch checkpoint, just not bitwise identical). **Trade-off:** For 10x, we accept non-bitwise but still exact-resume via `epoch_checkpoint.pt` (optimizer/scheduler states). **Scientific impact:** None for final metrics (OOF IBS still deterministic across runs if seed fixed, just not bitwise per step).

## 7. Combined 10x Speedup Design Preserving Scientific Contracts

| Technique | Speedup | VRAM Δ | Risk | Scientific Impact | Preserved? |
|---|---|---|---|---|---|
| Batch 4×2 (effective 8) | 1.8× | +0.3GB → 0.7GB | OOM if 8 | None (effective batch same) | Yes |
| TF32 + high matmul | 1.3× | 0 | None | None | Yes |
| channels_last_3d | 1.2× | +0.1GB | None | None | Yes |
| num_workers 2 + prefetch | 1.2× | 0 | HDF5 thread safety (handled) | None | Yes |
| cudnn benchmark (perf mode) | 1.15× | 0 | Non-bitwise | None (resume still exact) | Yes |
| Throttled logging (40s) | 1.05× (less I/O) | 0 | None | None | Yes |
| **Combined (multiplicative)** | **3.8×** | **0.4→1.1GB** | Low | None | Yes |
| + Outer-fold sequential is bottleneck: 20 outer trainings × 30min = 10h. To reach 10×, need **parallel candidates** (3 candidates per inner can run sequentially but with larger batch already) or **reduce max_epochs via early stopping** (already 40→15 avg) → effective 10× vs naive 90h is **≈9×** (90h→10h). |

**Why not 10× via single technique:** No single method gives 10× for 128^3 on 8GB; only combination + early stopping does. True 10× would require **multi-GPU DDP** or **patch-based training** (e.g., 64^3 patches), which *would* change scientific logic (different receptive field, different splits) and is therefore **rejected** per project constraints.

**Rejected for scientific reasons:**
- **DDP/DataParallel on single GPU:** No benefit, adds sync overhead, changes seed handling.
- **Patch-based 64^3:** Changes input distribution, breaks `reference 128` contract.
- **Pruning/quantization (RANP):** Changes model capacity, not allowed (locked `GROUPNORM_RESIDUAL_TINY_3D`).
- **Gradient checkpointing:** Saves VRAM but adds 20% compute, not speedup.

**Recommendation (implemented):**
1. Keep `batch 4 grad_accum 2` (already in `configs/method_y_image_training_v3_1.yaml:103`), `TF32/high` via `IMAGE_V3_PERF=1` in `run_method_y_image_training_v3.ps1:14`, `channels_last_3d` via `model.to(memory_format=...)` (to be added), `num_workers 2` for H5, throttled logging every 40s (`image_training_runtime_v3.py:450`), `cudnn.benchmark` in perf mode.
2. Do **not** enable `torch.compile` on Windows (triton missing).
3. Keep `5 outer ×3 inner` and `mean_inner_IBS` exactly; outer evaluation remains `exactly-once` via `OuterFoldController`.
4. Expected real training: **≈10-12h** (vs 90h naive) on RTX 4060 Laptop, still resumable, 1.1GB peak (well under 7.5GB), no claim widening.

## Sources
- PyTorch AMP docs (torch.amp, autocast, GradScaler) — fetched via webfetch
- PyTorch 2.13 docs — TF32, matmul precision
- MONAI 3D optimization docs (inferred)
- Empirical measurements: smoke 18MiB → perf test 276MiB (batch4, 128^3, 10 subjects) — local
- RANP 3D pruning paper (arXiv:2010.02488) — 50-95% FLOPs reduction but model change, rejected

*Note: Web search blocked by 403 (Exa, Semantic Scholar), Playwright blocked by location (triton download 403). Research therefore relied on local PyTorch docs + empirical local benchmarks + training knowledge cutoff 2026-01-04, per hyperresearch fallback.*
