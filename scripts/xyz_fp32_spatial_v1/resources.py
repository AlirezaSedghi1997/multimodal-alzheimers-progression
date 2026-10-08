"""Bounded aggressive read-only RAM banks; identical float32 voxels and row order."""
import json
import numpy as np
from scripts.xyz_fusion_postall_v2.context import Context
from scripts.xyz_fusion_v1.runtime import available_ram
from scripts.xyz_fusion_v2.training import ImageBank
from scripts.xyz_fusion_v2.inference import PETBank


def can_cache(nbytes, remaining, reserve):
    return nbytes <= min(remaining, max(0, available_ram() - reserve))


class FastMRI(ImageBank):
    def __init__(self, path, budget, reserve):
        mapped = np.load(path, mmap_mode='r', allow_pickle=False)
        self.array = np.array(mapped) if can_cache(mapped.nbytes, budget, reserve) else mapped
        self.mode = 'MEMMAP' if isinstance(self.array, np.memmap) else 'RAM_ONCE'
        self.cached_bytes = self.array.nbytes if self.mode == 'RAM_ONCE' else 0


class FastPET(PETBank):
    def __init__(self, frame, budget, reserve):
        super().__init__(frame, max_gib=0)
        size = len(frame) * 128**3 * 4
        if can_cache(size, budget, reserve):
            # One allocation, one source pass; no second complete temporary list.
            self.array = np.empty((len(frame), 1, 128, 128, 128), dtype=np.float32)
            for i, key in enumerate(self.links.index): self.array[i] = self._read(int(key))
            self.close()
        self.mode = 'RAM_ONCE' if self.array is not None else 'HDF5_LAZY'
        self.cached_bytes = size if self.array is not None else 0


class FastContext(Context):
    def banks(self):
        if self._banks is None:
            self.ensure_inputs()
            budget = int(self.options['ram_cache_budget_gib'] * 1024**3)
            reserve = int(self.options['ram_reserve_gib'] * 1024**3)
            mri = FastMRI(self.root / 'results/xyz_mri_baseline_fast_v1/private/mri_cache.npy', budget, reserve)
            pet = FastPET(self.pet, budget - mri.cached_bytes, reserve)
            self._banks = {'mri': mri, 'pet': pet}
            print(json.dumps({'event': 'FP32_BANKS_READY', 'mri': mri.mode, 'pet': pet.mode,
                'cached_gib': (mri.cached_bytes + pet.cached_bytes) / 1024**3,
                'budget_gib': budget / 1024**3, 'minimum_spare_at_allocation_gib': reserve / 1024**3,
                'voxel_precision_changed': False}), flush=True)
        return self._banks
