"""Storage-only acceleration; math policy, tensors, subjects and training unchanged."""
import json
from scripts.xyz_fusion_postall_v2.context import Context


class ResourceContext(Context):
    def banks(self):
        if self._banks is None:
            self.ensure_inputs()
            from scripts.xyz_fusion_v2.training import ImageBank
            from scripts.xyz_fusion_v2.inference import PETBank
            # Each existing bank measures AVAILABLE RAM at allocation time and
            # reserves 6 GiB. The PET allocation occurs after the MRI allocation.
            # Insufficient capacity falls back to the original read-only source.
            mri=ImageBank(self.root/'results/xyz_mri_baseline_fast_v1/private/mri_cache.npy',
                          self.resource_options['mri_ram_gib'])
            pet=PETBank(self.pet,max_gib=self.resource_options['pet_ram_gib'])
            self._banks={'mri':mri,'pet':pet}
            print(json.dumps({'event':'LMO_BANK_RESOURCES','mri':self._banks['mri'].mode,
                'pet':self._banks['pet'].mode,'reserve_gib':6,'image_precision_changed':False}),flush=True)
        return self._banks
