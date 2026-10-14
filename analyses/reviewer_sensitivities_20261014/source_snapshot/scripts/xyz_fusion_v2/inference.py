"""Read-only pattern-PET bank, exact subject/scan/date linking; no SUVR equivalence."""
from __future__ import annotations
from pathlib import Path
import h5py
import numpy as np
import pandas as pd
import torch
from scripts.xyz_postprimary_v1.feasibility import audit_pet_linkage
from scripts.xyz_fusion_v1.runtime import available_ram
from scripts.xyz_fusion_v1.store import file_sha


def pet_intake(ctx, frame):
    audit = audit_pet_linkage(ctx)
    rows = pd.DataFrame(audit.pop('private_rows'))
    membership_path = ctx['root']/'results/method_y_revival_v3_image_v2/private/amyloid_image_membership.parquet'
    membership = pd.read_parquet(membership_path)
    subset = frame.loc[frame.patient_hash.isin(rows.patient_hash)].copy()
    links = membership.set_index('patient_hash').loc[subset.patient_hash]
    required = ['image_path_private', 'h5_group', 'source_image_sha256']
    subset = subset.merge(links[required].reset_index(), on='patient_hash', validate='one_to_one')
    from .data import manufacturer
    subset['pet_manufacturer']=list(links.manufacturer.map(manufacturer))
    audit.update(status='EXACT_SCAN_PATTERN_IMAGE_ELIGIBLE', subjects=len(subset),
                 events_primary=int(subset.event.sum()), quantitative_equivalence=False,
                 membership_sha256=file_sha(membership_path),
                 hdf5_sha256={p:file_sha(p) for p in sorted(subset.image_path_private.unique())})
    return subset, audit


class PETBank:
    def __init__(self, frame, max_gib=6):
        self.links = frame.set_index('cache_row', verify_integrity=True)
        self.handles = {}
        self.array = None
        self.mapping = {int(k):i for i,k in enumerate(self.links.index)}
        size = len(frame)*128**3*4
        if size < min(max_gib*1024**3, available_ram()-6*1024**3):
            # Fill once: stacking a list would briefly double the whole PET bank.
            self.array = np.empty((len(frame),1,128,128,128),dtype=np.float32)
            for i,k in enumerate(self.links.index):self.array[i]=self._read(int(k))
            self.close()
        self.mode = 'RAM_ONCE' if self.array is not None else 'HDF5_LAZY'

    def _read(self, row):
        r = self.links.loc[row]; p = str(r.image_path_private)
        if p not in self.handles: self.handles[p] = h5py.File(p, 'r')
        group = self.handles[p][str(r.h5_group)]
        if str(group.attrs.get('source_sha256')) != str(r.source_image_sha256):
            raise ValueError('PET_IMAGE_PARENT_DRIFT')
        a = np.asarray(group['image'], dtype=np.float32)
        if a.shape != (1,128,128,128) or not np.isfinite(a).all():
            raise ValueError('PET_PATTERN_IMAGE_INVALID')
        return a

    def batch(self, rows, device):
        a = np.stack([self.array[self.mapping[int(k)]] if self.array is not None else self._read(int(k)) for k in rows])
        t = torch.from_numpy(a)
        if device.type == 'cuda': t = t.pin_memory()
        return t.to(device, non_blocking=device.type=='cuda')

    def close(self):
        for f in self.handles.values(): f.close()
        self.handles = {}
