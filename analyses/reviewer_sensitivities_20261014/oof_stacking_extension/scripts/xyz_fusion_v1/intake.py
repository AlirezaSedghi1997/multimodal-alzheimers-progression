"""Chronology and lineage invariants for the new common-landmark experiment."""
from __future__ import annotations

import numpy as np
import pandas as pd


def select_blood(rows, landmark, lookback):
    if rows.empty:
        return None
    valid = rows.loc[(rows.day <= landmark) & (rows.day >= landmark-lookback)].copy()
    valid = valid.loc[np.isfinite(valid.value) & (valid.value > 0)]
    for col in ('assay', 'units', 'specimen', 'sample'):
        valid = valid.loc[valid[col].notna() & valid[col].astype(str).str.strip().ne('')]
    valid = valid.loc[valid.specimen.str.casefold().eq('plasma') & valid.units.eq('U/mL')]
    if valid.empty:
        return None
    chosen = valid.loc[valid.day == valid.day.max()].drop_duplicates(
        ['day','value','assay','units','specimen','sample'])
    if len(chosen) != 1:
        raise ValueError('AMBIGUOUS_BASELINE_BLOOD')
    return chosen.iloc[0].to_dict()


def sustained_endpoint(visits, landmark, window):
    days = {}
    for day, cdr in visits:
        if not np.isfinite(day) or not np.isfinite(cdr):
            continue
        if day in days and days[day] != cdr:
            raise ValueError('CONFLICTING_SAME_DAY_CDR')
        days[day] = cdr
    if any(v >= .5 for d,v in days.items() if d <= landmark):
        raise ValueError('PREVALENT_CDR_PROGRESSION')
    followup = sorted((d,v) for d,v in days.items() if landmark < d <= landmark+window)
    if not followup:
        raise ValueError('NO_POST_LANDMARK_FOLLOWUP')
    for (day,value), (_,following) in zip(followup,followup[1:]):
        if value >= .5 and following >= .5:
            return 1, float(day-landmark)
    return 0, float(followup[-1][0]-landmark)


def ancestry_overlap(ancestors, evaluation_subjects):
    if not ancestors or any(not isinstance(v, (list, set, tuple)) for v in ancestors.values()):
        raise ValueError('ANCESTRY_UNVERIFIABLE')
    forbidden = set(evaluation_subjects)
    return {name:len(set(ids)&forbidden) for name,ids in ancestors.items() if set(ids)&forbidden}
