"""Observable two-visit confirmation and explicit, prespecified sensitivity estimands."""
from __future__ import annotations
import numpy as np


def endpoint_views(visits, origin, window=1640.):
    days = {}
    for day, cdr in visits:
        if not np.isfinite(day) or not np.isfinite(cdr):
            continue
        if day in days and days[day] != cdr:
            raise ValueError('CONFLICTING_SAME_DAY_CDR')
        days[float(day)] = float(cdr)
    if any(c >= .5 for d, c in days.items() if d <= origin):
        raise ValueError('PREVALENT_CDR_PROGRESSION')
    follow = sorted((d-origin, c) for d, c in days.items() if origin < d <= origin+window)
    if not follow:
        raise ValueError('NO_POST_LANDMARK_FOLLOWUP')
    first = next((d for d, c in follow if c >= .5), None)
    pair = next(((a, b) for (a, x), (b, y) in zip(follow, follow[1:]) if x >= .5 and y >= .5), None)
    last = follow[-1][0]
    negative = [d for d, c in follow if c < .5]
    pending = pair is None and follow[-1][1] >= .5
    def row(event, time):
        return {'event': int(event), 'time_days': float(time)}
    return {
        'confirmation_time': row(pair is not None, pair[1] if pair else last),
        'legacy_onset': row(pair is not None, pair[0] if pair else last),
        'single_crossing': row(first is not None, first if first is not None else last),
        'last_visit_positive': row(pair is not None or pending, pair[0] if pair else last),
        'last_negative_censor': row(pair is not None, pair[0] if pair else
                                   (negative[-1] if pending and negative else (0. if pending else last))),
        'unconfirmed_final_positive': pending,
        'confirmation_delay_days': None if pair is None else pair[1]-pair[0],
        'last_observed_cdr': follow[-1][1],
        'definition': 'CDR_PROGRESSION_NOT_DEMENTIA_DIAGNOSIS'}
