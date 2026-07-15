from __future__ import annotations


class ImageSurvivalError(ValueError):
    """Raised when a minibatch cannot define a censoring-aware Cox objective."""


def negative_cox_partial_log_likelihood(risk_score, event, observed_time):
    import torch

    risk = risk_score.reshape(-1)
    events = event.reshape(-1).to(dtype=risk.dtype)
    times = observed_time.reshape(-1).to(dtype=risk.dtype)
    if not (risk.numel() == events.numel() == times.numel()):
        raise ImageSurvivalError("COX_BATCH_LENGTH_MISMATCH")
    if not torch.isfinite(risk).all() or not torch.isfinite(times).all():
        raise ImageSurvivalError("COX_BATCH_NONFINITE")
    event_count = events.sum()
    if float(event_count.detach().cpu()) <= 0.0:
        raise ImageSurvivalError("COX_BATCH_REQUIRES_EVENT")
    order = torch.argsort(times, descending=True)
    ordered_risk = risk[order]
    ordered_event = events[order]
    log_risk_set = torch.logcumsumexp(ordered_risk, dim=0)
    contribution = (ordered_risk - log_risk_set) * ordered_event
    return -contribution.sum() / ordered_event.sum()
