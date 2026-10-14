"""The actual Cox survival link, not a surrogate dot product of expert scores."""
from __future__ import annotations

import numpy as np
import torch


class FixedInferenceBatch(torch.nn.Module):
    """Preserve the parent's AMP batch geometry; only one image is variable."""
    def __init__(self, model, context, position):
        super().__init__()
        self.model = model
        self.register_buffer('context', context.detach())
        self.position = int(position)

    def forward(self, image):
        if len(image) != 1:
            raise ValueError('SINGLE_EXPLAINED_IMAGE_REQUIRED')
        batch = torch.cat((self.context[:self.position], image, self.context[self.position + 1:]), dim=0)
        return self.model(batch)[self.position:self.position + 1]


class LockedSurvivalCurve(torch.autograd.Function):
    """Exact parent's GPU forward with an analytic deterministic CPU backward.

    For q_i=sigmoid(-z_i), S_j=prod(i<=j)q_i and
    dS_j/dz_i = -S_j*(1-q_i) for i<=j. No surrogate gradient or refitting.
    """
    @staticmethod
    def forward(ctx, logits):
        q = torch.sigmoid(-logits.float())
        survival = torch.cumprod(q, dim=1)
        ctx.save_for_backward(q.detach().cpu().double(), survival.detach().cpu().double())
        ctx.device, ctx.dtype = logits.device, logits.dtype
        return survival.detach().cpu().double()

    @staticmethod
    def backward(ctx, gradient):
        q, survival = ctx.saved_tensors
        weighted = gradient.double().cpu() * survival
        tail = torch.flip(torch.cumsum(torch.flip(weighted, dims=[1]), dim=1), dims=[1])
        return (-(1 - q) * tail).to(device=ctx.device, dtype=ctx.dtype)


def expert_scores(model, image, policy, times, grid):
    # Parent interpolation uses numpy.float64 after float32 GPU cumulative
    # products. Repeating interpolation in float32 can materially change a Cox
    # head with tightly distributed image scores, despite identical weights.
    logits = model(image).float()
    survival = LockedSurvivalCurve.apply(logits)
    boundary = torch.cat([torch.ones((len(logits), 1), dtype=torch.float64), survival], dim=1)
    def interpolate(table, coordinates, queries):
        coordinates = np.asarray(coordinates, dtype=float)
        values = []
        for time in queries:
            hi = min(int(np.searchsorted(coordinates, time, side='right')), len(coordinates) - 1)
            lo = max(0, hi - 1)
            weight = (time - coordinates[lo]) / (coordinates[hi] - coordinates[lo])
            values.append(table[:, lo] * (1 - weight) + table[:, hi] * weight)
        return torch.stack(values, dim=1)
    table = interpolate(boundary, policy['hazard_edges_days'], grid)
    return -torch.log(interpolate(table, grid, times).clamp(1e-6, 1.))


def cox_risk(head, raw_features, horizon):
    """Differentiable parity with the locked imputer/scaler/Cox pipeline.

    scikit-survival uses S(t|x)=exp(-H0(t)*exp(x beta)). Attribution is to
    1-S(t|x), not linear log hazard and not a single component's coefficient.
    """
    if len(head.steps) != 3:
        raise ValueError('UNSUPPORTED_LOCKED_HEAD_TRANSFORM')
    imputer, scaler, cox = [step[1] for step in head.steps]
    if getattr(imputer, 'add_indicator', False) or imputer.strategy != 'median':
        raise ValueError('UNSUPPORTED_LOCKED_IMPUTER')
    x = raw_features.to(dtype=torch.float64)
    stats = torch.as_tensor(imputer.statistics_, dtype=x.dtype, device=x.device)
    if x.shape[-1] != len(stats) or not np.isfinite(imputer.statistics_).all():
        raise ValueError('IMPUTER_FEATURE_GEOMETRY')
    x = torch.where(torch.isnan(x), stats, x)
    mean = torch.as_tensor(scaler.mean_, dtype=x.dtype, device=x.device)
    scale = torch.as_tensor(scaler.scale_, dtype=x.dtype, device=x.device)
    coef = torch.as_tensor(cox.coef_, dtype=x.dtype, device=x.device)
    eta = (((x - mean) / scale) * coef).sum(dim=-1)
    h0 = float(cox.cum_baseline_hazard_(float(horizon)))
    if not np.isfinite(h0) or h0 < 0:
        raise ValueError('INVALID_BASELINE_HAZARD')
    if h0 == 0:
        return eta * 0
    # Beyond exp(-745), survival underflows to zero in float64. Avoid an
    # intermediate inf and its NaN backward product without changing the risk.
    log_hazard = torch.clamp(eta + np.log(h0), max=np.log(745.))
    return -torch.expm1(-torch.exp(log_hazard))


class ComponentTarget:
    def __init__(self, model, head, raw_row, features, modality, policy, grid, times, horizon=1095.75):
        self.model, self.head = model, head
        self.amp = bool(policy['amp'])
        self.raw_row = torch.as_tensor(raw_row, dtype=torch.float64).reshape(1, -1)
        self.positions = [features.index(f'{modality}_risk_{i}') for i in (1, 2, 3)]
        self.policy, self.grid, self.times, self.horizon = policy, grid, times, horizon

    def __call__(self, image):
        # Match the trained execution's AMP forward; IG separately scales and
        # unscales input gradients to avoid fp16 backward underflow.
        with torch.autocast(image.device.type, enabled=self.amp and image.device.type == 'cuda'):
            score = expert_scores(self.model, image, self.policy, self.times, self.grid)
        values = self.raw_row.expand(len(image), -1).clone()
        values[:, self.positions] = score.to(dtype=torch.float64, device='cpu')
        return cox_risk(self.head, values, self.horizon).sum()


def ensemble_mean(targets, image):
    if not targets:
        raise ValueError('EMPTY_ENSEMBLE')
    return sum(target(image) for target in targets) / len(targets)
