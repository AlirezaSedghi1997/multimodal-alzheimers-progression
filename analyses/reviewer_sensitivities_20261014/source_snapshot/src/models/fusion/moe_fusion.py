from __future__ import annotations

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except Exception:  # pragma: no cover
    torch = None
    nn = object
    F = None


class MoEFusion(nn.Module):
    """Mixture of Experts for MRI-tabular fusion.

    Uses a learned gating network to weight contributions from multiple
    expert networks, allowing specialization on different input subspaces.
    """

    def __init__(
        self,
        mri_dim: int,
        tab_dim: int,
        n_experts: int = 4,
        out_dim: int = 128,
        dropout: float = 0.1,
    ) -> None:
        if torch is None:
            raise RuntimeError("PyTorch is required for MoEFusion")
        super().__init__()
        combined_dim = mri_dim + tab_dim
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(combined_dim, out_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                )
                for _ in range(n_experts)
            ]
        )
        self.gate = nn.Sequential(
            nn.Linear(combined_dim, n_experts),
        )
        self.out_dim = out_dim

    def forward(self, mri_feat: torch.Tensor, tab_feat: torch.Tensor, tab_presence=None) -> torch.Tensor:
        combined = torch.cat([mri_feat, tab_feat], dim=-1)
        if tab_presence is not None:
            combined = combined * tab_presence.reshape(-1, 1)
        gate_weights = F.softmax(self.gate(combined), dim=-1)
        expert_outputs = torch.stack([e(combined) for e in self.experts], dim=1)
        return (gate_weights.unsqueeze(-1) * expert_outputs).sum(dim=1)
