"""AMUSE-X Pillar A2 phenotypic contrastive loss with explicit cohort-proxy exclusion."""

from __future__ import annotations

try:
    import torch
    import torch.nn.functional as F
    from torch import Tensor
except Exception:  # pragma: no cover
    torch = None
    F = None
    Tensor = object


DEFAULT_FORBIDDEN_FEATURES = {"cohort_id", "site", "scanner_id", "study", "cohort"}


def _normalize_feature_names(feature_names: list[str]) -> list[str]:
    return [str(name).strip().lower() for name in feature_names]


def phenotypic_contrastive_loss(
    mri_latent: Tensor,
    phenotype_features: Tensor,
    phenotype_feature_names: list[str],
    forbidden_features: set[str] = DEFAULT_FORBIDDEN_FEATURES,
    temperature: float = 0.1,
    similarity_threshold: float = 0.7,
) -> Tensor:
    """
    Soft positive contrastive loss using cosine-similarity in phenotype space.

    Raises if any forbidden feature name is in phenotype_feature_names.
    """

    if torch is None or F is None:
        raise RuntimeError("PyTorch is required for phenotypic_contrastive_loss")
    normalized_names = _normalize_feature_names(phenotype_feature_names)
    forbidden = {str(name).strip().lower() for name in forbidden_features}
    overlap = sorted(set(normalized_names) & forbidden)
    if overlap:
        raise ValueError("Forbidden phenotype feature(s): " + ", ".join(overlap))
    if mri_latent.ndim != 2:
        raise ValueError("mri_latent must have shape [B, D]")
    if phenotype_features.ndim != 2:
        raise ValueError("phenotype_features must have shape [B, P]")
    if mri_latent.shape[0] != phenotype_features.shape[0]:
        raise ValueError("mri_latent and phenotype_features must have the same batch size")
    if phenotype_features.shape[1] != len(normalized_names):
        raise ValueError("phenotype_feature_names length must match phenotype_features width")
    if float(temperature) <= 0.0:
        raise ValueError("temperature must be positive")
    batch = int(mri_latent.shape[0])
    if batch < 2:
        return mri_latent.sum() * 0.0

    latent = F.normalize(mri_latent.float(), dim=-1)
    phenotype = phenotype_features.float()
    phenotype = phenotype - phenotype.mean(dim=0, keepdim=True)
    phenotype = phenotype / phenotype.std(dim=0, keepdim=True, unbiased=False).clamp_min(1e-6)
    phenotype = F.normalize(phenotype, dim=-1)

    eye = torch.eye(batch, dtype=torch.bool, device=mri_latent.device)
    phenotype_sim = torch.mm(phenotype, phenotype.t()).clamp(-1.0, 1.0)
    positive_mask = (phenotype_sim >= float(similarity_threshold)) & ~eye
    mask_value = torch.finfo(phenotype_sim.dtype).min / 2.0
    target_logits = phenotype_sim / float(temperature)
    target_logits = target_logits.masked_fill(eye, mask_value)
    has_threshold_positive = positive_mask.any(dim=1, keepdim=True)
    target_logits = torch.where(has_threshold_positive & ~positive_mask, torch.full_like(target_logits, mask_value), target_logits)
    target = F.softmax(target_logits, dim=1).detach()

    latent_logits = torch.mm(latent, latent.t()) / float(temperature)
    latent_logits = latent_logits.masked_fill(eye, mask_value)
    log_prob = F.log_softmax(latent_logits, dim=1)
    return -(target * log_prob).sum(dim=1).mean()
