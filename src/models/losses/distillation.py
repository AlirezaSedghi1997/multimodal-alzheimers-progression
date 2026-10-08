from __future__ import annotations

try:
    import torch
    import torch.nn.functional as F
except Exception:  # pragma: no cover
    torch = None
    F = None


def _require_torch():
    if torch is None or F is None:
        raise RuntimeError("PyTorch is required for Method Y distillation losses")


def kd_logit_loss(student_logits, teacher_logits, *, temperature: float = 2.0):
    """Temperature-scaled KL distillation between 2-class logits.

    Conventional Hinton KD: T^2 * KL( softmax(teacher/T) || softmax(student/T) ).
    The teacher distribution is the target; the student is trained to match it.
    Teacher logits are detached so gradients flow only through the student.
    """
    _require_torch()
    temp = float(temperature)
    if temp <= 0.0:
        raise ValueError("temperature must be > 0")
    teacher = teacher_logits.detach().to(dtype=student_logits.dtype, device=student_logits.device)
    log_s = F.log_softmax(student_logits / temp, dim=-1)
    p_t = F.softmax(teacher / temp, dim=-1)
    return (temp * temp) * F.kl_div(log_s, p_t, reduction="batchmean")


def cosine_feature_loss(student_feat, teacher_feat_mri_only):
    """1 - cosine_similarity(student_feat, teacher_feat_mri_only).

    ``teacher_feat_mri_only`` MUST be the teacher's MRI-branch latent (z_mri), never the
    PET-branch latent. This contract is enforced by the caller (see
    ``extract_teacher_mri_latent``); this function never touches a PET tensor.
    """
    _require_torch()
    s = student_feat.reshape(student_feat.size(0), -1)
    t = teacher_feat_mri_only.detach().to(dtype=s.dtype, device=s.device).reshape(s.size(0), -1)
    cos = F.cosine_similarity(s, t, dim=-1, eps=1e-8)
    return (1.0 - cos).mean()


def supervised_bce_loss(student_logits, labels, *, pos_weight=None):
    """BCE on the student positive-class logit (student_logits[:, 0])."""
    _require_torch()
    logit = student_logits[:, 0].reshape(-1)
    target = labels.to(dtype=logit.dtype, device=logit.device).reshape(-1)
    return F.binary_cross_entropy_with_logits(logit, target, pos_weight=pos_weight)


def method_y_distillation_loss(
    student_logits,
    teacher_logits,
    student_feat,
    teacher_feat_mri_only,
    labels,
    *,
    temperature: float = 2.0,
    lambda_kd_logits: float = 1.0,
    lambda_kd_feat: float = 0.5,
    lambda_ce: float = 1.0,
    pos_weight=None,
):
    """Stage S3 Y->X distillation loss (PET-free student).

    total = lambda_kd_logits * KL(student||teacher/T)
          + lambda_kd_feat * (1 - cos(student_feat, teacher_feat_mri_only))
          + lambda_ce * BCE(student_logits, label)

    Distillation uses ONLY the teacher's MRI-branch latent (``teacher_feat_mri_only``).
    The PET-branch latent of the teacher is never consumed here (non-negotiable PET-free
    contract for the deployed student).
    """
    _require_torch()
    kd = kd_logit_loss(student_logits, teacher_logits, temperature=temperature)
    feat = cosine_feature_loss(student_feat, teacher_feat_mri_only)
    ce = supervised_bce_loss(student_logits, labels, pos_weight=pos_weight)
    total = (
        float(lambda_kd_logits) * kd
        + float(lambda_kd_feat) * feat
        + float(lambda_ce) * ce
    )
    parts = {
        "total": float(total.detach().cpu()),
        "kd_logit": float(kd.detach().cpu()),
        "cosine_feature": float(feat.detach().cpu()),
        "supervised_bce": float(ce.detach().cpu()),
        "lambda_kd_logits": float(lambda_kd_logits),
        "lambda_kd_feat": float(lambda_kd_feat),
        "lambda_ce": float(lambda_ce),
        "temperature": float(temperature),
        "teacher_feature_source": "mri_branch_latent_only",
    }
    return total, parts


def extract_teacher_mri_latent(teacher, image_mri, tabular=None):
    """Return the teacher's MRI-branch latent (z_mri) under no_grad.

    Enforces the PET-free distillation contract: only the MRI-branch latent is extracted
    for feature alignment; the teacher's PET branch is never read by the student loss.
    """
    _require_torch()
    with torch.no_grad():
        out = teacher(image_mri, tabular=tabular) if _accepts_tabular(teacher) else teacher(image_mri)
    if isinstance(out, dict):
        if "z_mri" in out:
            return out["z_mri"].detach()
        if "mri_latent" in out:
            return out["mri_latent"].detach()
    raise ValueError("teacher output must expose z_mri / mri_latent for PET-free distillation")


def _accepts_tabular(module) -> bool:
    import inspect

    try:
        params = list(inspect.signature(module.forward).parameters)
        return "tabular" in params
    except (TypeError, ValueError):
        return False
