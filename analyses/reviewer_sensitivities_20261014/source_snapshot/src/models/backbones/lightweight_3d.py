from __future__ import annotations

try:
    import torch
    from torch import nn
except Exception:  # pragma: no cover
    torch = None
    nn = object

from src.models.backbones.backbone_factory import build_3d_backbone
from src.models.fusion.bottleneck_cross_attention import BottleneckCrossAttention
from src.models.fusion.moe_fusion import MoEFusion


class Lightweight3DEncoder(nn.Module):
    """Small 3D CNN encoder used for CPU smoke and tiny-backbone ablations."""

    def __init__(self, in_channels: int = 1, base_channels: int = 8, out_dim: int = 128) -> None:
        if torch is None:
            raise RuntimeError("PyTorch is required for Lightweight3DEncoder")
        super().__init__()
        channels = int(base_channels)
        hidden = max(channels * 2, 4)
        self.net = nn.Sequential(
            nn.Conv3d(in_channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(channels),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),
            nn.Conv3d(channels, hidden, kernel_size=3, padding=1),
            nn.BatchNorm3d(hidden),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(hidden, out_dim),
        )
        self.out_dim = out_dim

    def forward(self, image):
        return self.net(image)


class MethodX3DClassifier(nn.Module):
    """Method X 3D MRI + tabular classifier (PET-free)."""

    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 8,
        latent_dim: int = 128,
        tab_hidden_dim: int = 32,
        tab_dim: int = 5,
        fusion: str = "cross_attention",
        dropout: float = 0.1,
        backbone: str = "efficientnet3d_b0",
        align_dim: int | None = None,
        bottleneck_dim: int = 32,
        fusion_hidden_dim: int | None = None,
        gradient_checkpointing: bool = False,
        pretrained_checkpoint: str | None = None,
        attention: str = "none",
        alignment_head: str = "linear",
        alignment_head_mode: str = "linear",
        tab_encoder: str = "linear",
        causal_adjustment: str = "none",
        n_experts: int = 4,
        fusion_num_heads: int = 2,
        moe_load_balance_lambda: float = 0.0,
        channel_gate: str = "none",
        hybrid_variant: str = "baseline",
        hybrid_attention_heads: int = 2,
        hybrid_attention_window_size: int = 4,
    ) -> None:
        if torch is None:
            raise RuntimeError("PyTorch is required for MethodX3DClassifier")
        # Fail-closed surface: this constructor implements exactly the locked
        # default configuration. Any non-default research flag is rejected
        # instead of being silently ignored, so a contract can never drift
        # from the executed architecture.
        _defaults = {
            "attention": "none",
            "alignment_head": "linear",
            "alignment_head_mode": "linear",
            "tab_encoder": "linear",
            "causal_adjustment": "none",
            "channel_gate": "none",
            "n_experts": 4,
            "fusion_num_heads": 2,
            "moe_load_balance_lambda": 0.0,
        }
        _given = {
            "attention": str(attention or "none").lower(),
            "alignment_head": str(alignment_head or "linear").lower(),
            "alignment_head_mode": str(alignment_head_mode or "linear").lower(),
            "tab_encoder": str(tab_encoder or "linear").lower(),
            "causal_adjustment": str(causal_adjustment or "none").lower(),
            "channel_gate": str(channel_gate or "none").lower(),
            "n_experts": int(n_experts),
            "fusion_num_heads": int(fusion_num_heads),
            "moe_load_balance_lambda": float(moe_load_balance_lambda),
        }
        _violations = sorted(key for key in _defaults if _given[key] != _defaults[key])
        if _violations:
            raise ValueError(
                "MethodX3DClassifier non-default research flags are not implemented: "
                + ", ".join(f"{key}={_given[key]!r}" for key in _violations)
            )
        if pretrained_checkpoint:
            raise ValueError("MethodX3DClassifier pretrained loading is not implemented in this surface.")
        super().__init__()
        self.gradient_checkpointing = bool(gradient_checkpointing)
        self.pretrained_load_report: dict | None = None
        self.fusion = fusion.lower().strip()
        self.latent_dim = int(latent_dim)
        self.tab_hidden_dim = int(tab_hidden_dim)
        self.backbone_name = backbone.lower().strip()
        self.hybrid_variant = str(hybrid_variant).lower().strip().replace("-", "_")
        if self.backbone_name != "amuse_x_hybrid3d" and self.hybrid_variant != "baseline":
            raise ValueError("HYBRID_VARIANT_REQUIRES_HYBRID_BACKBONE")
        hidden = int(latent_dim if not fusion_hidden_dim else fusion_hidden_dim)
        concat_align = int(align_dim if align_dim is not None else max(latent_dim // 2, tab_hidden_dim))

        if self.backbone_name == "tiny":
            self.encoder = Lightweight3DEncoder(
                in_channels=in_channels,
                base_channels=base_channels,
                out_dim=self.latent_dim,
            )
        else:
            self.encoder = build_3d_backbone(
                self.backbone_name, in_channels=in_channels, out_dim=self.latent_dim,
                hybrid_variant=self.hybrid_variant,
                hybrid_attention_heads=hybrid_attention_heads,
                hybrid_attention_window_size=hybrid_attention_window_size,
            )

        self.tab_encoder = nn.Sequential(
            nn.Linear(tab_dim, tab_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(tab_hidden_dim, tab_hidden_dim),
        )
        if self.fusion == "concat":
            self.fuser = None
            self.alignment_projection = nn.Linear(tab_hidden_dim, concat_align)
            self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(self.latent_dim + concat_align, 1))
            self._concat_align_dim = concat_align
        elif self.fusion in {"cross_attention", "cross-attention"}:
            self.alignment_projection = nn.Linear(tab_hidden_dim, self.latent_dim)
            self.fuser = BottleneckCrossAttention(
                mri_dim=self.latent_dim,
                tab_dim=tab_hidden_dim,
                hidden_dim=hidden,
                bottleneck_dim=bottleneck_dim,
                num_heads=2,
                dropout=dropout,
            )
            self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, 1))
        elif self.fusion == "moe":
            self.alignment_projection = nn.Linear(tab_hidden_dim, self.latent_dim)
            self.fuser = MoEFusion(
                mri_dim=self.latent_dim,
                tab_dim=tab_hidden_dim,
                n_experts=4,
                out_dim=hidden,
                dropout=dropout,
            )
            self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, 1))
        else:
            raise ValueError(f"Unsupported fusion mode: {fusion}")

    def encode(self, image, tabular):
        if self.gradient_checkpointing:
            import torch.utils.checkpoint as torch_checkpoint

            mri_latent = torch_checkpoint.checkpoint(self.encoder, image, use_reentrant=False)
        else:
            mri_latent = self.encoder(image)
        if mri_latent.ndim > 2:
            mri_latent = mri_latent.view(mri_latent.size(0), -1)
        tab_latent = self.tab_encoder(tabular)
        return mri_latent, tab_latent

    def fused_features(self, mri_latent, tab_latent, tab_presence=None):
        if self.fuser is None:
            aligned = self.alignment_projection(tab_latent)
            return torch.cat([mri_latent, aligned], dim=-1)
        return self.fuser(mri_latent, tab_latent, tab_presence=tab_presence)

    def logits_from_latents(self, mri_latent, tab_latent, tab_presence=None):
        features = self.fused_features(mri_latent, tab_latent, tab_presence=tab_presence)
        return self.head(features).squeeze(-1)

    def forward_with_latents(self, image, tabular, tab_presence=None):
        mri_latent, tab_latent = self.encode(image, tabular)
        logits = self.logits_from_latents(mri_latent, tab_latent, tab_presence=tab_presence)
        aligned_tab = self.alignment_projection(tab_latent)
        return logits, mri_latent, aligned_tab

    def forward(self, image, tabular, tab_presence=None):
        return self.forward_with_latents(image, tabular, tab_presence=tab_presence)[0]
