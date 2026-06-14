from __future__ import annotations

try:
    import torch
    import torch.nn as nn
except Exception:  # pragma: no cover
    torch = None
    nn = object


class PatchEmbedding3D(nn.Module):
    """3D patch embedding for volumetric data."""

    def __init__(self, img_size: int = 96, patch_size: int = 16, in_channels: int = 1, embed_dim: int = 256) -> None:
        if torch is None:
            raise RuntimeError("PyTorch is required for PatchEmbedding3D")
        super().__init__()
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 3
        self.proj = nn.Conv3d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class MAE3D(nn.Module):
    """3D Masked Autoencoder for self-supervised pretraining on MRI volumes.

    Based on the MAE architecture (He et al., 2022) adapted for 3D medical imaging.
    Masks a high proportion of patches and reconstructs them, learning robust
    3D anatomical representations without labels.
    """

    def __init__(
        self,
        img_size: int = 96,
        patch_size: int = 16,
        in_channels: int = 1,
        embed_dim: int = 256,
        encoder_depth: int = 6,
        encoder_heads: int = 8,
        decoder_embed_dim: int = 128,
        decoder_depth: int = 4,
        decoder_heads: int = 4,
        mask_ratio: float = 0.75,
    ) -> None:
        if torch is None:
            raise RuntimeError("PyTorch is required for MAE3D")
        super().__init__()
        self.patch_size = patch_size
        self.mask_ratio = mask_ratio
        self.n_patches = (img_size // patch_size) ** 3

        self.patch_embed = PatchEmbedding3D(img_size, patch_size, in_channels, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.n_patches + 1, embed_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=encoder_heads, dim_feedforward=embed_dim * 4, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=encoder_depth)
        self.encoder_norm = nn.LayerNorm(embed_dim)

        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, self.n_patches + 1, decoder_embed_dim))

        decoder_layer = nn.TransformerEncoderLayer(
            d_model=decoder_embed_dim, nhead=decoder_heads, dim_feedforward=decoder_embed_dim * 4, batch_first=True
        )
        self.decoder = nn.TransformerEncoder(decoder_layer, num_layers=decoder_depth)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size**3 * in_channels)

    def random_masking(self, x: torch.Tensor, mask_ratio: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Random masking of patches."""
        batch_size, seq_len, _ = x.shape
        n_keep = int(seq_len * (1 - mask_ratio))
        noise = torch.rand(batch_size, seq_len, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        ids_keep = ids_shuffle[:, :n_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).expand(-1, -1, x.shape[-1]))
        mask = torch.ones(batch_size, seq_len, device=x.device)
        mask[:, :n_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)
        return x_masked, mask, ids_restore

    def forward_encoder(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.patch_embed(x)
        x = x + self.pos_embed[:, 1:, :]
        x, mask, ids_restore = self.random_masking(x, self.mask_ratio)
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = self.encoder(x)
        x = self.encoder_norm(x)
        return x, mask, ids_restore

    def forward_decoder(self, x: torch.Tensor, ids_restore: torch.Tensor) -> torch.Tensor:
        x = self.decoder_embed(x)
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).expand(-1, -1, x_.shape[-1]))
        x = torch.cat([x[:, :1, :], x_], dim=1)
        x = x + self.decoder_pos_embed
        x = self.decoder(x)
        x = self.decoder_pred(x)
        x = x[:, 1:, :]
        return x

    def patchify(self, imgs: torch.Tensor) -> torch.Tensor:
        p = self.patch_size
        b, c, d, h, w = imgs.shape
        d, h, w = d // p, h // p, w // p
        x = imgs.reshape(b, c, d, p, h, p, w, p)
        x = x.permute(0, 2, 4, 6, 1, 3, 5, 7).reshape(b, d * h * w, c * p**3)
        return x

    def unpatchify(self, x: torch.Tensor, b: int, c: int, d: int, h: int, w: int) -> torch.Tensor:
        p = self.patch_size
        d, h, w = d // p, h // p, w // p
        x = x.reshape(b, d, h, w, c, p, p, p)
        x = x.permute(0, 4, 1, 5, 2, 6, 3, 7).reshape(b, c, d * p, h * p, w * p)
        return x

    def forward(self, imgs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        latent, mask, ids_restore = self.forward_encoder(imgs)
        pred = self.forward_decoder(latent, ids_restore)
        target = self.patchify(imgs)
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)
        loss = (loss * mask).sum() / mask.sum()
        return loss, pred, mask
