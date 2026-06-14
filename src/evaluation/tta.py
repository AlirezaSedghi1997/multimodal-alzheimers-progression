from __future__ import annotations

try:
    import torch
    import torch.nn as nn
except Exception:  # pragma: no cover
    torch = None
    nn = object


class TTAPredictor:
    """Test-Time Augmentation wrapper for 3D medical image models.

    Applies augmentation at inference and averages predictions,
    reducing variance and improving calibration stability.
    """

    def __init__(self, model: nn.Module, n_augmentations: int = 5) -> None:
        if torch is None:
            raise RuntimeError("PyTorch is required for TTAPredictor")
        self.model = model
        self.n_augmentations = n_augmentations

    @torch.no_grad()
    def predict(self, image: torch.Tensor, tabular: torch.Tensor) -> torch.Tensor:
        """Predict with TTA by averaging over original + augmented versions.

        Args:
            image: MRI volume (batch_size, channels, D, H, W)
            tabular: Tabular features (batch_size, n_features)

        Returns:
            Averaged probability predictions (batch_size,)
        """
        self.model.eval()
        probs = [torch.sigmoid(self.model(image, tabular))]
        for _ in range(self.n_augmentations):
            aug_image = self._apply_test_augmentation(image)
            probs.append(torch.sigmoid(self.model(aug_image, tabular)))
        return torch.stack(probs).mean(dim=0)

    def _apply_test_augmentation(self, image: torch.Tensor) -> torch.Tensor:
        """Apply light augmentation for test-time augmentation."""
        aug = image.clone()
        if torch.rand(1).item() > 0.5:
            aug = torch.flip(aug, dims=[2])
        if torch.rand(1).item() > 0.5:
            aug = torch.flip(aug, dims=[3])
        if torch.rand(1).item() > 0.3:
            noise = torch.randn_like(aug) * 0.01
            aug = aug + noise
        return aug
