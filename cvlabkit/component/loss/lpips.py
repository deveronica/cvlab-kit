"""LPIPS (Learned Perceptual Image Patch Similarity) loss using VGG16."""

from __future__ import annotations

import torch.nn.functional as F
import torchvision
from einops import reduce
from torch import nn
from torchvision.models import VGG16_Weights

from cvlabkit.component.base import Loss


def _exists(v):
    """Check if value is not None (private helper)."""
    return v is not None


class LPIPS(Loss):
    """LPIPS loss using VGG16 for perceptual similarity.

    This loss computes the perceptual similarity between images using features
    from a pre-trained VGG16 network.
    """

    def __init__(self, cfg):
        """Initialize LPIPS loss.

        Args:
            cfg: Configuration object with optional parameters:
                - vgg_weights: VGG16 weights to use (default: VGG16_Weights.DEFAULT)
        """
        super().__init__()

        # Get VGG weights from config
        vgg_weights_name = cfg.get("vgg_weights", "DEFAULT")
        if vgg_weights_name == "DEFAULT":
            vgg_weights = VGG16_Weights.DEFAULT
        else:
            vgg_weights = getattr(VGG16_Weights, vgg_weights_name)

        # Create VGG16 and modify classifier
        vgg = torchvision.models.vgg16(weights=vgg_weights)
        vgg.classifier = nn.Sequential(*vgg.classifier[:-2])

        # Store in list to avoid registering as submodule (VGG is frozen)
        self._vgg = [vgg]

    def forward(self, pred_data, data, reduction="mean"):
        """Compute LPIPS loss.

        Args:
            pred_data: Predicted images
            data: Target images
            reduction: Reduction method ('mean', 'sum', or 'none')

        Returns:
            Loss value
        """
        (vgg,) = self._vgg
        vgg = vgg.to(data.device)

        pred_embed, embed = map(vgg, (pred_data, data))

        loss = F.mse_loss(embed, pred_embed, reduction=reduction)

        if reduction == "none":
            loss = reduce(loss, "b ... -> b", "mean")

        return loss
