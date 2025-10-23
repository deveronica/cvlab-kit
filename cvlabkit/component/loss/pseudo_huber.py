"""Pseudo-Huber loss for robust regression."""

from __future__ import annotations

import torch.nn.functional as F
from einops import reduce

from cvlabkit.component.base import Loss


def _default(v, d):
    """Return v if it exists, else d (private helper)."""
    return v if v is not None else d


class PseudoHuber(Loss):
    """Pseudo-Huber loss for robust regression.

    Implements the loss from section 4.2 of https://arxiv.org/abs/2405.20320v1
    This loss is more robust to outliers than MSE.
    """

    def __init__(self, cfg):
        """Initialize Pseudo-Huber loss.

        Args:
            cfg: Configuration object with parameters:
                - data_dim: Dimensionality of data (default: 3 for RGB images)
        """
        super().__init__()
        self.data_dim = cfg.get("data_dim", 3)

    def forward(self, pred, target, reduction="mean", **kwargs):
        """Compute Pseudo-Huber loss.

        Args:
            pred: Predicted values
            target: Target values
            reduction: Reduction method ('mean', 'sum', or 'none')
            **kwargs: Additional arguments (data_dim can override config)

        Returns:
            Loss value
        """
        data_dim = _default(kwargs.get("data_dim"), self.data_dim)

        c = 0.00054 * data_dim
        loss = (F.mse_loss(pred, target, reduction=reduction) + c * c).sqrt() - c

        if reduction == "none":
            loss = reduce(loss, "b ... -> b", "mean")

        return loss
