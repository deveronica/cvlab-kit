"""Mean-variance network loss using negative log probability."""

from __future__ import annotations

from torch.distributions import Normal

from cvlabkit.component.base import Loss


class MeanVariance(Loss):
    """Loss for mean-variance networks using negative log probability.

    This loss assumes the model outputs (mean, variance) and computes the
    negative log probability of the target under the predicted normal distribution.
    """

    def __init__(self, cfg):
        """Initialize mean-variance loss.

        Args:
            cfg: Configuration object (no parameters needed)
        """
        super().__init__()

    def forward(self, pred, target, **kwargs):
        """Compute negative log probability loss.

        Args:
            pred: Tuple of (mean, variance) predictions
            target: Target values
            **kwargs: Additional arguments (unused)

        Returns:
            Negative log probability loss
        """
        dist = Normal(*pred)
        return -dist.log_prob(target).mean()
