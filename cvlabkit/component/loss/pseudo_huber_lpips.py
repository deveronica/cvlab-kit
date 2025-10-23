"""Combined Pseudo-Huber and LPIPS loss with time weighting."""

from __future__ import annotations

from cvlabkit.component.base import Loss
from cvlabkit.component.loss.lpips import LPIPS
from cvlabkit.component.loss.pseudo_huber import PseudoHuber


class PseudoHuberLPIPS(Loss):
    """Combined Pseudo-Huber and LPIPS loss with time-based weighting.

    This loss combines Pseudo-Huber loss for flow matching with LPIPS perceptual
    loss, weighted by the time variable to balance different phases of training.
    """

    def __init__(self, cfg):
        """Initialize combined loss.

        Args:
            cfg: Configuration object with parameters:
                - data_dim: For Pseudo-Huber loss (default: 3)
                - lpips_kwargs: Dict of kwargs for LPIPS (default: {})
        """
        super().__init__()

        # Create sub-losses
        pseudo_huber_cfg = type(
            "Config", (), {"get": lambda _, k, d=None: cfg.get(k, d)}
        )()
        lpips_cfg_dict = cfg.get("lpips_kwargs", {})
        lpips_cfg = type(
            "Config", (), {"get": lambda _, k, d=None: lpips_cfg_dict.get(k, d)}
        )()

        self.pseudo_huber = PseudoHuber(pseudo_huber_cfg)
        self.lpips = LPIPS(lpips_cfg)

    def forward(
        self, pred_flow, target_flow, pred_data=None, times=None, data=None, **kwargs
    ):
        """Compute combined loss.

        Args:
            pred_flow: Predicted flow
            target_flow: Target flow
            pred_data: Predicted data (required for LPIPS)
            times: Time values for weighting (required)
            data: Target data (required for LPIPS)
            **kwargs: Additional arguments

        Returns:
            Combined loss value
        """
        # Compute Pseudo-Huber loss on flow
        huber_loss = self.pseudo_huber(pred_flow, target_flow, reduction="none")

        # Compute LPIPS loss on data
        lpips_loss = self.lpips(pred_data, data, reduction="none")

        # Time-weighted combination
        # Huber weighted by (1 - t), LPIPS weighted by (1 / t)
        time_weighted_loss = huber_loss * (1 - times) + lpips_loss * (
            1.0 / times.clamp(min=1e-1)
        )

        return time_weighted_loss.mean()
