"""Noise scheduling transforms for time-conditioned models."""

from __future__ import annotations

import torch
from torch import pi

from cvlabkit.component.base import Transform


class NoiseSchedule(Transform):
    """Time scheduling transform for diffusion and flow models.

    Applies various scheduling functions to time values, useful for
    controlling noise levels in diffusion models or flow matching.
    """

    def __init__(self, cfg):
        """Initialize noise schedule.

        Args:
            cfg: Configuration object with parameters:
                - schedule: Schedule type (default: 'linear')
                  Options: 'linear', 'cosmap', 'identity'
        """
        super().__init__()
        schedule_type = cfg.get("schedule", "linear")

        if schedule_type == "cosmap":
            self.schedule_fn = self._cosmap
        elif schedule_type == "linear":
            self.schedule_fn = self._linear
        elif schedule_type == "identity":
            self.schedule_fn = self._identity
        else:
            raise ValueError(f"Unknown schedule type: {schedule_type}")

    def _cosmap(self, t):
        """Cosine mapping schedule.

        Algorithm 21 from https://arxiv.org/abs/2403.03206
        """
        return 1.0 - (1.0 / (torch.tan(pi / 2 * t) + 1))

    def _linear(self, t):
        """Linear schedule (identity)."""
        return t

    def _identity(self, t):
        """Identity schedule (no transformation)."""
        return t

    def __call__(self, times):
        """Apply schedule to time values.

        Args:
            times: Time tensor of any shape

        Returns:
            Scheduled time tensor (same shape as input)
        """
        return self.schedule_fn(times)
