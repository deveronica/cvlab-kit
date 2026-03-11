"""Data normalization transforms for generative models."""

from __future__ import annotations

from cvlabkit.component.base import Transform


class DataNormalizer(Transform):
    """Normalize/unnormalize data to specific ranges.

    Useful for image generation models that expect specific input ranges.
    """

    def __init__(self, cfg):
        """Initialize data normalizer.

        Args:
            cfg: Configuration object with parameters:
                - mode: Normalization mode (default: 'neg_one_to_one')
                  Options:
                    - 'neg_one_to_one': [0, 1] → [-1, 1]
                    - 'zero_to_one': [-1, 1] → [0, 1]
                    - 'identity': No transformation
        """
        super().__init__()
        self.mode = cfg.get("mode", "neg_one_to_one")

        if self.mode not in ["neg_one_to_one", "zero_to_one", "identity"]:
            raise ValueError(f"Unknown normalization mode: {self.mode}")

    def __call__(self, data):
        """Apply normalization.

        Args:
            data: Input tensor

        Returns:
            Normalized tensor
        """
        if self.mode == "neg_one_to_one":
            # [0, 1] → [-1, 1]
            return data * 2 - 1
        elif self.mode == "zero_to_one":
            # [-1, 1] → [0, 1]
            return (data + 1) * 0.5
        else:  # identity
            return data
