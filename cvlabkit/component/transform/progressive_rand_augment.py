# cvlabkit/component/transform/progressive_rand_augment.py

from torchvision import transforms

from cvlabkit.component.base import Transform
from cvlabkit.core.config import Config


class ProgressiveRandAugment(Transform):
    """Apply RandAugment with progressive magnitude based on strength parameter.

    Unlike AdaptiveRandAugment which uses difficulty_score (higher = less augmentation),
    this uses strength (higher = more augmentation) for better semantic clarity.
    """

    def __init__(self, cfg: Config):
        """Initializes the ProgressiveRandAugment transform policies.

        Args:
            cfg (Config): The configuration object. Expected keys:
                - "magnitude_min" (int): The minimum magnitude (for strength=0.0).
                - "magnitude_max" (int): The maximum magnitude (for strength=1.0).
                - "num_ops" (int, optional): The number of augmentation operators. Defaults to 2.
        """
        self.magnitude_min = cfg.get("magnitude_min")
        self.magnitude_max = cfg.get("magnitude_max")
        self.num_ops = cfg.get("num_ops", 2)

        if self.magnitude_min is None or self.magnitude_max is None:
            raise ValueError(
                "ProgressiveRandAugment requires 'magnitude_min' and 'magnitude_max' parameters."
            )

    def __call__(self, sample, **kwargs):
        """Applies RandAugment with a magnitude interpolated from the augmentation strength.

        A low strength (e.g., 0.0) results in low magnitude (close to min),
        while a high strength (e.g., 1.0) results in high magnitude (close to max).

        Args:
            sample (PIL.Image): The input image.
            **kwargs: Expected to contain 'strength' (float between 0 and 1).

        Returns:
            PIL.Image: The augmented image.
        """
        # Default to no augmentation if no strength is provided
        strength = kwargs.get("strength", 0.0)

        # Linear interpolation for the magnitude
        # magnitude = min + (max - min) * strength
        magnitude = (
            self.magnitude_min + (self.magnitude_max - self.magnitude_min) * strength
        )

        # Ensure magnitude is an integer for RandAugment
        final_magnitude = int(round(magnitude))

        augmenter = transforms.RandAugment(
            num_ops=self.num_ops, magnitude=final_magnitude
        )
        return augmenter(sample)
