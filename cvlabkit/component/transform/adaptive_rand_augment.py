# cvlabkit/component/transform/adaptive_rand_augment.py

from torchvision import transforms

from cvlabkit.component.base import Transform
from cvlabkit.core.config import Config


class AdaptiveRandAugment(Transform):
    """Apply RandAugment with adaptive magnitude based on difficulty score.

    The magnitude is dynamically calculated from a continuous difficulty score,
    where higher difficulty results in stronger augmentation.
    """

    def __init__(self, cfg: Config):
        """Initializes the AdaptiveRandAugment transform policies.

        Args:
            cfg (Config): The configuration object. Expected keys:
                - "magnitude_min" (int): The minimum magnitude (for hardest samples, score=1.0).
                - "magnitude_max" (int): The maximum magnitude (for easiest samples, score=0.0).
                - "num_ops" (int, optional): The number of augmentation operators. Defaults to 2.
        """
        self.magnitude_min = cfg.get("magnitude_min")
        self.magnitude_max = cfg.get("magnitude_max")
        self.num_ops = cfg.get("num_ops", 2)

        if self.magnitude_min is None or self.magnitude_max is None:
            raise ValueError(
                "AdaptiveRandAugment requires 'magnitude_min' and 'magnitude_max' parameters."
            )

        # Pre-create RandAugment objects for all possible magnitudes (cache)
        self._augmenters = {
            mag: transforms.RandAugment(num_ops=self.num_ops, magnitude=mag)
            for mag in range(self.magnitude_min, self.magnitude_max + 1)
        }

    def __call__(self, sample, **kwargs):
        """Applies RandAugment with a magnitude interpolated from the difficulty score.

        A high difficulty score (e.g., 1.0) results in a high magnitude (strong augmentation),
        while a low score (e.g., 0.0) results in a low magnitude (weak augmentation).

        Args:
            sample (PIL.Image): The input image.
            **kwargs: Expected to contain 'difficulty_score' (float between 0 and 1).

        Returns:
            PIL.Image: The augmented image.
        """
        # Default to the easiest case if no score is provided.
        difficulty_score = kwargs.get("difficulty_score", 0.0)

        # Linear interpolation for the magnitude
        # High difficulty → High magnitude (strong augmentation)
        # magnitude = min + (max - min) * score
        magnitude = (
            self.magnitude_min
            + (self.magnitude_max - self.magnitude_min) * difficulty_score
        )

        # Ensure magnitude is an integer and clamp to valid range
        final_magnitude = int(round(magnitude))
        final_magnitude = max(self.magnitude_min, min(self.magnitude_max, final_magnitude))

        # Use cached augmenter instead of creating new object
        return self._augmenters[final_magnitude](sample)
