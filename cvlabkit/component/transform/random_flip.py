from torchvision import transforms

from cvlabkit.component.base import Transform
from cvlabkit.core.config import Config


class RandomFlip(Transform):
    """
    A transform component that horizontally flips an image with a given probability.
    """
    def __init__(self, cfg: Config):
        """
        Initializes the RandomFlip transform.

        Args:
            cfg (Config): The configuration object. Expected keys:
                - "p" (float, optional): The probability of the flip. Defaults to 0.5.
        """
        probability = cfg.get("p", 0.5)
        self.transform = transforms.RandomHorizontalFlip(p=probability)

    def __call__(self, sample, **kwargs):
        """
        Applies the random horizontal flip transformation.

        Args:
            sample (PIL.Image): The input image.
            **kwargs: Ignored, for compatibility.

        Returns:
            PIL.Image: The potentially flipped image.
        """
        return self.transform(sample)