from torchvision import transforms
from cvlabkit.component.base import Transform
from cvlabkit.core.config import Config

class Normalize(Transform):
    """
    A transform component that normalizes a tensor image with mean and standard deviation.
    """
    def __init__(self, cfg: Config):
        """
        Initializes the Normalize transform.

        Args:
            cfg (Config): The configuration object. Expected keys:
                - "mean" (list of floats): The sequence of means for each channel.
                - "std" (list of floats): The sequence of standard deviations for each channel.
        """
        mean = cfg.get("mean", [0.485, 0.456, 0.406])
        std = cfg.get("std", [0.229, 0.224, 0.225])
        self.transform = transforms.Normalize(mean=mean, std=std)

    def __call__(self, sample, **kwargs):
        """
        Applies the normalization transformation.

        Args:
            sample (torch.Tensor): The tensor image to be normalized.
            **kwargs: Ignored, for compatibility.

        Returns:
            torch.Tensor: The normalized tensor.
        """
        return self.transform(sample)