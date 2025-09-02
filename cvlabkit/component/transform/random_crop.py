from torchvision import transforms

from cvlabkit.component.base import Transform
from cvlabkit.core.config import Config


class RandomCrop(Transform):
    """
    A transform component that crops the given image at a random location.
    """
    def __init__(self, cfg: Config):
        """
        Initializes the RandomCrop transform.

        Args:
            cfg (Config): The configuration object. Expected keys:
                - "size" (int or sequence): The desired output size of the crop.
                - "padding" (int or sequence, optional): The optional padding on each border of the image. Defaults to None.
                - "pad_if_needed" (bool, optional): It will pad the image if smaller than the desired size to avoid raising an exception. Defaults to False.
                - "fill" (int or tuple, optional): Pixel fill value for constant fill. Defaults to 0.
                - "padding_mode" (str, optional): Type of padding. Should be: 'constant', 'edge', 'reflect' or 'symmetric'. Defaults to 'constant'.
        """
        size = cfg.size
        padding = cfg.get("padding", None)
        pad_if_needed = cfg.get("pad_if_needed", False)
        fill = cfg.get("fill", 0)
        padding_mode = cfg.get("padding_mode", "reflect")

        self.transform = transforms.RandomCrop(
            size=size,
            padding=padding,
            pad_if_needed=pad_if_needed,
            fill=fill,
            padding_mode=padding_mode
        )

    def __call__(self, sample, **kwargs):
        """
        Applies the random crop transformation.

        Args:
            sample (PIL.Image): The input image.
            **kwargs: Ignored, for compatibility.

        Returns:
            PIL.Image: The cropped image.
        """
        return self.transform(sample)
