from torchvision import transforms

from cvlabkit.component.base import Transform
from cvlabkit.core.config import Config


class Resize(Transform):
    """
    A transform component that resizes an input image to a specified size.
    """
    def __init__(self, cfg: Config):
        """
        Initializes the Resize transform.

        Args:
            cfg (Config): The configuration object. Expected keys:
                - "size" (int or tuple): The desired output size.
        """
        size = cfg.get("size")
        if size is None:
            raise ValueError("Resize component requires a 'size' parameter in the config.")
        
        # This component wraps the torchvision Resize transform.
        self.transform = transforms.Resize((size, size) if isinstance(size, int) else size)

    def __call__(self, sample, **kwargs):
        """
        Applies the resize transformation.

        Args:
            sample (PIL.Image): The input image to be resized.
            **kwargs: Ignored, for compatibility with other transforms.

        Returns:
            PIL.Image: The resized image.
        """
        return self.transform(sample)