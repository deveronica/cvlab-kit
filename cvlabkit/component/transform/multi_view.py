from cvlabkit.component.base import Transform


class MultiView(Transform):
    """A meta-transform that applies multiple transforms to the same image.

    This is useful for semi-supervised learning techniques like FixMatch
    that require multiple augmented "views" of a single image.
    """

    def __init__(self, cfg, transforms_to_apply=None):
        """Args:
        cfg: The configuration object.
        transforms_to_apply (list): A list of transform components to apply.
        """
        super().__init__()
        if transforms_to_apply is None or not isinstance(transforms_to_apply, list):
            raise ValueError(
                "MultiView transform requires a 'transforms_to_apply' list."
            )
        self.transforms = transforms_to_apply

    def __call__(self, img):
        """Applies each transform to the image and returns a tuple of results."""
        return tuple(tf(img.copy()) for tf in self.transforms)
