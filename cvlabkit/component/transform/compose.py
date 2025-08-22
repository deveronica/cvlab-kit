# cvlabkit/component/transform/compose.py

from typing import List

from torchvision import transforms

from cvlabkit.core.config import Config
from cvlabkit.component.base import Transform


class Compose(Transform):
    """
    A transform component that composes a list of other transform components.

    This component wraps a torchvision.transforms.Compose instance. Thanks to
    the InterfaceMeta metaclass, any calls to this component (like __call__)
    that are not explicitly defined here will be automatically delegated to the
    internal self.pipeline object. It is initialized with a list of already-instantiated
    transform components.
    """
    def __init__(self, cfg: Config, components: List[Transform]):
        """
        Initializes the Compose transform.

        Args:
            cfg (Config): The configuration object for this component.
            components (List[Transform]): A list of PRE-INSTANTIATED transform components.
        """
        # The 'components' argument is now a list of actual transform objects,
        # not a list of configurations.
        self.pipeline = transforms.Compose(transforms=components)

    def __call__(self, sample, **kwargs):
        """
        Applies the sequence of transformations to the sample.
        
        This explicit __call__ method ensures that any additional keyword
        arguments (like 'difficulty_score') are correctly passed down to each
        sub-component in the pipeline, which is not the default behavior of
        torchvision.transforms.Compose.
        """
        for t in self.pipeline.transforms:
            # Check if the component is a class from our framework to avoid errors
            # with standard torchvision transforms that don't accept kwargs.
            if isinstance(t, Transform):
                 sample = t(sample, **kwargs)
            else:
                 sample = t(sample)
        return sample