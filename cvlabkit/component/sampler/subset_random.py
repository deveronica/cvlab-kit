from torch.utils.data import SubsetRandomSampler

from cvlabkit.component.base import Sampler
from cvlabkit.core.config import Config


class SubsetRandom(Sampler):
    """A sampler that samples elements randomly from a given list of indices, without replacement.

    This component wraps `torch.utils.data.SubsetRandomSampler`.
    """

    def __init__(self, cfg: Config, indices: list):
        """Initializes the SubsetRandom sampler.

        Args:
            cfg (Config): The configuration object.
            indices (list): A sequence of indices.
        """
        self.sampler = SubsetRandomSampler(indices)

    def __iter__(self):
        return iter(self.sampler)

    def __len__(self):
        return len(self.sampler)
