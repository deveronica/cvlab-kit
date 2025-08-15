from __future__ import annotations
from abc import abstractmethod
from typing import Iterator
from torch.utils.data import Sampler as TorchSampler
from cvlabkit.core.interface_meta import InterfaceMeta


class Sampler(TorchSampler, metaclass=InterfaceMeta):
    """Abstract base class for data samplers.
    """
    @abstractmethod
    def __iter__(self) -> Iterator[int]:
        """Returns an iterator that yields the indices of samples.
        """
        pass

    @abstractmethod
    def __len__(self) -> int:
        """Returns the total number of samples to be drawn.
        """
        pass