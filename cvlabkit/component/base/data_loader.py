from __future__ import annotations
from abc import abstractmethod
from typing import Iterator
from torch.utils.data import DataLoader as TorchDataLoader
from cvlabkit.core.interface_meta import InterfaceMeta


class DataLoader(TorchDataLoader, metaclass=InterfaceMeta):
    """Base class for all CVLab-Kit data loaders.

    This class uses `InterfaceMeta` to act as a wrapper around a PyTorch
    `DataLoader` instance. Subclasses should define a `loader` attribute
    in their `__init__` method, which will be the target for attribute and
    method delegation.
    """

    @abstractmethod
    def __iter__(self) -> Iterator:
        """Returns an iterator over the dataset.
        """
        pass

    @abstractmethod
    def __len__(self) -> int:
        """Returns the number of batches in the data loader.
        """
        pass