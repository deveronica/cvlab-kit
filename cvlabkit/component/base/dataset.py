from __future__ import annotations
from abc import abstractmethod
from typing import Any
from torch.utils.data import Dataset as TorchDataset
from cvlabkit.core.interface_meta import InterfaceMeta


class Dataset(TorchDataset, metaclass=InterfaceMeta):
    """Base class for all CVLab-Kit datasets.

    This class uses `InterfaceMeta` to act as a wrapper around a PyTorch
    `Dataset` instance. Subclasses should define a `dataset` attribute
    in their `__init__` method, which will be the target for attribute and
    method delegation.
    """

    @abstractmethod
    def __getitem__(self, index) -> Any:
        """Retrieves the item at the given index.

        Args:
            index (int): The index of the item to retrieve.

        Returns:
            Any: The item at the specified index.
        """
        pass

    @abstractmethod
    def __len__(self) -> int:
        """Returns the total number of items in the dataset.

        Returns:
            int: The total number of items.
        """
        pass