from __future__ import annotations
from abc import ABC, abstractmethod

from torch.utils.data import Dataset as TorchDataset


class Dataset(TorchDataset):
    """Base class for all CVLab-Kit datasets.

    Subclasses must implement __getitem__ and __len__.
    """

    def __getitem__(self, index):
        raise NotImplementedError("Subclasses must implement __getitem__()")

    def __len__(self):
        raise NotImplementedError("Subclasses must implement __len__()")
