from __future__ import annotations
from abc import ABC, abstractmethod

import torch


class Dataset(ABC):
    root: str
    train: bool
    download: bool
    transform: Transform | None

    @abstractmethod
    def __len__(self) -> int: ...

    @abstractmethod
    def __getitem__(self, idx: int) -> Any: ...