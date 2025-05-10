from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Dict, Iterator

import torch


class DataLoader(ABC):
    dataset: Dataset
    batch_size: int
    num_workers: int
    pin_memory: bool
    drop_last: bool
    sampler: Sampler | None

    @abstractmethod
    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]: ...