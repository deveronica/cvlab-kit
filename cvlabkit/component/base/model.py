from __future__ import annotations
from abc import ABC, abstractmethod
import torch


class Model(ABC):
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor: ...
