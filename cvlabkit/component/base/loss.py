from __future__ import annotations
from abc import ABC, abstractmethod

import torch


class Loss(ABC):
    @abstractmethod
    def __call__(self, preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor: ...
