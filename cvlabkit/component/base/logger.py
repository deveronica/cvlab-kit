from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict, Iterator

import torch


class Logger(ABC):
    name: str
    version: str
    uri: str

    @abstractmethod
    def log_metrics(self, metrics: Dict[str, float], step: int) -> None: ...

    # @abstractmethod
    # def log_hyperparams(self, params: Dict[str, Any]) -> None: ...

    # @abstractmethod
    # def log_graph(self, model: Model, sample: torch.Tensor) -> None: ...

    # @abstractmethod
    # def save(self) -> None: ...

    @abstractmethod
    def finalize(self) -> None: ...