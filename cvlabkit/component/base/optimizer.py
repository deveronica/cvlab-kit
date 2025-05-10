from __future__ import annotations
from abc import ABC, abstractmethod


class Optimizer(ABC):
    @abstractmethod
    def step(self) -> None: ...

    @abstractmethod
    def zero_grad(self) -> None: ...