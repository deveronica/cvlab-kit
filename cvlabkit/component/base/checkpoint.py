from __future__ import annotations
from abc import ABC, abstractmethod


class Checkpoint(ABC):
    @abstractmethod
    def save(self, step: int) -> None: ...

    @abstractmethod
    def load_best(self) -> None: ...