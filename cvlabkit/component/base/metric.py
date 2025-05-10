from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict


class Metric(ABC):
    @abstractmethod
    def update(self, **kwargs: Any) -> None: ...

    @abstractmethod
    def compute(self) -> Dict[str, float]: ...
