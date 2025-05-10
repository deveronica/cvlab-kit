from __future__ import annotations
from abc import ABC, abstractmethod


class Transform(ABC):
    @abstractmethod
    def __call__(self, sample: Any) -> Any: ...