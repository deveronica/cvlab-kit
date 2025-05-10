from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Iterator


class Sampler(ABC):
    @abstractmethod
    def __iter__(self) -> Iterator[int]: ...
