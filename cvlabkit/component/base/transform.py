from __future__ import annotations
from abc import abstractmethod
from typing import Any
from cvlabkit.core.interface_meta import InterfaceMeta


class Transform(metaclass=InterfaceMeta):
    """Abstract base class for all transforms.
    """
    @abstractmethod
    def __call__(self, sample: Any) -> Any:
        """Applies the transform to the given sample.

        Args:
            sample (Any): The input data to transform.

        Returns:
            Any: The transformed data.
        """
        pass