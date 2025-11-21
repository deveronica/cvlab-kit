from __future__ import annotations

from abc import abstractmethod
from typing import Any, Dict

from cvlabkit.core.interface_meta import InterfaceMeta


class Metric(metaclass=InterfaceMeta):
    """Abstract base class for all metrics."""

    @abstractmethod
    def update(self, **kwargs: Any) -> None:
        """Updates the metric's internal state with new data.

        Args:
            **kwargs: Arbitrary keyword arguments representing the data to update the metric.
        """
        pass

    @abstractmethod
    def compute(self) -> Dict[str, float]:
        """Computes the final metric value(s).

        Returns:
            Dict[str, float]: A dictionary of metric names and their computed values.
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """Resets the metric's internal state."""
        pass
