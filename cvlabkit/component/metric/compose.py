# cvlabkit/component/metric/compose.py

from typing import Any, Dict, List
from cvlabkit.core.config import Config
from cvlabkit.component.base import Metric


class Compose(Metric):
    """
    A generic metric component that composes a list of other metric components.
    It is initialized with a list of already-instantiated metric components.
    """
    def __init__(self, cfg: Config, components: List[Metric]):
        """
        Initializes the Compose metric.

        Args:
            cfg (Config): The configuration object for this component.
            components (List[Metric]): A list of PRE-INSTANTIATED metric components.
        """
        self.metrics = components

    def update(self, **kwargs) -> None:
        """Updates all metric components in the collection."""
        for metric in self.metrics:
            metric.update(**kwargs)

    def compute(self) -> Dict[str, float]:
        """Computes and aggregates results from all metric components."""
        results = {}
        for metric in self.metrics:
            results.update(metric.compute())
        return results

    def reset(self) -> None:
        """Resets all metric components in the collection."""
        for metric in self.metrics:
            metric.reset()