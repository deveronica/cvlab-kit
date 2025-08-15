from __future__ import annotations
from abc import abstractmethod
from typing import Any, Dict
from cvlabkit.core.interface_meta import InterfaceMeta


class Logger(metaclass=InterfaceMeta):
    """Abstract base class for all loggers.
    """
    @abstractmethod
    def log_metrics(self, metrics: Dict[str, float], step: int) -> None:
        """Logs a dictionary of metrics at a given step.

        Args:
            metrics (Dict[str, float]): A dictionary of metric names and their values.
            step (int): The current step or epoch number.
        """
        pass

    @abstractmethod
    def log_hyperparams(self, params: Dict[str, Any]) -> None:
        """Logs a dictionary of hyperparameters.

        Args:
            params (Dict[str, Any]): A dictionary of hyperparameter names and their values.
        """
        pass

    @abstractmethod
    def finalize(self) -> None:
        """Finalizes the logging process, e.g., closes connections or saves data.
        """
        pass
