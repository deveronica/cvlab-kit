from __future__ import annotations

from abc import abstractmethod
from typing import Any, Dict

from cvlabkit.core.interface_meta import InterfaceMeta


class Checkpoint(metaclass=InterfaceMeta):
    """Abstract base class for checkpointing functionalities."""

    @abstractmethod
    def save(self, state: Dict[str, Any], file_path: str) -> None:
        """Saves the current state of the experiment.

        Args:
            state (Dict[str, Any]): The state dictionary to save.
            file_path (str): The path to the file where the state will be saved.
        """
        pass

    @abstractmethod
    def load(self, file_path: str) -> None:
        """Loads a previously saved state.

        Args:
            file_path (str): The path to the file from which the state will be loaded.
        """
        pass
