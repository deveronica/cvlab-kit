from __future__ import annotations

from abc import abstractmethod

import torch

from cvlabkit.core.interface_meta import InterfaceMeta


class Model(torch.nn.Module, metaclass=InterfaceMeta):
    """Abstract base class for all models."""

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Defines the forward pass of the model.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        pass
