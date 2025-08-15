from __future__ import annotations
from abc import abstractmethod
import torch
from cvlabkit.core.interface_meta import InterfaceMeta


class Loss(torch.nn.Module, metaclass=InterfaceMeta):
    """Abstract base class for all loss functions.

    Only distinguishes by argument count and type, not by argument names.
    """

    @abstractmethod
    def forward(self, *inputs: torch.Tensor) -> torch.Tensor:
        """Compute the loss.

        Args:
            *inputs (torch.Tensor): Tensors required for computation.
                - (x, y): two-tensor losses (CE, JSD 2-way)
                - (x, y, z): three-tensor losses (Triplet, JSD 3-way)
                - (student, teacher, target): KD variants

        Returns:
            torch.Tensor: loss value
        """
        raise NotImplementedError
