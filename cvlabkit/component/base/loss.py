from __future__ import annotations
from abc import abstractmethod
import torch
from cvlabkit.core.interface_meta import InterfaceMeta


class Loss(torch.nn.Module, metaclass=InterfaceMeta):
    """Abstract base class for all loss functions.

    This class defines the interface for loss components. When implementing a custom
    loss, the `forward` method should be defined to accept the necessary tensors.
    The framework distinguishes between losses based on the number of arguments
    their `forward` method accepts.
    """

    @abstractmethod
    def forward(self, *inputs: torch.Tensor) -> torch.Tensor:
        """Computes the loss based on a variable number of input tensors.

        The number and order of tensors depend on the specific loss function being
        implemented.

        Args:
            *inputs (torch.Tensor): A sequence of tensors required for the loss
                computation. Common patterns include:
                - `(predictions, targets)`: For standard losses like Cross-Entropy.
                - `(anchor, positive, negative)`: For triplet-based losses.
                - `(student_preds, teacher_preds, targets)`: For knowledge distillation.

        Returns:
            torch.Tensor: A scalar tensor representing the computed loss value.
        """
        raise NotImplementedError
