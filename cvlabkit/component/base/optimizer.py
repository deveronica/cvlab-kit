from __future__ import annotations
from abc import abstractmethod
from torch.optim import Optimizer as TorchOptimizer
from cvlabkit.core.interface_meta import InterfaceMeta


class Optimizer(TorchOptimizer, metaclass=InterfaceMeta):
    """Abstract base class for all optimizers.
    """
    @abstractmethod
    def step(self) -> None:
        """Performs a single optimization step.
        """
        pass

    @abstractmethod
    def zero_grad(self) -> None:
        """Clears the gradients of all optimized `torch.Tensor`s.
        """
        pass