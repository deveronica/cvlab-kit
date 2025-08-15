from __future__ import annotations
from abc import abstractmethod
from torch.optim.lr_scheduler import _LRScheduler
from cvlabkit.core.interface_meta import InterfaceMeta


class Scheduler(_LRScheduler, metaclass=InterfaceMeta):
    """Abstract base class for all learning rate schedulers.
    """
    @abstractmethod
    def step(self) -> None:
        """Performs a single learning rate update step.
        """
        pass
