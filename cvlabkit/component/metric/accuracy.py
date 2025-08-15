from cvlabkit.component.base import Metric
import torch
from typing import Dict


class Accuracy(Metric):
    """A metric component for calculating classification accuracy."""

    def __init__(self, cfg):
        """Initializes the Accuracy metric."""
        super().__init__()
        self.reset()

    def update(self, preds: torch.Tensor, targets: torch.Tensor) -> None:
        """Updates the metric's state with a new batch of predictions and targets."""
        _, predicted = torch.max(preds.data, 1)
        self.total += targets.size(0)
        self.correct += (predicted == targets).sum().item()

    def compute(self) -> Dict[str, float]:
        """Computes the final accuracy from the accumulated state."""
        if self.total == 0:
            return {"accuracy": 0.0}
        accuracy = self.correct / self.total
        return {"accuracy": accuracy}

    def reset(self) -> None:
        """Resets the metric's state to its initial values."""
        self.correct = 0
        self.total = 0
