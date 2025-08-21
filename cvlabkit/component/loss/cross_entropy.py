import torch.nn as nn

from cvlabkit.component.base import Loss
from cvlabkit.core.config import Config


class CrossEntropy(Loss):
    """
    A loss component that wraps the standard torch.nn.CrossEntropyLoss.
    """
    def __init__(self, cfg: Config):
        """
        Initializes the CrossEntropy loss function.
        """
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, preds, targets):
        """
        Computes the cross-entropy loss between predictions and targets.

        Args:
            preds (torch.Tensor): The model's output logits.
            targets (torch.Tensor): The ground truth labels.

        Returns:
            torch.Tensor: The calculated loss value.
        """
        return self.loss_fn(preds, targets)
