from typing import Dict, List

import torch
from sklearn.metrics import f1_score

from cvlabkit.component.base import Metric
from cvlabkit.core.config import Config


class F1(Metric):
    """A metric component that computes the F1 score.

    This component accumulates predictions and targets and uses scikit-learn's
    f1_score function for calculation. The averaging method (e.g., 'macro',
    'micro', 'weighted') can be configured.
    """

    def __init__(self, cfg: Config):
        """Initializes the F1 score metric.

        Args:
            cfg (Config): The configuration object. Expected keys:
                - "average" (str, optional): The averaging method for multi-class F1.
                                             Defaults to 'macro'.
        """
        self.average_method = cfg.get("average", "macro")
        # The key in the output dictionary will reflect the averaging method used.
        self.metric_name = f"f1_{self.average_method}"
        self.reset()

    def update(self, preds: torch.Tensor, targets: torch.Tensor) -> None:
        """Updates the metric's state with a new batch of predictions and targets.

        Args:
            preds (torch.Tensor): The output logits from the model.
            targets (torch.Tensor): The ground truth labels.
        """
        _, predicted_labels = torch.max(preds.data, 1)

        self.all_preds.extend(predicted_labels.cpu().numpy())
        self.all_targets.extend(targets.cpu().numpy())

    def compute(self) -> Dict[str, float]:
        """Computes the final F1 score from all accumulated predictions and targets.

        Returns:
            Dict[str, float]: A dictionary containing the calculated F1 score.
                              Returns zero if no samples have been updated.
        """
        if not self.all_targets:
            return {self.metric_name: 0.0}

        score = f1_score(
            self.all_targets,
            self.all_preds,
            average=self.average_method,
            zero_division=0,
        )

        return {self.metric_name: score}

    def reset(self) -> None:
        """Resets the internal state, clearing all accumulated predictions and targets."""
        self.all_preds: List[int] = []
        self.all_targets: List[int] = []
