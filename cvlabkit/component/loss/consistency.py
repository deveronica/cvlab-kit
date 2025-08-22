import torch
import torch.nn.functional as F
from cvlabkit.component.base import Loss
from cvlabkit.core.config import Config

class Consistency(Loss):
    """
    A loss component that calculates the consistency loss between two predictions,
    typically a pseudo-label and a prediction from a strongly augmented view.
    """
    def __init__(self, cfg: Config):
        """
        Initializes the Consistency loss. It does not require specific config parameters.
        """
        super().__init__()

    def forward(self, student_preds, teacher_preds):
        """
        Computes the consistency loss, typically using cross-entropy.

        The teacher's prediction is treated as the target (pseudo-label),
        and should not have gradients flowing back through it.

        Args:
            student_preds (torch.Tensor): The logits from the student model (strong augmentation).
            teacher_preds (torch.Tensor): The detached logits from the teacher model (weak augmentation).

        Returns:
            torch.Tensor: The consistency loss value.
        """
        # Ensure the teacher's predictions are treated as fixed targets
        pseudo_labels = torch.softmax(teacher_preds.detach(), dim=-1)
        
        # Calculate cross-entropy loss
        # The student's predictions should be log-softmaxed
        log_probs = F.log_softmax(student_preds, dim=-1)
        
        # The loss is the negative log-likelihood of the student's predictions
        # given the teacher's pseudo-labels.
        loss = -torch.sum(pseudo_labels * log_probs, dim=-1).mean()
        
        return loss