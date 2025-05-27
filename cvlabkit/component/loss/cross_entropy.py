import torch
import torch.nn as nn
from cvlabkit.component.base import Loss


class CrossEntropyLoss(Loss, nn.Module):
    def __init__(self):
        super().__init__()

    def __call__(self, preds, targets): 
        return torch.nn.functional.cross_entropy(preds, targets)
