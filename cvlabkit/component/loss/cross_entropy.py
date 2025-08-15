import torch.nn as nn
from cvlabkit.component.base import Loss


class CrossEntropy(Loss):
    def __init__(self, cfg):
        super().__init__()
        self.loss = nn.CrossEntropyLoss()

    def forward(self, preds, targets):
        return self.loss(preds, targets)
