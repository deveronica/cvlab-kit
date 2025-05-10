import torch
import torch.nn as nn
from cvlabkit.component.base import Loss


class IrgContrastiveLoss(Loss, nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.w_contrast = cfg.lambda_contrast if hasattr(cfg, "lambda_contrast") else 0.2

    def __call__(self, preds, irg_feats, targets):
        return torch.tensor(0.0)
