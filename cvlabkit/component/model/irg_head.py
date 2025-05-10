import torch
import torch.nn as nn
from cvlabkit.component.base import Model


class IrgHeadModel(Model, nn.Module):
    def __init__(self, cfg, backbone):
        super().__init__()
        dim = getattr(backbone, "out_channels", None)
        if dim is None or not isinstance(dim, int):
            raise ValueError(f"Invalid 'out_channels': {dim}")
        hid = 256
        self.net = nn.Sequential(
            nn.Linear(dim, hid),
            nn.ReLU(),
            nn.Linear(hid, hid)
        )

    def forward(self, roi_feats):
        return self.net(roi_feats)
