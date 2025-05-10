import os
import torch
import torchvision
import torch.nn as nn
from cvlabkit.component.base import Model


class FasterRcnnFpnModel(Model, nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.detector = torchvision.models.detection.fasterrcnn_resnet50_fpn(
            pretrained=True,
            num_classes=91
        )
        ckpt = cfg.source_ckpt if hasattr(cfg, "source_ckpt") else None
        if ckpt and os.path.exists(ckpt):
            self.detector.load_state_dict(torch.load(ckpt))
        else:
            print(f"[Warning] checkpoint not found at '{ckpt}', skipping load")

        self.backbone = nn.Identity()
        self.backbone.out_channels = 256

    def forward(self, x, y):
        return self.detector(x, y)

    def parameters(self):
        return self.detector.parameters()
