import torch.nn as nn
import torchvision


class Model(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        num_classes = cfg["num_classes"]
        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
            num_classes=num_classes
        )

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)
