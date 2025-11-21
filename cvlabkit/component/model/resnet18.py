import torchvision

from cvlabkit.component.base import Model


class ResNet18(Model):
    def __init__(self, cfg):
        super().__init__()
        num_classes = cfg.get("num_classes", 1000)
        self.model = torchvision.models.resnet18(num_classes=num_classes)

    def forward(self, x):
        return self.model(x)
