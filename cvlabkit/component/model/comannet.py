import torch.nn as nn
import torch.nn.functional as F

from cvlabkit.component.base import Model


class Comannet(Model):
    """Coman Suggested Model
    at 'A Deep Learning SAR Target Classification Experiment on MSTAR Dataset'
    (https://doi.org/10.23919/IRS.2018.8448048)
    """

    def __init__(self, cfg):
        super(Comannet, self).__init__()
        dense_in_features = cfg.get("coman_dense_in_features")
        num_classes = cfg.get("num_classes")

        self.conv1 = nn.Conv2d(1, 32, 3)
        self.conv2 = nn.Conv2d(32, 32, 3)
        self.pool1 = nn.MaxPool2d(2)
        self.drop1 = nn.Dropout(0.25)
        self.flat1 = nn.Flatten()
        self.dense1 = nn.Linear(dense_in_features, 128)
        self.drop2 = nn.Dropout(0.25)
        self.dense2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool1(x)
        x = self.drop1(x)
        x = self.flat1(x)
        x = F.relu(self.dense1(x))
        x = self.drop2(x)
        x = self.dense2(x)

        return x
