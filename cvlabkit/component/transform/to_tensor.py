from torchvision import transforms
from cvlabkit.component.base import Transform


class ToTensor(Transform):
    def __init__(self, cfg):
        self.transform = transforms.ToTensor()

    def __call__(self, sample):
        return self.transform(sample)
