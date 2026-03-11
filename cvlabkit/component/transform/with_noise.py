import numpy as np
from PIL import Image
from torchvision import transforms

from cvlabkit.component.base import Transform


def Noise(img):
    if not isinstance(img, np.ndarray):
        array = np.array(img)
    randomStream = np.random.RandomState()
    noise = randomStream.gamma(size=array.shape, shape=2, scale=1 / 2).astype(
        array.dtype
    )
    img_noise = img * noise

    return Image.fromarray(img_noise)


class WithNoise(Transform):
    def __init__(self, cfg):
        super().__init__()
        size = cfg.get("size")
        self.weak = transforms.Compose(
            [transforms.RandomHorizontalFlip(), transforms.Resize((size, size))]
        )
        self.strong = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.Resize((size, size)),
                # for FGSM
                # RandAugmentMC(n=1, m=10) # Note: RandAugmentMC is not defined here
            ]
        )
        self.normalize = transforms.Compose([transforms.ToTensor()])

    def __call__(self, x):
        # x = Noise(x)
        weak = self.weak(x)
        strong = self.strong(x)
        return self.normalize(weak), self.normalize(strong)
