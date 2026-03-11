import numpy as np
import torch
from PIL import Image, ImageOps
from torchvision import transforms

from cvlabkit.component.base import Transform

IMAGE_SIZE = 32


def int_parameter(level, maxval):
    return int(level * maxval / 10)


def float_parameter(level, maxval):
    return float(level) * maxval / 10.0


def sample_level(n):
    return np.random.uniform(low=0.1, high=n)


def autocontrast(pil_img, _):
    return ImageOps.autocontrast(pil_img)


def equalize(pil_img, _):
    return ImageOps.equalize(pil_img)


def posterize(pil_img, level):
    level = int_parameter(sample_level(level), 4)
    return ImageOps.posterize(pil_img, 4 - level)


def rotate(pil_img, level):
    degrees = int_parameter(sample_level(level), 30)
    if np.random.uniform() > 0.5:
        degrees = -degrees
    return pil_img.rotate(degrees, resample=Image.BILINEAR)


def solarize(pil_img, level):
    level = int_parameter(sample_level(level), 256)
    return ImageOps.solarize(pil_img, 256 - level)


def shear_x(pil_img, level):
    level = float_parameter(sample_level(level), 0.3)
    if np.random.uniform() > 0.5:
        level = -level
    return pil_img.transform(
        (IMAGE_SIZE, IMAGE_SIZE),
        Image.AFFINE,
        (1, level, 0, 0, 1, 0),
        resample=Image.BILINEAR,
    )


def shear_y(pil_img, level):
    level = float_parameter(sample_level(level), 0.3)
    if np.random.uniform() > 0.5:
        level = -level
    return pil_img.transform(
        (IMAGE_SIZE, IMAGE_SIZE),
        Image.AFFINE,
        (1, 0, 0, level, 1, 0),
        resample=Image.BILINEAR,
    )


def translate_x(pil_img, level):
    level = int_parameter(sample_level(level), IMAGE_SIZE / 3)
    if np.random.random() > 0.5:
        level = -level
    return pil_img.transform(
        (IMAGE_SIZE, IMAGE_SIZE),
        Image.AFFINE,
        (1, 0, level, 0, 1, 0),
        resample=Image.BILINEAR,
    )


def translate_y(pil_img, level):
    level = int_parameter(sample_level(level), IMAGE_SIZE / 3)
    if np.random.random() > 0.5:
        level = -level
    return pil_img.transform(
        (IMAGE_SIZE, IMAGE_SIZE),
        Image.AFFINE,
        (1, 0, 0, 0, 1, level),
        resample=Image.BILINEAR,
    )


augmentations = [
    autocontrast,
    equalize,
    posterize,
    rotate,
    solarize,
    shear_x,
    shear_y,
    translate_x,
    translate_y,
]


def _aug(image, preprocess):
    aug_list = augmentations
    ws = np.float32(np.random.dirichlet([1] * 3))
    m = np.float32(np.random.beta(1, 1))

    mix = torch.zeros_like(preprocess(image))
    for i in range(3):
        image_aug = image.copy()
        depth = np.random.randint(1, 4)
        for _ in range(depth):
            op = np.random.choice(aug_list)
            image_aug = op(image_aug, 3)
        mix += ws[i] * preprocess(image_aug)

    mixed = (1 - m) * preprocess(image) + m * mix
    return mixed


class _AugMix:
    def __init__(self, transform=None):
        self.preprocess = transform

    def __call__(self, img):
        return _aug(img, self.preprocess)


class FixmatchAugmix(Transform):
    def __init__(self, cfg):
        super().__init__()
        mean = cfg.get("mean")
        std = cfg.get("std")
        self.base = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(
                    size=32, padding=int(32 * 0.125), padding_mode="reflect"
                ),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )
        self.weak = self.base
        self.strong1 = _AugMix(transform=self.base)
        self.strong2 = _AugMix(transform=self.base)

    def __call__(self, x):
        weak = self.weak(x)
        strong1 = self.strong1(x)
        strong2 = self.strong2(x)
        return weak, strong1, strong2
