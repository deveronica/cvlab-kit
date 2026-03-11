"""Image folder dataset for loading images from a directory."""

from __future__ import annotations

from functools import partial
from pathlib import Path

import torchvision.transforms as T
from PIL import Image
from torch import nn

from cvlabkit.component.base import Dataset


def _exists(v):
    """Check if value is not None (private helper)."""
    return v is not None


class ImageFolder(Dataset):
    """Dataset for loading images from a folder.

    Loads all images with specified extensions from a folder and applies
    transformations.
    """

    def __init__(self, cfg):
        """Initialize image folder dataset.

        Args:
            cfg: Configuration object with parameters:
                - folder: Path to folder containing images (required)
                - image_size: Size to resize images to (required)
                - exts: List of file extensions to load (default: ['jpg', 'jpeg', 'png', 'tiff'])
                - augment_horizontal_flip: Apply random horizontal flip (default: False)
                - convert_image_to: Convert images to specific mode (e.g., 'RGB', default: None)
        """
        super().__init__()

        folder = cfg.folder
        if isinstance(folder, str):
            folder = Path(folder)

        assert folder.is_dir(), f"Folder {folder} does not exist or is not a directory"

        self.folder = folder
        self.image_size = cfg.image_size

        exts = cfg.get("exts", ["jpg", "jpeg", "png", "tiff"])
        self.paths = [p for ext in exts for p in folder.glob(f"**/*.{ext}")]

        assert len(self.paths) > 0, (
            f"No images found in {folder} with extensions {exts}"
        )

        def convert_image_to_fn(img_type, image):
            if image.mode == img_type:
                return image
            return image.convert(img_type)

        convert_image_to = cfg.get("convert_image_to")
        maybe_convert_fn = (
            partial(convert_image_to_fn, convert_image_to)
            if _exists(convert_image_to)
            else nn.Identity()
        )

        augment_horizontal_flip = cfg.get("augment_horizontal_flip", False)

        self.transform = T.Compose(
            [
                T.Lambda(maybe_convert_fn),
                T.Resize(self.image_size),
                (
                    T.RandomHorizontalFlip()
                    if augment_horizontal_flip
                    else nn.Identity()
                ),
                T.CenterCrop(self.image_size),
                T.ToTensor(),
            ]
        )

    def __len__(self):
        """Return number of images."""
        return len(self.paths)

    def __getitem__(self, index):
        """Get image at index.

        Args:
            index: Index of image

        Returns:
            Transformed image tensor
        """
        path = self.paths[index]
        img = Image.open(path)
        return self.transform(img)
