"""This module provides a YOLO format dataset component for the CVLab-Kit."""

from pathlib import Path

import numpy as np
from PIL import Image

from cvlabkit.component.base import Dataset
from cvlabkit.core.config import Config


class Yolo(Dataset):
    """A dataset component for loading data in YOLO format.

    This dataset assumes a directory structure where images and labels are in
    separate folders, but share the same filenames (with different extensions).
    For example:
    - `data_root/images/train/image1.jpg`
    - `data_root/labels/train/image1.txt`

    The label file is expected to contain one object per line, in the format:
    `<class_id> <x_center> <y_center> <width> <height>`
    where coordinates are normalized relative to the image size.
    """

    def __init__(self, cfg: Config, transform=None):
        """Initializes the Yolo dataset component.

        Args:
            cfg (Config): The configuration object. Expected to contain keys like
                'data_root' and 'split'.
            transform (callable, optional): A function/transform that takes in
                a PIL image and returns a transformed version. Defaults to None.
        """
        super().__init__()
        data_root = Path(cfg.get("data_root", "./data/yolo"))
        split = cfg.get("split", "train")
        self.transform = transform

        self.image_dir = data_root / "images" / split
        self.label_dir = data_root / "labels" / split

        self.image_files = sorted([f for f in self.image_dir.iterdir() if f.is_file()])

    def __len__(self):
        """Returns the total number of images in the dataset.

        Returns:
            int: The size of the dataset.
        """
        return len(self.image_files)

    def __getitem__(self, index):
        """Retrieves the dataset item at the given index.

        This method reads an image and its corresponding YOLO label file,
        parses the annotations, and returns them along with the image.

        Args:
            index (int): The index of the item to retrieve.

        Returns:
            tuple: A tuple `(image, target)`, where `target` is a dictionary
                   containing 'boxes' and 'labels' for the image.
        """
        image_path = self.image_files[index]
        label_path = self.label_dir / f"{image_path.stem}.txt"

        image = Image.open(image_path).convert("RGB")
        width, height = image.size

        boxes = []
        labels = []
        if label_path.exists():
            with open(label_path) as f:
                for line in f.readlines():
                    cls, x_c, y_c, w, h = map(float, line.strip().split())

                    # Convert YOLO format (center_x, center_y, width, height) to
                    # (xmin, ymin, xmax, ymax)
                    x1 = (x_c - w / 2) * width
                    y1 = (y_c - h / 2) * height
                    x2 = (x_c + w / 2) * width
                    y2 = (y_c + h / 2) * height

                    boxes.append([x1, y1, x2, y2])
                    labels.append(int(cls))

        target = {
            "boxes": np.array(boxes, dtype=np.float32),
            "labels": np.array(labels, dtype=np.int64),
        }

        if self.transform:
            # Note: Transforms for detection often need to handle both the image
            # and the bounding boxes. A simple torchvision transform might not work
            # as expected. Libraries like Albumentations are better suited for this.
            image = self.transform(image)

        return image, target
