import os
from typing import Any, Tuple

from PIL import Image
import torch

from cvlabkit.component.base import Dataset
from cvlabkit.core.config import Config


class MSTAR(Dataset):
    """
    MSTAR (Moving and Stationary Target Acquisition and Recognition) dataset component.

    This component is responsible for loading image chips and their corresponding labels
    from the MSTAR dataset directory structure. Its sole purpose is to serve as a
    data source.
    """
    def __init__(self, cfg: Config, transform: Any = None):
        """
        Initializes the MSTAR dataset.

        Args:
            cfg (Config): The configuration object. Expected keys:
                - "root" (str): The root directory of the dataset.
                - "split" (str): The dataset split to use (e.g., "TRAIN", "TEST").
            transform (callable, optional): A transform to apply to dataset samples.
        """
        self.root = cfg.get("root", "./data/mstar")
        self.split = cfg.get("split", "TRAIN")
        self.transform = transform

        self.samples = []
        self.targets = []
        self.class_names = []
        self._load_dataset()

    def _load_dataset(self):
        """
        Loads the file paths and labels from the directory structure.
        """
        split_dir = os.path.join(self.root, self.split)
        if not os.path.isdir(split_dir):
            raise FileNotFoundError(f"Directory not found at: {split_dir}")

        # Discover class names from subdirectories.
        try:
            self.class_names = sorted([d.name for d in os.scandir(split_dir) if d.is_dir()])
            if not self.class_names:
                raise StopIteration
        except StopIteration:
            raise FileNotFoundError(f"No class directories found in {split_dir}")

        class_to_idx = {name: i for i, name in enumerate(self.class_names)}

        # Recursively find all image files and assign labels.
        for root, _, files in os.walk(split_dir):
            for fname in files:
                # This assumes a flat structure within class directories for simplicity.
                # Adjust if the structure is deeper (e.g., contains serial numbers).
                class_name = os.path.basename(root)
                if class_name in class_to_idx:
                    self.samples.append(os.path.join(root, fname))
                    self.targets.append(class_to_idx[class_name])

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        """
        Retrieves an image and its corresponding label at a given index.

        Args:
            index (int): The index of the item to retrieve.

        Returns:
            Tuple[torch.Tensor, int]: A tuple containing the image tensor and its integer label.
        """
        img_path = self.samples[index]
        label = self.targets[index]
        
        # Open as grayscale ('L') and convert to RGB for wider compatibility.
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label

    def __len__(self) -> int:
        """
        Returns the total number of samples in the dataset.
        """
        return len(self.samples)