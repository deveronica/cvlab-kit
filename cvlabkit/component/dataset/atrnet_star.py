"""This module provides an ATR-Net STAR dataset component for the CVLab-Kit."""

import os

from torchvision.datasets import ImageFolder

from cvlabkit.component.base import Dataset
from cvlabkit.core.config import Config


class AtrnetStar(Dataset):
    """An ATR-Net STAR dataset component using torchvision's ImageFolder.

    This component assumes the ATR-Net STAR dataset is organized in a directory
    structure compatible with ImageFolder, where images are sorted into
    subdirectories named after their classes. For example:
    `data_root/train/CLASS_1/image1.jpg`
    `data_root/train/CLASS_2/image2.jpg`

    It handles train/test splits by looking for 'train' and 'test' subfolders
    within the `data_root` specified in the configuration.
    """

    def __init__(self, cfg: Config, transform=None):
        """Initializes the AtrnetStar dataset component.

        Args:
            cfg (Config): The configuration object. Expected to contain keys like
                'data_root' and 'split'.
            transform (callable, optional): A function/transform that takes in
                a PIL image and returns a transformed version. Defaults to None.
        """
        super().__init__()
        data_root = cfg.get("data_root", "./data/atrnet_star")
        split = cfg.get("split", "train")  # 'train' or 'test'

        path = os.path.join(data_root, split)

        # torchvision.datasets.ImageFolder is a generic data loader that
        # will be wrapped by this component. The InterfaceMeta metaclass will
        # automatically delegate unimplemented methods to this self.dataset object.
        self.dataset = ImageFolder(root=path, transform=transform)

    def __getitem__(self, index):
        """Retrieves the dataset item at the given index.

        Args:
            index (int): The index of the item to retrieve from the dataset.

        Returns:
            tuple: A tuple containing the image and its corresponding label.
        """
        return self.dataset[index]

    def __len__(self):
        """Returns the total number of samples in the dataset.

        Returns:
            int: The size of the dataset.
        """
        return len(self.dataset)

    @property
    def classes(self):
        """Provides the list of class names in the dataset.

        Returns:
            list[str]: A list containing the names of the classes.
        """
        return self.dataset.classes
