"""This module provides a KITTI dataset component for the CVLab-Kit."""

from torchvision.datasets import Kitti as TorchKitti

from cvlabkit.component.base import Dataset
from cvlabkit.core.config import Config


class Kitti(Dataset):
    """A KITTI dataset component that wraps the torchvision implementation.

    This component is designed for object detection tasks on the KITTI dataset.
    It handles downloading the dataset and applying transformations.
    Note that the KITTI dataset is large, and downloading may take a significant
    amount of time and disk space.
    """

    def __init__(self, cfg: Config, transform=None):
        """Initializes the Kitti dataset component.

        Args:
            cfg (Config): The configuration object. Expected to contain keys like
                'data_root', 'split', and 'download'.
            transform (callable, optional): A function/transform that takes in
                a PIL image and returns a transformed version. Defaults to None.
        """
        super().__init__()
        data_root = cfg.get("data_root", "./data/kitti")
        # In KITTI, the split is controlled by the 'train' boolean.
        is_train = cfg.get("split", "train") == "train"
        download = cfg.get(
            "download", False
        )  # KITTI download is not supported in torchvision

        # The torchvision Kitti dataset is assigned to self.dataset.
        # The InterfaceMeta metaclass will automatically delegate unimplemented
        # methods to this self.dataset object.
        self.dataset = TorchKitti(
            root=data_root, train=is_train, transform=transform, download=download
        )

    def __getitem__(self, index):
        """Retrieves the dataset item at the given index.

        The item is a tuple containing the image and its annotation target.

        Args:
            index (int): The index of the item to retrieve.

        Returns:
            tuple: A tuple `(image, target)`, where `target` is a list of
                   dictionaries, each containing annotations for an object.
        """
        return self.dataset[index]

    def __len__(self):
        """Returns the total number of samples in the dataset.

        Returns:
            int: The size of the dataset.
        """
        return len(self.dataset)
