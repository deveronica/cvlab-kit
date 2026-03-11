"""This module provides a PASCAL VOC 2012 Detection dataset component."""

from torchvision.datasets import VOCDetection

from cvlabkit.component.base import Dataset
from cvlabkit.core.config import Config


class Voc2012Detection(Dataset):
    """A PASCAL VOC 2012 Detection dataset component.

    This component wraps the `torchvision.datasets.VOCDetection` class to provide
    the VOC 2012 dataset for object detection tasks. It handles downloading,
    and applying transforms.

    The component returns images and their corresponding annotation data, which
    includes bounding boxes and labels for each object in the image.
    """

    def __init__(self, cfg: Config, transform=None):
        """Initializes the Voc2012Detection dataset component.

        Args:
            cfg (Config): The configuration object. Expected to contain keys like
                'data_root', 'split' (which maps to image_set), and 'download'.
            transform (callable, optional): A function/transform that takes in a
                PIL image and returns a transformed version. Defaults to None.
        """
        super().__init__()
        data_root = cfg.get("data_root", "./data/voc")
        # In VOC, the 'split' corresponds to the 'image_set' (e.g., 'train', 'val')
        image_set = cfg.get("split", "train")
        download = cfg.get("download", True)

        # The torchvision VOCDetection dataset is assigned to self.dataset.
        # The InterfaceMeta metaclass will automatically delegate unimplemented
        # methods to this self.dataset object.
        self.dataset = VOCDetection(
            root=data_root,
            year="2012",
            image_set=image_set,
            download=download,
            transform=transform,
        )

    def __getitem__(self, index):
        """Retrieves the dataset item at the given index.

        The item is a tuple containing the image and its annotation target.

        Args:
            index (int): The index of the item to retrieve.

        Returns:
            tuple: A tuple `(image, target)`, where `target` is a dictionary
                   containing the annotations for the image.
        """
        return self.dataset[index]

    def __len__(self):
        """Returns the total number of samples in the dataset.

        Returns:
            int: The size of the dataset.
        """
        return len(self.dataset)
