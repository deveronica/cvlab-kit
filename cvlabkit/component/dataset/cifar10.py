from torchvision.datasets import CIFAR10 as TorchCIFAR10
from cvlabkit.component.base import Dataset
from cvlabkit.core.config import Config


class Cifar10(Dataset):
    """CIFAR-10 dataset component.

    This class wraps the `torchvision.datasets.CIFAR10` dataset and handles
    downloading and applying transformations.
    """
    def __init__(self, cfg: Config, transform=None):
        """Initializes the Cifar10 dataset.

        Args:
            cfg (Config): The configuration object.
            transform (callable, optional): A transform to apply to the dataset. Defaults to None.
        """
        data_root = cfg.get("data_root", "./data")
        is_train = cfg.get("split", "train") == "train"
        download = cfg.get("download", False)

        self.dataset = TorchCIFAR10(
            root=data_root,
            train=is_train,
            download=download,
            transform=transform
        )
        self.transform = transform

    def __getitem__(self, index):
        """Retrieves the item at the given index.

        Args:
            index (int): The index of the item to retrieve.

        Returns:
            Any: The item at the specified index.
        """
        return self.dataset[index]

    def __len__(self):
        """Returns the total number of items in the dataset.

        Returns:
            int: The total number of items.
        """
        return len(self.dataset)