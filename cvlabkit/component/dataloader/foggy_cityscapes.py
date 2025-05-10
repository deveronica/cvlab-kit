import torch
from torch.utils.data import Dataset, DataLoader

from cvlabkit.component.base import DataLoader as BaseDataLoader


class DummyDataset(Dataset):
    def __getitem__(self, idx):
        return torch.zeros(3, 224, 224), torch.zeros(1)

    def __len__(self):
        return 1

class FoggyCityscapesLoader(BaseDataLoader):
    def __init__(self, cfg):
        self.loader = DataLoader(dataset=DummyDataset(), batch_size=1)

    def __iter__(self):
        return iter(self.loader)
